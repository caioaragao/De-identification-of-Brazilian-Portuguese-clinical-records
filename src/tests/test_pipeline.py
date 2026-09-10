"""
Testes unitários para PipelineOrquestrador
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from PipelineOrquestrador import PipelineOrquestrador


def _llm_tuple(resp: dict, numero_tentativa: int = 1) -> tuple:
    """Constrói o par (resultado_final, tentativas_log) retornado por consultar_com_retry()."""
    t = {
        "numero_tentativa": numero_tentativa,
        "resposta":        resp.get("resposta", ""),
        "raciocinio":      resp.get("raciocinio", ""),
        "controle":        "",
        "status_erro":     "",
        "houve_falha":     False,
        "prompt_enviado":  "mock_prompt",
        "metricas_ollama": resp.get("metricas_ollama", {}),
        "loop_tipo":       None,
        "tempo_segundos":  0.1,
    }
    return (resp, [t])


class TestPipelineOrquestrador:
    def test_execucao_completa_pipeline(self):
        """Testa o caminho feliz de execução do pipeline ponta a ponta."""

        # Mocks de dependências pesadas
        mock_supabase = MagicMock()
        mock_supabase.obter_ou_criar_sessao.return_value = 1  # sessao_id = 1
        mock_supabase.carregar_caracteres_tag.return_value = ("[", "]")
        mock_supabase.carregar_conhecimento_sessao.return_value = ({}, {}, {}, {}, {}, {}, -1, 5)
        # Mock do LLM
        mock_ollama = MagicMock()
        mock_ollama.consultar_com_retry.return_value = _llm_tuple({"resposta": "{}"})

        # Mock Configs
        mock_config = MagicMock()
        mock_config.NOME_SESSAO = "test_sessao"
        mock_config.VERMELHO = ""
        mock_config.VERDE = ""
        mock_config.AZUL = ""
        mock_config.RESET = ""
        mock_config.AMARELO = ""
        mock_config.LLM_TEMPERATURA = 0.1
        mock_config.DATASET_INICIO = 0
        mock_config.PROMPT_ANONIMIZACAO = "prompt {texto_entrada}"
        mock_config.PROMPT_REVISAO_FINAL = "revisao {texto_entrada}"
        mock_config.PROMPT_LOOP_PENSAMENTO = "AVISO OBRIGATÓRIO {historico_raciocinio}"

        # Mock Funcoes Utils
        mock_funcoes = MagicMock()
        mock_funcoes.calcular_tokens.return_value = 10
        mock_funcoes.sanitizar_lista_tags.return_value = []

        # Precisamos mockar Anonimizacao também para evitar uso de complexidade real
        mock_anonimizacao_obj = MagicMock()
        mock_anonimizacao_obj.gerar_ngramas_corpus.return_value = ["ngram1", "ngram2"]
        mock_anonimizacao_obj.preparar_para_llm.side_effect = lambda x: x
        mock_anonimizacao_obj.converter_tags_para_llm.side_effect = lambda x: x
        mock_anonimizacao_obj.map_revisoes = {"termo_pendente": 1}

        # Mock a classe (factory) que será chamada no orquestrador
        mock_anonimizacao_class = MagicMock(return_value=mock_anonimizacao_obj)

        df = pd.DataFrame({"descricao_raw": ["Paciente 1", "Paciente 2"], "descricao": ["Paciente 1", "Paciente 2"]})

        orquestrador = PipelineOrquestrador(
            supabase_client=mock_supabase,
            cliente_ollama=mock_ollama,
            configuracao=mock_config,
            funcoes_gerais_module=mock_funcoes,
            anonimizacao_factory=mock_anonimizacao_class,
        )

        with patch("PipelineOrquestrador.time"):
            orquestrador.executar("modelo_x", df)

        # Verificações Core
        mock_supabase.atualizar_status_sessao.assert_any_call(1, "analisando_ngrams")
        mock_supabase.atualizar_status_sessao.assert_any_call(1, "processando_linha_a_linha")
        mock_supabase.atualizar_status_sessao.assert_any_call(1, "resolvendo_duvidas")
        mock_supabase.atualizar_status_sessao.assert_any_call(1, "concluido")

        # Verifica chamadas ao LLM (devem ter sido chamadas para n-grams e linha_a_linha e resolver duvidas)
        assert mock_ollama.consultar_com_retry.call_count > 0

    def test_fallback_texto_original_duvidas(self):
        """Testa se o pipeline refaz a requisição com o texto original ao encontrar uma dúvida."""
        mock_supabase = MagicMock()
        mock_supabase.obter_ou_criar_sessao.return_value = 1
        mock_supabase.carregar_caracteres_tag.return_value = ("[", "]")
        mock_supabase.carregar_conhecimento_sessao.return_value = ({}, {}, {}, {}, {}, {}, -1, 5)

        mock_ollama = MagicMock()
        # 1ª Tentativa com dúvidas | 2ª Tentativa bem-sucedida sem dúvidas
        mock_ollama.consultar_com_retry.side_effect = [
            _llm_tuple({"resposta": '{"duvidas_pendentes": ["Carlos"]}'}),
            _llm_tuple({"resposta": '{"map_nomes": ["Carlos Silva"]}'}),
        ]

        mock_config = MagicMock()
        mock_config.DATASET_INICIO = 0
        mock_config.PROMPT_ANONIMIZACAO = "prompt {texto_entrada}"
        mock_config.PROMPT_REVISAO_FINAL = "revisao {texto_entrada}"
        mock_config.PROMPT_LOOP_PENSAMENTO = "AVISO OBRIGATÓRIO {historico_raciocinio}"

        mock_funcoes = MagicMock()
        mock_funcoes.calcular_tokens.return_value = 10
        mock_funcoes.sanitizar_lista_tags.side_effect = lambda tags, texto: tags

        mock_anonimizacao_obj = MagicMock()
        mock_anonimizacao_obj.preparar_para_llm.return_value = "TEXTO_PRE_LLM"
        mock_anonimizacao_obj.converter_tags_para_llm.return_value = "TEXTO_ANONIMIZADO"

        mock_anonimizacao_class = MagicMock(return_value=mock_anonimizacao_obj)

        df = pd.DataFrame(
            {"descricao_raw": ["Paciente Carlos Silva Original"], "descricao": ["Paciente Carlos Silva Original"]}
        )

        orquestrador = PipelineOrquestrador(
            supabase_client=mock_supabase,
            cliente_ollama=mock_ollama,
            configuracao=mock_config,
            funcoes_gerais_module=mock_funcoes,
            anonimizacao_factory=mock_anonimizacao_class,
        )

        with patch("PipelineOrquestrador.time"):
            orquestrador.executar("modelo_x", df)

        # O LLM foi chamado 2 vezes dentro do laço de processamento de linhas e possivelmente mais para resolução inicial?
        # A 1ª chamada usou "TEXTO_ANONIMIZADO" e a 2ª chamada usou "Paciente Carlos Silva Original"
        chamadas_llm = mock_ollama.consultar_com_retry.call_args_list

        assert len(chamadas_llm) >= 2
        primeiro_prompt = chamadas_llm[0][0][0]
        segundo_prompt = chamadas_llm[1][0][0]

        assert "TEXTO_ANONIMIZADO" in primeiro_prompt
        assert "Paciente Carlos Silva Original" in segundo_prompt

    def test_pipeline_fase_ngram_chamada_llm(self):
        """Testa a fase de avaliação de N-Grams (boilerplates) consultando o LLM para separar PIIs e dúvidas."""
        mock_supabase = MagicMock()
        mock_supabase.obter_ou_criar_sessao.return_value = 1
        mock_supabase.carregar_caracteres_tag.return_value = ("[", "]")
        # ultimo_idx = -1, ultimo_ngram = 5
        mock_supabase.carregar_conhecimento_sessao.return_value = ({}, {}, {}, {}, {}, {}, -1, 5)

        mock_ollama = MagicMock()
        # Mock do retorno do LLM para a avaliação do ngram_alvo
        mock_ollama.consultar_com_retry.return_value = _llm_tuple({
            "resposta": '{"map_nomes": ["Dra. Ana"], "duvidas_pendentes": ["CRM 123"]}'
        })

        mock_config = MagicMock()
        mock_config.NGRAM_MIN = 4
        mock_config.DATASET_INICIO = 0
        mock_config.PROMPT_ANONIMIZACAO = "prompt {texto_entrada}"
        mock_config.PROMPT_REVISAO_FINAL = "revisao {texto_entrada}"
        mock_config.PROMPT_LOOP_PENSAMENTO = "AVISO OBRIGATÓRIO {historico_raciocinio}"

        mock_funcoes = MagicMock()
        mock_funcoes.calcular_tokens.return_value = 10
        mock_funcoes.sanitizar_lista_tags.side_effect = lambda tags, texto: tags

        mock_anonimizacao_obj = MagicMock()
        mock_anonimizacao_obj.preparar_corpus_ngrams.return_value = ["Corpus mockado"]

        # Faz com que apenas 1 iteração de n-gram (ex n=4) retorne candidatos
        def mock_gerar_ngrams(corpus, ngram_size):
            if ngram_size == 4:
                return [{"frase": "Dra. Ana CRM 123"}]
            return []

        mock_anonimizacao_obj.gerar_ngrams_candidatos_boilerplate.side_effect = mock_gerar_ngrams
        mock_anonimizacao_obj.verificar_e_aprender_variacao_boilerplate.return_value = False
        mock_anonimizacao_obj.preparar_para_llm.return_value = "TEXTO_PRE_LLM"
        mock_anonimizacao_obj.converter_tags_para_llm.return_value = "TEXTO_ANONIMIZADO"

        mock_anonimizacao_class = MagicMock(return_value=mock_anonimizacao_obj)

        df = pd.DataFrame({"descricao_raw": ["Paciente 1"], "descricao": ["Paciente 1"]})

        orquestrador = PipelineOrquestrador(
            supabase_client=mock_supabase,
            cliente_ollama=mock_ollama,
            configuracao=mock_config,
            funcoes_gerais_module=mock_funcoes,
            anonimizacao_factory=mock_anonimizacao_class,
        )

        with patch("PipelineOrquestrador.time"):
            orquestrador.executar("modelo_x", df)

        # Verifica que o registrar_e_get_tag foi chamado para registrar as descobertas do N-Gram no LLM
        mock_anonimizacao_obj._registrar_e_get_tag.assert_any_call("Dra. Ana", "nome")
        mock_anonimizacao_obj._registrar_e_get_tag.assert_any_call("CRM 123", "revisao")

        # Garante que as descobertas foram removidas do corpus N-gram localmente
        mock_anonimizacao_obj.remover_do_corpus.assert_any_call(["Corpus mockado"], "Dra. Ana")

    def test_resume_ngram_nao_pula_tamanho_parcial(self):
        """
        BUG-7: com ultimo_ngram=5, o resume deve executar n=5 (inclusive),
        não apenas n=4. Antes do fix, inicio_ngram = ultimo_ngram - 1 = 4
        pulava completamente o tamanho 5 que havia crashado no meio.
        """
        mock_supabase = MagicMock()
        mock_supabase.obter_ou_criar_sessao.return_value = 1
        mock_supabase.carregar_caracteres_tag.return_value = ("[", "]")
        # Simula crash durante n=5: ultimo_ngram=5, mas n=5 não foi concluído
        mock_supabase.carregar_conhecimento_sessao.return_value = ({}, {}, {}, {}, {}, {}, -1, 5)

        mock_ollama = MagicMock()
        mock_ollama.consultar_com_retry.return_value = _llm_tuple({"resposta": "{}"})

        mock_config = MagicMock()
        mock_config.NGRAM_MIN = 4
        mock_config.DATASET_INICIO = 0
        mock_config.PROMPT_ANONIMIZACAO = "prompt {texto_entrada}"
        mock_config.PROMPT_REVISAO_FINAL = "revisao {texto_entrada}"
        mock_config.PROMPT_LOOP_PENSAMENTO = "AVISO {historico_raciocinio}"

        mock_funcoes = MagicMock()
        mock_funcoes.calcular_tokens.return_value = 10
        mock_funcoes.sanitizar_lista_tags.side_effect = lambda tags, texto: tags

        mock_anonimizacao_obj = MagicMock()
        mock_anonimizacao_obj.preparar_corpus_ngrams.return_value = ["Corpus mockado"]
        mock_anonimizacao_obj.gerar_ngrams_candidatos_boilerplate.return_value = []
        mock_anonimizacao_obj.verificar_e_aprender_variacao_boilerplate.return_value = False

        mock_anonimizacao_class = MagicMock(return_value=mock_anonimizacao_obj)

        df = pd.DataFrame({"descricao_raw": ["Texto"], "descricao": ["Texto"]})

        orquestrador = PipelineOrquestrador(
            supabase_client=mock_supabase,
            cliente_ollama=mock_ollama,
            configuracao=mock_config,
            funcoes_gerais_module=mock_funcoes,
            anonimizacao_factory=mock_anonimizacao_class,
        )

        with patch("PipelineOrquestrador.time"):
            orquestrador.executar("modelo_x", df)

        # Com o fix (inicio_ngram = ultimo_ngram = 5), n=5 deve ser chamado
        tamanhos_chamados = [
            call.kwargs.get("ngram_size") or call.args[1]
            for call in mock_anonimizacao_obj.gerar_ngrams_candidatos_boilerplate.call_args_list
        ]
        assert 5 in tamanhos_chamados, (
            f"n=5 deveria ser executado no resume (ultimo_ngram=5), mas apenas {tamanhos_chamados} foram chamados"
        )


class TestPipelineMetricasOllama:
    """Verifica que as métricas reais da API Ollama são persistidas no log de execução."""

    def _montar_base(self, retorno_llm):
        mock_supabase = MagicMock()
        mock_supabase.obter_ou_criar_sessao.return_value = 1
        mock_supabase.carregar_caracteres_tag.return_value = ("[", "]")
        mock_supabase.carregar_conhecimento_sessao.return_value = ({}, {}, {}, {}, {}, {}, 0, 5)
        mock_supabase.obter_detalhes_sessao.return_value = {"status": "iniciado"}

        mock_ollama = MagicMock()
        mock_ollama.consultar_com_retry.return_value = _llm_tuple(retorno_llm)

        mock_config = MagicMock()
        mock_config.DATASET_INICIO = 0
        mock_config.PROMPT_ANONIMIZACAO = "prompt {texto_entrada}"
        mock_config.PROMPT_REVISAO_FINAL = "revisao {texto_entrada}"
        mock_config.PROMPT_LOOP_PENSAMENTO = "AVISO {historico_raciocinio}"

        mock_funcoes = MagicMock()
        mock_funcoes.sanitizar_lista_tags.side_effect = lambda tags, texto: tags

        mock_anonimizacao_obj = MagicMock()
        mock_anonimizacao_obj.map_revisoes = {}
        mock_anonimizacao_obj.preparar_para_llm.side_effect = lambda x: x[:len(x)//2] if len(x) > 4 else x
        mock_anonimizacao_obj.converter_tags_para_llm.side_effect = lambda x: x

        mock_anonimizacao_class = MagicMock(return_value=mock_anonimizacao_obj)

        # 2 linhas: índice 0 já processado (ultimo_idx=0), índice 1 será processado
        df = pd.DataFrame({"descricao_raw": ["Texto A", "Texto longo"], "descricao": ["Texto A", "Texto longo"]})

        orquestrador = PipelineOrquestrador(
            supabase_client=mock_supabase,
            cliente_ollama=mock_ollama,
            configuracao=mock_config,
            funcoes_gerais_module=mock_funcoes,
            anonimizacao_factory=mock_anonimizacao_class,
        )
        return orquestrador, df, mock_supabase

    def test_metricas_ollama_persistidas_na_fase_linha_a_linha(self):
        """As métricas reais da Ollama devem ser extraídas do retorno LLM e salvas no log."""
        retorno_llm = {
            "resposta": "{}",
            "metricas_ollama": {
                "prompt_eval_count": 150,
                "eval_count": 80,
                "eval_tokens_per_second": 25.5,
                "total_duration_sec": 3.2,
                "load_duration_sec": 0.1,
                "prompt_eval_duration_sec": 1.5,
                "eval_duration_sec": 1.6,
                "done_reason": "stop",
            },
        }
        orquestrador, df, mock_supabase = self._montar_base(retorno_llm)

        with patch("PipelineOrquestrador.time"):
            orquestrador.executar("modelo_x", df)

        chamadas = mock_supabase.salvar_log_execucao.call_args_list
        chamada_linha = next(
            (c for c in chamadas if c.kwargs.get("fase_atual") == "linha_a_linha"
             or (c.args and len(c.args) > 3 and c.args[3] == "linha_a_linha")),
            None,
        )
        assert chamada_linha is not None, "salvar_log_execucao não foi chamado para fase linha_a_linha"

        kwargs = chamada_linha.kwargs
        assert kwargs.get("prompt_eval_count") == 150
        assert kwargs.get("eval_count") == 80
        assert kwargs.get("eval_tokens_per_second") == 25.5
        assert kwargs.get("done_reason") == "stop"

    def test_chars_reducao_boilerplate_calculado(self):
        """chars_reducao_boilerplate deve refletir a diferença entre texto original e pré-LLM."""
        retorno_llm = {"resposta": "{}"}
        orquestrador, df, mock_supabase = self._montar_base(retorno_llm)

        with patch("PipelineOrquestrador.time"):
            orquestrador.executar("modelo_x", df)

        chamadas = mock_supabase.salvar_log_execucao.call_args_list
        chamada_linha = next(
            (c for c in chamadas if c.kwargs.get("fase_atual") == "linha_a_linha"
             or (c.args and len(c.args) > 3 and c.args[3] == "linha_a_linha")),
            None,
        )
        assert chamada_linha is not None
        # preparar_para_llm retorna metade do texto → redução positiva
        assert chamada_linha.kwargs.get("chars_reducao_boilerplate", 0) > 0

    def test_qtde_tentativas_2_quando_ha_duvida(self):
        """Quando o LLM retorna dúvidas, deve haver 2 registros no banco com qtde_tentativas 1 e 2."""
        retornos = [
            _llm_tuple({"resposta": '{"duvidas_pendentes": ["Carlos"]}'}),
            _llm_tuple({"resposta": "{}"}),
        ]
        mock_supabase = MagicMock()
        mock_supabase.obter_ou_criar_sessao.return_value = 1
        mock_supabase.carregar_caracteres_tag.return_value = ("[", "]")
        mock_supabase.carregar_conhecimento_sessao.return_value = ({}, {}, {}, {}, {}, {}, 0, 5)
        mock_supabase.obter_detalhes_sessao.return_value = {"status": "iniciado"}

        mock_ollama = MagicMock()
        mock_ollama.consultar_com_retry.side_effect = retornos

        mock_config = MagicMock()
        mock_config.DATASET_INICIO = 0
        mock_config.PROMPT_ANONIMIZACAO = "prompt {texto_entrada}"
        mock_config.PROMPT_REVISAO_FINAL = "revisao {texto_entrada}"
        mock_config.PROMPT_LOOP_PENSAMENTO = "AVISO {historico_raciocinio}"

        mock_funcoes = MagicMock()
        mock_funcoes.sanitizar_lista_tags.side_effect = lambda tags, texto: tags

        mock_anonimizacao_obj = MagicMock()
        mock_anonimizacao_obj.map_revisoes = {}
        mock_anonimizacao_obj.preparar_para_llm.side_effect = lambda x: x
        mock_anonimizacao_obj.converter_tags_para_llm.side_effect = lambda x: x

        # 2 linhas: índice 0 já processado (ultimo_idx=0), índice 1 será processado
        df = pd.DataFrame({"descricao_raw": ["Texto A", "Texto"], "descricao": ["Texto A", "Texto"]})

        orquestrador = PipelineOrquestrador(
            supabase_client=mock_supabase,
            cliente_ollama=mock_ollama,
            configuracao=mock_config,
            funcoes_gerais_module=mock_funcoes,
            anonimizacao_factory=MagicMock(return_value=mock_anonimizacao_obj),
        )

        with patch("PipelineOrquestrador.time"):
            orquestrador.executar("modelo_x", df)

        chamadas = mock_supabase.salvar_log_execucao.call_args_list
        chamadas_linha = [
            c for c in chamadas
            if c.kwargs.get("fase_atual") == "linha_a_linha"
            or (c.args and len(c.args) > 3 and c.args[3] == "linha_a_linha")
        ]
        # 2 chamadas ao consultar_com_retry → 2 registros no banco
        assert len(chamadas_linha) == 2
        tentativas_salvas = [c.kwargs.get("qtde_tentativas") for c in chamadas_linha]
        assert 1 in tentativas_salvas
        assert 2 in tentativas_salvas

    def test_tokens_antes_depois_zerados(self):
        """tokens_antes e tokens_depois devem ser 0 (tiktoken descontinuado)."""
        retorno_llm = {"resposta": "{}"}
        orquestrador, df, mock_supabase = self._montar_base(retorno_llm)

        with patch("PipelineOrquestrador.time"):
            orquestrador.executar("modelo_x", df)

        chamadas = mock_supabase.salvar_log_execucao.call_args_list
        chamada_linha = next(
            (c for c in chamadas if c.kwargs.get("fase_atual") == "linha_a_linha"
             or (c.args and len(c.args) > 3 and c.args[3] == "linha_a_linha")),
            None,
        )
        assert chamada_linha is not None
        # O pipeline não deve mais passar tokens_antes/tokens_depois como kwarg
        assert "tokens_antes" not in chamada_linha.kwargs
        assert "tokens_depois" not in chamada_linha.kwargs

    def test_raciocinio_e_resposta_bruta_persistidos_fase_linha_a_linha(self):
        """raciocinio (CoT thinking) e resposta_bruta são extraídos do retorno do LLM e persistidos."""
        retorno_llm = {
            "resposta": '{"map_nomes": [], "map_enderecos": [], "map_outros": [], "duvidas_pendentes": []}',
            "raciocinio": "Analisei o texto e não encontrei PIIs relevantes.",
            "metricas_ollama": {"prompt_eval_count": 100, "eval_count": 50},
        }
        orquestrador, df, mock_supabase = self._montar_base(retorno_llm)

        with patch("PipelineOrquestrador.time"):
            orquestrador.executar("modelo_x", df)

        chamadas = mock_supabase.salvar_log_execucao.call_args_list
        chamada_linha = next(
            (c for c in chamadas if c.kwargs.get("fase_atual") == "linha_a_linha"
             or (c.args and len(c.args) > 3 and c.args[3] == "linha_a_linha")),
            None,
        )
        assert chamada_linha is not None
        assert chamada_linha.kwargs.get("raciocinio") == "Analisei o texto e não encontrei PIIs relevantes."
        assert chamada_linha.kwargs.get("resposta_bruta") == (
            '{"map_nomes": [], "map_enderecos": [], "map_outros": [], "duvidas_pendentes": []}'
        )

    def test_raciocinio_e_resposta_bruta_persistidos_fase_ngram(self):
        """raciocinio e resposta_bruta são extraídos do retorno LLM e persistidos na fase ngram."""
        mock_supabase = MagicMock()
        mock_supabase.obter_ou_criar_sessao.return_value = 1
        mock_supabase.carregar_caracteres_tag.return_value = ("[", "]")
        mock_supabase.carregar_conhecimento_sessao.return_value = ({}, {}, {}, {}, {}, {}, -1, 5)

        retorno_ngram = {
            "resposta": '{"map_nomes": ["Dra. Ana"], "duvidas_pendentes": []}',
            "raciocinio": "O trecho parece ser nome de médica.",
            "metricas_ollama": {"prompt_eval_count": 80, "eval_count": 30},
        }
        mock_ollama = MagicMock()
        mock_ollama.consultar_com_retry.return_value = _llm_tuple(retorno_ngram)

        mock_config = MagicMock()
        mock_config.NGRAM_MIN = 4
        mock_config.DATASET_INICIO = 0
        mock_config.PROMPT_ANONIMIZACAO = "prompt {texto_entrada}"
        mock_config.PROMPT_REVISAO_FINAL = "revisao {texto_entrada}"
        mock_config.PROMPT_LOOP_PENSAMENTO = "AVISO {historico_raciocinio}"

        mock_funcoes = MagicMock()
        mock_funcoes.sanitizar_lista_tags.side_effect = lambda tags, texto: tags

        mock_anonimizacao_obj = MagicMock()
        mock_anonimizacao_obj.preparar_corpus_ngrams.return_value = ["Corpus mockado"]
        mock_anonimizacao_obj.verificar_e_aprender_variacao_boilerplate.return_value = False
        mock_anonimizacao_obj.preparar_para_llm.return_value = "TEXTO_PRE_LLM"
        mock_anonimizacao_obj.converter_tags_para_llm.return_value = "TEXTO_ANONIMIZADO"
        mock_anonimizacao_obj.map_revisoes = {}

        def mock_gerar_ngrams(corpus, ngram_size):
            return [{"frase": "Dra. Ana"}] if ngram_size == 4 else []

        mock_anonimizacao_obj.gerar_ngrams_candidatos_boilerplate.side_effect = mock_gerar_ngrams
        mock_anonimizacao_class = MagicMock(return_value=mock_anonimizacao_obj)

        df = pd.DataFrame({"descricao_raw": ["Texto"], "descricao": ["Texto"]})
        orquestrador = PipelineOrquestrador(
            supabase_client=mock_supabase,
            cliente_ollama=mock_ollama,
            configuracao=mock_config,
            funcoes_gerais_module=mock_funcoes,
            anonimizacao_factory=mock_anonimizacao_class,
        )

        with patch("PipelineOrquestrador.time"):
            orquestrador.executar("modelo_x", df)

        chamadas = mock_supabase.salvar_log_execucao.call_args_list
        # O resumo por tamanho de N-gram não tem raciocinio; buscamos o call do item processado
        chamada_ngram = next(
            (c for c in chamadas if "raciocinio" in c.kwargs),
            None,
        )
        assert chamada_ngram is not None
        assert chamada_ngram.kwargs.get("raciocinio") == "O trecho parece ser nome de médica."
        assert chamada_ngram.kwargs.get("resposta_bruta") == '{"map_nomes": ["Dra. Ana"], "duvidas_pendentes": []}'


class TestPipelineRetornoCaminhoMask:
    def _montar_orquestrador(self):
        """Monta orquestrador mínimo com mocks básicos."""
        mock_supabase = MagicMock()
        mock_supabase.obter_ou_criar_sessao.return_value = 1
        mock_supabase.carregar_caracteres_tag.return_value = ("[", "]")
        mock_supabase.carregar_conhecimento_sessao.return_value = ({}, {}, {}, {}, {}, {}, 0, 5)
        mock_supabase.obter_detalhes_sessao.return_value = {"status": "iniciado"}

        mock_ollama = MagicMock()
        mock_ollama.consultar_com_retry.return_value = _llm_tuple({"resposta": "{}"})

        mock_config = MagicMock()
        mock_config.DATASET_INICIO = 0
        mock_config.PROMPT_ANONIMIZACAO = "prompt {texto_entrada}"
        mock_config.PROMPT_REVISAO_FINAL = "revisao {texto_entrada}"
        mock_config.PROMPT_LOOP_PENSAMENTO = "AVISO {historico_raciocinio}"

        mock_funcoes = MagicMock()
        mock_funcoes.calcular_tokens.return_value = 5

        mock_anonimizacao_obj = MagicMock()
        mock_anonimizacao_obj.map_revisoes = {}
        mock_anonimizacao_obj.preparar_para_llm.side_effect = lambda x: x
        mock_anonimizacao_obj.converter_tags_para_llm.side_effect = lambda x: x

        mock_anonimizacao_class = MagicMock(return_value=mock_anonimizacao_obj)

        df = pd.DataFrame({"descricao_raw": ["Texto 1"], "descricao": ["Texto 1"]})

        orquestrador = PipelineOrquestrador(
            supabase_client=mock_supabase,
            cliente_ollama=mock_ollama,
            configuracao=mock_config,
            funcoes_gerais_module=mock_funcoes,
            anonimizacao_factory=mock_anonimizacao_class,
        )
        return orquestrador, df

    def test_executar_retorna_caminho_mask_quando_relatorio_gerado(self):
        """executar() deve retornar o caminho do .mask quando gerar_relatorio_if_yes=True."""
        orquestrador, df = self._montar_orquestrador()

        caminho_esperado = "/saida/fake/mascara.mask"
        mock_gerador = MagicMock()
        mock_gerador.gerar.return_value = caminho_esperado

        with patch("PipelineOrquestrador.time"), patch(
            "PipelineOrquestrador.GeradorRelatorio", return_value=mock_gerador
        ):
            resultado = orquestrador.executar("modelo_x", df, gerar_relatorio_if_yes=True, is_yes=True)

        assert resultado == caminho_esperado

    def test_executar_retorna_none_quando_relatorio_nao_solicitado(self):
        """executar() deve retornar None quando gerar_relatorio_if_yes=False."""
        orquestrador, df = self._montar_orquestrador()

        with patch("PipelineOrquestrador.time"):
            resultado = orquestrador.executar("modelo_x", df, gerar_relatorio_if_yes=False, is_yes=False)

        assert resultado is None
