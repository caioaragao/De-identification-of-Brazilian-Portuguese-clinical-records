"""
Testes unitários para o ClienteOllama.py
"""

from unittest.mock import MagicMock, patch

from a01_platform import config_router as config
from ClienteOllama import ClienteOllama


class TestConsultarComRetry:
    def setup_method(self):
        # Evita a tentativa de conectar ao Ollama real
        with patch("ollama.Client"):
            self.cliente = ClienteOllama("http://fake-host")

    def _mock_executar_prompt(self, respostas):
        """Cria geradores sequenciais para o mock de executar_prompt."""
        geradores = []
        for resp in respostas:

            def make_gen(r=resp):
                yield r

            geradores.append(make_gen())
        return geradores

    @patch("ClienteOllama.time.sleep")
    def test_sucesso_na_primeira_tentativa(self, mock_sleep):
        config.LLM_TEMPERATURA = 0.1
        self.cliente.executar_prompt = MagicMock()
        self.cliente.executar_prompt.side_effect = self._mock_executar_prompt(
            [
                {"controle": "", "raciocinio": "", "resposta": '{"map_nomes": []}'},
            ]
        )

        resultado, tentativas = self.cliente.consultar_com_retry("prompt", "qwen3:14b")
        assert resultado["resposta"] == '{"map_nomes": []}'
        assert self.cliente.executar_prompt.call_count == 1
        mock_sleep.assert_not_called()
        assert len(tentativas) == 1
        assert tentativas[0]["numero_tentativa"] == 1
        assert tentativas[0]["houve_falha"] is False

    @patch("ClienteOllama.time.sleep")
    def test_sucesso_na_segunda_tentativa(self, mock_sleep):
        config.LLM_TEMPERATURA = 0.1
        self.cliente.executar_prompt = MagicMock()
        self.cliente.executar_prompt.side_effect = self._mock_executar_prompt(
            [
                {"controle": "", "raciocinio": "", "resposta": "", "status_erro": "erro_stop_reason"},
                {"controle": "", "raciocinio": "", "resposta": '{"map_nomes": []}', "status_erro": ""},
            ]
        )

        resultado, tentativas = self.cliente.consultar_com_retry("prompt", "qwen3:14b")
        assert resultado["resposta"] == '{"map_nomes": []}'
        assert self.cliente.executar_prompt.call_count == 2
        mock_sleep.assert_called_once_with(2)
        assert len(tentativas) == 2
        assert tentativas[0]["houve_falha"] is True
        assert tentativas[0]["status_erro"] == "erro_stop_reason"
        assert tentativas[1]["houve_falha"] is False

    @patch("ClienteOllama.time.sleep")
    def test_falha_em_todas_tentativas(self, mock_sleep):
        config.LLM_TEMPERATURA = 0.1
        self.cliente.executar_prompt = MagicMock()
        self.cliente.executar_prompt.side_effect = self._mock_executar_prompt(
            [
                {"controle": "", "raciocinio": "", "resposta": "", "status_erro": "erro_stop_reason"},
                {"controle": "", "raciocinio": "", "resposta": "", "status_erro": "erro"},
                {"controle": "", "raciocinio": "", "resposta": "", "status_erro": ""},
            ]
        )

        resultado, tentativas = self.cliente.consultar_com_retry("prompt", "qwen3:14b")
        assert "houve_falha" in resultado
        assert self.cliente.executar_prompt.call_count == 3
        assert mock_sleep.call_count == 2
        assert len(tentativas) == 3

    @patch("ClienteOllama.time.sleep")
    def test_consultar_com_retry_context_limit_injeta_pensamento(self, mock_sleep):
        config.LLM_TEMPERATURA = 0.1
        config.LLM_NUM_CTX = 8192
        config.LLM_NUM_PREDICT = 4096

        self.cliente.executar_prompt = MagicMock()
        self.cliente.executar_prompt.side_effect = self._mock_executar_prompt(
            [
                {
                    "controle": "",
                    "raciocinio": "Pensamento interrompido pela metade",
                    "resposta": "",
                    "status_erro": "erro_stop_reason",
                },
                {"controle": "", "raciocinio": "", "resposta": '{"sucesso": true}', "status_erro": ""},
            ]
        )

        resultado, tentativas = self.cliente.consultar_com_retry("Prompt Original", "qwen")
        assert resultado["resposta"] == '{"sucesso": true}'
        assert self.cliente.executar_prompt.call_count == 2

        # Verifica argumentos da tentativa 2
        chamada_2_kwargs = self.cliente.executar_prompt.call_args_list[1].kwargs

        assert "Pensamento interrompido pela metade" in chamada_2_kwargs["prompt"]
        assert "Prompt Original" in chamada_2_kwargs["prompt"]
        assert "AVISO OBRIGATÓRIO" in chamada_2_kwargs["prompt"]

        assert chamada_2_kwargs["think"] is False
        # num_ctx permanece fixo para evitar reload do modelo na VRAM; só num_predict aumenta
        assert chamada_2_kwargs["num_ctx"] == 8192
        assert chamada_2_kwargs["num_predict"] == 8192 - 68  # LLM_NUM_CTX - 68
        assert len(tentativas) == 2
        assert tentativas[0]["numero_tentativa"] == 1
        assert tentativas[1]["numero_tentativa"] == 2

    @patch("ClienteOllama.time.sleep")
    def test_timeout_em_uma_tentativa(self, mock_sleep):
        config.LLM_TEMPERATURA = 0.1
        self.cliente.executar_prompt = MagicMock()
        self.cliente.executar_prompt.side_effect = self._mock_executar_prompt(
            [
                {"controle": "", "raciocinio": "", "resposta": "", "status_erro": "timeout"},
                {"controle": "", "raciocinio": "", "resposta": '{"sucesso": true}', "status_erro": ""},
            ]
        )

        resultado, tentativas = self.cliente.consultar_com_retry("prompt", "qwen3:14b")
        assert resultado["resposta"] == '{"sucesso": true}'
        assert self.cliente.executar_prompt.call_count == 2
        mock_sleep.assert_called_once_with(2)
        assert len(tentativas) == 2


class TestExecutarPrompt:
    def setup_method(self):
        with patch("ollama.Client"):
            self.cliente = ClienteOllama("http://fake-host")

    def test_executar_prompt_sucesso_fluxo_normal(self):
        mock_response_stream = [
            {"message": {"thinking": "Pensando na resposta...", "content": ""}},
            {"message": {"thinking": "", "content": "Resposta "}},
            {"message": {"content": "Final.", "thinking": ""}},
            {"done": True, "done_reason": "stop"},
        ]
        self.cliente.cliente.chat.return_value = (item for item in mock_response_stream)

        gerador = self.cliente.executar_prompt("Oi", modelo="qwen")
        resultados = list(gerador)

        ultimo_estado = resultados[-1]
        assert ultimo_estado["raciocinio"] == "Pensando na resposta..."
        assert ultimo_estado["resposta"] == "Resposta Final."
        assert "RESPONDENDO" in ultimo_estado["controle"]
        assert "PENSANDO" in ultimo_estado["controle"]

    def test_loop_detectado_com_injection_false(self):
        """Loop no content deve ser detectado mesmo com PROMPT_INJECTION=False."""
        config.PROMPT_INJECTION = False
        janela = "A" * 100
        mock_response_stream = [
            {"message": {"content": janela}},
            {"message": {"content": janela}},
            {"message": {"content": janela}},
            {"done": False},
        ]
        self.cliente.cliente.chat.return_value = (item for item in mock_response_stream)

        try:
            gerador = self.cliente.executar_prompt("Oi", modelo="gemma3:4b")
            resultados = list(gerador)
            ultimo_estado = resultados[-1]
            assert ultimo_estado["status_erro"] == "erro_stop_reason"
            assert "LOOP DETECTADO" in ultimo_estado["controle"]
        finally:
            config.PROMPT_INJECTION = True

    def test_executar_prompt_loop_detection(self):
        # A janela de repeticao eh de 100 chars, e o threshold eh 3
        # Para que o algoritimo exato bata, precisamos mandar pedaços com exatos 100 chars (ou multiplos perfeitos)
        janela = (
            "Loop Test Repetitions Over and Over " * 3
        )  # len("Loop Test Repetitions Over and Over ") = 36. Not good.
        janela = "A" * 100

        mock_response_stream = [
            {"message": {"content": janela}},
            {"message": {"content": janela}},
            {"message": {"content": janela}},
            {"done": False},
        ]
        self.cliente.cliente.chat.return_value = (item for item in mock_response_stream)

        gerador = self.cliente.executar_prompt("Oi", modelo="qwen")
        resultados = list(gerador)

        ultimo_estado = resultados[-1]
        assert ultimo_estado["status_erro"] == "erro_stop_reason"
        assert "LOOP DETECTADO" in ultimo_estado["controle"]


class TestPromptInjectionFlag:
    """Verifica comportamento da flag PROMPT_INJECTION em overflow de contexto."""

    def setup_method(self):
        with patch("ollama.Client"):
            self.cliente = ClienteOllama("http://fake-host")

    def _mock_overflow(self):
        """Gera um generator que simula overflow (done_reason != 'stop')."""
        def make_gen():
            yield {
                "controle": "",
                "raciocinio": "pensamento incompleto...",
                "resposta": "",
                "status_erro": "erro_stop_reason",
            }
        return make_gen()

    @patch("ClienteOllama.time.sleep")
    def test_overflow_injection_false_retorna_deteccao_zerada(self, mock_sleep):
        """PROMPT_INJECTION=False: overflow retorna listas vazias e houve_falha=True sem retry."""
        config.LLM_TEMPERATURA = 0.1
        config.PROMPT_INJECTION = False

        self.cliente.executar_prompt = MagicMock(return_value=self._mock_overflow())

        resultado, tentativas = self.cliente.consultar_com_retry("prompt", "qwen3:14b", max_tentativas=3)

        # Deve ter chamado o LLM apenas 1 vez (sem retry)
        assert self.cliente.executar_prompt.call_count == 1
        assert resultado["houve_falha"] is True
        assert resultado["resposta"] == '{"map_nomes": [], "map_enderecos": [], "map_outros": [], "duvidas_pendentes": []}'
        # Sem sleep (não tentou novamente)
        mock_sleep.assert_not_called()
        assert len(tentativas) == 1
        assert tentativas[0]["houve_falha"] is True

    @patch("ClienteOllama.time.sleep")
    def test_overflow_injection_false_preserva_saida_real_llm(self, mock_sleep):
        """PROMPT_INJECTION=False: resposta_llm_real contém o conteúdo truncado real do LLM."""
        config.LLM_TEMPERATURA = 0.1
        config.PROMPT_INJECTION = False

        self.cliente.executar_prompt = MagicMock(return_value=self._mock_overflow())

        resultado, tentativas = self.cliente.consultar_com_retry("prompt", "qwen3:14b", max_tentativas=3)

        # resposta_llm_real deve ter o conteúdo real do LLM (do mock: resposta vazia, raciocínio incompleto)
        assert "resposta_llm_real" in resultado
        # raciocínio preservado do LLM real, não vazio forçado
        assert resultado["raciocinio"] == "pensamento incompleto..."

    @patch("ClienteOllama.time.sleep")
    def test_overflow_injection_true_resgata_raciocinio(self, mock_sleep):
        """PROMPT_INJECTION=True: overflow injeta raciocínio e tenta novamente."""
        config.LLM_TEMPERATURA = 0.1
        config.PROMPT_INJECTION = True

        overflow = {
            "controle": "",
            "raciocinio": "pensamento incompleto...",
            "resposta": "",
            "status_erro": "erro_stop_reason",
        }
        sucesso = {
            "controle": "",
            "raciocinio": "",
            "resposta": '{"map_nomes": ["Maria"]}',
            "status_erro": "",
        }

        def make_gen(r):
            yield r

        self.cliente.executar_prompt = MagicMock(
            side_effect=[make_gen(overflow), make_gen(sucesso)]
        )

        resultado, tentativas = self.cliente.consultar_com_retry("prompt", "qwen3:14b", max_tentativas=3)

        # Deve ter chamado o LLM 2 vezes (overflow + retry com raciocínio injetado)
        assert self.cliente.executar_prompt.call_count == 2
        assert resultado["resposta"] == '{"map_nomes": ["Maria"]}'
        assert resultado.get("houve_falha") is not True
        assert len(tentativas) == 2
        assert tentativas[0]["status_erro"] == "erro_stop_reason"
        assert tentativas[1]["status_erro"] == ""

    def test_llm_thinking_false_passa_think_false(self):
        """LLM_THINKING=False: executar_prompt deve ser chamado com think=False em todas as tentativas."""
        config.LLM_THINKING = False
        config.LLM_TEMPERATURA = 0.1
        config.PROMPT_INJECTION = True

        def make_gen():
            yield {"controle": "", "raciocinio": "", "resposta": '{"map_nomes": []}', "status_erro": ""}

        self.cliente.executar_prompt = MagicMock(return_value=make_gen())

        try:
            self.cliente.consultar_com_retry("prompt", "gemma3:4b")
            _, kwargs = self.cliente.executar_prompt.call_args
            assert kwargs.get("think") is False
        finally:
            config.LLM_THINKING = True


class TestRespostaLlmReal:
    """Garante que resposta_llm_real é preservada em todos os caminhos de falha."""

    def setup_method(self):
        with patch("ollama.Client"):
            self.cliente = ClienteOllama("http://fake-host")

    @patch("ClienteOllama.time.sleep")
    def test_falha_todas_tentativas_preserva_resposta_llm_real(self, mock_sleep):
        """Quando todas as tentativas falham (PROMPT_INJECTION=True), resposta_llm_real contém o último output."""
        config.LLM_TEMPERATURA = 0.1
        config.PROMPT_INJECTION = True

        saida_truncada = "json parcialmente gerado pelo llm..."

        def make_gen(r):
            yield r

        self.cliente.executar_prompt = MagicMock(
            side_effect=[
                make_gen({"controle": "", "raciocinio": "pens.", "resposta": saida_truncada, "status_erro": "erro_stop_reason"}),
                make_gen({"controle": "", "raciocinio": "",     "resposta": "",              "status_erro": "erro_stop_reason"}),
            ]
        )

        resultado, tentativas = self.cliente.consultar_com_retry("prompt", "qwen", max_tentativas=2)

        assert resultado["houve_falha"] is True
        assert "resposta_llm_real" in resultado
        # O último tentativa é a segunda (resposta vazia), então resposta_llm_real é string vazia
        assert resultado["resposta_llm_real"] == ""
        assert len(tentativas) == 2

    @patch("ClienteOllama.time.sleep")
    def test_injection_false_resposta_llm_real_preserva_truncado(self, mock_sleep):
        """PROMPT_INJECTION=False: resposta_llm_real contém o conteúdo truncado (não o sintético)."""
        config.LLM_TEMPERATURA = 0.1
        config.PROMPT_INJECTION = False

        conteudo_truncado = '{"map_nomes": ["Joao"'  # JSON incompleto que o LLM emitiu

        def make_gen():
            yield {
                "controle": "PENSANDO...",
                "raciocinio": "raciocínio incompleto",
                "resposta": conteudo_truncado,
                "status_erro": "erro_stop_reason",
            }

        self.cliente.executar_prompt = MagicMock(return_value=make_gen())

        resultado, tentativas = self.cliente.consultar_com_retry("prompt", "qwen", max_tentativas=3)

        assert resultado["houve_falha"] is True
        # 'resposta' ainda é o JSON sintético vazio (usado pelo pipeline downstream)
        assert '"map_nomes": []' in resultado["resposta"]
        # 'resposta_llm_real' é o conteúdo real truncado
        assert resultado["resposta_llm_real"] == conteudo_truncado
        # 'raciocínio' é preservado (não forçado para vazio)
        assert resultado["raciocinio"] == "raciocínio incompleto"
        # 'controle' é preservado
        assert resultado["controle"] == "PENSANDO..."
        assert len(tentativas) == 1


class TestRaciocinioPersistencia:
    """Verifica que raciocínio é preservado corretamente nos casos críticos."""

    def setup_method(self):
        with patch("ollama.Client"):
            self.cliente = ClienteOllama("http://fake-host")

    @patch("ClienteOllama.time.sleep")
    def test_raciocinio_da_primeira_tentativa_preservado_apos_retry(self, mock_sleep):
        """Após retry com PROMPT_INJECTION, raciocínio da tentativa 1 é incluído no retorno final."""
        config.LLM_TEMPERATURA = 0.1
        config.PROMPT_INJECTION = True
        config.LLM_NUM_CTX = 8192

        raciocinio_original = "Pensei na questão: há um nome de pessoa no texto."
        overflow = {
            "controle": "",
            "raciocinio": raciocinio_original,
            "resposta": "",
            "status_erro": "erro_stop_reason",
        }
        sucesso = {
            "controle": "",
            "raciocinio": "",  # retry com think=False → raciocinio vazio
            "resposta": '{"map_nomes": ["Ana"]}',
            "status_erro": "",
        }

        def make_gen(r):
            yield r

        self.cliente.executar_prompt = MagicMock(
            side_effect=[make_gen(overflow), make_gen(sucesso)]
        )

        resultado, tentativas = self.cliente.consultar_com_retry("prompt", "qwen", max_tentativas=2)

        assert resultado["resposta"] == '{"map_nomes": ["Ana"]}'
        # O raciocínio da tentativa 1 deve ser preservado no retorno final
        assert resultado["raciocinio"] == raciocinio_original
        assert len(tentativas) == 2
        assert tentativas[0]["raciocinio"] == raciocinio_original

    def test_think_tags_extraidas_para_raciocinio(self):
        """Modelos que embtem CoT em <think>...</think> dentro de content devem ter raciocínio separado."""
        mock_stream = [
            {"message": {"content": "<think>Analisei o texto.", "thinking": None}},
            {"message": {"content": " Não há PII.</think>", "thinking": None}},
            {"message": {"content": '{"map_nomes": []}', "thinking": None}},
            {"done": True, "done_reason": "stop"},
        ]
        self.cliente.cliente.chat.return_value = (item for item in mock_stream)

        gerador = self.cliente.executar_prompt("Texto", modelo="gemma3")
        resultados = list(gerador)

        ultimo = resultados[-1]
        assert ultimo["raciocinio"] == "Analisei o texto. Não há PII."
        assert ultimo["resposta"] == '{"map_nomes": []}'
