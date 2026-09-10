import json
import re
import time

from Anonimizacao import Anonimizacao
from funcoes_gerais import carregar_contador, sanitizar_lista_tags
from GeradorRelatorio import GeradorRelatorio
import logger
from logger import LogLevel


class PipelineOrquestrador:
    """
    Orquestra o pipeline completo de anonimização, integrando banco de dados (Supabase),
    LLM (ClienteOllama), controle de estado e a classe de Anonimizacao.
    """

    def __init__(
        self, supabase_client, cliente_ollama, configuracao, funcoes_gerais_module, anonimizacao_factory=Anonimizacao
    ):
        """
        Injeta as dependências necessárias para a execução do pipeline.

        Args:
            supabase_client: Instância do cliente Supabase para persistência/estado
            cliente_ollama: Interface com o modelo LLM
            configuracao: Módulo contendo variáveis de configuração globale e constantes de cor
            funcoes_gerais_module: Utilitários
            anonimizacao_factory: Factory (ou classe) para instanciar o Anonimizacao.
        """
        self._supabase = supabase_client
        self._llm = cliente_ollama
        self._config = configuracao
        self._funcoes_gerais = funcoes_gerais_module
        self._anonimizacao_factory = anonimizacao_factory

    def executar(self, modelo_atual: str, df, gerar_relatorio_if_yes: bool = False, is_yes: bool = False):
        """
        Executa a pipeline de extração N-GRAM e processamento LINHA a LINHA sobre o dataframe.
        """

        # --- 2. CONFIGURAÇÃO DE SESSÃO NO SUPABASE ---
        sessao_id = self._supabase.obter_ou_criar_sessao(modelo_atual)
        logger.log(f"Sessão ID vinculada: {sessao_id} (Modelo: {modelo_atual})", level=LogLevel.NORMAL)

        # VERIFICAÇÃO DE STATUS: Se já concluído, pula completamente (antes de qualquer escrita)
        detalhes_sessao = self._supabase.obter_detalhes_sessao(sessao_id)
        status_atual = detalhes_sessao.get("status", "desconhecido")

        if status_atual == "concluido":
            logger.log(f"Modelo '{modelo_atual}' já está CONCLUÍDO. Pulando.", level=LogLevel.NORMAL)
            return

        self._supabase.atualizar_status_sessao(sessao_id, "iniciado")

        # --- 3. TAGs Dinâmicas: Carrega ou seleciona caracteres únicos para esta sessão
        tag_ini, tag_fim = self._supabase.carregar_caracteres_tag(sessao_id)
        if tag_ini is None or tag_fim is None:
            # Sessão nova: seleciona caracteres únicos baseado no corpus
            texto_completo = "".join(df["descricao"].dropna().astype(str))
            tag_ini, tag_fim = self._funcoes_gerais.identificar_caracteres_mascara(texto_completo, 2)
            self._supabase.salvar_caracteres_tag(sessao_id, tag_ini, tag_fim)
            logger.log(f"TAGs selecionadas: INI='{tag_ini}' FIM='{tag_fim}'", level=LogLevel.NORMAL)
        else:
            logger.log(f"TAGs carregadas da sessão: INI='{tag_ini}' FIM='{tag_fim}'", level=LogLevel.NORMAL)

        # Inicializa estado com TAGs dinâmicas
        anonimizacao = self._anonimizacao_factory(tag_ini=tag_ini, tag_fim=tag_fim)

        # --- 3. RECUPERAÇÃO DE ESTADO RELACIONAL (RESUME) ---
        (m_nomes, m_ends, m_outros, m_boilers, m_revs, m_unknowns, ultimo_idx, ultimo_ngram) = (
            self._supabase.carregar_conhecimento_sessao(sessao_id)
        )

        # Restaura memória da classe Anonimizacao
        anonimizacao.map_nomes_encontrados = m_nomes
        anonimizacao.map_enderecos_encontrados = m_ends
        anonimizacao.map_outras_info_encontradas = m_outros
        anonimizacao.map_boilerplates_encontrados = m_boilers
        anonimizacao.map_revisoes = m_revs
        anonimizacao.map_unknown_encontrados = m_unknowns

        # Popula índice de otimização
        for b in m_boilers:
            anonimizacao.set_boilerplates_pontuacao_removida.add(anonimizacao._normalizar_para_comparacao(b))

        # Restaura contadores
        anonimizacao.contador_nomes = carregar_contador(m_nomes)
        anonimizacao.contador_enderecos = carregar_contador(m_ends)
        anonimizacao.contador_outros = carregar_contador(m_outros)
        anonimizacao.contador_boilerplates = carregar_contador(m_boilers)
        anonimizacao.contador_revisoes = carregar_contador(m_revs)
        anonimizacao.contador_unknown = carregar_contador(m_unknowns)

        # --- 4. FASE N-GRAM (SÓ RODA SE FOR SESSÃO NOVA - PROGRESSO ZERO) ---
        if ultimo_idx == -1:
            logger.log("Iniciando varredura de boilerplates (N-Grams)...", level=LogLevel.NORMAL)
            self._supabase.atualizar_status_sessao(
                sessao_id, "analisando_ngrams"
            )  # Usando analisando_ngrams como test_pipeline.py

            inicio_ngram = ultimo_ngram
            corpus_pacientes = anonimizacao.preparar_corpus_ngrams(df.copy())

            for n in range(inicio_ngram, self._config.NGRAM_MIN - 1, -1):
                t_inicio_n = time.time()
                logger.log(f"Pesquisando {n}-grams candidatos a boilerplate...", level=LogLevel.NORMAL)
                ngrams_candidatos = anonimizacao.gerar_ngrams_candidatos_boilerplate(corpus_pacientes, ngram_size=n)

                if ngrams_candidatos:
                    for item in ngrams_candidatos:
                        ngram_alvo = item["frase"]

                        if anonimizacao.verificar_e_aprender_variacao_boilerplate(ngram_alvo):
                            anonimizacao.remover_do_corpus(corpus_pacientes, ngram_alvo)
                            continue

                        resp, tentativas_llm_ngram = self._llm.consultar_com_retry(
                            self._config.PROMPT_ANONIMIZACAO.format(texto_entrada=ngram_alvo), modelo_atual
                        )
                        houve_duvida_ngram = False

                        try:
                            dados = json.loads(re.sub(r"```json|```", "", resp.get("resposta", "{}")).strip())
                            if not isinstance(dados, dict):
                                raise json.JSONDecodeError("resposta não é objeto JSON", "", 0)
                            duvidas = sanitizar_lista_tags(dados.get("duvidas_pendentes", []), ngram_alvo)
                            if duvidas:
                                houve_duvida_ngram = True
                            for termo_duvida in duvidas:
                                anonimizacao._registrar_e_get_tag(termo_duvida, "revisao")

                            pii_nomes      = sanitizar_lista_tags(dados.get("map_nomes", []), ngram_alvo)
                            pii_ends       = sanitizar_lista_tags(dados.get("map_enderecos", []), ngram_alvo)
                            pii_outros     = sanitizar_lista_tags(dados.get("map_outros", []), ngram_alvo)
                            pii_idades     = sanitizar_lista_tags(dados.get("map_idades", []), ngram_alvo)
                            pii_profissoes = sanitizar_lista_tags(dados.get("map_profissoes", []), ngram_alvo)
                            pii_outros    += pii_idades + pii_profissoes

                            if pii_nomes or pii_ends or pii_outros:
                                for termo in pii_nomes:
                                    anonimizacao._registrar_e_get_tag(termo, "nome")
                                for termo in pii_ends:
                                    anonimizacao._registrar_e_get_tag(termo, "endereco")
                                for termo in pii_outros:
                                    anonimizacao._registrar_e_get_tag(termo, "outros")

                                for termo in pii_nomes + pii_ends + pii_outros:
                                    anonimizacao.remover_do_corpus(corpus_pacientes, termo)
                            elif not duvidas:
                                anonimizacao.salvar_novo_boilerplate(ngram_alvo)
                                anonimizacao.remover_do_corpus(corpus_pacientes, ngram_alvo)
                        except json.JSONDecodeError:
                            pass

                        for t in tentativas_llm_ngram:
                            metricas = t.get("metricas_ollama", {})
                            log_dict = {
                                "raciocinio": t["raciocinio"],
                                "resposta": t["resposta"],
                                "metricas_ollama": metricas,
                                "controle": t["controle"],
                            }
                            self._supabase.salvar_log_execucao(
                                sessao_id,
                                0,
                                t["tempo_segundos"],
                                "ngram",
                                tamanho_ngram=int(n),
                                houve_duvida=houve_duvida_ngram if not t["houve_falha"] else False,
                                houve_falha=t["houve_falha"],
                                status_erro=t["status_erro"],
                                texto_original=ngram_alvo,
                                texto_pre_llm=t["prompt_enviado"],
                                llm_raw_response=json.dumps(log_dict, ensure_ascii=False),
                                prompt_eval_count=metricas.get("prompt_eval_count"),
                                eval_count=metricas.get("eval_count"),
                                eval_tokens_per_second=metricas.get("eval_tokens_per_second"),
                                total_duration_ollama_sec=metricas.get("total_duration_sec"),
                                load_duration_sec=metricas.get("load_duration_sec"),
                                prompt_eval_duration_sec=metricas.get("prompt_eval_duration_sec"),
                                eval_duration_sec=metricas.get("eval_duration_sec"),
                                done_reason=metricas.get("done_reason"),
                                qtde_tentativas=t["numero_tentativa"],
                                raciocinio=t["raciocinio"],
                                resposta_bruta=t["resposta"],
                            )

                self._supabase.salvar_conhecimento_lote(sessao_id, "nome", anonimizacao.map_nomes_encontrados)
                self._supabase.salvar_conhecimento_lote(sessao_id, "endereco", anonimizacao.map_enderecos_encontrados)
                self._supabase.salvar_conhecimento_lote(sessao_id, "info", anonimizacao.map_outras_info_encontradas)
                self._supabase.salvar_conhecimento_lote(
                    sessao_id, "boilerplate", anonimizacao.map_boilerplates_encontrados
                )
                self._supabase.salvar_conhecimento_lote(sessao_id, "revisao", anonimizacao.map_revisoes)
                self._supabase.salvar_conhecimento_lote(sessao_id, "unknown", anonimizacao.map_unknown_encontrados)
                self._supabase.salvar_log_execucao(sessao_id, 0, time.time() - t_inicio_n, "ngram_resumo", int(n))
            logger.log("Limpeza de N-Grams concluída.", level=LogLevel.NORMAL)

        # --- 5. LOOP PRINCIPAL DE PROCESSAMENTO (Linha a Linha) ---
        logger.log("Iniciando processamento linha a linha...", level=LogLevel.NORMAL)
        self._supabase.atualizar_status_sessao(sessao_id, "processando_linha_a_linha")

        indice_inicial = max(0, ultimo_idx + 1)
        if hasattr(self._config, "DATASET_INICIO") and indice_inicial < self._config.DATASET_INICIO:
            indice_inicial = self._config.DATASET_INICIO

        evolucao_raw = None

        for i, evolucao_raw in df["descricao"].dropna().items():
            try:
                idx_int = int(str(i))
                if idx_int < indice_inicial:
                    continue
            except ValueError:
                cor_aviso = getattr(self._config, "AMARELO", self._config.VERMELHO)
                logger.log(f"Aviso: Índice inválido/não-numérico '{i}' ignorado.", level=LogLevel.WARNING)
                continue

            logger.log(f"[{modelo_atual}] Processando Evolução {i}...", level=LogLevel.NORMAL)

            evolucao_pre_llm = anonimizacao.preparar_para_llm(evolucao_raw)
            chars_reducao_boilerplate = len(evolucao_raw) - len(evolucao_pre_llm)
            texto_para_llm = anonimizacao.converter_tags_para_llm(evolucao_pre_llm)

            # Laço de até 2 tentativas para resolver dúvidas com texto original.
            # Cada chamada a consultar_com_retry() pode ter N retries internos;
            # acumulamos todas as tentativas para logging por tentativa.
            todas_tentativas_llm = []
            offset_tentativa = 0
            houve_duvida = False

            for tentativa_resolver_duvida in range(1, 3):
                texto_prompt = texto_para_llm if tentativa_resolver_duvida == 1 else evolucao_raw

                if tentativa_resolver_duvida == 2:
                    logger.log("LLM teve dúvida. Re-enviando com texto original (Tentativa 2)...", level=LogLevel.WARNING)

                retorno_llm, tentativas_llm = self._llm.consultar_com_retry(
                    self._config.PROMPT_ANONIMIZACAO.format(texto_entrada=texto_prompt), modelo_atual
                )

                # Offset garante numeração sequencial entre chamadas externas (1, 2, 3, ...)
                for t in tentativas_llm:
                    t_offset = dict(t)
                    t_offset["numero_tentativa"] += offset_tentativa
                    todas_tentativas_llm.append(t_offset)
                offset_tentativa += len(tentativas_llm)

                houve_duvida = False
                if isinstance(retorno_llm, dict):
                    try:
                        dados_resp = json.loads(re.sub(r"```json|```", "", retorno_llm.get("resposta", "{}")).strip())
                        if not isinstance(dados_resp, dict):
                            raise json.JSONDecodeError("resposta não é objeto JSON", "", 0)

                        if "duvidas_pendentes" in dados_resp:
                            dados_resp["duvidas_pendentes"] = sanitizar_lista_tags(
                                dados_resp["duvidas_pendentes"], evolucao_raw
                            )
                        if "map_nomes" in dados_resp:
                            dados_resp["map_nomes"] = sanitizar_lista_tags(dados_resp["map_nomes"], evolucao_raw)
                        if "map_enderecos" in dados_resp:
                            dados_resp["map_enderecos"] = sanitizar_lista_tags(
                                dados_resp["map_enderecos"], evolucao_raw
                            )
                        idades     = sanitizar_lista_tags(dados_resp.pop("map_idades",     []), evolucao_raw)
                        profissoes = sanitizar_lista_tags(dados_resp.pop("map_profissoes", []), evolucao_raw)
                        if "map_outros" in dados_resp:
                            dados_resp["map_outros"] = sanitizar_lista_tags(dados_resp["map_outros"], evolucao_raw)
                        else:
                            dados_resp["map_outros"] = []
                        dados_resp["map_outros"] += idades + profissoes

                        if dados_resp.get("duvidas_pendentes"):
                            houve_duvida = True
                            if tentativa_resolver_duvida == 2:
                                logger.log(f"Dúvidas persistentes detectadas: {dados_resp['duvidas_pendentes']}", level=LogLevel.WARNING)
                                for termo_duvida in dados_resp.get("duvidas_pendentes"):
                                    anonimizacao._registrar_e_get_tag(termo_duvida, "revisao")

                        retorno_llm["resposta"] = json.dumps(dados_resp)

                        # Marca a última tentativa desta chamada com houve_duvida correto
                        if todas_tentativas_llm:
                            todas_tentativas_llm[-1]["_houve_duvida"] = houve_duvida

                        # Se não houve dúvida ou já estamos na segunda tentativa, podemos sair do laço de retentativa
                        if not houve_duvida or tentativa_resolver_duvida == 2:
                            break

                    except json.JSONDecodeError:
                        # Em caso de erro JSON, consideramos uma falha e tentamos de novo se for a primeira tentativa
                        pass

            if isinstance(retorno_llm, dict):
                anonimizacao.atualizar_piis_e_boilerplates(evolucao_raw, retorno_llm)

            self._supabase.salvar_conhecimento_lote(sessao_id, "nome", anonimizacao.map_nomes_encontrados)
            self._supabase.salvar_conhecimento_lote(sessao_id, "endereco", anonimizacao.map_enderecos_encontrados)
            self._supabase.salvar_conhecimento_lote(sessao_id, "info", anonimizacao.map_outras_info_encontradas)
            self._supabase.salvar_conhecimento_lote(sessao_id, "boilerplate", anonimizacao.map_boilerplates_encontrados)
            self._supabase.salvar_conhecimento_lote(sessao_id, "revisao", anonimizacao.map_revisoes)

            for t in todas_tentativas_llm:
                metricas = t.get("metricas_ollama", {})
                log_dict = {
                    "raciocinio": t["raciocinio"],
                    "resposta": t["resposta"],
                    "metricas_ollama": metricas,
                    "controle": t["controle"],
                }
                self._supabase.salvar_log_execucao(
                    sessao_id,
                    int(str(i)),
                    t["tempo_segundos"],
                    "linha_a_linha",
                    houve_duvida=t.get("_houve_duvida", False),
                    houve_falha=t["houve_falha"],
                    status_erro=t["status_erro"],
                    texto_original=evolucao_raw,
                    texto_pre_llm=t["prompt_enviado"],
                    llm_raw_response=json.dumps(log_dict, ensure_ascii=False),
                    prompt_eval_count=metricas.get("prompt_eval_count"),
                    eval_count=metricas.get("eval_count"),
                    eval_tokens_per_second=metricas.get("eval_tokens_per_second"),
                    total_duration_ollama_sec=metricas.get("total_duration_sec"),
                    load_duration_sec=metricas.get("load_duration_sec"),
                    prompt_eval_duration_sec=metricas.get("prompt_eval_duration_sec"),
                    eval_duration_sec=metricas.get("eval_duration_sec"),
                    done_reason=metricas.get("done_reason"),
                    qtde_tentativas=t["numero_tentativa"],
                    chars_reducao_boilerplate=chars_reducao_boilerplate,
                    raciocinio=t["raciocinio"],
                    resposta_bruta=t["resposta"],
                )

        logger.log(f"\nPROCESSO DE ANONIMIZAÇÃO CONCLUÍDO ({modelo_atual})", level=LogLevel.NORMAL)

        # --- 6. RESOLUÇÃO DE DÚVIDAS PENDENTES ---
        self._supabase.atualizar_status_sessao(sessao_id, "resolvendo_duvidas")
        self._resolver_revisoes_pendentes(anonimizacao, modelo_atual, sessao_id)

        # --- 6.1 PERSISTÊNCIA FINAL ---
        self._supabase.salvar_conhecimento_lote(sessao_id, "nome", anonimizacao.map_nomes_encontrados)
        self._supabase.salvar_conhecimento_lote(sessao_id, "endereco", anonimizacao.map_enderecos_encontrados)
        self._supabase.salvar_conhecimento_lote(sessao_id, "info", anonimizacao.map_outras_info_encontradas)
        self._supabase.salvar_conhecimento_lote(sessao_id, "boilerplate", anonimizacao.map_boilerplates_encontrados)
        self._supabase.salvar_conhecimento_lote(sessao_id, "unknown", anonimizacao.map_unknown_encontrados)
        self._supabase.salvar_conhecimento_lote(sessao_id, "revisao", anonimizacao.map_revisoes)

        self._supabase.atualizar_status_sessao(sessao_id, "concluido")

        # --- 7. EXPORTAÇÃO DE RESULTADOS ---
        caminho_mask = None
        if evolucao_raw is not None and gerar_relatorio_if_yes and is_yes:
            gerador = GeradorRelatorio(anonimizacao, self._config, self._funcoes_gerais)
            caminho_mask = gerador.gerar(df, modelo_atual, silencioso=False)
        return caminho_mask

    def _resolver_revisoes_pendentes(self, anonimizacao, modelo_atual: str, sessao_id: int):
        if not anonimizacao.map_revisoes:
            return

        logger.log(f"\nIniciando resolução final de {len(anonimizacao.map_revisoes)} termos...", level=LogLevel.NORMAL)
        termos_resolvidos_para_remocao = []

        for termo in list(anonimizacao.map_revisoes.keys()):
            logger.log(f"Auditando: '{termo}'...", level=LogLevel.NORMAL, end=" ")
            resp, _ = self._llm.consultar_com_retry(
                self._config.PROMPT_REVISAO_FINAL.format(texto_entrada=termo), modelo_atual, max_tentativas=2
            )

            try:
                dados = json.loads(re.sub(r"```json|```", "", resp.get("resposta", "{}")).strip())
                if not isinstance(dados, dict):
                    raise json.JSONDecodeError("resposta não é objeto JSON", "", 0)
                categoria = dados.get("classificacao", "outros").lower()

                if categoria == "seguro":
                    logger.log("Seguro.", level=LogLevel.NORMAL)
                    termos_resolvidos_para_remocao.append(termo)
                    if termo in anonimizacao.map_revisoes:
                        del anonimizacao.map_revisoes[termo]
                elif categoria in ["nome", "endereco", "outros"]:
                    logger.log(f"Classificado como {categoria}.", level=LogLevel.NORMAL)
                    anonimizacao._registrar_e_get_tag(termo, categoria)
                    termos_resolvidos_para_remocao.append(termo)
                else:
                    logger.log("Incerteza mantida -> UNKNOWN.", level=LogLevel.WARNING)
                    anonimizacao._registrar_e_get_tag(termo, "unknown")
                    termos_resolvidos_para_remocao.append(termo)
            except Exception as e:
                logger.log(f"Erro ({e}) -> UNKNOWN.", level=LogLevel.ERROR)
                anonimizacao._registrar_e_get_tag(termo, "unknown")
                termos_resolvidos_para_remocao.append(termo)

        if termos_resolvidos_para_remocao:
            logger.log(f"Arquivando {len(termos_resolvidos_para_remocao)} termos resolvidos no histórico...", level=LogLevel.NORMAL)
            self._supabase.arquivar_conhecimento_lote(
                sessao_id, "revisao", "revisao_resolvida", termos_resolvidos_para_remocao
            )
