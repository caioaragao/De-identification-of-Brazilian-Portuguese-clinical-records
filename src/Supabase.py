"""
Módulo de Persistência com Supabase.

Gerencia todas as interações com o banco de dados Supabase, responsável por:
1. Gerenciar Sessões de Teste (tabela p02_sessoes).
2. Armazenar o conhecimento aprendido (vocabulário de PIIs e Boilerplates) (tabela p02_conhecimento).
3. Logar métricas de execução para auditoria (tabela p02_logs_execucao).
"""

import json
from datetime import datetime
from typing import Any

from supabase import Client, create_client

from a01_platform import config_router as config
import logger
from logger import LogLevel


class Supabase:
    """
    Cliente para operações de banco de dados no Supabase.
    Abstrai a conexão e métodos específicos de inserção/consulta do projeto.
    """

    def __init__(self, caminho_cfg: str):
        """
        Inicializa a conexão com o Supabase.

        Args:
            caminho_cfg (str): Caminho para o arquivo JSON contendo URL e KEY da API.
        """
        try:
            with open(caminho_cfg) as f:
                cfg = json.load(f)
            # Suporta formato lista [url, key] ou dict {"url":..., "key":...}
            if isinstance(cfg, list):
                self.supabase: Client = create_client(cfg[0], cfg[1])
            else:
                self.supabase: Client = create_client(cfg["url"], cfg["key"])
        except Exception as e:
            raise ConnectionError(f"Erro ao conectar ao Supabase: {e}")

    def obter_detalhes_sessao(self, sessao_id: int) -> dict[str, Any]:
        """
        Retorna os metadados de uma sessão específica pelo ID.
        Útil para o modo relatório recuperar qual modelo foi usado.
        """
        try:
            res = self.supabase.table("p02_sessoes").select("*").eq("id", sessao_id).execute()
            return res.data[0] if res.data else {}
        except Exception as e:
            logger.log(f"Erro ao buscar detalhes da sessão {sessao_id}: {e}", level=LogLevel.ERROR)
            return {}

    def salvar_caracteres_tag(self, sessao_id: int, tag_ini: str, tag_fim: str):
        """
        Salva os caracteres de TAG selecionados para a sessão.
        Esses caracteres são únicos por sessão e garantem que não colidem com o corpus.
        """
        try:
            self.supabase.table("p02_sessoes").update({"tag_ini": tag_ini, "tag_fim": tag_fim}).eq(
                "id", sessao_id
            ).execute()
        except Exception as e:
            logger.log(f"Erro ao salvar caracteres de TAG: {e}", level=LogLevel.ERROR)

    def carregar_caracteres_tag(self, sessao_id: int) -> tuple[str | None, str | None]:
        """
        Carrega os caracteres de TAG da sessão.
        Retorna (None, None) se não existirem (sessão nova ou migração pendente).
        """
        try:
            res = self.supabase.table("p02_sessoes").select("tag_ini, tag_fim").eq("id", sessao_id).execute()
            if res.data and res.data[0].get("tag_ini"):
                return (res.data[0]["tag_ini"], res.data[0]["tag_fim"])
        except Exception as e:
            logger.log(f"Erro ao carregar caracteres de TAG: {e}", level=LogLevel.ERROR)
        return (None, None)

    def obter_sessao(self, modelo_llm: str) -> int:
        """
        Identifica a sessão atual baseada no 'NOME_SESSAO' do config e no 'modelo_llm'.
        O nome no banco é salvo como "NomeSessao [Modelo]".
        Se existir, retorna o ID e atualiza o timestamp.
        Se não, retorna -1

        Returns:
            int: ID numérico da sessão no banco de dados ou -1 caso inexistente.
        """
        # Constrói o nome composto único
        nome_composto = f"{config.NOME_SESSAO} [{modelo_llm}]"

        try:
            # Busca sessão existente pelo nome composto
            res = self.supabase.table("p02_sessoes").select("id").eq("nome_identificador", nome_composto).execute()

            if not res.data:
                # Sessão não existe
                return -1

            # Atualiza 'last_seen' e retorna ID
            agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sessao_id = res.data[0]["id"]
            self.supabase.table("p02_sessoes").update({"ultima_atualizacao": agora}).eq("id", sessao_id).execute()
            return sessao_id

        except Exception as e:
            logger.log(f"Erro na gestão de sessão: {e}", level=LogLevel.ERROR)
            raise

    def obter_ou_criar_sessao(self, modelo_llm: str) -> int:
        """
        Identifica a sessão atual baseada no 'NOME_SESSAO' do config e 'modelo_llm'.
        Se existir, retorna o ID e atualiza o timestamp.
        Se não, cria uma nova entrada na tabela p02_sessoes.

        Returns:
            int: ID numérico da sessão no banco de dados.
        """
        try:
            sessao_id = self.obter_sessao(modelo_llm)
            if sessao_id > 0:
                return sessao_id

            # Sessão nova: Cria registro com NOME COMPOSTO
            nome_composto = f"{config.NOME_SESSAO} [{modelo_llm}]"
            agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            dados = {
                "nome_identificador": nome_composto,
                "modelo_llm": modelo_llm,
                "temperatura": config.LLM_TEMPERATURA,
                "hardware_info": config.HARDWARE_INFO,
                "config_ngram_min": config.NGRAM_MIN,
                "config_ngram_max": config.NGRAM_MAX,
                "status": "iniciado",
                "data_inicio": agora,
                "ultima_atualizacao": agora,
            }
            res = self.supabase.table("p02_sessoes").insert(dados).execute()
            return res.data[0]["id"]
        except Exception as e:
            logger.log(f"Erro na gestão de sessão: {e}", level=LogLevel.ERROR)
            raise

    def carregar_conhecimento_sessao(self, sessao_id: int) -> tuple[dict, dict, dict, dict, dict, dict, int, int]:
        """
        Carrega todo o 'conhecimento' (termos identificados) associado a uma sessão.

        Isso permite retomar o processamento de onde parou (Resume Capability) e
        reutilizar o aprendizado (ex: um nome identificado na linha 1 será anonimizado
        automaticamente na linha 1000 sem gastar tokens LLM).

        Args:
            sessao_id (int): ID da sessão a carregar.

        Returns:
            Tuple: (mapas de conhecimento..., ultimo_index_linha, ultimo_tamanho_ngram)
        """
        logger.log(f"Carregando conhecimento da sessão {sessao_id} do banco...", level=LogLevel.NORMAL)

        m_nomes, m_ends, m_outros, m_boilers, m_revs, m_unknowns = {}, {}, {}, {}, {}, {}

        # Paginação manual para contornar limites de retorno da API do Supabase (geralmente 1000 linhas)
        inicio = 0
        passo = 1000
        while True:
            fim = inicio + passo - 1
            logger.log(f" -> Buscando registros {inicio} a {fim}...", level=LogLevel.VERBOSE, end="\r")

            res = (
                self.supabase.table("p02_conhecimento")
                .select("tipo, texto_original, id_local_sessao")
                .eq("sessao_id", sessao_id)
                .order("id")
                .range(inicio, fim)
                .execute()
            )

            registros = res.data
            if not registros:
                break

            # Popula os mapas locais
            for item in registros:
                t, txt, idx = item["tipo"], item["texto_original"], item["id_local_sessao"]
                if t == "nome":
                    m_nomes[txt] = idx
                elif t == "endereco":
                    m_ends[txt] = idx
                elif t == "info":
                    m_outros[txt] = idx
                elif t == "boilerplate":
                    m_boilers[txt] = idx
                elif t == "revisao":
                    m_revs[txt] = idx
                elif t == "unknown":
                    m_unknowns[txt] = idx

            if len(registros) < passo:
                break  # Fim dos dados

            inicio += passo

        logger.log(f"\nCarga completa. {len(m_boilers)} boilerplates carregados.", level=LogLevel.NORMAL)

        # 1. Busca último índice processado (Fase Linha a Linha)
        res_log = (
            self.supabase.table("p02_logs_execucao")
            .select("evolucao_index")
            .eq("sessao_id", sessao_id)
            .eq("fase_atual", "linha_a_linha")
            .order("evolucao_index", desc=True)
            .limit(1)
            .execute()
        )

        ultimo_index = res_log.data[0]["evolucao_index"] if res_log.data else -1

        # 2. Busca último tamanho de N-Gram processado (Fase N-Gram)
        # Queremos o MENOR tamanho já registrado, pois o loop é decrescente (MAX -> MIN)
        res_ngram = (
            self.supabase.table("p02_logs_execucao")
            .select("tamanho_ngram")
            .eq("sessao_id", sessao_id)
            .eq("fase_atual", "ngram")
            .order("tamanho_ngram", desc=False)
            .limit(1)
            .execute()
        )

        # Se não tiver registro, assume que não começou (retorna NGRAM_MAX + 1 para o loop começar do MAX)
        ultimo_ngram = res_ngram.data[0]["tamanho_ngram"] if res_ngram.data else config.NGRAM_MAX + 1

        return (m_nomes, m_ends, m_outros, m_boilers, m_revs, m_unknowns, ultimo_index, ultimo_ngram)

    def salvar_conhecimento_lote(self, sessao_id: int, tipo: str, itens: dict[str, int]):
        """
        Salva ou atualiza um lote de termos conhecidos no banco.
        Usa 'upsert' para garantir que não haja duplicatas.

        Args:
            sessao_id (int): ID da sessão.
            tipo (str): Categoria do dado ('nome', 'boilerplate', etc).
            itens (Dict[str, int]): Dicionário { 'texto_original': id_numerico }.
        """
        if not itens:
            return
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepara payload
        dados = [
            {"sessao_id": sessao_id, "tipo": tipo, "texto_original": k, "id_local_sessao": v, "created_at": agora}
            for k, v in itens.items()
        ]

        try:
            self.supabase.table("p02_conhecimento").upsert(
                dados, on_conflict="sessao_id, tipo, texto_original"
            ).execute()
        except Exception as e:
            logger.log(f"Erro ao salvar conhecimento ({tipo}): {e}", level=LogLevel.ERROR)

    def arquivar_conhecimento_lote(self, sessao_id: int, tipo_antigo: str, tipo_novo: str, lista_termos: list[str]):
        """
        Altera o tipo de termos específicos no banco de dados (arquivamento).
        Útil para manter histórico de revisões resolvidas sem que sejam re-auditadas.
        """
        if not lista_termos:
            return
        try:
            self.supabase.table("p02_conhecimento").update({"tipo": tipo_novo}).eq("sessao_id", sessao_id).eq(
                "tipo", tipo_antigo
            ).in_("texto_original", lista_termos).execute()
        except Exception as e:
            logger.log(f"Erro ao arquivar conhecimento ({tipo_antigo} -> {tipo_novo}): {e}", level=LogLevel.ERROR)

    def atualizar_status_sessao(self, sessao_id: int, status: str):
        """Atualiza a coluna 'status' da sessão na tabela p02_sessoes."""
        try:
            # Formato curto para caber em VARCHAR(20): "2026-02-08 19:55:37"
            agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.supabase.table("p02_sessoes").update({"status": status, "ultima_atualizacao": agora}).eq(
                "id", sessao_id
            ).execute()
        except Exception as e:
            logger.log(f"Erro ao atualizar status da sessão {sessao_id}: {e}", level=LogLevel.ERROR)

    def salvar_log_execucao(
        self,
        sessao_id: int,
        evolucao_index: int,
        tempo_segundos: float,
        fase_atual: str = "linha_a_linha",
        tamanho_ngram: int | None = None,
        houve_duvida: bool = False,
        qtde_piis_novos: int = 0,
        texto_original: str | None = None,
        texto_pre_llm: str | None = None,
        llm_raw_response: str | None = None,
        houve_falha: bool = False,
        status_erro: str | None = None,
        prompt_eval_count: int | None = None,
        eval_count: int | None = None,
        eval_tokens_per_second: float | None = None,
        total_duration_ollama_sec: float | None = None,
        load_duration_sec: float | None = None,
        prompt_eval_duration_sec: float | None = None,
        eval_duration_sec: float | None = None,
        done_reason: str | None = None,
        qtde_tentativas: int = 1,
        chars_reducao_boilerplate: int | None = None,
        raciocinio: str | None = None,
        resposta_bruta: str | None = None,
    ):
        """
        Registra métricas de desempenho e auditoria de uma execução.

        tokens_antes/tokens_depois foram removidos da assinatura (eram estimativas tiktoken/GPT-4o).
        Os tokens reais do modelo local ficam em prompt_eval_count e eval_count (API Ollama).

        Args:
            sessao_id: ID da sessão.
            evolucao_index: Índice da linha do dataset processada (0 para N-Grams).
            tempo_segundos: Duração total do processamento da linha/ngram.
            fase_atual: 'ngram' ou 'linha_a_linha'.
            tamanho_ngram: Tamanho do N-Gram (apenas fase ngram).
            houve_duvida: LLM retornou duvidas_pendentes.
            qtde_piis_novos: Novos termos aprendidos nesta iteração.
            texto_original: Texto cru (auditoria).
            texto_pre_llm: Prompt real enviado ao LLM (auditoria).
            llm_raw_response: JSON completo do retorno do ClienteOllama (sem campos de controle).
            houve_falha: Timeout ou overflow de contexto.
            status_erro: Código do erro ("erro_stop_reason", "timeout", "erro", ou None).
            prompt_eval_count: Tokens reais do prompt (API Ollama).
            eval_count: Tokens reais gerados (API Ollama).
            eval_tokens_per_second: Throughput de geração.
            total_duration_ollama_sec: Tempo total medido pelo Ollama.
            load_duration_sec: Tempo de carregamento do modelo.
            prompt_eval_duration_sec: Tempo de prefill.
            eval_duration_sec: Tempo de decoding.
            done_reason: Motivo de parada ("stop", "length", etc.).
            qtde_tentativas: Número de tentativas na fase linha_a_linha (1 ou 2).
            chars_reducao_boilerplate: Chars economizados por boilerplates no pré-LLM.
            raciocinio: CoT thinking do modelo (thinking tokens).
            resposta_bruta: Texto bruto retornado pelo LLM antes do parse JSON.
        """
        dados = {
            "sessao_id": sessao_id,
            "evolucao_index": evolucao_index,
            "fase_atual": fase_atual,
            "tamanho_ngram": tamanho_ngram,
            # tokens_antes/tokens_depois zerados — tiktoken descontinuado neste fluxo.
            # Usar prompt_eval_count / eval_count para contagens reais (API Ollama).
            "tokens_antes": 0,
            "tokens_depois": 0,
            "porcentagem_reducao": 0.0,
            "tempo_execucao_segundos": tempo_segundos,
            "qtde_piis_novos": qtde_piis_novos,
            "houve_duvida": houve_duvida,
            "houve_falha": houve_falha,
            "status_erro": status_erro,
            "texto_original": texto_original,
            "texto_pre_llm": texto_pre_llm,
            "llm_raw_response": llm_raw_response,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "eval_tokens_per_second": eval_tokens_per_second,
            "total_duration_ollama_sec": total_duration_ollama_sec,
            "load_duration_sec": load_duration_sec,
            "prompt_eval_duration_sec": prompt_eval_duration_sec,
            "eval_duration_sec": eval_duration_sec,
            "done_reason": done_reason,
            "qtde_tentativas": qtde_tentativas,
            "chars_reducao_boilerplate": chars_reducao_boilerplate,
            "raciocinio": raciocinio,
            "resposta_bruta": resposta_bruta,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            self.supabase.table("p02_logs_execucao").insert(dados).execute()
        except Exception as e:
            logger.log(f"Erro ao salvar log: {e}", level=LogLevel.ERROR)

    # ============================================================
    # MÉTODOS BRKGA (operam SOMENTE em tabelas p02_brkga_*)
    # ============================================================

    def obter_ou_criar_sessao_brkga(self, nome_identificador: str, modelo_llm: str, config_brkga: dict) -> int:
        """
        Obtém sessão BRKGA existente por nome_identificador, ou cria nova.
        Segue o mesmo padrão de `obter_ou_criar_sessao` do projeto principal.

        Args:
            nome_identificador: Chave da sessão (ex: "alfa_8 [qwen3.5:9b]").
            modelo_llm: Nome do modelo LLM utilizado.
            config_brkga: Dict com n_genes, n_repeticoes, indices_amostra, etc.

        Returns:
            int: ID da sessão BRKGA.
        """
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # Busca sessão existente pelo nome_identificador
            res = (
                self.supabase.table("p02_brkga_sessoes")
                .select("id")
                .eq("nome_identificador", nome_identificador)
                .limit(1)
                .execute()
            )

            if res.data:
                # Sessão existe → atualiza timestamp e retorna ID
                sessao_id = res.data[0]["id"]
                self.supabase.table("p02_brkga_sessoes").update({"created_at": agora}).eq("id", sessao_id).execute()
                return sessao_id

            # Sessão nova → cria registro
            dados = {
                "nome_identificador": nome_identificador,
                "modelo_llm": modelo_llm,
                "llm_num_ctx": config.LLM_NUM_CTX,
                "llm_num_predict": config.LLM_NUM_PREDICT,
                "n_genes": config_brkga["n_genes"],
                "n_repeticoes": config_brkga["n_repeticoes"],
                "indices_amostra": json.dumps(config_brkga["indices_amostra"]),
                "alfa_penalizacao": config_brkga["alfa_penalizacao"],
                "pop_size": config_brkga.get("pop_size"),
                "max_geracoes": config_brkga.get("max_geracoes"),
                "seed_brkga": config_brkga.get("seed_brkga"),
                "pe": config_brkga.get("pe"),
                "pm": config_brkga.get("pm"),
                "rhoe": config_brkga.get("rhoe"),
                "k_populacoes": config_brkga.get("k_populacoes"),
                "created_at": agora,
            }
            res = self.supabase.table("p02_brkga_sessoes").insert(dados).execute()
            return res.data[0]["id"]
        except Exception as e:
            logger.log(f"Erro ao obter/criar sessão BRKGA: {e}", level=LogLevel.ERROR)
            raise

    def buscar_cache_brkga(self, sessao_brkga_id: int, params: dict) -> tuple[float, int] | None:
        """
        Busca fitness cacheado para um cromossomo já avaliado.

        Args:
            sessao_brkga_id: ID da sessão BRKGA.
            params: Dict dos hiperparâmetros decodificados.

        Returns:
            Tuple (fitness, resultado_id), ou None se não encontrado.
            O resultado_id é necessário para salvar o log de cache hit.
        """
        try:
            res = (
                self.supabase.table("p02_brkga_resultados")
                .select("id, fitness")
                .eq("sessao_brkga_id", sessao_brkga_id)
                .eq("temperature", params["temperature"])
                .eq("top_k", params["top_k"])
                .eq("top_p", params["top_p"])
                .eq("repeat_penalty", params["repeat_penalty"])
                .eq("repeat_last_n", params["repeat_last_n"])
                .limit(1)
                .execute()
            )
            if res.data and res.data[0]["fitness"] is not None:
                return (res.data[0]["fitness"], res.data[0]["id"])
            return None

        except Exception as e:
            logger.log(f"Erro ao buscar cache BRKGA: {e}", level=LogLevel.ERROR)
            return None

    def salvar_resultado_brkga(self, sessao_brkga_id: int, dados: dict) -> int:
        """
        Salva resultado de avaliação de um cromossomo BRKGA.
        Usa upsert para idempotência (chave: sessao_brkga_id + campos dos params).

        Args:
            sessao_brkga_id: ID da sessão BRKGA.
            dados: Dict com cromossomo, params_decodificados, métricas.

        Returns:
            int: ID do registro persistido em p02_brkga_resultados.
        """
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        params = dados["params_decodificados"]
        registro = {
            "sessao_brkga_id": sessao_brkga_id,
            "temperature": params["temperature"],
            "top_k": params["top_k"],
            "top_p": params["top_p"],
            "repeat_penalty": params["repeat_penalty"],
            "repeat_last_n": params["repeat_last_n"],
            "cromossomo": json.dumps(dados["cromossomo"]),
            "params_decodificados": json.dumps(params),
            "recall_medio": dados.get("recall_medio"),
            "precision_medio": dados.get("precision_medio"),
            "f1_medio": dados.get("f1_medio"),
            "f2_medio": dados.get("f2_medio"),
            "desvio_padrao": dados.get("desvio_padrao"),
            "fitness": dados.get("fitness"),
            "contagem_falhas": dados.get("contagem_falhas", 0),
            "contagem_duvidas": dados.get("contagem_duvidas", 0),
            "created_at": agora,
        }
        try:
            res = self.supabase.table("p02_brkga_resultados").upsert(
                registro, on_conflict="sessao_brkga_id, temperature, top_k, top_p, repeat_penalty, repeat_last_n"
            ).execute()
            return res.data[0]["id"]
        except Exception as e:
            logger.log(f"Erro ao salvar resultado BRKGA: {e}", level=LogLevel.ERROR)
            return -1

    def salvar_logs_llm_brkga(self, resultado_id: int, logs: list[dict]) -> None:
        """
        Salva logs de auditoria LLM de cada repetição de um cromossomo avaliado.
        Chamado individualmente após cada repetição (comportamento eager).

        Args:
            resultado_id: ID do registro em p02_brkga_resultados.
            logs: Lista de dicts, um por repetição.
                  Campos: repeticao, texto_pre_llm, raciocinio, resposta_bruta,
                  llm_raw_response, status_erro, houve_falha, metricas_ollama.
        """
        if not logs:
            return
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dados = [
            {
                "resultado_id":    resultado_id,
                "repeticao":       log.get("repeticao"),
                "texto_pre_llm":   log.get("texto_pre_llm"),
                "raciocinio":      log.get("raciocinio"),
                "resposta_bruta":  log.get("resposta_bruta"),
                "llm_raw_response": log.get("llm_raw_response"),
                "status_erro":     log.get("status_erro"),
                "houve_falha":     log.get("houve_falha", False),
                
                "done_reason":              log.get("metricas_ollama", {}).get("done_reason"),
                "total_duration_sec":       log.get("metricas_ollama", {}).get("total_duration_sec", 0.0),
                "load_duration_sec":        log.get("metricas_ollama", {}).get("load_duration_sec", 0.0),
                "prompt_eval_count":        log.get("metricas_ollama", {}).get("prompt_eval_count", 0),
                "prompt_eval_duration_sec": log.get("metricas_ollama", {}).get("prompt_eval_duration_sec", 0.0),
                "eval_count":               log.get("metricas_ollama", {}).get("eval_count", 0),
                "eval_duration_sec":        log.get("metricas_ollama", {}).get("eval_duration_sec", 0.0),
                "eval_tokens_per_second":   log.get("metricas_ollama", {}).get("eval_tokens_per_second", 0.0),
                
                "created_at":      agora,
            }
            for log in logs
        ]
        try:
            self.supabase.table("p02_brkga_logs_llm").insert(dados).execute()
        except Exception as e:
            logger.log(f"Erro ao salvar logs LLM BRKGA: {e}", level=LogLevel.ERROR)

    def incrementar_cache_hit_brkga(self, resultado_id: int) -> None:
        """
        Incrementa atomicamente o contador cache_hits do cromossomo já avaliado.
        Chamado toda vez que o fitness é servido do cache (sem chamar o LLM).

        Args:
            resultado_id: ID do registro em p02_brkga_resultados.
        """
        try:
            self.supabase.rpc(
                "incrementar_cache_hits_brkga",
                {"p_resultado_id": resultado_id},
            ).execute()
        except Exception:
            # Fallback: UPDATE via client (sem RPC)
            try:
                res = (
                    self.supabase.table("p02_brkga_resultados")
                    .select("cache_hits")
                    .eq("id", resultado_id)
                    .single()
                    .execute()
                )
                atual = res.data["cache_hits"] if res.data else 0
                self.supabase.table("p02_brkga_resultados").update(
                    {"cache_hits": atual + 1}
                ).eq("id", resultado_id).execute()
            except Exception as e:
                logger.log(f"Erro ao incrementar cache_hits BRKGA {resultado_id}: {e}", level=LogLevel.ERROR)

    def criar_stub_resultado_brkga(self, sessao_brkga_id: int, params: dict, cromossomo: list) -> int:
        """
        Cria um registro stub em p02_brkga_resultados antes de iniciar as repetições.
        As métricas finais são preenchidas depois via atualizar_resultado_brkga.
        Se o registro já existir (upsert por chave de cache), retorna o ID existente.

        Args:
            sessao_brkga_id: ID da sessão BRKGA.
            params: Dict dos hiperparâmetros decodificados.
            cromossomo: Lista de genes do cromossomo.

        Returns:
            int: ID do registro criado/existente.
        """
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        registro = {
            "sessao_brkga_id": sessao_brkga_id,
            "temperature":    params["temperature"],
            "top_k":          params["top_k"],
            "top_p":          params["top_p"],
            "repeat_penalty": params["repeat_penalty"],
            "repeat_last_n":  params["repeat_last_n"],
            "cromossomo":     json.dumps(cromossomo),
            "params_decodificados": json.dumps(params),
            "created_at":     agora,
        }
        try:
            res = self.supabase.table("p02_brkga_resultados").upsert(
                registro,
                on_conflict="sessao_brkga_id, temperature, top_k, top_p, repeat_penalty, repeat_last_n",
            ).execute()
            return res.data[0]["id"]
        except Exception as e:
            logger.log(f"Erro ao criar stub resultado BRKGA: {e}", level=LogLevel.ERROR)
            return -1

    def atualizar_resultado_brkga(self, resultado_id: int, dados: dict) -> None:
        """
        Atualiza o registro de resultado BRKGA com as métricas finais após todas as repetições.

        Args:
            resultado_id: ID do registro em p02_brkga_resultados.
            dados: Dict com recall_medio, precision_medio, f1_medio, f2_medio, desvio_padrao,
                   fitness, contagem_falhas, contagem_duvidas.
        """
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        campos = {
            "recall_medio":       dados.get("recall_medio"),
            "precision_medio":    dados.get("precision_medio"),
            "f1_medio":           dados.get("f1_medio"),
            "f2_medio":           dados.get("f2_medio"),
            "desvio_padrao":      dados.get("desvio_padrao"),
            "fitness":            dados.get("fitness"),
            "contagem_falhas":    dados.get("contagem_falhas", 0),
            "contagem_duvidas":   dados.get("contagem_duvidas", 0),
            "ultima_atualizacao": agora,
        }
        try:
            self.supabase.table("p02_brkga_resultados").update(campos).eq("id", resultado_id).execute()
        except Exception as e:
            logger.log(f"Erro ao atualizar resultado BRKGA {resultado_id}: {e}", level=LogLevel.ERROR)

    def obter_sessao_brkga(self, sessao_brkga_id: int) -> dict:
        """
        Retorna os metadados de uma sessão BRKGA pelo ID.

        Args:
            sessao_brkga_id: ID da sessão BRKGA.

        Returns:
            Dict com campos da sessão, ou {} se não encontrada.
        """
        try:
            res = (
                self.supabase.table("p02_brkga_sessoes")
                .select("*")
                .eq("id", sessao_brkga_id)
                .single()
                .execute()
            )
            return res.data if res.data else {}
        except Exception as e:
            logger.log(f"Erro ao buscar sessão BRKGA {sessao_brkga_id}: {e}", level=LogLevel.ERROR)
            return {}

    def carregar_logs_llm_brkga_por_sessao(self, sessao_brkga_id: int) -> list[dict]:
        """
        Carrega todos os logs LLM de uma sessão BRKGA, enriquecidos com parâmetros do resultado pai.

        Dois passos (API REST sem JOINs):
        1. Busca resultados da sessão com seus hiperparâmetros.
        2. Busca logs de cada resultado, paginando se necessário.

        Cada log retornado contém o campo extra '_params' com
        {temperature, top_k, top_p, repeat_penalty, repeat_last_n} do resultado pai.

        Args:
            sessao_brkga_id: ID da sessão BRKGA.

        Returns:
            Lista de dicts (logs enriquecidos), ordenada por id.
        """
        try:
            # Passo 1: busca resultados e seus parâmetros
            res_resultados = (
                self.supabase.table("p02_brkga_resultados")
                .select("id, temperature, top_k, top_p, repeat_penalty, repeat_last_n")
                .eq("sessao_brkga_id", sessao_brkga_id)
                .execute()
            )
            if not res_resultados.data:
                return []

            # Mapeia resultado_id → params
            mapa_params = {}
            resultado_ids = []
            for r in res_resultados.data:
                resultado_ids.append(r["id"])
                mapa_params[r["id"]] = {
                    "temperature": r["temperature"],
                    "top_k": r["top_k"],
                    "top_p": r["top_p"],
                    "repeat_penalty": r["repeat_penalty"],
                    "repeat_last_n": r["repeat_last_n"],
                }

            # Passo 2: busca logs com paginação cursor-based (seguro contra inserções concorrentes)
            todos_logs = []
            ultimo_id = 0
            passo = 1000
            while True:
                res_logs = (
                    self.supabase.table("p02_brkga_logs_llm")
                    .select("*")
                    .in_("resultado_id", resultado_ids)
                    .order("id")
                    .gt("id", ultimo_id)
                    .limit(passo)
                    .execute()
                )
                if not res_logs.data:
                    break
                todos_logs.extend(res_logs.data)
                ultimo_id = res_logs.data[-1]["id"]
                if len(res_logs.data) < passo:
                    break

            # Enriquece cada log com params do resultado pai
            for log in todos_logs:
                log["_params"] = mapa_params.get(log["resultado_id"], {})

            return todos_logs

        except Exception as e:
            logger.log(f"Erro ao carregar logs LLM BRKGA da sessão {sessao_brkga_id}: {e}", level=LogLevel.ERROR)
            return []

    def salvar_simulacao_injection(self, log_llm_id: int, dados: dict) -> int:
        """
        Insere resultado de simulação de injection na tabela p02_brkga_simular_injection.

        Args:
            log_llm_id: ID do log original em p02_brkga_logs_llm.
            dados: Dict com campos da simulação (raciocinio, resposta_bruta, métricas, etc.).

        Returns:
            int: ID do registro criado, ou -1 em caso de erro.
        """
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metricas = dados.get("metricas_ollama", {})
        registro = {
            "log_llm_id":              log_llm_id,
            "raciocinio":              dados.get("raciocinio"),
            "resposta_bruta":          dados.get("resposta_bruta"),
            "llm_raw_response":        dados.get("llm_raw_response"),
            "status_erro":             dados.get("status_erro"),
            "done_reason":             metricas.get("done_reason") if metricas else dados.get("done_reason"),
            "total_duration_sec":      metricas.get("total_duration_sec", 0.0) if metricas else dados.get("total_duration_sec"),
            "load_duration_sec":       metricas.get("load_duration_sec", 0.0) if metricas else dados.get("load_duration_sec"),
            "prompt_eval_count":       metricas.get("prompt_eval_count", 0) if metricas else dados.get("prompt_eval_count"),
            "prompt_eval_duration_sec": metricas.get("prompt_eval_duration_sec", 0.0) if metricas else dados.get("prompt_eval_duration_sec"),
            "eval_count":              metricas.get("eval_count", 0) if metricas else dados.get("eval_count"),
            "eval_duration_sec":       metricas.get("eval_duration_sec", 0.0) if metricas else dados.get("eval_duration_sec"),
            "eval_tokens_per_second":  metricas.get("eval_tokens_per_second", 0.0) if metricas else dados.get("eval_tokens_per_second"),
            "recall":                  dados.get("recall"),
            "precision":               dados.get("precision"),
            "f1":                      dados.get("f1"),
            "f2":                      dados.get("f2"),
            "houve_falha":             dados.get("houve_falha", False),
            "prompt_enviado":          dados.get("prompt_enviado"),
            # Colunas de análise do loop (DetectorLoop offline)
            "loop_detectado":          dados.get("loop_detectado"),
            "loop_tipo":               dados.get("loop_tipo"),
            "posicao_corte":           dados.get("posicao_corte"),
            "tamanho_raciocinio_original":  dados.get("tamanho_raciocinio_original"),
            "tamanho_raciocinio_truncado":  dados.get("tamanho_raciocinio_truncado"),
            "ratio_raciocinio_aproveitado": dados.get("ratio_raciocinio_aproveitado"),
            "created_at":              agora,
        }
        try:
            res = self.supabase.table("p02_brkga_simular_injection").insert(registro).execute()
            return res.data[0]["id"]
        except Exception as e:
            logger.log(f"Erro ao salvar simulação injection (log {log_llm_id}): {e}", level=LogLevel.ERROR)
            return -1

    def verificar_simulacao_existente(self, log_llm_id: int) -> bool:
        """
        Verifica se já existe simulação para o log_llm_id dado (resume capability).

        Args:
            log_llm_id: ID do log original em p02_brkga_logs_llm.

        Returns:
            True se já existe, False caso contrário.
        """
        try:
            res = (
                self.supabase.table("p02_brkga_simular_injection")
                .select("id")
                .eq("log_llm_id", log_llm_id)
                .limit(1)
                .execute()
            )
            return bool(res.data)
        except Exception as e:
            logger.log(f"Erro ao verificar simulação existente (log {log_llm_id}): {e}", level=LogLevel.ERROR)
            return False

    def salvar_comparacao_deteccoes(self, log_llm_id: int, sessao_brkga_id: int, dados: dict) -> int:
        """
        Insere resultado de comparação de detectores na tabela p02_brkga_comparar_deteccoes_loop.

        Args:
            log_llm_id: ID do log original em p02_brkga_logs_llm.
            sessao_brkga_id: ID da sessão BRKGA.
            dados: Dict com campos da comparação.

        Returns:
            int: ID do registro criado, ou -1 em caso de erro.
        """
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        registro = {
            "log_llm_id":              log_llm_id,
            "sessao_brkga_id":         sessao_brkga_id,
            "tamanho_raciocinio":      dados.get("tamanho_raciocinio"),
            "artigo_detectou":         dados.get("artigo_detectou", False),
            "artigo_tipo":             dados.get("artigo_tipo"),
            "artigo_posicao":          dados.get("artigo_posicao"),
            "artigo_ratio_aproveitado": dados.get("artigo_ratio_aproveitado"),
            "det4_detectou":           dados.get("det4_detectou", False),
            "det4_tipo":               dados.get("det4_tipo"),
            "det4_posicao":            dados.get("det4_posicao"),
            "det4_ratio_aproveitado":  dados.get("det4_ratio_aproveitado"),
            "ambos_detectaram":        dados.get("ambos_detectaram", False),
            "apenas_artigo":           dados.get("apenas_artigo", False),
            "apenas_det4":             dados.get("apenas_det4", False),
            "nenhum_detectou":         dados.get("nenhum_detectou", False),
            "diferenca_posicao":       dados.get("diferenca_posicao"),
            "detector_mais_preciso":   dados.get("detector_mais_preciso"),
            "created_at":              agora,
        }
        try:
            res = self.supabase.table("p02_brkga_comparar_deteccoes_loop").insert(registro).execute()
            return res.data[0]["id"]
        except Exception as e:
            logger.log(f"Erro ao salvar comparação detecções (log {log_llm_id}): {e}", level=LogLevel.ERROR)
            return -1

    def verificar_comparacao_existente(self, log_llm_id: int) -> bool:
        """
        Verifica se já existe comparação para o log_llm_id dado (resume capability).

        Args:
            log_llm_id: ID do log original em p02_brkga_logs_llm.

        Returns:
            True se já existe, False caso contrário.
        """
        try:
            res = (
                self.supabase.table("p02_brkga_comparar_deteccoes_loop")
                .select("id")
                .eq("log_llm_id", log_llm_id)
                .limit(1)
                .execute()
            )
            return bool(res.data)
        except Exception as e:
            logger.log(f"Erro ao verificar comparação existente (log {log_llm_id}): {e}", level=LogLevel.ERROR)
            return False

    # ============================================================
    # MÉTODOS BRANCH-AND-BOUND (COMBINAÇÕES)
    # ============================================================

    def carregar_ids_combos_processados(self, modo: str, grupo: str) -> dict[str, float]:
        """
        Retorna quais combos (via label_combo) já foram executados neste modo e grupo,
        acompanhado do seu F2_score anterior. Se foi podado, F2_score = -1.
        """
        try:
            res = (self.supabase.table("p02_combinacoes")
                   .select("label_combo, f2_score, podado")
                   .in_("modo", [modo, "unico"])   # singles (k=1) sempre reutilizáveis
                   .eq("grupo", grupo)
                   .execute())
            cache = {}
            for row in res.data:
                cache[row["label_combo"]] = row["f2_score"] if not row["podado"] else -1.0
            return cache
        except Exception as e:
            logger.log(f"Erro ao carregar pré-existentes: {e}", level=LogLevel.ERROR)
            return {}

    def salvar_combinacao(
        self,
        sessao_ids: list[int],
        modo: str,
        nivel_k: int,
        label_combo: str,
        nome_arquivo: str | None,
        metricas: dict,
        podado: bool = False,
        grupo: str = "default",
    ) -> int:
        """
        Insere/Atualiza o resultado da combinação no banco (Idempotente).
        
        Observação sobre Poda e Execução Forçada:
        - Se a combinação foi podada pelo algoritmo: entra com matriz vazia e podado=True.
        - Se for forçada via CLI e o score for melhor, o upsert substitui os dados.
        """
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        registro = {
            "modo": modo,
            "grupo": grupo,
            "nivel_k": nivel_k,
            "label_combo": label_combo,
            "nome_arquivo": nome_arquivo,
            "precision": metricas.get("precision"),
            "recall": metricas.get("recall"),
            "f1_score": metricas.get("f1_score"),
            "f2_score": metricas.get("f2_score"),
            "podado": podado,
            "created_at": agora,
        }

        try:
            # Upsert na entidade principal
            res = self.supabase.table("p02_combinacoes").upsert(
                registro, on_conflict="label_combo, grupo"
            ).execute()
            
            combinacao_id = res.data[0]["id"]

            # Upsert na tabela pivot
            if sessao_ids:
                pivots = [{"combinacao_id": combinacao_id, "sessao_id": s} for s in sessao_ids]
                self.supabase.table("p02_combinacoes_sessoes").upsert(
                    pivots, on_conflict="combinacao_id, sessao_id"
                ).execute()

            return combinacao_id
        except Exception as e:
            logger.log(f"Erro ao salvar combinação: {e}", level=LogLevel.ERROR)
            return -1

    def buscar_todas_as_combinacoes_validas(self) -> list[dict]:
        """Busca todas as combinações ativas no banco para plotagem O(1) de gráficos (substituindo JSON file scan)."""
        if not self.supabase:
            return []
        try:
            r = self.supabase.table("p02_combinacoes").select("*").eq("podado", False).execute()
            return r.data
        except Exception as e:
            logger.log(f"Erro ao buscar combinações válidas no BD: {e}", level=LogLevel.ERROR)
            return []
