"""
Módulo de Cliente LLM (Ollama).

Este módulo fornece uma abstração para comunicação com a API do Ollama,
facilitando a execução de prompts, gerenciamento de modelos locais e
o tratamento de respostas em streaming com suporte a Chain of Thought (CoT).
"""

import re
import time
from collections.abc import Generator
from datetime import datetime
from typing import Any

import httpx
import ollama

import logger
from a01_platform import config_router as config
from detector_loop import DetectorLoop
from logger import LogLevel


class ClienteOllama:
    """
    Cliente wrapper para comunicação com o servidor Ollama.

    Gerencia a conexão, verificação de modelos e execução de chat/completions.
    Implementa suporte a streaming de respostas e separa o raciocínio (pensamento)
    do conteúdo final, útil para modelos que utilizam tokens de 'thinking'.
    """

    def __init__(self, url: str, timeout: float = 3600.0):
        """
        Inicializa o cliente Ollama.

        Args:
            url (str): URL do servidor Ollama (ex: "http://localhost:11434").
            timeout (float): Tempo máximo de espera da conexão/leitura em segundos.
        """
        self.cliente = ollama.Client(host=url, timeout=timeout)

    def modelo_existe(self, modelo: str) -> bool:
        """Verifica se o modelo especificado está disponível no servidor local."""
        try:
            self.cliente.show(modelo)
            return True
        except Exception:
            return False

    def listar_modelos(self) -> list[str]:
        """Retorna uma lista de nomes de modelos disponíveis formatados com prefixo 'ollama:'."""
        modelos = self.cliente.list()
        return ["ollama:" + dict(x)["model"] for x in modelos["models"]]

    def executar_prompt(
        self,
        prompt: str,
        historico: list[dict[str, str]] | None = None,
        modelo: str = "gpt-oss:120b",
        temperatura: float = 0.1,
        system_prompt: str = "",
        num_ctx: int | None = None,
        num_predict: int | None = None,
        think: bool = True,
        options_extra: dict | None = None,
    ) -> Generator[dict[str, str], None, None]:
        """
        Envia um prompt para o modelo e gera a resposta via streaming.

        Esta função gerencia a separação entre tokens de pensamento e a resposta final,
        além de monitorar o status de encerramento da geração.

        Args:
            prompt (str): A mensagem do usuário.
            historico (list): Histórico de mensagens anteriores (chat).
            modelo (str): Nome do modelo a ser usado.
            temperatura (float): Criatividade do modelo (0.0 a 1.0).
            system_prompt (str): Instrução de sistema para alinhar o comportamento.

        Yields:
            dict: Dicionário contendo:
                - 'controle': Logs de status e timestamps.
                - 'raciocinio': Conteúdo do pensamento (Chain of Thought).
                - 'resposta': O conteúdo final da resposta. Retorna 'erro_stop_reason'
                  se a geração for interrompida por limites técnicos (ex: estouro de contexto)
                  em vez de uma parada natural.
        """

        # Correção: inicializa lista vazia dentro da função para evitar compartilhamento
        if historico is None:
            historico = []

        # Prepara o histórico com System Prompt se houver
        mensagens = []
        if system_prompt:
            mensagens.append({"role": "system", "content": system_prompt})

        mensagens.extend(historico)
        mensagens.append({"role": "user", "content": prompt})

        options = {
            "temperature": temperatura,
            "num_ctx": num_ctx if num_ctx is not None else config.LLM_NUM_CTX,  # Contexto dinâmico baseado no config
            "num_predict": num_predict
            if num_predict is not None
            else config.LLM_NUM_PREDICT,  # Limite dinâmico de predição
        }
        if options_extra:
            options.update(options_extra)

        metricas_zeradas = {
            "done_reason": None,
            "total_duration_sec": 0.0,
            "load_duration_sec": 0.0,
            "prompt_eval_count": 0,
            "prompt_eval_duration_sec": 0.0,
            "eval_count": 0,
            "eval_duration_sec": 0.0,
            "eval_tokens_per_second": 0.0,
        }
        retorno = {"controle": "", "raciocinio": "", "resposta": "", "metricas_ollama": metricas_zeradas}
        retorno["controle"] += f"#### {datetime.now()} - INICIALIZANDO... ####\n"

        # Máquina de estados para separar "Thinking" (Raciocínio) de "Content" (Resposta)
        # 1: Inicial/Iniciando, 2: Pensando (Tokens de CoT), 3: Respondendo (Conteúdo final)
        fase_atual = 1

        # --- ANTI-LOOP: Detecção de repetição via DetectorLoop ---
        # detector_content sempre ativo; detector_thinking apenas quando LLM_THINKING=True.
        # Nenhum depende de PROMPT_INJECTION, que controla apenas o retry com injection.
        config_loop = {
            "janela_chars": config.LOOP_JANELA_CHARS,
            "limiar_janelas": config.LOOP_LIMIAR_JANELAS,
            "rep_n_tamanho": config.LOOP_REP_N_TAMANHO,
            "rep_n_threshold": config.LOOP_REP_N_THRESHOLD,
            "rep_n_min_tokens": config.LOOP_REP_N_MIN_TOKENS,
            "rep_n_intervalo": config.LOOP_REP_N_INTERVALO,
            "rep_n_janela_tokens": config.LOOP_REP_N_JANELA_TOKENS,
            "cosine_janela_linhas": config.LOOP_COSINE_JANELA_LINHAS,
            "cosine_threshold": config.LOOP_COSINE_THRESHOLD,
            "cosine_min_linhas": config.LOOP_COSINE_MIN_LINHAS,
            "cosine_ratio_pares": config.LOOP_COSINE_RATIO_PARES,
            "bloco_limiar": config.LOOP_BLOCO_LIMIAR,
            "bloco_min_linhas": config.LOOP_BLOCO_MIN_LINHAS,
            "bloco_min_chars_linha": config.LOOP_BLOCO_MIN_CHARS_LINHA,
        }
        detector_thinking = DetectorLoop(config_loop) if config.LLM_THINKING else None
        detector_content = DetectorLoop(config_loop)

        try:
            stream = self.cliente.chat(model=modelo, messages=mensagens, options=options, stream=True, think=think)

            # Contadores para métricas parciais em caso de abort por loop detector.
            # O chunk 'done' (com eval_count real) nunca é lido quando abortamos cedo;
            # contamos os tokens gerados via chunks recebidos (1 chunk ≈ 1 token no Ollama).
            _eval_count = 0
            _eval_t_primeiro_token: float | None = None

            for part in stream:
                msg_part = part.get("message", {})
                # SDK 0.6+: SubscriptableBaseModel.get() usa getattr(), retorna None p/ campos ausentes.
                # Usar `or ""` garante string vazia em vez de None para evitar TypeError no +=.
                content  = msg_part.get("content")  or ""
                thinking = msg_part.get("thinking") or ""

                if content or thinking:
                    _eval_count += 1
                    if _eval_t_primeiro_token is None:
                        _eval_t_primeiro_token = time.time()

                # Detecta início da atividade do modelo
                if fase_atual == 1:
                    retorno["controle"] += f"\n#### {datetime.now()} - PENSANDO... ####\n"
                    fase_atual = 2

                # Lógica de Distribuição de Conteúdo
                if not retorno["resposta"] and thinking:
                    # Se houver tokens de pensamento e ainda não iniciamos a resposta final
                    retorno["raciocinio"] += thinking
                else:
                    # Transição de fase se detectarmos conteúdo final (content)
                    if fase_atual == 2 and content:
                        fase_atual = 3
                        retorno["controle"] += f"\n#### {datetime.now()} - RESPONDENDO ####\n"

                    retorno["resposta"] += content

                # --- ANTI-LOOP: Detecção de repetição no thinking (raciocínio) ---
                if thinking and detector_thinking:
                    resultado_loop = detector_thinking.alimentar(thinking)
                    if resultado_loop:
                        retorno["controle"] += (
                            f"\n⚠ LOOP DETECTADO no raciocínio ({resultado_loop['tipo']}: "
                            f"{resultado_loop['detalhes']}) - Abortando geração\n"
                        )
                        retorno["status_erro"] = "erro_stop_reason"
                        retorno["loop_tipo"] = f"thinking_{resultado_loop['tipo']}"
                        retorno["loop_posicao_inicio"] = detector_thinking.obter_posicao_inicio_loop()
                        _eval_dur = time.time() - _eval_t_primeiro_token if _eval_t_primeiro_token else 0.0
                        retorno["metricas_ollama"]["eval_count"] = _eval_count
                        retorno["metricas_ollama"]["eval_duration_sec"] = _eval_dur
                        if _eval_dur > 0:
                            retorno["metricas_ollama"]["eval_tokens_per_second"] = _eval_count / _eval_dur
                        yield retorno
                        return

                # --- ANTI-LOOP: Detecção de repetição no content (resposta) ---
                if content and detector_content:
                    resultado_loop = detector_content.alimentar(content)
                    if resultado_loop:
                        retorno["controle"] += (
                            f"\n⚠ LOOP DETECTADO na resposta ({resultado_loop['tipo']}: "
                            f"{resultado_loop['detalhes']}) - Abortando geração\n"
                        )
                        retorno["status_erro"] = "erro_stop_reason"
                        retorno["loop_tipo"] = f"content_{resultado_loop['tipo']}"
                        _eval_dur = time.time() - _eval_t_primeiro_token if _eval_t_primeiro_token else 0.0
                        retorno["metricas_ollama"]["eval_count"] = _eval_count
                        retorno["metricas_ollama"]["eval_duration_sec"] = _eval_dur
                        if _eval_dur > 0:
                            retorno["metricas_ollama"]["eval_tokens_per_second"] = _eval_count / _eval_dur
                        yield retorno
                        return

                # Monitoramento de Segurança: Verifica se o modelo parou por erro ou limite
                # Se 'done_reason' não for 'stop' (fim natural), sinaliza interrupção anômala.
                if part.get("done"):
                    # Fallback: modelos sem campo thinking dedicado embtem CoT em <think>...</think>
                    # no content acumulado. Separa ao final do stream (não por token, pois vêm fragmentados).
                    if not retorno["raciocinio"] and "<think>" in retorno["resposta"]:
                        m = re.search(r"<think>(.*?)</think>\s*(.*)", retorno["resposta"], re.DOTALL)
                        if m:
                            retorno["raciocinio"] = m.group(1).strip()
                            retorno["resposta"]   = m.group(2).strip()

                    done_reason = part.get("done_reason")
                    if done_reason not in ["stop", None]:
                        retorno["status_erro"] = "erro_stop_reason"

                    # Extrai métricas em Nanossegundos e converte para Segundos (Float)
                    eval_count = part.get("eval_count", 0)
                    eval_duration_sec = part.get("eval_duration", 0) / 1e9 if part.get("eval_duration") else 0.0

                    tokens_per_second = 0.0
                    if eval_duration_sec > 0:
                        tokens_per_second = eval_count / eval_duration_sec

                    retorno["metricas_ollama"] = {
                        "done_reason": done_reason,
                        "total_duration_sec": part.get("total_duration", 0) / 1e9 if part.get("total_duration") else 0.0,
                        "load_duration_sec": part.get("load_duration", 0) / 1e9 if part.get("load_duration") else 0.0,
                        "prompt_eval_count": part.get("prompt_eval_count", 0) if part.get("prompt_eval_count") else 0,
                        "prompt_eval_duration_sec": part.get("prompt_eval_duration", 0) / 1e9 if part.get("prompt_eval_duration") else 0.0,
                        "eval_count": eval_count if eval_count else 0,
                        "eval_duration_sec": eval_duration_sec,
                        "eval_tokens_per_second": tokens_per_second,
                    }

                # Entrega o estado atualizado do buffer para o chamador
                yield retorno

        except httpx.TimeoutException:
            retorno["controle"] += "\nTIMEOUT NA EXECUÇÃO: O LLM demorou muito para responder.\n"
            retorno["status_erro"] = "timeout"
            yield retorno
        except Exception as e:
            retorno["controle"] += f"\nERRO NA EXECUÇÃO: {str(e)}\n"
            retorno["status_erro"] = "erro"
            yield retorno

    def consultar_com_retry(
        self, prompt: str, modelo_atual: str, max_tentativas: int = 3, options_extra: dict | None = None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Envia prompt ao LLM com lógica de retentativa em caso de falha.

        Args:
            prompt: Texto a ser enviado.
            modelo_atual: Nome do modelo LLM a ser utilizado.
            max_tentativas: Número máximo de tentativas antes de desistir.
            options_extra: Parâmetros extras para repassar à API do Ollama (como top_k, top_p, repeat_penalty).

        Returns:
            Tuple (resultado_final, tentativas_log):
                resultado_final: Resposta estruturada do LLM (usada pela lógica de negócio).
                tentativas_log: Lista de dicts, um por tentativa LLM, para persistência por tentativa.
        """
        historico_raciocinio = ""
        tentativas_log: list[dict[str, Any]] = []

        for tentativa in range(1, max_tentativas + 1):
            if tentativa > 1:
                logger.log(f"Tentativa {tentativa}/{max_tentativas} no LLM...", level=LogLevel.WARNING)

            t_inicio_tentativa = time.time()
            modelo_nome = modelo_atual.replace("ollama:", "")

            prompt_atual = prompt
            think_flag = config.LLM_THINKING
            ctx_atual = config.LLM_NUM_CTX
            predict_atual = config.LLM_NUM_PREDICT

            # Se resgatou raciocínio da tentativa anterior por erro contextual, desliga o think e amplia o predict.
            # num_ctx permanece fixo (config.LLM_NUM_CTX) para evitar que o Ollama descarregue e recarregue o
            # modelo da VRAM a cada retry — mudar num_ctx força realocação do KV-cache.
            if historico_raciocinio:
                prompt_atual += config.PROMPT_LOOP_PENSAMENTO.format(historico_raciocinio=historico_raciocinio)
                think_flag = False
                predict_atual = config.LLM_NUM_CTX - 68  # usa quase todo o contexto para a resposta final

            gerador = self.executar_prompt(
                prompt=prompt_atual,
                modelo=modelo_nome,
                temperatura=config.LLM_TEMPERATURA,
                num_ctx=ctx_atual,
                num_predict=predict_atual,
                think=think_flag,
                options_extra=options_extra,
            )

            # Consome o gerador de streaming até o final para obter a resposta completa
            resposta_final = {}
            for resposta_final in gerador:
                pass

            t_fim_tentativa = time.time()
            status_erro = resposta_final.get("status_erro", "")

            tentativas_log.append({
                "numero_tentativa": tentativa,
                "resposta":         resposta_final.get("resposta", ""),
                "raciocinio":       resposta_final.get("raciocinio", ""),
                "controle":         resposta_final.get("controle", ""),
                "status_erro":      status_erro,
                "houve_falha":      bool(status_erro),
                "prompt_enviado":   prompt_atual,
                "metricas_ollama":  resposta_final.get("metricas_ollama", {}),
                "loop_tipo":        resposta_final.get("loop_tipo"),
                "tempo_segundos":   t_fim_tentativa - t_inicio_tentativa,
            })

            # Verifica se o modelo retornou uma falha interna ou atingiu limite de tokens (verificando flag paralela)
            if not status_erro and resposta_final.get("resposta"):
                resposta_final["prompt_enviado"] = prompt_atual
                # Se houve retry com PROMPT_INJECTION, o raciocínio da tentativa anterior
                # (injetado no prompt) é preservado no retorno para auditoria.
                if historico_raciocinio and not resposta_final.get("raciocinio"):
                    resposta_final["raciocinio"] = historico_raciocinio
                return resposta_final, tentativas_log

            # Log específico por tipo de erro
            if status_erro == "erro_stop_reason":
                loop_tipo = resposta_final.get("loop_tipo", "overflow_natural")
                logger.log(
                    f"⚠ LLM interrompido: {loop_tipo} (done_reason != 'stop').",
                    level=LogLevel.WARNING,
                )

                if not config.PROMPT_INJECTION:
                    # Modo PROMPT_INJECTION = False: falha total imediata.
                    # Retorna detecção zerada para que precisão e recall = 0.
                    # resposta_llm_real preserva o conteúdo truncado real para auditoria.
                    logger.log(
                        "⚠ PROMPT_INJECTION=False: abortando com detecção zerada (sem retry).",
                        level=LogLevel.WARNING,
                    )
                    metricas_zeradas = {
                        "done_reason": None,
                        "total_duration_sec": 0.0,
                        "load_duration_sec": 0.0,
                        "prompt_eval_count": 0,
                        "prompt_eval_duration_sec": 0.0,
                        "eval_count": 0,
                        "eval_duration_sec": 0.0,
                        "eval_tokens_per_second": 0.0,
                    }
                    return {
                        "resposta": '{"map_nomes": [], "map_enderecos": [], "map_outros": [], "duvidas_pendentes": []}',
                        "resposta_llm_real": resposta_final.get("resposta", ""),
                        "raciocinio":        resposta_final.get("raciocinio", ""),
                        "controle":          resposta_final.get("controle", ""),
                        "prompt_enviado":    prompt_atual,
                        "houve_falha":       True,
                        "status_erro":       status_erro,
                        "loop_tipo":         loop_tipo,
                        "metricas_ollama":   resposta_final.get("metricas_ollama", metricas_zeradas),
                    }, tentativas_log
                else:
                    # Modo PROMPT_INJECTION = True: resgata raciocínio para injetar na próxima tentativa
                    if resposta_final.get("raciocinio"):
                        historico_raciocinio = resposta_final.get("raciocinio")
                        # Mit-2: trunca raciocínio no ponto onde o loop começou
                        posicao_corte = resposta_final.get("loop_posicao_inicio")
                        if posicao_corte:
                            historico_raciocinio = historico_raciocinio[:posicao_corte]

            elif status_erro == "erro":
                logger.log("⚠ Erro de conexão ou execução no LLM.", level=LogLevel.WARNING)
            elif status_erro == "timeout":
                logger.log("⚠ LLM não respondeu dentro do tempo limite (Timeout).", level=LogLevel.WARNING)
            else:
                logger.log("⚠ LLM retornou resposta vazia.", level=LogLevel.WARNING)

            if tentativa < max_tentativas:
                time.sleep(2)

        logger.log(
            f"Falha após {max_tentativas} tentativas. Último status: '{status_erro}'.",
            level=LogLevel.ERROR,
        )

        # Envia de volta com flag de auditoria pra logar banco preservando o texto truncado na key 'resposta'
        # resposta_llm_real duplica 'resposta' explicitamente para simetria com o caminho PROMPT_INJECTION=False
        resposta_final["prompt_enviado"] = prompt_atual
        resposta_final["houve_falha"] = True
        resposta_final["resposta_llm_real"] = resposta_final.get("resposta", "")
        return resposta_final, tentativas_log
