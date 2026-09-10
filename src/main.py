import argparse
import os
import shutil
import glob
import signal
import sys
from itertools import combinations

import pandas

# Bibliotecas Internas
import ClienteOllama
import funcoes_gerais
import Supabase
from a01_platform import config_router as config
from Anonimizacao import Anonimizacao
from funcoes_gerais import (
    _intersect_mapas,
    _merge_mapas,
    carregar_contador,
    sanitizar_nome_sessao,
)
from GeradorRelatorio import GeradorRelatorio
from PipelineOrquestrador import PipelineOrquestrador

import logger
from logger import LogLevel

# Importação live de métricas para não gerar bash sujo
from anonymed.gerar_metricas import calcular_e_salvar_metricas

# Variavel global para os args, será preenchida no main()
args = None


def gerar_sub_combinacao_len_k(elementos, k):
    return list(combinations(elementos, k))


def arq_nomes_base_limpar(mk_path):
    # base_path of metricas files
    return os.path.splitext(mk_path)[0]


def _encontrar_mask_existente(diretorio_saida: str, sessao: str, modelo_arquivo: str):
    """Procura por um .mask já gerado para esta combinação, ignorando o timestamp.

    Verifica primeiro a pasta de excluídos e depois a pasta de saída.

    Retorna:
        (caminho_mask, foi_excluido) se encontrado, ou (None, False) se não existe.
    """
    # Padrão determinístico (sem timestamp): *_mascara_{sessao}_{modelo}.mask
    padrao = f"*_mascara_{sessao}_{modelo_arquivo}.mask"

    pasta_excluidos = os.path.join(diretorio_saida, "01-ARQUIVOS-EXCLUIDOS")
    if os.path.isdir(pasta_excluidos):
        encontrados = glob.glob(os.path.join(pasta_excluidos, padrao))
        if encontrados:
            return encontrados[0], True

    pasta_gerados = os.path.join(diretorio_saida, "02-ARQUIVOS-GERADOS")
    if os.path.isdir(pasta_gerados):
        encontrados = glob.glob(os.path.join(pasta_gerados, padrao))
        if encontrados:
            return encontrados[0], False

    encontrados = glob.glob(os.path.join(diretorio_saida, padrao))
    if encontrados:
        return encontrados[0], False

    return None, False


def mover_arquivos_para_pasta(mk_path: str, nome_subpasta: str):
    """Move mascara, json, texto original, txt anonimizado etc gerados para uma subpasta específica (ex: 01-ARQUIVOS-EXCLUIDOS)."""
    base = os.path.splitext(mk_path)[0]
    out_dir = os.path.join(os.path.dirname(mk_path), nome_subpasta)
    os.makedirs(out_dir, exist_ok=True)

    movidos = 0

    # Arquivos possivelmente gerados:
    # _mascara_X.mask, .mask.meta, .json, .score, .fp, .fn, .fp.json, .fn.json, .groundtruth...
    # E o TXT original e anonimizado associados a este mask.
    prefixo_mask = os.path.basename(base)

    # Textos associados
    # Original: troca _mascara_ por _original_
    # Anonimizado: troca _mascara_ por _anonimizado_
    prefix_orig = prefixo_mask.replace("_mascara_", "_original_")
    prefix_anon = prefixo_mask.replace("_mascara_", "_anonimizado_")

    for f in glob.glob(os.path.join(os.path.dirname(mk_path), "*")):
        filename = os.path.basename(f)
        if filename.startswith(prefixo_mask) or filename.startswith(prefix_orig) or filename.startswith(prefix_anon):
            dest = os.path.join(out_dir, filename)
            shutil.move(f, dest)
            movidos += 1

    return movidos


def apagar_arquivos_gerados(mk_path: str):
    """Apaga máscara, meta, json, textos e demais artefatos gerados para esta combinação.
    Usado no modo headless (sem -r): os dados já foram persistidos no banco e os arquivos
    físicos não são necessários, evitando acúmulo em disco."""
    base = os.path.splitext(mk_path)[0]
    prefixo_mask = os.path.basename(base)
    prefix_orig = prefixo_mask.replace("_mascara_", "_original_")
    prefix_anon = prefixo_mask.replace("_mascara_", "_anonimizado_")

    apagados = 0
    for f in glob.glob(os.path.join(os.path.dirname(mk_path), "*")):
        filename = os.path.basename(f)
        if filename.startswith(prefixo_mask) or filename.startswith(prefix_orig) or filename.startswith(prefix_anon):
            try:
                os.remove(f)
                apagados += 1
            except OSError as e:
                logger.log(f"Aviso: não foi possível apagar '{f}': {e}", level=LogLevel.WARNING)

    return apagados


def processar_relatorios_branch_and_bound(modo: str, sessao_ids: list[int],
                                          df: pandas.DataFrame, supabase_client, config, funcoes_gerais, grupo: str = "default"):
    logger.log(f"\n--- MODO {modo.upper()} | GRUPO '{grupo}' | IDs {sessao_ids} ---", level=LogLevel.NORMAL)

    if not sessao_ids:
        return 0

    total_movidos_modo = 0

    # --- Carrega metadados de todas as sessões base ---
    sessoes_info = {}
    for sid in sessao_ids:
        dados_sessao = supabase_client.obter_detalhes_sessao(sid)
        if not dados_sessao:
            logger.log(f"Erro: Sessão ID {sid} não encontrada no banco.", level=LogLevel.ERROR)
            sys.exit(1)

        tag_ini, tag_fim = supabase_client.carregar_caracteres_tag(sid)
        if tag_ini is None or tag_fim is None:
            tag_ini, tag_fim = "[", "]"

        (m_nomes, m_ends, m_outros, m_boilers, m_revs, m_unknowns, _, _) = (
            supabase_client.carregar_conhecimento_sessao(sid)
        )

        nome = dados_sessao.get("nome_identificador", f"Sessao_{sid}")
        sessoes_info[sid] = {
            "nome": nome,
            "nome_sanitizado": sanitizar_nome_sessao(nome),
            "modelo": dados_sessao.get("modelo_llm", "modelo_desconhecido"),
            "tag_ini": tag_ini,
            "tag_fim": tag_fim,
            "m_nomes": m_nomes,
            "m_ends": m_ends,
            "m_outros": m_outros,
            "m_boilers": m_boilers,
            "m_revs": m_revs,
            "m_unknowns": m_unknowns,
        }

    db_cache = supabase_client.carregar_ids_combos_processados(modo, grupo) if supabase_client else {}

    # Estado de DFS/BFS: Dicionário arquivando F2 real
    # chaves são tuplas ordenadas de sids associadas, eval(f2_score)
    f2_history = {}

    # Etapa Iterativa BFS Nível por Nível
    max_k = len(sessao_ids)

    for k in range(1, max_k + 1):
        if k == 1:
            combos_atuais = [(sid,) for sid in sessao_ids]
        else:
            # Geração de combinações K a partir das de K-1 ativas (Apriori candidate generation estrito)
            todas_possiveis = list(combinations(sessao_ids, k))
            combos_atuais = []

            # Condição da poda Apriori: "Todas as N-1 subcombinações PRECISAM estar ativas (ter melhorado)"
            # Se a subcombinação não está no history, significa que ela sofreu poda ou não superou o maximo de seus pais.
            for combo in todas_possiveis:
                subcombos = list(combinations(combo, k - 1))
                valido = True
                for sc in subcombos:
                    if tuple(sorted(sc)) not in f2_history:
                        valido = False
                        break
                if valido:
                    combos_atuais.append(combo)

            if not combos_atuais:
                logger.log(f"Nenhuma combinação viável de tamanho {k} sobreviveu à poda. Fim do Branch.", level=LogLevel.NORMAL)
                break

        logger.log(f"\n--- NÍVEL {k} ({len(combos_atuais)} Combinações) ---", level=LogLevel.NORMAL)

        for combo in combos_atuais:
            c_sorted = tuple(sorted(combo))
            nomes_sanitizados = [sessoes_info[sid]["nome_sanitizado"] for sid in c_sorted]
            # K=1 não possui união nem interseção, unifica-se para otimizar cache
            if k == 1:
                prefixo_modo = "unico"
                modo_db = "unico"   # agnóstico: reutilizável por -u e -i sem sobrescrever
            else:
                prefixo_modo = modo
                modo_db = modo

            label_combo = f"{prefixo_modo}_" + "-".join(nomes_sanitizados)
            ids_label = f"{grupo}-s{'-s'.join(map(str, c_sorted))}"
            # Nome de arquivo curto: o identificador único do combo já está em ids_label
            # (grupo+IDs); aqui basta o modo para distinguir união/interseção e evitar
            # 'File name too long' com muitas sessões. label_combo segue só no banco.
            modelo_arquivo = prefixo_modo

            # --- Check do Cache do Banco DB O(1) ---
            f2_db = db_cache.get(label_combo, None)
            _regerar_db = False
            if f2_db is not None:
                if f2_db == -1.0:
                    if k < max_k:
                        logger.log(f"[{modo.upper()} Skip DB] '{label_combo}' podado anteriormente. Mantendo bloqueio.", level=LogLevel.WARNING)
                        continue # Não adiciona no history -> Subcombinações podadas
                    else:
                        logger.log(f"[{modo.upper()} Force DB] '{label_combo}' podado nativamente, mas forçado via CLI alvo (k={k}).", level=LogLevel.NORMAL)
                else:
                    # Tem métrica real. Apenas gerar caso mask_existente não esteja lá.
                    mask_existente, foi_excluido = _encontrar_mask_existente(config.DIRETORIO_SAIDA, ids_label, modelo_arquivo)
                    if not mask_existente:
                        logger.log(f"[{modo.upper()} Regerar DB] '{label_combo}' F2={f2_db:.4f} logado, mas Mask alvo na máquina atual não encontrada. Recriando pelo motor rápido...", level=LogLevel.WARNING)
                        # Seta historico para permitir progresso BFS (apriori condition) mas FORCA recriacao
                        f2_history[c_sorted] = f2_db
                        _regerar_db = True
                    else:
                        logger.log(f"[{modo.upper()} Skip DB] '{label_combo}' reutilizando conhecimento F2={f2_db:.4f} salvo no Banco.", level=LogLevel.NORMAL)
                        f2_history[c_sorted] = f2_db
                        continue

            # --- Verificação de arquivo legado (fallback ou pre-DB cache) ---
            mask_existente, foi_excluido = _encontrar_mask_existente(
                config.DIRETORIO_SAIDA, ids_label, modelo_arquivo
            )
            if mask_existente:
                if foi_excluido:
                    logger.log(
                        f"[{modo.upper()} Skip -> Sync DB] '{label_combo}' arquivado como PODADO na lixeira local. Migrando status para o Banco.",
                        level=LogLevel.WARNING,
                    )
                    if supabase_client:
                        # Realiza o lazy sync para o BD marcando como podado
                        supabase_client.salvar_combinacao(
                            sessao_ids=list(c_sorted), modo=modo_db, nivel_k=k, label_combo=label_combo,
                            nome_arquivo=None, metricas={}, podado=True, grupo=grupo
                        )
                    continue
                else:
                    caminho_json_existente = os.path.splitext(mask_existente)[0] + ".json"
                    if os.path.exists(caminho_json_existente):
                        import json as _json
                        try:
                            with open(caminho_json_existente, encoding="utf-8") as _f:
                                resultado_cache = _json.load(_f)
                            f2_cache = resultado_cache.get("metricas", {}).get("f2_score")
                            if f2_cache is not None:
                                logger.log(
                                    f"[{modo.upper()} Skip -> Sync DB] '{label_combo}' via Mask Legada. "
                                    f"Salvando no DB em tempo real e reaproveitando F2={f2_cache:.4f}.",
                                    level=LogLevel.NORMAL,
                                )
                                f2_history[c_sorted] = f2_cache
                                if supabase_client:
                                    supabase_client.salvar_combinacao(
                                        sessao_ids=list(c_sorted), modo=modo_db, nivel_k=k,
                                        label_combo=label_combo, nome_arquivo=os.path.basename(mask_existente),
                                        metricas=resultado_cache.get("metricas", {}), podado=False, grupo=grupo
                                    )
                                continue
                        except Exception as e:
                            logger.log(
                                f"[Skip] Falha ao ler JSON de cache ({caminho_json_existente}): {e}. Regenerando.",
                                level=LogLevel.WARNING,
                            )
                    else:
                        logger.log(
                            f"[Skip] Mask existente mas sem JSON ({mask_existente}). Regenerando métricas.",
                            level=LogLevel.WARNING,
                        )
            # --- fim do skip ---

            logger.log(f"\n[Gerando] {label_combo}...", level=LogLevel.NORMAL)

            # Instancia o motor com o tag_ini da primária
            primeiro = sessoes_info[c_sorted[0]]
            anonimizacao = Anonimizacao(tag_ini=primeiro["tag_ini"], tag_fim=primeiro["tag_fim"])

            if modo == "uniao":
                anonimizacao.map_nomes_encontrados = _merge_mapas([sessoes_info[s]["m_nomes"] for s in combo])
                anonimizacao.map_enderecos_encontrados = _merge_mapas([sessoes_info[s]["m_ends"] for s in combo])
                anonimizacao.map_outras_info_encontradas = _merge_mapas([sessoes_info[s]["m_outros"] for s in combo])
                anonimizacao.map_unknown_encontrados = _merge_mapas([sessoes_info[s]["m_unknowns"] for s in combo])
            else:
                anonimizacao.map_nomes_encontrados = _intersect_mapas([sessoes_info[s]["m_nomes"] for s in combo])
                anonimizacao.map_enderecos_encontrados = _intersect_mapas([sessoes_info[s]["m_ends"] for s in combo])
                anonimizacao.map_outras_info_encontradas = _intersect_mapas([sessoes_info[s]["m_outros"] for s in combo])
                # unknowns: UNION conservador — afetam char_unk na máscara
                anonimizacao.map_unknown_encontrados = _merge_mapas([sessoes_info[s]["m_unknowns"] for s in combo])
            # map_boilerplates e map_revisoes não são populados:
            # boilerplates não entram na máscara; revisões estão vazias em sessões concluídas.

            anonimizacao.contador_nomes = carregar_contador(anonimizacao.map_nomes_encontrados)
            anonimizacao.contador_enderecos = carregar_contador(anonimizacao.map_enderecos_encontrados)
            anonimizacao.contador_outros = carregar_contador(anonimizacao.map_outras_info_encontradas)
            anonimizacao.contador_unknown = carregar_contador(anonimizacao.map_unknown_encontrados)

            gerador = GeradorRelatorio(anonimizacao, config, funcoes_gerais)
            # Retorna caminho absoluto do .mask recem criado no disco
            caminho_mask = gerador.gerar(df, modelo_arquivo, silencioso=False, sessao_nome=ids_label)

            if not caminho_mask:
                logger.log("Erro: Caminho do Mask não gerado.", level=LogLevel.ERROR)
                continue

            # Roda as métricas Imediatamente usando RAM direct mode
            resultado = calcular_e_salvar_metricas(config.ARQUIVO_DS_ENTRADA, caminho_mask)

            if not resultado or "metricas" not in resultado:
                logger.log(f"Falha ao validar F2-Score do arquivo {caminho_mask}", level=LogLevel.ERROR)
                continue

            f2_atual = resultado["metricas"]["f2_score"]

            if _regerar_db:
                # Banco já tem a entrada correta (podado=False, F2 válido).
                # Só gerencia o arquivo físico — não reavalia Apriori nem sobrescreve o banco.
                if not getattr(args, 'report', False):
                    apagar_arquivos_gerados(caminho_mask)
                continue

            if k == 1:
                # Nível 1 sempre passa e é salvo no dicionário de parents. Se recuperado failover, atualiza.
                if c_sorted not in f2_history:
                    f2_history[c_sorted] = f2_atual
                logger.log(f"[{modo.upper()}] Base avaliada: {ids_label} alcançou F2={f2_atual:.4f}", level=LogLevel.NORMAL)
                if supabase_client:
                    supabase_client.salvar_combinacao(list(c_sorted), modo_db, k, label_combo, os.path.basename(caminho_mask), resultado["metricas"], podado=False, grupo=grupo)

                # HEADLESS: Se foi sucesso e não tiver -r explícito, apaga os artefatos
                if not getattr(args, 'report', False):
                    qtd = apagar_arquivos_gerados(caminho_mask)
                    logger.log(f"[{modo.upper()}] DB-Only (Headless): '{ids_label}' validada e {qtd} arquivos apagados.", level=LogLevel.VERBOSE)

            else:
                # Condição de Branch and Bound: O F2 desse combo K *deve* ser ESTRITAMENTE MAIOR que o máximo
                # F2 obtido entre todas as subcombinações (pais) de K-1 (ex: max(A+B, A+C, B+C)).
                subcombos = list(combinations(c_sorted, k - 1))
                f2_pais = []
                for sc in subcombos:
                    pai = tuple(sorted(sc))
                    if pai in f2_history:
                        f2_pais.append(f2_history[pai])

                max_f2_pais = max(f2_pais) if f2_pais else 0.0

                if f2_atual > max_f2_pais:
                    # Progresso confirmdo! Salva a ramificação e não move arquivos.
                    f2_history[c_sorted] = f2_atual
                    logger.log(f"[{modo.upper()}] Sucesso: '{ids_label}' com F2={f2_atual:.4f} superou max pai ({max_f2_pais:.4f}). Ramo validado!", level=LogLevel.NORMAL)
                    if supabase_client:
                        supabase_client.salvar_combinacao(list(c_sorted), modo_db, k, label_combo, os.path.basename(caminho_mask), resultado["metricas"], podado=False, grupo=grupo)

                    if not getattr(args, 'report', False):
                        qtd = apagar_arquivos_gerados(caminho_mask)
                        logger.log(f"[{modo.upper()}] DB-Only (Headless): '{ids_label}' crescimento validado e {qtd} arquivos apagados.", level=LogLevel.VERBOSE)

                else:
                    # Degradação no F2: ramo podado. Headless apaga; com -r preserva para auditoria.
                    if not getattr(args, 'report', False):
                        qtd = apagar_arquivos_gerados(caminho_mask)
                        total_movidos_modo += qtd
                        logger.log(f"[{modo.upper()}] Poda no Ramo: '{ids_label}' com F2={f2_atual:.4f} inferior ao max pai ({max_f2_pais:.4f}). {qtd} arquivos apagados.", level=LogLevel.WARNING)
                    else:
                        qtd = mover_arquivos_para_pasta(caminho_mask, "01-ARQUIVOS-EXCLUIDOS")
                        total_movidos_modo += qtd
                        logger.log(f"[{modo.upper()}] Poda no Ramo: '{ids_label}' com F2={f2_atual:.4f} inferior ao max pai ({max_f2_pais:.4f}). {qtd} arquivos movidos para 01-ARQUIVOS-EXCLUIDOS.", level=LogLevel.WARNING)
                    if supabase_client:
                        # Para os stats mantemos podado=True e nome=None já que excluímos.
                        supabase_client.salvar_combinacao(list(c_sorted), modo_db, k, label_combo, None, resultado["metricas"], podado=True, grupo=grupo)

    return total_movidos_modo


def _handler_abort(signum, frame):
    nome_sinal = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n[ABORTADO] Sistema interrompido ({nome_sinal}). Encerrando sem persistência.", flush=True)
    sys.exit(0)


def main():
    signal.signal(signal.SIGTERM, _handler_abort)
    signal.signal(signal.SIGINT, _handler_abort)

    global args
    parser = argparse.ArgumentParser(description="Sistema de Anonimização de Prontuários Médicos")
    parser.add_argument("-d", "--database", help="Caminho do arquivo de configuração do Supabase")
    parser.add_argument("-f", "--file_dataset", help="Caminho do dataset json ou csv")
    parser.add_argument(
        "-u", "--uniao",
        nargs="+",
        help="Gera relatório(s) via União incremental com Poda em F2. Ex: -u 63 65 72 ou -u 57"
    )
    parser.add_argument(
        "-i", "--interseccao",
        nargs="+",
        help="Gera relatório(s) via Intersecção. Ex: -i 63 65 72"
    )
    parser.add_argument("-l", "--limit", type=int, help="Limite de registros para o relatório ou teste")
    parser.add_argument("-y", "--yes", action="store_true", help="Pula a confirmação interativa inicial")
    parser.add_argument("--log-level", type=str, default=None,
        help="Nível de log: silent|error|warning|normal|verbose (padrão: config LOG_LEVEL)")
    parser.add_argument(
        "-g",
        "--grafico",
        nargs="?",
        const="relatorio_metricas.png",
        help="Gera relatório em imagem (relatorio_metricas.png)",
    )
    parser.add_argument(
        "-r", "--report",
        action="store_true",
        help="Mantém arquivos físicos (MASK/TXT) na pasta de saída. Sem a flag, o sistema atua em Headless Mode (escondendo-os na pasta 02-ARQUIVOS-GERADOS após alimentar o banco de dados)."
    )
    parser.add_argument(
        "-m", "--modelos",
        nargs="+",
        metavar="MODELO",
        help="Sobrescreve config.MODELOS_LLM. Ex: -m gpt-oss:20b gemma4:26b",
    )
    parser.add_argument(
        "--ollama-url",
        metavar="URL",
        help="Sobrescreve config.OLLAMA_URL. Ex: --ollama-url http://localhost:11434",
    )
    parser.add_argument(
        "--grupo",
        metavar="NOME",
        help="Nome do grupo de combinações. Obrigatório com -u/-i. Ex: --grupo cpu",
    )

    args = parser.parse_args()

    # Configura nível de log: CLI > config > padrão "normal"
    log_level = args.log_level or getattr(config, "LOG_LEVEL", "normal")
    logger.configurar(log_level)
    logger.instalar_print_global()

    if args.database:
        config.SUPABASE_CFG_PATH = args.database
        logger.log(f"Usando configuração do Supabase: {config.SUPABASE_CFG_PATH}")

    if args.file_dataset:
        config.ARQUIVO_DS_ENTRADA = args.file_dataset
        logger.log(f"Usando dataset: {config.ARQUIVO_DS_ENTRADA}")

    if args.modelos:
        config.MODELOS_LLM = args.modelos
        logger.log(f"Modelos sobrescritos: {config.MODELOS_LLM}")

    if args.ollama_url:
        config.OLLAMA_URL = args.ollama_url
        logger.log(f"Ollama URL sobrescrita: {config.OLLAMA_URL}")

    if args.limit:
        config.DATASET_FIM = args.limit + config.DATASET_INICIO

    logger.log("--- INICIALIZANDO SISTEMA DE ANONIMIZAÇÃO (MULTIMODELO) ---", level=LogLevel.NORMAL)
    pass

    if args.grafico:
        logger.log("MODO GRÁFICO ATIVADO", level=LogLevel.NORMAL)
        try:
            nome_saida = args.grafico
            if "/" not in nome_saida and "\\" not in nome_saida:
                from pathlib import Path
                caminho_saida = str(Path(config.DIRETORIO_SAIDA) / nome_saida)
            else:
                caminho_saida = nome_saida

            supabase_client = Supabase.Supabase(config.SUPABASE_CFG_PATH)
            if supabase_client and hasattr(supabase_client, 'supabase') and supabase_client.supabase:
                logger.log("Lendo métricas globalizadas diretamente do Banco de Dados O(1)...", level=LogLevel.NORMAL)
                funcoes_gerais.plotar_metricas_via_banco(supabase_client, caminho_saida)
            else:
                logger.log(f"Lendo JSONs de métricas legados em: {config.DIRETORIO_SAIDA}", level=LogLevel.NORMAL)
                funcoes_gerais.plotar_metricas_relatorios(config.DIRETORIO_SAIDA, caminho_saida)
            sys.exit(0)
        except Exception as e:
            logger.log(f"Erro ao gerar gráfico: {e}", level=LogLevel.ERROR)
            sys.exit(1)

    # Verifica modos solicitados (Union/Intersection)
    requests = []
    if args.uniao is not None:
        ids_flat = []
        for v in args.uniao:
            for s in str(v).split(","):
                ids_flat.append(int(s.strip()))
        requests.append(("uniao", list(set(ids_flat))))

    if args.interseccao is not None:
        ids_flat = []
        for v in args.interseccao:
            for s in str(v).split(","):
                ids_flat.append(int(s.strip()))
        requests.append(("interseccao", list(set(ids_flat))))

    if requests:
        if not args.grupo:
            parser.error("--grupo é obrigatório ao usar -u ou -i. Ex: --grupo cpu")
        try:
            if config.ARQUIVO_DS_ENTRADA[-4:] == ".csv":
                df = pandas.read_csv(config.ARQUIVO_DS_ENTRADA, header=1)
            elif config.ARQUIVO_DS_ENTRADA[-5:] == ".json":
                df = pandas.read_json(config.ARQUIVO_DS_ENTRADA)
            else:
                raise TypeError(f"Formato do dataset não é reconhecido: {config.ARQUIVO_DS_ENTRADA}")

            if args.limit:
                df = df.iloc[: args.limit]
                logger.log(f"Limitado a {args.limit} registros.", level=LogLevel.NORMAL)
            elif config.DATASET_FIM:
                df = df.iloc[config.DATASET_INICIO : config.DATASET_FIM]

            df["descricao_raw"] = df["descricao"].copy()
            df = funcoes_gerais.pre_processamento_geral(df)

            supabase_client = Supabase.Supabase(config.SUPABASE_CFG_PATH)

            total_geral_movidos = 0
            for modo, sid_list in requests:
                total_geral_movidos += processar_relatorios_branch_and_bound(modo, sid_list, df, supabase_client, config, funcoes_gerais, grupo=args.grupo)

            if total_geral_movidos > 0:
                logger.log(f"\n[SUMÁRIO] {total_geral_movidos} arquivos foram movidos para a pasta '01-ARQUIVOS-EXCLUIDOS' devido à poda por desempenho.", level=LogLevel.NORMAL)

            logger.log("\nWorkflows Concluídos com Sucesso!", level=LogLevel.NORMAL)
            sys.exit(0)
        except Exception as e:
            import traceback
            logger.log(f"Erro no processamento dos relatórios: {e}\n{traceback.format_exc()}", level=LogLevel.ERROR)
            sys.exit(1)

    # --- SETUP DE EXECUÇÃO NORMAL ---
    try:
        cliente_ollama = ClienteOllama.ClienteOllama(config.OLLAMA_URL)
        supabase_client = Supabase.Supabase(config.SUPABASE_CFG_PATH)

        logger.log("\nCONFIGURAÇÃO DO BATCH:", level=LogLevel.NORMAL)
        logger.log(f" > Nome Sessão: {config.NOME_SESSAO}", level=LogLevel.NORMAL)
        logger.log(f" > Modelos a Executar: {config.MODELOS_LLM}", level=LogLevel.NORMAL)

        if args.yes:
            logger.log("\nConfirmação automática (-y). Iniciando...", level=LogLevel.NORMAL)
        else:
            input("\nPressione ENTER para confirmar a execução em lote, ou CTRL+C para sair...")

        if config.ARQUIVO_DS_ENTRADA[-4:] == ".csv":
            df = pandas.read_csv(config.ARQUIVO_DS_ENTRADA, header=1)
        elif config.ARQUIVO_DS_ENTRADA[-5:] == ".json":
            df = pandas.read_json(config.ARQUIVO_DS_ENTRADA)
        else:
            raise TypeError(f"Formato do dataset não é reconhecido: {config.ARQUIVO_DS_ENTRADA}")

        fim = config.DATASET_FIM if config.DATASET_FIM is not None else len(df)
        df = df.iloc[config.DATASET_INICIO : fim]

        df["descricao_raw"] = df["descricao"].copy()
        df = funcoes_gerais.pre_processamento_geral(df)

        logger.log(f"Dataset carregado. Linhas a processar por modelo: {len(df)}", level=LogLevel.NORMAL)

        for modelo in config.MODELOS_LLM:
            try:
                orquestrador = PipelineOrquestrador(supabase_client, cliente_ollama, config, funcoes_gerais)
                caminho_mask = orquestrador.executar(modelo, df, args.report, args.yes)
                if caminho_mask:
                    calcular_e_salvar_metricas(config.ARQUIVO_DS_ENTRADA, caminho_mask)
            except Exception as e:
                logger.log(f"Erro ao processar modelo {modelo}: {e}", level=LogLevel.ERROR)

    except Exception as e:
        logger.log(f"ERRO CRÍTICO na inicialização: {e}", level=LogLevel.ERROR)
        sys.exit(1)

if __name__ == "__main__":
    main()
