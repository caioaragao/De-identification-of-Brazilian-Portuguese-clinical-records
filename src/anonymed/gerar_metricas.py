import argparse
import io
import json
import os
import sys

# Garante que o Python consegue importar os módulos raiz (logger, config, etc)
# independentemente de onde o script for chamado
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logger
from logger import LogLevel


class TeeWriter:
    """Escreve simultaneamente no terminal (stdout) e em um buffer de texto."""

    def __init__(self, original_stdout):
        self.terminal = original_stdout
        self.buffer = io.StringIO()

    def write(self, msg):
        self.terminal.write(msg)
        self.buffer.write(msg)

    def flush(self):
        self.terminal.flush()

    def getvalue(self) -> str:
        return self.buffer.getvalue()


# --- CONFIGURAÇÃO ---
CATEGORIAS_RELEVANTES = {
    "DOCTOR", "PATIENT",                          # NAME
    "CITY", "COUNTRY", "STATE", "STREET",         # LOCATION
    "HOSPITAL", "ORGANIZATION", "LOCATION_OTHER", # LOCATION
    "PHONE", "EMAIL", "ZIP",                      # CONTACT
    "DATE",                                       # DATE
    "IDNUM", "MEDICAL_RECORD", "HEALTH_PLAN",     # ID
    "AGE",                                        # AGE
    "PROFESSION",                                 # PROFESSION
    "OTHER",                                      # OTHER — incluído para comparação com autor do dataset
}

# Mínimo de caracteres para um grupo residual de máscara contar como FP.
# Valor 2 filtra espaços e pontuação isolados entre labels adjacentes,
# sem ignorar entidades curtas como "SP" (2 chars) ou "Ana" (3 chars).
MIN_CHARS_FP_GRUPO = 3

# Constantes de Cores para Terminal
AMARELO = "\033[93m"
VERDE = "\033[92m"
VERMELHO = "\033[91m"
AZUL = "\033[94m"
RESET = "\033[0m"

# --- FUNÇÕES DE CARGA ---


def carregar_json(caminho: str) -> list[dict]:
    if not caminho or not os.path.exists(caminho):
        logger.log(f"Erro: Arquivo JSON não encontrado: {caminho}", level=LogLevel.ERROR)
        return []
    logger.log(f"Carregando dataset (Gabarito): {caminho}...", level=LogLevel.NORMAL)
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def carregar_mask_buffer(caminho: str) -> str:
    """Carrega o arquivo de máscara como buffer contínuo."""
    if not caminho or not os.path.exists(caminho):
        logger.log(f"Erro: Arquivo MASK não encontrado: {caminho}", level=LogLevel.ERROR)
        return ""
    logger.log(f"Carregando máscara do sistema: {caminho}...", level=LogLevel.NORMAL)
    with open(caminho, encoding="utf-8") as f:
        return f.read()


def carregar_meta(caminho_mask: str) -> tuple[str, str]:
    caminho_meta = caminho_mask + ".meta"
    if not os.path.exists(caminho_meta):
        logger.log(f"Aviso: Metadados ({caminho_meta}) não encontrados. Usando padrões (&, *).", level=LogLevel.WARNING)
        return "&", "*"

    with open(caminho_meta) as f:
        dados = json.load(f)
    return dados.get("char_pii", "&"), dados.get("char_unk", "*")


def gerar_buffer_groundtruth(dataset: list[dict], char_pii: str, categorias: set[str] | None = None) -> str:
    """Gera buffer com labels do GT substituídos por char_pii.
    Se categorias=None, usa TODAS as categorias. Se fornecido, filtra."""
    partes = []
    for registro in dataset:
        texto = list(registro.get("descricao", ""))
        for label in registro.get("labels", []):
            if categorias and label.get("subcategory", "OTHER") not in categorias:
                continue
            l_start = label["first_position"]
            l_end = label["last_position"]
            for i in range(l_start, min(l_end, len(texto))):
                texto[i] = char_pii
        partes.append("".join(texto))
    return "".join(partes)


# --- FUNÇÃO PRINCIPAL ---


def calcular_metricas(dataset: list[dict], buffer_mascara: str, chars: tuple[str, str]) -> dict:
    if not dataset or not buffer_mascara:
        logger.log(f"Impossível calcular métricas: Dados ausentes.", level=LogLevel.ERROR)
        return {}

    char_pii, char_unk = chars
    chars_mascara = {char_pii, char_unk}

    # Valida tamanho total
    tamanho_esperado = sum(len(r.get("descricao", "")) for r in dataset)
    tamanho_buffer = len(buffer_mascara)

    if tamanho_esperado != tamanho_buffer:
        logger.log(
            f"AVISO: Buffer ({tamanho_buffer} chars) ≠ dataset ({tamanho_esperado} chars). Apenas registros cobertos pelo buffer serão avaliados.", level=LogLevel.WARNING
        )

    total_tp = 0
    total_tp_unk = 0
    total_fn = 0
    total_fp = 0
    total_chars_fp_residual = 0
    registros_avaliados = 0

    detalhe_cat = {cat: {"tp": 0, "tp_unk": 0, "fn": 0, "total": 0} for cat in CATEGORIAS_RELEVANTES}

    dicionario_fp = {}
    dicionario_fn = {}

    registros_desalinhados = []
    cursor = 0

    # Inicializa buffers com o texto original (para facilitar o diff)
    texto_completo = "".join(r.get("descricao", "") for r in dataset)
    texto_avaliado = texto_completo[:tamanho_buffer]
    buffer_fp = list(texto_avaliado)
    buffer_fn = list(texto_avaliado)

    logger.log(f"\nProcessando até {len(dataset)} registros (buffer de {tamanho_buffer} chars)...", level=LogLevel.NORMAL)

    for idx, registro in enumerate(dataset):
        texto_original = registro.get("descricao", "")
        tamanho_registro = len(texto_original)

        # Para quando o buffer acabar
        if cursor >= tamanho_buffer:
            break

        registros_avaliados += 1

        # Fatia o buffer e cria cópia mutável para consumo
        mascara_recorte = list(buffer_mascara[cursor : cursor + tamanho_registro])
        offset_registro = cursor  # Para referenciar a posição global no buffer
        cursor += tamanho_registro

        # Verifica alinhamento
        if len(mascara_recorte) != tamanho_registro:
            registros_desalinhados.append(
                {
                    "idx": idx,
                    "prontuario": registro.get("prontuario", "?"),
                    "len_texto": tamanho_registro,
                    "len_mascara": len(mascara_recorte),
                }
            )

        labels = registro.get("labels", [])

        # --- PASSO 1: Avalia cada label do GT (TP/FN) e CONSOME a região ---
        for label in labels:
            categoria = label.get("subcategory", "OTHER")
            l_start = label["first_position"]
            l_end = label["last_position"]

            # CONSOME SEMPRE: troca chars de máscara nessa região por '.'
            # Mesmo categorias fora de CATEGORIAS_RELEVANTES são consumidas
            # para evitar que detecções corretas (ex: AGE) virem FP falso.
            for i in range(l_start, min(l_end, len(mascara_recorte))):
                if mascara_recorte[i] in chars_mascara:
                    mascara_recorte[i] = "."

            # Só conta TP/FN para categorias relevantes
            if categoria not in CATEGORIAS_RELEVANTES:
                continue

            detalhe_cat[categoria]["total"] += 1

            if l_end > len(mascara_recorte):
                total_fn += 1
                detalhe_cat[categoria]["fn"] += 1
                continue

            trecho_original = buffer_mascara[cursor - tamanho_registro + l_start : cursor - tamanho_registro + l_end]

            qtd_pii = trecho_original.count(char_pii)
            qtd_unk = trecho_original.count(char_unk)
            qtd_mascarada = qtd_pii + qtd_unk
            tamanho_termo = l_end - l_start

            if tamanho_termo == 0:
                continue

            # Avalia TP ou FN
            if (qtd_mascarada / tamanho_termo) > 0.5:
                if qtd_pii >= qtd_unk:
                    total_tp += 1
                    detalhe_cat[categoria]["tp"] += 1
                else:
                    total_tp_unk += 1
                    detalhe_cat[categoria]["tp_unk"] += 1
            else:
                total_fn += 1
                detalhe_cat[categoria]["fn"] += 1

                termo_fn = texto_original[l_start:l_end]
                dicionario_fn[termo_fn] = dicionario_fn.get(termo_fn, 0) + 1

                # Preenche o buffer_fn com o texto original que vazou (ou char_pii para destaque visual se preferir)
                # Vamos injetar os caracteres da máscara (char_pii) na posição do *label* para indicar = "ISTO devia ter sido mascarado!"
                for i in range(l_start, l_end):
                    if offset_registro + i < tamanho_buffer:
                        buffer_fn[offset_registro + i] = char_pii

        # --- PASSO 2: Conta FP nos grupos RESIDUAIS de máscara ---
        grupo_atual = 0
        indices_grupo = []
        for i_char, char in enumerate(mascara_recorte):
            if char in chars_mascara:
                grupo_atual += 1
                indices_grupo.append(i_char)
            else:
                if grupo_atual >= MIN_CHARS_FP_GRUPO:
                    total_fp += 1
                    total_chars_fp_residual += grupo_atual

                    termo_fp = texto_original[indices_grupo[0] : indices_grupo[-1] + 1]
                    dicionario_fp[termo_fp] = dicionario_fp.get(termo_fp, 0) + 1

                    # Preenche o buffer de FP
                    for idx_fp in indices_grupo:
                        if offset_registro + idx_fp < tamanho_buffer:
                            buffer_fp[offset_registro + idx_fp] = buffer_mascara[offset_registro + idx_fp]
                grupo_atual = 0
                indices_grupo.clear()

        # Último grupo pendente
        if grupo_atual >= MIN_CHARS_FP_GRUPO:
            total_fp += 1
            total_chars_fp_residual += grupo_atual

            termo_fp = texto_original[indices_grupo[0] : indices_grupo[-1] + 1]
            dicionario_fp[termo_fp] = dicionario_fp.get(termo_fp, 0) + 1

            for idx_fp in indices_grupo:
                if offset_registro + idx_fp < tamanho_buffer:
                    buffer_fp[offset_registro + idx_fp] = buffer_mascara[offset_registro + idx_fp]

    # Alertas de alinhamento
    if registros_desalinhados:
        logger.log(
            f"\nALERTA: {len(registros_desalinhados)} registros com tamanho de máscara ≠ texto original:", level=LogLevel.WARNING
        )
        for r in registros_desalinhados[:5]:
            logger.log(f"  Prontuário {r['prontuario']}: texto={r['len_texto']} chars, máscara={r['len_mascara']} chars", level=LogLevel.NORMAL)
        if len(registros_desalinhados) > 5:
            logger.log(f"  ... e mais {len(registros_desalinhados) - 5}", level=LogLevel.NORMAL)

    # Calcula métricas
    total_tp_geral = total_tp + total_tp_unk

    precision = total_tp_geral / (total_tp_geral + total_fp) if (total_tp_geral + total_fp) > 0 else 0.0
    recall = total_tp_geral / (total_tp_geral + total_fn) if (total_tp_geral + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f2 = 5 * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0

    # --- IMPRESSÃO ---
    logger.log("\n" + "=" * 50, level=LogLevel.NORMAL)
    logger.log("RESULTADOS GERAIS (MÁSCARA POSICIONAL - RESIDUAL)", level=LogLevel.NORMAL)
    logger.log("=" * 50, level=LogLevel.NORMAL)
    logger.log(f"Caracteres Usados: PII='{char_pii}'  UNK='{char_unk}'", level=LogLevel.NORMAL)
    logger.log(f"Registros: {registros_avaliados}/{len(dataset)}  |  Buffer: {tamanho_buffer} chars", level=LogLevel.NORMAL)
    logger.log(f"Filtro FP: grupos ≥ {MIN_CHARS_FP_GRUPO} chars", level=LogLevel.NORMAL)
    logger.log("-" * 30, level=LogLevel.NORMAL)
    logger.log(f"TP (Certeza):       {total_tp}", level=LogLevel.NORMAL)
    logger.log(f"TP (Dúvida/Safe):   {total_tp_unk}", level=LogLevel.NORMAL)
    logger.log(f"False Negatives:    {total_fn}", level=LogLevel.NORMAL)
    logger.log(f"False Positives:    {total_fp}   (grupos residuais ≥ {MIN_CHARS_FP_GRUPO} chars)", level=LogLevel.NORMAL)
    logger.log(f"  └ Chars residuais: {total_chars_fp_residual}", level=LogLevel.NORMAL)
    logger.log("-" * 30, level=LogLevel.NORMAL)
    logger.log(f"PRECISION: {precision:.4f} ({precision:.2%})", level=LogLevel.NORMAL)
    logger.log(f"RECALL:    {recall:.4f} ({recall:.2%})", level=LogLevel.NORMAL)
    logger.log(f"F1-SCORE:  {f1:.4f} ({f1:.2%})", level=LogLevel.NORMAL)
    logger.log(f"F2-SCORE:  {f2:.4f} ({f2:.2%})", level=LogLevel.NORMAL)
    logger.log("=" * 50, level=LogLevel.NORMAL)

    # Detalhe por categoria
    logger.log(f"\n{'CATEGORIA':<20} | {'TOTAL':<6} | {'TP':<5} | {'UNK':<5} | {'FN':<5} | {'RECALL':<8}", level=LogLevel.NORMAL)
    logger.log("-" * 65, level=LogLevel.NORMAL)
    detalhe_cat_output = {}
    for cat, dados in sorted(detalhe_cat.items()):
        acertos = dados["tp"] + dados["tp_unk"]
        rec = acertos / dados["total"] if dados["total"] > 0 else 0.0
        if dados["total"] > 0:
            logger.log(
                f"{cat:<20} | {dados['total']:<6} | {dados['tp']:<5} | {dados['tp_unk']:<5} | {dados['fn']:<5} | {rec:.2%}", level=LogLevel.NORMAL
            )
            detalhe_cat_output[cat] = {
                "total": dados["total"],
                "tp": dados["tp"],
                "tp_unk": dados["tp_unk"],
                "fn": dados["fn"],
                "recall": round(rec, 4),
            }

    # Monta resultado para JSON
    resultado = {
        "metricas": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "f2_score": round(f2, 4),
        },
        "contagens": {
            "tp_certeza": total_tp,
            "tp_duvida": total_tp_unk,
            "tp_total": total_tp_geral,
            "false_negatives": total_fn,
            "false_positives": total_fp,
            "chars_fp_residuais": total_chars_fp_residual,
        },
        "config": {
            "char_pii": char_pii,
            "char_unk": char_unk,
            "min_chars_fp_grupo": MIN_CHARS_FP_GRUPO,
            "registros_avaliados": registros_avaliados,
            "registros_total_dataset": len(dataset),
            "tamanho_buffer": tamanho_buffer,
            "registros_desalinhados": len(registros_desalinhados),
        },
        "detalhe_por_categoria": detalhe_cat_output,
    }

    resultado["_registros_avaliados"] = registros_avaliados
    resultado["_buffer_fp"] = "".join(buffer_fp)
    resultado["_buffer_fn"] = "".join(buffer_fn)
    resultado["_dict_fp"] = dict(sorted(dicionario_fp.items(), key=lambda item: item[1], reverse=True))
    resultado["_dict_fn"] = dict(sorted(dicionario_fn.items(), key=lambda item: item[1], reverse=True))

    return resultado


def calcular_e_salvar_metricas(ds_path: str, mk_path: str) -> dict:
    """Carrega arquivos, calcula F2 e exporta relatórios físicos. Retorna métricas calculadas."""
    # Pulo rápido: se o JSON final já existir, não refaz todo o carregamento pesado
    base = os.path.splitext(mk_path)[0]
    caminho_json_existente = base + ".json"

    if os.path.exists(caminho_json_existente):
        logger.log(f"\nMétricas já calculadas para este arquivo ({caminho_json_existente}). Pulando...\n", level=LogLevel.NORMAL)
        try:
            with open(caminho_json_existente, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Ativa TeeWriter para capturar stdout em buffer
    tee = TeeWriter(sys.stdout)
    sys.stdout = tee

    try:
        dataset = carregar_json(ds_path)
        buffer_mascara = carregar_mask_buffer(mk_path)
        chars = carregar_meta(mk_path)
        resultado = calcular_metricas(dataset, buffer_mascara, chars)

        if resultado:
            # Salva JSON estruturado
            caminho_json = base + ".json"
            with open(caminho_json, "w", encoding="utf-8") as f:
                json.dump(resultado, f, ensure_ascii=False, indent=2)
            logger.log(f"\nMétricas salvas em: {caminho_json}", level=LogLevel.NORMAL)

            # Salva .score (espelho do stdout)
            caminho_score = base + ".score"
            with open(caminho_score, "w", encoding="utf-8") as f:
                f.write(tee.getvalue())
            logger.log(f"Score salvo em: {caminho_score}", level=LogLevel.NORMAL)

            # Slice do dataset limitado ao que foi avaliado
            n_avaliados = resultado.get("_registros_avaliados", len(dataset))
            ds_avaliado = dataset[:n_avaliados]

            # Gera groundtruth com TODAS as categorias
            gt_todas = gerar_buffer_groundtruth(ds_avaliado, chars[0])
            caminho_gt = base + ".groundtruth"
            with open(caminho_gt, "w", encoding="utf-8") as f:
                f.write(gt_todas)
            logger.log(f"GT (todas categorias): {caminho_gt}", level=LogLevel.NORMAL)

            # Salva Buffer de False Positives (FP)
            caminho_fp = base + ".fp"
            with open(caminho_fp, "w", encoding="utf-8") as f:
                f.write(resultado.get("_buffer_fp", ""))
            logger.log(f"FP apenas (Falsos Positivos): {caminho_fp}", level=LogLevel.NORMAL)

            # Salva Buffer de False Negatives (FN)
            caminho_fn = base + ".fn"
            with open(caminho_fn, "w", encoding="utf-8") as f:
                f.write(resultado.get("_buffer_fn", ""))
            logger.log(f"FN apenas (Falsos Negativos): {caminho_fn}", level=LogLevel.NORMAL)

            # Salva dicionário de Falsos Positivos (.fp.json)
            caminho_fp_json = base + ".fp.json"
            with open(caminho_fp_json, "w", encoding="utf-8") as f:
                json.dump(resultado.get("_dict_fp", {}), f, ensure_ascii=False, indent=2)
            logger.log(f"Frequência de Falsos Positivos: {caminho_fp_json}", level=LogLevel.NORMAL)

            # Salva dicionário de Falsos Negativos (.fn.json)
            caminho_fn_json = base + ".fn.json"
            with open(caminho_fn_json, "w", encoding="utf-8") as f:
                json.dump(resultado.get("_dict_fn", {}), f, ensure_ascii=False, indent=2)
            logger.log(f"Frequência de Falsos Negativos: {caminho_fn_json}", level=LogLevel.NORMAL)

            # Gera groundtruth com apenas CATEGORIAS_RELEVANTES
            gt_relevante = gerar_buffer_groundtruth(ds_avaliado, chars[0], CATEGORIAS_RELEVANTES)
            caminho_gt_rel = base + ".groundtruth_relevante"
            with open(caminho_gt_rel, "w", encoding="utf-8") as f:
                f.write(gt_relevante)
            logger.log(f"GT (relevantes): {caminho_gt_rel}", level=LogLevel.NORMAL)
            
            return resultado
    except Exception as e:
        logger.log(f"Falha ao processar arquivo: {e}", level=LogLevel.ERROR)
        return {}
    finally:
        sys.stdout = tee.terminal


# --- MAIN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculadora de Métricas (Máscara Posicional - Residual)")
    parser.add_argument("dataset_json", nargs="?", help="Caminho do JSON (Ground Truth)")
    parser.add_argument("arquivo_mask", nargs="?", help="Caminho do arquivo .mask gerado")

    args = parser.parse_args()

    ds_path = args.dataset_json
    mk_path = args.arquivo_mask

    # Entrada Interativa se os argumentos não foram passados
    if not ds_path:
        logger.log(f"\nCaminho do dataset JSON não informado.", level=LogLevel.WARNING)
        ds_path = input("Caminho do JSON (Gabarito): ").strip()

    if not mk_path:
        logger.log(f"\nCaminho do arquivo de máscara não informado.", level=LogLevel.WARNING)
        mk_path = input("Caminho do arquivo .mask: ").strip()

    if ds_path and mk_path:
        resultado = calcular_e_salvar_metricas(ds_path, mk_path)
        if not resultado:
            sys.exit(1)
    else:
        logger.log(f"Erro: Caminhos obrigatórios não fornecidos.", level=LogLevel.ERROR)
        sys.exit(1)
