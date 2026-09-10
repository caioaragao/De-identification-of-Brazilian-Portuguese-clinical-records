import os
import sys

# Corrige colisão de nomes com pacotes PyPI caso rodado como script solto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logger
from logger import LogLevel
import re

import pandas as pd

ID_INICIAL = 875  # 2019 e 2422
lista_dfs = []

# Diretório de datasets, resolvido em relação a este arquivo (portável entre ambientes)
DIR_DATASETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

JSONS_IN_OUT = tuple(
    (
        os.path.join(DIR_DATASETS, f"clinical_deid_{particao}_set.json"),  # INPUT_FILE
        os.path.join(DIR_DATASETS, f"clinical_deid_{particao}_SEM_TAGS.json"),  # OUTPUT_FILE
    )
    for particao in ("training", "validation", "test")
)


def limpar_tags_e_ajustar_labels(df):
    """
    Remove tags do texto baseando-se nos labels e verifica a integridade.
    Remove linhas com erros de consistência.
    """

    def process_row(row):
        original_text = row["text"]
        labels = row["labels"]
        _row_id = row.get("id", "?")  # noqa: F841 — usado em prints de debug comentados
        should_remove = False

        # Ordenar labels pela posição inicial para processamento sequencial
        labels.sort(key=lambda x: x["first_position"])

        new_text_parts = []
        new_labels = []
        cursor_original = 0

        for label in labels:
            word = label["word"]
            subcat = label["subcategory"]
            start = label["first_position"]
            end = label["last_position"]

            pre_text = original_text[cursor_original:start]

            # --- VERIFICAÇÃO DO PADRÃO DA TAG ABERTURA ---
            lookback_limit = 50
            search_start = max(0, start - lookback_limit)
            snippet_before = original_text[search_start:start]

            opening_tag_match = re.search(rf"<{re.escape(subcat)}\b[^>]*>$", snippet_before)

            if opening_tag_match:
                tag_len = len(opening_tag_match.group(0))
                pre_text = pre_text[:-tag_len]
            else:
                should_remove = True

            new_text_parts.append(pre_text)

            # --- CORREÇÃO: Remove espaços do word e ajusta posições ---
            word_stripped = word.strip()
            if word_stripped != word:
                leading = len(word) - len(word.lstrip())
                # O espaço leading vira parte do pre_text (já appendado acima)
                if leading > 0:
                    new_text_parts[-1] = new_text_parts[-1] + word[:leading]
                word = word_stripped

            # --- CALCULA NOVA POSIÇÃO ---
            current_new_text_len = sum(len(p) for p in new_text_parts)
            new_start = current_new_text_len
            new_end = new_start + len(word)

            new_text_parts.append(word)

            new_label = label.copy()
            new_label["word"] = word
            new_label["first_position"] = new_start
            new_label["last_position"] = new_end
            new_labels.append(new_label)

            # --- VERIFICAÇÃO DO PADRÃO DA TAG FECHAMENTO ---
            snippet_after = original_text[end : end + lookback_limit]
            closing_tag_match = re.match(rf"^</{re.escape(subcat)}\b[^>]*>", snippet_after)

            if closing_tag_match:
                tag_len_close = len(closing_tag_match.group(0))
                cursor_original = end + tag_len_close
            else:
                cursor_original = end
                should_remove = True

        new_text_parts.append(original_text[cursor_original:])
        full_new_text = "".join(new_text_parts)

        # --- CHECK FINAL DE TAGS SOBRANDO ---
        leftover_tags = list(re.finditer(r"<[^>]+>", full_new_text))
        if leftover_tags:
            should_remove = True

        row["text"] = full_new_text
        row["labels"] = new_labels
        row["__REMOVE__"] = should_remove
        return row

    # Aplica o processamento
    processed_df = df.apply(process_row, axis=1)

    # Filtra as linhas marcadas para remoção
    initial_count = len(processed_df)
    final_df = processed_df[~processed_df["__REMOVE__"]].copy()
    removed_count = initial_count - len(final_df)

    if removed_count > 0:
        logger.log(f"\n[RESUMO] Foram removidas {removed_count} linhas devido a erros de consistência.", level=LogLevel.ERROR)

    # Remove a coluna temporária
    final_df.drop(columns=["__REMOVE__"], inplace=True)

    return final_df


if __name__ == "__main__":
    for INPUT_FILE, OUTPUT_FILE in JSONS_IN_OUT:
        logger.log(f"ID: {ID_INICIAL}. Arquivo: {INPUT_FILE[:50]}", level=LogLevel.NORMAL)
        try:
            logger.log(f"Lendo {INPUT_FILE}...", level=LogLevel.NORMAL)
            df = pd.read_json(INPUT_FILE)

            logger.log("Iniciando processamento...", level=LogLevel.NORMAL)
            df_clean = limpar_tags_e_ajustar_labels(df)

            logger.log(f"Dataset processado: {len(df_clean)} registros restantes (de {len(df)} originais).", level=LogLevel.NORMAL)

            # 1. Remover coluna synthetic
            if "synthetic" in df_clean.columns:
                df_clean.drop(columns=["synthetic"], inplace=True)

            # 2. Gerar novos IDs a partir de 875
            df_clean["id"] = range(ID_INICIAL, ID_INICIAL + len(df_clean))

            # Validação final de todos os labels
            logger.log("\n--- Iniciando Validação Completa do Dataset Processado ---", level=LogLevel.NORMAL)

            total_labels_checked = 0
            total_errors = 0

            for index, row in df_clean.iterrows():
                row_id = row["id"]
                text = row["text"]
                labels = row["labels"]

                for label in labels:
                    total_labels_checked += 1
                    word = label["word"]
                    start = label["first_position"]
                    end = label["last_position"]

                    extracted_word = text[start:end]

                    if extracted_word != word:
                        total_errors += 1
                        logger.log(
                            f"[ERRO VALIDACAO] ID: {row_id} | Esperado: '{word}' | Extraído: '{extracted_word}' (Pos: {start}-{end})", level=LogLevel.NORMAL
                        )

            logger.log("\n--- Resultado da Validação ---", level=LogLevel.NORMAL)
            logger.log(
                f"Último ID de df: {len(df) + ID_INICIAL - 1}. Último ID de df_clean: {len(df_clean) + ID_INICIAL - 1}.", level=LogLevel.NORMAL
            )
            logger.log(f"Total de labels verificados: {total_labels_checked}", level=LogLevel.NORMAL)
            if total_errors == 0:
                logger.log("SUCESSO: Todos os labels correspondem exatamente ao texto processado.", level=LogLevel.NORMAL)

                # 3. Salvar em saida.json
                df_clean.rename(columns={"text": "descricao"}, inplace=True)
                df_clean.rename(columns={"id": "prontuario"}, inplace=True)
                df_clean.to_json(OUTPUT_FILE, orient="records", indent=4, force_ascii=False)
                logger.log(f"Arquivo salvo com sucesso em: {OUTPUT_FILE}", level=LogLevel.NORMAL)

                ID_INICIAL = len(df_clean) + ID_INICIAL
                lista_dfs.append(df_clean)
            else:
                logger.log(f"FALHA: Encontrados {total_errors} erros de mapeamento. O arquivo de saída NÃO foi gerado.", level=LogLevel.ERROR)

        except Exception as e:
            logger.log(f"Ocorreu um erro crítico: {e}", level=LogLevel.ERROR)
            import traceback

            traceback.print_exc()

    df_train_full_sem_tags = pd.concat(lista_dfs, ignore_index=True)
    df_train_full_sem_tags.to_json(
        os.path.join(DIR_DATASETS, "clinical_deid_FULL_SEM_TAGS.json"),
        orient="records",
        indent=4,
        force_ascii=False,
    )
