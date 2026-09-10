import logger
from logger import LogLevel
import json
import os
from datetime import datetime

import pandas as pd
from pandarallel import pandarallel


class GeradorRelatorio:
    """
    Classe responsável pela geração dos arquivos TXT finais (Original vs Anonimizado)
    e das métricas (máscara) para fins de auditoria.
    """

    def __init__(self, anonimizacao, configuracao, funcoes_gerais_module):
        """
        Inicializa o gerador de relatórios com as dependências necessárias.

        Args:
            anonimizacao: Instância do motor de anonimização configurado para a sessão
            configuracao: Módulo ou classe com configurações (contendo caminhos, cores, NOME_SESSAO, etc)
            funcoes_gerais_module: Módulo com utilitários gerais como identificar_caracteres_mascara
        """
        self._anonimizacao = anonimizacao
        self._config = configuracao
        self._funcoes_gerais = funcoes_gerais_module

    def gerar(self, df: pd.DataFrame, modelo_nome: str, silencioso: bool = False, sessao_nome: str | None = None):
        """
        Executa a geração dos arquivos paralelizada.

        Args:
            df: DataFrame com colunas 'descricao' e 'descricao_raw' limitadas ao escopo desejado
            modelo_nome: Nome do modelo que processou os dados (usado p/ gerar o arquivo)
            silencioso: Se True, oculta a barra de progresso do pandarallel
            sessao_nome: Nome da sessão para nomeação de arquivos. Se None, usa config.NOME_SESSAO.
        """
        try:
            pandarallel.initialize(nb_workers=11, progress_bar=not silencioso)
        except Exception as e:
            if not silencioso:
                logger.log(f"Aviso: pandarallel não inicializado ou já inicializado: {e}", level=LogLevel.WARNING)

        # Sanitiza o nome do modelo para uso em arquivo (remove caracteres proibidos)
        modelo_arquivo = str(modelo_nome).replace(":", "-").replace("/", "-")

        sessao = sessao_nome or self._config.NOME_SESSAO

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arq_original = self._config.ARQUIVO_SAIDA_ORIGINAL.format(
            timestamp=timestamp, sessao=sessao, modelo=modelo_arquivo
        )
        arq_anonimizado = self._config.ARQUIVO_SAIDA_ANONIMIZADO.format(
            timestamp=timestamp, sessao=sessao, modelo=modelo_arquivo
        )

        # Garante que o diretório de saída existe
        os.makedirs(os.path.dirname(arq_original), exist_ok=True)

        # Arquivos de Máscara (Métricas)
        arq_mascara = arq_original.replace("_original_", "_mascara_").replace(".txt", ".mask")
        arq_meta = arq_mascara + ".meta"

        if not silencioso:
            logger.log("Gerando arquivos de texto e máscara (Paralelizado)...", level=LogLevel.NORMAL)

        # Prepara Texto Completo para eleição de caracteres seguros
        texto_completo_concat = "".join(df["descricao_raw"].dropna().astype(str))
        char_pii, char_unk = self._funcoes_gerais.identificar_caracteres_mascara(texto_completo_concat, 2)

        if not silencioso:
            logger.log(f"Caracteres de Máscara Eleitos: PII='{char_pii}', UNK='{char_unk}'", level=LogLevel.NORMAL)

        # Salva Metadados da Máscara
        with open(arq_meta, "w") as f:
            json.dump({"char_pii": char_pii, "char_unk": char_unk}, f)

        # IMPORTANTE: Desativamos a interceptação global temporariamente para permitir
        # que o pandarallel consiga fazer o pickle/serializar objetos de sistema limpos
        logger.restaurar_print_global()

        # Passo 1: Gera versão com todas as tags (Anonimização Total Consolidada)
        coluna_anonimizada = df["descricao"].parallel_apply(self._anonimizacao.anonimizar_consolidado)

        # Passo 1.1: Gera Máscara (Posicional) sobre o texto BRUTO para alinhar com gabarito
        coluna_mascara = df["descricao_raw"].parallel_apply(
            lambda t: self._anonimizacao.gerar_mascara(t, (char_pii, char_unk))
        )

        # Passo 2: Revela boilerplates e revisões (Restaura Contexto Médico)
        coluna_anonimizada = coluna_anonimizada.parallel_apply(self._anonimizacao.restaurar_apenas_contexto)

        # Passo 3: Limpeza Estética das Tags PII para melhor leitura
        coluna_anonimizada = coluna_anonimizada.parallel_apply(self._anonimizacao.formatar_tags_para_relatorio)

        # Retorna o controle do logger para o modo interceptado
        logger.instalar_print_global()

        # Salva Arquivo Original
        with open(arq_original, "w", encoding="utf-8") as f:
            for item in df["descricao_raw"]:
                f.write(str(item))

        # Salva Arquivo Anonimizado
        with open(arq_anonimizado, "w", encoding="utf-8") as f:
            for item in coluna_anonimizada:
                f.write(str(item))

        # Salva Arquivo Máscara
        with open(arq_mascara, "w", encoding="utf-8") as f:
            for item in coluna_mascara:
                f.write(str(item))

        if not silencioso:
            verde = getattr(self._config, "VERDE", "")
            reset = getattr(self._config, "RESET", "")
            azul = getattr(self._config, "AZUL", "")
            logger.log(f"{verde}Arquivos salvos:\n - {arq_original}\n - {arq_anonimizado}\n - {arq_mascara}{reset}", level=LogLevel.NORMAL)

            script_metr = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anonymed", "gerar_metricas.py")
            ds_json = getattr(self._config, "ARQUIVO_DS_ENTRADA", "")
            abs_mask = os.path.abspath(arq_mascara).replace(" ", "\\ ")

            logger.log(f"\n{azul}Para calcular as métricas, execute:", level=LogLevel.NORMAL)
            logger.log(f"python {script_metr} {ds_json} {abs_mask}{reset}", level=LogLevel.NORMAL)

        return arq_mascara
