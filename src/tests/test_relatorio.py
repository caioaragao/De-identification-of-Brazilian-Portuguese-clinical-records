"""
Testes unitários para a classe GeradorRelatorio.py
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from GeradorRelatorio import GeradorRelatorio


class TestGeradorRelatorio:
    @patch("GeradorRelatorio.pandarallel.initialize")
    def test_geracao_arquivos(self, mock_initialize, tmp_path):
        """Verifica se o método gerar cria os arquivos .txt e .mask nos caminhos corretos."""

        # Evita erro de pickling (dill + MagicMock) forçando o uso de apply normal
        pd.Series.parallel_apply = pd.Series.apply

        # Classe dummy para evitar problemas do MagicMock com o pd.Series.apply (que tenta iterar mocks)
        class DummyAnonimizacao:
            def anonimizar_consolidado(self, x):
                return "Texto anonimizado"

            def gerar_mascara(self, x, chars):
                return "Mascara gerada"

            def restaurar_apenas_contexto(self, x):
                return "Contexto restaurado"

            def formatar_tags_para_relatorio(self, x):
                return "Final formatado"

        mock_anonimizacao = DummyAnonimizacao()

        mock_config = MagicMock()
        mock_config.ARQUIVO_SAIDA_ORIGINAL = str(tmp_path / "{timestamp}_original_{sessao}_{modelo}.txt")
        mock_config.ARQUIVO_SAIDA_ANONIMIZADO = str(tmp_path / "{timestamp}_anonimizado_{sessao}_{modelo}.txt")
        mock_config.NOME_SESSAO = "sessao_teste"
        mock_config.VERDE = ""
        mock_config.AZUL = ""
        mock_config.RESET = ""

        mock_funcoes = MagicMock()
        mock_funcoes.identificar_caracteres_mascara.return_value = ("X", "Y")

        df = pd.DataFrame({"descricao_raw": ["Texto teste"] * 5, "descricao": ["Texto pre-processado"] * 5})

        gerador = GeradorRelatorio(
            anonimizacao=mock_anonimizacao, configuracao=mock_config, funcoes_gerais_module=mock_funcoes
        )

        # Mock datetime para o nome do arquivo ser previsível
        with patch("GeradorRelatorio.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            # O parâmetro silencioso impede a barra de progresso do pandarallel
            gerador.gerar(df, "modelo:v1", silencioso=True)

        # Verificações
        arq_original = tmp_path / "20240101_120000_original_sessao_teste_modelo-v1.txt"
        arq_anonimizado = tmp_path / "20240101_120000_anonimizado_sessao_teste_modelo-v1.txt"
        arq_mascara = tmp_path / "20240101_120000_mascara_sessao_teste_modelo-v1.mask"
        arq_meta = tmp_path / "20240101_120000_mascara_sessao_teste_modelo-v1.mask.meta"

        assert arq_original.exists()
        assert arq_anonimizado.exists()
        assert arq_mascara.exists()
        assert arq_meta.exists()

        # Verifica conteúdo (os mocks devem ter percorrido o DF 5 vezes e gravado o retorno deles)
        assert open(arq_original).read() == "Texto teste" * 5
        assert open(arq_anonimizado).read() == "Final formatado" * 5
        assert open(arq_mascara).read() == "Mascara gerada" * 5


