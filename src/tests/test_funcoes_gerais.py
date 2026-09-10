"""
Testes unitários para o módulo funcoes_gerais.py

Cobre:
- Cálculo de tokens (tiktoken)
- Pré-processamento de DataFrame
- Identificação de caracteres de máscara
- carregar_contador
- sanitizar_lista_tags
- gerar_combinacoes
- _merge_mapas
- sanitizar_nome_sessao
"""

import json

import pandas as pd
import pytest

from funcoes_gerais import (
    _agregar_metricas_json,
    _intersect_mapas,
    _merge_mapas,
    _parsear_modelo_de_nome_arquivo,
    _parsear_modo_de_nome_arquivo,
    _parsear_nivel_de_nome_arquivo,
    calcular_tokens,
    carregar_contador,
    gerar_combinacoes,
    identificar_caracteres_mascara,
    pre_processamento_geral,
    sanitizar_lista_tags,
    sanitizar_nome_sessao,
    verificar_alinhamento_arquivos,
)

# =============================================
# calcular_tokens
# =============================================


class TestCalcularTokens:
    def test_texto_simples(self):
        """Texto curto deve retornar número positivo de tokens."""
        resultado = calcular_tokens("Paciente em bom estado geral")
        assert resultado > 0

    def test_texto_vazio(self):
        """Texto vazio deve retornar 0 tokens."""
        resultado = calcular_tokens("")
        assert resultado == 0

    def test_texto_longo_tem_mais_tokens(self):
        """Texto mais longo deve ter mais tokens que texto curto."""
        curto = calcular_tokens("Olá")
        longo = calcular_tokens("Paciente refere dor abdominal persistente há 3 dias, sem melhora com Dipirona")
        assert longo > curto

    def test_texto_com_caracteres_especiais(self):
        """Deve funcionar com acentos e caracteres especiais do português."""
        resultado = calcular_tokens("Prescrição médica: Amoxicilina 500mg — 8/8h por 7 dias")
        assert resultado > 0

    def test_texto_com_unicode(self):
        """Deve funcionar com caracteres Unicode raros (TAGs do sistema)."""
        resultado = calcular_tokens("⦃ PERSON 1 ⦄ refere dor")
        assert resultado > 0

    def test_numero_puro(self):
        """Deve funcionar com entrada numérica (conversão implícita para str)."""
        resultado = calcular_tokens(12345)
        assert resultado > 0


# =============================================
# pre_processamento_geral
# =============================================


class TestPreProcessamentoGeral:
    def test_normaliza_quebras_de_linha(self):
        """Deve converter \\r\\n para \\n."""
        df = pd.DataFrame({"descricao": ["Texto\r\ncom\r\nquebras"]})
        resultado = pre_processamento_geral(df)
        assert "\r" not in resultado["descricao"].iloc[0]

    def test_remove_espacos_antes_de_newline(self):
        """Deve remover espaços/tabs logo antes de \\n."""
        df = pd.DataFrame({"descricao": ["Texto   \ncom espaço antes"]})
        resultado = pre_processamento_geral(df)
        assert "   \n" not in resultado["descricao"].iloc[0]

    def test_remove_espacos_depois_de_newline(self):
        """Deve remover espaços/tabs logo depois de \\n."""
        df = pd.DataFrame({"descricao": ["Texto\n   com espaço depois"]})
        resultado = pre_processamento_geral(df)
        assert "\n   " not in resultado["descricao"].iloc[0]

    def test_colapsa_multiplas_quebras(self):
        """3+ \\n consecutivos devem virar exatamente 2 \\n."""
        df = pd.DataFrame({"descricao": ["Parágrafo 1\n\n\n\n\nParágrafo 2"]})
        resultado = pre_processamento_geral(df)
        texto = resultado["descricao"].iloc[0]
        assert "\n\n\n" not in texto
        assert "\n\n" in texto

    def test_preserva_colchetes(self):
        """Colchetes NÃO devem ser removidos (decisão de design: TAGs dinâmicas)."""
        df = pd.DataFrame({"descricao": ["Texto com [brackets] e mais texto"]})
        resultado = pre_processamento_geral(df)
        assert "[brackets]" in resultado["descricao"].iloc[0]

    def test_campo_nan(self):
        """Campos NaN devem virar string vazia."""
        df = pd.DataFrame({"descricao": [None]})
        resultado = pre_processamento_geral(df)
        assert resultado["descricao"].iloc[0] == ""

    def test_strip_extremidades(self):
        """Deve remover espaços/quebras nas extremidades."""
        df = pd.DataFrame({"descricao": ["  \n  Texto central  \n  "]})
        resultado = pre_processamento_geral(df)
        assert resultado["descricao"].iloc[0] == "Texto central"


# =============================================
# identificar_caracteres_mascara
# =============================================


class TestIdentificarCaracteresMascara:
    def test_retorna_dois_caracteres_por_padrao(self):
        """Deve retornar uma tupla de 2 caracteres."""
        resultado = identificar_caracteres_mascara("Texto simples sem caracteres especiais")
        assert len(resultado) == 2

    def test_caracteres_nao_existem_no_texto(self):
        """Os caracteres retornados NÃO devem existir no texto de entrada."""
        texto = "Texto com vários caracteres comuns: @#$%&*"
        resultado = identificar_caracteres_mascara(texto)
        for char in resultado:
            assert char not in texto

    def test_caracteres_sao_diferentes_entre_si(self):
        """Os dois caracteres retornados devem ser distintos."""
        resultado = identificar_caracteres_mascara("Texto qualquer")
        assert resultado[0] != resultado[1]

    def test_prioriza_unicode_raro(self):
        """Deve preferir chars Unicode raros (⦃, ⦄) quando disponíveis."""
        resultado = identificar_caracteres_mascara("Texto sem caracteres Unicode")
        # Os dois primeiros candidatos são ⦃ e ⦄
        assert resultado[0] == "⦃"
        assert resultado[1] == "⦄"

    def test_evita_chars_presentes_no_texto(self):
        """Se ⦃ já existe no texto, deve pular para o próximo candidato."""
        texto = "Texto com ⦃ e ⦄ já usados"
        resultado = identificar_caracteres_mascara(texto)
        assert "⦃" not in resultado
        assert "⦄" not in resultado

    def test_erro_quando_nao_ha_candidatos(self):
        """Deve lançar ValueError se todos os candidatos estiverem no texto."""
        # Cria texto que contém TODOS os candidatos
        todos_candidatos = "⦃⦄⟦⟧⟨⟩❮❯*#@&%§¶†‡ΨΩβΣ"
        with pytest.raises(ValueError, match="Não foi possível encontrar"):
            identificar_caracteres_mascara(todos_candidatos, 2)

    def test_quantidade_customizada(self):
        """Deve respeitar o parâmetro quantidade."""
        resultado = identificar_caracteres_mascara("Texto", quantidade=3)
        assert len(resultado) == 3


# =============================================
# carregar_contador
# =============================================


class TestCarregarContador:
    def test_mapa_vazio(self):
        assert carregar_contador({}) == 1

    def test_mapa_com_valores(self):
        assert carregar_contador({"a": 1, "b": 5, "c": 3}) == 6

    def test_mapa_com_unico(self):
        assert carregar_contador({"x": 10}) == 11


# =============================================
# sanitizar_lista_tags
# =============================================


class TestSanitizarListaTags:
    def test_lista_vazia(self):
        assert sanitizar_lista_tags([], "texto qualquer") == []

    def test_lista_none(self):
        assert sanitizar_lista_tags(None, "texto qualquer") == []

    def test_mantem_dados_reais(self):
        """Termos que não parecem tags devem ser mantidos."""
        resultado = sanitizar_lista_tags(["Maria Silva", "Maceió"], "algum texto")
        assert resultado == ["Maria Silva", "Maceió"]

    def test_remove_tag_vazada_com_colchetes_no_texto(self):
        """'PERSON 311' deve ser removido se [ PERSON 311 ] existir no texto."""
        texto = "O paciente [ PERSON 311 ] recebeu alta."
        resultado = sanitizar_lista_tags(["PERSON 311"], texto)
        assert resultado == []

    def test_mantem_tag_sem_colchetes_no_texto(self):
        """'PERSON 311' mantido se NÃO houver [ PERSON 311 ] no texto (pode ser dado real)."""
        texto = "Código PERSON 311 não encontrado."
        resultado = sanitizar_lista_tags(["PERSON 311"], texto)
        assert resultado == ["PERSON 311"]

    def test_remove_multiplas_tags_vazadas(self):
        """Vários tipos de tags devem ser removidos se presentes entre colchetes."""
        texto = "[ BOILERPLATE 5 ] texto [ LOCATION 12 ]"
        resultado = sanitizar_lista_tags(["BOILERPLATE 5", "LOCATION 12", "Maria"], texto)
        assert resultado == ["Maria"]

    def test_case_insensitive(self):
        """Regex de tags deve ser case insensitive."""
        texto = "[ person 42 ] está aqui"
        resultado = sanitizar_lista_tags(["PERSON 42"], texto)
        assert resultado == []

    def test_mantem_texto_que_parece_tag_mas_nao_esta_no_contexto(self):
        """EMAIL 20 mantido se não existe como tag no texto."""
        texto = "O código EMAIL 20 é do sistema."
        resultado = sanitizar_lista_tags(["EMAIL 20"], texto)
        assert resultado == ["EMAIL 20"]


# =============================================
# gerar_combinacoes (Multi-Sessão)
# =============================================


class TestGerarCombinacoes:
    def test_unico_id(self):
        assert gerar_combinacoes([63]) == [(63,)]

    def test_dois_ids(self):
        result = gerar_combinacoes([63, 65])
        assert result == [(63,), (65,), (63, 65)]

    def test_tres_ids(self):
        result = gerar_combinacoes([63, 65, 72])
        assert len(result) == 7  # 2^3 - 1
        assert (63,) in result
        assert (65,) in result
        assert (72,) in result
        assert (63, 65) in result
        assert (63, 72) in result
        assert (65, 72) in result
        assert (63, 65, 72) in result


# =============================================
# _merge_mapas (Multi-Sessão)
# =============================================


class TestMergeMapas:
    def test_sem_sobreposicao(self):
        """Mapas disjuntos: todos os itens entram na união."""
        m1 = {"Maria": 1, "João": 2}
        m2 = {"Dr. Santos": 1, "Ana": 2}
        result = _merge_mapas([m1, m2])
        assert len(result) == 4
        assert "Maria" in result
        assert "Dr. Santos" in result

    def test_com_sobreposicao(self):
        """Chave duplicada aparece uma vez, ID reindexado."""
        m1 = {"Maria": 1, "João": 2}
        m2 = {"Maria": 5, "Ana": 3}
        result = _merge_mapas([m1, m2])
        assert len(result) == 3  # Maria apenas 1x
        assert "Maria" in result
        assert "João" in result
        assert "Ana" in result

    def test_vazio(self):
        result = _merge_mapas([{}, {}])
        assert result == {}

    def test_ids_sequenciais(self):
        """IDs devem ser sequenciais de 1 a N."""
        m1 = {"A": 10, "B": 20}
        m2 = {"C": 30}
        result = _merge_mapas([m1, m2])
        assert sorted(result.values()) == [1, 2, 3]


# =============================================
# _intersect_mapas (Multi-Sessão)
# =============================================


class TestIntersectMapas:
    def test_sem_sobreposicao(self):
        """Mapas sem chaves comuns: resultado deve ser vazio."""
        m1 = {"Maria": 1, "João": 2}
        m2 = {"Dr. Santos": 1, "Ana": 2}
        assert _intersect_mapas([m1, m2]) == {}

    def test_sobreposicao_total(self):
        """Todos os mapas com as mesmas chaves: retorna todas reindexadas."""
        m1 = {"Maria": 1, "João": 2}
        m2 = {"Maria": 5, "João": 9}
        result = _intersect_mapas([m1, m2])
        assert set(result.keys()) == {"Maria", "João"}

    def test_sobreposicao_parcial(self):
        """Apenas chaves presentes em TODOS os mapas são mantidas."""
        m1 = {"Maria": 1, "João": 2, "Ana": 3}
        m2 = {"Maria": 5, "Ana": 6}
        m3 = {"Maria": 7, "Carlos": 8}
        result = _intersect_mapas([m1, m2, m3])
        assert set(result.keys()) == {"Maria"}

    def test_mapa_vazio_resulta_em_vazio(self):
        """Um mapa vazio em qualquer posição zera o resultado."""
        m1 = {"Maria": 1}
        m2 = {}
        assert _intersect_mapas([m1, m2]) == {}

    def test_lista_vazia(self):
        """Lista sem mapas retorna {}."""
        assert _intersect_mapas([]) == {}

    def test_unico_mapa_retorna_proprio(self):
        """Um único mapa: retorna os termos reindexados."""
        m = {"Ana": 10, "Maria": 20}
        result = _intersect_mapas([m])
        assert set(result.keys()) == {"Ana", "Maria"}

    def test_ids_sequenciais(self):
        """IDs do resultado devem ser sequenciais a partir de 1."""
        m1 = {"A": 10, "B": 20, "C": 30}
        m2 = {"B": 1, "C": 2, "D": 3}
        result = _intersect_mapas([m1, m2])
        assert set(result.keys()) == {"B", "C"}
        assert sorted(result.values()) == [1, 2]


# =============================================
# sanitizar_nome_sessao (Multi-Sessão)
# =============================================


class TestSanitizarNomeSessao:
    def test_qwen(self):
        assert sanitizar_nome_sessao("alfa 6 [qwen3.5:4b]") == "alfa-6-qwen3.5-4b"

    def test_deepseek(self):
        assert sanitizar_nome_sessao("alfa 4 [deepseek-r1:14b]") == "alfa-4-deepseek-r1-14b"

    def test_simples(self):
        assert sanitizar_nome_sessao("teste_123") == "teste_123"

    def test_sessao_desconhecida(self):
        assert sanitizar_nome_sessao("Sessao_42") == "Sessao_42"


# =============================================
# Plotagem de Relatórios
# =============================================


class TestPlotagemRelatorios:
    def test_parsear_modelo_de_nome_arquivo(self):
        nome_arquivo = "20260311_234536_mascara_s57-s63-s65_alfa-5-gpt-oss-120b-alfa-5-deepseek-r1-70b.json"
        modelo = _parsear_modelo_de_nome_arquivo(nome_arquivo)
        assert modelo == "alfa-5-gpt-oss-120b-alfa-5-deepseek-r1-70b"

        nome_curto = "20260311_232256_mascara_s57_alfa-5-gpt-oss-120b.json"
        modelo = _parsear_modelo_de_nome_arquivo(nome_curto)
        assert modelo == "alfa-5-gpt-oss-120b"

    def test_parsear_modo_uniao(self):
        """Arquivo com prefixo 'uniao_' deve retornar 'Uniao'."""
        nome = "20260408_120000_mascara_s57-s58_uniao_alfa-57-alfa-58.json"
        assert _parsear_modo_de_nome_arquivo(nome) == "Uniao"

    def test_parsear_modo_interseccao(self):
        """Arquivo com prefixo 'interseccao_' deve retornar 'Interseccao'."""
        nome = "20260408_130000_mascara_s57-s58_interseccao_alfa-57-alfa-58.json"
        assert _parsear_modo_de_nome_arquivo(nome) == "Interseccao"

    def test_parsear_modo_unico(self):
        """Arquivo com prefixo 'unico_' deve retornar 'Unico'."""
        nome = "20260408_130000_mascara_s57_unico_alfa-57.json"
        assert _parsear_modo_de_nome_arquivo(nome) == "Unico"

    def test_parsear_modo_default_uniao(self):
        """Arquivo sem prefixo reconhecido deve retornar 'Uniao' (default)."""
        nome = "1111_2222_mascara_s1_modelo-A.json"
        assert _parsear_modo_de_nome_arquivo(nome) == "Uniao"

    def test_parsear_nivel_k1(self):
        """IDs com umúnico símbolo de sessão = k=1."""
        nome = "20260408_120000_mascara_s57_uniao_alfa-57.json"
        assert _parsear_nivel_de_nome_arquivo(nome) == 1

    def test_parsear_nivel_k2(self):
        """IDs com dois símbolos de sessão = k=2."""
        nome = "20260408_120000_mascara_s57-s58_uniao_alfa-57-alfa-58.json"
        assert _parsear_nivel_de_nome_arquivo(nome) == 2

    def test_parsear_nivel_k3(self):
        """IDs com três símbolos de sessão = k=3."""
        nome = "20260408_120000_mascara_s57-s58-s59_uniao_alfa-57-alfa-58-alfa-59.json"
        assert _parsear_nivel_de_nome_arquivo(nome) == 3

    def test_agregar_metricas_json(self, tmp_path):
        import json

        arquivo_1 = tmp_path / "1111_2222_mascara_s1_uniao_modelo-A.json"
        arquivo_2 = tmp_path / "1111_2222_mascara_s1-s2_uniao_modelo-B.json"
        arquivo_3 = tmp_path / "1111_2222_mascara_s1-s2_interseccao_modelo-C.json"

        arquivo_1.write_text(json.dumps({"metricas": {"precision": 0.8, "recall": 0.9, "f1_score": 0.85, "f2_score": 0.88}}))
        arquivo_2.write_text(json.dumps({"metricas": {"precision": 0.9, "recall": 0.95, "f1_score": 0.92, "f2_score": 0.94}}))
        arquivo_3.write_text(json.dumps({"metricas": {"precision": 0.95, "recall": 0.95, "f1_score": 0.95, "f2_score": 0.95}}))

        df = _agregar_metricas_json(str(tmp_path))

        assert len(df) == 3
        # Colunas novas devem existir
        assert "Modo" in df.columns
        assert "Nivel" in df.columns

        # Verifica que os modos foram parseados corretamente
        modos = set(df["Modo"].tolist())
        assert "Uniao" in modos
        assert "Interseccao" in modos

        # Verifica níveis
        nivels = set(df["Nivel"].tolist())
        assert 1 in nivels   # s1 = k=1
        assert 2 in nivels   # s1-s2 = k=2

        # Ordenacao por F2_Score decrescente
        assert df.iloc[0]["F2_Score"] >= df.iloc[1]["F2_Score"] >= df.iloc[2]["F2_Score"]


# =============================================
# BUG-5: verificar_alinhamento_arquivos deve usar encoding="utf-8"
# =============================================


class TestVerificarAlinhamento:
    """
    Verifica que verificar_alinhamento_arquivos lê arquivos com encoding UTF-8
    explícito, necessário para prontuários com acentos e máscaras com chars
    Unicode (TAGs dinâmicas como ⦿).
    """

    def _criar_arquivos(self, tmp_path, original: str, mascara: str, char_pii: str, char_unk: str):
        arq_orig = tmp_path / "original.txt"
        arq_mask = tmp_path / "original.mask"
        arq_meta = tmp_path / "original.mask.meta"
        arq_orig.write_text(original, encoding="utf-8")
        arq_mask.write_text(mascara, encoding="utf-8")
        arq_meta.write_text(json.dumps({"char_pii": char_pii, "char_unk": char_unk}), encoding="utf-8")
        return str(arq_orig), str(arq_mask)

    def test_alinhamento_perfeito_retorna_zero(self, tmp_path):
        """Texto e máscara perfeitamente alinhados retornam 0 desalinhamentos."""
        texto = "Paciente com dor."
        mascara = "Paciente com dor."
        orig, mask = self._criar_arquivos(tmp_path, texto, mascara, "█", "?")
        assert verificar_alinhamento_arquivos(orig, mask) == 0

    def test_pii_mascarado_nao_conta_desalinhamento(self, tmp_path):
        """Posições com char_pii são consideradas alinhadas."""
        texto = "CPF 123.456.789-00 ok."
        mascara = "CPF ██████████████ ok."
        orig, mask = self._criar_arquivos(tmp_path, texto, mascara, "█", "?")
        assert verificar_alinhamento_arquivos(orig, mask) == 0

    def test_char_desalinhado_e_contado(self, tmp_path):
        """Caractere que não é o original nem char_pii/unk conta como desalinhamento."""
        texto = "AB"
        mascara = "AX"  # 'X' não é original 'B' nem char_pii nem char_unk
        orig, mask = self._criar_arquivos(tmp_path, texto, mascara, "█", "?")
        assert verificar_alinhamento_arquivos(orig, mask) == 1

    def test_com_acentos_e_unicode(self, tmp_path):
        """Texto com acentos e máscara com char Unicode não lança UnicodeDecodeError."""
        texto = "Paciente João com febre."
        char_pii = "⦿"
        mascara = "Paciente " + char_pii * 4 + " com febre."
        orig, mask = self._criar_arquivos(tmp_path, texto, mascara, char_pii, "?")
        assert verificar_alinhamento_arquivos(orig, mask) == 0
