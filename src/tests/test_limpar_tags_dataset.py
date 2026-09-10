"""
Testes para limpar_tags_dataset.py — limpeza de tags XML do dataset original,
correção de espaços em words, e validação de sincronismo posicional.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "anonymed"))
from limpar_tags_dataset import limpar_tags_e_ajustar_labels

# =============================================
# Helpers
# =============================================


def criar_df(registros: list[dict]) -> pd.DataFrame:
    """Cria DataFrame a partir de lista de dicts com 'text' e 'labels'."""
    return pd.DataFrame(registros)


def validar_sincronismo(df: pd.DataFrame):
    """Valida que desc[fp:lp] == word para todos os labels no resultado."""
    for _, row in df.iterrows():
        text = row["text"]
        for label in row["labels"]:
            word = label["word"]
            fp = label["first_position"]
            lp = label["last_position"]
            trecho = text[fp:lp]
            assert trecho == word, f"Dessincronismo: desc[{fp}:{lp}]={trecho!r} != word={word!r}"


# =============================================
# Linha simples sem erro → preserva texto e labels
# =============================================


class TestRegistroValido:
    def test_tag_simples_removida(self):
        """Tags XML devem ser removidas, texto limpo preservado, posições ajustadas."""
        #         pos: 0123456789...
        # Texto orig: 'Paciente <PATIENT>Maria</PATIENT/> atendida.'
        #                        ^^^^^^^^^ = tag abertura (9 chars)
        #                                  Maria = pos 18-23
        df = criar_df(
            [
                {
                    "text": "Paciente <PATIENT>Maria</PATIENT/> atendida.",
                    "labels": [
                        {
                            "word": "Maria",
                            "category": "NAME",
                            "subcategory": "PATIENT",
                            "first_position": 18,
                            "last_position": 23,
                        }
                    ],
                }
            ]
        )
        resultado = limpar_tags_e_ajustar_labels(df)
        assert len(resultado) == 1
        row = resultado.iloc[0]
        assert "<PATIENT>" not in row["text"]
        assert "</PATIENT/>" not in row["text"]
        assert "Maria" in row["text"]
        validar_sincronismo(resultado)

    def test_multiplos_labels(self):
        """Múltiplos labels no mesmo registro devem ser processados corretamente."""
        # 'Dr. <DOCTOR>Silva</DOCTOR/> atende <PATIENT>Ana</PATIENT/>.'
        df = criar_df(
            [
                {
                    "text": "Dr. <DOCTOR>Silva</DOCTOR/> atende <PATIENT>Ana</PATIENT/>.",
                    "labels": [
                        {
                            "word": "Silva",
                            "category": "NAME",
                            "subcategory": "DOCTOR",
                            "first_position": 12,
                            "last_position": 17,
                        },
                        {
                            "word": "Ana",
                            "category": "NAME",
                            "subcategory": "PATIENT",
                            "first_position": 44,
                            "last_position": 47,
                        },
                    ],
                }
            ]
        )
        resultado = limpar_tags_e_ajustar_labels(df)
        assert len(resultado) == 1
        row = resultado.iloc[0]
        assert row["text"] == "Dr. Silva atende Ana."
        validar_sincronismo(resultado)

    def test_texto_sem_labels(self):
        """Registro sem labels deve manter texto limpo (sem tags)."""
        df = criar_df(
            [
                {
                    "text": "Texto sem dados sensíveis.",
                    "labels": [],
                }
            ]
        )
        resultado = limpar_tags_e_ajustar_labels(df)
        assert len(resultado) == 1
        assert resultado.iloc[0]["text"] == "Texto sem dados sensíveis."


# =============================================
# Correção de espaço em word
# =============================================


class TestCorrecaoEspacoWord:
    def test_word_com_espaco_leading(self):
        """Word com espaço no início deve ser trimado e posições ajustadas."""
        # 'Hospital <HOSPITAL> Jardim</HOSPITAL/> Municipal.'
        # A tag <HOSPITAL> termina em pos 19, espaço em 19, 'Jardim' começa em 20
        # O word original é ' Jardim' (com espaço), first_position=19, last_position=26
        df = criar_df(
            [
                {
                    "text": "Hospital <HOSPITAL> Jardim</HOSPITAL/> Municipal.",
                    "labels": [
                        {
                            "word": " Jardim",
                            "category": "LOCATION",
                            "subcategory": "HOSPITAL",
                            "first_position": 19,
                            "last_position": 26,
                        }
                    ],
                }
            ]
        )
        resultado = limpar_tags_e_ajustar_labels(df)
        assert len(resultado) == 1
        row = resultado.iloc[0]
        label = row["labels"][0]
        assert label["word"] == "Jardim"
        validar_sincronismo(resultado)

    def test_word_com_espaco_trailing(self):
        """Word com espaço no fim deve ser trimado."""
        # 'Rua <STREET>Centro </STREET/>número 10.'
        df = criar_df(
            [
                {
                    "text": "Rua <STREET>Centro </STREET/>número 10.",
                    "labels": [
                        {
                            "word": "Centro ",
                            "category": "LOCATION",
                            "subcategory": "STREET",
                            "first_position": 12,
                            "last_position": 19,
                        }
                    ],
                }
            ]
        )
        resultado = limpar_tags_e_ajustar_labels(df)
        assert len(resultado) == 1
        label = resultado.iloc[0]["labels"][0]
        assert label["word"] == "Centro"
        validar_sincronismo(resultado)

    def test_word_com_espaco_ambos_lados(self):
        """Word com espaço em ambos os lados deve ser trimado."""
        # 'Hospital <HOSPITAL> Jardim </HOSPITAL/>Sul.'
        df = criar_df(
            [
                {
                    "text": "Hospital <HOSPITAL> Jardim </HOSPITAL/>Sul.",
                    "labels": [
                        {
                            "word": " Jardim ",
                            "category": "LOCATION",
                            "subcategory": "HOSPITAL",
                            "first_position": 19,
                            "last_position": 27,
                        }
                    ],
                }
            ]
        )
        resultado = limpar_tags_e_ajustar_labels(df)
        label = resultado.iloc[0]["labels"][0]
        assert label["word"] == "Jardim"
        validar_sincronismo(resultado)


# =============================================
# Detecção de erros → linhas removidas
# =============================================


class TestLinhasComErro:
    def test_tag_abertura_ausente(self):
        """Linha sem tag de abertura deve ser marcada para remoção."""
        df = criar_df(
            [
                {
                    "text": "Paciente Maria</PATIENT> atendida.",
                    "labels": [
                        {
                            "word": "Maria",
                            "category": "NAME",
                            "subcategory": "PATIENT",
                            "first_position": 9,
                            "last_position": 14,
                        }
                    ],
                }
            ]
        )
        resultado = limpar_tags_e_ajustar_labels(df)
        assert len(resultado) == 0

    def test_tag_fechamento_ausente(self):
        """Linha sem tag de fechamento deve ser marcada para remoção."""
        df = criar_df(
            [
                {
                    "text": "Paciente <PATIENT>Maria atendida.",
                    "labels": [
                        {
                            "word": "Maria",
                            "category": "NAME",
                            "subcategory": "PATIENT",
                            "first_position": 18,
                            "last_position": 23,
                        }
                    ],
                }
            ]
        )
        resultado = limpar_tags_e_ajustar_labels(df)
        assert len(resultado) == 0

    def test_tags_sobrando(self):
        """Texto com tags XML não mapeadas deve ser removido."""
        df = criar_df(
            [
                {
                    "text": "Paciente <PATIENT>Maria</PATIENT/> de <EXTRA>info</EXTRA/>.",
                    "labels": [
                        {
                            "word": "Maria",
                            "category": "NAME",
                            "subcategory": "PATIENT",
                            "first_position": 18,
                            "last_position": 23,
                        }
                    ],
                }
            ]
        )
        resultado = limpar_tags_e_ajustar_labels(df)
        assert len(resultado) == 0


# =============================================
# Mix válidas/inválidas → filtragem correta
# =============================================


class TestFiltragemMultiplasLinhas:
    def test_mix_validas_invalidas(self):
        """Linhas válidas são mantidas, inválidas removidas."""
        df = criar_df(
            [
                {
                    "text": "Dr. <DOCTOR>Silva</DOCTOR/> ok.",
                    "labels": [
                        {
                            "word": "Silva",
                            "category": "NAME",
                            "subcategory": "DOCTOR",
                            "first_position": 12,
                            "last_position": 17,
                        }
                    ],
                },
                {
                    "text": "Paciente Maria</PATIENT/> erro.",
                    "labels": [
                        {
                            "word": "Maria",
                            "category": "NAME",
                            "subcategory": "PATIENT",
                            "first_position": 9,
                            "last_position": 14,
                        }
                    ],
                },
                {
                    "text": "Rua <STREET>Centro</STREET/> 10.",
                    "labels": [
                        {
                            "word": "Centro",
                            "category": "LOCATION",
                            "subcategory": "STREET",
                            "first_position": 12,
                            "last_position": 18,
                        }
                    ],
                },
            ]
        )
        resultado = limpar_tags_e_ajustar_labels(df)
        assert len(resultado) == 2
        validar_sincronismo(resultado)


# =============================================
# Sincronismo com dataset real (subset)
# =============================================


class TestSincronismoDatasetReal:
    """Valida sincronismo posicional no dataset FULL gerado."""

    @pytest.fixture
    def dataset_full_subset(self):
        caminho = os.path.join(
            os.path.dirname(__file__),
            "..",
            "anonymed",
            "datasets",
            "clinical_deid_FULL_SEM_TAGS.json",
        )
        if not os.path.exists(caminho):
            pytest.skip("Dataset FULL não encontrado")
        import json

        with open(caminho, encoding="utf-8") as f:
            ds = json.load(f)
        return ds[:100]

    def test_sincronismo_posicional(self, dataset_full_subset):
        """desc[fp:lp] == word para todos os labels nos primeiros 100 registros."""
        total = 0
        erros = 0
        for r in dataset_full_subset:
            desc = r.get("descricao", "")
            for l in r.get("labels", []):
                total += 1
                word = l.get("word", "")
                fp = l["first_position"]
                lp = l["last_position"]
                trecho = desc[fp:lp]
                if word != trecho:
                    erros += 1
        assert total > 0, "Dataset sem labels"
        assert erros == 0, f"{erros}/{total} labels dessincronizados"

    def test_nenhum_word_com_espaco(self, dataset_full_subset):
        """Nenhum word deve ter espaço no início ou fim no dataset processado."""
        count = 0
        for r in dataset_full_subset:
            for l in r.get("labels", []):
                w = l.get("word", "")
                if w != w.strip():
                    count += 1
        assert count == 0, f"{count} labels com espaço no word"
