"""
Testes unitários para o módulo gerar_metricas.py

Cobre:
- Carga de dados (JSON, mask, meta)
- Cálculo de métricas (TP, FP, FN, Precision, Recall, F1)
- Geração de buffer de ground truth
- Resiliência a buffers parciais
"""

import json
import os
import tempfile

import pytest

from anonymed.gerar_metricas import (
    calcular_metricas,
    carregar_json,
    carregar_mask_buffer,
    carregar_meta,
    gerar_buffer_groundtruth,
)

# =============================================
# Fixtures Locais
# =============================================


@pytest.fixture
def dataset_simples():
    """Dataset mínimo com 2 registros e labels posicionais."""
    return [
        {
            "prontuario": "P001",
            "descricao": "Paciente Maria Silva mora em Maceió.",
            #              01234567890123456789012345678901234567
            #              0         1         2         3
            "labels": [
                {"first_position": 9, "last_position": 21, "subcategory": "PATIENT", "word": "Maria Silva"},
                {"first_position": 30, "last_position": 36, "subcategory": "CITY", "word": "Maceió"},
            ],
        },
        {
            "prontuario": "P002",
            "descricao": "Dr. Santos atendeu aqui.",
            #              0123456789012345678901234
            "labels": [
                {"first_position": 4, "last_position": 10, "subcategory": "DOCTOR", "word": "Santos"},
            ],
        },
    ]


@pytest.fixture
def chars_teste():
    return ("█", "?")


def gerar_mascara_perfeita(dataset, char_pii):
    """Gera máscara que cobre exatamente todos os labels (100% recall)."""
    partes = []
    for reg in dataset:
        texto = list(reg["descricao"])
        for label in reg.get("labels", []):
            for i in range(label["first_position"], min(label["last_position"], len(texto))):
                texto[i] = char_pii
        partes.append("".join(texto))
    return "".join(partes)


def gerar_mascara_vazia(dataset):
    """Gera máscara sem nenhuma detecção (0% recall)."""
    return "".join(reg["descricao"] for reg in dataset)


# =============================================
# gerar_buffer_groundtruth
# =============================================


class TestGerarBufferGroundtruth:
    def test_substitui_labels_por_char(self, dataset_simples):
        gt = gerar_buffer_groundtruth(dataset_simples, "█")
        # "Maria Silva" (pos 9-21) deve ser mascarada
        assert "█" in gt

    def test_preserva_texto_normal(self, dataset_simples):
        gt = gerar_buffer_groundtruth(dataset_simples, "█")
        assert "Paciente" in gt  # Antes do primeiro label

    def test_filtro_por_categoria(self, dataset_simples):
        # Somente PATIENT
        gt = gerar_buffer_groundtruth(dataset_simples, "█", categorias={"PATIENT"})
        # "Maria Silva" mascarada
        pos_maria = 9
        assert gt[pos_maria] == "█"
        # "Maceió" NÃO mascarada (é CITY, não PATIENT)
        pos_maceio = 30
        assert gt[pos_maceio] != "█" or gt[pos_maceio] == "M"

    def test_tamanho_preservado(self, dataset_simples):
        """O buffer GT deve ter o mesmo tamanho que a concatenação dos textos."""
        gt = gerar_buffer_groundtruth(dataset_simples, "█")
        esperado = sum(len(r["descricao"]) for r in dataset_simples)
        assert len(gt) == esperado


# =============================================
# calcular_metricas
# =============================================


class TestCalcularMetricas:
    def test_recall_perfeito(self, dataset_simples, chars_teste):
        """Máscara perfeita deve ter recall = 1.0."""
        mascara = gerar_mascara_perfeita(dataset_simples, chars_teste[0])
        resultado = calcular_metricas(dataset_simples, mascara, chars_teste)
        assert resultado["metricas"]["recall"] == 1.0

    def test_recall_zero(self, dataset_simples, chars_teste):
        """Máscara vazia deve ter recall = 0.0."""
        mascara = gerar_mascara_vazia(dataset_simples)
        resultado = calcular_metricas(dataset_simples, mascara, chars_teste)
        assert resultado["metricas"]["recall"] == 0.0

    def test_precision_perfeita_sem_fp(self, dataset_simples, chars_teste):
        """Máscara perfeita sem detecções extras deve ter precision = 1.0."""
        mascara = gerar_mascara_perfeita(dataset_simples, chars_teste[0])
        resultado = calcular_metricas(dataset_simples, mascara, chars_teste)
        assert resultado["metricas"]["precision"] == 1.0

    def test_contagens_corretas(self, dataset_simples, chars_teste):
        """Deve contar labels relevantes corretamente."""
        mascara = gerar_mascara_perfeita(dataset_simples, chars_teste[0])
        resultado = calcular_metricas(dataset_simples, mascara, chars_teste)
        # 3 labels relevantes: PATIENT (Maria Silva), CITY (Maceió), DOCTOR (Santos)
        tp_total = resultado["contagens"]["tp_total"]
        fn = resultado["contagens"]["false_negatives"]
        assert tp_total + fn == 3

    def test_dataset_vazio(self, chars_teste):
        """Dataset vazio deve retornar dicionário vazio."""
        resultado = calcular_metricas([], "", chars_teste)
        assert resultado == {}

    def test_buffer_parcial(self, dataset_simples, chars_teste):
        """Deve avaliar apenas registros cobertos pelo buffer."""
        # Mascara só do primeiro registro
        mascara = gerar_mascara_perfeita(dataset_simples[:1], chars_teste[0])
        resultado = calcular_metricas(dataset_simples, mascara, chars_teste)
        assert resultado["config"]["registros_avaliados"] == 1

    def test_detalhe_por_categoria(self, dataset_simples, chars_teste):
        """Resultado deve incluir detalhamento por categoria."""
        mascara = gerar_mascara_perfeita(dataset_simples, chars_teste[0])
        resultado = calcular_metricas(dataset_simples, mascara, chars_teste)
        assert "detalhe_por_categoria" in resultado
        assert "PATIENT" in resultado["detalhe_por_categoria"]


# =============================================
# Carga de Arquivos
# =============================================


class TestCargaArquivos:
    def test_carregar_json_valido(self, dataset_simples):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(dataset_simples, f)
            caminho = f.name
        try:
            dados = carregar_json(caminho)
            assert len(dados) == 2
        finally:
            os.unlink(caminho)

    def test_carregar_json_inexistente(self):
        dados = carregar_json("/caminho/inexistente.json")
        assert dados == []

    def test_carregar_mask_valido(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mask", delete=False) as f:
            f.write("conteúdo da máscara")
            caminho = f.name
        try:
            conteudo = carregar_mask_buffer(caminho)
            assert conteudo == "conteúdo da máscara"
        finally:
            os.unlink(caminho)

    def test_carregar_meta_padrao(self):
        """Se meta não existir, deve retornar padrões ('&', '*')."""
        char_pii, char_unk = carregar_meta("/caminho/inexistente.mask")
        assert char_pii == "&"
        assert char_unk == "*"

    def test_carregar_meta_customizado(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mask.meta", delete=False) as f:
            json.dump({"char_pii": "█", "char_unk": "?"}, f)
            # O meta é buscado como mask_path + ".meta"
            # Então precisamos criar o nome correto
            caminho_meta = f.name
        try:
            # carregar_meta espera o caminho do .mask, e adiciona .meta
            caminho_mask = caminho_meta.replace(".mask.meta", ".mask")
            # Renomeia para simular
            os.rename(caminho_meta, caminho_mask + ".meta")
            char_pii, char_unk = carregar_meta(caminho_mask)
            assert char_pii == "█"
            assert char_unk == "?"
        finally:
            if os.path.exists(caminho_mask + ".meta"):
                os.unlink(caminho_mask + ".meta")


# =============================================
# FP Residual (detecção espúria)
# =============================================


class TestFalsoPositivo:
    def test_fp_deteccao_espuria(self):
        """Máscara com região mascarada fora dos labels deve contar como FP."""
        dataset = [
            {
                "prontuario": "P001",
                "descricao": "Texto sem dados pessoais aqui.",
                "labels": [],
            }
        ]
        # Mascara "sem" (3 chars, posição 6-8) → grupo de 3 chars ≥ MIN_CHARS_FP_GRUPO → 1 FP
        mascara = "Texto ███ dados pessoais aqui."
        assert len(mascara) == len(dataset[0]["descricao"])
        resultado = calcular_metricas(dataset, mascara, ("█", "?"))
        assert resultado["contagens"]["false_positives"] >= 1

    def test_fp_ignorado_se_grupo_curto(self):
        """Grupos de máscara < MIN_CHARS_FP_GRUPO não devem contar como FP."""
        dataset = [
            {
                "prontuario": "P001",
                "descricao": "Texto AB normal.",
                "labels": [],
            }
        ]
        # Mascara "AB" (2 chars) → abaixo do limiar → 0 FP
        mascara = "Texto ██ normal."
        assert len(mascara) == len(dataset[0]["descricao"])
        resultado = calcular_metricas(dataset, mascara, ("█", "?"))
        assert resultado["contagens"]["false_positives"] == 0


# =============================================
# TP Parcial (threshold de 50%)
# =============================================


class TestTpParcial:
    def test_tp_acima_de_50_porcento(self):
        """Label com >50% mascarado deve ser TP."""
        dataset = [
            {
                "prontuario": "P001",
                "descricao": "Paciente Maria mora aqui.",
                "labels": [
                    {
                        "first_position": 9,
                        "last_position": 14,
                        "subcategory": "PATIENT",
                        "word": "Maria",
                    }
                ],
            }
        ]
        # "Maria" (5 chars) → mascarar 3/5 = 60% → TP
        texto = list(dataset[0]["descricao"])
        texto[9] = "█"
        texto[10] = "█"
        texto[11] = "█"
        mascara = "".join(texto)
        resultado = calcular_metricas(dataset, mascara, ("█", "?"))
        assert resultado["contagens"]["tp_total"] == 1

    def test_fn_abaixo_de_50_porcento(self):
        """Label com <50% mascarado deve ser FN."""
        dataset = [
            {
                "prontuario": "P001",
                "descricao": "Paciente Maria mora aqui.",
                "labels": [
                    {
                        "first_position": 9,
                        "last_position": 14,
                        "subcategory": "PATIENT",
                        "word": "Maria",
                    }
                ],
            }
        ]
        # "Maria" (5 chars) → mascarar 2/5 = 40% → FN
        texto = list(dataset[0]["descricao"])
        texto[9] = "█"
        texto[10] = "█"
        mascara = "".join(texto)
        resultado = calcular_metricas(dataset, mascara, ("█", "?"))
        assert resultado["contagens"]["false_negatives"] == 1


# =============================================
# Consumo de categorias irrelevantes
# =============================================


class TestCategoriaIrrelevante:
    def test_age_consumido_nao_vira_fp(self):
        """Categoria AGE (irrelevante) mascarada deve ser consumida e não virar FP."""
        dataset = [
            {
                "prontuario": "P001",
                "descricao": "Paciente tem 45 anos.",
                "labels": [
                    {
                        "first_position": 14,
                        "last_position": 16,
                        "subcategory": "AGE",
                        "word": "45",
                    }
                ],
            }
        ]
        # Mascara "45" (2 chars) → AGE, consumido → não FP
        texto = list(dataset[0]["descricao"])
        texto[14] = "█"
        texto[15] = "█"
        mascara = "".join(texto)
        resultado = calcular_metricas(dataset, mascara, ("█", "?"))
        # "45" é consumido pelo label AGE, não deve aparecer como FP
        assert resultado["contagens"]["false_positives"] == 0


# =============================================
# End-to-end: Máscara → Score
# =============================================


class TestEndToEnd:
    def test_pipeline_mascara_metricas(self, chars_teste):
        """Testa o pipeline completo: Anonimizacao.gerar_mascara → calcular_metricas."""
        from Anonimizacao import Anonimizacao

        anonimizacao = Anonimizacao(tag_ini="⦃", tag_fim="⦄")

        dataset = [
            {
                "prontuario": "P001",
                "descricao": "Paciente Maria Silva mora em Maceió.",
                "labels": [
                    {
                        "first_position": 9,
                        "last_position": 21,
                        "subcategory": "PATIENT",
                        "word": "Maria Silva",
                    },
                    {
                        "first_position": 30,
                        "last_position": 36,
                        "subcategory": "CITY",
                        "word": "Maceió",
                    },
                ],
            },
            {
                "prontuario": "P002",
                "descricao": "Dr. Santos atendeu aqui.",
                "labels": [
                    {
                        "first_position": 4,
                        "last_position": 10,
                        "subcategory": "DOCTOR",
                        "word": "Santos",
                    }
                ],
            },
        ]

        # Registra os PIIs na Anonimizacao
        anonimizacao._registrar_e_get_tag("Maria Silva", "nome")
        anonimizacao._registrar_e_get_tag("Maceió", "endereco")
        anonimizacao._registrar_e_get_tag("Santos", "nome")

        # Gera buffer de máscara contínuo (como gerar_relatorio_txt faz)
        buffer = ""
        for r in dataset:
            buffer += anonimizacao.gerar_mascara(r["descricao"], chars_teste)

        # Calcula métricas
        resultado = calcular_metricas(dataset, buffer, chars_teste)

        # Validações
        assert resultado["metricas"]["recall"] == 1.0
        assert resultado["metricas"]["precision"] == 1.0
        assert resultado["metricas"]["f1_score"] == 1.0
        assert resultado["contagens"]["tp_total"] == 3
        assert resultado["contagens"]["false_negatives"] == 0
        assert resultado["contagens"]["false_positives"] == 0


# =============================================
# Validação com dados reais (subset)
# =============================================


class TestAlinhamentoDatasetReal:
    """Valida que o formato do dataset JSON real está alinhado com as expectativas do código."""

    @pytest.fixture
    def dataset_real_subset(self):
        """Carrega os primeiros 50 registros do dataset real."""
        caminho = os.path.join(
            os.path.dirname(__file__),
            "..",
            "anonymed",
            "datasets",
            "clinical_deid_FULL_SEM_TAGS.json",
        )
        if not os.path.exists(caminho):
            pytest.skip("Dataset real não encontrado (esperado em CI)")
        with open(caminho, encoding="utf-8") as f:
            ds = json.load(f)
        return ds[:50]

    def test_last_position_exclusiva(self, dataset_real_subset):
        """Valida que last_position é exclusiva: desc[fp:lp] == word."""
        total = 0
        match = 0
        for r in dataset_real_subset:
            for l in r.get("labels", []):
                total += 1
                word = l.get("word", "")
                fp = l["first_position"]
                lp = l["last_position"]
                trecho = r["descricao"][fp:lp]
                if word == trecho:
                    match += 1
        assert total > 0, "Dataset subset sem labels"
        taxa = match / total
        assert taxa > 0.95, f"Apenas {taxa:.1%} dos labels alinham com boundary exclusiva"

    def test_groundtruth_tamanho_alinhado(self, dataset_real_subset):
        """GT buffer deve ter mesmo tamanho que concatenação dos textos."""
        gt = gerar_buffer_groundtruth(dataset_real_subset, "█")
        esperado = sum(len(r.get("descricao", "")) for r in dataset_real_subset)
        assert len(gt) == esperado

    def test_mascara_perfeita_recall_100(self, dataset_real_subset):
        """Recall = 1.0 com máscara perfeita sobre o dataset real."""
        chars = ("█", "?")
        mascara = gerar_mascara_perfeita(dataset_real_subset, chars[0])
        resultado = calcular_metricas(dataset_real_subset, mascara, chars)
        assert resultado["metricas"]["recall"] == 1.0, (
            f"Recall deveria ser 1.0, mas é {resultado['metricas']['recall']}"
        )

    def test_mascara_vazia_recall_0(self, dataset_real_subset):
        """Recall = 0.0 com máscara vazia sobre o dataset real."""
        chars = ("█", "?")
        mascara = gerar_mascara_vazia(dataset_real_subset)
        resultado = calcular_metricas(dataset_real_subset, mascara, chars)
        assert resultado["metricas"]["recall"] == 0.0
