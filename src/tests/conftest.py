"""
Fixtures compartilhadas para os testes do sistema de anonimização.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

import logger


def pytest_configure(config):
    """Configura o logger em nível ERROR para silenciar saída nos testes."""
    logger.configurar("error")


@pytest.fixture
def anonimizacao():
    """Retorna uma instância fresca de Anonimizacao com TAGs padrão."""
    from Anonimizacao import Anonimizacao

    return Anonimizacao(tag_ini="⦃", tag_fim="⦄")


@pytest.fixture
def anonimizacao_com_dados(anonimizacao):
    """Retorna instância com dados pré-carregados para testes de substituição."""
    anonimizacao._registrar_e_get_tag("Maria Silva", "nome")
    anonimizacao._registrar_e_get_tag("Dr. José Santos", "nome")
    anonimizacao._registrar_e_get_tag("Maceió", "endereco")
    anonimizacao._registrar_e_get_tag("Hospital Geral", "endereco")
    anonimizacao._registrar_e_get_tag("123.456.789-00", "cpf")
    anonimizacao.salvar_novo_boilerplate("Paciente em bom estado geral")
    return anonimizacao


@pytest.fixture
def df_prontuarios():
    """DataFrame de exemplo com prontuários fictícios para testes."""
    return pd.DataFrame(
        {
            "prontuario": ["P001", "P001", "P002", "P003"],
            "descricao": [
                "Paciente Maria Silva, 45 anos, natural de Maceió. CPF 123.456.789-00.",
                "Retorno da paciente Maria Silva. Paciente em bom estado geral.",
                "Dr. José Santos atendeu no Hospital Geral. Data: 15/03/2026.",
                "Paciente refere dor abdominal. Prescrito Dipirona 500mg 6/6h.",
            ],
        }
    )


@pytest.fixture
def mock_supabase():
    """Mock do cliente Supabase para testes sem banco de dados."""
    mock = MagicMock()
    mock.obter_ou_criar_sessao.return_value = 1
    mock.carregar_conhecimento_sessao.return_value = ({}, {}, {}, {}, {}, {}, -1, 102)
    mock.carregar_caracteres_tag.return_value = ("⦃", "⦄")
    mock.obter_detalhes_sessao.return_value = {
        "status": "iniciado",
        "nome_identificador": "teste",
        "modelo_llm": "qwen3:14b",
    }
    return mock


@pytest.fixture
def mock_ollama():
    """Mock do cliente Ollama para testes sem servidor LLM."""
    mock = MagicMock()
    mock.executar_prompt.return_value = iter(
        [
            {
                "controle": "OK",
                "raciocinio": "",
                "resposta": '{"map_nomes": [], "map_enderecos": [], "map_outros": [], "duvidas_pendentes": []}',
            }
        ]
    )
    return mock
