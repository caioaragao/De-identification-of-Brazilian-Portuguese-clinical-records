import pytest
from unittest.mock import MagicMock, patch
from Supabase import Supabase

@pytest.fixture
def mock_supabase():
    with patch("Supabase.create_client") as mock_create:
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        # Usa um arquivo json mockado para nao quebrar o __init__
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value={"url": "http://mock", "key": "mock"}):
                supa = Supabase("dummy_path")
                yield supa, mock_client

def test_carregar_ids_combos_processados(mock_supabase):
    supa, m_client = mock_supabase
    
    # Prepara um retorno de banco de dados mockado
    m_res = MagicMock()
    m_res.data = [
        {"label_combo": "uniao_A-B", "f2_score": 0.85, "podado": False},
        {"label_combo": "uniao_B-C", "f2_score": 0.0, "podado": True},
    ]
    # Simula mock_client.table().select().eq().execute()
    m_client.table.return_value.select.return_value.eq.return_value.execute.return_value = m_res

    cache = supa.carregar_ids_combos_processados("uniao")
    
    assert cache["uniao_A-B"] == 0.85
    assert cache["uniao_B-C"] == -1.0


def test_salvar_combinacao_sucesso(mock_supabase):
    supa, m_client = mock_supabase

    # Configs do mock de upsert do combo matriz
    m_res_combo = MagicMock()
    m_res_combo.data = [{"id": 999}]
    m_client.table.return_value.upsert.return_value.execute.return_value = m_res_combo

    # Chama salvar_combinacao
    metricas = {"precision": 0.9, "recall": 0.8, "f1_score": 0.84, "f2_score": 0.81}
    novo_id = supa.salvar_combinacao(
        sessao_ids=[1, 2],
        modo="uniao",
        nivel_k=2,
        label_combo="uniao_alfa-beta",
        nome_arquivo="dummy.mask",
        metricas=metricas,
        podado=False
    )
    
    assert novo_id == 999
    # Verifica o table upsert pro pivot foi chamado
    m_client.table.assert_any_call("p02_combinacoes_sessoes")
