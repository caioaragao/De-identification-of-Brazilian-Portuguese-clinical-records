import os
from itertools import combinations

import pytest

from main import _encontrar_mask_existente, mover_arquivos_para_pasta


@pytest.fixture
def temp_output_dir(tmp_path):
    di = tmp_path / "saida"
    di.mkdir()
    return str(di)


def test_mover_arquivos_excluidos_apenas_corretos(temp_output_dir):
    """
    Testa se a função move EXATAMENTE os arquivos prefixados pelo mask rejeitado
    e os arquivos '_original_' e '_anonimizado_' associados, deixando o resto intacto.
    """
    # Arquivos do combo rejeitado 57-59
    base_mask_rejeitado = "20260408_000000_mascara_s57-s59_modelo_mock.mask"
    arquivos_rejeitados = [
        base_mask_rejeitado,
        base_mask_rejeitado + ".meta",
        base_mask_rejeitado.replace(".mask", ".json"),
        base_mask_rejeitado.replace(".mask", ".fp.json"),
        base_mask_rejeitado.replace("_mascara_", "_original_").replace(".mask", ".txt"),
        base_mask_rejeitado.replace("_mascara_", "_anonimizado_").replace(".mask", ".txt"),
    ]

    # Arquivos de um combo válido que DEVE ficar (57-60)
    base_mask_bom = "20260408_000000_mascara_s57-s60_modelo_mock.mask"
    arquivos_bons = [
        base_mask_bom,
        base_mask_bom + ".meta",
        base_mask_bom.replace(".mask", ".json"),
        base_mask_bom.replace("_mascara_", "_original_").replace(".mask", ".txt"),
        base_mask_bom.replace("_mascara_", "_anonimizado_").replace(".mask", ".txt"),
    ]

    # Cria dummy files
    for a in arquivos_rejeitados + arquivos_bons:
        with open(os.path.join(temp_output_dir, a), "w") as f:
            f.write("dummy")

    # Caminho do parametro mk_path
    mk_path = os.path.join(temp_output_dir, base_mask_rejeitado)

    movidos = mover_arquivos_para_pasta(mk_path, "01-ARQUIVOS-EXCLUIDOS")

    assert movidos == len(arquivos_rejeitados), f"Moveu {movidos}, esperado {len(arquivos_rejeitados)}"

    out_folder = os.path.join(temp_output_dir, "01-ARQUIVOS-EXCLUIDOS")
    assert os.path.exists(out_folder)

    # Verifica os que foram movidos
    for arq in arquivos_rejeitados:
        assert os.path.exists(os.path.join(out_folder, arq)), f"Arquivo excluido nao encontrado na lixeira: {arq}"

    # Verifica os que NÃO devem ter sido movidos (ainda na raiz)
    for arq in arquivos_bons:
        assert os.path.exists(os.path.join(temp_output_dir, arq)), f"Arquivo util foi movido indevidamente: {arq}"
        assert not os.path.exists(os.path.join(out_folder, arq)), f"Arquivo util cruzou para lixeira: {arq}"


def test_mover_arquivos_evitar_colisao_substring(temp_output_dir):
    """
    Garante que ao podar 's57', não mova arquivos do 's570'.
    O algoritmo usa prefixos, mas como as IDs são s{id}, s57- e s570- são distintos.
    """
    rejeitado = "20260408_000000_mascara_s57_modelo.mask"
    vizinho_longo = "20260408_000000_mascara_s570_modelo.mask"

    for a in [rejeitado, vizinho_longo]:
        with open(os.path.join(temp_output_dir, a), "w") as f:
            f.write("data")

    mover_arquivos_para_pasta(os.path.join(temp_output_dir, rejeitado), "01-ARQUIVOS-EXCLUIDOS")

    out_folder = os.path.join(temp_output_dir, "01-ARQUIVOS-EXCLUIDOS")
    assert os.path.exists(os.path.join(out_folder, rejeitado))
    assert os.path.exists(os.path.join(temp_output_dir, vizinho_longo)), "s570 foi movido por engano ao podar s57!"


def test_apriori_subset_combination_logic():
    """
    Testa a lógia estrutural de branch and bound do nivel K.
    Se no nivel 2 o (1, 2) foi falho (não adicionado ao historico),
    no nivel 3 o (1, 2, 3) nem pode ser gerado.
    """
    # Mock do estado de historico_f2 que seria preenchido na vida real
    historico_f2 = {
        (1,): 0.90,
        (2,): 0.85,
        (3,): 0.80,
        (1, 3): 0.95, # Aprovado
        (2, 3): 0.88, # Aprovado
        # (1, 2) não está, ou seja, F2 dele diminuiu em relação a max(A,B), logo sofreu pruning!
    }

    # Tentativa de formar level k=3
    sessao_ids = [1, 2, 3]
    k = 3
    todas_possiveis = list(combinations(sessao_ids, k))
    combos_atuais = []

    for combo in todas_possiveis:
        subcombos = list(combinations(combo, k - 1))
        valido = True
        for sc in subcombos:
            if tuple(sorted(sc)) not in historico_f2:
                valido = False
                break
        if valido:
            combos_atuais.append(combo)

    # Assert: Se (1, 2) falhou (ausente do dict base), (1, 2, 3) morre sumariamente
    assert combos_atuais == [], "A combinação de Nível 3 deveria ser podada porque subset (1,2) não existia!"


def test_apriori_subset_combination_success():
    """Testa aprovação quando sub-ramos foram todos de sucesso."""
    historico_f2 = {
        (1,): 0.9, (2,): 0.8, (3,): 0.8,
        (1, 2): 0.91, (1, 3): 0.92, (2, 3): 0.86
    }

    sessao_ids = [1, 2, 3]
    k = 3
    todas_possiveis = list(combinations(sessao_ids, k))
    combos_atuais = []

    for combo in todas_possiveis:
        subcombos = list(combinations(combo, k - 1))
        valido = True
        for sc in subcombos:
            if tuple(sorted(sc)) not in historico_f2:
                valido = False
                break
        if valido:
            combos_atuais.append(combo)

    assert combos_atuais == [(1, 2, 3)], "A combinação plena deveria progredir"


def test_encontrar_mask_nao_existente(temp_output_dir):
    """Retorna (None, False) quando não há nenhum arquivo na pasta."""
    resultado, excluido = _encontrar_mask_existente(temp_output_dir, "s57", "uniao_s57")
    assert resultado is None
    assert excluido is False


def test_encontrar_mask_na_saida_ignora_timestamp(temp_output_dir):
    """Detecta mask na pasta de saída independente do timestamp no nome."""
    nome = "20260401_120000_mascara_s57_uniao_s57.mask"
    with open(os.path.join(temp_output_dir, nome), "w") as f:
        f.write("mask")

    caminho, excluido = _encontrar_mask_existente(temp_output_dir, "s57", "uniao_s57")
    assert caminho is not None
    assert os.path.basename(caminho) == nome
    assert excluido is False


def test_encontrar_mask_prioriza_excluidos(temp_output_dir):
    """Se o arquivo existe tanto na saída quanto nos excluídos, retorna o excluído com flag=True."""
    pasta_excluidos = os.path.join(temp_output_dir, "01-ARQUIVOS-EXCLUIDOS")
    os.makedirs(pasta_excluidos)

    nome_saida = "20260401_120000_mascara_s57_uniao_s57.mask"
    nome_excluido = "20260402_090000_mascara_s57_uniao_s57.mask"

    with open(os.path.join(temp_output_dir, nome_saida), "w") as f:
        f.write("mask saida")
    with open(os.path.join(pasta_excluidos, nome_excluido), "w") as f:
        f.write("mask excluido")

    caminho, excluido = _encontrar_mask_existente(temp_output_dir, "s57", "uniao_s57")
    assert excluido is True
    assert os.path.basename(caminho) == nome_excluido
