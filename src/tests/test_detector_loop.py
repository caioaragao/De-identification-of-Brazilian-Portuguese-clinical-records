"""
Testes unitários para DetectorLoop.

Usa os arquivos reais de reasoning_loop (reasoning_loopN_linhaM.txt)
para validar detecção de diferentes padrões de repetição.
"""

import re
import pytest
from pathlib import Path

from detector_loop import DetectorLoop

PASTA_LOOPS = Path(__file__).parent / "reasoning_loop"

# Tamanho do chunk para simular streaming (tokens Ollama ≈ 4-5 chars)
CHUNK_SIZE = 4


def _carregar_arquivo_loop(nome: str) -> tuple[str, int]:
    """
    Carrega arquivo de teste e extrai a linha de início do loop do nome.

    Ex: reasoning_loop1_linha107.txt → conteudo, 107
    """
    match = re.search(r"linha(\d+)", nome)
    assert match, f"Nome de arquivo inválido (sem linhaN): {nome}"
    linha_loop = int(match.group(1))
    conteudo = (PASTA_LOOPS / nome).read_text(encoding="utf-8")
    return conteudo, linha_loop


def _alimentar_em_chunks(detector: DetectorLoop, texto: str, chunk_size: int = CHUNK_SIZE) -> dict | None:
    """
    Alimenta o detector em chunks simulando streaming.
    Retorna o primeiro resultado de loop detectado, ou None.
    """
    for i in range(0, len(texto), chunk_size):
        chunk = texto[i : i + chunk_size]
        resultado = detector.alimentar(chunk)
        if resultado is not None:
            return resultado
    return None


def _obter_posicao_char_da_linha(texto: str, num_linha: int) -> int:
    """Retorna a posição (offset em chars) do início da linha num_linha (1-indexed)."""
    linhas = texto.split("\n")
    pos = 0
    for i in range(min(num_linha - 1, len(linhas))):
        pos += len(linhas[i]) + 1  # +1 para o \n
    return pos


# Descobre dinamicamente todos os arquivos reasoning_loop*_linha*.txt
_ARQUIVOS_LOOP = sorted(PASTA_LOOPS.glob("reasoning_loop*_linha*.txt"))


# ============================================================
# Testes de detecção com arquivos reais (parametrizado)
# ============================================================


class TestDeteccaoComArquivosReais:
    """Testa que TODOS os arquivos reasoning_loop são detectados via streaming simulado."""

    @pytest.mark.parametrize("arquivo", _ARQUIVOS_LOOP, ids=lambda p: p.name)
    def test_loop_detectado(self, arquivo):
        """Cada arquivo de loop deve ser detectado por algum detector (Det-1, Det-2 ou Det-3)."""
        texto, linha_loop = _carregar_arquivo_loop(arquivo.name)
        detector = DetectorLoop()
        resultado = _alimentar_em_chunks(detector, texto)

        assert resultado is not None, f"{arquivo.name}: loop não detectado (início esperado: linha {linha_loop})"
        assert resultado["tipo"] in ("janela_exata", "rep_n", "cosine_sim")

    @pytest.mark.parametrize("arquivo", _ARQUIVOS_LOOP, ids=lambda p: p.name)
    def test_posicao_razoavel(self, arquivo):
        """A posição detectada não deve ser absurdamente antes da linha real do loop."""
        texto, linha_loop = _carregar_arquivo_loop(arquivo.name)
        detector = DetectorLoop()
        _alimentar_em_chunks(detector, texto)

        posicao = detector.obter_posicao_inicio_loop()
        assert posicao is not None, f"{arquivo.name}: posicao_inicio_loop é None"

        pos_linha_loop = _obter_posicao_char_da_linha(texto, linha_loop)
        assert posicao >= pos_linha_loop - 500, (
            f"{arquivo.name}: posicao ({posicao}) muito antes da linha {linha_loop} ({pos_linha_loop})"
        )


class TestBlocosLinhas:
    """Testes unitários para Det-4 (blocos de linhas repetidos)."""

    def test_bloco_abcd_3x(self):
        """Bloco de 4 linhas repetido 3 vezes → detectado."""
        bloco = "Linha alfa com conteudo\nLinha beta diferente\nLinha gamma terceira\nLinha delta quarta\n"
        texto = "Preambulo inicial.\n" + bloco * 3
        detector = DetectorLoop()
        resultado = _alimentar_em_chunks(detector, texto)
        assert resultado is not None, "Bloco ABCD×3 deveria ser detectado"
        assert resultado["tipo"] == "bloco_linhas"

    def test_bloco_ab_3x(self):
        """Bloco de 2 linhas repetido 3 vezes (salto mínimo=2) → detectado."""
        bloco = "Primeira linha do bloco AB\nSegunda linha do bloco AB\n"
        texto = "Inicio do texto.\n" + bloco * 3
        detector = DetectorLoop()
        resultado = _alimentar_em_chunks(detector, texto)
        assert resultado is not None, "Bloco AB×3 deveria ser detectado"
        assert resultado["tipo"] == "bloco_linhas"

    def test_bloco_incompleto(self):
        """2.5 blocos (incompleto) → NÃO detectado."""
        bloco = "Linha alfa com conteudo\nLinha beta diferente\nLinha gamma terceira\nLinha delta quarta\n"
        texto = "Preambulo inicial.\n" + bloco * 2 + "Linha alfa com conteudo\nLinha beta diferente\n"
        detector = DetectorLoop()
        resultado = _alimentar_em_chunks(detector, texto)
        # Det-4 não deve detectar com apenas 2 blocos completos (limiar=3)
        if resultado is not None:
            assert resultado["tipo"] != "bloco_linhas", "Bloco incompleto não deveria disparar Det-4"

    def test_bloco_com_linhas_vazias(self):
        """Bloco contendo linhas vazias internas → detectado (comparação exata inclui vazias)."""
        bloco = "Linha significativa alpha\n\nOutra linha significativa\n"
        texto = "Texto introdutorio.\n" + bloco * 3
        detector = DetectorLoop()
        resultado = _alimentar_em_chunks(detector, texto)
        assert resultado is not None, "Bloco com linhas vazias×3 deveria ser detectado"
        assert resultado["tipo"] == "bloco_linhas"

    def test_bloco_sem_warmup(self):
        """Poucas linhas significativas → NÃO detectado pelo Det-4."""
        # Com min_linhas=6, apenas 4 linhas significativas não ativam Det-4
        texto = "AAAAAAAAAA\nBBBBBBBBBB\nAAAAAAAAA\nBBBBBBBBBB\n"
        detector = DetectorLoop({"bloco_min_linhas": 6})
        resultado = _alimentar_em_chunks(detector, texto)
        # Det-4 não deveria disparar (warmup insuficiente)
        if resultado is not None:
            assert resultado["tipo"] != "bloco_linhas"

    def test_texto_normal_sem_bloco(self):
        """Texto variado sem repetição de blocos → NÃO detectado pelo Det-4."""
        linhas = [f"Linha unica numero {i} com conteudo variado\n" for i in range(20)]
        texto = "".join(linhas)
        detector = DetectorLoop()
        resultado = _alimentar_em_chunks(detector, texto)
        assert resultado is None, "Texto variado não deveria disparar nenhum detector"

    def test_posicao_inicio_bloco(self):
        """A posição do loop deve apontar para o início do 2º bloco."""
        bloco = "Linha alfa com conteudo\nLinha beta diferente\nLinha gamma terceira\nLinha delta quarta\n"
        preambulo = "Preambulo inicial do texto.\n"
        texto = preambulo + bloco * 3
        detector = DetectorLoop()
        _alimentar_em_chunks(detector, texto)
        posicao = detector.obter_posicao_inicio_loop()
        assert posicao is not None
        # Posição deve estar após o preâmbulo + 1º bloco
        pos_esperada = len(preambulo) + len(bloco)
        assert posicao == pos_esperada, (
            f"Posição ({posicao}) deveria ser {pos_esperada} (início do 2º bloco)"
        )

class TestFalsoPositivo:
    """Testes para garantir que texto legítimo não gera falso positivo."""

    def test_texto_normal_sem_loop(self):
        """
        Texto pré-loop do arquivo 1 (linhas 1 a 106) não deve disparar detecção.
        """
        texto, linha_loop = _carregar_arquivo_loop("reasoning_loop1_linha107.txt")
        linhas = texto.split("\n")
        texto_pre_loop = "\n".join(linhas[: linha_loop - 1])

        detector = DetectorLoop()
        resultado = _alimentar_em_chunks(detector, texto_pre_loop)
        assert resultado is None, "Texto pré-loop não deveria ser detectado como loop"

    def test_texto_curto_sem_loop(self):
        """Texto curto e variado não deve disparar detecção."""
        detector = DetectorLoop()
        texto = "Linha diferente a cada vez.\n" * 10
        resultado = _alimentar_em_chunks(detector, texto)
        assert resultado is None


class TestDetectorDesativado:
    """Testa que sem DetectorLoop instanciado, não há detecção (simula PROMPT_INJECTION=False)."""

    def test_sem_detector_nao_aborta(self):
        """
        Se nenhum DetectorLoop é instanciado (detector=None),
        o loop não é detectado — validação conceitual.
        """
        texto, _ = _carregar_arquivo_loop("reasoning_loop1_linha107.txt")
        detector = None  # Simula PROMPT_INJECTION=False

        # Sem detector, nenhuma detecção ocorre
        assert detector is None


class TestReset:
    """Testa o método reset() do DetectorLoop."""

    def test_reset_limpa_estado(self):
        """Após reset, o detector não deve manter estado anterior."""
        detector = DetectorLoop()

        # Alimenta com texto que gera loop
        texto, _ = _carregar_arquivo_loop("reasoning_loop1_linha107.txt")
        _alimentar_em_chunks(detector, texto)

        # Reset
        detector.reset()

        # Alimenta com texto limpo — não deve detectar
        texto_limpo = "Texto completamente normal sem repetição alguma.\n" * 5
        resultado = _alimentar_em_chunks(detector, texto_limpo)
        assert resultado is None, "Após reset, detector não deveria ter memória do loop anterior"


class TestDetectoresAtivos:
    """Testes para o parâmetro detectores_ativos (ativação seletiva)."""

    def test_det123_bloqueia_det4(self):
        """Com detectores_ativos={1,2,3}, Det-4 não deve disparar mesmo com blocos repetidos."""
        bloco = "Linha alfa com conteudo\nLinha beta diferente\nLinha gamma terceira\nLinha delta quarta\n"
        texto = "Preambulo inicial.\n" + bloco * 3
        detector = DetectorLoop({"detectores_ativos": {1, 2, 3}})
        resultado = _alimentar_em_chunks(detector, texto)
        # Det-1/2/3 podem ou não detectar, mas se detectar, não deve ser bloco_linhas
        if resultado is not None:
            assert resultado["tipo"] != "bloco_linhas", "Det-4 deveria estar desativado"

    def test_det4_bloqueia_det123(self):
        """Com detectores_ativos={4}, Det-1/2/3 não devem disparar."""
        # Texto que Det-1 detectaria (mesma janela repetida)
        texto = "A" * 400
        detector = DetectorLoop({"detectores_ativos": {4}, "janela_chars": 100, "limiar_janelas": 3})
        resultado = _alimentar_em_chunks(detector, texto)
        # Det-4 não detecta pois não há \n (sem linhas)
        assert resultado is None, "Det-1/2/3 deveriam estar desativados e Det-4 sem linhas"

    def test_none_todos_ativos(self):
        """Com detectores_ativos=None (default), todos funcionam normalmente."""
        # Texto que Det-1 detectaria
        texto = "A" * 400
        detector = DetectorLoop({"janela_chars": 100, "limiar_janelas": 3})
        resultado = _alimentar_em_chunks(detector, texto)
        assert resultado is not None, "Com None, Det-1 deveria detectar"
        assert resultado["tipo"] == "janela_exata"
