"""
Testes para comportamento de log com timestamp (anteriormente testava _timestamped_print de main.py).
Agora testa o módulo logger.py que substituiu aquele mecanismo.
"""

import io

import logger
from logger import LogLevel


def test_log_normal_adiciona_timestamp(capsys):
    """
    logger.log() adiciona timestamp [YYYY-MM-DD HH:MM:SS] ao início da mensagem.
    """
    logger.configurar("normal")
    buf = io.StringIO()
    logger.log("Teste de mensagem comum", level=LogLevel.NORMAL, file=buf)
    output = buf.getvalue()

    assert output.startswith("[")
    assert "]" in output
    assert "Teste de mensagem comum" in output


def test_log_vazio_nao_adiciona_timestamp():
    """
    Strings vazias passam sem timestamp para manter compatibilidade com barras
    de progresso e separadores.
    """
    buf = io.StringIO()
    logger.configurar("normal")
    logger.log("", level=LogLevel.NORMAL, file=buf)
    output = buf.getvalue()

    assert "[20" not in output  # sem timestamp


def test_log_barra_progresso_escape_timestamp():
    """
    Mensagens com \\r (carriage return) não recebem timestamp.
    """
    buf = io.StringIO()
    logger.configurar("normal")
    logger.log("\r100% Completo", level=LogLevel.NORMAL, file=buf, end="")
    output = buf.getvalue()

    assert output == "\r100% Completo"
    assert "[" not in output
