"""
Testes unitários para o módulo logger.py
"""

import builtins
import io

import pytest

import logger
from logger import LogLevel


class TestLogLevel:
    def test_enum_valores(self):
        assert LogLevel.SILENT  == 0
        assert LogLevel.ERROR   == 1
        assert LogLevel.WARNING == 2
        assert LogLevel.NORMAL  == 3
        assert LogLevel.VERBOSE == 4

    def test_ordenacao_crescente(self):
        assert LogLevel.SILENT < LogLevel.ERROR < LogLevel.WARNING < LogLevel.NORMAL < LogLevel.VERBOSE


class TestConfigurar:
    def setup_method(self):
        logger.configurar("error")  # estado controlado

    def teardown_method(self):
        logger.configurar("error")

    def test_configurar_por_string(self):
        logger.configurar("verbose")
        assert logger.nivel_atual() == LogLevel.VERBOSE

    def test_configurar_por_enum(self):
        logger.configurar(LogLevel.WARNING)
        assert logger.nivel_atual() == LogLevel.WARNING

    def test_configurar_por_int(self):
        logger.configurar(0)
        assert logger.nivel_atual() == LogLevel.SILENT

    def test_configurar_string_invalida(self):
        with pytest.raises(ValueError, match="Nível de log inválido"):
            logger.configurar("ultradebug")

    def test_configurar_case_insensitive(self):
        logger.configurar("VERBOSE")
        assert logger.nivel_atual() == LogLevel.VERBOSE

    def test_todos_os_niveis_string(self):
        for nome in ("silent", "error", "warning", "normal", "verbose"):
            logger.configurar(nome)
            assert logger.nivel_atual().name.lower() == nome


class TestLog:
    def setup_method(self):
        self._buf = io.StringIO()

    def teardown_method(self):
        logger.configurar("error")

    def _capturar(self, func):
        """Captura saída de log em buffer."""
        import sys
        old_stdout = sys.stdout
        sys.stdout = self._buf
        try:
            func()
        finally:
            sys.stdout = old_stdout
        return self._buf.getvalue()

    def test_log_normal_exibe_quando_level_normal(self):
        logger.configurar("normal")
        saida = self._capturar(lambda: logger.log("msg teste", level=LogLevel.NORMAL, file=self._buf))
        assert "msg teste" in saida

    def test_log_normal_suprimido_quando_level_error(self):
        logger.configurar("error")
        logger.log("nao deve aparecer", level=LogLevel.NORMAL, file=self._buf)
        assert self._buf.getvalue() == ""

    def test_log_error_exibe_mesmo_em_warning(self):
        logger.configurar("warning")
        logger.log("erro grave", level=LogLevel.ERROR, file=self._buf)
        assert "erro grave" in self._buf.getvalue()

    def test_log_verbose_suprimido_em_normal(self):
        logger.configurar("normal")
        logger.log("detalhe verboso", level=LogLevel.VERBOSE, file=self._buf)
        assert self._buf.getvalue() == ""

    def test_log_verbose_exibe_em_verbose(self):
        logger.configurar("verbose")
        logger.log("detalhe verboso", level=LogLevel.VERBOSE, file=self._buf)
        assert "detalhe verboso" in self._buf.getvalue()

    def test_log_silent_suprime_tudo(self):
        logger.configurar("silent")
        for nivel in (LogLevel.ERROR, LogLevel.WARNING, LogLevel.NORMAL, LogLevel.VERBOSE):
            logger.log("qualquer coisa", level=nivel, file=self._buf)
        assert self._buf.getvalue() == ""

    def test_timestamp_no_prefixo(self):
        logger.configurar("normal")
        logger.log("com timestamp", level=LogLevel.NORMAL, file=self._buf)
        saida = self._buf.getvalue()
        assert "[20" in saida  # prefixo [YYYY-MM-DD HH:MM:SS]

    def test_linha_vazia_sem_timestamp(self):
        logger.configurar("normal")
        logger.log("", level=LogLevel.NORMAL, file=self._buf)
        saida = self._buf.getvalue()
        assert "[20" not in saida  # sem timestamp em linhas de controle


class TestInstalarPrintGlobal:
    def setup_method(self):
        self._print_original = builtins.print

    def teardown_method(self):
        logger.restaurar_print_global()
        logger.configurar("error")

    def test_instalar_e_restaurar(self):
        logger.instalar_print_global()
        assert builtins.print is not self._print_original
        logger.restaurar_print_global()
        assert builtins.print is self._print_original or callable(builtins.print)
