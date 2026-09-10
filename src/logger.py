"""
Sistema centralizado de log com níveis de verbosidade.

Níveis disponíveis (crescente):
    silent  = 0 — nada é exibido
    error   = 1 — erros fatais e exceções
    warning = 2 — avisos, falhas LLM, timeouts, retries
    normal  = 3 — progresso principal (padrão)
    verbose = 4 — debug, prompts, tokens, batches

Uso básico:
    from logger import log, LogLevel, configurar
    log("Mensagem normal")
    log("Aviso importante", LogLevel.WARNING)
    log("[DEBUG] detalhe", LogLevel.VERBOSE)

Integração com entrypoint CLI:
    import logger
    logger.configurar("verbose")          # via config ou argparse
    logger.instalar_print_global()        # redireciona builtins.print → log(NORMAL)
"""

import builtins
import sys
from datetime import datetime
from enum import IntEnum

# ---------------------------------------------------------------------------
# Enum de níveis
# ---------------------------------------------------------------------------

class LogLevel(IntEnum):
    SILENT  = 0
    ERROR   = 1
    WARNING = 2
    NORMAL  = 3
    VERBOSE = 4


_LEVEL_BY_NAME: dict[str, LogLevel] = {v.name.lower(): v for v in LogLevel}

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------

_current_level: LogLevel = LogLevel.NORMAL
_original_print = builtins.print


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def configurar(level: "LogLevel | str | int") -> None:
    """
    Define o nível de log global.

    Args:
        level: LogLevel enum, string ('silent','error','warning','normal','verbose')
               ou inteiro 0-4.
    """
    global _current_level
    if isinstance(level, str):
        level_lower = level.strip().lower()
        if level_lower not in _LEVEL_BY_NAME:
            raise ValueError(
                f"Nível de log inválido: '{level}'. "
                f"Use: {list(_LEVEL_BY_NAME.keys())}"
            )
        _current_level = _LEVEL_BY_NAME[level_lower]
    else:
        _current_level = LogLevel(int(level))


def nivel_atual() -> LogLevel:
    """Retorna o nível de log configurado atualmente."""
    return _current_level


def log(
    *args,
    level: LogLevel = LogLevel.NORMAL,
    sep: str = " ",
    end: str = "\n",
    file=None,
    flush: bool = True,
) -> None:
    """
    Imprime uma mensagem se level <= nível configurado.
    Prefixo de timestamp é adicionado exceto para linhas de controle
    (\\r, \\n, string vazia) para manter progresso de linha.

    Args:
        *args: Argumentos da mensagem (igual a print()).
        level: Nível de severidade da mensagem.
        sep, end, file, flush: Idêntico a print().
    """
    if level > _current_level:
        return

    msg = sep.join(str(a) for a in args)
    out = file or sys.stdout

    # Linhas de controle (carriage return, empty) sem timestamp
    if not msg or msg.startswith("\r") or msg in ("\n", ""):
        _original_print(msg, end=end, file=out, flush=flush)
        return

    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _original_print(f"[{agora}] {msg}", end=end, file=out, flush=flush)


def instalar_print_global() -> None:
    """
    Substitui builtins.print por uma função que chama log(NORMAL).
    Deve ser chamado apenas no entrypoint principal (main.py).
    Não deve ser chamado em testes.
    """
    import functools

    def _print_interceptado(*args, **kwargs):
        sep   = kwargs.get("sep", " ")
        end   = kwargs.get("end", "\n")
        file  = kwargs.get("file", None)
        flush = kwargs.get("flush", True)
        msg   = sep.join(str(a) for a in args)
        log(msg, level=LogLevel.NORMAL, end=end, file=file, flush=flush)

    builtins.print = _print_interceptado


def restaurar_print_global() -> None:
    """Restaura builtins.print ao original (útil em testes)."""
    builtins.print = _original_print
