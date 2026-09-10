"""
Router dinâmico de configuração por servidor.

Avalia o nome da máquina que executa o código e exporta as constantes do `config.py`
contido na pasta correspondente a esse ambiente. Se o hostname não constar em
`_PERFIS_VALIDOS`, adota-se o perfil `vertex` como fallback.
"""

import importlib
import socket

_HOSTNAME = socket.gethostname()

_PERFIS_VALIDOS = ["computador", "servidor-hospital", "servidor-universidade", "vertex"]
_perfil_ativo = _HOSTNAME if _HOSTNAME in _PERFIS_VALIDOS else "vertex"

try:
    _modulo = importlib.import_module(f"a01_platform.{_perfil_ativo}.config")
    # Exporta as constantes públicas do perfil para este namespace (equivalente a `from X import *`)
    globals().update({k: v for k, v in vars(_modulo).items() if not k.startswith("_")})
except Exception as e:
    raise ImportError(f"Falha ao injetar a configuração do ambiente '{_perfil_ativo}': {e}")
