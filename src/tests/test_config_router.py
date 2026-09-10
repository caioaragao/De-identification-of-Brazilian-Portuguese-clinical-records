import sys
import os
import socket
from unittest.mock import patch, MagicMock

import pytest

class TestConfigRouter:
    """Verifica se o config_router define _perfil_ativo corretamente baseado no socket."""
    
    def test_routing_para_ambiente_valido(self):
        """Se hostname for válido, ele mesmo é o perfil ativo."""
        with patch("socket.gethostname", return_value="servidor-hospital"):
            with patch("importlib.import_module") as mock_import:
                # Simula um módulo retornado com algumas variáveis
                mock_module = MagicMock()
                mock_module._variavel_lixo = "ignorar"
                mock_module.VARIAVEL_BOA = "manter"
                mock_import.return_value = mock_module
                
                # Importa de forma limpa
                import a01_platform.config_router as router
                import importlib
                importlib.reload(router)
                
                assert router._perfil_ativo == "servidor-hospital"
                mock_import.assert_called_with("a01_platform.servidor-hospital.config")
                
    def test_routing_fallback_para_vertex(self):
        """Se o hostname não constar na lista de perfis, cai no vertex."""
        with patch("socket.gethostname", return_value="5229de284b94"):
            with patch("importlib.import_module") as mock_import:
                # Simula um módulo retornado
                mock_import.return_value = MagicMock()
                
                import a01_platform.config_router as router
                import importlib
                importlib.reload(router)
                
                assert router._perfil_ativo == "vertex"
                mock_import.assert_called_with("a01_platform.vertex.config")
