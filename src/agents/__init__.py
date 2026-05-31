# agents module
# Este arquivo é crucial: ele dispara o carregamento dos agentes
from .base import BaseAgent
from .registry import register, instantiate_all, get_agent
from . import implementations  # 🚨 Linha obrigatória: executa os @register decorators

__all__ = ["BaseAgent", "register", "instantiate_all", "get_agent"]