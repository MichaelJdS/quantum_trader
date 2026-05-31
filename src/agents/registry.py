from typing import Type, Dict, List
from .base import BaseAgent

_registry: Dict[str, Type[BaseAgent]] = {}

def register(cls: Type[BaseAgent]) -> Type[BaseAgent]:
    """Decorator para registrar agentes automaticamente."""
    if cls.name in _registry:
        raise ValueError(f"Agent '{cls.name}' já está registrado.")
    _registry[cls.name] = cls
    return cls

def instantiate_all() -> List[BaseAgent]:
    """Cria instâncias de todos os agentes registrados."""
    if not _registry:
        raise RuntimeError("Nenhum agente registrado. Verifique se src/agents/implementations.py está sendo importado.")
    return [cls() for cls in _registry.values()]

def get_agent(name: str) -> BaseAgent:
    """Retorna um agente específico pelo nome."""
    if name not in _registry:
        raise KeyError(f"Agent '{name}' não encontrado.")
    return _registry[name]()

__all__ = ["register", "instantiate_all", "get_agent"]