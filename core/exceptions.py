from __future__ import annotations


class QuantumTraderError(Exception):
    """Exceção base do projeto."""


class ConfigurationError(QuantumTraderError):
    """Configuração inválida ou ausente."""


class DerivConnectionError(QuantumTraderError):
    """Falha na conexão com a API Deriv."""


class DerivAuthorizationError(QuantumTraderError):
    """Falha na autenticação com a API Deriv."""


class DerivContractError(QuantumTraderError):
    """Erro ao obter proposta ou executar contrato."""


class RiskViolationError(QuantumTraderError):
    """Operação bloqueada por regra de risco."""


class StopWinReachedError(QuantumTraderError):
    """Stop Win atingido na sessão."""


class StopLossReachedError(QuantumTraderError):
    """Stop Loss atingido na sessão."""


class DatabaseError(QuantumTraderError):
    """Erro de acesso ou integridade no banco de dados."""


class ModelNotReadyError(QuantumTraderError):
    """Modelo ML ainda não treinado ou carregado."""


class SymbolNotSupportedError(QuantumTraderError):
    """Símbolo não suportado pelo sistema."""