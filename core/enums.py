"""
Enums do domínio Quantum Trader.
"""
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TradeDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ContractType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"
    DIGITOVER = "DIGITOVER"
    DIGITUNDER = "DIGITUNDER"


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    WON = "WON"
    LOST = "LOST"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class StakeMode(str, Enum):
    FIXED = "FIXED"
    ADAPTIVE = "ADAPTIVE"
    KELLY = "KELLY"
    FRACTIONAL = "FRACTIONAL"
    FRACTIONAL_KELLY = "FRACTIONAL_KELLY"


class SignalStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"