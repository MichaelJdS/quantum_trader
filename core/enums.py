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
    PENDING = "PENDING"
    OPEN = "OPEN"
    WON = "WON"
    LOST = "LOST"
    CANCELLED = "CANCELLED"


class StakeMode(str, Enum):
    FIXED = "FIXED"
    KELLY = "KELLY"
    ADAPTIVE = "ADAPTIVE"