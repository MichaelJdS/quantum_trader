from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from core.enums import ContractType, StakeMode, TradeDirection, TradeStatus


@dataclass(slots=True)
class SymbolConfig:
    name: str
    enabled: bool = True
    granularity: int = 60


@dataclass(slots=True)
class RiskConfig:
    stake_mode: StakeMode
    base_stake: float
    stop_win_pct: float
    stop_loss_pct: float
    max_daily_drawdown_pct: float
    max_consecutive_losses: int
    kelly_fraction: float = 0.25


@dataclass(slots=True)
class Signal:
    symbol: str
    direction: TradeDirection
    confidence: float
    strategy_name: str
    model_name: str
    contract_type: ContractType
    entry_price: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass(slots=True)
class Trade:
    symbol: str
    stake: float
    contract_type: ContractType
    direction: TradeDirection
    status: TradeStatus
    confidence: float
    strategy_name: str
    model_name: str
    id: str = field(default_factory=lambda: str(uuid4()))
    entry_price: float | None = None
    exit_price: float | None = None
    payout: float | None = None
    pnl: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    closed_at: datetime | None = None


@dataclass(slots=True)
class SessionState:
    initial_balance: float
    current_balance: float
    win_rate: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0