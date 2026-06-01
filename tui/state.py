from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.execution_engine import ExecutionEngine
    from core.entities import Trade


@dataclass
class TUIState:
    """
    Estado compartilhado entre o ExecutionEngine e a TUI.

    Atualizado pelo engine a cada tick e lido pela TUI a cada 1s.
    Thread-safe via asyncio (single-threaded event loop).
    """

    # Conta / sessão.
    balance: float = 0.0
    initial_balance: float = 0.0
    session_pnl: float = 0.0
    session_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    dry_run: bool = True

    # Métricas de trading.
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    consecutive_losses: int = 0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    # Trades.
    open_trades: list[dict] = field(default_factory=list)
    recent_trades: list[dict] = field(default_factory=list)  # Últimos 50.
    all_trades: list[dict] = field(default_factory=list)    # Últimos 500.

    # Símbolos.
    symbol_data: dict[str, dict] = field(default_factory=dict)
    active_symbols: list[str] = field(default_factory=list)

    # Risco.
    risk_metrics: dict[str, Any] = field(default_factory=dict)
    stop_win_reached: bool = False
    stop_loss_reached: bool = False

    # Neurônios.
    neuron_summary: dict[str, Any] = field(default_factory=dict)

    # Online learner.
    online_metrics: dict[str, dict] = field(default_factory=dict)

    # Ensemble.
    ensemble_weights: dict[str, float] = field(default_factory=dict)

    # Últimas mensagens / alertas.
    alerts: list[str] = field(default_factory=list)

    def sync_from_engine(self, engine: "ExecutionEngine") -> None:
        """Sincroniza com o estado atual do ExecutionEngine."""
        sess = engine.session_state
        self.balance = sess.current_balance
        self.initial_balance = sess.initial_balance
        self.session_pnl = sess.current_balance - sess.initial_balance
        self.total_trades = sess.total_trades
        self.wins = sess.wins
        self.losses = sess.losses
        self.win_rate = sess.win_rate
        self.consecutive_losses = sess.consecutive_losses
        metrics = engine.metrics
        self.max_drawdown = metrics.get("max_drawdown", 0.0)
        self.sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        self.risk_metrics = metrics

    @property
    def balance_str(self) -> str:
        return f"${self.balance:,.2f}"

    @property
    def pnl_str(self) -> str:
        sign = "+" if self.session_pnl >= 0 else ""
        return f"{sign}${self.session_pnl:,.2f}"

    @property
    def win_rate_str(self) -> str:
        return f"{self.win_rate:.1%}"

    @property
    def session_duration(self) -> str:
        delta = datetime.now(tz=timezone.utc) - self.started_at
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"