from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import pandas as pd
from loguru import logger

from backtesting.metrics import PerformanceMetrics
from backtesting.vectorized_backtester import BacktestResult, BacktestTrade
from core.entities import RiskConfig, SessionState
from core.strategy_base import StrategyBase
from ml.feature_engineer import FeatureEngineer


class EventType(Enum):
    MARKET_DATA = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()
    CONTRACT_RESULT = auto()


@dataclass(order=True)
class Event:
    """Evento no barramento — ordenado por timestamp para simulação correta."""

    timestamp: float
    event_type: EventType = field(compare=False)
    data: dict[str, Any] = field(default_factory=dict, compare=False)


class EventBus:
    """Priority queue de eventos — garante ordem temporal."""

    def __init__(self) -> None:
        self._queue: list[Event] = []

    def publish(self, event: Event) -> None:
        heapq.heappush(self._queue, event)

    def consume(self) -> Event | None:
        return heapq.heappop(self._queue) if self._queue else None

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)


class EventDrivenBacktester:
    """
    Backtester event-driven com simulação fiel de latência e rejeições.

    Diferenças do vetorizado:
      - Cada candle gera um evento MARKET_DATA.
      - Sinal → evento SIGNAL → evento ORDER → evento FILL (com latência).
      - Permite simular: rejeições de proposta, timeout, slippage dinâmico.
      - Mais lento que o vetorizado (O(n log n)) mas mais realista.

    Quando usar:
      - Validação final de estratégia antes de ir para live.
      - Testar condições de borda (latência alta, mercado rápido).
    """

    FILL_LATENCY_CANDLES: int = 1  # Latência de execução = 1 candle.

    def __init__(
        self,
        risk_config: RiskConfig,
        payout_rate: float = 0.95,
        slippage_model: str = "fixed",
        rejection_rate: float = 0.0,
    ) -> None:
        self.risk_config = risk_config
        self.payout_rate = payout_rate
        self.slippage_model = slippage_model
        self.rejection_rate = rejection_rate
        self._fe = FeatureEngineer()

    def run(
        self,
        df: pd.DataFrame,
        strategy: StrategyBase,
        symbol: str = "R_50",
        granularity: int = 60,
        initial_balance: float = 1000.0,
    ) -> BacktestResult:
        """
        Executa backtesting event-driven para uma estratégia.
        """
        import random

        feat_df = self._fe.compute(df)
        bus = EventBus()
        trades: list[BacktestTrade] = []
        balance = initial_balance
        equity = []

        session = SessionState(
            initial_balance=initial_balance,
            current_balance=initial_balance,
        )

        # Injeta todos os eventos de mercado na fila.
        for i, (idx, row) in enumerate(feat_df.iterrows()):
            bus.publish(Event(
                timestamp=float(row.get("epoch", i)),
                event_type=EventType.MARKET_DATA,
                data={"idx": i, "row": row},
            ))

        pending_order: dict | None = None
        current_idx = 0

        while not bus.is_empty():
            event = bus.consume()

            if event.event_type == EventType.MARKET_DATA:
                current_idx = event.data["idx"]
                equity.append(balance)

                # Processa fill pendente (latência de 1 candle).
                if pending_order and current_idx >= pending_order["fill_at"]:
                    trade_data = pending_order
                    exit_price = float(feat_df["close"].iloc[current_idx])

                    won = (
                        exit_price > trade_data["entry_price"]
                        if trade_data["direction"] == "BUY"
                        else exit_price < trade_data["entry_price"]
                    )
                    pnl = (trade_data["stake"] * self.payout_rate) if won else -trade_data["stake"]
                    balance += pnl
                    balance = max(balance, 0.0)
                    session.current_balance = balance

                    trades.append(BacktestTrade(
                        entry_idx=trade_data["entry_idx"],
                        exit_idx=current_idx,
                        entry_price=trade_data["entry_price"],
                        exit_price=exit_price,
                        direction=trade_data["direction"],
                        strategy_name=strategy.name,
                        stake=trade_data["stake"],
                        pnl=pnl,
                        won=won,
                        confidence=trade_data["confidence"],
                        entry_time=trade_data["entry_time"],
                        exit_time=float(feat_df["epoch"].iloc[current_idx]),
                    ))
                    pending_order = None

                    if won:
                        session.wins += 1
                        session.consecutive_losses = 0
                    else:
                        session.losses += 1
                        session.consecutive_losses += 1
                    session.total_trades += 1
                    session.win_rate = session.wins / (session.total_trades or 1)

                # Gera sinal se não houver order pendente.
                if pending_order is None and current_idx >= 52:
                    window = feat_df.iloc[: current_idx + 1]
                    signal = strategy.generate_signal(window, symbol, session)

                    if signal:
                        # Simula rejeição de proposta.
                        if random.random() < self.rejection_rate:
                            logger.debug("Proposta rejeitada (simulado).", idx=current_idx)
                            continue

                        entry_price = float(feat_df["close"].iloc[current_idx])
                        slippage = self._compute_slippage(feat_df, current_idx)
                        if signal.direction.value == "BUY":
                            entry_price *= 1 + slippage
                        else:
                            entry_price *= 1 - slippage

                        stake = self._calc_stake(balance, signal.confidence)

                        pending_order = {
                            "entry_idx": current_idx,
                            "fill_at": current_idx + self.FILL_LATENCY_CANDLES,
                            "entry_price": entry_price,
                            "direction": signal.direction.value,
                            "stake": stake,
                            "confidence": signal.confidence,
                            "entry_time": float(feat_df["epoch"].iloc[current_idx]),
                        }

        while len(equity) < len(feat_df):
            equity.append(equity[-1] if equity else initial_balance)

        equity_series = pd.Series(equity[: len(feat_df)], name="equity")
        drawdown_series = self._compute_drawdown(equity_series)

        metrics = PerformanceMetrics.compute(
            trades=trades,
            equity_curve=equity_series,
            drawdown_series=drawdown_series,
            initial_balance=initial_balance,
            granularity=granularity,
            payout_rate=self.payout_rate,
        )

        logger.info(
            "Backtest event-driven concluído.",
            strategy=strategy.name,
            trades=len(trades),
            sharpe=round(metrics.get("sharpe_ratio", 0), 3),
        )

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            granularity=granularity,
            total_trades=len(trades),
            trades=trades,
            equity_curve=equity_series,
            drawdown_series=drawdown_series,
            metrics=metrics,
            params={"mode": "event_driven", "latency": self.FILL_LATENCY_CANDLES},
        )

    def _compute_slippage(self, df: pd.DataFrame, idx: int) -> float:
        """Slippage dinâmico proporcional à volatilidade do candle."""
        if self.slippage_model == "dynamic":
            atr = df["atr_14"].iloc[idx] if "atr_14" in df.columns else 0
            close = df["close"].iloc[idx]
            return min(float(atr / (close + 1e-10)) * 0.10, 0.005)
        return 0.0001  # Fixed 0.01%.

    def _calc_stake(self, balance: float, confidence: float) -> float:
        from backtesting.vectorized_backtester import VectorizedBacktester
        bt = VectorizedBacktester(self.risk_config, self.payout_rate)
        return bt._calc_stake(balance, confidence)

    @staticmethod
    def _compute_drawdown(equity: pd.Series) -> pd.Series:
        rolling_max = equity.expanding().max()
        return (equity - rolling_max) / (rolling_max + 1e-10)