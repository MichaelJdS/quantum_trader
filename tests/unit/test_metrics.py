from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.metrics import PerformanceMetrics
from backtesting.vectorized_backtester import BacktestTrade


def make_trades(n: int = 100, win_rate: float = 0.55, stake: float = 1.0, payout: float = 0.95) -> list[BacktestTrade]:
    trades = []
    for i in range(n):
        won = i / n < win_rate
        trades.append(BacktestTrade(
            entry_idx=i, exit_idx=i + 1,
            entry_price=1000.0, exit_price=1001.0,
            direction="BUY", strategy_name="test",
            stake=stake,
            pnl=(stake * payout) if won else -stake,
            won=won, confidence=0.60,
        ))
    return trades


def make_equity(trades: list[BacktestTrade], initial: float = 1000.0) -> pd.Series:
    bal = initial
    vals = [bal]
    for t in trades:
        bal += t.pnl
        vals.append(max(bal, 0))
    return pd.Series(vals)


def test_win_rate_computed_correctly():
    trades = make_trades(100, win_rate=0.60)
    equity = make_equity(trades)
    dd = PerformanceMetrics._compute_drawdown(equity)
    m = PerformanceMetrics.compute(trades, equity, dd, initial_balance=1000.0)
    assert abs(m["win_rate"] - 0.60) < 0.02


def test_profit_factor_above_1_for_positive_strategy():
    trades = make_trades(200, win_rate=0.60)
    equity = make_equity(trades)
    dd = PerformanceMetrics._compute_drawdown(equity)
    m = PerformanceMetrics.compute(trades, equity, dd, initial_balance=1000.0)
    assert m["profit_factor"] > 1.0


def test_no_nan_in_metrics():
    import math
    trades = make_trades(50, win_rate=0.50)
    equity = make_equity(trades)
    dd = PerformanceMetrics._compute_drawdown(equity)
    m = PerformanceMetrics.compute(trades, equity, dd, initial_balance=1000.0)
    for k, v in m.items():
        assert not math.isnan(v), f"NaN em {k}"
        assert not math.isinf(v), f"Inf em {k}"


def test_max_consecutive_losses():
    trades = [
        BacktestTrade(0, 1, 100, 101, "BUY", "t", 1, -1, False, 0.5),
        BacktestTrade(1, 2, 100, 101, "BUY", "t", 1, -1, False, 0.5),
        BacktestTrade(2, 3, 100, 101, "BUY", "t", 1, -1, False, 0.5),
        BacktestTrade(3, 4, 100, 101, "BUY", "t", 1, 0.95, True, 0.5),
        BacktestTrade(4, 5, 100, 101, "BUY", "t", 1, -1, False, 0.5),
        BacktestTrade(5, 6, 100, 101, "BUY", "t", 1, -1, False, 0.5),
    ]
    assert PerformanceMetrics._max_consecutive_losses(trades) == 3


@staticmethod
def _compute_drawdown(equity):
    rolling_max = equity.expanding().max()
    return (equity - rolling_max) / (rolling_max + 1e-10)


PerformanceMetrics._compute_drawdown = staticmethod(_compute_drawdown)