from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.entities import RiskConfig, SessionState
from core.enums import StakeMode, TradeDirection
from core.strategies import BollingerReversionStrategy, BreakoutStrategy, EmaRsiStrategy
from ml.feature_engineer import FeatureEngineer


def make_ohlcv(n: int = 200, trend: str = "up") -> pd.DataFrame:
    """Gera DataFrame OHLCV sintético para testes."""
    np.random.seed(42)
    price = 1000.0
    rows = []
    for i in range(n):
        if trend == "up":
            change = np.random.normal(0.3, 1.5)
        elif trend == "down":
            change = np.random.normal(-0.3, 1.5)
        else:
            change = np.random.normal(0, 1.5)

        open_ = price
        close = price + change
        high = max(open_, close) + abs(np.random.normal(0, 0.5))
        low = min(open_, close) - abs(np.random.normal(0, 0.5))
        rows.append({
            "open": open_, "high": high, "low": low,
            "close": close, "epoch": 1_000_000 + i * 60,
        })
        price = close
    return pd.DataFrame(rows)


@pytest.fixture
def risk_config():
    return RiskConfig(
        stake_mode=StakeMode.FIXED,
        base_stake=1.0,
        stop_win_pct=0.05,
        stop_loss_pct=0.03,
        max_daily_drawdown_pct=0.05,
        max_consecutive_losses=5,
    )


@pytest.fixture
def session():
    return SessionState(initial_balance=100.0, current_balance=100.0)


@pytest.fixture
def feat_df():
    raw = make_ohlcv(200, "up")
    return FeatureEngineer().compute(raw)


def test_feature_engineer_columns(feat_df):
    """Verifica que features essenciais são geradas."""
    essential = {
        "ema_9", "ema_21", "ema_50", "rsi_14", "macd_hist",
        "adx", "bb_upper", "bb_lower", "atr_14", "is_squeeze",
    }
    assert essential.issubset(set(feat_df.columns))


def test_feature_engineer_no_nan(feat_df):
    assert not feat_df.isnull().any().any()


def test_ema_rsi_returns_signal_or_none(risk_config, session, feat_df):
    strategy = EmaRsiStrategy(risk_config=risk_config)
    result = strategy.generate_signal(feat_df, "R_50", session)
    assert result is None or result.direction in (TradeDirection.BUY, TradeDirection.SELL)


def test_bollinger_reversion_returns_signal_or_none(risk_config, session, feat_df):
    strategy = BollingerReversionStrategy(risk_config=risk_config)
    result = strategy.generate_signal(feat_df, "R_50", session)
    assert result is None or result.direction in (TradeDirection.BUY, TradeDirection.SELL)


def test_breakout_returns_signal_or_none(risk_config, session, feat_df):
    strategy = BreakoutStrategy(risk_config=risk_config)
    result = strategy.generate_signal(feat_df, "R_50", session)
    assert result is None or result.direction in (TradeDirection.BUY, TradeDirection.SELL)


def test_insufficient_data_returns_none(risk_config, session):
    strategy = EmaRsiStrategy(risk_config=risk_config)
    small_df = make_ohlcv(10)
    result = strategy.generate_signal(small_df, "R_50", session)
    assert result is None