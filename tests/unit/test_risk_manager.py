from __future__ import annotations

import pytest

from core.entities import RiskConfig, SessionState, Trade
from core.enums import ContractType, StakeMode, TradeDirection, TradeStatus
from core.exceptions import StopLossReachedError, StopWinReachedError
from core.risk_manager import RiskManager


def make_trade(pnl: float) -> Trade:
    return Trade(
        symbol="R_50",
        stake=1.0,
        contract_type=ContractType.CALL,
        direction=TradeDirection.BUY,
        status=TradeStatus.WON if pnl > 0 else TradeStatus.LOST,
        confidence=0.7,
        strategy_name="test",
        model_name="test",
        pnl=pnl,
    )


def test_fixed_stake(default_risk_config, default_session):
    rm = RiskManager(config=default_risk_config, initial_balance=100.0)
    stake = rm.calculate_stake(current_balance=100.0)
    assert stake == default_risk_config.base_stake


def test_stop_loss_blocks_trade(default_risk_config):
    session = SessionState(initial_balance=100.0, current_balance=94.0)
    rm = RiskManager(config=default_risk_config, initial_balance=100.0)
    ok, reason = rm.can_trade(session)
    assert not ok
    assert "Stop Loss" in reason


def test_stop_win_blocks_trade(default_risk_config):
    session = SessionState(initial_balance=100.0, current_balance=106.0)
    rm = RiskManager(config=default_risk_config, initial_balance=100.0)
    ok, reason = rm.can_trade(session)
    assert not ok
    assert "Stop Win" in reason


def test_consecutive_losses_block(default_risk_config, default_session):
    rm = RiskManager(config=default_risk_config, initial_balance=100.0)
    for _ in range(default_risk_config.max_consecutive_losses):
        rm.register_trade(make_trade(-1.0))
    ok, reason = rm.can_trade(default_session)
    assert not ok
    assert "perdas consecutivas" in reason


def test_kelly_stake_positive(default_risk_config):
    cfg = RiskConfig(
        stake_mode=StakeMode.KELLY,
        base_stake=1.0,
        stop_win_pct=0.05,
        stop_loss_pct=0.03,
        max_daily_drawdown_pct=0.05,
        max_consecutive_losses=5,
        kelly_fraction=0.25,
    )
    rm = RiskManager(config=cfg, initial_balance=100.0)
    stake = rm.calculate_stake(100.0, win_probability=0.55, payout_multiplier=2.0)
    assert stake > 0.35


def test_assert_raises_stop_loss(default_risk_config):
    session = SessionState(initial_balance=100.0, current_balance=94.0)
    rm = RiskManager(config=default_risk_config, initial_balance=100.0)
    with pytest.raises(StopLossReachedError):
        rm.assert_can_trade(session)