from __future__ import annotations

import asyncio
import os

import pytest

# Força .env de teste para não usar credenciais reais.
os.environ.setdefault("DERIV_APP_ID", "test_app_id")
os.environ.setdefault("DERIV_API_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_quantum.db")
os.environ.setdefault("APP_ENV", "development")


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def default_risk_config():
    from core.entities import RiskConfig
    from core.enums import StakeMode
    return RiskConfig(
        stake_mode=StakeMode.FIXED,
        base_stake=1.0,
        stop_win_pct=0.05,
        stop_loss_pct=0.03,
        max_daily_drawdown_pct=0.05,
        max_consecutive_losses=5,
        kelly_fraction=0.25,
    )


@pytest.fixture
def default_session():
    from core.entities import SessionState
    return SessionState(
        initial_balance=100.0,
        current_balance=100.0,
    )