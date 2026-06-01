from __future__ import annotations

import pytest

from infra.db.database import init_db, close_db, get_session
from infra.db.repository import SessionRepository, TradeRepository


@pytest.fixture(autouse=True)
async def setup_db():
    await init_db()
    yield
    await close_db()


async def test_create_and_get_session():
    async with get_session() as db:
        repo = SessionRepository(db)
        session = await repo.create(
            initial_balance=100.0,
            stop_win_pct=0.05,
            stop_loss_pct=0.03,
            max_drawdown_pct=0.05,
            max_consecutive_losses=5,
            stake_mode="FIXED",
            base_stake=1.0,
            kelly_fraction=0.25,
            symbols=["R_50", "R_75"],
        )
        assert session.id is not None
        assert session.is_active is True

    async with get_session() as db:
        repo = SessionRepository(db)
        active = await repo.get_active()
        assert active is not None
        assert active.initial_balance == 100.0


async def test_close_session():
    async with get_session() as db:
        repo = SessionRepository(db)
        session = await repo.create(
            initial_balance=100.0,
            stop_win_pct=0.05,
            stop_loss_pct=0.03,
            max_drawdown_pct=0.05,
            max_consecutive_losses=5,
            stake_mode="FIXED",
            base_stake=1.0,
            kelly_fraction=0.25,
            symbols=["R_50"],
        )
        sid = session.id

    async with get_session() as db:
        repo = SessionRepository(db)
        await repo.close_session(
            session_id=sid,
            final_balance=103.0,
            metrics={"total_trades": 10, "win_rate": 0.6, "pnl_total": 3.0,
                     "sharpe_ratio": 1.2, "max_drawdown_pct": 0.01},
        )
        closed = await repo.get_by_id(sid)
        assert closed is not None
        assert closed.is_active is False
        assert closed.final_balance == 103.0