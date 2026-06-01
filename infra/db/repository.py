from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.entities import SessionState, Trade
from core.enums import TradeStatus
from infra.db.models_db import (
    CandleModel,
    ModelSnapshotModel,
    SessionModel,
    StopEventModel,
    TickModel,
    TradeModel,
)


# ── Session Repository ────────────────────────────────────────────────────────

class SessionRepository:
    """CRUD de sessões de trading."""

    def __init__(self, session: AsyncSession) -> None:
        self._db = session

    async def create(
        self,
        initial_balance: float,
        stop_win_pct: float,
        stop_loss_pct: float,
        max_drawdown_pct: float,
        max_consecutive_losses: int,
        stake_mode: str,
        base_stake: float,
        kelly_fraction: float,
        symbols: list[str],
    ) -> SessionModel:
        model = SessionModel(
            initial_balance=initial_balance,
            stop_win_pct=stop_win_pct,
            stop_loss_pct=stop_loss_pct,
            max_drawdown_pct=max_drawdown_pct,
            max_consecutive_losses=max_consecutive_losses,
            stake_mode=stake_mode,
            base_stake=base_stake,
            kelly_fraction=kelly_fraction,
            symbols=json.dumps(symbols),
        )
        self._db.add(model)
        await self._db.flush()
        logger.info("Sessão criada.", session_id=model.id)
        return model

    async def get_active(self) -> SessionModel | None:
        result = await self._db.execute(
            select(SessionModel)
            .where(SessionModel.is_active == True)  # noqa: E712
            .order_by(SessionModel.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, session_id: str) -> SessionModel | None:
        result = await self._db.execute(
            select(SessionModel).where(SessionModel.id == session_id)
        )
        return result.scalar_one_or_none()

    async def close_session(
        self,
        session_id: str,
        final_balance: float,
        metrics: dict,
    ) -> None:
        await self._db.execute(
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                is_active=False,
                final_balance=final_balance,
                ended_at=datetime.now(tz=timezone.utc),
                total_trades=metrics.get("total_trades", 0),
                wins=int(metrics.get("win_rate", 0) * metrics.get("total_trades", 0)),
                win_rate=metrics.get("win_rate", 0.0),
                pnl_total=metrics.get("pnl_total", 0.0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
                max_drawdown_reached=metrics.get("max_drawdown_pct", 0.0),
            )
        )
        logger.info("Sessão fechada.", session_id=session_id, final_balance=final_balance)

    async def list_recent(self, limit: int = 50) -> Sequence[SessionModel]:
        result = await self._db.execute(
            select(SessionModel)
            .order_by(SessionModel.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()


# ── Trade Repository ──────────────────────────────────────────────────────────

class TradeRepository:
    """CRUD de operações (trades)."""

    def __init__(self, session: AsyncSession) -> None:
        self._db = session

    async def save(self, trade: Trade, session_id: str) -> TradeModel:
        """Persiste um Trade (entidade de domínio) no banco."""
        model = TradeModel(
            id=trade.id,
            session_id=session_id,
            symbol=trade.symbol,
            contract_type=trade.contract_type.value,
            direction=trade.direction.value,
            status=trade.status.value,
            stake=trade.stake,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            payout=trade.payout,
            pnl=trade.pnl,
            confidence=trade.confidence,
            strategy_name=trade.strategy_name,
            model_name=trade.model_name,
            opened_at=trade.opened_at,
            closed_at=trade.closed_at,
        )
        self._db.add(model)
        await self._db.flush()
        logger.debug("Trade persistido.", trade_id=trade.id, symbol=trade.symbol)
        return model

    async def update_result(
        self,
        trade_id: str,
        status: TradeStatus,
        exit_price: float | None,
        pnl: float,
        payout: float | None,
        closed_at: datetime,
    ) -> None:
        await self._db.execute(
            update(TradeModel)
            .where(TradeModel.id == trade_id)
            .values(
                status=status.value,
                exit_price=exit_price,
                pnl=pnl,
                payout=payout,
                closed_at=closed_at,
            )
        )
        logger.debug("Trade atualizado.", trade_id=trade_id, status=status.value, pnl=pnl)

    async def list_by_session(
        self, session_id: str, limit: int = 200
    ) -> Sequence[TradeModel]:
        result = await self._db.execute(
            select(TradeModel)
            .where(TradeModel.session_id == session_id)
            .order_by(TradeModel.opened_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def list_by_symbol(
        self, symbol: str, limit: int = 100
    ) -> Sequence[TradeModel]:
        result = await self._db.execute(
            select(TradeModel)
            .where(TradeModel.symbol == symbol)
            .order_by(TradeModel.opened_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def win_rate_by_symbol(self, symbol: str) -> float:
        """Calcula win rate histórico de um símbolo."""
        trades = await self.list_by_symbol(symbol, limit=500)
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.status == TradeStatus.WON.value)
        return wins / len(trades)


# ── Tick Repository ───────────────────────────────────────────────────────────

class TickRepository:
    """Persistência de ticks brutos."""

    def __init__(self, session: AsyncSession) -> None:
        self._db = session

    async def bulk_save(self, ticks: list[dict]) -> None:
        """
        Insere múltiplos ticks em batch.

        Args:
            ticks: Lista de dicts com keys: symbol, price, epoch, pip_size.
        """
        models = [
            TickModel(
                symbol=t["symbol"],
                price=t["price"],
                epoch=t["epoch"],
                pip_size=t.get("pip_size"),
            )
            for t in ticks
        ]
        self._db.add_all(models)
        await self._db.flush()

    async def get_recent(self, symbol: str, limit: int = 500) -> Sequence[TickModel]:
        result = await self._db.execute(
            select(TickModel)
            .where(TickModel.symbol == symbol)
            .order_by(TickModel.epoch.desc())
            .limit(limit)
        )
        return result.scalars().all()


# ── Candle Repository ─────────────────────────────────────────────────────────

class CandleRepository:
    """Persistência de candles OHLCV."""

    def __init__(self, session: AsyncSession) -> None:
        self._db = session

    async def bulk_upsert(
        self, symbol: str, granularity: int, candles: list[dict]
    ) -> None:
        """
        Insere candles ignorando duplicatas (por epoch).

        Args:
            candles: Lista de dicts com keys: open, high, low, close, epoch.
        """
        existing_epochs = set()
        result = await self._db.execute(
            select(CandleModel.epoch)
            .where(
                CandleModel.symbol == symbol,
                CandleModel.granularity == granularity,
            )
        )
        existing_epochs = {row[0] for row in result.fetchall()}

        new_candles = [
            CandleModel(
                symbol=symbol,
                granularity=granularity,
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                epoch=int(c["epoch"]),
            )
            for c in candles
            if int(c["epoch"]) not in existing_epochs
        ]
        if new_candles:
            self._db.add_all(new_candles)
            await self._db.flush()
            logger.debug(
                "Candles inseridos.",
                symbol=symbol,
                count=len(new_candles),
            )

    async def get_recent(
        self, symbol: str, granularity: int, limit: int = 500
    ) -> Sequence[CandleModel]:
        result = await self._db.execute(
            select(CandleModel)
            .where(
                CandleModel.symbol == symbol,
                CandleModel.granularity == granularity,
            )
            .order_by(CandleModel.epoch.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))


# ── Model Snapshot Repository ─────────────────────────────────────────────────

class ModelSnapshotRepository:
    """Gerencia checkpoints de modelos ML."""

    def __init__(self, session: AsyncSession) -> None:
        self._db = session

    async def save_snapshot(
        self,
        model_name: str,
        version: str,
        symbol: str,
        weights_path: str,
        metrics: dict,
    ) -> ModelSnapshotModel:
        # Desativa snapshots anteriores deste modelo+símbolo.
        await self._db.execute(
            update(ModelSnapshotModel)
            .where(
                ModelSnapshotModel.model_name == model_name,
                ModelSnapshotModel.symbol == symbol,
            )
            .values(is_active=False)
        )

        model = ModelSnapshotModel(
            model_name=model_name,
            version=version,
            symbol=symbol,
            weights_path=weights_path,
            metrics_json=json.dumps(metrics),
            is_active=True,
            trained_at=datetime.now(tz=timezone.utc),
        )
        self._db.add(model)
        await self._db.flush()
        logger.success(
            "Snapshot de modelo salvo.",
            model=model_name,
            symbol=symbol,
            version=version,
        )
        return model

    async def get_active(
        self, model_name: str, symbol: str
    ) -> ModelSnapshotModel | None:
        result = await self._db.execute(
            select(ModelSnapshotModel)
            .where(
                ModelSnapshotModel.model_name == model_name,
                ModelSnapshotModel.symbol == symbol,
                ModelSnapshotModel.is_active == True,  # noqa: E712
            )
            .limit(1)
        )
        return result.scalar_one_or_none()


# ── Stop Event Repository ─────────────────────────────────────────────────────

class StopEventRepository:
    """Registra eventos de Stop Win / Stop Loss para auditoria."""

    def __init__(self, session: AsyncSession) -> None:
        self._db = session

    async def record(
        self,
        session_id: str,
        event_type: str,
        balance: float,
        pnl: float,
        reason: str,
    ) -> StopEventModel:
        model = StopEventModel(
            session_id=session_id,
            event_type=event_type,
            balance_at_event=balance,
            pnl_at_event=pnl,
            reason=reason,
        )
        self._db.add(model)
        await self._db.flush()
        logger.warning(
            "Stop event registrado.",
            type=event_type,
            session_id=session_id,
            reason=reason,
        )
        return model