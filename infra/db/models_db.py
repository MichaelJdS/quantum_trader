from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarativa compartilhada por todos os modelos ORM."""

    __abstract__ = True

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ── Sessões ───────────────────────────────────────────────────────────────────

class SessionModel(Base):
    """
    Representa uma sessão de trading.
    Uma sessão agrupa trades com os mesmos parâmetros de risco.
    """

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    initial_balance: Mapped[float] = mapped_column(Float, nullable=False)
    final_balance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_win_pct: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss_pct: Mapped[float] = mapped_column(Float, nullable=False)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False)
    max_consecutive_losses: Mapped[int] = mapped_column(Integer, nullable=False)
    stake_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    base_stake: Mapped[float] = mapped_column(Float, nullable=False)
    kelly_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    symbols: Mapped[str] = mapped_column(Text, nullable=False)  # JSON list
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    pnl_total: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown_reached: Mapped[float] = mapped_column(Float, default=0.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    trades: Mapped[list["TradeModel"]] = relationship(
        "TradeModel", back_populates="session", lazy="selectin"
    )
    stop_events: Mapped[list["StopEventModel"]] = relationship(
        "StopEventModel", back_populates="session", lazy="selectin"
    )

    __table_args__ = (
        Index("ix_sessions_is_active", "is_active"),
        Index("ix_sessions_created_at", "created_at"),
    )


# ── Trades ────────────────────────────────────────────────────────────────────

class TradeModel(Base):
    """
    Registro imutável de cada operação executada.
    Não deve ser atualizado após fechamento — apenas closed_at e campos de resultado.
    """

    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    contract_type: Mapped[str] = mapped_column(String(16), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    stake: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    payout: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    signal_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    deriv_contract_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    session: Mapped["SessionModel"] = relationship(
        "SessionModel", back_populates="trades"
    )

    __table_args__ = (
        Index("ix_trades_session_id", "session_id"),
        Index("ix_trades_symbol", "symbol"),
        Index("ix_trades_status", "status"),
        Index("ix_trades_opened_at", "opened_at"),
        Index("ix_trades_symbol_status", "symbol", "status"),
    )


# ── Ticks ─────────────────────────────────────────────────────────────────────

class TickModel(Base):
    """
    Armazena ticks brutos recebidos do WebSocket Deriv.
    Otimizado para leitura em série temporal.
    """

    __tablename__ = "ticks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)
    pip_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_ticks_symbol_epoch", "symbol", "epoch"),
        Index("ix_ticks_epoch", "epoch"),
    )


# ── Candles ───────────────────────────────────────────────────────────────────

class CandleModel(Base):
    """
    Armazena candles OHLCV históricos por símbolo e granularidade.
    """

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    granularity: Mapped[int] = mapped_column(Integer, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        Index("ix_candles_symbol_granularity_epoch", "symbol", "granularity", "epoch"),
    )


# ── Snapshots de Modelo ───────────────────────────────────────────────────────

class ModelSnapshotModel(Base):
    """
    Checkpoint de modelo ML para auditoria e rollback.
    O arquivo de pesos é referenciado por caminho — não armazenado no banco.
    """

    __tablename__ = "model_snapshots"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    weights_path: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_model_snapshots_symbol", "symbol"),
        Index("ix_model_snapshots_model_name", "model_name"),
        Index("ix_model_snapshots_is_active", "is_active"),
    )


# ── Stop Events ───────────────────────────────────────────────────────────────

class StopEventModel(Base):
    """
    Registra eventos de Stop Win / Stop Loss para auditoria.
    """

    __tablename__ = "stop_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(16), nullable=False)  # WIN | LOSS
    balance_at_event: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_at_event: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)

    session: Mapped["SessionModel"] = relationship(
        "SessionModel", back_populates="stop_events"
    )

    __table_args__ = (
        Index("ix_stop_events_session_id", "session_id"),
    )