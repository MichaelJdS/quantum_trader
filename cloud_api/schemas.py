"""
cloud_api/schemas.py — Modelos Pydantic para request/response da Cloud API
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class BotConfig(BaseModel):
    symbols: list[str] = Field(default=["R_50", "R_75"], description="Símbolos para operar")
    dry_run: bool = Field(default=True, description="Modo simulação (True = sem dinheiro real)")
    granularity: int = Field(default=60, ge=1, le=3600, description="Segundos por candle")
    stake: float = Field(default=1.0, gt=0, description="Stake base em USD")
    stop_win_pct: float = Field(default=5.0, gt=0, description="Stop Win % da banca")
    stop_loss_pct: float = Field(default=3.0, gt=0, description="Stop Loss % da banca")
    max_drawdown_pct: float = Field(default=5.0, gt=0, description="Max drawdown diário %")
    max_consecutive_losses: int = Field(default=5, ge=1, description="Max perdas consecutivas")
    kelly_enabled: bool = Field(default=False, description="Ativar Kelly Criterion")
    kelly_pct: float = Field(default=25.0, gt=0, le=100, description="Kelly Fraction %")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Mensagem para o Gemini")


# ── Responses ─────────────────────────────────────────────────────────────────

class CommandResponse(BaseModel):
    success: bool
    message: str


class BotStatus(BaseModel):
    is_running: bool
    session_id: str
    balance: float
    initial_balance: float
    win_rate: float
    total_trades: int
    wins: int
    losses: int
    consecutive_losses: int
    pnl: float
    dry_run: bool
    symbols: list[str]
    gemini_enabled: bool
    last_gemini_advice: dict | None = None


class ChatResponse(BaseModel):
    message: str


class TradeEvent(BaseModel):
    event: str          # "trade_opened" | "trade_closed" | "signal_rejected" etc.
    trade_id: str
    symbol: str
    direction: str
    stake: float
    strategy: str
    confidence: float
    pnl: float | None = None
    status: str | None = None
    ts: str


# ── WebSocket ─────────────────────────────────────────────────────────────────

class WSMessage(BaseModel):
    event: str          # "status" | "trade" | "log" | "gemini_advice" | etc.
    data: Any
    ts: str
