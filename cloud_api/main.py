"""
cloud_api/main.py — Backend FastAPI do Quantum Trader

Expõe o ExecutionEngine via HTTP REST + WebSocket para ser controlado
pelo App Windows. Roda 24/7 no Google Compute Engine e2-micro (free tier).

Endpoints:
  POST /start           — inicia o bot
  POST /stop            — para o bot
  GET  /status          — estado atual em JSON
  GET  /ws              — WebSocket para eventos em tempo real
  POST /gemini/chat     — chat direto com o Gemini Advisor
  PUT  /config          — atualiza configurações do bot em runtime
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from cloud_api.auth import verify_token
from cloud_api.schemas import (
    BotConfig,
    BotStatus,
    ChatRequest,
    ChatResponse,
    CommandResponse,
    TradeEvent,
    WSMessage,
)


# ── Estado Global do Servidor ────────────────────────────────────────────────

class AppState:
    """Estado singleton do servidor — compartilhado entre requests."""
    engine = None
    client = None
    symbol_manager = None
    gemini_advisor = None
    is_running: bool = False
    session_id: str = ""
    config: BotConfig | None = None
    _ws_clients: set[WebSocket] = set()

    @classmethod
    async def broadcast(cls, event: str, data: Any) -> None:
        """Envia mensagem para todos os clientes WebSocket conectados."""
        if not cls._ws_clients:
            return
        msg = WSMessage(event=event, data=data, ts=datetime.now(tz=timezone.utc).isoformat())
        payload = msg.model_dump_json()
        disconnected = set()
        for ws in cls._ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)
        cls._ws_clients -= disconnected


state = AppState()


# ── Lifespan (startup/shutdown) ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Quantum Trader Cloud API iniciando...")
    # Carrega variáveis de ambiente e inicializa dependências base
    from core.bootstrap import bootstrap
    await bootstrap()
    logger.success("Cloud API pronta.")
    yield
    # Shutdown limpo
    if state.is_running and state.engine is not None:
        await state.engine.stop()
    if state.client is not None:
        await state.client.disconnect()
    logger.info("Cloud API encerrada.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Quantum Trader Cloud API",
    version="1.0.0",
    description="Backend 24/7 do Quantum Trader rodando no Google Cloud",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints REST ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — sem autenticação para o GCP verificar se a VM está ativa."""
    return {"status": "ok", "ts": datetime.now(tz=timezone.utc).isoformat()}


@app.post("/start", response_model=CommandResponse, dependencies=[Depends(verify_token)])
async def start_bot(config: BotConfig):
    """Inicia o ExecutionEngine na nuvem."""
    if state.is_running:
        raise HTTPException(status_code=400, detail="Bot já está em execução.")

    state.config = config
    state.session_id = str(uuid.uuid4())

    try:
        await _boot_engine(config)
        state.is_running = True
        await state.broadcast("bot_started", {"session_id": state.session_id, "config": config.model_dump()})
        return CommandResponse(success=True, message=f"Bot iniciado. Session: {state.session_id}")
    except Exception as exc:
        logger.error("Falha ao iniciar bot.", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/stop", response_model=CommandResponse, dependencies=[Depends(verify_token)])
async def stop_bot():
    """Para o ExecutionEngine."""
    if not state.is_running or state.engine is None or state.client is None:
        raise HTTPException(status_code=400, detail="Bot não está em execução.")
    try:
        await state.engine.stop()
        await state.client.disconnect()
        state.is_running = False
        await state.broadcast("bot_stopped", {})
        return CommandResponse(success=True, message="Bot parado com sucesso.")
    except Exception as exc:
        logger.error("Falha ao parar bot.", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/status", response_model=BotStatus, dependencies=[Depends(verify_token)])
async def get_status():
    """Retorna o estado atual completo do bot."""
    if not state.is_running or state.engine is None:
        return BotStatus(
            is_running=False,
            session_id="",
            balance=0.0,
            initial_balance=0.0,
            win_rate=0.0,
            total_trades=0,
            wins=0,
            losses=0,
            consecutive_losses=0,
            pnl=0.0,
            dry_run=state.config.dry_run if state.config else True,
            symbols=state.config.symbols if state.config else [],
            gemini_enabled=False,
            last_gemini_advice=None,
        )
    s = state.engine.session_state
    gemini_advice = None
    if state.gemini_advisor and state.gemini_advisor.last_advice:
        adv = state.gemini_advisor.last_advice
        gemini_advice = {
            "strategy": adv.recommended_strategy,
            "multiplier": adv.confidence_multiplier,
            "risk_flag": adv.risk_flag,
            "reasoning": adv.reasoning,
        }
    return BotStatus(
        is_running=state.is_running,
        session_id=state.session_id,
        balance=round(s.current_balance, 4),
        initial_balance=round(s.initial_balance, 4),
        win_rate=round(s.win_rate, 4),
        total_trades=s.total_trades,
        wins=s.wins,
        losses=s.losses,
        consecutive_losses=s.consecutive_losses,
        pnl=round(s.current_balance - s.initial_balance, 4),
        dry_run=state.engine.dry_run,
        symbols=state.config.symbols if state.config else [],
        gemini_enabled=state.gemini_advisor.is_enabled if state.gemini_advisor else False,
        last_gemini_advice=gemini_advice,
    )


@app.post("/gemini/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def gemini_chat(req: ChatRequest):
    """Chat livre com o Gemini Advisor — para o painel do App Windows."""
    if state.gemini_advisor is None or not state.gemini_advisor.is_enabled:
        raise HTTPException(status_code=503, detail="Gemini Advisor não está ativo.")

    context = ""
    if state.is_running and state.engine:
        s = state.engine.session_state
        context = (
            f"[Contexto atual do bot]\n"
            f"Saldo: ${s.current_balance:.2f} | Win Rate: {s.win_rate:.1%} | "
            f"Trades: {s.total_trades} | Perdas consecutivas: {s.consecutive_losses}"
        )
    response = await state.gemini_advisor.chat(req.message, context=context)
    return ChatResponse(message=response)


@app.get("/council/status", dependencies=[Depends(verify_token)])
async def get_council_status():
    """Retorna o último status do Oracle Council."""
    if not state.is_running or state.engine is None or state.engine.grand_oracle is None:
        return {"status": "idle", "last_decision": None}
    return state.engine.grand_oracle.get_status()


@app.put("/config", response_model=CommandResponse, dependencies=[Depends(verify_token)])
async def update_config(config: BotConfig):
    """Atualiza configurações sem reiniciar (quando possível)."""
    state.config = config
    await state.broadcast("config_updated", config.model_dump())
    return CommandResponse(success=True, message="Configurações atualizadas.")


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str = ""):
    """
    WebSocket para stream de eventos em tempo real para o App Windows.
    O App deve enviar: ws://host/ws?token=<API_TOKEN>
    """
    if not verify_token_ws(token):
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    state._ws_clients.add(ws)
    logger.info("WebSocket conectado.", clients=len(state._ws_clients))

    try:
        # Envia status imediato ao conectar
        status_data = (await get_status()).model_dump()
        await ws.send_text(WSMessage(event="status", data=status_data, ts=datetime.now(tz=timezone.utc).isoformat()).model_dump_json())

        # Mantém conexão viva aguardando mensagens do cliente (ping/pong)
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                data = json.loads(raw)
                if data.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except TimeoutError:
                # Envia heartbeat
                await ws.send_text(json.dumps({"type": "heartbeat", "ts": datetime.now(tz=timezone.utc).isoformat()}))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WebSocket erro.", error=str(exc))
    finally:
        state._ws_clients.discard(ws)
        logger.info("WebSocket desconectado.", clients=len(state._ws_clients))


def verify_token_ws(token: str) -> bool:
    """Verifica token passado via query param no WebSocket."""
    api_token = os.getenv("API_TOKEN", "")
    if not api_token:
        return True  # Sem token configurado → modo dev
    return token == api_token


# ── Boot Engine ───────────────────────────────────────────────────────────────

async def _boot_engine(config: BotConfig) -> None:
    """Instancia e inicia todos os componentes do trading engine."""
    from core.entities import RiskConfig
    from core.enums import StakeMode
    from core.execution_engine import ExecutionEngine
    from core.strategies import BollingerReversionStrategy, BreakoutStrategy, EmaRsiStrategy
    from infra.deriv_client import DerivClient
    from infra.symbol_manager import SymbolManager
    from ml.gemini_advisor import GeminiAdvisor

    risk_config = RiskConfig(
        stake_mode=StakeMode.FRACTIONAL_KELLY if getattr(config, "kelly_enabled", False) else StakeMode.FIXED,
        base_stake=config.stake,
        stop_win_pct=config.stop_win_pct / 100,
        stop_loss_pct=config.stop_loss_pct / 100,
        max_daily_drawdown_pct=config.max_drawdown_pct / 100,
        max_consecutive_losses=config.max_consecutive_losses,
        kelly_fraction=getattr(config, "kelly_pct", 25.0) / 100,
    )

    state.client = DerivClient(dry_run=config.dry_run)
    await state.client.connect()

    state.symbol_manager = SymbolManager(
        client=state.client,
        symbols=config.symbols,
        granularity=config.granularity,
    )
    await state.symbol_manager.initialize()

    state.gemini_advisor = GeminiAdvisor()

    state.engine = ExecutionEngine(
        client=state.client,
        symbol_manager=state.symbol_manager,
        risk_config=risk_config,
        session_id=state.session_id,
        dry_run=config.dry_run,
        gemini_advisor=state.gemini_advisor,
        broadcast_fn=state.broadcast,
    )

    state.engine.register_strategy(EmaRsiStrategy(risk_config=risk_config))
    state.engine.register_strategy(BollingerReversionStrategy(risk_config=risk_config))
    state.engine.register_strategy(BreakoutStrategy(risk_config=risk_config))

    await state.engine.start()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        "cloud_api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
