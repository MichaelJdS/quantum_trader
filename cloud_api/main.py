"""
cloud_api/main.py — Backend FastAPI do Quantum Trader v2.0

Endpoints:
  POST /start                — inicia o bot
  POST /stop                 — para o bot
  GET  /status               — estado atual em JSON
  GET  /metrics              — PnL, win rate, agentes, circuit breaker
  GET  /council/status       — status do GrandOracle
  GET  /circuit_breaker      — status do circuit breaker
  POST /admin/reset_breaker  — força reset do circuit breaker
  GET  /market/profiles      — granularidade e duração ótimas por símbolo
  GET  /ws                   — WebSocket para eventos em tempo real
  POST /gemini/chat          — chat direto com o Gemini Advisor
  PUT  /config               — atualiza configurações do bot em runtime
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
from cloud_api.metrics import MetricsCollector
from cloud_api.schemas import (
    BotConfig,
    BotStatus,
    ChatRequest,
    ChatResponse,
    CommandResponse,
    TradeEvent,
    WSMessage,
)


# ── Estado Global ─────────────────────────────────────────────────────────────

class AppState:
    """Estado singleton do servidor — compartilhado entre requests."""
    engine          = None
    client          = None
    symbol_manager  = None
    gemini_advisor  = None
    circuit_breaker = None   # NOVO
    market_profiler = None   # NOVO
    metrics         = MetricsCollector()  # NOVO
    is_running: bool = False
    session_id: str  = ""
    config: BotConfig | None = None
    _ws_clients: set[WebSocket] = set()

    @classmethod
    async def broadcast(cls, event: str, data: Any) -> None:
        """Envia mensagem para todos os clientes WebSocket conectados.
        Também registra trades e decisões nas métricas automaticamente."""
        # Registra nas métricas antes de broadcast
        if event == "trade":
            cls.metrics.record_trade(data)
        elif event == "council_vote":
            cls.metrics.record_council_decision(data)

        if not cls._ws_clients:
            return
        msg     = WSMessage(event=event, data=data, ts=datetime.now(tz=timezone.utc).isoformat())
        payload = msg.model_dump_json()
        disconnected: set[WebSocket] = set()
        for ws in cls._ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)
        cls._ws_clients -= disconnected


state = AppState()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Quantum Trader Cloud API iniciando...")
    from core.bootstrap import bootstrap
    await bootstrap()
    logger.success("Cloud API pronta.")
    yield
    # Shutdown limpo
    if state.is_running and state.engine is not None:
        await state.engine.stop()
    if state.market_profiler is not None:
        state.market_profiler.stop()
    if state.client is not None:
        await state.client.disconnect()
    logger.info("Cloud API encerrada.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Quantum Trader Cloud API",
    version="2.0.0",
    description="Backend 24/7 do Quantum Trader rodando no Google Cloud",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — sem autenticação para o GCP verificar se a VM está ativa."""
    return {"status": "ok", "ts": datetime.now(tz=timezone.utc).isoformat()}


# ── Start / Stop ──────────────────────────────────────────────────────────────

@app.post("/start", response_model=CommandResponse, dependencies=[Depends(verify_token)])
async def start_bot(config: BotConfig):
    """Inicia o ExecutionEngine na nuvem."""
    if state.is_running:
        raise HTTPException(status_code=400, detail="Bot já está em execução.")

    state.config     = config
    state.session_id = str(uuid.uuid4())
    state.metrics    = MetricsCollector()  # Reseta as métricas em cada novo boot

    try:
        await _boot_engine(config)
        state.is_running = True
        await state.broadcast("bot_started", {"session_id": state.session_id, "config": config.model_dump()})
        return CommandResponse(success=True, message=f"Bot iniciado. Session: {state.session_id}")
    except Exception as exc:
        logger.exception("Falha ao iniciar bot.")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/stop", response_model=CommandResponse, dependencies=[Depends(verify_token)])
async def stop_bot():
    """Para o ExecutionEngine."""
    if not state.is_running or state.engine is None or state.client is None:
        raise HTTPException(status_code=400, detail="Bot não está em execução.")
    try:
        await state.engine.stop()
        if state.market_profiler:
            state.market_profiler.stop()
        await state.client.disconnect()
        state.is_running = False
        await state.broadcast("bot_stopped", {})
        return CommandResponse(success=True, message="Bot parado com sucesso.")
    except Exception as exc:
        logger.error("Falha ao parar bot.", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ── Status ────────────────────────────────────────────────────────────────────

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
            "strategy":   adv.recommended_strategy,
            "multiplier": adv.confidence_multiplier,
            "risk_flag":  adv.risk_flag,
            "reasoning":  adv.reasoning,
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


# ── Metrics (NOVO) ────────────────────────────────────────────────────────────

@app.get("/metrics", dependencies=[Depends(verify_token)])
async def get_metrics():
    """
    Dashboard completo em JSON:
    - PnL acumulado e win rate globais
    - Win rate e PnL por símbolo
    - Peso atual de cada agente do GrandOracle
    - Status do circuit breaker
    - Perfis de granularidade/duração do MarketProfiler
    - Últimas 10 decisões do council
    - Últimos 20 trades
    """
    return state.metrics.snapshot(
        engine=state.engine,
        circuit_breaker=state.circuit_breaker,
        market_profiler=state.market_profiler,
    )


# ── Council ───────────────────────────────────────────────────────────────────

@app.get("/council/status", dependencies=[Depends(verify_token)])
async def get_council_status():
    """Retorna o último status do Oracle Council."""
    if not state.is_running or state.engine is None or state.engine.grand_oracle is None:
        return {"status": "idle", "last_decision": None}
    return state.engine.grand_oracle.get_council_health()


# ── Circuit Breaker (NOVO) ────────────────────────────────────────────────────

@app.get("/circuit_breaker", dependencies=[Depends(verify_token)])
async def get_circuit_breaker():
    """Retorna o status atual do circuit breaker."""
    if state.circuit_breaker is None:
        return {"status": "not_initialized"}
    return state.circuit_breaker.status()


@app.post("/admin/reset_breaker", response_model=CommandResponse, dependencies=[Depends(verify_token)])
async def reset_circuit_breaker():
    """Força reset manual do circuit breaker (use com cautela)."""
    if state.circuit_breaker is None:
        raise HTTPException(status_code=404, detail="Circuit breaker não inicializado.")
    state.circuit_breaker.force_reset()
    logger.warning("⚠️ Circuit breaker resetado manualmente via API.")
    return CommandResponse(success=True, message="Circuit breaker resetado com sucesso.")


# ── Market Profiler (NOVO) ────────────────────────────────────────────────────

@app.get("/market/profiles", dependencies=[Depends(verify_token)])
async def get_market_profiles():
    """Retorna granularidade e duração ótimas por símbolo."""
    if state.market_profiler is None:
        return {"status": "not_initialized"}
    return state.market_profiler.all_profiles()


# ── Gemini Chat ───────────────────────────────────────────────────────────────

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


# ── Config ────────────────────────────────────────────────────────────────────

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
    if not _verify_token_ws(token):
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    state._ws_clients.add(ws)
    logger.info("WebSocket conectado.", clients=len(state._ws_clients))

    try:
        # Envia status imediato ao conectar
        status_data = (await get_status()).model_dump()
        await ws.send_text(
            WSMessage(event="status", data=status_data, ts=datetime.now(tz=timezone.utc).isoformat()).model_dump_json()
        )

        # Mantém conexão viva
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=60.0)
                data = json.loads(raw)
                if data.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                try:
                    await ws.send_text(json.dumps({
                        "type": "heartbeat",
                        "ts": datetime.now(tz=timezone.utc).isoformat()
                    }))
                except Exception:
                    break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning("Erro interno no websocket loop", error=str(e))
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WebSocket erro.", error=str(exc))
    finally:
        state._ws_clients.discard(ws)
        logger.info("WebSocket desconectado.", clients=len(state._ws_clients))


def _verify_token_ws(token: str) -> bool:
    api_token = os.getenv("API_TOKEN", "")
    if not api_token:
        return True  # Sem token configurado → modo dev
    return token == api_token


# ── Boot Engine ───────────────────────────────────────────────────────────────

async def _warm_up_oracle(engine) -> None:
    """Alimenta o GrandOracle com histórico de trades do banco."""
    if engine.grand_oracle is None:
        return
    try:
        from infra.db.database import get_session
        # Tenta importar o repositório — nome pode variar
        try:
            from infra.db.repository import TradeRepository
        except ImportError:
            from infra.db.repositories import TradeRepository

        async with get_session() as session:
            repo = TradeRepository(session)
            # Tenta nomes alternativos do método
            if hasattr(repo, "get_recent"):
                trades = await repo.get_recent(limit=200)
            elif hasattr(repo, "get_last"):
                trades = await repo.get_last(limit=200)
            elif hasattr(repo, "list_recent"):
                trades = await repo.list_recent(limit=200)
            else:
                # Fallback: query direta
                from infra.db.models_db import TradeModel
                from sqlalchemy import select, desc
                result = await session.execute(
                    select(TradeModel).order_by(desc(TradeModel.opened_at)).limit(200)
                )
                trades = result.scalars().all()

        if not trades:
            logger.info("Sem histórico de trades para warm-up do Oracle.")
            return

        # Popula as métricas do dashboard com histórico do banco (inverso: mais antigos primeiro)
        for trade in reversed(trades):
            try:
                status_val = trade.status.value if hasattr(trade.status, "value") else str(getattr(trade, "status", "CLOSED"))
                dir_val = trade.direction.value if hasattr(trade.direction, "value") else str(getattr(trade, "direction", "BUY"))
                state.metrics.record_trade({
                    "trade_id": str(getattr(trade, "id", "")),
                    "symbol": str(getattr(trade, "symbol", "unknown")),
                    "direction": dir_val,
                    "stake": float(getattr(trade, "stake", 0.0) or 0.0),
                    "strategy": getattr(trade, "strategy_name", "historical"),
                    "confidence": float(getattr(trade, "confidence", 0.0) or 0.0),
                    "pnl": float(getattr(trade, "pnl", 0.0) or 0.0),
                    "status": status_val,
                    "ts": trade.closed_at.isoformat() if getattr(trade, "closed_at", None) else (trade.opened_at.isoformat() if getattr(trade, "opened_at", None) else datetime.now(tz=timezone.utc).isoformat()),
                })
            except Exception as e:
                logger.debug(f"Erro ao popular histórico de dashboard: {e}")

        fed = 0
        for trade in trades:
            won = float(getattr(trade, "pnl", 0) or 0) > 0
            pnl = float(getattr(trade, "pnl", 0) or 0)

            # record_outcome pode ter assinatura diferente — tenta variações
            oracle = engine.grand_oracle
            if hasattr(oracle, "record_outcome"):
                try:
                    oracle.record_outcome(
                        agent_votes=getattr(trade, "agent_votes", {}) or {},
                        won=won,
                        pnl=pnl,
                    )
                except TypeError:
                    oracle.record_outcome(won=won, pnl=pnl)
            elif hasattr(oracle, "update_weights"):
                oracle.update_weights(won=won, pnl=pnl)
            fed += 1

        logger.success("🧠 GrandOracle aquecido com histórico.", trades_fed=fed)
    except Exception as exc:
        logger.warning("Warm-up do Oracle falhou (não crítico).", error=str(exc))

async def _boot_engine(config: BotConfig) -> None:
    """
    Instancia e inicia todos os componentes do trading engine.

    Ordem de boot:
      1. CircuitBreaker
      2. DerivClient (conecta WebSocket)
      3. SymbolManager (inicializa subscrições)
      4. MarketProfiler (avalia gran + duração ótimas)
      5. HistoricalLoader (popula candles históricos)
      6. GeminiAdvisor
      7. ExecutionEngine (injeta circuit breaker)
      8. Estratégias
      9. engine.start()
    """
    from core.circuit_breaker import CircuitBreaker
    from core.entities import RiskConfig
    from core.enums import StakeMode
    from core.execution_engine import ExecutionEngine
    from core.strategies import BollingerReversionStrategy, BreakoutStrategy, EmaRsiStrategy
    from infra.deriv_client import DerivClient
    from infra.historical_loader import HistoricalLoader
    from infra.symbol_manager import SymbolManager
    from ml.gemini_advisor import GeminiAdvisor
    from ml.market_profiler import MarketProfiler

    # 1. Circuit Breaker
    state.circuit_breaker = CircuitBreaker(
        max_consecutive_losses=int(os.getenv("CIRCUIT_BREAKER_MAX_LOSSES", "5")),
        max_drawdown_pct=float(os.getenv("CIRCUIT_BREAKER_MAX_DRAWDOWN",   "0.15")),
        cooldown_seconds=int(os.getenv("CIRCUIT_BREAKER_COOLDOWN",         "1800")),
    )

    # 2. Market Profiler e Historical Loader ANTES de conectar à Deriv
    #    (evita timeout do WS durante o boot longo)
    state.market_profiler = MarketProfiler(
        client=None,   # será injetado depois
        symbols=config.symbols,
    )

    # 3. Conecta à Deriv — só agora, boot pesado já passou
    state.client = DerivClient(dry_run=config.dry_run)
    await state.client.connect()

    # Injeta client no profiler após conectar
    state.market_profiler.client = state.client

    # 4. Symbol Manager
    state.symbol_manager = SymbolManager(
        client=state.client,
        symbols=config.symbols,
        granularity=config.granularity,
    )
    await state.symbol_manager.initialize()

    # 5. Market Profiler (agora com client)
    logger.info("🔍 MarketProfiler avaliando...")
    await state.market_profiler.run_once()
    await state.market_profiler.start_background()

    # 6. Historical Loader
    loader = HistoricalLoader(
        client=state.client,
        symbol_manager=state.symbol_manager,
        symbols=config.symbols,
    )
    logger.info("📥 HistoricalLoader baixando dados...")
    await loader.load_all(timeout=90.0)

    # 7. Advisors
    state.gemini_advisor = GeminiAdvisor()

    # 7. Execution Engine
    risk_config = RiskConfig(
        stake_mode=StakeMode.FRACTIONAL_KELLY if getattr(config, "kelly_enabled", False) else StakeMode.FIXED,
        base_stake=config.stake,
        stop_win_pct=config.stop_win_pct   / 100,
        stop_loss_pct=config.stop_loss_pct / 100,
        max_daily_drawdown_pct=config.max_drawdown_pct / 100,
        max_consecutive_losses=config.max_consecutive_losses,
        kelly_fraction=getattr(config, "kelly_pct", 25.0) / 100,
    )

    state.engine = ExecutionEngine(
        client=state.client,
        symbol_manager=state.symbol_manager,
        risk_config=risk_config,
        session_id=state.session_id,
        dry_run=config.dry_run,
        gemini_advisor=state.gemini_advisor,
        broadcast_fn=state.broadcast,
    )

    # Injeta circuit breaker no engine para checagem em _await_result
    state.engine._circuit_breaker = state.circuit_breaker
    
    # Injeta engine no symbol_manager para o streaming de eventos
    state.symbol_manager._engine = state.engine
    state.symbol_manager._broadcast_fn = state.broadcast

    # 8. Estratégias
    state.engine.register_strategy(EmaRsiStrategy(risk_config=risk_config))
    state.engine.register_strategy(BollingerReversionStrategy(risk_config=risk_config))
    state.engine.register_strategy(BreakoutStrategy(risk_config=risk_config))

    # 9. Inicia engine
    # Pré-aquece o GrandOracle com histórico de trades do banco
    await _warm_up_oracle(state.engine)
    await state.engine.start()

    logger.success(
        "🚀 Todos os componentes iniciados com sucesso.",
        symbols=config.symbols,
        dry_run=config.dry_run,
        profiles=state.market_profiler.all_profiles(),
    )


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        "cloud_api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )
