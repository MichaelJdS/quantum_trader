from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import pandas as pd
from loguru import logger

from infra.deriv_client import DerivClient

TickCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class SymbolState:
    symbol: str
    candles_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    recent_ticks: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=500))
    tick_listeners: list[TickCallback] = field(default_factory=list)
    is_ready: bool = False
    last_tick_epoch: int | None = None
    last_candle_epoch: int | None = None
    last_price: float | None = None
    last_update: float | None = None


class SymbolManager:
    """
    Responsável por:
      - carregar candles iniciais de cada símbolo
      - manter estado em memória por símbolo
      - assinar ticks em tempo real
      - expor DataFrame de candles e ticks recentes
    """

    def __init__(
        self,
        client: DerivClient,
        symbols: list[str],
        granularity: int = 60,
        candle_count: int = 500,
    ) -> None:
        self._client = client
        self._symbols = list(dict.fromkeys(symbols))
        self._granularity = granularity
        self._candle_count = max(100, min(candle_count, 5000))
        self._states: dict[str, SymbolState] = {
            symbol: SymbolState(symbol=symbol) for symbol in self._symbols
        }
        self._lock = asyncio.Lock()
        self._initialized = False
        self._engine: Any | None = None
        self._broadcast_fn: Callable | None = None

    # ── Inicialização ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Carrega candles históricos e registra subscriptions de ticks.
        """
        if self._initialized:
            return

        for symbol in self._symbols:
            await self._initialize_symbol(symbol)

        await self._start_candle_streams()

        self._initialized = True
        logger.success(
            "SymbolManager inicializado.",
            symbols=self.ready_symbols,
            granularity=self._granularity,
        )

    async def _initialize_symbol(self, symbol: str) -> None:
        try:
            candles = await self._client.get_candles(
                symbol=symbol,
                granularity=self._granularity,
                count=self._candle_count,
            )
            df = self._candles_to_df(candles)

            async with self._lock:
                state = self._states[symbol]
                state.candles_df = df
                state.is_ready = not df.empty
                if not df.empty:
                    state.last_candle_epoch = int(df.iloc[-1]["epoch"])

            import asyncio
            for attempt in range(3):
                try:
                    await self._client.subscribe_ticks(symbol, callback=self._handle_tick)
                    break
                except TimeoutError:
                    if attempt == 2:
                        logger.error("Falha ao inicializar símbolo após 3 tentativas.", symbol=symbol)
                        raise
                    else:
                        logger.warning(f"Timeout ao inscrever {symbol}, tentando novamente ({attempt+2}/3)...")
                        await asyncio.sleep(2)

            logger.info(
                "Símbolo inicializado.",
                symbol=symbol,
                candles=len(df),
                ready=not df.empty,
            )
        except Exception as exc:
            logger.exception(
                "Falha ao inicializar símbolo.",
                symbol=symbol,
                error=str(exc),
            )

    async def _start_candle_streams(self) -> None:
        """Inicia stream de candles em tempo real para cada símbolo."""
        for symbol in self._symbols:
            gran = self._granularity
            asyncio.create_task(
                self._client.stream_candles(
                    symbol=symbol,
                    granularity=gran,
                    callback=self._on_live_candle,
                ),
                name=f"candle_stream_{symbol}",
            )
        logger.info("📡 Streams de candles iniciados.", symbols=self._symbols)

    async def _on_live_candle(self, candle: dict) -> None:
        """Callback chamado instantaneamente a cada vela nova."""
        import time
        from datetime import datetime, timezone
        
        symbol = candle.get("symbol") or candle.get("underlying")
        if not symbol:
            return
            
        async with self._lock:
            state = self._states.get(symbol)
            if state is None:
                return
            # Atualiza DataFrame de candles
            new_row = {
                "epoch": int(candle.get("epoch", 0)),
                "open":  float(candle.get("open",  0)),
                "high":  float(candle.get("high",  0)),
                "low":   float(candle.get("low",   0)),
                "close": float(candle.get("close", 0)),
            }
            if state.candles_df is not None:
                # Atualiza a última vela se mesmo epoch, senão appenda
                if (len(state.candles_df) > 0 and
                        state.candles_df.iloc[-1]["epoch"] == new_row["epoch"]):
                    state.candles_df.iloc[-1] = new_row
                else:
                    import pandas as pd
                    state.candles_df = pd.concat(
                        [state.candles_df, pd.DataFrame([new_row])],
                        ignore_index=True,
                    ).tail(2000)  # mantém últimas 2000 velas
            state.last_price = new_row["close"]
            state.last_update = time.time()

        # Broadcast instantâneo de tick para o dashboard
        await self._broadcast_tick(symbol, new_row["close"], new_row["epoch"])
        
        # Avisa o motor de execução instantaneamente
        if hasattr(self, '_engine') and self._engine is not None:
            await self._engine._candle_queue.put((symbol, new_row))

    async def _broadcast_tick(self, symbol: str, price: float, epoch: int) -> None:
        from datetime import datetime, timezone
        if hasattr(self, '_broadcast_fn') and self._broadcast_fn:
            await self._broadcast_fn("tick", {
                "symbol": symbol,
                "price":  price,
                "epoch":  epoch,
                "ts":     datetime.now(tz=timezone.utc).isoformat(),
            })

    # ── Tick stream ───────────────────────────────────────────────────────────

    async def _handle_tick(self, data: dict[str, Any]) -> None:
        """
        Callback único para todos os ticks da Deriv.

        Formato esperado:
          {
            "msg_type": "tick",
            "tick": {
              "symbol": "R_50",
              "quote": 123.45,
              "epoch": 1710000000,
              ...
            }
          }
        """
        tick = data.get("tick", {})
        symbol = tick.get("symbol")
        if not symbol or symbol not in self._states:
            return

        normalized_tick = {
            "symbol": symbol,
            "price": self._extract_tick_price(tick),
            "epoch": int(tick.get("epoch", 0)),
            "raw": tick,
        }

        async with self._lock:
            state = self._states[symbol]

            # Evita duplicidade exata do mesmo tick
            if state.last_tick_epoch == normalized_tick["epoch"]:
                return

            state.last_tick_epoch = normalized_tick["epoch"]
            state.recent_ticks.append(normalized_tick)

            listeners = list(state.tick_listeners)

        # Fora do lock para não bloquear o pipeline
        for cb in listeners:
            try:
                await cb(normalized_tick)
            except Exception as exc:
                logger.exception(
                    "Tick listener falhou.",
                    symbol=symbol,
                    error=str(exc),
                )

    def _extract_tick_price(self, tick: dict[str, Any]) -> float:
        quote = tick.get("quote")
        if quote is None:
            return 0.0
        try:
            return float(quote)
        except (TypeError, ValueError):
            return 0.0

    # ── Candles ───────────────────────────────────────────────────────────────

    def _candles_to_df(self, candles: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Converte lista de candles da Deriv em DataFrame padronizado.

        Espera itens no formato:
          {"epoch": 123, "open": "...", "high": "...", "low": "...", "close": "..."}
        """
        if not candles:
            return pd.DataFrame(columns=["epoch", "open", "high", "low", "close"])

        df = pd.DataFrame(candles).copy()

        required = ["epoch", "open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[required]

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required)  # type: ignore

        if df.empty:
            return pd.DataFrame(columns=required)

        df = (
            df.sort_values("epoch")
            .drop_duplicates(subset=["epoch"], keep="last")
            .reset_index(drop=True)
        )

        # Sanidade OHLC
        df = df[
            (df["high"] >= df["low"])
            & (df["high"] >= df[["open", "close"]].max(axis=1))
            & (df["low"] <= df[["open", "close"]].min(axis=1))
        ].reset_index(drop=True)

        return df

    async def refresh_symbol(self, symbol: str) -> None:
        """
        Recarrega candles completos para um símbolo.
        Útil após reconexão ou se detectar staleness.
        """
        candles = await self._client.get_candles(
            symbol=symbol,
            granularity=self._granularity,
            count=self._candle_count,
        )
        df = self._candles_to_df(candles)

        async with self._lock:
            state = self._states[symbol]
            state.candles_df = df
            state.is_ready = not df.empty
            state.last_candle_epoch = int(df.iloc[-1]["epoch"]) if not df.empty else None

        logger.info(
            "Símbolo atualizado.",
            symbol=symbol,
            candles=len(df),
            ready=not df.empty,
        )

    # ── Listeners ─────────────────────────────────────────────────────────────

    def add_tick_listener(self, symbol: str, callback: TickCallback) -> None:
        if symbol not in self._states:
            raise ValueError(f"Símbolo não gerenciado: {symbol}")

        state = self._states[symbol]
        if callback not in state.tick_listeners:
            state.tick_listeners.append(callback)

    def remove_tick_listener(self, symbol: str, callback: TickCallback) -> None:
        if symbol not in self._states:
            return

        state = self._states[symbol]
        state.tick_listeners = [cb for cb in state.tick_listeners if cb is not callback]

    # ── Leitura de estado ─────────────────────────────────────────────────────

    def get_candles_df(self, symbol: str) -> pd.DataFrame:
        state = self._states.get(symbol)
        if state is None or state.candles_df.empty:
            return pd.DataFrame(columns=["epoch", "open", "high", "low", "close"])
        return state.candles_df.copy()

    def get_recent_ticks(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        state = self._states.get(symbol)
        if state is None:
            return []
        if limit <= 0:
            return list(state.recent_ticks)
        return list(state.recent_ticks)[-limit:]

    @property
    def ready_symbols(self) -> list[str]:
        return [symbol for symbol, state in self._states.items() if state.is_ready]

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)