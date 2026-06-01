from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from loguru import logger

from core.entities import SymbolConfig
from core.exceptions import SymbolNotSupportedError
from core.settings import get_settings
from infra.deriv_client import DerivClient, MessageCallback
from infra.db.database import get_session
from infra.db.repository import CandleRepository, TickRepository


# ── Dados de tick ao vivo por símbolo ─────────────────────────────────────────

@dataclass
class SymbolState:
    """Estado em tempo real de um símbolo."""

    config: SymbolConfig
    last_price: float = 0.0
    last_epoch: int = 0
    tick_count: int = 0
    candles_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    is_ready: bool = False  # True após carga inicial de candles.


# ── Symbol Manager ────────────────────────────────────────────────────────────

class SymbolManager:
    """
    Gerencia múltiplos símbolos simultaneamente.

    Responsabilidades:
      - Carregar candles históricos de todos os símbolos no início.
      - Manter stream de ticks ativos via WebSocket.
      - Atualizar DataFrame de candles ao vivo.
      - Persistir ticks e candles no banco de dados.
      - Notificar listeners externos sobre novos ticks/candles.
      - Fornecer acesso thread-safe ao estado de cada símbolo.
    """

    # Símbolos suportados pela plataforma Deriv (Volatility Index).
    SUPPORTED_SYMBOLS: frozenset[str] = frozenset({
        "R_10", "R_25", "R_50", "R_75", "R_100",
        "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",
        "V10", "V25", "V50", "V75", "V100",
        "BOOM300N", "BOOM500", "BOOM1000",
        "CRASH300N", "CRASH500", "CRASH1000",
        "STPRNG",
    })

    def __init__(
        self,
        client: DerivClient,
        symbols: list[str] | None = None,
        granularity: int | None = None,
    ) -> None:
        settings = get_settings()
        self._client = client
        self._granularity = granularity or settings.default_granularity
        self._states: dict[str, SymbolState] = {}
        self._tick_listeners: dict[str, list[MessageCallback]] = defaultdict(list)
        self._candle_listeners: dict[str, list[MessageCallback]] = defaultdict(list)
        self._lock = asyncio.Lock()

        raw_symbols = symbols or settings.symbols_list
        for sym in raw_symbols:
            self._register_symbol(sym)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _register_symbol(self, symbol: str) -> None:
        """Valida e registra símbolo para tracking."""
        if symbol not in self.SUPPORTED_SYMBOLS:
            logger.warning(
                "Símbolo não reconhecido na lista padrão — prosseguindo mesmo assim.",
                symbol=symbol,
            )
        self._states[symbol] = SymbolState(
            config=SymbolConfig(
                name=symbol,
                granularity=self._granularity,
            )
        )
        logger.debug("Símbolo registrado.", symbol=symbol)

    async def initialize(self) -> None:
        """
        Inicializa todos os símbolos em paralelo:
          1. Carrega candles históricos do banco (se existirem).
          2. Busca candles frescos da API Deriv.
          3. Inicia subscriptions de ticks.
        """
        logger.info(
            "Inicializando SymbolManager.",
            symbols=list(self._states.keys()),
            granularity=self._granularity,
        )

        tasks = [
            self._initialize_symbol(symbol)
            for symbol in self._states
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(self._states.keys(), results):
            if isinstance(result, Exception):
                logger.error(
                    "Falha ao inicializar símbolo.",
                    symbol=symbol,
                    error=str(result),
                )

        ready_count = sum(1 for s in self._states.values() if s.is_ready)
        logger.success(
            "SymbolManager inicializado.",
            ready=ready_count,
            total=len(self._states),
        )

    async def _initialize_symbol(self, symbol: str) -> None:
        """Inicialização de um símbolo individual."""
        state = self._states[symbol]

        # 1. Carrega candles da API.
        candles = await self._client.get_candles(
            symbol=symbol,
            granularity=self._granularity,
            count=500,
        )

        if candles:
            state.candles_df = self._candles_to_df(candles)
            state.is_ready = True
            logger.info(
                "Candles carregados.",
                symbol=symbol,
                count=len(candles),
            )

            # 2. Persiste candles no banco.
            async with get_session() as db:
                repo = CandleRepository(db)
                await repo.bulk_upsert(symbol, self._granularity, candles)

        # 3. Subscribe a ticks ao vivo.
        await self._client.subscribe_ticks(
            symbol=symbol,
            callback=None,
        )
        self._client.on("tick", self._handle_tick)

    # ── Handlers ──────────────────────────────────────────────────────────────

    async def _handle_tick(self, data: dict[str, Any]) -> None:
        """
        Callback chamado para cada tick recebido.

        Atualiza estado do símbolo, persiste no banco e notifica listeners.
        """
        tick = data.get("tick", {})
        symbol: str = tick.get("symbol", "")

        if symbol not in self._states:
            return

        price = float(tick.get("ask", tick.get("bid", 0.0)))
        epoch = int(tick.get("epoch", 0))
        pip_size = tick.get("pip_size")

        async with self._lock:
            state = self._states[symbol]
            state.last_price = price
            state.last_epoch = epoch
            state.tick_count += 1

            # Atualiza último candle do DataFrame com o preço atual.
            self._update_live_candle(state, price, epoch)

        # Persiste tick no banco (batch a cada 100 ticks por símbolo).
        if state.tick_count % 100 == 0:
            asyncio.create_task(
                self._persist_ticks(symbol, [{"symbol": symbol, "price": price,
                                               "epoch": epoch, "pip_size": pip_size}])
            )

        # Notifica listeners externos.
        for listener in self._tick_listeners.get(symbol, []):
            asyncio.create_task(listener(data))

        logger.trace(
            "Tick recebido.",
            symbol=symbol,
            price=price,
            epoch=epoch,
        )

    def _update_live_candle(
        self,
        state: SymbolState,
        price: float,
        epoch: int,
    ) -> None:
        """
        Atualiza o high/low/close do candle atual.
        Cria novo candle quando a janela de tempo encerrou.
        """
        if state.candles_df.empty:
            return

        last_epoch = int(state.candles_df.iloc[-1]["epoch"])
        candle_end = last_epoch + self._granularity

        if epoch >= candle_end:
            # Novo candle.
            new_row = pd.DataFrame([{
                "open": price, "high": price,
                "low": price, "close": price,
                "epoch": epoch,
            }])
            state.candles_df = pd.concat(
                [state.candles_df, new_row], ignore_index=True
            ).tail(1000)  # Mantém apenas últimos 1000 candles em memória.
        else:
            # Atualiza candle corrente.
            idx = len(state.candles_df) - 1
            state.candles_df.at[idx, "close"] = price
            state.candles_df.at[idx, "high"] = max(
                state.candles_df.at[idx, "high"], price
            )
            state.candles_df.at[idx, "low"] = min(
                state.candles_df.at[idx, "low"], price
            )

    # ── Persistência ──────────────────────────────────────────────────────────

    async def _persist_ticks(
        self,
        symbol: str,
        ticks: list[dict],
    ) -> None:
        try:
            async with get_session() as db:
                repo = TickRepository(db)
                await repo.bulk_save(ticks)
        except Exception as exc:
            logger.error("Falha ao persistir ticks.", symbol=symbol, error=str(exc))

    # ── API Pública ───────────────────────────────────────────────────────────

    def get_candles_df(self, symbol: str) -> pd.DataFrame:
        """Retorna DataFrame de candles ao vivo do símbolo."""
        if symbol not in self._states:
            raise SymbolNotSupportedError(f"Símbolo não registrado: {symbol}")
        return self._states[symbol].candles_df.copy()

    def get_last_price(self, symbol: str) -> float:
        """Retorna último preço recebido do símbolo."""
        if symbol not in self._states:
            raise SymbolNotSupportedError(f"Símbolo não registrado: {symbol}")
        return self._states[symbol].last_price

    def is_symbol_ready(self, symbol: str) -> bool:
        """True se símbolo tem candles carregados e está operacional."""
        return self._states.get(symbol, SymbolState(
            config=SymbolConfig(name=symbol)
        )).is_ready

    def add_tick_listener(self, symbol: str, callback: MessageCallback) -> None:
        """Registra callback para ticks de um símbolo específico."""
        self._tick_listeners[symbol].append(callback)

    def remove_tick_listener(self, symbol: str, callback: MessageCallback) -> None:
        self._tick_listeners[symbol] = [
            cb for cb in self._tick_listeners[symbol] if cb is not callback
        ]

    async def add_symbol(self, symbol: str) -> None:
        """Adiciona e inicializa um novo símbolo em runtime."""
        if symbol in self._states:
            logger.warning("Símbolo já registrado.", symbol=symbol)
            return
        self._register_symbol(symbol)
        await self._initialize_symbol(symbol)

    async def remove_symbol(self, symbol: str) -> None:
        """Remove símbolo e cancela sua subscription."""
        if symbol not in self._states:
            return
        await self._client.unsubscribe_ticks(symbol)
        self._states.pop(symbol)
        logger.info("Símbolo removido.", symbol=symbol)

    async def shutdown(self) -> None:
        """Cancela todas as subscriptions e libera recursos."""
        await self._client.unsubscribe_all()
        logger.info("SymbolManager encerrado.")

    @property
    def symbols(self) -> list[str]:
        return list(self._states.keys())

    @property
    def ready_symbols(self) -> list[str]:
        return [s for s, state in self._states.items() if state.is_ready]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _candles_to_df(candles: list[dict]) -> pd.DataFrame:
        """Converte lista de candles da API para DataFrame tipado."""
        df = pd.DataFrame(candles)
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df.sort_values("epoch", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df