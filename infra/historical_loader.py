"""
infra/historical_loader.py — Download de dados históricos no boot
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from infra.deriv_client import DerivClient
    from infra.symbol_manager import SymbolManager

CANDIDATE_GRANULARITIES = [60, 120, 180, 300]
CANDLES_PER_GRAN        = 1500   # mais candles = mais histórico
FALLBACK_GRAN           = 60
FALLBACK_COUNT          = 800
MIN_CANDLES_REQUIRED    = 200    # mínimo para Bollinger + EMA + RSI


@dataclass
class HistoricalLoader:
    client:         "DerivClient"
    symbol_manager: "SymbolManager"
    symbols:        list[str]
    _loaded: dict[str, bool] = field(default_factory=dict, init=False)

    async def load_all(self, timeout: float = 120.0) -> None:
        logger.info("📥 HistoricalLoader: iniciando...", symbols=self.symbols)
        tasks   = [asyncio.create_task(self._load_symbol(s, timeout)) for s in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for sym, res in zip(self.symbols, results):
            if isinstance(res, Exception):
                logger.warning("Fallback para símbolo.", symbol=sym, error=str(res))
                await self._fallback(sym)
            else:
                self._loaded[sym] = True

        # Após carregar todos, dispara cálculo de features em cada símbolo
        await self._warm_up_features()

        loaded = sum(1 for v in self._loaded.values() if v)
        logger.success("📥 HistoricalLoader concluído.", loaded=loaded, total=len(self.symbols))

    async def _warm_up_features(self) -> None:
        """Força cálculo de features injetando candles históricos no SymbolManager."""
        logger.info("🔥 Aquecendo features históricas...")
        for sym in self.symbols:
            try:
                async with self.symbol_manager._lock:
                    state = self.symbol_manager._states.get(sym)
                if state is None or state.candles_df is None or len(state.candles_df) < 20:
                    logger.warning("Candles insuficientes para warm-up.", symbol=sym)
                    continue

                df   = state.candles_df.copy()
                tail = df.tail(300)

                for _, row in tail.iterrows():
                    # Evita erro de tipagem Any | None do row.get() no Pylance
                    val = row["close"] if "close" in row else row.iloc[-1]
                    price = float(val) if val is not None else 0.0
                    # Tenta os nomes possíveis do método de tick no SymbolManager
                    if hasattr(self.symbol_manager, "_handle_tick"):
                        await self.symbol_manager._handle_tick({"tick": {"symbol": sym, "quote": price, "epoch": 0}})
                    elif hasattr(self.symbol_manager, "_process_tick"):
                        await self.symbol_manager._process_tick(sym, {"symbol": sym, "quote": price, "epoch": 0})
                    elif hasattr(self.symbol_manager, "on_tick"):
                        await self.symbol_manager.on_tick(sym, price)
                    elif hasattr(self.symbol_manager, "_update_symbol"):
                        await self.symbol_manager._update_symbol(sym, price)
                    else:
                        # Injeta diretamente no buffer de preços do estado
                        async with self.symbol_manager._lock:
                            s = self.symbol_manager._states.get(sym)
                            if s is not None and hasattr(s, "price_buffer"):
                                s.price_buffer.append(price)
                            elif s is not None and hasattr(s, "ticks"):
                                s.ticks.append(price)

                logger.info("✅ Features aquecidas.", symbol=sym, candles=len(tail))
            except Exception as exc:
                logger.warning("Warm-up falhou.", symbol=sym, error=str(exc))

    async def _load_symbol(self, symbol: str, timeout: float) -> None:
        best_df, best_gran, best_score = None, FALLBACK_GRAN, -1.0
        per_gran_timeout = timeout / len(CANDIDATE_GRANULARITIES)
        for gran in CANDIDATE_GRANULARITIES:
            try:
                candles = await asyncio.wait_for(
                    self.client.get_candles(symbol=symbol, granularity=gran, count=CANDLES_PER_GRAN),
                    timeout=per_gran_timeout,
                )
                if not candles or len(candles) < MIN_CANDLES_REQUIRED:
                    continue
                df    = self.symbol_manager._candles_to_df(candles)
                score = self._volatility_score(df)
                if score > best_score:
                    best_score, best_df, best_gran = score, df, gran
            except Exception as exc:
                logger.debug("Gran falhou.", symbol=symbol, gran=gran, error=str(exc))

        if best_df is None or len(best_df) < MIN_CANDLES_REQUIRED:
            raise RuntimeError(f"Candles insuficientes para {symbol}")

        async with self.symbol_manager._lock:
            state = self.symbol_manager._states.get(symbol)
            if state is not None:
                state.candles_df = best_df
                state.is_ready   = True
        logger.info("✅ Histórico carregado.", symbol=symbol, gran=best_gran, candles=len(best_df))

    async def _fallback(self, symbol: str) -> None:
        try:
            candles = await asyncio.wait_for(
                self.client.get_candles(symbol=symbol, granularity=FALLBACK_GRAN, count=FALLBACK_COUNT),
                timeout=30.0,
            )
            if candles and len(candles) >= MIN_CANDLES_REQUIRED:
                df = self.symbol_manager._candles_to_df(candles)
                async with self.symbol_manager._lock:
                    s = self.symbol_manager._states.get(symbol)
                    if s is not None:
                        s.candles_df = df
                        s.is_ready   = True
                self._loaded[symbol] = True
                logger.info("✅ Fallback carregado.", symbol=symbol, candles=len(df))
        except Exception as exc:
            logger.error("Fallback falhou.", symbol=symbol, error=str(exc))

    @staticmethod
    def _volatility_score(df: pd.DataFrame) -> float:
        if "close" not in df.columns or len(df) < 10:
            return 0.0
        returns = df["close"].pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return 0.0
        return returns.std() / (abs(returns.mean()) + 1e-9)
