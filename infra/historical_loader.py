"""
infra/historical_loader.py — HistoricalLoader v1.0

No boot do sistema:
  1. Baixa 1000 candles em TODAS as granularidades candidatas para cada símbolo
  2. Roda FeatureEngineer em cada DataFrame
  3. Envia tudo ao MarketProfiler → SymbolProfile(gran, duration, unit)
  4. Atualiza o SymbolManager com a granularidade ótima
  5. Retorna dict[symbol → SymbolProfile] para o ExecutionEngine
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from ml.feature_engineer import FeatureEngineer
from ml.market_profiler import MarketProfiler, CANDIDATE_GRANULARITIES

if TYPE_CHECKING:
    from infra.deriv_client import DerivClient
    from infra.symbol_manager import SymbolManager
    from ml.market_profiler import SymbolProfile


class HistoricalLoader:
    """
    Responsável pelo carregamento histórico inteligente no boot.
    """

    CANDLES_PER_GRAN = 1000   # candles por granularidade

    def __init__(
        self,
        client: "DerivClient",
        symbol_manager: "SymbolManager",
        profiler: MarketProfiler | None = None,
    ) -> None:
        self._client   = client
        self._sm       = symbol_manager
        self._profiler = profiler or MarketProfiler()
        self._fe       = FeatureEngineer()

    async def load_all(self) -> dict[str, SymbolProfile]:
        """
        Executa o boot histórico para todos os símbolos gerenciados.
        Retorna map symbol → SymbolProfile com a decisão ótima.
        """
        symbols = self._sm.symbols
        logger.info(
            "HistoricalLoader: iniciando boot multi-granularidade.",
            symbols=symbols,
            granularities=CANDIDATE_GRANULARITIES,
        )

        profiles: dict[str, SymbolProfile] = {}

        # Paraleliza por símbolo (max 3 simultâneos para não estressar a API)
        semaphore = asyncio.Semaphore(3)

        async def load_symbol(sym: str) -> None:
            async with semaphore:
                p = await self._load_symbol(sym)
                profiles[sym] = p

        await asyncio.gather(*[load_symbol(sym) for sym in symbols])

        logger.success(
            "HistoricalLoader: boot concluído.",
            profiles={
                sym: f"{p.granularity}s/{p.duration}{p.duration_unit}"
                for sym, p in profiles.items()
            },
        )
        return profiles

    async def _load_symbol(self, symbol: str) -> SymbolProfile:
        """Baixa dados em todas as granularidades e perfia o símbolo."""
        multi_dfs = {}

        for gran in CANDIDATE_GRANULARITIES:
            try:
                candles = await self._client.get_candles(
                    symbol=symbol,
                    granularity=gran,
                    count=self.CANDLES_PER_GRAN,
                )
                if not candles:
                    continue

                raw_df = self._sm._candles_to_df(candles)
                if raw_df.empty or len(raw_df) < 80:
                    continue

                feat_df = self._fe.compute(raw_df)
                if not feat_df.empty:
                    multi_dfs[gran] = feat_df

                logger.debug(
                    "Granularidade carregada.",
                    symbol=symbol,
                    gran=f"{gran}s",
                    candles=len(feat_df),
                )

                # Pequena pausa entre requisições para respeitar rate limit
                await asyncio.sleep(0.15)

            except Exception as exc:
                logger.warning(
                    "Falha ao carregar granularidade.",
                    symbol=symbol,
                    gran=gran,
                    error=str(exc),
                )

        if not multi_dfs:
            logger.warning(
                "Nenhuma granularidade carregada — usando fallback 60s/5t.",
                symbol=symbol,
            )
            from ml.market_profiler import SymbolProfile
            import time
            return SymbolProfile(
                symbol=symbol, granularity=60, duration=5,
                duration_unit="t", score=0.0, win_rate=0.5,
                evaluated_at=time.time(),
            )

        # Roda profiler
        profile = self._profiler.profile(symbol, multi_dfs)

        # Atualiza SymbolManager com a granularidade ótima + DataFrame já carregado
        best_df = multi_dfs[profile.granularity]
        async with self._sm._lock:
            self._sm._states[symbol].candles_df   = best_df
            self._sm._states[symbol].is_ready     = True
            self._sm._granularity                 = profile.granularity

        return profile
