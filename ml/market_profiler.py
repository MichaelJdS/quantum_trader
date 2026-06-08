"""
ml/market_profiler.py — Profiler de Granularidade e Duração Ótimas
Reavalia a cada 2h automaticamente via background loop.
"""
from __future__ import annotations
import asyncio, statistics, time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import pandas as pd
from loguru import logger

from infra.deriv_client import DerivClient

CANDIDATE_GRANULARITIES = [30, 60, 120, 180, 300]
CANDIDATE_DURATIONS     = [3, 5, 7, 10]
EVAL_CANDLES            = 300
REEVAL_INTERVAL         = 7200  # 2h


@dataclass
class SymbolProfile:
    symbol:      str
    granularity: int   = 60
    duration:    int   = 5
    score:       float = 0.0
    evaluated_at: float = field(default_factory=time.time)


@dataclass
class MarketProfiler:
    client:  DerivClient | None
    symbols: list[str]
    _profiles: dict[str, SymbolProfile] = field(default_factory=dict, init=False)
    _lock:     asyncio.Lock             = field(default_factory=asyncio.Lock, init=False)
    _running:  bool                     = field(default=False, init=False)

    def get_profile(self, symbol: str) -> SymbolProfile:
        return self._profiles.get(symbol, SymbolProfile(symbol=symbol))

    def get_granularity(self, symbol: str) -> int:
        return self.get_profile(symbol).granularity

    def get_duration(self, symbol: str) -> int:
        return self.get_profile(symbol).duration

    async def run_once(self) -> None:
        await asyncio.gather(*[self._profile_symbol(s) for s in self.symbols], return_exceptions=True)
        rows = [f"  {s:8s} → gran={p.granularity:3d}s  dur={p.duration}t  score={p.score:.4f}"
                for s, p in self._profiles.items()]
        logger.info("MarketProfiler resumo:\n" + "\n".join(rows))

    async def start_background(self) -> None:
        self._running = True
        asyncio.create_task(self._background_loop(), name="market_profiler")

    def stop(self) -> None:
        self._running = False

    async def _profile_symbol(self, symbol: str) -> None:
        if self.client is None:
            return
        best, best_score = SymbolProfile(symbol=symbol), -1.0
        for gran in CANDIDATE_GRANULARITIES:
            try:
                candles = await asyncio.wait_for(
                    self.client.get_candles(symbol=symbol, granularity=gran, count=EVAL_CANDLES),
                    timeout=20.0,
                )
            except Exception:
                continue
            if not candles or len(candles) < 50:
                continue
            df = self._to_df(candles)
            for dur in CANDIDATE_DURATIONS:
                score = self._backtest_score(df, dur)
                if score > best_score:
                    best_score = score
                    best = SymbolProfile(symbol=symbol, granularity=gran, duration=dur, score=score)
        async with self._lock:
            self._profiles[symbol] = best
        logger.info("Perfil calculado.", symbol=symbol, gran=best.granularity, dur=best.duration, score=round(best_score,4))

    def _backtest_score(self, df: pd.DataFrame, duration: int) -> float:
        if len(df) < duration + 22:
            return 0.0
        close = df["close"]
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        wins, total, rets = 0, 0, []
        for i in range(21, len(df) - duration):
            if   ema9.iloc[i] > ema21.iloc[i] and ema9.iloc[i-1] <= ema21.iloc[i-1]: direction = "BUY"
            elif ema9.iloc[i] < ema21.iloc[i] and ema9.iloc[i-1] >= ema21.iloc[i-1]: direction = "SELL"
            else: continue
            entry, exit_ = close.iloc[i], close.iloc[i + duration]
            ret = (exit_ - entry) / entry if direction == "BUY" else (entry - exit_) / entry
            wins += int(ret > 0); total += 1; rets.append(ret)
        if total < 5:
            return 0.0
        wr = wins / total
        try:
            std = statistics.stdev(rets) if len(rets) > 1 else 1e-9
            sharpe = (statistics.mean(rets) / (std + 1e-9)) * (total ** 0.5)
        except Exception:
            sharpe = 0.0
        return wr * max(0, sharpe) if wr > 0.50 else wr * 0.5

    async def _background_loop(self) -> None:
        while self._running:
            await asyncio.sleep(REEVAL_INTERVAL)
            logger.info("MarketProfiler: reavaliando...")
            await self.run_once()

    @staticmethod
    def _to_df(candles: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(candles)
        for col in ["open","high","low","close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["close"]).reset_index(drop=True)

    def all_profiles(self) -> dict:
        return {s: {"granularity": p.granularity, "duration": p.duration, "score": round(p.score,4)}
                for s, p in self._profiles.items()}
