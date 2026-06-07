"""
ml/market_profiler.py — MarketProfiler v1.0

Analisa dados históricos multi-granularidade e decide:
  - Granularidade ideal de candle (30s / 60s / 120s / 180s / 240s / 300s)
  - Duração ideal do contrato (5t–10t ou 30s–300s)

O resultado é persistido em disco e reavaliado a cada REEVAL_HOURS horas.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    pass

# Granularidades a testar (em segundos)
CANDIDATE_GRANULARITIES = [30, 60, 120, 180, 240, 300]

# Durações a testar: (valor, unidade)
CANDIDATE_DURATIONS = [
    (5, "t"), (6, "t"), (7, "t"), (8, "t"), (9, "t"), (10, "t"),
    (30, "s"), (60, "s"), (120, "s"), (180, "s"), (300, "s"),
]

PROFILES_PATH = os.path.join(os.path.dirname(__file__), "states", "market_profiles.json")
REEVAL_HOURS  = 2.0


@dataclass
class SymbolProfile:
    symbol:       str
    granularity:  int    # segundos
    duration:     int    # valor numérico
    duration_unit: str   # "t" ou "s"
    score:        float  # score do profiler
    win_rate:     float  # win rate backtestado
    evaluated_at: float  = 0.0

    def needs_reeval(self) -> bool:
        return (time.time() - self.evaluated_at) > REEVAL_HOURS * 3600


class MarketProfiler:
    """
    Recebe DataFrames de múltiplas granularidades e decide
    a combinação ótima para cada símbolo.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, SymbolProfile] = {}
        self._load_profiles()

    # ── API pública ──────────────────────────────────────────────────────────

    def profile(
        self,
        symbol: str,
        multi_dfs: dict[int, pd.DataFrame],  # {granularity_secs: df_com_features}
    ) -> SymbolProfile:
        """
        Analisa todos os DataFrames e retorna o SymbolProfile ótimo.
        Sempre reavalia se dados novos chegarem ou TTL expirou.
        """
        existing = self._profiles.get(symbol)
        if existing and not existing.needs_reeval() and multi_dfs:
            logger.debug("MarketProfiler: usando perfil cached.", symbol=symbol,
                         gran=existing.granularity, dur=existing.duration)
            return existing

        best_profile = self._evaluate(symbol, multi_dfs)
        self._profiles[symbol] = best_profile
        self._save_profiles()

        logger.success(
            "MarketProfiler: perfil definido.",
            symbol=symbol,
            granularity=f"{best_profile.granularity}s",
            duration=f"{best_profile.duration}{best_profile.duration_unit}",
            score=round(best_profile.score, 4),
            win_rate=f"{best_profile.win_rate:.1%}",
        )
        return best_profile

    def get_profile(self, symbol: str) -> SymbolProfile | None:
        return self._profiles.get(symbol)

    # ── Avaliação interna ────────────────────────────────────────────────────

    def _evaluate(
        self,
        symbol: str,
        multi_dfs: dict[int, pd.DataFrame],
    ) -> SymbolProfile:
        best_score = -1.0
        best = SymbolProfile(
            symbol=symbol,
            granularity=60,
            duration=5,
            duration_unit="t",
            score=0.5,
            win_rate=0.5,
            evaluated_at=time.time(),
        )

        for gran, df in multi_dfs.items():
            if df is None or len(df) < 80:
                continue

            gran_metrics = self._analyze_granularity(df)

            for dur, unit in CANDIDATE_DURATIONS:
                # Simula win_rate com a estratégia EMA crossover simples
                sim_wr = self._simulate_win_rate(df, dur, unit)
                # Score composto
                score = (
                    sim_wr            * 0.50 +
                    gran_metrics["sharpe"]     * 0.20 +
                    gran_metrics["frequency"]  * 0.15 +
                    gran_metrics["clarity"]    * 0.15
                )
                if score > best_score:
                    best_score = score
                    best = SymbolProfile(
                        symbol=symbol,
                        granularity=gran,
                        duration=dur,
                        duration_unit=unit,
                        score=round(score, 5),
                        win_rate=round(sim_wr, 4),
                        evaluated_at=time.time(),
                    )

        return best

    def _analyze_granularity(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Calcula métricas de qualidade para uma granularidade:
          - sharpe: retorno/risco dos candles
          - frequency: quantos sinais EMA gera por 100 candles (normalizado)
          - clarity: nitidez da tendência (ADX médio normalizado)
        """
        closes = df["close"].values.astype(float)
        returns = np.diff(closes) / closes[:-1]

        # Sharpe simples (diário, sem fator de escala)
        mean_r = np.mean(returns)
        std_r  = np.std(returns)
        sharpe = float(np.clip(mean_r / (std_r + 1e-9), 0, 3) / 3)  # normaliza 0-1

        # Frequência de sinais EMA (cruzamentos por 100 candles, cap 30)
        if "ema_short" in df.columns and "ema_long" in df.columns:
            ema_s = df["ema_short"].values
            ema_l = df["ema_long"].values
            crossings = np.sum(np.diff(np.sign(ema_s[1:] - ema_l[1:])) != 0)
            freq = min(crossings / max(len(df), 1) * 100, 30) / 30
        else:
            freq = 0.5

        # Clareza (ADX médio)
        if "adx" in df.columns:
            adx_mean = df["adx"].mean()
            clarity = np.clip(adx_mean / 50.0, 0, 1)
        else:
            clarity = 0.4

        return {"sharpe": sharpe, "frequency": freq, "clarity": clarity}

    def _simulate_win_rate(
        self, df: pd.DataFrame, duration: int, unit: str
    ) -> float:
        """
        Backtesta win_rate com EMA crossover simples.
        - Se unit == "t": projeta `duration` ticks à frente
        - Se unit == "s": converte em candles com base na granularidade
        """
        if len(df) < 60:
            return 0.5

        closes = df["close"].values.astype(float)

        if "ema_short" in df.columns and "ema_long" in df.columns:
            ema_s = df["ema_short"].values
            ema_l = df["ema_long"].values
        else:
            ema_s = pd.Series(closes).ewm(span=9).mean().values
            ema_l = pd.Series(closes).ewm(span=21).mean().values

        # Para "t" assume cada tick ≈ 1 candle (aproximação para backtesting)
        step = max(1, duration)
        wins, total = 0, 0

        for i in range(20, len(closes) - step):
            # Sinal
            if ema_s[i] > ema_l[i] and ema_s[i - 1] <= ema_l[i - 1]:
                direction = "BUY"
            elif ema_s[i] < ema_l[i] and ema_s[i - 1] >= ema_l[i - 1]:
                direction = "SELL"
            else:
                continue

            future_price = closes[i + step]
            current_price = closes[i]

            if direction == "BUY":
                won = future_price > current_price
            else:
                won = future_price < current_price

            wins += int(won)
            total += 1

        if total < 5:
            return 0.5
        return wins / total

    # ── Persistência ─────────────────────────────────────────────────────────

    def _save_profiles(self) -> None:
        os.makedirs(os.path.dirname(PROFILES_PATH), exist_ok=True)
        try:
            data = {sym: asdict(p) for sym, p in self._profiles.items()}
            with open(PROFILES_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning("MarketProfiler: falha ao salvar perfis.", error=str(exc))

    def _load_profiles(self) -> None:
        try:
            if os.path.exists(PROFILES_PATH):
                with open(PROFILES_PATH) as f:
                    data = json.load(f)
                for sym, d in data.items():
                    self._profiles[sym] = SymbolProfile(**d)
                logger.info(
                    "MarketProfiler: perfis carregados.",
                    symbols=list(self._profiles.keys()),
                )
        except Exception as exc:
            logger.warning("MarketProfiler: falha ao carregar perfis.", error=str(exc))
