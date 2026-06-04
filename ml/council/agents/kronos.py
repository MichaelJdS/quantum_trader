"""
ml/council/agents/kronos.py — KRONOS: Análise Temporal Multi-Escala

Agrega os candles 1m em timeframes 3m e 5m on-the-fly (sem banco de dados).
Verifica alinhamento de tendência entre os 3 timeframes.
Sinal válido apenas se ≥ 2 timeframes concordam.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from ml.council.base_agent import AgentVote, BaseAgent


class KronosAgent(BaseAgent):
    """Especialista em Análise Temporal Multi-Escala — KRONOS."""

    name = "KRONOS"
    weight = 0.12

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        if len(df) < 30:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Dados insuficientes para multi-TF")

        # ── Agrega candles 1m → 3m e 5m on-the-fly ──────────────────────────
        tf1_vote  = self._analyze_timeframe(df, "1m")
        tf3_vote  = self._analyze_timeframe(self._resample(df, 3), "3m")
        tf5_vote  = self._analyze_timeframe(self._resample(df, 5), "5m")

        votes = [tf1_vote, tf3_vote, tf5_vote]
        buy_count  = sum(1 for v in votes if v == "BUY")
        sell_count = sum(1 for v in votes if v == "SELL")

        # ── Decisão: precisa de maioria (2/3) ─────────────────────────────────
        if buy_count >= 2:
            score = 0.70 if buy_count == 3 else 0.60
            return AgentVote(
                self.name, "BUY", score,
                reasoning=f"Multi-TF BUY ({buy_count}/3 concordam: 1m={tf1_vote} 3m={tf3_vote} 5m={tf5_vote})"
            )
        if sell_count >= 2:
            score = 0.70 if sell_count == 3 else 0.60
            return AgentVote(
                self.name, "SELL", score,
                reasoning=f"Multi-TF SELL ({sell_count}/3 concordam: 1m={tf1_vote} 3m={tf3_vote} 5m={tf5_vote})"
            )

        return AgentVote(
            self.name, "NEUTRAL", 0.4,
            reasoning=f"Desacordo multi-TF: 1m={tf1_vote} 3m={tf3_vote} 5m={tf5_vote}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resample(self, df: pd.DataFrame, n_bars: int) -> pd.DataFrame:
        """Agrega n_bars candles de 1m em 1 candle do timeframe maior."""
        if len(df) < n_bars * 3:
            return df

        rows = []
        closes = df["close"].astype(float).values
        highs  = df["high"].astype(float).values if "high" in df.columns else closes
        lows   = df["low"].astype(float).values  if "low"  in df.columns else closes
        opens  = df["open"].astype(float).values  if "open" in df.columns else closes

        for i in range(0, len(df) - n_bars + 1, n_bars):
            chunk = slice(i, i + n_bars)
            rows.append({
                "open":  opens[i],
                "high":  float(highs[chunk].max()),
                "low":   float(lows[chunk].min()),
                "close": closes[i + n_bars - 1],
            })
            for col in ["ema_9", "ema_21", "ema_50", "rsi_14", "macd_hist"]:
                if col in df.columns:
                    rows[-1][col] = float(df[col].iloc[i + n_bars - 1])

        return pd.DataFrame(rows) if rows else df

    def _analyze_timeframe(self, df: pd.DataFrame, label: str) -> str:
        """Retorna BUY / SELL / NEUTRAL para um timeframe."""
        if len(df) < 5:
            return "NEUTRAL"

        last = df.iloc[-1]
        prev = df.iloc[-2]

        try:
            ema9  = float(last.get("ema_9",  last["close"]))
            ema21 = float(last.get("ema_21", last["close"]))
            ema50 = float(last.get("ema_50", last["close"]))
            rsi   = float(last.get("rsi_14", 50.0))
            macd  = float(last.get("macd_hist", 0.0))
            close = float(last["close"])
        except Exception:
            return "NEUTRAL"

        bull = (ema9 > ema21) and (rsi > 45) and (macd > 0)
        bear = (ema9 < ema21) and (rsi < 55) and (macd < 0)

        if bull:
            return "BUY"
        if bear:
            return "SELL"
        return "NEUTRAL"
