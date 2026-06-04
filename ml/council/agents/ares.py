"""
ml/council/agents/ares.py — ARES: Execução & Microestrutura

Analisa ticks recentes para detectar:
- Velocidade de ticks (aceleração ou desaceleração de preço)
- Momentum direcional nos últimos N ticks
- Qualidade de entrada (retracement ideal vs. breakout)
Sem ticks disponíveis, usa candle atual como proxy.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from ml.council.base_agent import AgentVote, BaseAgent


class AresAgent(BaseAgent):
    """Especialista em Execução & Microestrutura — ARES."""

    name = "ARES"
    weight = 0.10

    TICK_WINDOW = 15  # últimos N ticks para análise

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        sig_dir = signal.direction.value if hasattr(signal.direction, "value") else str(signal.direction)
        is_buy  = sig_dir in ("BUY", "buy", "CALL", "call")

        if ticks and len(ticks) >= 5:
            return self._analyze_ticks(ticks, is_buy)
        else:
            return self._analyze_from_candles(df, is_buy)

    # ── Análise via ticks reais ───────────────────────────────────────────────

    def _analyze_ticks(self, ticks: list[dict], is_buy: bool) -> AgentVote:
        recent = ticks[-self.TICK_WINDOW:]
        prices = []
        times  = []
        for t in recent:
            p = t.get("quote", t.get("price", t.get("bid")))
            ts = t.get("epoch", t.get("time", time.time()))
            if p is not None:
                prices.append(float(p))
                times.append(float(ts))

        if len(prices) < 3:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Ticks insuficientes")

        # Momentum de preço
        price_change = (prices[-1] - prices[0]) / (abs(prices[0]) + 1e-9)
        momentum_bull = price_change > 0.0001
        momentum_bear = price_change < -0.0001

        # Velocidade: ticks por segundo
        if len(times) >= 2:
            duration = max(times[-1] - times[0], 1)
            tick_speed = len(prices) / duration  # ticks/s
        else:
            tick_speed = 1.0

        # Aceleração: compara velocidade na 1a metade vs. 2a metade
        mid = len(prices) // 2
        first_half_change  = abs(prices[mid] - prices[0])
        second_half_change = abs(prices[-1] - prices[mid])
        accelerating = second_half_change > first_half_change * 1.2

        # Reversão: preço voltando após movimento
        retracing = (
            (is_buy  and price_change < 0 and prices[-1] > prices[0] - 3 * (prices[-1] - prices[-2])) or
            (not is_buy and price_change > 0)
        )

        # Score
        if is_buy:
            if momentum_bull and accelerating:
                score = 0.78; action = "BUY"; note = "Aceleração bullish"
            elif momentum_bull:
                score = 0.65; action = "BUY"; note = "Momentum bullish"
            elif retracing:
                score = 0.60; action = "BUY"; note = "Retracement ideal"
            else:
                score = 0.45; action = "NEUTRAL"; note = "Momentum contrário"
        else:
            if momentum_bear and accelerating:
                score = 0.78; action = "SELL"; note = "Aceleração bearish"
            elif momentum_bear:
                score = 0.65; action = "SELL"; note = "Momentum bearish"
            else:
                score = 0.45; action = "NEUTRAL"; note = "Momentum contrário"

        return AgentVote(self.name, action, score,
                         reasoning=f"Ticks: {note} Δ={price_change:.5f} speed={tick_speed:.1f}/s")

    # ── Fallback: análise via candles ─────────────────────────────────────────

    def _analyze_from_candles(self, df: pd.DataFrame, is_buy: bool) -> AgentVote:
        if len(df) < 5:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Candles insuficientes")

        closes = df["close"].tail(10).astype(float).values
        price_change = (closes[-1] - closes[0]) / (abs(closes[0]) + 1e-9)

        # Tendência de curto prazo dos últimos 3 candles
        last3 = closes[-3:]
        up_count   = sum(1 for i in range(1, len(last3)) if last3[i] > last3[i-1])
        down_count = len(last3) - 1 - up_count

        atr = self._safe_float(df.iloc[-1].get("atr_14"), 0.001)
        last_move = abs(closes[-1] - closes[-2])
        move_rel_atr = last_move / (atr + 1e-9)

        if is_buy:
            if up_count >= 2 and price_change > 0:
                score = min(0.55 + move_rel_atr * 0.1, 0.80)
                return AgentVote(self.name, "BUY", score,
                                 reasoning=f"Candles subindo ({up_count}/2), Δ={price_change:.4f}")
            elif down_count >= 2:
                return AgentVote(self.name, "NEUTRAL", 0.45,
                                 reasoning=f"Candles caindo — entrada contra-tendência de curto prazo")
        else:
            if down_count >= 2 and price_change < 0:
                score = min(0.55 + move_rel_atr * 0.1, 0.80)
                return AgentVote(self.name, "SELL", score,
                                 reasoning=f"Candles caindo ({down_count}/2), Δ={price_change:.4f}")
            elif up_count >= 2:
                return AgentVote(self.name, "NEUTRAL", 0.45,
                                 reasoning=f"Candles subindo — entrada contra-tendência de curto prazo")

        return AgentVote(self.name, "NEUTRAL", 0.5,
                         reasoning=f"Microestrutura neutra Δ={price_change:.4f}")
