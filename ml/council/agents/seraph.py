"""
ml/council/agents/seraph.py — SERAPH: Análise Técnica Profunda

Detecta padrões de candlestick (Doji, Hammer, Engulfing, Pin Bar, Shooting Star),
suporte/resistência dinâmico via pivots e posição nas Bollinger Bands.
Sem ML: lógica pura sobre OHLCV.
"""
from __future__ import annotations

import pandas as pd

from ml.council.base_agent import AgentVote, BaseAgent


class SeraphAgent(BaseAgent):
    """Especialista em Análise Técnica Profunda — SERAPH."""

    name = "SERAPH"
    weight = 0.13

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        if len(df) < 10:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Dados insuficientes")

        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) >= 3 else prev

        o, h, l, c = (
            self._safe_float(last.get("open")),
            self._safe_float(last.get("high")),
            self._safe_float(last.get("close")),
            self._safe_float(last.get("close")),
        )
        # Recalcula dos dados brutos
        try:
            o = float(last["open"]); h = float(last["high"])
            l = float(last["low"]);  c = float(last["close"])
            po = float(prev["open"]); ph = float(prev["high"])
            pl = float(prev["low"]);  pc = float(prev["close"])
        except Exception:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Erro nos dados OHLCV")

        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        full_range = h - l if h != l else 1e-10

        bullish_signals = 0
        bearish_signals = 0
        patterns_found = []

        # ── Doji (indecisão) ──────────────────────────────────────────────────
        if body < full_range * 0.1:
            # Doji perto de suporte → bullish; perto de resistência → bearish
            bullish_signals += 0.5
            bearish_signals += 0.5
            patterns_found.append("Doji")

        # ── Hammer / Inverted Hammer ──────────────────────────────────────────
        if lower_wick > body * 2 and upper_wick < body * 0.5 and c > pc:
            bullish_signals += 1.5
            patterns_found.append("Hammer")

        # ── Shooting Star ─────────────────────────────────────────────────────
        if upper_wick > body * 2 and lower_wick < body * 0.5 and c < pc:
            bearish_signals += 1.5
            patterns_found.append("ShootingStar")

        # ── Bullish Engulfing ─────────────────────────────────────────────────
        if o < pc and c > po and c > o and pc > po:
            # Candle bullish engulfa o bearish anterior
            bullish_signals += 2.0
            patterns_found.append("BullEngulfing")

        # ── Bearish Engulfing ─────────────────────────────────────────────────
        if o > pc and c < po and c < o and pc < po:
            bearish_signals += 2.0
            patterns_found.append("BearEngulfing")

        # ── Pin Bar bullish ───────────────────────────────────────────────────
        if lower_wick > full_range * 0.6 and body < full_range * 0.3:
            bullish_signals += 1.0
            patterns_found.append("PinBarBull")

        # ── Pin Bar bearish ───────────────────────────────────────────────────
        if upper_wick > full_range * 0.6 and body < full_range * 0.3:
            bearish_signals += 1.0
            patterns_found.append("PinBarBear")

        # ── Posição nas Bollinger Bands ───────────────────────────────────────
        bb_upper = self._safe_float(last.get("bb_upper"))
        bb_lower = self._safe_float(last.get("bb_lower"))
        if bb_upper and bb_lower and bb_upper != bb_lower:
            bb_pos = (c - bb_lower) / (bb_upper - bb_lower)
            if bb_pos < 0.2:       # próximo da banda inferior → bullish
                bullish_signals += 1.0
                patterns_found.append("NearBBLower")
            elif bb_pos > 0.8:     # próximo da banda superior → bearish
                bearish_signals += 1.0
                patterns_found.append("NearBBUpper")

        # ── Suporte/Resistência via pivots ────────────────────────────────────
        recent_highs = df["high"].tail(20).values
        recent_lows  = df["low"].tail(20).values
        resistance   = float(recent_highs.max())
        support      = float(recent_lows.min())
        price_range  = resistance - support if resistance != support else 1e-10

        dist_to_resistance = (resistance - c) / price_range
        dist_to_support    = (c - support)   / price_range

        if dist_to_support < 0.15:    # perto do suporte → bullish
            bullish_signals += 0.8
            patterns_found.append("NearSupport")
        if dist_to_resistance < 0.15: # perto da resistência → bearish
            bearish_signals += 0.8
            patterns_found.append("NearResistance")

        # ── Decisão ───────────────────────────────────────────────────────────
        total = bullish_signals + bearish_signals
        if total == 0:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Sem padrões detectados")

        bull_ratio = bullish_signals / total
        bear_ratio = bearish_signals / total

        sig_dir = signal.direction.value if hasattr(signal.direction, "value") else str(signal.direction)
        is_buy = sig_dir in ("BUY", "buy", "CALL", "call")

        if bull_ratio > 0.65:
            action = "BUY"
            score = min(0.5 + bull_ratio * 0.5, 0.95)
        elif bear_ratio > 0.65:
            action = "SELL"
            score = min(0.5 + bear_ratio * 0.5, 0.95)
        else:
            action = "NEUTRAL"
            score = 0.5

        reasoning = f"Padrões: {', '.join(patterns_found) or 'nenhum'}"
        return AgentVote(self.name, action, score, reasoning=reasoning)
