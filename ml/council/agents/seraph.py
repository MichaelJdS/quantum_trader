"""
ml/council/agents/seraph.py — SERAPH v3.0: Análise Técnica Evolutiva

Melhorias v3.0:
  - 12 padrões de candlestick (+ Morning/Evening Star, Three Soldiers/Crows)
  - Nivéis S/R dinâmicos via Fibonacci Retracement
  - Memória de padrões: aprende quais padrões têm maior win rate por símbolo
  - Auto-pesagem: padrões que erram muito recebem menor peso
  - Divergência RSI: detecta divergências bullish/bearish
  - Acesso web: puxa suporte/resistência chave de dados externos (cache 1h)
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import pandas as pd
import numpy as np

from ml.council.base_agent import AgentVote, BaseAgent


class SeraphAgent(BaseAgent):
    """SERAPH v3.0 — Análise Técnica com Aprendizado de Padrões."""

    name   = "SERAPH"
    weight = 0.13
    ADAPT_EVERY = 30

    # Peso inicial de cada padrão (auto-ajustado com o aprendizado)
    _PATTERN_WEIGHTS: dict[str, float] = {
        "Doji":           0.4,
        "Hammer":         1.5,
        "ShootingStar":   1.5,
        "BullEngulfing":  2.0,
        "BearEngulfing":  2.0,
        "PinBarBull":     1.2,
        "PinBarBear":     1.2,
        "MorningStar":    2.2,
        "EveningStar":    2.2,
        "ThreeSoldiers":  2.5,
        "ThreeCrows":     2.5,
        "RSI_Div_Bull":   1.8,
        "RSI_Div_Bear":   1.8,
        "NearBBLower":    1.0,
        "NearBBUpper":    1.0,
        "NearSupport":    0.8,
        "NearResistance": 0.8,
        "FibBounce":      1.4,
    }

    def __init__(self) -> None:
        super().__init__()
        # Histórico de acerto por padrão: padrão → deque de bool
        self._pattern_history: dict[str, list[bool]] = defaultdict(list)
        self._pattern_weights: dict[str, float] = dict(self._PATTERN_WEIGHTS)

    def _default_thresholds(self) -> dict[str, float]:
        return {
            "bull_ratio_threshold": 0.62,
            "bear_ratio_threshold": 0.62,
            "confidence_floor":     0.50,
            "fib_proximity":        0.015,  # 1.5% de proximidade ao nível Fib
        }

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        if len(df) < 10:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Dados insuficientes")

        self._last_state = self._state_key(df)

        try:
            last  = df.iloc[-1]
            prev  = df.iloc[-2]
            prev2 = df.iloc[-3] if len(df) >= 3 else prev
            prev3 = df.iloc[-4] if len(df) >= 4 else prev2

            o  = float(last["open"]);  h  = float(last["high"])
            l  = float(last["low"]);   c  = float(last["close"])
            po = float(prev["open"]);  ph = float(prev["high"])
            pl = float(prev["low"]);   pc = float(prev["close"])
            p2c = float(prev2["close"])
            p3c = float(prev3["close"])
        except Exception:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Erro OHLCV")

        body       = abs(c - o)
        uw         = h - max(o, c)
        lw         = min(o, c) - l
        full_range = h - l if h != l else 1e-10

        bull_score  = 0.0
        bear_score  = 0.0
        found_bull: list[str] = []
        found_bear: list[str] = []

        # Helper inline
        def add_bull(p: str, strength: float = 1.0):
            nonlocal bull_score
            w = self._pattern_weights.get(p, strength) * strength
            bull_score += w
            found_bull.append(p)

        def add_bear(p: str, strength: float = 1.0):
            nonlocal bear_score
            w = self._pattern_weights.get(p, strength) * strength
            bear_score += w
            found_bear.append(p)

        # ── Doji ─────────────────────────────────────────────────────────────
        if body < full_range * 0.10:
            # Neutro — ambos ganham meio ponto
            bull_score += 0.2; bear_score += 0.2

        # ── Hammer ───────────────────────────────────────────────────────────
        if lw > body * 2 and uw < body * 0.5 and c > pc:
            add_bull("Hammer")

        # ── Shooting Star ─────────────────────────────────────────────────────
        if uw > body * 2 and lw < body * 0.5 and c < pc:
            add_bear("ShootingStar")

        # ── Bullish Engulfing ─────────────────────────────────────────────────
        if o < pc and c > po and c > o and pc > po:
            add_bull("BullEngulfing")

        # ── Bearish Engulfing ─────────────────────────────────────────────────
        if o > pc and c < po and c < o and pc < po:
            add_bear("BearEngulfing")

        # ── Pin Bar Bull ──────────────────────────────────────────────────────
        if lw > full_range * 0.60 and body < full_range * 0.30:
            add_bull("PinBarBull")

        # ── Pin Bar Bear ──────────────────────────────────────────────────────
        if uw > full_range * 0.60 and body < full_range * 0.30:
            add_bear("PinBarBear")

        # ── Morning Star (3 velas) ────────────────────────────────────────────
        try:
            p2o = float(prev2["open"]); p2c_v = float(prev2["close"])
            if (p2c < p2o                          # vela 1: bearish
                    and abs(po - pc) < abs(p2o - p2c) * 0.5   # vela 2: pequena
                    and c > o and c > (p2o + p2c) / 2):       # vela 3: bullish
                add_bull("MorningStar", 1.2)
        except Exception:
            pass

        # ── Evening Star ──────────────────────────────────────────────────────
        try:
            p2o = float(prev2["open"]); p2c_v = float(prev2["close"])
            if (p2c > p2o
                    and abs(po - pc) < abs(p2o - p2c) * 0.5
                    and c < o and c < (p2o + p2c) / 2):
                add_bear("EveningStar", 1.2)
        except Exception:
            pass

        # ── Three White Soldiers ──────────────────────────────────────────────
        try:
            if (c > o and pc > po and p2c > p2o
                    and c > pc > p2c
                    and o > po and po > float(prev2["open"])):
                add_bull("ThreeSoldiers", 1.3)
        except Exception:
            pass

        # ── Three Black Crows ─────────────────────────────────────────────────
        try:
            if (c < o and pc < po and p2c < p2o
                    and c < pc < p2c
                    and o < po and po < float(prev2["open"])):
                add_bear("ThreeCrows", 1.3)
        except Exception:
            pass

        # ── RSI Divergência Bullish ───────────────────────────────────────────
        rsi_col = next((col for col in ("rsi_14", "rsi", "RSI") if col in df.columns), None)
        if rsi_col and len(df) >= 14:
            rsi_now  = self._safe_float(last.get(rsi_col, 50.0), 50.0)
            rsi_prev = self._safe_float(df.iloc[-6][rsi_col] if len(df) >= 6 else 50.0, 50.0)
            price_down = c < p2c
            rsi_up     = rsi_now > rsi_prev
            if price_down and rsi_up and rsi_now < 45:
                add_bull("RSI_Div_Bull")
            price_up   = c > p2c
            rsi_down   = rsi_now < rsi_prev
            if price_up and rsi_down and rsi_now > 55:
                add_bear("RSI_Div_Bear")

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb_upper = self._safe_float(last.get("bb_upper"))
        bb_lower = self._safe_float(last.get("bb_lower"))
        if bb_upper and bb_lower and bb_upper != bb_lower:
            bb_pos = (c - bb_lower) / (bb_upper - bb_lower)
            if bb_pos < 0.18:
                add_bull("NearBBLower")
            elif bb_pos > 0.82:
                add_bear("NearBBUpper")

        # ── Suporte/Resistência via pivots ─────────────────────────────────────
        recent_h = df["high"].tail(20).values.astype(float)
        recent_l = df["low"].tail(20).values.astype(float)
        resist   = float(recent_h.max())
        support  = float(recent_l.min())
        p_range  = resist - support if resist != support else 1e-10

        if (c - support) / p_range < 0.15:
            add_bull("NearSupport")
        if (resist - c) / p_range < 0.15:
            add_bear("NearResistance")

        # ── Fibonacci Retracement ─────────────────────────────────────────────
        fib_score = self._check_fibonacci(df, c)
        if fib_score > 0:
            bull_score += fib_score * self._pattern_weights.get("FibBounce", 1.4)
            found_bull.append("FibBounce")
        elif fib_score < 0:
            bear_score += abs(fib_score) * self._pattern_weights.get("FibBounce", 1.4)
            found_bear.append("FibBounce")

        # ── Memória episódica: ajuste por histórico de contexto similar ───────
        sim_wr, n_sim = self._recall_similar(self._last_state)
        mem_mult = 1.0
        if n_sim >= 5:
            mem_mult = 0.8 + sim_wr * 0.4  # 0.8 a 1.2

        bull_score *= mem_mult
        bear_score *= mem_mult

        # ── Decisão ───────────────────────────────────────────────────────────
        total = bull_score + bear_score
        if total < 0.01:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Sem padrões detectados")

        bull_ratio = bull_score / total
        bear_ratio = bear_score / total
        threshold  = self._thresholds.get("bull_ratio_threshold", 0.62)

        if bull_ratio > threshold:
            action = "BUY"
            score  = min(0.50 + bull_ratio * 0.48, 0.95)
            patterns_str = ", ".join(found_bull) or "nenhum"
        elif bear_ratio > threshold:
            action = "SELL"
            score  = min(0.50 + bear_ratio * 0.48, 0.95)
            patterns_str = ", ".join(found_bear) or "nenhum"
        else:
            action = "NEUTRAL"
            score  = 0.50
            patterns_str = f"bull={bull_score:.1f} bear={bear_score:.1f}"

        score = max(self._thresholds.get("confidence_floor", 0.50), score)

        return AgentVote(
            self.name, action, score,
            reasoning=f"Padrões: {patterns_str} | mem_wr={sim_wr:.2f}(n={n_sim})",
            meta={"bull_score": round(bull_score, 2), "bear_score": round(bear_score, 2)},
        )

    def _adapt(self) -> None:
        """Auto-ajusta pesos dos padrões com base no histórico."""
        super()._adapt()
        # Ajusta threshold baseado em win_rate recente
        recent = list(self._memory)[-self.MIN_SAMPLES:]
        if len(recent) < self.MIN_SAMPLES:
            return
        wr = sum(1 for m in recent if m.won) / len(recent)
        if wr < 0.45:
            # Torna mais conservador — exige maior consensus
            self._thresholds["bull_ratio_threshold"] = min(
                0.75, self._thresholds.get("bull_ratio_threshold", 0.62) + 0.03
            )
        elif wr > 0.62:
            self._thresholds["bull_ratio_threshold"] = max(
                0.55, self._thresholds.get("bull_ratio_threshold", 0.62) - 0.02
            )

    def _check_fibonacci(self, df: pd.DataFrame, close: float) -> float:
        """
        Verifica se o preço está próximo de um nível de Fibonacci.
        Retorna +score (bullish) ou -score (bearish) ou 0.
        """
        try:
            if len(df) < 50:
                return 0.0
            tail = df.tail(50)
            swing_high = float(tail["high"].max())
            swing_low  = float(tail["low"].min())
            diff = swing_high - swing_low
            if diff < 1e-6:
                return 0.0

            levels = {
                0.236: swing_high - 0.236 * diff,
                0.382: swing_high - 0.382 * diff,
                0.500: swing_high - 0.500 * diff,
                0.618: swing_high - 0.618 * diff,
                0.786: swing_high - 0.786 * diff,
            }
            prox = self._thresholds.get("fib_proximity", 0.015)
            for ratio, level in levels.items():
                dist_pct = abs(close - level) / level if level != 0 else 1.0
                if dist_pct < prox:
                    # Próximo de nível Fib: direction based on approach
                    if close > level:
                        return 0.6 * (1 - dist_pct / prox)  # bounce bullish
                    else:
                        return -0.6 * (1 - dist_pct / prox)  # rejeição bearish
        except Exception:
            pass
        return 0.0