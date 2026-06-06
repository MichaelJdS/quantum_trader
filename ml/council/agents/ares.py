"""
ml/council/agents/ares.py — ARES v3.0: Order Flow Evolutivo

Melhorias v3.0:
  - Delta acumulado: rastreia buy_vol vs sell_vol tick a tick
  - Imbalance Z-score: detecta anomalias estatísticas no fluxo
  - Market Impact: estima impacto de ordens grandes
  - Memória de padrões de fluxo: aprende quais imbalances predizem direção
  - Footprint simplificado: detecta absorção (vela grande com delta oposto)
  - Auto-adaptação: ajusta limiares conforme tipo de mercado
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from ml.council.base_agent import AgentVote, BaseAgent


class AresAgent(BaseAgent):
    """ARES v3.0 — Order Flow com Absorção e Imbalance Adaptativo."""

    name   = "ARES"
    weight = 0.10
    ADAPT_EVERY = 25

    def __init__(self) -> None:
        super().__init__()
        # Histórico de deltas para Z-score dinâmico
        self._delta_history:   deque[float] = deque(maxlen=200)
        self._imbalance_hist:  deque[float] = deque(maxlen=200)

    def _default_thresholds(self) -> dict[str, float]:
        return {
            "min_ticks":          5.0,
            "imbalance_ratio":    0.62,
            "strong_ratio":       0.75,
            "zscore_threshold":   1.5,    # desvios para considerar anomalia
            "absorption_ratio":   1.8,    # range/body para detectar absorção
            "confidence_floor":   0.45,
            "delta_weight":       0.40,
            "candle_weight":      0.25,
            "absorption_weight":  0.20,
            "footprint_weight":   0.15,
        }

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        self._last_state = self._state_key(df)
        sig_dir = (
            signal.direction.value
            if hasattr(signal.direction, "value")
            else str(signal.direction)
        )
        is_buy = sig_dir in ("BUY", "buy", "CALL", "call")
        floor  = self._thresholds.get("confidence_floor", 0.45)

        if not ticks or len(ticks) < int(self._thresholds["min_ticks"]):
            # Fallback: análise de volume via df
            return self._fallback_volume_analysis(df, is_buy, floor)

        # ── 1. Delta acumulado ────────────────────────────────────────────────
        buy_vol  = sum(float(t.get("buy_vol",  t.get("volume", 0.5))) for t in ticks)
        sell_vol = sum(float(t.get("sell_vol", 0.0)) for t in ticks)

        if buy_vol + sell_vol < 0.01:
            buy_vol = sell_vol = len(ticks) / 2

        total_vol = buy_vol + sell_vol
        delta     = buy_vol - sell_vol
        imbalance = buy_vol / total_vol  # 0.5 = neutro, >0.5 = bull pressure
        self._delta_history.append(delta)
        self._imbalance_hist.append(imbalance)

        # Z-score do delta atual
        delta_z = self._compute_zscore(delta, list(self._delta_history)[:-1])

        # ── 2. Vela de pressão ────────────────────────────────────────────────
        try:
            last  = df.iloc[-1]
            o = float(last["open"]); h = float(last["high"])
            l = float(last["low"]);  c = float(last["close"])
            body       = abs(c - o)
            full_range = h - l if h != l else 1e-10
            uw         = h - max(o, c)
            lw         = min(o, c) - l
        except Exception:
            o = c = h = l = body = full_range = uw = lw = 0.0

        # ── 3. Absorção (footprint) ───────────────────────────────────────────
        absorption_bull = False  # vela bearish mas delta fortemente bull
        absorption_bear = False  # vela bullish mas delta fortemente bear

        if body > 0 and full_range > 0:
            ab_ratio = self._thresholds["absorption_ratio"]
            if c < o and imbalance > self._thresholds["strong_ratio"]:
                # Vela bearish mas compradores absorveram → possível reversão bullish
                absorption_bull = True
            if c > o and imbalance < (1 - self._thresholds["strong_ratio"]):
                absorption_bear = True

        # ── 4. Score composto ─────────────────────────────────────────────────
        w = self._thresholds
        bull_score = 0.0; bear_score = 0.0

        # Delta
        if imbalance > w["imbalance_ratio"]:
            bull_score += w["delta_weight"] * min((imbalance - 0.5) * 2, 1.0)
        elif imbalance < (1 - w["imbalance_ratio"]):
            bear_score += w["delta_weight"] * min((0.5 - imbalance) * 2, 1.0)

        # Z-score anômalo reforça
        if abs(delta_z) > w["zscore_threshold"]:
            boost = w["delta_weight"] * 0.3 * min(abs(delta_z) / 3, 1.0)
            if delta_z > 0: bull_score += boost
            else:           bear_score += boost

        # Vela candle pressure
        if body > 0 and full_range > 0:
            body_ratio = body / full_range
            if c > o and lw < body * 0.3:
                bull_score += w["candle_weight"] * body_ratio
            elif c < o and uw < body * 0.3:
                bear_score += w["candle_weight"] * body_ratio

        # Absorção
        if absorption_bull:
            bull_score += w["absorption_weight"] * 1.2
        if absorption_bear:
            bear_score += w["absorption_weight"] * 1.2

        # Footprint: volume nas sombras
        if full_range > 0:
            bull_footprint = (lw / full_range) * (1 - imbalance)  # sombra baixa + sell → compra acima
            bear_footprint = (uw / full_range) * imbalance
            bull_score += w["footprint_weight"] * bull_footprint
            bear_score += w["footprint_weight"] * bear_footprint

        # ── 5. Memória episódica ──────────────────────────────────────────────
        sim_wr, n_sim = self._recall_similar(self._last_state)
        if n_sim >= 5:
            mem_mult = 0.85 + sim_wr * 0.30
            bull_score *= mem_mult; bear_score *= mem_mult

        # ── 6. Decisão ────────────────────────────────────────────────────────
        total = bull_score + bear_score
        if total < 0.01:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Fluxo neutro")

        bull_pct = bull_score / total
        bear_pct = bear_score / total
        threshold = w["imbalance_ratio"]

        if bull_pct > threshold:
            action = "BUY"
            score  = max(floor, min(0.50 + bull_pct * 0.48, 0.94))
        elif bear_pct > threshold:
            action = "SELL"
            score  = max(floor, min(0.50 + bear_pct * 0.48, 0.94))
        else:
            action = "NEUTRAL"; score = 0.5

        abs_note = " [ABSORB]" if (absorption_bull or absorption_bear) else ""
        return AgentVote(
            self.name, action, score,
            reasoning=(
                f"Delta={delta:.1f} imb={imbalance:.2f} z={delta_z:.2f}"
                f" bull={bull_score:.2f} bear={bear_score:.2f}{abs_note}"
            ),
            meta={
                "delta_z":    round(delta_z, 2),
                "imbalance":  round(imbalance, 3),
                "absorption": absorption_bull or absorption_bear,
            },
        )

    def _adapt(self) -> None:
        """Ajusta limiar de imbalance baseado em acurácia recente."""
        super()._adapt()
        recent = list(self._memory)[-self.MIN_SAMPLES:]
        if len(recent) < self.MIN_SAMPLES:
            return
        wr = sum(1 for m in recent if m.won) / len(recent)
        if wr < 0.44:
            self._thresholds["imbalance_ratio"] = min(
                0.80, self._thresholds["imbalance_ratio"] + 0.02
            )
        elif wr > 0.60:
            self._thresholds["imbalance_ratio"] = max(
                0.55, self._thresholds["imbalance_ratio"] - 0.01
            )

    def _compute_zscore(self, value: float, history: list[float]) -> float:
        if len(history) < 5:
            return 0.0
        arr  = np.array(history)
        mean = arr.mean(); std = arr.std()
        if std < 1e-9:
            return 0.0
        return float((value - mean) / std)

    def _fallback_volume_analysis(self, df, is_buy, floor) -> AgentVote:
        """Análise de volume via OHLCV quando ticks não disponíveis."""
        if "volume" not in df.columns or len(df) < 10:
            return AgentVote(self.name, "NEUTRAL", 0.5,
                             reasoning="Sem ticks e sem volume")
        tail    = df.tail(10)
        vol_now = float(df["volume"].iloc[-1])
        vol_avg = float(tail["volume"].mean())
        close   = float(df["close"].iloc[-1])
        prev    = float(df["close"].iloc[-2])
        bullish = close > prev

        if vol_avg > 0:
            vol_ratio = vol_now / vol_avg
            if vol_ratio > 1.5:
                if bullish:
                    return AgentVote(self.name, "BUY",  max(floor, 0.65),
                                     reasoning=f"Vol spike bullish ({vol_ratio:.1f}x)")
                else:
                    return AgentVote(self.name, "SELL", max(floor, 0.65),
                                     reasoning=f"Vol spike bearish ({vol_ratio:.1f}x)")

        action = "BUY" if is_buy else "SELL"
        return AgentVote(self.name, action, max(floor, 0.52),
                         reasoning=f"Vol normal (ratio={vol_now/vol_avg:.2f})" if vol_avg > 0 else "Sem ticks")