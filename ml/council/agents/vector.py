"""
ml/council/agents/vector.py — VECTOR: Machine Learning Preditivo (Ensemble Heurístico)

Simula um ensemble XGBoost/LightGBM via combinação linear ponderada de features
técnicas. Pesos ajustados com base em correlações conhecidas para índices sintéticos Deriv.
Plug-and-play: substitua _score_features() por um modelo treinado na Fase 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.council.base_agent import AgentVote, BaseAgent


class VectorAgent(BaseAgent):
    """Especialista em ML Ensemble Preditivo — VECTOR."""

    name = "VECTOR"
    weight = 0.12

    # Pesos heurísticos (baseados em correlações conhecidas para índices sintéticos)
    # Serão substituídos por XGBoost real na Fase 2
    _FEATURE_WEIGHTS = {
        "rsi_distance":   0.25,  # RSI distante de 50 = sinal mais forte
        "macd_momentum":  0.25,  # MACD hist positivo/negativo
        "ema_alignment":  0.20,  # Alinhamento EMA9/21/50
        "bb_breakout":    0.15,  # Preço saindo das Bollinger Bands
        "momentum_10":    0.15,  # Momentum de 10 períodos
    }

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        if len(df) < 15:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Features insuficientes")

        last = df.iloc[-1]
        sig_dir = signal.direction.value if hasattr(signal.direction, "value") else str(signal.direction)
        is_buy  = sig_dir in ("BUY", "buy", "CALL", "call")

        bull_score, bear_score, details = self._score_features(last, df)

        # Normaliza os scores
        total = bull_score + bear_score
        if total < 0.01:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Features neutras")

        bull_pct = bull_score / total
        bear_pct = bear_score / total

        if bull_pct > 0.60:
            action = "BUY"
            score  = min(0.5 + bull_pct * 0.5, 0.92)
        elif bear_pct > 0.60:
            action = "SELL"
            score  = min(0.5 + bear_pct * 0.5, 0.92)
        else:
            action = "NEUTRAL"
            score  = 0.5

        reasoning = f"Ensemble: bull={bull_score:.2f} bear={bear_score:.2f} [{details}]"
        return AgentVote(self.name, action, score, reasoning=reasoning)

    # ── Feature scoring (substituível por modelo treinado) ────────────────────

    def _score_features(self, last: pd.Series, df: pd.DataFrame):
        """
        Calcula scores bullish e bearish com base nas features.
        Retorna (bull_score, bear_score, details_str).
        """
        bull = 0.0
        bear = 0.0
        details = []

        close = self._safe_float(last.get("close"))

        # 1. RSI distance from 50
        rsi = self._safe_float(last.get("rsi_14", 50.0), 50.0)
        rsi_w = self._FEATURE_WEIGHTS["rsi_distance"]
        if rsi > 52:
            bull += rsi_w * min((rsi - 50) / 30, 1.0)
            details.append(f"RSI↑{rsi:.0f}")
        elif rsi < 48:
            bear += rsi_w * min((50 - rsi) / 30, 1.0)
            details.append(f"RSI↓{rsi:.0f}")

        # 2. MACD histogram momentum
        macd = self._safe_float(last.get("macd_hist", 0.0))
        macd_w = self._FEATURE_WEIGHTS["macd_momentum"]
        macd_std = float(df["macd_hist"].std()) if "macd_hist" in df.columns else 0.001
        if macd_std > 0:
            norm_macd = macd / (macd_std * 2 + 1e-9)
            if norm_macd > 0:
                bull += macd_w * min(abs(norm_macd), 1.0)
                details.append(f"MACD↑")
            else:
                bear += macd_w * min(abs(norm_macd), 1.0)
                details.append(f"MACD↓")

        # 3. EMA alignment
        e9  = self._safe_float(last.get("ema_9",  close), close)
        e21 = self._safe_float(last.get("ema_21", close), close)
        e50 = self._safe_float(last.get("ema_50", close), close)
        ema_w = self._FEATURE_WEIGHTS["ema_alignment"]
        if e9 > e21 > e50:
            bull += ema_w
            details.append("EMA↑↑↑")
        elif e9 > e21:
            bull += ema_w * 0.6
            details.append("EMA↑↑")
        elif e9 < e21 < e50:
            bear += ema_w
            details.append("EMA↓↓↓")
        elif e9 < e21:
            bear += ema_w * 0.6
            details.append("EMA↓↓")

        # 4. Bollinger Band breakout
        bb_upper = self._safe_float(last.get("bb_upper"))
        bb_lower = self._safe_float(last.get("bb_lower"))
        bb_w = self._FEATURE_WEIGHTS["bb_breakout"]
        if bb_upper and bb_lower and bb_upper != bb_lower:
            bb_pos = (close - bb_lower) / (bb_upper - bb_lower)
            if bb_pos < 0.2:
                bull += bb_w * (1 - bb_pos / 0.2)
                details.append("BB_low")
            elif bb_pos > 0.8:
                bear += bb_w * (bb_pos - 0.8) / 0.2
                details.append("BB_high")

        # 5. Momentum (10 períodos)
        mom_w = self._FEATURE_WEIGHTS["momentum_10"]
        if "momentum_10" in last.index:
            mom = self._safe_float(last.get("momentum_10", 0.0))
        else:
            # Calcula on-the-fly
            if len(df) > 10:
                c_now  = float(df["close"].iloc[-1])
                c_10   = float(df["close"].iloc[-11])
                mom    = (c_now / c_10 - 1) if c_10 else 0.0
            else:
                mom = 0.0

        if mom > 0.001:
            bull += mom_w * min(mom / 0.01, 1.0)
        elif mom < -0.001:
            bear += mom_w * min(abs(mom) / 0.01, 1.0)

        return bull, bear, " ".join(details)
