"""
ml/council/agents/nexus.py — NEXUS: Detecção de Regime de Mercado

Classifica o mercado em: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CALM.
Usa: ADX, BB width, desvio padrão dos retornos, ATR relativo.
Sugere qual tipo de estratégia é mais adequada para o regime atual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.council.base_agent import AgentVote, BaseAgent


class NexusAgent(BaseAgent):
    """Especialista em Detecção de Regime de Mercado — NEXUS."""

    name = "NEXUS"
    weight = 0.10

    # Limiares de regime
    ADX_TREND_THRESHOLD   = 22   # ADX > 22 → tendência presente
    ADX_STRONG_THRESHOLD  = 30   # ADX > 30 → tendência forte
    BB_WIDE_THRESHOLD     = 0.04 # BB width > 4% do preço → volátil
    BB_NARROW_THRESHOLD   = 0.01 # BB width < 1% → comprimido (ranging)
    VOLATILITY_LOOKBACK   = 20   # períodos para calcular volatilidade

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        if len(df) < 25:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Dados insuficientes para regime")

        last = df.iloc[-1]
        close = df["close"].tail(self.VOLATILITY_LOOKBACK).astype(float)

        # ── Métricas de regime ────────────────────────────────────────────────
        adx        = self._safe_float(last.get("adx", 15.0))
        bb_upper   = self._safe_float(last.get("bb_upper"))
        bb_lower   = self._safe_float(last.get("bb_lower"))
        price      = self._safe_float(last.get("close"))
        ema9       = self._safe_float(last.get("ema_9"))
        ema21      = self._safe_float(last.get("ema_21"))
        ema50      = self._safe_float(last.get("ema_50"))
        atr        = self._safe_float(last.get("atr_14"))

        # BB width relativo
        bb_width = (bb_upper - bb_lower) / price if price else 0.02

        # Retornos e volatilidade realizada
        returns  = close.pct_change().dropna()
        vol_real = float(returns.std()) if len(returns) > 1 else 0.01

        # ── Classificação de regime ───────────────────────────────────────────
        trending_up   = (adx > self.ADX_TREND_THRESHOLD and ema9 > ema21 > ema50)
        trending_down = (adx > self.ADX_TREND_THRESHOLD and ema9 < ema21 < ema50)
        ranging       = (adx < self.ADX_TREND_THRESHOLD and bb_width < self.BB_WIDE_THRESHOLD)
        volatile      = (bb_width > self.BB_WIDE_THRESHOLD or vol_real > 0.008)
        calm          = (adx < 15 and bb_width < self.BB_NARROW_THRESHOLD)

        sig_dir = signal.direction.value if hasattr(signal.direction, "value") else str(signal.direction)
        is_buy  = sig_dir in ("BUY", "buy", "CALL", "call")

        # ── Decisão de voto baseada no regime ─────────────────────────────────
        if calm:
            # Mercado calmo — sinais fracos, reduz confiança
            return AgentVote(
                self.name, "NEUTRAL", 0.4,
                reasoning=f"Regime CALM (ADX={adx:.1f}, BBw={bb_width:.3f}) — sinais fracos"
            )

        if volatile and not (trending_up or trending_down):
            # Volátil sem tendência — perigoso
            return AgentVote(
                self.name, "NEUTRAL", 0.35,
                reasoning=f"Regime VOLATILE sem tendência (BBw={bb_width:.3f}) — risco elevado"
            )

        if trending_up:
            action = "BUY"
            score  = 0.75 if adx > self.ADX_STRONG_THRESHOLD else 0.65
            return AgentVote(
                self.name, action, score,
                reasoning=f"Regime TRENDING_UP (ADX={adx:.1f}) — favorece CALL"
            )

        if trending_down:
            action = "SELL"
            score  = 0.75 if adx > self.ADX_STRONG_THRESHOLD else 0.65
            return AgentVote(
                self.name, action, score,
                reasoning=f"Regime TRENDING_DOWN (ADX={adx:.1f}) — favorece PUT"
            )

        if ranging:
            # Ranging — estratégia de reversão à média
            bb_pos = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) else 0.5
            if bb_pos < 0.25:
                action = "BUY";  score = 0.62
            elif bb_pos > 0.75:
                action = "SELL"; score = 0.62
            else:
                action = "NEUTRAL"; score = 0.5
            return AgentVote(
                self.name, action, score,
                reasoning=f"Regime RANGING (ADX={adx:.1f}, BBpos={bb_pos:.2f}) — reversão"
            )

        return AgentVote(self.name, "NEUTRAL", 0.5, reasoning=f"Regime INDEFINIDO (ADX={adx:.1f})")
