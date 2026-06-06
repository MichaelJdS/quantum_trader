"""
ml/council/agents/nexus.py — NEXUS v3.0: Detector de Regime Evolutivo

Melhorias v3.0:
  - Memória de transições de regime: aprende padrões de mudança
  - Hurst Exponent: determina se o mercado é mean-reverting ou trending
  - Volatility Forecasting: EWMA para prever volatilidade futura
  - Auto-calibração de limiares ADX/BBW por símbolo
  - Acesso web: ciclo econômico global via API pública (cache 6h)
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
from collections import deque
from threading import Thread

import numpy as np
import pandas as pd
from loguru import logger

from ml.council.base_agent import AgentVote, BaseAgent


class NexusAgent(BaseAgent):
    """NEXUS v3.0 — Detector de Regime com Hurst + Memória de Transições."""

    name   = "NEXUS"
    weight = 0.10
    ADAPT_EVERY = 30

    def _default_thresholds(self) -> dict[str, float]:
        return {
            "adx_trend":        22.0,
            "adx_strong":       30.0,
            "bb_wide":          0.040,
            "bb_narrow":        0.010,
            "vol_lookback":     20.0,
            "hurst_trend":      0.60,   # H > 0.6 → trending
            "hurst_revert":     0.40,   # H < 0.4 → mean-reverting
            "confidence_floor": 0.40,
        }

    def __init__(self) -> None:
        super().__init__()
        # Memória de regimes anteriores: deque de str
        self._regime_history: deque[str] = deque(maxlen=100)
        # Transições observadas: (from, to) → count
        self._transitions:  dict[str, int] = {}
        self._current_regime: str = "UNKNOWN"
        self._vol_forecast:   float = 0.0
        self._cycle_risk:     float = 0.5   # risco do ciclo econômico
        self._cycle_fetched:  float = 0.0

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        if len(df) < 25:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Dados insuf.")

        self._last_state = self._state_key(df)
        self._refresh_cycle_background()

        last  = df.iloc[-1]
        close = df["close"].tail(int(self._thresholds["vol_lookback"])).astype(float)
        t     = self._thresholds

        adx   = self._safe_float(last.get("adx", 15.0))
        bbu   = self._safe_float(last.get("bb_upper"))
        bbl   = self._safe_float(last.get("bb_lower"))
        price = self._safe_float(last.get("close"))
        e9    = self._safe_float(last.get("ema_9"))
        e21   = self._safe_float(last.get("ema_21"))
        e50   = self._safe_float(last.get("ema_50"))

        bb_width = (bbu - bbl) / price if price and bbu > bbl else 0.02
        returns  = close.pct_change().dropna()
        vol_real = float(returns.std()) if len(returns) > 1 else 0.01

        # ── Hurst Exponent ────────────────────────────────────────────────────
        hurst = self._compute_hurst(close.values)

        # ── Volatility EWMA Forecast ──────────────────────────────────────────
        self._vol_forecast = self._ewma_vol_forecast(returns)

        # ── Classifica regime ─────────────────────────────────────────────────
        trend_up   = adx > t["adx_trend"] and e9 > e21 > e50
        trend_down = adx > t["adx_trend"] and e9 < e21 < e50
        volatile   = bb_width > t["bb_wide"] or vol_real > 0.008
        ranging    = adx < t["adx_trend"] and bb_width < t["bb_wide"]
        calm       = adx < 15 and bb_width < t["bb_narrow"]

        # Sobrepõe com Hurst
        if hurst > t["hurst_trend"] and not (trend_up or trend_down):
            trend_up   = e9 > e21
            trend_down = e9 < e21
        elif hurst < t["hurst_revert"] and not (ranging or calm):
            ranging = True

        # Detecta regime e registra transição
        new_regime = self._classify_regime(trend_up, trend_down, volatile, ranging, calm)
        self._record_regime_transition(new_regime)

        sig_dir = (
            signal.direction.value
            if hasattr(signal.direction, "value")
            else str(signal.direction)
        )
        is_buy = sig_dir in ("BUY", "buy", "CALL", "call")

        # ── Previsão de transição de regime ───────────────────────────────────
        regime_risk = self._predict_regime_change()

        # Penalidade pelo risco do ciclo macro
        macro_penalty = self._cycle_risk * 0.10

        # ── Gera voto ─────────────────────────────────────────────────────────
        vote = self._generate_vote(
            new_regime, trend_up, trend_down, volatile, ranging, calm,
            adx, bb_width, bbu, bbl, price, is_buy, hurst,
            regime_risk, macro_penalty,
        )

        # Memória: ajuste por similar
        sim_wr, n_sim = self._recall_similar(self._last_state)
        if n_sim >= 5 and vote.action != "NEUTRAL":
            adj = (sim_wr - 0.5) * 0.12
            new_score = max(0.30, min(0.90, vote.score + adj))
            vote = AgentVote(
                vote.agent_name, vote.action, new_score,
                reasoning=vote.reasoning + f" mem={sim_wr:.2f}(n={n_sim})",
                meta=vote.meta,
            )

        return vote

    def _adapt(self) -> None:
        """Auto-calibra ADX threshold baseado no histórico do regime."""
        super()._adapt()
        if not self._regime_history:
            return
        # Se o regime TRENDING foi seguido de muitas perdas, aumenta adx_trend
        recent_mem = list(self._memory)[-self.MIN_SAMPLES:]
        if not recent_mem:
            return
        trend_losses = [m for m in recent_mem
                        if not m.won and "TREND" in self._current_regime]
        if len(trend_losses) > len(recent_mem) * 0.4:
            self._thresholds["adx_trend"] = min(28.0, self._thresholds["adx_trend"] + 1.0)
            logger.info("NEXUS auto-adaptou: adx_trend aumentado.",
                       new=self._thresholds["adx_trend"])

    # ── Regime helpers ────────────────────────────────────────────────────────

    def _classify_regime(self, tu, td, vol, rang, calm) -> str:
        if calm:       return "CALM"
        if td:         return "TREND_DOWN"
        if tu:         return "TREND_UP"
        if vol:        return "VOLATILE"
        if rang:       return "RANGING"
        return "NEUTRAL"

    def _record_regime_transition(self, new_regime: str) -> None:
        if self._current_regime and self._current_regime != new_regime:
            key = f"{self._current_regime}→{new_regime}"
            self._transitions[key] = self._transitions.get(key, 0) + 1
        self._regime_history.append(new_regime)
        self._current_regime = new_regime

    def _predict_regime_change(self) -> float:
        """
        Probabilidade de mudança de regime iminente (0–1).
        Baseada na frequência histórica da transição atual.
        """
        if len(self._regime_history) < 3:
            return 0.0
        # Padrão recente: últimos 3 regimes
        recent = list(self._regime_history)[-3:]
        # Se oscilando entre 2 regimes → alta chance de mudança
        if len(set(recent)) == 2 and recent[-1] != recent[-2]:
            return 0.6
        if len(set(recent)) == 3:
            return 0.7
        return 0.1

    def _generate_vote(
        self, regime, tu, td, vol, rang, calm,
        adx, bb_w, bbu, bbl, price, is_buy, hurst,
        regime_risk, macro_penalty
    ) -> AgentVote:
        t = self._thresholds
        floor = t.get("confidence_floor", 0.40)

        if calm:
            return AgentVote(
                self.name, "NEUTRAL", max(floor, 0.38 - macro_penalty),
                reasoning=f"CALM (ADX={adx:.1f}, BBw={bb_w:.3f}) — sinais fracos",
                meta={"regime": "CALM", "hurst": round(hurst, 3)},
            )

        if vol and not (tu or td):
            pen = min(0.20, regime_risk * 0.15 + macro_penalty)
            return AgentVote(
                self.name, "NEUTRAL", max(floor - 0.05, 0.33),
                reasoning=f"VOLATILE sem tendência (BBw={bb_w:.3f}, H={hurst:.2f}) — risco {pen:.2f}",
                meta={"regime": "VOLATILE", "hurst": round(hurst, 3)},
            )

        if tu:
            score = 0.78 if adx > t["adx_strong"] else 0.66
            score = max(floor, score - regime_risk * 0.10 - macro_penalty)
            return AgentVote(
                self.name, "BUY", score,
                reasoning=f"TREND_UP (ADX={adx:.1f}, H={hurst:.2f})",
                meta={"regime": "TREND_UP", "hurst": round(hurst, 3)},
            )

        if td:
            score = 0.78 if adx > t["adx_strong"] else 0.66
            score = max(floor, score - regime_risk * 0.10 - macro_penalty)
            return AgentVote(
                self.name, "SELL", score,
                reasoning=f"TREND_DOWN (ADX={adx:.1f}, H={hurst:.2f})",
                meta={"regime": "TREND_DOWN", "hurst": round(hurst, 3)},
            )

        if rang and bbu and bbl and bbu > bbl:
            bb_pos = (price - bbl) / (bbu - bbl)
            if bb_pos < 0.25:
                action = "BUY";  score = max(floor, 0.63 - macro_penalty)
            elif bb_pos > 0.75:
                action = "SELL"; score = max(floor, 0.63 - macro_penalty)
            else:
                action = "NEUTRAL"; score = max(floor, 0.50)
            return AgentVote(
                self.name, action, score,
                reasoning=f"RANGING (ADX={adx:.1f}, H={hurst:.2f}, BBpos={bb_pos:.2f})",
                meta={"regime": "RANGING", "hurst": round(hurst, 3)},
            )

        return AgentVote(
            self.name, "NEUTRAL", max(floor, 0.50),
            reasoning=f"Regime INDEFINIDO (ADX={adx:.1f}, H={hurst:.2f})",
            meta={"regime": "UNKNOWN", "hurst": round(hurst, 3)},
        )

    # ── Indicadores quantitativos ─────────────────────────────────────────────

    def _compute_hurst(self, prices: np.ndarray) -> float:
        """
        Hurst Exponent via R/S Analysis (simplificado).
        H > 0.5: trending (persistente)
        H < 0.5: mean-reverting (anti-persistente)
        H ≈ 0.5: random walk
        """
        try:
            n = len(prices)
            if n < 20:
                return 0.5
            lags  = [2, 4, 8, 16]
            rs_vals: list[float] = []
            for lag in lags:
                if lag >= n:
                    continue
                chunks = [prices[i:i+lag] for i in range(0, n - lag, lag)]
                rs_per_chunk = []
                for chunk in chunks:
                    mean  = np.mean(chunk)
                    dev   = np.cumsum(chunk - mean)
                    r     = dev.max() - dev.min()
                    s     = np.std(chunk, ddof=1)
                    if s > 0:
                        rs_per_chunk.append(r / s)
                if rs_per_chunk:
                    rs_vals.append(np.mean(rs_per_chunk))

            if len(rs_vals) < 2:
                return 0.5

            # Regressão log-log: H = slope
            lags_valid = lags[:len(rs_vals)]
            x = np.log(lags_valid)
            y = np.log(rs_vals)
            H = float(np.polyfit(x, y, 1)[0])
            return max(0.1, min(0.9, H))
        except Exception:
            return 0.5

    def _ewma_vol_forecast(self, returns: pd.Series, lambda_: float = 0.94) -> float:
        """EWMA volatility forecast (como em RiskMetrics)."""
        try:
            if len(returns) < 5:
                return 0.01
            var = float(returns.iloc[0] ** 2)
            for r in returns.iloc[1:]:
                var = lambda_ * var + (1 - lambda_) * float(r) ** 2
            return math.sqrt(var)
        except Exception:
            return 0.01

    def _refresh_cycle_background(self) -> None:
        """Busca indicador de ciclo econômico via web. TTL: 6h."""
        if time.time() - self._cycle_fetched < 21600:
            return
        cached = self._get_web_cache("cycle_risk")
        if cached is not None:
            self._cycle_risk = cached
            return

        def _fetch():
            try:
                # Fear & Greed como proxy de ciclo
                url = "https://api.alternative.me/fng/?limit=1&format=json"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    fg   = float(data["data"][0]["value"])
                    risk = max(0.0, min(1.0, (50 - fg) / 50)) if fg < 50 else 0.0
                    self._cycle_risk    = risk
                    self._cycle_fetched = time.time()
                    self._set_web_cache("cycle_risk", risk)
                    logger.debug("NEXUS: cycle_risk atualizado.", risk=round(risk, 3))
            except Exception:
                pass

        Thread(target=_fetch, daemon=True).start()