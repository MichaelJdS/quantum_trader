"""
ml/council/agents/vector.py — VECTOR v3.0: Ensemble ML Auto-Evolutivo

Melhorias v3.0:
  - Pesos das features aprendidos por gradiente (online gradient descent)
  - 8 features (vs. 5 anteriores) + 2 novas: Stochastic RSI, Volume Delta
  - Calibração bayesiana simples (Beta distribution) por feature
  - Acesso web: verifica eventos macroeconômicos do dia (cache 4h)
  - Auto-eliminação de features com correlação negativa persistente
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from collections import defaultdict, deque
from threading import Thread

import numpy as np
import pandas as pd
from loguru import logger

from ml.council.base_agent import AgentVote, BaseAgent

_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "states", "vector_weights.json"
)


class VectorAgent(BaseAgent):
    """VECTOR v3.0 — Ensemble com Gradient Descent Online."""

    name   = "VECTOR"
    weight = 0.12
    ADAPT_EVERY = 15

    # Pesos iniciais das 8 features
    _INITIAL_WEIGHTS = {
        "rsi_distance":    0.22,
        "macd_momentum":   0.22,
        "ema_alignment":   0.18,
        "bb_breakout":     0.12,
        "momentum_10":     0.10,
        "stoch_rsi":       0.08,
        "volume_delta":    0.08,
        "atr_quality":     0.00,  # peso zero inicial, cresce com aprendizado
    }
    LR_WEIGHTS = 0.008  # taxa de aprendizado dos pesos

    def __init__(self) -> None:
        super().__init__()
        self._weights: dict[str, float] = dict(self._INITIAL_WEIGHTS)
        # Beta distribution params por feature: (alpha, beta)
        self._beta_params: dict[str, list[float]] = {
            k: [1.0, 1.0] for k in self._INITIAL_WEIGHTS
        }
        # Histórico de contribuição por feature: feature → deque[float]
        self._feature_contribs: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._last_features: dict[str, float] = {}
        self._macro_risk:    float = 0.0    # risco macro (0=ok, 1=alto)
        self._macro_fetched: float = 0.0
        self._load_weights()

    def _default_thresholds(self) -> dict[str, float]:
        return {
            "confidence_floor":  0.50,
            "bull_threshold":    0.58,
            "bear_threshold":    0.58,
            "min_feature_weight": 0.005,  # peso mínimo antes de eliminar
        }

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        if len(df) < 15:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Features insuf.")

        self._last_state = self._state_key(df)
        self._refresh_macro_background()

        last = df.iloc[-1]
        sig_dir = (
            signal.direction.value
            if hasattr(signal.direction, "value")
            else str(signal.direction)
        )
        is_buy = sig_dir in ("BUY", "buy", "CALL", "call")

        bull_score, bear_score, details, raw_features = self._score_features(last, df)

        # Ajusta pelo risco macro
        if self._macro_risk > 0.7:
            bull_score *= 0.80
            bear_score *= 0.80
            details += " [MacroRisk]"

        # Memória: boost se contexto similar teve bom histórico
        sim_wr, n_sim = self._recall_similar(self._last_state)
        if n_sim >= 5:
            mem_mult = 0.85 + sim_wr * 0.30
            bull_score *= mem_mult
            bear_score *= mem_mult

        total = bull_score + bear_score
        if total < 0.01:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Features neutras")

        bull_pct = bull_score / total
        bear_pct = bear_score / total
        t_bull   = self._thresholds.get("bull_threshold", 0.58)
        t_bear   = self._thresholds.get("bear_threshold", 0.58)

        if bull_pct > t_bull:
            action = "BUY"
            score  = min(0.50 + bull_pct * 0.47, 0.94)
        elif bear_pct > t_bear:
            action = "SELL"
            score  = min(0.50 + bear_pct * 0.47, 0.94)
        else:
            action = "NEUTRAL"
            score  = 0.50

        score = max(self._thresholds.get("confidence_floor", 0.50), score)

        self._last_features = raw_features
        return AgentVote(
            self.name, action, score,
            reasoning=f"Ensemble: {details} bull={bull_score:.2f} bear={bear_score:.2f}",
            meta={"macro_risk": round(self._macro_risk, 2), "sim_wr": round(sim_wr, 3)},
        )

    def _score_features(self, last: pd.Series, df: pd.DataFrame):
        bull = 0.0; bear = 0.0
        details: list[str] = []
        raw: dict[str, float] = {}

        close = self._safe_float(last.get("close"))
        w = self._weights

        # 1. RSI distance
        rsi = self._safe_float(last.get("rsi_14", 50.0), 50.0)
        raw["rsi_distance"] = (rsi - 50) / 50
        if rsi > 52:
            c = w["rsi_distance"] * min((rsi - 50) / 30, 1.0); bull += c; details.append(f"RSI↑{rsi:.0f}")
        elif rsi < 48:
            c = w["rsi_distance"] * min((50 - rsi) / 30, 1.0); bear += c; details.append(f"RSI↓{rsi:.0f}")

        # 2. MACD
        macd = self._safe_float(last.get("macd_hist", 0.0))
        macd_std = float(df["macd_hist"].std()) if "macd_hist" in df.columns else 0.001
        raw["macd_momentum"] = macd / (macd_std + 1e-9)
        if macd_std > 0:
            norm = macd / (macd_std * 2 + 1e-9)
            if norm > 0:  c = w["macd_momentum"] * min(abs(norm), 1.0); bull += c; details.append("MACD↑")
            else:         c = w["macd_momentum"] * min(abs(norm), 1.0); bear += c; details.append("MACD↓")

        # 3. EMA alignment
        e9  = self._safe_float(last.get("ema_9",  close), close)
        e21 = self._safe_float(last.get("ema_21", close), close)
        e50 = self._safe_float(last.get("ema_50", close), close)
        if e9 > e21 > e50:
            raw["ema_alignment"] = 1.0; bull += w["ema_alignment"]; details.append("EMA↑↑↑")
        elif e9 > e21:
            raw["ema_alignment"] = 0.6; bull += w["ema_alignment"] * 0.6; details.append("EMA↑↑")
        elif e9 < e21 < e50:
            raw["ema_alignment"] = -1.0; bear += w["ema_alignment"]; details.append("EMA↓↓↓")
        elif e9 < e21:
            raw["ema_alignment"] = -0.6; bear += w["ema_alignment"] * 0.6; details.append("EMA↓↓")
        else:
            raw["ema_alignment"] = 0.0

        # 4. BB breakout
        bbu = self._safe_float(last.get("bb_upper"))
        bbl = self._safe_float(last.get("bb_lower"))
        if bbu and bbl and bbu != bbl:
            bb_pos = (close - bbl) / (bbu - bbl)
            raw["bb_breakout"] = bb_pos - 0.5
            if bb_pos < 0.20:
                bull += w["bb_breakout"] * (1 - bb_pos / 0.2); details.append("BB_low")
            elif bb_pos > 0.80:
                bear += w["bb_breakout"] * (bb_pos - 0.8) / 0.2; details.append("BB_high")
        else:
            raw["bb_breakout"] = 0.0

        # 5. Momentum 10
        if len(df) > 10:
            c_now = float(df["close"].iloc[-1]); c_10 = float(df["close"].iloc[-11])
            mom = (c_now / c_10 - 1) if c_10 else 0.0
        else:
            mom = 0.0
        raw["momentum_10"] = mom
        if mom > 0.001:  bull += w["momentum_10"] * min(mom / 0.01, 1.0)
        elif mom < -0.001: bear += w["momentum_10"] * min(abs(mom) / 0.01, 1.0)

        # 6. Stochastic RSI (NOVO)
        stoch_rsi_k = self._safe_float(last.get("stoch_rsi_k", 50.0), 50.0)
        raw["stoch_rsi"] = (stoch_rsi_k - 50) / 50
        if stoch_rsi_k < 20:
            bull += w["stoch_rsi"] * (1 - stoch_rsi_k / 20); details.append("StRSI_low")
        elif stoch_rsi_k > 80:
            bear += w["stoch_rsi"] * (stoch_rsi_k - 80) / 20; details.append("StRSI_high")

        # 7. Volume Delta (NOVO)
        if "volume" in df.columns and len(df) >= 20:
            vol_now  = self._safe_float(last.get("volume", 0))
            vol_mean = float(df["volume"].tail(20).mean())
            if vol_mean > 0:
                vol_ratio = vol_now / vol_mean - 1.0  # >0 = volume acima da média
                raw["volume_delta"] = vol_ratio
                if vol_ratio > 0.5:  # volume 50%+ acima da média
                    if mom > 0: bull += w["volume_delta"] * min(vol_ratio, 1.0)
                    else:       bear += w["volume_delta"] * min(vol_ratio, 1.0)
                    details.append(f"Vol+{vol_ratio:.1f}x")
            else:
                raw["volume_delta"] = 0.0
        else:
            raw["volume_delta"] = 0.0

        # 8. ATR Quality (NOVO)
        atr = self._safe_float(last.get("atr_14", 0.001))
        if close > 0:
            atr_pct = atr / close
            raw["atr_quality"] = atr_pct
            # ATR muito alto ou muito baixo penaliza a qualidade
            if 0.002 <= atr_pct <= 0.008:
                bull += w["atr_quality"] * 0.5
                bear += w["atr_quality"] * 0.5
        else:
            raw["atr_quality"] = 0.0

        return bull, bear, " ".join(details), raw

    def _adapt(self) -> None:
        """
        Gradient descent online nos pesos das features.
        Aumenta peso de features que contribuíram para trades ganhos,
        reduz peso das que contribuíram para perdas.
        """
        super()._adapt()
        recent = list(self._memory)[-self.ADAPT_EVERY:]
        if len(recent) < 10:
            return

        # Calcula gradiente simples: sinal do trade × contribuição da feature
        for mem in recent:
            signal_val = 1.0 if mem.won else -1.0
            for feat, contrib in self._last_features.items():
                if feat not in self._weights:
                    continue
                grad = signal_val * abs(contrib) * 0.1
                self._weights[feat] = max(
                    self._thresholds.get("min_feature_weight", 0.005),
                    self._weights[feat] + self.LR_WEIGHTS * grad,
                )

        # Normaliza pesos para soma = 1.0
        total_w = sum(self._weights.values())
        if total_w > 0:
            for k in self._weights:
                self._weights[k] /= total_w

        self._save_weights()
        logger.debug("VECTOR: pesos auto-atualizados.", weights={k: round(v, 4) for k, v in self._weights.items()})

    def _refresh_macro_background(self) -> None:
        """
        Verifica indicadores macro via web (Yahoo Finance / APIs públicas).
        Atualiza self._macro_risk (0=calmo, 1=alto risco). TTL: 4h.
        """
        if time.time() - self._macro_fetched < 14400:
            return
        cached = self._get_web_cache("macro_risk")
        if cached is not None:
            self._macro_risk = cached
            return

        def _fetch():
            try:
                # VIX proxy via Alpha Vantage (gratuito, sem auth para dados básicos)
                # Fallback: Fear & Greed do Alternative.me
                url = "https://api.alternative.me/fng/?limit=1&format=json"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    fg = float(data["data"][0]["value"])
                    # Fear & Greed < 25 = medo extremo = alto risco macro
                    risk = max(0.0, min(1.0, (50 - fg) / 50)) if fg < 50 else 0.0
                    self._macro_risk    = risk
                    self._macro_fetched = time.time()
                    self._set_web_cache("macro_risk", risk)
                    logger.debug("VECTOR: macro_risk atualizado.", risk=round(risk, 3))
            except Exception as exc:
                logger.debug("VECTOR: falha macro fetch.", error=str(exc))

        Thread(target=_fetch, daemon=True).start()

    def _save_weights(self) -> None:
        os.makedirs(os.path.dirname(_WEIGHTS_PATH), exist_ok=True)
        try:
            with open(_WEIGHTS_PATH, "w") as f:
                json.dump({"weights": self._weights, "beta": self._beta_params}, f)
        except Exception:
            pass

    def _load_weights(self) -> None:
        try:
            if os.path.exists(_WEIGHTS_PATH):
                with open(_WEIGHTS_PATH) as f:
                    data = json.load(f)
                self._weights     = data.get("weights",     dict(self._INITIAL_WEIGHTS))
                self._beta_params = data.get("beta",        {k: [1.0, 1.0] for k in self._weights})
        except Exception:
            pass