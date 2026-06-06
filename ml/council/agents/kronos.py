"""
ml/council/agents/kronos.py — KRONOS v3.0: Multi-Timeframe Evolutivo

Melhorias v3.0:
  - 5 timeframes: 1m, 3m, 5m, 10m, 15m (vs. 3 anteriores)
  - Pesos dos TFs auto-ajustados por acurácia individual
  - Volume Profile: acumulação de volume por faixa de preço (VPVR simplificado)
  - Memória: aprende quais timeframes dominam em cada regime
  - Auto-adaptação: se 3m ou 5m gera mais falsos, reduz seu peso
"""
from __future__ import annotations

import json
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from loguru import logger

from ml.council.base_agent import AgentVote, BaseAgent

_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "states", "kronos_tf_weights.json"
)


class KronosAgent(BaseAgent):
    """KRONOS v3.0 — Multi-Timeframe com 5 TFs e Volume Profile."""

    name   = "KRONOS"
    weight = 0.12
    ADAPT_EVERY = 20

    # TFs: (tamanho em candles 1m, label)
    TIMEFRAMES = [(1, "1m"), (3, "3m"), (5, "5m"), (10, "10m"), (15, "15m")]

    # Pesos iniciais dos TFs (auto-ajustados)
    _INITIAL_TF_WEIGHTS = {"1m": 0.15, "3m": 0.20, "5m": 0.25, "10m": 0.20, "15m": 0.20}

    def __init__(self) -> None:
        super().__init__()
        self._tf_weights: dict[str, float]     = dict(self._INITIAL_TF_WEIGHTS)
        # Histórico de acurácia por TF: tf → deque de bool
        self._tf_accuracy: dict[str, deque]    = {
            tf: deque(maxlen=100) for _, tf in self.TIMEFRAMES
        }
        self._last_tf_votes: dict[str, str]    = {}
        self._load_tf_weights()

    def _default_thresholds(self) -> dict[str, float]:
        return {
            "approval_threshold":  0.55,   # score mínimo ponderado para aprovação
            "confidence_floor":    0.50,
            "vpvr_proximity":      0.01,    # 1% de proximidade ao POC do VPVR
        }

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        if len(df) < 30:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Dados insuf. multi-TF")

        self._last_state = self._state_key(df)

        # ── Gera votos para cada TF ───────────────────────────────────────────
        tf_votes: dict[str, str] = {}
        tf_scores: dict[str, float] = {}

        for n_bars, tf_label in self.TIMEFRAMES:
            resampled = self._resample(df, n_bars) if n_bars > 1 else df
            vote, score = self._analyze_timeframe(resampled, tf_label)
            tf_votes[tf_label]  = vote
            tf_scores[tf_label] = score

        self._last_tf_votes = tf_votes

        # ── Volume Profile (VPVR) ─────────────────────────────────────────────
        vpvr_signal, vpvr_score = self._compute_vpvr(df, signal)

        # ── Score ponderado ────────────────────────────────────────────────────
        sig_dir = (
            signal.direction.value
            if hasattr(signal.direction, "value")
            else str(signal.direction)
        )
        is_buy       = sig_dir in ("BUY", "buy", "CALL", "call")
        target_action = "BUY" if is_buy else "SELL"

        weighted_bull = 0.0
        weighted_bear = 0.0
        total_w = 0.0

        for _, tf in self.TIMEFRAMES:
            w  = self._tf_weights[tf]
            vote = tf_votes[tf]
            sc   = tf_scores[tf]
            if vote == "BUY":
                weighted_bull += w * sc
            elif vote == "SELL":
                weighted_bear += w * sc
            else:
                # NEUTRAL contribui com neutro
                weighted_bull += w * 0.5
                weighted_bear += w * 0.5
            total_w += w

        # Adiciona contribuição do VPVR
        if vpvr_signal == "BUY":
            weighted_bull += 0.10 * vpvr_score
        elif vpvr_signal == "SELL":
            weighted_bear += 0.10 * vpvr_score
        total_w += 0.10

        bull_ratio = weighted_bull / total_w if total_w > 0 else 0.5
        bear_ratio = weighted_bear / total_w if total_w > 0 else 0.5

        th = self._thresholds.get("approval_threshold", 0.55)

        # Memória: ajuste por histórico similar
        sim_wr, n_sim = self._recall_similar(self._last_state)
        mem_adj = (sim_wr - 0.5) * 0.10 if n_sim >= 5 else 0.0

        if bull_ratio > th:
            action = "BUY"
            score  = max(self._thresholds["confidence_floor"],
                         min(0.50 + bull_ratio * 0.48, 0.94) + mem_adj)
        elif bear_ratio > th:
            action = "SELL"
            score  = max(self._thresholds["confidence_floor"],
                         min(0.50 + bear_ratio * 0.48, 0.94) + mem_adj)
        else:
            action = "NEUTRAL"
            score  = max(self._thresholds["confidence_floor"], 0.45)

        tf_str = " ".join(f"{tf}:{v}" for tf, v in tf_votes.items())
        return AgentVote(
            self.name, action, score,
            reasoning=f"MTF: {tf_str} VPVR:{vpvr_signal}",
            meta={"tf_votes": tf_votes, "bull_ratio": round(bull_ratio, 3)},
        )

    def _adapt(self) -> None:
        """Auto-ajusta pesos dos TFs baseado na acurácia individual."""
        super()._adapt()
        # Calcula acurácia de cada TF
        for _, tf in self.TIMEFRAMES:
            hist = list(self._tf_accuracy[tf])
            if len(hist) < 10:
                continue
            acc = sum(hist) / len(hist)
            # Pesos maiores para TFs mais acurados
            self._tf_weights[tf] = max(0.05, min(0.40, acc * 0.40))

        # Normaliza
        total = sum(self._tf_weights.values())
        if total > 0:
            for tf in self._tf_weights:
                self._tf_weights[tf] /= total

        self._save_tf_weights()
        logger.debug("KRONOS: TF weights atualizados.", weights={k: round(v, 3) for k, v in self._tf_weights.items()})

    def record_outcome(self, action: str, signal: str, won: bool, pnl: float) -> None:
        """Registra resultado também nos históricos por TF."""
        super().record_outcome(action, signal, won, pnl)
        for _, tf in self.TIMEFRAMES:
            last_vote = self._last_tf_votes.get(tf, "NEUTRAL")
            if last_vote == action:
                self._tf_accuracy[tf].append(won)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resample(self, df: pd.DataFrame, n_bars: int) -> pd.DataFrame:
        if len(df) < n_bars * 3:
            return df
        rows = []
        cl = df["close"].astype(float).values
        hi = df["high"].astype(float).values  if "high" in df.columns else cl
        lo = df["low"].astype(float).values   if "low"  in df.columns else cl
        op = df["open"].astype(float).values  if "open" in df.columns else cl

        for i in range(0, len(df) - n_bars + 1, n_bars):
            s = slice(i, i + n_bars)
            row = {
                "open":  op[i],
                "high":  float(hi[s].max()),
                "low":   float(lo[s].min()),
                "close": cl[i + n_bars - 1],
            }
            for col in ("ema_9", "ema_21", "ema_50", "rsi_14", "macd_hist", "adx", "volume"):
                if col in df.columns:
                    row[col] = float(df[col].iloc[i + n_bars - 1])
            rows.append(row)
        return pd.DataFrame(rows) if rows else df

    def _analyze_timeframe(self, df: pd.DataFrame, label: str) -> tuple[str, float]:
        if len(df) < 5:
            return "NEUTRAL", 0.5
        try:
            last  = df.iloc[-1]
            close = self._safe_float(last.get("close", 0.0))
            ema9  = self._safe_float(last.get("ema_9",  close))
            ema21 = self._safe_float(last.get("ema_21", close))
            ema50 = self._safe_float(last.get("ema_50", close))
            rsi   = self._safe_float(last.get("rsi_14", 50.0))
            macd  = self._safe_float(last.get("macd_hist", 0.0))
            adx   = self._safe_float(last.get("adx", 20.0))

            bull_pts = sum([
                ema9 > ema21,
                ema21 > ema50,
                rsi > 50,
                macd > 0,
                adx > 20,
            ])
            bear_pts = sum([
                ema9 < ema21,
                ema21 < ema50,
                rsi < 50,
                macd < 0,
                adx > 20,
            ])

            if bull_pts >= 4:
                score = 0.60 + (bull_pts - 3) * 0.07
                return "BUY", min(score, 0.88)
            elif bear_pts >= 4:
                score = 0.60 + (bear_pts - 3) * 0.07
                return "SELL", min(score, 0.88)
            elif bull_pts == 3:
                return "BUY", 0.58
            elif bear_pts == 3:
                return "SELL", 0.58
            else:
                return "NEUTRAL", 0.50
        except Exception:
            return "NEUTRAL", 0.5

    def _compute_vpvr(self, df: pd.DataFrame, signal) -> tuple[str, float]:
        """
        Volume Profile (VPVR simplificado):
        Divide o range em 10 faixas, acumula volume em cada faixa.
        O POC (Point of Control) é a faixa com mais volume.
        """
        try:
            if "volume" not in df.columns or len(df) < 20:
                return "NEUTRAL", 0.5
            tail = df.tail(50)
            price_min = float(tail["low"].min())
            price_max = float(tail["high"].max())
            if price_max <= price_min:
                return "NEUTRAL", 0.5
            last_close = float(tail["close"].iloc[-1])
            n_bins     = 10
            bin_size   = (price_max - price_min) / n_bins
            bins: list[float] = [0.0] * n_bins

            for _, row in tail.iterrows():
                c   = float(row["close"])
                vol = float(row.get("volume", 1.0))
                idx = min(int((c - price_min) / bin_size), n_bins - 1)
                bins[idx] += vol

            poc_idx   = int(np.argmax(bins))
            poc_price = price_min + poc_idx * bin_size + bin_size / 2

            prox = self._thresholds.get("vpvr_proximity", 0.01)
            dist = abs(last_close - poc_price) / last_close if last_close > 0 else 1.0

            if dist < prox:
                # Preço no POC → zona de decisão
                sig_dir = (
                    signal.direction.value
                    if hasattr(signal.direction, "value")
                    else str(signal.direction)
                )
                is_buy = sig_dir in ("BUY", "buy", "CALL", "call")
                score  = 0.65 * (1 - dist / prox)
                return ("BUY" if is_buy else "SELL"), score
            elif last_close < poc_price:
                return "BUY", 0.58   # Preço abaixo do POC → pullback esperado
            else:
                return "SELL", 0.55  # Preço acima do POC → resistência
        except Exception:
            return "NEUTRAL", 0.5

    def _save_tf_weights(self) -> None:
        os.makedirs(os.path.dirname(_WEIGHTS_PATH), exist_ok=True)
        try:
            with open(_WEIGHTS_PATH, "w") as f:
                json.dump(self._tf_weights, f)
        except Exception:
            pass

    def _load_tf_weights(self) -> None:
        try:
            if os.path.exists(_WEIGHTS_PATH):
                with open(_WEIGHTS_PATH) as f:
                    self._tf_weights = json.load(f)
        except Exception:
            pass