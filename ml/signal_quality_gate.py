"""
ml/signal_quality_gate.py — Filtro de Qualidade de Sinal

Gatekeeper que avalia a consistência do sinal ANTES de chegar ao Oracle Council.
Elimina sinais que:
  1. Contradizem múltiplos timeframes (multi-TF filter)
  2. Têm baixa consistência histórica para o setup atual
  3. Geram em "limpe" do setup (zona de ruído perto de EMAs)
  4. São gerados durante horários/padrões historicamente perdedores
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from core.entities import Signal, SessionState, Trade


@dataclass
class SignalQualityReport:
    """Resultado da avaliação de qualidade do sinal."""
    passed:         bool
    score:          float           # 0.0 a 1.0
    rejection_reason: str           = ""
    checks:         dict[str, bool] = field(default_factory=dict)
    boost:          float           = 0.0  # multiplicador de confiança se aprovado


class SignalQualityGate:
    """
    Filtro multicamada de qualidade de sinal.

    Checks implementados:
      ✓ MTF_ALIGNMENT   — alinhamento com timeframe maior (20 candles = TF longo)
      ✓ EMA_DISTANCE    — preço não está em zona de ruído (< 0.05% das EMAs)
      ✓ RSI_EXTREME     — RSI não está em zona de reversão contra o sinal
      ✓ VOLUME_CONFIRM  — volume acima da média (se disponível)
      ✓ CONSECUTIVE_LOSS — para se há 3+ perdas consecutivas no símbolo
      ✓ SETUP_HISTORY   — taxa de vitória histórica do setup > threshold
    """

    # Limiares configuráveis
    MIN_QUALITY_SCORE   = 0.55   # score mínimo para aprovação
    EMA_NOISE_THRESHOLD = 0.0005 # 0.05% — distância mínima do preço às EMAs
    RSI_BUY_MAX         = 75.0   # RSI máximo para sinal de COMPRA
    RSI_SELL_MIN        = 25.0   # RSI mínimo para sinal de VENDA
    VOLUME_MA_MULT      = 0.8    # volume mínimo = 80% da média
    MIN_SETUP_WINRATE   = 0.45   # win rate histórica mínima do setup
    MIN_SETUP_SAMPLES   = 15     # amostras mínimas para avaliar setup history

    def __init__(self) -> None:
        # Histórico por símbolo+setup: deque de bool (True=ganhou)
        self._setup_history: dict[str, deque] = {}
        # Perdas consecutivas por símbolo
        self._consecutive_losses: dict[str, int] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        signal:  "Signal",
        df:      pd.DataFrame,
        session: "SessionState",
    ) -> SignalQualityReport:
        """
        Avalia a qualidade do sinal. Retorna SignalQualityReport.
        Se passed=False, o sinal deve ser descartado.
        """
        checks: dict[str, bool] = {}
        scores: list[float]     = []

        sig_dir = (
            signal.direction.value
            if hasattr(signal.direction, "value")
            else str(signal.direction)
        )
        is_buy = sig_dir in ("BUY", "buy", "CALL", "call")
        symbol = signal.symbol

        # ── Check 1: Alinhamento MTF ─────────────────────────────────────────
        mtf_ok, mtf_score = self._check_mtf_alignment(df, is_buy)
        checks["MTF_ALIGNMENT"] = mtf_ok
        scores.append(mtf_score)

        # ── Check 2: Distância das EMAs (não está em zona de ruído) ──────────
        ema_ok, ema_score = self._check_ema_distance(df)
        checks["EMA_DISTANCE"] = ema_ok
        scores.append(ema_score)

        # ── Check 3: RSI não extremo contra o sinal ───────────────────────────
        rsi_ok, rsi_score = self._check_rsi(df, is_buy)
        checks["RSI_EXTREME"] = rsi_ok
        scores.append(rsi_score)

        # ── Check 4: Volume acima da média ────────────────────────────────────
        vol_ok, vol_score = self._check_volume(df)
        checks["VOLUME_CONFIRM"] = vol_ok
        scores.append(vol_score)

        # ── Check 5: Perdas consecutivas no símbolo ───────────────────────────
        consec_ok = self._consecutive_losses.get(symbol, 0) < 3
        checks["CONSECUTIVE_LOSS"] = consec_ok
        scores.append(1.0 if consec_ok else 0.0)

        # ── Check 6: Histórico do setup ───────────────────────────────────────
        setup_key = f"{symbol}_{signal.strategy_name}_{sig_dir}"
        setup_ok, setup_score = self._check_setup_history(setup_key)
        checks["SETUP_HISTORY"] = setup_ok
        scores.append(setup_score)

        # ── Score final (média ponderada) ─────────────────────────────────────
        weights = [0.25, 0.15, 0.20, 0.10, 0.15, 0.15]
        final_score = float(np.dot(scores, weights))

        passed = final_score >= self.MIN_QUALITY_SCORE

        # Calcula boost se todos os checks críticos passaram
        boost = 1.0
        critical_passed = checks["MTF_ALIGNMENT"] and checks["RSI_EXTREME"] and checks["CONSECUTIVE_LOSS"]
        if passed and critical_passed and final_score >= 0.75:
            boost = 1.08  # +8% confiança em sinais de alta qualidade

        rejection_reason = ""
        if not passed:
            failed = [k for k, v in checks.items() if not v]
            rejection_reason = f"Qualidade insuficiente (score={final_score:.3f}): {', '.join(failed)}"
            logger.debug(
                "SignalQualityGate: REJEITADO",
                symbol=symbol,
                score=round(final_score, 3),
                failed_checks=failed,
                strategy=signal.strategy_name,
            )
        else:
            logger.debug(
                "SignalQualityGate: APROVADO",
                symbol=symbol,
                score=round(final_score, 3),
                boost=boost,
            )

        return SignalQualityReport(
            passed=passed,
            score=final_score,
            rejection_reason=rejection_reason,
            checks=checks,
            boost=boost,
        )

    def record_outcome(self, signal: "Signal | Trade", won: bool) -> None:
        """Registra resultado do trade para aprendizado do filtro."""
        sig_dir = (
            signal.direction.value
            if hasattr(signal.direction, "value")
            else str(signal.direction)
        )
        # Handle difference between Signal and Trade property names if needed
        strategy_name = getattr(signal, "strategy_name", "")
        setup_key = f"{signal.symbol}_{strategy_name}_{sig_dir}"

        if setup_key not in self._setup_history:
            self._setup_history[setup_key] = deque(maxlen=200)
        self._setup_history[setup_key].append(won)

        # Atualiza perdas consecutivas
        if won:
            self._consecutive_losses[signal.symbol] = 0
        else:
            self._consecutive_losses[signal.symbol] = (
                self._consecutive_losses.get(signal.symbol, 0) + 1
            )

    def get_stats(self) -> dict:
        """Retorna estatísticas do filtro para o dashboard."""
        stats = {}
        for key, hist in self._setup_history.items():
            if hist:
                stats[key] = {
                    "samples": len(hist),
                    "win_rate": round(sum(hist) / len(hist), 3),
                }
        return stats

    # ── Checks internos ───────────────────────────────────────────────────────

    def _check_mtf_alignment(
        self, df: pd.DataFrame, is_buy: bool
    ) -> tuple[bool, float]:
        """
        Verifica alinhamento com tendência de longo prazo.
        Usa EMA rápida vs lenta nos últimos 20 candles (proxy de TF maior).
        """
        try:
            if "ema_fast" in df.columns and "ema_slow" in df.columns:
                ema_fast = float(df["ema_fast"].iloc[-1])
                ema_slow = float(df["ema_slow"].iloc[-1])
                long_bias_up = ema_fast > ema_slow
                aligned = (is_buy and long_bias_up) or (not is_buy and not long_bias_up)
                score = 0.75 if aligned else 0.35
                return aligned, score
            # Fallback: close vs SMA20
            if "close" in df.columns and len(df) >= 20:
                sma20 = float(df["close"].tail(20).mean())
                last  = float(df["close"].iloc[-1])
                up    = last > sma20
                aligned = (is_buy and up) or (not is_buy and not up)
                score   = 0.70 if aligned else 0.40
                return aligned, score
        except Exception:
            pass
        return True, 0.60

    def _check_ema_distance(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Verifica se o preço está suficientemente longe das EMAs.
        Sinais gerados em cima de EMAs (< threshold %) são ruído.
        """
        try:
            close = float(df["close"].iloc[-1])
            emas  = []
            for col in ("ema_fast", "ema_slow", "ema_9", "ema_21"):
                if col in df.columns:
                    emas.append(float(df[col].iloc[-1]))
            if not emas:
                return True, 0.60

            min_dist = min(abs(close - e) / close for e in emas)
            if min_dist >= self.EMA_NOISE_THRESHOLD:
                score = min(1.0, min_dist / self.EMA_NOISE_THRESHOLD * 0.7)
                return True, score
            else:
                return False, min_dist / self.EMA_NOISE_THRESHOLD * 0.5
        except Exception:
            return True, 0.60

    def _check_rsi(self, df: pd.DataFrame, is_buy: bool) -> tuple[bool, float]:
        """
        Verifica se RSI não está em zona extrema contra a direção do sinal.
        Compra com RSI > 75 = sinal fraco. Venda com RSI < 25 = sinal fraco.
        """
        try:
            rsi_col = next(
                (c for c in ("rsi", "rsi_14", "RSI") if c in df.columns), None
            )
            if rsi_col is None:
                return True, 0.60
            rsi = float(df[rsi_col].iloc[-1])

            if is_buy:
                if rsi > self.RSI_BUY_MAX:
                    return False, max(0.2, (100 - rsi) / 100)
                score = 0.5 + (self.RSI_BUY_MAX - rsi) / (2 * self.RSI_BUY_MAX)
            else:
                if rsi < self.RSI_SELL_MIN:
                    return False, max(0.2, rsi / 100)
                score = 0.5 + (rsi - self.RSI_SELL_MIN) / (2 * (100 - self.RSI_SELL_MIN))

            return True, min(1.0, score)
        except Exception:
            return True, 0.60

    def _check_volume(self, df: pd.DataFrame) -> tuple[bool, float]:
        """Volume atual deve estar acima de VOLUME_MA_MULT × média."""
        try:
            vol_col = next(
                (c for c in ("volume", "vol", "Volume") if c in df.columns), None
            )
            if vol_col is None:
                return True, 0.60  # sem volume disponível → ignora check
            vol_now  = float(df[vol_col].iloc[-1])
            vol_mean = float(df[vol_col].tail(20).mean())
            if vol_mean <= 0:
                return True, 0.60
            ratio = vol_now / vol_mean
            ok    = ratio >= self.VOLUME_MA_MULT
            score = min(1.0, ratio * 0.6)
            return ok, score
        except Exception:
            return True, 0.60

    def _check_setup_history(self, setup_key: str) -> tuple[bool, float]:
        """Win rate histórica do setup deve ser > MIN_SETUP_WINRATE."""
        hist = self._setup_history.get(setup_key)
        if hist is None or len(hist) < self.MIN_SETUP_SAMPLES:
            # Sem dados suficientes → aprovação neutra
            return True, 0.60
        wr = sum(hist) / len(hist)
        ok = wr >= self.MIN_SETUP_WINRATE
        return ok, max(0.1, wr)


# Singleton global
_gate_instance: SignalQualityGate | None = None


def get_signal_gate() -> SignalQualityGate:
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = SignalQualityGate()
    return _gate_instance