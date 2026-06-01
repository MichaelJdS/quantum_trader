from __future__ import annotations

import pandas as pd
from loguru import logger

from core.entities import RiskConfig, SessionState, Signal
from core.enums import ContractType, TradeDirection
from core.strategy_base import StrategyBase


class BreakoutStrategy(StrategyBase):
    """
    Breakout de Squeeze com confirmação de ATR.

    Lógica:
      1. Squeeze liberado (squeeze_release == 1): BB saiu do Keltner.
      2. Candle de breakout com corpo > 60% do range total.
      3. ATR14 acima da média dos últimos 20 períodos (volatilidade expansiva).
      4. Direção determinada pela posição do close em relação ao BB mid.

    Filtro de falso breakout:
      - Close deve romper definitivamente (close > bb_upper para CALL,
        close < bb_lower para PUT).
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        duration: int = 5,
        duration_unit: str = "t",
        min_confidence: float = 0.62,
    ) -> None:
        super().__init__(name="breakout_squeeze", risk_config=risk_config)
        self.duration = duration
        self.duration_unit = duration_unit
        self.min_confidence = min_confidence

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        session: SessionState,
    ) -> Signal | None:
        if len(df) < 22:
            return None

        last = df.iloc[-1]
        required = {
            "squeeze_release", "close", "bb_upper", "bb_lower", "bb_mid",
            "candle_body_pct", "atr_14",
        }
        if not required.issubset(df.columns):
            return None

        if last["squeeze_release"] != 1:
            return None

        atr_avg = df["atr_14"].tail(20).mean()
        if last["atr_14"] < atr_avg:
            return None  # Sem expansão de volatilidade — falso breakout.

        strong_body = last["candle_body_pct"] > 0.60

        # ── CALL: rompimento para cima ────────────────────────────────────────
        if (
            last["close"] > last["bb_upper"]
            and last["is_bullish"] == 1
            and strong_body
        ):
            confidence = self._confidence(last, atr_avg, "bull")
            if confidence >= self.min_confidence:
                logger.info(
                    "Sinal CALL (breakout).",
                    symbol=symbol,
                    confidence=confidence,
                    atr_ratio=round(last["atr_14"] / atr_avg, 3),
                )
                return Signal(
                    symbol=symbol,
                    direction=TradeDirection.BUY,
                    confidence=confidence,
                    strategy_name=self.name,
                    model_name="technical",
                    contract_type=ContractType.CALL,
                    entry_price=float(last["close"]),
                )

        # ── PUT: rompimento para baixo ────────────────────────────────────────
        if (
            last["close"] < last["bb_lower"]
            and last["is_bearish"] == 1
            and strong_body
        ):
            confidence = self._confidence(last, atr_avg, "bear")
            if confidence >= self.min_confidence:
                logger.info(
                    "Sinal PUT (breakout).",
                    symbol=symbol,
                    confidence=confidence,
                    atr_ratio=round(last["atr_14"] / atr_avg, 3),
                )
                return Signal(
                    symbol=symbol,
                    direction=TradeDirection.SELL,
                    confidence=confidence,
                    strategy_name=self.name,
                    model_name="technical",
                    contract_type=ContractType.PUT,
                    entry_price=float(last["close"]),
                )

        return None

    def _confidence(
        self,
        last: pd.Series,
        atr_avg: float,
        direction: str,
    ) -> float:
        atr_expansion = min(last["atr_14"] / (atr_avg + 1e-10), 2.0) / 2.0
        body_score = last["candle_body_pct"]
        bb_ext = (
            (last["close"] - last["bb_upper"]) / (last["bb_upper"] + 1e-10)
            if direction == "bull"
            else (last["bb_lower"] - last["close"]) / (last["bb_lower"] + 1e-10)
        )
        score = (
            atr_expansion * 0.40
            + body_score * 0.35
            + max(bb_ext, 0) * 0.25
        )
        return round(min(0.50 + score * 0.50, 1.0), 4)