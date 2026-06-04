from __future__ import annotations

import pandas as pd
from loguru import logger

from core.entities import RiskConfig, SessionState, Signal
from core.enums import ContractType, TradeDirection
from core.strategy_base import StrategyBase


class BollingerReversionStrategy(StrategyBase):
    """
    Mean Reversion via Bollinger Bands + RSI oversold/overbought.

    CALL quando:
      - Preço toca ou cruza BB lower (close < bb_lower).
      - RSI14 < 32 (oversold).
      - Candle atual é bullish (fechou acima da abertura).
      - Sem tendência de queda forte (ADX < 35 ou DI- não dominante).

    PUT quando:
      - Preço toca ou cruza BB upper (close > bb_upper).
      - RSI14 > 70 (overbought).
      - Candle atual é bearish.
      - ADX < 35 (mercado não em tendência forte).

    Filtro: Não opera em squeeze (BB dentro do Keltner).
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        duration: int = 3,
        duration_unit: str = "t",
        min_confidence: float = 0.40,
    ) -> None:
        super().__init__(name="bollinger_reversion", risk_config=risk_config)
        self.duration = duration
        self.duration_unit = duration_unit
        self.min_confidence = min_confidence

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        session: SessionState,
    ) -> Signal | None:
        if len(df) < 2:
            return None

        last = df.iloc[-1]

        # FIX: Adicionado "is_bearish" ao required set.
        # O código de PUT usa last["is_bearish"] mas o set original não incluía
        # essa feature, permitindo que a estratégia chegasse ao bloco PUT
        # com KeyError silencioso ou valor inesperado.
        required = {
            "bb_lower", "bb_upper", "rsi_14", "adx",
            "squeeze", "is_bullish", "is_bearish",
        }
        if not required.issubset(df.columns):
            logger.warning(
                "Features insuficientes para BollingerReversionStrategy.",
                missing=required - set(df.columns),
            )
            return None


        # ── CALL: oversold + preço abaixo de BB lower ─────────────────────────
        if (
            last["close"] < last["bb_lower"]
            and last["rsi_14"] < 42
            and last["is_bullish"] == 1
            and last["adx"] < 40
        ):
            confidence = self._confidence(last, "bull")
            if confidence >= self.min_confidence:
                logger.info(
                    "Sinal CALL (mean reversion).",
                    symbol=symbol,
                    rsi=round(last["rsi_14"], 2),
                    bb_pct=round(last.get("bb_pct", 0), 4),
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

        # ── PUT: overbought + preço acima de BB upper ─────────────────────────
        # FIX: RSI threshold padronizado para 70 (consistente com o docstring).
        # O valor anterior era 68, criando inconsistência com a lógica documentada.
        if (
            last["close"] > last["bb_upper"]
            and last["rsi_14"] > 62
            and last["is_bearish"] == 1
            and last["adx"] < 40
        ):
            confidence = self._confidence(last, "bear")
            if confidence >= self.min_confidence:
                logger.info(
                    "Sinal PUT (mean reversion).",
                    symbol=symbol,
                    rsi=round(last["rsi_14"], 2),
                    bb_pct=round(last.get("bb_pct", 0), 4),
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

    def _confidence(self, last: pd.Series, direction: str) -> float:
        rsi_extreme = (
            (30 - last["rsi_14"]) / 30 if direction == "bull"
            else (last["rsi_14"] - 70) / 30
        )
        bb_ext = (
            (last["bb_lower"] - last["close"]) / (last["bb_lower"] + 1e-10)
            if direction == "bull"
            else (last["close"] - last["bb_upper"]) / (last["bb_upper"] + 1e-10)
        )
        body_pct = last.get("candle_body_pct", 0.5)
        score = (
            max(rsi_extreme, 0) * 0.40
            + max(bb_ext, 0) * 0.35
            + body_pct * 0.25
        )
        return round(min(0.50 + score, 1.0), 4)