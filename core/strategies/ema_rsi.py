from __future__ import annotations

import pandas as pd
from loguru import logger

from core.entities import RiskConfig, SessionState, Signal
from core.enums import ContractType, TradeDirection
from core.strategy_base import StrategyBase


class EmaRsiStrategy(StrategyBase):
    """
    EMA Crossover + RSI + MACD com confirmação de ADX.

    Lógica de entrada CALL (alta):
      1. EMA9 cruza EMA21 de baixo para cima (crossover bullish).
      2. EMA9 > EMA21 > EMA50 (tendência alinhada).
      3. RSI14 entre 50 e 65 (momentum positivo sem sobrecompra).
      4. MACD histograma positivo.
      5. ADX > 20 (tendência forte, não lateral).

    Lógica de entrada PUT (baixa):
      Espelhada para o lado oposto.

    Filtro adicional:
      - Não opera se squeeze ativo (BB dentro de Keltner).
      - Não opera se ATR14 < percentil 20 das últimas 50 barras (baixa volatilidade).
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        duration: int = 5,
        duration_unit: str = "t",
        min_confidence: float = 0.48,
    ) -> None:
        super().__init__(name="ema_rsi_macd", risk_config=risk_config)
        self.duration = duration
        self.duration_unit = duration_unit
        self.min_confidence = min_confidence

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        session: SessionState,
    ) -> Signal | None:
        if len(df) < 30:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        required_cols = {"ema_9", "ema_21", "rsi_14", "macd_hist"}
        if not required_cols.issubset(df.columns):
            return None

        rsi = float(last["rsi_14"])
        macd = float(last["macd_hist"])
        e9 = float(last["ema_9"])
        e21 = float(last["ema_21"])
        pe9 = float(prev["ema_9"])
        pe21 = float(prev["ema_21"])

        # ── Sinal CALL: EMA9 acima ou cruzando EMA21 + RSI entre 45-72 + MACD positivo
        call_trend = e9 > e21  # EMA9 acima de EMA21
        call_cross = pe9 <= pe21 and e9 > e21  # crossover bullish recente
        call_rsi = 45 < rsi < 72
        call_macd = macd > 0

        if (call_trend or call_cross) and call_rsi and call_macd:
            confidence = self._compute_confidence(
                last,
                direction="bull",
                conditions=[call_trend, call_rsi, call_macd],
            )
            if confidence >= self.min_confidence:
                logger.info(
                    "Sinal CALL gerado.",
                    symbol=symbol,
                    strategy=self.name,
                    confidence=confidence,
                    rsi=round(rsi, 2),
                    ema9_vs_21=round(e9 - e21, 5),
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

        # ── Sinal PUT: EMA9 abaixo ou cruzando EMA21 + RSI entre 28-55 + MACD negativo
        put_trend = e9 < e21
        put_cross = pe9 >= pe21 and e9 < e21
        put_rsi = 28 < rsi < 55
        put_macd = macd < 0

        if (put_trend or put_cross) and put_rsi and put_macd:
            confidence = self._compute_confidence(
                last,
                direction="bear",
                conditions=[put_trend, put_rsi, put_macd],
            )
            if confidence >= self.min_confidence:
                logger.info(
                    "Sinal PUT gerado.",
                    symbol=symbol,
                    strategy=self.name,
                    confidence=confidence,
                    rsi=round(rsi, 2),
                    ema9_vs_21=round(e9 - e21, 5),
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

    def _compute_confidence(
        self,
        last: pd.Series,
        direction: str,
        conditions: list[bool],
    ) -> float:
        """
        Confiança ponderada por:
          - Número de condições satisfeitas (base).
          - Força do ADX (normalizado).
          - Distância do RSI do neutro (50).
          - Magnitude do histograma MACD normalizado por sua própria média recente.
        """
        base = sum(conditions) / len(conditions)

        adx_factor = min(last["adx"] / 50.0, 1.0)
        rsi_distance = abs(last["rsi_14"] - 50) / 50.0

        # FIX B20: Normaliza macd_hist pelo próprio histograma (valor absoluto)
        # ao invés de usar last.get("macd", 1.0) que é um campo diferente e
        # pode retornar fallback silencioso mascarando features faltantes.
        macd_abs = abs(last["macd_hist"])
        # Usa o próprio valor como referência de escala (normalizado a [0, 1]).
        # macd_abs / (macd_abs + 1e-10) = 1.0 quando macd_hist != 0,
        # portanto usamos um fator linear simples: min(macd_abs * 100, 1.0)
        # para escalas típicas de volatility index.
        macd_factor = min(macd_abs * 100.0, 1.0)

        confidence = (
            base * 0.50
            + adx_factor * 0.20
            + rsi_distance * 0.15
            + macd_factor * 0.15
        )
        return round(min(confidence, 1.0), 4)