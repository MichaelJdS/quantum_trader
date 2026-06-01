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
        min_confidence: float = 0.60,
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
        # Guard mínimo interno: o ExecutionEngine garante 55+ candles,
        # mas mantemos o guard para uso isolado/testes da estratégia.
        if len(df) < 3:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        required_cols = {
            "ema_9", "ema_21", "ema_50", "rsi_14",
            "macd_hist", "adx", "squeeze", "atr_14",
        }
        if not required_cols.issubset(df.columns):
            logger.warning(
                "Features insuficientes para EmaRsiStrategy.",
                missing=required_cols - set(df.columns),
            )
            return None

        # ── Filtros de condição de mercado ────────────────────────────────────

        if last["squeeze"] == 1:
            return None  # Mercado comprimido — aguardar breakout.

        atr_threshold = df["atr_14"].tail(50).quantile(0.20)
        if last["atr_14"] < atr_threshold:
            return None  # Volatilidade muito baixa.

        # ── Sinal CALL ────────────────────────────────────────────────────────

        ema_cross_bull = prev["ema_9"] <= prev["ema_21"] and last["ema_9"] > last["ema_21"]
        trend_aligned = last["ema_9"] > last["ema_21"] > last["ema_50"]
        # FIX B7: Range ajustado para 50-65 (era 45-65).
        # O range anterior sobrepunha o range de PUT (35-55) entre 45 e 55,
        # criando ambiguidade de direção quando RSI estava nessa faixa.
        rsi_ok_bull = 50 < last["rsi_14"] < 65
        macd_bull = last["macd_hist"] > 0
        adx_trending = last["adx"] > 20

        if ema_cross_bull and trend_aligned and rsi_ok_bull and macd_bull and adx_trending:
            confidence = self._compute_confidence(
                last,
                direction="bull",
                conditions=[ema_cross_bull, trend_aligned, rsi_ok_bull, macd_bull, adx_trending],
            )
            if confidence >= self.min_confidence:
                logger.info(
                    "Sinal CALL gerado.",
                    symbol=symbol,
                    strategy=self.name,
                    confidence=confidence,
                    rsi=round(last["rsi_14"], 2),
                    adx=round(last["adx"], 2),
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

        # ── Sinal PUT ─────────────────────────────────────────────────────────

        ema_cross_bear = prev["ema_9"] >= prev["ema_21"] and last["ema_9"] < last["ema_21"]
        trend_aligned_bear = last["ema_9"] < last["ema_21"] < last["ema_50"]
        # FIX B7: Range ajustado para 35-50 (era 35-55).
        # Ranges CALL (50-65) e PUT (35-50) agora são mutuamente exclusivos —
        # RSI = 50 não pertence a nenhum (zona neutra intencional).
        rsi_ok_bear = 35 < last["rsi_14"] < 50
        macd_bear = last["macd_hist"] < 0

        if ema_cross_bear and trend_aligned_bear and rsi_ok_bear and macd_bear and adx_trending:
            confidence = self._compute_confidence(
                last,
                direction="bear",
                conditions=[ema_cross_bear, trend_aligned_bear, rsi_ok_bear, macd_bear, adx_trending],
            )
            if confidence >= self.min_confidence:
                logger.info(
                    "Sinal PUT gerado.",
                    symbol=symbol,
                    strategy=self.name,
                    confidence=confidence,
                    rsi=round(last["rsi_14"], 2),
                    adx=round(last["adx"], 2),
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