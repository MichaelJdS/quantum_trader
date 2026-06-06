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
      1. EMA9 cruza EMA21 de baixo para cima (crossover bullish) OU EMA9 > EMA21.
      2. RSI14 entre 52 e 68 (momentum positivo, fora de sobrecompra).
      3. MACD histograma positivo.
      4. Não opera se squeeze ativo (Bollinger dentro de Keltner).
      5. Não opera se ATR baixo (baixa volatilidade).

    Lógica de entrada PUT (baixa):
      1. EMA9 cruza EMA21 de cima para baixo (crossover bearish) OU EMA9 < EMA21.
      2. RSI14 entre 32 e 48 (momentum negativo, fora de sobrevenda).
      3. MACD histograma negativo.
      4. Filtros de squeeze e ATR idênticos.

    Nota sobre bandas de RSI:
      As zonas CALL (52–68) e PUT (32–48) são propositalmente separadas com
      uma zona neutra entre 48–52, eliminando a sobreposição anterior (45–55)
      que gerava ambiguidade de sinal com MACD como único desempate.

    Filtros adicionais:
      - Não opera se squeeze ativo (BB dentro de Keltner).
      - Não opera se ATR14 < percentil 20 das últimas 50 barras (baixa volatilidade).
    """

    # ── Zonas RSI sem sobreposição ─────────────────────────────────────────────
    #
    #   ┌─────────────────────────────────────────────────────────────┐
    #   │  0   32   48   52   68  100                                 │
    #   │  |    |    |    |    |                                       │
    #   │  [─── PUT ──]  NEUTRO  [──── CALL ───]                     │
    #   │       32–48     48–52       52–68                           │
    #   └─────────────────────────────────────────────────────────────┘
    #
    RSI_CALL_LOW  = 52.0
    RSI_CALL_HIGH = 68.0
    RSI_PUT_LOW   = 32.0
    RSI_PUT_HIGH  = 48.0

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
            logger.debug(
                "Colunas obrigatórias ausentes para EmaRsiStrategy.",
                symbol=symbol,
                missing=required_cols - set(df.columns),
            )
            return None

        rsi  = float(last["rsi_14"])
        macd = float(last["macd_hist"])
        e9   = float(last["ema_9"])
        e21  = float(last["ema_21"])
        pe9  = float(prev["ema_9"])
        pe21 = float(prev["ema_21"])

        # ── Filtros globais ────────────────────────────────────────────────────

        # Filtro de squeeze: BB dentro de Keltner → mercado lateral, sem operar.
        if self._is_squeeze_active(last):
            return None

        # Filtro de ATR baixo: mercado sem volatilidade suficiente.
        if self._is_low_volatility(df):
            return None

        # ── Sinal CALL ──────────────────────────────────────────────────────────
        #
        #  Condições:
        #    - EMA9 acima de EMA21 (tendência) OU crossover bullish recente
        #    - RSI entre 52 e 68 (zona positiva sem sobrecompra)
        #    - MACD histograma positivo
        #
        call_trend = e9 > e21
        call_cross = pe9 <= pe21 and e9 > e21
        call_rsi   = self.RSI_CALL_LOW < rsi < self.RSI_CALL_HIGH
        call_macd  = macd > 0

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
                    crossover=call_cross,
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

        # ── Sinal PUT ───────────────────────────────────────────────────────────
        #
        #  Condições simétricas às de CALL:
        #    - EMA9 abaixo de EMA21 (tendência) OU crossover bearish recente
        #    - RSI entre 32 e 48 (zona negativa sem sobrevenda extrema)
        #    - MACD histograma negativo
        #
        put_trend = e9 < e21
        put_cross = pe9 >= pe21 and e9 < e21
        put_rsi   = self.RSI_PUT_LOW < rsi < self.RSI_PUT_HIGH
        put_macd  = macd < 0

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
                    crossover=put_cross,
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

    # ── Helpers de filtro ──────────────────────────────────────────────────────

    def _is_squeeze_active(self, last: pd.Series) -> bool:
        """
        Retorna True se squeeze estiver ativo (BB dentro de Keltner).
        Se as colunas não existirem, retorna False (conservador: não bloqueia).
        """
        required = {"bb_upper", "bb_lower", "kc_upper", "kc_lower"}
        for col in required:
            if col not in last.index or pd.isna(last[col]):
                return False
        return (
            float(last["bb_upper"]) < float(last["kc_upper"])
            and float(last["bb_lower"]) > float(last["kc_lower"])
        )

    def _is_low_volatility(self, df: pd.DataFrame, lookback: int = 50) -> bool:
        """
        Retorna True se o ATR14 atual estiver abaixo do percentil 20
        das últimas `lookback` barras.
        Se a coluna não existir, retorna False (conservador: não bloqueia).
        """
        if "atr_14" not in df.columns:
            return False

        recent = df["atr_14"].dropna().tail(lookback)
        if len(recent) < 10:
            return False

        threshold = recent.quantile(0.20)
        return float(df["atr_14"].iloc[-1]) < threshold

    # ── Cálculo de confiança ───────────────────────────────────────────────────

    def _compute_confidence(
        self,
        last: pd.Series,
        direction: str,
        conditions: list[bool],
    ) -> float:
        """
        Confiança ponderada por:
          - Número de condições satisfeitas (base 50%).
          - Força do ADX normalizado (20%).
          - Distância do RSI da zona neutra (15%).
          - Magnitude do MACD histograma normalizado (15%).

        Notas:
          - ADX: normalizado pelo máximo empírico de 50. ADX > 50 é tratado como 1.0.
          - RSI distance: distância absoluta do neutro (50), normalizada para [0, 1].
            Tanto zona CALL (52–68) quanto PUT (32–48) são equidistantes de 50,
            garantindo que a função de confiança seja simétrica.
          - MACD factor: escala linear min(|macd_hist| × 100, 1.0) para volatility
            index típico (ex: R_50 = 50 pontos de movimento base).
        """
        base = sum(conditions) / len(conditions)

        # ADX: segurança contra coluna ausente
        adx_raw = last.get("adx", 0.0)
        adx_factor = min(float(adx_raw) / 50.0, 1.0) if adx_raw is not None else 0.0

        # RSI distance: centrado em 50, zona neutra (48–52) tem distância baixa
        rsi_raw = last.get("rsi_14", 50.0)
        rsi_distance = abs(float(rsi_raw) - 50.0) / 50.0

        # MACD factor: magnitude como proxy de força do sinal
        macd_abs = abs(float(last.get("macd_hist", 0.0)))
        macd_factor = min(macd_abs * 100.0, 1.0)

        confidence = (
            base         * 0.50
            + adx_factor   * 0.20
            + rsi_distance * 0.15
            + macd_factor  * 0.15
        )
        return round(min(confidence, 1.0), 4)