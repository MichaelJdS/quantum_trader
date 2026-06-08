from __future__ import annotations

import math

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """
    Calcula features técnicas para estratégias e ML.

    Convenções:
      - Espera colunas: epoch, open, high, low, close
      - Retorna DataFrame ordenado por epoch crescente
      - Nunca muta o DataFrame de entrada
      - Remove NaNs apenas no final, preservando o máximo de histórico possível
    """

    REQUIRED_COLUMNS = {"epoch", "open", "high", "low", "close"}

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline principal de feature engineering.

        Features geradas:
          - Retornos e ranges
          - EMA 9/21/50
          - RSI 14
          - MACD, signal, hist
          - ATR 14
          - ADX 14
          - Bollinger Bands 20
          - Keltner Channels 20
          - Flags auxiliares para squeeze/volatilidade
        """
        if df is None or df.empty:
            return pd.DataFrame()

        if not self.REQUIRED_COLUMNS.issubset(df.columns):
            missing = self.REQUIRED_COLUMNS - set(df.columns)
            logger.warning(
                "FeatureEngineer recebeu DataFrame sem colunas obrigatórias.",
                missing=sorted(missing),
            )
            return pd.DataFrame()

        data = df.copy()

        data = self._normalize_types(data)
        data = self._sort_and_deduplicate(data)

        if len(data) < 30:
            return pd.DataFrame()

        # ── Base price features ────────────────────────────────────────────────
        data["return_1"] = data["close"].pct_change()
        data["log_return_1"] = np.log(data["close"] / data["close"].shift(1))
        data["range"] = data["high"] - data["low"]
        data["body"] = (data["close"] - data["open"]).abs()
        data["upper_wick"] = data["high"] - data[["open", "close"]].max(axis=1)
        data["lower_wick"] = data[["open", "close"]].min(axis=1) - data["low"]

        # ── EMAs ───────────────────────────────────────────────────────────────
        data["ema_9"] = self._ema(data["close"], span=9)
        data["ema_21"] = self._ema(data["close"], span=21)
        data["ema_50"] = self._ema(data["close"], span=50)

        # ── RSI ────────────────────────────────────────────────────────────────
        data["rsi_14"] = self._rsi(data["close"], period=14)

        # ── MACD ───────────────────────────────────────────────────────────────
        macd, macd_signal, macd_hist = self._macd(data["close"])
        data["macd"] = macd
        data["macd_signal"] = macd_signal
        data["macd_hist"] = macd_hist

        # ── ATR / ADX ──────────────────────────────────────────────────────────
        data["tr"] = self._true_range(data)
        data["atr_14"] = self._atr(data, period=14)
        data["adx"] = self._adx(data, period=14)

        # ── Bollinger Bands ────────────────────────────────────────────────────
        bb_mid, bb_upper, bb_lower, bb_width = self._bollinger_bands(
            data["close"], period=20, std_mult=2.0
        )
        data["bb_mid"] = bb_mid
        data["bb_upper"] = bb_upper
        data["bb_lower"] = bb_lower
        data["bb_width"] = bb_width

        # ── Keltner Channels ───────────────────────────────────────────────────
        kc_mid, kc_upper, kc_lower = self._keltner_channels(
            close=data["close"],
            atr=data["atr_14"],
            ema_period=20,
            atr_mult=1.5,
        )
        data["kc_mid"] = kc_mid
        data["kc_upper"] = kc_upper
        data["kc_lower"] = kc_lower

        # ── Flags auxiliares ───────────────────────────────────────────────────
        data["is_squeeze"] = (
            (data["bb_upper"] < data["kc_upper"])
            & (data["bb_lower"] > data["kc_lower"])
        )
        data["squeeze"] = data["is_squeeze"]  # Alias para compatibilidade com estratégias antigas
        
        data["is_bullish"] = (data["close"] > data["open"]).astype(int)
        data["is_bearish"] = (data["close"] < data["open"]).astype(int)

        atr_q20 = data["atr_14"].rolling(50, min_periods=20).quantile(0.20)
        data["low_volatility"] = data["atr_14"] < atr_q20

        # ── Sanity cleanup ─────────────────────────────────────────────────────
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Mantém apenas linhas com features suficientes para operar
        keep_cols = [
            "epoch", "open", "high", "low", "close",
            "return_1", "log_return_1", "range", "body",
            "upper_wick", "lower_wick",
            "ema_9", "ema_21", "ema_50",
            "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "tr", "atr_14", "adx",
            "bb_mid", "bb_upper", "bb_lower", "bb_width",
            "kc_mid", "kc_upper", "kc_lower",
            "is_squeeze", "squeeze", "is_bullish", "is_bearish", "low_volatility",
        ]
        data = data[keep_cols].dropna().reset_index(drop=True)

        return data

    # ── Normalização ──────────────────────────────────────────────────────────

    def _normalize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        numeric_cols = ["epoch", "open", "high", "low", "close"]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna(subset=numeric_cols)

        # Garante consistência OHLC
        out = out[
            (out["high"] >= out["low"])
            & (out["high"] >= out[["open", "close"]].max(axis=1))
            & (out["low"] <= out[["open", "close"]].min(axis=1))
        ]

        return out

    def _sort_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.sort_values("epoch")
            .drop_duplicates(subset=["epoch"], keep="last")
            .reset_index(drop=True)
        )

    # ── Indicadores ───────────────────────────────────────────────────────────

    def _ema(self, series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=span).mean()

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()

        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Casos extremos:
        # loss=0 e gain>0 => RSI=100
        # gain=0 e loss>0 => RSI=0
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
        rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0.0)

        return rsi.clip(0, 100)

    def _macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr_components = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        )
        return tr_components.max(axis=1)

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = self._true_range(df)
        return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
        )

        tr = self._true_range(df)
        atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        plus_di = 100 * (
            plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
            / atr.replace(0, np.nan)
        )
        minus_di = 100 * (
            minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
            / atr.replace(0, np.nan)
        )

        dx = (
            ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
            * 100
        )
        adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        return adx.clip(lower=0, upper=100)

    def _bollinger_bands(
        self,
        close: pd.Series,
        period: int = 20,
        std_mult: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        mid = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std(ddof=0)
        upper = mid + std_mult * std
        lower = mid - std_mult * std
        width = (upper - lower) / mid.replace(0, np.nan)
        return mid, upper, lower, width

    def _keltner_channels(
        self,
        close: pd.Series,
        atr: pd.Series,
        ema_period: int = 20,
        atr_mult: float = 1.5,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid = close.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
        upper = mid + atr_mult * atr
        lower = mid - atr_mult * atr
        return mid, upper, lower