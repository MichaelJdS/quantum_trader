from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """
    Pipeline completo de feature engineering para séries temporais de preço.

    Indicadores calculados (sem dependência de pandas-ta para o núcleo):
      - Tendência  : EMA (9, 21, 50, 200), SMA (20, 50), MACD, ADX
      - Momentum   : RSI (7, 14), Stochastic, CCI, Williams %R
      - Volatilidade: Bollinger Bands, ATR (7, 14), Keltner Channel, Squeeze
      - Price Action: candle body, wick ratio, gap, higher-high/lower-low
      - Rolling stats: mean, std, skew, kurt, min, max (janelas 5,10,20,50)
      - Lag features: lags 1,2,3,5,10
      - Streak: contagem de candles consecutivos de alta/baixa
    """

    EMA_PERIODS: tuple[int, ...] = (9, 21, 50, 200)
    SMA_PERIODS: tuple[int, ...] = (20, 50)
    ATR_PERIODS: tuple[int, ...] = (7, 14)
    RSI_PERIODS: tuple[int, ...] = (7, 14)
    ROLLING_WINDOWS: tuple[int, ...] = (5, 10, 20, 50)
    LAG_PERIODS: tuple[int, ...] = (1, 2, 3, 5, 10)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todo o pipeline de features sobre um DataFrame OHLCV.

        Args:
            df: DataFrame com colunas obrigatórias: open, high, low, close, epoch.
                Coluna volume é opcional.

        Returns:
            DataFrame enriquecido com todas as features. Sem NaN nas primeiras linhas
            (dropna aplicado ao final).
        """
        if df.empty or len(df) < 52:
            logger.warning("DataFrame insuficiente para feature engineering.", rows=len(df))
            return df

        feat = df.copy()
        feat = self._ema_sma(feat)
        feat = self._macd(feat)
        feat = self._adx(feat)
        feat = self._rsi(feat)
        feat = self._stochastic(feat)
        feat = self._cci(feat)
        feat = self._williams_r(feat)
        feat = self._bollinger_bands(feat)
        feat = self._atr(feat)
        feat = self._keltner_channel(feat)
        feat = self._squeeze(feat)
        feat = self._price_action(feat)
        feat = self._rolling_stats(feat)
        feat = self._lag_features(feat)
        feat = self._streak(feat)
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        feat.reset_index(drop=True, inplace=True)
        return feat

    # ── Tendência ─────────────────────────────────────────────────────────────

    def _ema_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        for p in self.EMA_PERIODS:
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
        for p in self.SMA_PERIODS:
            df[f"sma_{p}"] = df["close"].rolling(p).mean()

        df["ema_cross_9_21"] = (
            (df["ema_9"] > df["ema_21"]).astype(int)
            - (df["ema_9"] < df["ema_21"]).astype(int)
        )
        df["ema_bull_9_21_50"] = (
            (df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])
        ).astype(int)
        df["ema_bear_9_21_50"] = (
            (df["ema_9"] < df["ema_21"]) & (df["ema_21"] < df["ema_50"])
        ).astype(int)
        df["price_vs_ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
        df["price_vs_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"]
        return df

    def _macd(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_bull"] = (
            (df["macd"] > df["macd_signal"]) & (df["macd_hist"] > 0)
        ).astype(int)
        df["macd_bear"] = (
            (df["macd"] < df["macd_signal"]) & (df["macd_hist"] < 0)
        ).astype(int)
        return df

    def _adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        dm_plus = high.diff().clip(lower=0)
        dm_minus = (-low.diff()).clip(lower=0)
        dm_plus = dm_plus.where(dm_plus > dm_minus, 0.0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0.0)

        atr_adx = tr.ewm(alpha=1 / period, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(alpha=1 / period, adjust=False).mean() / atr_adx
        di_minus = 100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / atr_adx
        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10))
        df["adx"] = dx.ewm(alpha=1 / period, adjust=False).mean()
        df["di_plus"] = di_plus
        df["di_minus"] = di_minus
        df["adx_trending"] = (df["adx"] > 25).astype(int)
        return df

    # ── Momentum ──────────────────────────────────────────────────────────────

    def _rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.RSI_PERIODS:
            delta = df["close"].diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_bull"] = (df["rsi_14"] > 50).astype(int)
        return df

    def _stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> pd.DataFrame:
        low_min = df["low"].rolling(k_period).min()
        high_max = df["high"].rolling(k_period).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
        df["stoch_bull"] = (
            (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"] < 80)
        ).astype(int)
        return df

    def _cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df["cci"] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        return df

    def _williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high_max = df["high"].rolling(period).max()
        low_min = df["low"].rolling(period).min()
        df["williams_r"] = -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)
        return df

    # ── Volatilidade ──────────────────────────────────────────────────────────

    def _bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> pd.DataFrame:
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        df["bb_upper"] = sma + std_dev * std
        df["bb_mid"] = sma
        df["bb_lower"] = sma - std_dev * std
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-10
        )
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-10)
        df["price_below_bb_lower"] = (df["close"] < df["bb_lower"]).astype(int)
        df["price_above_bb_upper"] = (df["close"] > df["bb_upper"]).astype(int)
        return df

    def _atr(self, df: pd.DataFrame) -> pd.DataFrame:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        for period in self.ATR_PERIODS:
            df[f"atr_{period}"] = tr.ewm(alpha=1 / period, adjust=False).mean()
        df["atr_ratio"] = df["atr_14"] / (df["close"] + 1e-10)
        return df

    def _keltner_channel(
        self,
        df: pd.DataFrame,
        period: int = 20,
        multiplier: float = 2.0,
    ) -> pd.DataFrame:
        ema = df["close"].ewm(span=period, adjust=False).mean()
        df["kc_upper"] = ema + multiplier * df["atr_14"]
        df["kc_lower"] = ema - multiplier * df["atr_14"]
        return df

    def _squeeze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Squeeze Momentum: BB dentro do Keltner Channel indica compressão.
        squeeze = 1 → volatilidade comprimida (possível breakout iminente).
        """
        df["squeeze"] = (
            (df["bb_upper"] <= df["kc_upper"]) & (df["bb_lower"] >= df["kc_lower"])
        ).astype(int)
        df["squeeze_release"] = (
            df["squeeze"].shift(1) == 1
        ) & (df["squeeze"] == 0)
        df["squeeze_release"] = df["squeeze_release"].astype(int)
        return df

    # ── Price Action ──────────────────────────────────────────────────────────

    def _price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        body = (df["close"] - df["open"]).abs()
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
        candle_range = df["high"] - df["low"] + 1e-10

        df["candle_body"] = body
        df["candle_body_pct"] = body / candle_range
        df["upper_wick_pct"] = upper_wick / candle_range
        df["lower_wick_pct"] = lower_wick / candle_range
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)
        df["is_bearish"] = (df["close"] < df["open"]).astype(int)
        df["gap"] = df["open"] - df["close"].shift(1)
        df["gap_pct"] = df["gap"] / (df["close"].shift(1) + 1e-10)
        df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
        df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)
        df["pct_change"] = df["close"].pct_change()
        return df

    # ── Rolling Statistics ────────────────────────────────────────────────────

    def _rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in self.ROLLING_WINDOWS:
            df[f"roll_mean_{w}"] = df["close"].rolling(w).mean()
            df[f"roll_std_{w}"] = df["close"].rolling(w).std()
            df[f"roll_skew_{w}"] = df["close"].rolling(w).skew()
            df[f"roll_kurt_{w}"] = df["close"].rolling(w).kurt()
            df[f"roll_min_{w}"] = df["close"].rolling(w).min()
            df[f"roll_max_{w}"] = df["close"].rolling(w).max()
            df[f"roll_range_{w}"] = df[f"roll_max_{w}"] - df[f"roll_min_{w}"]
            df[f"roll_ret_{w}"] = df["close"].pct_change(w)
        return df

    # ── Lags ──────────────────────────────────────────────────────────────────

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in self.LAG_PERIODS:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"ret_lag_{lag}"] = df["pct_change"].shift(lag)
            df[f"rsi_lag_{lag}"] = df["rsi_14"].shift(lag)
            df[f"macd_hist_lag_{lag}"] = df["macd_hist"].shift(lag)
        return df

    # ── Streak ────────────────────────────────────────────────────────────────

    def _streak(self, df: pd.DataFrame) -> pd.DataFrame:
        """Contagem de candles consecutivos de alta/baixa."""
        streak = []
        current = 0
        for is_bull in df["is_bullish"]:
            if is_bull == 1:
                current = current + 1 if current > 0 else 1
            else:
                current = current - 1 if current < 0 else -1
            streak.append(current)
        df["candle_streak"] = streak
        df["streak_abs"] = df["candle_streak"].abs()
        return df