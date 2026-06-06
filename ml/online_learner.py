from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    from river import drift, linear_model, metrics, preprocessing, stream
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    drift = linear_model = metrics = preprocessing = stream = None  # type: ignore
    logger.warning("River não instalado — online learning desabilitado.")


class OnlineLearner:
    """
    Aprendizado contínuo incremental usando River (Passive-Aggressive Classifier).

    Atualiza pesos a cada novo resultado de trade, sem retreinar do zero.
    Detecta mudança de regime via ADWIN (Adaptive Windowing).

    Fluxo:
        tick features → predict() → executar trade → resultado → learn()

    O modelo nunca para de aprender — cada trade é uma amostra nova.
    """

    FEATURE_COLS: list[str] = [
        "rsi_14", "macd_hist", "adx", "bb_pct", "atr_ratio",
        "ema_cross_9_21", "ema_bull_9_21_50", "stoch_k",
        "candle_body_pct", "squeeze", "candle_streak",
        "roll_ret_5", "roll_ret_10", "roll_std_20",
        "williams_r", "cci", "price_vs_ema50",
    ]

    def __init__(
        self,
        symbol: str,
        model_dir: str = "./models_store",
        drift_threshold: float = 0.002,
    ) -> None:
        if not RIVER_AVAILABLE:
            raise ImportError("Instale river: pip install river")

        self.symbol = symbol
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold

        # Modelo principal: PA com normalização online.
        self._pipeline = (
            preprocessing.StandardScaler()
            | linear_model.PAClassifier(C=0.1, mode=1)
        )

        # Detector de drift ADWIN por feature.
        self._drift_detector = drift.ADWIN(delta=drift_threshold)
        self._drift_count: int = 0

        # Métricas online acumuladas.
        self._accuracy = metrics.Accuracy()
        self._f1 = metrics.F1()
        self._kappa = metrics.CohenKappa()

        # Histórico leve (últimas 500 amostras) para recalibração.
        self._recent_x: list[dict] = []
        self._recent_y: list[int] = []
        self._max_recent: int = 500

        # Estatísticas de sessão.
        self._total_learned: int = 0
        self._total_drift_events: int = 0

        self._try_load()

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, features: pd.Series | dict) -> dict[str, Any]:
        """
        Prediz probabilidade de WIN (1) / LOSS (0) para um dado de entrada.

        Returns:
            dict com keys: prediction (int), proba_win (float),
                           proba_loss (float), confidence (float).
        """
        x = self._extract_features(features)

        try:
            pred = self._pipeline.predict_one(x)
            proba = self._pipeline.predict_proba_one(x)
        except Exception:
            # Modelo ainda sem dados suficientes — retorna neutro.
            return {
                "prediction": 1,
                "proba_win": 0.50,
                "proba_loss": 0.50,
                "confidence": 0.50,
                "model": "online_pa",
                "symbol": self.symbol,
            }

        proba_win = proba.get(1, 0.50)
        proba_loss = proba.get(0, 0.50)
        confidence = abs(proba_win - 0.50) * 2  # 0 = neutro, 1 = certeza.

        return {
            "prediction": int(pred) if pred is not None else 1,
            "proba_win": round(proba_win, 4),
            "proba_loss": round(proba_loss, 4),
            "confidence": round(confidence, 4),
            "model": "online_pa",
            "symbol": self.symbol,
        }

    # ── Learn ─────────────────────────────────────────────────────────────────

    def learn(self, features: pd.Series | dict, won: bool) -> dict[str, Any]:
        """
        Atualiza o modelo com resultado de um trade finalizado.

        Args:
            features: Features do momento da entrada no trade.
            won: True se o trade foi vencedor, False se perdedor.

        Returns:
            dict com métricas atualizadas e flag de drift detectado.
        """
        x = self._extract_features(features)
        y = int(won)

        # Atualiza métricas ANTES de aprender (avaliação honesta).
        try:
            pred = self._pipeline.predict_one(x)
            if pred is not None:
                self._accuracy.update(y, pred)
                self._f1.update(y, pred)
                self._kappa.update(y, pred)
        except Exception:
            pass

        # Aprende.
        self._pipeline.learn_one(x, y)
        self._total_learned += 1

        # Atualiza histórico recente.
        self._recent_x.append(x)
        self._recent_y.append(y)
        if len(self._recent_x) > self._max_recent:
            self._recent_x.pop(0)
            self._recent_y.pop(0)

        # Detecta drift.
        error = float(pred != y) if pred is not None else 0.0
        self._drift_detector.update(error)
        drift_detected = self._drift_detector.drift_detected

        if drift_detected:
            self._total_drift_events += 1
            self._drift_count += 1
            logger.warning(
                "Drift detectado — regime de mercado mudou.",
                symbol=self.symbol,
                total_drifts=self._total_drift_events,
                total_learned=self._total_learned,
            )
            if self._drift_count >= 3:
                self._reset_model()
                self._drift_count = 0

        # Salva checkpoint a cada 50 trades.
        if self._total_learned % 50 == 0:
            self.save()

        return {
            "total_learned": self._total_learned,
            "accuracy": round(float(self._accuracy.get()), 4),
            "f1": round(float(self._f1.get()), 4),
            "kappa": round(float(self._kappa.get()), 4),
            "drift_detected": drift_detected,
            "total_drifts": self._total_drift_events,
        }

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_model(self) -> None:
        """
        Reinicia o modelo após drift severo.
        Re-treina nas últimas `_max_recent` amostras para
        manter contexto recente.
        """
        logger.info(
            "Reiniciando modelo após drift severo.",
            symbol=self.symbol,
            retraining_samples=len(self._recent_x),
        )
        self._pipeline = (
            preprocessing.StandardScaler()
            | linear_model.PAClassifier(C=0.1, mode=1)
        )
        # Re-treina no histórico recente.
        for x, y in zip(self._recent_x, self._recent_y):
            self._pipeline.learn_one(x, y)

        logger.success(
            "Modelo reiniciado e re-treinado.",
            symbol=self.symbol,
            samples=len(self._recent_x),
        )

    # ── Persistência ──────────────────────────────────────────────────────────

    def save(self) -> Path:
        """Serializa o pipeline e métricas para disco."""
        path = self.model_dir / f"online_{self.symbol}.pkl"
        state = {
            "pipeline": self._pipeline,
            "accuracy": self._accuracy,
            "f1": self._f1,
            "kappa": self._kappa,
            "total_learned": self._total_learned,
            "total_drift_events": self._total_drift_events,
            "recent_x": self._recent_x[-100:],  # Salva apenas últimas 100.
            "recent_y": self._recent_y[-100:],
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.debug("OnlineLearner salvo.", path=str(path), symbol=self.symbol)
        return path

    def _try_load(self) -> None:
        """Tenta carregar checkpoint existente."""
        path = self.model_dir / f"online_{self.symbol}.pkl"
        if not path.exists():
            logger.debug("Nenhum checkpoint encontrado.", symbol=self.symbol)
            return
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._pipeline = state["pipeline"]
            self._accuracy = state["accuracy"]
            self._f1 = state["f1"]
            self._kappa = state["kappa"]
            self._total_learned = state["total_learned"]
            self._total_drift_events = state["total_drift_events"]
            self._recent_x = state.get("recent_x", [])
            self._recent_y = state.get("recent_y", [])
            logger.success(
                "OnlineLearner carregado do checkpoint.",
                symbol=self.symbol,
                total_learned=self._total_learned,
            )
        except Exception as exc:
            logger.error("Falha ao carregar checkpoint.", error=str(exc))

    # ── Métricas ──────────────────────────────────────────────────────────────

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "total_learned": self._total_learned,
            "accuracy": round(float(self._accuracy.get()), 4),
            "f1": round(float(self._f1.get()), 4),
            "kappa": round(float(self._kappa.get()), 4),
            "total_drifts": self._total_drift_events,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_features(self, features: pd.Series | dict) -> dict[str, float]:
        """Extrai e normaliza features para o modelo River (dict float)."""
        if hasattr(features, "to_dict"):
            row = features.to_dict()
        else:
            row = features
        return {
            col: float(row.get(col, 0.0))
            for col in self.FEATURE_COLS
            if col in row or True  # Inclui com 0.0 se ausente.
        }