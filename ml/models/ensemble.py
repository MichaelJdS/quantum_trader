from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class EnsemblePrediction:
    """Resultado agregado do ensemble."""
    proba_up: float
    proba_down: float
    confidence: float
    prediction: int
    votes: dict[str, int]
    weights_used: dict[str, float]


class EnsembleModel:
    """
    Ensemble de modelos com soft voting ponderado por performance recente.

    Modelos suportados:
      - lstm       : LSTMTrainer.predict()
      - online_pa  : OnlineLearner.predict()
      - technical  : confiança da estratégia técnica

    Peso de cada modelo é dinâmico:
      - Começa em 1/N.
      - Aumenta proporcionalmente ao accuracy recente (janela 50 trades).
      - Mínimo de 0.05 por modelo (nunca excluído completamente).
    """

    def __init__(self, model_names: list[str]) -> None:
        self._models = model_names
        n = len(model_names)
        self._weights: dict[str, float] = {m: 1.0 / n for m in model_names}
        self._performance: dict[str, list[int]] = {m: [] for m in model_names}
        self._window = 50

    def predict(self, predictions: dict[str, dict]) -> EnsemblePrediction:
        """
        Agrega predições de múltiplos modelos via soft voting ponderado.

        Args:
            predictions: {model_name: {"proba_up": float, "proba_down": float, ...}}

        Returns:
            EnsemblePrediction com resultado agregado.
        """
        if not predictions:
            return EnsemblePrediction(
                proba_up=0.50, proba_down=0.50,
                confidence=0.0, prediction=1,
                votes={}, weights_used={},
            )

        weighted_up = 0.0
        weighted_down = 0.0
        total_weight = 0.0
        votes: dict[str, int] = {}

        for model_name, pred in predictions.items():
            w = self._weights.get(model_name, 0.10)
            proba_up = pred.get("proba_up", 0.50)
            proba_down = pred.get("proba_down", 1 - proba_up)
            weighted_up += proba_up * w
            weighted_down += proba_down * w
            total_weight += w
            votes[model_name] = int(proba_up > 0.50)

        if total_weight > 0:
            weighted_up /= total_weight
            weighted_down /= total_weight

        prediction = int(weighted_up > 0.50)
        confidence = abs(weighted_up - 0.50) * 2

        logger.debug(
            "Ensemble prediction.",
            proba_up=round(weighted_up, 4),
            confidence=round(confidence, 4),
            votes=votes,
            weights=self._weights,
        )

        return EnsemblePrediction(
            proba_up=round(weighted_up, 4),
            proba_down=round(weighted_down, 4),
            confidence=round(confidence, 4),
            prediction=prediction,
            votes=votes,
            weights_used=dict(self._weights),
        )

    def update_performance(self, model_name: str, correct: bool) -> None:
        """
        Atualiza performance de um modelo após resultado de trade.
        Recalcula pesos dinamicamente.
        """
        if model_name not in self._performance:
            return

        self._performance[model_name].append(int(correct))
        if len(self._performance[model_name]) > self._window:
            self._performance[model_name].pop(0)

        self._recalculate_weights()

    def _recalculate_weights(self) -> None:
        """
        Recalcula pesos baseado em accuracy recente.
        Usa softmax para normalizar — modelos com melhor
        performance recente recebem mais peso.
        """
        import math
        accuracies = {}
        for model, results in self._performance.items():
            if len(results) >= 10:
                accuracies[model] = sum(results) / len(results)
            else:
                accuracies[model] = 0.50  # Prior neutro.

        # Softmax dos accuracies.
        exp_vals = {m: math.exp(acc * 5) for m, acc in accuracies.items()}
        total = sum(exp_vals.values())
        raw_weights = {m: v / total for m, v in exp_vals.items()}

        # Garante mínimo de 0.05 por modelo.
        min_w = 0.05
        for model in self._weights:
            self._weights[model] = max(raw_weights.get(model, 1 / len(self._models)), min_w)

        # Renormaliza.
        total_w = sum(self._weights.values())
        for model in self._weights:
            self._weights[model] = round(self._weights[model] / total_w, 4)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)