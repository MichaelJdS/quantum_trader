from __future__ import annotations

import pytest

try:
    from ml.online_learner import OnlineLearner
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="River não instalado")
class TestOnlineLearner:
    def test_predict_before_learning(self, tmp_path):
        learner = OnlineLearner(symbol="R_50", model_dir=str(tmp_path))
        features = {col: 0.5 for col in OnlineLearner.FEATURE_COLS}
        result = learner.predict(features)
        assert "proba_win" in result
        assert 0.0 <= result["proba_win"] <= 1.0
        assert "confidence" in result

    def test_learn_updates_metrics(self, tmp_path):
        learner = OnlineLearner(symbol="R_50", model_dir=str(tmp_path))
        features = {col: 0.5 for col in OnlineLearner.FEATURE_COLS}
        for i in range(20):
            learner.learn(features, won=(i % 2 == 0))
        assert learner.metrics["total_learned"] == 20

    def test_save_and_load(self, tmp_path):
        learner = OnlineLearner(symbol="R_50", model_dir=str(tmp_path))
        features = {col: 0.3 for col in OnlineLearner.FEATURE_COLS}
        for _ in range(10):
            learner.learn(features, won=True)
        learner.save()

        learner2 = OnlineLearner(symbol="R_50", model_dir=str(tmp_path))
        assert learner2.metrics["total_learned"] == 10

    def test_drift_detection(self, tmp_path):
        learner = OnlineLearner(
            symbol="R_50",
            model_dir=str(tmp_path),
            drift_threshold=0.5,  # Sensível para forçar drift em teste.
        )
        features = {col: 0.5 for col in OnlineLearner.FEATURE_COLS}
        for i in range(100):
            learner.learn(features, won=(i % 3 != 0))
        assert learner.metrics["total_learned"] == 100


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="River não instalado")
def test_ensemble_soft_voting():
    from ml.models.ensemble import EnsembleModel
    ens = EnsembleModel(model_names=["lstm", "online_pa", "technical"])
    predictions = {
        "lstm": {"proba_up": 0.70, "proba_down": 0.30},
        "online_pa": {"proba_up": 0.65, "proba_down": 0.35},
        "technical": {"proba_up": 0.60, "proba_down": 0.40},
    }
    result = ens.predict(predictions)
    assert result.proba_up > 0.50
    assert result.prediction == 1
    assert abs(sum(ens.weights.values()) - 1.0) < 0.01