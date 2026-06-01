from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class MLOpsManager:
    """
    Gerencia experimentos MLFlow e otimização de hiperparâmetros com Optuna.

    Responsabilidades:
      - Loga métricas, parâmetros e artefatos de cada run.
      - Registra snapshots de modelos com versão.
      - Executa busca de hiperparâmetros via TPE (Tree-structured Parzen Estimator).
      - Purged Walk-Forward CV para evitar data leakage em séries temporais.
    """

    def __init__(
        self,
        experiment_name: str = "quantum_trader",
        tracking_uri: str = "sqlite:///mlflow.db",
        model_dir: str = "./models_store",
    ) -> None:
        self.experiment_name = experiment_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._active_run_id: str | None = None

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info("MLFlow configurado.", tracking_uri=tracking_uri)
        else:
            logger.warning("MLFlow não instalado — tracking desabilitado.")

    # ── Logging de Experimento ─────────────────────────────────────────────────

    def start_run(self, run_name: str, params: dict[str, Any]) -> str | None:
        """Inicia um run MLFlow e loga parâmetros."""
        if not MLFLOW_AVAILABLE:
            return None
        run = mlflow.start_run(run_name=run_name)
        self._active_run_id = run.info.run_id
        mlflow.log_params(params)
        logger.info("MLFlow run iniciado.", run_id=self._active_run_id, name=run_name)
        return self._active_run_id

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Loga métricas no run ativo."""
        if not MLFLOW_AVAILABLE or not self._active_run_id:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_name: str) -> None:
        """Loga modelo PyTorch como artefato MLFlow."""
        if not MLFLOW_AVAILABLE:
            return
        try:
            import torch
            mlflow.pytorch.log_model(model, artifact_name)
            logger.debug("Modelo logado no MLFlow.", artifact=artifact_name)
        except Exception as exc:
            logger.error("Falha ao logar modelo.", error=str(exc))

    def end_run(self) -> None:
        if MLFLOW_AVAILABLE:
            mlflow.end_run()
            logger.info("MLFlow run encerrado.", run_id=self._active_run_id)
            self._active_run_id = None

    # ── Purged Walk-Forward CV ────────────────────────────────────────────────

    @staticmethod
    def purged_walk_forward_splits(
        n_samples: int,
        n_splits: int = 5,
        purge_gap: int = 10,
        embargo: int = 5,
    ) -> list[tuple[list[int], list[int]]]:
        """
        Gera índices de treino/validação para Purged Walk-Forward CV.

        Evita data leakage em séries temporais:
          - purge_gap: candles removidos antes da validação (anti-lookahead).
          - embargo: candles removidos após a validação (anti-contaminação).

        Returns:
            Lista de (train_indices, val_indices).
        """
        splits = []
        fold_size = n_samples // (n_splits + 1)

        for fold in range(n_splits):
            train_end = fold_size * (fold + 1)
            val_start = train_end + purge_gap
            val_end = val_start + fold_size - embargo

            if val_end > n_samples:
                break

            train_idx = list(range(0, train_end))
            val_idx = list(range(val_start, val_end))
            splits.append((train_idx, val_idx))

        return splits

    # ── Hyperparameter Optimization ───────────────────────────────────────────

    def optimize_lstm_hyperparams(
        self,
        X: Any,
        y: Any,
        symbol: str,
        n_trials: int = 30,
    ) -> dict[str, Any]:
        """
        Busca os melhores hiperparâmetros para o LSTM via Optuna TPE.

        Parâmetros otimizados:
          - hidden_size: [64, 256]
          - num_layers: [1, 3]
          - dropout: [0.1, 0.5]
          - lr: [1e-4, 1e-2]
          - batch_size: [32, 128]

        Returns:
            dict com melhores parâmetros encontrados.
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna não instalado — usando defaults.")
            return {
                "hidden_size": 128, "dropout": 0.30,
                "lr": 1e-3, "batch_size": 64,
            }

        import torch

        def objective(trial: "optuna.Trial") -> float:
            from ml.models.lstm_model import LSTMTrainer

            params = {
                "hidden_size": trial.suggest_categorical(
                    "hidden_size", [64, 128, 256]
                ),
                "dropout": trial.suggest_float("dropout", 0.10, 0.50),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [32, 64, 128]
                ),
            }

            trainer = LSTMTrainer(
                symbol=symbol,
                hidden_size=params["hidden_size"],
                dropout=params["dropout"],
                lr=params["lr"],
                batch_size=params["batch_size"],
                max_epochs=15,
                patience=5,
                model_dir=str(self.model_dir / "optuna_trials"),
            )

            try:
                history = trainer.fit(X, y, val_split=0.15)
                return min(history["val_loss"])
            except Exception as exc:
                logger.error("Trial falhou.", error=str(exc))
                raise optuna.exceptions.TrialPruned()

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            study_name=f"lstm_{symbol}",
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, timeout=300)

        best = study.best_params
        logger.success(
            "Otimização concluída.",
            symbol=symbol,
            best_params=best,
            best_val_loss=round(study.best_value, 5),
        )
        return best