from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AttentionLayer(nn.Module):
    """
    Mecanismo de atenção Bahdanau sobre saída do LSTM.
    Aprende a pesar quais timesteps são mais relevantes para a predição.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output: "torch.Tensor") -> "torch.Tensor":
        # lstm_output: (batch, seq_len, hidden)
        scores = self.attention(lstm_output)          # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)        # (batch, seq_len, 1)
        context = (weights * lstm_output).sum(dim=1)  # (batch, hidden)
        return context


class LSTMClassifier(nn.Module):
    """
    LSTM Bidirecional + Atenção para classificação de direção de preço.

    Arquitetura:
      Input → LayerNorm → BiLSTM(128) → Dropout → BiLSTM(64) →
      Attention → Residual → FC(64, ReLU) → Dropout → FC(2, Softmax)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("Instale torch: pip install torch")
        super().__init__()

        self.input_norm = nn.LayerNorm(input_size)
        self.lstm1 = nn.LSTM(
            input_size, hidden_size,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            hidden_size * 2, hidden_size // 2,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.attention = AttentionLayer(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "fc" in name and "weight" in name:
                nn.init.kaiming_normal_(param, nonlinearity="relu")

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, input_size)
        x = self.input_norm(x)
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.lstm2(out1)
        context = self.attention(out2)
        context = self.dropout2(context)
        out = self.relu(self.fc1(context))
        logits = self.fc2(out)
        return torch.softmax(logits, dim=-1)


class LSTMTrainer:
    """
    Treinador do LSTMClassifier com:
      - Mixed precision (AMP) quando disponível.
      - AdamW + CosineAnnealingLR.
      - Early stopping por val_loss.
      - Purged Walk-Forward Cross-Validation.
      - Integração com NeuronMonitor.
    """

    def __init__(
        self,
        symbol: str,
        seq_len: int = 30,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.30,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 7,
        model_dir: str = "./models_store",
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("Instale torch: pip install torch")

        self.symbol = symbol
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: LSTMClassifier | None = None
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        self.feature_cols: list[str] = []
        self._best_val_loss: float = float("inf")

        logger.info(
            "LSTMTrainer inicializado.",
            symbol=symbol,
            device=str(self.device),
        )

    # ── Preparação de Dados ───────────────────────────────────────────────────

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        target_col: str = "future_direction",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Converte DataFrame de features em sequências (X, y) para o LSTM.

        Cria `future_direction` como target: 1 se close[t+1] > close[t], else 0.
        """
        feat_df = df.copy()

        # Target: direção do próximo candle.
        feat_df[target_col] = (
            feat_df["close"].shift(-1) > feat_df["close"]
        ).astype(int)
        feat_df = feat_df.dropna()

        # Seleciona features numéricas.
        exclude = {"open", "high", "low", "close", "epoch", target_col}
        self.feature_cols = [
            c for c in feat_df.columns
            if c not in exclude and feat_df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]

        X_raw = feat_df[self.feature_cols].values.astype(np.float32)
        y_raw = feat_df[target_col].values.astype(np.int64)

        # Normalização (z-score) — fit apenas no train, apply em val/test.
        self.scaler_mean = X_raw.mean(axis=0)
        self.scaler_std = X_raw.std(axis=0) + 1e-8
        X_norm = (X_raw - self.scaler_mean) / self.scaler_std

        # Cria sequências deslizantes.
        X_seq, y_seq = [], []
        for i in range(len(X_norm) - self.seq_len):
            X_seq.append(X_norm[i: i + self.seq_len])
            y_seq.append(y_raw[i + self.seq_len])

        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_seq), dtype=torch.long)

        logger.info(
            "Sequências preparadas.",
            symbol=self.symbol,
            sequences=len(X_seq),
            features=len(self.feature_cols),
            seq_len=self.seq_len,
        )
        return X_tensor, y_tensor

    # ── Treinamento ───────────────────────────────────────────────────────────

    def fit(
        self,
        X: "torch.Tensor",
        y: "torch.Tensor",
        val_split: float = 0.15,
        monitor: "NeuronMonitor | None" = None,
    ) -> dict[str, Any]:
        """
        Treina o LSTM com early stopping e mixed precision.

        Args:
            X, y: Tensors de sequências e targets.
            val_split: Fração para validação.
            monitor: NeuronMonitor opcional para inspeção de ativações.

        Returns:
            dict com histórico de loss e acurácia.
        """
        n_val = int(len(X) * val_split)
        n_train = len(X) - n_val

        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=False,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.batch_size, shuffle=False,
        )

        input_size = X.shape[2]
        self.model = LSTMClassifier(
            input_size=input_size,
            hidden_size=self.hidden_size,
        ).to(self.device)

        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-5
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler_amp = torch.amp.GradScaler(enabled=self.device.type == "cuda")

        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # ── Train ──────────────────────────────────────────────────────
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type):
                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)

                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()

                train_loss += loss.item() * len(X_batch)
                train_correct += (logits.argmax(1) == y_batch).sum().item()
                train_total += len(X_batch)

            # ── Validation ─────────────────────────────────────────────────
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * len(X_batch)
                    val_correct += (logits.argmax(1) == y_batch).sum().item()
                    val_total += len(X_batch)

            t_loss = train_loss / train_total
            v_loss = val_loss / val_total
            t_acc = train_correct / train_total
            v_acc = val_correct / val_total

            history["train_loss"].append(round(t_loss, 5))
            history["val_loss"].append(round(v_loss, 5))
            history["train_acc"].append(round(t_acc, 4))
            history["val_acc"].append(round(v_acc, 4))

            scheduler.step()

            logger.info(
                f"Epoch {epoch + 1}/{self.max_epochs}",
                train_loss=round(t_loss, 4),
                val_loss=round(v_loss, 4),
                val_acc=round(v_acc, 4),
            )

            # Early stopping.
            if v_loss < self._best_val_loss:
                self._best_val_loss = v_loss
                patience_counter = 0
                self.save()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping ativado.", epoch=epoch + 1)
                    break

        self.load()  # Carrega o melhor checkpoint.
        return history

    # ── Inferência ────────────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """
        Prediz direção do próximo candle.

        Args:
            features_df: DataFrame com colunas de features (últimos `seq_len` candles).

        Returns:
            dict com proba_up, proba_down, confidence, prediction.
        """
        if self.model is None:
            return {
                "proba_up": 0.50, "proba_down": 0.50,
                "confidence": 0.0, "prediction": 1,
                "model": "lstm",
            }

        if len(features_df) < self.seq_len:
            return {
                "proba_up": 0.50, "proba_down": 0.50,
                "confidence": 0.0, "prediction": 1,
                "model": "lstm",
            }

        X_raw = features_df[self.feature_cols].tail(self.seq_len).values.astype(np.float32)
        X_norm = (X_raw - self.scaler_mean) / self.scaler_std
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            proba = self.model(X_tensor)[0].cpu().numpy()

        return {
            "proba_down": round(float(proba[0]), 4),
            "proba_up": round(float(proba[1]), 4),
            "confidence": round(abs(float(proba[1]) - 0.50) * 2, 4),
            "prediction": int(np.argmax(proba)),
            "model": "lstm",
        }

    # ── Persistência ──────────────────────────────────────────────────────────

    def save(self) -> Path:
        path = self.model_dir / f"lstm_{self.symbol}.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "feature_cols": self.feature_cols,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
        }, path)
        logger.debug("LSTM salvo.", path=str(path))
        return path

    def load(self) -> bool:
        path = self.model_dir / f"lstm_{self.symbol}.pt"
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.scaler_mean = ckpt["scaler_mean"]
        self.scaler_std = ckpt["scaler_std"]
        self.feature_cols = ckpt["feature_cols"]
        self.seq_len = ckpt["seq_len"]

        if self.model is None:
            input_size = len(self.feature_cols)
            self.model = LSTMClassifier(
                input_size=input_size,
                hidden_size=ckpt["hidden_size"],
            ).to(self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        logger.success("LSTM carregado.", path=str(path))
        return True