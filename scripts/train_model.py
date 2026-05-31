import os
import sys
import logging
import multiprocessing
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.lstm_predictor import MarketLSTM, prepare_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

torch.set_num_threads(multiprocessing.cpu_count())

# ─── Épocas separadas: poucas na busca, mais no treino final ──────────────────
SEARCH_EPOCHS = 7   # era 15 → reduz tempo de busca de 3.5min para ~1.5min/trial
FINAL_EPOCHS  = 60  # treino completo só uma vez, com os melhores params


def _make_loaders(X_train, y_train, X_val, y_val, batch_size=512):
    """Cria os DataLoaders uma única vez para reutilizar em todos os trials."""
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def _run_epochs(model, train_loader, val_loader, criterion, optimizer,
                n_epochs, trial=None):
    """
    Treina por n_epochs e retorna (melhor_accuracy, melhor_state_dict).
    Se `trial` for fornecido, ativa o pruning do Optuna.
    """
    best_accuracy = 0.0
    best_state    = None

    for epoch in range(n_epochs):
        # ── Treino ────────────────────────────────────────────────────────────
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        # ── Validação ─────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for batch_X_val, batch_y_val in val_loader:
                preds   = torch.argmax(model(batch_X_val), dim=1)
                correct += (preds == batch_y_val).sum().item()
                total   += batch_y_val.size(0)

        accuracy = correct / total

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Guarda cópia dos pesos do melhor epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # ── Pruning (apenas durante a busca Optuna) ───────────────────────────
        if trial is not None:
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_accuracy, best_state


def objective(trial, X_train, y_train, X_val, y_val,
              device, class_weights, train_loader, val_loader):
    hidden_size = trial.suggest_int('hidden', 32, 64)
    num_layers  = trial.suggest_int('layers', 1, 2)
    dropout     = trial.suggest_float('dropout', 0.1, 0.4)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = MarketLSTM(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc, _ = _run_epochs(
        model, train_loader, val_loader,
        criterion, optimizer,
        n_epochs=SEARCH_EPOCHS,   # ← busca rápida
        trial=trial
    )
    return best_acc


def train():
    device = torch.device("cpu")
    logger.info(f"🖥️  Dispositivo : {device}")
    logger.info(f"⚡  Threads      : {torch.get_num_threads()}")

    # ── Dados ─────────────────────────────────────────────────────────────────
    df = pd.read_csv("data/history.csv")
    logger.info("🧠 Memória: 180 ticks | 🎯 Alvo: 60 ticks (1 Minuto)")

    X, y, scaler = prepare_features(df, seq_len=180, future_steps=60)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    split    = int(0.8 * len(X_tensor))
    X_train, X_val = X_tensor[:split], X_tensor[split:]
    y_train, y_val = y_tensor[:split], y_tensor[split:]

    class_counts  = np.bincount(y)
    class_weights = len(y) / (len(class_counts) * class_counts)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(f"⚖️  Pesos de classes: {class_weights}")

    # ── Loaders criados UMA VEZ e compartilhados em todos os trials ───────────
    train_loader, val_loader = _make_loaders(X_train, y_train, X_val, y_val)

    # ── Busca de hiperparâmetros ───────────────────────────────────────────────
    logger.info(f"🔍 Optuna: {20} trials × {SEARCH_EPOCHS} épocas cada…")
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=3, interval_steps=1
    )
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_val, y_val,
            device, weights_tensor, train_loader, val_loader
        ),
        n_trials=20,
        show_progress_bar=True
    )

    best_params = study.best_params
    logger.info(f"🏆 Melhores parâmetros: {best_params}")
    logger.info(f"📊 Melhor acurácia na busca: {study.best_value:.4f}")

    # ── Treino final com os melhores params e mais épocas ─────────────────────
    #  BUG CORRIGIDO: antes o modelo era criado mas NUNCA treinado antes de salvar
    logger.info(f"🚀 Treino final: {FINAL_EPOCHS} épocas com os melhores parâmetros…")
    final_model = MarketLSTM(
        input_size=X_train.shape[2],
        hidden_size=best_params['hidden'],
        num_layers=best_params['layers'],
        dropout=best_params['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])

    final_acc, best_state = _run_epochs(
        final_model, train_loader, val_loader,
        criterion, optimizer,
        n_epochs=FINAL_EPOCHS,
        trial=None  # sem pruning no treino final
    )
    logger.info(f"✅ Melhor acurácia no treino final: {final_acc:.4f}")

    # Restaura os pesos do melhor epoch (não necessariamente o último)
    final_model.load_state_dict(best_state)

    # ── Persistência ──────────────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    torch.save(final_model.state_dict(), 'models/best_lstm.pth')

    # BUG CORRIGIDO: scaler nunca era salvo → inferência usava escala errada
    joblib.dump(scaler, 'models/scaler.pkl')

    logger.info("💾 Modelo  → models/best_lstm.pth")
    logger.info("💾 Scaler  → models/scaler.pkl")
    logger.info("🎯 Pronto para produção!")


if __name__ == "__main__":
    train()