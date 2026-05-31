import sys
import os

# Garante que o Python reconheça o diretório raiz para importar o pacote 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import optuna
from loguru import logger

from src.models.lstm_predictor import MarketLSTM, prepare_features


def load_data(path: str = "data/history.csv"):
    df = pd.read_csv(path)
    df["price"] = df["price"].astype(float)
    return df["price"].values


def compute_class_weights(y_tensor, device):
    y_np = y_tensor.cpu().numpy()
    class_counts = np.bincount(y_np)
    total = len(y_np)
    weights = total / (len(class_counts) * (class_counts + 1e-5))
    return torch.tensor(weights, dtype=torch.float32).to(device)


def objective(trial, X_train, y_train, X_val, y_val, device, class_weights):
    hidden = trial.suggest_int("hidden", 32, 128)
    layers = trial.suggest_int("layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    
    model = MarketLSTM(input_dim=5, hidden=hidden, layers=layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    best_acc = 0.0
    
    for epoch in range(15):
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            acc = (torch.argmax(val_pred, dim=1) == y_val).float().mean().item()
            best_acc = max(best_acc, acc)
            
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return best_acc


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Processamento alocado no dispositivo: {device}")
    
    prices = load_data()
    
    # Configuração explícita para o aprendizado de padrões de 1 Minuto
    seq_len = 180  # Observar os últimos 3 minutos de histórico
    horizon = 60   # Prever o preço em 60 ticks (aproximadamente 1 minuto no ativo R_100)
    
    logger.info(f"🧠 Memória configurada para {seq_len} ticks de histórico.")
    logger.info(f"🎯 Alinhando o alvo da predição para {horizon} ticks (1 Minuto) no futuro...")
    
    X, y = prepare_features(
        [{"quote": p} for p in prices], 
        seq_len=seq_len, 
        forecast_horizon=horizon
    )
    
    if X is None: 
        logger.error("❌ Dados insuficientes para treino. Execute download_history.py primeiro.")
        return
        
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx].to(device), X[split_idx:].to(device)
    y_train, y_val = y[:split_idx].to(device), y[split_idx:].to(device)
    
    class_weights = compute_class_weights(y_train, device)
    logger.info(f"⚖️ Pesos de balanceamento ajustados: {class_weights.cpu().numpy()}")
        
    logger.info("🔍 Otimizando hiperparâmetros da LSTM para 1 minuto...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, device, class_weights), n_trials=25)
    
    logger.info(f"✅ Melhor acurácia de validação: {study.best_value:.4f}")
    
    best_model = MarketLSTM(
        input_dim=5, 
        hidden=study.best_params["hidden"], 
        layers=study.best_params["layers"], 
        dropout=study.best_params["dropout"]
    ).to(device)
    
    os.makedirs("models", exist_ok=True)
    torch.save(best_model.state_dict(), "models/best_lstm.pth")
    logger.info("💾 Modelo treinado para Operações de 1 Minuto salvo com sucesso!")


if __name__ == "__main__":
    train()