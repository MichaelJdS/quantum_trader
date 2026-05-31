import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from loguru import logger
import os

from src.models.lstm_predictor import MarketLSTM, prepare_features

def load_data(path: str = "data/history.csv"):
    df = pd.read_csv(path)
    df["price"] = df["price"].astype(float)
    return df["price"].values

def build_dataloader(X, y, batch_size=32, shuffle=False):
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)

def objective(trial, X, y, device):
    hidden = trial.suggest_int("hidden", 32, 128)
    layers = trial.suggest_int("layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    model = MarketLSTM(input_dim=5, hidden=hidden, layers=layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(15):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            pred = model(X)
            acc = (torch.argmax(pred, dim=1) == y).float().mean().item()
        trial.report(acc, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        
    return acc

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prices = load_data()
    X, y = prepare_features([{"quote": p} for p in prices], seq_len=30)
    if X is None: 
        logger.error("❌ Dados insuficientes para treino. Execute download_history.py primeiro.")
        return
        
    logger.info("🔍 Otimizando hiperparâmetros com Optuna...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, X, y, device), n_trials=25)
    
    logger.info(f"✅ Melhor acurácia: {study.best_value:.4f}")
    logger.info(f"⚙️ Melhores params: {study.best_params}")
    
    best_model = MarketLSTM(**study.best_params).to(device)
    torch.save(best_model.state_dict(), "models/best_lstm.pth")
    logger.info("💾 Modelo salvo em models/best_lstm.pth")

if __name__ == "__main__":
    train()