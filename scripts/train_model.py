import os
import sys
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset

# Garante que o Python encontre o pacote 'src' para os imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.lstm_predictor import MarketLSTM, prepare_features

# Configuração de observabilidade do terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)

def objective(trial, X_train, y_train, X_val, y_val, device, class_weights):
    """
    Função objetivo para a otimização de hiperparâmetros via Optuna.
    """
    hidden_size = trial.suggest_int('hidden', 32, 128)
    num_layers = trial.suggest_int('layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = MarketLSTM(
        input_size=X_train.shape[2], 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        dropout=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(15):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_preds = torch.argmax(val_output, dim=1)
        accuracy = (val_preds == y_val).float().mean().item()

    return accuracy

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Processamento alocado no dispositivo: {device}")

    # 1. Carregamento e Preparação dos Dados
    df = pd.read_csv("data/history.csv")
    logger.info("🧠 Memória configurada para 180 ticks de histórico.")
    logger.info("🎯 Alinhando o alvo da predição para 60 ticks (1 Minuto) no futuro...")

    # CORREÇÃO APLICADA: Substituição de forecast_horizon por future_steps
    X, y, scaler = prepare_features(df, seq_len=180, future_steps=60)

    # 2. Conversão para Tensores do PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    # Divisão entre Treino (80%) e Validação (20%)
    split = int(0.8 * len(X_tensor))
    X_train, X_val = X_tensor[:split], X_tensor[split:]
    y_train, y_val = y_tensor[:split], y_tensor[split:]

    # 3. Cálculo de Pesos para Balanceamento de Classes (Prevenção de Colapso)
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    logger.info(f"⚖️ Pesos de balanceamento ajustados: {class_weights}")
    logger.info("🔍 Otimizando hiperparâmetros da LSTM para 1 minuto...")

    # 4. Execução do Otimizador (Optuna)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, device, class_weights_tensor), n_trials=25)

    best_params = study.best_params
    logger.info(f"🏆 Melhores hiperparâmetros encontrados: {best_params}")

    # 5. Salvar o Melhor Modelo
    best_model = MarketLSTM(
        input_size=X_train.shape[2],
        hidden_size=best_params['hidden'],
        num_layers=best_params['layers'],
        dropout=best_params['dropout']
    ).to(device)

    os.makedirs('models', exist_ok=True)
    torch.save(best_model.state_dict(), 'models/best_lstm.pth')
    logger.info("💾 Modelo treinado para Operações de 1 Minuto salvo com sucesso!")

if __name__ == "__main__":
    train()