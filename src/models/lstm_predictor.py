import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class MarketLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def prepare_features(ticks: list[dict], seq_len=180, forecast_horizon=60):
    quotes = [t["quote"] for t in ticks]
    
    if len(quotes) < seq_len + 100: 
        return None, None
        
    df = pd.DataFrame({"close": quotes})
    
    df["returns"] = df["close"].pct_change()
    df["diffs"] = df["close"].diff() / df["close"]
    
    # Média Móvel Exponencial adaptada para fluxo de 1 minuto (60 períodos)
    ema60 = df["close"].ewm(span=60, adjust=False).mean()
    df["ema_dist"] = (df["close"] - ema60) / df["close"]
    
    # Volatilidade baseada nos últimos 30 segundos
    df["vol"] = df["returns"].rolling(window=30).std()
    df["signs"] = np.sign(df["close"].diff())
    
    # Target travado em 60 ticks no futuro (Aproximadamente 1 minuto no ativo R_100)
    df["target"] = (df["close"].shift(-forecast_horizon) > df["close"]).astype(int)
    
    df.dropna(inplace=True)
    
    features_cols = ["returns", "diffs", "ema_dist", "vol", "signs"]
    for col in features_cols:
        if col != "signs":
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
    
    features = df[features_cols].values
    targets = df["target"].values
    
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(targets[i+seq_len-1])
        
    if not X:
        return None, None
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)