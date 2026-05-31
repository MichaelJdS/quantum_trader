import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class MarketLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        # O PyTorch emite um aviso se o dropout for definido em LSTM de 1 camada.
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
        # Extrai apenas o último passo temporal da sequência
        return torch.softmax(self.head(out[:, -1, :]), dim=1)

def prepare_features(ticks: list[dict], seq_len=30):
    quotes = [t["quote"] for t in ticks]
    # Necessita de uma margem segura para gerar a Média Móvel Exponencial (EMA)
    if len(quotes) < seq_len + 15: 
        return None, None
        
    # Modelagem de dados segura usando pandas para alinhamento perfeito de índices
    df = pd.DataFrame({"close": quotes})
    df["diffs"] = df["close"].diff()
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["vol"] = df["diffs"].rolling(window=5).std()
    df["returns"] = df["close"].pct_change()
    df["signs"] = np.sign(df["diffs"])
    
    # Desloca o target em -1 para que a predição seja o movimento futuro
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    
    # Remove as linhas com NaN resultantes dos cálculos de features
    df.dropna(inplace=True)
    
    features = df[["diffs", "ema10", "vol", "returns", "signs"]].values
    targets = df["target"].values
    
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(targets[i+seq_len-1])
        
    if not X:
        return None, None
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)