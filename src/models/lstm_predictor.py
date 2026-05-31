import torch
import torch.nn as nn
import numpy as np

class MarketLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.softmax(self.head(out[:, -1, :]), dim=1)

def prepare_features(ticks: list[dict], seq_len=30):
    quotes = np.array([t["quote"] for t in ticks])
    if len(quotes) < seq_len + 1: return None, None
    
    diffs = np.diff(quotes)
    ema10 = np.convolve(quotes, np.ones(10)/10, 'valid')
    vol = np.lib.stride_tricks.sliding_window_view(np.diff(quotes), 5).std(axis=1)
    returns = quotes[1:] / quotes[:-1] - 1
    signs = np.sign(diffs)
    
    X, y = [], []
    for i in range(len(quotes) - seq_len):
        feat = np.column_stack([diffs[i:i+seq_len], ema10[i:i+seq_len], vol[i:i+seq_len], returns[i:i+seq_len], signs[i:i+seq_len]])
        X.append(feat)
        y.append(1 if diffs[i+seq_len-1] > 0 else 0)
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)