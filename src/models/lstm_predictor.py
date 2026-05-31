import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class MarketLSTM(nn.Module):
    """
    Arquitetura de 'Super IA': LSTM associada a um mecanismo de Self-Attention.
    O Attention permite que a rede identifique quais micro-movimentos dentro
    dos 180 ticks passados são cruciais para o rompimento no futuro.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(MarketLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # Camada Recorrente (Memória Temporal)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Mecanismo de Atenção (Pesos Dinâmicos)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Classificador Focado
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 2)  # Target: CALL (1) ou PUT (0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        
        # Calculando os pesos de importância para cada tick do passado
        attn_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Multiplicando a saída da LSTM pelos pesos de atenção
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_size)
        
        # Passagem final pela rede densa
        out = self.fc1(context_vector)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def prepare_features(df, seq_len=180, future_steps=60):
    """
    Motor de Engenharia de Features.
    Injeta lógica de Momentum, Volatilidade e Tendência nos Tensores.
    Agora com detecção dinâmica Universal (suporta Ticks e Candles).
    """
    df = df.copy()
    
    # 0. Identificação Dinâmica da Coluna Alvo
    # Padroniza todas as colunas para minúsculo removendo espaços para evitar erros de formatação no CSV
    df.columns = [str(col).lower().strip() for col in df.columns]
    
    if 'close' in df.columns:
        target_col = 'close'
    elif 'price' in df.columns:
        target_col = 'price'
    elif 'quote' in df.columns:
        target_col = 'quote'
    else:
        # Fallback de segurança: Assume que a última coluna contém os valores numéricos de cotação
        target_col = df.columns[-1]
        
    close = df[target_col].astype(float)
    
    # 1. Log-Retornos (Garante que a IA leia variações percentuais puras)
    df['log_return'] = np.log(close / close.shift(1))
    
    # 2. RSI - Relative Strength Index (Momentum)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 3. MACD - Convergência e Divergência (Força da Tendência)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 4. Volatilidade e Distância de Bollinger
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    df['bb_upper_dist'] = (sma_20 + (std_20 * 2)) - close
    df['bb_lower_dist'] = close - (sma_20 - (std_20 * 2))
    df['volatility_30'] = df['log_return'].rolling(window=30).std()
    
    # 5. O Alvo (Target) -> 60 ticks (1 minuto) no futuro
    df['future_price'] = close.shift(-future_steps)
    df['target'] = (df['future_price'] > close).astype(int)
    
    # Limpando as pontas que ficaram sem dados
    df.dropna(inplace=True)
    
    # Selecionando colunas para a Rede Neural
    feature_cols = [
        'log_return', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 
        'bb_upper_dist', 'bb_lower_dist', 'volatility_30'
    ]
    
    data_values = df[feature_cols].values
    target_values = df['target'].values
    
    # Normalização Estatística (Z-Score)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i+seq_len])
        y.append(target_values[i+seq_len])
        
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler