import sys
import os

# Garante que o Python reconheça o diretório raiz para importar o pacote 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from tqdm import tqdm
from src.backtest.engine import WalkForwardEngine, MonteCarloAnalyzer
from loguru import logger


def simple_strategy(train_df):
    class DummyModel:
        def predict(self, row):
            return 1 if row["close"] > row["open"] else -1
    return DummyModel()


def run_backtest():
    csv_path = "data/history.csv"
    
    if not os.path.exists(csv_path) or not pd.api.types.is_numeric_dtype(pd.read_csv(csv_path)["price"]):
        logger.error("❌ Arquivo de histórico inválido ou não encontrado.")
        return
        
    df = pd.read_csv(csv_path)
    df["close"] = df["price"]
    df["open"] = df["price"].shift(1).fillna(df["price"])
    df.dropna(inplace=True)
    
    logger.info("🔄 Executando Walk-Forward Backtest...")
    wf = WalkForwardEngine(df, window=800, step=100, stake=10, fee=0.01)
    equity_curve = wf.run(simple_strategy)
    
    logger.info("🎲 Executando Monte Carlo (10k simulações)...")
    mc = MonteCarloAnalyzer(equity_curve, sims=10000)
    metrics = mc.analyze()
    
    logger.info("📈 Métricas Finais:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
        
    # Salvar curva de equity
    pd.Series(equity_curve).to_csv("data/equity_curve.csv", index=False)
    logger.info("✅ Resultados salvos em data/")


if __name__ == "__main__":
    run_backtest()