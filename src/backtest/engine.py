import numpy as np
import pandas as pd
from tqdm import tqdm

class WalkForwardEngine:
    def __init__(self, df: pd.DataFrame, window=500, step=100, stake=10, fee=0.01):
        self.df = df
        self.window = window
        self.step = step
        self.stake = stake
        self.fee = fee

    def run(self, signal_func):
        equity = [1000.0]
        for start in tqdm(range(0, len(self.df) - self.window, self.step), desc="Walk-Forward"):
            train = self.df.iloc[start:start+self.window]
            test = self.df.iloc[start+self.window:start+self.window+self.step]
            model = signal_func(train)
            for _, row in test.iterrows():
                sig = model.predict(row)
                ret = (row["close"] / row["open"] - 1) if sig == 1 else (row["open"] / row["close"] - 1) if sig == -1 else 0
                equity.append(equity[-1] + self.stake * ret * (1 - self.fee))
        return equity

class MonteCarloAnalyzer:
    def __init__(self, equity_curve: list[float], sims=10000):
        self.ec = np.array(equity_curve)
        self.sims = sims

    def analyze(self):
        returns = np.diff(self.ec) / self.ec[:-1]
        paths = np.cumprod(1 + np.random.choice(returns, (self.sims, len(returns))), axis=1)
        dd = np.max(np.maximum.accumulate(paths, axis=1) - paths, axis=1) / np.maximum.accumulate(paths, axis=1)
        return {
            "mean_max_dd": np.mean(dd), "p95_max_dd": np.percentile(dd, 95),
            "prob_ruin": np.mean(dd > 0.25), "sharpe_approx": np.mean(returns)/np.std(returns)*np.sqrt(252)
        }