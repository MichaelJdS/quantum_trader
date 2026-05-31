from src.config import settings
from loguru import logger

class RiskManager:
    def __init__(self):
        self.start_balance = 0.0
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.drawdown = 0.0
        self.circuit_breaker = False
        self.daily_pnl = 0.0

    def update(self, balance: float) -> dict:
        if self.start_balance == 0: self.start_balance = balance
        self.current_balance = balance
        self.peak_balance = max(self.peak_balance, balance)
        self.drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance * 100) if self.peak_balance else 0
        self.circuit_breaker = self.drawdown >= settings.MAX_DRAWDOWN_PCT
        return {"balance": balance, "drawdown": self.drawdown, "circuit": self.circuit_breaker}

    def get_stake(self) -> float:
        if self.circuit_breaker: return 0.0
        base = self.current_balance * settings.MAX_POSITION_PCT / 100
        return max(1.0, min(base, 50.0))

    def log_trade(self, pnl: float):
        self.daily_pnl += pnl
        if self.daily_pnl < -self.start_balance * 0.05:
            self.circuit_breaker = True
            logger.warning("🛑 Daily Loss Limit atingido. Circuit Breaker ativado.")