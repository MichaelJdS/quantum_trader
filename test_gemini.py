import asyncio
import os
import sys

# Load env vars
with open('.env') as f:
    for line in f:
        if '=' in line and not line.startswith('#'):
            k, v = line.strip().split('=', 1)
            os.environ[k] = v.strip('\"\'')

from ml.gemini_advisor import GeminiAdvisor, StrategyContext

def main():
    advisor = GeminiAdvisor()
    ctx = StrategyContext(
        symbol='R_50',
        candles_summary='epoch,open,high,low,close,volume\n1,1,2,1,2,10',
        indicators={'rsi': 50},
        session_state_summary={'balance': 100},
        available_strategies=['ema_rsi_macd', 'bollinger'],
        last_strategy_results={'ema_rsi_macd': 0.5}
    )
    try:
        res = advisor._sync_consult(ctx)
        print('FINAL PARSED ADVICE:', res)
    except Exception as e:
        import traceback
        traceback.print_exc()

main()
