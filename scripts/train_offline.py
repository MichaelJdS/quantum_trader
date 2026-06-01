"""
Treino offline inicial do LSTM com dados históricos do banco.

Uso:
    python scripts/train_offline.py --symbol R_50 --granularity 60 --candles 2000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main(symbol: str, granularity: int, candles: int) -> None:
    from core.bootstrap import bootstrap
    await bootstrap()

    from infra.db.database import get_session
    from infra.db.repository import CandleRepository
    from ml.feature_engineer import FeatureEngineer
    from ml.mlops import MLOpsManager
    from ml.models.lstm_model import LSTMTrainer

    # 1. Carrega candles do banco.
    async with get_session() as db:
        repo = CandleRepository(db)
        candle_rows = await repo.get_recent(symbol, granularity, limit=candles)

    if len(candle_rows) < 200:
        print(f"Candles insuficientes: {len(candle_rows)}. Mínimo: 200.")
        return

    import pandas as pd
    df = pd.DataFrame([{
        "open": c.open, "high": c.high, "low": c.low,
        "close": c.close, "epoch": c.epoch,
    } for c in candle_rows])

    # 2. Feature engineering.
    fe = FeatureEngineer()
    feat_df = fe.compute(df)
    print(f"Features calculadas: {len(feat_df)} amostras, {len(feat_df.columns)} colunas.")

    # 3. MLOps: inicia run.
    mlops = MLOpsManager()

    # 4. Otimiza hiperparâmetros (opcional — descomente para usar).
    trainer = LSTMTrainer(symbol=symbol, max_epochs=30, patience=7)
    X, y = trainer.prepare_sequences(feat_df)

    run_id = mlops.start_run(
        run_name=f"lstm_{symbol}_offline",
        params={"symbol": symbol, "seq_len": trainer.seq_len,
                "hidden_size": trainer.hidden_size, "samples": len(X)},
    )

    # 5. Treina.
    history = trainer.fit(X, y)

    # 6. Loga métricas.
    best_val_loss = min(history["val_loss"])
    best_val_acc = max(history["val_acc"])
    mlops.log_metrics({
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["val_loss"]),
    })
    mlops.end_run()

    print(f"\n✅ Treino concluído!")
    print(f"   Val Loss: {best_val_loss:.4f}")
    print(f"   Val Acc:  {best_val_acc:.4f}")
    print(f"   Modelo salvo em: models_store/lstm_{symbol}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino offline do LSTM.")
    parser.add_argument("--symbol", default="R_50")
    parser.add_argument("--granularity", type=int, default=60)
    parser.add_argument("--candles", type=int, default=2000)
    args = parser.parse_args()
    asyncio.run(main(args.symbol, args.granularity, args.candles))