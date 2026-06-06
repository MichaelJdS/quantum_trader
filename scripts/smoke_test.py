from __future__ import annotations

import asyncio

from core.bootstrap import bootstrap
from core.settings import get_settings
from infra.deriv_client import DerivClient
from infra.symbol_manager import SymbolManager
from ml.feature_engineer import FeatureEngineer


async def main() -> None:
    await bootstrap()
    settings = get_settings()

    client = DerivClient(dry_run=True)
    await client.connect()

    try:
        sm = SymbolManager(
            client=client,
            symbols=settings.default_symbols_list,
            granularity=settings.default_granularity,
        )
        await sm.initialize()

        fe = FeatureEngineer()

        for symbol in sm.ready_symbols:
            df = sm.get_candles_df(symbol)
            feat_df = fe.compute(df)

            print(
                {
                    "symbol": symbol,
                    "candles": len(df),
                    "features": len(feat_df),
                    "last_epoch": int(df.iloc[-1]["epoch"]) if not df.empty else None,
                    "feature_cols": list(feat_df.columns[-8:]) if not feat_df.empty else [],
                }
            )
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())