import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from core.settings import get_settings
from infra.deriv_client import DerivClient
from ml.feature_engineer import FeatureEngineer

async def main():
    s = get_settings()
    client = DerivClient(app_id=s.deriv_app_id, api_token=s.deriv_api_token, url=s.deriv_websocket_url)
    await client.connect()
    fe = FeatureEngineer()

    for sym in ["R_50", "R_75", "R_100"]:
        candles = await client.get_candles(symbol=sym, granularity=60, count=200)
        if not candles:
            print(sym + ": sem candles!")
            continue

        df = pd.DataFrame([{"epoch": c["epoch"], "open": float(c["open"]),
                             "high": float(c["high"]), "low": float(c["low"]),
                             "close": float(c["close"])} for c in candles])
        df.sort_values("epoch", inplace=True)
        df.reset_index(drop=True, inplace=True)
        feat = fe.compute(df)
        if feat.empty:
            print(sym + ": feat_df vazio!")
            continue

        last = feat.iloc[-1]
        rsi  = float(last.get("rsi_14", 50))
        macd = float(last.get("macd_hist", 0))
        adx  = float(last.get("adx", 0))
        sq   = int(last.get("squeeze", 0))
        e9   = float(last.get("ema_9", 0))
        e21  = float(last.get("ema_21", 0))
        e50  = float(last.get("ema_50", 0))
        close= float(last.get("close", 0))
        bb_lo= float(last.get("bb_lower", 0))
        bb_hi= float(last.get("bb_upper", 0))
        bull_c = int(last.get("is_bullish", 0))

        tb = e9 > e21 > e50
        te = e9 < e21 < e50

        ok = lambda v: "OK" if v else "FAIL"
        print("\n== " + sym + " ==")
        print("  RSI=" + str(round(rsi,1)) + "  MACD=" + str(round(macd,6)) + "  ADX=" + str(round(adx,1)) + "  squeeze=" + str(sq))
        print("  EMA9=" + str(round(e9,5)) + "  EMA21=" + str(round(e21,5)) + "  EMA50=" + str(round(e50,5)))
        print("  close=" + str(round(close,5)) + "  BB_lo=" + str(round(bb_lo,5)) + "  BB_hi=" + str(round(bb_hi,5)))
        print("  SQUEEZE filter: " + ("BLOQUEANDO TUDO!" if sq==1 else "OK (nao bloqueia)"))
        print("  EMA CALL: squeeze=" + ok(sq==0) + " trend=" + ok(tb) + " rsi48-68=" + ok(48<rsi<68) + " macd+=" + ok(macd>0) + " adx>18=" + ok(adx>18))
        print("  EMA PUT:  squeeze=" + ok(sq==0) + " trend=" + ok(te) + " rsi32-52=" + ok(32<rsi<52) + " macd-=" + ok(macd<0) + " adx>18=" + ok(adx>18))
        print("  BOLL CALL: squeeze=" + ok(sq==0) + " c<bb_lo=" + ok(close<bb_lo) + " rsi<42=" + ok(rsi<42) + " bull=" + ok(bull_c==1))
        print("  BOLL PUT:  squeeze=" + ok(sq==0) + " c>bb_hi=" + ok(close>bb_hi) + " rsi>62=" + ok(rsi>62))

    await client.disconnect()
    print("\nDiagnostico concluido.")

asyncio.run(main())
