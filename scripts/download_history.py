import asyncio
import json
import csv
import os
import time
from datetime import datetime
import websockets
from loguru import logger
from src.config import settings

async def fetch_ticks_history(symbol: str = "R_100", count: int = 5000, output_path: str = "data/history.csv"):
    os.makedirs("data", exist_ok=True)
    url = f"wss://ws.derivws.com/websockets/v3?app_id={settings.DERIV_APP_ID}"
    all_ticks = []
    batch_size = 1000
    start = int(time.time()) - (count * 2)  # Estimativa de 2s por tick

    logger.info(f"📥 Baixando {count} ticks de {symbol}...")
    
    async with websockets.connect(url) as ws:
        for i in range(0, count, batch_size):
            req = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": min(batch_size, count - i),
                "end": "latest",
                "start": start,
                "style": "ticks"
            }
            await ws.send(json.dumps(req))
            resp = json.loads(await ws.recv())
            
            if "error" in resp:
                logger.error(f"❌ Erro Deriv: {resp['error']['message']}")
                break
                
            ticks = resp.get("history", {}).get("prices", [])
            times = resp.get("history", {}).get("times", [])
            
            if not ticks:
                logger.warning("⚠️ Sem dados retornados. Aguardando rate limit...")
                await asyncio.sleep(2)
                continue
                
            all_ticks.extend(zip(times, ticks))
            start = times[-1] + 1
            await asyncio.sleep(0.5)  # Respeita rate limit da Deriv

    if not all_ticks:
        logger.error("❌ Nenhum dado baixado.")
        return

    all_ticks.sort(key=lambda x: x[0])
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "price"])
        writer.writerows(all_ticks)
        
    logger.info(f"✅ {len(all_ticks)} ticks salvos em {output_path}")

if __name__ == "__main__":
    asyncio.run(fetch_ticks_history())