import sys
import os

# Garante que o Python reconheça o diretório raiz para importar o pacote 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import json
import csv
import websockets
from loguru import logger
from src.config import settings


async def fetch_ticks_history(symbol: str = "R_100", total_count: int = 50000, output_path: str = "data/history.csv"):
    os.makedirs("data", exist_ok=True)
    url = f"wss://ws.derivws.com/websockets/v3?app_id={settings.DERIV_APP_ID}"
    
    # O uso de um dicionário garante 100% que não haverá timestamps repetidos
    unique_ticks = {}
    
    # A Deriv permite no máximo 5000 ticks por chamada no histórico
    batch_size = 5000
    current_end = "latest"
    
    logger.info(f"📥 Iniciando o download de {total_count} registros de {symbol} em lotes (de trás para frente)...")
    
    async with websockets.connect(url) as ws:
        while len(unique_ticks) < total_count:
            remaining = total_count - len(unique_ticks)
            # Busca o lote máximo ou o que falta para completar
            fetch_count = min(batch_size, remaining)
            
            req = {
                "ticks_history": symbol,
                "count": fetch_count,
                "end": current_end,
                "style": "ticks"
            }
            
            await ws.send(json.dumps(req))
            resp = json.loads(await ws.recv())
            
            if "error" in resp:
                logger.error(f"❌ Erro Deriv: {resp['error']['message']}")
                break
                
            ticks = resp.get("history", {}).get("prices", [])
            times = resp.get("history", {}).get("times", [])
            
            if not ticks or not times:
                logger.warning("⚠️ Nenhum dado retornado neste lote. Fim do histórico alcançado ou rate limit.")
                break
            
            # Adiciona ao dicionário para remover qualquer duplicidade de timestamp
            added_in_batch = 0
            for t_time, t_price in zip(times, ticks):
                if t_time not in unique_ticks:
                    unique_ticks[t_time] = t_price
                    added_in_batch += 1
            
            logger.info(
                f"🔄 Lote processado: Recebidos {len(ticks)} | Novos: {added_in_batch} | "
                f"Total Únicos: {len(unique_ticks)}/{total_count}"
            )
            
            # O próximo lote deve terminar 1 segundo antes do timestamp mais antigo deste lote
            oldest_timestamp = times[0]
            current_end = str(oldest_timestamp - 1)
            
            # Se não adicionamos nenhum tick novo, chegamos ao limite de dados antigos do ativo
            if added_in_batch == 0:
                logger.warning("⚠️ Nenhum dado novo neste lote. Interrompendo para evitar loop infinito.")
                break
            
            # Respeita o limite de requisições da corretora (Evita block de IP/Token)
            await asyncio.sleep(0.5)

    if not unique_ticks:
        logger.error("❌ Nenhum dado baixado.")
        return

    # Converte o dicionário em lista e ordena cronologicamente (do mais antigo para o mais recente)
    sorted_ticks = sorted(unique_ticks.items(), key=lambda item: item[0])
    
    # Se a corretora enviar alguns ticks a mais no último lote, cortamos para ter exatamente o total_count
    sorted_ticks = sorted_ticks[-total_count:]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "price"])
        writer.writerows(sorted_ticks)
        
    logger.info(f"✅ Download concluído: {len(sorted_ticks)} registros únicos salvos em {output_path}")


if __name__ == "__main__":
    asyncio.run(fetch_ticks_history())