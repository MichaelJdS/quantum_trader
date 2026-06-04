"""
download_ticks_r100.py
======================
Baixa até 500.000 ticks históricos de R_100 da API Deriv e salva no banco
de dados SQLite do projeto (quantum_trader.db).

Uso:
    python download_ticks_r100.py                # baixa 500.000 ticks
    python download_ticks_r100.py --count 100000 # baixa X ticks
    python download_ticks_r100.py --check        # apenas mostra status do DB

O script trabalha de forma incremental: se o banco já tem ticks do R_100,
continua de onde parou (sem duplicatas).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

# ── Config ────────────────────────────────────────────────────────────────────

APP_ID       = "111625"
API_TOKEN    = "ASnUzqqVuOaRIEd"
WS_URL       = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
SYMBOL       = "R_100"
BATCH_SIZE   = 5000   # máximo por chamada da API Deriv
TARGET_TICKS = 500_000
DB_PATH      = Path(__file__).parent / "quantum_trader.db"


# ── Banco de dados ────────────────────────────────────────────────────────────

def get_db_stats(conn: sqlite3.Connection) -> tuple[int, int | None, int | None]:
    """Retorna (total, epoch_min, epoch_max) para R_100 no banco."""
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*), MIN(epoch), MAX(epoch) FROM ticks WHERE symbol = ?",
        (SYMBOL,),
    )
    row = cur.fetchone()
    return row[0] or 0, row[1], row[2]


def insert_ticks(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    """Insere ticks ignorando duplicatas. Retorna quantos foram inseridos."""
    cur = conn.cursor()
    cur.executemany(
        """INSERT OR IGNORE INTO ticks (symbol, epoch, price, pip_size)
           VALUES (?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return cur.rowcount


# ── Download ──────────────────────────────────────────────────────────────────

async def fetch_ticks_batch(ws, end_epoch: int, req_id: int) -> list[dict]:
    """Baixa 1 lote de até 5.000 ticks encerrado em end_epoch."""
    req = {
        "ticks_history": SYMBOL,
        "end":           str(end_epoch),
        "count":         BATCH_SIZE,
        "style":         "ticks",
        "req_id":        req_id,
    }
    await ws.send(json.dumps(req))

    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(raw)
        if msg.get("req_id") != req_id:
            continue
        if msg.get("error"):
            raise RuntimeError(f"API error: {msg['error']['message']}")
        history = msg.get("history", {})
        times  = history.get("times",  [])
        prices = history.get("prices", [])
        if not times:
            return []
        return [
            {"epoch": int(t), "quote": float(p)}
            for t, p in zip(times, prices)
        ]


async def download(target: int, conn: sqlite3.Connection) -> None:
    total_before, _, epoch_max = get_db_stats(conn)
    print(f"\n📊 R_100 no banco: {total_before:,} ticks")

    need = target - total_before
    if need <= 0:
        print(f"✅ Já tem {total_before:,} ticks — nada a baixar.")
        return

    print(f"📥 Precisa baixar: {need:,} ticks")
    print(f"📦 Lotes de {BATCH_SIZE} ticks  |  ~{(need // BATCH_SIZE) + 1} chamadas à API\n")

    async with websockets.connect(
        WS_URL, ping_interval=30, ping_timeout=10, max_size=2**21
    ) as ws:
        # Autoriza
        await ws.send(json.dumps({"authorize": API_TOKEN, "req_id": 0}))
        auth = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        if auth.get("error"):
            print(f"❌ Erro de autorização: {auth['error']['message']}")
            return
        print(f"✅ Autorizado: {auth['authorize'].get('loginid')}\n")

        # Ponto de partida — continua de onde parou
        end_epoch = epoch_max - 1 if epoch_max else int(time.time())

        downloaded = 0
        req_id = 1
        t0 = time.time()

        while downloaded < need:
            try:
                ticks = await fetch_ticks_batch(ws, end_epoch, req_id)
            except Exception as exc:
                print(f"\n⚠️  Erro no lote {req_id}: {exc}. Tentando novamente em 3s...")
                await asyncio.sleep(3)
                continue

            if not ticks:
                print("\n⚠️  API retornou lote vazio — fim dos dados históricos.")
                break

            rows = [(SYMBOL, t["epoch"], t["quote"], None) for t in ticks]
            inserted = insert_ticks(conn, rows)
            downloaded += inserted

            oldest_epoch = min(t["epoch"] for t in ticks)
            end_epoch = oldest_epoch - 1  # próximo lote vai ainda mais atrás

            elapsed = time.time() - t0
            rate = downloaded / elapsed if elapsed > 0 else 0
            eta = (need - downloaded) / rate if rate > 0 else 0

            oldest_dt = datetime.fromtimestamp(oldest_epoch, tz=timezone.utc)
            print(
                f"  Lote {req_id:3d} | +{inserted:5,} ticks | "
                f"Total: {total_before + downloaded:,} | "
                f"Mais antigo: {oldest_dt.strftime('%Y-%m-%d')} | "
                f"ETA: {int(eta//60)}m{int(eta%60):02d}s",
                flush=True,
            )

            req_id += 1

            # Pausa curta para não sobrecarregar a API
            await asyncio.sleep(0.3)

    total_after, epoch_min, epoch_max2 = get_db_stats(conn)
    dt_min = datetime.fromtimestamp(epoch_min, tz=timezone.utc) if epoch_min else None
    dt_max = datetime.fromtimestamp(epoch_max2, tz=timezone.utc) if epoch_max2 else None

    print(f"\n{'='*60}")
    print(f"✅ Download concluído!")
    print(f"   R_100 no banco: {total_after:,} ticks")
    if dt_min and dt_max:
        print(f"   De: {dt_min.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"   Até: {dt_max.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"   Tempo total: {int((time.time()-t0)//60)}m{int((time.time()-t0)%60):02d}s")
    print(f"{'='*60}\n")


# ── Check ─────────────────────────────────────────────────────────────────────

def show_db_status(conn: sqlite3.Connection) -> None:
    """Mostra status atual do banco para todos os símbolos."""
    cur = conn.cursor()
    cur.execute(
        """SELECT symbol, COUNT(*) as cnt,
                  MIN(epoch) as oldest, MAX(epoch) as newest
           FROM ticks GROUP BY symbol ORDER BY symbol"""
    )
    rows = cur.fetchall()
    if not rows:
        print("❌ Nenhum tick encontrado no banco.")
        return
    print("\n📊 Status do banco de ticks:\n")
    print(f"  {'Símbolo':<12} {'Ticks':>12}  {'Mais antigo':<22} {'Mais recente'}")
    print(f"  {'-'*12} {'-'*12}  {'-'*22} {'-'*22}")
    for sym, cnt, oldest, newest in rows:
        d1 = datetime.fromtimestamp(oldest, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC') if oldest else '—'
        d2 = datetime.fromtimestamp(newest, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC') if newest else '—'
        print(f"  {sym:<12} {cnt:>12,}  {d1:<22} {d2}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa ticks históricos do R_100 da Deriv")
    parser.add_argument("--count",  type=int, default=TARGET_TICKS, help="Total de ticks a ter no banco")
    parser.add_argument("--check",  action="store_true", help="Apenas mostra status do banco")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    if args.check:
        show_db_status(conn)
        conn.close()
        return

    print("=" * 60)
    print(f"🚀 Downloader de Ticks  |  Símbolo: {SYMBOL}")
    print(f"   Target: {args.count:,} ticks  |  DB: {DB_PATH.name}")
    print("=" * 60)

    show_db_status(conn)

    asyncio.run(download(args.count, conn))
    conn.close()


if __name__ == "__main__":
    main()
