import sqlite3
import os

db_path = "quantum_trader.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Lista tabelas
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("Tabelas:", tables)

for table in tables:
    try:
        cur.execute(f"SELECT COUNT(*) FROM [{table}]")
        total = cur.fetchone()[0]
        print(f"  {table}: {total:,} registros")
        try:
            cur.execute(f"SELECT symbol, COUNT(*) as cnt FROM [{table}] GROUP BY symbol ORDER BY cnt DESC")
            for sym, cnt in cur.fetchall():
                print(f"    {sym}: {cnt:,}")
        except Exception:
            pass
    except Exception as e:
        print(f"  {table}: {e}")

# Verifica range de datas dos ticks R_100
try:
    cur.execute("SELECT MIN(epoch), MAX(epoch), COUNT(*) FROM ticks WHERE symbol='R_100'")
    row = cur.fetchone()
    if row and row[0]:
        from datetime import datetime, timezone
        mn = datetime.fromtimestamp(row[0], tz=timezone.utc)
        mx = datetime.fromtimestamp(row[1], tz=timezone.utc)
        print(f"\nR_100 ticks: {row[2]:,}")
        print(f"  De: {mn.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Ate: {mx.strftime('%Y-%m-%d %H:%M:%S UTC')}")
except Exception as e:
    print(f"Erro R_100: {e}")

conn.close()
