"""
Script de diagnóstico — testa authorize isolado do mesmo jeito que o deriv_client faz.
Salve em: test_ws.py (raiz do projeto)
"""
import asyncio
import json
import websockets


async def main() -> None:
    from core.settings import get_settings
    s = get_settings()

    url = f"{s.deriv_websocket_url}?app_id={s.deriv_app_id}"
    print(f"Conectando: {url}")

    async with websockets.connect(
        url,
        ping_interval=30,
        ping_timeout=10,
        close_timeout=5,
        max_size=2**20,
    ) as ws:
        print("WebSocket aberto!")

        # Authorize
        req_id = 1
        await ws.send(json.dumps({"authorize": s.deriv_api_token, "req_id": req_id}))
        print("Authorize enviado, aguardando resposta...")

        resp_raw = await asyncio.wait_for(ws.recv(), timeout=15)
        resp = json.loads(resp_raw)

        if resp.get("error"):
            print(f"ERRO na autorizacao: {resp['error']}")
            return

        acct = resp.get("authorize", {})
        print(f"Autorizado! Login: {acct.get('loginid')} | Balance: {acct.get('balance')} {acct.get('currency')}")

        # Ping
        await ws.send(json.dumps({"ping": 1, "req_id": 2}))
        pong = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        print(f"Ping OK: {pong}")

        # Subscribe ticks R_50
        await ws.send(json.dumps({"ticks": "R_50", "subscribe": 1, "req_id": 3}))
        print("Aguardando 3 ticks de R_50...")
        for i in range(3):
            tick_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            tick = json.loads(tick_raw)
            if "tick" in tick:
                t = tick["tick"]
                print(f"  Tick {i+1}: {t.get('symbol')} = {t.get('quote')} @ {t.get('epoch')}")
            else:
                print(f"  Msg {i+1}: {tick_raw[:100]}")

        print("\nTudo OK! Conexao e subscricao funcionando.")


if __name__ == "__main__":
    asyncio.run(main())