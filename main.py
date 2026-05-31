"""
Quantum Trader v2 — Entrypoint principal.
Inicia dashboard NiceGUI + métricas Prometheus.
"""
from nicegui import ui
from config.settings import settings
import dashboard.app  # registra páginas

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        title="⚡ Quantum Trader",
        dark=True,
        reload=False,
        favicon="⚡",
    )