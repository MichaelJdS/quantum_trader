"""
Quantum Trader v2 — Entrypoint principal.
NiceGUI 3.x  |  Compatível Windows/Linux
"""
from __future__ import annotations
import sys
import os

# ── Garante raiz no sys.path ─────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Imports ──────────────────────────────────────────────────
from nicegui import ui
from config.settings import settings

# Registra todas as páginas do dashboard
import dashboard.app  # noqa: F401  (efeito colateral: @ui.page decorators)

# ── Boot ─────────────────────────────────────────────────────
# NiceGUI 3.x: ui.run() deve ser chamado no escopo global,
# SEM proteção "if __name__" — o multiprocessing do NiceGUI
# reimporta o módulo com __name__ != "__main__" nos workers.
ui.run(
    host=settings.dashboard_host,
    port=settings.dashboard_port,
    title="⚡ Quantum Trader",
    dark=True,
    reload=False,
    favicon="⚡",
    show=True,          # abre o browser automaticamente
)
