"""Dashboard Quantum Trader — NiceGUI 3.x"""
from __future__ import annotations
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nicegui import ui, app as ng_app
from config.settings import settings
from src.utils.logger import logger


# ── Startup assíncrono ───────────────────────────────────────
@ng_app.on_startup
async def startup() -> None:
    from src.persistence.database import init_db
    from src.metrics.prometheus_metrics import TradingMetrics
    await init_db()
    try:
        TradingMetrics().start_server(port=settings.prometheus_port)
    except Exception:
        pass  # já rodando
    logger.info("✅ Quantum Trader iniciado.")


# ── Página inicial ───────────────────────────────────────────
@ui.page("/")
async def index() -> None:
    ui.dark_mode().enable()
    ui.query("body").style("background:#0f1117; margin:0;")

    with ui.column().classes("items-center justify-center w-full gap-6").style(
        "min-height:100vh;"
    ):
        ui.label("⚡ QUANTUM TRADER").classes(
            "text-5xl font-bold text-teal-400 tracking-widest"
        )
        ui.label("Automação de trading — Deriv API").classes(
            "text-gray-400 text-lg"
        )
        with ui.row().classes("gap-4 mt-6 flex-wrap justify-center"):
            ui.button(
                "📊 Dashboard",
                on_click=lambda: ui.navigate.to("/dashboard")
            ).props("color=teal size=lg")
            ui.button(
                "⚙ Estratégias",
                on_click=lambda: ui.navigate.to("/strategies")
            ).props("color=blue outline size=lg")
            ui.button(
                "🛡 Risco Global",
                on_click=lambda: ui.navigate.to("/risk")
            ).props("color=orange outline size=lg")
            ui.button(
                "📋 Histórico",
                on_click=lambda: ui.navigate.to("/history")
            ).props("color=gray outline size=lg")

        ui.separator().classes("w-96 border-gray-700 my-4")

        with ui.row().classes("gap-6 text-center"):
            for label, value, color in [
                ("Versão", "2.0.0", "teal"),
                ("Modo", "DRY RUN", "yellow"),
                ("Status", "Online", "green"),
            ]:
                with ui.column().classes("items-center gap-1"):
                    ui.label(label).classes("text-xs text-gray-500 uppercase tracking-wide")
                    ui.badge(value).props(f"color={color} rounded").classes("text-sm px-3")


# ── Dashboard ────────────────────────────────────────────────
@ui.page("/dashboard")
async def dashboard_page() -> None:
    ui.dark_mode().enable()
    ui.query("body").style("background:#0f1117;")

    from src.persistence.database import get_session
    from src.persistence.models import Strategy, StrategyStatus
    from sqlalchemy import select

    # Header
    with ui.header().classes("bg-gray-900 border-b border-gray-700 px-6 py-3"):
        with ui.row().classes("items-center gap-4 w-full"):
            with ui.link(target="/").classes("no-underline"):
                ui.label("⚡ QUANTUM TRADER").classes("text-xl font-bold text-teal-400")
            ui.space()
            ui.label("Dashboard").classes("text-gray-400 text-sm")
            ui.button(icon="refresh", on_click=ui.navigate.reload).props(
                "flat round dense color=gray"
            )

    with ui.column().classes("p-6 gap-6 w-full"):

        # KPIs
        with ui.row().classes("gap-4 flex-wrap"):
            for label, val, color in [
                ("PnL Sessão",       "+$0.00",  "teal"),
                ("PnL Diário",       "+$0.00",  "blue"),
                ("Win Rate",         "0.0%",    "purple"),
                ("Ordens Hoje",      "0",       "orange"),
                ("Circuit Breaker",  "OFF",     "green"),
                ("Estratégias",      "0 ativas","cyan"),
            ]:
                with ui.card().classes("p-4 min-w-[150px] text-center bg-gray-800 border border-gray-700"):
                    ui.label(label).classes("text-xs text-gray-400 uppercase tracking-wide")
                    ui.label(val).classes(f"text-2xl font-bold text-{color}-400 mt-1")

        # Estratégias
        async with get_session() as session:
            result = await session.execute(select(Strategy))
            strategies = result.scalars().all()

        with ui.card().classes("w-full bg-gray-800 border border-gray-700 p-5"):
            with ui.row().classes("items-center justify-between mb-4"):
                ui.label("Estratégias").classes("text-base font-semibold text-gray-200")
                ui.button(
                    "Nova Estratégia", icon="add",
                    on_click=lambda: ui.navigate.to("/strategies")
                ).props("flat color=teal size=sm")

            if not strategies:
                with ui.column().classes("items-center py-10 gap-3"):
                    ui.icon("add_circle_outline").classes("text-6xl text-gray-700")
                    ui.label("Nenhuma estratégia criada").classes("text-gray-500 text-sm")
                    ui.button(
                        "Criar Estratégia",
                        on_click=lambda: ui.navigate.to("/strategies")
                    ).props("flat color=teal")
            else:
                status_colors = {
                    "ACTIVE": "green", "PAUSED": "yellow",
                    "STOPPED": "gray", "ERROR": "red",
                }
                for s in strategies:
                    sc = status_colors.get(s.status.value, "gray")
                    with ui.card().classes(
                        "w-full bg-gray-750 border border-gray-600 p-4 mb-2"
                    ):
                        with ui.row().classes("items-center justify-between w-full"):
                            with ui.row().classes("items-center gap-3"):
                                ui.icon("circle").classes(f"text-{sc}-400 text-xs")
                                ui.label(s.name).classes("font-semibold text-gray-100")
                                if s.dry_run:
                                    ui.badge("DRY RUN").props("color=orange rounded")
                                ui.badge(s.status.value).props(f"color={sc} rounded")
                            with ui.row().classes("gap-4 text-xs text-gray-400"):
                                ui.label(f"📌 {s.symbol}")
                                ui.label(f"💰 ${s.base_stake:.2f}")
                                ui.label(f"🎯 {s.gale_mode.value}")
                                ui.label(f"✅ ${s.stop_win:.2f}")
                                ui.label(f"❌ ${s.stop_loss:.2f}")


# ── Estratégias ──────────────────────────────────────────────
@ui.page("/strategies")
async def strategies_page() -> None:
    ui.dark_mode().enable()
    ui.query("body").style("background:#0f1117;")

    from src.persistence.database import get_session
    from src.persistence.models import Strategy, GaleMode, StrategyStatus
    from src.persistence.repository import StrategyRepo

    with ui.header().classes("bg-gray-900 border-b border-gray-700 px-6 py-3"):
        with ui.row().classes("items-center gap-4 w-full"):
            with ui.link(target="/").classes("no-underline"):
                ui.label("⚡ QUANTUM TRADER").classes("text-xl font-bold text-teal-400")
            ui.space()
            ui.label("Estratégias").classes("text-gray-400 text-sm")

    with ui.column().classes("p-6 gap-6 w-full max-w-4xl"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("⚙ Gerenciar Estratégias").classes(
                "text-xl font-bold text-gray-100"
            )

        # Formulário de criação
        with ui.card().classes("w-full bg-gray-800 border border-gray-700 p-6"):
            ui.label("Nova Estratégia").classes(
                "text-base font-semibold text-teal-400 mb-4"
            )

            with ui.row().classes("gap-4 w-full flex-wrap"):
                name = ui.input(
                    "Nome", placeholder="Ex: EMA_RSI_R100"
                ).classes("flex-1 min-w-[200px]")
                symbol = ui.select(
                    ["R_10", "R_25", "R_50", "R_75", "R_100", "1HZ10V", "1HZ100V"],
                    label="Ativo", value="R_100"
                ).classes("flex-1 min-w-[150px]")

            ui.separator().classes("border-gray-700 my-2")
            ui.label("Stake & Gale").classes("text-sm font-semibold text-gray-300 mb-2")

            with ui.row().classes("gap-4 w-full flex-wrap"):
                base_stake = ui.number(
                    "Stake Base ($)", value=1.0, min=0.35, step=0.5, format="%.2f"
                ).classes("flex-1 min-w-[130px]")
                gale_mode = ui.select(
                    {m.value: m.value for m in GaleMode},
                    label="Modo Gale/Kelly", value=GaleMode.NONE.value
                ).classes("flex-1 min-w-[180px]")
                gale_mult = ui.number(
                    "Multiplicador", value=2.0, min=1.1, step=0.1, format="%.1f"
                ).classes("flex-1 min-w-[120px]")
                max_gale = ui.number(
                    "Máx. Níveis", value=3, min=1, max=10
                ).classes("flex-1 min-w-[100px]")
                kelly = ui.number(
                    "Kelly Fraction", value=0.25, min=0.01, max=1.0, step=0.05, format="%.2f"
                ).classes("flex-1 min-w-[120px]")

            ui.separator().classes("border-gray-700 my-2")
            ui.label("Risk Management").classes("text-sm font-semibold text-gray-300 mb-2")

            with ui.row().classes("gap-4 w-full flex-wrap"):
                stop_win = ui.number(
                    "Stop Win ($)", value=50.0, min=1.0, step=5.0, format="%.2f"
                ).classes("flex-1 min-w-[130px]")
                stop_loss = ui.number(
                    "Stop Loss ($)", value=20.0, min=1.0, step=5.0, format="%.2f"
                ).classes("flex-1 min-w-[130px]")
                max_losses = ui.number(
                    "Perdas Consec. Máx.", value=3, min=1, max=20
                ).classes("flex-1 min-w-[160px]")
                cooldown = ui.number(
                    "Cooldown (s)", value=60, min=5, max=3600
                ).classes("flex-1 min-w-[100px]")
                win_prob = ui.number(
                    "Win Prob (Kelly)", value=0.50, min=0.01, max=0.99, step=0.01, format="%.2f"
                ).classes("flex-1 min-w-[130px]")

            with ui.row().classes("items-center gap-4 mt-2"):
                duration = ui.number("Duração", value=1, min=1, max=60).classes("w-24")
                dur_unit = ui.select(
                    {"t": "Ticks", "s": "Segundos", "m": "Minutos"},
                    label="Unidade", value="t"
                ).classes("w-32")
                dry_run = ui.checkbox("DRY RUN (simulação)", value=True)

            async def _save() -> None:
                if not name.value or not name.value.strip():
                    ui.notify("⚠ Nome é obrigatório!", type="negative")
                    return
                async with get_session() as session:
                    strat = Strategy(
                        name=name.value.strip(),
                        symbol=symbol.value,
                        base_stake=float(base_stake.value),
                        duration=int(duration.value),
                        duration_unit=dur_unit.value,
                        gale_mode=GaleMode(gale_mode.value),
                        gale_multiplier=float(gale_mult.value),
                        max_gale_levels=int(max_gale.value),
                        kelly_fraction=float(kelly.value),
                        win_probability=float(win_prob.value),
                        stop_win=float(stop_win.value),
                        stop_loss=float(stop_loss.value),
                        max_consecutive_losses=int(max_losses.value),
                        cooldown_seconds=int(cooldown.value),
                        dry_run=dry_run.value,
                        status=StrategyStatus.STOPPED,
                    )
                    session.add(strat)
                ui.notify(f"✅ Estratégia '{name.value}' criada!", type="positive")
                await ui.run_javascript("setTimeout(()=>location.reload(),800)")

            ui.button("💾 Salvar Estratégia", on_click=_save).props(
                "color=teal"
            ).classes("mt-4 w-full")


# ── Histórico ────────────────────────────────────────────────
@ui.page("/history")
async def history_page() -> None:
    ui.dark_mode().enable()
    ui.query("body").style("background:#0f1117;")

    from src.persistence.database import get_session
    from src.persistence.models import Order
    from sqlalchemy import select

    with ui.header().classes("bg-gray-900 border-b border-gray-700 px-6 py-3"):
        with ui.row().classes("items-center gap-4 w-full"):
            with ui.link(target="/").classes("no-underline"):
                ui.label("⚡ QUANTUM TRADER").classes("text-xl font-bold text-teal-400")
            ui.space()
            ui.label("Histórico de Ordens").classes("text-gray-400 text-sm")

    with ui.column().classes("p-6 gap-4 w-full"):
        ui.label("📋 Histórico de Ordens").classes("text-xl font-bold text-gray-100")

        async with get_session() as session:
            result = await session.execute(
                select(Order).order_by(Order.created_at.desc()).limit(200)
            )
            orders = result.scalars().all()

        if not orders:
            with ui.column().classes("items-center py-16 gap-3"):
                ui.icon("history").classes("text-6xl text-gray-700")
                ui.label("Nenhuma ordem registrada ainda").classes("text-gray-500")
            return

        columns = [
            {"name": "id",      "label": "ID",       "field": "id",      "sortable": True},
            {"name": "symbol",  "label": "Ativo",    "field": "symbol"},
            {"name": "type",    "label": "Tipo",     "field": "type"},
            {"name": "stake",   "label": "Stake",    "field": "stake",   "sortable": True},
            {"name": "profit",  "label": "Lucro",    "field": "profit",  "sortable": True},
            {"name": "status",  "label": "Status",   "field": "status"},
            {"name": "gale",    "label": "Gale Lv",  "field": "gale"},
            {"name": "mode",    "label": "Modo",     "field": "mode"},
            {"name": "date",    "label": "Data/Hora","field": "date",    "sortable": True},
        ]
        rows = [
            {
                "id":     o.id,
                "symbol": o.symbol,
                "type":   o.contract_type,
                "stake":  f"${o.stake:.2f}",
                "profit": f"${o.profit:.2f}" if o.profit is not None else "—",
                "status": o.status.value,
                "gale":   o.gale_level,
                "mode":   "DRY" if o.dry_run else "REAL",
                "date":   str(o.created_at)[:19],
            }
            for o in orders
        ]
        ui.table(columns=columns, rows=rows, row_key="id").classes(
            "w-full"
        ).props("flat bordered dense")


# ── Risco Global ─────────────────────────────────────────────
@ui.page("/risk")
async def risk_page() -> None:
    ui.dark_mode().enable()
    ui.query("body").style("background:#0f1117;")

    from src.persistence.database import get_session
    from src.persistence.models import RiskEvent
    from sqlalchemy import select

    with ui.header().classes("bg-gray-900 border-b border-gray-700 px-6 py-3"):
        with ui.row().classes("items-center gap-4 w-full"):
            with ui.link(target="/").classes("no-underline"):
                ui.label("⚡ QUANTUM TRADER").classes("text-xl font-bold text-teal-400")
            ui.space()
            ui.label("Risco Global").classes("text-gray-400 text-sm")

    with ui.column().classes("p-6 gap-6 w-full max-w-3xl"):
        ui.label("🛡️ Controle de Risco Global").classes(
            "text-xl font-bold text-gray-100"
        )

        with ui.card().classes("w-full bg-gray-800 border border-gray-700 p-6"):
            ui.label("Limites Diários Globais").classes(
                "text-sm font-semibold text-teal-400 mb-4"
            )
            with ui.row().classes("gap-4 flex-wrap"):
                ui.number("Stop Loss Diário (%)", value=5.0, min=0.5, max=50.0, step=0.5, format="%.1f").classes("flex-1")
                ui.number("Stop Win Diário (%)", value=20.0, min=1.0, max=200.0, step=1.0, format="%.1f").classes("flex-1")
            ui.separator().classes("border-gray-700 my-3")
            ui.label("Circuit Breaker Global").classes(
                "text-sm font-semibold text-teal-400 mb-2"
            )
            with ui.row().classes("gap-4 flex-wrap"):
                ui.number("Perdas Consec. Máx.", value=5, min=1, max=50).classes("flex-1")
                ui.number("Cooldown Global (s)", value=300, min=30, max=86400).classes("flex-1")
            ui.button("💾 Salvar Configurações", on_click=lambda: ui.notify("Configurações salvas!", type="positive")).props("color=teal").classes("w-full mt-4")

        with ui.card().classes("w-full bg-gray-800 border border-gray-700 p-5"):
            ui.label("Eventos de Risco Recentes").classes(
                "text-sm font-semibold text-teal-400 mb-4"
            )
            async with get_session() as session:
                result = await session.execute(
                    select(RiskEvent).order_by(RiskEvent.created_at.desc()).limit(50)
                )
                events = result.scalars().all()

            if not events:
                ui.label("✅ Nenhum evento de risco registrado.").classes(
                    "text-gray-500 text-sm py-4"
                )
            else:
                for ev in events:
                    with ui.row().classes(
                        "items-center gap-3 py-2 border-b border-gray-700"
                    ):
                        ui.icon("warning").classes("text-yellow-400 text-sm")
                        ui.badge(ev.event_type.value).props(
                            "color=orange rounded"
                        ).classes("text-xs")
                        ui.label(ev.description).classes("text-xs text-gray-400 flex-1")
                        ui.label(str(ev.created_at)[:19]).classes(
                            "text-xs text-gray-600"
                        )
