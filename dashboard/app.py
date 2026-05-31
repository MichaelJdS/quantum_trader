"""
Painel de controle avançado — NiceGUI
Controla estratégias, risco, gale/kelly, stop win/loss e monitora em tempo real.
"""
from __future__ import annotations
import asyncio
from nicegui import ui, app
from config.settings import settings
from src.persistence.database import get_session, init_db
from src.persistence.models import Strategy, GaleMode, StrategyStatus
from src.persistence.repository import StrategyRepo, OrderRepo, RiskEventRepo
from src.utils.logger import logger

# ── Estado global do painel ──────────────────────────────────
_engines: dict[int, object] = {}   # strategy_id → TradingEngine


# ────────────────────────────────────────────────────────────
# HELPERS DE UI
# ────────────────────────────────────────────────────────────
def badge(text: str, color: str = "blue") -> None:
    ui.badge(text).props(f"color={color} rounded")


def card_metric(label: str, value: str, color: str = "teal") -> None:
    with ui.card().classes("p-4 min-w-[140px] text-center"):
        ui.label(label).classes("text-xs text-gray-400 uppercase tracking-wide")
        ui.label(value).classes(f"text-2xl font-bold text-{color}-400 mt-1")


# ────────────────────────────────────────────────────────────
# PÁGINA PRINCIPAL
# ────────────────────────────────────────────────────────────
@ui.page("/")
async def index():
    ui.dark_mode().enable()
    ui.query("body").style("background:#0f1117")

    # ── Header ──────────────────────────────────────────────
    with ui.header().classes("bg-gray-900 border-b border-gray-700 px-6 py-3"):
        with ui.row().classes("items-center gap-4 w-full"):
            ui.label("⚡ QUANTUM TRADER").classes(
                "text-xl font-bold text-teal-400 tracking-widest"
            )
            ui.space()
            with ui.row().classes("gap-2 items-center"):
                ws_dot = ui.icon("circle").classes("text-red-400 text-sm")
                ws_label = ui.label("Desconectado").classes("text-xs text-gray-400")
            ui.button(
                "Nova Estratégia", icon="add",
                on_click=lambda: dialog_new_strategy.open()
            ).props("flat color=teal").classes("text-sm")

    # ── Layout principal ─────────────────────────────────────
    with ui.row().classes("w-full gap-0"):

        # Sidebar
        with ui.column().classes(
            "w-56 bg-gray-900 border-r border-gray-700 min-h-screen p-3 gap-1"
        ):
            for icon, label, path in [
                ("dashboard", "Dashboard", "/"),
                ("show_chart", "Estratégias", "/strategies"),
                ("security", "Risco Global", "/risk"),
                ("history", "Histórico", "/history"),
                ("analytics", "Analytics", "/analytics"),
                ("settings", "Configurações", "/settings"),
            ]:
                with ui.link(target=path):
                    with ui.row().classes(
                        "items-center gap-2 px-3 py-2 rounded-lg "
                        "hover:bg-gray-800 cursor-pointer"
                    ):
                        ui.icon(icon).classes("text-teal-400 text-lg")
                        ui.label(label).classes("text-sm text-gray-300")

        # Conteúdo
        with ui.column().classes("flex-1 p-6 gap-6"):
            await _dashboard_content()

    # ── Dialog nova estratégia ───────────────────────────────
    with ui.dialog() as dialog_new_strategy, ui.card().classes(
        "bg-gray-800 p-6 min-w-[480px]"
    ):
        await _form_new_strategy(dialog_new_strategy)


async def _dashboard_content():
    """Conteúdo principal do dashboard."""
    # KPIs superiores
    with ui.row().classes("gap-4"):
        card_metric("PnL Sessão", "+R$ 0.00", "teal")
        card_metric("PnL Diário", "+R$ 0.00", "blue")
        card_metric("Win Rate", "0.0%", "purple")
        card_metric("Ordens Hoje", "0", "orange")
        card_metric("Circuit Breaker", "OFF", "green")

    # Gráfico de equidade (placeholder)
    with ui.card().classes("w-full bg-gray-800 p-4"):
        ui.label("📈 Curva de Equidade").classes("text-sm text-gray-400 mb-3")
        equity_chart = ui.highchart({
            "chart": {"type": "area", "backgroundColor": "#1f2937", "height": 220},
            "title": {"text": ""},
            "series": [{"name": "PnL", "data": [], "color": "#14b8a6"}],
            "xAxis": {"type": "datetime"},
            "yAxis": {"title": {"text": ""}},
            "legend": {"enabled": False},
            "credits": {"enabled": False},
        }).classes("w-full")

    # Estratégias ativas
    ui.label("Estratégias Ativas").classes(
        "text-base font-semibold text-gray-200 mt-2"
    )

    async with get_session() as session:
        repo = StrategyRepo(session)
        strategies = await repo.all()

    if not strategies:
        with ui.card().classes("w-full bg-gray-800 p-8 text-center"):
            ui.icon("add_circle_outline").classes("text-5xl text-gray-600")
            ui.label("Nenhuma estratégia criada ainda").classes(
                "text-gray-500 mt-2"
            )
            ui.button(
                "Criar primeira estratégia", icon="add"
            ).props("flat color=teal")
        return

    for strat in strategies:
        await _strategy_card(strat)


async def _strategy_card(strat: Strategy):
    """Card de uma estratégia com controles operacionais."""
    status_colors = {
        StrategyStatus.ACTIVE: "green",
        StrategyStatus.PAUSED: "yellow",
        StrategyStatus.STOPPED: "gray",
        StrategyStatus.ERROR: "red",
    }
    color = status_colors.get(strat.status, "gray")

    with ui.card().classes("w-full bg-gray-800 border border-gray-700 p-4"):
        with ui.row().classes("items-center justify-between w-full"):

            # Nome + status
            with ui.row().classes("items-center gap-3"):
                ui.icon("circle").classes(f"text-{color}-400 text-sm")
                ui.label(strat.name).classes("font-semibold text-gray-100 text-base")
                if strat.dry_run:
                    ui.badge("DRY RUN").props("color=orange rounded").classes("text-xs")
                ui.badge(strat.status.value).props(
                    f"color={color} rounded"
                ).classes("text-xs")

            # Controles
            with ui.row().classes("gap-2"):
                ui.button(
                    "▶ Iniciar", on_click=lambda s=strat: _start_strategy(s.id)
                ).props("flat color=green").classes("text-xs").bind_visibility_from(
                    strat, "status",
                    backward=lambda v: v != StrategyStatus.ACTIVE
                )
                ui.button(
                    "⏹ Parar", on_click=lambda s=strat: _stop_strategy(s.id)
                ).props("flat color=red").classes("text-xs").bind_visibility_from(
                    strat, "status",
                    backward=lambda v: v == StrategyStatus.ACTIVE
                )
                ui.button(
                    "⚙", on_click=lambda s=strat: _open_config(s.id)
                ).props("flat color=gray").classes("text-xs")

        ui.separator().classes("my-2 border-gray-700")

        # Métricas da estratégia
        with ui.row().classes("gap-6 flex-wrap"):
            for label, value in [
                ("Ativo", strat.symbol),
                ("Stake Base", f"${strat.base_stake:.2f}"),
                ("Modo", strat.gale_mode.value),
                ("Stop Win", f"${strat.stop_win:.2f}"),
                ("Stop Loss", f"${strat.stop_loss:.2f}"),
                ("Max Gale", str(strat.max_gale_levels)),
                ("Kelly", f"{strat.kelly_fraction*100:.0f}%"),
            ]:
                with ui.column().classes("gap-0"):
                    ui.label(label).classes("text-xs text-gray-500")
                    ui.label(value).classes("text-sm font-medium text-gray-200")


async def _form_new_strategy(dialog):
    """Formulário de criação/edição de estratégia."""
    ui.label("Nova Estratégia").classes(
        "text-lg font-bold text-teal-400 mb-4"
    )

    name = ui.input("Nome da estratégia", placeholder="Ex: EMA_RSI_R100").classes(
        "w-full"
    )
    symbol = ui.select(
        ["R_10", "R_25", "R_50", "R_75", "R_100", "1HZ10V", "1HZ100V"],
        label="Ativo (Símbolo Deriv)",
        value="R_100",
    ).classes("w-full")
    strategy_type = ui.select(
        {"ema_rsi": "EMA/RSI Crossover", "bollinger_mr": "Bollinger Mean Reversion"},
        label="Tipo de estratégia",
        value="ema_rsi",
    ).classes("w-full")

    ui.separator().classes("my-3 border-gray-700")
    ui.label("Gestão de Stake").classes("text-sm font-semibold text-gray-300")

    with ui.row().classes("gap-4 w-full"):
        base_stake = ui.number(
            "Stake Base ($)", value=1.0, min=0.35, step=0.5, format="%.2f"
        ).classes("flex-1")
        duration = ui.number(
            "Duração", value=1, min=1, max=60
        ).classes("flex-1")
        duration_unit = ui.select(
            {"t": "Ticks", "s": "Segundos", "m": "Minutos"},
            label="Unidade",
            value="t",
        ).classes("flex-1")

    gale_mode = ui.select(
        {m.value: m.value for m in GaleMode},
        label="Modo de Gale/Staking",
        value=GaleMode.NONE.value,
    ).classes("w-full")

    with ui.row().classes("gap-4 w-full"):
        gale_multiplier = ui.number(
            "Multiplicador Gale", value=2.0, min=1.1, step=0.1, format="%.1f"
        ).classes("flex-1")
        max_gale = ui.number(
            "Máx. Níveis Gale", value=3, min=1, max=10
        ).classes("flex-1")
        kelly_frac = ui.number(
            "Fração Kelly (0-1)", value=0.25, min=0.01, max=1.0, step=0.05, format="%.2f"
        ).classes("flex-1")

    ui.separator().classes("my-3 border-gray-700")
    ui.label("Gestão de Risco").classes("text-sm font-semibold text-gray-300")

    with ui.row().classes("gap-4 w-full"):
        stop_win = ui.number(
            "Stop Win ($)", value=50.0, min=1.0, step=5.0, format="%.2f"
        ).classes("flex-1")
        stop_loss = ui.number(
            "Stop Loss ($)", value=20.0, min=1.0, step=5.0, format="%.2f"
        ).classes("flex-1")

    with ui.row().classes("gap-4 w-full"):
        max_losses = ui.number(
            "Perdas Consecutivas Máx.", value=3, min=1, max=20
        ).classes("flex-1")
        cooldown = ui.number(
            "Cooldown após circuit (s)", value=60, min=5, max=3600
        ).classes("flex-1")
        win_prob = ui.number(
            "Win Prob estimada (Kelly)", value=0.50, min=0.01, max=0.99, step=0.01
        ).classes("flex-1")

    dry_run = ui.checkbox("Modo DRY RUN (simulação)", value=True).classes("mt-2")

    ui.separator().classes("my-3 border-gray-700")
    with ui.row().classes("gap-3 justify-end"):
        ui.button("Cancelar", on_click=dialog.close).props("flat color=gray")

        async def _save():
            if not name.value:
                ui.notify("Nome obrigatório!", type="negative")
                return
            async with get_session() as session:
                strat = Strategy(
                    name=name.value,
                    symbol=symbol.value,
                    base_stake=float(base_stake.value),
                    duration=int(duration.value),
                    duration_unit=duration_unit.value,
                    gale_mode=GaleMode(gale_mode.value),
                    gale_multiplier=float(gale_multiplier.value),
                    max_gale_levels=int(max_gale.value),
                    kelly_fraction=float(kelly_frac.value),
                    win_probability=float(win_prob.value),
                    stop_win=float(stop_win.value),
                    stop_loss=float(stop_loss.value),
                    max_consecutive_losses=int(max_losses.value),
                    cooldown_seconds=int(cooldown.value),
                    dry_run=dry_run.value,
                    status=StrategyStatus.STOPPED,
                )
                session.add(strat)
            ui.notify(f"Estratégia '{name.value}' criada!", type="positive")
            dialog.close()
            await asyncio.sleep(0.5)
            ui.navigate.reload()

        ui.button("Salvar Estratégia", on_click=_save).props("color=teal")


async def _start_strategy(strategy_id: int):
    ui.notify("Iniciando estratégia...", type="positive")
    async with get_session() as session:
        repo = StrategyRepo(session)
        await repo.update_field(strategy_id, "status", StrategyStatus.ACTIVE)
    ui.navigate.reload()


async def _stop_strategy(strategy_id: int):
    ui.notify("Parando estratégia...", type="warning")
    async with get_session() as session:
        repo = StrategyRepo(session)
        await repo.update_field(strategy_id, "status", StrategyStatus.STOPPED)
    ui.navigate.reload()


async def _open_config(strategy_id: int):
    ui.notify(f"Editando estratégia #{strategy_id}")


# ────────────────────────────────────────────────────────────
# PÁGINA HISTÓRICO
# ────────────────────────────────────────────────────────────
@ui.page("/history")
async def history_page():
    ui.dark_mode().enable()
    ui.query("body").style("background:#0f1117")
    with ui.column().classes("p-6 gap-4 w-full"):
        ui.label("📋 Histórico de Ordens").classes(
            "text-xl font-bold text-teal-400"
        )
        async with get_session() as session:
            from sqlalchemy import select
            from src.persistence.models import Order
            result = await session.execute(
                select(Order).order_by(Order.created_at.desc()).limit(200)
            )
            orders = result.scalars().all()

        columns = [
            {"name": "id", "label": "ID", "field": "id", "sortable": True},
            {"name": "strategy_id", "label": "Estratégia", "field": "strategy_id"},
            {"name": "symbol", "label": "Ativo", "field": "symbol"},
            {"name": "contract_type", "label": "Tipo", "field": "contract_type"},
            {"name": "stake", "label": "Stake", "field": "stake"},
            {"name": "profit", "label": "Lucro", "field": "profit"},
            {"name": "status", "label": "Status", "field": "status"},
            {"name": "gale_level", "label": "Gale", "field": "gale_level"},
            {"name": "dry_run", "label": "Dry", "field": "dry_run"},
            {"name": "created_at", "label": "Data/Hora", "field": "created_at"},
        ]
        rows = [
            {
                "id": o.id,
                "strategy_id": o.strategy_id,
                "symbol": o.symbol,
                "contract_type": o.contract_type,
                "stake": f"${o.stake:.2f}",
                "profit": f"${o.profit:.2f}" if o.profit is not None else "-",
                "status": o.status.value,
                "gale_level": o.gale_level,
                "dry_run": "✓" if o.dry_run else "Real",
                "created_at": str(o.created_at)[:19],
            }
            for o in orders
        ]
        ui.table(columns=columns, rows=rows, row_key="id").classes(
            "w-full bg-gray-800"
        ).props("flat bordered")


# ────────────────────────────────────────────────────────────
# PÁGINA RISCO GLOBAL
# ────────────────────────────────────────────────────────────
@ui.page("/risk")
async def risk_page():
    ui.dark_mode().enable()
    ui.query("body").style("background:#0f1117")
    with ui.column().classes("p-6 gap-6 w-full max-w-3xl"):
        ui.label("🛡️ Controle de Risco Global").classes(
            "text-xl font-bold text-teal-400"
        )

        with ui.card().classes("w-full bg-gray-800 p-5"):
            ui.label("Limites Diários Globais").classes(
                "text-sm font-semibold text-gray-300 mb-4"
            )
            max_loss = ui.number(
                "Stop Loss Diário (%)",
                value=5.0, min=1.0, max=50.0, step=0.5
            ).classes("w-full")
            max_win = ui.number(
                "Stop Win Diário (%)",
                value=20.0, min=1.0, max=100.0, step=1.0
            ).classes("w-full")
            ui.separator().classes("my-3 border-gray-700")
            ui.label("Circuit Breaker Global").classes(
                "text-sm font-semibold text-gray-300 mb-2"
            )
            ui.label(
                "Ativa se qualquer estratégia atingir os limites configurados."
            ).classes("text-xs text-gray-500 mb-3")
            max_cons = ui.number(
                "Perdas consecutivas globais máx.",
                value=5, min=1, max=50
            ).classes("w-full")
            cooldown_global = ui.number(
                "Cooldown global (segundos)",
                value=300, min=30, max=86400
            ).classes("w-full")
            ui.button("Salvar Configurações").props("color=teal").classes("mt-4 w-full")

        with ui.card().classes("w-full bg-gray-800 p-5"):
            ui.label("Eventos de Risco Recentes").classes(
                "text-sm font-semibold text-gray-300 mb-4"
            )
            async with get_session() as session:
                from sqlalchemy import select
                from src.persistence.models import RiskEvent
                result = await session.execute(
                    select(RiskEvent).order_by(RiskEvent.created_at.desc()).limit(50)
                )
                events = result.scalars().all()

            if not events:
                ui.label("Nenhum evento de risco registrado.").classes(
                    "text-gray-500 text-sm"
                )
            else:
                for ev in events:
                    with ui.row().classes("items-center gap-3 py-1"):
                        ui.icon("warning").classes("text-yellow-400 text-sm")
                        ui.label(ev.event_type.value).classes(
                            "text-xs text-yellow-300 font-mono w-40"
                        )
                        ui.label(ev.description).classes("text-xs text-gray-400")
                        ui.space()
                        ui.label(str(ev.created_at)[:19]).classes(
                            "text-xs text-gray-600"
                        )


# ────────────────────────────────────────────────────────────
# STARTUP
# ────────────────────────────────────────────────────────────
async def startup():
    await init_db()
    from src.metrics.prometheus_metrics import TradingMetrics
    metrics = TradingMetrics()
    metrics.start_server(port=settings.prometheus_port)
    logger.info("Dashboard Quantum Trader iniciado.")


app.on_startup(startup)