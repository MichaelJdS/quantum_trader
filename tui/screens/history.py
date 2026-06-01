from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, Select

from tui.state import TUIState


class HistoryScreen(Widget):
    """
    Histórico completo de trades com filtros de símbolo, status e período.
    Exibe até 500 trades com totais na rodapé.
    """

    def __init__(self, state: TUIState) -> None:
        super().__init__()
        self.state = state
        self._filter_symbol: str = "ALL"
        self._filter_status: str = "ALL"
        self._initialized: bool = False

    def compose(self) -> ComposeResult:
        yield Label("📋 Histórico de Trades", classes="section-title")

        with Horizontal(id="filter_row", classes="filter-row"):
            yield Label("Símbolo:")
            yield Select(
                options=[("Todos", "ALL")],
                id="filter_symbol",
                value="ALL",
            )
            yield Label("Status:")
            yield Select(
                options=[
                    ("Todos", "ALL"),
                    ("Ganhos", "WON"),
                    ("Perdidos", "LOST"),
                    ("Cancelados", "CANCELLED"),
                ],
                id="filter_status",
                value="ALL",
            )
            yield Button("Aplicar", id="btn_filter", variant="primary")
            yield Button("Exportar CSV", id="btn_export", variant="default")

        yield DataTable(id="history_table", zebra_stripes=True, cursor_type="row")
        yield Label(id="history_footer", classes="history-footer")

    def on_mount(self) -> None:
        table = self.query_one("#history_table", DataTable)
        table.add_columns(
            "ID", "Símbolo", "Dir", "Stake", "Payout", "PnL",
            "Status", "Estratégia", "Confiança", "Abertura", "Fechamento"
        )
        self._initialized = True
        self._update_symbol_filter()
        self._render_table()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_filter":
            self._render_table()
        elif event.button.id == "btn_export":
            self._export_csv()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "filter_symbol":
            self._filter_symbol = str(event.value)
        elif event.select.id == "filter_status":
            self._filter_status = str(event.value)

    def _update_symbol_filter(self) -> None:
        symbols = ["ALL"] + self.state.active_symbols
        sel = self.query_one("#filter_symbol", Select)
        sel.set_options([(s if s != "ALL" else "Todos", s) for s in symbols])

    def _render_table(self) -> None:
        if not self._initialized:
            return

        table = self.query_one("#history_table", DataTable)
        table.clear()

        trades = self.state.all_trades
        if self._filter_symbol != "ALL":
            trades = [t for t in trades if t.get("symbol") == self._filter_symbol]
        if self._filter_status != "ALL":
            trades = [t for t in trades if t.get("status") == self._filter_status]

        total_pnl = 0.0
        wins = 0
        for trade in trades[:500]:
            pnl = trade.get("pnl", 0.0) or 0.0
            total_pnl += pnl
            if trade.get("status") == "WON":
                wins += 1
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            payout = trade.get("payout") or 0.0
            table.add_row(
                trade.get("id", "")[:8],
                trade.get("symbol", ""),
                "🟢 CALL" if trade.get("direction") == "BUY" else "🔴 PUT",
                f"${trade.get('stake', 0):.2f}",
                f"${payout:.2f}" if payout else "-",
                pnl_str,
                trade.get("status", ""),
                trade.get("strategy_name", ""),
                f"{trade.get('confidence', 0):.1%}",
                (trade.get("opened_at") or "")[:16],
                (trade.get("closed_at") or "")[:16],
            )

        win_rate = wins / len(trades) if trades else 0.0
        footer = (
            f"Total: {len(trades)} trades | "
            f"Win Rate: {win_rate:.1%} | "
            f"PnL Total: {'+'if total_pnl>=0 else ''}${total_pnl:.2f}"
        )
        self.query_one("#history_footer", Label).update(footer)

    def _export_csv(self) -> None:
        import csv
        from datetime import datetime

        path = f"exports/history_{datetime.now():%Y%m%d_%H%M%S}.csv"
        import os
        os.makedirs("exports", exist_ok=True)

        trades = self.state.all_trades
        if not trades:
            self.app.notify("Nenhum trade para exportar.", severity="warning")
            return

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)

        self.app.notify(f"Exportado: {path}", severity="information")