from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable, Label

from tui.state import TUIState


class TradesScreen(Widget):
    """
    Tela de trades abertos e recentes.

    Tabelas:
      - Trades Abertos : id (curto), símbolo, direção, stake, abertura, status.
      - Recentes (50)  : id, símbolo, direção, stake, PnL, status, fechamento.
    """

    def __init__(self, state: TUIState) -> None:
        super().__init__()
        self.state = state
        self._open_initialized = False
        self._recent_initialized = False

    def compose(self) -> ComposeResult:
        yield Label("🔓 Trades Abertos", classes="section-title")
        yield DataTable(id="open_trades_table", zebra_stripes=True, cursor_type="row")

        yield Label("📋 Trades Recentes", classes="section-title")
        yield DataTable(id="recent_trades_table", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        self._init_tables()

    def _init_tables(self) -> None:
        open_t = self.query_one("#open_trades_table", DataTable)
        open_t.add_columns("ID", "Símbolo", "Dir", "Stake", "Abertura", "Status")
        self._open_initialized = True

        recent_t = self.query_one("#recent_trades_table", DataTable)
        recent_t.add_columns(
            "ID", "Símbolo", "Dir", "Stake", "PnL", "Status", "Fechamento"
        )
        self._recent_initialized = True

    def refresh_data(self) -> None:
        if not self._open_initialized or not self._recent_initialized:
            return

        # Trades abertos.
        open_t = self.query_one("#open_trades_table", DataTable)
        open_t.clear()
        for trade in self.state.open_trades:
            open_t.add_row(
                trade.get("id", "")[:8],
                trade.get("symbol", ""),
                self._dir_badge(trade.get("direction", "")),
                f"${trade.get('stake', 0):.2f}",
                trade.get("opened_at", "")[:19],
                trade.get("status", ""),
            )

        # Trades recentes.
        recent_t = self.query_one("#recent_trades_table", DataTable)
        recent_t.clear()
        for trade in self.state.recent_trades[:50]:
            pnl = trade.get("pnl", 0.0) or 0.0
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            recent_t.add_row(
                trade.get("id", "")[:8],
                trade.get("symbol", ""),
                self._dir_badge(trade.get("direction", "")),
                f"${trade.get('stake', 0):.2f}",
                pnl_str,
                trade.get("status", ""),
                (trade.get("closed_at") or "")[:19],
            )

    @staticmethod
    def _dir_badge(direction: str) -> str:
        return "🟢 CALL" if direction == "BUY" else "🔴 PUT"