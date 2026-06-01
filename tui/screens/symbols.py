from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Label, Select, Static

from tui.state import TUIState
from tui.widgets.candlestick_chart import CandlestickChart
from tui.widgets.indicator_panel import IndicatorPanel


class SymbolsScreen(Widget):
    """
    Tela de análise de símbolo individual.

    Exibe:
      - Seletor de símbolo.
      - Gráfico de candles ASCII.
      - Painel de indicadores: EMA, RSI, MACD, ATR, BB, ADX.
      - Status do sinal atual (CALL / PUT / NEUTRO).
      - Predição do ensemble (proba_up, proba_down, weights).
    """

    def __init__(self, state: TUIState) -> None:
        super().__init__()
        self.state = state
        self._selected_symbol: str = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="symbol_header"):
            yield Label("Símbolo: ")
            yield Select(
                options=[],
                id="symbol_select",
                prompt="Selecione um símbolo...",
            )

        with Horizontal(id="symbol_body"):
            with Vertical(id="chart_panel", classes="panel"):
                yield Label("📉 Candles (últimos 60)", classes="section-title")
                yield CandlestickChart(id="candlestick", state=self.state)

            with Vertical(id="indicators_panel", classes="panel"):
                yield Label("🔬 Indicadores", classes="section-title")
                yield IndicatorPanel(id="indicators", state=self.state)

        with Horizontal(id="signal_row"):
            yield Static(id="signal_display", classes="signal-box")
            yield Static(id="ensemble_display", classes="ensemble-box")

    def on_mount(self) -> None:
        self._update_symbol_select()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "symbol_select" and event.value:
            self._selected_symbol = str(event.value)
            self._refresh_symbol()

    def _update_symbol_select(self) -> None:
        sel = self.query_one("#symbol_select", Select)
        options = [(s, s) for s in self.state.active_symbols]
        if options:
            sel.set_options(options)
            if not self._selected_symbol:
                self._selected_symbol = options[0][0]
            self._refresh_symbol()

    def _refresh_symbol(self) -> None:
        if not self._selected_symbol:
            return

        sym_data = self.state.symbol_data.get(self._selected_symbol, {})

        # Candles.
        self.query_one(CandlestickChart).update_symbol(
            self._selected_symbol, sym_data.get("candles", [])
        )

        # Indicadores.
        self.query_one(IndicatorPanel).update_indicators(sym_data.get("indicators", {}))

        # Sinal.
        signal = sym_data.get("last_signal")
        self._render_signal(signal)

        # Ensemble.
        ensemble = sym_data.get("ensemble_prediction", {})
        self._render_ensemble(ensemble)

    def _render_signal(self, signal: dict | None) -> None:
        box = self.query_one("#signal_display", Static)
        if not signal:
            box.update("⚪ NEUTRO — Sem sinal ativo")
            box.remove_class("signal-call", "signal-put")
            return

        direction = signal.get("direction", "")
        conf = signal.get("confidence", 0.0)
        strategy = signal.get("strategy_name", "")

        if direction == "BUY":
            box.update(
                f"🟢 CALL\n"
                f"Confiança : {conf:.1%}\n"
                f"Estratégia: {strategy}"
            )
            box.add_class("signal-call")
            box.remove_class("signal-put")
        else:
            box.update(
                f"🔴 PUT\n"
                f"Confiança : {conf:.1%}\n"
                f"Estratégia: {strategy}"
            )
            box.add_class("signal-put")
            box.remove_class("signal-call")

    def _render_ensemble(self, ensemble: dict) -> None:
        box = self.query_one("#ensemble_display", Static)
        if not ensemble:
            box.update("🧠 Ensemble: aguardando dados...")
            return

        proba_up = ensemble.get("proba_up", 0.50)
        proba_down = ensemble.get("proba_down", 0.50)
        conf = ensemble.get("confidence", 0.0)
        weights = ensemble.get("weights_used", {})

        bar_up = "█" * int(proba_up * 20)
        bar_down = "█" * int(proba_down * 20)

        weights_str = "\n".join(
            f"  {k:<12}: {v:.1%}" for k, v in weights.items()
        )
        box.update(
            f"🧠 Ensemble — Confiança: {conf:.1%}\n"
            f"↑ UP   [{bar_up:<20}] {proba_up:.1%}\n"
            f"↓ DOWN [{bar_down:<20}] {proba_down:.1%}\n"
            f"\nPesos dinâmicos:\n{weights_str}"
        )