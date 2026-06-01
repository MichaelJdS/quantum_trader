from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class IndicatorPanel(Widget):
    """Painel de indicadores técnicos com valores formatados."""

    def __init__(self, state=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(id="indicators_text")

    def update_indicators(self, indicators: dict) -> None:
        if not indicators:
            self.query_one("#indicators_text", Static).update(
                "Aguardando dados de indicadores..."
            )
            return

        def fmt(val, dec=2):
            return f"{val:.{dec}f}" if val is not None else "—"

        rsi = indicators.get("rsi_14", 0)
        rsi_icon = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "⚪"
        adx = indicators.get("adx", 0)
        adx_icon = "📈" if adx > 25 else "➡️"
        squeeze = indicators.get("squeeze", 0)
        sq_icon = "🔒" if squeeze else "🔓"

        text = (
            f"{'─' * 30}\n"
            f"EMA  9     : {fmt(indicators.get('ema_9'))}\n"
            f"EMA 21     : {fmt(indicators.get('ema_21'))}\n"
            f"EMA 50     : {fmt(indicators.get('ema_50'))}\n"
            f"EMA 200    : {fmt(indicators.get('ema_200'))}\n"
            f"{'─' * 30}\n"
            f"RSI 14     : {rsi_icon} {fmt(rsi)}\n"
            f"MACD Hist  : {fmt(indicators.get('macd_hist'), 4)}\n"
            f"Stoch K    : {fmt(indicators.get('stoch_k'))}\n"
            f"CCI        : {fmt(indicators.get('cci'))}\n"
            f"{'─' * 30}\n"
            f"ADX        : {adx_icon} {fmt(adx)}\n"
            f"BB%%        : {fmt(indicators.get('bb_pct'), 4)}\n"
            f"ATR 14     : {fmt(indicators.get('atr_14'), 4)}\n"
            f"Squeeze    : {sq_icon} {'ATIVO' if squeeze else 'LIVRE'}\n"
            f"Streak     : {indicators.get('candle_streak', 0):+d}\n"
            f"{'─' * 30}\n"
        )
        self.query_one("#indicators_text", Static).update(text)
