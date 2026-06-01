from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Label, Static


class KPICard(Widget):
    """Card de KPI com título, valor principal e subtítulo."""

    DEFAULT_CSS = """
    KPICard {
        border: solid $primary;
        padding: 1 2;
        margin: 0 1;
        min-width: 18;
        height: 7;
        background: $surface;
    }
    KPICard .kpi-title { color: $text-muted; text-style: bold; }
    KPICard .kpi-value { color: $text; text-style: bold; font-size: 1.2; }
    KPICard .kpi-value.positive { color: $success; }
    KPICard .kpi-value.negative { color: $error; }
    KPICard .kpi-subtitle { color: $text-muted; }
    """

    def __init__(
        self,
        title: str,
        value: str,
        subtitle: str = "",
        color_value: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._value = value
        self._subtitle = subtitle
        self._color_value = color_value

    def compose(self) -> ComposeResult:
        yield Label(self._title, classes="kpi-title")
        yield Static(self._value, id="kpi_val", classes="kpi-value")
        yield Label(self._subtitle, classes="kpi-subtitle")

    def update_value(self, value: str) -> None:
        widget = self.query_one("#kpi_val", Static)
        widget.update(value)
        if self._color_value:
            is_positive = not value.startswith("-")
            widget.set_class(is_positive, "positive")
            widget.set_class(not is_positive, "negative")

    def update_subtitle(self, subtitle: str) -> None:
        self.query_one(Label).update(subtitle)