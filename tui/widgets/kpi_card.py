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
    KPICard .kpi-title    { color: $text-muted; text-style: bold; }
    KPICard .kpi-value    { color: $text;       text-style: bold; }
    KPICard .kpi-value.positive { color: $success; }
    KPICard .kpi-value.negative { color: $error;   }
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
        # FIX: id dedicado para que update_subtitle não confunda com o título.
        yield Label(self._subtitle, id="kpi_sub", classes="kpi-subtitle")

    def update_value(self, value: str) -> None:
        widget = self.query_one("#kpi_val", Static)
        widget.update(value)

        # FIX: Guard movido para antes da query — evita trabalho desnecessário
        # quando color_value=False (era feito depois, inútil).
        if not self._color_value:
            return

        # FIX: Trata valor zero como neutro (sem classe positiva nem negativa).
        # Antes "-0.00" era negativo e "0.00" positivo, causando coloração
        # errada em resultados de empate ou valores ainda não calculados.
        try:
            numeric = float(value.replace("%", "").replace("$", "").strip())
        except ValueError:
            numeric = 0.0

        widget.set_class(numeric > 0, "positive")
        widget.set_class(numeric < 0, "negative")

    def update_subtitle(self, subtitle: str) -> None:
        # FIX: Usa id "#kpi_sub" ao invés de query_one(Label).
        # query_one(Label) retornava o PRIMEIRO Label encontrado (o título),
        # sobrescrevendo o título ao invés do subtítulo.
        self.query_one("#kpi_sub", Label).update(subtitle)