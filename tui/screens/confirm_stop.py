from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ConfirmStopScreen(ModalScreen[bool]):
    """Modal de confirmação antes do stop de emergência."""

    DEFAULT_CSS = """
    ConfirmStopScreen {
        align: center middle;
    }
    ConfirmStopScreen > * {
        background: $surface;
        border: thick $error;
        padding: 2 4;
        width: 50;
        height: 12;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label(
            "⛔ STOP DE EMERGÊNCIA\n\n"
            "Isso irá encerrar TODOS os trades e\n"
            "parar o sistema imediatamente.\n\n"
            "Confirma?",
            id="confirm_label",
        )
        with Horizontal(id="button_row"):
            yield Button("⛔ Confirmar", id="btn_confirm", variant="error")
            yield Button("Cancelar", id="btn_cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn_confirm")