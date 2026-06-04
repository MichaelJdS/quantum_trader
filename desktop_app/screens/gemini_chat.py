"""
desktop_app/screens/gemini_chat.py — Painel de Chat com o Gemini Advisor

Interface de chat para o usuário conversar diretamente com o Gemini,
com contexto do estado atual do bot.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable, Awaitable

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ChatBubble(QWidget):
    """Bolha de mensagem individual no chat."""

    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextFormat(Qt.TextFormat.PlainText)
        bubble.setMaximumWidth(600)
        bubble.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        if is_user:
            bubble.setStyleSheet("""
                background-color: #1f6feb;
                color: #ffffff;
                border-radius: 12px 12px 4px 12px;
                padding: 10px 14px;
                font-size: 13px;
            """)
            layout.addStretch()
            layout.addWidget(bubble)
        else:
            bubble.setStyleSheet("""
                background-color: #161b22;
                color: #e6edf3;
                border: 1px solid #21262d;
                border-radius: 12px 12px 12px 4px;
                padding: 10px 14px;
                font-size: 13px;
            """)
            layout.addWidget(bubble)
            layout.addStretch()


class AsyncWorker(QObject):
    """Worker para executar co-rotinas assíncronas e emitir resultado para a GUI."""
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, coro_fn: Callable, *args, **kwargs):
        super().__init__()
        self._coro_fn = coro_fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(self._coro_fn(*self._args, **self._kwargs))
            loop.close()
            self.result_ready.emit(str(result))
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class GeminiChatScreen(QWidget):
    """Tela de chat com o Gemini Advisor."""

    def __init__(self, api_client=None, parent=None):
        super().__init__(parent)
        self._api_client = api_client
        self._thread: QThread | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        title = QLabel("🤖 Gemini Advisor Chat")
        title.setObjectName("section_title")
        subtitle = QLabel(
            "Converse diretamente com o Gemini. Ele tem acesso ao estado atual do bot."
        )
        subtitle.setObjectName("section_subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # Área de mensagens (scroll)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setObjectName("chat_display")
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._messages_widget = QWidget()
        self._messages_layout = QVBoxLayout(self._messages_widget)
        self._messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._messages_layout.setSpacing(4)
        self._messages_layout.setContentsMargins(8, 8, 8, 8)
        self._scroll.setWidget(self._messages_widget)
        layout.addWidget(self._scroll)

        # Mensagem inicial
        self._add_message(
            "Olá! Sou o Gemini Advisor do Quantum Trader. Posso analisar o mercado, "
            "explicar estratégias, alertar sobre riscos e ajudar com qualquer dúvida "
            "sobre o bot. O que você gostaria de saber?",
            is_user=False,
        )

        # Input bar
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self._input = QTextEdit()
        self._input.setObjectName("chat_input")
        self._input.setPlaceholderText("Pergunte algo ao Gemini... (Enter para enviar)")
        self._input.setMaximumHeight(80)
        self._input.setAcceptRichText(False)
        self._input.keyPressEvent = self._on_key_press
        input_layout.addWidget(self._input)

        self._send_btn = QPushButton("Enviar")
        self._send_btn.setObjectName("btn_primary")
        self._send_btn.setFixedWidth(80)
        self._send_btn.setFixedHeight(80)
        self._send_btn.clicked.connect(self._on_send)
        input_layout.addWidget(self._send_btn)

        layout.addLayout(input_layout)

    def set_api_client(self, client):
        self._api_client = client

    def _on_key_press(self, event):
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QKeyEvent
        if event.key() == Qt.Key.Key_Return and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._on_send()
        else:
            QTextEdit.keyPressEvent(self._input, event)

    def _on_send(self):
        message = self._input.toPlainText().strip()
        if not message:
            return

        self._input.clear()
        self._add_message(message, is_user=True)
        self._send_btn.setEnabled(False)
        self._send_btn.setText("...")

        if self._api_client is None:
            self._add_message("❌ Não conectado ao backend. Configure o servidor nas Configurações.", is_user=False)
            self._send_btn.setEnabled(True)
            self._send_btn.setText("Enviar")
            return

        # Executa em thread separado para não bloquear a GUI
        self._thread = QThread()
        self._worker = AsyncWorker(self._api_client.chat, message)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.result_ready.connect(self._on_response)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.result_ready.connect(self._thread.quit)
        self._worker.error_occurred.connect(self._thread.quit)
        self._thread.start()

    def _on_response(self, text: str):
        self._add_message(text, is_user=False)
        self._send_btn.setEnabled(True)
        self._send_btn.setText("Enviar")

    def _on_error(self, error: str):
        self._add_message(f"❌ Erro: {error}", is_user=False)
        self._send_btn.setEnabled(True)
        self._send_btn.setText("Enviar")

    def _add_message(self, text: str, is_user: bool):
        bubble = ChatBubble(text, is_user)
        self._messages_layout.addWidget(bubble)
        # Scrola para baixo
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))
