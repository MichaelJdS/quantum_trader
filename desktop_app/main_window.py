"""
desktop_app/main_window.py — Janela Principal do Quantum Trader Desktop

Layout:
  ┌─────────────────────────────────────────────────────┐
  │  Sidebar (logo, nav, status, controles)             │
  ├─────────────────────────────────────────────────────┤
  │  Content Area (Dashboard / Trades / Gemini / Settings) │
  └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from desktop_app.api_client import CloudAPIClient
from desktop_app.screens.dashboard import DashboardScreen
from desktop_app.screens.gemini_chat import GeminiChatScreen
from desktop_app.screens.settings import SettingsScreen
from desktop_app.styles.theme import DARK_THEME


# ── Signal Bridge ────────────────────────────────────────────────────────────
# Bridge para passar eventos do thread do WebSocket para o thread da GUI

class SignalBridge(QObject):
    """Ponte entre eventos do WebSocket (thread background) e a GUI (thread principal)."""
    event_received = pyqtSignal(str, object)
    status_received = pyqtSignal(dict)
    council_status_received = pyqtSignal(dict)


# ── Sidebar Button ────────────────────────────────────────────────────────────

class NavButton(QPushButton):
    def __init__(self, icon: str, label: str, parent=None):
        super().__init__(f"  {icon}  {label}", parent)
        self.setObjectName("nav_btn")
        self.setCheckable(True)
        self.setFixedHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Janela principal do Quantum Trader Desktop."""

    def __init__(self):
        super().__init__()
        self._api_client: CloudAPIClient | None = None
        self._bridge = SignalBridge()
        self._is_bot_running = False
        self._poll_timer = QTimer()
        self._current_nav: NavButton | None = None

        self.setWindowTitle("⚡ Quantum Trader")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)
        self.setStyleSheet(DARK_THEME)

        self._setup_ui()
        self._connect_signals()
        self._navigate(self._nav_dashboard)

        # Tenta reconectar ao backend automaticamente na abertura
        QTimer.singleShot(500, self._auto_connect)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Logo
        logo = QLabel("⚡ Quantum")
        logo.setObjectName("logo_label")
        version = QLabel("Trader v2.0 · Cloud")
        version.setObjectName("version_label")
        sidebar_layout.addWidget(logo)
        sidebar_layout.addWidget(version)

        # Divisor
        div1 = QFrame()
        div1.setFrameShape(QFrame.Shape.HLine)
        sidebar_layout.addWidget(div1)
        sidebar_layout.addSpacing(8)

        # Botões de navegação
        self._nav_dashboard = NavButton("📊", "Dashboard")
        self._nav_trades = NavButton("📋", "Trades")
        self._nav_gemini = NavButton("🤖", "Gemini Chat")
        self._nav_settings = NavButton("⚙️", "Configurações")
        self._nav_buttons = [
            self._nav_dashboard,
            self._nav_trades,
            self._nav_gemini,
            self._nav_settings,
        ]
        for btn in self._nav_buttons:
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        # Divisor
        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.HLine)
        sidebar_layout.addWidget(div2)
        sidebar_layout.addSpacing(8)

        # Status de conexão
        self._conn_badge = QLabel("● Desconectado")
        self._conn_badge.setStyleSheet("color: #f85149; font-size: 11px; padding: 0 16px 4px 16px;")
        sidebar_layout.addWidget(self._conn_badge)

        # Status do bot
        self._bot_badge = QLabel("Bot Parado")
        self._bot_badge.setObjectName("status_stopped")
        self._bot_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bot_badge.setStyleSheet(
            "background-color: #2d1f1f; color: #f85149; border-radius: 4px; "
            "padding: 4px; font-size: 11px; font-weight: 700; margin: 0 8px 8px 8px;"
        )
        sidebar_layout.addWidget(self._bot_badge)

        # Botões Start/Stop
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(8, 0, 8, 12)
        btn_row.setSpacing(6)
        self._btn_start = QPushButton("▶ Iniciar")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.setFixedHeight(36)
        self._btn_start.clicked.connect(self._on_start)

        self._btn_stop = QPushButton("■ Parar")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setFixedHeight(36)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)

        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        sidebar_layout.addLayout(btn_row)

        root.addWidget(sidebar)

        # ── Content Area ──
        self._stack = QStackedWidget()
        self._dashboard = DashboardScreen()
        self._trades = self._build_trades_screen()
        self._gemini_chat = GeminiChatScreen()
        self._settings = SettingsScreen()

        self._stack.addWidget(self._dashboard)   # index 0
        self._stack.addWidget(self._trades)       # index 1
        self._stack.addWidget(self._gemini_chat)  # index 2
        self._stack.addWidget(self._settings)     # index 3

        root.addWidget(self._stack)

    def _build_trades_screen(self) -> QWidget:
        """Tela de trades — tabela em modo fullscreen."""
        from PyQt6.QtWidgets import QTableWidget
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        title = QLabel("Trades ao Vivo")
        title.setObjectName("section_title")
        subtitle = QLabel("Histórico de trades da sessão atual")
        subtitle.setObjectName("section_subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        self._full_trades_table = QTableWidget(0, 6)
        self._full_trades_table.setHorizontalHeaderLabels(["HORA", "SÍMBOLO", "DIR", "STAKE", "PNL", "CONF"])
        h_header = self._full_trades_table.horizontalHeader()
        if h_header is not None:
            h_header.setStretchLastSection(True)
            
        v_header = self._full_trades_table.verticalHeader()
        if v_header is not None:
            v_header.setVisible(False)
        self._full_trades_table.setAlternatingRowColors(True)
        self._full_trades_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._full_trades_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        layout.addWidget(self._full_trades_table)
        return w

    def _connect_signals(self):
        # Navegação
        self._nav_dashboard.clicked.connect(lambda: self._navigate(self._nav_dashboard))
        self._nav_trades.clicked.connect(lambda: self._navigate(self._nav_trades))
        self._nav_gemini.clicked.connect(lambda: self._navigate(self._nav_gemini))
        self._nav_settings.clicked.connect(lambda: self._navigate(self._nav_settings))

        # Eventos do WebSocket → GUI
        self._bridge.event_received.connect(self._on_ws_event)
        self._bridge.status_received.connect(self._dashboard.update_status)
        self._bridge.council_status_received.connect(self._dashboard.on_council_status)

        # Settings salvas → reconectar
        self._settings.config_changed.connect(self._on_config_changed)

        # Poll de status a cada 10s
        self._poll_timer.timeout.connect(self._poll_status)
        self._poll_timer.start(10_000)

    # ── Navegação ─────────────────────────────────────────────────────────────

    def _navigate(self, btn: NavButton):
        if self._current_nav:
            self._current_nav.setChecked(False)
        btn.setChecked(True)
        self._current_nav = btn

        idx = self._nav_buttons.index(btn)
        self._stack.setCurrentIndex(idx)

    # ── Conexão ───────────────────────────────────────────────────────────────

    def _auto_connect(self):
        url, token = self._settings.get_server_config()
        if url:
            self._connect_to_server(url, token)

    def _on_config_changed(self, _data: dict):
        url, token = self._settings.get_server_config()
        if url:
            self._connect_to_server(url, token)

    def _connect_to_server(self, url: str, token: str):
        if self._api_client:
            self._api_client.stop_stream()

        def on_event(event: str, data):
            self._bridge.event_received.emit(event, data)

        self._api_client = CloudAPIClient(
            base_url=url,
            api_token=token,
            on_event=on_event,
        )
        self._api_client.start_stream()
        self._gemini_chat.set_api_client(self._api_client)

    def _on_ws_event(self, event: str, data):
        if event == "connected":
            self._conn_badge.setText("● Conectado")
            self._conn_badge.setStyleSheet("color: #3fb950; font-size: 11px; padding: 0 16px 4px 16px;")
            # Busca status imediato
            self._poll_status()

        elif event == "connection_lost":
            self._conn_badge.setText("● Reconectando...")
            self._conn_badge.setStyleSheet("color: #d29922; font-size: 11px; padding: 0 16px 4px 16px;")

        elif event == "status":
            self._update_bot_state(data)
            self._dashboard.update_status(data)

        elif event == "trade_opened":
            self._dashboard.on_trade_event(data)
            
            # Adiciona na tabela de trades (modo fullscreen)
            from PyQt6.QtWidgets import QTableWidgetItem
            from PyQt6.QtGui import QColor
            from datetime import datetime
            
            self._full_trades_table.insertRow(0)
            
            ts = data.get("ts", "")
            if ts:
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M:%S")
                except: pass

            def item(t, c=QColor("#e6edf3")):
                it = QTableWidgetItem(str(t))
                it.setForeground(c)
                it.setData(Qt.ItemDataRole.UserRole, data.get("trade_id"))
                return it

            self._full_trades_table.setItem(0, 0, item(ts))
            self._full_trades_table.setItem(0, 1, item(data.get("symbol", "")))
            self._full_trades_table.setItem(0, 2, item(data.get("direction", ""), QColor("#d29922")))
            self._full_trades_table.setItem(0, 3, item(f"${float(data.get('stake', 0)):.2f}"))
            self._full_trades_table.setItem(0, 4, item("Em aberto", QColor("#d29922")))
            self._full_trades_table.setItem(0, 5, item(f"{float(data.get('confidence', 0)):.1%}"))

        elif event == "trade":
            # UPDATE existing row on close
            from PyQt6.QtWidgets import QTableWidgetItem
            from PyQt6.QtGui import QColor
            
            trade_id = data.get("trade_id")
            for row in range(self._full_trades_table.rowCount()):
                it = self._full_trades_table.item(row, 0)
                if it and it.data(Qt.ItemDataRole.UserRole) == trade_id:
                    pnl = float(data.get("pnl", 0.0))
                    status = data.get("status", "")
                    color = QColor("#3fb950") if status == "WON" else QColor("#f85149") if status == "LOST" else QColor("#8b949e")
                    
                    def item(t, c=QColor("#e6edf3")):
                        new_it = QTableWidgetItem(str(t))
                        new_it.setForeground(c)
                        new_it.setData(Qt.ItemDataRole.UserRole, trade_id)
                        return new_it
                        
                    self._full_trades_table.setItem(row, 2, item(data.get("direction", ""), color))
                    self._full_trades_table.setItem(row, 4, item(f"${pnl:+.2f}", color))
                    break
            
            self._dashboard.on_trade_closed(data)

        elif event == "gemini_advice":
            self._dashboard.on_gemini_advice(data)

        elif event in ("bot_started", "bot_stopped"):
            self._poll_status()

    def _poll_status(self):
        if self._api_client is None:
            return

        def _fetch():
            try:
                status = self._api_client.get_status_sync()
                self._bridge.status_received.emit(status)
                self._update_bot_state(status)
                
                # Fetch council status as well
                if status.get("is_running"):
                    c_status = self._api_client.get_council_status_sync()
                    self._bridge.council_status_received.emit(c_status)
            except Exception:
                pass

        threading.Thread(target=_fetch, daemon=True).start()

    def _update_bot_state(self, status: dict):
        running = status.get("is_running", False)
        dry_run = status.get("dry_run", True)
        self._is_bot_running = running

        if running and dry_run:
            self._bot_badge.setText("DRY RUN")
            self._bot_badge.setStyleSheet(
                "background-color: #2d2a1f; color: #d29922; border-radius: 4px; "
                "padding: 4px; font-size: 11px; font-weight: 700; margin: 0 8px 8px 8px;"
            )
        elif running:
            self._bot_badge.setText("● AO VIVO")
            self._bot_badge.setStyleSheet(
                "background-color: #1a4731; color: #3fb950; border-radius: 4px; "
                "padding: 4px; font-size: 11px; font-weight: 700; margin: 0 8px 8px 8px;"
            )
        else:
            self._bot_badge.setText("Bot Parado")
            self._bot_badge.setStyleSheet(
                "background-color: #2d1f1f; color: #f85149; border-radius: 4px; "
                "padding: 4px; font-size: 11px; font-weight: 700; margin: 0 8px 8px 8px;"
            )

        self._btn_start.setEnabled(not running)
        self._btn_stop.setEnabled(running)

    # ── Controles do Bot ──────────────────────────────────────────────────────

    def _on_start(self):
        if self._api_client is None:
            QMessageBox.warning(self, "Não conectado", "Configure o servidor nas Configurações primeiro.")
            self._navigate(self._nav_settings)
            return

        config = self._settings.get_bot_config()
        dry = config.get("dry_run", True)
        if not dry:
            reply = QMessageBox.question(
                self, "⚠️ Modo Live",
                "Você está prestes a iniciar o bot em modo LIVE com dinheiro real.\n\nDeseja continuar?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._btn_start.setEnabled(False)
        self._btn_start.setText("Iniciando...")

        def _start():
            try:
                self._api_client.start_bot_sync(config)
            except Exception as exc:
                self._btn_start.setEnabled(True)
                self._btn_start.setText("▶ Iniciar")

        threading.Thread(target=_start, daemon=True).start()

    def _on_stop(self):
        self._btn_stop.setEnabled(False)
        self._btn_stop.setText("Parando...")

        def _stop():
            try:
                self._api_client.stop_bot_sync()
            except Exception as exc:
                self._btn_stop.setEnabled(True)
                self._btn_stop.setText("■ Parar")

        threading.Thread(target=_stop, daemon=True).start()

    def closeEvent(self, event):
        if self._api_client:
            self._api_client.close()
        super().closeEvent(event)
