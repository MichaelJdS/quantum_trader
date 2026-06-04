"""
desktop_app/screens/settings.py — Tela de Configurações do Bot

Funcionalidades de auto-preenchimento:
  1. Lê CLOUD_SERVER_URL, API_TOKEN e GEMINI_API_KEY do .env automaticamente
  2. Botão "🔍 Auto-detectar VM" consulta o gcloud para pegar o IP externo da VM
  3. Salva config em ~/.quantum_trader_config.json E atualiza o .env
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Localiza o .env na raiz do projeto (dois níveis acima de desktop_app/screens/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
_CONFIG_FILE = Path.home() / ".quantum_trader_config.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_env() -> dict[str, str]:
    """Lê todas as variáveis do .env e retorna como dict."""
    env: dict[str, str] = {}
    if not _ENV_FILE.exists():
        return env
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _write_env_value(key: str, value: str) -> None:
    """Atualiza (ou adiciona) uma variável no .env sem apagar as outras."""
    if not _ENV_FILE.exists():
        return
    content = _ENV_FILE.read_text(encoding="utf-8")
    lines = content.splitlines()
    updated = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped == key:
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key}={value}")
    _ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ── Tela de Configurações ────────────────────────────────────────────────────

class SettingsScreen(QWidget):
    """Tela de configurações com auto-preenchimento a partir do .env e gcloud."""

    config_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._auto_fill()   # Preenche automaticamente ao abrir

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(0, 0, 16, 0)
        main_layout.setSpacing(20)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addWidget(scroll)
        scroll.setWidget(content)

        # Header
        title = QLabel("Configurações")
        title.setObjectName("section_title")
        subtitle = QLabel("Parâmetros do servidor, risco e integrações")
        subtitle.setObjectName("section_subtitle")
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # ── Servidor Cloud ─────────────────────────────────────────────────
        server_group = QGroupBox("🌐 Servidor Cloud (Google Compute Engine)")
        server_form = QFormLayout(server_group)
        server_form.setSpacing(10)

        self._server_url = QLineEdit()
        self._server_url.setPlaceholderText("http://EXTERNAL_IP:8080  (auto-preenchido após deploy)")
        server_form.addRow("URL do Servidor:", self._server_url)

        self._api_token = QLineEdit()
        self._api_token.setPlaceholderText("Token de autenticação (gerado pelo deploy_vm.sh)")
        self._api_token.setEchoMode(QLineEdit.EchoMode.Password)
        server_form.addRow("API Token:", self._api_token)

        # Linha de botões: Auto-detectar + Testar
        btns_layout = QHBoxLayout()
        self._btn_autodetect = QPushButton("🔍 Auto-detectar VM")
        self._btn_autodetect.setObjectName("btn_primary")
        self._btn_autodetect.setToolTip(
            "Consulta o gcloud para encontrar o IP externo da VM automaticamente"
        )
        self._btn_autodetect.clicked.connect(self._auto_detect_vm)

        self._btn_test = QPushButton("✓ Testar Conexão")
        self._btn_test.setObjectName("btn_primary")
        self._btn_test.clicked.connect(self._test_connection)

        self._conn_status = QLabel("")
        self._conn_status.setFixedWidth(200)

        btns_layout.addWidget(self._btn_autodetect)
        btns_layout.addWidget(self._btn_test)
        btns_layout.addWidget(self._conn_status)
        btns_layout.addStretch()
        server_form.addRow("", btns_layout)
        main_layout.addWidget(server_group)

        # ── Gemini AI ──────────────────────────────────────────────────────
        gemini_group = QGroupBox("🤖 Gemini Advisor (Google AI Studio — Gratuito)")
        gemini_form = QFormLayout(gemini_group)
        gemini_form.setSpacing(10)

        key_layout = QHBoxLayout()
        self._gemini_key = QLineEdit()
        self._gemini_key.setPlaceholderText("AIzaSy... (lido do .env automaticamente)")
        self._gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._btn_get_key = QPushButton("🔑 Obter Chave Grátis")
        self._btn_get_key.setObjectName("btn_primary")
        self._btn_get_key.setToolTip("Abre o Google AI Studio para gerar sua chave gratuita")
        self._btn_get_key.clicked.connect(self._open_gemini_key_page)
        key_layout.addWidget(self._gemini_key)
        key_layout.addWidget(self._btn_get_key)
        gemini_form.addRow("GEMINI_API_KEY:", key_layout)

        self._gemini_interval = QSpinBox()
        self._gemini_interval.setRange(60, 3600)
        self._gemini_interval.setSingleStep(60)
        self._gemini_interval.setValue(300)
        self._gemini_interval.setSuffix(" s")
        self._gemini_interval.setToolTip(
            "Frequência com que o Gemini é consultado para recomendar estratégias"
        )
        gemini_form.addRow("Intervalo de consulta:", self._gemini_interval)
        main_layout.addWidget(gemini_group)

        # ── Trading ────────────────────────────────────────────────────────
        trading_group = QGroupBox("📈 Configurações de Trading")
        trading_form = QFormLayout(trading_group)
        trading_form.setSpacing(10)

        self._symbols_input = QLineEdit()
        self._symbols_input.setPlaceholderText("R_50, R_75, R_100")
        trading_form.addRow("Símbolos:", self._symbols_input)

        self._dry_run = QCheckBox("Ativar Dry Run — simulação sem dinheiro real")
        self._dry_run.setChecked(True)
        trading_form.addRow("Modo:", self._dry_run)

        self._stake = QDoubleSpinBox()
        self._stake.setRange(0.35, 1000.0)
        self._stake.setSingleStep(0.5)
        self._stake.setValue(1.0)
        self._stake.setPrefix("$ ")
        trading_form.addRow("Stake Base:", self._stake)

        # ── Kelly Criterion ──
        kelly_layout = QHBoxLayout()
        self._kelly_enabled = QCheckBox("Ativar Kelly Criterion")
        self._kelly_enabled.setChecked(False)
        self._kelly_pct = QDoubleSpinBox()
        self._kelly_pct.setRange(1.0, 100.0)
        self._kelly_pct.setSingleStep(1.0)
        self._kelly_pct.setValue(25.0)
        self._kelly_pct.setSuffix(" %")
        self._kelly_pct.setToolTip("Fração do Kelly pleno a ser utilizada (ex: 25% = Quarter Kelly)")
        
        # Desabilita o spinbox se Kelly estiver desativado
        self._kelly_pct.setEnabled(False)
        self._kelly_enabled.toggled.connect(self._kelly_pct.setEnabled)

        kelly_layout.addWidget(self._kelly_enabled)
        kelly_layout.addWidget(self._kelly_pct)
        kelly_layout.addStretch()
        trading_form.addRow("Gestão de Banca:", kelly_layout)

        self._granularity = QSpinBox()
        self._granularity.setRange(1, 3600)
        self._granularity.setValue(60)
        self._granularity.setSuffix(" s")
        trading_form.addRow("Granularidade (candle):", self._granularity)

        self._stop_win = QDoubleSpinBox()
        self._stop_win.setRange(0.1, 100.0)
        self._stop_win.setSingleStep(0.5)
        self._stop_win.setValue(5.0)
        self._stop_win.setSuffix(" %")
        trading_form.addRow("Stop Win:", self._stop_win)

        self._stop_loss = QDoubleSpinBox()
        self._stop_loss.setRange(0.1, 100.0)
        self._stop_loss.setSingleStep(0.5)
        self._stop_loss.setValue(3.0)
        self._stop_loss.setSuffix(" %")
        trading_form.addRow("Stop Loss:", self._stop_loss)

        self._max_drawdown = QDoubleSpinBox()
        self._max_drawdown.setRange(0.1, 100.0)
        self._max_drawdown.setSingleStep(0.5)
        self._max_drawdown.setValue(5.0)
        self._max_drawdown.setSuffix(" %")
        trading_form.addRow("Max Drawdown Diário:", self._max_drawdown)

        self._max_losses = QSpinBox()
        self._max_losses.setRange(1, 50)
        self._max_losses.setValue(5)
        trading_form.addRow("Max Perdas Consecutivas:", self._max_losses)
        main_layout.addWidget(trading_group)

        # Botão Salvar
        save_btn = QPushButton("💾  Salvar Configurações")
        save_btn.setObjectName("btn_start")
        save_btn.setFixedHeight(44)
        save_btn.clicked.connect(self._save_config)
        main_layout.addWidget(save_btn)
        main_layout.addStretch()

    # ── Auto-preenchimento ────────────────────────────────────────────────────

    def _auto_fill(self):
        """
        Preenche automaticamente os campos a partir de (em ordem de prioridade):
        1. ~/.quantum_trader_config.json (salvo pelo usuário)
        2. .env do projeto (CLOUD_SERVER_URL, API_TOKEN, GEMINI_API_KEY, etc.)
        """
        # Primeiro tenta o config salvo pelo usuário
        if _CONFIG_FILE.exists():
            try:
                data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
                self._apply_saved_config(data)
                return  # Config salvo tem prioridade
            except Exception:
                pass

        # Fallback: lê do .env
        self._fill_from_env()

    def _fill_from_env(self):
        """Preenche campos a partir do .env."""
        env = _read_env()

        cloud_url = env.get("CLOUD_SERVER_URL", "").strip()
        api_token = env.get("API_TOKEN", "").strip()
        gemini_key = env.get("GEMINI_API_KEY", "").strip()
        gemini_interval = env.get("GEMINI_INTERVAL", "300").strip()
        symbols = env.get("DEFAULT_SYMBOLS", "R_50,R_75").strip()
        stake = env.get("DEFAULT_STAKE", "1.0").strip()
        granularity = env.get("DEFAULT_GRANULARITY", "60").strip()
        stop_win = env.get("DEFAULT_STOP_WIN_PCT", "0.05").strip()
        stop_loss = env.get("DEFAULT_STOP_LOSS_PCT", "0.03").strip()
        max_drawdown = env.get("MAX_DAILY_DRAWDOWN_PCT", "0.05").strip()
        max_losses = env.get("MAX_CONSECUTIVE_LOSSES", "5").strip()

        if cloud_url:
            self._server_url.setText(cloud_url)
        if api_token:
            self._api_token.setText(api_token)
        if gemini_key:
            self._gemini_key.setText(gemini_key)
        try:
            self._gemini_interval.setValue(int(gemini_interval))
        except ValueError:
            pass

        # Símbolos: converte "R_50,R_75" → "R_50, R_75"
        self._symbols_input.setText(", ".join(s.strip() for s in symbols.split(",") if s.strip()))

        try:
            self._stake.setValue(float(stake))
        except ValueError:
            pass
        try:
            self._granularity.setValue(int(granularity))
        except ValueError:
            pass
        try:
            # .env pode ter 0.05 (fração) ou 5.0 (percentual) — normaliza para percentual
            sw = float(stop_win)
            self._stop_win.setValue(sw * 100 if sw <= 1.0 else sw)
        except ValueError:
            pass
        try:
            sl = float(stop_loss)
            self._stop_loss.setValue(sl * 100 if sl <= 1.0 else sl)
        except ValueError:
            pass
        try:
            md = float(max_drawdown)
            self._max_drawdown.setValue(md * 100 if md <= 1.0 else md)
        except ValueError:
            pass
        try:
            self._max_losses.setValue(int(max_losses))
        except ValueError:
            pass

        # Marca campos preenchidos com indicador visual
        if cloud_url:
            self._set_field_ok(self._server_url, "✓ Lido do .env")
        if gemini_key:
            self._set_field_ok(self._gemini_key, "✓ Lido do .env")

    def _apply_saved_config(self, data: dict):
        """Aplica config salvo em JSON."""
        self._server_url.setText(data.get("server_url", ""))
        self._api_token.setText(data.get("api_token", ""))
        self._gemini_key.setText(data.get("gemini_key", ""))
        self._gemini_interval.setValue(data.get("gemini_interval", 300))
        cfg = data.get("bot_config", {})
        if cfg:
            self._symbols_input.setText(", ".join(cfg.get("symbols", ["R_50"])))
            self._dry_run.setChecked(cfg.get("dry_run", True))
            self._stake.setValue(cfg.get("stake", 1.0))
            self._granularity.setValue(cfg.get("granularity", 60))
            self._stop_win.setValue(cfg.get("stop_win_pct", 5.0))
            self._stop_loss.setValue(cfg.get("stop_loss_pct", 3.0))
            self._max_drawdown.setValue(cfg.get("max_drawdown_pct", 5.0))
            self._max_losses.setValue(cfg.get("max_consecutive_losses", 5))
            self._kelly_enabled.setChecked(cfg.get("kelly_enabled", False))
            self._kelly_pct.setValue(cfg.get("kelly_pct", 25.0))

    # ── Auto-detecção da VM ───────────────────────────────────────────────────

    def _auto_detect_vm(self):
        """
        Detecta automaticamente onde o backend está rodando:
        1. Se URL atual é localhost → testa conexão local diretamente
        2. Se gcloud está instalado → busca IP da VM na nuvem (timeout: 5s)
        3. Se gcloud não está instalado → mostra instrução para deploy
        """
        current_url = self._server_url.text().strip()

        # Se já tem URL local configurada, apenas testa a conexão
        if current_url and ("localhost" in current_url or "127.0.0.1" in current_url):
            self._set_status("✓ Modo local (localhost)", "#58a6ff")
            self._test_connection()
            return

        self._btn_autodetect.setEnabled(False)
        self._btn_autodetect.setText("⏳ Buscando...")
        self._set_status("Verificando gcloud...", "#d29922")

        env = _read_env()
        instance = env.get("GCP_INSTANCE_NAME", "quantum-trader-backend")
        zone = env.get("GCP_ZONE", "us-central1-a")
        port = env.get("GCP_PORT", "8080")

        def _detect():
            # Verifica se gcloud existe antes de tentar
            try:
                chk = subprocess.run(
                    ["gcloud", "version"],
                    capture_output=True, text=True, timeout=4,
                )
                if chk.returncode != 0:
                    raise FileNotFoundError
            except (FileNotFoundError, subprocess.TimeoutExpired):
                QTimer.singleShot(0, lambda: self._on_vm_detect_failed(
                    "gcloud não instalado. VM não criada ainda."
                ))
                return

            # gcloud existe — tenta buscar IP da VM
            try:
                result = subprocess.run(
                    [
                        "gcloud", "compute", "instances", "describe", instance,
                        f"--zone={zone}",
                        "--format=get(networkInterfaces[0].accessConfigs[0].natIP)",
                    ],
                    capture_output=True, text=True, timeout=5,
                )
                ip = result.stdout.strip()
                if ip and result.returncode == 0:
                    url = f"http://{ip}:{port}"
                    QTimer.singleShot(0, lambda: self._on_vm_detected(url, instance, zone))
                else:
                    QTimer.singleShot(0, lambda: self._on_vm_detect_failed(
                        "VM não criada ainda. Execute: bash scripts/deploy_vm.sh"
                    ))
            except subprocess.TimeoutExpired:
                QTimer.singleShot(0, lambda: self._on_vm_detect_failed(
                    "Timeout — verifique sua conexão com a internet."
                ))
            except Exception as exc:
                QTimer.singleShot(0, lambda: self._on_vm_detect_failed(str(exc)))

        threading.Thread(target=_detect, daemon=True).start()

    def _on_vm_detected(self, url: str, instance: str, zone: str):
        self._server_url.setText(url)
        self._set_status(f"✓ VM encontrada!", "#3fb950")
        self._btn_autodetect.setEnabled(True)
        self._btn_autodetect.setText("🔍 Auto-detectar VM")
        _write_env_value("CLOUD_SERVER_URL", url)

    def _on_vm_detect_failed(self, error: str):
        self._btn_autodetect.setEnabled(True)
        self._btn_autodetect.setText("🔍 Auto-detectar VM")
        # Se não tem URL ainda, sugere localhost para teste local
        if not self._server_url.text().strip():
            self._server_url.setText("http://localhost:8080")
            self._set_status("ℹ️ Usando localhost para teste local", "#58a6ff")
        else:
            self._set_status(f"ℹ️ {error[:60]}", "#8b949e")

    # ── Testar Conexão ────────────────────────────────────────────────────────

    def _test_connection(self):
        url = self._server_url.text().strip()
        if not url:
            self._set_status("❌ URL não preenchida", "#f85149")
            return
        self._btn_test.setEnabled(False)
        self._set_status("Testando...", "#d29922")

        def _check():
            import httpx
            try:
                resp = httpx.get(f"{url.rstrip('/')}/health", timeout=6.0)
                ok = resp.status_code == 200
            except Exception:
                ok = False
            if ok:
                QTimer.singleShot(0, lambda: self._set_status("✓ Conectado ao backend!", "#3fb950"))
            else:
                QTimer.singleShot(0, lambda: self._set_status("❌ Backend inacessível", "#f85149"))
            QTimer.singleShot(0, lambda: self._btn_test.setEnabled(True))

        threading.Thread(target=_check, daemon=True).start()

    # ── Abrir página de chave Gemini ─────────────────────────────────────────

    def _open_gemini_key_page(self):
        import webbrowser
        webbrowser.open("https://aistudio.google.com/apikey")

    # ── Salvar ────────────────────────────────────────────────────────────────

    def _save_config(self):
        data = {
            "server_url": self._server_url.text().strip(),
            "api_token": self._api_token.text().strip(),
            "gemini_key": self._gemini_key.text().strip(),
            "gemini_interval": self._gemini_interval.value(),
            "bot_config": self.get_bot_config(),
        }
        _CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Atualiza o .env também
        if data["server_url"]:
            _write_env_value("CLOUD_SERVER_URL", data["server_url"])
        if data["api_token"]:
            _write_env_value("API_TOKEN", data["api_token"])
        if data["gemini_key"]:
            _write_env_value("GEMINI_API_KEY", data["gemini_key"])
        _write_env_value("GEMINI_INTERVAL", str(data["gemini_interval"]))

        self._set_status("✓ Configurações salvas!", "#3fb950")
        self.config_changed.emit(data)

    # ── API Pública ───────────────────────────────────────────────────────────

    def get_bot_config(self) -> dict:
        symbols = [s.strip() for s in self._symbols_input.text().split(",") if s.strip()]
        return {
            "symbols": symbols or ["R_50"],
            "dry_run": self._dry_run.isChecked(),
            "granularity": self._granularity.value(),
            "stake": self._stake.value(),
            "stop_win_pct": self._stop_win.value(),
            "stop_loss_pct": self._stop_loss.value(),
            "max_drawdown_pct": self._max_drawdown.value(),
            "max_consecutive_losses": self._max_losses.value(),
            "kelly_enabled": self._kelly_enabled.isChecked(),
            "kelly_pct": self._kelly_pct.value(),
        }

    def get_server_config(self) -> tuple[str, str]:
        return self._server_url.text().strip(), self._api_token.text().strip()

    def get_gemini_key(self) -> str:
        return self._gemini_key.text().strip()

    def get_gemini_interval(self) -> int:
        return self._gemini_interval.value()

    # ── Helpers visuais ───────────────────────────────────────────────────────

    def _set_status(self, msg: str, color: str):
        self._conn_status.setText(msg)
        self._conn_status.setStyleSheet(f"color: {color}; font-size: 12px;")

    def _set_field_ok(self, field: QLineEdit, tooltip: str):
        field.setStyleSheet("border-color: #1a4731;")
        field.setToolTip(tooltip)
