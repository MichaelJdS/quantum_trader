"""
desktop_app/styles/theme.py — Dark theme profissional para o Quantum Trader Desktop
"""

DARK_THEME = """
/* ═══════════════════════════════════════════════════
   QUANTUM TRADER — Dark Theme Premium
   ═══════════════════════════════════════════════════ */

QMainWindow, QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

/* ── Sidebar ── */
#sidebar {
    background-color: #161b22;
    border-right: 1px solid #21262d;
    min-width: 200px;
    max-width: 200px;
}

#logo_label {
    color: #58a6ff;
    font-size: 18px;
    font-weight: 700;
    padding: 20px 16px 8px 16px;
}

#version_label {
    color: #8b949e;
    font-size: 11px;
    padding: 0px 16px 16px 16px;
}

#nav_btn {
    background-color: transparent;
    color: #8b949e;
    border: none;
    border-radius: 6px;
    padding: 10px 16px;
    text-align: left;
    font-size: 13px;
    margin: 2px 8px;
}

#nav_btn:hover {
    background-color: #21262d;
    color: #e6edf3;
}

#nav_btn:checked {
    background-color: #1f6feb;
    color: #ffffff;
    font-weight: 600;
}

/* ── Status Badge ── */
#status_running {
    background-color: #1a4731;
    color: #3fb950;
    border-radius: 4px;
    padding: 4px 10px;
    font-weight: 600;
    font-size: 11px;
}

#status_stopped {
    background-color: #2d1f1f;
    color: #f85149;
    border-radius: 4px;
    padding: 4px 10px;
    font-weight: 600;
    font-size: 11px;
}

#status_dryrun {
    background-color: #2d2a1f;
    color: #d29922;
    border-radius: 4px;
    padding: 4px 10px;
    font-weight: 600;
    font-size: 11px;
}

/* ── Cards de Métrica ── */
#metric_card {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px;
}

#metric_value {
    font-size: 28px;
    font-weight: 700;
    color: #e6edf3;
}

#metric_value_green {
    font-size: 28px;
    font-weight: 700;
    color: #3fb950;
}

#metric_value_red {
    font-size: 28px;
    font-weight: 700;
    color: #f85149;
}

#metric_label {
    color: #8b949e;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Botões de Ação ── */
#btn_start {
    background-color: #238636;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 13px;
}

#btn_start:hover {
    background-color: #2ea043;
}

#btn_start:disabled {
    background-color: #21262d;
    color: #8b949e;
}

#btn_stop {
    background-color: #b62324;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 13px;
}

#btn_stop:hover {
    background-color: #da3633;
}

#btn_stop:disabled {
    background-color: #21262d;
    color: #8b949e;
}

#btn_primary {
    background-color: #1f6feb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 13px;
}

#btn_primary:hover {
    background-color: #388bfd;
}

/* ── Tabela de Trades ── */
QTableWidget {
    background-color: #0d1117;
    gridline-color: #21262d;
    border: 1px solid #21262d;
    border-radius: 6px;
    selection-background-color: #1f6feb;
}

QTableWidget::item {
    padding: 6px 10px;
    border-bottom: 1px solid #21262d;
}

QHeaderView::section {
    background-color: #161b22;
    color: #8b949e;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    padding: 8px 10px;
    border: none;
    border-bottom: 1px solid #21262d;
}

/* ── Feed de Logs ── */
#log_display {
    background-color: #0d1117;
    color: #8b949e;
    border: 1px solid #21262d;
    border-radius: 6px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    padding: 8px;
}

/* ── Chat Gemini ── */
#chat_display {
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
}

#chat_input {
    background-color: #161b22;
    color: #e6edf3;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 13px;
}

#chat_input:focus {
    border-color: #1f6feb;
}

/* ── Inputs / Spinboxes / Combos ── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #161b22;
    color: #e6edf3;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #1f6feb;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #21262d;
    selection-background-color: #1f6feb;
}

/* ── Labels de Seção ── */
#section_title {
    font-size: 18px;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 4px;
}

#section_subtitle {
    color: #8b949e;
    font-size: 12px;
    margin-bottom: 16px;
}

/* ── Divider ── */
QFrame[frameShape="4"] {
    color: #21262d;
}

/* ── Scrollbars ── */
QScrollBar:vertical {
    background: #0d1117;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #21262d;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #30363d;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* ── Tab Widget ── */
QTabWidget::pane {
    border: 1px solid #21262d;
    border-radius: 6px;
    background: #0d1117;
}

QTabBar::tab {
    background: #161b22;
    color: #8b949e;
    padding: 8px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #1f6feb;
    color: #ffffff;
    font-weight: 600;
}

/* ── Checkbox ── */
QCheckBox {
    color: #e6edf3;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #21262d;
    background: #161b22;
}
QCheckBox::indicator:checked {
    background-color: #1f6feb;
    border-color: #1f6feb;
}

/* ── GroupBox ── */
QGroupBox {
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-weight: 600;
    color: #8b949e;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}

/* ── ToolTip ── */
QToolTip {
    background-color: #161b22;
    color: #e6edf3;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 4px 8px;
}
"""
