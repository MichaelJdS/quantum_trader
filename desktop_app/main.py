"""
desktop_app/main.py — Entry point do Quantum Trader Desktop App (PyQt6)

Uso:
    python -m desktop_app.main
    ou
    python desktop_app/main.py
"""
from __future__ import annotations

import sys
import os

# Fix: força renderização por software para evitar crash no plugin Windows do Qt
# (necessário em alguns sistemas sem OpenGL/DirectX completo)
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("QT_QPA_PLATFORM", "windows")

# Garante que o diretório raiz do projeto está no path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from desktop_app.main_window import MainWindow


def main():
    # High DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Quantum Trader")
    app.setOrganizationName("SuperNovaIA")

    # Fonte padrão
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
