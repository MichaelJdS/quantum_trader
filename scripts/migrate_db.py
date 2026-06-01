"""
Script para executar migrations Alembic programaticamente.

Uso:
    python scripts/migrate_db.py upgrade head
    python scripts/migrate_db.py downgrade -1
    python scripts/migrate_db.py revision --autogenerate -m "descricao"
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    args = sys.argv[1:] or ["upgrade", "head"]
    root = Path(__file__).parent.parent
    cmd = [
        sys.executable, "-m", "alembic",
        "-c", str(root / "infra" / "db" / "migrations" / "alembic.ini"),
        *args,
    ]
    print(f"Executando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(root))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()