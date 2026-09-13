"""Точка входа для сборки в exe (PyInstaller)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from braintool.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
