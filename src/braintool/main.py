"""Точка входа в программу."""
from __future__ import annotations

# ВАЖНО: pandas и matplotlib (через dateutil/six) должны быть импортированы раньше
# PySide6 — иначе система хуков shiboken ломает их импорт (AttributeError у six).
import matplotlib as _matplotlib_import_order_guard  # noqa: F401
_matplotlib_import_order_guard.use("Agg")
import matplotlib.pyplot as _pyplot_import_order_guard  # noqa: F401
import pandas as _pandas_import_order_guard  # noqa: F401

import sys

from PySide6.QtWidgets import QApplication

from .review_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Обработка фото мозга")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
