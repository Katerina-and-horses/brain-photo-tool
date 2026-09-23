"""Точка входа в программу."""
from __future__ import annotations

# ВАЖНО: pandas и matplotlib (через dateutil/six) должны быть импортированы раньше
# PySide6 — иначе система хуков shiboken ломает их импорт (AttributeError у six).
import matplotlib as _matplotlib_import_order_guard  # noqa: F401
_matplotlib_import_order_guard.use("Agg")
import matplotlib.pyplot as _pyplot_import_order_guard  # noqa: F401
import pandas as _pandas_import_order_guard  # noqa: F401

import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from .review_window import MainWindow


def _install_excepthook() -> None:
    """Показывает необработанные исключения диалогом, а не тихо в stderr.

    Сборка PyInstaller — `console=False` (BrainPhotoTool.spec), то есть у
    пользователя (не программиста) stderr никуда не выведен и не виден вообще.
    Без этого хука необработанное исключение где угодно в программе — это полный
    крах без единого объяснения, что случилось и что делать (а несохранённая
    разметка при этом теряется без возможности восстановить)."""

    def hook(exc_type, exc_value, exc_tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            QMessageBox.critical(
                None,
                "Непредвиденная ошибка",
                "Программа столкнулась с ошибкой и не смогла продолжить это действие:\n\n"
                f"{exc_value}\n\n"
                "Если ошибка повторяется — сообщите разработчику текст ниже.\n\n" + text,
            )
        except Exception:  # noqa: BLE001 — сам диалог ошибок не должен уронить процесс
            pass
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = hook


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Обработка фото мозга")
    _install_excepthook()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
