"""Runtime-хук для PyInstaller: полностью импортирует pandas/matplotlib ДО того,
как хук PySide6 (pyi_rth_pyside6) вызовет импорт PySide6/shiboken. Иначе глобальный
хук shiboken ломает ленивый импорт six.moves внутри dateutil (используется pandas
и matplotlib) ошибкой "'_SixMetaPathImporter' object has no attribute '_path'".

Пользовательские runtime-хуки PyInstaller выполняются раньше хуков, добавленных
автоматически для найденных пакетов (в т.ч. pyi_rth_pyside6.py), поэтому этот файл
успевает "прогреть" импорт до того, как shiboken испортит машинерию импорта.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  # noqa: F401
import pandas  # noqa: F401
