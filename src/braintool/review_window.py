"""Главное окно программы: два независимых пайплайна («Срезы» и «Обонятельный
эпителий»), каждый — свой цикл из трёх вкладок: Проект, Проверка масок, Результаты."""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QBrush, QColor, QKeySequence, QPixmap, QShortcut, QStandardItem, QStandardItemModel,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTableView, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

from .canvas import MaskCanvas
from .colors import color_for_animal
from .editing import add_mask, animal_count, delete_mask, move_mask, slices_of
from .imaging import (
    MASK_SATURATION_FRACTION_LIMIT, choose_mask_exposure, contrast_stretch_to_uint8, load_image,
    mask_saturation_fraction, scan_group_folder, unsaturated_exposures,
)
from .measurements import export_table, measure_shot, per_animal_average, rows_to_dataframe
from .models import GroupConfig, MaskMode, MeasurementRow, ShotReview
from .project_io import FILE_SUFFIX, autosave_path, load_markup, save_markup
from .segmentation import build_masks_for_image, find_blob_at
from .stats import boxplot_png_bytes, color_for_group, compare_groups, save_boxplot

PIPELINE_TITLES = {
    MaskMode.WHOLE_BLOB: "Срезы",
    MaskMode.ROSTRAL_CUT: "Обонятельный эпителий",
}

_P_VALUE_RE = re.compile(r"(p(?:-value)?\s*[:=]\s*[0-9.]+)")


def _format_stats_html(lines: list[str]) -> str:
    """HTML-версия текстового вывода сравнения групп: жирным — p-value (чтобы не
    искать глазами по строке), цветом предупреждения — уже принятым в программе
    для похожих предупреждений (`exposure_warning_label`, `slice_warning_label`),
    а не новым произвольным цветом."""
    text = "\n".join(lines)
    html_lines = []
    for raw_line in text.split("\n"):
        escaped = html.escape(raw_line)
        escaped = _P_VALUE_RE.sub(r"<b>\1</b>", escaped)
        if raw_line.strip().startswith("ВНИМАНИЕ"):
            escaped = f'<span style="color:#b34700; font-weight:bold;">{escaped}</span>'
        html_lines.append(escaped or "&nbsp;")
    return "<br>".join(html_lines)


class ProjectTab(QWidget):
    """Вкладка настройки групп: какие папки, какая сетка — для ОДНОГО пайплайна
    (что обводить зафиксировано пайплайном, а не выбирается по группе)."""

    def __init__(self, mode: MaskMode, on_start_review, on_open_markup=None, on_restore_autosave=None):
        super().__init__()
        self.mode = mode
        self._on_start_review = on_start_review
        self.groups: list[GroupConfig] = []

        layout = QVBoxLayout(self)
        info = QLabel(
            "Добавьте по одной папке на каждую группу животных. Для каждой папки укажите:\n"
            "— сколько животных на одном фото (по горизонтали, слева направо);\n"
            "— контрольная это группа или опытная (для сравнения «контроль vs опыт»);\n"
            "— при желании — произвольное условие (например, дата съёмки), чтобы потом "
            "сравнить группы «все со всеми» по этой метке.\n"
            "Количество срезов/повторов на одно животное программа определяет сама по фото."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Группа", "Папка", "Животных в ряд", "Контроль?", "Условие (метка)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Добавить папку с группой...")
        add_btn.clicked.connect(self._add_group)
        remove_btn = QPushButton("Удалить выбранную группу")
        remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        start_btn = QPushButton("Начать проверку масок ▶")
        start_btn.setMinimumHeight(36)
        start_btn.clicked.connect(self._start_review)
        layout.addWidget(start_btn)

        # продолжить прошлую работу — разметка сохраняется в файл (сессия 7)
        resume_row = QHBoxLayout()
        open_btn = QPushButton("Открыть сохранённую разметку…")
        open_btn.setMinimumHeight(32)
        if on_open_markup is not None:
            open_btn.clicked.connect(on_open_markup)
        self.restore_btn = QPushButton("↺ Продолжить с автосохранения")
        self.restore_btn.setMinimumHeight(32)
        self.restore_btn.setToolTip("Разметка, сохранённая автоматически в прошлый раз")
        if on_restore_autosave is not None:
            self.restore_btn.clicked.connect(on_restore_autosave)
        resume_row.addWidget(open_btn)
        resume_row.addWidget(self.restore_btn)
        resume_row.addStretch(1)
        layout.addLayout(resume_row)

    def _add_group(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с фото группы")
        if not folder:
            return
        name, ok = QInputDialog.getText(self, "Название группы", "Как назвать эту группу?", text=Path(folder).name)
        if not ok or not name.strip():
            return
        self._append_row(name.strip(), folder)

    def set_groups(self, groups: list[GroupConfig]) -> None:
        """Заполнить таблицу группами из загруженной разметки."""
        self.table.setRowCount(0)
        for g in groups:
            self._append_row(g.name, str(g.folder), g.cols, g.is_control, g.condition)

    def _append_row(
        self, name: str, folder: str, cols: int = 5, is_control: bool = False, condition: str = "",
    ) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        self.table.setItem(row, 1, QTableWidgetItem(folder))

        cols_spin = QSpinBox()
        cols_spin.setRange(1, 200)
        cols_spin.setValue(cols)  # в лаборатории в группе обычно 5 животных
        self.table.setCellWidget(row, 2, cols_spin)

        control_checkbox = QCheckBox()
        control_checkbox.setChecked(is_control)
        control_cell = QWidget()
        control_layout = QHBoxLayout(control_cell)
        control_layout.addWidget(control_checkbox)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 3, control_cell)

        self.table.setItem(row, 4, QTableWidgetItem(condition))

    def _remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _collect_groups(self) -> list[GroupConfig] | None:
        groups: list[GroupConfig] = []
        seen_names: set[str] = set()
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().strip()
            folder = self.table.item(row, 1).text()
            cols = self.table.cellWidget(row, 2).value()
            is_control = self.table.cellWidget(row, 3).findChild(QCheckBox).isChecked()
            condition_item = self.table.item(row, 4)
            condition = condition_item.text().strip() if condition_item else ""
            # название группы редактируется прямо в ячейке таблицы (двойной клик) —
            # пустое или повторяющееся имя молча сливает разные папки/животных в один
            # ряд статистики (per_animal_average группирует именно по этому имени)
            if not name:
                QMessageBox.warning(
                    self, "Пустое название группы",
                    f"Строка {row + 1}: укажите название группы (сейчас пусто).",
                )
                return None
            if name in seen_names:
                QMessageBox.warning(
                    self, "Повторяющееся название группы",
                    f"Название «{name}» использовано больше одного раза — разные папки "
                    "с одинаковым именем группы молча объединятся в статистике в одну "
                    "группу. Переименуйте одну из них.",
                )
                return None
            seen_names.add(name)
            try:
                groups.append(
                    GroupConfig(
                        name=name, folder=Path(folder), mode=self.mode,
                        cols=cols, is_control=is_control, condition=condition,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "Ошибка настройки группы", f"Группа «{name}»: {exc}")
                return None
        if not groups:
            QMessageBox.information(self, "Нет групп", "Добавьте хотя бы одну папку с группой.")
            return None
        return groups

    def _start_review(self) -> None:
        groups = self._collect_groups()
        if groups is None:
            return
        self._on_start_review(groups)


class ReassignDialog(QDialog):
    """Перенос маски к другому животному и/или на другой номер среза."""

    def __init__(self, parent, masks: list, target, allow_slices: bool):
        super().__init__(parent)
        self.setWindowTitle("Переназначить маску")
        self._masks = masks
        self._target = target
        n_animals = animal_count(masks)
        layout = QFormLayout(self)

        layout.addRow(QLabel(
            f"Сейчас: животное {target.animal_index + 1}"
            + (f", срез {target.slice_index + 1}" if allow_slices else "")
        ))

        self.animal_combo = QComboBox()
        for a in range(n_animals):
            r, g, b = color_for_animal(a)
            self.animal_combo.addItem(f"Животное {a + 1}", ("existing", a))
            self.animal_combo.setItemData(
                self.animal_combo.count() - 1, QColor(r, g, b), Qt.ItemDataRole.DecorationRole
            )
        for pos in range(n_animals + 1):
            if pos == 0:
                where = "самым левым (станет животным 1)"
            elif pos == n_animals:
                where = f"самым правым (станет животным {n_animals + 1})"
            else:
                where = f"между {pos} и {pos + 1}"
            self.animal_combo.addItem(f"Новое животное — {where}", ("new", pos))
        self.animal_combo.setCurrentIndex(target.animal_index)
        layout.addRow("Животное:", self.animal_combo)

        self.slice_combo = QComboBox()
        layout.addRow("Номер среза:", self.slice_combo)
        self.slice_combo.setVisible(allow_slices)
        layout.labelForField(self.slice_combo).setVisible(allow_slices)
        self.animal_combo.currentIndexChanged.connect(self._refresh_slices)
        self._refresh_slices()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _refresh_slices(self) -> None:
        kind, a = self.animal_combo.currentData()
        self.slice_combo.clear()
        if kind == "new":
            self.slice_combo.addItem("1", 0)
            return
        others = [m for m in slices_of(self._masks, a) if m is not self._target]
        self.slice_combo.addItem("по положению на фото (сверху вниз)", None)
        for i in range(len(others) + 1):
            self.slice_combo.addItem(str(i + 1), i)
        if a == self._target.animal_index:
            idx = self.slice_combo.findData(self._target.slice_index)
            if idx >= 0:
                self.slice_combo.setCurrentIndex(idx)

    def choice(self) -> tuple[int, int | None, bool]:
        """(животное, срез или None = по высоте, новое_ли_животное)"""
        kind, a = self.animal_combo.currentData()
        return a, self.slice_combo.currentData(), kind == "new"


def _section(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    box = QGroupBox(title)
    box.setStyleSheet("QGroupBox { font-weight: bold; } QGroupBox QWidget { font-weight: normal; }")
    lay = QVBoxLayout(box)
    lay.setSpacing(4)
    return box, lay


HELP_HTML = """
<table cellspacing="0" cellpadding="2">
<tr><td><b>Левая кнопка</b></td><td>рисовать кистью (или стирать, если включён ластик)</td></tr>
<tr><td><b>Правая кнопка</b></td><td>стирать кистью</td></tr>
<tr><td><b>Обе кнопки</b> (зажать и вести)</td><td>сдвинуть фото; средняя кнопка — тоже</td></tr>
<tr><td><b>Колесо</b></td><td>размер кисти</td></tr>
<tr><td><b>Ctrl + колесо</b></td><td>приблизить / отдалить под курсором</td></tr>
<tr><td><b>Клик по маске</b></td><td>выбрать её (обводка и подпись)</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>отменить последнюю правку</td></tr>
<tr><td><b>Delete</b></td><td>удалить выбранную маску</td></tr>
<tr><td><b>Esc</b></td><td>выйти из «Поищи здесь»</td></tr>
<tr><td><b>Ctrl+S</b></td><td>сохранить разметку</td></tr>
</table>
<p style="margin-top:6px"><b>Режим точек:</b> тянуть точку контура мышью; двойной клик на
линии — добавить точку; правый клик по точке — убрать её.</p>
<p><b>Черепа:</b> линию отреза тянуть за белые точки.</p>
"""


class ReviewTab(QWidget):
    """Вкладка постраничного просмотра и правки автоматически предложенных масок."""

    def __init__(self, mode: MaskMode, on_all_reviewed, on_changed=None, on_save=None, on_navigated=None):
        super().__init__()
        self.mode = mode
        self._on_all_reviewed = on_all_reviewed
        self._on_changed = on_changed or (lambda: None)
        self._on_save = on_save or (lambda: None)
        # переход между фото — момент для тихого автосохранения
        self._on_navigated = on_navigated or (lambda: None)
        self.reviews: list[ShotReview] = []
        self.current_index = 0

        layout = QVBoxLayout(self)

        self.progress_label = QLabel("Кадров пока нет")
        self.progress_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.progress_label)

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: #b34700; font-weight: bold;")
        self.warning_label.setWordWrap(True)
        layout.addWidget(self.warning_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        self.canvas = MaskCanvas()
        self.canvas.maskEdited.connect(self._on_mask_edited)
        self.canvas.activeChanged.connect(self._sync_tree_selection)
        self.canvas.findHereRequested.connect(self._find_here)
        splitter.addWidget(self.canvas)

        # боковая панель — в прокрутке: раньше длинная подсказка уходила вниз под кнопку
        # «Далее» и её не было видно (жалоба сессии 7)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 4, 4, 4)

        # --- маски ---
        box, lay = _section("Маски на этом фото")
        self.mask_tree = QTreeWidget()
        self.mask_tree.setHeaderHidden(True)
        self.mask_tree.setMinimumHeight(180)
        self.mask_tree.itemClicked.connect(self._on_tree_clicked)
        self.mask_tree.itemDoubleClicked.connect(lambda *_: self._reassign_active())
        lay.addWidget(self.mask_tree)
        row = QHBoxLayout()
        self.reassign_btn = QPushButton("Переназначить…")
        self.reassign_btn.setToolTip("Отнести выбранную маску к другому животному или поменять номер среза")
        self.reassign_btn.clicked.connect(self._reassign_active)
        self.delete_btn = QPushButton("Удалить маску")
        self.delete_btn.setToolTip("Удалить выбранную маску (мусор, блик). Отменяется Ctrl+Z")
        self.delete_btn.clicked.connect(self._delete_active)
        row.addWidget(self.reassign_btn)
        row.addWidget(self.delete_btn)
        lay.addLayout(row)
        side_layout.addWidget(box)

        # --- добавить (только срезы: у черепов маска на животное одна) ---
        box, lay = _section("Добавить пропущенное")
        self.find_here_btn = QPushButton("Поищи здесь")
        self.find_here_btn.setCheckable(True)
        self.find_here_btn.setToolTip(
            "Включите и щёлкните по фото там, где срез не нашёлся — программа сама найдёт "
            "его контур и решит, к какому животному он относится"
        )
        self.find_here_btn.toggled.connect(self._toggle_find_here)
        lay.addWidget(self.find_here_btn)
        self.new_slice_btn = QPushButton("✏ Новый срез кистью")
        self.new_slice_btn.setToolTip(
            "Нарисовать кистью срез, которого программа не нашла; он добавится к животному "
            "выбранной сейчас маски"
        )
        self.new_slice_btn.clicked.connect(self._new_slice_by_brush)
        lay.addWidget(self.new_slice_btn)
        side_layout.addWidget(box)
        box.setVisible(mode == MaskMode.WHOLE_BLOB)

        # --- инструменты ---
        box, lay = _section("Инструменты")
        self.erase_checkbox = QCheckBox("Ластик (левая кнопка стирает)")
        self.erase_checkbox.toggled.connect(self.canvas.set_erase_enabled)
        lay.addWidget(self.erase_checkbox)
        self.point_mode_checkbox = QCheckBox("Режим точек (тянуть контур за точки)")
        self.point_mode_checkbox.toggled.connect(self.canvas.set_point_mode_enabled)
        # по умолчанию включён для носа/эпителия (ROSTRAL_CUT) — там форма
        # сложная, полигон обязателен для точной обводки; для срезов (WHOLE_BLOB)
        # авто-маска обычно неплохая и по умолчанию правится кистью
        self.point_mode_checkbox.setChecked(mode == MaskMode.ROSTRAL_CUT)
        lay.addWidget(self.point_mode_checkbox)
        row = QHBoxLayout()
        self.undo_btn = QPushButton("↶ Отменить")
        self.undo_btn.setToolTip("Отменить последнюю правку на этом фото (Ctrl+Z)")
        self.undo_btn.clicked.connect(self._undo)
        reset_view_btn = QPushButton("⤢ Общий вид")
        reset_view_btn.setToolTip("Показать фото целиком (сбросить приближение и сдвиг)")
        reset_view_btn.clicked.connect(self.canvas.reset_view)
        row.addWidget(self.undo_btn)
        row.addWidget(reset_view_btn)
        lay.addLayout(row)
        side_layout.addWidget(box)

        # --- выдержка ---
        box, lay = _section("Выдержка, на которой ищутся маски")
        self.exposure_combo = QComboBox()
        # не шире панели: иначе длинный пункт раздувает всю панель за правый край
        self.exposure_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.exposure_combo.setMinimumContentsLength(12)
        self.exposure_combo.activated.connect(self._on_exposure_chosen)
        lay.addWidget(self.exposure_combo)
        note = QLabel(
            "По умолчанию — вторая по длине незасвеченная: на ней меньше свечения вокруг "
            "срезов. Если масок-мусора много — возьмите выдержку короче. На измерения это "
            "не влияет: яркость считается на общей выдержке (вкладка «Результаты»)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666666;")
        lay.addWidget(note)
        recompute_btn = QPushButton("Найти маски заново")
        recompute_btn.setToolTip(
            "Заново найти все маски этого фото автоматически на выбранной выдержке "
            "(ручные правки этого фото пропадут, отменяется Ctrl+Z)"
        )
        recompute_btn.clicked.connect(self._recompute_current)
        lay.addWidget(recompute_btn)
        side_layout.addWidget(box)

        # --- подсказка ---
        box, lay = _section("Мышь и клавиши")
        help_label = QLabel(HELP_HTML)
        help_label.setWordWrap(True)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(help_label)
        side_layout.addWidget(box)
        side_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # ширина панели: весь текст помещается без обрезки справа (жалоба сессии 7 —
        # содержимое шире окна прокрутки уходило за правый край)
        scroll.setMinimumWidth(340)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1000, 400])

        # --- навигация: «Принять» рядом с «Далее», крупно (раньше терялась посередине) ---
        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Назад")
        self.prev_btn.clicked.connect(self._go_prev)
        self.save_btn = QPushButton("Сохранить разметку")
        self.save_btn.setToolTip("Сохранить всю разметку в файл, чтобы продолжить позже (Ctrl+S)")
        self.save_btn.clicked.connect(lambda: self._on_save())
        self.accept_btn = QPushButton("✔ Принять всё на этом фото")
        self.accept_btn.clicked.connect(self._accept_all)
        self.next_btn = QPushButton("Далее ▶")
        self.next_btn.clicked.connect(self._go_next)
        for b in (self.prev_btn, self.save_btn, self.accept_btn, self.next_btn):
            b.setMinimumHeight(40)
        self.accept_btn.setMinimumWidth(260)
        self.next_btn.setMinimumWidth(140)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.save_btn)
        nav_row.addStretch(1)
        nav_row.addWidget(self.accept_btn)
        nav_row.addWidget(self.next_btn)
        layout.addLayout(nav_row)

        for seq, slot in (
            ("Ctrl+Z", self._undo),
            ("Ctrl+S", lambda: self._on_save()),
            ("Delete", self._delete_active),
            ("Escape", lambda: self.find_here_btn.setChecked(False)),
        ):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

    # ---------- показ кадра ----------

    def load_reviews(self, reviews: list[ShotReview], current_index: int = 0) -> None:
        self.reviews = reviews
        self.current_index = min(max(current_index, 0), max(len(reviews) - 1, 0))
        self._show_current()

    def _review(self) -> ShotReview | None:
        return self.reviews[self.current_index] if self.reviews else None

    def _show_current(self, keep_view: bool = False) -> None:
        review = self._review()
        if review is None:
            self.progress_label.setText("Кадров нет — вернитесь на вкладку «Проект»")
            return
        self.progress_label.setText(
            f"Кадр {self.current_index + 1} из {len(self.reviews)} — {review.shot.display_name}"
        )
        self.warning_label.setText(review.warning or "")
        self.find_here_btn.setChecked(False)
        self.canvas.set_shot(review.display_image, review.masks, review.undo_stack, keep_view=keep_view)
        self._refresh_exposure_combo(review)
        self._refresh_side()

    def _refresh_side(self) -> None:
        review = self._review()
        if review is None:
            return
        self._rebuild_mask_tree(review)
        self._refresh_accept_button_style()
        has_active = self.canvas.active_mask() is not None
        self.reassign_btn.setEnabled(has_active)
        self.delete_btn.setEnabled(has_active)
        self.undo_btn.setEnabled(self.canvas.can_undo())
        self.prev_btn.setEnabled(self.current_index > 0)
        last = self.current_index >= len(self.reviews) - 1
        self.next_btn.setText("К результатам ▶" if last else "Далее ▶")

    def _rebuild_mask_tree(self, review: ShotReview) -> None:
        self.mask_tree.blockSignals(True)
        self.mask_tree.clear()
        active = self.canvas.active_mask()
        by_animal: dict[int, list] = {}
        for m in review.masks:
            by_animal.setdefault(m.animal_index, []).append(m)
        active_item = None
        for ai in sorted(by_animal):
            r, g, b = color_for_animal(ai)
            slices = sorted(by_animal[ai], key=lambda m: m.slice_index)
            n_ok = sum(1 for m in slices if m.accepted and m.mask.any())
            n_nonempty = sum(1 for m in slices if m.mask.any())
            parent = QTreeWidgetItem([f"Животное {ai + 1}   ({n_ok}/{n_nonempty} принято)"])
            parent.setBackground(0, QBrush(QColor(r, g, b)))
            parent.setForeground(0, QBrush(QColor(0, 0, 0)))
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            parent.setData(0, Qt.ItemDataRole.UserRole, (ai, slices[0].slice_index))
            self.mask_tree.addTopLevelItem(parent)
            for m in slices:
                if self.mode == MaskMode.ROSTRAL_CUT:
                    label = "маска носа"
                else:
                    label = f"срез {m.slice_index + 1}"
                if not m.mask.any():
                    label += " — пусто, нарисуйте"
                elif m.accepted:
                    label += "  ✓ принят"
                child = QTreeWidgetItem([label])
                child.setData(0, Qt.ItemDataRole.UserRole, (m.animal_index, m.slice_index))
                if m.accepted and m.mask.any():
                    child.setForeground(0, QBrush(QColor("#2e7d32")))
                parent.addChild(child)
                if m is active:
                    active_item = child
            parent.setExpanded(True)
        if active_item is not None:
            self.mask_tree.setCurrentItem(active_item)
        self.mask_tree.blockSignals(False)

    def _sync_tree_selection(self) -> None:
        self._refresh_side()

    def _on_tree_clicked(self, item, _column=0) -> None:
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key is not None:
            self.canvas.set_active_mask(*key)
            self._refresh_side()

    def _refresh_exposure_combo(self, review: ShotReview) -> None:
        self.exposure_combo.clear()
        ok = set(unsaturated_exposures(review.shot))
        for exposure in sorted(review.shot.exposure_files, reverse=True):
            text = f"{exposure} мс"
            if exposure not in ok:
                text += " — засвет"
            if exposure == review.shot.chosen_exposure:
                text += "  (сейчас)"
            self.exposure_combo.addItem(text, exposure)
        idx = self.exposure_combo.findData(review.shot.chosen_exposure)
        self.exposure_combo.setCurrentIndex(max(idx, 0))
        self.exposure_combo.setEnabled(len(review.shot.exposure_files) > 1)

    # ---------- правка ----------

    def _mark_changed(self) -> None:
        self._on_changed()

    def _on_mask_edited(self) -> None:
        pending = self.canvas.pending_new_mask
        if pending is not None and pending.mask.any() and pending in self.canvas.masks:
            # новый срез кистью нарисован — ставим его на место по высоте среди срезов
            # того же животного (срезы нумеруются сверху вниз)
            self.canvas.pending_new_mask = None
            move_mask(self.canvas.masks, pending, pending.animal_index, None)
            self.canvas.set_active_mask(pending.animal_index, pending.slice_index)
        self._refresh_side()
        self._mark_changed()

    def _undo(self) -> None:
        if self.canvas.undo():
            self._refresh_side()

    def _toggle_find_here(self, enabled: bool) -> None:
        self.canvas.set_find_here_mode(enabled)
        self.find_here_btn.setStyleSheet(
            "background-color: #1565c0; color: white; font-weight: bold;" if enabled else ""
        )
        self.find_here_btn.setText(
            "Щёлкните по фото, где пропущен срез (Esc — выйти)" if enabled else "Поищи здесь"
        )

    def _find_here(self, x: float, y: float) -> None:
        review = self._review()
        if review is None:
            return
        taken = self.canvas.mask_at_point((x, y))
        if taken is not None:
            self.canvas.set_active_mask(taken.animal_index, taken.slice_index)
            self._refresh_side()
            QMessageBox.information(
                self, "Здесь уже есть маска",
                f"В этом месте уже есть маска (животное {taken.animal_index + 1}, срез "
                f"{taken.slice_index + 1}) — она выбрана. Щёлкните по пропущенному срезу.",
            )
            return
        found = find_blob_at(review.image, (x, y), [m.mask for m in review.masks if m.mask.any()])
        if found is None:
            QMessageBox.information(
                self, "Ничего не нашлось",
                "Рядом с этой точкой не нашлось ничего похожего на срез. Щёлкните точнее по "
                "срезу или нарисуйте его кистью («Новый срез кистью»).",
            )
            return
        self.canvas.push_undo()
        new = add_mask(review.masks, found)
        self.canvas.set_active_mask(new.animal_index, new.slice_index)
        self.canvas.update()
        self._refresh_side()
        self._mark_changed()

    def _new_slice_by_brush(self) -> None:
        review = self._review()
        if review is None:
            return
        active = self.canvas.active_mask()
        animal = active.animal_index if active is not None else 0
        new_animal = active is None
        self.find_here_btn.setChecked(False)
        self.canvas.push_undo()
        empty = np.zeros(review.image.shape[:2], dtype=bool)
        m = add_mask(review.masks, empty, animal_index=animal, new_animal=new_animal)
        # пустая маска встаёт последней у животного; после первого мазка переедет на
        # своё место по высоте (_on_mask_edited)
        move_mask(review.masks, m, animal, len(slices_of(review.masks, animal)))
        self.canvas.set_active_mask(m.animal_index, m.slice_index)
        self.canvas.pending_new_mask = m
        self.erase_checkbox.setChecked(False)
        self.point_mode_checkbox.setChecked(False)
        self._refresh_side()
        self._mark_changed()

    def _reassign_active(self) -> None:
        review = self._review()
        m = self.canvas.active_mask()
        if review is None or m is None:
            return
        dialog = ReassignDialog(self, review.masks, m, allow_slices=self.mode == MaskMode.WHOLE_BLOB)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        animal, slice_pos, new_animal = dialog.choice()
        self.canvas.push_undo()
        if self.mode == MaskMode.ROSTRAL_CUT and not new_animal:
            # у черепов на животное одна маска — меняемся местами с маской того животного
            other = next((o for o in review.masks if o.animal_index == animal and o is not m), None)
            if other is not None:
                other.animal_index, m.animal_index = m.animal_index, animal
                other.accepted = m.accepted = False
                review.masks.sort(key=lambda x: (x.animal_index, x.slice_index))
            else:
                move_mask(review.masks, m, animal, 0)
        else:
            move_mask(review.masks, m, animal, slice_pos, new_animal=new_animal)
        self.canvas.set_active_mask(m.animal_index, m.slice_index)
        self.canvas.update()
        self._refresh_side()
        self._mark_changed()

    def _delete_active(self) -> None:
        review = self._review()
        m = self.canvas.active_mask()
        if review is None or m is None:
            return
        self.canvas.push_undo()
        delete_mask(review.masks, m)
        if review.masks:
            first = review.masks[0]
            self.canvas.set_active_mask(first.animal_index, first.slice_index)
        else:
            self.canvas.active_key = None
        self.canvas.update()
        self._refresh_side()
        self._mark_changed()

    def _on_exposure_chosen(self, _index: int) -> None:
        review = self._review()
        exposure = self.exposure_combo.currentData()
        if review is None or exposure is None or exposure == review.shot.chosen_exposure:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Сменить выдержку")
        box.setText(
            f"Показать фото на выдержке {exposure} мс.\n\n"
            "Найти маски заново на этой выдержке? Текущие маски и отметки «принято» на "
            "этом фото заменятся (вернуть — Ctrl+Z)."
        )
        redo_btn = box.addButton("Найти маски заново", QMessageBox.ButtonRole.AcceptRole)
        keep_btn = box.addButton("Только показать, маски оставить", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (redo_btn, keep_btn):
            self._refresh_exposure_combo(review)
            return
        try:
            image = load_image(review.shot.exposure_files[exposure])
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Не удалось открыть файл", str(exc))
            self._refresh_exposure_combo(review)
            return
        review.image = image
        review.display_image = contrast_stretch_to_uint8(image)
        review.shot.chosen_exposure = exposure
        if clicked is redo_btn:
            self.canvas.push_undo()
            masks, warning = build_masks_for_image(image, review.shot.group)
            review.masks[:] = masks
            review.warning = warning
        self._show_current(keep_view=True)
        self._mark_changed()

    def _accept_all(self) -> None:
        self.canvas.accept_all()
        self._refresh_side()
        self._mark_changed()

    def _refresh_accept_button_style(self) -> None:
        """Кнопка «Принять всё» — крупная, рядом с «Далее». Пока есть непринятые
        маски, она синяя (главное действие на экране); когда всё принято — зелёная с
        галочкой. Любая правка снимает «принято» с изменённой маски (canvas.py)."""
        all_accepted = bool(self.reviews) and not self._has_unaccepted_nonempty_masks()
        if all_accepted:
            self.accept_btn.setText("✓ Всё на этом фото принято")
            self.accept_btn.setStyleSheet(
                "background-color: #2e7d32; color: white; font-weight: bold; font-size: 11pt;"
            )
        else:
            self.accept_btn.setText("✔ Принять всё на этом фото")
            self.accept_btn.setStyleSheet(
                "background-color: #1565c0; color: white; font-weight: bold; font-size: 11pt;"
            )

    def _recompute_current(self) -> None:
        review = self._review()
        if review is None:
            return
        if any(m.mask.any() for m in review.masks):
            box = QMessageBox(self)
            box.setWindowTitle("Пересчитать автоматически заново?")
            box.setText(
                "Это заменит ВСЕ маски на этом фото свежими автоматическими — ручная "
                "правка и отметки «принято» на этом фото пропадут (вернуть — Ctrl+Z). "
                "Продолжить?"
            )
            box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            if box.exec() != QMessageBox.StandardButton.Yes:
                return
        self.canvas.push_undo()
        masks, warning = build_masks_for_image(review.image, review.shot.group)
        review.masks[:] = masks
        review.warning = warning
        self._show_current(keep_view=True)
        self._mark_changed()

    def _has_unaccepted_nonempty_masks(self) -> bool:
        review = self._review()
        if review is None:
            return False
        return any((not m.accepted) and m.mask.any() for m in review.masks)

    def _confirm_leave_unaccepted(self) -> bool:
        """Возвращает True, если можно уходить с текущего кадра (нет непринятых
        непустых масок, либо пользователь явно согласился принять их или уйти без них)."""
        if not self._has_unaccepted_nonempty_masks():
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Есть непринятые маски")
        box.setText(
            "На этом фото есть непустые маски, которые вы не приняли кнопкой "
            "«Принять всё на этом фото». Если уйти сейчас, они НЕ попадут в результаты."
        )
        accept_btn = box.addButton("Принять всё и продолжить", QMessageBox.ButtonRole.AcceptRole)
        leave_btn = box.addButton("Всё равно уйти без них", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is accept_btn:
            self.canvas.accept_all()
            self._mark_changed()
            return True
        if clicked is leave_btn:
            return True
        return False

    def _go_prev(self) -> None:
        if self.current_index > 0 and self._confirm_leave_unaccepted():
            self.current_index -= 1
            self._show_current()
            self._mark_changed()
            self._on_navigated()

    def _go_next(self) -> None:
        if not self._confirm_leave_unaccepted():
            return
        if self.current_index < len(self.reviews) - 1:
            self.current_index += 1
            self._show_current()
            self._mark_changed()
            self._on_navigated()
        else:
            self._on_all_reviewed(self.reviews)


class ResultsTab(QWidget):
    """Вкладка итоговой таблицы, экспорта и сравнения групп — для ОДНОГО пайплайна
    (что анализируем зафиксировано пайплайном, фильтр по режиму не нужен)."""

    def __init__(self):
        super().__init__()
        self.reviews: list[ShotReview] = []
        self.current_df = pd.DataFrame()
        self.slice_df = pd.DataFrame()  # таблица по срезам (до усреднения по животным) — основа для статистики
        self.recommended_exposure: int | None = None
        self._saturated_by_exposure: dict[int, list[str]] = {}
        # таблица по срезам на выбранной выдержке — пересчитывается в _refresh_table,
        # остальные контролы (срез, «сравнивать по») берут её отсюда, не перечитывая файлы
        self._filtered_cache: pd.DataFrame | None = None

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        refresh_btn = QPushButton("Собрать таблицу по принятым маскам")
        refresh_btn.clicked.connect(self._refresh_table)
        top_row.addWidget(refresh_btn)

        top_row.addWidget(QLabel("Выдержка:"))
        self.exposure_combo = QComboBox()
        self.exposure_combo.setMinimumWidth(260)
        self.exposure_combo.currentIndexChanged.connect(self._refresh_table)
        top_row.addWidget(self.exposure_combo)

        self.average_checkbox = QCheckBox(
            "Показывать таблицу, усреднённую по животным (не влияет на статистику ниже — "
            "сравнение групп ВСЕГДА считается по животным, а не по отдельным срезам)"
        )
        self.average_checkbox.stateChanged.connect(self._refresh_table)
        top_row.addWidget(self.average_checkbox)
        top_row.addStretch(1)

        export_csv_btn = QPushButton("Сохранить таблицу (CSV/Excel)...")
        export_csv_btn.clicked.connect(self._export_table)
        top_row.addWidget(export_csv_btn)
        layout.addLayout(top_row)

        self.exposure_warning_label = QLabel("")
        self.exposure_warning_label.setStyleSheet("color: #b34700; font-weight: bold;")
        self.exposure_warning_label.setWordWrap(True)
        layout.addWidget(self.exposure_warning_label)

        self.table_view = QTableView()
        self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table_view, 2)

        stats_box = QGroupBox("Сравнение групп")
        stats_layout = QVBoxLayout(stats_box)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Что сравнивать:"))
        self.metric_combo = QComboBox()
        # площадь в сравнении не нужна (сессия 7: «не то, что надо сравнивать») — она
        # остаётся колонкой в таблице и выгрузке, но не пунктом сравнения
        self.metric_combo.addItems(["mean_intensity (средняя яркость)", "std_intensity (разброс яркости)"])
        controls_row.addWidget(self.metric_combo)

        controls_row.addWidget(QLabel("Сравнивать по:"))
        self.compare_by_combo = QComboBox()
        self.compare_by_combo.addItem("Группе (как есть)", "group")
        self.compare_by_combo.addItem("Контроль vs Опыт", "control")
        self.compare_by_combo.addItem("Условию (все со всеми)", "condition")
        controls_row.addWidget(self.compare_by_combo)

        controls_row.addWidget(QLabel("Срез:"))
        self.slice_combo = QComboBox()
        self.slice_combo.addItem("Среднее по животному (все срезы)", None)
        self.slice_combo.currentIndexChanged.connect(self._refresh_slice_warning)
        controls_row.addWidget(self.slice_combo)
        compare_btn = QPushButton("Сравнить группы")
        compare_btn.clicked.connect(self._run_comparison)
        controls_row.addWidget(compare_btn)
        save_plot_btn = QPushButton("Сохранить график...")
        save_plot_btn.clicked.connect(self._save_plot)
        controls_row.addWidget(save_plot_btn)
        controls_row.addStretch(1)
        stats_layout.addLayout(controls_row)

        multi_metric_note = QLabel(
            "Учтите: средняя яркость и разброс яркости — два разных вопроса к одним и тем же "
            "группам. Если проверяете оба подряд в поисках значимого различия, относитесь к "
            "этому как к серии из 2 сравнений (риск случайно найти «значимость» растёт), а не "
            "выбирайте после факта только тот показатель, где p < 0.05."
        )
        multi_metric_note.setWordWrap(True)
        multi_metric_note.setStyleSheet("color: #666666; font-style: italic;")
        stats_layout.addWidget(multi_metric_note)

        self.slice_warning_label = QLabel("")
        self.slice_warning_label.setStyleSheet("color: #b34700; font-weight: bold;")
        self.slice_warning_label.setWordWrap(True)
        stats_layout.addWidget(self.slice_warning_label)

        # текст и график — рядом, а не график отдельным файлом после сохранения:
        # так сравнение видно целиком, не переключаясь между таблицей и файлом на
        # диске (раньше график был доступен только через "Сохранить график...")
        results_row = QHBoxLayout()
        self.stats_output = QTextEdit()
        self.stats_output.setReadOnly(True)
        self.stats_output.setMaximumHeight(260)
        results_row.addWidget(self.stats_output, 1)

        self.plot_label = QLabel("График появится здесь после «Сравнить группы»")
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_label.setStyleSheet("color: #898781; font-style: italic;")
        self.plot_label.setMinimumSize(360, 260)
        self.plot_label.setMaximumHeight(260)
        results_row.addWidget(self.plot_label, 1)
        stats_layout.addLayout(results_row)

        layout.addWidget(stats_box)

    def set_reviews(self, reviews: list[ShotReview]) -> None:
        self.reviews = reviews
        self._refresh_table()

    def _current_metric_key(self) -> str:
        return self.metric_combo.currentText().split(" ")[0]

    # --- выдержка анализа (сессия 7) ---
    # Сравнивать яркость можно только внутри ОДНОЙ выдержки: у всех кадров всех групп
    # берётся один и тот же файл exp<N>. Рекомендуемая — самая длинная из общих для
    # всех кадров, на которой ни в одной принятой маске нет засвета (> 0,1% пикселей
    # маски у предела шкалы). Пункт «все выдержки» — только для выгрузки таблицы.

    ALL_EXPOSURES = "all"

    def _accepted_union(self, review: ShotReview) -> np.ndarray | None:
        masks = [m.mask for m in review.masks if m.accepted and m.mask.any()]
        if not masks:
            return None
        union = np.zeros_like(masks[0])
        for m in masks:
            union |= m
        return union

    def _common_exposures(self) -> list[int]:
        sets = [set(r.shot.exposure_files) for r in self.reviews if self._accepted_union(r) is not None]
        return sorted(set.intersection(*sets)) if sets else []

    def _saturated_shots(self, exposure: int) -> list[str]:
        """Кадры, у которых на этой выдержке засвет внутри принятых масок."""
        bad: list[str] = []
        for r in self.reviews:
            union = self._accepted_union(r)
            if union is None or exposure not in r.shot.exposure_files:
                continue
            frac = mask_saturation_fraction(load_image(r.shot.exposure_files[exposure]), union)
            if frac > MASK_SATURATION_FRACTION_LIMIT:
                bad.append(r.shot.display_name)
        return bad

    def _refresh_exposure_options(self) -> None:
        common = self._common_exposures()
        self.recommended_exposure = None
        saturated: dict[int, list[str]] = {}
        for exposure in sorted(common, reverse=True):
            saturated[exposure] = self._saturated_shots(exposure)
            if not saturated[exposure] and self.recommended_exposure is None:
                self.recommended_exposure = exposure

        current = self.exposure_combo.currentData()
        self.exposure_combo.blockSignals(True)
        self.exposure_combo.clear()
        for exposure in sorted(common, reverse=True):
            text = f"{exposure} мс"
            if exposure == self.recommended_exposure:
                text += " — рекомендуемая"
            elif saturated.get(exposure):
                text += " — есть засвет в масках"
            self.exposure_combo.addItem(text, exposure)
        self.exposure_combo.addItem("Все выдержки (только таблица для выгрузки)", self.ALL_EXPOSURES)
        idx = self.exposure_combo.findData(current)
        if idx < 0:
            default = self.recommended_exposure if self.recommended_exposure is not None else (
                min(common) if common else self.ALL_EXPOSURES
            )
            idx = self.exposure_combo.findData(default)
        self.exposure_combo.setCurrentIndex(max(idx, 0))
        self.exposure_combo.blockSignals(False)
        self._saturated_by_exposure = saturated

    def _exposure_note(self) -> str:
        exposure = self.exposure_combo.currentData()
        common = self._common_exposures()
        if exposure == self.ALL_EXPOSURES:
            return (
                "Показаны все выдержки каждого кадра (колонка exposure_ms) — это для выгрузки. "
                "Для сравнения групп выберите одну выдержку."
            )
        if not common:
            return (
                "Нет выдержки, которая есть у ВСЕХ кадров с принятыми масками — сравнивать "
                "яркость между группами не на чем. Проверьте имена файлов (exp<число>)."
            )
        notes = []
        if self.recommended_exposure is None:
            notes.append(
                "ВНИМАНИЕ: на всех общих выдержках есть засвет внутри масок хотя бы у одного "
                "кадра — выбрана самая короткая. Яркость в засвеченных местах занижена."
            )
        bad = self._saturated_by_exposure.get(exposure, [])
        if bad and self.recommended_exposure is not None:
            notes.append(
                f"ВНИМАНИЕ: на {exposure} мс засвет внутри масок у кадров: {', '.join(bad)}. "
                f"Рекомендуемая выдержка — {self.recommended_exposure} мс."
            )
        return "\n".join(notes)

    def _exposure_filtered(self) -> tuple[pd.DataFrame, list[str]]:
        """Таблица по срезам на выбранной выдержке (или на всех — для выгрузки).
        Возвращает (таблица, кадры без файла с этой выдержкой — исключены)."""
        exposure = self.exposure_combo.currentData()
        rows: list[MeasurementRow] = []
        skipped: list[str] = []
        for review in self.reviews:
            if self._accepted_union(review) is None:
                continue
            if exposure == self.ALL_EXPOSURES:
                exposures = sorted(review.shot.exposure_files)
            elif exposure in review.shot.exposure_files:
                exposures = [exposure]
            else:
                skipped.append(review.shot.display_name)
                continue
            for e in exposures:
                image = load_image(review.shot.exposure_files[e])
                rows.extend(measure_shot(review, image=image, exposure_ms=e))
        return rows_to_dataframe(rows), skipped

    def _refresh_table(self) -> None:
        # базовая таблица (для списка номеров срезов) — на выдержке, на которой
        # показаны маски; значения яркости берутся потом на выбранной выдержке
        all_rows: list[MeasurementRow] = []
        for review in self.reviews:
            all_rows.extend(measure_shot(review))
        self.slice_df = rows_to_dataframe(all_rows)

        self._refresh_exposure_options()
        self._refresh_slice_options()

        filtered, skipped = self._exposure_filtered()
        self._filtered_cache = filtered
        warnings = []
        note = self._exposure_note()
        if note:
            warnings.append(note)
        if skipped:
            warnings.append("Нет файла с выбранной выдержкой — исключены: " + ", ".join(skipped))
        self.exposure_warning_label.setText("\n".join(warnings))
        if self.average_checkbox.isChecked():
            df = per_animal_average(filtered)
        else:
            df = filtered
        # сортируем здесь, ДО сохранения в self.current_df — иначе таблица на экране
        # (сортированная в _show_dataframe) и файл, который уходит при экспорте
        # (self.current_df), расходятся по порядку строк
        if "group" in df.columns and not df.empty:
            sort_cols = [c for c in ("group", "animal_index", "slice_index", "exposure_ms") if c in df.columns]
            df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        self.current_df = df
        self._show_dataframe(df)
        self._refresh_slice_warning()

    def _refresh_slice_options(self) -> None:
        """Заполняет список конкретных номеров среза, реально найденных в текущих
        измерениях — для сравнения "по срезу №N" вместо "среднее по животному"
        (номера появляются/исчезают по мере разметки, поэтому список строится
        динамически, а не фиксированным набором)."""
        values: list[int] = []
        if not self.slice_df.empty and "slice_index" in self.slice_df.columns:
            values = sorted(int(v) for v in self.slice_df["slice_index"].dropna().unique())

        current = self.slice_combo.currentData()
        self.slice_combo.blockSignals(True)
        self.slice_combo.clear()
        self.slice_combo.addItem("Среднее по животному (все срезы)", None)
        for slice_index in values:
            self.slice_combo.addItem(f"Только срез №{slice_index}", slice_index)
        idx = self.slice_combo.findData(current)
        self.slice_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.slice_combo.blockSignals(False)

    def _slice_filter_info(self) -> tuple[pd.DataFrame, int]:
        """Таблица по срезам (уже отфильтрованная по выдержке), при необходимости
        суженная до одного выбранного номера среза. Возвращает (таблица, число
        животных, у которых при этом выборе среза не осталось ни одной строки —
        то есть молча исключённых из сравнения)."""
        filtered = self._filtered_cache
        if filtered is None:
            filtered, _ = self._exposure_filtered()
        slice_index = self.slice_combo.currentData()
        if slice_index is None or filtered.empty:
            return filtered, 0
        animal_cols = ["group", "mode", "source_file", "animal_index"]
        total = filtered[animal_cols].drop_duplicates().shape[0]
        sliced = filtered[filtered["slice_index"] == slice_index]
        kept = sliced[animal_cols].drop_duplicates().shape[0]
        return sliced, total - kept

    def _refresh_slice_warning(self) -> None:
        slice_index = self.slice_combo.currentData()
        if slice_index is None:
            self.slice_warning_label.setText("")
            return
        _, dropped = self._slice_filter_info()
        if dropped:
            self.slice_warning_label.setText(
                f"Срез №{slice_index}: исключено из сравнения животных без этого среза — {dropped}."
            )
        else:
            self.slice_warning_label.setText(
                f"Срез №{slice_index}: у всех животных есть этот срез, никто не исключён."
            )

    def _show_dataframe(self, df: pd.DataFrame) -> None:
        # сортировка по группе уже применена вызывающим кодом (_refresh_table) —
        # тут только рендер, чтобы таблица на экране и self.current_df (экспорт)
        # были гарантированно в одном и том же порядке строк
        model = QStandardItemModel(df.shape[0], df.shape[1])
        model.setHorizontalHeaderLabels(list(df.columns))
        group_colors: dict[str, QColor] = {}
        if "group" in df.columns:
            for i, g in enumerate(sorted(df["group"].unique())):
                color = QColor(color_for_group(i))
                color.setAlpha(45)
                group_colors[g] = color
        for r in range(df.shape[0]):
            row_color = group_colors.get(df.iat[r, df.columns.get_loc("group")]) if group_colors else None
            for c, col in enumerate(df.columns):
                item = QStandardItem(str(df.iat[r, c]))
                if row_color is not None:
                    item.setBackground(row_color)
                model.setItem(r, c, item)
        self.table_view.setModel(model)

    def _export_table(self) -> None:
        if self.current_df.empty:
            QMessageBox.information(self, "Пусто", "Сначала соберите таблицу — нет ни одной принятой маски.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить таблицу", "результаты.xlsx", "Excel (*.xlsx);;CSV (*.csv)"
        )
        if not path:
            return
        try:
            export_table(self.current_df, Path(path))
        except Exception as exc:  # noqa: BLE001
            # без этого исключение (файл занят в Excel, диск недоступен и т.п.) гасится
            # где-то на границе диспетчера сигналов Qt МОЛЧА — ни ошибки, ни "Готово" не
            # появляется, а собранный exe без консоли (console=False) не показывает даже
            # traceback в stderr; пользователь считает, что данные сохранены, хотя нет
            QMessageBox.warning(
                self, "Не удалось сохранить",
                f"Таблица не сохранена:\n{exc}\n\nЕсли файл открыт в Excel или другой "
                "программе — закройте его и попробуйте снова.",
            )
            return
        QMessageBox.information(self, "Готово", f"Таблица сохранена:\n{path}")

    def _animal_df(self) -> pd.DataFrame:
        """Таблица для статистики — ВСЕГДА агрегированная по животным (срез — не
        независимое наблюдение), и (если выбрано) на одной зафиксированной выдержке
        и/или одном конкретном номере среза для всех групп.

        Если выбран конкретный срез №N, `per_animal_average` получает уже суженную
        до этого среза таблицу — животных с несколькими срезами это не меняет
        (после сужения на животное остаётся ровно один срез, "усреднение" по нему
        тривиально), а животных БЕЗ среза №N молча исключает — количество таких
        животных показывается отдельно в `slice_warning_label`/`_run_comparison`,
        не молча.
        """
        filtered, _ = self._slice_filter_info()
        return per_animal_average(filtered)

    def _comparison_df(self, animal_df: pd.DataFrame) -> pd.DataFrame:
        """Подменяет колонку "group" на выбранное измерение сравнения ("Сравнивать
        по"). compare_groups/save_boxplot всегда работают с колонкой "group" — так
        сравнение "контроль vs опыт" или "по условию" не требует их менять, только
        один раз здесь переименовать нужную колонку в "group"."""
        key = self.compare_by_combo.currentData()
        if key == "group" or animal_df.empty:
            return animal_df
        out = animal_df.copy()
        if key == "control":
            out["group"] = out["is_control"].map({True: "Контроль", False: "Опыт"})
        elif key == "condition":
            out = out[out["condition"].astype(str).str.strip() != ""]
            out["group"] = out["condition"]
        return out

    def _not_comparable_reason(self, animal_df: pd.DataFrame) -> str | None:
        """Почему сравнение невозможно — конкретно, а не общим «нужно 2 группы»
        (сессия 7: сообщение сбивало с толку, когда масок было достаточно, а групп
        для выбранного «Сравнивать по» получалась одна)."""
        if self.exposure_combo.currentData() == self.ALL_EXPOSURES:
            return (
                "Выбран пункт «Все выдержки» — он только для выгрузки таблицы. Для сравнения "
                "выберите одну выдержку (рекомендуемая отмечена в списке)."
            )
        key = self.compare_by_combo.currentData()
        # при «по условию» группы с пустой меткой уже отброшены (_comparison_df), поэтому
        # пустая таблица тут может значить «метки не заполнены», а не «масок нет»
        if animal_df.empty and not (key == "condition" and not self._animal_df().empty):
            return "Нет ни одной принятой маски на выбранной выдержке / выбранном срезе."
        if not animal_df.empty and animal_df["group"].nunique() >= 2:
            return None
        if key == "condition":
            filled = self._animal_df()["condition"].astype(str).str.strip()
            if (filled == "").all():
                return (
                    "Выбрано «Сравнивать по: Условию», но колонка «Условие (метка)» на вкладке "
                    "«Проект» не заполнена ни у одной группы. Заполните её или выберите "
                    "«Сравнивать по: Группе»."
                )
            return (
                "Выбрано «Сравнивать по: Условию», но у всех групп с принятыми масками одно и "
                "то же условие (или оно заполнено только у одной) — сравнивать не с чем."
            )
        if key == "control":
            return (
                "Выбрано «Контроль vs Опыт», но галочка «Контроль?» на вкладке «Проект» стоит "
                "у всех групп или ни у одной — получается только одна сторона сравнения."
            )
        return "Принятые маски есть только у одной группы — сравнивать не с чем."

    def _run_comparison(self) -> None:
        animal_df = self._comparison_df(self._animal_df())
        reason = self._not_comparable_reason(animal_df)
        if reason:
            QMessageBox.information(self, "Сравнить не получится", reason)
            self._clear_plot_preview()
            return
        metric = self._current_metric_key()
        try:
            result = compare_groups(animal_df, metric)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Не удалось сравнить", str(exc))
            self._clear_plot_preview()
            return

        slice_index = self.slice_combo.currentData()
        unit_note = (
            "единица анализа — животное, срезы усреднены" if slice_index is None
            else f"единица анализа — животное, только срез №{slice_index}"
        )
        lines = [
            f"Метод: {result.test_name} ({unit_note})",
            f"Группы: {', '.join(f'{g} (n={result.n_per_group[g]})' for g in result.groups)}",
            f"Общий p-value: {result.p_value:.4f}" + ("  (есть значимое различие, p < 0.05)" if result.p_value < 0.05 else "  (значимого различия не обнаружено)"),
        ]
        if slice_index is not None:
            _, dropped = self._slice_filter_info()
            if dropped:
                lines.append(
                    f"\nВНИМАНИЕ: при выборе среза №{slice_index} исключено животных без "
                    f"этого среза — {dropped}. Результат относится только к оставшимся."
                )

        exposure = self.exposure_combo.currentData()
        lines.insert(1, f"Выдержка: {exposure} мс (одна и та же у всех кадров)")
        bad = self._saturated_by_exposure.get(exposure, [])
        if bad:
            lines.append(
                f"\nВНИМАНИЕ: на этой выдержке есть засвет внутри масок ({', '.join(bad)}) — "
                "яркость там занижена. Лучше взять рекомендуемую выдержку."
            )
        min_p = max((pw.min_possible_p for pw in result.pairwise), default=0.0)
        if min_p > 0.05:
            lines.append(
                f"\nВНИМАНИЕ: при таком числе животных наименьшее в принципе достижимое "
                f"p-value = {min_p:.3f} — тест физически не может показать p < 0.05, даже "
                f"если реальный эффект есть. \"Незначимо\" здесь не означает \"эффекта нет\"."
            )
        if len(result.pairwise) > 1:
            note = " (с поправкой Холма на множественные сравнения)" if result.holm_applied else ""
            lines.append(f"\nПопарные сравнения{note}:")
            for pw in result.pairwise:
                extra = f", было бы p = {pw.p_value_raw:.4f} без поправки" if result.holm_applied else ""
                lines.append(
                    f"  {pw.group_a} vs {pw.group_b}: p = {pw.p_value:.4f}{extra}; "
                    f"размер эффекта (ранговая бисериальная корреляция) = {pw.effect_size:+.2f}"
                )
        else:
            pw = result.pairwise[0]
            lines.append(f"Размер эффекта (ранговая бисериальная корреляция): {pw.effect_size:+.2f}")
        self.stats_output.setHtml(_format_stats_html(lines))

        png_bytes = boxplot_png_bytes(animal_df, metric, title=self.metric_combo.currentText() + self._plot_title_suffix())
        pixmap = QPixmap()
        pixmap.loadFromData(png_bytes, "PNG")
        self.plot_label.setPixmap(
            pixmap.scaled(
                self.plot_label.width() or 360, self.plot_label.height() or 260,
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _clear_plot_preview(self) -> None:
        self.plot_label.clear()
        self.plot_label.setText("График появится здесь после «Сравнить группы»")

    def _plot_title_suffix(self) -> str:
        slice_index = self.slice_combo.currentData()
        where = " (по животным" if slice_index is None else f" (срез №{slice_index}"
        return f"{where}, {self.exposure_combo.currentData()} мс)"

    def _save_plot(self) -> None:
        animal_df = self._comparison_df(self._animal_df())
        reason = self._not_comparable_reason(animal_df)
        if reason:
            QMessageBox.information(self, "Сохранить график не получится", reason)
            return
        metric = self._current_metric_key()
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить график", "сравнение_групп.png", "PNG (*.png)")
        if not path:
            return
        try:
            save_boxplot(animal_df, metric, Path(path), title=self.metric_combo.currentText() + self._plot_title_suffix())
        except Exception as exc:  # noqa: BLE001
            # тот же класс тихого отказа, что и в _export_table — без try/except
            # исключение гасится без единого диалога, пользователь решает, что график
            # сохранён, хотя файла нет
            QMessageBox.warning(
                self, "Не удалось сохранить",
                f"График не сохранён:\n{exc}\n\nЕсли файл открыт в другой программе — "
                "закройте его и попробуйте снова.",
            )
            return
        QMessageBox.information(self, "Готово", f"График сохранён:\n{path}")


class PipelineTabs(QTabWidget):
    """Один независимый цикл Проект → Проверка масок → Результаты для одного
    пайплайна (режим обводки зафиксирован на весь пайплайн)."""

    AUTOSAVE_INTERVAL_MS = 60_000

    def __init__(self, mode: MaskMode):
        super().__init__()
        self.mode = mode
        self.groups: list[GroupConfig] = []
        # файл, куда сохраняет «Сохранить разметку» (выбран пользователем); пока не
        # выбран — автосохранение пишет в autosave_path(mode)
        self.save_path: Path | None = None
        self._dirty = False

        self.project_tab = ProjectTab(
            mode=mode, on_start_review=self._start_review,
            on_open_markup=self._open_markup, on_restore_autosave=self._restore_autosave,
        )
        self.review_tab = ReviewTab(
            mode=mode, on_all_reviewed=self._finish_review,
            on_changed=self._mark_dirty, on_save=self.save_markup_interactive,
            on_navigated=self.autosave,
        )
        self.results_tab = ResultsTab()

        self.addTab(self.project_tab, "1. Проект")
        self.addTab(self.review_tab, "2. Проверка масок")
        self.addTab(self.results_tab, "3. Результаты")
        self.setTabEnabled(1, False)
        self.setTabEnabled(2, False)
        self.project_tab.restore_btn.setEnabled(autosave_path(mode).exists())

        # страховка от сбоя/зависания: раз в минуту, если что-то менялось
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self.autosave)
        self._autosave_timer.start(self.AUTOSAVE_INTERVAL_MS)

    # ---------- сохранение ----------

    def _mark_dirty(self) -> None:
        self._dirty = True

    def has_markup(self) -> bool:
        return bool(self.review_tab.reviews)

    def _write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_markup(path, self.mode, self.groups, self.review_tab.reviews, self.review_tab.current_index)

    def autosave(self) -> None:
        """Тихое сохранение: в выбранный файл, а если его нет — в автосохранение.
        Ошибка не показывается диалогом (не мешать работе), но и не роняет программу."""
        if not self._dirty or not self.has_markup():
            return
        try:
            self._write(self.save_path or autosave_path(self.mode))
            self._dirty = False
        except Exception:  # noqa: BLE001
            pass

    def save_markup_interactive(self) -> None:
        if not self.has_markup():
            QMessageBox.information(self, "Нечего сохранять", "Сначала начните проверку масок.")
            return
        start = str(self.save_path) if self.save_path else f"разметка_{PIPELINE_TITLES[self.mode].lower()}{FILE_SUFFIX}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить разметку", start, f"Разметка BrainPhotoTool (*{FILE_SUFFIX})"
        )
        if not path:
            return
        path = Path(path)
        if path.suffix != FILE_SUFFIX:
            path = path.with_name(path.name + FILE_SUFFIX)
        try:
            self._write(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Не удалось сохранить", f"Разметка не сохранена:\n{exc}")
            return
        self.save_path = path
        self._dirty = False
        QMessageBox.information(
            self, "Сохранено",
            f"Разметка сохранена:\n{path}\n\nДальше программа будет сама сохранять в этот "
            "файл — при переходе между фото, раз в минуту и при закрытии.",
        )

    def _confirm_replace_markup(self) -> bool:
        # повторный запуск/загрузка заменяет всю текущую разметку
        if not (self.review_tab.reviews and any(m.mask.any() for r in self.review_tab.reviews for m in r.masks)):
            return True
        self.autosave()
        box = QMessageBox(self)
        box.setWindowTitle("Заменить текущую разметку?")
        box.setText(
            "На вкладке «2. Проверка масок» уже есть разметка — она будет заменена. "
            "Если она нужна, сначала сохраните её («Сохранить разметку»). Продолжить?"
        )
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _open_markup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть разметку", "", f"Разметка BrainPhotoTool (*{FILE_SUFFIX})"
        )
        if path:
            self._load_markup_file(Path(path), remember_path=True)

    def _restore_autosave(self) -> None:
        path = autosave_path(self.mode)
        if not path.exists():
            QMessageBox.information(self, "Нет автосохранения", "Автосохранённой разметки пока нет.")
            return
        self._load_markup_file(path, remember_path=False)

    def _load_markup_file(self, path: Path, remember_path: bool) -> None:
        if not self._confirm_replace_markup():
            return
        try:
            loaded = load_markup(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Не удалось открыть разметку", f"{path}\n\n{exc}")
            return
        if loaded.mode != self.mode:
            QMessageBox.warning(
                self, "Другой пайплайн",
                f"Этот файл — разметка пайплайна «{PIPELINE_TITLES[loaded.mode]}». Откройте его "
                f"на вкладке «{PIPELINE_TITLES[loaded.mode]}».",
            )
            return
        if loaded.problems:
            QMessageBox.warning(self, "Открыто не всё", "\n".join(loaded.problems))
        if not loaded.reviews:
            return
        self.groups = loaded.groups
        self.project_tab.set_groups(loaded.groups)
        self.save_path = path if remember_path else None
        self._dirty = False
        self.review_tab.load_reviews(loaded.reviews, loaded.current_index)
        self.setTabEnabled(1, True)
        if any(m.accepted for r in loaded.reviews for m in r.masks):
            self.results_tab.set_reviews(loaded.reviews)
            self.setTabEnabled(2, True)
        self.setCurrentIndex(1)

    # ---------- проверка ----------

    def _start_review(self, groups: list[GroupConfig]) -> None:
        if not self._confirm_replace_markup():
            return

        reviews: list[ShotReview] = []
        problems: list[str] = []
        for group in groups:
            try:
                shots = scan_group_folder(group)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"Группа «{group.name}»: не удалось прочитать папку ({exc})")
                continue
            if not shots:
                problems.append(f"Группа «{group.name}»: в папке не найдено подходящих фото")
                continue
            for shot in shots:
                # один битый/нестандартный файл не должен ронять весь запуск — такой
                # кадр пропускается со списком остальных проблем
                try:
                    exposure = choose_mask_exposure(shot)
                    shot.chosen_exposure = exposure
                    image = load_image(shot.exposure_files[exposure])
                    masks, warning = build_masks_for_image(image, group)
                    display = contrast_stretch_to_uint8(image)
                except Exception as exc:  # noqa: BLE001
                    problems.append(
                        f"Группа «{group.name}», кадр «{shot.shot_key}»: не удалось "
                        f"обработать ({exc}) — кадр пропущен."
                    )
                    continue
                reviews.append(ShotReview(shot=shot, image=image, display_image=display, masks=masks, warning=warning))

        if problems:
            QMessageBox.warning(self, "Есть проблемы", "\n".join(problems))
        if not reviews:
            return

        self.groups = groups
        self.save_path = None
        self._dirty = True
        self.review_tab.load_reviews(reviews)
        self.setTabEnabled(1, True)
        self.setTabEnabled(2, False)
        self.setCurrentIndex(1)

    def _finish_review(self, reviews: list[ShotReview]) -> None:
        self.autosave()
        self.results_tab.set_reviews(reviews)
        self.setTabEnabled(2, True)
        self.setCurrentIndex(2)


class MainWindow(QTabWidget):
    """Верхний уровень: выбор пайплайна («Срезы» / «Обонятельный эпителий») —
    каждый со своим полностью независимым циклом Проект→Проверка→Результаты
    и своим списком групп/фото, без общего хранилища между пайплайнами."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обработка фото мозга")
        self.resize(1280, 820)

        self.pipelines: list[PipelineTabs] = []
        for mode, title in PIPELINE_TITLES.items():
            pipeline = PipelineTabs(mode)
            self.pipelines.append(pipeline)
            self.addTab(pipeline, title)

    def closeEvent(self, event) -> None:  # noqa: N802
        # при закрытии разметка сохраняется сама — раньше закрытие окна молча теряло
        # всю проверку масок
        for pipeline in self.pipelines:
            pipeline._dirty = pipeline._dirty or pipeline.has_markup()
            pipeline.autosave()
        super().closeEvent(event)
