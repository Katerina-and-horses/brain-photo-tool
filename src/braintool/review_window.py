"""Главное окно программы: два независимых пайплайна («Срезы» и «Обонятельный
эпителий»), каждый — свой цикл из трёх вкладок: Проект, Проверка масок, Результаты."""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QMessageBox, QPushButton, QSizePolicy,
    QSpinBox, QSplitter, QTableView, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)

from .canvas import MaskCanvas
from .colors import color_for_animal
from .imaging import choose_best_exposure, contrast_stretch_to_uint8, load_image, scan_group_folder
from .measurements import export_table, measure_shot, per_animal_average, rows_to_dataframe
from .models import GroupConfig, MaskMode, MeasurementRow, ShotReview
from .segmentation import build_masks_for_image
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

    def __init__(self, mode: MaskMode, on_start_review):
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

    def _add_group(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с фото группы")
        if not folder:
            return
        name, ok = QInputDialog.getText(self, "Название группы", "Как назвать эту группу?", text=Path(folder).name)
        if not ok or not name.strip():
            return

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name.strip()))
        self.table.setItem(row, 1, QTableWidgetItem(folder))

        cols_spin = QSpinBox()
        cols_spin.setRange(1, 200)
        cols_spin.setValue(5)  # в лаборатории в группе обычно 5 животных
        self.table.setCellWidget(row, 2, cols_spin)

        control_checkbox = QCheckBox()
        control_cell = QWidget()
        control_layout = QHBoxLayout(control_cell)
        control_layout.addWidget(control_checkbox)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 3, control_cell)

        self.table.setItem(row, 4, QTableWidgetItem(""))

    def _remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _collect_groups(self) -> list[GroupConfig] | None:
        groups: list[GroupConfig] = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text()
            folder = self.table.item(row, 1).text()
            cols = self.table.cellWidget(row, 2).value()
            is_control = self.table.cellWidget(row, 3).findChild(QCheckBox).isChecked()
            condition_item = self.table.item(row, 4)
            condition = condition_item.text().strip() if condition_item else ""
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


class ReviewTab(QWidget):
    """Вкладка постраничного просмотра и правки автоматически предложенных масок."""

    def __init__(self, on_all_reviewed):
        super().__init__()
        self._on_all_reviewed = on_all_reviewed
        self.reviews: list[ShotReview] = []
        self.current_index = 0

        layout = QVBoxLayout(self)

        self.progress_label = QLabel("Кадров пока нет")
        layout.addWidget(self.progress_label)

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: #b34700; font-weight: bold;")
        self.warning_label.setWordWrap(True)
        layout.addWidget(self.warning_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        self.canvas = MaskCanvas()
        self.canvas.maskEdited.connect(self._on_mask_edited)
        splitter.addWidget(self.canvas)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(QLabel("Животные на этом фото (цвет = маска этого животного):"))
        self.animal_list = QVBoxLayout()
        animal_container = QWidget()
        animal_container.setLayout(self.animal_list)
        side_layout.addWidget(animal_container)

        self.erase_checkbox = QCheckBox("Ластик (стирать кистью, а не рисовать)")
        self.erase_checkbox.toggled.connect(self.canvas.set_erase_enabled)
        side_layout.addWidget(self.erase_checkbox)

        self.point_mode_checkbox = QCheckBox(
            "Режим точек (тянуть контур за вершины вместо кисти)"
        )
        self.point_mode_checkbox.toggled.connect(self.canvas.set_point_mode_enabled)
        # включён по умолчанию — оказался удобнее кисти для большинства правок,
        # человек сам выключит галочку, если для конкретного кадра нужна кисть
        self.point_mode_checkbox.setChecked(True)
        side_layout.addWidget(self.point_mode_checkbox)

        help_label = QLabel(
            "Подсказка:\n"
            "— левая кнопка мыши рисует, правая стирает;\n"
            "— колесо мыши меняет размер кисти;\n"
            "— Ctrl + колесо — приблизить/отдалить (под курсором);\n"
            "— средняя кнопка мыши (зажать и вести) — сдвинуть картинку;\n"
            "— у линии отреза (для черепов) можно потянуть за белые точки;\n"
            "— в «Режиме точек»: контур сразу виден у ВСЕХ животных/срезов на\n"
            "  фото (активный — ярче); тянуть точку — зажать и вести мышью;\n"
            "  добавить точку — двойной клик на линии контура;\n"
            "  убрать точку — клик правой кнопкой по ней (кисть в этом режиме не рисует)."
        )
        help_label.setWordWrap(True)
        side_layout.addWidget(help_label)
        side_layout.addStretch(1)

        recompute_btn = QPushButton("Пересчитать автоматически заново")
        recompute_btn.clicked.connect(self._recompute_current)
        side_layout.addWidget(recompute_btn)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Назад")
        self.prev_btn.clicked.connect(self._go_prev)
        self.accept_btn = QPushButton("Принять всё на этом фото")
        self.accept_btn.clicked.connect(self._accept_all)
        self.next_btn = QPushButton("Далее ▶")
        self.next_btn.clicked.connect(self._go_next)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.accept_btn)
        nav_row.addWidget(self.next_btn)
        layout.addLayout(nav_row)

    def load_reviews(self, reviews: list[ShotReview]) -> None:
        self.reviews = reviews
        self.current_index = 0
        self._show_current()

    def _show_current(self) -> None:
        if not self.reviews:
            self.progress_label.setText("Кадров нет — вернитесь на вкладку «Проект»")
            return
        review = self.reviews[self.current_index]
        self.progress_label.setText(
            f"Кадр {self.current_index + 1} из {len(self.reviews)} — {review.shot.display_name}"
        )
        self.warning_label.setText(review.warning or "")
        self.canvas.set_shot(review.display_image, review.masks)
        self._rebuild_animal_list(review)
        self._refresh_accept_button_style()

    def _rebuild_animal_list(self, review: ShotReview) -> None:
        while self.animal_list.count():
            item = self.animal_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        by_animal: dict[int, list] = {}
        for m in review.masks:
            by_animal.setdefault(m.animal_index, []).append(m)

        for ai in sorted(by_animal):
            r, g, b = color_for_animal(ai)
            slices = sorted(by_animal[ai], key=lambda m: m.slice_index)
            if len(slices) == 1:
                m = slices[0]
                label = f"Животное {ai + 1}" + (" ✓" if m.accepted else "")
                btn = QPushButton(label)
                btn.setStyleSheet(
                    f"background-color: rgb({r},{g},{b}); font-weight: bold; text-align: left; padding: 4px;"
                )
                btn.clicked.connect(lambda _checked=False, a=ai, s=m.slice_index: self.canvas.set_active_mask(a, s))
                self.animal_list.addWidget(btn)
            else:
                header = QLabel(f"Животное {ai + 1}:")
                header.setStyleSheet(f"background-color: rgb({r},{g},{b}); font-weight: bold; padding: 2px;")
                self.animal_list.addWidget(header)
                for m in slices:
                    label = f"  срез {m.slice_index + 1}" + (" ✓" if m.accepted else "")
                    btn = QPushButton(label)
                    btn.setStyleSheet("text-align: left; padding: 3px;")
                    btn.clicked.connect(
                        lambda _checked=False, a=ai, s=m.slice_index: self.canvas.set_active_mask(a, s)
                    )
                    self.animal_list.addWidget(btn)

    def _accept_all(self) -> None:
        self.canvas.accept_all()
        self._rebuild_animal_list(self.reviews[self.current_index])
        self._refresh_accept_button_style()

    def _on_mask_edited(self) -> None:
        self._rebuild_animal_list(self.reviews[self.current_index])
        self._refresh_accept_button_style()

    def _refresh_accept_button_style(self) -> None:
        """Кнопка "Принять всё" горит зелёным, пока все непустые маски на этом фото
        приняты; любая правка (кистью или перетаскиванием линии отреза) снимает
        "принято" с изменённой маски (см. canvas.py), и кнопка гаснет сама собой."""
        all_accepted = bool(self.reviews) and not self._has_unaccepted_nonempty_masks()
        if all_accepted:
            self.accept_btn.setStyleSheet(
                "background-color: #2e7d32; color: white; font-weight: bold;"
            )
        else:
            self.accept_btn.setStyleSheet("")

    def _recompute_current(self) -> None:
        if not self.reviews:
            return
        review = self.reviews[self.current_index]
        masks, warning = build_masks_for_image(review.image, review.shot.group)
        review.masks = masks
        review.warning = warning
        self._show_current()

    def _has_unaccepted_nonempty_masks(self) -> bool:
        if not self.reviews:
            return False
        review = self.reviews[self.current_index]
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
            return True
        if clicked is leave_btn:
            return True
        return False

    def _go_prev(self) -> None:
        if self.current_index > 0 and self._confirm_leave_unaccepted():
            self.current_index -= 1
            self._show_current()

    def _go_next(self) -> None:
        if not self._confirm_leave_unaccepted():
            return
        if self.current_index < len(self.reviews) - 1:
            self.current_index += 1
            self._show_current()
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

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        refresh_btn = QPushButton("Собрать таблицу по принятым маскам")
        refresh_btn.clicked.connect(self._refresh_table)
        top_row.addWidget(refresh_btn)

        top_row.addWidget(QLabel("Выдержка для сравнения:"))
        self.exposure_combo = QComboBox()
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
        self.metric_combo.addItems(["area_px (площадь)", "mean_intensity (средняя яркость)", "std_intensity (разброс яркости)"])
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
            "Учтите: площадь, средняя яркость и разброс яркости — три разных вопроса к одним "
            "и тем же группам. Если вы проверяете все три подряд в поисках значимого различия, "
            "относитесь к этому как к одной серии из 3 сравнений (риск случайно найти "
            "«значимость» растёт), а не выбирайте после факта только тот показатель, где p < 0.05."
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

    def _refresh_exposure_options(self) -> None:
        """Заполняет список выдержек, ОБЩИХ для всех кадров этого пайплайна.

        В папке группы обычно лежит несколько файлов с разной выдержкой на один и тот
        же кадр (`Shot.exposure_files`); программа по умолчанию сама выбирает для
        КАЖДОГО кадра свою "лучшую" (самую длинную незасвеченную). Для сравнения групп
        по яркости это может быть некорректно, если у групп в итоге выбралась разная
        выдержка — здесь можно явно зафиксировать одну выдержку на все группы сразу
        (например, все "срезы" на 250мс, все "черепа" на 150мс).
        """
        exposure_sets = [set(r.shot.exposure_files.keys()) for r in self.reviews if r.shot.exposure_files]
        common = sorted(set.intersection(*exposure_sets)) if exposure_sets else []

        current = self.exposure_combo.currentData()
        self.exposure_combo.blockSignals(True)
        self.exposure_combo.clear()
        self.exposure_combo.addItem("Авто (у каждого кадра своя)", None)
        for exposure in common:
            self.exposure_combo.addItem(f"{exposure} мс (общая для всех групп)", exposure)
        idx = self.exposure_combo.findData(current)
        self.exposure_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.exposure_combo.blockSignals(False)

    def _exposure_filtered(self) -> tuple[pd.DataFrame, list[str]]:
        """Таблица по срезам, при необходимости пересчитанная на одной зафиксированной
        выдержке. Возвращает (таблица, список кадров, у которых нет файла с выбранной
        выдержкой и которые поэтому исключены)."""
        exposure = self.exposure_combo.currentData()
        if exposure is None:
            return self.slice_df, []

        rows: list[MeasurementRow] = []
        skipped: list[str] = []
        for review in self.reviews:
            if exposure not in review.shot.exposure_files:
                skipped.append(review.shot.display_name)
                continue
            image = load_image(review.shot.exposure_files[exposure])
            rows.extend(measure_shot(review, image=image, exposure_ms=exposure))
        return rows_to_dataframe(rows), skipped

    def _refresh_table(self) -> None:
        all_rows: list[MeasurementRow] = []
        for review in self.reviews:
            all_rows.extend(measure_shot(review))
        self.slice_df = rows_to_dataframe(all_rows)

        self._refresh_exposure_options()
        self._refresh_slice_options()

        filtered, skipped = self._exposure_filtered()
        self.exposure_warning_label.setText(
            "Без выбранной выдержки исключены из сравнения: " + ", ".join(skipped) if skipped else ""
        )
        if self.average_checkbox.isChecked():
            df = per_animal_average(filtered)
        else:
            df = filtered
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
        # визуально группируем строки по группе — раньше порядок был "как пришло",
        # из-за чего строки одной группы могли перемежаться со строками другой
        if "group" in df.columns and not df.empty:
            sort_cols = [c for c in ("group", "animal_index", "slice_index") if c in df.columns]
            df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

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
        export_table(self.current_df, Path(path))
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

    def _run_comparison(self) -> None:
        animal_df = self._comparison_df(self._animal_df())
        if animal_df.empty or animal_df["group"].nunique() < 2:
            QMessageBox.information(self, "Недостаточно данных", "Нужно минимум 2 группы с принятыми масками.")
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

        if metric in ("mean_intensity", "std_intensity") and "exposure_ms" in animal_df.columns:
            compared = animal_df[animal_df["group"].isin(result.groups)]
            per_group_exposures = {
                g: sorted(set(compared.loc[compared["group"] == g, "exposure_ms"].dropna()))
                for g in result.groups
            }
            all_exposures = sorted({e for exps in per_group_exposures.values() for e in exps})
            if len(all_exposures) > 1:
                details = ", ".join(f"{g}: {exps} мс" for g, exps in per_group_exposures.items())
                lines.append(
                    f"\nВНИМАНИЕ: группы сняты на РАЗНОЙ выдержке ({details}) — программа "
                    "сама выбирает наибольшую незасвеченную выдержку для каждого кадра "
                    "отдельно. Сравнение яркости между группами при разной выдержке может "
                    "быть искажено (более долгая выдержка — систематически выше сигнал), "
                    "даже если реального биологического различия нет."
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
        return " (по животным)" if slice_index is None else f" (срез №{slice_index})"

    def _save_plot(self) -> None:
        animal_df = self._comparison_df(self._animal_df())
        if animal_df.empty:
            QMessageBox.information(self, "Пусто", "Сначала соберите таблицу.")
            return
        metric = self._current_metric_key()
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить график", "сравнение_групп.png", "PNG (*.png)")
        if not path:
            return
        save_boxplot(animal_df, metric, Path(path), title=self.metric_combo.currentText() + self._plot_title_suffix())
        QMessageBox.information(self, "Готово", f"График сохранён:\n{path}")


class PipelineTabs(QTabWidget):
    """Один независимый цикл Проект → Проверка масок → Результаты для одного
    пайплайна (режим обводки зафиксирован на весь пайплайн)."""

    def __init__(self, mode: MaskMode):
        super().__init__()
        self.mode = mode

        self.project_tab = ProjectTab(mode=mode, on_start_review=self._start_review)
        self.review_tab = ReviewTab(on_all_reviewed=self._finish_review)
        self.results_tab = ResultsTab()

        self.addTab(self.project_tab, "1. Проект")
        self.addTab(self.review_tab, "2. Проверка масок")
        self.addTab(self.results_tab, "3. Результаты")
        self.setTabEnabled(1, False)
        self.setTabEnabled(2, False)

    def _start_review(self, groups: list[GroupConfig]) -> None:
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
                path = choose_best_exposure(shot)
                image = load_image(path)
                masks, warning = build_masks_for_image(image, group)
                display = contrast_stretch_to_uint8(image)
                reviews.append(ShotReview(shot=shot, image=image, display_image=display, masks=masks, warning=warning))

        if problems:
            QMessageBox.warning(self, "Есть проблемы", "\n".join(problems))
        if not reviews:
            return

        self.review_tab.load_reviews(reviews)
        self.setTabEnabled(1, True)
        self.setCurrentIndex(1)

    def _finish_review(self, reviews: list[ShotReview]) -> None:
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

        for mode, title in PIPELINE_TITLES.items():
            self.addTab(PipelineTabs(mode), title)
