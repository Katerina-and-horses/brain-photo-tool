"""Главное окно программы: три вкладки — Проект, Проверка масок, Результаты."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QStandardItem, QStandardItemModel
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
from .stats import compare_groups, save_boxplot

MODE_LABELS = {
    MaskMode.WHOLE_BLOB: "Весь срез целиком",
    MaskMode.ROSTRAL_CUT: "Носовая (передняя) часть черепа",
}
MODE_BY_LABEL = {v: k for k, v in MODE_LABELS.items()}


class ProjectTab(QWidget):
    """Вкладка настройки групп: какие папки, какая сетка, какой режим обводки."""

    def __init__(self, on_start_review):
        super().__init__()
        self._on_start_review = on_start_review
        self.groups: list[GroupConfig] = []

        layout = QVBoxLayout(self)
        info = QLabel(
            "Добавьте по одной папке на каждую группу животных. Для каждой папки укажите:\n"
            "— что обводить (весь срез мозга или носовую часть черепа сверху);\n"
            "— сколько животных на одном фото (по горизонтали, слева направо).\n"
            "Количество срезов/повторов на одно животное программа определяет сама по фото."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Группа", "Папка", "Что обводить", "Животных в ряд"])
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

        mode_combo = QComboBox()
        mode_combo.addItems(list(MODE_LABELS.values()))
        self.table.setCellWidget(row, 2, mode_combo)

        cols_spin = QSpinBox()
        cols_spin.setRange(1, 200)
        cols_spin.setValue(5)  # в лаборатории в группе обычно 5 животных
        self.table.setCellWidget(row, 3, cols_spin)

    def _remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _collect_groups(self) -> list[GroupConfig] | None:
        groups: list[GroupConfig] = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text()
            folder = self.table.item(row, 1).text()
            mode_label = self.table.cellWidget(row, 2).currentText()
            cols = self.table.cellWidget(row, 3).value()
            try:
                groups.append(
                    GroupConfig(
                        name=name, folder=Path(folder), mode=MODE_BY_LABEL[mode_label],
                        cols=cols,
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
        splitter.addWidget(self.canvas)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(QLabel("Животные на этом фото (цвет = маска этого животного):"))
        self.animal_list = QVBoxLayout()
        animal_container = QWidget()
        animal_container.setLayout(self.animal_list)
        side_layout.addWidget(animal_container)

        self.erase_checkbox = QCheckBox("Ластик (стирать кистью, а не рисовать)")
        side_layout.addWidget(self.erase_checkbox)

        help_label = QLabel(
            "Подсказка:\n"
            "— левая кнопка мыши рисует, правая стирает;\n"
            "— колесо мыши меняет размер кисти;\n"
            "— у линии отреза (для черепов) можно потянуть за белые точки."
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
    """Вкладка итоговой таблицы, экспорта и сравнения групп."""

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
        compare_btn = QPushButton("Сравнить группы")
        compare_btn.clicked.connect(self._run_comparison)
        controls_row.addWidget(compare_btn)
        save_plot_btn = QPushButton("Сохранить график...")
        save_plot_btn.clicked.connect(self._save_plot)
        controls_row.addWidget(save_plot_btn)
        controls_row.addStretch(1)
        stats_layout.addLayout(controls_row)

        self.stats_output = QTextEdit()
        self.stats_output.setReadOnly(True)
        self.stats_output.setMaximumHeight(160)
        stats_layout.addWidget(self.stats_output)

        layout.addWidget(stats_box)

    def set_reviews(self, reviews: list[ShotReview]) -> None:
        self.reviews = reviews
        self._refresh_table()

    def _current_metric_key(self) -> str:
        return self.metric_combo.currentText().split(" ")[0]

    def _refresh_table(self) -> None:
        all_rows: list[MeasurementRow] = []
        for review in self.reviews:
            all_rows.extend(measure_shot(review))
        self.slice_df = rows_to_dataframe(all_rows)
        if self.average_checkbox.isChecked():
            df = per_animal_average(self.slice_df)
        else:
            df = self.slice_df
        self.current_df = df
        self._show_dataframe(df)

    def _show_dataframe(self, df: pd.DataFrame) -> None:
        model = QStandardItemModel(df.shape[0], df.shape[1])
        model.setHorizontalHeaderLabels(list(df.columns))
        for r in range(df.shape[0]):
            for c, col in enumerate(df.columns):
                model.setItem(r, c, QStandardItem(str(df.iat[r, c])))
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
        """Таблица для статистики — ВСЕГДА агрегированная по животным, независимо от
        того, что сейчас показано в таблице выше (срез — не независимое наблюдение)."""
        return per_animal_average(self.slice_df)

    def _run_comparison(self) -> None:
        animal_df = self._animal_df()
        if animal_df.empty or animal_df["group"].nunique() < 2:
            QMessageBox.information(self, "Недостаточно данных", "Нужно минимум 2 группы с принятыми масками.")
            return
        metric = self._current_metric_key()
        try:
            result = compare_groups(animal_df, metric)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Не удалось сравнить", str(exc))
            return

        lines = [
            f"Метод: {result.test_name} (единица анализа — животное, срезы усреднены)",
            f"Группы: {', '.join(f'{g} (n={result.n_per_group[g]})' for g in result.groups)}",
            f"Общий p-value: {result.p_value:.4f}" + ("  (есть значимое различие, p < 0.05)" if result.p_value < 0.05 else "  (значимого различия не обнаружено)"),
        ]
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
        self.stats_output.setPlainText("\n".join(lines))

    def _save_plot(self) -> None:
        animal_df = self._animal_df()
        if animal_df.empty:
            QMessageBox.information(self, "Пусто", "Сначала соберите таблицу.")
            return
        metric = self._current_metric_key()
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить график", "сравнение_групп.png", "PNG (*.png)")
        if not path:
            return
        save_boxplot(animal_df, metric, Path(path), title=self.metric_combo.currentText() + " (по животным)")
        QMessageBox.information(self, "Готово", f"График сохранён:\n{path}")


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обработка фото мозга")
        self.resize(1280, 820)

        self.project_tab = ProjectTab(on_start_review=self._start_review)
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
