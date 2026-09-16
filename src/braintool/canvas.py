"""Виджет для просмотра и ручной правки масок поверх фото."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from .colors import color_for_animal
from .models import SpecimenMask
from .segmentation import recompute_rostral_mask_from_line

HANDLE_HIT_RADIUS_PX = 14  # в экранных пикселях


class MaskCanvas(QWidget):
    """Показывает фото с полупрозрачными масками; поддерживает кисть и перетаскивание
    линии отреза для режима "носовая часть черепа"."""

    maskEdited = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)

        self.image_u8: np.ndarray | None = None
        self.masks: list[SpecimenMask] = []
        # активна ОДНА конкретная маска (животное, срез) — не всё животное сразу,
        # иначе при нескольких срезах на животное правка всегда попадала бы в первый срез
        self.active_key: tuple[int, int] | None = None
        self.brush_radius_img = 14.0
        self.erase_mode = False
        # состояние чекбокса "Ластик" в боковой панели — независимо от кнопки мыши,
        # чтобы можно было стирать левой кнопкой, не только правой
        self._erase_toggle = False

        self._draw_rect: QRectF | None = None
        self._scale = 1.0
        self._dragging_handle: tuple[SpecimenMask, int] | None = None
        self._dragging_brush = False

    # ---------- публичный API ----------

    def set_shot(self, image_u8: np.ndarray, masks: list[SpecimenMask]) -> None:
        self.image_u8 = image_u8
        self.masks = masks
        if masks:
            self.active_key = (masks[0].animal_index, masks[0].slice_index)
        else:
            self.active_key = None
        self.update()

    def set_active_mask(self, animal_index: int, slice_index: int) -> None:
        self.active_key = (animal_index, slice_index)
        self.update()

    def set_erase_enabled(self, enabled: bool) -> None:
        self._erase_toggle = bool(enabled)

    def accept_all(self) -> None:
        for m in self.masks:
            m.accepted = True
        self.update()

    def _active_mask(self) -> SpecimenMask | None:
        if self.active_key is None:
            return None
        for m in self.masks:
            if (m.animal_index, m.slice_index) == self.active_key:
                return m
        return None

    # ---------- геометрия ----------

    def _recompute_draw_rect(self) -> None:
        if self.image_u8 is None:
            self._draw_rect = None
            return
        ih, iw = self.image_u8.shape[:2]
        aw, ah = max(1, self.width()), max(1, self.height())
        scale = min(aw / iw, ah / ih)
        dw, dh = iw * scale, ih * scale
        x0, y0 = (aw - dw) / 2, (ah - dh) / 2
        self._draw_rect = QRectF(x0, y0, dw, dh)
        self._scale = scale

    def _screen_to_image(self, pt: QPointF) -> tuple[float, float] | None:
        if self._draw_rect is None or self._scale == 0:
            return None
        if not self._draw_rect.contains(pt):
            return None
        ix = (pt.x() - self._draw_rect.x()) / self._scale
        iy = (pt.y() - self._draw_rect.y()) / self._scale
        return ix, iy

    # ---------- отрисовка ----------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        if self.image_u8 is None:
            painter.end()
            return

        self._recompute_draw_rect()
        composite = self._build_composite_rgb()
        h, w, _ = composite.shape
        qimg = QImage(composite.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        painter.drawPixmap(self._draw_rect.toRect(), pixmap)

        # черепа на фото не лежат идеально по линейке, поэтому у каждого животного
        # своя независимая линия отреза — показываем и даём тянуть их ВСЕ сразу,
        # а не только у активного животного (иначе остальные 4 не видно и не поправить)
        active = self._active_mask()
        for m in self.masks:
            if m.cut_line is not None:
                self._draw_handles(painter, m, is_active=(m is active))

        painter.end()

    def _build_composite_rgb(self) -> np.ndarray:
        base = self.image_u8.astype(np.float32)
        rgb = np.stack([base, base, base], axis=-1)
        for m in self.masks:
            color = np.array(color_for_animal(m.animal_index), dtype=np.float32)
            alpha = 0.7 if (m.animal_index, m.slice_index) == self.active_key else 0.55
            sel = m.mask
            if sel.any():
                rgb[sel] = rgb[sel] * (1 - alpha) + color * alpha
        return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))

    def _draw_handles(self, painter: QPainter, m: SpecimenMask, is_active: bool) -> None:
        assert m.cut_line is not None
        # активная линия — яркая и толще, остальные — притушенные, чтобы не мешали
        # глазу, но были видны и доступны для перетаскивания напрямую с фото
        line_alpha = 255 if is_active else 120
        line_width = 2 if is_active else 1
        handle_radius = 6 if is_active else 4

        for p in m.cut_line:
            sx = self._draw_rect.x() + p[0] * self._scale
            sy = self._draw_rect.y() + p[1] * self._scale
            painter.setPen(QPen(QColor(255, 255, 255, line_alpha), line_width))
            painter.setBrush(QBrush(QColor(255, 255, 255, line_alpha if is_active else 90)))
            painter.drawEllipse(QPointF(sx, sy), handle_radius, handle_radius)
        p1, p2 = m.cut_line
        sx1 = self._draw_rect.x() + p1[0] * self._scale
        sy1 = self._draw_rect.y() + p1[1] * self._scale
        sx2 = self._draw_rect.x() + p2[0] * self._scale
        sy2 = self._draw_rect.y() + p2[1] * self._scale
        painter.setPen(QPen(QColor(255, 255, 255, line_alpha), line_width, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

    # ---------- взаимодействие мышью ----------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        img_pt = self._screen_to_image(event.position())
        if img_pt is None:
            return

        # ручки видны у всех животных сразу (не только у активного) — ищем среди всех,
        # чтобы можно было потянуть за нужный череп прямо на фото, не щёлкая сначала
        # по нему в списке слева; это заодно и переключает активное животное
        for m in self.masks:
            if m.cut_line is None:
                continue
            for idx, p in enumerate(m.cut_line):
                dx = (p[0] - img_pt[0]) * self._scale
                dy = (p[1] - img_pt[1]) * self._scale
                if (dx * dx + dy * dy) ** 0.5 <= HANDLE_HIT_RADIUS_PX:
                    self._dragging_handle = (m, idx)
                    self.active_key = (m.animal_index, m.slice_index)
                    self.update()
                    return

        self._dragging_brush = True
        # правая кнопка стирает всегда; левая — стирает, только если включён чекбокс
        # "Ластик" (раньше чекбокс ни на что не влиял — это и была жалоба "ластик не
        # работает": включаешь чекбокс, жмёшь левой, а получаешь рисование)
        self.erase_mode = self._erase_toggle or event.button() == Qt.MouseButton.RightButton
        self._paint_brush(img_pt)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        img_pt = self._screen_to_image(event.position())
        if img_pt is None:
            return
        if self._dragging_handle is not None:
            m, idx = self._dragging_handle
            other = m.cut_line[1 - idx]
            new_line = (img_pt, other) if idx == 0 else (other, img_pt)
            m.cut_line = new_line
            self.update()
            return
        if self._dragging_brush:
            self._paint_brush(img_pt)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._dragging_handle is not None:
            m, _ = self._dragging_handle
            if m.source_blob is not None and m.cut_line is not None:
                # опорная точка фиксируется один раз при автоопределении (центроид
                # изначальной носовой части) и не пересчитывается — иначе при
                # перетаскивании линии сторона выбора могла неожиданно инвертироваться
                ref = m.rostral_anchor if m.rostral_anchor is not None else (0.0, 0.0)
                m.mask = recompute_rostral_mask_from_line(m.source_blob, m.cut_line, ref)
                # перетаскивание линии — тоже правка; раньше только кисть сбрасывала
                # "принято", и перетащенная-но-непроверенная маска могла остаться
                # помеченной как принятая
                m.accepted = False
            self._dragging_handle = None
            self.maskEdited.emit()
            self.update()
        if self._dragging_brush:
            self._dragging_brush = False
            self.maskEdited.emit()

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1 / 1.15)
        self.brush_radius_img = float(np.clip(self.brush_radius_img * factor, 3, 150))
        self.update()

    def _paint_brush(self, img_pt: tuple[float, float]) -> None:
        target = self._active_mask()
        if target is None:
            return
        # рисование кистью "отвязывает" маску от параметрической линии отреза —
        # дальше это обычная растровая маска, которую правят только кистью
        target.cut_line = None
        target.source_blob = None

        h, w = target.mask.shape
        yy, xx = np.ogrid[:h, :w]
        cx, cy = img_pt
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        brush = dist2 <= self.brush_radius_img ** 2
        if self.erase_mode:
            target.mask = target.mask & ~brush
        else:
            target.mask = target.mask | brush
        target.accepted = False
        self.update()
