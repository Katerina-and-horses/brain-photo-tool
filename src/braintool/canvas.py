"""Виджет для просмотра и ручной правки масок поверх фото."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from .colors import color_for_animal
from .models import SpecimenMask
from .segmentation import mask_to_polygon, polygon_to_mask, recompute_rostral_mask_from_line

HANDLE_HIT_RADIUS_PX = 14  # в экранных пикселях — попадание по вершине (клик/тяга)
EDGE_HIT_RADIUS_PX = 10    # попадание по ребру полигона (двойной клик — добавить точку)
MIN_ZOOM = 1.0             # 1.0 = фото вписано целиком (как раньше, до зума)
MAX_ZOOM = 8.0


class MaskCanvas(QWidget):
    """Показывает фото с полупрозрачными масками; поддерживает кисть, перетаскивание
    линии отреза (режим "носовая часть черепа") и правку контура точками, плюс зум/
    панораму для рассмотрения деталей вблизи."""

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
        # состояние чекбокса "Режим точек" — альтернатива кисти: маска редактируется
        # перетаскиванием вершин полигона, а не закрашиванием. Пока включён, кисть
        # не рисует (мышь целиком уходит на работу с точками).
        self._point_mode_enabled = False

        self._draw_rect: QRectF | None = None
        self._scale = 1.0
        self._zoom = 1.0
        self._pan_offset = QPointF(0.0, 0.0)  # экранные пиксели, поверх центрирования
        self._panning = False
        self._pan_last_screen: QPointF | None = None

        self._dragging_handle: tuple[SpecimenMask, int] | None = None
        self._dragging_polygon_vertex: tuple[SpecimenMask, int] | None = None
        self._dragging_brush = False

    # ---------- публичный API ----------

    def set_shot(self, image_u8: np.ndarray, masks: list[SpecimenMask]) -> None:
        self.image_u8 = image_u8
        self.masks = masks
        if masks:
            self.active_key = (masks[0].animal_index, masks[0].slice_index)
        else:
            self.active_key = None
        # новое фото — старый зум/пан почти наверняка не туда указывает
        self._zoom = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        if self._point_mode_enabled:
            self._ensure_all_polygons()
        self.update()

    def set_active_mask(self, animal_index: int, slice_index: int) -> None:
        self.active_key = (animal_index, slice_index)
        self.update()

    def set_erase_enabled(self, enabled: bool) -> None:
        self._erase_toggle = bool(enabled)

    def set_point_mode_enabled(self, enabled: bool) -> None:
        self._point_mode_enabled = bool(enabled)
        if self._point_mode_enabled:
            self._ensure_all_polygons()
        self.update()

    def _ensure_polygon(self, m: SpecimenMask) -> None:
        """Если у маски ещё нет полигона — строит его по текущей растровой маске
        (отправная точка для правки точками: подтянуть готовый контур, не обводить
        форму заново с нуля). Пустую маску (нечего обводить) не трогает — полигон
        появится только после того, как в маске будет хоть что-то (автообнаружение
        или предварительная правка кистью)."""
        if m.polygon is None and m.mask.any():
            m.polygon = mask_to_polygon(m.mask)

    def _ensure_all_polygons(self) -> None:
        """Полигоны нужны у ВСЕХ масок сразу (не только активной) — иначе при
        нескольких животных/срезах на фото пришлось бы щёлкать по каждому в списке
        слева, чтобы просто увидеть его контур; переключение между ними в списке
        неудобно и легко пропустить, кого не проверил."""
        for m in self.masks:
            self._ensure_polygon(m)

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
        base_scale = min(aw / iw, ah / ih)
        self._scale = base_scale * self._zoom
        dw, dh = iw * self._scale, ih * self._scale
        x0 = (aw - dw) / 2 + self._pan_offset.x()
        y0 = (ah - dh) / 2 + self._pan_offset.y()
        self._draw_rect = QRectF(x0, y0, dw, dh)

    def _screen_to_image(self, pt: QPointF) -> tuple[float, float] | None:
        if self._draw_rect is None or self._scale == 0:
            return None
        if not self._draw_rect.contains(pt):
            return None
        ix = (pt.x() - self._draw_rect.x()) / self._scale
        iy = (pt.y() - self._draw_rect.y()) / self._scale
        return ix, iy

    def _to_screen(self, p: tuple[float, float]) -> QPointF:
        return QPointF(
            self._draw_rect.x() + p[0] * self._scale,
            self._draw_rect.y() + p[1] * self._scale,
        )

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

        # полигон точек — по той же логике, что и линия отреза выше: ВСЕ маски сразу,
        # активная ярче. Раньше показывалась только активная — оказалось неудобно
        # переключаться между животными/срезами в списке слева, чтобы просто увидеть
        # контур и понять, норм он или нет
        if self._point_mode_enabled:
            self._ensure_all_polygons()
            for m in self.masks:
                if m.polygon:
                    self._draw_polygon(painter, m.polygon, is_active=(m is active))

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

    def _draw_polygon(self, painter: QPainter, polygon: list[tuple[float, float]], is_active: bool) -> None:
        line_alpha = 230 if is_active else 110
        line_width = 2 if is_active else 1
        handle_radius = 5 if is_active else 3

        screen_pts = [self._to_screen(p) for p in polygon]
        painter.setPen(QPen(QColor(255, 255, 255, line_alpha), line_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(len(screen_pts)):
            painter.drawLine(screen_pts[i], screen_pts[(i + 1) % len(screen_pts)])
        painter.setBrush(QBrush(QColor(255, 255, 255, line_alpha)))
        for sp in screen_pts:
            painter.drawEllipse(sp, handle_radius, handle_radius)

    # ---------- взаимодействие мышью ----------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_last_screen = event.position()
            return

        img_pt = self._screen_to_image(event.position())
        if img_pt is None:
            return

        # режим точек полностью забирает мышь себе — кисть в этом режиме не рисует
        # (переключение между режимами — отдельный чекбокс в боковой панели). Ищем
        # среди ВСЕХ масок сразу (как и с cut_line ниже) — так можно поправить любое
        # животное прямо на фото, не переключаясь сначала на него в списке слева;
        # попадание заодно переключает активное животное
        if self._point_mode_enabled:
            hit = self._hit_test_polygon_vertex_any(img_pt)
            if hit is not None:
                m, idx = hit
                self.active_key = (m.animal_index, m.slice_index)
                if event.button() == Qt.MouseButton.RightButton:
                    if len(m.polygon) > 3:
                        del m.polygon[idx]
                        m.mask = polygon_to_mask(m.polygon, m.mask.shape)
                        m.accepted = False
                        self.maskEdited.emit()
                    self.update()
                    return
                self._dragging_polygon_vertex = (m, idx)
                self.update()
                return
            # клик мимо вершины в режиме точек ничего не делает — добавление точки
            # только двойным кликом на ребре (mouseDoubleClickEvent), одиночный клик
            # по пустому месту не должен случайно создавать новую точку
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
        if self._panning and self._pan_last_screen is not None:
            delta = event.position() - self._pan_last_screen
            self._pan_offset += delta
            self._pan_last_screen = event.position()
            self.update()
            return

        img_pt = self._screen_to_image(event.position())
        if img_pt is None:
            return
        if self._dragging_polygon_vertex is not None:
            # во время перетаскивания растр НЕ пересчитывается на каждый пиксель
            # движения (только контур-превью) — как и с cut_line ниже, растеризация
            # в mask происходит один раз на отпускании кнопки (mouseReleaseEvent)
            m, idx = self._dragging_polygon_vertex
            polygon = list(m.polygon)
            polygon[idx] = img_pt
            m.polygon = polygon
            self.update()
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
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._pan_last_screen = None
            return
        if self._dragging_polygon_vertex is not None:
            m, _ = self._dragging_polygon_vertex
            m.mask = polygon_to_mask(m.polygon, m.mask.shape)
            m.accepted = False
            self._dragging_polygon_vertex = None
            self.maskEdited.emit()
            self.update()
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

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        """Двойной клик на ребре полигона (только в режиме точек, у любой маски) —
        добавляет вершину в этом месте. Одиночный клик рядом с ребром намеренно
        ничего не делает (см. mousePressEvent) — иначе случайная новая точка при
        попытке просто перетащить существующую была бы неотличима от намеренного
        добавления."""
        if not self._point_mode_enabled:
            return
        img_pt = self._screen_to_image(event.position())
        if img_pt is None:
            return
        hit = self._hit_test_polygon_edge_any(img_pt)
        if hit is None:
            return
        m, insert_at = hit
        polygon = list(m.polygon)
        polygon.insert(insert_at, img_pt)
        m.polygon = polygon
        m.mask = polygon_to_mask(m.polygon, m.mask.shape)
        m.accepted = False
        self.active_key = (m.animal_index, m.slice_index)
        self.maskEdited.emit()
        self.update()

    def _hit_test_polygon_vertex_any(self, img_pt: tuple[float, float]) -> tuple[SpecimenMask, int] | None:
        for m in self.masks:
            if not m.polygon:
                continue
            for idx, p in enumerate(m.polygon):
                dx = (p[0] - img_pt[0]) * self._scale
                dy = (p[1] - img_pt[1]) * self._scale
                if (dx * dx + dy * dy) ** 0.5 <= HANDLE_HIT_RADIUS_PX:
                    return m, idx
        return None

    def _hit_test_polygon_edge_any(self, img_pt: tuple[float, float]) -> tuple[SpecimenMask, int] | None:
        """Возвращает (маску, индекс КУДА вставить новую вершину) для ближайшего
        ребра среди ВСЕХ полигонов, если клик достаточно близко к какому-то из них,
        иначе None."""
        best: tuple[SpecimenMask, int] | None = None
        best_dist = EDGE_HIT_RADIUS_PX / max(self._scale, 1e-6)  # порог в координатах изображения
        p = np.array(img_pt)
        for m in self.masks:
            if not m.polygon:
                continue
            n = len(m.polygon)
            for i in range(n):
                a, b = np.array(m.polygon[i]), np.array(m.polygon[(i + 1) % n])
                ab = b - a
                ab_len2 = float(ab @ ab)
                t = 0.0 if ab_len2 == 0 else float(np.clip((p - a) @ ab / ab_len2, 0.0, 1.0))
                closest = a + t * ab
                dist = float(np.linalg.norm(p - closest))
                if dist <= best_dist:
                    best_dist = dist
                    best = (m, i + 1)
        return best

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._zoom_at(event.position(), event.angleDelta().y())
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1 / 1.15)
        self.brush_radius_img = float(np.clip(self.brush_radius_img * factor, 3, 150))
        self.update()

    def _zoom_at(self, screen_pos: QPointF, wheel_delta: float) -> None:
        """Зум с "заякориванием" под курсором — точка фото под мышью остаётся на
        месте на экране (как в обычных просмотрщиках изображений), а не съезжает
        каждый раз к центру. Без этого не понять, норм ли легли точки полигона на
        край ткани — нужно приближать именно то место, куда смотришь, не в центр."""
        if self._draw_rect is None:
            return
        anchor_img = self._screen_to_image(screen_pos)
        factor = 1.15 if wheel_delta > 0 else (1 / 1.15)
        self._zoom = float(np.clip(self._zoom * factor, MIN_ZOOM, MAX_ZOOM))
        self._recompute_draw_rect()
        if anchor_img is not None:
            new_screen = self._to_screen(anchor_img)
            self._pan_offset += screen_pos - new_screen
            self._recompute_draw_rect()
        self.update()

    def _paint_brush(self, img_pt: tuple[float, float]) -> None:
        target = self._active_mask()
        if target is None:
            return
        # рисование кистью "отвязывает" маску от параметрической линии отреза и от
        # полигона — дальше это обычная растровая маска, которую правят только
        # кистью (полигон при повторном включении режима точек будет перестроен
        # заново по уже подправленному кистью растру)
        target.cut_line = None
        target.source_blob = None
        target.polygon = None

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
