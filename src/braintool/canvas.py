"""Виджет для просмотра и ручной правки масок поверх фото."""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QWidget

from .colors import color_for_animal
from .models import SpecimenMask
from .segmentation import apply_polygon_edit, mask_to_polygon, recompute_rostral_mask_from_line

HANDLE_HIT_RADIUS_PX = 14  # в экранных пикселях — попадание по вершине (клик/тяга)
EDGE_HIT_RADIUS_PX = 10    # попадание по ребру полигона (двойной клик — добавить точку)
MIN_ZOOM = 1.0             # 1.0 = фото вписано целиком (как раньше, до зума)
MAX_ZOOM = 8.0
UNDO_LIMIT = 60            # сколько последних правок на кадр помнит Ctrl+Z


class MaskCanvas(QWidget):
    """Показывает фото с полупрозрачными масками; поддерживает кисть, перетаскивание
    линии отреза (режим "носовая часть черепа") и правку контура точками, плюс зум/
    панораму для рассмотрения деталей вблизи."""

    maskEdited = Signal()
    # активная маска сменилась кликом по фото — боковой список подсвечивает её
    activeChanged = Signal()
    # клик в режиме «Поищи здесь» — координаты на изображении; поиск делает ReviewTab
    # (ему нужен исходный кадр, а у холста только картинка для показа)
    findHereRequested = Signal(float, float)

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
        # режим «Поищи здесь»: следующий левый клик по фото — запрос поиска среза
        self._find_here_mode = False
        # история правок текущего кадра (список ShotReview.undo_stack, см. set_shot)
        self._undo_stack: list = []
        # маска, созданная кнопкой «Новый срез кистью» и ещё не нарисованная — после
        # первого мазка ReviewTab ставит её на место по высоте
        self.pending_new_mask: SpecimenMask | None = None

        self._draw_rect: QRectF | None = None
        self._scale = 1.0
        self._zoom = 1.0
        self._pan_offset = QPointF(0.0, 0.0)  # экранные пиксели, поверх центрирования
        self._panning = False
        self._pan_last_screen: QPointF | None = None

        self._dragging_handle: tuple[SpecimenMask, int] | None = None
        self._dragging_polygon_vertex: tuple[SpecimenMask, int] | None = None
        self._drag_polygon_before: list[tuple[float, float]] | None = None
        self._dragging_brush = False

        # положение курсора на экране — для превью круга кисти (None = курсор вне
        # виджета). Раньше размер кисти было не видно до первого мазка
        self._cursor_screen: QPointF | None = None
        # кеш наложения масок на фото: раньше пересобиралось по всему кадру на КАЖДЫЙ
        # paintEvent, включая каждое движение мыши. Ключ — сами объекты картинки и
        # массивов масок (все правки масок в программе ПЕРЕПРИСВАИВАЮТ m.mask новым
        # массивом, а не меняют на месте) + активная маска. Сравнение через `is`, а
        # не id(): ключ держит ссылки на массивы, поэтому id не может переиспользоваться
        self._composite_key: tuple | None = None
        self._composite_pixmap: QPixmap | None = None
        self._active_outline_key: tuple | None = None
        self._active_outline: list[np.ndarray] = []
        self._bbox_cache: dict[int, tuple[np.ndarray, tuple[int, int, int, int] | None]] = {}
        self._base_rgb_src: np.ndarray | None = None
        self._base_rgb: np.ndarray | None = None

    # ---------- публичный API ----------

    def set_shot(
        self, image_u8: np.ndarray, masks: list[SpecimenMask], undo_stack: list | None = None,
        keep_view: bool = False,
    ) -> None:
        same_frame = self.image_u8 is not None and self.image_u8.shape == image_u8.shape
        self.image_u8 = image_u8
        self.masks = masks
        self._undo_stack = undo_stack if undo_stack is not None else []
        self.pending_new_mask = None
        if self._active_mask() is None or not keep_view:
            self.active_key = (masks[0].animal_index, masks[0].slice_index) if masks else None
        if not (keep_view and same_frame):
            # новое фото — старый зум/пан почти наверняка не туда указывает
            self.reset_view()
        if self._point_mode_enabled:
            self._ensure_all_polygons()
        self.update()

    def set_active_mask(self, animal_index: int, slice_index: int) -> None:
        self.active_key = (animal_index, slice_index)
        self.update()

    def active_mask(self) -> SpecimenMask | None:
        return self._active_mask()

    def reset_view(self) -> None:
        """«Вернуть общий вид» — всё фото целиком в окне, без зума и сдвига."""
        self._zoom = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self.update()

    def set_find_here_mode(self, enabled: bool) -> None:
        self._find_here_mode = bool(enabled)
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    # ---------- отмена (Ctrl+Z) ----------

    def _snapshot(self) -> tuple:
        # маски копируются объектами: номера/«принято» меняются на месте (editing.py,
        # accept_all), а сами растры — нет (любая правка присваивает НОВЫЙ массив),
        # поэтому массивы можно не копировать — снимок дешёвый
        copies = [
            dataclasses.replace(m, polygon=list(m.polygon) if m.polygon else None)
            for m in self.masks
        ]
        return copies, self.active_key

    def push_undo(self) -> None:
        """Запомнить текущее состояние масок кадра ПЕРЕД правкой."""
        self._undo_stack.append(self._snapshot())
        del self._undo_stack[:-UNDO_LIMIT]

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        copies, active_key = self._undo_stack.pop()
        # на место — тот же список, что и ShotReview.masks; копии копий, чтобы снимок в
        # истории не менялся от последующих правок
        self.masks[:] = [
            dataclasses.replace(m, polygon=list(m.polygon) if m.polygon else None) for m in copies
        ]
        self.active_key = active_key
        self._cancel_drags()
        self.maskEdited.emit()
        self.update()
        return True

    def _cancel_drags(self) -> None:
        self._dragging_brush = False
        self._dragging_handle = None
        self._dragging_polygon_vertex = None
        self._drag_polygon_before = None

    def set_erase_enabled(self, enabled: bool) -> None:
        self._erase_toggle = bool(enabled)
        self.update()  # превью кисти: пунктир для ластика

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
        или предварительная правка кистью).

        НЕ трогает `cut_line`/`source_blob` — это только построение временного
        отображаемого контура (вызывается при каждом показе фото, до какой-либо
        правки пользователя), а не правка маски. cut_line/source_blob остаются
        рабочими и отвязываются от маски только при РЕАЛЬНОЙ правке точками (см.
        mousePressEvent/mouseReleaseEvent/mouseDoubleClickEvent) — иначе включение
        режима точек по умолчанию (пайплайн носа) безвозвратно убивало бы
        перетаскиваемую линию отреза ещё до того, как пользователь вообще
        посмотрел на кадр, даже если он тут же выключит "Режим точек" обратно."""
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
        if any(not m.accepted for m in self.masks):
            self.push_undo()
        for m in self.masks:
            m.accepted = True
        self.update()

    def mask_at_point(self, img_pt: tuple[float, float]) -> SpecimenMask | None:
        """Первая маска (в порядке self.masks), чей растр покрывает эту точку
        изображения — используется, чтобы клик по маске выбирал её активной."""
        x, y = int(round(img_pt[0])), int(round(img_pt[1]))
        for m in self.masks:
            h, w = m.mask.shape
            if 0 <= y < h and 0 <= x < w and m.mask[y, x]:
                return m
        return None

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
        painter.drawPixmap(self._draw_rect.toRect(), self._composite_pixmap_cached())

        # черепа на фото не лежат идеально по линейке, поэтому у каждого животного
        # своя независимая линия отреза — показываем и даём тянуть их ВСЕ сразу,
        # а не только у активного животного (иначе остальные 4 не видно и не поправить).
        # В режиме точек мышь на cut_line всё равно не реагирует (mousePressEvent
        # обрабатывает точки раньше и не доходит до хит-теста ручек) — не рисуем
        # их поверх контура полигона, только загромождали бы картинку
        active = self._active_mask()
        if not self._point_mode_enabled:
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

        # активная маска раньше отличалась от остальных только чуть большей
        # непрозрачностью заливки (0.7 против 0.55) — на глаз почти незаметно, в какую
        # маску сейчас пойдёт правка. Теперь — контрастная обводка + подпись. В режиме
        # точек активный полигон и так рисуется ярче остальных, обводка не нужна
        if active is not None and active.mask.any():
            if not self._point_mode_enabled or not active.polygon:
                self._draw_active_outline(painter, active)
            self._draw_active_label(painter, active)

        if self._brush_would_paint():
            self._draw_brush_preview(painter)

        painter.end()

    def _composite_pixmap_cached(self) -> QPixmap:
        key = (self.image_u8, self.active_key, tuple((m.mask, m.animal_index, m.slice_index) for m in self.masks))
        if not self._cache_key_matches(self._composite_key, key):
            composite = self._build_composite_rgb()
            h, w, _ = composite.shape
            qimg = QImage(composite.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            # copy(): QImage не владеет буфером numpy, а composite живёт только до
            # конца этой функции
            self._composite_pixmap = QPixmap.fromImage(qimg.copy())
            self._composite_key = key
        return self._composite_pixmap

    @staticmethod
    def _cache_key_matches(old: tuple | None, new: tuple) -> bool:
        if old is None:
            return False
        old_img, old_active, old_masks = old
        new_img, new_active, new_masks = new
        if old_img is not new_img or old_active != new_active or len(old_masks) != len(new_masks):
            return False
        return all(a[0] is b[0] and a[1:] == b[1:] for a, b in zip(old_masks, new_masks))

    def _brush_would_paint(self) -> bool:
        """Показывать ли круг кисти под курсором — только там, где клик реально
        рисует кистью: вне режима точек, либо в режиме точек у активной маски без
        контура (пустая ячейка, см. mousePressEvent)."""
        if self._cursor_screen is None or self._panning or self._find_here_mode:
            return False
        if self._dragging_handle is not None or self._dragging_polygon_vertex is not None:
            return False
        if self._screen_to_image(self._cursor_screen) is None:
            return False
        if not self._point_mode_enabled or self._erase_toggle:
            return True
        active = self._active_mask()
        return active is not None and active.polygon is None

    def _draw_brush_preview(self, painter: QPainter) -> None:
        r = self.brush_radius_img * self._scale
        erase = self._erase_toggle
        # тёмная подложка + светлая линия — круг виден и на ярком, и на тёмном фоне;
        # пунктир для ластика, чтобы отличать режим до клика
        style = Qt.PenStyle.DashLine if erase else Qt.PenStyle.SolidLine
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, 160), 3))
        painter.drawEllipse(self._cursor_screen, r, r)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1.5, style))
        painter.drawEllipse(self._cursor_screen, r, r)

    def _active_outline_contours(self, m: SpecimenMask) -> list[np.ndarray]:
        key = (m.mask,)
        if self._active_outline_key is None or self._active_outline_key[0] is not m.mask:
            contours, _ = cv2.findContours(m.mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            self._active_outline = [c.reshape(-1, 2) for c in contours if len(c) >= 2]
            self._active_outline_key = key
        return self._active_outline

    def _draw_active_outline(self, painter: QPainter, m: SpecimenMask) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        polys = []
        for c in self._active_outline_contours(m):
            polys.append(QPolygonF([self._to_screen((float(x) + 0.5, float(y) + 0.5)) for x, y in c]))
        for pen in (QPen(QColor(0, 0, 0, 200), 3.5), QPen(QColor(255, 255, 255, 240), 1.5)):
            painter.setPen(pen)
            for poly in polys:
                painter.drawPolygon(poly)

    def _draw_active_label(self, painter: QPainter, m: SpecimenMask) -> None:
        """Подпись "Животное N[, срез M]" над активной маской — те же номера (с 1),
        что и в списке слева."""
        ys, xs = np.nonzero(m.mask)
        top = self._to_screen((float(xs.min()), float(ys.min())))
        text = f"Животное {m.animal_index + 1}"
        # как в списке слева: номер среза — только если у животного их несколько
        # (у черепов в пайплайне носа маска на животное одна)
        if sum(1 for o in self.masks if o.animal_index == m.animal_index) > 1:
            text += f", срез {m.slice_index + 1}"
        font = QFont(painter.font())
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 8, fm.height() + 4
        x = min(max(top.x(), 2.0), max(2.0, self.width() - w - 2))
        y = max(top.y() - h - 4, 2.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 170)))
        painter.drawRoundedRect(QRectF(x, y, w, h), 3, 3)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(x, y, w, h), Qt.AlignmentFlag.AlignCenter, text)

    def _mask_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        """(y0, y1, x0, x1) закрашенной области маски или None для пустой. Кешируется
        по объекту массива (маски не меняются на месте — см. _composite_key)."""
        cached = self._bbox_cache.get(id(mask))
        if cached is not None and cached[0] is mask:
            return cached[1]
        rows = np.flatnonzero(mask.any(axis=1))
        if rows.size == 0:
            bbox = None
        else:
            cols = np.flatnonzero(mask[rows[0]:rows[-1] + 1].any(axis=0))
            bbox = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
        self._bbox_cache[id(mask)] = (mask, bbox)
        return bbox

    def _build_composite_rgb(self) -> np.ndarray:
        # серое фото в float-RGB зависит только от самого фото — считаем раз на кадр
        if self._base_rgb_src is not self.image_u8:
            base = self.image_u8.astype(np.float32)
            self._base_rgb = np.stack([base, base, base], axis=-1)
            self._base_rgb_src = self.image_u8
        rgb = self._base_rgb.copy()
        # каждая маска смешивается только в своём bbox, а не по всему кадру — раньше
        # 30 срезов на фото давали ~0.3 с на каждую перерисовку во время мазка кистью
        live = {id(m.mask) for m in self.masks}
        self._bbox_cache = {k: v for k, v in self._bbox_cache.items() if k in live}
        for m in self.masks:
            bbox = self._mask_bbox(m.mask)
            if bbox is None:
                continue
            y0, y1, x0, x1 = bbox
            color = np.array(color_for_animal(m.animal_index), dtype=np.float32)
            alpha = 0.7 if (m.animal_index, m.slice_index) == self.active_key else 0.55
            sel = m.mask[y0:y1, x0:x1]
            crop = rgb[y0:y1, x0:x1]
            crop[sel] = crop[sel] * (1 - alpha) + color * alpha
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

    def _start_pan(self, event) -> None:
        self._panning = True
        self._pan_last_screen = event.position()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.update()

    def _set_active(self, m: SpecimenMask) -> None:
        key = (m.animal_index, m.slice_index)
        if key != self.active_key:
            self.active_key = key
            self.activeChanged.emit()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        buttons = event.buttons()
        both = (buttons & Qt.MouseButton.LeftButton) and (buttons & Qt.MouseButton.RightButton)
        if event.button() == Qt.MouseButton.MiddleButton:
            self._start_pan(event)
            return
        if both:
            # обе кнопки мыши сразу — сдвиг картинки. Первая из двух кнопок уже успела
            # начать мазок кистью (или тянуть точку) — откатываем его, как будто его
            # не было, иначе каждое перемещение оставляло бы кляксу в маске
            if self._dragging_brush or self._dragging_polygon_vertex is not None or self._dragging_handle is not None:
                self.undo()
            self._cancel_drags()
            self._start_pan(event)
            return

        img_pt = self._screen_to_image(event.position())
        if img_pt is None:
            return

        if self._find_here_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                self.findHereRequested.emit(float(img_pt[0]), float(img_pt[1]))
            return

        # ластик важнее режима точек: включённый ластик должен стирать с ПЕРВОГО клика
        # по маске (жалоба сессии 7: «первый тык просто выбирает, хотя ластик включён» —
        # в режиме точек клик по маске мимо вершины не делал ничего)
        if self._erase_toggle:
            hit_mask = self.mask_at_point(img_pt)
            if hit_mask is not None:
                self._set_active(hit_mask)
            self._begin_brush(img_pt, erase=True)
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
                self._set_active(m)
                if event.button() == Qt.MouseButton.RightButton:
                    if len(m.polygon) > 3:
                        self.push_undo()
                        old_polygon = list(m.polygon)
                        m.polygon = list(m.polygon)
                        del m.polygon[idx]
                        m.mask = apply_polygon_edit(m.mask, old_polygon, m.polygon)
                        m.cut_line = None
                        m.source_blob = None
                        m.accepted = False
                        self.maskEdited.emit()
                    self.update()
                    return
                self.push_undo()
                self._dragging_polygon_vertex = (m, idx)
                # контур ДО перетаскивания — нужен на отпускании, чтобы понять, какую
                # область маски он представлял (см. apply_polygon_edit)
                self._drag_polygon_before = list(m.polygon)
                self.update()
                return
            # клик мимо вершины в режиме точек НАМЕРЕННО ничего не делает для маски,
            # у которой уже есть контур — добавление точки только двойным кликом на
            # ребре (mouseDoubleClickEvent), одиночный клик по пустому месту не должен
            # случайно создавать новую точку. НО если у активной маски контура ещё нет
            # (пустая ячейка — например, авто-детекция не нашла череп, см. секцию 3.3/3.4
            # документации), то обвести нечего, и хит-тест выше никогда не сработает —
            # без этой ветки такую маску было вообще невозможно нарисовать в режиме
            # точек (клик молча ничего не делал). Кисть остаётся способом создать
            # первый мазок; контур для него построится сам при следующей отрисовке
            # клик внутри другой маски мимо вершин — просто выбрать её
            hit_mask = self.mask_at_point(img_pt)
            if hit_mask is not None:
                self._set_active(hit_mask)
                self.update()
                return
            active = self._active_mask()
            if active is not None and active.polygon is None:
                self._begin_brush(img_pt, erase=event.button() == Qt.MouseButton.RightButton)
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
                    self.push_undo()
                    self._dragging_handle = (m, idx)
                    self._set_active(m)
                    self.update()
                    return

        # клик по уже размеченной маске (кисть, вне режима точек) сначала выбирает
        # её активной, как и с ручками cut_line/полигона выше — раньше рисование/
        # стирание всегда шло в маску, выбранную ранее в списке слева, даже если
        # курсор был над СОВСЕМ ДРУГИМ животным на фото; неудобно на многоживотных
        # фото по сравнению с пайплайном "нос", где клик по линии сам переключает
        # активное животное. Клик по пустому фону активное животное не меняет —
        # так по-прежнему можно дорисовывать маску наружу за её текущий край
        hit_mask = self.mask_at_point(img_pt)
        if hit_mask is not None:
            self._set_active(hit_mask)

        # правая кнопка стирает всегда; левая — стирает, только если включён чекбокс
        # "Ластик" (раньше чекбокс ни на что не влиял — это и была жалоба "ластик не
        # работает": включаешь чекбокс, жмёшь левой, а получаешь рисование)
        self._begin_brush(img_pt, erase=event.button() == Qt.MouseButton.RightButton)

    def _begin_brush(self, img_pt: tuple[float, float], erase: bool) -> None:
        if self._active_mask() is None:
            return
        self.push_undo()
        self._dragging_brush = True
        self.erase_mode = self._erase_toggle or erase
        self._paint_brush(img_pt)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._cursor_screen = event.position()
        if not (self._panning or self._dragging_brush or self._dragging_handle is not None
                or self._dragging_polygon_vertex is not None):
            # простое движение без нажатых кнопок — только сдвинуть превью кисти
            self.update()
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
        if self._panning:
            # сдвиг заканчивается, когда отпущена любая из кнопок, которыми он начат;
            # оставшаяся зажатой кнопка не должна тут же начать рисовать
            self._panning = False
            self._pan_last_screen = None
            self.setCursor(Qt.CursorShape.CrossCursor if self._find_here_mode else Qt.CursorShape.ArrowCursor)
            self._cancel_drags()
            self.update()
            return
        if self._dragging_polygon_vertex is not None:
            m, _ = self._dragging_polygon_vertex
            m.mask = apply_polygon_edit(m.mask, self._drag_polygon_before or m.polygon, m.polygon)
            self._drag_polygon_before = None
            m.cut_line = None
            m.source_blob = None
            m.accepted = False
            self._dragging_polygon_vertex = None
            self.maskEdited.emit()
            self.update()
        if self._dragging_handle is not None:
            m, _ = self._dragging_handle
            if m.source_blob is not None and m.cut_line is not None:
                p1, p2 = m.cut_line
                degenerate = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 < 1.0
                if degenerate:
                    # обе ручки сведены практически в одну точку — направление линии
                    # не определено; пересчёт по такой линии молча отдал бы под "нос"
                    # весь исходный blob целиком (весь череп). Оставляем маску как
                    # была — пользователь должен развести ручки снова, а не получить
                    # тихо неверный результат
                    pass
                else:
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
        self.push_undo()
        old_polygon = list(m.polygon)
        polygon = list(m.polygon)
        polygon.insert(insert_at, img_pt)
        m.polygon = polygon
        m.mask = apply_polygon_edit(m.mask, old_polygon, m.polygon)
        m.cut_line = None
        m.source_blob = None
        m.accepted = False
        self._set_active(m)
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
                # клик почти точно на одном из концов ребра — это попытка попасть по
                # СУЩЕСТВУЮЩЕЙ вершине (двойной клик по ней проходит Press на первом
                # клике, который уже начинает перетаскивание), а не по середине ребра;
                # не считаем это ребро кандидатом на вставку новой точки, иначе почти
                # каждый двойной клик по вершине незаметно дублирует её вырожденной
                # соседней точкой (нулевой длины ребро)
                edge_hit_radius_img = EDGE_HIT_RADIUS_PX / max(self._scale, 1e-6)
                if np.linalg.norm(p - a) <= edge_hit_radius_img or np.linalg.norm(p - b) <= edge_hit_radius_img:
                    continue
                ab = b - a
                ab_len2 = float(ab @ ab)
                t = 0.0 if ab_len2 == 0 else float(np.clip((p - a) @ ab / ab_len2, 0.0, 1.0))
                closest = a + t * ab
                dist = float(np.linalg.norm(p - closest))
                if dist <= best_dist:
                    best_dist = dist
                    best = (m, i + 1)
        return best

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._cursor_screen = None
        self.update()
        super().leaveEvent(event)

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
