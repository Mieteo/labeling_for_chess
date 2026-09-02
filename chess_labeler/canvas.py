"""QGraphicsView-based image canvas: draw/select/move/resize boxes, zoom,
and the two drag gestures used elsewhere (new box, reference-radius
measurement). Kept free of file I/O and undo policy -- it only exposes
primitives and emits signals for user gestures; `main_window.py` decides
what those gestures mean (push undo snapshot, mark dirty, assign class...).
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable, Mapping

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QStyleOptionGraphicsItem,
    QWidget,
)

from .key_shortcuts import BOARD_REGION_CLASS, display_label_for_class

HANDLE_SCREEN_SIZE = 9.0  # px, kept constant on screen regardless of zoom
LABEL_FONT_SCREEN_PX = 9.0  # class-label glyph size, kept constant on screen
LABEL_PADDING_SCREEN_PX = 2.0
SELECTED_LABEL_FONT_SCREEN_PX = 16.0
SELECTED_LABEL_PADDING_SCREEN_PX = 6.0
SELECTED_LABEL_GAP_SCREEN_PX = 5.0
SELECTED_LABEL_BORDER_SCREEN_PX = 1.0

# `board_region` spans nearly the whole board and overlaps every piece box on
# top of it -- see yeu_cau_tu_app_ky_nhan.md section 1. Left at the default
# z-value it would sit above (or below, depending on add order) piece boxes
# arbitrarily, so a click meant for a piece could hit the region box instead.
# Keep it behind everything else while it isn't selected (a click always
# reaches the smaller piece box first), but bring it in front while selected
# so its own resize handles stay reachable even where they overlap a piece
# box -- see BoxItem._sync_z_value.
_DEFAULT_BOX_Z_VALUE = 0.0
_UNSELECTED_BOARD_REGION_Z_VALUE = -0.5
_SELECTED_BOARD_REGION_Z_VALUE = 1.0
MIN_DRAG_PX = 3
CORNER_MARKER_SCREEN_RADIUS = 3.0

CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")
_CORNER_COLORS = {
    "top_left": QColor(76, 175, 80),
    "top_right": QColor(33, 150, 243),
    "bottom_right": QColor(255, 152, 0),
    "bottom_left": QColor(156, 39, 176),
}

_PALETTE_BY_NAME = {
    "red_king": QColor(211, 47, 47),
    "red_advisor": QColor(229, 57, 53),
    "red_elephant": QColor(239, 83, 80),
    "red_horse": QColor(244, 67, 54),
    "red_cannon": QColor(216, 27, 96),
    "red_rook": QColor(230, 74, 25),
    "red_pawn": QColor(255, 112, 67),
    "black_king": QColor(38, 50, 56),
    "black_advisor": QColor(69, 90, 100),
    "black_elephant": QColor(84, 110, 122),
    "black_horse": QColor(55, 71, 79),
    "black_cannon": QColor(63, 81, 181),
    "black_rook": QColor(48, 63, 159),
    "black_pawn": QColor(92, 107, 192),
    "hand": QColor(251, 192, 45),
    "board_region": QColor(0, 150, 136),
}
_FALLBACK_COLORS = [QColor(h) for h in ("#8e24aa", "#3949ab", "#00897b", "#7cb342", "#fdd835")]
_UNASSIGNED_COLOR = QColor(158, 158, 158)


def class_color(name: str | None) -> QColor:
    if not name:
        return _UNASSIGNED_COLOR
    if name in _PALETTE_BY_NAME:
        return _PALETTE_BY_NAME[name]
    return _FALLBACK_COLORS[hash(name) % len(_FALLBACK_COLORS)]


_LABEL_TEXT_RED = _PALETTE_BY_NAME["red_king"]
_LABEL_TEXT_BLACK = QColor(0, 0, 0)


def _label_text_color(class_name: str | None) -> QColor:
    """Text color for the in-box class glyph: red for a red_* class, black
    otherwise (black pieces, `hand`, `board_region`, or anything
    unrecognized) -- color alone needs to carry red/black since the glyph
    itself is always lowercase (see key_shortcuts.display_label_for_class)."""
    if class_name and class_name.startswith("red_"):
        return _LABEL_TEXT_RED
    return _LABEL_TEXT_BLACK


class Handle(Enum):
    TOP_LEFT = auto()
    TOP = auto()
    TOP_RIGHT = auto()
    RIGHT = auto()
    BOTTOM_RIGHT = auto()
    BOTTOM = auto()
    BOTTOM_LEFT = auto()
    LEFT = auto()


_HANDLE_CURSORS = {
    Handle.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
    Handle.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
    Handle.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
    Handle.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
    Handle.TOP: Qt.CursorShape.SizeVerCursor,
    Handle.BOTTOM: Qt.CursorShape.SizeVerCursor,
    Handle.LEFT: Qt.CursorShape.SizeHorCursor,
    Handle.RIGHT: Qt.CursorShape.SizeHorCursor,
}


class BoxItem(QGraphicsRectItem):
    """A bounding box. `confirmed=False` renders as a dashed "suggestion"
    (from circle-detect) that is excluded from saving until the user
    confirms it. Position is always kept in `rect()` (item pos stays at the
    scene origin) so resize/move math never has to juggle two coordinate
    systems.
    """

    def __init__(self, rect: QRectF, class_name: str | None, confirmed: bool, image_bounds: QRectF):
        super().__init__(rect)
        self._class_name: str | None = None
        self.confirmed = confirmed
        self._selection_overlay_visible = True
        self.image_bounds = image_bounds
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        self._active_handle: Handle | None = None
        self._moving = False
        self._press_rect: QRectF | None = None
        self._press_scene_pos: QPointF | None = None

        # Set by ImageCanvas so undo snapshots can be captured around drags.
        self.on_drag_begin: Callable[[], None] | None = None
        self.on_drag_end: Callable[[bool], None] | None = None

        self.class_name = class_name  # through the property, to set the initial z-value

    @property
    def class_name(self) -> str | None:
        return self._class_name

    @class_name.setter
    def class_name(self, value: str | None) -> None:
        self.prepareGeometryChange()
        self._class_name = value
        self._sync_z_value()

    def itemChange(self, change, value):  # noqa: N802 (Qt override)
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            # The selected label is part of boundingRect(), so announce the
            # geometry change before Qt switches the selection state.
            self.prepareGeometryChange()
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._sync_z_value()
        return super().itemChange(change, value)

    def set_selection_overlay_visible(self, visible: bool) -> None:
        self._selection_overlay_visible = bool(visible)
        self.update()

    def _sync_z_value(self) -> None:
        if self._class_name != BOARD_REGION_CLASS:
            self.setZValue(_DEFAULT_BOX_Z_VALUE)
        elif self.isSelected():
            self.setZValue(_SELECTED_BOARD_REGION_Z_VALUE)
        else:
            self.setZValue(_UNSELECTED_BOARD_REGION_Z_VALUE)

    def handle_scene_size(self) -> float:
        scene = self.scene()
        return getattr(scene, "handle_scene_size", HANDLE_SCREEN_SIZE)

    def _handle_points(self) -> dict[Handle, QPointF]:
        r = self.rect()
        cx = (r.left() + r.right()) / 2
        cy = (r.top() + r.bottom()) / 2
        return {
            Handle.TOP_LEFT: QPointF(r.left(), r.top()),
            Handle.TOP: QPointF(cx, r.top()),
            Handle.TOP_RIGHT: QPointF(r.right(), r.top()),
            Handle.RIGHT: QPointF(r.right(), cy),
            Handle.BOTTOM_RIGHT: QPointF(r.right(), r.bottom()),
            Handle.BOTTOM: QPointF(cx, r.bottom()),
            Handle.BOTTOM_LEFT: QPointF(r.left(), r.bottom()),
            Handle.LEFT: QPointF(r.left(), cy),
        }

    def _handle_rects(self) -> dict[Handle, QRectF]:
        s = self.handle_scene_size()
        half = s / 2
        return {h: QRectF(p.x() - half, p.y() - half, s, s) for h, p in self._handle_points().items()}

    def _handle_at(self, pos: QPointF) -> Handle | None:
        for h, rect in self._handle_rects().items():
            if rect.contains(pos):
                return h
        return None

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt override)
        s = self.handle_scene_size()
        return (
            self.rect()
            .adjusted(-s, -s, s, s)
            .united(self._label_text_rect())
            .united(self._selected_label_rect())
        )

    def _label_font(self) -> QFont:
        inv_scale = self.handle_scene_size() / HANDLE_SCREEN_SIZE
        font = QFont()
        font.setPixelSize(max(1, round(LABEL_FONT_SCREEN_PX * inv_scale)))
        font.setBold(True)
        return font

    def _selected_label_font(self) -> QFont:
        inv_scale = self.handle_scene_size() / HANDLE_SCREEN_SIZE
        font = QFont()
        font.setPixelSize(max(1, round(SELECTED_LABEL_FONT_SCREEN_PX * inv_scale)))
        font.setBold(True)
        return font

    def _label_text_rect(self) -> QRectF:
        """Where the class glyph is drawn: just inside the box's top-left
        corner, no background -- an unlabeled box (empty glyph) draws
        nothing, so this degenerates to a zero-size rect at that corner
        (harmless: united()/drawText() both no-op on an empty rect)."""
        text = display_label_for_class(self.class_name)
        top_left = self.rect().topLeft()
        if not text:
            return QRectF(top_left, top_left)
        metrics = QFontMetricsF(self._label_font())
        pad = LABEL_PADDING_SCREEN_PX * (self.handle_scene_size() / HANDLE_SCREEN_SIZE)
        text_height = max(metrics.height(), 1.0)
        text_width = max(metrics.horizontalAdvance(text), 1.0)
        return QRectF(top_left.x() + pad, top_left.y() + pad, text_width, text_height)

    def _selected_label_rect(self) -> QRectF:
        """Return a large, in-image label rectangle for the selected box."""
        text = self.class_name
        if not self.isSelected() or not text:
            return QRectF()

        inv_scale = self.handle_scene_size() / HANDLE_SCREEN_SIZE
        pad = SELECTED_LABEL_PADDING_SCREEN_PX * inv_scale
        gap = SELECTED_LABEL_GAP_SCREEN_PX * inv_scale
        font = self._selected_label_font()
        metrics = QFontMetricsF(font)
        label_width = max(metrics.horizontalAdvance(text) + 2 * pad, 1.0)
        label_height = max(metrics.height() + 2 * pad, 1.0)
        box = self.rect()
        cx = (box.left() + box.right()) / 2
        cy = (box.top() + box.bottom()) / 2
        candidates = (
            QRectF(cx - label_width / 2, box.top() - gap - label_height, label_width, label_height),
            QRectF(cx - label_width / 2, box.bottom() + gap, label_width, label_height),
            QRectF(box.right() + gap, cy - label_height / 2, label_width, label_height),
            QRectF(box.left() - gap - label_width, cy - label_height / 2, label_width, label_height),
        )
        bounds = self.image_bounds
        for candidate in candidates:
            if bounds.contains(candidate):
                return candidate

        # If the box is so close to multiple edges that no side fits, keep
        # the label inside the image and choose the side with the most room.
        def visible_area(candidate: QRectF) -> float:
            clipped = candidate.intersected(bounds)
            return max(0.0, clipped.width()) * max(0.0, clipped.height())

        candidate = max(candidates, key=visible_area)
        width = min(candidate.width(), max(bounds.width(), 1.0))
        height = min(candidate.height(), max(bounds.height(), 1.0))
        left = min(max(candidate.left(), bounds.left()), bounds.right() - width)
        top = min(max(candidate.top(), bounds.top()), bounds.bottom() - height)
        return QRectF(left, top, width, height)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:  # noqa: N802
        color = class_color(self.class_name)
        pen = QPen(color)
        selected_overlay = self.isSelected() and self._selection_overlay_visible
        pen.setWidth(5 if selected_overlay else (3 if self.isSelected() else 2))
        pen.setStyle(Qt.PenStyle.SolidLine if self.confirmed else Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        if selected_overlay:
            painter.fillRect(self.rect(), QColor(30, 120, 255, 128))
        painter.drawRect(self.rect())

        if self.isSelected():
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setBrush(QBrush(color))
            for h_rect in self._handle_rects().values():
                painter.drawRect(h_rect)

        self._paint_class_label(painter)
        self._paint_selected_label(painter)

    def _paint_class_label(self, painter: QPainter) -> None:
        text = display_label_for_class(self.class_name)
        if not text:
            return
        painter.setFont(self._label_font())
        painter.setPen(_label_text_color(self.class_name))
        painter.drawText(self._label_text_rect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, text)

    def _paint_selected_label(self, painter: QPainter) -> None:
        text = self.class_name
        label_rect = self._selected_label_rect()
        if not text or label_rect.isEmpty():
            return

        inv_scale = self.handle_scene_size() / HANDLE_SCREEN_SIZE
        border_width = max(1.0, SELECTED_LABEL_BORDER_SCREEN_PX * inv_scale)
        radius = 3.0 * inv_scale
        painter.save()
        painter.setFont(self._selected_label_font())
        painter.setPen(QPen(_label_text_color(self.class_name), border_width))
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawRoundedRect(label_rect, radius, radius)
        painter.setPen(_label_text_color(self.class_name))
        pad = SELECTED_LABEL_PADDING_SCREEN_PX * inv_scale
        painter.drawText(
            label_rect.adjusted(pad, pad, -pad, -pad),
            Qt.AlignmentFlag.AlignCenter,
            text,
        )
        painter.restore()

    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        if self.isSelected():
            handle = self._handle_at(event.pos())
            if handle is not None:
                self.setCursor(_HANDLE_CURSORS[handle])
            elif self.rect().contains(event.pos()):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.unsetCursor()
        super().hoverMoveEvent(event)

    def _select_exclusively(self) -> None:
        scene = self.scene()
        if scene is not None:
            for other in scene.selectedItems():
                if other is not self:
                    other.setSelected(False)
        self.setSelected(True)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        self._select_exclusively()

        handle = self._handle_at(event.pos())
        self._press_rect = QRectF(self.rect())
        self._press_scene_pos = event.scenePos()
        if handle is not None:
            self._active_handle = handle
            self._moving = False
        elif self.rect().contains(event.pos()):
            self._active_handle = None
            self._moving = True
        else:
            self._active_handle = None
            self._moving = False

        if (self._active_handle is not None or self._moving) and self.on_drag_begin:
            self.on_drag_begin()
        event.accept()

    def _clamp_edges(self, left: float, top: float, right: float, bottom: float) -> QRectF:
        b = self.image_bounds
        left = min(max(left, b.left()), b.right())
        right = min(max(right, b.left()), b.right())
        top = min(max(top, b.top()), b.bottom())
        bottom = min(max(bottom, b.top()), b.bottom())
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        return QRectF(QPointF(left, top), QPointF(right, bottom))

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self._active_handle is None and not self._moving:
            event.ignore()
            return
        delta = event.scenePos() - self._press_scene_pos
        pr = self._press_rect
        if self._moving:
            b = self.image_bounds
            w, h = pr.width(), pr.height()
            new_left = pr.left() + delta.x()
            new_top = pr.top() + delta.y()
            max_left = max(b.left(), b.right() - w)
            max_top = max(b.top(), b.bottom() - h)
            new_left = min(max(new_left, b.left()), max_left)
            new_top = min(max(new_top, b.top()), max_top)
            self.prepareGeometryChange()
            self.setRect(QRectF(new_left, new_top, w, h))
        else:
            left, top, right, bottom = pr.left(), pr.top(), pr.right(), pr.bottom()
            h = self._active_handle
            if h in (Handle.TOP_LEFT, Handle.LEFT, Handle.BOTTOM_LEFT):
                left += delta.x()
            if h in (Handle.TOP_RIGHT, Handle.RIGHT, Handle.BOTTOM_RIGHT):
                right += delta.x()
            if h in (Handle.TOP_LEFT, Handle.TOP, Handle.TOP_RIGHT):
                top += delta.y()
            if h in (Handle.BOTTOM_LEFT, Handle.BOTTOM, Handle.BOTTOM_RIGHT):
                bottom += delta.y()
            self.prepareGeometryChange()
            self.setRect(self._clamp_edges(left, top, right, bottom))
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        changed = (self._active_handle is not None or self._moving) and self.rect() != self._press_rect
        self._active_handle = None
        self._moving = False
        self._press_rect = None
        self._press_scene_pos = None
        if self.on_drag_end:
            self.on_drag_end(changed)
        event.accept()


class CornerMarkerItem(QGraphicsRectItem):
    """Small, filled corner marker anchored in image-pixel coordinates.

    The marker geometry is recalculated by :class:`ImageCanvas` whenever
    zoom changes, so it remains a compact three-pixel-radius dot on screen
    while its centre stays exactly on the source image coordinate.
    """

    def __init__(self, corner_name: str, point: QPointF):
        super().__init__()
        self.corner_name = corner_name
        self.point = QPointF(point)
        color = _CORNER_COLORS.get(corner_name, QColor(255, 255, 255))
        self.setPen(QPen(QColor(255, 255, 255)))
        self.setBrush(QBrush(color))
        self.setZValue(10000)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{corner_name}: ({point.x():.1f}, {point.y():.1f})")

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self.setOpacity(1.0)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self.setOpacity(0.78)
        super().hoverLeaveEvent(event)

    def set_scene_radius(self, radius: float) -> None:
        self.setRect(QRectF(self.point.x() - radius, self.point.y() - radius, radius * 2, radius * 2))
        self.setOpacity(0.78)


class CanvasMode(Enum):
    SELECT = auto()
    DRAW_BOX = auto()
    MEASURE_RADIUS = auto()


class ImageCanvas(QGraphicsView):
    boxDrawn = Signal(QRectF)
    radiusMeasured = Signal(float)
    deleteRequested = Signal()
    confirmRequested = Signal()
    boxSelected = Signal(object)  # BoxItem | None
    sceneModified = Signal()
    cornerRequested = Signal(str, QPointF)  # canonical name, raw image pixel position
    clearCornersRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._scene.handle_scene_size = HANDLE_SCREEN_SIZE
        self.setScene(self._scene)
        self._scene.selectionChanged.connect(self._on_selection_changed)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_size = (0, 0)
        self._mode = CanvasMode.SELECT
        self._drawing = False
        self._draw_start_scene: QPointF | None = None
        self._temp_rect_item: QGraphicsRectItem | None = None
        self._current_scale = 1.0
        self._selection_overlay_visible = True
        self._last_scene_pos: QPointF | None = None
        self._corner_items: dict[str, CornerMarkerItem] = {}

        # Wired up by MainWindow for undo-snapshot bracketing of live drags.
        self.on_drag_begin: Callable[[], None] | None = None
        self.on_drag_end: Callable[[bool], None] | None = None

    # ---- image loading ----------------------------------------------
    def load_image(self, pixmap: QPixmap) -> None:
        self._scene.clear()
        self._corner_items.clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setZValue(-1)
        self._scene.addItem(self._pixmap_item)
        self._image_size = (pixmap.width(), pixmap.height())
        self._scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self._mode = CanvasMode.SELECT
        self._drawing = False
        self._temp_rect_item = None
        self._last_scene_pos = None

    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def image_bounds(self) -> QRectF:
        w, h = self._image_size
        return QRectF(0, 0, w, h)

    # ---- board-corner overlay ------------------------------------------
    def set_corner_points(self, points: Mapping[str, tuple[float, float] | QPointF] | None) -> None:
        """Replace the four-corner overlay without changing image metadata.

        ``points`` uses source-image pixels, never view/zoom coordinates.
        This method is deliberately presentation-only; callers own validation
        and persistence through the metadata sidecar.
        """
        for item in self._corner_items.values():
            self._scene.removeItem(item)
        self._corner_items.clear()
        if not points:
            return
        for name in CORNER_ORDER:
            value = points.get(name)
            if value is None:
                continue
            point = value if isinstance(value, QPointF) else QPointF(float(value[0]), float(value[1]))
            item = CornerMarkerItem(name, point)
            self._scene.addItem(item)
            self._corner_items[name] = item
        self._update_corner_marker_sizes()

    def corner_points(self) -> dict[str, QPointF]:
        return {name: QPointF(item.point) for name, item in self._corner_items.items()}

    def _update_corner_marker_sizes(self) -> None:
        radius = CORNER_MARKER_SCREEN_RADIUS / max(self._current_scale, 0.0001)
        for item in self._corner_items.values():
            item.set_scene_radius(radius)
        self.viewport().update()

    # ---- boxes --------------------------------------------------------
    def add_box_item(
        self,
        rect: QRectF,
        class_name: str | None,
        confirmed: bool,
        select: bool = False,
        emit_modified: bool = True,
    ) -> BoxItem:
        item = BoxItem(QRectF(rect), class_name, confirmed, self.image_bounds())
        item.set_selection_overlay_visible(self._selection_overlay_visible)
        item.on_drag_begin = lambda: self.on_drag_begin() if self.on_drag_begin else None
        item.on_drag_end = self._handle_item_drag_end
        self._scene.addItem(item)
        if select:
            self._scene.clearSelection()
            item.setSelected(True)
        if emit_modified:
            self.sceneModified.emit()
        return item

    def _handle_item_drag_end(self, changed: bool) -> None:
        if self.on_drag_end:
            self.on_drag_end(changed)
        if changed:
            self.sceneModified.emit()

    def box_items(self) -> list[BoxItem]:
        return [it for it in self._scene.items() if isinstance(it, BoxItem)]

    def selected_box(self) -> BoxItem | None:
        sel = [it for it in self._scene.selectedItems() if isinstance(it, BoxItem)]
        return sel[0] if sel else None

    def select_box(self, item: BoxItem | None) -> None:
        self._scene.clearSelection()
        if item is not None:
            item.setSelected(True)

    def remove_box(self, item: BoxItem) -> None:
        self._scene.removeItem(item)
        self.sceneModified.emit()

    def clear_unconfirmed_suggestions(self) -> None:
        for item in self.box_items():
            if not item.confirmed:
                self._scene.removeItem(item)
        self.sceneModified.emit()

    def clear_all_boxes(self) -> None:
        """Remove every box without emitting sceneModified -- used by undo/
        redo snapshot restore, where the caller marks dirty exactly once."""
        for item in self.box_items():
            self._scene.removeItem(item)

    # ---- modes ----------------------------------------------------------
    def set_mode(self, mode: CanvasMode) -> None:
        self._mode = mode

    def mode(self) -> CanvasMode:
        return self._mode

    # ---- zoom -------------------------------------------------------
    def current_scale(self) -> float:
        return self._current_scale

    def selection_overlay_visible(self) -> bool:
        return self._selection_overlay_visible

    def set_selection_overlay_visible(self, visible: bool) -> None:
        self._selection_overlay_visible = bool(visible)
        for box in self.box_items():
            box.set_selection_overlay_visible(self._selection_overlay_visible)
        self.viewport().update()

    def toggle_selection_overlay(self) -> bool:
        self.set_selection_overlay_visible(not self._selection_overlay_visible)
        return self._selection_overlay_visible

    def set_zoom(self, scale: float) -> None:
        scale = max(0.05, min(scale, 20.0))
        self._current_scale = scale
        self.resetTransform()
        self.scale(scale, scale)
        self._scene.handle_scene_size = HANDLE_SCREEN_SIZE / scale
        self._update_corner_marker_sizes()

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self._current_scale * factor)

    def fit_to_window(self) -> None:
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._current_scale = self.transform().m11()
        self._scene.handle_scene_size = HANDLE_SCREEN_SIZE / self._current_scale
        self._update_corner_marker_sizes()

    def zoom_to_board_region(self) -> bool:
        """Fit the board_region box plus a safety margin to the viewport."""
        board_boxes = [b for b in self.box_items() if b.class_name == "board_region"]
        if not board_boxes or self._pixmap_item is None:
            return False
        board = max(board_boxes, key=lambda item: item.rect().width() * item.rect().height())
        board_rect = board.rect()
        # Pieces sitting on the outer intersections extend beyond the board
        # region itself. Keep roughly one half-piece of breathing room so the
        # edge pieces remain fully visible after zooming.
        margin_x = board_rect.width() * 0.08
        margin_y = board_rect.height() * 0.08
        zoom_rect = board_rect.adjusted(-margin_x, -margin_y, margin_x, margin_y)
        self.fitInView(zoom_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._current_scale = self.transform().m11()
        self._scene.handle_scene_size = HANDLE_SCREEN_SIZE / max(self._current_scale, 0.0001)
        self.centerOn(board.rect().center())
        self._update_corner_marker_sizes()
        return True

    def reset_to_original_size(self) -> None:
        """Show the image at 1:1 source-pixel scale."""
        if self._pixmap_item is None:
            return
        self.set_zoom(1.0)
        self.centerOn(self.image_bounds().center())

    def fit_to_width(self) -> None:
        if self._pixmap_item is None or self._image_size[0] == 0:
            return
        viewport_w = self.viewport().width()
        self.set_zoom(viewport_w / self._image_size[0])

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.25 if event.angleDelta().y() > 0 else 0.8
            self.zoom_by(factor)
            event.accept()
        else:
            super().wheelEvent(event)

    # ---- draw-box / measure-radius drag gestures -----------------------
    def _begin_drag(self, event) -> None:
        self._drawing = True
        self._draw_start_scene = self.mapToScene(event.pos())
        self._temp_rect_item = QGraphicsRectItem(QRectF(self._draw_start_scene, self._draw_start_scene))
        pen = QPen(QColor(0, 200, 255))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(2)
        self._temp_rect_item.setPen(pen)
        self._temp_rect_item.setZValue(1000)
        self._scene.addItem(self._temp_rect_item)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._last_scene_pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and self._mode in (
            CanvasMode.DRAW_BOX,
            CanvasMode.MEASURE_RADIUS,
        ):
            item = self.itemAt(event.pos())
            if item is None or item is self._pixmap_item:
                self._begin_drag(event)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._last_scene_pos = self.mapToScene(event.position().toPoint())
        if self._drawing and self._temp_rect_item is not None:
            cur = self.mapToScene(event.pos())
            rect = QRectF(self._draw_start_scene, cur).normalized()
            rect = rect.intersected(self.image_bounds())
            self._temp_rect_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._last_scene_pos = self.mapToScene(event.position().toPoint())
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            rect = self._temp_rect_item.rect()
            self._scene.removeItem(self._temp_rect_item)
            self._temp_rect_item = None
            self._drawing = False
            mode = self._mode
            self._mode = CanvasMode.SELECT
            if rect.width() >= MIN_DRAG_PX and rect.height() >= MIN_DRAG_PX:
                if mode == CanvasMode.DRAW_BOX:
                    self.boxDrawn.emit(rect)
                elif mode == CanvasMode.MEASURE_RADIUS:
                    r0 = (rect.width() + rect.height()) / 4.0
                    self.radiusMeasured.emit(r0)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        corner_by_key = {
            Qt.Key.Key_1: "top_left",
            Qt.Key.Key_2: "top_right",
            Qt.Key.Key_3: "bottom_right",
            Qt.Key.Key_4: "bottom_left",
        }
        corner_name = corner_by_key.get(event.key())
        if corner_name is not None:
            pos = self._last_scene_pos
            if pos is not None and self.image_bounds().contains(pos):
                self.cornerRequested.emit(corner_name, QPointF(pos))
            event.accept()
            return
        if event.key() == Qt.Key.Key_0:
            self.clearCornersRequested.emit()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.selected_box() is not None:
                self.deleteRequested.emit()
                event.accept()
                return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            box = self.selected_box()
            if box is not None and not box.confirmed:
                self.confirmRequested.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    def _on_selection_changed(self) -> None:
        self.boxSelected.emit(self.selected_box())
