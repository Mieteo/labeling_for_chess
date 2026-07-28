"""Side dock panels: the image file browser and the current image's box list."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from . import yolo_io
from .canvas import BoxItem, class_color
from .key_shortcuts import CLASS_DISPLAY_ORDER, HAND_CLASS

_LABELED_COLOR = QColor(46, 125, 50)
_UNLABELED_COLOR = QColor(130, 130, 130)

# Two columns instead of one flat list: left = black pieces + not-yet-labeled
# boxes, right = red pieces + the colorless `hand` class. Each column keeps
# the same fixed mnemonic sub-order as before (see CLASS_DISPLAY_ORDER),
# with its "leftover" bucket (unassigned / hand) sorted last within it.
_BLACK_COLUMN_ORDER = CLASS_DISPLAY_ORDER[:7]
_RED_COLUMN_ORDER = CLASS_DISPLAY_ORDER[7:14]
_BLACK_SORT_INDEX = {name: i for i, name in enumerate(_BLACK_COLUMN_ORDER)}
_RED_SORT_INDEX = {name: i for i, name in enumerate(_RED_COLUMN_ORDER)}
_BLACK_UNASSIGNED_SORT_INDEX = len(_BLACK_COLUMN_ORDER)
_RED_HAND_SORT_INDEX = len(_RED_COLUMN_ORDER)


def _is_red_column(class_name: str | None) -> bool:
    return class_name == HAND_CLASS or class_name in _RED_SORT_INDEX


def _black_sort_key(box: BoxItem) -> int:
    return _BLACK_SORT_INDEX.get(box.class_name, _BLACK_UNASSIGNED_SORT_INDEX)


def _red_sort_key(box: BoxItem) -> int:
    if box.class_name == HAND_CLASS:
        return _RED_HAND_SORT_INDEX
    return _RED_SORT_INDEX.get(box.class_name, _RED_HAND_SORT_INDEX)


class FileListPanel(QWidget):
    """Flat list of every image in the open directory. Bold + checkmark
    marks images that already have a sibling `.txt` (reviewed), matching
    the labeled/unlabeled distinction required by the spec. Click (or
    Enter) jumps to that image.
    """

    imageActivated = Signal(int)  # row index

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget(self)
        self._list.itemActivated.connect(self._on_activated)
        self._list.itemClicked.connect(self._on_activated)
        layout.addWidget(self._list)
        self._paths: list[Path] = []

    def set_images(self, paths: list[Path]) -> None:
        self._paths = list(paths)
        self._list.clear()
        for p in self._paths:
            self._list.addItem(QListWidgetItem(p.name))
        self.refresh_all_labels()

    def refresh_all_labels(self) -> None:
        for i in range(len(self._paths)):
            self._refresh_label_at(i)

    def refresh_label_at(self, index: int) -> None:
        if 0 <= index < len(self._paths):
            self._refresh_label_at(index)

    def _refresh_label_at(self, index: int) -> None:
        item = self._list.item(index)
        if item is None:
            return
        path = self._paths[index]
        labeled = yolo_io.has_label(path)
        font = item.font()
        font.setBold(labeled)
        item.setFont(font)
        item.setForeground(_LABELED_COLOR if labeled else _UNLABELED_COLOR)
        item.setText(("✓ " if labeled else "    ") + path.name)

    def set_current_index(self, index: int) -> None:
        self._list.blockSignals(True)
        self._list.setCurrentRow(index)
        self._list.blockSignals(False)
        item = self._list.item(index)
        if item is not None:
            self._list.scrollToItem(item)

    def _on_activated(self, item: QListWidgetItem) -> None:
        self.imageActivated.emit(self._list.row(item))


class BoxListPanel(QWidget):
    """Lists every box drawn on the current image (class name, suggestion
    marker), split into two side-by-side columns -- left for black pieces
    and not-yet-labeled boxes, right for red pieces and `hand` -- each
    independently sorted (see the `_*_sort_key` functions above). Click
    highlights/selects the matching box on the canvas; only one column ever
    shows a highlighted row at a time, matching the single canvas selection.
    """

    boxActivated = Signal(object)  # BoxItem

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        self._black_list = self._build_column(columns, "Đen / chưa gán")
        self._red_list = self._build_column(columns, "Đỏ / hand")
        layout.addLayout(columns)

        self._black_items: list[BoxItem] = []
        self._red_items: list[BoxItem] = []

    def _build_column(self, columns: QHBoxLayout, title: str) -> QListWidget:
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title, self)
        label.setStyleSheet("font-weight: bold; color: #666;")
        column.addWidget(label)
        list_widget = QListWidget(self)
        list_widget.itemClicked.connect(lambda item, lw=list_widget: self._on_clicked(lw, item))
        column.addWidget(list_widget)
        columns.addLayout(column)
        return list_widget

    def set_boxes(self, boxes: list[BoxItem]) -> None:
        black = [b for b in boxes if not _is_red_column(b.class_name)]
        red = [b for b in boxes if _is_red_column(b.class_name)]
        # sorted() is stable, so boxes sharing a class keep their relative order.
        self._black_items = sorted(black, key=_black_sort_key)
        self._red_items = sorted(red, key=_red_sort_key)
        self._populate(self._black_list, self._black_items)
        self._populate(self._red_list, self._red_items)

    @staticmethod
    def _populate(list_widget: QListWidget, items: list[BoxItem]) -> None:
        list_widget.clear()
        for box in items:
            label = box.class_name or "(chưa gán lớp)"
            if not box.confirmed:
                label = f"[gợi ý] {label}"
            item = QListWidgetItem(label)
            item.setForeground(class_color(box.class_name))
            list_widget.addItem(item)

    def select_box(self, box: BoxItem | None) -> None:
        for list_widget, items in ((self._black_list, self._black_items), (self._red_list, self._red_items)):
            idx = -1
            if box is not None:
                for i, b in enumerate(items):
                    if b is box:
                        idx = i
                        break
            list_widget.blockSignals(True)
            list_widget.setCurrentRow(idx)
            list_widget.blockSignals(False)

    def _on_clicked(self, list_widget: QListWidget, item: QListWidgetItem) -> None:
        items = self._black_items if list_widget is self._black_list else self._red_items
        idx = list_widget.row(item)
        if 0 <= idx < len(items):
            self.boxActivated.emit(items[idx])
