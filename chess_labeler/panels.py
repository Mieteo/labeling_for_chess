"""Side dock panels: the image file browser and the current image's box list."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from . import yolo_io
from .canvas import BoxItem, class_color

_LABELED_COLOR = QColor(46, 125, 50)
_UNLABELED_COLOR = QColor(130, 130, 130)


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
    marker). Click highlights/selects the matching box on the canvas.
    """

    boxActivated = Signal(object)  # BoxItem

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget(self)
        self._list.itemClicked.connect(self._on_clicked)
        layout.addWidget(self._list)
        self._items: list[BoxItem] = []

    def set_boxes(self, boxes: list[BoxItem]) -> None:
        self._items = list(boxes)
        self._list.clear()
        for box in self._items:
            label = box.class_name or "(chua gan lop)"
            if not box.confirmed:
                label = f"[goi y] {label}"
            item = QListWidgetItem(label)
            item.setForeground(class_color(box.class_name))
            self._list.addItem(item)

    def select_box(self, box: BoxItem | None) -> None:
        idx = -1
        if box is not None:
            for i, b in enumerate(self._items):
                if b is box:
                    idx = i
                    break
        self._list.blockSignals(True)
        self._list.setCurrentRow(idx)
        self._list.blockSignals(False)

    def _on_clicked(self, item: QListWidgetItem) -> None:
        idx = self._list.row(item)
        if 0 <= idx < len(self._items):
            self.boxActivated.emit(self._items[idx])
