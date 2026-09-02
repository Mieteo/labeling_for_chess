"""Side dock panels: the image file browser and the current image's box list."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from . import metadata, yolo_io
from .canvas import BoxItem, class_color
from .key_shortcuts import BOARD_REGION_CLASS, CLASS_DISPLAY_ORDER, HAND_CLASS

_LABELED_COLOR = QColor(46, 125, 50)
_UNLABELED_COLOR = QColor(130, 130, 130)

# Two columns instead of one flat list: left = black pieces + not-yet-labeled
# boxes, right = red pieces + the colorless `hand` and `board_region`
# classes. Each column keeps the same fixed mnemonic sub-order as before (see
# CLASS_DISPLAY_ORDER), with its "leftover" bucket (unassigned / hand /
# board_region) sorted last within it.
_BLACK_COLUMN_ORDER = CLASS_DISPLAY_ORDER[:7]
_RED_COLUMN_ORDER = CLASS_DISPLAY_ORDER[7:14]
_BLACK_SORT_INDEX = {name: i for i, name in enumerate(_BLACK_COLUMN_ORDER)}
_RED_SORT_INDEX = {name: i for i, name in enumerate(_RED_COLUMN_ORDER)}
_BLACK_UNASSIGNED_SORT_INDEX = len(_BLACK_COLUMN_ORDER)
_RED_HAND_SORT_INDEX = len(_RED_COLUMN_ORDER)
_RED_BOARD_REGION_SORT_INDEX = _RED_HAND_SORT_INDEX + 1


def _is_red_column(class_name: str | None) -> bool:
    return class_name in (HAND_CLASS, BOARD_REGION_CLASS) or class_name in _RED_SORT_INDEX


def _black_sort_key(box: BoxItem) -> int:
    return _BLACK_SORT_INDEX.get(box.class_name, _BLACK_UNASSIGNED_SORT_INDEX)


def _red_sort_key(box: BoxItem) -> int:
    if box.class_name == HAND_CLASS:
        return _RED_HAND_SORT_INDEX
    if box.class_name == BOARD_REGION_CLASS:
        return _RED_BOARD_REGION_SORT_INDEX
    return _RED_SORT_INDEX.get(box.class_name, _RED_BOARD_REGION_SORT_INDEX)


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
        layout.setSpacing(4)
        self._cohort_filter = QComboBox(self)
        self._cohort_filter.setObjectName("contentCohortFilter")
        self._cohort_filter.currentIndexChanged.connect(self._apply_filter)
        layout.addWidget(self._cohort_filter)
        self._cohort_summary = QLabel("Cohort: chưa có ảnh", self)
        self._cohort_summary.setObjectName("contentCohortSummary")
        self._cohort_summary.setWordWrap(True)
        self._cohort_summary.setStyleSheet("color: #555;")
        layout.addWidget(self._cohort_summary)
        self._list = QListWidget(self)
        self._list.itemActivated.connect(self._on_activated)
        self._list.itemClicked.connect(self._on_activated)
        layout.addWidget(self._list)
        self._paths: list[Path] = []
        self._visible_indices: list[int] = []
        self._content_cohorts: dict[Path, str | None] = {}

    def set_images(self, paths: list[Path]) -> None:
        self._paths = list(paths)
        self._content_cohorts = {path: self._read_content_cohort(path) for path in self._paths}
        self._refresh_filter_options()
        self._apply_filter()

    @staticmethod
    def _read_content_cohort(path: Path) -> str | None:
        try:
            record = metadata.load_metadata(path)
        except metadata.MetadataError:
            # A malformed sidecar stays visible as unassigned for discovery;
            # it is never repaired or inferred here.
            return None
        return record.capture.content_cohort if record is not None else None

    def content_cohort_counts(self) -> dict[str, int]:
        counts = {"unassigned": 0, **{cohort: 0 for cohort in sorted(metadata.CONTENT_COHORTS)}}
        for cohort in self._content_cohorts.values():
            counts[cohort if cohort is not None else "unassigned"] += 1
        return counts

    def _refresh_filter_options(self) -> None:
        selected = self._cohort_filter.currentData()
        counts = self.content_cohort_counts()
        options: list[tuple[str, str]] = [
            (f"Tất cả ({len(self._paths)})", "all"),
            (f"Chưa gán nhãn ({counts['unassigned']})", "unassigned"),
            (f"Bàn cờ thật ({counts['real']})", "real"),
            (f"Ảnh chụp màn hình ({counts['native_screenshot']})", "native_screenshot"),
            (f"Render tạo bằng code ({counts['procedural_render']})", "procedural_render"),
            (f"Ảnh chụp lại màn hình ({counts['screen_photo']})", "screen_photo"),
            (f"Chưa xác định ({counts['unknown']})", "unknown"),
        ]
        self._cohort_filter.blockSignals(True)
        try:
            self._cohort_filter.clear()
            for label, value in options:
                self._cohort_filter.addItem(label, value)
            index = self._cohort_filter.findData(selected)
            self._cohort_filter.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self._cohort_filter.blockSignals(False)
        self._cohort_summary.setText(
            "Cohort: "
            f"chưa gán {counts['unassigned']} • thật {counts['real']} • "
            f"screenshot {counts['native_screenshot']} • render {counts['procedural_render']} • "
            f"chụp màn hình {counts['screen_photo']} • chưa xác định {counts['unknown']}"
        )

    def _apply_filter(self, *_: object) -> None:
        selected = self._cohort_filter.currentData() or "all"
        self._visible_indices = [
            index
            for index, path in enumerate(self._paths)
            if selected == "all"
            or (selected == "unassigned" and self._content_cohorts.get(path) is None)
            or self._content_cohorts.get(path) == selected
        ]
        self._list.clear()
        for index in self._visible_indices:
            self._list.addItem(QListWidgetItem(self._paths[index].name))
        self.refresh_all_labels()

    def refresh_all_labels(self) -> None:
        for i in range(len(self._visible_indices)):
            self._refresh_label_at(i)

    def refresh_label_at(self, index: int) -> None:
        if index in self._visible_indices:
            self._refresh_label_at(self._visible_indices.index(index))

    def refresh_content_cohort_at(self, index: int) -> None:
        if not (0 <= index < len(self._paths)):
            return
        path = self._paths[index]
        self._content_cohorts[path] = self._read_content_cohort(path)
        self._refresh_filter_options()
        self._apply_filter()

    def refresh_all_content_cohorts(self) -> None:
        self._content_cohorts = {path: self._read_content_cohort(path) for path in self._paths}
        self._refresh_filter_options()
        self._apply_filter()

    def _refresh_label_at(self, visible_index: int) -> None:
        item = self._list.item(visible_index)
        if item is None:
            return
        path = self._paths[self._visible_indices[visible_index]]
        labeled = yolo_io.has_label(path)
        font = item.font()
        font.setBold(labeled)
        item.setFont(font)
        item.setForeground(_LABELED_COLOR if labeled else _UNLABELED_COLOR)
        item.setText(("✓ " if labeled else "    ") + path.name)

    def set_current_index(self, index: int) -> None:
        if index not in self._visible_indices:
            self._list.blockSignals(True)
            self._list.setCurrentRow(-1)
            self._list.blockSignals(False)
            return
        visible_index = self._visible_indices.index(index)
        self._list.blockSignals(True)
        self._list.setCurrentRow(visible_index)
        self._list.blockSignals(False)
        item = self._list.item(visible_index)
        if item is not None:
            self._list.scrollToItem(item)

    def _on_activated(self, item: QListWidgetItem) -> None:
        visible_index = self._list.row(item)
        if 0 <= visible_index < len(self._visible_indices):
            self.imageActivated.emit(self._visible_indices[visible_index])


class BoxListPanel(QWidget):
    """Lists every box drawn on the current image (class name, suggestion
    marker), split into two side-by-side columns -- left for black pieces
    and not-yet-labeled boxes, right for red pieces, `hand`, and
    `board_region` -- each independently sorted (see the `_*_sort_key`
    functions above). Click
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

    def review_order(self) -> list[BoxItem]:
        """Return boxes in the same black-first order used by the lists."""
        return [*self._black_items, *self._red_items]

    def _on_clicked(self, list_widget: QListWidget, item: QListWidgetItem) -> None:
        items = self._black_items if list_widget is self._black_list else self._red_items
        idx = list_widget.row(item)
        if 0 <= idx < len(items):
            self.boxActivated.emit(items[idx])
