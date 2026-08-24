"""Compact board-level metadata controls for ``<stem>.meta.json`` sidecars.

The interface deliberately shows annotators only the information they need to
enter: board orientation, whether corners/FEN are present, and capture
conditions.  The version-1 sidecar keeps a few legacy quality/review fields
for backwards compatibility, but those technical fields are derived by the
main window rather than being exposed as confusing choices in this panel.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# Display text is Vietnamese; the value stored in the JSON remains the stable
# English schema code in itemData.  Do not use itemText() when persisting.
CAPTURE_DISPLAY_OPTIONS: dict[str, list[tuple[str, str]]] = {
    "lighting": [
        ("Chưa xác định", "unknown"),
        ("Rất tối", "very_dark"),
        ("Ánh sáng yếu", "dim"),
        ("Ánh sáng đều", "even"),
        ("Sáng", "bright"),
        ("Ánh sáng pha trộn", "mixed"),
    ],
    "shadow": [
        ("Chưa xác định", "unknown"),
        ("Không có", "none"),
        ("Nhẹ", "mild"),
        ("Nhiều", "strong"),
    ],
    "glare": [
        ("Chưa xác định", "unknown"),
        ("Không có", "none"),
        ("Nhẹ", "mild"),
        ("Nhiều", "strong"),
    ],
    "perspective": [
        ("Chưa xác định", "unknown"),
        ("Chính diện", "frontal"),
        ("Nghiêng nhẹ", "mild"),
        ("Nghiêng nhiều", "strong"),
        ("Cực nghiêng", "extreme"),
    ],
    "board_material": [
        ("Chưa xác định", "unknown"),
        ("Gỗ", "wood"),
        ("Nhựa", "plastic"),
        ("Giấy", "paper"),
        ("Đá", "stone"),
        ("Khác", "other"),
    ],
    "board_fill": [
        ("Chưa xác định", "unknown"),
        ("Rất nhỏ trong ảnh", "tiny"),
        ("Nhỏ trong ảnh", "small"),
        ("Vừa trong ảnh", "medium"),
        ("Lớn trong ảnh", "large"),
        ("Rất lớn trong ảnh", "very_large"),
    ],
    "distance": [
        ("Chưa xác định", "unknown"),
        ("Gần", "near"),
        ("Vừa", "medium"),
        ("Xa", "far"),
    ],
    "blur": [
        ("Chưa xác định", "unknown"),
        ("Không mờ", "none"),
        ("Mờ nhẹ", "mild"),
        ("Mờ nhiều", "strong"),
    ],
    "occlusion": [
        ("Chưa xác định", "unknown"),
        ("Không bị che", "none"),
        ("Bị tay che", "hand"),
        ("Bị quân cờ che", "piece"),
        ("Bị vật thể che", "object"),
        ("Bị nhiều thứ che", "multiple"),
    ],
    "occlusion_severity": [
        ("Chưa xác định", "unknown"),
        ("Không bị che", "none"),
        ("Che nhẹ", "mild"),
        ("Che nhiều", "strong"),
    ],
    "environment": [
        ("Chưa xác định", "unknown"),
        ("Trong nhà", "indoor"),
        ("Ngoài trời", "outdoor"),
        ("Cả trong và ngoài nhà", "mixed"),
    ],
}

# Kept as a small compatibility surface for callers/tests that need the
# persisted vocabulary, not the translated presentation text.
CAPTURE_OPTIONS: dict[str, list[str]] = {
    field: [value for _label, value in options]
    for field, options in CAPTURE_DISPLAY_OPTIONS.items()
}

ORIENTATION_OPTIONS: list[tuple[str, str]] = [
    ("Chưa xác định", "unknown"),
    ("Đỏ ở phía dưới", "red_at_bottom"),
    ("Đỏ ở phía trên", "red_at_top"),
    ("Đỏ ở bên trái", "red_at_left"),
    ("Đỏ ở bên phải", "red_at_right"),
]

# A blank value deliberately means "not labeled yet".  It is not persisted as
# ``unknown`` because old sidecars and new drafts must remain distinguishable
# from an annotator explicitly choosing the unknown cohort.
CONTENT_COHORT_OPTIONS: list[tuple[str, str | None]] = [
    ("Chưa gán nhãn", None),
    ("Bàn cờ thật", "real"),
    ("Ảnh chụp màn hình", "native_screenshot"),
    ("Render tạo bằng code", "procedural_render"),
    ("Ảnh chụp lại màn hình", "screen_photo"),
    ("Chưa xác định", "unknown"),
]

_FIELD_LABELS = {
    "lighting": "Ánh sáng",
    "shadow": "Bóng đổ",
    "glare": "Chói/lóa",
    "perspective": "Góc chụp",
    "board_material": "Chất liệu bàn",
    "board_fill": "Bàn trong ảnh",
    "distance": "Khoảng cách",
    "blur": "Độ mờ",
    "occlusion": "Che khuất",
    "occlusion_severity": "Mức che",
    "environment": "Môi trường",
}

_CORNER_ERROR_LABELS = {
    "missing_corners": "Chưa đủ bốn góc",
    "corner_out_of_bounds": "Có góc nằm ngoài ảnh",
    "duplicate_corners": "Có hai góc trùng nhau",
    "self_intersecting_polygon": "Tứ giác tự cắt (bow-tie)",
    "non_convex_polygon": "Bốn góc không tạo tứ giác lồi",
    "wrong_corner_order": "Thứ tự góc không theo chiều kim đồng hồ",
    "board_area_too_small": "Diện tích bàn quá nhỏ (< 1% ảnh)",
}


def _nested(mapping: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    value = (mapping or {}).get(key, {})
    return value if isinstance(value, Mapping) else {}


class MetadataPanel(QWidget):
    """One compact, no-scroll form for board and capture metadata."""

    metadataChanged = Signal()
    applyNextRequested = Signal()
    applyRangeRequested = Signal()
    openFenRequested = Signal()
    confirmFenRequested = Signal()
    clearFenRequested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._loading = False
        self._corner_errors: list[str] = []
        self._fen_errors: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(5)

        self._completeness = QLabel("Chưa đủ dữ liệu benchmark", self)
        self._completeness.setObjectName("metadataCompleteness")
        self._completeness.setStyleSheet("font-weight: bold; color: #b36b00; padding: 3px;")
        root.addWidget(self._completeness)

        corner_hint = QLabel(
            "<b>Đánh dấu 4 góc:</b> đưa chuột lên giao điểm lưới ngoài cùng trên ảnh, "
            "rồi bấm <b>1</b> trên-trái, <b>2</b> trên-phải, <b>3</b> dưới-phải, "
            "<b>4</b> dưới-trái. Bấm <b>0</b> để xóa cả 4 góc.",
            self,
        )
        corner_hint.setObjectName("cornerInstruction")
        corner_hint.setWordWrap(True)
        corner_hint.setStyleSheet("color: #555;")
        root.addWidget(corner_hint)

        board_grid = QGridLayout()
        board_grid.setHorizontalSpacing(8)
        board_grid.setVerticalSpacing(4)
        root.addLayout(board_grid)
        self._orientation = self._combo(ORIENTATION_OPTIONS, self)
        self._add_compact_field(board_grid, 0, 0, "Hướng ảnh:", self._orientation, self)

        self._corner_presence = self._status_label("Chưa đủ 4 góc")
        self._corner_presence.setObjectName("cornerPresence")
        self._add_compact_field(board_grid, 0, 2, "4 góc bàn:", self._corner_presence, self)

        self._fen_presence = self._status_label("Chưa có FEN")
        self._fen_presence.setObjectName("fenPresence")
        self._add_compact_field(board_grid, 1, 0, "FEN bàn cờ:", self._fen_presence, self)
        fen_actions = QWidget(self)
        fen_actions_layout = QHBoxLayout(fen_actions)
        fen_actions_layout.setContentsMargins(0, 0, 0, 0)
        fen_actions_layout.setSpacing(3)
        open_fen = QPushButton("Mở FEN", fen_actions)
        open_fen.setToolTip("Mở tab để nhập/chỉnh thế cờ; FEN được tạo tự động từ bàn cờ")
        open_fen.clicked.connect(self.openFenRequested)
        fen_actions_layout.addWidget(open_fen)
        confirm_fen = QPushButton("Xác nhận FEN", fen_actions)
        confirm_fen.setToolTip("Lưu FEN của bàn cờ đang hiển thị, kể cả thế cờ đầu")
        confirm_fen.clicked.connect(self.confirmFenRequested)
        fen_actions_layout.addWidget(confirm_fen)
        clear_fen = QPushButton("Bỏ FEN", fen_actions)
        clear_fen.setToolTip("Bỏ FEN đã xác nhận cho ảnh này")
        clear_fen.clicked.connect(self.clearFenRequested)
        fen_actions_layout.addWidget(clear_fen)
        board_grid.addWidget(fen_actions, 1, 2, 1, 2)

        self._content_cohort = self._combo(CONTENT_COHORT_OPTIONS, self)
        self._content_cohort.setObjectName("capture_content_cohort")
        self._add_compact_field(board_grid, 2, 0, "Loại ảnh:", self._content_cohort, self)

        self._corner_validation = QLabel("", self)
        self._corner_validation.setObjectName("cornerValidation")
        self._corner_validation.setWordWrap(True)
        root.addWidget(self._corner_validation)
        self._fen_validation = QLabel("", self)
        self._fen_validation.setObjectName("fenValidation")
        self._fen_validation.setWordWrap(True)
        root.addWidget(self._fen_validation)

        capture_title = QLabel("<b>Điều kiện chụp</b>", self)
        root.addWidget(capture_title)
        capture_grid = QGridLayout()
        capture_grid.setHorizontalSpacing(8)
        capture_grid.setVerticalSpacing(4)
        root.addLayout(capture_grid)
        self._capture_controls: dict[str, QComboBox] = {}
        capture_fields = (
            "lighting",
            "perspective",
            "board_fill",
            "blur",
            "occlusion",
            "environment",
            "shadow",
            "glare",
            "board_material",
            "distance",
            "occlusion_severity",
        )
        for index, field in enumerate(capture_fields):
            combo = self._combo(CAPTURE_DISPLAY_OPTIONS[field], self)
            combo.setObjectName(f"capture_{field}")
            self._capture_controls[field] = combo
            self._add_compact_field(
                capture_grid,
                index // 2,
                (index % 2) * 2,
                f"{_FIELD_LABELS[field]}:",
                combo,
                self,
            )

        self._capture_group = QComboBox(self)
        self._capture_group.setObjectName("capture_capture_group")
        self._capture_group.setEditable(True)
        self._capture_group.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._capture_group.lineEdit().setPlaceholderText("Để trống nếu chưa biết")
        self._capture_group.addItem("Chưa gán", "")
        self._add_compact_field(
            capture_grid,
            (len(capture_fields) + 1) // 2,
            0,
            "Nhóm chụp:",
            self._capture_group,
            self,
        )

        self._style_or_app = QLineEdit(self)
        self._style_or_app.setObjectName("capture_style_or_app")
        self._style_or_app.setPlaceholderText("Ví dụ: Xiangqi app A, Ky Nhan marble")
        self._add_compact_field(
            capture_grid,
            (len(capture_fields) + 1) // 2,
            2,
            "App / style:",
            self._style_or_app,
            self,
        )

        extra_row = (len(capture_fields) + 3) // 2
        self._capture_session = QLineEdit(self)
        self._capture_session.setObjectName("capture_capture_session")
        self._capture_session.setPlaceholderText("Buổi / thiết bị (tùy chọn)")
        self._add_compact_field(
            capture_grid, extra_row, 0, "Buổi chụp:", self._capture_session, self
        )
        self._position_id = QLineEdit(self)
        self._position_id.setObjectName("capture_position_id")
        self._position_id.setPlaceholderText("Mã nhóm thế cờ (tùy chọn)")
        self._add_compact_field(
            capture_grid, extra_row, 2, "Mã thế cờ:", self._position_id, self
        )

        template_row = QHBoxLayout()
        template_row.setContentsMargins(0, 0, 0, 0)
        next_button = QPushButton("Áp dụng cho ảnh tiếp", self)
        next_button.setToolTip("Áp dụng các điều kiện chụp này cho ảnh mới kế tiếp")
        next_button.clicked.connect(self.applyNextRequested)
        template_row.addWidget(next_button)
        range_button = QPushButton("Áp dụng cho dải…", self)
        range_button.setToolTip("Áp dụng các điều kiện chụp cho một dải ảnh có xác nhận")
        range_button.clicked.connect(self.applyRangeRequested)
        template_row.addWidget(range_button)
        template_row.addStretch(1)
        root.addLayout(template_row)

        self._notes = QLineEdit(self)
        self._notes.setObjectName("metadataNotes")
        self._notes.setPlaceholderText("Ghi chú (tùy chọn)")
        root.addWidget(self._notes)

        for combo in [self._orientation, self._content_cohort, *self._capture_controls.values()]:
            combo.currentIndexChanged.connect(self._emit_changed)
        self._capture_controls["occlusion"].currentIndexChanged.connect(self._sync_occlusion_severity)
        self._capture_group.currentTextChanged.connect(self._emit_changed)
        self._capture_group.lineEdit().editingFinished.connect(self._emit_changed)
        for edit in (self._style_or_app, self._capture_session, self._position_id):
            edit.editingFinished.connect(self._emit_changed)
        self._notes.editingFinished.connect(self._emit_changed)

        self.set_values({})

    @staticmethod
    def _combo(options: list[tuple[str, str | None]], parent: QWidget) -> QComboBox:
        combo = QComboBox(parent)
        for label, value in options:
            combo.addItem(label, value)
        return combo

    @staticmethod
    def _status_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: 600; color: #8a5a00;")
        return label

    @staticmethod
    def _add_compact_field(
        layout: QGridLayout,
        row: int,
        column: int,
        label: str,
        widget: QWidget,
        parent: QWidget,
    ) -> None:
        """Place a label and input in one of two compact grid columns."""

        layout.addWidget(QLabel(label, parent), row, column)
        layout.addWidget(widget, row, column + 1)
        layout.setColumnStretch(column + 1, 1)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: object, fallback: str = "unknown") -> None:
        code = value if isinstance(value, str) and value else fallback
        index = combo.findData(code)
        combo.setCurrentIndex(index if index >= 0 else 0)

    @staticmethod
    def _set_optional_combo_data(combo: QComboBox, value: object) -> None:
        code = value if isinstance(value, str) and value else None
        index = combo.findData(code)
        combo.setCurrentIndex(index if index >= 0 else 0)

    @staticmethod
    def _combo_value(combo: QComboBox, fallback: str = "unknown") -> str:
        value = combo.currentData()
        return value if isinstance(value, str) and value else fallback

    def _capture_group_value(self) -> str | None:
        if (
            self._capture_group.currentIndex() == 0
            and self._capture_group.currentData() == ""
            and self._capture_group.currentText() == self._capture_group.itemText(0)
        ):
            return None
        value = self._capture_group.currentText().strip()
        return value or None

    def _set_capture_group(self, value: object) -> None:
        text = value.strip() if isinstance(value, str) else ""
        self._capture_group.blockSignals(True)
        try:
            if text and self._capture_group.findData(text) < 0:
                self._capture_group.addItem(text, text)
            if text:
                self._capture_group.setCurrentText(text)
            else:
                self._capture_group.setCurrentIndex(0)
        finally:
            self._capture_group.blockSignals(False)

    def set_values(self, values: Mapping[str, Any] | None) -> None:
        """Load schema-shaped values without emitting dirty notifications."""

        board = _nested(values, "board")
        capture = _nested(values, "capture")
        review = _nested(values, "review")
        corners = _nested(board, "corners_px")
        all_corners_present = bool(corners) and all(
            corners.get(name) is not None
            for name in ("top_left", "top_right", "bottom_right", "bottom_left")
        )
        corner_count = sum(
            corners.get(name) is not None
            for name in ("top_left", "top_right", "bottom_right", "bottom_left")
        )
        has_fen = isinstance(board.get("board_fen"), str) and bool(board.get("board_fen"))

        self._loading = True
        try:
            self._set_combo_data(self._orientation, board.get("image_orientation"))
            self._set_optional_combo_data(self._content_cohort, capture.get("content_cohort"))
            for field, combo in self._capture_controls.items():
                self._set_combo_data(combo, capture.get(field))
            self._set_capture_group(capture.get("capture_group"))
            self._style_or_app.setText(str(capture.get("style_or_app") or ""))
            self._capture_session.setText(str(capture.get("capture_session") or ""))
            self._position_id.setText(str(capture.get("position_id") or ""))
            self._notes.setText(str(review.get("notes") or ""))
            self._set_presence(
                self._corner_presence,
                "Đã đánh dấu (4/4)" if all_corners_present else f"Chưa đủ ({corner_count}/4)",
                all_corners_present,
            )
            self._set_presence(self._fen_presence, "Đã có FEN" if has_fen else "Chưa có FEN", has_fen)
        finally:
            self._loading = False
        self._refresh_validation_labels()

    @staticmethod
    def _set_presence(label: QLabel, text: str, present: bool) -> None:
        label.setText(text)
        label.setStyleSheet(
            "font-weight: 600; color: #2e7d32;" if present else "font-weight: 600; color: #8a5a00;"
        )

    def values(self) -> dict[str, dict[str, Any]]:
        """Return only the fields this simplified panel owns.

        Legacy review/status/device fields intentionally stay out of this
        return value so changing a visible dropdown cannot erase them.
        """

        capture = {
            field: self._combo_value(combo)
            for field, combo in self._capture_controls.items()
        }
        capture["capture_group"] = self._capture_group_value()
        capture["content_cohort"] = self._content_cohort.currentData()
        capture["style_or_app"] = self._optional_text(self._style_or_app)
        capture["capture_session"] = self._optional_text(self._capture_session)
        capture["position_id"] = self._optional_text(self._position_id)
        return {
            "board": {"image_orientation": self._combo_value(self._orientation)},
            "capture": capture,
            "review": {"notes": self._notes.text()},
        }

    def capture_values(self) -> dict[str, Any]:
        return self.values()["capture"]

    def set_capture_values(self, capture: Mapping[str, Any]) -> None:
        self._loading = True
        try:
            for field, combo in self._capture_controls.items():
                self._set_combo_data(combo, capture.get(field))
            self._set_capture_group(capture.get("capture_group"))
            self._set_optional_combo_data(self._content_cohort, capture.get("content_cohort"))
            self._style_or_app.setText(str(capture.get("style_or_app") or ""))
            self._capture_session.setText(str(capture.get("capture_session") or ""))
            self._position_id.setText(str(capture.get("position_id") or ""))
        finally:
            self._loading = False

    def set_recent_values(self, _devices: list[str], capture_groups: list[str]) -> None:
        """Populate the editable capture-group combobox from session MRU values."""

        current = self._capture_group_value()
        self._capture_group.blockSignals(True)
        try:
            self._capture_group.clear()
            self._capture_group.addItem("Chưa gán", "")
            seen: set[str] = set()
            for value in capture_groups:
                text = value.strip()
                if text and text not in seen:
                    self._capture_group.addItem(text, text)
                    seen.add(text)
            if current:
                if current not in seen:
                    self._capture_group.addItem(current, current)
                self._capture_group.setCurrentText(current)
            else:
                self._capture_group.setCurrentIndex(0)
        finally:
            self._capture_group.blockSignals(False)

    @staticmethod
    def _optional_text(edit: QLineEdit) -> str | None:
        text = edit.text().strip()
        return text or None

    def set_validation(self, corner_errors: list[str], fen_errors: list[str]) -> None:
        self._corner_errors = list(corner_errors)
        self._fen_errors = list(fen_errors)
        self._refresh_validation_labels()

    def set_completeness(self, is_complete: bool, detail: str = "") -> None:
        if is_complete:
            self._completeness.setText("Đủ dữ liệu benchmark")
            self._completeness.setStyleSheet("font-weight: bold; color: #2e7d32; padding: 3px;")
        else:
            suffix = f": thiếu {detail}" if detail else ""
            self._completeness.setText("Chưa đủ dữ liệu benchmark" + suffix)
            self._completeness.setStyleSheet("font-weight: bold; color: #b36b00; padding: 3px;")

    def _sync_occlusion_severity(self, *_: object) -> None:
        if self._loading:
            return
        severity = self._capture_controls["occlusion_severity"]
        if self._combo_value(self._capture_controls["occlusion"]) == "none":
            self._set_combo_data(severity, "none")
        elif self._combo_value(severity) == "none":
            self._set_combo_data(severity, "unknown")

    def _refresh_validation_labels(self) -> None:
        if self._corner_errors:
            self._corner_validation.setText(
                "Góc bàn: "
                + "; ".join(_CORNER_ERROR_LABELS.get(error, error.replace("_", " ")) for error in self._corner_errors)
            )
            self._corner_validation.setStyleSheet("color: #c62828;")
        else:
            self._corner_validation.setText("Góc bàn: hợp lệ khi đã đánh dấu đủ 4 góc.")
            self._corner_validation.setStyleSheet("color: #2e7d32;")
        if self._fen_errors:
            self._fen_validation.setText("FEN: " + "; ".join(self._fen_errors))
            self._fen_validation.setStyleSheet("color: #c62828;")
        else:
            self._fen_validation.setText("FEN: không có lỗi cấu trúc.")
            self._fen_validation.setStyleSheet("color: #2e7d32;")

    def _emit_changed(self, *_: object) -> None:
        if not self._loading:
            self.metadataChanged.emit()
