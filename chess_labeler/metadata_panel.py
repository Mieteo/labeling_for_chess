"""Board-level metadata controls for the optional ``<stem>.meta.json`` sidecar.

The widget intentionally exchanges plain nested dictionaries with the main
window.  That keeps Qt presentation concerns separate from schema validation
and atomic persistence in :mod:`chess_labeler.metadata`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


CAPTURE_OPTIONS: dict[str, list[str]] = {
    "lighting": ["unknown", "very_dark", "dim", "even", "bright", "mixed"],
    "shadow": ["unknown", "none", "mild", "strong"],
    "glare": ["unknown", "none", "mild", "strong"],
    "perspective": ["unknown", "frontal", "mild", "strong", "extreme"],
    "board_material": ["unknown", "wood", "plastic", "paper", "stone", "other"],
    "board_fill": ["unknown", "tiny", "small", "medium", "large", "very_large"],
    "distance": ["unknown", "near", "medium", "far"],
    "blur": ["unknown", "none", "mild", "strong"],
    "occlusion": ["unknown", "none", "hand", "piece", "object", "multiple"],
    "occlusion_severity": ["unknown", "none", "mild", "strong"],
    "environment": ["unknown", "indoor", "outdoor", "mixed"],
}

ORIENTATION_OPTIONS = ["unknown", "red_at_bottom", "red_at_top", "red_at_left", "red_at_right"]
CORNER_STATUS_OPTIONS = [
    "unmarked",
    "partial",
    "auto_suggested",
    "human_marked",
    "human_verified",
    "not_applicable",
]
FEN_STATUS_OPTIONS = ["not_started", "human_marked", "human_verified", "not_applicable"]
REVIEW_STATUS_OPTIONS = ["unreviewed", "annotated", "self_checked", "gold_verified", "needs_review"]

_FIELD_LABELS = {
    "lighting": "Ánh sáng",
    "shadow": "Bóng",
    "glare": "Chói",
    "perspective": "Góc xiên",
    "board_material": "Chất liệu bàn",
    "board_fill": "Độ phủ bàn",
    "distance": "Khoảng cách",
    "blur": "Mờ",
    "occlusion": "Che khuất",
    "occlusion_severity": "Mức che khuất",
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
    """Fast dropdown editor for capture and review fields.

    ``metadataChanged`` is emitted only as a result of a user-facing control
    change.  ``set_values`` is therefore safe during image navigation.
    """

    metadataChanged = Signal()
    applyNextRequested = Signal()
    applyRangeRequested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._loading = False
        self._corner_errors: list[str] = []
        self._fen_errors: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)
        content = QWidget(scroll)
        scroll.setWidget(content)
        layout = QVBoxLayout(content)

        self._completeness = QLabel("Metadata chưa đủ", content)
        self._completeness.setStyleSheet("font-weight: bold; color: #b36b00; padding: 4px;")
        layout.addWidget(self._completeness)

        self._workflow_tabs = QTabWidget(content)
        self._workflow_tabs.setObjectName("metadataWorkflowTabs")
        layout.addWidget(self._workflow_tabs, 1)

        essential = QWidget(self._workflow_tabs)
        essential_layout = QVBoxLayout(essential)
        essential_layout.setContentsMargins(6, 6, 6, 6)
        essential_layout.setSpacing(6)
        essential_hint = QLabel(
            "Các mục cần kiểm tra trước khi đánh dấu ảnh đã hoàn tất.", essential
        )
        essential_hint.setWordWrap(True)
        essential_hint.setStyleSheet("color: #666;")
        essential_layout.addWidget(essential_hint)
        board_grid = QGridLayout()
        board_grid.setHorizontalSpacing(8)
        board_grid.setVerticalSpacing(4)
        essential_layout.addLayout(board_grid)
        self._orientation = self._combo(ORIENTATION_OPTIONS, essential)
        self._add_compact_field(board_grid, 0, 0, "Hướng ảnh:", self._orientation, essential)
        self._corner_status = self._combo(CORNER_STATUS_OPTIONS, essential)
        self._add_compact_field(board_grid, 0, 2, "Trạng thái góc:", self._corner_status, essential)
        self._fen_status = self._combo(FEN_STATUS_OPTIONS, essential)
        self._add_compact_field(board_grid, 1, 0, "Trạng thái FEN:", self._fen_status, essential)
        self._side_to_move = QComboBox(essential)
        self._side_to_move.addItem("Không biết", None)
        self._side_to_move.addItem("Đỏ", "red")
        self._side_to_move.addItem("Đen", "black")
        self._add_compact_field(board_grid, 1, 2, "Lượt đi:", self._side_to_move, essential)
        self._position_complete = QCheckBox("Đủ toàn bộ thế cờ", essential)
        self._fen_verified = QCheckBox("FEN đã đối chiếu ảnh", essential)
        self._corners_verified = QCheckBox("Góc đã đối chiếu", essential)
        verification_row = QHBoxLayout()
        verification_row.setContentsMargins(0, 0, 0, 0)
        verification_row.addWidget(self._position_complete)
        verification_row.addWidget(self._corners_verified)
        verification_row.addWidget(self._fen_verified)
        verification_row.addStretch(1)
        essential_layout.addLayout(verification_row)
        self._corner_validation = QLabel("", essential)
        self._corner_validation.setWordWrap(True)
        essential_layout.addWidget(self._corner_validation)
        self._fen_validation = QLabel("", essential)
        self._fen_validation.setWordWrap(True)
        essential_layout.addWidget(self._fen_validation)
        self._review_status = self._combo(REVIEW_STATUS_OPTIONS, essential)
        review_grid = QGridLayout()
        self._add_compact_field(review_grid, 0, 0, "Review:", self._review_status, essential)
        essential_layout.addLayout(review_grid)
        essential_layout.addStretch(1)

        capture = QWidget(self._workflow_tabs)
        capture_layout = QVBoxLayout(capture)
        capture_layout.setContentsMargins(6, 6, 6, 6)
        capture_layout.setSpacing(6)
        capture_hint = QLabel("Mô tả nhanh ảnh và điều kiện chụp chính.", capture)
        capture_hint.setStyleSheet("color: #666;")
        capture_layout.addWidget(capture_hint)
        capture_grid = QGridLayout()
        capture_grid.setHorizontalSpacing(8)
        capture_grid.setVerticalSpacing(4)
        capture_layout.addLayout(capture_grid)
        self._capture_controls: dict[str, QComboBox] = {}
        primary_capture_fields = (
            "lighting",
            "perspective",
            "board_fill",
            "blur",
            "occlusion",
            "environment",
        )
        advanced_capture_fields = tuple(field for field in CAPTURE_OPTIONS if field not in primary_capture_fields)
        for index, field in enumerate(primary_capture_fields):
            combo = self._combo(CAPTURE_OPTIONS[field], capture)
            self._capture_controls[field] = combo
            self._add_compact_field(
                capture_grid, index // 2, (index % 2) * 2, f"{_FIELD_LABELS[field]}:", combo, capture
            )

        self._device_model = self._editable_combo(["unknown"], capture)
        self._add_compact_field(capture_grid, 3, 0, "Thiết bị:", self._device_model, capture)
        self._capture_group = self._editable_combo([], capture)
        self._capture_group.setPlaceholderText("Không nhóm")
        self._add_compact_field(capture_grid, 3, 2, "Nhóm chụp:", self._capture_group, capture)
        template_row = QHBoxLayout()
        next_button = QPushButton("Áp dụng cho ảnh tiếp", capture)
        next_button.setToolTip("Áp dụng điều kiện chụp này cho ảnh tiếp theo")
        next_button.clicked.connect(self.applyNextRequested)
        template_row.addWidget(next_button)
        range_button = QPushButton("Áp dụng cho dải…", capture)
        range_button.clicked.connect(self.applyRangeRequested)
        template_row.addWidget(range_button)
        template_row.addStretch(1)
        capture_layout.addLayout(template_row)
        capture_layout.addStretch(1)

        details = QWidget(self._workflow_tabs)
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(6, 6, 6, 6)
        details_layout.setSpacing(6)
        details_hint = QLabel("Chỉ điền khi quan sát thấy hoặc khi review cần ghi chú thêm.", details)
        details_hint.setWordWrap(True)
        details_hint.setStyleSheet("color: #666;")
        details_layout.addWidget(details_hint)
        details_grid = QGridLayout()
        details_grid.setHorizontalSpacing(8)
        details_grid.setVerticalSpacing(4)
        details_layout.addLayout(details_grid)
        for index, field in enumerate(advanced_capture_fields):
            combo = self._combo(CAPTURE_OPTIONS[field], details)
            self._capture_controls[field] = combo
            self._add_compact_field(
                details_grid, index // 2, (index % 2) * 2, f"{_FIELD_LABELS[field]}:", combo, details
            )
        self._exclude_gold = QCheckBox("Loại khỏi gold set", details)
        details_layout.addWidget(self._exclude_gold)
        self._exclusion_reason = QLineEdit(details)
        self._exclusion_reason.setPlaceholderText("Lý do nếu loại")
        exclusion_grid = QGridLayout()
        self._add_compact_field(exclusion_grid, 0, 0, "Lý do:", self._exclusion_reason, details)
        details_layout.addLayout(exclusion_grid)
        self._notes = QTextEdit(details)
        self._notes.setAcceptRichText(False)
        self._notes.setPlaceholderText("Ghi chú review (tùy chọn)")
        self._notes.setFixedHeight(60)
        details_layout.addWidget(self._notes)
        self._workflow_tabs.addTab(essential, "Cần hoàn tất")
        self._workflow_tabs.addTab(capture, "Điều kiện chụp")
        self._workflow_tabs.addTab(details, "Bổ sung & review")

        for combo in [self._orientation, self._corner_status, self._fen_status, self._side_to_move, *self._capture_controls.values(),
                      self._device_model, self._capture_group, self._review_status]:
            combo.currentTextChanged.connect(self._emit_changed)
            if combo.isEditable() and combo.lineEdit() is not None:
                combo.lineEdit().editingFinished.connect(self._emit_changed)
        for checkbox in [self._position_complete, self._fen_verified, self._corners_verified, self._exclude_gold]:
            checkbox.toggled.connect(self._emit_changed)
        self._exclusion_reason.editingFinished.connect(self._emit_changed)
        self._notes.textChanged.connect(self._emit_changed)
        self._capture_controls["occlusion"].currentTextChanged.connect(self._sync_occlusion_severity)

        self.set_values({})

    @staticmethod
    def _combo(values: list[str], parent: QWidget) -> QComboBox:
        combo = QComboBox(parent)
        combo.addItems(values)
        return combo

    @staticmethod
    def _editable_combo(values: list[str], parent: QWidget) -> QComboBox:
        combo = QComboBox(parent)
        combo.setEditable(True)
        combo.addItems(values)
        return combo

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
    def _set_combo(combo: QComboBox, value: object, fallback: str = "unknown") -> None:
        text = value if isinstance(value, str) and value else fallback
        index = combo.findText(text)
        if index < 0 and combo.isEditable():
            combo.addItem(text)
            index = combo.findText(text)
        combo.setCurrentIndex(index if index >= 0 else 0)
        if combo.isEditable():
            combo.setEditText(text)

    def set_values(self, values: Mapping[str, Any] | None) -> None:
        """Load schema-shaped values without emitting dirty notifications."""
        board = _nested(values, "board")
        capture = _nested(values, "capture")
        review = _nested(values, "review")
        self._loading = True
        try:
            self._set_combo(self._orientation, board.get("image_orientation"))
            self._set_combo(self._corner_status, board.get("corners_status"), "unmarked")
            self._set_combo(self._fen_status, board.get("fen_status"), "not_started")
            self._position_complete.setChecked(bool(board.get("position_complete", False)))
            side = board.get("side_to_move")
            self._side_to_move.setCurrentIndex(max(0, self._side_to_move.findData(side)))
            for field, combo in self._capture_controls.items():
                self._set_combo(combo, capture.get(field))
            self._set_combo(self._device_model, capture.get("device_model"))
            self._set_combo(self._capture_group, capture.get("capture_group"), "")
            self._set_combo(self._review_status, review.get("status"), "unreviewed")
            self._fen_verified.setChecked(bool(review.get("fen_verified", False)))
            self._corners_verified.setChecked(bool(review.get("corners_verified", False)))
            self._exclude_gold.setChecked(bool(review.get("exclude_from_gold", False)))
            self._exclusion_reason.setText(str(review.get("exclusion_reason") or ""))
            self._notes.setPlainText(str(review.get("notes") or ""))
        finally:
            self._loading = False
        self._refresh_validation_labels()

    def values(self) -> dict[str, dict[str, Any]]:
        """Return metadata fields owned by this panel, in schema shape."""
        capture = {field: combo.currentText() or "unknown" for field, combo in self._capture_controls.items()}
        device = self._device_model.currentText().strip() or "unknown"
        group = self._capture_group.currentText().strip() or None
        capture.update({"device_model": device, "capture_group": group})
        return {
            "board": {
                "image_orientation": self._orientation.currentText(),
                "corners_status": self._corner_status.currentText(),
                "fen_status": self._fen_status.currentText(),
                "position_complete": self._position_complete.isChecked(),
                "side_to_move": self._side_to_move.currentData(),
            },
            "capture": capture,
            "review": {
                "status": self._review_status.currentText(),
                "fen_verified": self._fen_verified.isChecked(),
                "corners_verified": self._corners_verified.isChecked(),
                "exclude_from_gold": self._exclude_gold.isChecked(),
                "exclusion_reason": self._exclusion_reason.text().strip() or None,
                "notes": self._notes.toPlainText(),
            },
        }

    def capture_values(self) -> dict[str, Any]:
        return self.values()["capture"]

    def set_capture_values(self, capture: Mapping[str, Any]) -> None:
        current = self.values()
        current["capture"] = dict(capture)
        self.set_values(current)

    def set_recent_values(self, devices: list[str], capture_groups: list[str]) -> None:
        self._replace_recent_items(self._device_model, ["unknown", *devices], preserve="unknown")
        self._replace_recent_items(self._capture_group, capture_groups, preserve="")

    @staticmethod
    def _replace_recent_items(combo: QComboBox, values: list[str], preserve: str) -> None:
        current = combo.currentText()
        unique: list[str] = []
        for value in values:
            if isinstance(value, str) and value not in unique:
                unique.append(value)
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(unique)
        combo.setEditText(current or preserve)
        combo.blockSignals(False)

    def set_validation(self, corner_errors: list[str], fen_errors: list[str]) -> None:
        self._corner_errors = list(corner_errors)
        self._fen_errors = list(fen_errors)
        # Invalid geometry / position cannot be promoted to verified status.
        if self._corner_errors and self._corner_status.currentText() == "human_verified":
            self._corner_status.setCurrentText("human_marked")
        if self._fen_errors and self._fen_status.currentText() == "human_verified":
            self._fen_status.setCurrentText("human_marked")
        if self._fen_errors and self._fen_verified.isChecked():
            self._fen_verified.setChecked(False)
        self._refresh_validation_labels()

    def set_completeness(self, is_complete: bool, detail: str = "") -> None:
        if is_complete:
            self._completeness.setText("Metadata đủ điều kiện review" + (f": {detail}" if detail else ""))
            self._completeness.setStyleSheet("font-weight: bold; color: #2e7d32; padding: 4px;")
        else:
            self._completeness.setText("Metadata chưa đủ" + (f": {detail}" if detail else ""))
            self._completeness.setStyleSheet("font-weight: bold; color: #b36b00; padding: 4px;")

    def _sync_occlusion_severity(self, value: str) -> None:
        if self._loading:
            return
        severity = self._capture_controls["occlusion_severity"]
        if value == "none":
            severity.setCurrentText("none")
        elif severity.currentText() == "none":
            severity.setCurrentText("unknown")

    def _refresh_validation_labels(self) -> None:
        if self._corner_errors:
            self._corner_validation.setText(
                "; ".join(_CORNER_ERROR_LABELS.get(error, error.replace("_", " ")) for error in self._corner_errors)
            )
            self._corner_validation.setStyleSheet("color: #c62828;")
        else:
            self._corner_validation.setText("OK hoặc chưa nhập đủ 4 góc")
            self._corner_validation.setStyleSheet("color: #2e7d32;")
        if self._fen_errors:
            self._fen_validation.setText("; ".join(self._fen_errors))
            self._fen_validation.setStyleSheet("color: #c62828;")
        else:
            self._fen_validation.setText("Không có lỗi board")
            self._fen_validation.setStyleSheet("color: #2e7d32;")

    def _emit_changed(self, *_: object) -> None:
        if not self._loading:
            self.metadataChanged.emit()
