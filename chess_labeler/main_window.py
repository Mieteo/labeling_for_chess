"""Main application window: wires the canvas, side panels, class-assignment
shortcuts, circle-detect assist, and file I/O together. See
labeling_tool_requirements.md for the full behavioral spec this implements.
"""

from __future__ import annotations

import dataclasses
import copy
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QFile, QPointF, QRectF, QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from . import circle_detect, metadata, session, yolo_io
from .board_editor import STARTING_BOARD_FEN, XiangqiBoardEditor, describe_validation_issue
from .canvas import BoxItem, CanvasMode, ImageCanvas
from .constants import DEFAULT_RADIUS_TOLERANCE_PCT
from .key_shortcuts import HAND_CLASS, resolve_piece_class
from .metadata_panel import MetadataPanel
from .panels import BoxListPanel, FileListPanel

_ROLE_KEY_CODES = {
    Qt.Key.Key_P: "P",
    Qt.Key.Key_C: "C",
    Qt.Key.Key_R: "R",
    Qt.Key.Key_H: "H",
    Qt.Key.Key_E: "E",
    Qt.Key.Key_A: "A",
    Qt.Key.Key_K: "K",
}

# QSettings key for the last opened image directory -- app-level machine
# preference (see labeling_tool_requirements.md section 5 update), distinct
# from the per-folder ".labeling_session.json" (session.py), which is about
# labeling progress and travels with the folder across machines.
_LAST_DIR_SETTINGS_KEY = "last_image_dir"


@dataclasses.dataclass(frozen=True)
class _BoxSnapshot:
    rect: tuple[float, float, float, float]
    class_name: str | None
    confirmed: bool


@dataclasses.dataclass(frozen=True)
class _CornerSnapshot:
    corners: tuple[tuple[str, float, float] | None, ...]


def _read_image_bgr(path: Path) -> np.ndarray | None:
    """Unicode-path-safe read (cv2.imread mishandles non-ASCII paths on Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        self.setWindowTitle("Xiangqi Labeler")
        self.resize(1440, 920)

        self._app_settings = settings if settings is not None else QSettings("ChessLabeler", "XiangqiLabeler")
        self._shown_once = False

        self._image_dir: Path | None = None
        self._classes: list[str] = []
        self._images: list[Path] = []
        self._current_index: int = -1
        self._current_image_path: Path | None = None
        self._current_image_bgr: np.ndarray | None = None
        self._metadata: metadata.ImageMetadata | None = None
        self._metadata_load_error: metadata.MetadataError | None = None
        # Keep image annotations and board-level metadata independently dirty.
        # This prevents a metadata-only save from rewriting a legacy YOLO file.
        self._boxes_dirty = False
        self._metadata_dirty = False
        self._dirty = False

        self._undo_stack: list[list[_BoxSnapshot]] = []
        self._redo_stack: list[list[_BoxSnapshot]] = []
        self._pending_drag_snapshot: list[_BoxSnapshot] | None = None
        self._corner_undo_stack: list[_CornerSnapshot] = []
        self._corner_redo_stack: list[_CornerSnapshot] = []
        self._pending_capture_template: dict[str, object] | None = None
        self._recent_device_models: list[str] = []
        self._recent_capture_groups: list[str] = []

        self._reference_radius_px: float | None = None

        self._canvas = ImageCanvas(self)
        self.setCentralWidget(self._canvas)
        self._canvas.on_drag_begin = self._on_drag_begin
        self._canvas.on_drag_end = self._on_drag_end
        self._canvas.boxDrawn.connect(self._on_box_drawn)
        self._canvas.radiusMeasured.connect(self._on_radius_measured)
        self._canvas.deleteRequested.connect(self._on_delete_requested)
        self._canvas.confirmRequested.connect(self._on_confirm_requested)
        self._canvas.boxSelected.connect(self._on_box_selected)
        self._canvas.sceneModified.connect(self._mark_boxes_dirty)
        self._canvas.cornerRequested.connect(self._on_corner_requested)
        self._canvas.clearCornersRequested.connect(self._clear_corners_with_confirmation)

        self._build_dock_panels()
        self._build_toolbar_and_actions()
        self.setStatusBar(QStatusBar(self))

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self._update_window_title()
        self._update_undo_redo_actions()
        self._update_corner_undo_actions()
        self._resume_last_directory()

    def _resume_last_directory(self) -> None:
        last_dir = self._app_settings.value(_LAST_DIR_SETTINGS_KEY, "", type=str)
        if last_dir and Path(last_dir).is_dir():
            self._open_directory(Path(last_dir))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_dock_panels(self) -> None:
        self._file_list_panel = FileListPanel(self)
        file_dock = QDockWidget("Danh sách ảnh", self)
        file_dock.setWidget(self._file_list_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, file_dock)
        self._file_list_panel.imageActivated.connect(self._go_to_index)

        self._right_tabs = QTabWidget(self)
        self._right_tabs.setObjectName("annotationTabs")
        self._right_tabs.setMinimumWidth(520)

        annotation_tab = QWidget(self._right_tabs)
        annotation_tab.setObjectName("annotationTab")
        annotation_layout = QVBoxLayout(annotation_tab)
        annotation_layout.setContentsMargins(6, 6, 6, 6)
        annotation_layout.setSpacing(6)

        boxes_group = QGroupBox("Các box", annotation_tab)
        boxes_group.setObjectName("boxesGroup")
        boxes_layout = QVBoxLayout(boxes_group)
        boxes_layout.setContentsMargins(6, 6, 6, 6)
        self._box_list_panel = BoxListPanel(boxes_group)
        self._box_list_panel.setMinimumHeight(110)
        boxes_layout.addWidget(self._box_list_panel)
        annotation_layout.addWidget(boxes_group)
        self._box_list_panel.boxActivated.connect(self._canvas.select_box)

        assist_group = QGroupBox("Gán lớp & hỗ trợ phát hiện", annotation_tab)
        assist_group.setObjectName("classAssistGroup")
        assist_layout = QGridLayout(assist_group)
        assist_layout.setContentsMargins(6, 6, 6, 6)
        assist_layout.setHorizontalSpacing(6)
        assist_layout.setVerticalSpacing(4)

        self._class_combo = QComboBox(assist_group)
        self._class_combo.setMaximumWidth(190)
        self._class_combo.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._class_combo.activated.connect(self._on_class_combo_activated)
        assist_layout.addWidget(QLabel("Lớp:", assist_group), 0, 0)
        assist_layout.addWidget(self._class_combo, 0, 1)

        measure_btn = QPushButton("Đo bán kính (kéo chuột)", assist_group)
        measure_btn.setToolTip("Kéo chuột quanh một quân cờ mẫu để đo bán kính tham chiếu")
        measure_btn.clicked.connect(self._start_measure_radius)
        assist_layout.addWidget(measure_btn, 0, 2, 1, 3)

        self._radius_spin = QDoubleSpinBox(assist_group)
        self._radius_spin.setRange(1.0, 5000.0)
        self._radius_spin.setDecimals(1)
        self._radius_spin.setMaximumWidth(105)
        self._radius_spin.valueChanged.connect(self._on_radius_spin_changed)
        assist_layout.addWidget(QLabel("Bán kính:", assist_group), 1, 0)
        assist_layout.addWidget(self._radius_spin, 1, 1)

        self._tolerance_spin = QDoubleSpinBox(assist_group)
        self._tolerance_spin.setRange(1.0, 100.0)
        self._tolerance_spin.setDecimals(1)
        self._tolerance_spin.setSuffix(" %")
        self._tolerance_spin.setValue(DEFAULT_RADIUS_TOLERANCE_PCT)
        self._tolerance_spin.setMaximumWidth(105)
        assist_layout.addWidget(QLabel("Dung sai:", assist_group), 1, 2)
        assist_layout.addWidget(self._tolerance_spin, 1, 3)

        rerun_btn = QPushButton("Chạy lại", assist_group)
        rerun_btn.setToolTip("Chạy lại phát hiện theo bán kính tham chiếu")
        rerun_btn.clicked.connect(lambda: self._run_detection("radius_guided"))
        assist_layout.addWidget(rerun_btn, 2, 0, 1, 2)

        autoscan_btn = QPushButton("Tự quét", assist_group)
        autoscan_btn.setToolTip("Tự động quét toàn bộ ảnh")
        autoscan_btn.clicked.connect(lambda: self._run_detection("auto_scan"))
        assist_layout.addWidget(autoscan_btn, 2, 2)

        clear_btn = QPushButton("Xóa gợi ý", assist_group)
        clear_btn.setToolTip("Xóa các gợi ý chưa xác nhận")
        clear_btn.clicked.connect(self._clear_unconfirmed_suggestions)
        assist_layout.addWidget(clear_btn, 2, 3)
        assist_layout.setColumnStretch(4, 1)
        annotation_layout.addWidget(assist_group)

        self._board_editor = XiangqiBoardEditor(self)
        self._board_editor.boardChanged.connect(self._on_board_changed)

        self._metadata_panel = MetadataPanel(self)
        self._metadata_panel.metadataChanged.connect(self._on_metadata_panel_changed)
        self._metadata_panel.applyNextRequested.connect(self._remember_capture_template_for_next)
        self._metadata_panel.applyRangeRequested.connect(self._apply_capture_template_to_range)
        metadata_group = QGroupBox("Bàn cờ & điều kiện chụp", annotation_tab)
        metadata_group.setObjectName("captureConditionsGroup")
        metadata_layout = QVBoxLayout(metadata_group)
        metadata_layout.setContentsMargins(6, 6, 6, 6)
        metadata_layout.addWidget(self._metadata_panel)
        annotation_layout.addWidget(metadata_group, 1)

        fen_tab = QWidget(self._right_tabs)
        fen_tab.setObjectName("fenTab")
        fen_layout = QVBoxLayout(fen_tab)
        fen_layout.setContentsMargins(0, 0, 0, 0)
        fen_layout.addWidget(self._board_editor)

        self._right_tabs.addTab(annotation_tab, "Gán nhãn")
        self._right_tabs.addTab(fen_tab, "Bàn cờ & FEN")
        self._metadata_panel.openFenRequested.connect(lambda: self._right_tabs.setCurrentWidget(fen_tab))
        self._metadata_panel.confirmFenRequested.connect(self._confirm_current_board_fen)
        self._metadata_panel.clearFenRequested.connect(self._clear_current_board_fen)
        self._annotation_dock = QDockWidget("Thông tin gán nhãn", self)
        self._annotation_dock.setObjectName("annotationDock")
        self._annotation_dock.setMinimumWidth(520)
        self._annotation_dock.setWidget(self._right_tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._annotation_dock)
        self.resizeDocks([self._annotation_dock], [540], Qt.Orientation.Horizontal)

    def _build_toolbar_and_actions(self) -> None:
        toolbar = QToolBar("Main", self)
        self.addToolBar(toolbar)

        open_action = QAction("Mở thư mục...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_directory)
        toolbar.addAction(open_action)

        save_action = QAction("Lưu", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._save_current_image)
        toolbar.addAction(save_action)

        save_empty_action = QAction("Lưu: không có đối tượng nào", self)
        save_empty_action.triggered.connect(self._save_current_image_as_empty)
        toolbar.addAction(save_empty_action)

        self._autosave_action = QAction("Auto-save khi chuyển ảnh", self)
        self._autosave_action.setCheckable(True)
        self._autosave_action.setChecked(True)
        toolbar.addAction(self._autosave_action)

        delete_image_action = QAction("Xóa ảnh (Shift+Delete)", self)
        delete_image_action.setToolTip(
            "Chuyển ảnh hiện tại cùng file .txt và .meta.json vào Thùng rác"
        )
        delete_image_action.triggered.connect(self._delete_current_image)
        toolbar.addAction(delete_image_action)

        toolbar.addSeparator()

        prev_action = QAction("Ảnh trước (←)", self)
        prev_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        prev_action.triggered.connect(self.prev_image)
        toolbar.addAction(prev_action)

        next_action = QAction("Ảnh sau (→)", self)
        next_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        next_action.triggered.connect(self.next_image)
        toolbar.addAction(next_action)

        toolbar.addSeparator()

        draw_action = QAction("Vẽ box (W)", self)
        draw_action.setShortcut(QKeySequence("W"))
        draw_action.triggered.connect(self._enter_draw_mode)
        toolbar.addAction(draw_action)

        corner_action = QAction("Đánh dấu góc (1–4)", self)
        corner_action.setToolTip(
            "Đưa chuột lên ảnh rồi bấm 1=trên-trái, 2=trên-phải, 3=dưới-phải, 4=dưới-trái"
        )
        corner_action.triggered.connect(self._focus_canvas_for_corners)
        toolbar.addAction(corner_action)

        self._duplicate_action = QAction("Nhân bản box (Ctrl+D)", self)
        self._duplicate_action.setShortcut(QKeySequence("Ctrl+D"))
        self._duplicate_action.setEnabled(False)
        self._duplicate_action.triggered.connect(self._duplicate_selected_box)
        toolbar.addAction(self._duplicate_action)

        toolbar.addSeparator()

        self._undo_action = QAction("Undo", self)
        self._undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        self._undo_action.triggered.connect(self.undo)
        toolbar.addAction(self._undo_action)

        self._redo_action = QAction("Redo", self)
        self._redo_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        self._redo_action.triggered.connect(self.redo)
        toolbar.addAction(self._redo_action)

        self._corner_undo_action = QAction("Undo góc", self)
        self._corner_undo_action.setShortcut(QKeySequence("Ctrl+Alt+Z"))
        self._corner_undo_action.triggered.connect(self.undo_corners)
        toolbar.addAction(self._corner_undo_action)

        self._corner_redo_action = QAction("Redo góc", self)
        self._corner_redo_action.setShortcut(QKeySequence("Ctrl+Alt+Shift+Z"))
        self._corner_redo_action.triggered.connect(self.redo_corners)
        toolbar.addAction(self._corner_redo_action)

        toolbar.addSeparator()

        zoom_in = QAction("Zoom +", self)
        zoom_in.setShortcut(QKeySequence("Ctrl+="))
        zoom_in.triggered.connect(lambda: self._canvas.zoom_by(1.25))
        toolbar.addAction(zoom_in)

        zoom_out = QAction("Zoom -", self)
        zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out.triggered.connect(lambda: self._canvas.zoom_by(0.8))
        toolbar.addAction(zoom_out)

        fit_window = QAction("Fit window", self)
        fit_window.setShortcut(QKeySequence("Ctrl+0"))
        fit_window.triggered.connect(self._canvas.fit_to_window)
        toolbar.addAction(fit_window)

        fit_width = QAction("Fit width", self)
        fit_width.setShortcut(QKeySequence("Ctrl+9"))
        fit_width.triggered.connect(self._canvas.fit_to_width)
        toolbar.addAction(fit_width)

    # ------------------------------------------------------------------
    # Directory / image loading
    # ------------------------------------------------------------------
    def open_directory(self) -> None:
        if not self._confirm_leave_current_image():
            return
        directory = QFileDialog.getExistingDirectory(self, "Chọn thư mục ảnh")
        if not directory:
            return
        self._open_directory(Path(directory))

    def _open_directory(self, directory: Path) -> None:
        self._image_dir = directory
        self._app_settings.setValue(_LAST_DIR_SETTINGS_KEY, str(directory))
        self._classes = yolo_io.load_or_create_classes(directory)
        self._images = yolo_io.list_images(directory)

        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItems(self._classes)
        self._class_combo.setCurrentIndex(-1)
        self._class_combo.blockSignals(False)

        self._file_list_panel.set_images(self._images)

        state = session.load_session(directory)
        if state.last_radius_px:
            self._reference_radius_px = state.last_radius_px
            self._radius_spin.blockSignals(True)
            self._radius_spin.setValue(state.last_radius_px)
            self._radius_spin.blockSignals(False)
        if state.last_tolerance_pct:
            self._tolerance_spin.setValue(state.last_tolerance_pct)
        self._recent_device_models = list(state.recent_device_models)
        self._recent_capture_groups = list(state.recent_capture_groups)
        self._metadata_panel.set_recent_values(self._recent_device_models, self._recent_capture_groups)

        if not self._images:
            QMessageBox.information(
                self, "Thư mục trống", "Không tìm thấy ảnh .jpg/.jpeg/.png nào trong thư mục này."
            )
            self._current_index = -1
            self._current_image_path = None
            return

        resume_path = session.find_resume_image(directory, self._images)
        idx = self._images.index(resume_path) if resume_path in self._images else 0
        self._load_image_at(idx)

    def _load_image_at(self, index: int) -> None:
        if not (0 <= index < len(self._images)):
            return
        path = self._images[index]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Lỗi", f"Không thể mở ảnh: {path}")
            return

        self._current_index = index
        self._current_image_path = path
        self._current_image_bgr = None
        self._canvas.load_image(pixmap)
        sidecar_found = self._load_metadata_for_current_image(path, pixmap.width(), pixmap.height())

        for box in yolo_io.load_boxes(path):
            left, top, w, h = box.to_pixel_rect(pixmap.width(), pixmap.height())
            name = yolo_io.class_name(self._classes, box.class_id)
            self._canvas.add_box_item(
                QRectF(left, top, w, h), class_name=name, confirmed=True, emit_modified=False
            )

        self._undo_stack.clear()
        self._redo_stack.clear()
        self._pending_drag_snapshot = None
        self._corner_undo_stack.clear()
        self._corner_redo_stack.clear()
        self._boxes_dirty = False
        self._metadata_dirty = False
        self._dirty = False

        # "Apply conditions to next" only fills a genuinely new sidecar; it
        # never silently changes tags a user already stored for this image.
        if self._pending_capture_template is not None and not sidecar_found and self._metadata is not None:
            for field, value in self._pending_capture_template.items():
                setattr(self._metadata.capture, field, copy.deepcopy(value))
            self._pending_capture_template = None
            self._metadata_panel.set_values(self._metadata.to_dict())
            self._refresh_metadata_validation()
            self._metadata_dirty = True
            self._dirty = True

        self._file_list_panel.set_current_index(index)
        self._refresh_box_list()
        self._canvas.fit_to_window()
        self._canvas.setFocus()
        self._update_window_title()
        self._update_undo_redo_actions()
        self._update_corner_undo_actions()
        self._save_session_state()

    def _ensure_image_bgr(self) -> np.ndarray | None:
        if self._current_image_bgr is None and self._current_image_path is not None:
            self._current_image_bgr = _read_image_bgr(self._current_image_path)
        return self._current_image_bgr

    # ------------------------------------------------------------------
    # Optional board-level metadata sidecar
    # ------------------------------------------------------------------
    def _load_metadata_for_current_image(self, path: Path, width: int, height: int) -> bool:
        """Load a valid sidecar or construct a clean, unsaved draft.

        A malformed existing file is never silently treated as a normal
        missing sidecar: the UI stays usable with a fresh in-memory draft but
        remembers the error and requires an explicit overwrite confirmation
        at save time.
        """
        result = metadata.try_load_metadata(path, expected_image_size=(width, height))
        self._metadata_load_error = result.error
        self._metadata = result.metadata or metadata.new_metadata(path, width, height)

        self._canvas.set_corner_points(self._corner_points_for_canvas())
        board_fen = self._metadata.board.board_fen or STARTING_BOARD_FEN
        try:
            self._board_editor.set_board_fen(board_fen)
        except ValueError:
            # This should only be reachable if a future schema becomes less
            # strict. Preserve the sidecar warning rather than crashing the
            # labeling workflow.
            self._board_editor.set_starting_position()
        self._metadata_panel.set_values(self._metadata.to_dict())
        self._refresh_metadata_validation()
        if result.error is not None:
            self.statusBar().showMessage(
                f"Metadata lỗi ở {metadata.metadata_path_for_image(path).name}: {result.error}", 7000
            )
        return result.found

    def _corner_points_for_canvas(self) -> dict[str, tuple[float, float]]:
        if self._metadata is None:
            return {}
        return {
            name: (point.x, point.y)
            for name, point in self._metadata.board.corners_px.items()
            if point is not None
        }

    def _corner_validation(self) -> metadata.CornerValidation:
        if self._metadata is None:
            return metadata.CornerValidation(errors=("metadata_not_loaded",))
        return metadata.validate_corners(
            self._metadata.board.corners_px,
            self._metadata.image.width_px,
            self._metadata.image.height_px,
        )

    def _refresh_metadata_validation(self) -> None:
        if self._metadata is None:
            return
        corner_result = self._corner_validation()
        board_issues = self._board_editor.validation_issues if self._metadata.board.board_fen else []
        self._metadata_panel.set_validation(
            list(corner_result.errors), [describe_validation_issue(issue) for issue in board_issues]
        )

        # The annotator sees only the meaningful binary state: four corners
        # exist or they do not; a board FEN exists or it does not.  Geometry
        # and board-validation warnings remain visible separately.
        corners_ready = corner_result.is_valid
        fen_ready = self._metadata.board.board_fen is not None and not board_issues
        ready = corners_ready and fen_ready
        missing: list[str] = []
        if not corners_ready:
            missing.append("góc")
        if not fen_ready:
            missing.append("FEN")
        self._metadata_panel.set_completeness(ready, " & ".join(missing))

    def _sync_legacy_metadata_state(self) -> None:
        """Derive v1 status/review fields from the simplified UI state.

        Existing sidecars remain schema-v1 compatible, but annotators no
        longer need to understand or choose these implementation details.
        """

        if self._metadata is None:
            return
        record = self._metadata
        corner_result = self._corner_validation()
        corner_count = sum(point is not None for point in record.board.corners_px.values())
        if corner_count == 0:
            record.board.corners_status = "unmarked"
            record.review.corners_verified = False
        elif corner_result.is_valid:
            record.board.corners_status = "human_verified"
            record.review.corners_verified = True
        else:
            record.board.corners_status = "partial" if corner_count < len(metadata.CORNER_NAMES) else "human_marked"
            record.review.corners_verified = False

        board_issues = self._board_editor.validation_issues if record.board.board_fen else []
        if record.board.board_fen is None:
            record.board.fen_status = "not_started"
            record.board.position_complete = False
            record.review.fen_verified = False
        elif board_issues:
            record.board.fen_status = "human_marked"
            record.board.position_complete = False
            record.review.fen_verified = False
        else:
            record.board.fen_status = "human_verified"
            record.board.position_complete = True
            record.review.fen_verified = True

        # Preserve old review decisions when they are still valid.  A gold
        # record changed into an invalid/incomplete one must not remain gold.
        if record.review.status == "gold_verified" and not (
            record.review.corners_verified and record.review.fen_verified
        ):
            record.review.status = "needs_review"

    def _apply_panel_values_to_metadata(self) -> None:
        if self._metadata is None:
            return
        values = self._metadata_panel.values()
        board_values = values["board"]
        capture_values = values["capture"]
        review_values = values["review"]

        self._metadata.board.image_orientation = str(board_values["image_orientation"])
        for field, value in capture_values.items():
            setattr(self._metadata.capture, field, value)
        self._metadata.review.notes = str(review_values["notes"])
        self._sync_legacy_metadata_state()

    def _on_metadata_panel_changed(self) -> None:
        if self._metadata is None:
            return
        self._apply_panel_values_to_metadata()
        # Reflect any guardrail correction (e.g. no verified FEN while the
        # board has a structural warning) back into the controls.
        self._metadata_panel.set_values(self._metadata.to_dict())
        self._refresh_metadata_validation()
        self._mark_metadata_dirty()

    def _on_board_changed(self, board_fen: str, issues: list[str]) -> None:
        if self._metadata is None:
            return
        # Retain visible conditions, then make the direct board edit
        # authoritative. A static photo does not establish whose turn it is,
        # so old full-FEN fields must not be left stale after an edit.
        self._apply_panel_values_to_metadata()
        self._metadata.board.board_fen = board_fen
        self._metadata.board.side_to_move = None
        self._metadata.board.full_fen = None
        self._sync_legacy_metadata_state()
        self._metadata_panel.set_values(self._metadata.to_dict())
        self._refresh_metadata_validation()
        self._mark_metadata_dirty()

    def _confirm_current_board_fen(self) -> None:
        """Persist the editor's current board, including an unchanged scaffold."""

        if self._metadata is None:
            return
        self._apply_panel_values_to_metadata()
        self._metadata.board.board_fen = self._board_editor.board_fen
        self._metadata.board.side_to_move = None
        self._metadata.board.full_fen = None
        self._sync_legacy_metadata_state()
        self._metadata_panel.set_values(self._metadata.to_dict())
        self._refresh_metadata_validation()
        self._mark_metadata_dirty()
        self.statusBar().showMessage("Đã xác nhận FEN đang hiển thị.", 2500)

    def _clear_current_board_fen(self) -> None:
        if self._metadata is None or self._metadata.board.board_fen is None:
            return
        response = QMessageBox.question(
            self,
            "Bỏ FEN",
            "Bỏ FEN đã xác nhận cho ảnh hiện tại? Bàn cờ sẽ trở về thế cờ đầu để làm khung nhập.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        self._metadata.board.board_fen = None
        self._metadata.board.side_to_move = None
        self._metadata.board.full_fen = None
        self._board_editor.set_starting_position()
        self._sync_legacy_metadata_state()
        self._metadata_panel.set_values(self._metadata.to_dict())
        self._refresh_metadata_validation()
        self._mark_metadata_dirty()
        self.statusBar().showMessage("Đã bỏ FEN của ảnh hiện tại.", 2500)

    def _snapshot_corners(self) -> _CornerSnapshot:
        if self._metadata is None:
            return _CornerSnapshot(())
        corners: list[tuple[str, float, float] | None] = []
        for name in metadata.CORNER_NAMES:
            point = self._metadata.board.corners_px[name]
            corners.append((name, point.x, point.y) if point is not None else None)
        return _CornerSnapshot(tuple(corners))

    def _apply_corner_snapshot(self, snapshot: _CornerSnapshot) -> None:
        if self._metadata is None:
            return
        corners: dict[str, metadata.Point | None] = {}
        for name, value in zip(metadata.CORNER_NAMES, snapshot.corners):
            corners[name] = None if value is None else metadata.Point(value[1], value[2])
        self._metadata.board.corners_px = corners
        self._sync_legacy_metadata_state()
        self._canvas.set_corner_points(self._corner_points_for_canvas())
        self._metadata_panel.set_values(self._metadata.to_dict())
        self._refresh_metadata_validation()
        self._mark_metadata_dirty()

    def _push_corner_undo(self) -> None:
        self._corner_undo_stack.append(self._snapshot_corners())
        self._corner_redo_stack.clear()
        self._update_corner_undo_actions()

    def _on_corner_requested(self, name: str, point: QPointF) -> None:
        if self._metadata is None or name not in metadata.CORNER_NAMES:
            return
        self._push_corner_undo()
        self._metadata.board.corners_px[name] = metadata.Point(point.x(), point.y())
        self._sync_legacy_metadata_state()
        self._canvas.set_corner_points(self._corner_points_for_canvas())
        self._metadata_panel.set_values(self._metadata.to_dict())
        self._refresh_metadata_validation()
        self._mark_metadata_dirty()
        self.statusBar().showMessage(f"Đã đặt {name}: ({point.x():.1f}, {point.y():.1f})", 2500)

    def _clear_corners_with_confirmation(self) -> None:
        if self._metadata is None or not any(self._metadata.board.corners_px.values()):
            return
        response = QMessageBox.question(
            self,
            "Xóa 4 góc",
            "Xóa toàn bộ bốn góc của ảnh hiện tại?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        self._push_corner_undo()
        self._metadata.board.corners_px = {name: None for name in metadata.CORNER_NAMES}
        self._sync_legacy_metadata_state()
        self._canvas.set_corner_points({})
        self._metadata_panel.set_values(self._metadata.to_dict())
        self._refresh_metadata_validation()
        self._mark_metadata_dirty()

    def undo_corners(self) -> None:
        if not self._corner_undo_stack or self._metadata is None:
            return
        current = self._snapshot_corners()
        previous = self._corner_undo_stack.pop()
        self._corner_redo_stack.append(current)
        self._apply_corner_snapshot(previous)
        self._update_corner_undo_actions()

    def redo_corners(self) -> None:
        if not self._corner_redo_stack or self._metadata is None:
            return
        current = self._snapshot_corners()
        following = self._corner_redo_stack.pop()
        self._corner_undo_stack.append(current)
        self._apply_corner_snapshot(following)
        self._update_corner_undo_actions()

    def _remember_capture_template_for_next(self) -> None:
        self._pending_capture_template = copy.deepcopy(self._metadata_panel.capture_values())
        self.statusBar().showMessage("Đã ghi nhớ điều kiện chụp cho ảnh mới kế tiếp.", 3000)

    def _apply_capture_template_to_range(self) -> None:
        if not self._images:
            return
        raw, accepted = QInputDialog.getText(
            self,
            "Áp dụng điều kiện chụp",
            f"Nhập dải số theo danh sách ảnh 1–{len(self._images)} (ví dụ 5-12):",
        )
        if not accepted:
            return
        try:
            first_text, last_text = raw.split("-", 1)
            first, last = int(first_text.strip()), int(last_text.strip())
        except ValueError:
            QMessageBox.warning(self, "Dải không hợp lệ", "Dùng định dạng số-số, ví dụ 5-12.")
            return
        if first < 1 or last < first or last > len(self._images):
            QMessageBox.warning(self, "Dải không hợp lệ", "Dải phải nằm trong danh sách ảnh hiện tại.")
            return
        targets = self._images[first - 1 : last]
        confirm = QMessageBox.question(
            self,
            "Xác nhận batch apply",
            f"Áp dụng điều kiện chụp hiện tại cho {len(targets)} ảnh?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        overwrite = QMessageBox.question(
            self,
            "Ghi đè dữ liệu đã có?",
            "Có ghi đè các điều kiện khác 'unknown' / nhóm / thiết bị đã có không?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

        template = copy.deepcopy(self._metadata_panel.capture_values())
        changed_count = 0
        skipped_count = 0
        current_reloaded: metadata.ImageMetadata | None = None
        for image_path in targets:
            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                skipped_count += 1
                continue
            result = metadata.try_load_metadata(
                image_path, expected_image_size=(pixmap.width(), pixmap.height())
            )
            if result.error is not None:
                skipped_count += 1
                continue
            record = result.metadata or metadata.new_metadata(image_path, pixmap.width(), pixmap.height())
            changed = False
            for field, value in template.items():
                existing = getattr(record.capture, field)
                if overwrite or existing in {"unknown", "", None}:
                    if existing != value:
                        setattr(record.capture, field, copy.deepcopy(value))
                        changed = True
            if not changed:
                continue
            try:
                metadata.save_metadata_atomic(
                    image_path, record, expected_image_size=(pixmap.width(), pixmap.height())
                )
            except metadata.MetadataError:
                skipped_count += 1
                continue
            changed_count += 1
            if image_path == self._current_image_path:
                current_reloaded = record

        if current_reloaded is not None:
            self._metadata = current_reloaded
            self._metadata_load_error = None
            self._metadata_dirty = False
            self._metadata_panel.set_values(self._metadata.to_dict())
            self._refresh_metadata_validation()
            self._sync_dirty_state()
        if changed_count:
            self._file_list_panel.refresh_all_content_cohorts()
        self.statusBar().showMessage(
            f"Đã áp dụng điều kiện cho {changed_count} ảnh"
            + (f"; bỏ qua {skipped_count} ảnh lỗi." if skipped_count else "."),
            5000,
        )

    # ------------------------------------------------------------------
    # Navigation with unsaved-change guard
    # ------------------------------------------------------------------
    def _confirm_leave_current_image(self) -> bool:
        if not self._dirty:
            return True
        if self._autosave_action.isChecked():
            self._save_current_image()
            return True
        resp = QMessageBox.question(
            self,
            "Thay đổi chưa lưu",
            "Ảnh hiện tại có thay đổi chưa lưu. Lưu trước khi chuyển ảnh?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if resp == QMessageBox.StandardButton.Save:
            self._save_current_image()
            return True
        return resp == QMessageBox.StandardButton.Discard

    def _go_to_index(self, index: int) -> None:
        if index == self._current_index or not (0 <= index < len(self._images)):
            return
        if not self._confirm_leave_current_image():
            return
        self._load_image_at(index)

    def next_image(self) -> None:
        self._go_to_index(self._current_index + 1)

    def prev_image(self) -> None:
        self._go_to_index(self._current_index - 1)

    def _delete_current_image(self) -> None:
        """Move the current image and its two label sidecars to the Recycle Bin.

        This deliberately bypasses the normal leave-image guard: when the
        user asks to delete an image, unsaved canvas/metadata changes must not
        be auto-saved immediately before the destructive action.
        """

        if self._current_image_path is None or self._current_index < 0:
            return
        image_path = self._current_image_path
        label_path = yolo_io.label_path_for_image(image_path)
        sidecar_path = metadata.metadata_path_for_image(image_path)
        companions = [path for path in (label_path, sidecar_path) if path.exists()]

        same_stem = [
            path
            for path in self._images
            if path != image_path and path.stem.casefold() == image_path.stem.casefold()
        ]
        if same_stem and companions:
            QMessageBox.warning(
                self,
                "Không thể xóa an toàn",
                "Có ảnh khác cùng tên gốc nên đang dùng chung file nhãn/metadata:\n"
                + "\n".join(f"• {path.name}" for path in same_stem)
                + "\n\nHãy đổi tên hoặc xử lý ảnh trùng tên trước để không làm mất dữ liệu của ảnh còn lại.",
            )
            return

        targets = [image_path, *companions]
        target_lines = "\n".join(f"• {path.name}" for path in targets)
        response = QMessageBox.question(
            self,
            "Xóa ảnh hiện tại",
            f"Chuyển các file sau vào Thùng rác?\n\n{target_lines}"
            "\n\nCác thay đổi chưa lưu của ảnh này sẽ bị bỏ.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return

        moved: list[Path] = []
        failures: list[str] = []
        for path in targets:
            try:
                outcome = QFile.moveToTrash(str(path))
                success = outcome[0] if isinstance(outcome, tuple) else bool(outcome)
            except Exception as exc:  # pragma: no cover - platform/Qt boundary
                success = False
                failures.append(f"{path.name}: {exc}")
            if success:
                moved.append(path)
            elif not failures or not failures[-1].startswith(f"{path.name}:"):
                failures.append(path.name)

        image_moved = image_path in moved
        if image_moved:
            self._remove_deleted_image_from_ui(image_path)

        if failures:
            moved_text = ", ".join(path.name for path in moved) or "không có file nào"
            QMessageBox.warning(
                self,
                "Xóa chưa hoàn tất",
                "Đã chuyển vào Thùng rác: "
                + moved_text
                + "\nKhông thể chuyển: "
                + ", ".join(failures)
                + "\n\nCác file đã chuyển có thể khôi phục từ Thùng rác.",
            )
            return

        self.statusBar().showMessage(f"Đã chuyển {image_path.name} và dữ liệu liên quan vào Thùng rác.", 3500)

    def _remove_deleted_image_from_ui(self, image_path: Path) -> None:
        """Refresh navigation after the primary image has left its folder."""

        try:
            deleted_index = self._images.index(image_path)
        except ValueError:
            return
        self._images.pop(deleted_index)
        self._file_list_panel.set_images(self._images)
        if self._images:
            # Same index means the next image; after the final image, use the
            # now-last entry (the prior image).
            self._load_image_at(min(deleted_index, len(self._images) - 1))
            return

        self._current_index = -1
        self._current_image_path = None
        self._current_image_bgr = None
        self._metadata = None
        self._metadata_load_error = None
        self._boxes_dirty = False
        self._metadata_dirty = False
        self._dirty = False
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._corner_undo_stack.clear()
        self._corner_redo_stack.clear()
        self._canvas.load_image(QPixmap())
        self._box_list_panel.set_boxes([])
        self._board_editor.set_starting_position()
        self._metadata_panel.set_values({})
        self._metadata_panel.set_validation([], [])
        self._metadata_panel.set_completeness(False, "chưa có ảnh")
        self._update_window_title()
        self._update_undo_redo_actions()
        self._update_corner_undo_actions()
        self._save_session_state()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _save_metadata_if_dirty(self) -> bool:
        if not self._metadata_dirty:
            return True
        if self._metadata is None or self._current_image_path is None:
            return True
        if self._metadata_load_error is not None:
            response = QMessageBox.warning(
                self,
                "Metadata sidecar lỗi",
                "File metadata cũ có lỗi và chưa bị thay đổi. Bạn có muốn ghi đè nó bằng metadata đang mở không?\n\n"
                f"Chi tiết: {self._metadata_load_error}",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response != QMessageBox.StandardButton.Save:
                return False
        try:
            metadata.save_metadata_atomic(
                self._current_image_path,
                self._metadata,
                expected_image_size=self._canvas.image_size(),
            )
        except metadata.MetadataError as exc:
            QMessageBox.warning(self, "Không thể lưu metadata", str(exc))
            return False
        self._metadata_load_error = None
        self._metadata_dirty = False
        self._board_editor.mark_clean()
        self._recent_device_models = session.add_recent_value(
            self._recent_device_models, self._metadata.capture.device_model
        )
        self._recent_capture_groups = session.add_recent_value(
            self._recent_capture_groups, self._metadata.capture.capture_group
        )
        self._metadata_panel.set_recent_values(self._recent_device_models, self._recent_capture_groups)
        if self._current_index >= 0:
            self._file_list_panel.refresh_content_cohort_at(self._current_index)
        return True

    def _save_current_image(self) -> bool:
        if self._current_image_path is None:
            return False
        if not self._boxes_dirty:
            if not self._save_metadata_if_dirty():
                return False
            self._sync_dirty_state()
            self._save_session_state()
            self.statusBar().showMessage(f"Đã lưu {self._current_image_path.name}.", 2500)
            return True
        items = self._canvas.box_items()
        unclassified = [b for b in items if b.confirmed and not b.class_name]
        if unclassified:
            resp = QMessageBox.warning(
                self,
                "Còn box chưa gán lớp",
                f"{len(unclassified)} box đã xác nhận nhưng CHƯA gán lớp -- các box này sẽ "
                "KHÔNG được ghi vào file .txt. Vẫn lưu?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
            )
            if resp != QMessageBox.StandardButton.Save:
                return False

        w, h = self._canvas.image_size()
        boxes_to_save = []
        for item in items:
            if not item.confirmed or not item.class_name or item.class_name not in self._classes:
                continue
            class_id = self._classes.index(item.class_name)
            r = item.rect()
            boxes_to_save.append(yolo_io.Box.from_pixel_rect(class_id, r.x(), r.y(), r.width(), r.height(), w, h))

        yolo_io.save_boxes(self._current_image_path, boxes_to_save)
        self._boxes_dirty = False
        if not self._save_metadata_if_dirty():
            # The YOLO save has already succeeded. Keep only the metadata
            # dirty flag so the next explicit save can safely retry it.
            self._sync_dirty_state()
            self._file_list_panel.refresh_label_at(self._current_index)
            self._save_session_state()
            return False
        self._sync_dirty_state()
        self._file_list_panel.refresh_label_at(self._current_index)
        self._save_session_state()
        self.statusBar().showMessage(f"Đã lưu {self._current_image_path.name} ({len(boxes_to_save)} box).", 2500)
        return True

    def _save_current_image_as_empty(self) -> None:
        if self._current_image_path is None:
            return
        boxes = self._canvas.box_items()
        if boxes:
            resp = QMessageBox.question(
                self,
                "Xác nhận",
                f"Ảnh này đang có {len(boxes)} box trên canvas. Lưu như KHÔNG có đối tượng nào sẽ "
                "bỏ qua toàn bộ box hiện tại khi ghi file (không xoá trên canvas). Tiếp tục?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
        yolo_io.save_boxes(self._current_image_path, [])
        self._boxes_dirty = False
        self._save_metadata_if_dirty()
        self._sync_dirty_state()
        self._file_list_panel.refresh_label_at(self._current_index)
        self._save_session_state()
        self.statusBar().showMessage("Đã lưu: không có đối tượng nào.", 2500)

    def _save_session_state(self) -> None:
        if self._image_dir is None:
            return
        state = session.SessionState(
            last_image=self._current_image_path.name if self._current_image_path else None,
            last_radius_px=self._reference_radius_px,
            last_tolerance_pct=self._tolerance_spin.value(),
            recent_device_models=self._recent_device_models,
            recent_capture_groups=self._recent_capture_groups,
        )
        session.save_session(self._image_dir, state)

    # ------------------------------------------------------------------
    # Dirty / title
    # ------------------------------------------------------------------
    def _sync_dirty_state(self) -> None:
        self._dirty = self._boxes_dirty or self._metadata_dirty
        self._update_window_title()

    def _mark_boxes_dirty(self) -> None:
        self._boxes_dirty = True
        self._sync_dirty_state()

    def _mark_metadata_dirty(self) -> None:
        self._metadata_dirty = True
        self._sync_dirty_state()

    # Existing box operations and third-party scripts call this name; it is
    # deliberately kept as the box-dirty compatibility alias.
    def _mark_dirty(self) -> None:
        self._mark_boxes_dirty()

    def _update_window_title(self) -> None:
        name = self._current_image_path.name if self._current_image_path else "(chưa mở thư mục)"
        star = "*" if self._dirty else ""
        self.setWindowTitle(f"Xiangqi Labeler - {name}{star}")

    def _refresh_box_list(self) -> None:
        self._box_list_panel.set_boxes(self._canvas.box_items())
        # Rebuilding the list drops its old row selection -- since boxes are
        # now grouped/sorted by class, a box can also change row entirely
        # right after being assigned a class, so re-sync the highlight.
        self._box_list_panel.select_box(self._canvas.selected_box())

    # ------------------------------------------------------------------
    # Undo / redo (per-image snapshot stacks)
    # ------------------------------------------------------------------
    def _snapshot(self) -> list[_BoxSnapshot]:
        result = []
        for item in self._canvas.box_items():
            r = item.rect()
            result.append(_BoxSnapshot((r.x(), r.y(), r.width(), r.height()), item.class_name, item.confirmed))
        return result

    def _apply_snapshot(self, snapshot: list[_BoxSnapshot]) -> None:
        self._canvas.clear_all_boxes()
        for snap in snapshot:
            x, y, w, h = snap.rect
            self._canvas.add_box_item(
                QRectF(x, y, w, h), snap.class_name, snap.confirmed, emit_modified=False
            )
        self._canvas.viewport().update()

    def _push_undo(self, before_snapshot: list[_BoxSnapshot]) -> None:
        self._undo_stack.append(before_snapshot)
        self._redo_stack.clear()
        self._update_undo_redo_actions()

    def _update_undo_redo_actions(self) -> None:
        self._undo_action.setEnabled(bool(self._undo_stack))
        self._redo_action.setEnabled(bool(self._redo_stack))

    def _update_corner_undo_actions(self) -> None:
        self._corner_undo_action.setEnabled(bool(self._corner_undo_stack))
        self._corner_redo_action.setEnabled(bool(self._corner_redo_stack))

    def undo(self) -> None:
        if not self._undo_stack:
            return
        current = self._snapshot()
        previous = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._apply_snapshot(previous)
        self._mark_dirty()
        self._refresh_box_list()
        self._update_undo_redo_actions()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        current = self._snapshot()
        nxt = self._redo_stack.pop()
        self._undo_stack.append(current)
        self._apply_snapshot(nxt)
        self._mark_dirty()
        self._refresh_box_list()
        self._update_undo_redo_actions()

    # ------------------------------------------------------------------
    # Canvas gesture handlers
    # ------------------------------------------------------------------
    def _on_drag_begin(self) -> None:
        self._pending_drag_snapshot = self._snapshot()

    def _on_drag_end(self, changed: bool) -> None:
        if changed and self._pending_drag_snapshot is not None:
            self._push_undo(self._pending_drag_snapshot)
            self._refresh_box_list()
        self._pending_drag_snapshot = None

    def _on_box_drawn(self, rect: QRectF) -> None:
        self._push_undo(self._snapshot())
        self._canvas.add_box_item(rect, class_name=None, confirmed=True, select=True)
        self._mark_dirty()
        self._refresh_box_list()

    def _on_delete_requested(self) -> None:
        box = self._canvas.selected_box()
        if box is None:
            return
        self._push_undo(self._snapshot())
        self._canvas.remove_box(box)
        self._mark_dirty()
        self._refresh_box_list()

    def _on_confirm_requested(self) -> None:
        box = self._canvas.selected_box()
        if box is None or box.confirmed:
            return
        self._push_undo(self._snapshot())
        box.confirmed = True
        box.update()
        self._mark_dirty()
        self._refresh_box_list()

    def _on_box_selected(self, box: BoxItem | None) -> None:
        self._box_list_panel.select_box(box)
        self._duplicate_action.setEnabled(box is not None)
        self._class_combo.blockSignals(True)
        if box is not None and box.class_name in self._classes:
            self._class_combo.setCurrentIndex(self._classes.index(box.class_name))
        else:
            self._class_combo.setCurrentIndex(-1)
        self._class_combo.blockSignals(False)

    def _duplicate_selected_box(self) -> None:
        box = self._canvas.selected_box()
        if box is None:
            return
        self._push_undo(self._snapshot())
        r = box.rect()
        bounds = self._canvas.image_bounds()
        offset = 12.0
        new_x = min(r.x() + offset, max(bounds.left(), bounds.right() - r.width()))
        new_y = min(r.y() + offset, max(bounds.top(), bounds.bottom() - r.height()))
        new_rect = QRectF(new_x, new_y, r.width(), r.height())
        self._canvas.add_box_item(new_rect, box.class_name, box.confirmed, select=True)
        self._mark_dirty()
        self._refresh_box_list()

    def _on_class_combo_activated(self, index: int) -> None:
        box = self._canvas.selected_box()
        if box is None or not (0 <= index < len(self._classes)):
            return
        self._push_undo(self._snapshot())
        box.class_name = self._classes[index]
        box.confirmed = True
        box.update()
        self._mark_dirty()
        self._refresh_box_list()

    # ------------------------------------------------------------------
    # Modes: draw box / measure radius
    # ------------------------------------------------------------------
    def _enter_draw_mode(self) -> None:
        self._canvas.set_mode(CanvasMode.DRAW_BOX)
        self.statusBar().showMessage("Chế độ vẽ box: kéo chuột để vẽ 1 box mới.", 3000)

    def _focus_canvas_for_corners(self) -> None:
        if self._current_image_path is None:
            self.statusBar().showMessage("Hãy mở thư mục ảnh trước khi đánh dấu góc.", 2500)
            return
        self._canvas.setFocus()
        self.statusBar().showMessage(
            "Đưa chuột lên ảnh: 1=trên-trái, 2=trên-phải, 3=dưới-phải, 4=dưới-trái.",
            4500,
        )

    def _start_measure_radius(self) -> None:
        self._canvas.set_mode(CanvasMode.MEASURE_RADIUS)
        self.statusBar().showMessage("Kéo chuột quanh 1 quân cờ mẫu rõ nét để đo bán kính...", 4000)

    def _on_radius_measured(self, r0: float) -> None:
        self._reference_radius_px = r0
        self._radius_spin.blockSignals(True)
        self._radius_spin.setValue(r0)
        self._radius_spin.blockSignals(False)
        self.statusBar().showMessage(f"Đã đo bán kính tham chiếu: {r0:.1f}px", 3000)
        self._run_detection("radius_guided")

    def _on_radius_spin_changed(self, value: float) -> None:
        self._reference_radius_px = value

    # ------------------------------------------------------------------
    # Circle detection
    # ------------------------------------------------------------------
    def _run_detection(self, mode: str) -> None:
        image_bgr = self._ensure_image_bgr()
        if image_bgr is None:
            return

        if mode == "auto_scan":
            circles = circle_detect.auto_scan(image_bgr)
        else:
            if self._reference_radius_px is None:
                QMessageBox.information(
                    self,
                    "Cần đo bán kính",
                    "Hãy đo bán kính tham chiếu trước (nút 'Đo bán kính tham chiếu', kéo chuột quanh 1 quân cờ mẫu).",
                )
                return
            circles = circle_detect.radius_guided(image_bgr, self._reference_radius_px, self._tolerance_spin.value())

        self._push_undo(self._snapshot())
        self._canvas.clear_unconfirmed_suggestions()

        confirmed = [(b.rect().center().x(), b.rect().center().y(), max(b.rect().width(), b.rect().height()) / 2)
                     for b in self._canvas.box_items() if b.confirmed]

        bounds = self._canvas.image_bounds()
        added = 0
        for c in circles:
            too_close = any(
                ((c.cx - cx) ** 2 + (c.cy - cy) ** 2) ** 0.5 < 0.3 * max(r, c.r) for cx, cy, r in confirmed
            )
            if too_close:
                continue
            left, top, side_w, side_h = circle_detect.circle_to_pixel_box(c)
            rect = QRectF(left, top, side_w, side_h).intersected(bounds)
            if rect.width() < 2 or rect.height() < 2:
                continue
            self._canvas.add_box_item(rect, class_name=None, confirmed=False)
            added += 1

        self._mark_dirty()
        self._refresh_box_list()
        self._save_session_state()
        self.statusBar().showMessage(f"Phát hiện {added} gợi ý hình tròn.", 3000)

    def _clear_unconfirmed_suggestions(self) -> None:
        self._push_undo(self._snapshot())
        self._canvas.clear_unconfirmed_suggestions()
        self._mark_dirty()
        self._refresh_box_list()

    # ------------------------------------------------------------------
    # Class-assignment shortcuts (global key event filter)
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            modifiers = event.modifiers()
            shift_delete = (
                key == Qt.Key.Key_Delete
                and bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                and not bool(
                    modifiers
                    & (
                        Qt.KeyboardModifier.ControlModifier
                        | Qt.KeyboardModifier.AltModifier
                        | Qt.KeyboardModifier.MetaModifier
                    )
                )
            )
            focus = QApplication.focusWidget()
            if shift_delete and QApplication.activeModalWidget() is None and not isinstance(
                focus, (QLineEdit, QTextEdit)
            ):
                # Consume this before ImageCanvas/board-editor sees Delete;
                # plain Delete keeps its existing meaning of deleting a box or
                # a selected chess piece.
                self._delete_current_image()
                return True

            ctrl_held = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            other_modifiers_held = bool(
                modifiers & (Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
            )

            # `hand` has no color, so it keeps its own Ctrl+H shortcut --
            # none of the 7 role letters is free for it (H is "horse").
            if key == Qt.Key.Key_H and ctrl_held and not other_modifiers_held:
                self._assign_class_to_selected(HAND_CLASS)
                return True

            if key in _ROLE_KEY_CODES and not ctrl_held and not other_modifiers_held:
                # Case carries the color: lowercase = black, UPPERCASE
                # (Caps Lock or Shift) = red. event.text() reflects the
                # OS's Caps-Lock-aware translation; event.modifiers()
                # would not (Caps Lock isn't a Qt modifier flag).
                class_name = resolve_piece_class(_ROLE_KEY_CODES[key], is_red=event.text().isupper())
                if class_name is not None:
                    self._assign_class_to_selected(class_name)
                    return True
        return super().eventFilter(obj, event)

    def _assign_class_to_selected(self, class_name: str) -> None:
        box = self._canvas.selected_box()
        if box is None:
            self.statusBar().showMessage("Không có box nào đang chọn -- bỏ qua phím tắt gán lớp.", 2000)
            return
        self._push_undo(self._snapshot())
        box.class_name = class_name
        box.confirmed = True
        box.update()
        self._mark_dirty()
        self._refresh_box_list()
        self.statusBar().showMessage(f"Đã gán lớp: {class_name}", 1200)

    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._shown_once:
            # An image can get loaded (e.g. via _resume_last_directory, run
            # from __init__) before the window is ever shown, while the
            # canvas viewport still has its pre-layout placeholder size --
            # fit_to_window computed then would zoom to that wrong size.
            # Redo it once real geometry exists, the first time we're shown.
            self._shown_once = True
            self._canvas.fit_to_window()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._confirm_leave_current_image():
            event.accept()
        else:
            event.ignore()
