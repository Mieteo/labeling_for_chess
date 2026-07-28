"""Main application window: wires the canvas, side panels, chord shortcuts,
circle-detect assist, and file I/O together. See
labeling_tool_requirements.md for the full behavioral spec this implements.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QToolBar,
    QWidget,
)

from . import circle_detect, session, yolo_io
from .canvas import BoxItem, CanvasMode, ImageCanvas
from .chord import ChordOutcome, ChordShortcutManager
from .constants import DEFAULT_RADIUS_TOLERANCE_PCT
from .panels import BoxListPanel, FileListPanel

_CHORD_LETTER_KEYS = {"R", "B", "K", "A", "E", "H", "C", "P"}


@dataclasses.dataclass(frozen=True)
class _BoxSnapshot:
    rect: tuple[float, float, float, float]
    class_name: str | None
    confirmed: bool


def _read_image_bgr(path: Path) -> np.ndarray | None:
    """Unicode-path-safe read (cv2.imread mishandles non-ASCII paths on Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Xiangqi Labeler")
        self.resize(1440, 920)

        self._image_dir: Path | None = None
        self._classes: list[str] = []
        self._images: list[Path] = []
        self._current_index: int = -1
        self._current_image_path: Path | None = None
        self._current_image_bgr: np.ndarray | None = None
        self._dirty = False

        self._undo_stack: list[list[_BoxSnapshot]] = []
        self._redo_stack: list[list[_BoxSnapshot]] = []
        self._pending_drag_snapshot: list[_BoxSnapshot] | None = None

        self._reference_radius_px: float | None = None
        self._chord = ChordShortcutManager()

        self._canvas = ImageCanvas(self)
        self.setCentralWidget(self._canvas)
        self._canvas.on_drag_begin = self._on_drag_begin
        self._canvas.on_drag_end = self._on_drag_end
        self._canvas.boxDrawn.connect(self._on_box_drawn)
        self._canvas.radiusMeasured.connect(self._on_radius_measured)
        self._canvas.deleteRequested.connect(self._on_delete_requested)
        self._canvas.confirmRequested.connect(self._on_confirm_requested)
        self._canvas.boxSelected.connect(self._on_box_selected)
        self._canvas.sceneModified.connect(self._mark_dirty)

        self._build_dock_panels()
        self._build_toolbar_and_actions()
        self.setStatusBar(QStatusBar(self))
        self._chord_indicator = QLabel("")
        self._chord_indicator.setStyleSheet("font-weight: bold; color: #d81b60; padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._chord_indicator)

        self._chord_timer = QTimer(self)
        self._chord_timer.timeout.connect(self._on_chord_timer)
        self._chord_timer.start(250)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self._update_window_title()
        self._update_undo_redo_actions()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_dock_panels(self) -> None:
        self._file_list_panel = FileListPanel(self)
        file_dock = QDockWidget("Danh sach anh", self)
        file_dock.setWidget(self._file_list_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, file_dock)
        self._file_list_panel.imageActivated.connect(self._go_to_index)

        self._box_list_panel = BoxListPanel(self)
        box_dock = QDockWidget("Box trong anh nay", self)
        box_dock.setWidget(self._box_list_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, box_dock)
        self._box_list_panel.boxActivated.connect(self._canvas.select_box)

        assist = QWidget(self)
        form = QFormLayout(assist)

        self._class_combo = QComboBox(assist)
        self._class_combo.activated.connect(self._on_class_combo_activated)
        form.addRow("Lop:", self._class_combo)

        measure_btn = QPushButton("Do ban kinh tham chieu (keo chuot)", assist)
        measure_btn.clicked.connect(self._start_measure_radius)
        form.addRow(measure_btn)

        self._radius_spin = QDoubleSpinBox(assist)
        self._radius_spin.setRange(1.0, 5000.0)
        self._radius_spin.setDecimals(1)
        self._radius_spin.valueChanged.connect(self._on_radius_spin_changed)
        form.addRow("Ban kinh (px):", self._radius_spin)

        self._tolerance_spin = QDoubleSpinBox(assist)
        self._tolerance_spin.setRange(1.0, 100.0)
        self._tolerance_spin.setDecimals(1)
        self._tolerance_spin.setSuffix(" %")
        self._tolerance_spin.setValue(DEFAULT_RADIUS_TOLERANCE_PCT)
        form.addRow("Dung sai ban kinh:", self._tolerance_spin)

        rerun_btn = QPushButton("Chay lai phat hien (radius-guided)", assist)
        rerun_btn.clicked.connect(lambda: self._run_detection("radius_guided"))
        form.addRow(rerun_btn)

        autoscan_btn = QPushButton("Auto-scan toan anh", assist)
        autoscan_btn.clicked.connect(lambda: self._run_detection("auto_scan"))
        form.addRow(autoscan_btn)

        clear_btn = QPushButton("Xoa goi y chua xac nhan", assist)
        clear_btn.clicked.connect(self._clear_unconfirmed_suggestions)
        form.addRow(clear_btn)

        assist_dock = QDockWidget("Gan lop / Circle-assist", self)
        assist_dock.setWidget(assist)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, assist_dock)

    def _build_toolbar_and_actions(self) -> None:
        toolbar = QToolBar("Main", self)
        self.addToolBar(toolbar)

        open_action = QAction("Mo thu muc...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_directory)
        toolbar.addAction(open_action)

        save_action = QAction("Luu", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._save_current_image)
        toolbar.addAction(save_action)

        save_empty_action = QAction("Luu: khong co doi tuong nao", self)
        save_empty_action.triggered.connect(self._save_current_image_as_empty)
        toolbar.addAction(save_empty_action)

        self._autosave_action = QAction("Auto-save khi chuyen anh", self)
        self._autosave_action.setCheckable(True)
        self._autosave_action.setChecked(True)
        toolbar.addAction(self._autosave_action)

        toolbar.addSeparator()

        prev_action = QAction("Anh truoc (A)", self)
        prev_action.setShortcut(QKeySequence("A"))
        prev_action.triggered.connect(self.prev_image)
        toolbar.addAction(prev_action)

        next_action = QAction("Anh sau (D)", self)
        next_action.setShortcut(QKeySequence("D"))
        next_action.triggered.connect(self.next_image)
        toolbar.addAction(next_action)

        toolbar.addSeparator()

        draw_action = QAction("Ve box (W)", self)
        draw_action.setShortcut(QKeySequence("W"))
        draw_action.triggered.connect(self._enter_draw_mode)
        toolbar.addAction(draw_action)

        self._duplicate_action = QAction("Nhan doi box (Ctrl+D)", self)
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
        directory = QFileDialog.getExistingDirectory(self, "Chon thu muc anh")
        if not directory:
            return
        self._open_directory(Path(directory))

    def _open_directory(self, directory: Path) -> None:
        self._image_dir = directory
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

        if not self._images:
            QMessageBox.information(
                self, "Thu muc trong", "Khong tim thay anh .jpg/.jpeg/.png nao trong thu muc nay."
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
            QMessageBox.warning(self, "Loi", f"Khong the mo anh: {path}")
            return

        self._current_index = index
        self._current_image_path = path
        self._current_image_bgr = None
        self._canvas.load_image(pixmap)

        for box in yolo_io.load_boxes(path):
            left, top, w, h = box.to_pixel_rect(pixmap.width(), pixmap.height())
            name = yolo_io.class_name(self._classes, box.class_id)
            self._canvas.add_box_item(QRectF(left, top, w, h), class_name=name, confirmed=True)

        self._undo_stack.clear()
        self._redo_stack.clear()
        self._pending_drag_snapshot = None
        self._dirty = False

        self._file_list_panel.set_current_index(index)
        self._refresh_box_list()
        self._canvas.fit_to_window()
        self._canvas.setFocus()
        self._update_window_title()
        self._update_undo_redo_actions()
        self._save_session_state()

    def _ensure_image_bgr(self) -> np.ndarray | None:
        if self._current_image_bgr is None and self._current_image_path is not None:
            self._current_image_bgr = _read_image_bgr(self._current_image_path)
        return self._current_image_bgr

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
            "Thay doi chua luu",
            "Anh hien tai co thay doi chua luu. Luu truoc khi chuyen anh?",
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

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _save_current_image(self) -> bool:
        if self._current_image_path is None:
            return False
        items = self._canvas.box_items()
        unclassified = [b for b in items if b.confirmed and not b.class_name]
        if unclassified:
            resp = QMessageBox.warning(
                self,
                "Con box chua gan lop",
                f"{len(unclassified)} box da xac nhan nhung CHUA gan lop -- cac box nay se "
                "KHONG duoc ghi vao file .txt. Van luu?",
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
        self._dirty = False
        self._update_window_title()
        self._file_list_panel.refresh_label_at(self._current_index)
        self._save_session_state()
        self.statusBar().showMessage(f"Da luu {self._current_image_path.name} ({len(boxes_to_save)} box).", 2500)
        return True

    def _save_current_image_as_empty(self) -> None:
        if self._current_image_path is None:
            return
        boxes = self._canvas.box_items()
        if boxes:
            resp = QMessageBox.question(
                self,
                "Xac nhan",
                f"Anh nay dang co {len(boxes)} box tren canvas. Luu nhu KHONG co doi tuong nao se "
                "bo qua toan bo box hien tai khi ghi file (khong xoa tren canvas). Tiep tuc?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
        yolo_io.save_boxes(self._current_image_path, [])
        self._dirty = False
        self._update_window_title()
        self._file_list_panel.refresh_label_at(self._current_index)
        self._save_session_state()
        self.statusBar().showMessage("Da luu: khong co doi tuong nao.", 2500)

    def _save_session_state(self) -> None:
        if self._image_dir is None:
            return
        state = session.SessionState(
            last_image=self._current_image_path.name if self._current_image_path else None,
            last_radius_px=self._reference_radius_px,
            last_tolerance_pct=self._tolerance_spin.value(),
        )
        session.save_session(self._image_dir, state)

    # ------------------------------------------------------------------
    # Dirty / title
    # ------------------------------------------------------------------
    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_window_title()

    def _update_window_title(self) -> None:
        name = self._current_image_path.name if self._current_image_path else "(chua mo thu muc)"
        star = "*" if self._dirty else ""
        self.setWindowTitle(f"Xiangqi Labeler - {name}{star}")

    def _refresh_box_list(self) -> None:
        self._box_list_panel.set_boxes(self._canvas.box_items())

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
            self._canvas.add_box_item(QRectF(x, y, w, h), snap.class_name, snap.confirmed)
        self._canvas.viewport().update()

    def _push_undo(self, before_snapshot: list[_BoxSnapshot]) -> None:
        self._undo_stack.append(before_snapshot)
        self._redo_stack.clear()
        self._update_undo_redo_actions()

    def _update_undo_redo_actions(self) -> None:
        self._undo_action.setEnabled(bool(self._undo_stack))
        self._redo_action.setEnabled(bool(self._redo_stack))

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
        self.statusBar().showMessage("Che do ve box: keo chuot de ve 1 box moi.", 3000)

    def _start_measure_radius(self) -> None:
        self._canvas.set_mode(CanvasMode.MEASURE_RADIUS)
        self.statusBar().showMessage("Keo chuot quanh 1 quan co mau ro net de do ban kinh...", 4000)

    def _on_radius_measured(self, r0: float) -> None:
        self._reference_radius_px = r0
        self._radius_spin.blockSignals(True)
        self._radius_spin.setValue(r0)
        self._radius_spin.blockSignals(False)
        self.statusBar().showMessage(f"Da do ban kinh tham chieu: {r0:.1f}px", 3000)
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
                    "Can do ban kinh",
                    "Hay do ban kinh tham chieu truoc (nut 'Do ban kinh tham chieu', keo chuot quanh 1 quan co mau).",
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
        self.statusBar().showMessage(f"Phat hien {added} goi y hinh tron.", 3000)

    def _clear_unconfirmed_suggestions(self) -> None:
        self._push_undo(self._snapshot())
        self._canvas.clear_unconfirmed_suggestions()
        self._mark_dirty()
        self._refresh_box_list()

    # ------------------------------------------------------------------
    # Chord shortcuts (global key event filter)
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Escape:
                ev = self._chord.escape_pressed()
                if ev.outcome != ChordOutcome.IGNORED:
                    self._handle_chord_event(ev)
                    return True
                return False
            if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
                letter = chr(key)
                ctrl_held = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                if letter in _CHORD_LETTER_KEYS and (ctrl_held or self._chord.is_pending):
                    ev = self._chord.key_pressed(letter, ctrl_held)
                    if ev.outcome != ChordOutcome.IGNORED:
                        self._handle_chord_event(ev)
                        return True
        return super().eventFilter(obj, event)

    def _handle_chord_event(self, ev) -> None:
        if ev.outcome == ChordOutcome.PENDING:
            self._chord_indicator.setText(ev.pending_label or "")
        elif ev.outcome == ChordOutcome.CANCELLED:
            self._chord_indicator.setText("")
        elif ev.outcome == ChordOutcome.ASSIGNED:
            self._chord_indicator.setText("")
            self._assign_class_to_selected(ev.class_name)

    def _on_chord_timer(self) -> None:
        ev = self._chord.check_timeout()
        if ev.outcome == ChordOutcome.CANCELLED:
            self._chord_indicator.setText("")

    def _assign_class_to_selected(self, class_name: str | None) -> None:
        box = self._canvas.selected_box()
        if box is None:
            self.statusBar().showMessage("Khong co box nao dang chon -- chord bi bo qua.", 2000)
            return
        self._push_undo(self._snapshot())
        box.class_name = class_name
        box.confirmed = True
        box.update()
        self._mark_dirty()
        self._refresh_box_list()

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        if self._confirm_leave_current_image():
            event.accept()
        else:
            event.ignore()
