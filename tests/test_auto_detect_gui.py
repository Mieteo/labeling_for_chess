"""End-to-end wiring tests for the Digital additions:
- image-mode inference/override + circle-detect vs auto-detect UI visibility
  (yeu_cau_tu_app_ky_nhan.md section 2),
- single-image auto-detect reusing the existing suggestion/confirm/save
  machinery (section 3.4),
- the batch worker's sidecar output (section 3.4, item 1/6),
- pending-suggestion pickup on image open (section 3.4).

Uses a fake detector (duck-typing AutoDetector's public surface: `.detect()`
and `.model_path`) so these tests do not depend on a real .onnx file being
present on the machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QRectF, QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from chess_labeler import auto_detect, image_mode, suggestions, yolo_io
from chess_labeler.auto_detect_worker import AutoDetectBatchWorker
from chess_labeler.constants import DEFAULT_CLASSES, DEFAULT_CLASSES_DIGITAL
from chess_labeler.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def isolated_settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def main_window(qapp, isolated_settings):
    win = MainWindow(settings=isolated_settings)
    yield win
    win._dirty = False
    win.close()
    qapp.removeEventFilter(win)


def _digital_dataset(tmp_path: Path) -> Path:
    d = tmp_path / "digitalImg"
    d.mkdir()
    Image.new("RGB", (640, 480), (220, 220, 220)).save(d / "d0001.png")
    Image.new("RGB", (640, 480), (220, 220, 220)).save(d / "d0002.png")
    return d


class _FakeDetector:
    def __init__(self, detections, model_path="fake.onnx"):
        self.model_path = model_path
        self._detections = detections

    def detect(self, image_bgr, conf_threshold, iou_threshold):
        return list(self._detections)


class _QueueDetector:
    """Returns a different (possibly-exception) result per call, in order."""

    def __init__(self, results, model_path="fake.onnx"):
        self.model_path = model_path
        self._results = list(results)
        self._i = 0

    def detect(self, image_bgr, conf_threshold, iou_threshold):
        result = self._results[self._i]
        self._i += 1
        if isinstance(result, Exception):
            raise result
        return result


def _press_key(widget, key, text=""):
    event = QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier, text)
    QApplication.sendEvent(widget, event)


# ---------------------------------------------------------------------
# Image mode inference + UI visibility
# ---------------------------------------------------------------------
def test_physical_dirname_shows_circle_detect_hides_auto_detect(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)  # dir name "chessImg"
    assert main_window._image_mode == image_mode.PHYSICAL
    # The window is never shown in these headless tests, so isVisible() is
    # always False regardless of setVisible() -- isHidden() reflects the
    # widget's own explicit visibility flag independent of ancestor state.
    assert all(not w.isHidden() for w in main_window._physical_assist_widgets)
    assert all(w.isHidden() for w in main_window._digital_assist_widgets)


def test_digital_dirname_shows_auto_detect_hides_circle_detect(main_window, tmp_path):
    main_window._open_directory(_digital_dataset(tmp_path))
    assert main_window._image_mode == image_mode.DIGITAL
    assert all(w.isHidden() for w in main_window._physical_assist_widgets)
    assert all(not w.isHidden() for w in main_window._digital_assist_widgets)


def test_manual_mode_override_persists_across_reopen(qapp, isolated_settings, labelimg_dataset):
    win1 = MainWindow(settings=isolated_settings)
    win1._open_directory(labelimg_dataset)
    assert win1._image_mode == image_mode.PHYSICAL

    win1._on_mode_combo_changed(1)  # force Digital despite the "chessImg" name
    assert win1._image_mode == image_mode.DIGITAL
    win1.close()
    qapp.removeEventFilter(win1)

    win2 = MainWindow(settings=isolated_settings)
    win2._open_directory(labelimg_dataset)
    assert win2._image_mode == image_mode.DIGITAL  # override wins over inference
    win2.close()
    qapp.removeEventFilter(win2)


# ---------------------------------------------------------------------
# Single-image auto-detect reuses the existing suggestion/confirm/save flow
# ---------------------------------------------------------------------
def test_auto_detect_current_image_adds_unconfirmed_suggestions_with_class(main_window, tmp_path):
    main_window._open_directory(_digital_dataset(tmp_path))
    detection = auto_detect.Detection(cx=100, cy=80, w=40, h=60, class_name="red_cannon", score=0.83)
    main_window._auto_detector = _FakeDetector([detection])
    main_window._model_path_edit.setText("fake.onnx")

    main_window._run_auto_detect_current()

    boxes = main_window._canvas.box_items()
    assert len(boxes) == 1
    assert boxes[0].class_name == "red_cannon"
    assert boxes[0].confirmed is False


def test_auto_detect_suggestion_excluded_until_confirmed_then_saved(main_window, tmp_path):
    d = _digital_dataset(tmp_path)
    main_window._open_directory(d)
    detection = auto_detect.Detection(cx=100, cy=80, w=40, h=60, class_name="red_cannon", score=0.83)
    main_window._auto_detector = _FakeDetector([detection])
    main_window._model_path_edit.setText("fake.onnx")
    main_window._run_auto_detect_current()

    main_window._save_current_image()
    assert yolo_io.load_boxes(main_window._current_image_path) == []  # unconfirmed -> not saved

    box = main_window._canvas.box_items()[0]
    assert box.class_name == "red_cannon" and box.confirmed is False
    main_window._canvas.select_box(box)  # suggestions aren't auto-selected, same as circle-detect
    _press_key(main_window._canvas, Qt.Key.Key_Return)  # confirm, keeping the model's class
    assert box.confirmed is True

    main_window._save_current_image()
    saved = yolo_io.load_boxes(main_window._current_image_path)
    assert len(saved) == 1
    assert saved[0].class_id == main_window._classes.index("red_cannon")


def test_auto_detect_rerun_clears_previous_unconfirmed_suggestions(main_window, tmp_path):
    main_window._open_directory(_digital_dataset(tmp_path))
    first = auto_detect.Detection(cx=100, cy=80, w=40, h=60, class_name="red_cannon", score=0.83)
    main_window._auto_detector = _FakeDetector([first])
    main_window._model_path_edit.setText("fake.onnx")
    main_window._run_auto_detect_current()
    assert len(main_window._canvas.box_items()) == 1

    second = auto_detect.Detection(cx=200, cy=150, w=30, h=30, class_name="black_pawn", score=0.7)
    main_window._auto_detector = _FakeDetector([second])
    main_window._run_auto_detect_current()

    boxes = main_window._canvas.box_items()
    assert len(boxes) == 1
    assert boxes[0].class_name == "black_pawn"


# ---------------------------------------------------------------------
# Pending suggestions sidecar is picked up on open, and consumed
# ---------------------------------------------------------------------
def test_pending_suggestions_sidecar_loaded_as_unconfirmed_on_open(main_window, tmp_path):
    d = _digital_dataset(tmp_path)
    image_path = d / "d0001.png"
    suggestions.save_suggestions(
        image_path,
        [suggestions.PendingBox(class_name="black_king", xc=0.5, yc=0.5, w=0.1, h=0.15, score=0.6)],
        model_path="fake.onnx",
        conf_threshold=0.25,
        iou_threshold=0.45,
    )

    main_window._open_directory(d)
    assert main_window._current_image_path == image_path
    boxes = main_window._canvas.box_items()
    assert len(boxes) == 1
    assert boxes[0].class_name == "black_king"
    assert boxes[0].confirmed is False
    # Consumed -- reopening the same image must not duplicate it.
    assert not suggestions.has_pending_suggestions(image_path)


# ---------------------------------------------------------------------
# Batch worker (run() called directly/synchronously -- see
# yeu_cau_tu_app_ky_nhan.md section 3.4 item 6 for the cancel/progress
# contract exercised via signals here)
# ---------------------------------------------------------------------
def test_batch_worker_writes_sidecars_and_reports_counts(tmp_path):
    d = tmp_path / "digitalImg"
    d.mkdir()
    ok_with_detections = d / "a.png"
    ok_no_detections = d / "b.png"
    unreadable = d / "c.png"
    Image.new("RGB", (100, 80), (200, 200, 200)).save(ok_with_detections)
    Image.new("RGB", (100, 80), (200, 200, 200)).save(ok_no_detections)
    unreadable.write_bytes(b"")  # 0 bytes -> read_image_bgr returns None

    detection = auto_detect.Detection(cx=50, cy=40, w=20, h=16, class_name="red_king", score=0.7)
    detector = _QueueDetector([[detection], [], RuntimeError("boom")])

    worker = AutoDetectBatchWorker(detector, [ok_with_detections, ok_no_detections, unreadable], 0.25, 0.45)
    results: list[tuple[int, int]] = []
    worker.finishedBatch.connect(lambda done, skipped: results.append((done, skipped)))
    worker.run()  # call directly (synchronous) instead of start() -- no real thread needed

    assert results == [(1, 1)]  # 1 processed, 1 skipped; "no detections" counts as neither
    assert suggestions.has_pending_suggestions(ok_with_detections)
    assert not suggestions.has_pending_suggestions(ok_no_detections)
    assert not suggestions.has_pending_suggestions(unreadable)

    loaded = suggestions.load_and_consume_suggestions(ok_with_detections)
    assert len(loaded) == 1
    assert loaded[0].class_name == "red_king"
    assert loaded[0].xc == pytest.approx(0.5)
    assert loaded[0].yc == pytest.approx(0.5)


def test_batch_worker_cancel_stops_early(tmp_path):
    d = tmp_path / "digitalImg"
    d.mkdir()
    paths = []
    for i in range(3):
        p = d / f"{i}.png"
        Image.new("RGB", (100, 80), (200, 200, 200)).save(p)
        paths.append(p)

    detection = auto_detect.Detection(cx=50, cy=40, w=20, h=16, class_name="red_king", score=0.7)
    detector = _QueueDetector([[detection]] * 3)
    worker = AutoDetectBatchWorker(detector, paths, 0.25, 0.45)

    seen = []

    def on_progress(done, total, filename):
        seen.append(done)
        if done == 1:
            worker.cancel()

    worker.progress.connect(on_progress)
    worker.run()

    # Cancelled after the 2nd image's progress tick (index 1) -- fewer than
    # all 3 images should have been processed.
    assert sum(suggestions.has_pending_suggestions(p) for p in paths) < 3


# ---------------------------------------------------------------------
# classes.txt schema fork: Digital gets the extra `board_region` 16th
# class, Physical keeps the plain 15 (yeu_cau_tu_app_ky_nhan.md section 1)
# ---------------------------------------------------------------------
def test_opening_fresh_digital_directory_creates_16_class_classes_txt(main_window, tmp_path):
    d = _digital_dataset(tmp_path)
    main_window._open_directory(d)
    assert main_window._classes == DEFAULT_CLASSES_DIGITAL
    assert "board_region" in main_window._classes
    on_disk = (d / "classes.txt").read_text(encoding="utf-8").strip("\n").split("\n")
    assert on_disk == DEFAULT_CLASSES_DIGITAL


def test_opening_fresh_physical_directory_still_creates_15_class_classes_txt(main_window, tmp_path):
    d = tmp_path / "chessImgFresh"
    d.mkdir()
    Image.new("RGB", (640, 480), (220, 220, 220)).save(d / "p0001.jpg")
    main_window._open_directory(d)
    assert main_window._classes == DEFAULT_CLASSES
    assert "board_region" not in main_window._classes


# ---------------------------------------------------------------------
# Ctrl+B assigns board_region, the colorless 16th Digital-only class
# (yeu_cau_tu_app_ky_nhan.md section 1)
# ---------------------------------------------------------------------
def test_ctrl_b_assigns_board_region(main_window, tmp_path):
    main_window._open_directory(_digital_dataset(tmp_path))
    box = main_window._canvas.add_box_item(QRectF(10, 10, 20, 20), class_name=None, confirmed=False, select=True)

    QTest.keyClick(main_window._canvas, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)

    assert box.class_name == "board_region"
    assert box.confirmed is True
