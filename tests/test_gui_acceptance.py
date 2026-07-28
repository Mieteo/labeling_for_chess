"""End-to-end acceptance tests driving the real MainWindow headlessly
(QT_QPA_PLATFORM=offscreen, set in conftest.py). Covers acceptance criteria
2-5 from labeling_tool_requirements.md section 8.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from chess_labeler import yolo_io
from chess_labeler.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def main_window(qapp):
    win = MainWindow()
    yield win
    # Teardown just disposes of the Qt object -- never let a leftover dirty/
    # unclassified-box state pop a blocking QMessageBox (closeEvent ->
    # autosave -> "unclassified boxes" warning) with nobody there to click it.
    win._dirty = False
    win.close()
    qapp.removeEventFilter(win)


def _index_of(win: MainWindow, name: str) -> int:
    for i, p in enumerate(win._images):
        if p.name == name:
            return i
    raise AssertionError(f"{name} not found in {win._images}")


def test_open_directory_resumes_at_first_unlabeled_image(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    assert main_window._current_image_path.name == "0005.jpg"


def test_loaded_boxes_match_labelimg_txt_exactly(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))

    boxes = main_window._canvas.box_items()
    assert len(boxes) == 2
    names = sorted(b.class_name for b in boxes)
    assert names == ["black_pawn", "red_king"]
    assert all(b.confirmed for b in boxes)


def test_chord_shortcut_assigns_class_end_to_end(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))

    box = main_window._canvas.add_box_item(QRectF(400, 300, 30, 30), class_name=None, confirmed=True, select=True)

    QTest.keyClick(main_window._canvas, Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)
    assert main_window._chord.is_pending
    QTest.keyClick(main_window._canvas, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)

    assert box.class_name == "red_cannon"
    assert box.confirmed is True
    assert not main_window._chord.is_pending


def test_ctrl_h_alone_assigns_hand_not_pending(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    box = main_window._canvas.add_box_item(QRectF(10, 10, 20, 20), class_name=None, confirmed=True, select=True)

    QTest.keyClick(main_window._canvas, Qt.Key.Key_H, Qt.KeyboardModifier.ControlModifier)

    assert box.class_name == "hand"
    assert not main_window._chord.is_pending


def test_save_round_trip_and_suggestions_excluded(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    assert main_window._current_image_path.name == "0005.jpg"

    box = main_window._canvas.add_box_item(QRectF(50, 50, 40, 40), class_name=None, confirmed=True, select=True)
    QTest.keyClick(main_window._canvas, Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(main_window._canvas, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert box.class_name == "red_cannon"

    assert main_window._save_current_image() is True
    saved = yolo_io.load_boxes(labelimg_dataset / "0005.jpg")
    assert len(saved) == 1
    assert saved[0].class_id == main_window._classes.index("red_cannon")

    # An unconfirmed suggestion must never make it into the saved .txt.
    main_window._canvas.add_box_item(QRectF(200, 200, 30, 30), class_name=None, confirmed=False)
    main_window._save_current_image()
    saved_again = yolo_io.load_boxes(labelimg_dataset / "0005.jpg")
    assert len(saved_again) == 1


def test_resume_advances_after_labeling_a_fresh_window(qapp, labelimg_dataset):
    win1 = MainWindow()
    win1._open_directory(labelimg_dataset)
    box = win1._canvas.add_box_item(QRectF(10, 10, 20, 20), class_name=None, confirmed=True, select=True)
    QTest.keyClick(win1._canvas, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(win1._canvas, Qt.Key.Key_P, Qt.KeyboardModifier.ControlModifier)
    assert box.class_name == "black_pawn"
    win1._save_current_image()
    win1.close()
    qapp.removeEventFilter(win1)

    win2 = MainWindow()
    win2._open_directory(labelimg_dataset)
    assert win2._current_image_path.name == "0006.jpg"
    win2.close()
    qapp.removeEventFilter(win2)


def test_explicit_save_as_empty_writes_empty_txt(main_window, labelimg_dataset, monkeypatch):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    assert len(main_window._canvas.box_items()) == 2  # pre-existing boxes still on canvas

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    main_window._save_current_image_as_empty()

    assert yolo_io.has_label(labelimg_dataset / "0001.jpg") is True
    assert yolo_io.load_boxes(labelimg_dataset / "0001.jpg") == []


def test_undo_redo_restores_box_state(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    assert len(main_window._canvas.box_items()) == 2

    main_window._on_box_drawn(QRectF(400, 300, 25, 25))
    assert len(main_window._canvas.box_items()) == 3

    main_window.undo()
    assert len(main_window._canvas.box_items()) == 2

    main_window.redo()
    assert len(main_window._canvas.box_items()) == 3


def test_delete_selected_box(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    box = main_window._canvas.box_items()[0]
    main_window._canvas.select_box(box)

    main_window._on_delete_requested()

    assert len(main_window._canvas.box_items()) == 1


def test_duplicate_selected_box(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    box = main_window._canvas.box_items()[0]
    main_window._canvas.select_box(box)

    main_window._duplicate_selected_box()

    assert len(main_window._canvas.box_items()) == 3
