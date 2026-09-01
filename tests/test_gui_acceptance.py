"""End-to-end acceptance tests driving the real MainWindow headlessly
(QT_QPA_PLATFORM=offscreen, set in conftest.py). Covers acceptance criteria
2-5 from labeling_tool_requirements.md section 8.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QRectF, QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from chess_labeler import metadata, yolo_io
import chess_labeler.main_window as main_window_module
from chess_labeler.main_window import MainWindow


def _press_letter(widget, key, text):
    """Simulate a bare letter keypress with a specific case, e.g. `text="C"`
    for a Caps-Lock/Shift-typed 'C'. QTest.keyClick's Qt.Key overload doesn't
    populate event.text() (case doesn't matter for its own Ctrl-based
    shortcuts), so the letter shortcuts -- which read event.text() to tell
    red from black -- need a manually-built QKeyEvent instead."""
    event = QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier, text)
    QApplication.sendEvent(widget, event)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def isolated_settings(tmp_path):
    """A QSettings backed by a throwaway ini file -- MainWindow persists the
    last-opened directory (see labeling_tool_requirements.md section 5
    update) via QSettings, which defaults to the real Windows registry.
    Tests must never touch that real, machine-wide key."""
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def main_window(qapp, isolated_settings):
    win = MainWindow(settings=isolated_settings)
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


def test_reopening_app_auto_opens_last_used_folder(qapp, isolated_settings, labelimg_dataset):
    win1 = MainWindow(settings=isolated_settings)
    win1._open_directory(labelimg_dataset)
    win1.close()
    qapp.removeEventFilter(win1)

    win2 = MainWindow(settings=isolated_settings)  # simulates relaunching the app
    assert win2._image_dir == labelimg_dataset
    win2.close()
    qapp.removeEventFilter(win2)


def test_fresh_app_with_no_saved_folder_opens_nothing(qapp, isolated_settings):
    win = MainWindow(settings=isolated_settings)
    assert win._image_dir is None
    win.close()
    qapp.removeEventFilter(win)


def test_right_sidebar_groups_annotation_tools_and_fen_into_two_tabs(main_window):
    tabs = main_window._right_tabs

    assert tabs.count() == 2
    assert [tabs.tabText(index) for index in range(tabs.count())] == ["Gán nhãn", "Bàn cờ & FEN"]
    assert main_window._annotation_dock.minimumWidth() >= 520
    assert main_window._box_list_panel.parentWidget().objectName() == "boxesGroup"
    assert main_window._metadata_panel.parentWidget().objectName() == "captureConditionsGroup"
    assert main_window._board_editor.parentWidget().objectName() == "fenTab"

    # Compact class-assist controls keep the frequently used inputs on the
    # same short rows instead of consuming one full row each.
    assert main_window._class_combo.maximumWidth() == 190
    assert main_window._radius_spin.maximumWidth() == 105
    assert main_window._tolerance_spin.maximumWidth() == 105


def test_stale_saved_folder_that_no_longer_exists_is_ignored(qapp, isolated_settings, tmp_path):
    isolated_settings.setValue("last_image_dir", str(tmp_path / "deleted_folder"))
    win = MainWindow(settings=isolated_settings)  # must not crash on a missing directory
    assert win._image_dir is None
    win.close()
    qapp.removeEventFilter(win)


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


def test_letter_shortcut_assigns_black_class_end_to_end(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))

    box = main_window._canvas.add_box_item(QRectF(400, 300, 30, 30), class_name=None, confirmed=True, select=True)

    _press_letter(main_window._canvas, Qt.Key.Key_C, "c")

    assert box.class_name == "black_cannon"
    assert box.confirmed is True


def test_letter_shortcut_uppercase_assigns_red_class_end_to_end(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))

    box = main_window._canvas.add_box_item(QRectF(400, 300, 30, 30), class_name=None, confirmed=True, select=True)

    _press_letter(main_window._canvas, Qt.Key.Key_C, "C")

    assert box.class_name == "red_cannon"
    assert box.confirmed is True


def test_ctrl_h_assigns_hand(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    box = main_window._canvas.add_box_item(QRectF(10, 10, 20, 20), class_name=None, confirmed=True, select=True)

    QTest.keyClick(main_window._canvas, Qt.Key.Key_H, Qt.KeyboardModifier.ControlModifier)

    assert box.class_name == "hand"


def test_save_round_trip_and_suggestions_excluded(main_window, labelimg_dataset, monkeypatch):
    main_window._open_directory(labelimg_dataset)
    assert main_window._current_image_path.name == "0005.jpg"

    box = main_window._canvas.add_box_item(QRectF(50, 50, 40, 40), class_name=None, confirmed=True, select=True)
    _press_letter(main_window._canvas, Qt.Key.Key_C, "C")
    assert box.class_name == "red_cannon"

    assert main_window._save_current_image() is True
    saved = yolo_io.load_boxes(labelimg_dataset / "0005.jpg")
    assert len(saved) == 1
    assert saved[0].class_id == main_window._classes.index("red_cannon")

    # An unconfirmed suggestion must never make it into the saved .txt --
    # saving past the "you still have unconfirmed suggestions" warning
    # (answered here as "save anyway") must still exclude it.
    main_window._canvas.add_box_item(QRectF(200, 200, 30, 30), class_name=None, confirmed=False)
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Save)
    )
    main_window._save_current_image()
    saved_again = yolo_io.load_boxes(labelimg_dataset / "0005.jpg")
    assert len(saved_again) == 1


def test_saving_with_unconfirmed_suggestions_warns_first(main_window, labelimg_dataset, monkeypatch):
    # Regression test: silently saving over unconfirmed suggestions used to
    # look like "Save does nothing" (nothing gets written for them) or,
    # worse, overwrite a previously-correct .txt with an empty one if none
    # of the boxes on screen happened to be confirmed yet.
    main_window._open_directory(labelimg_dataset)
    assert main_window._current_image_path.name == "0005.jpg"
    main_window._canvas.add_box_item(QRectF(50, 50, 40, 40), class_name="red_cannon", confirmed=False)

    warned = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: (warned.append(True), QMessageBox.StandardButton.Cancel)[1]),
    )
    result = main_window._save_current_image()

    assert warned  # the user was actually told about the unconfirmed box
    assert result is False
    assert not yolo_io.has_label(labelimg_dataset / "0005.jpg")  # cancelling writes nothing


def test_resume_advances_after_labeling_a_fresh_window(qapp, isolated_settings, labelimg_dataset):
    win1 = MainWindow(settings=isolated_settings)
    win1._open_directory(labelimg_dataset)
    box = win1._canvas.add_box_item(QRectF(10, 10, 20, 20), class_name=None, confirmed=True, select=True)
    _press_letter(win1._canvas, Qt.Key.Key_P, "p")
    assert box.class_name == "black_pawn"
    win1._save_current_image()
    win1.close()
    qapp.removeEventFilter(win1)

    win2 = MainWindow(settings=isolated_settings)
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


def _move_to_trash_by_unlink(path_text: str):
    """Deterministic stand-in for the platform Recycle Bin in test folders."""

    Path(path_text).unlink()
    return True, ""


def test_shift_delete_moves_image_and_both_sidecars_then_selects_next_image(
    main_window, labelimg_dataset, monkeypatch
):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    image_path = labelimg_dataset / "0001.jpg"
    label_path = yolo_io.label_path_for_image(image_path)
    sidecar_path = metadata.metadata_path_for_image(image_path)
    record = metadata.new_metadata(image_path, 640, 480)
    metadata.save_metadata_atomic(image_path, record, expected_image_size=(640, 480))
    assert label_path.exists() and sidecar_path.exists()

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(
        main_window_module.QFile,
        "moveToTrash",
        staticmethod(_move_to_trash_by_unlink),
    )

    QTest.keyClick(main_window._canvas, Qt.Key.Key_Delete, Qt.KeyboardModifier.ShiftModifier)

    assert not image_path.exists()
    assert not label_path.exists()
    assert not sidecar_path.exists()
    assert main_window._current_image_path.name == "0002.jpg"
    assert all(path.name != "0001.jpg" for path in main_window._images)


def test_shift_delete_cancel_keeps_current_image_and_selected_box(main_window, labelimg_dataset, monkeypatch):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    box = main_window._canvas.box_items()[0]
    main_window._canvas.select_box(box)
    image_path = main_window._current_image_path

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.No),
    )
    QTest.keyClick(main_window._canvas, Qt.Key.Key_Delete, Qt.KeyboardModifier.ShiftModifier)

    assert image_path.exists()
    assert main_window._current_image_path == image_path
    assert len(main_window._canvas.box_items()) == 2


def test_delete_only_image_resets_the_active_annotation_state(main_window, tmp_path, monkeypatch):
    dataset = tmp_path / "one-image"
    dataset.mkdir()
    Image.new("RGB", (20, 20), (200, 200, 200)).save(dataset / "only.jpg")
    main_window._open_directory(dataset)

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(
        main_window_module.QFile,
        "moveToTrash",
        staticmethod(_move_to_trash_by_unlink),
    )
    main_window._delete_current_image()

    assert main_window._images == []
    assert main_window._current_image_path is None
    assert main_window._metadata is None
    assert main_window._canvas.image_size() == (0, 0)


def test_duplicate_selected_box(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    box = main_window._canvas.box_items()[0]
    main_window._canvas.select_box(box)

    main_window._duplicate_selected_box()

    assert len(main_window._canvas.box_items()) == 3
