"""End-to-end checks for the optional sidecar flow in the real main window."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

from chess_labeler import metadata
from chess_labeler.board_editor import STARTING_BOARD_FEN
from chess_labeler.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def main_window(qapp, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    yield window
    window._dirty = False
    window.close()
    qapp.removeEventFilter(window)


def _index_of(win, name: str) -> int:
    return next(index for index, image in enumerate(win._images) if image.name == name)


def _mark_valid_corners(win) -> None:
    for name, point in (
        ("top_left", QPointF(50, 40)),
        ("top_right", QPointF(590, 45)),
        ("bottom_right", QPointF(600, 440)),
        ("bottom_left", QPointF(40, 435)),
    ):
        win._on_corner_requested(name, point)


def test_new_image_shows_starting_board_scaffold_without_creating_fen(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)

    assert main_window._current_image_path.name == "0005.jpg"
    assert main_window._metadata.board.board_fen is None
    assert main_window._metadata.board.fen_status == "not_started"
    assert main_window._board_editor.board_fen == STARTING_BOARD_FEN
    assert not (labelimg_dataset / "0005.meta.json").exists()


def test_metadata_only_save_preserves_yolo_bytes_and_round_trips_simplified_fields(main_window, labelimg_dataset):
    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    yolo_before = (labelimg_dataset / "0001.txt").read_bytes()

    _mark_valid_corners(main_window)
    main_window._board_editor.set_starting_position()  # no user signal for setup
    main_window._on_board_changed(STARTING_BOARD_FEN, [])

    panel = main_window._metadata_panel
    panel._orientation.setCurrentIndex(panel._orientation.findData("red_at_bottom"))
    panel._capture_controls["lighting"].setCurrentIndex(
        panel._capture_controls["lighting"].findData("even")
    )
    panel._capture_controls["board_material"].setCurrentIndex(
        panel._capture_controls["board_material"].findData("wood")
    )
    panel._notes.setText("Ảnh test")
    main_window._on_metadata_panel_changed()

    assert main_window._boxes_dirty is False
    assert main_window._metadata_dirty is True
    assert main_window._save_current_image() is True

    assert (labelimg_dataset / "0001.txt").read_bytes() == yolo_before
    saved = metadata.load_metadata(labelimg_dataset / "0001.jpg", expected_image_size=(640, 480))
    assert saved is not None
    assert saved.board.corners_status == "human_verified"
    assert saved.board.board_fen == STARTING_BOARD_FEN
    assert saved.board.fen_status == "human_verified"
    assert saved.board.position_complete is True
    assert saved.board.side_to_move is None
    assert saved.board.full_fen is None
    assert saved.review.status == "unreviewed"
    assert saved.review.corners_verified is True
    assert saved.review.fen_verified is True
    assert saved.capture.lighting == "even"
    assert saved.capture.device_model == "unknown"
    assert saved.review.notes == "Ảnh test"

    # Re-opening restores the raw points, the FEN state, and dropdown values.
    main_window._go_to_index(_index_of(main_window, "0002.jpg"))
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    assert main_window._canvas.corner_points()["top_left"] == QPointF(50, 40)
    assert main_window._board_editor.board_fen == STARTING_BOARD_FEN
    assert main_window._metadata_panel._capture_controls["lighting"].currentData() == "even"


def test_confirm_and_clear_fen_turn_the_scaffold_into_an_explicit_value(main_window, labelimg_dataset, monkeypatch):
    main_window._open_directory(labelimg_dataset)
    assert main_window._metadata.board.board_fen is None
    assert main_window._metadata_panel._fen_presence.text() == "Chưa có FEN"

    main_window._metadata_panel.openFenRequested.emit()
    assert main_window._right_tabs.currentWidget().objectName() == "fenTab"

    main_window._confirm_current_board_fen()

    assert main_window._metadata.board.board_fen == STARTING_BOARD_FEN
    assert main_window._metadata.board.side_to_move is None
    assert main_window._metadata.board.full_fen is None
    assert main_window._metadata_panel._fen_presence.text() == "Đã có FEN"

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Yes),
    )
    main_window._clear_current_board_fen()

    assert main_window._metadata.board.board_fen is None
    assert main_window._metadata_panel._fen_presence.text() == "Chưa có FEN"


def test_corrupt_sidecar_is_not_overwritten_without_explicit_confirmation(
    main_window, labelimg_dataset, monkeypatch
):
    target = labelimg_dataset / "0001.meta.json"
    original = b"{not-json"
    target.write_bytes(original)

    main_window._open_directory(labelimg_dataset)
    main_window._go_to_index(_index_of(main_window, "0001.jpg"))
    assert main_window._metadata_load_error is not None
    main_window._on_corner_requested("top_left", QPointF(10, 10))

    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Cancel),
    )
    assert main_window._save_current_image() is False
    assert target.read_bytes() == original
