from __future__ import annotations

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt
from PySide6.QtGui import QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from chess_labeler.board_editor import (
    EMPTY_BOARD_FEN,
    STARTING_BOARD_FEN,
    XiangqiBoard,
    XiangqiBoardEditor,
)


def _ensure_qapp():
    return QApplication.instance() or QApplication([])


def test_starting_position_round_trips_to_the_documented_board_fen():
    board = XiangqiBoard()

    assert board.to_fen() == STARTING_BOARD_FEN
    assert board.validation_issues() == []


def test_parser_accepts_the_board_field_of_a_full_fen_and_rejects_bad_ranks():
    board = XiangqiBoard.from_fen(f"{STARTING_BOARD_FEN} w - - 0 1")

    assert board.fen == STARTING_BOARD_FEN
    with pytest.raises(ValueError, match="10 ranks"):
        XiangqiBoard.from_fen("9/9")
    with pytest.raises(ValueError, match="Invalid FEN token"):
        XiangqiBoard.from_fen("x8/9/9/9/9/9/9/9/9/9")
    with pytest.raises(ValueError, match="expand to 9"):
        XiangqiBoard.from_fen("8/9/9/9/9/9/9/9/9/9")


def test_mutation_operations_move_capture_remove_reset_and_clear():
    board = XiangqiBoard.empty()
    assert board.fen == EMPTY_BOARD_FEN

    assert board.set_piece((0, 0), "r")
    assert board.set_piece((0, 1), "P")
    assert board.move((0, 0), (0, 1))  # captures the red pawn
    assert board.piece_at((0, 0)) is None
    assert board.piece_at((0, 1)) == "r"
    assert board.remove((0, 1)) == "r"
    assert board.remove((0, 1)) is None
    assert board.clear() is False

    assert board.reset()
    assert board.fen == STARTING_BOARD_FEN
    assert board.clear()
    assert board.fen == EMPTY_BOARD_FEN


def test_validation_catches_required_structural_problems():
    board = XiangqiBoard.empty()
    board.set_piece((4, 0), "K")  # red king outside its palace
    board.set_piece((4, 9), "k")
    board.set_piece((0, 1), "B")  # red elephant over the river
    board.set_piece((0, 6), "b")  # black elephant over the river
    board.set_piece((0, 8), "A")  # red advisor outside its palace
    for file in (0, 1, 2, 3, 5, 6):
        board.set_piece((file, 3), "P")  # six pawns, away from the kings' file

    issues = board.validation_issues()

    assert "red_king_outside_palace_4_0" in issues
    assert "red_elephant_crossed_river_0_1" in issues
    assert "black_elephant_crossed_river_0_6" in issues
    assert "red_advisor_outside_palace_0_8" in issues
    assert "too_many_red_pawn_6" in issues
    assert "facing_kings" in issues


def test_programmatic_fen_load_is_clean_but_click_move_emits_a_live_fen():
    app = _ensure_qapp()
    editor = XiangqiBoardEditor()
    editor.resize(420, 620)
    editor.show()
    app.processEvents()
    try:
        emitted: list[tuple[str, list[str]]] = []
        editor.boardChanged.connect(lambda fen, issues: emitted.append((fen, issues)))

        editor.set_board_fen(STARTING_BOARD_FEN)
        assert emitted == []

        # Click a red pawn then its forward intersection.  The view owns the
        # coordinate mapping, so the test stays valid if the dock is resized.
        view = editor._view
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=view.square_center((0, 6)))
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=view.square_center((0, 5)))
        app.processEvents()

        assert len(emitted) == 1
        assert editor.board_fen == emitted[0][0]
        assert editor.board().piece_at((0, 5)) == "P"
        assert editor.board().piece_at((0, 6)) is None

        fen_before_flip = editor.board_fen
        editor._flip_button.click()
        assert editor.is_flipped
        assert editor.board_fen == fen_before_flip
        assert len(emitted) == 1
    finally:
        editor.close()


def test_dragging_a_board_piece_moves_it_and_captures_the_target_occupant():
    app = _ensure_qapp()
    editor = XiangqiBoardEditor()
    editor.resize(420, 620)
    editor.show()
    app.processEvents()
    try:
        editor.set_empty_position()
        editor._place_user((0, 6), "P")
        editor._place_user((0, 5), "c")
        view = editor._view
        QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=view.square_center((0, 6)))
        QTest.mouseMove(view, view.square_center((0, 5)), delay=10)
        QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=view.square_center((0, 5)))
        app.processEvents()

        assert editor.board().piece_at((0, 6)) is None
        assert editor.board().piece_at((0, 5)) == "P"
    finally:
        editor.close()


def test_selected_piece_can_be_removed_with_delete_from_the_board_surface():
    app = _ensure_qapp()
    editor = XiangqiBoardEditor()
    editor.resize(420, 620)
    editor.show()
    app.processEvents()
    try:
        view = editor._view
        source = (0, 6)
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=view.square_center(source))
        QTest.keyClick(view, Qt.Key.Key_Delete)
        app.processEvents()

        assert editor.board().piece_at(source) is None
        assert editor.can_undo
    finally:
        editor.close()


def test_piece_trays_are_visual_and_dropping_a_piece_places_or_replaces_it():
    app = _ensure_qapp()
    editor = XiangqiBoardEditor()
    editor.resize(520, 620)
    editor.show()
    app.processEvents()
    try:
        assert len(editor._palette_buttons) == 14
        assert set(editor._palette_buttons) == set("KABNRCPkabnrcp")
        assert editor._palette_buttons["k"].isCheckable()
        assert editor._palette_buttons["K"].isCheckable()

        editor.set_empty_position()
        view = editor._view
        target = (4, 4)
        mime = QMimeData()
        mime.setData("application/x-xiangqi-piece", b"C")
        drop = QDropEvent(
            QPointF(view.square_center(target)),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        view.dropEvent(drop)
        app.processEvents()
        assert drop.isAccepted()
        assert editor.board().piece_at(target) == "C"

        # A second drop on the same intersection deliberately replaces it.
        mime.setData("application/x-xiangqi-piece", b"p")
        replacement = QDropEvent(
            QPointF(view.square_center(target)),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        view.dropEvent(replacement)
        assert editor.board().piece_at(target) == "p"
        assert editor.can_undo
    finally:
        editor.close()


def test_editor_undo_redo_tracks_board_state_and_keyboard_shortcuts():
    app = _ensure_qapp()
    editor = XiangqiBoardEditor()
    editor.resize(420, 620)
    editor.show()
    app.processEvents()
    try:
        emitted: list[str] = []
        editor.boardChanged.connect(lambda fen, _issues: emitted.append(fen))
        initial = editor.board_fen

        # Use the same slot connected to the board surface: this is a user
        # mutation, unlike set_board_fen which deliberately resets history.
        editor._move_user((0, 6), (0, 5))
        moved = editor.board_fen
        assert moved != initial
        assert editor.is_dirty
        assert editor.can_undo and not editor.can_redo
        assert editor._undo_button.isEnabled()

        editor.setFocus()
        QTest.keyClick(editor, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        app.processEvents()
        assert editor.board_fen == initial
        assert not editor.is_dirty
        assert not editor.can_undo and editor.can_redo

        QTest.keyClick(
            editor,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        app.processEvents()
        assert editor.board_fen == moved
        assert editor.can_undo and not editor.can_redo
        assert emitted == [moved, initial, moved]
    finally:
        editor.close()


def test_copy_fen_and_destructive_actions_respect_confirmation_when_dirty():
    app = _ensure_qapp()
    editor = XiangqiBoardEditor()
    editor.resize(420, 620)
    editor.show()
    app.processEvents()
    try:
        editor._move_user((0, 6), (0, 5))
        edited_fen = editor.board_fen
        assert editor.copy_board_fen() == edited_fen
        assert QApplication.clipboard().text() == edited_fen

        confirmations: list[str] = []
        editor._confirm_destructive_change = lambda action: confirmations.append(action) or False
        editor._empty_button.click()
        assert editor.board_fen == edited_fen
        assert confirmations == ["Xóa toàn bộ quân trên bàn"]

        editor._confirm_destructive_change = lambda action: confirmations.append(action) or True
        editor._empty_button.click()
        assert editor.board_fen == EMPTY_BOARD_FEN
        assert editor.can_undo

        # Empty is a normal history state, so undo restores the actual board,
        # not merely a visual preview.
        assert editor.undo()
        assert editor.board_fen == edited_fen

        # A newly loaded FEN is the clean baseline: Reset changes it without
        # an unnecessary confirmation prompt.
        editor.set_board_fen(EMPTY_BOARD_FEN)
        editor._confirm_destructive_change = lambda _action: pytest.fail("unexpected confirmation")
        editor._reset_button.click()
        assert editor.board_fen == STARTING_BOARD_FEN
    finally:
        editor.close()
