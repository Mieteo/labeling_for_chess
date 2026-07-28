from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from chess_labeler.canvas import BoxItem
from chess_labeler.panels import BoxListPanel


def _ensure_qapp():
    return QApplication.instance() or QApplication([])


def _box(class_name):
    return BoxItem(QRectF(0, 0, 10, 10), class_name, confirmed=True, image_bounds=QRectF(0, 0, 100, 100))


def test_black_column_has_black_pieces_then_unassigned():
    _ensure_qapp()
    panel = BoxListPanel()
    boxes = [
        _box("hand"),
        _box(None),
        _box("red_king"),
        _box("black_pawn"),
        _box("red_pawn"),
        _box("black_king"),
    ]
    panel.set_boxes(boxes)
    assert [b.class_name for b in panel._black_items] == ["black_pawn", "black_king", None]


def test_red_column_has_red_pieces_then_hand():
    _ensure_qapp()
    panel = BoxListPanel()
    boxes = [
        _box("hand"),
        _box(None),
        _box("red_king"),
        _box("black_pawn"),
        _box("red_pawn"),
        _box("black_king"),
    ]
    panel.set_boxes(boxes)
    assert [b.class_name for b in panel._red_items] == ["red_pawn", "red_king", "hand"]


def test_columns_keep_relative_order_within_same_group():
    _ensure_qapp()
    panel = BoxListPanel()
    a = _box("black_rook")
    b = _box("black_rook")
    c = _box(None)
    d = _box(None)
    panel.set_boxes([c, a, d, b])
    assert panel._black_items == [a, b, c, d]


def test_unrecognized_class_name_sorts_into_black_column_with_unassigned():
    _ensure_qapp()
    panel = BoxListPanel()
    known = _box("black_king")
    unknown = _box("some_legacy_label")
    panel.set_boxes([known, unknown])
    assert panel._black_items == [known, unknown]
    assert panel._red_items == []


def test_select_box_highlights_correct_column_and_clears_the_other():
    _ensure_qapp()
    panel = BoxListPanel()
    black_box = _box("black_pawn")
    red_box = _box("red_pawn")
    panel.set_boxes([black_box, red_box])

    panel.select_box(red_box)
    assert panel._red_list.currentRow() == panel._red_items.index(red_box)
    assert panel._black_list.currentRow() == -1

    panel.select_box(black_box)
    assert panel._black_list.currentRow() == panel._black_items.index(black_box)
    assert panel._red_list.currentRow() == -1


def test_clicking_an_item_emits_the_right_box_from_either_column():
    _ensure_qapp()
    panel = BoxListPanel()
    black_box = _box("black_pawn")
    red_box = _box("red_pawn")
    panel.set_boxes([black_box, red_box])

    emitted = []
    panel.boxActivated.connect(emitted.append)

    panel._on_clicked(panel._red_list, panel._red_list.item(0))
    assert emitted == [red_box]

    panel._on_clicked(panel._black_list, panel._black_list.item(0))
    assert emitted == [red_box, black_box]
