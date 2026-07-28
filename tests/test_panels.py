from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from chess_labeler.canvas import BoxItem
from chess_labeler.panels import BoxListPanel


def _ensure_qapp():
    return QApplication.instance() or QApplication([])


def _box(class_name):
    return BoxItem(QRectF(0, 0, 10, 10), class_name, confirmed=True, image_bounds=QRectF(0, 0, 100, 100))


def test_box_list_groups_black_then_red_then_hand_then_unassigned():
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
    ordered_classes = [b.class_name for b in panel._items]
    assert ordered_classes == ["black_pawn", "black_king", "red_pawn", "red_king", "hand", None]


def test_box_list_keeps_relative_order_within_same_group():
    _ensure_qapp()
    panel = BoxListPanel()
    a = _box("black_rook")
    b = _box("black_rook")
    c = _box(None)
    d = _box(None)
    panel.set_boxes([c, a, d, b])
    assert panel._items == [a, b, c, d]


def test_box_list_unrecognized_class_name_sorts_with_unassigned():
    _ensure_qapp()
    panel = BoxListPanel()
    known = _box("black_king")
    unknown = _box("some_legacy_label")
    panel.set_boxes([known, unknown])
    assert panel._items == [known, unknown]
