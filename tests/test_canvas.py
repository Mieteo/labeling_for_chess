from PySide6.QtCore import QRectF
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from chess_labeler.canvas import BoxItem


def _ensure_qapp():
    return QApplication.instance() or QApplication([])


def _render(box: BoxItem) -> None:
    pixmap = QPixmap(200, 200)
    pixmap.fill()
    painter = QPainter(pixmap)
    try:
        box.paint(painter, None)
    finally:
        painter.end()


def test_paint_does_not_crash_for_labeled_and_unlabeled_boxes():
    _ensure_qapp()
    for class_name in (None, "black_pawn", "red_king", "hand", "unknown_legacy_label"):
        box = BoxItem(QRectF(0, 0, 40, 40), class_name, confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
        _render(box)  # must not raise


def test_paint_does_not_crash_for_a_tiny_box():
    # The label text can be bigger than a very small box -- must not crash
    # even though it then overflows the box's own rect.
    _ensure_qapp()
    box = BoxItem(QRectF(0, 0, 2, 2), "hand", confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
    _render(box)


def test_bounding_rect_covers_the_label_text():
    _ensure_qapp()
    box = BoxItem(QRectF(0, 0, 2, 2), "hand", confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
    assert box.boundingRect().contains(box._label_text_rect())


def test_label_text_sits_inside_the_box_at_the_top_left_corner():
    _ensure_qapp()
    box = BoxItem(QRectF(50, 60, 40, 40), "black_pawn", confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
    text_rect = box._label_text_rect()
    top_left = box.rect().topLeft()
    assert text_rect.left() >= top_left.x()
    assert text_rect.top() >= top_left.y()
    assert text_rect.width() > 0
    assert text_rect.height() > 0
    # Inside the box's own interior, not overlapping/outside it.
    assert box.rect().contains(text_rect)


def test_label_text_rect_is_empty_for_an_unlabeled_box():
    _ensure_qapp()
    box = BoxItem(QRectF(50, 60, 40, 40), None, confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
    assert box._label_text_rect().isEmpty()


def test_paint_and_bounding_rect_do_not_crash_for_a_box_glued_to_the_image_corner():
    # A piece flush against the board edge -- the label then sits partly
    # (or entirely) outside the image, but must never raise.
    _ensure_qapp()
    for rect in (QRectF(0, 0, 40, 40), QRectF(0, 0, 1, 1), QRectF(160, 160, 40, 40)):
        box = BoxItem(rect, "red_king", confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
        bounding = box.boundingRect()
        assert bounding.width() > 0 and bounding.height() > 0
        _render(box)
