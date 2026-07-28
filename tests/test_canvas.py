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
    # The label patch can be bigger than a very small box -- must not crash
    # even though it then overflows the box's own rect.
    _ensure_qapp()
    box = BoxItem(QRectF(0, 0, 2, 2), "hand", confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
    _render(box)


def test_bounding_rect_covers_the_label_patch():
    _ensure_qapp()
    box = BoxItem(QRectF(0, 0, 2, 2), "hand", confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
    assert box.boundingRect().contains(box._label_patch_rect())
