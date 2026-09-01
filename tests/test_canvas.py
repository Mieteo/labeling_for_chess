from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QKeyEvent, QPainter, QPixmap, QTransform
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from chess_labeler.canvas import BoxItem, ImageCanvas


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


def test_corner_hotkeys_store_raw_image_coordinates_and_overlay_tracks_zoom():
    app = _ensure_qapp()
    canvas = ImageCanvas()
    pixmap = QPixmap(100, 80)
    pixmap.fill()
    canvas.load_image(pixmap)
    observed: list[tuple[str, QPointF]] = []
    canvas.cornerRequested.connect(lambda name, point: observed.append((name, point)))

    # The canvas receives a scene/image coordinate, independent of the view
    # scale.  This is what metadata persistence must use as its canonical
    # ground-truth coordinate system.
    canvas._last_scene_pos = QPointF(37.25, 41.5)
    app.sendEvent(canvas, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier, "1"))
    assert observed == [("top_left", QPointF(37.25, 41.5))]

    canvas.set_corner_points({"top_left": (37.25, 41.5), "bottom_right": (90.0, 70.0)})
    before = canvas._corner_items["top_left"].rect().width()
    canvas.set_zoom(3.0)
    after = canvas._corner_items["top_left"].rect().width()
    assert canvas.corner_points()["top_left"] == QPointF(37.25, 41.5)
    assert after < before  # remains a three-screen-pixel-radius marker


def test_corner_hotkey_maps_the_actual_pointer_through_zoom_and_pan():
    app = _ensure_qapp()
    canvas = ImageCanvas()
    canvas.resize(280, 200)
    pixmap = QPixmap(800, 600)
    pixmap.fill()
    canvas.load_image(pixmap)
    canvas.show()
    app.processEvents()
    try:
        target = QPointF(631.0, 417.0)
        canvas.set_zoom(2.5)
        canvas.centerOn(target)  # a non-default scroll/pan position
        app.processEvents()

        observed: list[QPointF] = []
        canvas.cornerRequested.connect(lambda _name, point: observed.append(point))
        QTest.mouseMove(canvas.viewport(), canvas.mapFromScene(target))
        app.sendEvent(canvas, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_3, Qt.KeyboardModifier.NoModifier, "3"))

        assert len(observed) == 1
        # A viewport mouse event is integer-pixel based; after 2.5x zoom its
        # inverse mapping can differ by a fractional source pixel, well below
        # the one-image-pixel acceptance tolerance.
        assert abs(observed[0].x() - target.x()) <= 1.0
        assert abs(observed[0].y() - target.y()) <= 1.0
    finally:
        canvas.close()


# ---------------------------------------------------------------------
# board_region z-ordering: it spans nearly the whole board and overlaps
# every piece box, so it must never steal a click meant for a piece
# underneath it (yeu_cau_tu_app_ky_nhan.md section 1).
# ---------------------------------------------------------------------
def test_board_region_box_defaults_behind_piece_boxes():
    _ensure_qapp()
    board = BoxItem(QRectF(0, 0, 200, 200), "board_region", confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
    piece = BoxItem(QRectF(50, 50, 20, 20), "black_pawn", confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
    assert board.zValue() < piece.zValue()


def test_board_region_box_moves_in_front_only_while_selected():
    # Selected board_region still needs its own resize handles reachable,
    # even where they sit under a piece box -- so it comes forward while
    # selected, and drops back behind pieces the instant it's deselected.
    _ensure_qapp()
    board = BoxItem(QRectF(0, 0, 200, 200), "board_region", confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
    piece = BoxItem(QRectF(50, 50, 20, 20), "black_pawn", confirmed=True, image_bounds=QRectF(0, 0, 200, 200))

    board.setSelected(True)
    assert board.zValue() > piece.zValue()

    board.setSelected(False)
    assert board.zValue() < piece.zValue()


def test_reassigning_a_box_to_or_from_board_region_updates_its_z_value():
    _ensure_qapp()
    box = BoxItem(QRectF(0, 0, 40, 40), "black_pawn", confirmed=True, image_bounds=QRectF(0, 0, 200, 200))
    default_z = box.zValue()

    box.class_name = "board_region"
    assert box.zValue() < default_z

    box.class_name = "black_pawn"
    assert box.zValue() == default_z


def test_clicking_a_piece_under_the_board_region_box_hits_the_piece_not_the_region():
    # Regression test for the reported bug: a giant unselected board_region
    # box covering the whole board intercepted clicks meant for the piece
    # underneath it.
    _ensure_qapp()
    canvas = ImageCanvas()
    pixmap = QPixmap(200, 200)
    pixmap.fill()
    canvas.load_image(pixmap)

    canvas.add_box_item(QRectF(0, 0, 200, 200), class_name="board_region", confirmed=True)
    piece = canvas.add_box_item(QRectF(80, 80, 20, 20), class_name="black_pawn", confirmed=True)

    hit = canvas._scene.itemAt(QPointF(90, 90), QTransform())
    assert hit is piece


def test_corner_hotkey_outside_the_source_image_is_ignored_and_zero_requests_clear():
    app = _ensure_qapp()
    canvas = ImageCanvas()
    pixmap = QPixmap(100, 80)
    pixmap.fill()
    canvas.load_image(pixmap)
    observed: list[str] = []
    cleared: list[bool] = []
    canvas.cornerRequested.connect(lambda name, _point: observed.append(name))
    canvas.clearCornersRequested.connect(lambda: cleared.append(True))

    canvas._last_scene_pos = QPointF(101.0, 10.0)
    app.sendEvent(canvas, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_2, Qt.KeyboardModifier.NoModifier, "2"))
    app.sendEvent(canvas, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_0, Qt.KeyboardModifier.NoModifier, "0"))
    assert observed == []
    assert cleared == [True]
