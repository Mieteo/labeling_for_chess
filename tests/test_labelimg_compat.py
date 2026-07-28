"""Acceptance criterion #1 (labeling_tool_requirements.md section 8):
opening a directory partially labeled by real labelImg must show the exact
same boxes/classes/positions, and saving one image must not touch the
others.
"""

from __future__ import annotations

from pathlib import Path

from chess_labeler import session, yolo_io
from chess_labeler.constants import DEFAULT_CLASSES


def test_classes_txt_matches_labelimg_order_despite_crlf(labelimg_dataset: Path):
    classes = yolo_io.load_or_create_classes(labelimg_dataset)
    assert classes == DEFAULT_CLASSES


def test_list_images_includes_all_six(labelimg_dataset: Path):
    names = [p.name for p in yolo_io.list_images(labelimg_dataset)]
    assert names == ["0001.jpg", "0002.jpg", "0003.jpg", "0004.jpg", "0005.jpg", "0006.jpg"]


def test_0001_boxes_match_exactly(labelimg_dataset: Path):
    boxes = yolo_io.load_boxes(labelimg_dataset / "0001.jpg")
    assert len(boxes) == 2

    b0 = boxes[0]
    assert b0.class_id == 0  # red_king
    left, top, w, h = b0.to_pixel_rect(640, 480)
    assert abs(left - 80) < 0.5 and abs(top - 80) < 0.5
    assert abs(w - 40) < 0.5 and abs(h - 40) < 0.5

    b1 = boxes[1]
    assert b1.class_id == 13  # black_pawn
    left, top, w, h = b1.to_pixel_rect(640, 480)
    assert abs(left - 282) < 0.5 and abs(top - 132) < 0.5


def test_0003_hand_box(labelimg_dataset: Path):
    boxes = yolo_io.load_boxes(labelimg_dataset / "0003.jpg")
    assert len(boxes) == 1
    assert boxes[0].class_id == 14  # hand


def test_0004_is_reviewed_but_empty(labelimg_dataset: Path):
    assert yolo_io.has_label(labelimg_dataset / "0004.jpg") is True
    assert yolo_io.load_boxes(labelimg_dataset / "0004.jpg") == []


def test_0005_0006_are_unreviewed(labelimg_dataset: Path):
    assert yolo_io.has_label(labelimg_dataset / "0005.jpg") is False
    assert yolo_io.has_label(labelimg_dataset / "0006.jpg") is False


def test_resume_point_is_0005(labelimg_dataset: Path):
    resume = session.find_resume_image(labelimg_dataset)
    assert resume.name == "0005.jpg"


def test_editing_one_image_does_not_touch_others_byte_for_byte(labelimg_dataset: Path):
    other_files = {
        "0002.txt": (labelimg_dataset / "0002.txt").read_bytes(),
        "0003.txt": (labelimg_dataset / "0003.txt").read_bytes(),
        "0004.txt": (labelimg_dataset / "0004.txt").read_bytes(),
        "classes.txt": (labelimg_dataset / "classes.txt").read_bytes(),
    }

    # Re-save 0001 with one extra box (simulating a real edit session).
    boxes = yolo_io.load_boxes(labelimg_dataset / "0001.jpg")
    boxes.append(yolo_io.Box(4, 0.5, 0.5, 0.05, 0.05))  # red_cannon
    yolo_io.save_boxes(labelimg_dataset / "0001.jpg", boxes)

    for name, original_bytes in other_files.items():
        assert (labelimg_dataset / name).read_bytes() == original_bytes, f"{name} was modified!"

    reloaded = yolo_io.load_boxes(labelimg_dataset / "0001.jpg")
    assert len(reloaded) == 3
    assert reloaded[2].class_id == 4


def test_saved_txt_is_still_parseable_as_lf_only(labelimg_dataset: Path):
    yolo_io.save_boxes(labelimg_dataset / "0005.jpg", [yolo_io.Box(6, 0.5, 0.5, 0.1, 0.1)])
    raw = (labelimg_dataset / "0005.txt").read_bytes()
    assert b"\r\n" not in raw
    assert yolo_io.load_boxes(labelimg_dataset / "0005.jpg") == [yolo_io.Box(6, 0.5, 0.5, 0.1, 0.1)]
