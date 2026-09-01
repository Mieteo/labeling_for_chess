from pathlib import Path

import pytest

from chess_labeler import yolo_io
from chess_labeler.constants import DEFAULT_CLASSES, DEFAULT_CLASSES_DIGITAL


def test_load_or_create_classes_creates_default(tmp_path: Path):
    classes = yolo_io.load_or_create_classes(tmp_path)
    assert classes == DEFAULT_CLASSES
    assert (tmp_path / "classes.txt").exists()


def test_load_or_create_classes_creates_digital_default_with_board_region(tmp_path: Path):
    classes = yolo_io.load_or_create_classes(tmp_path, DEFAULT_CLASSES_DIGITAL)
    assert classes == DEFAULT_CLASSES_DIGITAL
    assert len(classes) == 16
    assert classes[:15] == DEFAULT_CLASSES
    assert classes[15] == "board_region"
    on_disk = (tmp_path / "classes.txt").read_text(encoding="utf-8").strip("\n").split("\n")
    assert on_disk == DEFAULT_CLASSES_DIGITAL


def test_load_or_create_classes_preserves_existing_order(tmp_path: Path):
    custom = ["foo", "bar", "hand"]
    (tmp_path / "classes.txt").write_text("\n".join(custom) + "\n", encoding="utf-8")
    loaded = yolo_io.load_or_create_classes(tmp_path)
    assert loaded == custom  # must NOT be reordered/replaced with defaults


def test_load_or_create_classes_reads_crlf_from_real_labelimg(tmp_path: Path):
    # labelImg writes classes.txt via Python text mode -> CRLF on Windows.
    (tmp_path / "classes.txt").write_bytes(
        ("\r\n".join(DEFAULT_CLASSES) + "\r\n").encode("utf-8")
    )
    loaded = yolo_io.load_or_create_classes(tmp_path)
    assert loaded == DEFAULT_CLASSES
    assert all("\r" not in c for c in loaded)


def test_box_line_roundtrip():
    line = "4 0.512300 0.489900 0.041200 0.061500"
    box = yolo_io.Box.from_line(line)
    assert box == yolo_io.Box(4, 0.5123, 0.4899, 0.0412, 0.0615)
    assert box.to_line() == line


def test_box_from_line_blank_returns_none():
    assert yolo_io.Box.from_line("") is None
    assert yolo_io.Box.from_line("   \n") is None


def test_box_from_line_malformed_raises():
    with pytest.raises(ValueError):
        yolo_io.Box.from_line("1 2 3")


def test_pixel_rect_roundtrip():
    box = yolo_io.Box.from_pixel_rect(4, 100, 200, 50, 60, img_w=1000, img_h=800)
    left, top, w, h = box.to_pixel_rect(1000, 800)
    assert left == pytest.approx(100, abs=1e-6)
    assert top == pytest.approx(200, abs=1e-6)
    assert w == pytest.approx(50, abs=1e-6)
    assert h == pytest.approx(60, abs=1e-6)


def test_pixel_rect_clamps_out_of_bounds():
    box = yolo_io.Box.from_pixel_rect(0, -10, -10, 50, 50, img_w=1000, img_h=800)
    assert 0.0 <= box.xc <= 1.0
    assert 0.0 <= box.yc <= 1.0
    assert 0.0 <= box.w <= 1.0
    assert 0.0 <= box.h <= 1.0


def test_list_images_sorted_case_insensitive(tmp_path: Path):
    names = ["0003.jpg", "0001.png", "B.jpeg", "a.jpg", "0002.JPG"]
    for n in names:
        (tmp_path / n).write_bytes(b"fake")
    (tmp_path / "classes.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignore me", encoding="utf-8")

    listed = [p.name for p in yolo_io.list_images(tmp_path)]
    assert listed == sorted(names, key=str.lower)


def test_has_label_distinguishes_missing_vs_empty(tmp_path: Path):
    img = tmp_path / "0001.jpg"
    img.write_bytes(b"fake")
    assert yolo_io.has_label(img) is False

    yolo_io.save_boxes(img, [])
    assert yolo_io.has_label(img) is True
    assert yolo_io.load_boxes(img) == []


def test_save_boxes_only_touches_target_file(tmp_path: Path):
    img_a = tmp_path / "0001.jpg"
    img_b = tmp_path / "0002.jpg"
    img_a.write_bytes(b"fake")
    img_b.write_bytes(b"fake")

    box = yolo_io.Box(0, 0.5, 0.5, 0.1, 0.1)
    yolo_io.save_boxes(img_a, [box])
    yolo_io.save_boxes(img_b, [box])

    before = (tmp_path / "0002.txt").read_text(encoding="utf-8")
    # Re-saving image A must not rewrite image B's label file.
    yolo_io.save_boxes(img_a, [box, box])
    after = (tmp_path / "0002.txt").read_text(encoding="utf-8")
    assert before == after


def test_save_boxes_writes_lf_only(tmp_path: Path):
    img = tmp_path / "0001.jpg"
    img.write_bytes(b"fake")
    yolo_io.save_boxes(img, [yolo_io.Box(0, 0.5, 0.5, 0.1, 0.1)])
    raw = (tmp_path / "0001.txt").read_bytes()
    assert b"\r\n" not in raw


def test_class_name_out_of_range_is_safe():
    assert yolo_io.class_name(["a", "b"], 0) == "a"
    assert yolo_io.class_name(["a", "b"], 5) == "unknown_class_5"
