"""Shared fixture: a small directory that mimics real labelImg 1.8.6 output,
used by the backward-compatibility acceptance tests (see
labeling_tool_requirements.md section 8, item 1).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from chess_labeler.constants import DEFAULT_CLASSES

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

IMG_W, IMG_H = 640, 480


def _yolo_line(class_id: int, cx_px: float, cy_px: float, w_px: float, h_px: float) -> str:
    xc = cx_px / IMG_W
    yc = cy_px / IMG_H
    w = w_px / IMG_W
    h = h_px / IMG_H
    return "%d %.6f %.6f %.6f %.6f" % (class_id, xc, yc, w, h)


def _write_lf(path: Path, lines: list[str]) -> None:
    """Mimic labelImg's YOLOWriter, which writes via codecs.open (LF-only,
    even on Windows) -- see libs/yolo_io.py in the installed labelImg
    package."""
    content = "".join(line + "\n" for line in lines)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def _plain_image(color=(200, 200, 200)) -> Image.Image:
    return Image.new("RGB", (IMG_W, IMG_H), color)


@pytest.fixture
def labelimg_dataset(tmp_path: Path) -> Path:
    d = tmp_path / "chessImg"
    d.mkdir()

    # classes.txt exactly as real labelImg would emit on Windows (Python
    # text-mode write -> CRLF line endings).
    classes_bytes = ("\r\n".join(DEFAULT_CLASSES) + "\r\n").encode("utf-8")
    (d / "classes.txt").write_bytes(classes_bytes)

    # 0001.jpg: two labeled pieces (red_king, black_pawn).
    img1 = _plain_image()
    draw = ImageDraw.Draw(img1)
    draw.ellipse((80, 80, 120, 120), fill=(180, 30, 30))  # circle1 center (100,100) r=20
    draw.ellipse((282, 132, 318, 168), fill=(20, 20, 20))  # circle2 center (300,150) r=18
    img1.save(d / "0001.jpg", quality=95)
    _write_lf(
        d / "0001.txt",
        [
            _yolo_line(0, 100, 100, 40, 40),  # red_king
            _yolo_line(13, 300, 150, 36, 36),  # black_pawn
        ],
    )

    # 0002.jpg: one labeled piece (black_cannon). Used as the "must stay
    # byte-identical after saving a different image" control file.
    img2 = _plain_image()
    ImageDraw.Draw(img2).ellipse((200, 200, 236, 236), fill=(20, 20, 20))
    img2.save(d / "0002.jpg", quality=95)
    _write_lf(d / "0002.txt", [_yolo_line(11, 218, 218, 36, 36)])  # black_cannon

    # 0003.jpg: a "hand" occlusion box.
    img3 = _plain_image()
    img3.save(d / "0003.jpg", quality=95)
    _write_lf(d / "0003.txt", [_yolo_line(14, 320, 240, 100, 80)])  # hand

    # 0004.jpg: reviewed, explicitly zero objects (empty .txt must exist).
    img4 = _plain_image()
    img4.save(d / "0004.jpg", quality=95)
    (d / "0004.txt").write_text("", encoding="utf-8")

    # 0005.jpg, 0006.jpg: never reviewed -- no sibling .txt at all.
    _plain_image().save(d / "0005.jpg", quality=95)
    _plain_image().save(d / "0006.jpg", quality=95)

    return d


CIRCLE_TEST_CENTERS_RADII = [
    (150, 150, 30),
    (400, 150, 30),
    (650, 150, 30),
    (275, 400, 30),
    (525, 400, 30),
]


@pytest.fixture
def circle_test_image(tmp_path: Path) -> Path:
    """A high-contrast PNG with clearly separated filled circles, for
    exercising the Hough circle-detect assist (no JPEG compression
    artifacts to keep this deterministic). Circle centers/radii are in
    CIRCLE_TEST_CENTERS_RADII."""
    img = Image.new("RGB", (800, 600), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    for cx, cy, r in CIRCLE_TEST_CENTERS_RADII:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(30, 30, 30))
    path = tmp_path / "circles.png"
    img.save(path)
    return path
