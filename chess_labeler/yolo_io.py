"""Flat-directory YOLO label I/O, byte-compatible with labelImg 1.8.6.

This module is the single most safety-critical piece of the tool: see
labeling_tool_requirements.md section 1 ("tuong thich du lieu 100% voi
labelImg"). It intentionally mirrors the on-disk format produced by
``libs/yolo_io.py`` inside the labelImg package (YOLOWriter/YoloReader):

- ``classes.txt``: one class name per line, order = class index. Written
  with '\\n'-only line endings, read back with universal-newline text mode
  (so files created on Windows with '\\r\\n' still parse correctly).
- ``<image>.txt``: one line per box, ``%d %.6f %.6f %.6f %.6f`` (class_id,
  x_center, y_center, width, height, all normalized to [0, 1]), '\\n'-only
  line endings, no header/footer, no extra fields.
- Presence of ``<image>.txt`` (even empty) means "reviewed"; absence means
  "not yet reviewed". This distinction must never collapse.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from .constants import CLASSES_FILENAME, DEFAULT_CLASSES, IMAGE_EXTENSIONS


@dataclasses.dataclass
class Box:
    """A single YOLO-format bounding box, normalized to the image size."""

    class_id: int
    xc: float
    yc: float
    w: float
    h: float

    def clamped(self) -> "Box":
        xc = min(max(self.xc, 0.0), 1.0)
        yc = min(max(self.yc, 0.0), 1.0)
        w = min(max(self.w, 0.0), 1.0)
        h = min(max(self.h, 0.0), 1.0)
        return Box(self.class_id, xc, yc, w, h)

    def to_line(self) -> str:
        return "%d %.6f %.6f %.6f %.6f" % (self.class_id, self.xc, self.yc, self.w, self.h)

    @classmethod
    def from_line(cls, line: str) -> "Box | None":
        stripped = line.strip()
        if not stripped:
            return None
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"malformed YOLO label line: {line!r}")
        class_id = int(parts[0])
        xc, yc, w, h = (float(p) for p in parts[1:])
        return cls(class_id, xc, yc, w, h)

    def to_pixel_rect(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """(left, top, width, height) in pixel coordinates."""
        bw = self.w * img_w
        bh = self.h * img_h
        left = self.xc * img_w - bw / 2
        top = self.yc * img_h - bh / 2
        return left, top, bw, bh

    @classmethod
    def from_pixel_rect(
        cls,
        class_id: int,
        left: float,
        top: float,
        width: float,
        height: float,
        img_w: int,
        img_h: int,
    ) -> "Box":
        xc = (left + width / 2) / img_w
        yc = (top + height / 2) / img_h
        w = width / img_w
        h = height / img_h
        return cls(class_id, xc, yc, w, h).clamped()


def classes_path(image_dir: Path | str) -> Path:
    return Path(image_dir) / CLASSES_FILENAME


def load_or_create_classes(
    image_dir: Path | str, default_classes: list[str] = DEFAULT_CLASSES
) -> list[str]:
    """Load classes.txt if present (preserving its exact existing order),
    otherwise create it from `default_classes` and return that default list.
    Callers pass `constants.DEFAULT_CLASSES_DIGITAL` for a Digital directory
    so a brand-new one gets the 16th `board_region` class from the start
    (see yeu_cau_tu_app_ky_nhan.md section 1); the plain 15-class default
    covers Physical directories.

    Never call this on a directory whose classes.txt you intend to keep
    untouched but do not want re-created -- it only writes when the file is
    absent.
    """
    path = classes_path(image_dir)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        return text.strip("\n").split("\n") if text.strip("\n") else []
    save_classes(image_dir, default_classes)
    return list(default_classes)


def save_classes(image_dir: Path | str, classes: list[str]) -> None:
    """Overwrite classes.txt. Only ever call this for a brand-new directory
    (no pre-existing classes.txt) -- section 6 forbids reordering/rewriting
    an existing one.
    """
    path = classes_path(image_dir)
    content = "\n".join(classes) + "\n"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def class_name(classes: list[str], class_id: int) -> str:
    if 0 <= class_id < len(classes):
        return classes[class_id]
    return f"unknown_class_{class_id}"


def list_images(image_dir: Path | str) -> list[Path]:
    """All .jpg/.jpeg/.png files directly in image_dir, sorted case-
    insensitively by filename (matches labelImg's scan_all_images order)."""
    image_dir = Path(image_dir)
    files = [
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda p: p.name.lower())
    return files


def label_path_for_image(image_path: Path | str) -> Path:
    image_path = Path(image_path)
    return image_path.with_suffix(".txt")


def has_label(image_path: Path | str) -> bool:
    """True iff <name>.txt exists next to the image (reviewed, possibly with
    zero boxes) -- NOT the same as "has at least one box"."""
    return label_path_for_image(image_path).exists()


def load_boxes(image_path: Path | str) -> list[Box]:
    """Boxes saved for this image, or [] if never reviewed (no .txt) or
    reviewed-empty (.txt with zero lines) -- callers needing to distinguish
    "never reviewed" vs "reviewed, empty" should check has_label() first."""
    txt_path = label_path_for_image(image_path)
    if not txt_path.exists():
        return []
    boxes: list[Box] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        box = Box.from_line(line)
        if box is not None:
            boxes.append(box)
    return boxes


def save_boxes(image_path: Path | str, boxes: list[Box]) -> None:
    """Write (possibly empty) box list to <name>.txt. Writing with an empty
    list is the explicit "reviewed, no objects" action -- it must still
    create the file, never skip it."""
    txt_path = label_path_for_image(image_path)
    lines = [box.to_line() for box in boxes]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    with open(txt_path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
