"""Sidecar that ferries pending, unconfirmed auto-detect suggestions from a
batch run into the canvas the next time that image is opened -- see
yeu_cau_tu_app_ky_nhan.md section 3.4.

This file is NEVER a source of truth for anything and is deliberately
throwaway: `load_and_consume_suggestions` deletes it as soon as its boxes
have been handed to the canvas. From that point on, an unconfirmed
suggestion behaves exactly like a live circle-detect or single-image
auto-detect suggestion always has in this tool -- if the user navigates away
without confirming and saving it, it is simply gone. Real progress is always
the sibling `<stem>.txt` YOLO file; this sidecar only exists so a batch run
across many images doesn't require the canvas to be open for all of them at
once.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

SUGGESTIONS_SUFFIX = ".autodetect.json"
SCHEMA_VERSION = 1


@dataclasses.dataclass
class PendingBox:
    """A single pending suggestion, normalized to [0, 1] like a YOLO box."""

    class_name: str
    xc: float
    yc: float
    w: float
    h: float
    score: float = 0.0


def suggestions_path_for_image(image_path: Path | str) -> Path:
    return Path(image_path).with_suffix(SUGGESTIONS_SUFFIX)


def save_suggestions(
    image_path: Path | str,
    boxes: list[PendingBox],
    model_path: str,
    conf_threshold: float,
    iou_threshold: float,
) -> None:
    """Atomic write: a crash mid-write must never leave a half-written
    sidecar that a later load could misparse."""
    path = suggestions_path_for_image(image_path)
    data = {
        "schema_version": SCHEMA_VERSION,
        "model_path": model_path,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "boxes": [dataclasses.asdict(b) for b in boxes],
    }
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def load_and_consume_suggestions(image_path: Path | str) -> list[PendingBox]:
    """Load pending suggestions for this image, if any, and delete the
    sidecar -- its only job is to seed the canvas once. A malformed sidecar
    is treated as empty rather than raised, matching how a missing sidecar
    behaves; it is simply skipped and removed."""
    path = suggestions_path_for_image(image_path)
    if not path.exists():
        return []
    boxes: list[PendingBox] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("boxes", []):
            boxes.append(
                PendingBox(
                    class_name=str(entry["class_name"]),
                    xc=float(entry["xc"]),
                    yc=float(entry["yc"]),
                    w=float(entry["w"]),
                    h=float(entry["h"]),
                    score=float(entry.get("score", 0.0)),
                )
            )
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        boxes = []
    try:
        path.unlink()
    except OSError:
        pass
    return boxes


def has_pending_suggestions(image_path: Path | str) -> bool:
    return suggestions_path_for_image(image_path).exists()


def discard_suggestions(image_path: Path | str) -> None:
    try:
        suggestions_path_for_image(image_path).unlink()
    except OSError:
        pass
