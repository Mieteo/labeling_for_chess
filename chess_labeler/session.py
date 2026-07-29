"""Convenience session state (`.labeling_session.json`), stored inside the
image directory itself so it travels with the folder when copied to another
machine (see labeling_tool_requirements.md section 5).

This file is NEVER the source of truth for labeling progress -- that is
always derived from which images have a sibling `<name>.txt` (see
yolo_io.has_label). If this file is missing, corrupt, or stale, the tool
must still resume correctly by re-deriving the stopping point from the
filesystem.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from . import yolo_io
from .constants import SESSION_FILENAME


@dataclasses.dataclass
class SessionState:
    last_image: str | None = None  # filename only, relative to the image dir
    last_radius_px: float | None = None
    last_tolerance_pct: float | None = None
    # Convenience-only values for the metadata UI. They are never labels,
    # never used to decide progress, and contain only annotator-entered device
    # family / capture-session names (not EXIF, GPS, or personal IDs).
    recent_device_models: list[str] = dataclasses.field(default_factory=list)
    recent_capture_groups: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "last_image": self.last_image,
            "last_radius_px": self.last_radius_px,
            "last_tolerance_pct": self.last_tolerance_pct,
            "recent_device_models": list(self.recent_device_models),
            "recent_capture_groups": list(self.recent_capture_groups),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        return cls(
            last_image=data.get("last_image"),
            last_radius_px=data.get("last_radius_px"),
            last_tolerance_pct=data.get("last_tolerance_pct"),
            recent_device_models=_clean_recent_values(data.get("recent_device_models")),
            recent_capture_groups=_clean_recent_values(data.get("recent_capture_groups")),
        )


def _clean_recent_values(value: object, limit: int = 12) -> list[str]:
    """Recover a small, safe recent-value list from old/corrupt sessions."""
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def add_recent_value(values: list[str], value: str | None, limit: int = 12) -> list[str]:
    """Return an MRU-style recent list without mutating the caller's list."""
    text = (value or "").strip()
    if not text or text == "unknown":
        return _clean_recent_values(values, limit)
    return _clean_recent_values([text, *values], limit)


def session_path(image_dir: Path | str) -> Path:
    return Path(image_dir) / SESSION_FILENAME


def load_session(image_dir: Path | str) -> SessionState:
    """Best-effort load. A missing or corrupt file is not an error -- it
    just means we have no hints, so return an empty state."""
    path = session_path(image_dir)
    if not path.exists():
        return SessionState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return SessionState()
        return SessionState.from_dict(data)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return SessionState()


def save_session(image_dir: Path | str, state: SessionState) -> None:
    path = session_path(image_dir)
    path.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def find_resume_image(
    image_dir: Path | str, images: list[Path] | None = None
) -> Path | None:
    """Image to jump to when opening image_dir: the first (by filename
    order) image that has no `<name>.txt` yet. Derived purely from
    filesystem state, independent of the session file.

    Falls back to the session's last_image hint if every image is already
    labeled, then to the first image, then None for an empty directory.
    """
    image_dir = Path(image_dir)
    if images is None:
        images = yolo_io.list_images(image_dir)
    for img in images:
        if not yolo_io.has_label(img):
            return img
    if not images:
        return None
    state = load_session(image_dir)
    if state.last_image:
        candidate = image_dir / state.last_image
        if candidate.exists():
            for img in images:
                if img.name == state.last_image:
                    return img
    return images[0]
