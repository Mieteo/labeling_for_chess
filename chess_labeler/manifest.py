"""Export a portable micro-gold manifest from image metadata sidecars.

This module deliberately only reads the label directory.  It does not infer
content cohorts, mutate sidecars, select scanner routes, or touch YOLO files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import metadata, yolo_io


MANIFEST_SCHEMA_VERSION = 1
UNASSIGNED_COHORT = "unassigned"


def content_cohort_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count explicit cohorts plus the distinct unassigned bucket."""

    counts = {
        UNASSIGNED_COHORT: 0,
        **{cohort: 0 for cohort in sorted(metadata.CONTENT_COHORTS)},
    }
    for record in records:
        cohort = record.get("content_cohort")
        counts[cohort if cohort is not None else UNASSIGNED_COHORT] += 1
    return counts


def manifest_record(image_path: Path) -> dict[str, Any]:
    """Build one record; invalid sidecars fail loudly instead of being hidden."""

    record = metadata.load_metadata(image_path)
    if record is None:
        return {
            "image": {"filename": image_path.name},
            "board_fen": None,
            "corners_px": None,
            "content_cohort": None,
        }

    capture = record.capture
    exported: dict[str, Any] = {
        "image": record.image.to_dict(),
        "board_fen": record.board.board_fen,
        "corners_px": record.board.to_dict()["corners_px"],
        "content_cohort": capture.content_cohort,
    }
    for field in ("style_or_app", "capture_session", "position_id", "capture_group"):
        value = getattr(capture, field)
        if value is not None:
            exported[field] = value
    return exported


def build_manifest(image_dir: Path | str) -> dict[str, Any]:
    """Return all directory images and cohort counts in stable filename order."""

    directory = Path(image_dir)
    records = [manifest_record(image_path) for image_path in yolo_io.list_images(directory)]
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "images": records,
        "content_cohort_counts": content_cohort_counts(records),
    }


def export_manifest(image_dir: Path | str, output_path: Path | str) -> Path:
    """Write the manifest as UTF-8 JSON, creating only the requested output."""

    target = Path(output_path)
    target.write_text(
        json.dumps(build_manifest(image_dir), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Xiangqi micro-gold metadata manifest")
    parser.add_argument("image_dir", type=Path, help="Flat directory containing image sidecars")
    parser.add_argument("output", type=Path, help="Output JSON manifest path")
    args = parser.parse_args(argv)
    output = export_manifest(args.image_dir, args.output)
    print(f"Đã export manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
