from __future__ import annotations

import json

from chess_labeler import metadata
from chess_labeler.manifest import build_manifest, export_manifest


def test_manifest_exports_cohorts_optional_fields_and_counts(tmp_path):
    first = tmp_path / "001.jpg"
    second = tmp_path / "002.jpg"
    first.write_bytes(b"")
    second.write_bytes(b"")
    record = metadata.new_metadata(first, 10, 10)
    record.capture.content_cohort = "screen_photo"
    record.capture.style_or_app = "Xiangqi app A"
    record.capture.capture_session = "pixel-9"
    record.capture.position_id = "fen-001"
    metadata.save_metadata_atomic(first, record, expected_image_size=(10, 10))

    manifest = build_manifest(tmp_path)

    assert manifest["content_cohort_counts"]["screen_photo"] == 1
    assert manifest["content_cohort_counts"]["unassigned"] == 1
    assert manifest["content_cohort_counts"]["unknown"] == 0
    assert manifest["images"][0]["content_cohort"] == "screen_photo"
    assert manifest["images"][0]["style_or_app"] == "Xiangqi app A"
    assert manifest["images"][1]["content_cohort"] is None

    output = export_manifest(tmp_path, tmp_path / "manifest.json")
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
