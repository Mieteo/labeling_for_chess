"""Focused tests for the versioned metadata sidecar backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_labeler import metadata


IMAGE_SIZE = (640, 480)
STARTING_BOARD_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"


def _valid_corners() -> dict[str, metadata.Point]:
    return {
        "top_left": metadata.Point(50.0, 40.0),
        "top_right": metadata.Point(590.0, 45.0),
        "bottom_right": metadata.Point(600.0, 440.0),
        "bottom_left": metadata.Point(40.0, 435.0),
    }


def _complete_metadata(image_path: Path) -> metadata.ImageMetadata:
    result = metadata.new_metadata(image_path, *IMAGE_SIZE)
    result.board.corners_px = _valid_corners()
    result.board.corners_status = "human_verified"
    result.board.image_orientation = "red_at_bottom"
    result.board.position_complete = True
    result.board.board_fen = STARTING_BOARD_FEN
    result.board.fen_status = "human_verified"
    result.capture.lighting = "even"
    result.capture.board_material = "wood"
    result.capture.device_model = "Redmi Note 11"
    result.review.corners_verified = True
    result.review.fen_verified = True
    result.review.status = "gold_verified"
    return result


def test_new_metadata_has_safe_unannotated_defaults(tmp_path: Path):
    image_path = tmp_path / "0105.jpg"
    result = metadata.new_metadata(image_path, *IMAGE_SIZE)

    assert result.schema_version == 1
    assert result.image == metadata.ImageFingerprint("0105.jpg", 640, 480)
    assert result.board.corners_px == {name: None for name in metadata.CORNER_NAMES}
    assert result.board.corners_status == "unmarked"
    assert result.board.board_fen is None
    assert result.board.side_to_move is None
    assert result.board.full_fen is None
    assert result.capture.lighting == "unknown"
    assert result.capture.device_model == "unknown"
    assert result.review.status == "unreviewed"


def test_new_metadata_defaults_capture_group_to_own_image(tmp_path: Path):
    image_path = tmp_path / "0007.jpg"
    result = metadata.new_metadata(image_path, *IMAGE_SIZE)

    assert result.capture.capture_group == "M0007"


def test_metadata_path_uses_stem_not_image_extension(tmp_path: Path):
    assert metadata.metadata_path_for_image(tmp_path / "board.photo.JPEG").name == "board.photo.meta.json"


def test_atomic_roundtrip_preserves_yolo_file_byte_for_byte(labelimg_dataset: Path):
    image_path = labelimg_dataset / "0001.jpg"
    yolo_before = (labelimg_dataset / "0001.txt").read_bytes()
    source = _complete_metadata(image_path)

    target = metadata.save_metadata_atomic(
        image_path, source, expected_image_size=IMAGE_SIZE
    )
    loaded = metadata.load_metadata(image_path, expected_image_size=IMAGE_SIZE)

    assert target == labelimg_dataset / "0001.meta.json"
    assert loaded == source
    assert (labelimg_dataset / "0001.txt").read_bytes() == yolo_before
    assert json.loads(target.read_text(encoding="utf-8"))["image"]["filename"] == "0001.jpg"
    assert not list(labelimg_dataset.glob(".0001.meta.json.*.tmp"))


def test_missing_sidecar_is_normal_and_safe_result_has_no_error(tmp_path: Path):
    image_path = tmp_path / "0105.jpg"

    assert metadata.load_metadata(image_path, expected_image_size=IMAGE_SIZE) is None
    result = metadata.try_load_metadata(image_path, expected_image_size=IMAGE_SIZE)
    assert result.metadata is None
    assert result.found is False
    assert result.error is None


def test_corrupt_json_is_reported_without_overwriting_it(tmp_path: Path):
    image_path = tmp_path / "0105.jpg"
    sidecar = metadata.metadata_path_for_image(image_path)
    original = b"{bad json"
    sidecar.write_bytes(original)

    with pytest.raises(metadata.MetadataDecodeError):
        metadata.load_metadata(image_path, expected_image_size=IMAGE_SIZE)
    result = metadata.try_load_metadata(image_path, expected_image_size=IMAGE_SIZE)

    assert result.metadata is None
    assert result.found is True
    assert isinstance(result.error, metadata.MetadataDecodeError)
    assert sidecar.read_bytes() == original


def test_filename_and_dimension_fingerprint_mismatches_are_rejected(tmp_path: Path):
    image_path = tmp_path / "0105.jpg"
    source = metadata.new_metadata(image_path, *IMAGE_SIZE)
    metadata.save_metadata_atomic(image_path, source, expected_image_size=IMAGE_SIZE)

    with pytest.raises(metadata.MetadataImageMismatchError, match="dimensions"):
        metadata.load_metadata(image_path, expected_image_size=(1280, 720))

    payload = json.loads(metadata.metadata_path_for_image(image_path).read_text(encoding="utf-8"))
    payload["image"]["filename"] = "wrong.jpg"
    metadata.metadata_path_for_image(image_path).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(metadata.MetadataImageMismatchError, match="belongs to"):
        metadata.load_metadata(image_path, expected_image_size=IMAGE_SIZE)


def test_sidecar_rejects_unknown_fields_instead_of_silently_dropping_them(tmp_path: Path):
    image_path = tmp_path / "0105.jpg"
    payload = metadata.new_metadata(image_path, *IMAGE_SIZE).to_dict()
    payload["unexpected_future_field"] = True
    metadata.metadata_path_for_image(image_path).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(metadata.MetadataSchemaError, match="invalid keys"):
        metadata.load_metadata(image_path, expected_image_size=IMAGE_SIZE)


def test_valid_corners_pass_and_include_area_metrics():
    result = metadata.validate_corners(_valid_corners(), *IMAGE_SIZE)

    assert result.is_valid
    assert result.errors == ()
    assert result.area_px2 == pytest.approx(217_250.0)
    assert result.area_ratio == pytest.approx(217_250.0 / (640 * 480))


@pytest.mark.parametrize(
    ("corners", "expected_error"),
    [
        ({**_valid_corners(), "bottom_left": None}, "missing_corners"),
        ({**_valid_corners(), "top_right": metadata.Point(50.0, 40.0)}, "duplicate_corners"),
        (
            {
                "top_left": metadata.Point(50, 40),
                "top_right": metadata.Point(590, 440),
                "bottom_right": metadata.Point(600, 45),
                "bottom_left": metadata.Point(40, 435),
            },
            "self_intersecting_polygon",
        ),
        (
            {
                "top_left": metadata.Point(0, 0),
                "top_right": metadata.Point(30, 0),
                "bottom_right": metadata.Point(30, 30),
                "bottom_left": metadata.Point(0, 30),
            },
            "board_area_too_small",
        ),
        ({**_valid_corners(), "bottom_right": metadata.Point(700, 440)}, "corner_out_of_bounds"),
    ],
)
def test_corner_validation_reports_bad_geometry(corners, expected_error: str):
    result = metadata.validate_corners(corners, *IMAGE_SIZE)

    assert result.is_valid is False
    assert expected_error in result.errors


def test_human_verified_corners_cannot_be_saved_when_geometry_is_bad(tmp_path: Path):
    image_path = tmp_path / "0105.jpg"
    source = metadata.new_metadata(image_path, *IMAGE_SIZE)
    source.board.corners_px = {**_valid_corners(), "bottom_right": metadata.Point(700, 440)}
    source.board.corners_status = "human_verified"

    with pytest.raises(metadata.MetadataSchemaError, match="human_verified"):
        metadata.save_metadata_atomic(image_path, source, expected_image_size=IMAGE_SIZE)
    assert not metadata.metadata_path_for_image(image_path).exists()


def test_human_marked_draft_with_incomplete_corners_can_be_saved(tmp_path: Path):
    image_path = tmp_path / "0105.jpg"
    source = metadata.new_metadata(image_path, *IMAGE_SIZE)
    source.board.corners_px["top_left"] = metadata.Point(10, 20)
    source.board.corners_status = "partial"

    metadata.save_metadata_atomic(image_path, source, expected_image_size=IMAGE_SIZE)
    assert metadata.load_metadata(image_path, expected_image_size=IMAGE_SIZE) == source


def test_verified_fen_and_gold_review_require_their_prerequisites(tmp_path: Path):
    image_path = tmp_path / "0105.jpg"
    source = metadata.new_metadata(image_path, *IMAGE_SIZE)
    source.board.fen_status = "human_verified"

    with pytest.raises(metadata.MetadataSchemaError, match="complete position"):
        metadata.validate_metadata(source)

    source = _complete_metadata(image_path)
    source.review.exclude_from_gold = True
    with pytest.raises(metadata.MetadataSchemaError, match="cannot be excluded"):
        metadata.validate_metadata(source)


def test_full_fen_must_match_board_fen_and_known_side(tmp_path: Path):
    image_path = tmp_path / "0105.jpg"
    source = metadata.new_metadata(image_path, *IMAGE_SIZE)
    source.board.board_fen = STARTING_BOARD_FEN
    source.board.side_to_move = "red"
    source.board.full_fen = f"{STARTING_BOARD_FEN} b - - 0 1"

    with pytest.raises(metadata.MetadataSchemaError, match="side-to-move"):
        metadata.validate_metadata(source)


def test_unsupported_schema_version_is_not_silently_accepted(tmp_path: Path):
    image_path = tmp_path / "0105.jpg"
    payload = metadata.new_metadata(image_path, *IMAGE_SIZE).to_dict()
    payload["schema_version"] = 2
    metadata.metadata_path_for_image(image_path).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(metadata.UnsupportedMetadataSchemaError):
        metadata.load_metadata(image_path, expected_image_size=IMAGE_SIZE)
