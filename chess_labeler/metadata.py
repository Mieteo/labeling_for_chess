"""Versioned board-level metadata sidecars for Xiangqi label images.

The YOLO ``<stem>.txt`` file remains the source of truth for piece boxes and
must never contain board metadata.  This module owns the optional
``<stem>.meta.json`` sidecar described in ``labeling_tool_requirements.md``
section 10.  It deliberately has no Qt dependency so the UI and downstream
tools can share the same schema, validation, and safe file I/O.

The public loading API distinguishes three states:

* a missing sidecar is normal and ``load_metadata`` returns ``None``;
* a valid sidecar returns an :class:`ImageMetadata` instance;
* malformed JSON, an unsupported schema, or a mismatched image fingerprint
  raises :class:`MetadataError` (or is reported by ``try_load_metadata``).

Callers must not automatically save over an invalid sidecar.  The UI can use
``try_load_metadata`` to show the error while still opening the image and its
YOLO boxes in a safe mode; an explicit user repair/save action may then call
``save_metadata_atomic``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


METADATA_SCHEMA_VERSION = 1
METADATA_SUFFIX = ".meta.json"

CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
CORNER_STATUSES = frozenset(
    {
        "unmarked",
        "partial",
        "auto_suggested",
        "human_marked",
        "human_verified",
        "not_applicable",
    }
)
IMAGE_ORIENTATIONS = frozenset(
    {"red_at_bottom", "red_at_top", "red_at_left", "red_at_right", "unknown"}
)
FEN_STATUSES = frozenset(
    {"not_started", "human_marked", "human_verified", "not_applicable"}
)
REVIEW_STATUSES = frozenset(
    {"unreviewed", "annotated", "self_checked", "gold_verified", "needs_review"}
)
SIDES_TO_MOVE = frozenset({"red", "black"})

CAPTURE_ENUMS: dict[str, frozenset[str]] = {
    "lighting": frozenset({"unknown", "very_dark", "dim", "even", "bright", "mixed"}),
    "shadow": frozenset({"unknown", "none", "mild", "strong"}),
    "glare": frozenset({"unknown", "none", "mild", "strong"}),
    "perspective": frozenset({"unknown", "frontal", "mild", "strong", "extreme"}),
    "board_material": frozenset({"unknown", "wood", "plastic", "paper", "stone", "other"}),
    "board_fill": frozenset({"unknown", "tiny", "small", "medium", "large", "very_large"}),
    "distance": frozenset({"unknown", "near", "medium", "far"}),
    "blur": frozenset({"unknown", "none", "mild", "strong"}),
    "occlusion": frozenset({"unknown", "none", "hand", "piece", "object", "multiple"}),
    "occlusion_severity": frozenset({"unknown", "none", "mild", "strong"}),
    "environment": frozenset({"unknown", "indoor", "outdoor", "mixed"}),
}

_TOP_LEVEL_KEYS = frozenset({"schema_version", "image", "board", "capture", "review"})
_IMAGE_KEYS = frozenset({"filename", "width_px", "height_px"})
_BOARD_KEYS = frozenset(
    {
        "corners_px",
        "corners_status",
        "image_orientation",
        "position_complete",
        "board_fen",
        "side_to_move",
        "full_fen",
        "fen_status",
    }
)
_CAPTURE_KEYS = frozenset({*CAPTURE_ENUMS, "device_model", "capture_group"})
_REVIEW_KEYS = frozenset(
    {
        "status",
        "fen_verified",
        "corners_verified",
        "exclude_from_gold",
        "exclusion_reason",
        "notes",
    }
)
_XQ_FEN_PIECES = frozenset("KABRCNPkabrcnp")


class MetadataError(ValueError):
    """Base exception for invalid, unreadable, or unwritable sidecars."""


class MetadataDecodeError(MetadataError):
    """The sidecar is not valid UTF-8 JSON."""


class MetadataSchemaError(MetadataError):
    """The sidecar JSON is valid but does not conform to schema version 1."""


class UnsupportedMetadataSchemaError(MetadataSchemaError):
    """The sidecar declares a schema version this tool cannot safely edit."""


class MetadataImageMismatchError(MetadataSchemaError):
    """The sidecar fingerprint does not belong to the image being opened."""


class MetadataSaveError(MetadataError):
    """The metadata could not be atomically persisted."""


@dataclasses.dataclass(frozen=True)
class Point:
    """One point in original-image pixel coordinates."""

    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y)}


def _empty_corners() -> dict[str, Point | None]:
    return {name: None for name in CORNER_NAMES}


@dataclasses.dataclass
class ImageFingerprint:
    """A minimal, portable check that a sidecar belongs to its image."""

    filename: str
    width_px: int
    height_px: int

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "width_px": self.width_px,
            "height_px": self.height_px,
        }


@dataclasses.dataclass
class BoardMetadata:
    """Human-provided board geometry and FEN ground truth."""

    corners_px: dict[str, Point | None] = dataclasses.field(default_factory=_empty_corners)
    corners_status: str = "unmarked"
    image_orientation: str = "unknown"
    position_complete: bool = False
    board_fen: str | None = None
    side_to_move: str | None = None
    full_fen: str | None = None
    fen_status: str = "not_started"

    def to_dict(self) -> dict[str, object]:
        return {
            "corners_px": {
                name: self.corners_px[name].to_dict() if self.corners_px[name] is not None else None
                for name in CORNER_NAMES
            },
            "corners_status": self.corners_status,
            "image_orientation": self.image_orientation,
            "position_complete": self.position_complete,
            "board_fen": self.board_fen,
            "side_to_move": self.side_to_move,
            "full_fen": self.full_fen,
            "fen_status": self.fen_status,
        }


@dataclasses.dataclass
class CaptureMetadata:
    """Controlled-vocabulary capture-condition tags for coverage analysis."""

    lighting: str = "unknown"
    shadow: str = "unknown"
    glare: str = "unknown"
    perspective: str = "unknown"
    board_material: str = "unknown"
    board_fill: str = "unknown"
    distance: str = "unknown"
    blur: str = "unknown"
    occlusion: str = "unknown"
    occlusion_severity: str = "unknown"
    environment: str = "unknown"
    device_model: str = "unknown"
    capture_group: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "lighting": self.lighting,
            "shadow": self.shadow,
            "glare": self.glare,
            "perspective": self.perspective,
            "board_material": self.board_material,
            "board_fill": self.board_fill,
            "distance": self.distance,
            "blur": self.blur,
            "occlusion": self.occlusion,
            "occlusion_severity": self.occlusion_severity,
            "environment": self.environment,
            "device_model": self.device_model,
            "capture_group": self.capture_group,
        }


@dataclasses.dataclass
class ReviewMetadata:
    """Review and gold-set eligibility state, separate from raw labels."""

    status: str = "unreviewed"
    fen_verified: bool = False
    corners_verified: bool = False
    exclude_from_gold: bool = False
    exclusion_reason: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "fen_verified": self.fen_verified,
            "corners_verified": self.corners_verified,
            "exclude_from_gold": self.exclude_from_gold,
            "exclusion_reason": self.exclusion_reason,
            "notes": self.notes,
        }


@dataclasses.dataclass
class ImageMetadata:
    """The complete version-1 sidecar model."""

    image: ImageFingerprint
    board: BoardMetadata = dataclasses.field(default_factory=BoardMetadata)
    capture: CaptureMetadata = dataclasses.field(default_factory=CaptureMetadata)
    review: ReviewMetadata = dataclasses.field(default_factory=ReviewMetadata)
    schema_version: int = METADATA_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return metadata_to_dict(self)


@dataclasses.dataclass(frozen=True)
class CornerValidation:
    """Result of validating the fixed-order board quadrilateral.

    ``errors`` deliberately uses stable machine-readable codes.  The UI can
    translate these codes into Vietnamese text without parsing exception
    strings, and tests/downstream scripts can make deterministic decisions.
    """

    errors: tuple[str, ...] = ()
    area_px2: float | None = None
    area_ratio: float | None = None

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclasses.dataclass(frozen=True)
class MetadataLoadResult:
    """Non-throwing result used by a UI that must keep an image open safely."""

    metadata: ImageMetadata | None
    found: bool
    error: MetadataError | None = None

    @property
    def is_valid(self) -> bool:
        return self.found and self.metadata is not None and self.error is None


def metadata_path_for_image(image_path: Path | str) -> Path:
    """Return the required ``<stem>.meta.json`` sibling path.

    ``Path.with_suffix`` intentionally removes only the final image suffix,
    so ``board.photo.jpg`` maps to ``board.photo.meta.json``.
    """

    return Path(image_path).with_suffix(METADATA_SUFFIX)


def new_metadata(
    image_path: Path | str,
    image_width_px: int,
    image_height_px: int,
) -> ImageMetadata:
    """Build the safe, explicitly-unannotated metadata state for an image."""

    width, height = _validate_image_size((image_width_px, image_height_px))
    image_name = Path(image_path).name
    if not image_name:
        raise MetadataSchemaError("image path must include a filename")
    return ImageMetadata(image=ImageFingerprint(image_name, width, height))


def metadata_to_dict(metadata: ImageMetadata) -> dict[str, object]:
    """Serialize a model without writing it. ``save_metadata_atomic`` also
    validates the model before persisting it.
    """

    return {
        "schema_version": metadata.schema_version,
        "image": metadata.image.to_dict(),
        "board": metadata.board.to_dict(),
        "capture": metadata.capture.to_dict(),
        "review": metadata.review.to_dict(),
    }


def metadata_from_dict(data: object) -> ImageMetadata:
    """Parse and validate a schema-version-1 JSON object into a model."""

    root = _as_mapping(data, "metadata")
    _require_exact_keys(root, _TOP_LEVEL_KEYS, "metadata")

    version = _as_int(root["schema_version"], "schema_version")
    if version != METADATA_SCHEMA_VERSION:
        raise UnsupportedMetadataSchemaError(
            f"unsupported metadata schema_version {version}; expected {METADATA_SCHEMA_VERSION}"
        )

    image_data = _as_mapping(root["image"], "image")
    _require_exact_keys(image_data, _IMAGE_KEYS, "image")
    filename = _as_filename(image_data["filename"], "image.filename")
    width = _as_positive_int(image_data["width_px"], "image.width_px")
    height = _as_positive_int(image_data["height_px"], "image.height_px")
    image = ImageFingerprint(filename=filename, width_px=width, height_px=height)

    board_data = _as_mapping(root["board"], "board")
    _require_exact_keys(board_data, _BOARD_KEYS, "board")
    corners_data = _as_mapping(board_data["corners_px"], "board.corners_px")
    _require_exact_keys(corners_data, frozenset(CORNER_NAMES), "board.corners_px")
    corners = {
        name: _optional_point(corners_data[name], f"board.corners_px.{name}")
        for name in CORNER_NAMES
    }
    board = BoardMetadata(
        corners_px=corners,
        corners_status=_as_enum(board_data["corners_status"], CORNER_STATUSES, "board.corners_status"),
        image_orientation=_as_enum(
            board_data["image_orientation"], IMAGE_ORIENTATIONS, "board.image_orientation"
        ),
        position_complete=_as_bool(board_data["position_complete"], "board.position_complete"),
        board_fen=_optional_string(board_data["board_fen"], "board.board_fen"),
        side_to_move=_optional_enum(board_data["side_to_move"], SIDES_TO_MOVE, "board.side_to_move"),
        full_fen=_optional_string(board_data["full_fen"], "board.full_fen"),
        fen_status=_as_enum(board_data["fen_status"], FEN_STATUSES, "board.fen_status"),
    )

    capture_data = _as_mapping(root["capture"], "capture")
    _require_exact_keys(capture_data, _CAPTURE_KEYS, "capture")
    capture_values = {
        name: _as_enum(capture_data[name], allowed, f"capture.{name}")
        for name, allowed in CAPTURE_ENUMS.items()
    }
    capture = CaptureMetadata(
        **capture_values,
        device_model=_as_string(capture_data["device_model"], "capture.device_model"),
        capture_group=_optional_string(capture_data["capture_group"], "capture.capture_group"),
    )

    review_data = _as_mapping(root["review"], "review")
    _require_exact_keys(review_data, _REVIEW_KEYS, "review")
    review = ReviewMetadata(
        status=_as_enum(review_data["status"], REVIEW_STATUSES, "review.status"),
        fen_verified=_as_bool(review_data["fen_verified"], "review.fen_verified"),
        corners_verified=_as_bool(review_data["corners_verified"], "review.corners_verified"),
        exclude_from_gold=_as_bool(
            review_data["exclude_from_gold"], "review.exclude_from_gold"
        ),
        exclusion_reason=_optional_string(
            review_data["exclusion_reason"], "review.exclusion_reason"
        ),
        notes=_as_string(review_data["notes"], "review.notes"),
    )

    metadata = ImageMetadata(
        image=image,
        board=board,
        capture=capture,
        review=review,
        schema_version=version,
    )
    validate_metadata(metadata)
    return metadata


def load_metadata(
    image_path: Path | str,
    *,
    expected_image_size: tuple[int, int] | None = None,
) -> ImageMetadata | None:
    """Load a sibling sidecar, returning ``None`` when it has not been made.

    A bad sidecar is intentionally *not* treated as missing: this function
    raises a descriptive :class:`MetadataError` so callers do not overwrite
    it by accident.  Pass the original image size when known to catch a
    sidecar copied to a resized/different image.
    """

    path = metadata_path_for_image(image_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as exc:
        raise MetadataDecodeError(f"{path.name}: metadata is not UTF-8") from exc
    except OSError as exc:
        raise MetadataError(f"could not read metadata sidecar {path}: {exc}") from exc

    try:
        data = json.loads(raw, parse_constant=_reject_non_standard_json_constant)
    except json.JSONDecodeError as exc:
        raise MetadataDecodeError(f"{path.name}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc
    except ValueError as exc:
        raise MetadataDecodeError(f"{path.name}: invalid JSON constant ({exc})") from exc

    metadata = metadata_from_dict(data)
    _validate_image_binding(metadata, Path(image_path), expected_image_size)
    return metadata


def try_load_metadata(
    image_path: Path | str,
    *,
    expected_image_size: tuple[int, int] | None = None,
) -> MetadataLoadResult:
    """Safely load a sidecar without hiding why an existing file is invalid."""

    path = metadata_path_for_image(image_path)
    try:
        metadata = load_metadata(image_path, expected_image_size=expected_image_size)
    except MetadataError as exc:
        return MetadataLoadResult(metadata=None, found=path.exists(), error=exc)
    return MetadataLoadResult(metadata=metadata, found=metadata is not None, error=None)


def save_metadata_atomic(
    image_path: Path | str,
    metadata: ImageMetadata,
    *,
    expected_image_size: tuple[int, int] | None = None,
) -> Path:
    """Validate and atomically replace only this image's ``.meta.json``.

    The temporary file is created in the destination directory and replaced
    with :func:`os.replace`, so a crash cannot leave a partially-written
    target sidecar.  This function never opens, rewrites, or otherwise
    touches the sibling YOLO ``.txt`` file.
    """

    validate_metadata(metadata)
    image = Path(image_path)
    _validate_image_binding(metadata, image, expected_image_size)
    target = metadata_path_for_image(image)
    try:
        content = json.dumps(
            metadata_to_dict(metadata), ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise MetadataSchemaError(f"metadata cannot be serialized as JSON: {exc}") from exc

    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        temp_path = None  # os.replace consumed it; do not try to unlink target.
    except OSError as exc:
        raise MetadataSaveError(f"could not atomically save metadata sidecar {target}: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                # The original target remains untouched; there is no safe
                # recovery action for a leftover temp file at this layer.
                pass
    return target


def validate_metadata(metadata: ImageMetadata) -> None:
    """Validate a manually-created model before it can be saved.

    ``metadata_from_dict`` already calls this, but UI code mutates dataclass
    instances in memory.  Re-validating before every write prevents a UI bug
    from producing a syntactically valid yet unusable gold-set sidecar.
    """

    if not isinstance(metadata, ImageMetadata):
        raise MetadataSchemaError("metadata must be an ImageMetadata instance")
    if _as_int(metadata.schema_version, "schema_version") != METADATA_SCHEMA_VERSION:
        raise UnsupportedMetadataSchemaError(
            f"unsupported metadata schema_version {metadata.schema_version}; "
            f"expected {METADATA_SCHEMA_VERSION}"
        )

    if not isinstance(metadata.image, ImageFingerprint):
        raise MetadataSchemaError("image must be an ImageFingerprint")
    _as_filename(metadata.image.filename, "image.filename")
    _as_positive_int(metadata.image.width_px, "image.width_px")
    _as_positive_int(metadata.image.height_px, "image.height_px")

    if not isinstance(metadata.board, BoardMetadata):
        raise MetadataSchemaError("board must be a BoardMetadata")
    _validate_corners_mapping(metadata.board.corners_px)
    _as_enum(metadata.board.corners_status, CORNER_STATUSES, "board.corners_status")
    _as_enum(metadata.board.image_orientation, IMAGE_ORIENTATIONS, "board.image_orientation")
    _as_bool(metadata.board.position_complete, "board.position_complete")
    _validate_board_fen(metadata.board.board_fen)
    _optional_enum(metadata.board.side_to_move, SIDES_TO_MOVE, "board.side_to_move")
    _validate_full_fen(metadata.board)
    _as_enum(metadata.board.fen_status, FEN_STATUSES, "board.fen_status")

    if not isinstance(metadata.capture, CaptureMetadata):
        raise MetadataSchemaError("capture must be a CaptureMetadata")
    for name, allowed in CAPTURE_ENUMS.items():
        _as_enum(getattr(metadata.capture, name), allowed, f"capture.{name}")
    _as_string(metadata.capture.device_model, "capture.device_model")
    _optional_string(metadata.capture.capture_group, "capture.capture_group")

    if not isinstance(metadata.review, ReviewMetadata):
        raise MetadataSchemaError("review must be a ReviewMetadata")
    _as_enum(metadata.review.status, REVIEW_STATUSES, "review.status")
    _as_bool(metadata.review.fen_verified, "review.fen_verified")
    _as_bool(metadata.review.corners_verified, "review.corners_verified")
    _as_bool(metadata.review.exclude_from_gold, "review.exclude_from_gold")
    _optional_string(metadata.review.exclusion_reason, "review.exclusion_reason")
    _as_string(metadata.review.notes, "review.notes")

    corner_validation = validate_corners(
        metadata.board.corners_px,
        metadata.image.width_px,
        metadata.image.height_px,
    )
    if metadata.board.corners_status == "human_verified" and not corner_validation.is_valid:
        raise MetadataSchemaError(
            "board.corners_status='human_verified' requires valid corners: "
            + ", ".join(corner_validation.errors)
        )
    if metadata.review.corners_verified and not corner_validation.is_valid:
        raise MetadataSchemaError(
            "review.corners_verified=true requires valid corners: "
            + ", ".join(corner_validation.errors)
        )
    if metadata.review.corners_verified and metadata.board.corners_status != "human_verified":
        raise MetadataSchemaError(
            "review.corners_verified=true requires board.corners_status='human_verified'"
        )
    if metadata.board.fen_status == "human_verified":
        if not metadata.board.position_complete or metadata.board.board_fen is None:
            raise MetadataSchemaError(
                "board.fen_status='human_verified' requires a complete position and board_fen"
            )
    if metadata.review.fen_verified:
        if metadata.board.fen_status != "human_verified":
            raise MetadataSchemaError(
                "review.fen_verified=true requires board.fen_status='human_verified'"
            )
        if not metadata.board.position_complete or metadata.board.board_fen is None:
            raise MetadataSchemaError(
                "review.fen_verified=true requires a complete position and board_fen"
            )
    if metadata.review.status == "gold_verified":
        if metadata.review.exclude_from_gold:
            raise MetadataSchemaError("review.status='gold_verified' cannot be excluded from gold")
        if not (metadata.review.fen_verified and metadata.review.corners_verified):
            raise MetadataSchemaError(
                "review.status='gold_verified' requires verified FEN and corners"
            )


def validate_corners(
    corners_px: Mapping[str, Point | None],
    image_width_px: int,
    image_height_px: int,
    *,
    min_area_ratio: float = 0.01,
) -> CornerValidation:
    """Validate four fixed-order grid-intersection corners.

    The expected point order is ``top_left → top_right → bottom_right →
    bottom_left`` (clockwise in image coordinates, where ``y`` increases
    downward).  The result is intentionally non-throwing for ordinary bad
    user input, so a UI can display all relevant problems and still permit a
    non-verified ``human_marked`` draft.
    """

    errors: list[str] = []
    try:
        width, height = _validate_image_size((image_width_px, image_height_px))
    except MetadataSchemaError:
        return CornerValidation(errors=("invalid_image_size",))
    if not isinstance(min_area_ratio, (int, float)) or isinstance(min_area_ratio, bool):
        return CornerValidation(errors=("invalid_min_area_ratio",))
    if not math.isfinite(float(min_area_ratio)) or min_area_ratio < 0:
        return CornerValidation(errors=("invalid_min_area_ratio",))

    try:
        _validate_corners_mapping(corners_px)
    except MetadataSchemaError:
        return CornerValidation(errors=("invalid_corner_mapping",))

    points = [corners_px[name] for name in CORNER_NAMES]
    if any(point is None for point in points):
        return CornerValidation(errors=("missing_corners",))
    # The preceding condition makes this cast logically safe without using a
    # typing-only import or exposing an Optional point to geometry helpers.
    actual_points = [point for point in points if point is not None]

    for point in actual_points:
        if not (0.0 <= point.x <= width and 0.0 <= point.y <= height):
            errors.append("corner_out_of_bounds")
            break

    # A separation smaller than one image pixel is not a meaningful separate
    # grid corner and typically comes from duplicate-entry mistakes.
    for i, first in enumerate(actual_points):
        for second in actual_points[i + 1 :]:
            if _distance_squared(first, second) < 1.0:
                errors.append("duplicate_corners")
                break
        if "duplicate_corners" in errors:
            break

    if "duplicate_corners" in errors:
        return CornerValidation(errors=tuple(errors))

    if _has_self_intersection(actual_points):
        errors.append("self_intersecting_polygon")

    signed_area = _signed_polygon_area(actual_points)
    area = abs(signed_area)
    ratio = area / (width * height)
    if area <= 1e-9:
        errors.append("degenerate_polygon")
    elif not _is_strictly_convex(actual_points):
        errors.append("non_convex_polygon")
    if not _has_self_intersection(actual_points) and signed_area <= 0:
        errors.append("wrong_corner_order")
    if ratio < float(min_area_ratio):
        errors.append("board_area_too_small")

    return CornerValidation(errors=tuple(_unique_preserving_order(errors)), area_px2=area, area_ratio=ratio)


def _validate_image_binding(
    metadata: ImageMetadata,
    image_path: Path,
    expected_image_size: tuple[int, int] | None,
) -> None:
    if not _filenames_match(metadata.image.filename, image_path.name):
        raise MetadataImageMismatchError(
            f"sidecar belongs to {metadata.image.filename!r}, not {image_path.name!r}"
        )
    if expected_image_size is not None:
        width, height = _validate_image_size(expected_image_size)
        if (metadata.image.width_px, metadata.image.height_px) != (width, height):
            raise MetadataImageMismatchError(
                "sidecar dimensions "
                f"{metadata.image.width_px}x{metadata.image.height_px} do not match "
                f"image dimensions {width}x{height}"
            )


def _filenames_match(left: str, right: str) -> bool:
    # Use the platform's filename comparison semantics.  On Windows this is
    # case-insensitive; on case-sensitive systems a differently cased name is
    # treated as a likely sidecar mix-up.
    return os.path.normcase(left) == os.path.normcase(right)


def _validate_corners_mapping(corners_px: object) -> None:
    if not isinstance(corners_px, Mapping):
        raise MetadataSchemaError("board.corners_px must be an object")
    keys = set(corners_px)
    expected = set(CORNER_NAMES)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise MetadataSchemaError(
            "board.corners_px must contain exactly "
            f"{', '.join(CORNER_NAMES)} (missing={missing}, extra={extra})"
        )
    for name in CORNER_NAMES:
        point = corners_px[name]
        if point is not None and not isinstance(point, Point):
            raise MetadataSchemaError(f"board.corners_px.{name} must be a Point or null")
        if point is not None:
            _as_finite_number(point.x, f"board.corners_px.{name}.x")
            _as_finite_number(point.y, f"board.corners_px.{name}.y")


def _validate_board_fen(value: object) -> None:
    if value is None:
        return
    fen = _as_string(value, "board.board_fen")
    ranks = fen.split("/")
    if len(ranks) != 10:
        raise MetadataSchemaError("board.board_fen must contain exactly 10 ranks")
    for rank_index, rank in enumerate(ranks):
        width = 0
        for char in rank:
            if char in "123456789":
                width += int(char)
            elif char in _XQ_FEN_PIECES:
                width += 1
            else:
                raise MetadataSchemaError(
                    f"board.board_fen rank {rank_index + 1} contains invalid character {char!r}"
                )
        if width != 9:
            raise MetadataSchemaError(
                f"board.board_fen rank {rank_index + 1} has width {width}, expected 9"
            )


def _validate_full_fen(board: BoardMetadata) -> None:
    full_fen = _optional_string(board.full_fen, "board.full_fen")
    if full_fen is None:
        return
    if board.board_fen is None:
        raise MetadataSchemaError("board.full_fen requires board.board_fen")
    if board.side_to_move is None:
        raise MetadataSchemaError("board.full_fen requires board.side_to_move")
    fields = full_fen.split()
    if len(fields) < 2 or fields[0] != board.board_fen:
        raise MetadataSchemaError("board.full_fen must start with the same board.board_fen")
    expected_side_fields = (
        {"red", "w"} if board.side_to_move == "red" else {"black", "b"}
    )
    if fields[1] not in expected_side_fields:
        raise MetadataSchemaError(
            "board.full_fen side-to-move field does not match board.side_to_move"
        )


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MetadataSchemaError(f"{field} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise MetadataSchemaError(f"{field} must use string keys")
    return value  # type: ignore[return-value]


def _require_exact_keys(data: Mapping[str, object], expected: frozenset[str], field: str) -> None:
    actual = set(data)
    if actual == set(expected):
        return
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    raise MetadataSchemaError(f"{field} has invalid keys (missing={missing}, extra={extra})")


def _as_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise MetadataSchemaError(f"{field} must be a string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _as_string(value, field)


def _as_filename(value: object, field: str) -> str:
    filename = _as_string(value, field)
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise MetadataSchemaError(f"{field} must be a filename, not a path")
    return filename


def _as_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise MetadataSchemaError(f"{field} must be a boolean")
    return value


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MetadataSchemaError(f"{field} must be an integer")
    return value


def _as_positive_int(value: object, field: str) -> int:
    integer = _as_int(value, field)
    if integer <= 0:
        raise MetadataSchemaError(f"{field} must be greater than zero")
    return integer


def _as_finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetadataSchemaError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise MetadataSchemaError(f"{field} must be finite")
    return number


def _as_enum(value: object, allowed: frozenset[str], field: str) -> str:
    string = _as_string(value, field)
    if string not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise MetadataSchemaError(f"{field} must be one of: {allowed_text}")
    return string


def _optional_enum(value: object, allowed: frozenset[str], field: str) -> str | None:
    if value is None:
        return None
    return _as_enum(value, allowed, field)


def _optional_point(value: object, field: str) -> Point | None:
    if value is None:
        return None
    data = _as_mapping(value, field)
    _require_exact_keys(data, frozenset({"x", "y"}), field)
    return Point(
        x=_as_finite_number(data["x"], f"{field}.x"),
        y=_as_finite_number(data["y"], f"{field}.y"),
    )


def _validate_image_size(value: tuple[int, int] | object) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise MetadataSchemaError("expected_image_size must be a (width_px, height_px) tuple")
    return (
        _as_positive_int(value[0], "image.width_px"),
        _as_positive_int(value[1], "image.height_px"),
    )


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(value)


def _distance_squared(first: Point, second: Point) -> float:
    return (first.x - second.x) ** 2 + (first.y - second.y) ** 2


def _signed_polygon_area(points: list[Point]) -> float:
    total = 0.0
    for first, second in zip(points, points[1:] + points[:1]):
        total += first.x * second.y - second.x * first.y
    return total / 2.0


def _cross(origin: Point, first: Point, second: Point) -> float:
    return (first.x - origin.x) * (second.y - origin.y) - (first.y - origin.y) * (
        second.x - origin.x
    )


def _is_strictly_convex(points: list[Point]) -> bool:
    crosses = [
        _cross(points[index], points[(index + 1) % 4], points[(index + 2) % 4])
        for index in range(4)
    ]
    if any(abs(cross) <= 1e-9 for cross in crosses):
        return False
    return all(cross > 0 for cross in crosses) or all(cross < 0 for cross in crosses)


def _has_self_intersection(points: list[Point]) -> bool:
    # Opposite quadrilateral edges must not meet.  Adjacent edges share a
    # legitimate vertex and are deliberately excluded.
    return _segments_intersect(points[0], points[1], points[2], points[3]) or _segments_intersect(
        points[1], points[2], points[3], points[0]
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    ab_c = _cross(a, b, c)
    ab_d = _cross(a, b, d)
    cd_a = _cross(c, d, a)
    cd_b = _cross(c, d, b)
    epsilon = 1e-9

    if (
        abs(ab_c) <= epsilon
        and _on_segment(a, b, c)
        or abs(ab_d) <= epsilon
        and _on_segment(a, b, d)
        or abs(cd_a) <= epsilon
        and _on_segment(c, d, a)
        or abs(cd_b) <= epsilon
        and _on_segment(c, d, b)
    ):
        return True
    return (ab_c > 0) != (ab_d > 0) and (cd_a > 0) != (cd_b > 0)


def _on_segment(a: Point, b: Point, point: Point) -> bool:
    epsilon = 1e-9
    return (
        min(a.x, b.x) - epsilon <= point.x <= max(a.x, b.x) + epsilon
        and min(a.y, b.y) - epsilon <= point.y <= max(a.y, b.y) + epsilon
    )


def _unique_preserving_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
