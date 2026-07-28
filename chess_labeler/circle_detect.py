"""Classical (non-ML) circle detection to speed up box drawing on real
photographed Xiangqi boards -- see labeling_tool_requirements.md section 4.

Two modes, both backed by the same OpenCV HoughCircles core:

- auto_scan(): radius range guessed from image width (1.5%-6%).
- radius_guided(): radius range = one measured reference radius +/-
  tolerance% (default +/-15%, user-adjustable).

Detected circles are only *suggestions*; turning them into real boxes and
assigning a class is a separate, explicit user action handled elsewhere.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from .constants import AUTO_SCAN_MAX_RADIUS_FRACTION, AUTO_SCAN_MIN_RADIUS_FRACTION

# Downscale large photos before running Hough -- keeps detection under the
# ~1-2s/image budget and Hough tends to behave better at moderate resolution
# anyway. Detected coordinates are scaled back to the original image size.
_MAX_DETECTION_DIM = 1600


@dataclasses.dataclass
class DetectedCircle:
    cx: float
    cy: float
    r: float


def _scale_for_detection(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= _MAX_DETECTION_DIM:
        return image_bgr, 1.0
    scale = _MAX_DETECTION_DIM / longest
    resized = cv2.resize(
        image_bgr,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _detect_circles_in_radius_range(
    image_bgr: np.ndarray,
    min_radius_px: int,
    max_radius_px: int,
    min_dist_px: float | None = None,
    param1: float = 100.0,
    param2: float = 20.0,
) -> list[DetectedCircle]:
    min_radius_px = max(1, int(min_radius_px))
    max_radius_px = max(min_radius_px + 1, int(max_radius_px))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    if min_dist_px is None:
        min_dist_px = max(min_radius_px * 1.5, 10.0)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min_dist_px,
        param1=param1,
        param2=param2,
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
    )
    if circles is None:
        return []
    return [DetectedCircle(float(c[0]), float(c[1]), float(c[2])) for c in circles[0]]


def auto_scan(
    image_bgr: np.ndarray,
    min_fraction: float = AUTO_SCAN_MIN_RADIUS_FRACTION,
    max_fraction: float = AUTO_SCAN_MAX_RADIUS_FRACTION,
) -> list[DetectedCircle]:
    """Full-range scan: radius bounds guessed as a fraction of image width."""
    h, w = image_bgr.shape[:2]
    min_r = max(1.0, w * min_fraction)
    max_r = max(min_r + 1.0, w * max_fraction)

    scaled_img, scale = _scale_for_detection(image_bgr)
    circles = _detect_circles_in_radius_range(
        scaled_img,
        min_radius_px=int(round(min_r * scale)),
        max_radius_px=int(round(max_r * scale)),
    )
    if scale != 1.0:
        circles = [DetectedCircle(c.cx / scale, c.cy / scale, c.r / scale) for c in circles]
    return circles


def radius_guided(
    image_bgr: np.ndarray,
    reference_radius_px: float,
    tolerance_pct: float,
) -> list[DetectedCircle]:
    """Narrow scan around a user-measured reference radius (original-image
    pixel units) +/- tolerance_pct percent."""
    tol = max(tolerance_pct, 0.0) / 100.0
    min_r = max(1.0, reference_radius_px * (1 - tol))
    max_r = max(min_r + 1.0, reference_radius_px * (1 + tol))

    scaled_img, scale = _scale_for_detection(image_bgr)
    circles = _detect_circles_in_radius_range(
        scaled_img,
        min_radius_px=int(round(min_r * scale)),
        max_radius_px=int(round(max_r * scale)),
    )
    if scale != 1.0:
        circles = [DetectedCircle(c.cx / scale, c.cy / scale, c.r / scale) for c in circles]
    return circles


def circle_to_pixel_box(circle: DetectedCircle) -> tuple[float, float, float, float]:
    """(left, top, width, height) of the square box enclosing a circle."""
    side = circle.r * 2
    left = circle.cx - circle.r
    top = circle.cy - circle.r
    return left, top, side, side
