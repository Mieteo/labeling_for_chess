"""ONNX auto-detect assist for Digital images, using an existing exported
Xiangqi piece detector (e.g. the source app's
``xiangqi_piece_detector_v1.onnx``) purely for *inference* -- see
yeu_cau_tu_app_ky_nhan.md section 3. No training/fine-tuning happens here.

Detections are only *suggestions*, exactly like circle_detect.py: turning
one into a real, saved box is a separate, explicit user action handled in
main_window.py (reusing the same confirm/reassign/delete mechanics already
built for circle-detect suggestions).

Model contract (confirmed against the real .onnx file, see section 3.1):

- input ``images``, float32, shape [1, 3, 640, 640] (NCHW), RGB, [0, 1].
  Preprocessing: letterbox resize into a 640x640 square, pad color (114,
  114, 114).
- output ``output0``, float32, shape [1, 19, 8400] = 4 box channels
  (cx, cy, w, h, in letterboxed-640 pixel units) + 15 class-score channels.
  Ultralytics YOLOv8 single-tensor format: no separate objectness channel,
  class scores are already probabilities (no sigmoid needed on decode).
- NMS runs per-class (never a single global NMS across all classes).
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from .constants import (
    AUTO_DETECT_INPUT_SIZE,
    AUTO_DETECT_LETTERBOX_COLOR,
    DEFAULT_AUTO_DETECT_CONF_THRESHOLD,
    DEFAULT_AUTO_DETECT_IOU_THRESHOLD,
)

# Model class order (index -> FEN-style letter), confirmed from the source
# app's `digitalBoardModelLabels` constant
# (lib/src/core/scanner/xiangqi_pwa_baseline.dart). This order is COMPLETELY
# DIFFERENT from this tool's classes.txt order (see
# yeu_cau_tu_app_ky_nhan.md section 3.2) -- mapping MUST go by name, never by
# index, or every auto-detect suggestion is silently mislabeled.
MODEL_LABELS: tuple[str, ...] = (
    "n", "b", "a", "k", "r", "c", "p",
    "R", "N", "A", "K", "B", "C", "P",
    "0",
)

# FEN-style letter -> this tool's classes.txt class name. Index 14 ('0') is
# the model's internal "board region" channel, not a piece -- it is
# intentionally absent from this table and must never be mapped to any
# class (not `hand`, not anything else).
MODEL_LABEL_TO_CLASS_NAME: dict[str, str] = {
    "n": "black_horse",
    "b": "black_elephant",
    "a": "black_advisor",
    "k": "black_king",
    "r": "black_rook",
    "c": "black_cannon",
    "p": "black_pawn",
    "R": "red_rook",
    "N": "red_horse",
    "A": "red_advisor",
    "K": "red_king",
    "B": "red_elephant",
    "C": "red_cannon",
    "P": "red_pawn",
}

NON_PIECE_MODEL_LABEL = "0"


@dataclasses.dataclass
class Detection:
    """A single decoded, un-letterboxed detection in original-image pixel
    coordinates -- (cx, cy) center, (w, h) size, already mapped to this
    tool's class name."""

    cx: float
    cy: float
    w: float
    h: float
    class_name: str
    score: float


class ModelLoadError(RuntimeError):
    """The .onnx file could not be opened, or onnxruntime is missing."""


class AutoDetector:
    """Wraps one onnxruntime session. Session creation is the expensive
    part, so callers should keep one instance alive per chosen model path
    rather than recreating it for every image."""

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ModelLoadError(
                "Thiếu thư viện onnxruntime. Cài bằng: pip install onnxruntime"
            ) from exc
        try:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ModelLoadError(f"Không thể mở model ONNX: {exc}") from exc
        self.model_path = model_path
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

    def detect(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = DEFAULT_AUTO_DETECT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_AUTO_DETECT_IOU_THRESHOLD,
    ) -> list[Detection]:
        tensor, scale, pad_x, pad_y = _preprocess(image_bgr)
        raw = self._session.run([self._output_name], {self._input_name: tensor})[0]
        boxes = _decode(raw[0], conf_threshold)
        boxes = _nms_per_class(boxes, iou_threshold)
        img_h, img_w = image_bgr.shape[:2]
        detections = [_un_letterbox(b, scale, pad_x, pad_y, img_w, img_h) for b in boxes]
        return [d for d in detections if d is not None]


@dataclasses.dataclass
class _RawBox:
    cx: float
    cy: float
    w: float
    h: float
    class_idx: int
    score: float


def _preprocess(image_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Letterbox `image_bgr` into a centered AUTO_DETECT_INPUT_SIZE square
    and build the NCHW float32 [0,1] RGB input tensor. Returns
    (tensor, scale, pad_x, pad_y) so callers can un-letterbox decoded boxes
    back to original-image pixel coordinates using the exact same integer
    padding actually used to place the resized image on the canvas."""
    h, w = image_bgr.shape[:2]
    size = AUTO_DETECT_INPUT_SIZE
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), AUTO_DETECT_LETTERBOX_COLOR, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])  # NCHW
    return tensor, scale, float(pad_x), float(pad_y)


def _decode(output: np.ndarray, conf_threshold: float) -> list[_RawBox]:
    """`output`: [19, 8400] = 4 box channels + 15 class-score channels
    (already probabilities, no objectness/sigmoid step needed)."""
    boxes_xywh = output[:4, :]
    class_scores = output[4:, :]
    class_idx = np.argmax(class_scores, axis=0)
    scores = class_scores[class_idx, np.arange(class_scores.shape[1])]
    keep = np.nonzero(scores >= conf_threshold)[0]
    return [
        _RawBox(
            cx=float(boxes_xywh[0, i]),
            cy=float(boxes_xywh[1, i]),
            w=float(boxes_xywh[2, i]),
            h=float(boxes_xywh[3, i]),
            class_idx=int(class_idx[i]),
            score=float(scores[i]),
        )
        for i in keep
    ]


def _box_corners(b: _RawBox) -> tuple[float, float, float, float]:
    return (b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2)


def _iou(a: _RawBox, b: _RawBox) -> float:
    ax0, ay0, ax1, ay1 = _box_corners(a)
    bx0, by0, bx1, by1 = _box_corners(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_per_class(boxes: list[_RawBox], iou_threshold: float) -> list[_RawBox]:
    """NMS run independently within each class channel -- never a single
    global NMS across all 14 piece classes (see module docstring)."""
    by_class: dict[int, list[_RawBox]] = {}
    for b in boxes:
        by_class.setdefault(b.class_idx, []).append(b)
    kept: list[_RawBox] = []
    for class_boxes in by_class.values():
        class_boxes.sort(key=lambda b: b.score, reverse=True)
        chosen: list[_RawBox] = []
        for b in class_boxes:
            if all(_iou(b, c) <= iou_threshold for c in chosen):
                chosen.append(b)
        kept.extend(chosen)
    return kept


def _un_letterbox(
    b: _RawBox, scale: float, pad_x: float, pad_y: float, img_w: int, img_h: int
) -> Detection | None:
    label = MODEL_LABELS[b.class_idx] if 0 <= b.class_idx < len(MODEL_LABELS) else None
    class_name = MODEL_LABEL_TO_CLASS_NAME.get(label) if label is not None else None
    if class_name is None:
        # Either the non-piece "board region" channel (index 14, '0') or an
        # out-of-range index -- never surfaced as a suggestion.
        return None

    cx = (b.cx - pad_x) / scale
    cy = (b.cy - pad_y) / scale
    w = b.w / scale
    h = b.h / scale
    left = max(0.0, cx - w / 2)
    top = max(0.0, cy - h / 2)
    right = min(float(img_w), cx + w / 2)
    bottom = min(float(img_h), cy + h / 2)
    if right <= left or bottom <= top:
        return None
    return Detection(
        cx=(left + right) / 2,
        cy=(top + bottom) / 2,
        w=right - left,
        h=bottom - top,
        class_name=class_name,
        score=b.score,
    )
