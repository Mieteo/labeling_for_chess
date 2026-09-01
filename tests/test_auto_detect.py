from pathlib import Path

import numpy as np
import pytest

from chess_labeler import auto_detect
from chess_labeler.constants import AUTO_DETECT_INPUT_SIZE, DEFAULT_CLASSES_DIGITAL

# A real model dropped locally for manual verification against the actual
# app model (see yeu_cau_tu_app_ky_nhan.md section 3.1). Not part of the
# repo -- the smoke test below is skipped everywhere it isn't present.
_LOCAL_MODEL_PATH = Path(r"D:\Workspace_Flutter\3th\model\test.onnx")


def test_model_label_order_matches_confirmed_app_source():
    # Exact order confirmed from digitalBoardModelLabels in the source app
    # -- see yeu_cau_tu_app_ky_nhan.md section 3.2.
    assert auto_detect.MODEL_LABELS == (
        "n", "b", "a", "k", "r", "c", "p",
        "R", "N", "A", "K", "B", "C", "P",
        "0",
    )


def test_class_mapping_covers_every_piece_plus_board_region():
    # Every one of the model's 15 labels (14 pieces + the "0" board-region
    # channel) must map to a class name, and that set must be exactly this
    # tool's 15 Digital classes minus `hand` -- mapping by name, never by
    # index.
    assert auto_detect.BOARD_REGION_MODEL_LABEL in auto_detect.MODEL_LABEL_TO_CLASS_NAME
    assert auto_detect.MODEL_LABEL_TO_CLASS_NAME[auto_detect.BOARD_REGION_MODEL_LABEL] == "board_region"
    mapped_classes = set(auto_detect.MODEL_LABEL_TO_CLASS_NAME.values())
    expected_classes = set(DEFAULT_CLASSES_DIGITAL) - {"hand"}
    assert mapped_classes == expected_classes
    assert len(auto_detect.MODEL_LABEL_TO_CLASS_NAME) == 15


def test_class_mapping_matches_documented_table():
    expected = {
        "n": "black_horse", "b": "black_elephant", "a": "black_advisor", "k": "black_king",
        "r": "black_rook", "c": "black_cannon", "p": "black_pawn",
        "R": "red_rook", "N": "red_horse", "A": "red_advisor", "K": "red_king",
        "B": "red_elephant", "C": "red_cannon", "P": "red_pawn",
        "0": "board_region",
    }
    assert auto_detect.MODEL_LABEL_TO_CLASS_NAME == expected


def _raw_box(cx, cy, w, h, class_idx, score):
    return auto_detect._RawBox(cx=cx, cy=cy, w=w, h=h, class_idx=class_idx, score=score)


def test_un_letterbox_maps_class_by_name_not_index():
    # class_idx 0 -> 'n' -> black_horse (never red_king, which is index 0 of
    # this tool's own classes.txt -- the whole point of section 3.2).
    b = _raw_box(cx=320, cy=320, w=40, h=60, class_idx=0, score=0.9)
    detection = auto_detect._un_letterbox(b, scale=1.0, pad_x=0.0, pad_y=0.0, img_w=640, img_h=640)
    assert detection is not None
    assert detection.class_name == "black_horse"


def test_un_letterbox_maps_board_region_channel():
    b = _raw_box(cx=320, cy=320, w=100, h=100, class_idx=14, score=0.99)
    detection = auto_detect._un_letterbox(b, scale=1.0, pad_x=0.0, pad_y=0.0, img_w=640, img_h=640)
    assert detection is not None
    assert detection.class_name == "board_region"


def test_un_letterbox_drops_out_of_range_class_idx():
    b = _raw_box(cx=320, cy=320, w=100, h=100, class_idx=99, score=0.99)
    assert auto_detect._un_letterbox(b, scale=1.0, pad_x=0.0, pad_y=0.0, img_w=640, img_h=640) is None


def test_un_letterbox_inverts_preprocess_padding():
    # Simulate a 800x400 original image (2:1) letterboxed into 640x640:
    # scale = min(640/800, 640/400) = 0.8, new_w=640, new_h=320,
    # pad_x=0, pad_y=160 -- matches _preprocess's own math exactly.
    img_w, img_h = 800, 400
    scale, pad_x, pad_y = 0.8, 0.0, 160.0
    orig_cx, orig_cy, orig_w, orig_h = 200.0, 100.0, 200.0, 100.0
    lb_box = _raw_box(
        cx=orig_cx * scale + pad_x,
        cy=orig_cy * scale + pad_y,
        w=orig_w * scale,
        h=orig_h * scale,
        class_idx=6,  # 'p' -> black_pawn
        score=0.9,
    )
    detection = auto_detect._un_letterbox(lb_box, scale, pad_x, pad_y, img_w, img_h)
    assert detection is not None
    assert detection.class_name == "black_pawn"
    assert detection.cx == pytest.approx(orig_cx, abs=0.5)
    assert detection.cy == pytest.approx(orig_cy, abs=0.5)
    assert detection.w == pytest.approx(orig_w, abs=0.5)
    assert detection.h == pytest.approx(orig_h, abs=0.5)


def test_preprocess_letterbox_shape_and_padding():
    image_bgr = np.zeros((400, 800, 3), dtype=np.uint8)  # h=400, w=800
    tensor, scale, pad_x, pad_y = auto_detect._preprocess(image_bgr)

    assert tensor.shape == (1, 3, AUTO_DETECT_INPUT_SIZE, AUTO_DETECT_INPUT_SIZE)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    assert scale == pytest.approx(0.8)
    assert pad_x == 0.0
    assert pad_y == 160.0

    # Padding rows (outside the pasted resized image) must be the letterbox
    # gray, not black -- a common off-by-one source of silently wrong boxes.
    pad_value = 114 / 255.0
    assert tensor[0, 0, 0, 0] == pytest.approx(pad_value, abs=1e-4)
    # Interior (the actual all-black source image) stays 0.
    assert tensor[0, 0, 320, 320] == pytest.approx(0.0, abs=1e-4)


def test_decode_and_nms_per_class_deduplicates_overlaps_and_filters_threshold():
    # 8400 anchors, 4 box channels + 15 class-score channels.
    output = np.zeros((19, 8400), dtype=np.float32)

    def set_anchor(i, cx, cy, w, h, class_idx, score):
        output[0, i] = cx
        output[1, i] = cy
        output[2, i] = w
        output[3, i] = h
        output[4 + class_idx, i] = score

    set_anchor(0, cx=320, cy=320, w=40, h=60, class_idx=0, score=0.9)  # kept (best of the pair)
    set_anchor(1, cx=322, cy=321, w=40, h=60, class_idx=0, score=0.6)  # near-duplicate -> NMS'd away
    set_anchor(2, cx=100, cy=100, w=200, h=200, class_idx=14, score=0.99)  # board-region channel
    set_anchor(3, cx=500, cy=500, w=20, h=20, class_idx=7, score=0.1)  # below conf threshold
    set_anchor(4, cx=500, cy=500, w=30, h=40, class_idx=7, score=0.8)  # kept, different class

    boxes = auto_detect._decode(output, conf_threshold=0.25)
    boxes = auto_detect._nms_per_class(boxes, iou_threshold=0.45)
    detections = [
        d for d in (
            auto_detect._un_letterbox(b, scale=1.0, pad_x=0.0, pad_y=0.0, img_w=640, img_h=640) for b in boxes
        )
        if d is not None
    ]

    by_class = {d.class_name: d for d in detections}
    assert set(by_class) == {"black_horse", "red_rook", "board_region"}
    assert by_class["black_horse"].score == pytest.approx(0.9)
    assert by_class["red_rook"].score == pytest.approx(0.8)
    assert by_class["board_region"].score == pytest.approx(0.99)


@pytest.mark.skipif(not _LOCAL_MODEL_PATH.exists(), reason="local .onnx model not present on this machine")
def test_real_model_smoke_run_returns_valid_class_names():
    detector = auto_detect.AutoDetector(str(_LOCAL_MODEL_PATH))
    image_bgr = np.full((720, 1280, 3), 200, dtype=np.uint8)  # plain screenshot-ish canvas
    detections = detector.detect(image_bgr)
    assert isinstance(detections, list)
    for d in detections:
        assert d.class_name in DEFAULT_CLASSES_DIGITAL
        assert d.class_name != "hand"
