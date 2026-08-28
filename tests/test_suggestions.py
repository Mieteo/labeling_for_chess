from pathlib import Path

from chess_labeler import suggestions


def test_suggestions_path_uses_autodetect_suffix():
    path = suggestions.suggestions_path_for_image(Path("0105.jpg"))
    assert path.name == "0105.autodetect.json"


def test_save_and_load_consumes_sidecar(tmp_path: Path):
    image_path = tmp_path / "0001.jpg"
    image_path.write_bytes(b"fake")
    boxes = [
        suggestions.PendingBox(class_name="red_cannon", xc=0.5, yc=0.4, w=0.05, h=0.06, score=0.83),
        suggestions.PendingBox(class_name="black_pawn", xc=0.2, yc=0.3, w=0.04, h=0.05, score=0.61),
    ]
    suggestions.save_suggestions(image_path, boxes, model_path="C:/model.onnx", conf_threshold=0.25, iou_threshold=0.45)

    assert suggestions.has_pending_suggestions(image_path)
    loaded = suggestions.load_and_consume_suggestions(image_path)
    assert loaded == boxes
    # Consuming deletes the sidecar -- it is never a source of truth.
    assert not suggestions.has_pending_suggestions(image_path)
    assert suggestions.load_and_consume_suggestions(image_path) == []


def test_missing_sidecar_returns_empty(tmp_path: Path):
    assert suggestions.load_and_consume_suggestions(tmp_path / "none.jpg") == []


def test_corrupt_sidecar_is_treated_as_empty_and_removed(tmp_path: Path):
    image_path = tmp_path / "0002.jpg"
    image_path.write_bytes(b"fake")
    sidecar = suggestions.suggestions_path_for_image(image_path)
    sidecar.write_text("{not valid json", encoding="utf-8")

    assert suggestions.load_and_consume_suggestions(image_path) == []
    assert not sidecar.exists()


def test_discard_suggestions_removes_file_without_loading(tmp_path: Path):
    image_path = tmp_path / "0003.jpg"
    image_path.write_bytes(b"fake")
    suggestions.save_suggestions(
        image_path,
        [suggestions.PendingBox(class_name="hand", xc=0.5, yc=0.5, w=0.2, h=0.2)],
        model_path="m.onnx",
        conf_threshold=0.25,
        iou_threshold=0.45,
    )
    suggestions.discard_suggestions(image_path)
    assert not suggestions.has_pending_suggestions(image_path)
