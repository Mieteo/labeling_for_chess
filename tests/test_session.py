import json
from pathlib import Path

from chess_labeler import session, yolo_io


def _make_images(tmp_path: Path, labeled: list[str], unlabeled: list[str]) -> None:
    for name in labeled + unlabeled:
        (tmp_path / name).write_bytes(b"fake")
    for name in labeled:
        yolo_io.save_boxes(tmp_path / name, [])


def test_find_resume_image_picks_first_unlabeled(tmp_path: Path):
    _make_images(
        tmp_path,
        labeled=["0001.jpg", "0002.jpg"],
        unlabeled=["0003.jpg", "0004.jpg"],
    )
    resume = session.find_resume_image(tmp_path)
    assert resume.name == "0003.jpg"


def test_find_resume_image_all_labeled_falls_back_to_session_last_image(tmp_path: Path):
    _make_images(tmp_path, labeled=["0001.jpg", "0002.jpg", "0003.jpg"], unlabeled=[])
    session.save_session(tmp_path, session.SessionState(last_image="0002.jpg"))
    resume = session.find_resume_image(tmp_path)
    assert resume.name == "0002.jpg"


def test_find_resume_image_all_labeled_no_session_falls_back_to_first(tmp_path: Path):
    _make_images(tmp_path, labeled=["0001.jpg", "0002.jpg"], unlabeled=[])
    resume = session.find_resume_image(tmp_path)
    assert resume.name == "0001.jpg"


def test_find_resume_image_empty_dir_returns_none(tmp_path: Path):
    assert session.find_resume_image(tmp_path) is None


def test_load_session_missing_file_returns_default(tmp_path: Path):
    state = session.load_session(tmp_path)
    assert state.last_image is None
    assert state.last_radius_px is None


def test_load_session_corrupt_file_does_not_raise(tmp_path: Path):
    (tmp_path / ".labeling_session.json").write_text("{not valid json", encoding="utf-8")
    state = session.load_session(tmp_path)
    assert state.last_image is None


def test_save_and_load_session_roundtrip(tmp_path: Path):
    state = session.SessionState(last_image="0007.jpg", last_radius_px=42.5, last_tolerance_pct=20.0)
    session.save_session(tmp_path, state)
    loaded = session.load_session(tmp_path)
    assert loaded == state


def test_session_file_is_not_source_of_truth_for_progress(tmp_path: Path):
    # Even if the session file claims a stale/wrong last_image, resume logic
    # must still be driven by which .txt files actually exist.
    _make_images(tmp_path, labeled=["0001.jpg"], unlabeled=["0002.jpg", "0003.jpg"])
    session.save_session(tmp_path, session.SessionState(last_image="0003.jpg"))
    resume = session.find_resume_image(tmp_path)
    assert resume.name == "0002.jpg"


def test_recent_metadata_values_roundtrip_and_stay_small_mru_lists(tmp_path: Path):
    state = session.SessionState(
        recent_device_models=["Phone A", "Phone B"],
        recent_capture_groups=["set-1"],
    )
    session.save_session(tmp_path, state)
    assert session.load_session(tmp_path) == state

    updated = session.add_recent_value(state.recent_device_models, "Phone B")
    assert updated == ["Phone B", "Phone A"]
    assert session.add_recent_value(updated, "unknown") == updated


def test_image_mode_and_auto_detect_thresholds_roundtrip(tmp_path: Path):
    state = session.SessionState(image_mode="digital", last_auto_detect_conf=0.3, last_auto_detect_iou=0.5)
    session.save_session(tmp_path, state)
    assert session.load_session(tmp_path) == state


def test_default_session_has_no_image_mode_override(tmp_path: Path):
    state = session.load_session(tmp_path)
    assert state.image_mode is None


def test_invalid_image_mode_in_file_falls_back_to_none(tmp_path: Path):
    (tmp_path / ".labeling_session.json").write_text(
        json.dumps({"image_mode": "not_a_real_mode"}), encoding="utf-8"
    )
    state = session.load_session(tmp_path)
    assert state.image_mode is None
