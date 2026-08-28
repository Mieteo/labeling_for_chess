from chess_labeler import image_mode


def test_digital_dirname_infers_digital():
    assert image_mode.infer_mode_from_dirname("D:/data/digitalImg") == image_mode.DIGITAL
    assert image_mode.infer_mode_from_dirname("DigitalScreens") == image_mode.DIGITAL


def test_non_digital_dirname_infers_physical():
    assert image_mode.infer_mode_from_dirname("D:/data/chessImg") == image_mode.PHYSICAL
    assert image_mode.infer_mode_from_dirname("photos") == image_mode.PHYSICAL


def test_inference_is_case_insensitive():
    assert image_mode.infer_mode_from_dirname("DIGITALIMG") == image_mode.DIGITAL
