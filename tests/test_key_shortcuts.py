from chess_labeler.key_shortcuts import (
    BOARD_REGION_CLASS,
    CLASS_DISPLAY_ORDER,
    HAND_CLASS,
    ROLE_BY_LETTER,
    display_label_for_class,
    resolve_piece_class,
)


def test_all_role_letters_lowercase_are_black():
    expected = {
        "p": "black_pawn",
        "c": "black_cannon",
        "r": "black_rook",
        "h": "black_horse",
        "e": "black_elephant",
        "a": "black_advisor",
        "k": "black_king",
    }
    for letter, class_name in expected.items():
        assert resolve_piece_class(letter, is_red=False) == class_name


def test_all_role_letters_uppercase_are_red():
    expected = {
        "P": "red_pawn",
        "C": "red_cannon",
        "R": "red_rook",
        "H": "red_horse",
        "E": "red_elephant",
        "A": "red_advisor",
        "K": "red_king",
    }
    for letter, class_name in expected.items():
        assert resolve_piece_class(letter, is_red=True) == class_name


def test_color_is_independent_of_letter_case_passed_in():
    # `letter` itself may arrive in either case (Qt key codes are always
    # the uppercase constant); `is_red` alone decides color.
    assert resolve_piece_class("r", is_red=True) == "red_rook"
    assert resolve_piece_class("R", is_red=False) == "black_rook"


def test_unrecognized_letter_returns_none():
    assert resolve_piece_class("z", is_red=False) is None
    assert resolve_piece_class("b", is_red=True) is None  # "b" no longer means a color step


def test_role_table_has_exactly_the_seven_spec_letters():
    assert set(ROLE_BY_LETTER) == {"P", "C", "R", "H", "E", "A", "K"}


def test_hand_class_constant():
    assert HAND_CLASS == "hand"


def test_board_region_class_constant():
    assert BOARD_REGION_CLASS == "board_region"


def test_display_label_uses_lowercase_letter_regardless_of_color():
    assert display_label_for_class("black_rook") == "r"
    assert display_label_for_class("red_rook") == "r"
    assert display_label_for_class("black_cannon") == "c"
    assert display_label_for_class("red_king") == "k"


def test_display_label_for_hand_is_the_word_hand():
    assert display_label_for_class("hand") == "hand"


def test_display_label_for_board_region_is_board():
    assert display_label_for_class("board_region") == "board"


def test_display_label_empty_for_unassigned_or_unknown():
    assert display_label_for_class(None) == ""
    assert display_label_for_class("") == ""
    assert display_label_for_class("not_a_real_class") == ""


def test_class_display_order_has_all_16_classes_black_then_red_then_hand_then_board_region():
    assert len(CLASS_DISPLAY_ORDER) == 16
    assert len(set(CLASS_DISPLAY_ORDER)) == 16  # no duplicates
    assert CLASS_DISPLAY_ORDER[:7] == [
        "black_pawn", "black_rook", "black_cannon", "black_horse",
        "black_elephant", "black_advisor", "black_king",
    ]
    assert CLASS_DISPLAY_ORDER[7:14] == [
        "red_pawn", "red_rook", "red_cannon", "red_horse",
        "red_elephant", "red_advisor", "red_king",
    ]
    assert CLASS_DISPLAY_ORDER[14] == "hand"
    assert CLASS_DISPLAY_ORDER[15] == "board_region"
