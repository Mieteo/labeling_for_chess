from chess_labeler.key_shortcuts import HAND_CLASS, ROLE_BY_LETTER, resolve_piece_class


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
