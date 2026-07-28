from chess_labeler.chord import ChordOutcome, ChordShortcutManager


def test_red_cannon_chord_matches_spec_example():
    mgr = ChordShortcutManager()
    ev1 = mgr.key_pressed("R", ctrl_held=True)
    assert ev1.outcome == ChordOutcome.PENDING
    assert mgr.is_pending
    ev2 = mgr.key_pressed("C", ctrl_held=True)
    assert ev2.outcome == ChordOutcome.ASSIGNED
    assert ev2.class_name == "red_cannon"
    assert not mgr.is_pending


def test_all_chords_from_spec_table():
    expected = {
        ("R", "K"): "red_king",
        ("R", "A"): "red_advisor",
        ("R", "E"): "red_elephant",
        ("R", "H"): "red_horse",
        ("R", "C"): "red_cannon",
        ("R", "R"): "red_rook",
        ("R", "P"): "red_pawn",
        ("B", "K"): "black_king",
        ("B", "A"): "black_advisor",
        ("B", "E"): "black_elephant",
        ("B", "H"): "black_horse",
        ("B", "C"): "black_cannon",
        ("B", "R"): "black_rook",
        ("B", "P"): "black_pawn",
    }
    for (step1, step2), class_name in expected.items():
        mgr = ChordShortcutManager()
        mgr.key_pressed(step1, ctrl_held=True)
        ev = mgr.key_pressed(step2, ctrl_held=True)
        assert ev.class_name == class_name, f"Ctrl,{step1},{step2}"


def test_ctrl_h_alone_is_hand_immediately_no_pending():
    mgr = ChordShortcutManager()
    ev = mgr.key_pressed("H", ctrl_held=True)
    assert ev.outcome == ChordOutcome.ASSIGNED
    assert ev.class_name == "hand"
    assert not mgr.is_pending


def test_ctrl_h_after_ctrl_r_is_red_horse_not_hand():
    mgr = ChordShortcutManager()
    mgr.key_pressed("R", ctrl_held=True)
    ev = mgr.key_pressed("H", ctrl_held=True)
    assert ev.outcome == ChordOutcome.ASSIGNED
    assert ev.class_name == "red_horse"


def test_escape_cancels_pending_chord():
    mgr = ChordShortcutManager()
    mgr.key_pressed("B", ctrl_held=True)
    assert mgr.is_pending
    ev = mgr.escape_pressed()
    assert ev.outcome == ChordOutcome.CANCELLED
    assert not mgr.is_pending


def test_escape_when_idle_is_ignored():
    mgr = ChordShortcutManager()
    ev = mgr.escape_pressed()
    assert ev.outcome == ChordOutcome.IGNORED


def test_releasing_ctrl_mid_chord_cancels():
    mgr = ChordShortcutManager()
    mgr.key_pressed("R", ctrl_held=True)
    ev = mgr.key_pressed("C", ctrl_held=False)
    assert ev.outcome == ChordOutcome.CANCELLED
    assert not mgr.is_pending


def test_timeout_cancels_pending_chord():
    mgr = ChordShortcutManager(timeout_seconds=1.0)
    mgr.key_pressed("R", ctrl_held=True, now=0.0)
    ev = mgr.check_timeout(now=2.0)
    assert ev.outcome == ChordOutcome.CANCELLED
    assert not mgr.is_pending


def test_unrecognized_step2_key_cancels():
    mgr = ChordShortcutManager()
    mgr.key_pressed("R", ctrl_held=True)
    ev = mgr.key_pressed("Z", ctrl_held=True)
    assert ev.outcome == ChordOutcome.CANCELLED
    assert not mgr.is_pending


def test_pending_indicator_shows_color():
    mgr = ChordShortcutManager()
    mgr.key_pressed("B", ctrl_held=True)
    assert mgr.pending_indicator == "B…"
