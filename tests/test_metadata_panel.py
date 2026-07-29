from PySide6.QtWidgets import QApplication

from chess_labeler import metadata
from chess_labeler.metadata_panel import MetadataPanel


def _ensure_qapp():
    return QApplication.instance() or QApplication([])


def test_capture_dropdown_taxonomy_matches_the_persisted_schema():
    _ensure_qapp()
    panel = MetadataPanel()
    try:
        for field, allowed in metadata.CAPTURE_ENUMS.items():
            combo = panel._capture_controls[field]
            displayed = {combo.itemText(index) for index in range(combo.count())}
            assert displayed == allowed
            assert combo.currentText() == "unknown"
        assert panel.values()["board"]["side_to_move"] is None
    finally:
        panel.close()


def test_metadata_workflow_splits_required_capture_and_optional_review_fields():
    _ensure_qapp()
    panel = MetadataPanel()
    try:
        tabs = panel._workflow_tabs
        assert [tabs.tabText(index) for index in range(tabs.count())] == [
            "Cần hoàn tất",
            "Điều kiện chụp",
            "Bổ sung & review",
        ]
        assert tabs.currentIndex() == 0

        essential, capture, details = (tabs.widget(index) for index in range(3))
        assert essential.isAncestorOf(panel._orientation)
        assert capture.isAncestorOf(panel._capture_controls["lighting"])
        assert details.isAncestorOf(panel._capture_controls["shadow"])
        assert details.isAncestorOf(panel._notes)
    finally:
        panel.close()
