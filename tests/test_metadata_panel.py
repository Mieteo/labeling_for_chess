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
