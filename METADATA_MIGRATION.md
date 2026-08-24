# Metadata schema v2 migration

Schema v2 adds optional fields under `capture`:

```json
{
  "content_cohort": null,
  "style_or_app": null,
  "capture_session": null,
  "position_id": null
}
```

`content_cohort` accepts only `real`, `native_screenshot`,
`procedural_render`, `screen_photo`, or `unknown`. `null` means the image has
not been classified. It is intentionally different from `unknown`, which is
an explicit annotator choice for unclear or mixed content.

The labeler safely reads v1 and v2 sidecars. Loading a v1 file changes nothing
on disk and exposes its cohort as **Chưa gán nhãn** in the UI. When the user
edits and saves that sidecar, it is atomically written as v2, retaining FEN,
corners, review fields, `capture_group`, and all other v1 values; the new
fields remain `null` unless entered by the annotator. The tool never derives a
cohort from EXIF, filenames, or image heuristics.

The optional manifest exporter is available as:

```powershell
python -m chess_labeler.manifest <image_directory> <output_manifest.json>
```

It includes `content_cohort`, available optional fields, and counts for each
cohort plus `unassigned`. It does not choose scanner routes or modify labels.
