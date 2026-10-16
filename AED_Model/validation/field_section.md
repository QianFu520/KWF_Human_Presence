## Using real field audio (KWF AudioMoth)

Paths and label files come from `AED_Model/config.py`:
`LABELS_PROGRESS_PATH` (outputs/labels_progress.csv), `EVENT_TABLE_PATH`
(data/event_table.csv) and `AUDIO_DIR` (the streamed Google Drive folder
`Audio Moth Data/Segmented Foldered`). Only clips listed in the manifest are
read, so only those stream down from Drive.

From `AED_Model/`:

```bash
# 1. Pick a labelled sample (default: 2 events per type, 500 BirdNET, 500 background)
python validation/build_field_manifest.py --check_files

# 2. In a second terminal: start the service
uvicorn api.main:app --port 8000

# 3. Run the checks against the running service (task 4)
python validation/validate_aed.py --aed_root . \
    --manifest validation/field/manifest.csv \
    --api_url http://localhost:8000 --skip_workflow --out validation/results

# 4. Score per event type and threshold; writes field_scores.md in the run folder
python validation/score_field.py --run validation/results/<stamp> \
    --manifest validation/field/manifest.csv

# 5. Digital Twin (task 5): pick raw recordings and write their ground-truth logs
python validation/make_twin_logs.py
```

Labels used: event windows from `event_table.csv` (meaningful),
`birdnet_species` (meaningful), `background_flatness` / `background_energy`
(not_meaningful). Labels from `model_inference_v1..v3` are left out because
earlier AED versions produced them.

A manifest is a CSV with these columns (paths absolute or relative to the CSV):

```csv
clip_name,path,expected_label,group,source,event_id,recorder
```

`expected_label` is `meaningful`, `not_meaningful`, or blank / `ambiguous`
(scored but left out of metrics). `group` drives the per-group tables.
