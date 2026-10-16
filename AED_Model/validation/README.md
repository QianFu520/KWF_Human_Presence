# AED validation harness

Checks that the AED component (TinyCNN v3) works correctly as a deployable
service and writes a reproducible validation record. It does not retrain or
modify the model.

| File                            | Purpose                                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `validate_aed.py`             | Runs every check and writes the report                                                                                    |
| `make_sample_set.py`          | Builds the sample audio set and its labelled `manifest.csv`                                                             |
| `build_field_manifest.py`     | Picks a labelled sample of KWF AudioMoth clips from the Drive folder                                                      |
| `score_field.py`              | Scores a field run per event type and threshold                                                                           |
| `make_twin_logs.py`           | Picks raw recordings for the Digital Twin and writes their ground-truth logs                                              |
| `score_twin_run.py`           | Runs the Digital Twin on those recordings and scores recall per event type against the logs                               |
| `submit_twin_runs.py`         | Uploads each recording with its log to a running Digital Twin and saves the dashboard's ground-truth result               |
| `requirements-validation.txt` | Pinned environment the record was produced in                                                                             |
| `results/<timestamp>/`        | One folder per run: `report.md`, `checks.csv`, `per_clip.csv`, `summary.json`, plus Digital Twin workflow outputs |

## What it checks

| Section                  | Checks                                                                                                                                                                                              |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A Environment            | Library versions; whether they match `AED_Model/requirements.txt`                                                                                                                                 |
| B Model artifact         | SHA-256 of `tinycnn_v3.pth`, checkpoint metadata, strict load and parameter count, which file the API and pipeline load, ONNX export equals the PyTorch model, Digital Twin bundles the same file |
| C Preprocessing parity   | The mel code in the training notebook (`05_train.ipynb`) is re-run and compared with the labelling notebook, the CLI, the FastAPI service and the Digital Twin's loader, on every clip            |
| D API contract           | `/health`, OpenAPI schema, `/predict` on every clip (status, schema, label vs threshold), error codes, determinism, latency                                                                     |
| E Cross-path consistency | Same clip through CLI, API, ONNX and the Digital Twin gate; probabilities must agree within 1e-4                                                                                                    |
| F Behaviour probes       | End-of-clip blind spot, gain invariance, zero-padding, silence (reported, not failed)                                                                                                               |
| G Metrics                | Confusion matrix, threshold sweep and per-group results against `manifest.csv` labels                                                                                                             |
| H Digital Twin workflow  | Runs the twin's own `run_node_audio_workflow` on a long file built from the clips, re-scores every AED event independently, and reports which clips reached AED                                   |

A check is PASS, FAIL, WARN (works, but something to raise) or INFO. The run
exits 1 if anything FAILs.

## Run it

From `AED_Model/`:

```bash
# 1. Environment (Python 3.10+)
python -m venv .venv-validation && source .venv-validation/bin/activate
pip install -r validation/requirements-validation.txt

# 2. Sample audio (downloads 128 ESC-50 clips, about 40 MB; --no_esc50 for synthetic only)
python validation/make_sample_set.py --out validation/sample_audio

# 3. Digital Twin backend (only for sections B6-B7, C5-C6, E4, G4, H)
git clone --depth 1 https://github.com/KashmirWorld/digital_twin_mvp ../../digital_twin_mvp

# 4. Validate
python validation/validate_aed.py \
    --aed_root . \
    --manifest validation/sample_audio/manifest.csv \
    --twin_backend ../../digital_twin_mvp/backend \
    --out validation/results
```

To test a running server instead of the in-process app, start
`uvicorn api.main:app --port 8000` in `AED_Model/` and add
`--api_url http://localhost:8000`.

## Using real field audio (KWF AudioMoth)

Paths and label files come from `AED_Model/config.py`:
`LABELS_PROGRESS_PATH` (outputs/labels_progress.csv), `EVENT_TABLE_PATH`
(data/event_table.csv) and `AUDIO_DIR` (the streamed Google Drive folder
`Audio Moth Data/Segmented Foldered`). Only clips listed in the manifest are
read, so only those stream down from Drive.

From `AED_Model/`:

```bash
# 1. Pick a labelled sample (default: 2 events per type, 500 BirdNET, 500 background).
#    --check_files stats every clip on Drive and can stall for an hour; validate_aed.py
#    reports unreadable clips anyway, so it is optional.
python validation/build_field_manifest.py

# 1b. Report metrics on clips v3 never trained on. Every labelled KWF clip was in v3's
#     training or validation split, so the default manifest is mostly training data.
#     --heldout reproduces the split (v3_split.py, checked against the checkpoint's
#     n_train/n_val) and keeps only validation clips; this takes every held-out event.
python validation/build_field_manifest.py --heldout --events_per_type 0 --max_clips_per_event 0 \
    --n_background 0 --out validation/field/manifest_heldout.csv

# 2. Every check on KWF clips, including the Digital Twin parity checks (C5, E4, G4, H1)
python validation/validate_aed.py --aed_root . \
    --manifest validation/field/manifest.csv \
    --twin_backend ../../digital_twin_mvp/backend --out validation/results

# 3. The API checks again against a running server (uvicorn api.main:app --port 8000)
python validation/validate_aed.py --aed_root . \
    --manifest validation/field/manifest.csv \
    --api_url http://localhost:8000 --skip_workflow --out validation/results

# 4. Score per event type and threshold; writes field_scores.md in the run folder
python validation/score_field.py --run validation/results/<stamp> \
    --manifest validation/field/manifest.csv

# 5. Digital Twin (task 5): pick raw recordings and write their ground-truth logs
python validation/make_twin_logs.py

# 6. Run them through a running twin (cd digital_twin_mvp && docker compose up --build -d),
#    one run per recording with its log, exactly as the Create Run page does
python validation/submit_twin_runs.py --out validation/results/twin_deployed_<stamp>

# 7. Add the chance baseline and stage breakdown to those runs. Copy each run's
#    digital_twin_mvp/backend/audio_workflow_outputs/run_<id>/SHAMAN/backend_payload.json
#    to <out>/<recording>/ first; without it the script runs the twin's code itself.
python validation/score_twin_run.py --twin_backend ../../digital_twin_mvp/backend \
    --threshold 0.75 --out validation/results/twin_deployed_<stamp>
```

The Create Run page sends its *Confidence Threshold* (default 0.75) as the AED
gate threshold; 0.3 is only the backend fallback when none is sent. Use 0.75
to match the dashboard.

### Uploading to the Digital Twin

`validation/field/twin_logs/recordings_to_upload.csv` lists each raw recording
and its `*_log.json`. Create **one run per recording**: upload the WAV as the
only node and choose its log under *Ground Truth Log (optional)*. The twin
scores every detection in a run against the single log, so putting two
recordings in one run mixes their timelines.

The log times are seconds from the start of the uploaded file, so upload the
recording unedited. A trimmed file needs its `timestamp_seconds` shifted by the
amount cut from the start.

### Reading the results

AED passes all meaningful audio, including birds, so on field recordings:

- **Recall per event type** is the number to use, but read it against
  `chance_recall` in `score_twin_run.py`'s `report.md`. A 5-minute window
  almost always contains some bird detection, so a high recall can come from
  birds rather than from the staged event. `density_ratio` (detections per
  minute inside the windows relative to outside) shows whether the event
  actually raised the detection rate.
- **Precision** looks low: every bird detection outside a staged human event
  counts as a false positive.
- Staged events overlap (gunshots and chainsaws happen during human-presence
  walks), so one detection can count for several types.
- The twin's stage-1 prefilter runs before AED. `report.md` shows whether a
  missed event had no stage-1 trigger or had triggers that AED rejected.

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

## About the sample audio

- `sample_audio/esc50/`: 128 real recordings from
  [ESC-50](https://github.com/karolpiczak/ESC-50) (CC BY-NC 3.0), 8 per class
  from 16 classes. Each was resampled to 48 kHz and cut to its loudest 3 s, so
  it matches the pipeline's clip contract (144,000 samples).
- `sample_audio/synthetic/`: 10 generated files for contract and edge cases.
- ESC-50 is not AudioMoth field audio. Results on it show the service behaves
  sensibly on real sound; they are not a measure of field accuracy.
