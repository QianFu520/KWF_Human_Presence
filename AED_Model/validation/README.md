# AED validation harness

Checks that the AED component (TinyCNN v3) works correctly as a deployable
service and writes a reproducible validation record. It does not retrain or
modify the model.

| File                            | Purpose                                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `validate_aed.py`             | Runs every check and writes the report                                                                                    |
| `build_field_manifest.py`     | Picks a labelled sample of KWF AudioMoth clips from the Drive folder                                                      |
| `score_field.py`              | Scores a field run per event type and threshold                                                                           |
| `make_twin_logs.py`           | Picks raw recordings for the Digital Twin and writes their ground-truth logs                                              |
| `score_twin_run.py`           | Runs the Digital Twin on those recordings and scores recall per event type against the logs                               |
| `submit_twin_runs.py`         | Uploads each recording with its log to a running Digital Twin and saves the dashboard's ground-truth result               |
| `requirements-validation.txt` | Pinned environment the record was produced in                                                                             |
| `results/<timestamp>/`        | One folder per run. Only `report.md` (and `field_scores.md`) are committed; see [Full results](#full-results) |

## Full results

The repository keeps the scripts and the summary reports (`results/*/report.md`,
`field_scores.md`). Everything else from the recorded runs is on Google Drive:
[AED validation results](https://drive.google.com/drive/folders/1zgaQq6ZMgiorRA2y9es7NyehX__0ooFi). It holds, in the same folder layout:

- per-run `checks.csv`, `per_clip.csv`, `cli_predictions.csv`, `summary.json`;
- the Digital Twin workflow timelines (`twin_workflow/`) and the deployed-twin
  dashboard results (`twin_deployed_20261016/`);
- `field/v3_split.csv` (regenerate with `v3_split.py`);
- the presentation (`AED_Validation_Results.pptx`) and the step 2 evidence
  spreadsheet (`AED_step2_artifact_and_preprocessing.xlsx`).

New runs write the full set locally; `validation/.gitignore` keeps all but the
summary reports out of git.

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

A check is PASS, FAIL, KNOWN (an explained difference within stated bounds),
WARN (works, but something to raise) or INFO. The run exits 1 if anything FAILs.

Known exceptions are listed at the top of `report.md`. There is one:

- **C5, Digital Twin preprocessing.** The twin peak-normalises audio before
  computing the mel spectrogram. On near-silent clips that moves one cell across
  `power_to_db`'s absolute floor (`amin`), so it differs from training by up to
  ~1 dB in that cell. C5 reports KNOWN only while every differing clip has at
  most one cell over tolerance and the twin's label equals the CLI's; anything
  else is still a FAIL.

All paths label a clip `not_meaningful` when `not_meaningful_prob >= threshold`
(default 0.5).

## Run it

All validation runs on KWF AudioMoth field clips. The audio is not in the
repository: it stays on Google Drive (or any local copy) and is found through
`AED_AUDIO_DIR`. Labels come from `AED_Model/config.py`:
`LABELS_PROGRESS_PATH` (`outputs/labels_progress.csv`) and `EVENT_TABLE_PATH`
(`data/event_table.csv`), both local and git-ignored. Only clips listed in the
manifest are read, so only those stream down from Drive.

From `AED_Model/`:

```bash
# 1. Environment (Python 3.10+)
python -m venv .venv-validation && source .venv-validation/bin/activate
pip install -r validation/requirements-validation.txt

# 2. Where the clips are: your copy of "Audio Moth Data/Segmented Foldered"
#    (without it, data/raw_audio/ is used). With Google Drive for desktop:
export AED_AUDIO_DIR="$HOME/Library/CloudStorage/GoogleDrive-<your account>/My Drive/Audio Moth Data/Segmented Foldered"

# 3. Digital Twin backend (only for sections B6-B7, C5-C6, E4, G4, H)
git clone --depth 1 https://github.com/KashmirWorld/digital_twin_mvp ../../digital_twin_mvp

# 4. Validate on the 4,027 held-out clips (v3's validation split, never trained on).
#    The manifest is committed and stores {AUDIO_DIR} placeholders, so it works as is.
python validation/validate_aed.py --aed_root . \
    --manifest validation/field/manifest_heldout.csv \
    --twin_backend ../../digital_twin_mvp/backend --out validation/results
```

To test a running server instead of the in-process app, start
`uvicorn api.main:app --port 8000` in `AED_Model/` and add
`--api_url http://localhost:8000`.

The first read of each clip from Drive is slow; mark the folders *Make available
offline* first, or expect a long first run.

## More field runs

From `AED_Model/`:

```bash
# 1. Rebuild the held-out manifest (needs outputs/labels_progress.csv and data/event_table.csv).
#    --heldout reproduces v3's train/validation split (v3_split.py, checked against the
#    checkpoint's n_train/n_val) and keeps only validation clips.
python validation/build_field_manifest.py --heldout --events_per_type 0 --max_clips_per_event 0 \
    --n_background 0 --out validation/field/manifest_heldout.csv

# 1b. A smaller sample (2 events per type, 500 BirdNET, 500 background). Every labelled
#     KWF clip was in v3's training or validation split, so this one is mostly training data.
#     --check_files stats every clip on Drive and can stall for an hour; it is optional.
python validation/build_field_manifest.py

# 2. Every check, including the Digital Twin parity checks (C5, E4, G4, H1)
python validation/validate_aed.py --aed_root . \
    --manifest validation/field/manifest_heldout.csv \
    --twin_backend ../../digital_twin_mvp/backend --out validation/results

# 3. The API checks again against a running server (uvicorn api.main:app --port 8000)
python validation/validate_aed.py --aed_root . \
    --manifest validation/field/manifest_heldout.csv \
    --api_url http://localhost:8000 --skip_workflow --out validation/results

# 4. Score per event type and threshold; writes field_scores.md in the run folder
python validation/score_field.py --run validation/results/<stamp> \
    --manifest validation/field/manifest_heldout.csv

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
