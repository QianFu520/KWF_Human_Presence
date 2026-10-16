# AED validation: handoff

Status as of 2026-10-16. Covers getting the AED service running locally, pointing
the project at the audio on Google Drive, validating every check on KWF field
audio, and task 5: field recordings through the deployed Digital Twin with
ground-truth logs. **None of it is committed yet**; see
[Uncommitted changes](#uncommitted-changes).

## Summary

- The FastAPI service (`api/main.py`) runs locally. `/` now redirects to `/docs`.
- `config.AUDIO_DIR` and every `audio_path` in `outputs/labels_progress.csv`
  point at the Google Drive clip folder.
- **All audio-based checks now run on KWF AudioMoth clips; ESC-50 is retired.**
  The headline set is 4,027 clips TinyCNN v3 never trained on (its reproduced
  validation split), covering all five staged event types.
- The deployed twin (Docker) was run on 5 field recordings, one run per recording
  with its ground-truth log. The dashboard reports 15/16 events found; an
  independent re-score reproduces it exactly. Recall is no better than chance on
  field audio, because AED passes all wildlife; see [Task 5 results](#task-5-results).

## Task checklist and final result

| # | Task | Status | Evidence |
|---|---|---|---|
| 1 | Review implementation and docs | Done | Code, notebooks, docs and checkpoints read |
| 2 | Correct model artifact; preprocessing matches | Done | B1–B7 PASS (SHA-256 `913a78a4…`, twin bundles the identical file); C2–C4 bit-identical to training over 4,027 KWF clips; C5 see issues |
| 3 | FastAPI endpoint, request and response | Done | D1–D5 PASS on 4,027 KWF clips, in-process **and** against a live uvicorn server |
| 4 | Test locally | Done | KWF held-out clips: CLI/API/ONNX/twin agree within 5e-5; metrics below |
| 5 | Rerun through the Digital Twin | Done | Twin parity on KWF clips (C5, E4, G4, H1); 5 field recordings through the deployed twin, dashboard = independent re-score |
| 6 | Behaviour, consistency, interface/preprocessing issues | Done | See [Issues found](#issues-found) |
| 7 | Document setup, procedure, outputs, metrics, issues, result | Done | This file, `README.md`, each run's `report.md` |

**Final result: PASS.** The service loads the correct v3 checkpoint;
preprocessing is identical across training, labelling, CLI and API, and the twin
agrees to within 3e-6 in probability; the API contract holds over HTTP; the
deployed twin's dashboard is reproduced exactly. The harness reports C5 as FAIL
on field audio: one spectrogram cell in 3 very quiet clips differs, with no
effect on any probability (explained under issues). On held-out KWF clips AED
keeps 99.0% of meaningful clips and drops 100% of background. Weak spot: the
quiet off-trail walk on Audio_Moth_3 (66% of its clips kept at 0.5).

### Field-clip metrics (held out from training)

`results/20261016_044454` + `score_field.py`. 4,027 clips from v3's validation
split (`v3_split.py` reproduces it; the counts match the checkpoint exactly).

| Group | Clips | Events | Correct @0.5 | @0.7 | @0.95 |
|---|---|---|---|---|---|
| Background (rule-based) | 586 | – | 100% | 99.8% | 99.8% |
| BirdNET birds | 500 | – | 100% | 100% | 100% |
| Chainsaw | 198 | 2 | 100% | 100% | 100% |
| Gunshot | 198 | 2 | 100% | 100% | 100% |
| Human presence, off trail | 99 | 1 | 65.7% | 73.7% | 84.8% |
| Human presence, on trail | 2,248 | 8 | 100% | 100% | 100% |
| Vehicle | 198 | 2 | 100% | 100% | 100% |

At 0.5: meaningful recall 99.0% (3,407/3,441), background recall 100%
(586/586). Twin gate: at 0.3 keeps 3,415/3,441 meaningful and drops 585/586
background; at the UI default 0.75 keeps 3,397/3,441 and drops 586/586.

Read with care: a clip inside an event window is "meaningful" whether AED keyed
on the human sound or on birds and insects in the same clip, so 100% here shows
AED does not drop event audio, not that it recognises the human activity. v3 was
also chosen over v4 on this split, so it is held out from training, not from
model selection.

The default 1,818-clip field manifest (`results/20261016_042947`) is 77%
training data and is kept only as a check; do not quote its metrics.

### Validation runs

| Run folder | Audio | Target | Result |
|---|---|---|---|
| `results/20261016_044454` | **4,027 held-out KWF clips** | In-process API + twin backend, section H | 23 pass, **1 fail (C5, explained)**, 2 warn |
| `results/20261016_045042` | 4,027 held-out KWF clips | Live server `uvicorn --port 8011` (`--api_url`, `--skip_workflow`) | PASS: 19 pass, 0 fail, 2 warn |
| `results/20261016_042947` | 1,818 KWF clips (default manifest, mostly training data) | In-process + twin | 23 pass, 1 fail (C5), 2 warn |
| `results/twin_deployed_20261016` | 5 raw field recordings, 46 h | Deployed twin (Docker), dashboard | Dashboard = independent re-score, exactly |
| `results/20261015_231906`, `20261015_231946` | ESC-50 sample set (retired) | Mac in-process / live server | PASS; kept for the record |

Commands (from `AED_Model/`):

```bash
python validation/build_field_manifest.py --heldout --events_per_type 0 --max_clips_per_event 0 \
    --n_background 0 --out validation/field/manifest_heldout.csv
python validation/validate_aed.py --aed_root . --manifest validation/field/manifest_heldout.csv \
    --twin_backend ../../digital_twin_mvp/backend --out validation/results
python validation/score_field.py --run validation/results/<stamp> --manifest validation/field/manifest_heldout.csv
# live server: uvicorn api.main:app --port 8011, then
python validation/validate_aed.py --aed_root . --manifest validation/field/manifest_heldout.csv \
    --api_url http://localhost:8011 --skip_workflow --out validation/results
```

Reading 4,000 clips one at a time from Drive took hours (each first read waits
seconds). Pre-reading them with 32 parallel threads took 10 minutes; after that
the harness runs in about 10 minutes.

### Harness fixes made

- **Manifest clips were never scored by the CLI.** `validate_aed.py` ran
  `score_folder()` on the folder holding the manifest file. That worked for the
  sample set (clips beside the manifest) but scored 0 clips for Drive paths, and
  E2–E4 then passed on empty comparisons because `max()` ignores NaN. Now manifest
  clips go through the CLI's own `ClipFolderDataset` → `predict` path
  (`cli_score_clips`), NaN scores are skipped, and E2–E4 fail if nothing was compared.
- **Overlapping events lost their clips.** `build_field_manifest.py` kept the
  first label per clip, so vehicles (staged during walks) vanished into
  "on trail". Specific events now win over the surrounding walk.
- `build_field_manifest.py --heldout` (via `v3_split.py`); `--n_birdnet 0` /
  `--n_background 0` now mean "all".

### Issues found

None block deployment.

| Check | Issue |
|---|---|
| C5 | **Twin preprocessing differs on very quiet clips.** 3 of 4,027 KWF clips (pre-dawn background, peak ≈0.026 vs typical 0.18) have one mel cell below librosa's power floor (`amin` 1e-10) in training; the twin peak-normalises first, so that cell is not clamped: up to 1.08 dB in 1 of 32,768 cells. Probabilities identical (E4 ≤ 3e-6). The other 4,024 clips differ by < 1e-3 dB |
| G3 | **Quiet off-trail walk partly scored as background.** Audio_Moth_3, 19 March 08:01: 34 of 99 clips scored not_meaningful at 0.5. The twin also missed this event (no stage-1 trigger) |
| F1 | **End-of-clip blind spot.** The mel is cropped to 256 frames (2.73 s of 3 s); a short sound after ~2.6 s is not seen |
| E5 | **Threshold comparison differs.** CLI `prob >= t`, API `prob > t`; 0 disagreements on 4,027 clips |
| — | **Operating points.** API 0.5 on not_meaningful_prob; the twin keeps meaningful confidence ≥ 0.75 when the run comes from the UI (0.3 is only the backend fallback) |
| Field | AED kept 98% of the twin's stage-1 triggers on field recordings; the prefilter decides what is flagged |
| C6 | Twin resampling for non-48 kHz input differs from training (5.6 dB on a 44.1 kHz test file). Not applicable to AudioMoth audio, which is always 48 kHz; not exercised on KWF data |
| A2 | This Mac's torch/librosa/numpy are newer than the pins; scores unchanged |

## Data paths in committed files

Manifests and ground-truth logs store audio paths as placeholders, expanded by
`validation/paths.py` from `config.AUDIO_DIR`:

- `{AUDIO_DIR}/Audio Moth N/<range>/<clip>.wav`: the 3 s clips
  (`Segmented Foldered`)
- `{RAW_AUDIO}/Audio Moth N/Audio_Moth_N_MMDDYY/<file>.WAV`: raw recordings
  (the folder above `AUDIO_DIR`)

**Set `AUDIO_DIR` in `AED_Model/config.py` to your own copy of
`Audio Moth Data/Segmented Foldered` before running anything**; the committed
`config.py` still points at `/Users/qian/KWF/Segmented_Foldered`. The run
reports use the same placeholders, plus `<repo>` and `<digital_twin_mvp>` for
local checkouts.

## Environment

- Use the conda Python (`/opt/miniconda3/bin/python`, 3.13). `python3` on this
  Mac is the system 3.9 with no project packages.
- Work from `AED_Model/`. Paths such as `outputs/models/tinycnn_v3.pth` are
  relative to it.
- `python-multipart` must be installed for the API (it is in
  `api/requirements.txt`, not in the top-level `requirements.txt`).
- Digital Twin backend: cloned to `~/digital_twin_mvp` (commit `7340e82`), i.e.
  `../../digital_twin_mvp/backend` from `AED_Model/`. Its audio workflow
  imports fine in the conda Python.
- Audio is on Google Drive (streamed, not local):
  `~/Library/CloudStorage/GoogleDrive-<your account>/My Drive/Audio Moth Data/`
  - `Segmented Foldered/Audio Moth N/<range>/<clip>.wav`: the 3 s clips
  - `Audio Moth N/Audio_Moth_N_MMDDYY/YYYYMMDD_HHMMSS.WAV`: raw recordings
    (2–4 GB each)
  - Drive delivered 1.3–3 MB/s. Long jobs are I/O bound; mark folders
    *Make available offline* first. Drive occasionally stops serving files
    entirely; restarting the Drive app fixed it.

## Running the API

```bash
cd AED_Model
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# browser: http://127.0.0.1:8000  (redirects to /docs)
curl -X POST http://127.0.0.1:8000/predict -F "file=@/path/to/clip.wav"
# -> {"label":"meaningful","not_meaningful_prob":0.0123}
```

Only `GET /`, `GET /health`, `GET /docs` and `POST /predict` (`.wav` only)
exist; anything else returns `{"detail":"Not Found"}`.

## Data paths

- `config.py`: `AUDIO_DIR` now points at the Drive `Segmented Foldered` folder.
  Nothing in the code reads `AUDIO_DIR` directly; notebooks use `audio_path`
  from the CSV.
- `outputs/labels_progress.csv` (gitignored): `audio_path` rewritten from
  `/Users/qian/KWF/Segmented_Foldered` to the Drive folder for all 631,317 rows,
  by plain text replacement (every other column byte-identical). The backup made
  at the time has since been removed; reverse the replacement if the old paths
  are ever needed.
- Label state in that file: 118,940 meaningful, 16,845 not_meaningful (including
  5,632 `model_inference_v3`), 495,532 unknown.

### Inputs still missing (only matter for rerunning the notebooks)

| Input | Used by | Notes |
|---|---|---|
| `data/clip_catalog.csv` | `01_labeling` | Drive `bioacoustics_dataset.csv` has the same columns except `audio_path` |
| Team Jacama `full_predictions.csv` | `01_labeling` | Its `model_confidence` is already merged into labels_progress |
| `outputs/spectrograms/*.npy` | `05_train` | Regenerable from the clips (~136k files over Drive) |
| `outputs/inference_v1.csv` (and v2/v3) | `06_inference_labeling` | Drive has `all_inference_v3.csv` (no `audio_path`) |
| `birdnet_audit.csv`, `lowscore_audit.csv`, `inference_audit.csv`, `data/fp_review/` | `02`, `03`, `06`, diagnosis | Hand-tagged; cannot be regenerated. Ask Qian |
| Hard-coded paths | `04_background_harvest` (`/Users/qian/KWF/raw_recordings/...`), `01_labeling`, ONNX notebook (`/Volumes/TOSHIBA EXT/...`) | Raw WAVs are on Drive under `Audio Moth N/` |

Retraining from labels_progress.csv as it stands trains on v4's data; drop the
`model_inference_v3` rows to reproduce v3.

## Task 5: Digital Twin with ground truth

### Ground-truth logs

`validation/make_twin_logs.py` writes one `*_log.json` per raw recording into
`validation/field/twin_logs/`, plus `recordings_to_upload.csv`. Format matches
the twin's `backend/app/services/aed/ground_truth.py` (identical to the copy in
`validation/ground_truth.py`):

```json
{"metadata": {"duration_seconds": 26620.582, "node_id": "Audio_Moth_5", "source_audio": "...", "recording_start": "2025-03-21T06:25:45"},
 "events": [{"timestamp_seconds": 8109.0, "duration_seconds": 297.0, "type": "CHAINSAW", "event_id": "..."}]}
```

`timestamp_seconds` = `Sim_Start` − recording start (from the WAV file name);
`duration_seconds` = `Sim_End` − `Sim_Start`, clipped to the recording.
Verified: for all 16 events the log window covers exactly the clips that
labels_progress tags with that simulation (event table and AudioMoth file names
use the same local clock).

| Recording | Hours | Events |
|---|---|---|
| Audio_Moth_2_20250318_062543 | 11.6 | Human presence on trail |
| Audio_Moth_3_20250319_080049 | 10.0 | Gunshot, human presence off + on trail |
| Audio_Moth_5_20250317_112812 | 6.5 | Chainsaw, gunshot, human presence on trail |
| Audio_Moth_5_20250319_062544 | 10.1 | Human presence on trail ×2, vehicle |
| Audio_Moth_5_20250321_062545 | 7.4 | Chainsaw, gunshot ×2, human presence off + on trail, vehicle |

### How the twin uses the log (from its code)

- `routes/runs.py::_evaluate_ground_truth_safe` scores **every** detection in a
  run (all nodes) against the **one** log. So: **one run per recording**, with
  that recording as the only node.
- Detections are the dashboard events (`bird_confirmed`, `birdnet_skipped`,
  `human_presence_confirmed`) at `trigger_time_s`, i.e. seconds from the start
  of the uploaded file. Upload the WAV unedited, or shift the log if trimmed.
- A detection hits an event if it falls in [start − 1.5 s, end + 1.5 s].
- Pipeline: stage-1 bird-like prefilter → AED gate → human presence. Audio the
  prefilter does not trigger on never reaches AED.
- The Create Run page sends its *Confidence Threshold* (default **0.75**) as the
  AED gate threshold; 0.3 is only the backend fallback when none is sent.
- `TYPE_LABELS` only maps `BIRD` and `GUNSHOT`, so field types display as raw
  names (e.g. `Human_Presence_On_Trail`). One-line fix in the twin repo.

### Running it on the deployed twin

```bash
cd ~/digital_twin_mvp && docker compose up --build -d     # backend :8000, UI :5173
cd ~/KWF_Human_Presence/AED_Model
python validation/submit_twin_runs.py --out validation/results/twin_deployed_<stamp>
```

`submit_twin_runs.py` does per recording what the Create Run page does: upload
the WAV (`/api/audio/upload`, streamed with curl), create a run with one SHAMAN
sensor node, Confidence Threshold 0.75, the stage-1 defaults and the log as
`groundTruth`, poll `/api/runs/{id}/status`, then save `/api/runs/{id}/dashboard`.
The twin writes its workflow output to
`digital_twin_mvp/backend/audio_workflow_outputs/run_<id>/SHAMAN/backend_payload.json`.

`score_twin_run.py` adds what the dashboard lacks. Copy each run's payload to
`<out>/<recording>/backend_payload.json` and run it with `--threshold 0.75`
(without a payload it runs the twin's code itself):

- `chance_recall`: expected recall if detections fell at the recording's
  background rate (outside all events): `1 − exp(−rate × window)`.
- `density_ratio`: detections/min inside an event's window ÷ the same
  recording's background rate (1.0 = no different from background).
- `outcome`: detected / missed with no stage-1 trigger / missed because AED
  rejected every trigger.

### Task 5 results

Output: `validation/results/twin_deployed_20261016/` (`dashboard.json` per run,
`dashboard_ground_truth.csv`, `report.md`, `by_type.csv`, `events.csv`).
Runs 1–5 in the twin; each took 4.5–35 min after a ≤1 min upload.

Cross-checks:

- The deployed twin (Linux CPU in Docker) and the earlier local run (Mac) scored
  the same 4,527 stage-1 triggers at the same timestamps; AED probabilities agree
  within 1.2e-6.
- `score_twin_run.py` on the deployed payloads reproduces the dashboard's recall,
  precision and detection count exactly for all 5 runs.

Dashboard at threshold 0.75, with the baseline:

| Event type | Events | Found | Recall | Chance recall | Rate vs background |
|---|---|---|---|---|---|
| Chainsaw | 2 | 2 | 100% | 100% | 0.29× |
| Gunshot | 4 | 4 | 100% | 99% | 1.68× |
| Human presence, off trail | 2 | 1 | 50% | 98% | 0.10× |
| Human presence, on trail | 6 | 6 | 100% | 100% | 1.71× |
| Vehicle | 2 | 2 | 100% | 100% | 2.87× |

Per recording: precision 3–10%, background 0.7–3.8 detections/min. AED kept
4,441 of 4,527 triggers (98%); its scores sit near 0 or 1, so 0.3 vs 0.75 changes
almost nothing.

Findings:

1. **Recall is not evidence of detection here.** Chance recall is 0.98–1.00 for
   every type; any 5-minute window is hit by wildlife detections.
2. **The detection-rate ratio is the informative number.** Vehicles, gunshots
   and on-trail walks raised the rate in their windows; chainsaws and off-trail
   walks were below background. On Audio_Moth_5 21 March an overlapping
   chainsaw, gunshot and off-trail walk at 08:40 drew only 1–3 detections yet
   counts as detected for all three types.
3. **The one miss is a prefilter miss.** The Audio_Moth_3 off-trail walk had no
   stage-1 trigger. It is also AED's weakest held-out event at clip level (66%).
4. **AED is a pass-through on field audio** behind the prefilter.
5. **Overlapping events** mean one detection can count for several types.
6. The dashboard shows field types as raw names (`Human_Presence_Off_Trail`)
   because `TYPE_LABELS` only maps BIRD and GUNSHOT.

## Open items

- Add field event types to the twin's `TYPE_LABELS`.
- Decide whether the twin should report a background-adjusted metric (chance
  recall or density ratio) for field runs; plain recall is misleading here.
- Cover all 63 staged events: 22 recordings, 67 GB (`make_twin_logs.py` takes a
  count per type; needs a small change to take every recording).
- For cleaner per-type recall, stage events that do not overlap, or add a
  quiet control window per recording.
- Look at the quiet off-trail walk (Audio_Moth_3, 19 March 08:01): partly
  scored as background by AED and missed by the twin's prefilter.
- Recover the hand-tagged audit CSVs from Qian if the labelling history needs
  to be reproducible.
- Stop the twin when done: `cd ~/digital_twin_mvp && docker compose down`. Its
  uploads (~15 GB) are under `backend/uploads/`.

## What is in the aed-validation branch

| File | Change |
|---|---|
| `validation/validate_aed.py` | Manifest clips scored through the CLI path; empty comparisons fail; placeholder paths |
| `validation/build_field_manifest.py` | `--heldout`; specific event wins over walk; `0` = all; writes placeholder paths |
| `validation/v3_split.py` | New: reproduces v3's train/validation split |
| `validation/paths.py` | New: `{AUDIO_DIR}` / `{RAW_AUDIO}` placeholders |
| `validation/score_twin_run.py` | New: twin field scoring with chance baseline |
| `validation/submit_twin_runs.py` | New: runs recordings through the deployed twin |
| `validation/make_twin_logs.py`, `ground_truth.py`, `score_field.py`, `make_sample_set.py` | Logs with placeholder paths; the rest as before |
| `validation/README.md`, `HANDOFF.md`, `field_section.md` | Docs |
| `validation/field/` | Manifests, ground-truth logs, `v3_split.csv` |
| `validation/results/` | Run reports, checks, per-clip scores, twin dashboard results |

Not in the branch:

- `*.wav` (ignored by `AED_Model/.gitignore`; `twin_long_input.wav` is 1.9 GB for
  the 4,027-clip run) and, via `validation/.gitignore`, the twin's
  `backend_payload.json` files (~10 MB each) and `results/logs/`. Rerun to
  regenerate them.
- Local edits outside `validation/`: `config.py` (`AUDIO_DIR` → Drive),
  `api/main.py` (`GET /` → `/docs`), and edits to `docs/MODEL_TRAINING.md` and
  `docs/aed_inference.md` that were not part of this work.
