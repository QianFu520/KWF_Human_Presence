# Digital Twin baseline: BirdNET skipped

What the Digital Twin does today, before any change. BirdNET is **not run** in
these results, so they say nothing about BirdNET's output. They are the "before"
that the BirdNET-on run is compared with.

## Twin version
- Repo: `KashmirWorld/digital_twin_mvp`, commit `7340e820195ca868d68c503366f3dfd9a0b7c9a6`
- Local copy had one modified file: `frontend/package-lock.json` (changed by installing the frontend, not a code change).

## Why BirdNET is off
- `backend/app/routes/runs.py` (about line 230) sets `skip_birdnet=bool(True)`. It is hardcoded; the UI has no switch.
- With the skip on, an event that passes the AED gate is not sent to BirdNET. The stage output records `birdnet_status: skipped`.
- The dashboard label "Wildlife" is made by `_event_label` in `backend/app/services/audio_processing.py` when the event is saved.

## Run
- Run 3, node `SHAMAN`, input `node_SHAMAN.WAV`
  - size 2,399,338,821 bytes
  - SHA-256 `D564289D5925AF1744F0AE41B5721AA8CC1BE309969047F239AFC596CB1E98F8`
  - about 6.9 hours of audio
- UI settings, entered on the run form (no screenshot saved): confidence threshold 0.75 (this is the AED gate threshold), sample rate 48,000. The "detection sensitivity" field is not used by the backend.
- Processing time 22.5 minutes; 8,331 three-second windows scanned (dashboard).

## Result (`node_run_summary.json`, verbatim except the user folder in the path, shown as `~`)
```json
{
  "node_id": "SHAMAN",
  "run_id": "3",
  "input_audio": "~\\twin\\uploads\\run_3\\node_SHAMAN.WAV",
  "n_stage_1_candidates": 1820,
  "n_stage_2_birdcall_confirmed": 1777,
  "n_birdnet_events": 1777,
  "n_human_presence_events": 1777,
  "tinycnn_model_status": "aed_tinycnn",
  "birdnet_status": "skipped",
  "human_presence_status": "proxy_feature_scorer"
}
```
- 1,820 candidates from stage 1, 1,777 confirmed as bird calls by the AED model.
- All 1,777 pass through the BirdNET stage as skipped, and appear on the dashboard as "Wildlife".
- Dashboard figures for the same run: 1,777 detections, 1,820 clips scored by the AED model, mean detection confidence 97.9% (this is the AED model's confidence, not BirdNET's), inference latency 67 ms on CPU, 6.9 h processed in 22.5 min, 8,331 windows scanned, 1,820 prefilter triggers.
- No species name and no BirdNET confidence exists anywhere in this output.

## Earlier check: short clip
- A single 3-second clip produced 0 stage-1 triggers and no result. Stage 1 needs long recordings, so short clips cannot exercise the pipeline.

## What this baseline does and does not show
- Shows: the integration point, the label the Twin gives when BirdNET is skipped, and the counts to compare against.
- Does not show: anything about BirdNET's species or confidences. That needs the BirdNET-on run.

## Evidence (`evidence/`)
- `Digital Twin _ Model Performance Dashboard.pdf`: the dashboard for run 3, printed 2026-10-16 from `localhost:5173`. Shows the detection, clip, audio-time and window counts quoted above.
- `node_run_summary.json`: the Twin's own run summary, quoted verbatim above.
- Not saved as files: the large stage outputs (`stage_1` to `stage_4`, `backend_payload.json`, 1 to 17 MB each) and the 2.4 GB input audio. The note gives their counts and the audio hash instead.
