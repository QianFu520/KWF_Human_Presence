# AED in the inference pipeline (Weeks 6–7)

Status: 2026-11-05, first pass on `main` at `3656528`. Covers the Weeks 6–7
AED tasks: AED in `inference_pipeline/run_pipeline.py`, its preprocessing and
output schema in the integrated run, the AED → BirdNET handoff, and an
integration test.

## What the pipeline runs today

- **Step 2** (`step2_filter`) calls `score_folder` from
  `inference_pipeline/scripts/aed_inference.py` with
  `inference_pipeline/models/tinycnn_v3.pth`, writes `tinycnn_predictions.csv`
  and returns the list of clips labelled `meaningful`.
- **Step 3** (`step3_birdnet`) runs BirdNET on **every** clip in the folder. It
  takes the meaningful list as an argument but does not use it.
- **Step 4** runs on every clip; **step 7** keeps only the meaningful clips
  (`keep_clips=meaningful_clips`). That is the one place clips are dropped.

## Checks (all pass)

| Check | Result |
|---|---|
| Pipeline checkpoint is the validated v3 file | SHA-256 `913a78a4…`, identical to `AED_Model/outputs/models/tinycnn_v3.pth` |
| Pipeline AED code = `AED_Model/aed_inference.py` | Only the import lines differ (`architecture.py` sits next to the script); `preprocess_clip`, `predict` and all constants identical. `architecture.py` identical |
| Step 2 output schema | `clip_name` (with `.wav`), `audio_path`, `not_meaningful_prob` in [0, 1], `predicted_label` (`not_meaningful` if prob >= 0.5), `threshold`, `model_version`; one row per clip |
| Step 2 → Step 3 | Same clip set, same `clip_name` format (with `.wav`), one row per clip in each, joins one-to-one |
| Real clips | Steps 2–3 on Ben's 12 AM1 setup clips: AED scores identical to Ben's notebook and to `QUIET_AUDIO.md` (0.56–1.00); BirdNET found no allow-list species; files join on all 12 |

Test: `inference_pipeline/tests/test_aed_handoff.py` (6 tests; generated
3 s clips, no field audio, about 3 s). Full suite: 28 passed. Environment:
`inference_pipeline/requirements.txt` in a fresh venv (Python 3.13).

## Issues for the integration

1. **Mark, don't drop (step 7, Patrick).** Step 7 still drops non-meaningful
   clips. Under the new decision it should keep every clip and carry
   `not_meaningful_prob` (and `predicted_label`) into `features.csv` so HP
   sees the score. AED's output already has what step 7 needs; the join key is
   `clip_name` with `.wav`.
2. **Unused argument (step 3, Ben).** `step3_birdnet(clips_dir, meaningful_clips, …)`
   ignores `meaningful_clips`, and the docstring and comment still say
   "meaningful clips only". Running BirdNET on every clip is right under the
   new decision; the argument and comment should go. Same for step 4.
3. **`clip_name` format.** Steps 2 and 3 write `.wav`; step 4 does not (Ben's
   notebook). Slide 8 decision 5. AED can follow either; the test checks that
   steps 2 and 3 agree.
4. **`model_version` is a free-text note**, not an identifier:
   "TinyCNN v3 — 11213 not_meaningful clips (…)". Propose `tinycnn_v3` plus the
   checkpoint hash prefix (`913a78a4`), matching whatever Patrick's contract
   uses for versions.
5. **Two copies of the AED code and checkpoint** (`AED_Model/` and
   `inference_pipeline/`). They match today; the test now fails if they drift.
   Longer term, one copy imported by both would be safer.
6. **Row order.** `tinycnn_predictions.csv` is not in time order (documented by
   Ben). Not a problem for joins; step 7 sorts before `Volume_Spike_15s`.
7. **How AED is called (slide 8, decision 1).** The pipeline imports AED as a
   Python function, matching Qian's direction (plain functions with a thin
   FastAPI wrapper). The API change to file paths waits for Patrick's contract.

## Not done yet

- A full step 1–3 run on Ben's shared test recording (Audio_Moth_5, 19 March)
  is waiting on Google Drive, which is not serving uncached files.
