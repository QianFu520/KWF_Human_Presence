# Compressed BirdNET validation

Validation of the compressed BirdNET model (FP16, pruned from 6,522 to 493
species) as served by `src/bridge`. Goal: show, reproducibly, that the
compressed component behaves like the original BirdNET v2.4 and that the
service around it adds no error.

**Status:** pass criteria are PROPOSED and wait for team lead approval
(`config/criteria.json`). Results below are not final until then.

## What is validated, and against what

| Question | Reference | Where |
|---|---|---|
| Is this the right model and label file? | sizes, shapes, dtypes, label count and order checks | `scripts/check_model_pairing.py` |
| Does the service return what the recorded Phase 4 FP16 run returned? | `verification_data/phase4_detail_Audio_Moth_6.csv` | `src/bridge/verify_bridge_parity.py` |
| Does the service agree with the original BirdNET? | `ref_top1`, `ref_conf` and recorded FP16 detection sets in the same CSV | `validation/scripts/service_vs_reference.py` |
| Does it work inside the Digital Twin? | local, throwaway patch | `validation/digital_twin/` (not done yet) |

FP32 is out of scope; the compressed model under test is FP16.

Ground truth is the original model's output, not human labels. This is a
fidelity test: did compression change what the model says.

Interface and preprocessing checks: wrong-length and stereo input (is it
scored, and does the caller see the rejection), and the effect of the Twin's
peak normalization and 16-bit write on the model's output.

Not covered yet: recorders other than Audio_Moth_6 and the Digital Twin run.

## Layout

```
validation/
  README.md
  ISSUES.md               problems found, with evidence and the decision needed
  requirements.txt        client-side packages for these scripts
  config/criteria.json    pass criteria, with the source of each number
  config/audio_manifest_am6_709.csv   clip names, sizes and SHA-256 of the test audio
  scripts/
    stage_clips.py        copy the clips a CSV names into one flat folder
    run_info.py           record hashes, versions, git commit, audio manifest
    service_vs_reference.py
    interface_and_preprocessing.py   wrong-length, stereo, and the Twin's normalization / 16-bit steps
    run_validation.py     runs everything below and writes a results folder
  results/<UTC time>/     one folder per run: raw output, JSON, CSV, SUMMARY.md
  digital_twin/           Digital Twin patch and results
```

## Test data

Audio is not in this repo. It lives in the team Google Drive, so you need
access to it first (ask the data owners).

Placeholders used below:
- `{AUDIO_DIR}`: the team Drive folder `Audio Moth Data / Segmented Foldered`, or your own copy
  of it (the same `AUDIO_DIR` as in `AED_Model/config.py`).
- `{DRIVE_DATA_DIR}`: the team Drive data folder that holds `filtered_clips_6` (ask the data
  owner for the path). Not the same as `{AUDIO_DIR}`.

**Correct clips (used for every pass/fail test)**
- Drive: `{AUDIO_DIR}/Audio Moth 6`
- The 709 clips are the ones named in `verification_data/phase4_detail_Audio_Moth_6.csv`.
- Each is a 3 s, 48 kHz mono clip of exactly 144,000 samples (288,044 bytes).
- `config/audio_manifest_am6_709.csv` lists every clip with its size and SHA-256,
  so you can confirm you have the same 709 files.

**Wrong-length clips (extra, real-world example)**
- Source: `{DRIVE_DATA_DIR}/filtered_clips_6`, another team's clips on the team Drive.
- They are valid for that team's use, but they are not 144,000 samples, so they are
  the wrong length for this validation.
- Download any few and point the interface test at them. The test reads each
  header and records the real length, so the case is checked, not assumed.

**Get the 709 clips onto your machine**
1. Make the `Audio Moth 6` folder reachable as a local folder (Google Drive for
   Desktop), or copy it into a Colab session with the Drive mount.
2. Stage only the clips you need and check them against the manifest:
   ```
   python validation/scripts/stage_clips.py --source "{AUDIO_DIR}/Audio Moth 6" \
       --csv verification_data/phase4_detail_Audio_Moth_6.csv \
       --manifest validation/config/audio_manifest_am6_709.csv --out <path>/am6_709
   ```
   Expect `requested 709, found 709, missing 0` and `manifest check: 0 file(s) differ`.

**What to expect:** with the same 709 files and the pinned package versions,
the summary table matches the committed result (same pass/fail, same
percentages). Tiny float differences are possible on another CPU, which is why
C2 allows 1e-6.

## How to run

1. Python 3.11 or newer. Create a clean environment and install:
   `pip install -r src/bridge/requirements.txt -r validation/requirements.txt`
2. Copy `src/bridge/models.json.example` to a models.json whose paths point at
   the real model, label and allow-list files.
3. Start the service in its own terminal and leave it running:
   ```
   set BRIDGE_MODELS_CONFIG=<path>\models.json
   uvicorn --app-dir src/bridge app.main:app --port 8000
   ```
   On Windows do not click inside that window. Selecting text pauses it.
4. Run:
   ```
   python validation/scripts/run_validation.py --audio-root <path>/am6_709 \
       --models-dir <path>/models --models-config <path>/models.json \
       --wrong-length-dir <path>/wrong_length
   ```
   `wrong_length` holds a few real wrong-length WAVs, downloaded from
   `{DRIVE_DATA_DIR}/filtered_clips_6` (see Test data). Without it the wrong-length case is skipped.
5. Read `validation/results/<time>/SUMMARY.md`. Exit code 0 means every gated
   criterion passed.

## Results

Run `20261022_160717` (commit `b208532`): all gated criteria pass. See `results/20261022_160717/SUMMARY.md` and `docs/BirdNET_Validation_Report.md`.

## Known gaps in this validation

Problems found in the service are listed in `ISSUES.md`.

- Pass criteria are not yet approved.
- C5 (confidence within 0.01) is reported, not gated, because the recorded
  Phase 4 value for this recorder is already below the population figure.
- The service and the recorded Phase 4 run share preprocessing code, so
  `C2` cannot detect a mistake in that shared step. `C3` to `C6` compare
  against the original model's recorded output, which covers it for these clips.
- One recorder, one date, 709 clips.
