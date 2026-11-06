# Compressed BirdNET (FP16, 493 species): Validation Report

Validation of the compressed BirdNET model as served by `src/bridge` (FastAPI)
and as used inside the Digital Twin.

| | |
|---|---|
| Model under test | `birdnet_fp16_pruned493.onnx` |
| Results run | `validation/results/20261106_162306/`, run from commit `219bc94` with no uncommitted changes (after the V-001 fix) |
| Criteria status | Approved by the team lead (`validation/config/criteria.json`) |

## 1. Summary

- **Verdict:** every gated criterion passes. The API reproduces the original BirdNET v2.4 on all 709 test clips.
- **Fidelity:** top-1 species matches on 709 of 709 clips. Confidence is within 0.05 on 709 of 709 and within 0.01 on 689 of 709. Largest difference: 0.029.
- **Digital Twin:** with the compressed model switched on, checks T1 to T5 pass on a 6.9 h recording (1,777 events, no interface, label or hand-off errors).
- **Issues:** seven recorded in `validation/ISSUES.md` (V-001 to V-007). None changes the scores of files that are read. Three are resolved: V-001 is fixed and verified in this run, V-004 has a traced cause (the Twin's placeholder human-presence formula, not BirdNET), and V-006 is closed by decision. Four are open: V-002, V-003 and V-005 sit with the Twin owner, and V-007 is record only.

## 2. Scope

In scope:
- The FP16 pruned model and its 493-species label file.
- The API in `src/bridge`, used unchanged.
- The BirdNET step of the Digital Twin.

Out of scope: the FP32 model, accuracy against human labels, and any change to existing repo code.

The ground truth is the output of the original BirdNET v2.4 (`birdnet` package v0.1.7), recorded in `verification_data/phase4_detail_Audio_Moth_6.csv`. The test shows that compression did not change what the model says. It does not show that BirdNET is correct.

## 3. Artifacts under test

| Item | Value |
|---|---|
| Model | `birdnet_fp16_pruned493.onnx`, 18,878,769 bytes |
| Model SHA-256 | `c6a5dcbbd2f23a5886b0abe640d37e16993359746383ede139be9c735db714d8` |
| Labels | `birdnet_fp16_pruned493.labels.txt`, 493 entries |
| Labels SHA-256 (LF-normalized) | begins `72b910ab9eae575b`, matches `docs/phase4_dual_model_notes.md` |
| Input | float32 raw audio, shape (N, 144000): 3 s at 48 kHz |
| Output | (N, 493) logits, converted to confidence by the API |
| Detection threshold | 0.25 in the API; 0.1 floor in the response |
| Allow-list | 218 species, marked as `in_allow_list`, not filtered |

No model hash is documented in the repo. The hash above was recorded in this validation.

Allow-list behaviour differs between components. The API marks species outside the allow-list (`in_allow_list: false`) and still returns them. The pipeline script `inference_pipeline/scripts/run_birdnet.py` drops them by default (`--no_allow_list` keeps them). The team lead plans to standardize this.

## 4. Test data

| | |
|---|---|
| Clips | 709 from `{AUDIO_DIR}/Audio Moth 6`, where `{AUDIO_DIR}` is `Audio Moth Data / Segmented Foldered` on the team Drive |
| Format | 3 s, 48 kHz, mono, 144,000 samples (288,044 bytes) |
| Identity | `validation/config/audio_manifest_am6_709.csv` lists name, size and SHA-256 for each clip |
| Reference | `ref_top1` and `ref_conf` columns of `verification_data/phase4_detail_Audio_Moth_6.csv` |

| Wrong-length clips | 5 real clips from `{DRIVE_DATA_DIR}/filtered_clips_6`, another team's older cut (143,360 and 147,456 samples) |
| Digital Twin input | `node_SHAMAN.WAV`, 2,399,338,821 bytes, SHA-256 `D564289D5925AF1744F0AE41B5721AA8CC1BE309969047F239AFC596CB1E98F8`, 6.9 h |

## 5. Pass criteria

Thresholds come from numbers already recorded in this repo (Phase 4, 51,720 clips). None was tuned to this result. Sources are in `validation/config/criteria.json`.

| ID | Check | Threshold | Gated |
|---|---|---|---|
| C0 | Model size and label hash match the documented values | exact | Yes |
| C1 | FP16 model and labels are a valid pair (14 checks in `scripts/check_model_pairing.py`) | 0 failures | Yes |
| C2 | API reproduces the recorded Phase 4 FP16 output (`src/bridge/verify_bridge_parity.py`) | max difference <= 1e-6 | Yes |
| C3 | API top-1 species equals the original model's | >= 99.8% | Yes |
| C4 | Confidence within 0.05 of the original | >= 99.9% | Yes |
| C5 | Confidence within 0.01 of the original | >= 98.6% | No |
| C6 | Detection set at 0.25 equals the recorded FP16 set | 100% | Yes |
| C8 | Wrong-length and stereo files are never scored | 10 of 10 | Yes |
| C9 | The caller is told when a file is rejected, and a partly failed job lists the skipped files | 10 of 10, plus the mixed job | No |

C5 is reported but not gated. The recorded Phase 4 value for this recorder alone is 97.18%, already below the 98.6% population figure.

## 6. Results

### 6.1 API against the original BirdNET (C3 to C6)

| Metric | Result | Criterion |
|---|---|---|
| Top-1 species agrees | 709 / 709 (100.0%) | C3: pass |
| Confidence within 0.05 | 709 / 709 (100.0%) | C4: pass |
| Confidence within 0.01 | 689 / 709 (97.18%) | C5: below 98.6%, not gated |
| Detection set equals recorded FP16 | 709 / 709 (100.0%) | C6: pass |
| Reference species missing from output | 0 | |
| Largest confidence difference | 0.0294 (`Audio_Moth_6_20250321_094620.wav`) | |

The 20 clips outside 0.01 are all within 0.03. This is FP16 rounding and matches the recorded Phase 4 value (97.18%).

### 6.2 Identity and parity (C0 to C2)

- C0: model is 18,878,769 bytes and the label hash matches. Pass.
- C1: 14 checks on the FP16 pair, 0 failed. Pass.
- C2: API output equals the recorded Phase 4 FP16 output on all 709 clips. Maximum difference 0, zero species mismatches. Pass.

### 6.3 Interface and preprocessing

| Case | Result |
|---|---|
| 5 real wrong-length clips | Not scored (0 windows). Job ends `error`, and the error names the file. |
| 5 stereo files made from good clips | Not scored (0 windows). Job ends `error`, and the error names the file. |
| Mixed job (1 good, 1 wrong length, 1 stereo) | `done`, 3 of 3 completed, 1 window scored, `skipped` = 2 with both files in `failures`. The good clip is scored correctly. |

C8 passes: bad files are never scored. C9 passes: every rejection is visible to the caller (V-001 fixed). In the previous run, before the fix, all 10 bad files ended as `done` with no error, so C9 failed. The scores in 6.1 and 6.2 are identical in both runs.

Effect of the Twin's preprocessing on the 709 clips, compared with unchanged clips:

| Transform | Top-1 changed | Largest confidence shift |
|---|---|---|
| Peak normalization | 0 / 709 | 0.0012 |
| Normalization then 16-bit write (as the Twin does) | 0 / 709 | 0.0068 |

The peak-normalization gain ranged from 1.0 to 25.7 (median 7.0). The effect on the model output is negligible.

### 6.4 Digital Twin (run 3 against run 4)

Same recording, same settings. Run 3 has BirdNET skipped (`skip_birdnet` is hardcoded True in `backend/app/routes/runs.py`). Run 4 has BirdNET on, sent to the FP16 model through the API on port 8001. The Twin was patched locally with `validation/digital_twin/birdnet_compressed.patch`. The patch was never committed or pushed. Twin commit: `7340e82`.

| | Run 3 (skipped) | Run 4 (compressed BirdNET) |
|---|---|---|
| Stage-1 candidates | 1,820 | 1,820 |
| Confirmed bird calls (AED gate) | 1,777 | 1,777 |
| Events with a species (>= 0.5) | 0 | 84 (11 species, confidence 0.503 to 0.999) |
| Dashboard "Detections" | 1,777 | 218 |
| `human_presence_confirmed` | 0 | 134 |
| Processing time | 22.5 min | 51.0 min |

Checks on run 4 (`validation/scripts/twin_check.py`, output in `validation/digital_twin/results/run4_twin_check.txt`):

| Check | Result |
|---|---|
| T1 interface: one API call per event, none missing or repeated | 1,777 of 1,777 |
| T2 label mapping: every label returned is in the 493 file | 613 of 613 |
| T3 hand-off: Twin result equals best logged detection >= 0.5 | 1,777 of 1,777 |
| T4 confidence range | 0.1000 to 0.9987 |
| T5 re-score 50 saved clips directly through the API | identical |

Model identity in run 4 is proven by hash: the file loaded by the API is the validated one.

### 6.5 Serving cost

Measured on 709 clips, one thread, Windows 11 CPU.

| Metric | This run (`20261106_162306`) | Earlier run (`20261022_160717`) |
|---|---|---|
| Throughput | 4.95 windows/s | 17.1 windows/s |
| Inference per window | mean 199.6 ms, p95 219.3 ms | mean 57.5 ms, p95 81.2 ms |
| Peak memory | 212 MB | 211 MB |

These are one-machine figures, recorded for reference and not pass/fail. The two runs used the same model, code path and machine, and speed differed by about 3.5 times. The cause was not investigated (machine load and power state were not recorded), so treat the speed figures as indicative only. Peak memory was stable.

## 7. Open items

Issues (details, evidence and suggested fixes in `validation/ISSUES.md`):

| ID | Issue | Severity | Status |
|---|---|---|---|
| V-001 | Skipped files are not reported in the job result | Medium | Fixed, verified in run `20261106_162306` |
| V-002 | Species allow-list not applied in the Digital Twin | Medium | Open, with the Twin owner |
| V-003 | Twin threshold (0.5) differs from the validated threshold (0.25) | Medium | Decided: 0.25 on our side. Twin owner to change 0.5 |
| V-004 | Human-presence confirmations rose from 0 to 134 with BirdNET on | High | Cause found: the Twin's placeholder formula, not BirdNET |
| V-005 | Dashboard detections fell from 1,777 to 218 with BirdNET on | Medium | Open, with the Twin owner |
| V-006 | The API takes file paths only, no upload | Medium | Closed: file paths for all services |
| V-007 | Jobs are kept in memory only | Low | Open, record only |

Decisions made by the team lead:
- Pass criteria approved. C5 stays reported, not gated.
- V-001: a job ends `done` if at least one file was scored and `error` if none was. Adding response fields is acceptable.
- V-003: 0.25 everywhere on our side. The Twin's 0.5 is raised with its owner, together with V-002 and V-005.
- V-002: recommended for the Twin is to drop detections with `in_allow_list: false`, as `inference_pipeline/scripts/run_birdnet.py` does.
- V-006: file paths, not uploads, for all services.

Still open:
- Twin threshold, allow-list and dashboard count (V-002, V-003, V-005): with the Twin owner.
- Whether the Twin calls the API over HTTP or loads the ONNX model itself: decided in the integration work (Weeks 6 to 8).
- Shared API contract (`/health`, `/predict`, `/jobs`) for AED, BirdNET and human presence: Patrick is drafting it. BirdNET has no `/predict` endpoint today.
- The pruned model dropped the non-bird classes (human vocal, human non-vocal, engine, siren). A full-model test is under way and a re-export may be requested. The validation scripts do not depend on the model file.

## 8. Not covered

- Recorders other than Audio_Moth_6, and one recording on one Twin node.
- A Twin run where the model is loaded in process instead of through the API.
- Accuracy against human-labelled data.

## 9. How to reproduce

Full steps are in `validation/README.md`. In short:

1. Get the 709 clips from Drive and stage them with `validation/scripts/stage_clips.py`. Use `--manifest` to confirm the files match.
2. Start the API (`src/bridge`) with a `models.json` that points at the FP16 model and labels.
3. Run `validation/scripts/run_validation.py`. Results are written to `validation/results/<UTC time>/`, with `SUMMARY.md` as the overview.
4. For the Twin: apply `validation/digital_twin/birdnet_compressed.patch` to a local Twin clone with `git apply --ignore-whitespace`, run the API on port 8001, then check with `validation/scripts/twin_check.py`.

Expected: the same pass/fail and percentages. Small float differences are possible on another CPU, which is why C2 allows 1e-6.

If the label file hash does not match on Windows, check for CRLF line endings. The validation compares the hash after converting to LF.

## 10. Files

| Path | Content |
|---|---|
| `validation/README.md` | How to run the validation |
| `validation/ISSUES.md` | Issues V-001 to V-007, open and resolved |
| `validation/config/criteria.json` | Pass criteria with sources |
| `validation/config/audio_manifest_am6_709.csv` | Test audio identity |
| `validation/scripts/` | Staging, run, comparison and Twin check scripts |
| `validation/results/20261106_162306/` | Raw output, JSON, CSV and `SUMMARY.md` of the run after the V-001 fix |
| `validation/results/20261022_160717/` | The earlier run, before the V-001 fix |
| `validation/digital_twin/` | Patch, baseline and run 4 notes, evidence PDFs, raw run 4 files |
| `docs/BirdNET_Service_Setup_and_Validation.md` | Earlier setup and test plan (written before the real-model results) |
