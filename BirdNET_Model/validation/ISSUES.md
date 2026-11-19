# Issues found during validation

Open issues are listed first. Resolved issues are kept at the end so the IDs used in the
report and slides stay valid.

## Open

| ID | Title | Severity | Status |
|---|---|---|---|
| V-002 | Species allow-list not applied in the Digital Twin | Medium | Open, with the Twin owner. Recommended: drop `in_allow_list: false` detections |
| V-003 | Twin threshold (0.5) differs from the validated threshold (0.25) | Medium | Decided: 0.25 on our side. Twin owner to change 0.5 |
| V-005 | Dashboard detection count changed from 1,777 to 218 when BirdNET was switched on | Medium | Open, with the Twin owner |
| V-007 | Jobs are kept in memory only | Low | Open, record only |

## Resolved

| ID | Title | Severity | Resolution |
|---|---|---|---|
| V-001 | Skipped files are not reported in the job result | Medium | Fixed. Verified in run `20261106_162306` |
| V-004 | Human-presence confirmations rose from 0 to 134 when BirdNET was switched on | High | Cause found. Not a BirdNET defect |
| V-006 | The API takes file paths only, no upload | Medium | Closed by decision: file paths for all services |

---

## Open issues

### V-002: Species allow-list is not applied in the Digital Twin

**Where:** Digital Twin `backend/app/audio_workflow/` (BirdNET confirm step); the API only marks `in_allow_list`

**What happens**
- The Twin has no species allow-list, before or after the validation patch.
- The original Twin BirdNET step runs birdnetlib with no location or date, so no regional filter is applied.
- The patched step keeps this behaviour on purpose (one change at a time). The API returns `in_allow_list` per detection, and the patch records it but does not act on it.
- The 218-species allow-list is only used by this repo's API and validation, not by the Twin.

**Evidence**
- Search of the Twin `backend/` for allow-list terms: no match (only CORS `allow_origins`, `allow_methods` in `main.py`).
- Run 4 (compressed model, BirdNET on): 2 of 84 confirmed events are Oilbird, which is outside the 218-species list.

**Impact**
- Species outside the study-area list can be confirmed and counted on the dashboard.
- Small in run 4 (2 of 84), but not checked at scale.

**Suggested fix (not applied)**
- Option A: in the Twin hand-off, drop detections where `in_allow_list` is false.
- Option B: pass the node location and date to BirdNET, as the reference results were produced.

**Status**
- Open. The Twin is owned by another team. The team lead is raising it with the Twin owner, together with V-003 and V-005.
- Recommended: Option A, so the Twin matches `inference_pipeline/scripts/run_birdnet.py`, which drops species outside the allow-list by default.
- Whether the API itself should drop them is decided later, in the integration work (Weeks 6 to 8).

---

### V-003: Twin threshold (0.5) differs from the validated threshold (0.25)

**Where:** Digital Twin BirdNET step (`birdnet_threshold`); API default 0.25

**What happens**
- The Twin keeps a BirdNET detection only at 0.5 or above. This is the Twin's existing default, and the validation patch did not change it.
- The validation (709 clips) used 0.25, the API's detection threshold.
- Run 4 was run at 0.5 for that reason: it is what the Twin does today.

**Evidence (run 4)**
- At 0.5: 84 events get a species.
- At 0.25 it would be 172. 88 events have a best detection between 0.25 and 0.5.

**Impact**
- The Twin runs at a setting the validation did not cover. Accuracy at 0.5 was not measured separately.
- The count of species events roughly halves compared with 0.25.

**Decision**
- 0.25 everywhere on our side (API default, validation, `inference_pipeline`).
- The Twin's 0.5 is an existing default that this repo does not own. The team lead is raising it with the Twin owner, together with V-002 and V-005. Recommended threshold for the Twin: 0.25.

**Next step (not done)**
- Once the Twin owner changes it, re-run the Twin at 0.25. The patch passes `birdnet_threshold` through, so no code change is needed on our side.

---

### V-005: Dashboard detection count changed from 1,777 to 218 when BirdNET was switched on

**Where:** Digital Twin dashboard and `_event_label` (`backend/app/services/audio_processing.py`)

**What happens**
- Run 3 (BirdNET skipped): "Detections" 1,777. Every AED-confirmed event is labelled Wildlife and counted.
- Run 4 (BirdNET on): "Detections" 218 = 84 `bird_confirmed` + 134 `human_presence_confirmed`.
- The 1,693 `bird_candidate` events are not counted, although the AED gate confirmed them as bird calls.

**Impact**
- Switching BirdNET on makes the headline number fall by about 88% with no change in the audio.
- The two runs cannot be compared on the dashboard count alone.

**Status**
- Open. Twin code, owned by another team. The team lead is raising it with the Twin owner, together with V-002 and V-003.
- Open question for the Twin owner: should `bird_candidate` events be counted, and which count is the Twin's result?

---

### V-007: Jobs are kept in memory only

**Where:** `src/bridge/app/jobs.py` (job store held in a dictionary in the running process)

**What happens**
- Job status and results live in the service's memory.
- They do not survive a restart of the service.

**Impact**
- A restart loses all job results, so a caller must read a result before the service stops.
- Low for benchmarking and the validation runs. It matters for a long-running deployment.

**Status**
- Open, low. Recorded only, no change planned.
- Revisit in the integration work if the service runs long-term.

---

## Resolved issues

### V-001: Skipped files are not reported in the job result (fixed)

**Where:** `src/bridge/app/jobs.py`, `src/bridge/app/schemas.py`

**Problem**
- If a file cannot be read (wrong length, stereo, corrupt), the API skipped it and the job still ended as `status: done`. The only trace was a warning in the service's console log, so a caller could not tell "no species detected" from "file rejected".
- Confirmed on the real FP16 model (run `20261022_160717`, `interface_cases.csv`): 10 of 10 files that cannot be scored finished as `done` with 0 windows and an empty `error`. These were 5 real wrong-length clips from `{DRIVE_DATA_DIR}/filtered_clips_6` and 5 stereo copies of good clips. A mixed job (1 good, 1 wrong length, 1 stereo) returned `completed 3/3` with 1 window scored, and the good clip was scored correctly.
- Scores of files that were read were never affected.

**Fix**
- `JobState` (the `GET /jobs/{id}` response) now has `skipped` (count) and `failures` (one `<file>: <reason>` string per skipped file).
- A job ends `done` if at least one file was scored, and `error` if none was. The error message names the first failure.
- `smoke_test.py` has three new checks (19 of 19 pass). `src/bridge/README.md` documents the new fields and statuses.
- `validation/scripts/interface_and_preprocessing.py` and C9 now also check a partial skip: a job with 1 good and 2 rejected files must end `done` with `skipped` = 2.
- `verify_bridge_parity.py` now fails when `skipped` is above 0. Before, it compared `completed` to the file count, which skipped files also increment.
- `src/bridge/models.json.example` now points to `inference_pipeline/models/` and has no FP32 entry (that file is not in the repo).

**Verification**
- Run `20261106_162306`, commit `219bc94`, no uncommitted changes. All gated criteria pass. Scores are identical to the run before the fix (top-1 709 of 709, within 0.05 on 709, within 0.01 on 689, largest difference 0.0294).
- C9 passes. All 10 files that cannot be scored now end as `error` with the file named in the message. The mixed job (1 good, 2 rejected) ends `done` with `skipped` = 2, both files in `failures`, and the good clip scored.
- Before the fix (run `20261022_160717`) all 10 ended as `done` with no error.

**Not changed**
- The `done` event on `/jobs/{id}/stream` does not carry `skipped`. Only `GET /jobs/{id}` does.
- `failures` is not capped, so a job over thousands of bad files returns a long list on every poll.
- `metrics.clips` still counts skipped files. `metrics.windows` does not.
- Each `failures` entry repeats the file name (`X.wav: X.wav: 143360 samples ...`), because the audio error message already contains it. Cosmetic.
- These are for the shared API contract review.

### V-004: Human-presence confirmations rose from 0 to 134 when BirdNET was switched on (cause found)

**Where:** Digital Twin human-presence stage (`HumanPresenceAdapter` in `backend/app/audio_workflow/node_audio_workflow.py`)

**Cause**
- `HumanPresenceAdapter` is a placeholder scorer ("proxy_feature_scorer"), not the trained human-presence model. It uses BirdNET output as input.
- It sets `Eerie_Silence` = 1 when an event has no sentinel species and either of two conditions holds: the recent sentinel rate (last 100 events) is above 0.05, or the audio is quiet (RMS energy below the larger of 0.005 and 65% of the mean of the last 5 events). A set flag adds 0.65 to the score.
- The sentinel-rate part never fires with BirdNET off, because no event has a species. The quiet-audio part does not depend on BirdNET and fires either way: the 226 flags in run 3 come from it, and none of those events was confirmed.
- With BirdNET on, sentinel species raise the rate, and later events without a sentinel species get flagged by the sentinel-rate part. That is what produces the 134 confirmations.

**Evidence (Twin output `run_3` and `run_4`, `stage_4_human_presence_timeline.json`, 1,777 events each)**

| | Run 3 (BirdNET off) | Run 4 (BirdNET on) |
|---|---|---|
| Events with a sentinel species | 0 | 62 |
| Events with `Eerie_Silence` = 1 | 226 | 531 |
| Events confirmed as human presence | 0 | 134 |

- All 134 confirmed events have `Eerie_Silence` = 1. None has a sentinel species itself.
- Which part set the flag (recomputed from the recorded features, matching the recorded flag on all 1,777 events in both runs):
  - Run 3: 226 flags, all from the quiet-audio part. 0 confirmed.
  - Run 4: 531 flags: 312 from the sentinel-rate part only, 171 from the quiet-audio part only, 48 from both. All 134 confirmed events come from the sentinel-rate part only.
- Recomputing the score with `Eerie_Silence` forced to 0 confirms 0 of 134. The recomputation matches the recorded decision on all 1,777 events in both runs.
- The Twin output files are not copied into this repo (`stage_4` is 5.4 MB).

**Conclusion**
- The 134 confirmations say nothing about BirdNET accuracy or about a trained human-presence model. They come from the placeholder formula.
- No BirdNET change is needed. The Twin's human-presence result should not be used as evidence for or against BirdNET until the trained model replaces the proxy.

### V-006: The API takes file paths only, no upload (closed)

**Decision:** file paths, not uploads, for all services. No upload endpoint is needed. The AED, BirdNET and human-presence services will share one API contract (`/health`, `/predict`, `/jobs`, same result and error format). Models are plain Python functions behind a thin FastAPI wrapper.

**Follow-up:** BirdNET has no `/predict` endpoint today. It will need one when the shared contract is agreed. Not started.
