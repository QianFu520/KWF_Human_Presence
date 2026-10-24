# Open issues found during validation

Issues are recorded here when found and are not fixed unless stated. Each one
needs a decision from the team lead before code changes.

| ID | Title | Severity | Status |
|---|---|---|---|
| V-001 | Skipped files are not reported in the job result | Medium | Open, not fixed (decision: record only) |
| V-002 | Species allow-list not applied in the Digital Twin | Medium | Open, not fixed (decision: record only) |
| V-003 | Twin threshold (0.5) differs from the validated threshold (0.25) | Medium | Open, not fixed (decision: record only) |
| V-004 | Human-presence confirmations rose from 0 to 134 when BirdNET was switched on | High | Open, cause not traced |
| V-005 | Dashboard detection count changed from 1,777 to 218 when BirdNET was switched on | Medium | Open, not fixed (decision: record only) |
| V-006 | The API takes file paths only, no upload | Medium | Open, decision needed before integration |
| V-007 | Jobs are kept in memory only | Low | Open, not fixed (decision: record only) |

---

## V-001: Skipped files are not reported in the job result

**Where:** `src/bridge/app/jobs.py`, `src/bridge/app/schemas.py`

**What happens**
- If a file cannot be read (wrong length, stereo, corrupt), the API skips it and continues.
- The job still ends as `status: done`. Nothing in the response says a file was skipped.
- The only trace is a warning line in the service's console log.
- The skipped file is also counted in `completed`.

**Evidence (from the code)**
- `jobs.py`, job loop: on `AudioError` or `RuntimeError`, `job.failures.append(...)`, `job.completed += 1`, `continue`.
- `jobs.py`, end of the job: `job.status = JobStatus.done`, then `log.warning("... files skipped")`.
- `schemas.py`, `JobState` (the `GET /jobs/{id}` response) has no `failures` field, so the collected messages never leave the service.
- `src/bridge/README.md` documents the skipping ("Bad files are skipped, logged, counted in `completed`"), but not that the caller cannot see it.
- `verify_bridge_parity.py` tells the user to look at "failures on the /jobs response". That field does not exist.
- Earlier test with a stub model: a wrong-length clip returned `done` with 0 windows and no error.
- Confirmed on the real FP16 model (run `20261022_160717`, `interface_cases.csv`):
  - 10 of 10 files that cannot be scored finished as `status: done`, 0 windows scored, empty `error`.
  - 5 real wrong-length clips from `{DRIVE_DATA_DIR}/filtered_clips_6` (3 with 143,360 samples, 2 with 147,456; 48 kHz mono 16-bit) and 5 stereo copies of good clips.
  - A mixed job (1 good, 1 wrong length, 1 stereo) returned `completed 3/3` with 1 window scored. The good clip was scored correctly.

**Impact**
- A caller, such as the Digital Twin, cannot tell "no species detected" from "file rejected".
- A job where every file fails still finishes as `done`.
- Does not affect the scores of files that are read. All validation results so far are unaffected, because all 709 clips were read.

**Suggested fix (not applied)**
- Add `failures` (and a skipped count) to `JobState` and return them from `to_state()`.
- Decide the final status: `done` with failures when some files fail, `error` when no file is scored.
- Optionally include the skipped count in the `done` stream event.
- Update `src/bridge/README.md` and the saved OpenAPI contract, and extend `smoke_test.py`.
- Re-run `run_validation.py` afterwards, because the response format changes.

**Decision needed from the team lead**
- Should a job with any skipped file end as `done` or `error`?
- Is adding fields to the response acceptable for existing callers? (Adding fields should not break them.)

---

## V-002: Species allow-list is not applied in the Digital Twin

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

**Decision needed from the team lead**
- Keep the Twin unfiltered, or apply the allow-list?
- If applied, Option A or B?

---

## V-003: Twin threshold (0.5) differs from the validated threshold (0.25)

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

**Next step (not done)**
- If the lead names a threshold, re-run the Twin with it. The patch passes `birdnet_threshold` through, so no code change is needed.

**Decision needed from the team lead**
- Which threshold should the Twin use with the compressed model?

---

## V-004: Human-presence confirmations rose from 0 to 134 when BirdNET was switched on

**Where:** Digital Twin human-presence stage (`HumanPresenceAdapter`)

**What happens**
- Same recording, same settings.
- Run 3 (BirdNET skipped): 0 `human_presence_confirmed`, 1,777 `human_presence_candidate`.
- Run 4 (compressed BirdNET): 134 `human_presence_confirmed`, 1,643 candidate.
- The stage reads BirdNET's species and confidence, so BirdNET output changes its result.

**Not known**
- Which species or events led to the 134 confirmations.
- Whether the 134 are correct. The original BirdNET was never run through the Twin, so there is no comparison.

**Impact**
- Human presence is the purpose of the project. A change of this size from enabling BirdNET needs an explanation before the Twin is relied on.

**Next step (not done)**
- Read `stage_3_birdnet_timeline.json` and the human-presence stage output for run 4. List the species and confidences on the 134 events and check them against the stage's sentinel-species rules.

**Decision needed from the team lead**
- Is a human-presence result driven by BirdNET output expected?

---

## V-005: Dashboard detection count changed from 1,777 to 218 when BirdNET was switched on

**Where:** Digital Twin dashboard and `_event_label` (`backend/app/services/audio_processing.py`)

**What happens**
- Run 3 (BirdNET skipped): "Detections" 1,777. Every AED-confirmed event is labelled Wildlife and counted.
- Run 4 (BirdNET on): "Detections" 218 = 84 `bird_confirmed` + 134 `human_presence_confirmed`.
- The 1,693 `bird_candidate` events are not counted, although the AED gate confirmed them as bird calls.

**Impact**
- Switching BirdNET on makes the headline number fall by about 88% with no change in the audio.
- The two runs cannot be compared on the dashboard count alone.

**Decision needed from the team lead**
- Should `bird_candidate` events be counted as detections?
- Which count should be treated as the Twin's result?

---

## V-006: The API takes file paths only, no upload

**Where:** `src/bridge/app/schemas.py` (`paths` in the job request), `src/bridge/app/jobs.py`

**What happens**
- `POST /jobs` takes `paths`: absolute paths to WAV files or folders on the API's own disk.
- There is no way to send audio bytes in the request.
- The Digital Twin patch works around this by writing each clip to a temporary WAV and sending its path. That only works when the Twin and the API share a filesystem.

**Impact**
- A caller on another machine or in another container cannot use the API.
- The AED API takes uploaded files, so the two components differ in how they are called.

**Suggested fix (not applied)**
- Add an upload endpoint (multipart WAV) next to the path input. Keep paths for benchmarking.

**Decision needed from the team lead**
- Path-only or upload for integration? Needed before the integration work.

---

## V-007: Jobs are kept in memory only

**Where:** `src/bridge/app/jobs.py` (job store held in a dictionary in the running process)

**What happens**
- Job status and results live in the service's memory.
- They do not survive a restart of the service.

**Impact**
- A restart loses all job results, so a caller must read a result before the service stops.
- Low for benchmarking and the validation runs. It matters for a long-running deployment.

**Decision needed from the team lead**
- Is in-memory enough for the integration, or should results be stored?
