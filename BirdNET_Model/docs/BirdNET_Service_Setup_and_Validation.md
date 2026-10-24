# BirdNET FastAPI Service: Setup, Test Procedure, Results, Open Issues

Date: 2026-10-16
Scope: BirdNET only (pruned 493-species ONNX). Digital Twin validation not started (no access yet).

---

## 1. Status at a glance

| # | Task | Status |
|---|---|---|
| 1 | Review implementation and docs | Done |
| 2 | Confirm model + 493-species label pairing | Half done. Labels verified. ONNX output width NOT verified (the .onnx files are not in this folder) |
| 3 | Review existing serving code | Done. Reuse `src/bridge/`, nothing new built |
| 4 | FastAPI service with clear I/O | Done (existing service). I/O documented in section 4 |
| 5 | Local test on sample audio | Partly done. Service tested with the stand-in model. Real model + real audio still needed |
| 6 | Digital Twin validation | Blocked on access. Checklist ready in section 7 |
| 7 | Check species, confidences, mapping issues | Blocked on the real model and audio |
| 8 | Documentation | This file |

---

## 2. What BirdNET looks like in this repo

- Model: BirdNET v2.4 exported to ONNX, class-pruned from 6,522 to 493 species.
- Two variants, same labels content:
  - `birdnet_fp16_pruned493.onnx` (18.0 MB). Docs recommend this one for deployment.
  - `birdnet_fp32_pruned493.onnx` (35.82 MB). Use for exact reproducibility or raw-logit work.
- Input: `inputs`, shape `(batch, 144000)`, float32, raw audio. 3.000 s at 48 kHz, mono.
- Output: `scores`, shape `(batch, 493)`, float32, raw logits (not probabilities).
- Confidence = `flat_sigmoid(logits, sensitivity=-1.0)`. Applied in the service, not in the model.
- The mel spectrogram is computed inside the graph. No spectrogram preprocessing is needed.
- Preprocessing chain (bit-exact vs reference): read float32, resample to 48 kHz (scipy FFT resample), split into 144,000-sample windows, model, flat_sigmoid.
- Wrong-length clips are rejected, never padded. Stereo is rejected, never downmixed.
- Operational threshold: 0.25. The reference CSV has no confidence below 0.25.

---

## 3. Model and label pairing (task 2)

Checked on the files that are present:

| Check | Result |
|---|---|
| fp16 labels: 493 lines, 493 unique, none blank | Pass |
| fp32 labels: same | Pass |
| fp16 and fp32 label files identical | Pass |
| fp16 and fp32 mapping CSVs identical | Pass |
| Mapping CSV: 493 rows, `new_index` runs 0..492 with no gaps | Pass |
| Mapping species column equals label file, line by line | Pass |
| Geomodel allow-list: 218 species, all 218 found in labels | Pass |
| Label file is not `labels.txt` (6,362) or `en_us.txt` (6,522) | Pass |

NOT checked (needs the .onnx files):
- ONNX output width is 493.
- ONNX input shape is `(dynamic, 144000)` float32.

The service checks both at startup and refuses to boot on a mismatch, so loading the real models through the service completes this task.

Correct pairing:
- `birdnet_fp16_pruned493.onnx` with `birdnet_fp16_pruned493.labels.txt`
- `birdnet_fp32_pruned493.onnx` with `birdnet_fp32_pruned493.labels.txt`

Assumption to confirm: "the compressed model" means the FP16 pruned model. If you meant FP32 pruned, the steps are identical.

---

## 4. Existing serving code (task 3) and I/O format (task 4)

Decision: reuse `src/bridge/`. It is already a FastAPI service with startup safety gates, the validated preprocessing, and job endpoints. Building a second service would duplicate it and lose the bit-exact validation it inherits.

Startup gates (service will not start if any fails):
- Banned label file (`labels.txt`, or any file with 6,362 entries)
- Label count equals model output width
- Pruned model must use its own `<stem>.labels.txt`
- Input must be `(dynamic, 144000)` float32

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Ready flag, loaded variants, ORT version |
| GET | `/models` | Per variant: class count, file size, allow-list size |
| POST | `/jobs` | Submit analysis, returns `job_id` (HTTP 202) |
| GET | `/jobs/{id}` | Status, metrics, detections |
| GET | `/jobs/{id}/stream` | Server-sent events: progress, result, done, error |
| DELETE | `/jobs/{id}` | Cancel |

Request (`POST /jobs`):

```json
{
  "paths": ["C:/audio/clip_000.wav"],
  "variant": "fp16_pruned493",
  "min_confidence": 0.25,
  "apply_allow_list": true,
  "limit": null,
  "collect_detections": true
}
```

- `paths`: WAV files, or folders (scanned for `*.wav`, not recursive). Paths are on the machine running the service.
- `variant`: key from `models.json`.
- Response: `{"job_id": "...", "status": "queued"}`

Result (`GET /jobs/{id}`), shape confirmed on a live uvicorn server:

```json
{
  "job_id": "...",
  "status": "done",
  "variant": "fp16_pruned493",
  "total": 1,
  "completed": 1,
  "error": null,
  "metrics": {
    "clips": 1, "windows": 1,
    "inference_ms_p50": 0.049,
    "model_file_mb": 0.0,
    "intra_op_num_threads": 1, "inter_op_num_threads": 1
  },
  "detections": [
    {
      "data_id": "clip_000",
      "species": "Accipiter bicolor_Bicolored Hawk",
      "confidence": 0.93,
      "start_time": 0.0,
      "end_time": 3.0,
      "in_allow_list": true
    }
  ]
}
```

(The detection above is an illustrative example of the schema. The live test used a stand-in model, so no real species result exists yet.)

- `species` is the label string exactly as in the label file: `Scientific name_Common name`.
- `in_allow_list` flags species outside the 218-species GeoModel list. Nothing is removed.
- `start_time` / `end_time` are seconds within the source file.

---

## 5. Setup

Python 3.11 is needed. `onnxruntime==1.27.0` does not install on Python 3.10.

```powershell
cd src\bridge
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy models.json.example models.json
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- `models.json` paths are relative to `src/bridge/`. The example points at `../../models/`.
- Interactive docs at http://localhost:8000/docs
- Threads default to 1 for deterministic output. Do not raise them for validation runs.

Pins used for the tests below: Python 3.11.16, onnxruntime 1.27.0, numpy 1.26.4, fastapi 0.143.2.

---

## 6. Test procedure and results (task 5)

### 6a. Done: plumbing test with the stand-in model

Command: `python smoke_test.py` (from `src/bridge/`)

Result: ALL CHECKS PASSED (16 of 16).
- Startup gates: label/output mismatch rejected, deprecated `labels.txt` rejected, valid pair loads with 493 classes, allow-list containment (218).
- Audio gates: wrong-length clip rejected, correct-length clip gives one window.
- API: health, models, job submit/complete, 7 files attempted, bad clip excluded, allow-list flag present, unknown job 404, finished job cancel 409, SSE stream emits frames.

Also run: a real uvicorn server with curl against `/health`, `/models`, `POST /jobs`, `GET /jobs/{id}`. All returned the documented shapes. Unknown variant returns HTTP 400 with a clear message.

What this proves: the service starts, gates fire, endpoints and schemas work.
What this does NOT prove: anything about real predictions. The stand-in model has random weights.

### 6b. Still to do: real model on real audio

Needs: the two .onnx files in `models/`, and a few real clips (exactly 144,000 samples, 48 kHz, mono WAV).

1. Put both .onnx files in `models/`.
2. Start the service. Startup passing means output width = 493 and input contract are confirmed (closes task 2).
3. `GET /models`: expect both variants, `num_classes` 493, `allow_list_size` 218, `file_mb` about 18.00 and 35.82.
4. `POST /jobs` with 3 to 5 real clips, FP16 then FP32.
5. Run the existing parity script on Audio_Moth_6 clips:
   `python verify_bridge_parity.py --audio-root <path to AM6 clips>`
   Expected: 709/709 compared, max abs difference `0.000e+00`, 0 top-1 flips, both variants.
6. Run `python src/pruning/verify_pruned_model.py` if the full 6,522-class FP32 model is available (expected max abs diff 0.000e+00).

---

## 7. Digital Twin validation checklist (tasks 6 and 7)

Do these once access exists. Pass criteria are what the existing docs say to expect.

Interface:
- [ ] Twin calls the service with the request in section 4 and parses the result without changes.
- [ ] Twin can reach the audio paths. The service reads files from its own disk, it does not accept uploads.
- [ ] Twin handles a job that finishes `done` with zero windows (see issue 3).

Predictions:
- [ ] Take 10 to 20 clips from `phase4_detail_Audio_Moth_6.csv`. Species and confidence returned by the service match the CSV (exact match expected).
- [ ] Every returned `species` string exists in the label file character for character.
- [ ] Every confidence is between 0.25 and 1.0 (threshold 0.25).
- [ ] FP16 and FP32 give the same detection set on most clips (docs: 99.05% identical, differences sit near the 0.25 boundary).
- [ ] Check the common name matches the scientific name for 5 to 10 species by eye. This is the quickest catch for an index shift.

Preprocessing:
- [ ] A 143,360-sample and a 147,456-sample clip are rejected, not scored.
- [ ] A stereo clip is rejected.
- [ ] A non-48 kHz clip is resampled and scored.
- [ ] A silent clip: expect no detections at 0.25 (expectation only, not yet observed).

Label mapping:
- [ ] Twin displays or stores the full `Scientific_Common` string, or splits it on `_` deliberately.
- [ ] Twin does not load `en_us.txt` or `labels.txt` anywhere.

Scope note from the bridge README: the bridge is a benchmarking service (size, latency, CPU time, memory). It says it cannot be pointed at the Blackbird pass/fail scenarios, because HP is the user-facing output. Confirm with the Twin owner exactly what "validate BirdNET" means there before starting.

---

## 8. Issues found and what still needs fixing

Blockers:
1. **The .onnx files are not in `models/`.** Only labels, mapping CSVs and the allow-list are. `models/README.md` says both models are "currently in this folder", so the copy is incomplete. Needed from whoever has them (Coleman's export, or the repo's original location).
2. **No sample audio.** The repo does not ship audio (about 39 GB, not publishable). Need a few clips at exactly 144,000 samples, plus the AM6 clips for the parity script.

Worth fixing in the service:
3. **Rejected files are invisible to API clients.** A job where every file is rejected returns `status: done`, `error: null`, `windows: 0`. The skip appears only in the server log (`1 files skipped`). A client cannot tell "no birds" from "file rejected". Suggested fix: add `skipped: [{file, reason}]` to `JobState`. Not done, because it changes the response schema and the Twin may already be coding against it. Say if you want it.
4. **Path-based input only.** No file upload endpoint. Fine when the Twin and the service share a disk. If the Twin calls from another machine, an upload endpoint is needed. Decide after seeing the Twin's contract.

Minor, documentation only:
5. `app/inference.py` docstring still says the 0.1 floor is "taken from the project handoff". The bridge README says this is resolved by measurement (reference minimum is exactly 0.250000). Docstring is stale.
6. `models/README.md` validation table says "2,000 real clips" but also "2011/2011 argmax". The counts disagree.
7. The reference CSV in `verification_data/` has absolute paths from a collaborator's machine. The main README already flags this. No change made.

Known limits of the existing validation (from the repo's own docs, not new findings):
- 51,720 clips validated, which is 47.9% of the 108,069 reference rows. Four of six recorders are partial.
- Bridge parity was checked on 709 AM6 clips, top allow-listed species per clip only, not the full 493-value vector.
- FP16 on hardware with real FP16 units is unmeasured.

Environment note:
- This test ran on Linux with Python 3.11.16. Your Windows run is not yet confirmed. The README's Windows commands are used above as written.
