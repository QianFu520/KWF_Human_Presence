# Shared API contract: AED, BirdNET and HP

For Cheryl (AED), Rashi (BirdNET), Ben (pipeline) and Coleman (HP). Qian approved the principles on 2026-11-05; the details below are my proposal. Nothing in the AED or BirdNET services changes until Cheryl and Rashi answer the open questions at the end.

## Principles

1. **One Python function per model.** By default `run_pipeline.py` calls it directly, with no HTTP:
   - AED: `score_folder()` in `inference_pipeline/scripts/aed_inference.py`
   - BirdNET: `process_all_clips()` in `inference_pipeline/scripts/run_birdnet.py`
   - HP: `predict()` in `inference_pipeline/scripts/hp_inference.py` (new, in my HP loader PR)
2. **A thin FastAPI wrapper** serves the same code over HTTP for other systems such as the Twin. It does no preprocessing or scoring of its own, so a clip gets the same score either way.
3. **Same shape everywhere:** the same endpoints, file paths as input, one response format and one error format.
4. **Mark, don't drop.** A service returns a result for every clip it is given. AED's label is a column for later steps, not a filter.

## Endpoints

| Endpoint | Use | Request body | Response |
|---|---|---|---|
| `GET /health` | Is the model loaded, and which one | none | `{"name", "version", "threshold", "status"}` |
| `POST /predict` | Synchronous, a few clips | `{"paths": [...]}` | the response envelope below |
| `POST /jobs` | Batch, e.g. a recorder-day | `{"paths": [...]}` | HTTP 202, `{"job_id", "status"}` |
| `GET /jobs/{job_id}` | Poll a batch | none | job fields plus the response envelope |

- `/health` example: `{"name": "aed", "version": "tinycnn_v3+913a78a4", "threshold": 0.5, "status": "ok"}`. A service can add its own fields: BirdNET keeps `ready`, `variants_loaded`, `onnxruntime_version` and `providers`; HP adds `feature_set` and `role`.
- `/predict` takes at most 100 clips. Anything bigger goes to `/jobs`.
- The job fields are BirdNET's existing ones: `paths` in the request; `job_id`, `status` (`queued`, `running`, `done`, `error`, `cancelled`), `total`, `completed` and `error` in the response. `results` and `errors` fill in as clips finish and are final when `status` is `done`. BirdNET's extras (`variant`, `started_at`, `finished_at`, `metrics`, `DELETE /jobs/{id}`, `/jobs/{id}/stream`) stay; other services don't need them.

## Inputs

`paths` is a list of absolute paths the service can read: WAV files, or folders scanned non-recursively for `*.wav`, as BirdNET's `JobRequest` already does. Example: `{"paths": ["<clips_dir>/Audio_Moth_1_20250321_062544.wav", "<clips_dir>"]}`. Nothing is uploaded. When a service runs in Docker, mount the clip folder into it at the same path the caller uses.

Two exceptions: HP takes features, not audio (see the table below), and AED keeps an upload endpoint, `POST /predict/upload` with one WAV, only as an optional extra for one-off tests.

## Response envelope

```json
{
  "model":   {"name": "aed", "version": "tinycnn_v3+913a78a4", "threshold": 0.5},
  "results": [
    {"clip_name": "Audio_Moth_1_20250321_062544.wav", "not_meaningful_prob": 0.031, "predicted_label": "meaningful"}
  ],
  "errors":  [
    {"clip_name": "Audio_Moth_1_20250321_062547.wav", "error": "could not read audio: file is empty"}
  ]
}
```

- `clip_name` is the clip's file name **with `.wav`**: `Audio_Moth_N_YYYYMMDD_HHMMSS.wav`, where the time is the clip's start time in local Costa Rica time. This is Ben's naming rule, which Qian approved, documented in `inference_pipeline/docs/pipeline_io.md` (branch `notebook-paths-and-tests`, not merged yet). CSVs and APIs follow the same rule, and `clip_name` is the join key across AED, BirdNET, HP and other systems such as the Twin.
- BirdNET's `data_id` is already the file name with `.wav`, so it maps directly to `clip_name`.
- HP results use `clip_name` with `.wav`. If an input row's `clip_name` lacks `.wav`, HP adds it: `features.csv` on `main` still strips `.wav` in step 7 until the pipeline changes.
- Every clip the service is given (each file, or each WAV found in a folder) appears exactly once, in `results` or in `errors`.

## Versions

Every service uses one version format: model name, `+`, then the first 8 hex characters of the SHA-256 of the model file. AED: `tinycnn_v3+913a78a4` (Cheryl's suggestion; `913a78a4` is the pipeline's checkpoint). BirdNET: `fp16_pruned493+<sha8 of the .onnx>`. HP: the bundle's `version` plus the hash of the `.joblib`, e.g. `hp-23-20261101+<sha8>`. The version is in `/health` and in `model.version` of every response, so any stored result says which model produced it.

## Errors

- **Bad request: HTTP 422** naming the fields, in FastAPI's standard format: `{"detail": [{"loc": ["body", "paths"], "msg": "...", "type": "..."}]}`. Examples: `paths` missing or empty, more than 100 clips on `/predict`, HP feature columns missing.
- **One clip fails: an entry in `errors`**, and the rest still return with HTTP 200. Examples: a missing or unreadable file, an HP row with a missing or non-numeric value.
- **The whole job fails** (e.g. the model crashes): `status` is `error` and `error` holds the message.
- Unknown `job_id`: 404. Service still starting: 503.

## Per service

| | AED | BirdNET | HP |
|---|---|---|---|
| Inputs | WAV paths | WAV paths | `/predict`: `{"rows": [...]}`, one object per clip with `clip_name` (`.wav` added if missing) and the bundle's features. `/jobs`: `{"paths": ["<features_csv>"]}` |
| Result fields (besides `clip_name`) | `not_meaningful_prob` (higher means more likely background) and `predicted_label` (`meaningful` / `not_meaningful`), for every clip | `detections`: list of `{species, confidence, start_time, end_time, in_allow_list}`, empty when nothing reaches the threshold | `probability` and `label` (1 = human presence); `not_meaningful_prob` and `predicted_label` are passed through when the input has them |
| Threshold | 0.5 today; `not_meaningful` when `not_meaningful_prob` ≥ threshold | 0.25 | read from the bundle; may be re-tuned during integration |
| What changes from today | Upload becomes paths; upload stays only as `/predict/upload`. Add `/jobs`. The response renames `label` to `predicted_label`, as in `tinycnn_predictions.csv`. Unreadable clips go in `errors` instead of being skipped. `/health` adds name, version and threshold. In the pipeline, step 7 no longer drops `not_meaningful` clips and carries both AED columns into `features.csv` (PR #11, merged) | Add the envelope to `GET /jobs/{id}`, next to the existing fields. Per-file failures, already collected in `job.failures`, are returned as `errors`. Add `/predict`. `/health` adds name, version, threshold and status. Threshold stays 0.25, already the default in the bridge and in `run_birdnet.py`. Scoring doesn't change | New service, after my HP loader PR. The candidate is Coleman's 23-feature bundle `hp-23-20261101` (no hour or weather features), for integration testing only, not field use |

## Open questions

**Cheryl (AED)**
1. Does anything call AED's upload `/predict` today, e.g. the Twin? If so, how much notice does it need before upload moves to `/predict/upload`?
2. Is 0.5 still the threshold you want? Since nothing is dropped, it only sets `predicted_label`. In the Nov 1 sweep only 1 of 15 staged events changed between 0.5 and 0.99.
3. `api/main.py` has its own copy of the preprocessing (same parameters today) and rounds the score to 4 decimals. Can the wrapper call `preprocess_clip()` and `predict()` from `aed_inference.py` instead, so both always give the same number?

**Rashi (BirdNET)**
1. Should results be grouped per clip (an empty `detections` list when nothing reaches 0.25), or should we keep the flat detection list and add `clip_name` to each detection? Grouping is what lets every clip appear exactly once.
2. The bridge marks species outside the allow-list with `in_allow_list`; the pipeline's `run_birdnet.py` drops them. I'd keep marking in the API. Does that work for you?
3. Is it OK to add a synchronous `/predict` that runs the same code as a job and returns directly?
