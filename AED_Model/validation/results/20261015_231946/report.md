# AED validation report — 20261015_231946

**Result: PASS** — 19 pass, 0 fail, 3 warn, 16 info.

Command: `validation/validate_aed.py --aed_root . --manifest validation/sample_audio/manifest.csv --api_url http://localhost:8011 --skip_workflow --out validation/results`

Environment: Python 3.13.13, torch 2.13.0, librosa 1.0.0, numpy 2.5.4, macOS-26.5-arm64-arm-64bit-Mach-O

| ID | Check | Status | Detail |
| --- | --- | --- | --- |
| A1 | Library versions | INFO | python 3.13.13, machine arm64, torch 2.13.0, librosa 1.0.0, numpy 2.5.4, soundfile 0.14.0, onnxruntime 1.31.0, fastapi 0.139.0, pandas 3.0.3 |
| A2 | Matches AED_Model/requirements.txt pins | WARN | differs: {'torch': '2.13.0', 'librosa': '1.0.0', 'numpy': '2.5.4'} (pins {'torch': '2.2.2', 'librosa': '0.11.0', 'numpy': '1.26.4'}) |
| B1 | Checkpoint SHA-256 | PASS | tinycnn_v3.pth 913a78a468ff42e2… |
| B2 | Checkpoint metadata says v3 | PASS | notes='TinyCNN v3 — 11213 not_meaningful clips (acoustic scan + model_inference_v1 + model_inference_v2)', val_acc=0.9984, nm_precision=0.98, nm_recall=1.0, n_train=101820, n_val=28333 |
| B3 | Strict load into TinyCNN + parameter count | PASS | 547,937 parameters, strict load ok (<All keys matched successfully>) |
| B4 | Consumers point at v3 | PASS | API default MODEL_PATH='outputs/models/tinycnn_v3.pth'; run_pipeline.py refs=['tinycnn_v3.pth'] |
| B5 | ONNX export = same model | PASS | in mel_spectrogram['batch_size', 1, 128, 256] -> out not_meaningful_prob['batch_size']; max /pt-onnx/ on 32 random inputs = 0.00e+00 |
| C1 | Mel contract (128x256, float32, finite, peak 0 dB) | PASS | 137 readable clips; violations: none; unreadable: ['syn_corrupt.wav'] |
| C1b | Loudest moment falls in the cropped tail (last 0.27 s) | INFO | 7/137 clips: ['syn_late_burst_2p8s.wav', 'esc50_hand_saw_5-253094-A-49.wav', 'esc50_hand_saw_2-51630-A-49.wav', 'esc50_footsteps_3-94342-A-25.wav', 'esc50_wind_4-144083-C-16.wav', 'esc50_insects_5-194932-A-7.wav', 'esc50_crickets_4-171823-A-13.wav'] |
| C2 | Labelling notebook (06) = training (05) | PASS | bit-identical over 137 clips |
| C3 | CLI preprocess_clip = training | PASS | bit-identical over 137 clips |
| C4 | FastAPI compute_mel = training | PASS | bit-identical over 137 clips |
| D0 | Target | INFO | live server http://localhost:8011 |
| D1 | GET /health | PASS | 200 {'status': 'ok'} |
| D2 | OpenAPI schema | PASS | paths=['/health', '/predict']; request multipart/form-data with 'file'; response fields=['label', 'not_meaningful_prob'] |
| D3 | POST /predict on every readable clip | PASS | 137 requests, 200 + valid schema + label consistent with threshold 0.5; problems: none |
| D4 | Error handling | PASS | non-.wav filename -> 400; corrupt .wav -> 422; empty .wav -> 422; no file field -> 422; upper-case .WAV -> 200 |
| D5 | Determinism | PASS | 5 identical requests -> 1 distinct response(s) |
| D6 | Latency per request (this machine) | INFO | p50 6.6 ms, p95 16.1 ms, max 88.3 ms over 137 requests |
| E1 | CLI scored every readable clip | PASS | 137 rows in CLI CSV for 137 readable clips |
| E2 | API vs CLI probability | PASS | max /diff/ 4.83e-05 (API rounds to 4 dp; tol 0.0001) |
| E3 | ONNX vs CLI probability | PASS | max /diff/ 1.43e-06 |
| E5 | CLI vs API labels at the same threshold | PASS | 0 disagreements [] (CLI uses >=, API uses >) |
| F1 | End-of-clip blind spot (0.2 s 5 kHz burst in noise, by onset) | WARN | 0.5s→0.000; 1.5s→0.000; 2.5s→0.003; 2.6s→0.066; 2.7s→1.000; 2.8s→1.000; noise alone→1.000. Frames kept cover 2.731 s of 3.0 s |
| F2 | Gain invariance | PASS | amp 0.001→1.0000; amp 0.05→1.0000; amp 0.5→1.0000 |
| F3 | Zero-padding changes the decision | INFO | 3 s noise → 1.000; first 1 s of same noise, zero-padded → 0.000 |
| F4 | Digital silence | INFO | 0.724 |
| G1 | Confusion at threshold 0.5 (115 labelled clips) | INFO | meaningful kept 79, meaningful dropped 3, background kept 20, background dropped 13; accuracy 0.800, meaningful recall 0.963, not_meaningful precision 0.812, recall 0.394 |
| G2 | Human-activity clips kept | WARN | 53/56 kept; dropped: ['esc50_footsteps_4-194979-A-25.wav', 'esc50_footsteps_3-103597-A-25.wav', 'esc50_fireworks_2-117617-A-48.wav'] |
| G3 | group background_rain | INFO | 6/8 as expected (not_meaningful); median p 0.851 |
| G3 | group background_storm | INFO | 0/8 as expected (not_meaningful); median p 0.000 |
| G3 | group background_water | INFO | 4/8 as expected (not_meaningful); median p 0.500 |
| G3 | group background_wind | INFO | 2/8 as expected (not_meaningful); median p 0.000 |
| G3 | group bird | INFO | 16/16 as expected (meaningful); median p 0.000 |
| G3 | group human_activity | INFO | 53/56 as expected (meaningful); median p 0.000 |
| G3 | group other_animal | INFO | 8/8 as expected (meaningful); median p 0.000 |
| G3 | group synthetic | INFO | 2/2 as expected (meaningful); median p 0.080 |
| G3 | group synthetic | INFO | 1/1 as expected (not_meaningful); median p 1.000 |

## Threshold sweep (labelled clips, CLI scores)

| threshold | meaningful kept | meaningful dropped | background dropped | background kept | accuracy |
| --- | --- | --- | --- | --- | --- |
| 0.3 | 79 | 3 | 14 | 19 | 0.809 |
| 0.5 | 79 | 3 | 13 | 20 | 0.800 |
| 0.7 | 79 | 3 | 12 | 21 | 0.791 |
| 0.9 | 80 | 2 | 11 | 22 | 0.791 |
| 0.95 | 80 | 2 | 11 | 22 | 0.791 |
| 0.99 | 81 | 1 | 10 | 23 | 0.791 |

## By group

| group | expected | n | as expected | median p | min p | max p |
| --- | --- | --- | --- | --- | --- | --- |
| background_rain | not_meaningful | 8 | 6 | 0.851 | 0.000 | 1.000 |
| background_storm | not_meaningful | 8 | 0 | 0.000 | 0.000 | 0.000 |
| background_water | not_meaningful | 8 | 4 | 0.500 | 0.000 | 1.000 |
| background_wind | not_meaningful | 8 | 2 | 0.000 | 0.000 | 1.000 |
| bird | meaningful | 16 | 16 | 0.000 | 0.000 | 0.000 |
| human_activity | meaningful | 56 | 53 | 0.000 | 0.000 | 1.000 |
| other_animal | meaningful | 8 | 8 | 0.000 | 0.000 | 0.000 |
| synthetic | meaningful | 2 | 2 | 0.080 | 0.000 | 0.159 |
| synthetic | not_meaningful | 1 | 1 | 1.000 | 1.000 | 1.000 |

Files: `checks.csv` (all checks), `per_clip.csv` (every clip, every path), `summary.json` (environment, model metadata, metrics), `cli_predictions.csv`.
