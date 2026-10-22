# Digital Twin run 4: compressed BirdNET ON

The same recording and settings as the baseline (`baseline_birdnet_off.md`), with
the Twin's BirdNET step switched on and sent to the compressed FP16 model through
the FastAPI service. This is the Twin validation of the BirdNET component.

Result: the compressed model works through the Twin with no interface, label
mapping or hand-off errors (checks T1 to T5 pass). Enabling it changes what the
Twin reports downstream; those integration findings are listed below and need
decisions from the team lead.

## Setup
- Twin: `KashmirWorld/digital_twin_mvp`, commit `7340e820195ca868d68c503366f3dfd9a0b7c9a6`, plus the local patch `birdnet_compressed.patch` (applied with `git apply --ignore-whitespace`, never committed or pushed).
- The patch sets `skip_birdnet` to False and replaces the BirdNET step with `ApiBirdNETConfirm`. For each triggered 3 s clip it writes a 16-bit WAV (as the Twin does today), sends it to the API, and returns the best detection at or above the Twin's threshold (0.5) as a common name.
- API: this repo's `src/bridge` service on port 8001, variant `fp16_pruned493`.
- Model file loaded by the API (hashed after the run): `birdnet_fp16_pruned493.onnx`, 18,878,769 bytes, SHA-256 `c6a5dcbbd2f23a5886b0abe640d37e16993359746383ede139be9c735db714d8`. Identical to the file validated in `results/`.
- Input: `node_SHAMAN.WAV` (2,399,338,821 bytes, SHA-256 `D564289D5925AF1744F0AE41B5721AA8CC1BE309969047F239AFC596CB1E98F8`, from the baseline). The copy the Twin stored for run 4 was not hashed separately. Its stage-1 output has the same size and counts as run 3.
- Same run settings as the baseline.

## Results

| | Run 3 (BirdNET skipped) | Run 4 (compressed BirdNET) |
|---|---|---|
| Stage-1 candidates | 1,820 | 1,820 |
| Confirmed bird calls (AED gate) | 1,777 | 1,777 |
| BirdNET events | 1,777 skipped | 1,777 scored |
| Events with a species (>= 0.5) | 0 | 84 |
| Events without a species | 1,777 | 1,693 |
| Dashboard "Detections" | 1,777 | 218 |
| Processed in | 22.5 min | 51.0 min |
| Dashboard inference latency | 67 ms | 721 ms |
| `bird_confirmed` / `bird_candidate` | none / none | 84 / 1,693 |
| `human_presence_confirmed` / `_candidate` | 0 / 1,777 | 134 / 1,643 |

Checks (`scripts/twin_check.py`, output in `results/run4_twin_check.txt`):
- T1 interface: 1,777 BirdNET events and 1,777 logged API calls, one each, none missing or repeated. PASS
- T2 label mapping: all 613 labels returned are in the 493-species label file. PASS
- T3 hand-off: the Twin's result equals the best logged detection at or above 0.5 on 1,777 of 1,777 events, confidence difference 0. PASS
- T4 confidence: 0.1000 to 0.9987 (the API floor is 0.1). PASS
- T5 re-score: 50 saved clips scored again directly through the API gave identical detections, difference 0. PASS

Species (84 events, 11 species, confidence 0.503 to 0.999, median 0.73): Buff-rumped Warbler 56, Black-capped Pygmy-Tyrant 9, Stripe-throated Hermit 4, Blue-throated Goldentail 4, Band-tailed Barbthroat 3, Scarlet Macaw 2, Oilbird 2, and one each of Black-cheeked Ant-Tanager, Streak-chested Antpitta, Violet-headed Hummingbird, Red-capped Manakin. Several also appear in the Audio_Moth_6 clips used for the main validation.

## Findings for the integration (decisions needed)
1. **Threshold (V-003).** The Twin uses 0.5; the validation used 0.25. At 0.5, 84 events get a species. At 0.25 it would be 172. 88 events have a best detection between 0.25 and 0.5.
2. **Dashboard (V-005).** "Detections" fell from 1,777 to 218. The count is the confirmed events: 84 `bird_confirmed` + 134 `human_presence_confirmed`. The 1,693 `bird_candidate` events do not appear as detections.
3. **Human-presence stage changes (V-004).** With BirdNET off, no event was human-presence confirmed. With it on, 134 are. The stage reads BirdNET's species and confidence (`HumanPresenceAdapter`). The cause of the 134 was not traced.
4. **Allow-list.** The patch does not filter by the 218-species allow-list, matching the original Twin. 2 of the 84 confirmed events (Oilbird) are outside the allow-list. Recorded as V-002 in `validation/ISSUES.md`.
5. **Label format.** The API returns `Scientific_Common`; the patch passes the common name on, as the original BirdNET step did. The human-presence stage matches sentinel species by substring on either name.
6. **Run time.** 51.0 min against 22.5 min. The API averaged 78 ms per call, so inference is not the cause. The patch sends one job per clip, and the API runs 3 warm-up passes at the start of each job (`warmup(model)` in `jobs.py`). That is a likely cause of the extra ~0.65 s per clip, but it was not measured.
7. **V-001 (silent skip) was not triggered.** All 1,777 clips were scored. The patch raises an error if the API scores zero windows, so a skipped clip could not pass unnoticed. See `../ISSUES.md`.

## Evidence
- `evidence/Digital Twin _ Model Performance Dashboard.pdf`: dashboard for run 3 (BirdNET skipped).
- `evidence/Digital Twin _ Model Performance Dashboard run4.pdf`: dashboard for run 4 (compressed BirdNET on).
- `evidence/node_run_summary.json`: the run 3 summary.
- `run4_raw/birdnet_calls.jsonl`: every API response received by the Twin (1,777 lines).
- `run4_raw/node_run_summary.json`: the Twin's run summary.
- `results/run4_twin_check.txt`: output of `scripts/twin_check.py`.
- Not committed: `stage_3_birdnet_timeline.json` (1.7 MB). Copy it from the Twin's `run_4\SHAMAN` folder into `run4_raw/` to re-run the check.

## Not covered
- Behaviour on clips with no bird, and the effect of the Twin's normalization on them.
- A second recording or node.
- A run where the Twin loads the ONNX model itself instead of calling the API.
