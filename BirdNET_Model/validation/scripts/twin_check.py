"""Checks on a Digital Twin run made with the BirdNET-on patch.

Inputs (from the run):
  --calls    birdnet_calls.jsonl written by the patch (one line per BirdNET call)
  --stage3   stage_3_birdnet_timeline.json from the Twin's output folder
  --summary  node_run_summary.json
Optional:
  --clips DIR --host URL   re-score the saved temp clips directly through the API
                           and compare with what the Twin received

Checks
  T1 interface    every BirdNET event has exactly one logged call; no call missing or repeated
  T2 label map    every label the API returned is in the 493-species label file
  T3 hand-off     the Twin's stage-3 result equals the best logged detection at or above
                  its threshold (species, confidence); no result where none qualifies
  T4 confidence   all logged confidences are in [0.1, 1.0]
  T5 re-score     (optional) saved clips scored directly give the same detections as logged
Also reports: how many events got a species, species list, near misses below the threshold.
Exit code 0 only if T1 to T4 (and T5 when run) pass.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
THRESHOLD = 0.5      # the Twin's birdnet_threshold default


def load_calls(path: Path) -> list:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def check(calls: list, events: list, labels: set, threshold: float) -> tuple[dict, list]:
    res, notes = {}, []
    by_time = collections.Counter(c["trigger_time_s"] for c in calls)
    ev_times = collections.Counter(e["trigger_time_s"] for e in events)
    res["T1"] = (by_time == ev_times and [c["call"] for c in calls] == list(range(1, len(calls) + 1)),
                 f"{len(calls)} calls, {len(events)} events, call numbers contiguous, trigger times match")
    outside = collections.Counter(d["label"] for c in calls for d in c["detections"] if d["label"] not in labels)
    n_det = sum(len(c["detections"]) for c in calls)
    res["T2"] = (not outside, f"{n_det - sum(outside.values())}/{n_det} returned labels are in the label file")
    log_by_time = {c["trigger_time_s"]: c for c in calls}
    bad, max_diff, confirmed = [], 0.0, []
    for e in events:
        c = log_by_time.get(e["trigger_time_s"])
        r = e["shaman_ii"]
        kept = [d for d in (c["detections"] if c else []) if d["confidence"] >= threshold]
        if kept:
            best = max(kept, key=lambda d: d["confidence"])
            ok = (e["event_type"] == "bird_confirmed" and r["species"] == best["label"].partition("_")[2]
                  and abs(r["confidence"] - best["confidence"]) < 1e-9)
            max_diff = max(max_diff, abs(r["confidence"] - best["confidence"]))
            confirmed.append(best)
        else:
            ok = e["event_type"] == "bird_candidate" and r["species"] is None
        if not ok:
            bad.append(e["trigger_time_s"])
    res["T3"] = (not bad, f"{len(events) - len(bad)}/{len(events)} Twin results equal the best logged detection at or above {threshold}"
                          f" (max confidence difference {max_diff:g})")
    confs = [d["confidence"] for c in calls for d in c["detections"]]
    res["T4"] = (bool(confs) and 0.1 <= min(confs) and max(confs) <= 1.0,
                 f"confidence range {min(confs):.4f} to {max(confs):.4f}" if confs else "no detections")
    near = sum(1 for c in calls if c["detections"] and max(d["confidence"] for d in c["detections"]) < threshold
               and max(d["confidence"] for d in c["detections"]) >= 0.25)
    sp = collections.Counter(d["label"] for d in confirmed)
    notes = [f"events with a species (>= {threshold}): {len(confirmed)}/{len(events)}",
             f"events without a species: {len(events) - len(confirmed)} "
             f"(of these, {near} had a best detection between 0.25 and {threshold})",
             f"calls with no detection at all (>= 0.1): {sum(1 for c in calls if not c['detections'])}",
             f"distinct species among confirmed: {len(sp)}"]
    notes += [f"   {n:>3}  {k}" for k, n in sp.most_common(15)]
    if confirmed:
        cs = sorted(d["confidence"] for d in confirmed)
        notes.append(f"confirmed confidence min/median/max: {cs[0]:.3f} / {cs[len(cs)//2]:.3f} / {cs[-1]:.3f}")
    ms = [c["inference_ms"] for c in calls if c.get("inference_ms") is not None]
    if ms:
        notes.append(f"API inference time per call, mean: {sum(ms)/len(ms):.1f} ms")
    return res, notes


def rescore(calls: list, clips: Path, host: str) -> tuple[bool, str]:
    import httpx
    sys.path.insert(0, str(HERE))
    import interface_and_preprocessing as ip
    by_call = {c["call"]: c for c in calls}
    worst, n, mism = 0.0, 0, []
    with httpx.Client(base_url=host, timeout=30.0) as client:
        for wav in sorted(clips.glob("*.wav")):
            call = by_call[int(wav.name.split("_")[0])]
            st = ip.submit(client, [wav])
            got = {d["species"]: d["confidence"] for d in (st.get("detections") or [])}
            want = {d["label"]: d["confidence"] for d in call["detections"]}
            n += 1
            if set(got) != set(want):
                mism.append(wav.name)
                continue
            for k in want:
                worst = max(worst, abs(got[k] - want[k]))
    ok = not mism and worst <= 1e-6
    return ok, f"{n} saved clips re-scored; same detections: {n - len(mism)}/{n}; max confidence difference {worst:.2e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--stage3", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=REPO / "models" / "birdnet_fp16_pruned493.labels.txt")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--clips", type=Path, default=None)
    ap.add_argument("--host", default="http://127.0.0.1:8001")
    ap.add_argument("--out", type=Path, default=None, help="also write the report to this file")
    a = ap.parse_args()

    calls = load_calls(a.calls)
    events = json.loads(a.stage3.read_text(encoding="utf-8"))
    summary = json.loads(a.summary.read_text(encoding="utf-8"))
    labels = {x.strip() for x in a.labels.read_text(encoding="utf-8-sig").splitlines() if x.strip()}
    res, notes = check(calls, events, labels, a.threshold)
    if a.clips:
        res["T5"] = rescore(calls, a.clips, a.host)

    out = [f"Twin run {summary.get('run_id')} | birdnet_status={summary.get('birdnet_status')} | "
           f"candidates {summary.get('n_stage_1_candidates')} | confirmed bird calls {summary.get('n_stage_2_birdcall_confirmed')} | "
           f"BirdNET events {summary.get('n_birdnet_events')}", ""]
    for k in sorted(res):
        out.append(f"  {k}: {'PASS' if res[k][0] else 'FAIL'}  {res[k][1]}")
    out += [""] + notes + ["", f"OVERALL: {'PASS' if all(v[0] for v in res.values()) else 'FAIL'}"]
    text = "\n".join(out)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    return 0 if all(v[0] for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
