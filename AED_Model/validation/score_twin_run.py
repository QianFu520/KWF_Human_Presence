"""
Score Digital Twin runs on field recordings against their ground-truth logs
(task 5). Uses the twin's own code end to end, so the headline numbers equal
what the dashboard shows after uploading the recording with its *_log.json:

  - detections  = the twin's dashboard events (bird_confirmed, birdnet_skipped,
                  human_presence_confirmed) from run_node_audio_workflow,
                  with the run settings the twin uses (AED threshold 0.3,
                  BirdNET skipped)
  - matching    = app.services.aed.ground_truth.evaluate (detection within
                  the event window +/- 1.5 s)

AED flags all meaningful audio, birds included, so on field recordings almost
every 5-minute event window contains some detection. Recall alone therefore
overstates how well events are detected. Two extra columns put it in context:

  chance_recall   expected recall if detections fell at the rate seen outside
                  all events (Poisson): 1 - exp(-rate_outside * (duration + 3 s)),
                  averaged over the type's events
  density_ratio   detections per minute inside an event's window divided by the
                  rate outside all events in the same recording, averaged over
                  the type's events (1.0 = no different from background)

and a stage breakdown shows whether a miss came from the twin's stage-1
prefilter (no trigger in the window) or from AED (triggers scored, none kept).

Precision is reported but not meaningful on field audio: every bird detection
outside a staged human event counts as a false positive.

Usage (from AED_Model/):
    python validation/score_twin_run.py --twin_backend ../../digital_twin_mvp/backend
    python validation/score_twin_run.py --twin_backend ... --only Audio_Moth_5_20250317_112812
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import sys
import time

import pandas as pd

from paths import expand

DASHBOARD_EVENTS = {"bird_confirmed", "birdnet_skipped", "human_presence_confirmed"}
WINDOW_PAD_S = 1.5  # same widening as ground_truth.HALF_CLIP_S


def merged_length(intervals):
    """Total length of the union of [a, b] intervals."""
    total, end = 0.0, -math.inf
    for a, b in sorted(intervals):
        if a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def inside(t, intervals):
    return any(a <= t <= b for a, b in intervals)


def run_twin(wf, log, out_dir, threshold):
    """Run the twin workflow on the log's source recording, or reuse a saved payload."""
    payload_path = os.path.join(out_dir, "backend_payload.json")
    if os.path.exists(payload_path):
        print(f"  reusing {payload_path}")
        with open(payload_path) as f:
            return json.load(f), None
    audio = expand(log["metadata"]["source_audio"])
    print(f"  running twin workflow on {audio}")
    t0 = time.perf_counter()
    res = wf.run_node_audio_workflow(
        input_audio=audio, out_dir=out_dir, node_id=log["metadata"]["node_id"],
        run_id="field_ground_truth", tinycnn_threshold=threshold, skip_birdnet=True,
    )
    wall = time.perf_counter() - t0
    with open(payload_path, "w") as f:
        json.dump(res["backend_payload"], f)
    return res["backend_payload"], wall


def score(gt, log, payload):
    duration = float(log["metadata"]["duration_seconds"])
    events = log["events"]
    combined = payload.get("combined_timeline") or []
    stage2 = (payload.get("stage_timelines") or {}).get("stage_2_tinycnn_birdcall") or []
    det = sorted(float(e["trigger_time_s"]) for e in combined if e.get("event_type") in DASHBOARD_EVENTS)

    twin = gt.evaluate_run(log, det)  # exactly what the dashboard stores

    windows = [(e["timestamp_seconds"] - WINDOW_PAD_S,
                e["timestamp_seconds"] + e["duration_seconds"] + WINDOW_PAD_S) for e in events]
    outside_s = max(1.0, duration - merged_length(windows))
    det_out = [t for t in det if not inside(t, windows)]
    rate_out = len(det_out) / outside_s

    trig = [(float(e["trigger_time_s"]), e.get("event_type") == "birdcall_stage2_confirmed") for e in stage2]
    trig_out = [k for t, k in trig if not inside(t, windows)]

    rows = []
    for e, (a, b) in zip(events, windows):
        n_det = sum(a <= t <= b for t in det)
        t_in = [k for t, k in trig if a <= t <= b]
        if n_det:
            stage = "detected"
        elif not t_in:
            stage = "missed: no stage-1 trigger"
        else:
            stage = "missed: AED rejected all triggers"
        rows.append({
            "event_id": e.get("event_id", ""), "type": e["type"],
            "start_s": e["timestamp_seconds"], "duration_s": e["duration_seconds"],
            "detections": n_det, "detected": n_det > 0,
            "chance_hit": 1 - math.exp(-rate_out * (b - a)),
            "det_per_min": n_det / ((b - a) / 60),
            "density_ratio": (n_det / (b - a)) / rate_out if rate_out > 0 else float("nan"),
            "stage1_triggers": len(t_in), "aed_kept": sum(t_in), "outcome": stage,
        })

    background = {
        "duration_h": duration / 3600,
        "detections": len(det),
        "detections_outside_events": len(det_out),
        "det_per_min_outside": rate_out * 60,
        "stage1_triggers_outside": len(trig_out),
        "aed_keep_rate_outside": (sum(trig_out) / len(trig_out)) if trig_out else float("nan"),
    }
    return twin, pd.DataFrame(rows), background


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twin_backend", required=True, help="digital_twin_mvp/backend")
    ap.add_argument("--logs", default="validation/field/twin_logs")
    ap.add_argument("--only", nargs="*", help="log stems to run (default: all)")
    ap.add_argument("--threshold", type=float, default=0.3, help="AED gate threshold (twin default 0.3)")
    ap.add_argument("--out", default=None, help="default validation/results/twin_field_<timestamp>")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.twin_backend))
    from app.audio_workflow import node_audio_workflow as wf
    from app.services.aed import ground_truth as gt

    out_root = a.out or os.path.join("validation", "results", f"twin_field_{dt.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_root, exist_ok=True)

    logs = sorted(glob.glob(os.path.join(a.logs, "*_log.json")))
    if a.only:
        logs = [p for p in logs if os.path.basename(p)[: -len("_log.json")] in a.only]
    if not logs:
        sys.exit(f"No logs found in {a.logs}")

    all_events, recordings = [], []
    for path in logs:
        stem = os.path.basename(path)[: -len("_log.json")]
        print(f"\n{stem}")
        with open(path) as f:
            log = json.load(f)
        rec_dir = os.path.join(out_root, stem)
        os.makedirs(rec_dir, exist_ok=True)
        payload, wall = run_twin(wf, log, rec_dir, a.threshold)
        twin, ev, bg = score(gt, log, payload)
        ev.insert(0, "recording", stem)
        all_events.append(ev)
        recordings.append({"recording": stem, **bg, "wall_min": None if wall is None else wall / 60,
                           "twin_recall": twin["recall"], "twin_precision": twin["precision"],
                           "events": twin["total_events"], "matched": twin["matched_events"]})
        with open(os.path.join(rec_dir, "ground_truth_eval.json"), "w") as f:
            json.dump({"twin_dashboard": twin, "background": bg, "events": ev.to_dict("records")}, f, indent=2)
        print(f"  {twin['matched_events']}/{twin['total_events']} events detected, "
              f"{bg['detections']} detections, {bg['det_per_min_outside']:.2f}/min outside events")

    ev = pd.concat(all_events, ignore_index=True)
    rec = pd.DataFrame(recordings)
    by_type = ev.groupby("type").agg(
        events=("detected", "size"), detected=("detected", "sum"),
        chance_recall=("chance_hit", "mean"), det_per_min=("det_per_min", "mean"),
        density_ratio=("density_ratio", "mean"),
        no_stage1_trigger=("outcome", lambda s: (s == "missed: no stage-1 trigger").sum()),
        aed_rejected=("outcome", lambda s: (s == "missed: AED rejected all triggers").sum()),
    ).reset_index()
    by_type["recall"] = by_type.detected / by_type.events
    by_type = by_type[["type", "events", "detected", "recall", "chance_recall", "det_per_min",
                       "density_ratio", "no_stage1_trigger", "aed_rejected"]]

    ev.to_csv(os.path.join(out_root, "events.csv"), index=False)
    by_type.to_csv(os.path.join(out_root, "by_type.csv"), index=False)
    rec.to_csv(os.path.join(out_root, "recordings.csv"), index=False)

    def fmt(df):
        try:
            return df.to_markdown(index=False, floatfmt=".3f")
        except ImportError:  # tabulate not installed
            return "```\n" + df.to_string(index=False, float_format="%.3f") + "\n```"
    report = [
        "# Digital Twin on field recordings: measured against ground truth", "",
        f"AED gate threshold {a.threshold}, BirdNET skipped (twin defaults). Twin backend: `{a.twin_backend}`.", "",
        "## Recall per event type", "",
        "`recall` is what the dashboard reports. `chance_recall` is what detections at the background rate "
        "would score by luck; `density_ratio` is detections per minute inside the windows relative to outside "
        "(1.0 = no different from background). Read recall against both.", "",
        fmt(by_type), "",
        "Staged events often overlap (e.g. a gunshot during a chainsaw run inside a human-presence walk), "
        "so a detection can count for several types at once.", "",
        "## Recordings", "", fmt(rec), "",
        "Precision is low by design here: AED flags birds and other wildlife, and every detection outside a staged "
        "human event counts as a false positive.", "",
        "## Events", "", fmt(ev.drop(columns=["event_id"])), "",
    ]
    with open(os.path.join(out_root, "report.md"), "w") as f:
        f.write("\n".join(report))
    print("\n" + fmt(by_type))
    print(f"\nWritten to {out_root}")


if __name__ == "__main__":
    main()
