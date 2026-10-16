"""
Run each field recording through a running Digital Twin (docker compose up),
one run per recording with its ground-truth log, exactly as the Create Run
page does, and save what the dashboard reports.

Per recording:
  1. POST /api/audio/upload           (the WAV, node SHAMAN; streamed by curl)
  2. POST /api/runs/create            (same body the UI builds: one SHAMAN sensor
                                       node, Confidence Threshold -> AED gate,
                                       stage-1 defaults, groundTruth = the log)
  3. poll GET /api/runs/{id}/status   until complete / failed
  4. GET /api/runs/{id}/dashboard     -> ground_truth block shown on the dashboard

Writes <out>/<recording>/dashboard.json and <out>/dashboard_ground_truth.csv.
The twin's own workflow output (backend_payload.json) is written by the twin
under digital_twin_mvp/backend/audio_workflow_outputs/run_<id>/SHAMAN/.

Usage (from AED_Model/, twin running on localhost:8000):
    python validation/submit_twin_runs.py
    python validation/submit_twin_runs.py --only Audio_Moth_5_20250317_112812
"""
import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

import pandas as pd

from paths import expand

# Create Run page defaults (frontend/src/components/CreateRun.jsx)
UI_CONFIDENCE_THRESHOLD = 0.75
UI_STAGE1_CONFIG = {"target_sample_rate": 48000, "block_seconds": 60, "clip_s": 3.0,
                    "rms_z_thresh": 1.2, "centroid_hz_thresh": 1400, "bandwidth_hz_thresh": 700,
                    "snr_db_thresh": 4.0, "min_gap_s": 1.5}


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def post_json(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def upload(api, path):
    """Stream the WAV with curl (requests would hold a 4 GB body in memory)."""
    out = subprocess.run(
        ["curl", "-sS", "--fail-with-body", "-X", "POST", f"{api}/audio/upload",
         "-F", f"file=@{path}", "-F", "node_id=SHAMAN"],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8000/api")
    ap.add_argument("--logs", default="validation/field/twin_logs")
    ap.add_argument("--only", nargs="*", help="log stems to run (default: all)")
    ap.add_argument("--threshold", type=float, default=UI_CONFIDENCE_THRESHOLD,
                    help="Confidence Threshold field (the AED gate); UI default 0.75")
    ap.add_argument("--out", default=None, help="default validation/results/twin_deployed_<timestamp>")
    ap.add_argument("--poll_s", type=float, default=30)
    a = ap.parse_args()

    out_root = a.out or os.path.join("validation", "results", f"twin_deployed_{dt.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_root, exist_ok=True)
    logs = sorted(glob.glob(os.path.join(a.logs, "*_log.json")))
    if a.only:
        logs = [p for p in logs if os.path.basename(p)[: -len("_log.json")] in a.only]
    if not logs:
        sys.exit(f"No logs found in {a.logs}")

    rows = []
    for path in logs:
        stem = os.path.basename(path)[: -len("_log.json")]
        rec_dir = os.path.join(out_root, stem)
        os.makedirs(rec_dir, exist_ok=True)
        done = os.path.join(rec_dir, "dashboard.json")
        if os.path.exists(done):
            print(f"{stem}: reusing {done}")
            with open(done) as f:
                saved = json.load(f)
        else:
            with open(path) as f:
                log = json.load(f)
            audio = expand(log["metadata"]["source_audio"])
            print(f"{stem}: uploading {os.path.getsize(audio) / 1e9:.2f} GB", flush=True)
            t0 = time.time()
            up = upload(a.api, audio)
            print(f"  uploaded in {(time.time() - t0) / 60:.1f} min -> {up['saved_path']}", flush=True)

            body = {
                "name": f"AED field GT {stem}",
                "description": "Field recording with ground-truth log (AED validation task 5)",
                "scenario": "MVP Simulation", "environment": "Tropical Forest",
                "confidenceThreshold": a.threshold, "sampleRate": 48000, "duration": "24h",
                "status": "pass",
                "nodes": [{"id": "SHAMAN", "label": "Shaman Node", "role": "sensor", "x": 0.5, "y": 0.5}],
                "edges": [],
                "mediaFiles": {"SHAMAN": up["saved_path"]},
                "shamanConfig": {},
                "stage1Config": UI_STAGE1_CONFIG,
                "groundTruth": log,
            }
            run = post_json(f"{a.api}/runs/create", body)
            run_id = run["id"]
            print(f"  run {run_id} created; processing", flush=True)
            t1 = time.time()
            while True:
                status = get_json(f"{a.api}/runs/{run_id}/status").get("status")
                if status in ("complete", "failed"):
                    break
                time.sleep(a.poll_s)
            print(f"  run {run_id} {status} after {(time.time() - t1) / 60:.1f} min", flush=True)
            dash = get_json(f"{a.api}/runs/{run_id}/dashboard")
            saved = {"run_id": run_id, "status": status, "threshold": a.threshold,
                     "saved_path": up["saved_path"], "dashboard": dash}
            with open(done, "w") as f:
                json.dump(saved, f, indent=2)

        g = (saved["dashboard"] or {}).get("ground_truth") or {}
        rows.append({"recording": stem, "run_id": saved["run_id"], "status": saved["status"],
                     "events": g.get("total_events"), "matched": g.get("matched_events"),
                     "recall": g.get("recall"), "precision": g.get("precision"),
                     "detections": g.get("total_detections"),
                     "by_type": json.dumps(g.get("by_type"))})
        print(f"  dashboard: {g.get('matched_events')}/{g.get('total_events')} events, "
              f"recall {g.get('recall')}, precision {g.get('precision')}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(out_root, "dashboard_ground_truth.csv"), index=False)
    print(f"\nWritten to {out_root}")


if __name__ == "__main__":
    main()
