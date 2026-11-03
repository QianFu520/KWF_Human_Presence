"""
Pick raw AudioMoth recordings for the Digital Twin run (task 5) and write a
ground-truth log for each one, in the format read by
digital_twin_mvp/backend/app/services/aed/ground_truth.py:

    {"metadata": {"duration_seconds": ..., "node_id": ...},
     "events": [{"timestamp_seconds": ..., "duration_seconds": ..., "type": ...}]}

timestamp_seconds = Sim_Start - recording start (from the file name YYYYMMDD_HHMMSS.WAV)
duration_seconds  = Sim_End - Sim_Start
Every event that overlaps the recording is included, not only the one it was picked for.

Only WAV headers are read (to get the length), so nothing large is downloaded.

Defaults come from AED_Model/config.py (event table; raw audio = the folder above AUDIO_DIR).

Usage (from AED_Model/):
    python validation/make_twin_logs.py
"""
import argparse
import datetime as dt
import glob
import json
import os
import re

import sys

import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402  (AED_Model/config.py: EVENT_TABLE_PATH, AUDIO_DIR)
from paths import portable  # noqa: E402


def type_slug(t):
    return t.strip().upper().replace(" ", "_")


def recordings_for(raw_root, recorder, day):
    """Raw WAVs for this recorder on `day` and the day before (night recordings start the evening before)."""
    n = recorder.rsplit("_", 1)[-1]
    out = []
    for d in (day - dt.timedelta(days=1), day):
        folder = os.path.join(raw_root, f"Audio Moth {n}", f"Audio_Moth_{n}_{d:%m%d%y}")
        for p in glob.glob(os.path.join(folder, "*.[Ww][Aa][Vv]")):
            m = re.match(r"(\d{8}_\d{6})", os.path.basename(p))
            if m:
                out.append((dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S"), p))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default=config.EVENT_TABLE_PATH)
    ap.add_argument("--raw_root", default=os.path.dirname(config.AUDIO_DIR.rstrip("/")), help="'.../Audio Moth Data' folder that holds 'Audio Moth 1'..'6'")
    ap.add_argument("--out", default="validation/field/twin_logs")
    ap.add_argument("--per_type", type=int, default=1, help="recordings to pick per event type")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    ev = pd.read_csv(a.events, parse_dates=["Sim_Start", "Sim_End"])
    os.makedirs(a.out, exist_ok=True)
    picked, summary = {}, []
    for etype, d in ev.groupby("Sim_Type_Label"):
        n_done = 0
        for e in d.sample(frac=1, random_state=a.seed).itertuples():
            if n_done >= a.per_type:
                break
            recs = [r for r in recordings_for(a.raw_root, e.Recorder, e.Sim_Start.date()) if r[0] <= e.Sim_Start]
            if not recs:
                print(f"  no recording found for {e.Event_ID}")
                continue
            start, path = recs[-1]
            if path in picked:
                n_done += 1
                continue
            info = sf.info(path)
            end = start + dt.timedelta(seconds=info.duration)
            if e.Sim_End > end:
                print(f"  {e.Event_ID} runs past the end of {os.path.basename(path)}; trying another event")
                continue
            picked[path] = (e.Recorder, start, info.duration)
            n_done += 1
        if n_done < a.per_type:
            print(f"WARNING: only {n_done} recording(s) found for {etype}")

    for path, (rec, start, dur) in picked.items():
        end = start + dt.timedelta(seconds=dur)
        inside = ev[(ev.Recorder == rec) & (ev.Sim_End > start) & (ev.Sim_Start < end)]
        events = [{"timestamp_seconds": round(max(0.0, (r.Sim_Start - start).total_seconds()), 3),
                   "duration_seconds": round((min(r.Sim_End, end) - max(r.Sim_Start, start)).total_seconds(), 3),
                   "type": type_slug(r.Sim_Type_Label), "event_id": r.Event_ID}
                  for r in inside.itertuples()]
        stem = f"{rec}_{os.path.splitext(os.path.basename(path))[0]}"
        log = {"metadata": {"duration_seconds": round(dur, 3), "node_id": rec,
                            "source_audio": portable(path), "recording_start": start.isoformat()},
               "events": events}
        log_path = os.path.join(a.out, f"{stem}_log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        summary.append({"recording": portable(path), "recorder": rec, "start": start, "hours": round(dur / 3600, 2),
                        "size_gb": round(os.path.getsize(path) / 1e9, 2), "n_events": len(events),
                        "types": ", ".join(sorted({x["type"] for x in events})), "log": log_path})

    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(a.out, "recordings_to_upload.csv"), index=False)
    print(s.drop(columns=["recording", "log"]).to_string(index=False))
    print(f"\nLogs and recordings_to_upload.csv written to {a.out}")


if __name__ == "__main__":
    main()
