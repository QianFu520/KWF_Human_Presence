"""
build_training_set.py: labelled train/val/test table for one label window.

  * Positives: clips on an event's own recorder inside its label window (--window,
    from label_windows.csv), for events with audio. Split = the event group's split.
  * Negatives: negatives.csv from plan_negative_split.py (outside every event buffer
    and trim exclusion), split by 20-minute shared blocks; guard/conflict drops removed.
  * Everything else (inside a buffer but not a positive) is left out.
  * AED-rejected clips are KEPT; not_meaningful_prob / predicted_label are carried
    along so results can also be reported on AED-passed clips only.

Output: <out>/training_set_<window>.csv with id columns, label, split, sim_type,
AED columns and the 25 no-weather features (FEATURES_25, in the order of
HP_Model/docs/feature_pipeline_progress.md).

Usage (PowerShell):
  python build_training_set.py --pipeline-out "C:\\BirdNET ONNX\\pipeline_out" `
      --split-dir "C:\\BirdNET ONNX\\hp_audit\\split" --window w10 --out "C:\\BirdNET ONNX\\hp_audit\\training"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "inference_pipeline" / "scripts"))
from feature_engineering import SENTINEL_SPECIES  # noqa: E402

ACOUSTIC = ["RMS_Energy", "Spectral_Contrast", "Spectral_Flatness", "Spectral_Bandwidth",
            "Spectral_Rolloff_85", "Onset_Strength", "MFCC_8", "MFCC_9", "MFCC_12", "MFCC_13"]
FEATURES_25 = ACOUSTIC + SENTINEL_SPECIES + ["confidence", "hour_sin", "hour_cos", "Eerie_Silence",
                                             "Volume_Spike_15s"]
NAMES = ["train", "val", "test"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline-out", required=True, type=Path)
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--window", default="w10", choices=["w10", "per_type", "tight"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cols = ["clip_name", "Recorder", "Timestamp", "not_meaningful_prob", "predicted_label", *FEATURES_25]
    df = pd.concat(pd.read_csv(args.pipeline_out / f"Audio_Moth_{n}" / "outputs" / "features_all.csv", usecols=cols)
                   for n in range(1, 7))
    df["ts"] = pd.to_datetime(df.Timestamp.astype(str), format="%Y%m%d_%H%M%S")
    assert len(FEATURES_25) == 25 and df.clip_name.is_unique

    # positives
    ev = pd.read_csv(args.split_dir / "events.csv")
    lw = pd.read_csv(args.split_dir / "label_windows.csv", parse_dates=["start_local", "end_local"])
    lw = lw[lw.window == args.window].merge(ev[["event_id", "sim_type", "split", "has_audio"]], on="event_id")
    lw = lw[lw.has_audio & lw.split.isin(NAMES)]
    by_rec = {r: g.set_index("ts").sort_index() for r, g in df.groupby("Recorder")}
    rows = []
    for w in lw.itertuples():
        hit = by_rec[w.recorder].loc[w.start_local:w.end_local]  # inclusive, as in build_event_split.py
        rows.append(pd.DataFrame({"clip_name": hit.clip_name.values, "split": w.split,
                                  "sim_type": w.sim_type, "event_id": w.event_id}))
    pos = pd.concat(rows)
    # a clip inside two events' windows (same group, so same split): keep one row, list both types
    pos = pos.groupby("clip_name").agg(split=("split", "first"), n_splits=("split", "nunique"),
                                       sim_type=("sim_type", lambda s: ";".join(sorted(set(s)))),
                                       event_id=("event_id", lambda s: ";".join(map(str, sorted(set(s))))))
    assert (pos.n_splits == 1).all(), "a positive clip belongs to events in different splits"
    pos = pos.drop(columns="n_splits").assign(label=1)

    # negatives
    neg = pd.read_csv(args.split_dir / "negatives.csv")
    neg = neg[neg.neg_split.isin(NAMES)].set_index("clip_name")
    assert not neg.index.isin(pos.index).any(), "a clip is both positive and negative"
    neg = pd.DataFrame({"split": neg.neg_split, "sim_type": "", "event_id": "", "label": 0}, index=neg.index)

    lab = pd.concat([pos, neg])
    out = df.drop(columns="ts").merge(lab, left_on="clip_name", right_index=True, how="inner")
    front = ["clip_name", "Recorder", "Timestamp", "split", "label", "sim_type", "event_id",
             "not_meaningful_prob", "predicted_label"]
    out = out[front + FEATURES_25].sort_values(["Recorder", "Timestamp"])
    path = args.out / f"training_set_{args.window}.csv"
    out.to_csv(path, index=False)

    t = out.groupby("split").agg(clips=("label", "size"), positives=("label", "sum"))
    t["negatives"] = t.clips - t.positives
    t["aed_rejected"] = out.groupby("split").predicted_label.apply(lambda s: (s == "not_meaningful").sum())
    print(f"Window {args.window}: wrote {len(out):,} rows to {path}")
    print(t.loc[NAMES].to_string())


if __name__ == "__main__":
    main()
