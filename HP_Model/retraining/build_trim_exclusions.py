"""
build_trim_exclusions.py: intervals near recording edges to exclude from negatives.

Deployment, pickup and recorder restarts add handling noise that isn't a staged
event, so those clips shouldn't be used as negatives. This script finds each
recorder's continuous recording segments from clip filenames and marks the first
and last --trim-minutes of every segment. A new segment starts wherever the
gap between consecutive clips is longer than --gap-minutes.

Exclusion is from NEGATIVES ONLY: positives inside these intervals stay
positives, and no audio is deleted or moved.

Clip names: Audio_Moth_N_YYYYMMDD_HHMMSS.wav (local Costa Rica time, clip start).
Outputs (in --out):
  coverage_segments.csv  recorder, segment, first_clip_local, end_local, hours, n_clips
  trim_exclusions.csv    recorder, start_local, end_local, reason
                         (a clip is excluded if start_local <= clip start < end_local)
  trim_config.json       settings used

Usage (PowerShell):
  python build_trim_exclusions.py --clips-root "C:\\BirdNET ONNX\\clips" --out "C:\\BirdNET ONNX\\hp_audit\\split"
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

CLIP_SECONDS = 3
CLIP_RE = re.compile(r"^(Audio_Moth_\d)_(\d{8}_\d{6})\.wav$", re.IGNORECASE)


def load_clip_times(clips_root: Path) -> pd.DataFrame:
    rows = []
    for rec_dir in sorted(clips_root.iterdir()):
        if not rec_dir.is_dir():
            continue
        with os.scandir(rec_dir) as it:
            for entry in it:
                m = CLIP_RE.match(entry.name)
                if m:
                    rows.append((m.group(1), m.group(2)))
    clips = pd.DataFrame(rows, columns=["recorder", "ts"])
    clips["ts"] = pd.to_datetime(clips.ts, format="%Y%m%d_%H%M%S")
    return clips.sort_values(["recorder", "ts"]).reset_index(drop=True)


def find_segments(clips: pd.DataFrame, gap_s: int) -> pd.DataFrame:
    clips = clips.copy()
    gap = clips.groupby("recorder").ts.diff().dt.total_seconds()
    clips["segment"] = (gap.isna() | (gap > gap_s)).groupby(clips.recorder).cumsum()
    seg = clips.groupby(["recorder", "segment"]).agg(
        first_clip_local=("ts", "min"), last_clip_local=("ts", "max"), n_clips=("ts", "size")
    ).reset_index()
    seg["end_local"] = seg.last_clip_local + pd.Timedelta(seconds=CLIP_SECONDS)
    seg["hours"] = ((seg.end_local - seg.first_clip_local).dt.total_seconds() / 3600).round(2)
    return seg


def trim_intervals(seg: pd.DataFrame, trim_s: int) -> pd.DataFrame:
    trim = pd.Timedelta(seconds=trim_s)
    rows = []
    for s in seg.itertuples():
        n = seg[seg.recorder == s.recorder].segment.max()
        head = "deployment" if s.segment == 1 else "after_gap"
        tail = "pickup" if s.segment == n else "before_gap"
        rows.append((s.recorder, s.first_clip_local, s.first_clip_local + trim, head))
        rows.append((s.recorder, s.end_local - trim, s.end_local, tail))
    return pd.DataFrame(rows, columns=["recorder", "start_local", "end_local", "reason"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-root", required=True, type=Path, help="folder with one subfolder of clips per recorder")
    ap.add_argument("--out", required=True, type=Path, help="output folder")
    ap.add_argument("--trim-minutes", type=float, default=30, help="minutes excluded at each segment edge")
    ap.add_argument("--gap-minutes", type=float, default=10, help="gaps longer than this split a segment")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    trim_s, gap_s = int(args.trim_minutes * 60), int(args.gap_minutes * 60)

    clips = load_clip_times(args.clips_root)
    seg = find_segments(clips, gap_s)
    excl = trim_intervals(seg, trim_s)

    seg[["recorder", "segment", "first_clip_local", "end_local", "hours", "n_clips"]].to_csv(
        args.out / "coverage_segments.csv", index=False)
    excl.to_csv(args.out / "trim_exclusions.csv", index=False)
    (args.out / "trim_config.json").write_text(json.dumps({
        "trim_minutes": args.trim_minutes, "gap_minutes": args.gap_minutes,
        "clips_root": str(args.clips_root), "n_clips_scanned": len(clips),
        "applies_to": "negatives only; positives and audio untouched",
    }, indent=2))

    # ---- summary: how many clips each recorder loses from the negative pool
    clips["excluded"] = False
    for e in excl.itertuples():
        m = (clips.recorder == e.recorder) & (clips.ts >= e.start_local) & (clips.ts < e.end_local)
        clips.loc[m, "excluded"] = True
    summ = clips.groupby("recorder").agg(clips=("ts", "size"), excluded=("excluded", "sum"))
    summ["segments"] = seg.groupby("recorder").size()
    print(f"Scanned {len(clips):,} clips | trim {args.trim_minutes:g} min | gap > {args.gap_minutes:g} min splits segments")
    print("\nSegments:\n" + seg[["recorder", "segment", "first_clip_local", "end_local", "hours", "n_clips"]].to_string(index=False))
    print("\nExcluded from negatives:\n" + summ.to_string())
    print(f"Total excluded: {int(summ.excluded.sum()):,}")


if __name__ == "__main__":
    main()
