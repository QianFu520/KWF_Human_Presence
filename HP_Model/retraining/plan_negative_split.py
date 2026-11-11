"""
plan_negative_split.py: assign negative clips to train/val/test in whole time blocks.

Rules
  * Negative = clip not inside any event buffer (negative_buffers.csv, all recorders)
    and not inside a trim exclusion (trim_exclusions.csv, its own recorder).
    The ±30 min buffer contains every label window, so negatives are the SAME set for
    w10 / per_type / tight; only the positives change between windows.
  * Block = --block-min minutes of clock time (local), shared by ALL recorders, so the same moment
    recorded on two recorders can never land in different splits.
  * A block that overlaps an event's buffer takes that event's group split (forced),
    so audio around a test event stays in test. A block touching groups from two
    different splits is a conflict and its negatives are dropped.
  * Free blocks are assigned 60/20/20 by a seeded random search (seed 42) that keeps
    negative-clip shares near 60/20/20 and puts every hour of day in each split
    (hour_sin/hour_cos are features, so val/test must cover the whole day).
  * Optional guard: drop negatives within --guard-min minutes of a boundary between
    blocks of different splits (adjacent 3 s clips are near-duplicates).

Approved Oct 29: 20-minute shared blocks, 1-minute guard, seed 42 (the defaults).

Outputs (in --out, default --split-dir):
  neg_blocks.csv  one row per block: n_neg, hour, forced, conflict, split
  negatives.csv   one row per negative clip: clip_name, Recorder, neg_split
                  (train / val / test / dropped_guard / dropped_conflict)

Usage (PowerShell):
  python plan_negative_split.py --pipeline-out "C:\\BirdNET ONNX\\pipeline_out" --split-dir "C:\\BirdNET ONNX\\hp_audit\\split"
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = ["train", "val", "test"]
FRACS = [0.6, 0.2, 0.2]


def in_intervals(ts: pd.Series, starts, ends, right_inclusive=True) -> np.ndarray:
    """Boolean mask: ts inside any [start, end] (or [start, end)). ts must be sorted."""
    t = ts.values
    mask = np.zeros(len(t), bool)
    for s, e in zip(starts, ends):
        lo = np.searchsorted(t, np.datetime64(s), "left")
        hi = np.searchsorted(t, np.datetime64(e), "right" if right_inclusive else "left")
        mask[lo:hi] = True
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline-out", required=True, type=Path, help="folder with Audio_Moth_N/outputs/features_all.csv")
    ap.add_argument("--split-dir", required=True, type=Path,
                    help="folder with events.csv, negative_buffers.csv, trim_exclusions.csv, label_windows.csv")
    ap.add_argument("--out", type=Path, help="output folder (default: --split-dir)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tries", type=int, default=5000, help="random-search tries (5000 = the approved split)")
    ap.add_argument("--guard-min", type=float, default=1, help="minutes dropped either side of a split boundary")
    ap.add_argument("--block-min", type=int, default=20, help="block length in minutes")
    args = ap.parse_args()
    PIPE, SPLIT, OUT = args.pipeline_out, args.split_dir, args.out or args.split_dir

    df = pd.concat(pd.read_csv(PIPE / f"Audio_Moth_{n}" / "outputs" / "features_all.csv",
                               usecols=["clip_name", "Recorder", "Timestamp", "predicted_label"])
                   for n in range(1, 7))
    df["ts"] = pd.to_datetime(df.Timestamp.astype(str), format="%Y%m%d_%H%M%S")
    df = df.sort_values(["Recorder", "ts"]).reset_index(drop=True)
    blen = pd.Timedelta(minutes=args.block_min)
    df["block"] = df.ts.dt.floor(blen)

    ev = pd.read_csv(SPLIT / "events.csv", parse_dates=["time_local"])
    buf = pd.read_csv(SPLIT / "negative_buffers.csv", parse_dates=["start_local", "end_local"]).merge(
        ev[["event_id", "split", "split_group"]], on="event_id")
    trim = pd.read_csv(SPLIT / "trim_exclusions.csv", parse_dates=["start_local", "end_local"])
    lw = pd.read_csv(SPLIT / "label_windows.csv", parse_dates=["start_local", "end_local"]).merge(
        ev[["event_id", "split", "sim_type", "has_audio"]], on="event_id")
    lw = lw[lw.has_audio]

    # ---- negatives (same for every window)
    excl_buf = np.zeros(len(df), bool)
    excl_trim = np.zeros(len(df), bool)
    for rec, idx in df.groupby("Recorder").indices.items():
        ts = df.ts.iloc[idx]
        excl_buf[idx] = in_intervals(ts, buf.start_local, buf.end_local)
        t = trim[trim.recorder == rec]
        excl_trim[idx] = in_intervals(ts, t.start_local, t.end_local, right_inclusive=False)
    df["negative"] = ~excl_buf & ~excl_trim

    # ---- forced blocks: any block overlapping an event buffer takes that event's split
    forced = {}
    for b in buf.itertuples():
        split = b.split if b.split in NAMES else None  # no_audio events: buffer only, no split
        for blk in pd.date_range(b.start_local.floor(blen), b.end_local.floor(blen), freq=blen):
            if split:
                forced.setdefault(blk, set()).add(split)
    conflict = {k for k, v in forced.items() if len(v) > 1}
    forced = {k: next(iter(v)) for k, v in forced.items() if len(v) == 1}

    neg = df[df.negative]
    blocks = neg.groupby("block").size().rename("n_neg").to_frame()
    blocks["hour"] = blocks.index.hour
    blocks["forced"] = blocks.index.map(forced)
    blocks["conflict"] = blocks.index.isin(conflict)
    free = blocks[blocks.forced.isna() & ~blocks.conflict]
    fixed = blocks[blocks.forced.notna()]
    total = blocks.loc[~blocks.conflict, "n_neg"].sum()

    # ---- seeded random search over free blocks
    rng = random.Random(args.seed)
    ids = list(free.index)
    best, best_score = None, float("inf")
    for _ in range(args.tries):
        assign = {b: rng.choices(NAMES, weights=FRACS)[0] for b in ids}
        s = pd.Series(assign)
        score = 0.0
        for k, name in enumerate(NAMES):
            n = free.n_neg[s[s == name].index].sum() + fixed.n_neg[fixed.forced == name].sum()
            score += (n / total - FRACS[k]) ** 2 * 100
            hours = set(free.hour[s[s == name].index]) | set(fixed.hour[fixed.forced == name])
            score += 0.05 * (24 - len(hours))
        if score < best_score:
            best, best_score = assign, score
    blocks["split"] = blocks.forced
    blocks.loc[free.index, "split"] = pd.Series(best)
    blocks.loc[blocks.conflict, "split"] = "dropped_conflict"
    blocks.to_csv(OUT / "neg_blocks.csv")

    df["neg_split"] = df.block.map(blocks.split)
    df.loc[~df.negative, "neg_split"] = None

    # ---- optional guard at boundaries between blocks of different splits
    if args.guard_min > 0:
        g = pd.Timedelta(minutes=args.guard_min)
        sp = blocks.split.to_dict()
        prev_diff = df.block.map(lambda b: sp.get(b - blen, sp.get(b)) != sp.get(b))
        next_diff = df.block.map(lambda b: sp.get(b + blen, sp.get(b)) != sp.get(b))
        near = (prev_diff & (df.ts - df.block < g)) | (next_diff & (df.block + blen - df.ts <= g))
        df.loc[df.negative & near, "neg_split"] = "dropped_guard"

    df.loc[df.negative, ["clip_name", "Recorder", "neg_split"]].to_csv(OUT / "negatives.csv", index=False)

    # ---- report
    print(f"Clips {len(df):,} | in event buffers {excl_buf.sum():,} | trim-only {(excl_trim & ~excl_buf).sum():,} "
          f"| negatives {df.negative.sum():,}")
    print(f"Blocks with negatives: {len(blocks)} | forced by events: {len(fixed)} | conflict (dropped): "
          f"{int(blocks.conflict.sum())} | free: {len(free)}")
    print(f"Seed {args.seed}, block {args.block_min} min, best score {best_score:.4f}, guard {args.guard_min:g} min")

    nm = df[df.negative]
    tab = nm.groupby("neg_split").agg(negatives=("clip_name", "size"),
                                      aed_rejected=("predicted_label", lambda s: (s == "not_meaningful").sum()))
    tab["share"] = (tab.negatives / tab.loc[tab.index.isin(NAMES), "negatives"].sum()).round(3)
    tab["blocks"] = blocks.split.value_counts()
    tab["hours_of_day"] = blocks.groupby("split").hour.nunique()
    print("\n== Negatives per split ==\n" + tab.to_string())
    print("\n== Negatives per recorder and split ==\n" +
          pd.crosstab(nm.Recorder, nm.neg_split, margins=True).to_string())

    # positives per window (by event group split)
    by_rec = {r: g.set_index("ts") for r, g in df.groupby("Recorder")}
    rows = []
    for w in lw.itertuples():
        hit = by_rec[w.recorder].loc[w.start_local:w.end_local]
        rows.append(pd.DataFrame({"window": w.window, "split": w.split, "clip_name": hit.clip_name.values,
                                  "recorder": w.recorder,
                                  "rejected": (hit.predicted_label == "not_meaningful").values}))
    pos = pd.concat(rows).drop_duplicates(["window", "clip_name"]).reset_index(drop=True)
    pt = pos.pivot_table(index="window", columns="split", values="clip_name", aggfunc="size")[NAMES]
    print("\n== Positives per window and split (unique clips) ==\n" + pt.to_string())
    negs = nm[nm.neg_split.isin(NAMES)].neg_split.value_counts()[NAMES]
    ratio = (negs / pt).round(1)
    print("\n== Negatives per positive ==\n" + ratio.to_string())
    print("\n== Positives per recorder and split (w10) ==\n" +
          pd.crosstab(pos[pos.window == "w10"].recorder, pos[pos.window == "w10"].split).to_string())


if __name__ == "__main__":
    main()
