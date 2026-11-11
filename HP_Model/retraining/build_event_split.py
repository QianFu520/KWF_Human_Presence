"""
build_event_split.py: event groups, label windows, negative buffers and the
train/validation/test split for HP retraining.

Needs only simulations.csv (no audio, no pipeline). Optionally takes CSVs with a
clip_name column (e.g. train/test_acoustic_features.csv) to mark which events
have audio and count their positive clips.

Design choices (see HP_Model/docs/feature_pipeline_progress.md, Decision section):
  * Clip filenames and event times are both handled in local Costa Rica time (UTC-6).
  * Split groups are built ONCE, using the WIDEST label window being compared, so
    every window variant shares the same split and results stay comparable.
  * Events on the same recorder whose widest windows overlap go in one group.
  * Events on different recorders within --cross-gap of each other also go in one
    group, since staged sounds can carry to neighbouring recorders.
  * Buffers apply to ALL recorders: clips inside any event's buffer are excluded
    from negatives (unless they are positives for their own recorder).

Usage (PowerShell):
  python build_event_split.py --sim "C:\\...\\simulations.csv" --out "C:\\BirdNET ONNX\\hp_audit\\split" `
      --clips "C:\\...\\train_acoustic_features.csv" "C:\\...\\test_acoustic_features.csv"
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from pathlib import Path

import pandas as pd

LOCAL_OFFSET = pd.Timedelta(hours=-6)  # Costa Rica, no daylight saving
PRESENCE = {"Human Presence on Trail", "Human Presence off Trail"}

# Label windows to compare, in seconds relative to the event timestamp: (start, end).
#   w10       ±10 min for every type (the current training labels)
#   per_type  original simulation_matching.py windows (also in sentinal_prob_model.ipynb)
#   tight     a tighter proposal: ±5 min for presence, 0 to +2 min for other events
WINDOWS = {
    "w10":      {"presence": (-600, 600), "event": (-600, 600)},
    "per_type": {"presence": (-900, 900), "event": (0, 300)},
    "tight":    {"presence": (-300, 300), "event": (0, 120)},
}
CLIP_SECONDS = 3
CLIP_RE = re.compile(r"^(Audio_Moth_\d)_(\d{8}_\d{6})(?:_aug_\d+)?(?:\.wav)?$")


# ---------------------------------------------------------------- events
def load_events(sim_csv: str) -> pd.DataFrame:
    sim = pd.read_csv(sim_csv)
    ev = pd.DataFrame({
        "event_id": sim["group"].astype(int),
        "name": sim["name"],
        "sim_type": sim["simulation_name"],
        "kind": sim["simulation_name"].map(lambda t: "presence" if t in PRESENCE else "event"),
        "recorder": sim["mic"],
        "time_local": (pd.to_datetime(sim["timestamp"], unit="ms") + LOCAL_OFFSET).dt.floor("s"),
        "lat": sim["latitude"], "lon": sim["longitude"],
    })
    return ev.sort_values("time_local").reset_index(drop=True)


def window_bounds(ev: pd.DataFrame, window: str) -> pd.DataFrame:
    w = WINDOWS[window]
    start = ev.apply(lambda r: r.time_local + pd.Timedelta(seconds=w[r.kind][0]), axis=1)
    end = ev.apply(lambda r: r.time_local + pd.Timedelta(seconds=w[r.kind][1]), axis=1)
    return pd.DataFrame({"event_id": ev.event_id, "recorder": ev.recorder, "window": window,
                         "start_local": start, "end_local": end})


# ---------------------------------------------------------------- audio presence (optional)
def load_clip_times(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        names = pd.read_csv(p, usecols=["clip_name"])["clip_name"].astype(str)
        m = names.str.extract(CLIP_RE)
        frames.append(pd.DataFrame({"recorder": m[0], "ts": pd.to_datetime(m[1], format="%Y%m%d_%H%M%S")}))
    clips = pd.concat(frames).dropna().drop_duplicates()
    return clips


def clips_in_windows(clips: pd.DataFrame, bounds: pd.DataFrame) -> dict:
    """event_id -> set of (recorder, ts) clips inside that event's window on its own recorder."""
    out = {}
    for b in bounds.itertuples():
        c = clips[(clips.recorder == b.recorder) & (clips.ts >= b.start_local) & (clips.ts <= b.end_local)]
        out[b.event_id] = set(zip(c.recorder, c.ts))
    return out


# ---------------------------------------------------------------- groups
def build_groups(ev: pd.DataFrame, cross_gap_s: int) -> pd.Series:
    """Union-find over events: same recorder with overlapping widest windows, or
    different recorders within cross_gap_s of each other."""
    widest = {
        kind: (min(WINDOWS[w][kind][0] for w in WINDOWS), max(WINDOWS[w][kind][1] for w in WINDOWS))
        for kind in ("presence", "event")
    }
    lo = ev.apply(lambda r: r.time_local + pd.Timedelta(seconds=widest[r.kind][0]), axis=1)
    hi = ev.apply(lambda r: r.time_local + pd.Timedelta(seconds=widest[r.kind][1]), axis=1)
    parent = list(range(len(ev)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(len(ev)), 2):
        a, b = ev.iloc[i], ev.iloc[j]
        if a.recorder == b.recorder:
            linked = lo[i] <= hi[j] and lo[j] <= hi[i]
        else:
            linked = abs((a.time_local - b.time_local).total_seconds()) <= cross_gap_s
        if linked:
            parent[find(i)] = find(j)

    roots = [find(i) for i in range(len(ev))]
    order = {r: k + 1 for k, r in enumerate(dict.fromkeys(roots))}  # number groups in time order
    return pd.Series([order[r] for r in roots], index=ev.index, name="split_group")


# ---------------------------------------------------------------- split
KINDS_REQUIRED = ["Human Presence on Trail", "Vehicle", "Gunshot", "Chainsaw"]


def assign_split(groups: pd.DataFrame, fractions=(0.6, 0.2, 0.2), seed=42, tries=20000) -> dict:
    """Random search for a group-level split whose positive-clip shares are close to
    `fractions`, where validation and test each contain every main event type."""
    rng = random.Random(seed)
    names = ["train", "val", "test"]
    total = groups.n_clips.sum()
    best, best_score = None, float("inf")
    ids = list(groups.index)
    for _ in range(tries):
        assign = {g: rng.choices(names, weights=fractions)[0] for g in ids}
        score = 0.0
        for k, name in enumerate(names):
            members = [g for g in ids if assign[g] == name]
            if not members:
                score += 10
                continue
            share = groups.loc[members, "n_clips"].sum() / total
            score += (share - fractions[k]) ** 2 * 100
            if name != "train":
                types = set().union(*groups.loc[members, "types"])
                score += sum(1 for t in KINDS_REQUIRED if t not in types)
        if score < best_score:
            best, best_score = assign, score
    return best


def assign_folds(groups: pd.DataFrame, k=5) -> dict:
    """Greedy grouped K-fold: largest groups first, each to the fold with fewest clips."""
    load = [0] * k
    out = {}
    for g in groups.sort_values("n_clips", ascending=False).index:
        f = min(range(k), key=lambda i: load[i])
        out[g] = f + 1
        load[f] += groups.loc[g, "n_clips"]
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True, help="simulations.csv")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--clips", nargs="*", default=[], help="CSVs with a clip_name column (positives), to mark events with audio")
    ap.add_argument("--cross-gap", type=int, default=600, help="seconds; events on different recorders this close share a group")
    ap.add_argument("--buffer", type=int, default=1800, help="seconds either side of every event excluded from negatives, on all recorders")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = load_events(args.sim)

    # label windows for every variant
    bounds = pd.concat([window_bounds(ev, w) for w in WINDOWS], ignore_index=True)
    bounds.to_csv(out / "label_windows.csv", index=False)

    # audio presence + positive clip counts (under w10, which matches the current labels)
    if args.clips:
        clips = load_clip_times(args.clips)
        per_event = clips_in_windows(clips, window_bounds(ev, "w10"))
        ev["n_clips"] = ev.event_id.map(lambda e: len(per_event.get(e, ()))).astype(int)
        ev["has_audio"] = ev.n_clips > 0
    else:
        per_event = None
        ev["n_clips"] = 400  # assume a full ±10 min window
        ev["has_audio"] = True
        print("No --clips given: assuming every event has audio.")

    # groups (built on all events, then events without audio are dropped from the split)
    ev["split_group"] = build_groups(ev, args.cross_gap)
    used = ev[ev.has_audio]
    groups = used.groupby("split_group").agg(
        n_events=("event_id", "size"),
        recorders=("recorder", lambda s: ",".join(sorted(set(s)))),
        types=("sim_type", lambda s: frozenset(s)),
        first_local=("time_local", "min"),
        last_local=("time_local", "max"),
        n_clips=("n_clips", "sum"),
    )
    if per_event is not None:
        groups["n_clips"] = [
            len(set().union(*(per_event[e] for e in used.loc[used.split_group == g, "event_id"])))
            for g in groups.index
        ]
    split = assign_split(groups, seed=args.seed)
    folds = assign_folds(groups, k=args.folds)
    groups["split"] = groups.index.map(split)
    groups["fold"] = groups.index.map(folds)
    groups["day"] = groups.first_local.dt.strftime("%m-%d")

    ev["split"] = ev.split_group.map(split)
    ev["fold"] = ev.split_group.map(folds)
    ev.loc[~ev.has_audio, ["split", "fold"]] = ["no_audio", None]

    # negative buffers: one interval per event, applied to every recorder
    buf = pd.DataFrame({
        "event_id": ev.event_id,
        "start_local": ev.time_local - pd.Timedelta(seconds=args.buffer),
        "end_local": ev.time_local + pd.Timedelta(seconds=args.buffer),
    })
    buf.to_csv(out / "negative_buffers.csv", index=False)

    ev.drop(columns=["lat", "lon"]).to_csv(out / "events.csv", index=False)
    g_out = groups.copy()
    g_out["types"] = g_out.types.map(lambda s: ";".join(sorted(s)))
    g_out.to_csv(out / "split_groups.csv")
    (out / "split_config.json").write_text(json.dumps({
        "windows": WINDOWS, "cross_gap_s": args.cross_gap, "buffer_s": args.buffer,
        "folds": args.folds, "seed": args.seed, "fractions": [0.6, 0.2, 0.2],
        "group_rule": "same recorder with overlapping widest windows, or different recorders within cross_gap_s",
    }, indent=2, default=str))

    # ---- summary
    print(f"\nEvents: {len(ev)} | with audio: {int(ev.has_audio.sum())} | split groups with audio: {len(groups)}")
    print(f"Group sizes (events per group): {groups.n_events.value_counts().sort_index().to_dict()}")
    summ = groups.groupby("split").agg(groups=("n_events", "size"), events=("n_events", "sum"), clips=("n_clips", "sum"))
    summ["clip_share"] = (summ.clips / summ.clips.sum()).round(3)
    print("\nSplit:\n" + summ.loc[["train", "val", "test"]].to_string())
    for name in ["val", "test"]:
        types = set().union(*groups[groups.split == name].types)
        print(f"{name} event types: {sorted(types)}")
    print("\nGroups per day and split:\n" + pd.crosstab(groups.day, groups.split).to_string())
    print(f"\nFolds (clips): {groups.groupby('fold').n_clips.sum().to_dict()}")
    print(f"\nWrote: {', '.join(p.name for p in sorted(out.iterdir()))}")


if __name__ == "__main__":
    main()
