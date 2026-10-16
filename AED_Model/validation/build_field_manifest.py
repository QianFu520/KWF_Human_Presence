"""
Build a labelled manifest of KWF AudioMoth 3-second clips for validate_aed.py.

Reads Qian's labels_progress.csv and event_table.csv, picks a manageable sample,
and maps each clip's original path (/Users/qian/KWF/Segmented_Foldered/...) onto
your streamed Google Drive folder. Only the clips in the manifest are ever read,
so only those stream down from Drive.

Sample (all sizes adjustable):
  event clips       every clip inside N events per event type      -> meaningful
  birdnet_species   random clips labelled by BirdNET, outside events -> meaningful
  background        background_flatness / background_energy clips  -> not_meaningful

Left out on purpose:
  model_inference_v1..v3   labelled by earlier AED versions (circular)
  unknown / unlabeled      no ground truth

Defaults come from AED_Model/config.py (labels, event table, Drive audio folder).

Usage (from AED_Model/):
    python validation/build_field_manifest.py
    # only clips v3 never trained on, every held-out event in full:
    python validation/build_field_manifest.py --heldout --events_per_type 0 --max_clips_per_event 0 \
        --n_background 0 --out validation/field/manifest_heldout.csv
"""
import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402  (AED_Model/config.py: LABELS_PROGRESS_PATH, EVENT_TABLE_PATH, AUDIO_DIR)

# audio_path may point at Qian's laptop or at a Drive mount; keep only the part after this folder
SEG_RE = re.compile(r"^.*?Segmented[ _]Foldered/")


def type_slug(t):
    return t.strip().upper().replace(" ", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=config.LABELS_PROGRESS_PATH)
    ap.add_argument("--events", default=config.EVENT_TABLE_PATH)
    ap.add_argument("--seg_root", default=config.AUDIO_DIR, help="streamed Drive folder '.../Audio Moth Data/Segmented Foldered'")
    ap.add_argument("--out", default="validation/field/manifest.csv")
    ap.add_argument("--events_per_type", type=int, default=2, help="events sampled per event type (0 = all)")
    ap.add_argument("--max_clips_per_event", type=int, default=100, help="cap clips per event (0 = no cap)")
    ap.add_argument("--n_birdnet", type=int, default=500, help="0 = all")
    ap.add_argument("--n_background", type=int, default=500, help="0 = all")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--heldout", action="store_true",
                    help="only clips in TinyCNN v3's validation split (never trained on); see v3_split.py")
    ap.add_argument("--check_files", action="store_true",
                    help="confirm each mapped path exists (fast; does not download audio)")
    a = ap.parse_args()

    lab = pd.read_csv(a.labels)
    ev = pd.read_csv(a.events, parse_dates=["Sim_Start", "Sim_End"])
    lab["ts"] = pd.to_datetime(lab["Timestamp"], format="%Y%m%d_%H%M%S")
    if a.heldout:
        from v3_split import v3_split
        split = v3_split(lab)
        lab = lab[lab.clip_name.map(split).eq("val")]
        print(f"Held-out mode: {len(lab):,} clips in v3's validation split")

    # 1. event clips: clips whose start falls inside an event window on that recorder
    ev = ev.sort_values(["Sim_Type_Label", "Sim_Start"]).reset_index(drop=True)
    if a.events_per_type > 0:
        ev = pd.concat([d.sample(min(len(d), a.events_per_type), random_state=a.seed)
                        for _, d in ev.groupby("Sim_Type_Label")])
    rows, in_event = [], set()
    for e in ev.itertuples():
        m = lab[(lab.Recorder == e.Recorder) & (lab.ts >= e.Sim_Start) & (lab.ts < e.Sim_End)]
        if a.max_clips_per_event > 0 and len(m) > a.max_clips_per_event:
            m = m.sample(a.max_clips_per_event, random_state=a.seed)
        if a.heldout and m.empty:
            continue  # event on a recorder-day v3 trained on
        for r in m.itertuples():
            rows.append(dict(clip_name=r.clip_name, src=r.audio_path, expected_label="meaningful",
                             group=type_slug(e.Sim_Type_Label), source="event_table.csv",
                             event_id=e.Event_ID, recorder=r.Recorder))
        print(f"  {e.Event_ID}: {len(m)} clips")
    # exclude every event window (not only sampled ones) from the other groups
    all_ev = pd.read_csv(a.events, parse_dates=["Sim_Start", "Sim_End"])
    for e in all_ev.itertuples():
        m = lab[(lab.Recorder == e.Recorder) & (lab.ts >= e.Sim_Start) & (lab.ts < e.Sim_End)]
        in_event.update(m.clip_name)
    outside = lab[~lab.clip_name.isin(in_event)]

    # 2. BirdNET-labelled meaningful clips
    b = outside[outside.meaningful_source == "birdnet_species"]
    if a.n_birdnet > 0:
        b = b.sample(min(len(b), a.n_birdnet), random_state=a.seed)
    rows += [dict(clip_name=r.clip_name, src=r.audio_path, expected_label="meaningful", group="BIRDNET_SPECIES",
                  source="birdnet_species", event_id="", recorder=r.Recorder) for r in b.itertuples()]

    # 3. background (rule-based) not-meaningful clips
    g = outside[outside.meaningful_source.isin(["background_flatness", "background_energy"])]
    if a.n_background > 0:
        g = g.sample(min(len(g), a.n_background), random_state=a.seed)
    rows += [dict(clip_name=r.clip_name, src=r.audio_path, expected_label="not_meaningful", group="BACKGROUND",
                  source=r.meaningful_source, event_id="", recorder=r.Recorder) for r in g.itertuples()]

    # A clip inside two events keeps the more specific one: chainsaws, gunshots and
    # vehicles were staged during human-presence walks, and would otherwise vanish
    # into the walk's group.
    man = pd.DataFrame(rows)
    walk = man.group.str.startswith("HUMAN_PRESENCE") & (man.source == "event_table.csv")
    man = man.assign(_walk=walk).sort_values("_walk", kind="stable").drop_duplicates("clip_name")
    man = man.drop(columns="_walk").sort_index()
    has = man.src.str.contains(SEG_RE)
    if not has.all():
        sys.exit(f"audio_path without a Segmented Foldered part, e.g. {man.src[~has].iloc[0]}")
    man["path"] = a.seg_root.rstrip("/") + "/" + man.src.str.replace(SEG_RE, "", regex=True)
    man = man[["clip_name", "path", "expected_label", "group", "source", "event_id", "recorder"]]

    if a.check_files:
        ok = man.path.map(os.path.exists)
        if (~ok).any():
            print(f"WARNING: {(~ok).sum()} mapped paths not found, e.g. {man.path[~ok].iloc[0]}")
            man = man[ok]

    from paths import portable
    man["path"] = man.path.map(portable)  # {AUDIO_DIR}/... so the manifest works on any machine
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    man.to_csv(a.out, index=False)
    print(f"\nWrote {len(man)} clips -> {a.out}")
    print(man.groupby(["group", "expected_label"]).size().to_string())


if __name__ == "__main__":
    main()
