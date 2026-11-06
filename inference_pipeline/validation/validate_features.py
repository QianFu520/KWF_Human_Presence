"""
Runtime feature validation for the HP pipeline.

Runs pipeline steps 1, 4 and 5 (split, acoustic features, metadata) on raw AudioMoth
recordings and compares the acoustic features with a reference CSV computed for
training (e.g. test_acoustic_features.csv).

Usage:
    python inference_pipeline/validation/validate_features.py \
        --raw_audio /path/to/raw --reference /path/to/test_acoustic_features.csv \
        --output /path/to/validation_out

Checks:
    1. Every clip is exactly 144000 samples at 48000 Hz. 143360 or 147456 samples
       means the old millisecond-based cut.
    2. Clips are matched to the reference on clip_name (.wav stripped). If the
       reference uses old notebook column names, a rename map is built from its
       header and printed, and the script stops until it is rerun with
       --accept_column_map. This happens before the slow steps run.
       The reference clips this run should produce are those from the recorder and
       date in each raw folder name (Audio_Moth_6_032025 -> Audio_Moth_6 on 2025-03-20).
    3. For each feature in hp_features.ACOUSTIC: median and max relative difference
       |pipeline - reference| / |reference|; under 1e-3 counts as a match. A reference
       Spectral_Rolloff_85 above 11025 Hz (Nyquist at the pipeline's 22050 Hz) is
       flagged: the reference was computed at a higher sample rate.
    4. Which hours each recorder covers, from the metadata.

Raw folders must be named <recorder>_MMDDYY, as in the Drive (e.g. Audio_Moth_6_032025).

Writes <output>/clips/, <output>/outputs/acoustic_features.csv and metadata.csv (the
same layout as run_pipeline.py), <output>/validation_report.md and <output>/summary.json
(the same results, machine-readable, for combine_reports.py).
The report and summary name the recorder folders and the reference file but contain no local paths.
The reference CSV and the outputs are data: keep them outside the repo.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# ── Path setup (same as run_pipeline.py) ──────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR     = PIPELINE_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "scripts"))

import config                                                  # noqa: E402
from audio_split_rename        import split_all_audio          # noqa: E402
from extract_acoustic_features import process_split, SR as ACOUSTIC_SR  # noqa: E402
from metadata_extraction       import extract_metadata         # noqa: E402
from hp_features               import ACOUSTIC                 # noqa: E402

MATCH_TOL        = 1e-3
OLD_CUT_LENGTHS  = {143360, 147456}       # clip lengths from the old millisecond-based cut
ROLLOFF_LIMIT_HZ = ACOUSTIC_SR / 2        # 11025 Hz: highest rolloff possible at 22050 Hz
CLIP_NAME_ALIASES = {"clipname", "clip", "clipid", "filename", "file"}
CLIP_NAME_RE     = re.compile(r"^(?P<recorder>.+)_(?P<ts>\d{8}_\d{6})$")
FOLDER_RE        = re.compile(r"^(?P<recorder>.+)_(?P<mmddyy>\d{6})$")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def build_column_map(columns):
    """
    Map reference column names to pipeline names when they differ only in case,
    spaces or punctuation (e.g. "Spectral Rolloff 85" -> "Spectral_Rolloff_85"),
    plus common names for the clip column (e.g. "filename" -> "clip_name").
    """
    columns = list(columns)
    targets = {_norm(t): t for t in ["clip_name"] + ACOUSTIC}
    rename = {}
    for col in columns:
        key = _norm(col)
        target = targets.get(key) or ("clip_name" if key in CLIP_NAME_ALIASES else None)
        if target and col != target and target not in columns:
            rename[col] = target
    clashes = [t for t in set(rename.values()) if list(rename.values()).count(t) > 1]
    if clashes:
        raise SystemExit(f"Several reference columns map to {clashes}; fix the reference header. Map: {rename}")
    return rename


def strip_wav(names):
    return names.astype(str).str.replace(r"\.wav$", "", regex=True, case=False)


def raw_folder_days(raw_audio):
    """
    (folder, recorder, date) for each recorder/day folder, e.g. Audio_Moth_6_032025 ->
    ("Audio_Moth_6_032025", "Audio_Moth_6", 2025-03-20). The recorder part is what
    audio_split_rename puts in the clip names.
    """
    days = []
    for folder in sorted(p.name for p in Path(raw_audio).iterdir() if p.is_dir()):
        m = FOLDER_RE.match(folder)
        try:
            day = datetime.strptime(m["mmddyy"], "%m%d%y").date() if m else None
        except ValueError:
            day = None
        if day is None:
            raise SystemExit(f"Raw folder {folder!r} is not named <recorder>_MMDDYY (e.g. Audio_Moth_6_032025).")
        days.append((folder, m["recorder"], day))
    if not days:
        raise SystemExit(f"No recorder/day folders in {raw_audio}.")
    return days


def expected_reference_clips(reference_names, folder_days):
    """
    Reference clips from the recorders and dates in the raw folder names. A folder's recordings
    run into the next morning, but that day's reference clips come from the next folder.
    """
    wanted = {(recorder, day) for _, recorder, day in folder_days}
    expected = []
    for name in reference_names:
        m = CLIP_NAME_RE.match(name)
        if m and (m["recorder"], datetime.strptime(m["ts"][:8], "%Y%m%d").date()) in wanted:
            expected.append(name)
    return expected


def _num(x):
    """JSON has no NaN; write null instead."""
    return None if np.isnan(x) else x


def _inside_repo(path):
    path = Path(path).resolve()
    return path == REPO_DIR or REPO_DIR in path.parents


def _git_commit():
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def format_hours(hours):
    """Compress clock hours into per-day ranges, e.g. '2025-03-18: 05–08, 17'."""
    parts = []
    for day, g in pd.Series(hours).groupby(pd.Series(hours).dt.date):
        hrs = sorted(g.dt.hour.unique())
        ranges, start, prev = [], hrs[0], hrs[0]
        for h in hrs[1:] + [None]:
            if h is not None and h == prev + 1:
                prev = h
                continue
            ranges.append(f"{start:02d}" if start == prev else f"{start:02d}–{prev:02d}")
            if h is not None:
                start = prev = h
        parts.append(f"{day}: {', '.join(ranges)}")
    return "<br>".join(parts)


# ── Checks ────────────────────────────────────────────────────────────────────
def check_clip_lengths(clips_dir):
    rows = []
    for path in sorted(Path(clips_dir).glob("*.wav")):
        info = sf.info(str(path))
        rows.append((path.name, info.frames, info.samplerate))
    clips = pd.DataFrame(rows, columns=["clip", "samples", "sample_rate"])
    bad = clips[
        (clips["samples"] != config.SAMPLES_PER_CLIP)
        | (clips["sample_rate"] != config.TARGET_SAMPLE_RATE)
    ].copy()
    bad["note"] = np.where(bad["samples"].isin(OLD_CUT_LENGTHS), "old millisecond-based cut", "")
    return clips, bad


def feature_stats(merged):
    stats = []
    for feat in ACOUSTIC:
        pipe = merged[f"{feat}_pipeline"].to_numpy(dtype=float)
        ref = merged[f"{feat}_reference"].to_numpy(dtype=float)
        rel = np.abs(pipe - ref) / np.maximum(np.abs(ref), 1e-12)
        n_nan = int(np.isnan(rel).sum())
        valid = ~np.isnan(rel)
        if valid.any():
            median, worst = float(np.median(rel[valid])), float(rel[valid].max())
            pct = 100.0 * float((rel[valid] < MATCH_TOL).mean())
            worst_clip = merged.loc[valid, "clip_name"].iloc[int(rel[valid].argmax())] if worst > 0 else "-"
        else:
            median = worst = pct = float("nan")
            worst_clip = ""
        stats.append({"feature": feat, "median": median, "max": worst, "pct": pct, "nan": n_nan,
                      "worst_clip": worst_clip, "match": n_nan == 0 and worst < MATCH_TOL})
    return stats


def compare_features(stats):
    return pd.DataFrame([{
        "Feature": s["feature"],
        "Median rel. diff": f"{s['median']:.2e}",
        "Max rel. diff": f"{s['max']:.2e}",
        "Clips < 1e-3": f"{s['pct']:.1f}%",
        "NaN": s["nan"],
        "Worst clip": s["worst_clip"],
        "Result": "match" if s["match"] else "DIFFERS",
    } for s in stats])


def recorder_coverage(metadata, matched_names):
    meta = metadata.copy()
    meta["clip_name"] = strip_wav(meta["clip_name"])
    meta["dt"] = pd.to_datetime(meta["Timestamp"].astype(str), format="%Y%m%d_%H%M%S")
    rows = []
    for recorder, g in meta.groupby("Recorder"):
        rows.append({
            "Recorder": recorder,
            "Clips": len(g),
            "Matched to reference": int(g["clip_name"].isin(matched_names).sum()),
            "First clip": g["dt"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "Last clip": g["dt"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "Hours covered (local)": format_hours(g["dt"].dt.floor("h").drop_duplicates()),
        })
    return pd.DataFrame(rows), meta


def unmatched_inside_audio(unmatched_names, meta):
    """
    Reference clips the pipeline did not produce, although a pipeline clip from the same
    recorder starts less than 3 s away. The audio is there, so these point to a
    different cut grid, not missing audio.
    """
    starts = {rec: np.sort(g["dt"].to_numpy(dtype="datetime64[s]")) for rec, g in meta.groupby("Recorder")}
    hits = []
    for name in unmatched_names:
        m = CLIP_NAME_RE.match(name)
        if not m or m["recorder"] not in starts:
            continue
        t = np.datetime64(datetime.strptime(m["ts"], "%Y%m%d_%H%M%S"), "s")
        s = starts[m["recorder"]]
        i = np.searchsorted(s, t)
        nearest = min(abs(s[j] - t) for j in (i - 1, i) if 0 <= j < len(s))
        if nearest < np.timedelta64(3, "s"):
            hits.append(name)
    return hits


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Validate pipeline acoustic features against a reference CSV")
    parser.add_argument("--raw_audio", required=True, help="Folder of raw AudioMoth recordings (one sub-folder per recorder/day)")
    parser.add_argument("--reference", required=True, help="Reference acoustic features CSV (keep it outside the repo)")
    parser.add_argument("--output",    required=True, help="Empty output folder for clips, CSVs and the report")
    parser.add_argument("--accept_column_map", action="store_true",
                        help="Use the printed reference column rename map (only needed if the reference uses old names)")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_audio     = Path(args.raw_audio)
    reference_csv = Path(args.reference)
    output_dir    = Path(args.output)
    clips_dir     = output_dir / "clips"
    outputs_dir   = output_dir / "outputs"

    for path, flag in ((reference_csv, "--reference"), (output_dir, "--output")):
        if _inside_repo(path):
            print(f"[WARNING] {flag} is inside the repo ({path}). Do not commit data.")

    # Folder names give the recorder and date; check them before the slow steps
    folder_days = raw_folder_days(raw_audio)

    # Reference header first, so any rename map is confirmed before the slow steps
    reference = pd.read_csv(reference_csv)
    column_map = build_column_map(reference.columns)
    if column_map:
        print("Reference column rename map (reference -> pipeline):")
        for old, new in column_map.items():
            print(f"  {old!r} -> {new!r}")
        if not args.accept_column_map:
            print("Check the map above, then rerun with --accept_column_map to use it.")
            sys.exit(2)
        reference = reference.rename(columns=column_map)
    missing = [c for c in ["clip_name"] + ACOUSTIC if c not in reference.columns]
    if missing:
        raise SystemExit(f"Reference is missing columns {missing}. Its header is {list(reference.columns)}")
    reference = reference[["clip_name"] + ACOUSTIC].copy()
    reference["clip_name"] = strip_wav(reference["clip_name"])
    n_ref_dupes = int(reference["clip_name"].duplicated().sum())
    reference = reference.drop_duplicates("clip_name")

    if clips_dir.exists() and any(clips_dir.glob("*.wav")):
        raise SystemExit(f"{clips_dir} already has clips; use an empty --output folder so runs don't mix.")
    clips_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Same calls and output names as run_pipeline.py steps 1, 4 and 5
    print("STEP 1 - Splitting raw audio into 3-second clips...")
    split_all_audio(main_folder=str(raw_audio), output_folder=str(clips_dir))
    print("STEP 4 - Acoustic feature extraction...")
    acoustic_csv = outputs_dir / "acoustic_features.csv"
    process_split(split_name="Inference", split_dir=str(clips_dir), output_csv=str(acoustic_csv))
    print("STEP 5 - Metadata extraction...")
    metadata_csv = outputs_dir / "metadata.csv"
    extract_metadata(input_folder=str(clips_dir), output_csv=str(metadata_csv))

    # 1. Clip lengths
    clips, bad_clips = check_clip_lengths(clips_dir)

    # 2. Merge with the reference
    pipeline = pd.read_csv(acoustic_csv)
    pipeline["clip_name"] = strip_wav(pipeline["clip_name"])
    merged = pipeline.merge(reference, on="clip_name", suffixes=("_pipeline", "_reference"))
    matched = set(merged["clip_name"])
    unmatched_ref = sorted(set(reference["clip_name"]) - matched)
    expected = set(expected_reference_clips(reference["clip_name"], folder_days))
    recorders = sorted({recorder for _, recorder, _ in folder_days})
    dates = sorted({day.isoformat() for _, _, day in folder_days})

    # 3. Feature comparison
    stats = feature_stats(merged) if len(merged) else []
    comparison = compare_features(stats) if stats else None
    rolloff_max = float(reference["Spectral_Rolloff_85"].max())
    rolloff_flag = rolloff_max > ROLLOFF_LIMIT_HZ

    # 4. Recorder coverage
    metadata = pd.read_csv(metadata_csv)
    coverage, meta = recorder_coverage(metadata, matched)
    grid_suspects = unmatched_inside_audio(unmatched_ref, meta)

    # ── Report ────────────────────────────────────────────────────────────────
    # No local paths: placeholders for directories, plus the recorder folder and reference file names
    raw_folders = [folder for folder, _, _ in folder_days]
    generated, commit = f"{datetime.now():%Y-%m-%d %H:%M}", _git_commit()
    lines = [
        "# Runtime feature validation",
        "",
        f"- Generated: {generated}, repo commit `{commit}`",
        f"- Raw audio: `<raw_audio>` ({', '.join(raw_folders)})",
        f"- Reference: `<reference_csv>` ({reference_csv.name}, {len(reference):,} clips"
        + (f", {n_ref_dupes} duplicate clip_name rows dropped" if n_ref_dupes else "") + ")",
        "- Pipeline outputs: `<output>/outputs`",
        "",
        "## 1. Clip lengths",
        "",
        f"{len(clips):,} clips; expected {config.SAMPLES_PER_CLIP} samples at {config.TARGET_SAMPLE_RATE} Hz.",
        "",
    ]
    if len(bad_clips):
        lines += [f"**{len(bad_clips):,} clips have the wrong length or sample rate:**", "", md_table(bad_clips), ""]
    else:
        lines += ["All clips are exactly the expected length and sample rate.", ""]

    lines += ["## 2. Match with the reference", ""]
    if column_map:
        lines += ["Reference columns renamed (confirmed with --accept_column_map):", "",
                  md_table(pd.DataFrame(list(column_map.items()), columns=["Reference", "Pipeline"])), ""]
    else:
        lines += ["Reference already uses the pipeline column names; no renaming.", ""]
    lines += [
        f"- Pipeline clips: {len(pipeline):,}",
        f"- Reference clips: {len(reference):,}",
        f"- **Matched on clip_name: {len(merged):,}**",
        f"- Reference clips not produced by the pipeline: {len(unmatched_ref):,}"
        + (f" (e.g. {', '.join(unmatched_ref[:5])})" if unmatched_ref else ""),
        f"- Of those, inside the recorded audio (a pipeline clip from the same recorder starts < 3 s away): "
        f"{len(grid_suspects):,}"
        + (f" (e.g. {', '.join(grid_suspects[:5])}). These suggest a different cut grid, not missing audio."
           if grid_suspects else ""),
        "",
    ]

    lines += ["## 3. Acoustic features", "",
              f"Relative difference = |pipeline − reference| / |reference|; under {MATCH_TOL:g} counts as a match.", ""]
    if comparison is not None:
        lines += [md_table(comparison), ""]
    else:
        lines += ["No clips matched, so no features were compared.", ""]
    if rolloff_flag:
        lines += [f"**Flag:** the reference Spectral_Rolloff_85 goes up to {rolloff_max:,.0f} Hz, above "
                  f"{ROLLOFF_LIMIT_HZ:,.0f} Hz. The reference was computed at a higher sample rate than the "
                  f"pipeline's {ACOUSTIC_SR} Hz.", ""]
    else:
        lines += [f"Reference Spectral_Rolloff_85 max is {rolloff_max:,.0f} Hz (≤ {ROLLOFF_LIMIT_HZ:,.0f} Hz), "
                  f"consistent with {ACOUSTIC_SR} Hz.", ""]

    lines += ["## 4. Recorder coverage", "",
              "Clock hours (local time) with at least one clip; 05–08 means 05:00–08:59.", "",
              md_table(coverage), ""]

    report_path = output_dir / "validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # ── summary.json (for combine_reports.py; no local paths) ─────────────────
    summary = {
        "raw_folders": raw_folders,
        "recorders": recorders,
        "dates": dates,
        "generated": generated,
        "repo_commit": commit,
        "reference_file": reference_csv.name,
        "reference_clips_total": len(reference),
        "clips_produced": len(clips),
        "clips_wrong_length_or_rate": len(bad_clips),
        "reference_clips_expected": len(expected),
        "reference_clips_matched": len(expected & matched),
        "clips_compared": len(merged),
        "match_tolerance": MATCH_TOL,
        "features": {
            s["feature"]: {"median_rel_diff": _num(s["median"]), "max_rel_diff": _num(s["max"]),
                           "nan": s["nan"], "match": s["match"]}
            for s in stats
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Clips with wrong length/sample rate: {len(bad_clips):,} of {len(clips):,}")
    print(f"Matched to reference: {len(merged):,} of {len(reference):,} reference clips")
    print(f"Reference clips for {', '.join(recorders)} on {', '.join(dates)}: "
          f"matched {len(expected & matched):,} of {len(expected):,}")
    if comparison is not None:
        print(comparison[["Feature", "Median rel. diff", "Max rel. diff", "Result"]].to_string(index=False))
    if rolloff_flag:
        print(f"FLAG: reference Spectral_Rolloff_85 max {rolloff_max:,.0f} Hz > {ROLLOFF_LIMIT_HZ:,.0f} Hz")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
