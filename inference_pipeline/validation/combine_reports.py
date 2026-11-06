"""
Combine the summary.json files written by validate_features.py into one markdown table.

Usage:
    python inference_pipeline/validation/combine_reports.py /path/to/kwf_val --output combined_report.md
    python inference_pipeline/validation/combine_reports.py report_am4 report_am6 --output combined_report.md

Each argument is a validate_features.py --output folder (with summary.json in it), or a parent
folder whose sub-folders are (e.g. report_am1 ... report_am6). A run passes when every clip has
the expected length and sample rate, every reference clip for its recorder/date was matched, and
every acoustic feature matches. The table and totals are printed and written to --output.
"""

import argparse
import json
from pathlib import Path


def find_summaries(folders):
    paths = []
    for folder in map(Path, folders):
        if (folder / "summary.json").is_file():
            paths.append(folder / "summary.json")
            continue
        found = sorted(folder.glob("*/summary.json"))
        if not found:
            raise SystemExit(f"No summary.json in {folder} or its sub-folders")
        paths += found
    return paths


def load_summaries(paths):
    summaries, seen = [], {}
    for path in paths:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        # the same recorder/day twice would count its clips twice in the totals
        for folder in summary["raw_folders"]:
            if folder in seen:
                raise SystemExit(f"{folder} is in both {seen[folder]} and {path}")
            seen[folder] = path
        summaries.append(summary)
    return summaries


def run_result(s):
    """'pass', or 'FAIL: ' and what failed."""
    problems = []
    if s["clips_wrong_length_or_rate"]:
        problems.append(f"{s['clips_wrong_length_or_rate']:,} clips with the wrong length or sample rate")
    if not s["reference_clips_expected"]:
        problems.append("no reference clips for this recorder/date")
    elif s["reference_clips_matched"] < s["reference_clips_expected"]:
        missing = s["reference_clips_expected"] - s["reference_clips_matched"]
        problems.append(f"{missing:,} of {s['reference_clips_expected']:,} reference clips not matched")
    differ = [feature for feature, v in s["features"].items() if not v["match"]]
    if differ:
        problems.append("features differ: " + ", ".join(differ))
    return "FAIL: " + "; ".join(problems) if problems else "pass"


def largest_difference(s):
    """(max relative difference, feature), or None when no clips were compared."""
    diffs = [(v["max_rel_diff"], feature) for feature, v in s["features"].items() if v["max_rel_diff"] is not None]
    return max(diffs) if diffs else None


def combine(summaries):
    """The combined markdown table, and the totals lines."""
    rows = sorted(summaries, key=lambda s: (s["recorders"], s["dates"]))
    lines = [
        "| Recorder | Date | Clips checked | Reference clips matched / expected | Largest difference (feature) | Result |",
        "|---|---|---|---|---|---|",
    ]
    for s in rows:
        worst = largest_difference(s)
        largest = f"{worst[0]:.2e} ({worst[1]})" if worst else "-"
        lines.append(
            f"| {', '.join(s['recorders'])} | {', '.join(s['dates'])} | {s['clips_produced']:,} | "
            f"{s['reference_clips_matched']:,} / {s['reference_clips_expected']:,} | {largest} | {run_result(s)} |"
        )

    matched = sum(s["reference_clips_matched"] for s in rows)
    expected = sum(s["reference_clips_expected"] for s in rows)
    n_recorders = len({recorder for s in rows for recorder in s["recorders"]})
    recorders = f"{n_recorders} recorder{'s' if n_recorders != 1 else ''}"
    references = {(s["reference_file"], s["reference_clips_total"]) for s in rows}
    if len(references) == 1:
        (ref_file, ref_total), = references
        coverage = f"Matched {matched:,} of {ref_total:,} reference clips in {ref_file} across {recorders}"
    else:
        coverage = f"Matched {matched:,} reference clips across {recorders} (the runs used different reference files)"
    totals = [
        f"{coverage}; {expected:,} were expected for the recorders and dates tested.",
        f"{sum(run_result(s) == 'pass' for s in rows)} of {len(rows)} runs pass; "
        f"{sum(s['clips_produced'] for s in rows):,} clips checked, "
        f"{sum(s['clips_wrong_length_or_rate'] for s in rows):,} with the wrong length or sample rate.",
    ]
    compared = [(largest_difference(s), s) for s in rows if largest_difference(s)]
    if compared:
        (value, feature), s = max(compared, key=lambda pair: pair[0])
        totals.append(f"Largest difference overall: {value:.2e} ({feature}, {', '.join(s['recorders'])}).")
    return "\n".join(lines), totals


def main(argv=None):
    parser = argparse.ArgumentParser(description="Combine validate_features.py summary.json files into one table")
    parser.add_argument("folders", nargs="+", help="Output folders with summary.json, or a parent folder of them")
    parser.add_argument("--output", required=True, help="Markdown file to write the combined table to")
    args = parser.parse_args(argv)

    table, totals = combine(load_summaries(find_summaries(args.folders)))
    text = "\n".join([table, "", *totals, ""])
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    print(f"Combined report: {args.output}")


if __name__ == "__main__":
    main()
