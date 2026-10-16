"""
Score a validate_aed.py field run (task 4) against the field manifest.

Reads <run>/per_clip.csv and the manifest, and writes <run>/field_scores.md and
field_scores.csv with:
  - API vs in-process model agreement (does the service add any error?)
  - clip-level results per group at thresholds 0.5 / 0.7 / 0.95
  - event-level recall per event type (event hit = at least one clip kept)

Decision rule matches the API: label = not_meaningful if p > threshold.
The Digital Twin gate (confidence = 1 - p >= 0.3) equals threshold 0.7.

Usage:
    python validation/score_field.py --run validation/results/<stamp> \
        --manifest validation/field/manifest.csv
"""
import argparse
import os

import pandas as pd

THRESHOLDS = [0.5, 0.7, 0.95]


def md(d):
    try:
        return d.to_markdown(index=False)
    except ImportError:  # tabulate not installed
        return "```\n" + d.to_string(index=False) + "\n```"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results/<stamp> folder from validate_aed.py")
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()

    pc = pd.read_csv(os.path.join(a.run, "per_clip.csv"))
    man = pd.read_csv(a.manifest, keep_default_na=False)
    df = man.merge(pc, on="clip_name", how="left")
    src = "p_api" if "p_api" in df and df["p_api"].notna().any() else "p_cli"
    df["p"] = df[src]
    lines = [f"# Field scores\n", f"Run: `{a.run}`  ", f"Scores from: `{src}`  ",
             f"Clips in manifest: {len(man)}; scored: {df.p.notna().sum()}; "
             f"errors: {(df.get('preprocess_error', pd.Series(dtype=str)).fillna('') != '').sum()}\n"]

    # API vs model
    if {"p_api", "p_cli"} <= set(df.columns) and df.p_api.notna().any():
        both = df.dropna(subset=["p_api", "p_cli"])
        diff = (both.p_api - both.p_cli).abs()
        lab_diff = ((both.p_api > 0.5) != (both.p_cli > 0.5)).sum()
        lines += ["## API vs model", f"- clips compared: {len(both)}",
                  f"- max |p_api - p_model|: {diff.max():.2e} (API rounds to 4 decimals)",
                  f"- label disagreements at 0.5: {lab_diff}\n"]

    ok = df[df.p.notna()]
    rows = []
    for (g, exp), d in ok.groupby(["group", "expected_label"]):
        r = {"group": g, "expected": exp, "clips": len(d)}
        for t in THRESHOLDS:
            correct = (d.p > t) if exp == "not_meaningful" else (d.p <= t)
            r[f"clip_correct@{t}"] = round(correct.mean(), 3)
        ev = d[d.event_id != ""]
        if len(ev):
            r["events"] = ev.event_id.nunique()
            for t in THRESHOLDS:
                hit = ev.groupby("event_id").p.apply(lambda s: (s <= t).any())
                r[f"event_recall@{t}"] = round(hit.mean(), 3)
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.run, "field_scores.csv"), index=False)
    lines += ["## Results by group",
              "clip_correct = share of clips given the expected label; "
              "event_recall = share of events with at least one clip kept as meaningful.\n",
              md(tab), "",
              "## Misses at 0.5 (meaningful clips scored not_meaningful), lowest-confidence first\n",
              ok[(ok.expected_label == "meaningful") & (ok.p > 0.5)]
                .sort_values("p", ascending=False)[["clip_name", "group", "p"]].head(20).pipe(md)]
    with open(os.path.join(a.run, "field_scores.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
