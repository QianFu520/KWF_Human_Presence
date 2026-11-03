"""
Threshold sweep on an existing validate_aed.py run: how many clips each cutoff
keeps (meaningful groups) or drops (background), per group and per staged event.

No rescoring: reads the run's per_clip.csv (CLI scores) and the manifest.
Rule: not_meaningful if not_meaningful_prob >= threshold.

Usage (from AED_Model/):
    python validation/threshold_sweep.py --run validation/results/20261016_044454 \
        --manifest validation/field/manifest_heldout.csv \
        --out validation/results/threshold_sweep_<date>.md
"""
import argparse

import pandas as pd

THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def table(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pc = pd.read_csv(f"{a.run}/per_clip.csv")[["clip_name", "p_cli"]]
    d = pd.read_csv(a.manifest, keep_default_na=False).merge(pc, on="clip_name", how="inner")
    d = d[d.expected_label.isin(["meaningful", "not_meaningful"]) & d.p_cli.notna()]

    def correct(x, t):
        return (x.p_cli >= t) if x.expected_label.iloc[0] == "not_meaningful" else (x.p_cli < t)

    grp = []
    for g, x in d.groupby("group"):
        r = {"group": g, "expected": x.expected_label.iloc[0], "clips": len(x),
             "events": x.event_id[x.event_id != ""].nunique()}
        r.update({f"{t}": f"{correct(x, t).mean():.1%}" for t in THRESHOLDS})
        grp.append(r)

    tot = []
    mean, bg = d[d.expected_label == "meaningful"], d[d.expected_label == "not_meaningful"]
    for t in THRESHOLDS:
        tot.append({"threshold": t, "meaningful kept": f"{(mean.p_cli < t).sum()} / {len(mean)}",
                    "background dropped": f"{(bg.p_cli >= t).sum()} / {len(bg)}"})

    ev = mean[mean.event_id != ""]
    per_ev = []
    for (g, e), x in ev.groupby(["group", "event_id"]):
        r = {"event": e, "clips": len(x)}
        r.update({f"{t}": f"{(x.p_cli < t).mean():.1%}" for t in THRESHOLDS})
        per_ev.append(r)
    per_ev = pd.DataFrame(per_ev)
    varies = per_ev[per_ev[[f"{t}" for t in THRESHOLDS]].nunique(axis=1) > 1]

    lines = ["# Threshold sweep", "",
             f"Run: `{a.run}` ({len(d)} labelled clips). Manifest: `{a.manifest}`.",
             "Rule: `not_meaningful` if `not_meaningful_prob >= threshold`. Cells: share of clips given the expected "
             "label (kept for meaningful groups, dropped for background).", "",
             "## Totals", "", table(pd.DataFrame(tot)), "",
             "## By group", "", table(pd.DataFrame(grp)), "",
             f"## By staged event ({ev.event_id.nunique()} events; share of clips kept)", "",
             f"Events whose result changes with the threshold: {len(varies)} of {len(per_ev)}.", "",
             table(per_ev), ""]
    open(a.out, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
