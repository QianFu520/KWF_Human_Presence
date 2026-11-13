"""
score_test.py: final run. Retrain the LOCKED configs on train, choose each threshold
on validation (most recall with <= --episode-budget false-alert episodes per
recorder per day), score test ONCE, and save deployment bundles.

Feature sets (tune_val.FEATURE_SETS, built from the hp_features.py lists):
  23     candidate (no hour features, no weather)
  23rel  diagnostic only (23 with recorder-relative loudness), no bundle
  25     reference: relies on time-of-day shortcuts (schedule-inflated upper bound)
  27     reference: no hour features, but weather identifies the session
  29     reference: relies on time-of-day and weather/session shortcuts
--sets limits the run to some sets, so a set added later is scored on test once
without re-scoring the others. The 23/23rel/25/29 results (Nov 1) were produced
when 23 was still labelled "27"; the model and numbers are unchanged.

Results (in --out): results.json, test_summary.csv, test_by_event_type.csv,
val_summary.csv, test_scores_<set>.csv. Bundles (in --bundles, outside the repo):
<version>.joblib, one dict per model in Patrick's proposed format plus extra keys.

--dry-run runs the whole code path with validation standing in for test and
writes nothing to --bundles, so bugs show up before test is touched.

Usage (PowerShell):
  python score_test.py --data "C:\\BirdNET ONNX\\hp_audit\\training\\training_set_w10_weather.csv" `
      --out "C:\\BirdNET ONNX\\hp_audit\\models\\final_w10" --bundles "C:\\BirdNET ONNX\\hp_audit\\bundles"
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_val import CLIP_S, FEATURE_SETS, add_relative_loudness, episodes_per_day, threshold_table, \
    within_hour_pr_auc  # noqa: E402

# Locked Nov 1 2026 from tune_val.py (best validation PR-AUC per set). Do not change.
LOCKED = {
    "23": {"max_depth": None, "min_samples_leaf": 500, "l2_regularization": 0.0, "max_iter": 200},
    "23rel": {"max_depth": None, "min_samples_leaf": 20, "l2_regularization": 10.0, "max_iter": 200},
    "25": {"max_depth": 6, "min_samples_leaf": 5000, "l2_regularization": 10.0, "max_iter": 50},
    "29": {"max_depth": 6, "min_samples_leaf": 5000, "l2_regularization": 0.0, "max_iter": 200},
    # Locked Nov 1 2026 (tune_val.py --sets 27, same grid).
    "27": {"max_depth": 4, "min_samples_leaf": 500, "l2_regularization": 10.0, "max_iter": 200},
}
ROLE = {"23": "candidate", "23rel": "diagnostic", "25": "reference", "27": "reference", "29": "reference"}
BUNDLED = ["23", "25", "27", "29"]
USES_WEATHER = {"27", "29"}
TRAINED_ON = ("pipeline_out features_all{weather}.csv: 616,381 clips, AM1-AM6, 2025-03-17..21; "
              "split: event groups + 20-min shared negative blocks, 1-min guard, seed 42 (fit on train only); "
              "label window: w10 (+/-10 min)")
SHORTCUT_NOTE = {"25": " | REFERENCE ONLY: relies on time-of-day shortcuts (staging schedule)",
                 "27": " | REFERENCE ONLY: relies on weather/session shortcuts (weather identifies day and hour)",
                 "29": " | REFERENCE ONLY: relies on time-of-day and weather/session shortcuts"}


def metrics(d: pd.DataFrame, s: np.ndarray, thr: float, gap_s: float, max_s: float) -> dict:
    y = d.label.values
    p = s >= thr
    order = np.lexsort((d.ts.values, d.Recorder.values))
    neg = y == 0
    days = neg.sum() * CLIP_S / 86400
    both = 0 < y.sum() < len(y)
    return {
        "clips": int(len(y)), "positives": int(y.sum()),
        "pr_auc": average_precision_score(y, s) if both else None,
        "within_hour_pr_auc": within_hour_pr_auc(d, s)["weighted"] if both else None,
        "roc_auc": roc_auc_score(y, s) if both else None,
        "precision": precision_score(y, p, zero_division=0) if p.any() else None,
        "recall": recall_score(y, p) if y.any() else None,
        "f1": f1_score(y, p) if y.any() else None,
        "fa_clips_per_rec_day": float((p & neg).sum() / days) if days else None,
        "fa_episodes_per_rec_day": episodes_per_day(d.Recorder.values[order], d.ts.values[order],
                                                    (p & neg)[order], days, gap_s, max_s) if days else None,
    }


def summary(d: pd.DataFrame, s: np.ndarray, thr: float, gap_s: float, max_s: float) -> dict:
    out = {"ALL clips": metrics(d, s, thr, gap_s, max_s)}
    m = (d.predicted_label == "meaningful").values
    out["AED-passed only"] = metrics(d[m], s[m], thr, gap_s, max_s)
    for rec, idx in d.groupby("Recorder").indices.items():
        out[rec] = metrics(d.iloc[idx], s[idx], thr, gap_s, max_s)
    return out


def by_event_type(d: pd.DataFrame, s: np.ndarray, thr: float) -> dict:
    """Each type's positives against ALL negatives of the set."""
    neg = d.label.values == 0
    out = {}
    types = sorted({t for v in d.sim_type.dropna() for t in str(v).split(";") if t})
    for t in types:
        pos = d.sim_type.fillna("").str.split(";").apply(lambda v: t in v).values & (d.label.values == 1)
        m = pos | neg
        y = pos[m].astype(int)
        out[t] = {"positives": int(pos.sum()), "recall": float((s[pos] >= thr).mean()),
                  "pr_auc_vs_all_negatives": average_precision_score(y, s[m]),
                  "recorders": ",".join(sorted(d.Recorder[pos].unique()))}
    return out


def git_commit() -> str:
    repo = Path(__file__).resolve().parents[2]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "HP_Model/retraining", "inference_pipeline"],
                           cwd=repo, capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bundles", required=True, type=Path)
    ap.add_argument("--episode-budget", type=float, default=5)
    ap.add_argument("--episode-gap", type=float, default=30)
    ap.add_argument("--episode-max", type=float, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="use validation as a stand-in for test; no bundles")
    ap.add_argument("--sets", nargs="*", default=list(LOCKED), help="feature sets to run (default: all locked)")
    args = ap.parse_args()
    assert all(s in LOCKED for s in args.sets), f"unlocked set in {args.sets}; lock its config first"
    args.out.mkdir(parents=True, exist_ok=True)
    assert args.episode_budget in (2, 5, 10), "threshold_table only computes budgets 2/5/10"

    df = pd.read_csv(args.data, dtype={"sim_type": str, "event_id": str})
    if args.dry_run:
        df = df[df.split != "test"]
    df["ts"] = pd.to_datetime(df.Timestamp.astype(str), format="%Y%m%d_%H%M%S")
    commit, today = git_commit(), dt.date.today()

    results, test_rows, type_rows, val_rows = {}, [], [], []
    for name in args.sets:
        cfg, feats = LOCKED[name], FEATURE_SETS[name]
        d = add_relative_loudness(df) if name.endswith("rel") else df
        tr = d[d.split == "train"]
        va = d[d.split == "val"].reset_index(drop=True)
        te = (va if args.dry_run else d[d.split == "test"]).reset_index(drop=True)

        scaler = StandardScaler().fit(tr[feats])  # DataFrame -> feature_names_in_ is set
        model = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, learning_rate=0.1,
                                               random_state=args.seed, **cfg).fit(scaler.transform(tr[feats]),
                                                                                    tr.label)
        assert list(model.classes_) == [0, 1]
        s_va = model.predict_proba(scaler.transform(va[feats]))[:, 1]
        row = next(r for r in threshold_table(va, s_va, args.episode_gap, args.episode_max)
                   if r["episode_budget"] == args.episode_budget)
        thr = float(row["threshold"])
        rule = (f"most validation recall with <= {args.episode_budget:g} false-alert episodes per recorder per day "
                f"(episode = FP clips <{args.episode_gap:g} s apart on one recorder, capped at {args.episode_max:g} s)")

        s_te = model.predict_proba(scaler.transform(te[feats]))[:, 1]  # the one test scoring
        val_sum, test_sum = summary(va, s_va, thr, args.episode_gap, args.episode_max), \
            summary(te, s_te, thr, args.episode_gap, args.episode_max)
        types = by_event_type(te, s_te, thr)
        results[name] = {"role": ROLE[name], "config": cfg, "threshold": thr, "threshold_rule": rule,
                         "val_within_hour_detail": within_hour_pr_auc(va, s_va),
                         "test_within_hour_detail": within_hour_pr_auc(te, s_te),
                         "val": val_sum, "test": test_sum, "test_by_event_type": types}
        test_rows += [{"set": name, "subset": k, **v} for k, v in test_sum.items()]
        val_rows += [{"set": name, "subset": k, **v} for k, v in val_sum.items()]
        type_rows += [{"set": name, "event_type": k, **v} for k, v in types.items()]
        te.assign(score=s_te)[["clip_name", "Recorder", "Timestamp", "label", "sim_type", "predicted_label",
                               "score"]].to_csv(args.out / f"test_scores_{name}.csv", index=False)
        print(f"[{name}] threshold {thr:.4f} | test PR-AUC {test_sum['ALL clips']['pr_auc']:.3f} "
              f"within-hour {test_sum['ALL clips']['within_hour_pr_auc']:.3f}", flush=True)

        if name in BUNDLED and not args.dry_run:
            version = f"hp-{name}-{today:%Y%m%d}"
            v = val_sum["ALL clips"]
            bundle = {
                "model": model, "scaler": scaler, "features": list(feats), "feature_set": name,
                "threshold": thr, "version": version,
                "trained_on": TRAINED_ON.format(weather="_weather" if name in USES_WEATHER else "")
                + SHORTCUT_NOTE.get(name, ""),
                "created": f"{today:%Y-%m-%d}",
                "sklearn_version": sklearn.__version__,
                "val_metrics": {"precision": v["precision"], "recall": v["recall"],
                                "fa_episodes_per_rec_day": v["fa_episodes_per_rec_day"]},
                "threshold_rule": rule, "git_commit": commit, "role": ROLE[name],
            }
            args.bundles.mkdir(parents=True, exist_ok=True)
            path = args.bundles / f"{version}.joblib"
            assert not path.exists(), f"{path} exists; use a b/c suffix for a second bundle the same day"
            joblib.dump(bundle, path)
            print(f"  saved {path}")

    pd.DataFrame(test_rows).to_csv(args.out / "test_summary.csv", index=False)
    pd.DataFrame(val_rows).to_csv(args.out / "val_summary.csv", index=False)
    pd.DataFrame(type_rows).to_csv(args.out / "test_by_event_type.csv", index=False)
    (args.out / "results.json").write_text(json.dumps({"git_commit": commit, "dry_run": args.dry_run,
                                                        "models": results}, indent=2, default=float))
    print(f"Wrote results to {args.out}{' (DRY RUN: test = validation)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
