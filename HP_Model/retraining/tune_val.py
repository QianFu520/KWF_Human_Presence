"""
tune_val.py: regularisation search on VALIDATION ONLY for several feature sets.
Test is never loaded.

Feature sets (same rows, split and labels; training_set_<window>_weather.csv):
  23      HP_FEATURES_25 without hour_sin / hour_cos (no hour, no weather)
  25      HP_FEATURES_25
  27      HP_FEATURES_29 without hour_sin / hour_cos (Qian's approved 27: weather, no hour)
  29      HP_FEATURES_29
  25rel   diagnostic: 25, with RMS_Energy and Volume_Spike_15s divided by the
          recorder's median RMS_Energy over its TRAIN clips (not a pipeline change)
  23rel   diagnostic: 23 with the same relative loudness

Caution: most positives are at 16:00-17:59 and event buffers remove nearly all
negatives at those hours, so hour features (and weather, which tracks the hour)
can separate classes without hearing anything. val_within_hour_pr_auc scores each
hour of day that has both classes separately, which removes that shortcut.

Model: StandardScaler + HistGradientBoostingClassifier(class_weight="balanced",
early_stopping=False). Grid over max_depth, min_samples_leaf, l2_regularization and
max_iter (learning_rate 0.1), plus the Baseline 1 defaults for reference.
The config with the best validation PR-AUC is kept per feature set.

For each kept config: validation PR-AUC overall and per recorder, and the
threshold giving the most validation recall with <= 2 / 5 / 10 false-alert
EPISODES per recorder per day (false-positive negative clips on one recorder
< --episode-gap s apart = one episode, capped at --episode-max s; per recorder-day
of negative audio).

Outputs (in --out): tuning_grid.csv, best_configs.json, threshold_tables.csv,
per_recorder_val.csv, importance_<set>.csv, val_scores_<set>.csv.

Usage (PowerShell):
  python tune_val.py --data "C:\\BirdNET ONNX\\hp_audit\\training\\training_set_w10_weather.csv" `
      --out "C:\\BirdNET ONNX\\hp_audit\\models\\tuning_w10"
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "inference_pipeline" / "scripts"))
from hp_features import HP_FEATURES_23, HP_FEATURES_25, HP_FEATURES_27, HP_FEATURES_29  # noqa: E402

CLIP_S = 3
FEATURE_SETS = {
    "23": list(HP_FEATURES_23),  # no hour, no weather
    "25": list(HP_FEATURES_25),
    "27": list(HP_FEATURES_27),  # Qian's approved 27: weather, no hour
    "29": list(HP_FEATURES_29),
    "25rel": list(HP_FEATURES_25),  # same names; two columns rescaled per recorder
    "23rel": list(HP_FEATURES_23),
}
# Spans small and large models: the 25-feature set prefers few, shallow trees (hour
# shortcut), while 27/29 do better with more capacity.
GRID = {"max_depth": [3, 4, 6, None], "min_samples_leaf": [20, 500, 5000],
        "l2_regularization": [0.0, 10.0], "max_iter": [20, 50, 100, 200]}
BASELINE1 = {"max_depth": None, "min_samples_leaf": 20, "l2_regularization": 0.0, "max_iter": 100}
BUDGETS = [2, 5, 10]


def add_relative_loudness(df: pd.DataFrame) -> pd.DataFrame:
    med = df[df.split == "train"].groupby("Recorder").RMS_Energy.median()
    out = df.copy()
    m = out.Recorder.map(med)
    out["RMS_Energy"] = out.RMS_Energy / m
    out["Volume_Spike_15s"] = out.Volume_Spike_15s / m
    return out


def episodes_per_day(rec: np.ndarray, ts: np.ndarray, fp: np.ndarray, days: float, gap_s: float,
                     max_s: float) -> float:
    """rec/ts sorted by (recorder, time); fp marks false-positive negative clips.
    A new episode starts on a new recorder, after a gap > gap_s, or once the current
    episode has lasted max_s (so an hour of continuous false alerts is not one alert)."""
    r, t = rec[fp], ts[fp].astype("datetime64[s]").astype(np.int64)
    n, start = 0, None
    for i in range(len(r)):
        if i == 0 or r[i] != r[i - 1] or t[i] - t[i - 1] > gap_s or t[i] - start >= max_s:
            n, start = n + 1, t[i]
    return float(n / days)


def within_hour_pr_auc(va: pd.DataFrame, s: np.ndarray) -> dict:
    """PR-AUC computed separately in each hour of day that has both classes, so
    hour-of-day alone can't separate them; plus a positive-weighted mean."""
    hour = va.ts.dt.hour.values
    out, w = {}, {}
    for h in np.unique(hour):
        m = hour == h
        y = va.label.values[m]
        if 0 < y.sum() < m.sum():
            out[f"h{h:02d}"] = average_precision_score(y, s[m])
            w[f"h{h:02d}"] = y.sum()
    out["weighted"] = sum(out[k] * w[k] for k in w) / sum(w.values()) if w else np.nan
    return out


def threshold_table(va: pd.DataFrame, s: np.ndarray, gap_s: float, max_s: float) -> list[dict]:
    order = np.lexsort((va.ts.values, va.Recorder.values))
    rec, ts, y, s = va.Recorder.values[order], va.ts.values[order], va.label.values[order], s[order]
    am3 = rec == "Audio_Moth_3"
    neg = y == 0
    days = neg.sum() * CLIP_S / 86400
    cands = np.unique(np.quantile(s, 1 - np.logspace(-5, -0.3, 400)))
    stats = []
    for t in cands:
        p = s >= t
        tp = (p & ~neg).sum()
        stats.append((t, episodes_per_day(rec, ts, p & neg, days, gap_s, max_s), tp / max(y.sum(), 1),
                      tp / p.sum() if p.sum() else np.nan, (p & neg).sum() / days,
                      (p & ~neg & am3).sum() / max((~neg & am3).sum(), 1)))
    st = pd.DataFrame(stats, columns=["threshold", "episodes", "recall", "precision", "fa_clips", "am3_recall"])
    rows = []
    for b in BUDGETS:
        ok = st[st.episodes <= b]
        r = ok.loc[ok.recall.idxmax()] if len(ok) else st.loc[st.threshold.idxmax()]
        rows.append({"episode_budget": b, **r.round(4).to_dict()})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sets", nargs="*", default=list(FEATURE_SETS))
    ap.add_argument("--episode-gap", type=float, default=30)
    ap.add_argument("--episode-max", type=float, default=300, help="seconds; longer alerting counts as more episodes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    df = pd.read_csv(args.data, dtype={"sim_type": str, "event_id": str})
    df = df[df.split.isin(["train", "val"])].reset_index(drop=True)  # test is never used here
    df["ts"] = pd.to_datetime(df.Timestamp.astype(str), format="%Y%m%d_%H%M%S")

    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())] + [BASELINE1]
    grid_rows, best, tables, per_rec = [], {}, [], []
    for name in args.sets:
        feats = FEATURE_SETS[name]
        d = add_relative_loudness(df) if name.endswith("rel") else df
        tr, va = d[d.split == "train"], d[d.split == "val"].reset_index(drop=True)
        scaler = StandardScaler().fit(tr[feats])
        Xtr, Xva = scaler.transform(tr[feats]), scaler.transform(va[feats])
        best_ap, best_model, best_cfg = -1, None, None
        for i, cfg in enumerate(configs):
            t0 = time.time()
            m = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, learning_rate=0.1,
                                               random_state=args.seed, **cfg).fit(Xtr, tr.label)
            s_va = m.predict_proba(Xva)[:, 1]
            s_tr = m.predict_proba(Xtr)[:, 1]
            row = {"set": name, **cfg, "baseline1_defaults": cfg is BASELINE1,
                   "val_pr_auc": average_precision_score(va.label, s_va), "val_roc_auc": roc_auc_score(va.label, s_va),
                   "train_pr_auc": average_precision_score(tr.label, s_tr), "fit_s": round(time.time() - t0, 1)}
            grid_rows.append(row)
            print(f"[{name}] {i + 1}/{len(configs)} {cfg} val PR-AUC {row['val_pr_auc']:.3f} "
                  f"(train {row['train_pr_auc']:.3f}) {row['fit_s']}s", flush=True)
            if cfg is not BASELINE1 and row["val_pr_auc"] > best_ap:
                best_ap, best_model, best_cfg = row["val_pr_auc"], m, cfg
        s = best_model.predict_proba(Xva)[:, 1]
        best[name] = {"config": best_cfg, "val_pr_auc": best_ap, "val_roc_auc": roc_auc_score(va.label, s),
                      "val_within_hour_pr_auc": within_hour_pr_auc(va, s)}
        for r in threshold_table(va, s, args.episode_gap, args.episode_max):
            tables.append({"set": name, **r})
        for rec, g in va.assign(score=s).groupby("Recorder"):
            per_rec.append({"set": name, "recorder": rec, "val_positives": int(g.label.sum()),
                            "val_pr_auc": average_precision_score(g.label, g.score) if g.label.any() else np.nan})
        pi = permutation_importance(best_model, Xva, va.label, scoring="average_precision", n_repeats=3,
                                    random_state=args.seed, n_jobs=-1)
        pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False) \
            .to_csv(args.out / f"importance_{name}.csv", header=["pr_auc_drop"])
        va.assign(score=s)[["clip_name", "Recorder", "Timestamp", "label", "predicted_label", "score"]] \
            .to_csv(args.out / f"val_scores_{name}.csv", index=False)

    pd.DataFrame(grid_rows).to_csv(args.out / "tuning_grid.csv", index=False)
    pd.DataFrame(tables).to_csv(args.out / "threshold_tables.csv", index=False)
    pd.DataFrame(per_rec).to_csv(args.out / "per_recorder_val.csv", index=False)
    (args.out / "best_configs.json").write_text(json.dumps(best, indent=2, default=float))
    print("\n" + json.dumps(best, indent=2, default=float))
    print(pd.DataFrame(tables).to_string(index=False))
    print(pd.DataFrame(per_rec).pivot(index="recorder", columns="set", values="val_pr_auc").round(3).to_string())


if __name__ == "__main__":
    main()
