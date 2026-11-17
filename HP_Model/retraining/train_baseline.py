"""
train_baseline.py: train the HP model on one training set, pick the threshold on
validation, and score test ONCE.

Model: StandardScaler (fit on train, saved as hp_scaler.pkl for the deployment
interface) + HistGradientBoostingClassifier with class_weight="balanced".
Early stopping is OFF: sklearn's internal early-stopping split is random by clip,
which would leak neighbouring clips.

Thresholds (both chosen on validation, before test is scored):
  * primary   = maximum F1 on validation (all clips)
  * fa_budget = highest validation recall with <= --fa-budget false alerts per recorder per day

False alerts per recorder per day = false-positive negative clips / recorder-days of
negative audio in that set (clips x 3 s / 86,400). "Episodes" merge false-positive
clips on the same recorder less than --episode-gap seconds apart into one alert.

Reports on test (all clips, AED-passed only, per recorder) and on validation per
recorder (AM3's positives are only in validation; validation was used for the
threshold, so those numbers are optimistic).

Usage (PowerShell):
  python train_baseline.py --data "C:\\BirdNET ONNX\\hp_audit\\training\\training_set_w10.csv" `
      --out "C:\\BirdNET ONNX\\hp_audit\\models\\hp25_w10_baseline"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, \
    recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_training_set import FEATURES_25  # noqa: E402

CLIP_S = 3


def fa_per_rec_day(d: pd.DataFrame, pred: np.ndarray) -> float:
    neg = d.label.values == 0
    days = neg.sum() * CLIP_S / 86400
    return float((pred & neg).sum() / days) if days else float("nan")


def episodes_per_rec_day(d: pd.DataFrame, pred: np.ndarray, gap_s: float) -> float:
    neg = d.label.values == 0
    days = neg.sum() * CLIP_S / 86400
    fp = d.loc[pred & neg, ["Recorder", "ts"]].sort_values(["Recorder", "ts"])
    if fp.empty:
        return 0.0
    new = fp.groupby("Recorder").ts.diff().dt.total_seconds().fillna(np.inf) > gap_s
    return float(new.sum() / days)


def metrics(d: pd.DataFrame, score: np.ndarray, thr: float, gap_s: float) -> dict:
    y = d.label.values
    pred = score >= thr
    has_both = 0 < y.sum() < len(y)
    return {
        "clips": int(len(y)), "positives": int(y.sum()),
        "pr_auc": float(average_precision_score(y, score)) if has_both else None,
        "roc_auc": float(roc_auc_score(y, score)) if has_both else None,
        "precision": float(precision_score(y, pred, zero_division=0)) if pred.any() else None,
        "recall": float(recall_score(y, pred, zero_division=0)) if y.sum() else None,
        "f1": float(f1_score(y, pred, zero_division=0)) if y.sum() else None,
        "false_alert_clips_per_rec_day": fa_per_rec_day(d, pred),
        "false_alert_episodes_per_rec_day": episodes_per_rec_day(d, pred, gap_s),
    }


def report(d: pd.DataFrame, score: np.ndarray, thr: float, gap_s: float) -> pd.DataFrame:
    rows = {"ALL clips": metrics(d, score, thr, gap_s)}
    m = (d.predicted_label == "meaningful").values
    rows["AED-passed only"] = metrics(d[m], score[m], thr, gap_s)
    for rec, idx in d.groupby("Recorder").indices.items():
        rows[rec] = metrics(d.iloc[idx], score[idx], thr, gap_s)
    return pd.DataFrame(rows).T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fa-budget", type=float, default=10, help="false alerts per recorder per day (clips)")
    ap.add_argument("--episode-gap", type=float, default=30, help="seconds; closer false alerts = one episode")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    df = pd.read_csv(args.data, dtype={"sim_type": str, "event_id": str})
    df["ts"] = pd.to_datetime(df.Timestamp.astype(str), format="%Y%m%d_%H%M%S")
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"])
    print(f"train {len(tr):,} ({tr.label.sum():,} pos) | val {len(va):,} ({va.label.sum():,} pos) | "
          f"test {len(te):,} ({te.label.sum():,} pos)")

    scaler = StandardScaler().fit(tr[FEATURES_25])
    model = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=args.seed)
    model.fit(scaler.transform(tr[FEATURES_25]), tr.label)

    def score(d):
        return model.predict_proba(scaler.transform(d[FEATURES_25]))[:, 1]

    s_tr, s_va = score(tr), score(va)

    # ---- thresholds on validation
    prec, rec, thr = precision_recall_curve(va.label, s_va)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    thr_f1 = float(thr[np.argmax(f1)])
    days_va = (va.label == 0).sum() * CLIP_S / 86400
    neg_scores = np.sort(s_va[va.label.values == 0])[::-1]
    k = int(np.floor(args.fa_budget * days_va))  # allowed false-positive clips on val
    thr_fa = float(np.nextafter(neg_scores[k], 1)) if k < len(neg_scores) else 0.0
    thresholds = {"primary_max_f1": thr_f1, f"fa_budget_{args.fa_budget:g}": thr_fa}

    grid = []
    for t in sorted(set(np.quantile(s_va, np.linspace(0.5, 0.9999, 60))) | set(thresholds.values())):
        p = s_va >= t
        grid.append({"threshold": t, "precision": precision_score(va.label, p, zero_division=0),
                     "recall": recall_score(va.label, p), "f1": f1_score(va.label, p),
                     "fa_clips_per_rec_day": fa_per_rec_day(va, p),
                     "fa_episodes_per_rec_day": episodes_per_rec_day(va, p, args.episode_gap)})
    pd.DataFrame(grid).to_csv(args.out / "val_threshold_curve.csv", index=False)

    # ---- score test once (both pre-chosen thresholds)
    s_te = score(te)
    results = {}
    for name, t in thresholds.items():
        results[name] = {"threshold": t,
                         "train_all": metrics(tr, s_tr, t, args.episode_gap),
                         "val": report(va, s_va, t, args.episode_gap).to_dict("index"),
                         "test": report(te, s_te, t, args.episode_gap).to_dict("index")}
        print(f"\n===== threshold {name} = {t:.4f} =====")
        print("-- TEST --\n" + report(te, s_te, t, args.episode_gap).to_string())
        print("-- VAL (threshold chosen here; optimistic) --\n" + report(va, s_va, t, args.episode_gap).to_string())
    tr_auc = roc_auc_score(tr.label, s_tr)
    print(f"\nTrain ROC-AUC {tr_auc:.3f}, PR-AUC {average_precision_score(tr.label, s_tr):.3f} (overfitting check)")

    # ---- permutation importance on validation (PR-AUC drop)
    pi = permutation_importance(model, scaler.transform(va[FEATURES_25]), va.label, scoring="average_precision",
                                n_repeats=3, random_state=args.seed, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=FEATURES_25).sort_values(ascending=False)
    imp.to_csv(args.out / "val_permutation_importance.csv", header=["pr_auc_drop"])
    print("\nTop permutation importances (val PR-AUC drop):\n" + imp.head(10).to_string())

    # ---- save
    joblib.dump(model, args.out / "hp_model.pkl")
    joblib.dump(scaler, args.out / "hp_scaler.pkl")
    (args.out / "feature_order.json").write_text(json.dumps(FEATURES_25, indent=2))
    (args.out / "threshold.json").write_text(json.dumps({"threshold": thr_f1, "rule": "max F1 on validation",
                                                         "alternatives": thresholds}, indent=2))
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    for name, d, s in [("val", va, s_va), ("test", te, s_te)]:
        d.assign(score=s)[["clip_name", "Recorder", "Timestamp", "label", "sim_type", "predicted_label", "score"]] \
            .to_csv(args.out / f"{name}_predictions.csv", index=False)
    print(f"\nSaved model, scaler, feature order, thresholds, metrics and predictions to {args.out}")


if __name__ == "__main__":
    main()
