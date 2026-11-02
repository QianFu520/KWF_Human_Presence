"""
check_bundles.py: reload each HP model bundle and verify the format.

Checks (Patrick's proposed format + the extra keys):
  * keys present; model has predict_proba and classes_ == [0, 1]
  * scaler is a StandardScaler with feature_names_in_ == features (same order)
  * feature_set in {"23", "25", "27", "29"}, len(features) == int(feature_set), and
    features match hp_features.py (23 = HP_FEATURES_25 without hour_sin/hour_cos;
    27 = HP_FEATURES_29 without hour_sin/hour_cos, Qian's approved definition)
  * threshold is a float in (0, 1); version == "hp-<feature_set>-YYYYMMDD[b|c...]"
    and matches the file name; created == "YYYY-MM-DD" and matches the version date
  * trained_on is a non-empty string; sklearn_version == installed scikit-learn
  * val_metrics has precision, recall, fa_episodes_per_rec_day; threshold_rule and
    git_commit are non-empty
  * optional --data: predict_proba on a few rows works and returns values in [0, 1]

Usage (PowerShell):
  python check_bundles.py "C:\\BirdNET ONNX\\hp_audit\\bundles" --data "C:\\BirdNET ONNX\\hp_audit\\training\\training_set_w10_weather.csv"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "inference_pipeline" / "scripts"))
from hp_features import HP_FEATURES_25, HP_FEATURES_29  # noqa: E402

HOUR = ("hour_sin", "hour_cos")
EXPECTED = {"23": [f for f in HP_FEATURES_25 if f not in HOUR],  # no hour, no weather
            "25": list(HP_FEATURES_25),
            "27": [f for f in HP_FEATURES_29 if f not in HOUR],  # Qian's approved 27: weather, no hour
            "29": list(HP_FEATURES_29)}
KEYS = ["model", "scaler", "features", "feature_set", "threshold", "version", "trained_on", "created",
        "sklearn_version", "val_metrics", "threshold_rule", "git_commit"]


def check(path: Path, sample: pd.DataFrame | None) -> list[str]:
    errs = []
    b = joblib.load(path)
    if not isinstance(b, dict):
        return ["not a dict"]
    errs += [f"missing key {k}" for k in KEYS if k not in b]
    if errs:
        return errs
    m, sc, feats, fs = b["model"], b["scaler"], b["features"], b["feature_set"]
    if not hasattr(m, "predict_proba"):
        errs.append("model has no predict_proba")
    if list(getattr(m, "classes_", [])) != [0, 1]:
        errs.append(f"classes_ = {getattr(m, 'classes_', None)}")
    if not isinstance(sc, StandardScaler):
        errs.append(f"scaler is {type(sc).__name__}")
    if list(getattr(sc, "feature_names_in_", [])) != list(feats):
        errs.append("scaler.feature_names_in_ != features")
    if fs not in EXPECTED:
        errs.append(f"feature_set {fs!r}")
    elif list(feats) != EXPECTED[fs]:
        errs.append(f"features differ from hp_features for set {fs}")
    if not (isinstance(b["threshold"], float) and 0 < b["threshold"] < 1):
        errs.append(f"threshold {b['threshold']!r}")
    if fs in EXPECTED and len(feats) != int(fs):
        errs.append(f"feature_set {fs} but {len(feats)} features")
    mv = re.fullmatch(r"hp-(23|25|27|29)-(\d{8})([b-z]?)", str(b["version"]))
    if not mv or mv.group(1) != fs:
        errs.append(f"version {b['version']!r}")
    if path.stem != b["version"]:
        errs.append(f"file name {path.name} != version")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(b["created"])) or \
            (mv and b["created"].replace("-", "") != mv.group(2)):
        errs.append(f"created {b['created']!r}")
    if not (isinstance(b["trained_on"], str) and b["trained_on"]):
        errs.append("trained_on empty")
    if b["sklearn_version"] != sklearn.__version__:
        errs.append(f"sklearn_version {b['sklearn_version']} != installed {sklearn.__version__}")
    vm = b["val_metrics"]
    if not (isinstance(vm, dict) and all(k in vm for k in ["precision", "recall", "fa_episodes_per_rec_day"])):
        errs.append(f"val_metrics {vm!r}")
    if not b["threshold_rule"] or not b["git_commit"]:
        errs.append("threshold_rule / git_commit empty")
    if sample is not None and not errs:
        p = m.predict_proba(sc.transform(sample[list(feats)]))[:, 1]
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            errs.append("predict_proba out of range")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundles", type=Path, help="folder of .joblib bundles (or one file)")
    ap.add_argument("--data", type=Path, help="CSV with feature columns for a predict smoke test")
    args = ap.parse_args()
    files = [args.bundles] if args.bundles.is_file() else sorted(args.bundles.glob("*.joblib"))
    sample = pd.read_csv(args.data, nrows=500) if args.data else None
    bad = 0
    for f in files:
        errs = check(f, sample)
        b = joblib.load(f)
        info = f"set {b.get('feature_set')}, {len(b.get('features', []))} features, threshold {b.get('threshold'):.4f}"
        print(f"{'OK  ' if not errs else 'FAIL'} {f.name}: {info}" + ("" if not errs else "\n     " + "\n     ".join(errs)))
        bad += bool(errs)
    print(f"{len(files) - bad}/{len(files)} bundles pass")
    sys.exit(1 if bad or not files else 0)


if __name__ == "__main__":
    main()
