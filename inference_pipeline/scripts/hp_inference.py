"""
HP model inference: load a model bundle and predict human presence per clip.

A bundle is one joblib file holding a dict; the format is in
HP_Model/docs/hp_model_bundle.md. load_bundle() checks it against hp_features.py
before anything is scored, and predict() applies it to step 7's features.

Used by run_pipeline.py step 8 (--bundle) and, later, the HP FastAPI service.
"""

import logging

import joblib
import numpy as np
import pandas as pd
import sklearn

from hp_features import (
    HP_FEATURES_23,
    HP_FEATURES_25,
    HP_FEATURES_27,
    HP_FEATURES_29,
    check_scaler_alignment,
    select_features,
)

log = logging.getLogger(__name__)

FEATURE_LISTS = {"23": HP_FEATURES_23, "25": HP_FEATURES_25, "27": HP_FEATURES_27, "29": HP_FEATURES_29}
WEATHER_SETS = {"27", "29"}
REQUIRED_KEYS = ["model", "scaler", "features", "feature_set", "threshold", "version", "trained_on", "created"]
PASS_THROUGH = ["not_meaningful_prob", "predicted_label"]  # AED columns from step 7, if present


class BundleError(ValueError):
    """The bundle can't be used: missing keys, wrong features, scaler, classes or threshold."""


def load_bundle(path):
    """Load an HP bundle and check it. Raise BundleError naming the problem.

    Checks: required keys; feature_set is "23", "25", "27" or "29" and features equal
    the matching hp_features list exactly (names and order); the scaler was fitted on
    those features in that order; model.classes_ is [0, 1]; threshold is in (0, 1).
    Logs a warning, but still loads, if sklearn_version differs from the installed
    scikit-learn or if the bundle has no role.
    """
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise BundleError(f"{path}: expected a dict, got {type(bundle).__name__}")
    missing = [key for key in REQUIRED_KEYS if key not in bundle]
    if missing:
        raise BundleError(f"{path}: missing keys {missing}")

    feature_set = bundle["feature_set"]
    if feature_set not in FEATURE_LISTS:
        raise BundleError(
            f"{path}: unknown feature_set {feature_set!r}; expected one of {sorted(FEATURE_LISTS)} (as a string)"
        )
    expected = list(FEATURE_LISTS[feature_set])
    if list(bundle["features"]) != expected:
        raise BundleError(
            f"{path}: features don't match HP_FEATURES_{feature_set} in hp_features.py (same names, same order)\n"
            f"  bundle:   {list(bundle['features'])}\n"
            f"  expected: {expected}"
        )
    try:
        check_scaler_alignment(bundle["scaler"], expected)
    except ValueError as exc:
        raise BundleError(f"{path}: {exc}") from exc

    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        raise BundleError(f"{path}: model {type(model).__name__} has no predict_proba")
    classes = getattr(model, "classes_", None)
    try:
        classes_ok = classes is not None and [int(c) for c in classes] == [0, 1]
    except (TypeError, ValueError):
        classes_ok = False
    if not classes_ok:
        raise BundleError(f"{path}: model.classes_ must be [0, 1] (1 = human presence), got {classes}")

    threshold = bundle["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise BundleError(f"{path}: threshold must be a number between 0 and 1, got {threshold!r}")

    trained_with = bundle.get("sklearn_version")
    if trained_with is not None and trained_with != sklearn.__version__:
        log.warning(
            f"HP bundle {bundle['version']} was saved with scikit-learn {trained_with}, "
            f"but {sklearn.__version__} is installed; predictions may differ."
        )
    role = bundle.get("role")
    if role is None:
        log.warning(f"HP bundle {bundle['version']} has no role (candidate / reference).")
    log.info(
        f"HP bundle {bundle['version']} loaded: feature_set {feature_set}, "
        f"threshold {threshold:.4f}, role {role or 'not set'}"
    )
    return bundle


def check_weather_mode(bundle, no_weather):
    """Raise BundleError if the bundle's feature set doesn't match the weather setting."""
    feature_set, version = bundle["feature_set"], bundle["version"]
    if no_weather and feature_set in WEATHER_SETS:
        raise BundleError(
            f"Bundle {version} uses weather features (feature_set {feature_set}), but --no_weather was given. "
            "Run with weather, or use a 23- or 25-feature bundle."
        )
    if not no_weather and feature_set not in WEATHER_SETS:
        raise BundleError(
            f"Bundle {version} has no weather features (feature_set {feature_set}). "
            "Add --no_weather, or use a 27- or 29-feature bundle."
        )


def with_wav(names):
    """Clip names with .wav, the naming rule for CSVs and APIs; adds it where missing."""
    names = pd.Series(names, dtype=str)
    has_wav = names.str.lower().str.endswith(".wav")
    return names.where(has_wav, names + ".wav")


def predict(bundle, df):
    """Score every row of df with the bundle.

    Returns one row per clip: clip_name (with .wav), probability (of human presence),
    label (1 if probability >= the bundle threshold, else 0), plus not_meaningful_prob
    and predicted_label when df has them. Raises KeyError naming missing columns.
    """
    if "clip_name" not in df.columns:
        raise KeyError("Missing column: clip_name")
    X = select_features(df, bundle["features"])
    probability = bundle["model"].predict_proba(bundle["scaler"].transform(X))[:, 1]

    out = pd.DataFrame({
        "clip_name": with_wav(df["clip_name"].to_numpy()).to_numpy(),
        "probability": probability,
        "label": (probability >= bundle["threshold"]).astype(np.int64),
    })
    for col in PASS_THROUGH:
        if col in df.columns:
            out[col] = df[col].to_numpy()
    return out


def predict_csv(bundle, features_csv, output_csv):
    """Read step 7's features.csv, predict every clip and write predictions.csv."""
    predictions = predict(bundle, pd.read_csv(features_csv))
    predictions.to_csv(output_csv, index=False)
    return predictions
