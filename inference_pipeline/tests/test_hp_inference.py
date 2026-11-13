"""
Tests for the HP bundle loader (hp_inference.py) and pipeline step 8.

Builds small fake bundles in the format of Coleman's hp-*-20261101 bundles
(StandardScaler fitted on a DataFrame + HistGradientBoostingClassifier) on
random data. No real bundles, no audio.

Run with: python -m pytest inference_pipeline/tests -v
"""

import logging
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import hp_inference
from hp_inference import BundleError, check_weather_mode, load_bundle, predict, predict_csv

sys.argv = sys.argv[:1]  # run_pipeline parses arguments only in main(), but keep pytest's out of the way
import run_pipeline  # noqa: E402

FEATURE_SETS = ["23", "25", "27", "29"]
ROLES = {"23": "candidate", "25": "reference", "27": "reference", "29": "reference"}  # as in Coleman's bundles
N_ROWS = 40


def features_df(features, n=N_ROWS, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random((n, len(features))), columns=list(features))


def make_bundle(feature_set="25", features=None, scaler_features=None, labels=(0, 1), **overrides):
    """A bundle dict like Coleman's: scaler fitted on a DataFrame, model on the scaled values."""
    features = list(hp_inference.FEATURE_LISTS[feature_set]) if features is None else list(features)
    X = features_df(scaler_features or features)
    y = np.array([labels[i % 2] for i in range(len(X))])
    scaler = StandardScaler().fit(X)
    model = HistGradientBoostingClassifier(max_iter=5, random_state=0).fit(scaler.transform(X), y)
    bundle = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "feature_set": feature_set,
        "threshold": 0.5,
        "version": f"hp-{feature_set}-20261101",
        "trained_on": "fake data for tests",
        "created": "2026-11-01",
        "sklearn_version": sklearn.__version__,
        "val_metrics": {"precision": 0.5, "recall": 0.5, "fa_episodes_per_rec_day": 1.0},
        "threshold_rule": "test",
        "git_commit": "0000000",
        "role": ROLES[feature_set] if feature_set in ROLES else "candidate",
    }
    bundle.update(overrides)
    return bundle


def save(tmp_path, bundle, name=None):
    path = tmp_path / f"{name or bundle['version']}.joblib"
    joblib.dump(bundle, path)
    return path


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_each_feature_set_loads(tmp_path, caplog, feature_set):
    with caplog.at_level(logging.INFO, logger="hp_inference"):
        bundle = load_bundle(save(tmp_path, make_bundle(feature_set)))
    assert bundle["feature_set"] == feature_set
    assert len(bundle["features"]) == int(feature_set)
    assert f"hp-{feature_set}-20261101 loaded" in caplog.text
    assert f"role {ROLES[feature_set]}" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]  # loading a reference bundle is fine


def test_candidate_role_loads_without_warning(tmp_path, caplog):
    path = save(tmp_path, make_bundle("23", role="candidate"))
    with caplog.at_level(logging.INFO, logger="hp_inference"):
        bundle = load_bundle(path)
    assert bundle["role"] == "candidate"
    assert "role candidate" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def swapped(features):
    features = list(features)
    features[0], features[1] = features[1], features[0]
    return features


@pytest.mark.parametrize("bundle_kwargs, message", [
    # features in the wrong order (scaler fitted the same way)
    (lambda: {"features": swapped(hp_inference.FEATURE_LISTS["25"]),
              "scaler_features": swapped(hp_inference.FEATURE_LISTS["25"])}, "don't match HP_FEATURES_25"),
    # features right, scaler fitted in the wrong order
    (lambda: {"scaler_features": swapped(hp_inference.FEATURE_LISTS["25"])}, "Scaler feature order"),
    # model trained on other labels
    (lambda: {"labels": (1, 2)}, "classes_ must be [0, 1]"),
    (lambda: {"labels": ("no", "yes")}, "classes_ must be [0, 1]"),
    # unknown feature set, and a feature set stored as a number
    (lambda: {"feature_set": "24", "features": hp_inference.FEATURE_LISTS["25"]}, "unknown feature_set '24'"),
    (lambda: {"feature_set": 25, "features": hp_inference.FEATURE_LISTS["25"]}, "unknown feature_set 25"),
    (lambda: {"threshold": 1.5}, "threshold must be a number between 0 and 1"),
], ids=["column_order", "scaler_order", "classes_1_2", "classes_strings",
        "unknown_feature_set", "feature_set_not_string", "threshold"])
def test_bad_bundles_are_rejected(tmp_path, bundle_kwargs, message):
    path = save(tmp_path, make_bundle(**bundle_kwargs()), name="bad")
    with pytest.raises(BundleError) as exc:
        load_bundle(path)
    assert message in str(exc.value)


def test_missing_keys_are_named(tmp_path):
    bundle = make_bundle()
    del bundle["threshold"], bundle["scaler"]
    with pytest.raises(BundleError, match=r"missing keys \['scaler', 'threshold'\]"):
        load_bundle(save(tmp_path, bundle))


def test_sklearn_version_mismatch_warns(tmp_path, caplog):
    path = save(tmp_path, make_bundle(sklearn_version="0.0.1"))
    with caplog.at_level(logging.WARNING, logger="hp_inference"):
        load_bundle(path)
    assert f"scikit-learn 0.0.1, but {sklearn.__version__} is installed" in caplog.text


def test_missing_role_warns(tmp_path, caplog):
    bundle = make_bundle()
    del bundle["role"]
    with caplog.at_level(logging.INFO, logger="hp_inference"):
        load_bundle(save(tmp_path, bundle))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["HP bundle hp-25-20261101 has no role (candidate / reference)."]
    assert "role not set" in caplog.text


@pytest.mark.parametrize("feature_set, no_weather, allowed", [
    ("23", True, True), ("25", True, True), ("27", False, True), ("29", False, True),
    ("23", False, False), ("25", False, False), ("27", True, False), ("29", True, False),
])
def test_weather_guard(feature_set, no_weather, allowed):
    bundle = {"feature_set": feature_set, "version": f"hp-{feature_set}-20261101"}
    if allowed:
        check_weather_mode(bundle, no_weather)
    else:
        with pytest.raises(BundleError, match="Add --no_weather" if not no_weather else "but --no_weather was given"):
            check_weather_mode(bundle, no_weather)


def test_pipeline_refuses_weather_bundle_with_no_weather(tmp_path, monkeypatch, capsys):
    path = save(tmp_path, make_bundle("29"))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--raw_audio", str(tmp_path / "raw"),
                                      "--output", str(out), "--no_weather", "--bundle", str(path)])
    with pytest.raises(SystemExit):
        run_pipeline.main()
    assert "uses weather features (feature_set 29), but --no_weather was given" in capsys.readouterr().err
    assert not out.exists()  # stopped before step 1


def step7_features(feature_set, names, with_aed=True):
    """features.csv as step 7 writes it: clip_name without .wav, every feature, the AED columns."""
    df = features_df(hp_inference.FEATURE_LISTS[feature_set], n=len(names), seed=1)
    df.insert(0, "clip_name", names)
    df["species"] = "[]"
    if with_aed:
        df["not_meaningful_prob"] = np.linspace(0, 1, len(names))
        df["predicted_label"] = np.where(df["not_meaningful_prob"] >= 0.5, "not_meaningful", "meaningful")
    return df


def test_step8_writes_predictions_csv(tmp_path, caplog):
    bundle = make_bundle("23")  # role candidate
    names = [f"Audio_Moth_1_20250321_1251{s:02d}" for s in range(0, 18, 3)]
    features = step7_features("23", names)
    features_csv = tmp_path / "features.csv"
    features.to_csv(features_csv, index=False)

    with caplog.at_level(logging.INFO):
        predictions_csv = run_pipeline.step8_hp_model(features_csv, tmp_path, bundle)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    assert predictions_csv == tmp_path / "predictions.csv"
    written = pd.read_csv(predictions_csv)
    assert list(written.columns) == ["clip_name", "probability", "label", "not_meaningful_prob", "predicted_label"]
    assert list(written["clip_name"]) == [f"{n}.wav" for n in names]
    assert written["probability"].between(0, 1).all()
    assert (written["label"] == (written["probability"] >= bundle["threshold"]).astype(int)).all()
    pd.testing.assert_series_equal(written["predicted_label"], features["predicted_label"])


@pytest.mark.parametrize("feature_set", ["25", "27", "29"])
def test_step8_warns_for_reference_bundle(tmp_path, caplog, feature_set):
    bundle = make_bundle(feature_set)  # role reference
    features_csv = tmp_path / "features.csv"
    step7_features(feature_set, ["Audio_Moth_1_20250321_125100"]).to_csv(features_csv, index=False)

    with caplog.at_level(logging.INFO):
        predictions_csv = run_pipeline.step8_hp_model(features_csv, tmp_path, bundle)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert f"hp-{feature_set}-20261101 has role 'reference'" in warnings[0]
    assert "not for deployment" in warnings[0]
    assert predictions_csv.exists()  # still predicts; the warning doesn't stop the run


def test_step8_skips_without_bundle(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert run_pipeline.step8_hp_model(tmp_path / "features.csv", tmp_path, None) is None
    assert "No --bundle given" in caplog.text
    assert not (tmp_path / "predictions.csv").exists()


def test_predict_keeps_wav_and_works_without_aed_columns(tmp_path):
    bundle = make_bundle("25")
    names = ["Audio_Moth_2_20250321_140200.wav", "Audio_Moth_2_20250321_140203"]
    out = predict(bundle, step7_features("25", names, with_aed=False))
    assert list(out.columns) == ["clip_name", "probability", "label"]
    assert list(out["clip_name"]) == ["Audio_Moth_2_20250321_140200.wav", "Audio_Moth_2_20250321_140203.wav"]


def test_predict_names_missing_features(tmp_path):
    bundle = make_bundle("25")
    df = step7_features("25", ["a", "b"]).drop(columns=["MFCC_8"])
    with pytest.raises(KeyError, match="MFCC_8"):
        predict(bundle, df)


def test_predictions_match_the_model_directly(tmp_path):
    bundle = make_bundle("27")
    df = step7_features("27", [f"clip_{i}" for i in range(10)])
    out = predict_csv(bundle, _write(df, tmp_path / "features.csv"), tmp_path / "predictions.csv")
    expected = bundle["model"].predict_proba(bundle["scaler"].transform(df[bundle["features"]]))[:, 1]
    np.testing.assert_allclose(out["probability"], expected)


def _write(df, path):
    df.to_csv(path, index=False)
    return path
