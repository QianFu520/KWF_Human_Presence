"""
Tests for step 7 marking clips with AED results instead of dropping them.

Uses small made-up CSVs in the formats steps 2-5 write (6 clips, one recorder).
No audio and no models.

Run with: python -m pytest inference_pipeline/tests -v
"""

import numpy as np
import pandas as pd
import pytest

import hp_features
from feature_engineering import AED_COLUMNS, SENTINEL_SPECIES, run_feature_engineering

ACOUSTIC_COLS = [
    "RMS_Energy", "Spectral_Contrast", "Spectral_Flatness", "Spectral_Bandwidth",
    "Spectral_Rolloff_85", "Onset_Strength", "MFCC_8", "MFCC_9", "MFCC_12", "MFCC_13",
]
NAMES = [f"Audio_Moth_1_20250321_1251{s:02d}" for s in range(0, 18, 3)]
PROBS = [0.02, 0.97, 0.10, 0.99, 0.51, 0.30]
LABELS = ["not_meaningful" if p >= 0.5 else "meaningful" for p in PROBS]


@pytest.fixture
def csvs(tmp_path):
    """Acoustic, BirdNET and metadata CSVs as steps 3-5 write them."""
    rng = np.random.default_rng(0)
    acoustic = pd.DataFrame(rng.random((len(NAMES), len(ACOUSTIC_COLS))), columns=ACOUSTIC_COLS)
    acoustic.insert(0, "clip_name", NAMES)  # step 4 writes the stem
    birdnet = pd.DataFrame({
        "clip_name": [f"{n}.wav" for n in NAMES],
        "species": [str([SENTINEL_SPECIES[0]]) if i % 2 else "[]" for i in range(len(NAMES))],
        "confidence": ["[0.8]" if i % 2 else "[]" for i in range(len(NAMES))],
    })
    metadata = pd.DataFrame({
        "clip_name": [f"{n}.wav" for n in NAMES],
        "Recorder": "Audio_Moth_1",
        "Timestamp": [n.removeprefix("Audio_Moth_1_") for n in NAMES],
    })
    paths = {k: tmp_path / f"{k}.csv" for k in ("acoustic", "birdnet", "metadata")}
    acoustic.to_csv(paths["acoustic"], index=False)
    birdnet.to_csv(paths["birdnet"], index=False)
    metadata.to_csv(paths["metadata"], index=False)
    return paths


def write_aed_csv(path, names, probs=PROBS):
    """tinycnn_predictions.csv as step 2 writes it."""
    pd.DataFrame({
        "clip_name": names,
        "audio_path": [f"clips/{n}" for n in names],
        "not_meaningful_prob": probs,
        "predicted_label": ["not_meaningful" if p >= 0.5 else "meaningful" for p in probs],
        "threshold": 0.5,
        "model_version": "tinycnn_v3",
    }).to_csv(path, index=False)
    return path


def run(csvs, tmp_path, **kwargs):
    return run_feature_engineering(
        acoustic_csv=csvs["acoustic"],
        birdnet_csv=csvs["birdnet"],
        metadata_csv=csvs["metadata"],
        weather_csv=None,
        output_csv=tmp_path / "features.csv",
        **kwargs,
    )


def aed_by_clip(df):
    return df.set_index("clip_name")[AED_COLUMNS].to_dict("index")


def expected_by_clip():
    return {n: {"not_meaningful_prob": p, "predicted_label": l} for n, p, l in zip(NAMES, PROBS, LABELS)}


def test_every_clip_kept_with_aed_columns(tmp_path, csvs):
    aed_csv = write_aed_csv(tmp_path / "tinycnn_predictions.csv", [f"{n}.wav" for n in NAMES])
    df = run(csvs, tmp_path, aed_csv=aed_csv)

    assert sorted(df["clip_name"]) == sorted(NAMES)
    assert set(df["predicted_label"]) == {"meaningful", "not_meaningful"}
    assert aed_by_clip(df) == expected_by_clip()

    written = pd.read_csv(tmp_path / "features.csv")
    assert len(written) == len(NAMES)
    assert written[AED_COLUMNS].notna().all().all()


@pytest.mark.parametrize("aed_names", [
    [f"{n}.wav" for n in NAMES],                                   # as step 2 writes them
    list(NAMES),                                                   # without .wav
    [f"{n}.wav" if i % 2 else n for i, n in enumerate(NAMES)],     # mixed
], ids=["with_wav", "without_wav", "mixed"])
def test_wav_and_bare_names_join(tmp_path, csvs, aed_names):
    aed_csv = write_aed_csv(tmp_path / "tinycnn_predictions.csv", aed_names)
    df = run(csvs, tmp_path, aed_csv=aed_csv)
    assert aed_by_clip(df) == expected_by_clip()


def test_missing_aed_row_kept_blank_with_warning(tmp_path, csvs, capsys):
    # AED skips clips it can't read, so the last clip has no row.
    aed_csv = write_aed_csv(tmp_path / "tinycnn_predictions.csv",
                            [f"{n}.wav" for n in NAMES[:-1]], PROBS[:-1])
    df = run(csvs, tmp_path, aed_csv=aed_csv)

    assert sorted(df["clip_name"]) == sorted(NAMES)
    missing = df.set_index("clip_name").loc[NAMES[-1], AED_COLUMNS]
    assert missing.isna().all()
    assert df.set_index("clip_name").loc[NAMES[:-1], AED_COLUMNS].notna().all().all()
    assert "WARNING: 1 of 6 clips have no AED result" in capsys.readouterr().out


def test_feature_values_identical_with_and_without_marking(tmp_path, csvs):
    plain = run(csvs, tmp_path)
    aed_csv = write_aed_csv(tmp_path / "tinycnn_predictions.csv", [f"{n}.wav" for n in NAMES])
    marked = run(csvs, tmp_path, aed_csv=aed_csv)

    assert list(marked.columns) == list(plain.columns) + AED_COLUMNS
    pd.testing.assert_frame_equal(marked[plain.columns], plain)


def test_keep_clips_still_drops_and_matches_marked_rows(tmp_path, csvs):
    aed_csv = write_aed_csv(tmp_path / "tinycnn_predictions.csv", [f"{n}.wav" for n in NAMES])
    marked = run(csvs, tmp_path, aed_csv=aed_csv)
    meaningful = [f"{n}.wav" for n, l in zip(NAMES, LABELS) if l == "meaningful"]
    kept = run(csvs, tmp_path, keep_clips=meaningful)

    expected = marked[marked["predicted_label"] == "meaningful"][kept.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, expected)


def test_aed_columns_are_not_hp_features(tmp_path, csvs):
    aed_csv = write_aed_csv(tmp_path / "tinycnn_predictions.csv", [f"{n}.wav" for n in NAMES])
    df = run(csvs, tmp_path, aed_csv=aed_csv)
    for col in hp_features.WEATHER:  # so the weather lists can be selected too
        df[col] = 1.0

    feature_lists = {k: v for k, v in vars(hp_features).items() if k.startswith("HP_FEATURES_")}
    assert feature_lists
    for name, features in feature_lists.items():
        assert not set(AED_COLUMNS) & set(features), name
        assert list(hp_features.select_features(df, features).columns) == list(features), name
