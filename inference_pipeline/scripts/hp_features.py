"""
HP model feature names and order: the single source of truth.

Training and inference must both select columns with these lists so the model
and scaler always see the same features in the same order. Names match the
columns written by extract_acoustic_features.py and feature_engineering.py.

  HP_FEATURES_25  without weather
  HP_FEATURES_29  with weather (adds Temperature, Humidity, Windspeed and
                  Volume_Wind_Ratio, which is computed from Windspeed)

See the "Decision" section of HP_Model/docs/feature_pipeline_progress.md.
"""

from feature_engineering import SENTINEL_SPECIES

ACOUSTIC = [
    "RMS_Energy",
    "Spectral_Contrast",
    "Spectral_Flatness",
    "Spectral_Bandwidth",
    "Spectral_Rolloff_85",
    "Onset_Strength",
    "MFCC_8",
    "MFCC_9",
    "MFCC_12",
    "MFCC_13",
]

SENTINEL = list(SENTINEL_SPECIES)

BIRDNET = ["confidence"]

ENGINEERED_BASE = ["hour_sin", "hour_cos", "Eerie_Silence", "Volume_Spike_15s"]

WEATHER = ["Temperature", "Humidity", "Windspeed", "Volume_Wind_Ratio"]

HP_FEATURES_25 = ACOUSTIC + SENTINEL + BIRDNET + ENGINEERED_BASE
HP_FEATURES_29 = HP_FEATURES_25 + WEATHER

assert len(HP_FEATURES_25) == 25, f"HP_FEATURES_25 has {len(HP_FEATURES_25)} features, expected 25"
assert len(HP_FEATURES_29) == 29, f"HP_FEATURES_29 has {len(HP_FEATURES_29)} features, expected 29"


def select_features(df, feature_list):
    """Return df[feature_list] in exact order. Raise KeyError naming any missing columns."""
    missing = [name for name in feature_list if name not in df.columns]
    if missing:
        raise KeyError(f"Missing HP feature columns: {missing}")
    return df[list(feature_list)]


def check_scaler_alignment(scaler, feature_list):
    """Raise ValueError unless the scaler was fitted on exactly feature_list, in order."""
    fitted = getattr(scaler, "feature_names_in_", None)
    if fitted is None:
        raise ValueError(
            "Scaler has no feature_names_in_; fit it on a DataFrame so feature order can be checked."
        )
    if list(fitted) != list(feature_list):
        raise ValueError(
            "Scaler feature order does not match the HP feature list.\n"
            f"  scaler:   {list(fitted)}\n"
            f"  expected: {list(feature_list)}"
        )
