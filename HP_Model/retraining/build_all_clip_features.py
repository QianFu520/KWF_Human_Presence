"""
build_all_clip_features.py: features for ALL clips, with the AED result joined in.

run_pipeline.py drops AED not-meaningful clips at the end of step 7, so its
features.csv only has meaningful clips and no AED column. For training we also
want the rejected clips (e.g. positives AED misses), so this script re-runs the
pipeline's own step 7 on the intermediate CSVs it already wrote, without the
AED filter, and joins not_meaningful_prob and predicted_label from step 2.
The pipeline itself is unchanged.

Inputs (in <pipeline output>/outputs/): acoustic_features.csv, birdnet_output.csv,
metadata.csv, tinycnn_predictions.csv, and weather.csv if present (unless --no_weather).
Output: <pipeline output>/outputs/features_all.csv (same columns as features.csv,
plus not_meaningful_prob and predicted_label).

Usage (PowerShell):
  python build_all_clip_features.py --pipeline_out "C:\\BirdNET ONNX\\pipeline_out\\Audio_Moth_4" --no_weather
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "inference_pipeline" / "scripts"))
from feature_engineering import run_feature_engineering  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline_out", required=True, type=Path, help="the --output folder given to run_pipeline.py")
    ap.add_argument("--no_weather", action="store_true", help="build the no-weather feature set")
    args = ap.parse_args()

    out = args.pipeline_out / "outputs"
    weather_csv = None if args.no_weather else out / "weather.csv"
    if weather_csv is not None and not weather_csv.exists():
        sys.exit(f"{weather_csv} not found; pass --no_weather or run the pipeline with weather.")

    features_all = out / "features_all.csv"
    df = run_feature_engineering(
        acoustic_csv=str(out / "acoustic_features.csv"),
        birdnet_csv=str(out / "birdnet_output.csv"),
        metadata_csv=str(out / "metadata.csv"),
        weather_csv=None if weather_csv is None else str(weather_csv),
        output_csv=str(features_all),
        keep_clips=None,
    )

    aed = pd.read_csv(out / "tinycnn_predictions.csv", usecols=["clip_name", "not_meaningful_prob", "predicted_label"])
    aed["clip_name"] = aed["clip_name"].str.replace(".wav", "", regex=False)
    df = df.merge(aed, on="clip_name", how="left", validate="one_to_one")
    n_missing = int(df.predicted_label.isna().sum())
    if n_missing:
        # AED skips clips it can't read; keep them, flagged by a blank label.
        print(f"WARNING: {n_missing:,} clips have no AED result (unreadable during step 2).")
    df.to_csv(features_all, index=False)

    counts = df.predicted_label.value_counts(dropna=False).to_dict()
    print(f"Wrote {len(df):,} rows to {features_all} | AED labels: {counts}")


if __name__ == "__main__":
    main()
