"""
KWF Human Presence — Inference Pipeline
========================================
Master script that runs all steps in order.

Usage:
    python run_pipeline.py --raw_audio /path/to/raw_audio --output /path/to/output
    python run_pipeline.py --raw_audio /path/to/raw_audio --output /path/to/output --no_weather
    python run_pipeline.py --raw_audio /path/to/raw_audio --output /path/to/output --no_weather --bundle /path/to/hp-23-YYYYMMDD.joblib

Pipeline steps:
    1. Split raw audio into 3-second clips
    2. Tiny CNN (AED) — score every clip meaningful / not_meaningful
    3. Compressed BirdNET — species + confidence on every clip
    4. Acoustic feature extraction
    5. Metadata extraction
    6. Weather data fetch + matching (skipped with --no_weather; needs internet)
    7. Feature engineering (sentinel species encoding + Patrick's features)
       — 29 features with weather, 25 with --no_weather. Every clip is kept and
       marked with AED's not_meaningful_prob and predicted_label.
    8. HP Model inference with --bundle (outputs/predictions.csv); skipped without it.
       A 27/29-feature bundle needs weather, a 23/25-feature bundle needs --no_weather.
    9. Output results

Dependencies:
    - PyTorch       (Tiny CNN)
    - onnxruntime   (Compressed BirdNET)
    - librosa       (acoustic features)
    - scikit-learn  (HP Model)
    - requests      (Open-Meteo weather API; not used with --no_weather)
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
SCRIPTS_DIR  = PIPELINE_DIR / "scripts"
MODELS_DIR   = PIPELINE_DIR / "models"
sys.path.insert(0, str(SCRIPTS_DIR))

# ── Script imports ────────────────────────────────────────────────────────────
from audio_split_rename      import split_all_audio
from aed_inference           import score_folder
from run_birdnet             import process_all_clips
from extract_acoustic_features import process_split
from metadata_extraction     import extract_metadata
from weather_info            import get_weather_info_all
from weather_matching        import weather_match
from feature_engineering     import run_feature_engineering
from hp_inference            import BundleError, check_weather_mode, load_bundle, predict_csv

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(PIPELINE_DIR / "pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step1_split(raw_audio_dir, clips_dir):
    log.info("STEP 1 — Splitting raw audio into 3-second clips...")
    split_all_audio(
        main_folder=str(raw_audio_dir),
        output_folder=str(clips_dir),
    )
    log.info(f"Clips saved to: {clips_dir}")


def step2_filter(clips_dir, outputs_dir):
    """Run Tiny CNN (AED) on every clip. Nothing is dropped; step 7 marks each clip with the result."""
    log.info("STEP 2 — Tiny CNN (AED): scoring every clip...")
    cnn_csv = outputs_dir / "tinycnn_predictions.csv"
    score_folder(
        folder_path=str(clips_dir),
        model_path=str(MODELS_DIR / "tinycnn_v3.pth"),
        output_csv=str(cnn_csv),
    )

    # Return the meaningful clip names (steps 3 and 4 take them but don't filter on
    # them) and the AED CSV, which step 7 joins onto every clip.
    df = pd.read_csv(cnn_csv)
    meaningful = df[df["predicted_label"] == "meaningful"]["clip_name"].tolist()
    log.info(f"Meaningful clips: {len(meaningful):,} / {len(df):,} total")
    return meaningful, cnn_csv


def step3_birdnet(clips_dir, meaningful_clips, outputs_dir):
    """
    Run compressed BirdNET (birdnet_fp16_pruned493.onnx via onnxruntime)
    on every clip, filtered to the regional allow-list.
    """
    log.info("STEP 3 — Compressed BirdNET: species + confidence...")
    birdnet_csv = outputs_dir / "birdnet_output.csv"

    # Runs on every clip; meaningful_clips is not used.
    process_all_clips(
        clips_folder=str(clips_dir),
        output_csv=str(birdnet_csv),
    )
    log.info(f"BirdNET output saved to: {birdnet_csv}")
    return birdnet_csv


def step4_acoustic(clips_dir, meaningful_clips, outputs_dir):
    """Extract 10 acoustic features from all clips (Volume_Spike_15s in step 7 needs every clip)."""
    log.info("STEP 4 — Acoustic feature extraction...")
    acoustic_csv = outputs_dir / "acoustic_features.csv"
    process_split(
        split_name="Inference",
        split_dir=str(clips_dir),
        output_csv=str(acoustic_csv),
    )
    log.info(f"Acoustic features saved to: {acoustic_csv}")
    return acoustic_csv


def step5_metadata(clips_dir, outputs_dir):
    """Extract metadata (clip_name, Recorder, Timestamp, audio properties)."""
    log.info("STEP 5 — Metadata extraction...")
    metadata_csv = outputs_dir / "metadata.csv"
    extract_metadata(
        input_folder=str(clips_dir),
        output_csv=str(metadata_csv),
    )
    log.info(f"Metadata saved to: {metadata_csv}")
    return metadata_csv


def step6_weather(raw_audio_dir, metadata_csv, outputs_dir):
    """Fetch hourly weather from the Open-Meteo API for the raw recordings and match it to clips."""
    log.info("STEP 6 — Weather data fetch + matching...")
    weather_hourly_csv = outputs_dir / "weather_hourly.csv"
    weather_csv        = outputs_dir / "weather.csv"
    get_weather_info_all(audio_folder=str(raw_audio_dir), output_csv=str(weather_hourly_csv))
    log.info(f"Hourly weather saved to: {weather_hourly_csv}")
    weather_match(
        weather_csv=str(weather_hourly_csv),
        metadata_csv=str(metadata_csv),
        output_csv=str(weather_csv),
    )
    log.info(f"Per-clip weather saved to: {weather_csv}")
    return weather_csv


def step7_feature_engineering(acoustic_csv, birdnet_csv, metadata_csv, weather_csv, outputs_dir, aed_csv):
    """
    Encode sentinel species and engineer Patrick's cross-signal features, then mark every
    clip with its AED result (not_meaningful_prob, predicted_label from aed_csv). No clip is dropped.
    weather_csv=None (--no_weather) leaves out the weather columns and Volume_Wind_Ratio.
    """
    if weather_csv is None:
        log.info("STEP 7 — Feature engineering (sentinel species + Patrick's features, no weather)...")
    else:
        log.info("STEP 7 — Feature engineering (sentinel species + Patrick's features)...")
    features_csv = outputs_dir / "features.csv"
    df = run_feature_engineering(
        acoustic_csv=str(acoustic_csv),
        birdnet_csv=str(birdnet_csv),
        metadata_csv=str(metadata_csv),
        weather_csv=None if weather_csv is None else str(weather_csv),
        output_csv=str(features_csv),
        aed_csv=str(aed_csv),
    )
    labels = df["predicted_label"].fillna("no AED result").value_counts()
    log.info(f"Kept all {len(df):,} clips. AED labels: "
             + ", ".join(f"{label} {n:,}" for label, n in labels.items()))
    log.info(f"Full feature set saved to: {features_csv}")
    return features_csv


def step8_hp_model(features_csv, outputs_dir, bundle):
    """
    Score every clip in features.csv with the HP bundle (loaded and checked by
    hp_inference.load_bundle in main) and write predictions.csv: clip_name,
    probability, label, plus the AED columns. Skipped when no bundle is given.
    Feature names and order come only from the bundle and scripts/hp_features.py.
    """
    if bundle is None:
        log.info("STEP 8 — HP Model inference: skipped")
        log.warning("No --bundle given; no HP predictions. Pass --bundle <path to .joblib> to run step 8.")
        log.info(f"Feature set ready at: {features_csv}")
        return None

    log.info(f"STEP 8 — HP Model inference ({bundle['version']})...")
    predictions_csv = outputs_dir / "predictions.csv"
    predictions = predict_csv(bundle, features_csv, predictions_csv)
    log.info(f"Clips flagged: {int(predictions['label'].sum()):,} / {len(predictions):,} "
             f"(threshold {bundle['threshold']:.4f})")
    log.info(f"Predictions saved to: {predictions_csv}")
    return predictions_csv


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KWF Human Presence Inference Pipeline")
    parser.add_argument("--raw_audio", required=True, help="Folder containing raw AudioMoth recordings")
    parser.add_argument("--output",    required=True, help="Output folder for all intermediate and final results")
    parser.add_argument("--no_weather", action="store_true", help="Skip step 6 (no internet needed) and build features without weather")
    parser.add_argument("--bundle", help="HP model bundle (.joblib) for step 8; without it, step 8 is skipped")
    args = parser.parse_args()

    # Load and check the bundle before step 1, so a wrong bundle fails in seconds, not after steps 1-7.
    bundle = None
    if args.bundle:
        try:
            bundle = load_bundle(args.bundle)
            check_weather_mode(bundle, args.no_weather)
        except (BundleError, OSError) as exc:
            parser.error(f"--bundle: {exc}")

    raw_audio_dir = Path(args.raw_audio)
    output_dir    = Path(args.output)
    clips_dir     = output_dir / "clips"
    outputs_dir   = output_dir / "outputs"

    clips_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("KWF Human Presence Inference Pipeline — Starting")
    log.info(f"Raw audio : {raw_audio_dir}")
    log.info(f"Output    : {output_dir}")
    log.info(f"Weather   : {'off (--no_weather)' if args.no_weather else 'on'}")
    log.info(f"HP bundle : {bundle['version'] if bundle else 'none (step 8 skipped)'}")
    log.info("=" * 60)

    step1_split(raw_audio_dir, clips_dir)
    meaningful_clips, aed_csv = step2_filter(clips_dir, outputs_dir)
    birdnet_csv          = step3_birdnet(clips_dir, meaningful_clips, outputs_dir)
    acoustic_csv         = step4_acoustic(clips_dir, meaningful_clips, outputs_dir)
    metadata_csv         = step5_metadata(clips_dir, outputs_dir)
    if args.no_weather:
        log.info("STEP 6 — Skipped (--no_weather)")
        weather_csv = None
    else:
        weather_csv = step6_weather(raw_audio_dir, metadata_csv, outputs_dir)
    features_csv         = step7_feature_engineering(acoustic_csv, birdnet_csv, metadata_csv, weather_csv, outputs_dir, aed_csv)
    step8_hp_model(features_csv, outputs_dir, bundle)

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
