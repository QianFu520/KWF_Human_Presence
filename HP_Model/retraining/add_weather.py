"""
add_weather.py: run the pipeline's weather step on already-processed recorders.

run_pipeline.py's step 6 reads raw recordings to decide which hours to fetch, but
the re-cut AM2-AM6 audio exists only as clips. This script fetches the same
Open-Meteo archive data (same location, timezone, variables and columns as
inference_pipeline/scripts/weather_info.py) for the hours the clips cover, then
uses the pipeline's own weather_match() for each recorder and rebuilds the
all-clip features with weather (build_all_clip_features.py).

Per recorder, in <pipeline_out>/Audio_Moth_N/outputs/:
  weather.csv                 per-clip weather (pipeline format)
  features_all_weather.csv    all clips, 29-feature columns + AED columns
Shared: <pipeline_out>/weather_hourly.csv. The pipeline itself is unchanged.

Usage (PowerShell):
  python add_weather.py --pipeline_out "C:\\BirdNET ONNX\\pipeline_out"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "inference_pipeline" / "scripts"))
sys.path.insert(0, str(REPO / "inference_pipeline"))
from config import LATITUDE, LONGITUDE, TIMEZONE, WEATHER_CODE_LEGEND  # noqa: E402
from weather_info import get_time_of_day  # noqa: E402
from weather_matching import weather_match  # noqa: E402


def fetch_hourly(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Same request and columns as weather_info.get_weather_info, for [start, end] local hours."""
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,precipitation,windspeed_10m,relative_humidity_2m,weathercode",
        "timezone": TIMEZONE,
    }
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    w = pd.DataFrame({
        "Datetime": pd.to_datetime(h["time"]),
        "Temperature": h["temperature_2m"],
        "Windspeed": h["windspeed_10m"],
        "precipitation": h["precipitation"],
        "Humidity": h["relative_humidity_2m"],
        "weathercode": h["weathercode"],
    })
    w["weather_desc"] = w.weathercode.map(WEATHER_CODE_LEGEND)
    w = w[(w.Datetime >= start) & (w.Datetime <= end)].reset_index(drop=True)
    tf = w.Datetime.apply(lambda ts: get_time_of_day(LATITUDE, LONGITUDE, ts.strftime("%Y-%m-%d %H:%M:%S")))
    for k in ["timestamp_local", "sunrise", "sunset", "time_of_day"]:
        w[k] = tf.apply(lambda x: x[k])
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline_out", required=True, type=Path)
    args = ap.parse_args()
    recs = sorted(p for p in args.pipeline_out.glob("Audio_Moth_*") if (p / "outputs" / "metadata.csv").exists())

    # hours covered by any clip; clips round to the nearest hour in weather_match
    ts = pd.concat(pd.to_datetime(pd.read_csv(r / "outputs" / "metadata.csv", usecols=["Timestamp"]).Timestamp
                                  .astype(str), format="%Y%m%d_%H%M%S") for r in recs)
    start, end = ts.min().floor("h"), ts.max().ceil("h")
    hourly_csv = args.pipeline_out / "weather_hourly.csv"
    w = fetch_hourly(start, end)
    w.to_csv(hourly_csv, index=False)
    expected = int((end - start) / pd.Timedelta(hours=1)) + 1
    print(f"Fetched {len(w)} of {expected} hours ({start} to {end}); "
          f"missing values: {w[['Temperature', 'Windspeed', 'Humidity']].isna().sum().to_dict()}")

    for r in recs:
        out = r / "outputs"
        m = weather_match(weather_csv=str(hourly_csv), metadata_csv=str(out / "metadata.csv"),
                          output_csv=str(out / "weather.csv"))
        n_na = int(m[["Temperature", "Windspeed", "Humidity"]].isna().any(axis=1).sum())
        res = subprocess.run([sys.executable, str(Path(__file__).with_name("build_all_clip_features.py")),
                              "--pipeline_out", str(r), "--output_name", "features_all_weather.csv"],
                             capture_output=True, text=True)
        if res.returncode:
            sys.exit(f"{r.name}: build_all_clip_features failed:\n{res.stderr[-2000:]}")
        print(f"{r.name}: {len(m):,} clips matched, {n_na} without weather | {res.stdout.strip().splitlines()[-1]}")


if __name__ == "__main__":
    main()
