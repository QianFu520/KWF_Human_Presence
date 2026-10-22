"""
Tests for step 6 (weather) and the weather-optional feature engineering in step 7.

Uses small made-up CSVs (10 clips across 2 recorders). No network calls:
get_weather_info is replaced with a fake that returns fixed hourly rows, or
requests.get is replaced with a failing request.

Run with: python -m pytest inference_pipeline/tests -v
"""

import numpy as np
import pandas as pd
import pytest
import requests
import soundfile as sf

import weather_info
from weather_info import get_weather_info_all
from weather_matching import add_weather_data, weather_match
from feature_engineering import run_feature_engineering, SENTINEL_SPECIES

WEATHER_COLS = ["Temperature", "Windspeed", "Humidity", "Volume_Wind_Ratio"]
ACOUSTIC_COLS = [
    "RMS_Energy", "Spectral_Contrast", "Spectral_Flatness", "Spectral_Bandwidth",
    "Spectral_Rolloff_85", "Onset_Strength", "MFCC_8", "MFCC_9", "MFCC_12", "MFCC_13",
]

# 5 clips per recorder. Moth 1 clips (12:51) round to 13:00, Moth 2 clips (14:02) to 14:00.
CLIPS = (
    [("Audio_Moth_1", f"20250321_1251{s:02d}") for s in range(0, 15, 3)]
    + [("Audio_Moth_2", f"20250321_1402{s:02d}") for s in range(0, 15, 3)]
)


def fake_hourly(hours):
    """Hourly rows shaped like get_weather_info's output; values depend only on the hour."""
    dt = pd.to_datetime([f"2025-03-21 {h:02d}:00:00" for h in hours])
    return pd.DataFrame({
        "Datetime": dt,
        "Temperature": [20.0 + h for h in hours],
        "Windspeed": [1.0 + h / 10 for h in hours],
        "precipitation": 0.0,
        "Humidity": [80.0 - h for h in hours],
        "weathercode": 0,
    })


@pytest.fixture
def raw_audio(tmp_path):
    """Raw folder layout from audio_split_rename.py; the files are empty because get_weather_info is faked."""
    raw = tmp_path / "raw"
    (raw / "Audio_Moth_1_032125").mkdir(parents=True)
    (raw / "Audio_Moth_2_032125").mkdir(parents=True)
    (raw / "Audio_Moth_1_032125" / "20250321_120000.wav").touch()
    (raw / "Audio_Moth_2_032125" / "20250321_130000.WAV").touch()
    (raw / "Audio_Moth_2_032125" / "._20250321_130000.wav").touch()  # macOS resource fork
    return raw


@pytest.fixture
def fake_weather_api(monkeypatch):
    """Recordings overlap at 13:00 and 14:00, so the combined hourly rows contain duplicate hours."""
    blocks = {
        "20250321_120000": [12, 13, 14],
        "20250321_130000": [13, 14, 15],
    }
    seen = []

    def fake_get_weather_info(wav_path, lat, lon):
        seen.append(wav_path.name)
        return fake_hourly(blocks[wav_path.stem])

    monkeypatch.setattr(weather_info, "get_weather_info", fake_get_weather_info)
    return seen


@pytest.fixture
def pipeline_csvs(tmp_path):
    """Acoustic, BirdNET and metadata CSVs in the formats steps 3-5 write."""
    rng = np.random.default_rng(0)
    names = [f"{rec}_{ts}" for rec, ts in CLIPS]

    acoustic = pd.DataFrame(rng.random((len(names), len(ACOUSTIC_COLS))), columns=ACOUSTIC_COLS)
    acoustic.insert(0, "clip_name", names)  # step 4 writes the stem

    birdnet = pd.DataFrame({
        "clip_name": [f"{n}.wav" for n in names],
        "species": [str([SENTINEL_SPECIES[0]]) if i % 2 else "[]" for i in range(len(names))],
        "confidence": ["[0.8]" if i % 2 else "[]" for i in range(len(names))],
    })

    metadata = pd.DataFrame({
        "clip_name": [f"{n}.wav" for n in names],
        "Recorder": [rec for rec, _ in CLIPS],
        "Timestamp": [ts for _, ts in CLIPS],
    })

    paths = {
        "acoustic": tmp_path / "acoustic_features.csv",
        "birdnet": tmp_path / "birdnet_output.csv",
        "metadata": tmp_path / "metadata.csv",
    }
    acoustic.to_csv(paths["acoustic"], index=False)
    birdnet.to_csv(paths["birdnet"], index=False)
    metadata.to_csv(paths["metadata"], index=False)
    return paths


@pytest.fixture
def weather_csv(tmp_path, raw_audio, fake_weather_api, pipeline_csvs):
    """Run step 6 the way run_pipeline does: hourly CSV, then per-clip weather.csv."""
    hourly_csv = tmp_path / "weather_hourly.csv"
    out_csv = tmp_path / "weather.csv"
    get_weather_info_all(audio_folder=raw_audio, output_csv=hourly_csv)
    weather_match(weather_csv=hourly_csv, metadata_csv=pipeline_csvs["metadata"], output_csv=out_csv)
    return out_csv


def test_without_weather_has_no_weather_columns(tmp_path, pipeline_csvs):
    df = run_feature_engineering(
        acoustic_csv=pipeline_csvs["acoustic"],
        birdnet_csv=pipeline_csvs["birdnet"],
        metadata_csv=pipeline_csvs["metadata"],
        weather_csv=None,
        output_csv=tmp_path / "features.csv",
    )
    assert not set(WEATHER_COLS) & set(df.columns)
    assert len(df) == len(CLIPS)


def test_with_weather_has_all_columns_without_nans(tmp_path, pipeline_csvs, weather_csv):
    df = run_feature_engineering(
        acoustic_csv=pipeline_csvs["acoustic"],
        birdnet_csv=pipeline_csvs["birdnet"],
        metadata_csv=pipeline_csvs["metadata"],
        weather_csv=weather_csv,
        output_csv=tmp_path / "features.csv",
    )
    for col in WEATHER_COLS:
        assert col in df.columns
        assert df[col].notna().all(), f"{col} has NaNs"


def test_duplicate_hours_do_not_duplicate_clips(tmp_path, pipeline_csvs, weather_csv):
    hourly = pd.read_csv(tmp_path / "weather_hourly.csv")
    assert hourly["Datetime"].is_unique
    assert list(hourly["Datetime"]) == sorted(hourly["Datetime"])

    df = run_feature_engineering(
        acoustic_csv=pipeline_csvs["acoustic"],
        birdnet_csv=pipeline_csvs["birdnet"],
        metadata_csv=pipeline_csvs["metadata"],
        weather_csv=weather_csv,
        output_csv=tmp_path / "features.csv",
    )
    assert len(df) == len(pd.read_csv(pipeline_csvs["acoustic"]))


def test_wav_matching_is_case_insensitive_and_skips_hidden(tmp_path, raw_audio, fake_weather_api):
    get_weather_info_all(audio_folder=raw_audio, output_csv=tmp_path / "weather_hourly.csv")
    assert sorted(fake_weather_api) == ["20250321_120000.wav", "20250321_130000.WAV"]


def test_no_weather_collected_raises(tmp_path, monkeypatch, raw_audio):
    monkeypatch.setattr(weather_info, "get_weather_info", lambda *args: None)
    with pytest.raises(RuntimeError):
        get_weather_info_all(audio_folder=raw_audio, output_csv=tmp_path / "weather_hourly.csv")


def test_1251_clip_matches_1300_weather():
    metadata = pd.DataFrame({"clip_name": ["Audio_Moth_1_20250321_125100.wav"], "Timestamp": ["20250321_125100"]})
    # Datetime as strings, the way it comes back from weather_hourly.csv
    weather = pd.DataFrame({
        "Datetime": ["2025-03-21 12:00:00", "2025-03-21 13:00:00"],
        "Temperature": [21.0, 25.0],
        "Windspeed": [1.0, 3.0],
        "Humidity": [90.0, 70.0],
    })
    out = add_weather_data(metadata, weather)
    assert len(out) == 1
    assert out.loc[0, "Datetime"] == pd.Timestamp("2025-03-21 13:00:00")
    assert out.loc[0, "Temperature"] == 25.0


def test_unmatched_clips_are_counted(capsys):
    metadata = pd.DataFrame({
        "clip_name": ["a.wav", "b.wav"],
        "Timestamp": ["20250321_125100", "20250321_200000"],
    })
    weather = pd.DataFrame({"Datetime": ["2025-03-21 13:00:00"], "Temperature": [25.0]})
    out = add_weather_data(metadata, weather)
    assert len(out) == 2
    assert "1 of 2 clips got no weather match" in capsys.readouterr().out


class _ErrorResponse:
    """What Open-Meteo sends back for a bad request: HTTP 400 and an error body, no "hourly"."""
    status_code = 400

    def json(self):
        return {"error": True, "reason": "Parameter 'start_date' is out of allowed range"}


def _raise_connection_error(*args, **kwargs):
    raise requests.ConnectionError("Failed to resolve archive-api.open-meteo.com")


@pytest.mark.parametrize(
    "fake_get",
    [_raise_connection_error, lambda *args, **kwargs: _ErrorResponse()],
    ids=["request_raises", "error_response"],
)
def test_failed_open_meteo_request_raises(tmp_path, monkeypatch, fake_get):
    # real (1-second) recordings, so the real get_weather_info runs up to the API call
    raw = tmp_path / "raw"
    for folder, name in [("Audio_Moth_1_032125", "20250321_120000.wav"),
                         ("Audio_Moth_2_032125", "20250321_130000.wav")]:
        (raw / folder).mkdir(parents=True)
        sf.write(raw / folder / name, np.zeros(8000), 8000)

    monkeypatch.setattr(weather_info.requests, "get", fake_get)
    hourly_csv = tmp_path / "weather_hourly.csv"
    with pytest.raises(RuntimeError) as excinfo:
        get_weather_info_all(audio_folder=raw, output_csv=hourly_csv)

    message = str(excinfo.value)
    assert "Audio_Moth_1_032125/20250321_120000.wav" in message
    assert "Audio_Moth_2_032125/20250321_130000.wav" in message
    assert not hourly_csv.exists()


def test_one_failed_recording_is_not_skipped(tmp_path, monkeypatch, raw_audio):
    # Moth 1 gets weather, Moth 2's request fails: stop instead of writing partial weather
    def fake_get_weather_info(wav_path, lat, lon):
        return fake_hourly([12, 13, 14]) if wav_path.stem == "20250321_120000" else None

    monkeypatch.setattr(weather_info, "get_weather_info", fake_get_weather_info)
    with pytest.raises(RuntimeError, match="1 recording") as excinfo:
        get_weather_info_all(audio_folder=raw_audio, output_csv=tmp_path / "weather_hourly.csv")
    assert "Audio_Moth_2_032125/20250321_130000.WAV" in str(excinfo.value)
    assert "Audio_Moth_1_032125" not in str(excinfo.value)
