"""
Configuration for the inference pipeline scripts.

Adapted from the original KWF pipeline config (KashmirWorld/Bioacoustics,
pipeline_scripts/config.py), keeping only the values these scripts import.

run_pipeline.py passes its own paths for steps 1 and 5 (--raw_audio / --output).
The paths below are only used when running a script on its own, and by step 6
(weather), which still reads them from here.
"""

## --------------audio_split_rename--------------
RAW_AUDIO_INPUT_FOLDER = r"path/to/raw"
CLIPPED_AUDIO_OUTPUT_FOLDER = r"path/to/clipped"
TARGET_SAMPLE_RATE = 48000
SAMPLES_PER_CLIP = 144000  # TARGET_SAMPLE_RATE * 3, exactly 3.0s per clip

## --------------metadata_extraction--------------
METADATA_OUTPUT_CSV = r"path/to/metadata.csv"

## ------------weather_info / weather_matching----------------
# Folder of raw AudioMoth recordings, one sub-folder per day (read by weather_info.py)
AUDIO_MOTH_FOLDER = r"path/to/raw"

WEATHER_DATA_PATH = r"path/to/weather_data.csv"     # hourly weather from Open-Meteo
WEATHER_OUTPUT_CSV = r"path/to/weather_matches.csv" # weather matched to each clip

## location
# Mean of the simulation GPS points in simulations.csv; all six recorders are
# within ~500 m of this point.
LATITUDE = 8.6602
LONGITUDE = -83.6502

## timezone
# AudioMoth filenames are in this local time (UTC-6), not UTC.
TIMEZONE = "America/Costa_Rica"

# Weather code legend (Open-Meteo API codes)
WEATHER_CODE_LEGEND = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}
