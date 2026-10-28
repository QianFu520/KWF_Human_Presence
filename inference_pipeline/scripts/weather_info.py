# %%
import requests
import soundfile as sf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from astral import LocationInfo
from astral.sun import sun

from config import LATITUDE, LONGITUDE, WEATHER_DATA_PATH, AUDIO_MOTH_FOLDER, WEATHER_CODE_LEGEND, TIMEZONE

"""the purpose of this script is to extract hourly weather info for each wav file in each recorder/day folder
     and save one combined hourly weather CSV (one row per hour). """


## the below function is used to add time of day classification
## timestamp_str is local time (TIMEZONE), matching the AudioMoth filenames
def get_time_of_day(lat, lon, timestamp_str):
    local_tz = ZoneInfo(TIMEZONE)
    local_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=local_tz)

    location = LocationInfo(latitude=lat, longitude=lon)
    s = sun(location.observer, date=local_dt.date(), tzinfo=local_tz)

    hour = local_dt.hour
    if 4 <= hour < 6:
        label = "dawn"
    elif 6 <= hour < 12:
        label = "morning"
    elif 12 <= hour < 16:
        label = "afternoon"
    elif 16 <= hour < 19:
        label = "evening"
    else:
        label = "night"

    return {
        "timestamp_local": local_dt,
        "sunrise": s["sunrise"],
        "sunset": s["sunset"],
        "time_of_day": label
    }


# the below function retrieve weather info for one wav file

def get_weather_info(wav_path, lat, lon):
    base_name = wav_path.stem  #  20250317_180003
    


    try:
        start_time = datetime.strptime(base_name, "%Y%m%d_%H%M%S")
        info = sf.info(str(wav_path))
        duration_hours = int(info.frames / info.samplerate // 3600) + 1
        end_time = start_time + timedelta(hours=duration_hours)

        start_date = start_time.strftime("%Y-%m-%d")
        end_date = end_time.strftime("%Y-%m-%d")

        # making open meteo request
        # filenames are local time, so request hourly data in local time too;
        # the returned hours then line up with the filename timestamps

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,windspeed_10m,relative_humidity_2m,weathercode",
            "timezone": TIMEZONE
        }

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        #convert API data to a df
        # column names match what feature_engineering.py reads
        weather_df = pd.DataFrame({
            "Datetime": data["hourly"]["time"],
            "Temperature": data["hourly"]["temperature_2m"],
            "Windspeed": data["hourly"]["windspeed_10m"],
            "precipitation": data["hourly"]["precipitation"],
            "Humidity": data["hourly"]["relative_humidity_2m"],
            "weathercode": data["hourly"]["weathercode"]
        })


        weather_df["Datetime"] = pd.to_datetime(weather_df["Datetime"])

        weather_df["weather_desc"] = weather_df["weathercode"].map(WEATHER_CODE_LEGEND)

        # filter only the hours that fall within the .wav file's duration
        # for example, if the wav file spans only 6 hours, then it keeps only 4 rows, not all 24 hours

        # one extra hour: clips near the end round up to the next hour in weather_matching
        # (e.g. 23:40 -> 00:00). It is never later than end_time, so end_date already covers it.
        hourly_timestamps = [start_time + timedelta(hours=i) for i in range(duration_hours + 1)]
        hour_set = set([ts.replace(minute=0, second=0, microsecond=0) for ts in hourly_timestamps])
        weather_df = weather_df[weather_df["Datetime"].isin(hour_set)]

        ## add time of day features
        time_features = weather_df["Datetime"].apply(
            lambda ts: get_time_of_day(lat, lon, ts.strftime("%Y-%m-%d %H:%M:%S"))
        )

        weather_df["timestamp_local"] = time_features.apply(lambda x: x["timestamp_local"])
        weather_df["sunrise"] = time_features.apply(lambda x: x["sunrise"])
        weather_df["sunset"] = time_features.apply(lambda x: x["sunset"])
        weather_df["time_of_day"] = time_features.apply(lambda x: x["time_of_day"])

        return weather_df

    except Exception as e:
        print(f"[ERROR] Failed to process {wav_path.name}: {e}")
        return None

## the below function loop through all the wav file in all 5 days folder 
# call above function on each of the wav file
def get_weather_info_all(audio_folder=AUDIO_MOTH_FOLDER, output_csv=WEATHER_DATA_PATH):
    root_folder = Path(audio_folder)
    all_weather_dfs = []
    failed = []

    for day_folder in sorted(root_folder.iterdir()):
        if not day_folder.is_dir():
            continue

        # recordings end in lowercase .wav; match any case and skip hidden files (e.g. macOS ._ files)
        wav_files = sorted(
            p for p in day_folder.iterdir()
            if p.is_file() and p.suffix.lower() == ".wav" and not p.name.startswith(".")
        )
        for wav_file in wav_files:
            print(f"Processing: {wav_file.name}")
            weather_df = get_weather_info(wav_file, LATITUDE, LONGITUDE)
            if weather_df is None:
                failed.append(f"{day_folder.name}/{wav_file.name}")
            else:
                all_weather_dfs.append(weather_df)

    # a recording without weather would leave its clips with blank weather; stop instead of skipping it
    if failed:
        raise RuntimeError(
            f"Could not get weather for {len(failed)} recording(s) (see [ERROR] lines above). "
            "Stopped so no clip gets blank weather; use --no_weather to run without weather. "
            "Failed recordings:\n  " + "\n  ".join(failed)
        )
    if not all_weather_dfs:
        raise RuntimeError(f"No weather data collected from {root_folder}")

    # recorders running at the same time return the same hours; keep one row per hour
    combined_df = pd.concat(all_weather_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="Datetime").sort_values("Datetime").reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"[OK] Combined weather data saved to {output_csv} ({len(combined_df)} hours)")
    return combined_df


if __name__ == "__main__":
    get_weather_info_all()


# %%



