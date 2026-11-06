# Pipeline inputs and outputs

This page says what goes into each step of the inference pipeline and what comes out.

Checked on: one 87-minute recording from Audio Moth 5 (1,747 clips), using `run_pipeline.py` on `main` (commit `4104a17`). Items marked "not tested" were read from the code, not run.

## Overview

| Step | What it does | Output file |
|---|---|---|
| 1. Split | Cuts long recordings into 3-second clips | clips folder |
| 2. Tiny CNN | Labels each clip "meaningful" or "not meaningful" | tinycnn_predictions.csv |
| 3. BirdNET | Finds bird species in each clip | birdnet_output.csv |
| 4. Acoustic features | Measures the sound of each clip (10 numbers) | acoustic_features.csv |
| 5. Metadata | Records the recorder, time and file facts for each clip | metadata.csv |
| 6. Weather | Looks up the weather for each clip's time (needs internet) | weather_hourly.csv, weather.csv |
| 7. Features | Combines everything into the table for the model | features.csv |
| 8. HP model | Predicts human presence | not available yet |

## Step 1: Split

- **In:** a folder of recordings. Each recorder has its own subfolder named like `Audio_Moth_5_031925`, holding long files named like `20250319_163237.WAV` (date, then time).
- **Out:** one flat folder of clips named like `Audio_Moth_5_20250319_163237.wav`. Every clip is 3.0 seconds, 48 kHz, one channel.
- **Good to know:**
  - The recorder name comes from the subfolder name (the last part is dropped), so the folder name matters.
  - The end of a recording that is shorter than 3 seconds is dropped. For our file, that was the last 2 seconds.
  - Clip times count up in exact 3-second steps from the recording's start time.
  - Recordings at other sample rates are converted to 48 kHz (not tested).

## Step 2: Tiny CNN

- **In:** the clips folder (it also searches subfolders) and the model file `tinycnn_v3.pth`.
- **Out:** `tinycnn_predictions.csv`, one row per clip:
  - `clip_name` (with `.wav`)
  - `audio_path`
  - `not_meaningful_prob` (a number from 0 to 1)
  - `predicted_label` (`meaningful` or `not_meaningful`; a clip is `not_meaningful` if the number is 0.5 or higher)
  - `threshold` and `model_version`
- **Good to know:** rows are not in time order.

## Step 3: BirdNET

- **In:** the clips folder (it also searches subfolders), the model file (`birdnet_fp16_pruned493.onnx`), the species names file and the regional species list (218 species).
- **Out:** `birdnet_output.csv`, one row per clip:
  - `clip_name` (with `.wav`)
  - `species`: the species found, written as a list
  - `confidence`: one number per species, also a list
- **Good to know:**
  - Only species with confidence of 0.25 or higher, and on the regional list, are kept.
  - The lists are saved as text, so the next step has to turn them back into lists.
  - An empty list can mean no birds, only birds outside the regional list, or a clip that failed. The file can't tell these apart.

## Step 4: Acoustic features

- **In:** the clips folder (it also searches subfolders).
- **Out:** `acoustic_features.csv`, one row per clip:
  - `clip_name` (without `.wav`)
  - 10 numbers: `RMS_Energy`, `Spectral_Contrast`, `Spectral_Flatness`, `Spectral_Bandwidth`, `Spectral_Rolloff_85`, `Onset_Strength`, `MFCC_8`, `MFCC_9`, `MFCC_12`, `MFCC_13`
- **Good to know:** a clip that fails is left out of the file, and the run summary reports how many failed.

## Step 5: Metadata

- **In:** the clips folder. It only looks at files directly in this folder, not subfolders, and only names ending in lowercase `.wav`.
- **Out:** `metadata.csv`, one row per clip, with:
  - `clip_name` (with `.wav`), `Recorder`, `Timestamp` (text like `20250319_163237`, local time)
  - `Duration (sec)`, `Sample Rate (Hz)`, `Channels`, `Bit Depth`, `File Size (KB)`
  - `Battery Voltage`, `Gain`
  - `Spectral Centroid`, `Zero Crossing Rate`
- **Good to know:**
  - `Battery Voltage` and `Gain` always say `Unknown`, because Step 1 doesn't copy that information from the original recording.
  - If the folder has subfolders or uppercase `.WAV` names, the step skips those files without an error, and can produce an empty file.
  - Rows are not in time order.

## Step 6: Weather

- **In:** the folder of raw recordings (to find the recording times), the Step 5 metadata file, and the location and timezone set in `config.py`. It needs internet.
- **Out:**
  - `weather_hourly.csv`: one row per hour of the recording
  - `weather.csv`: one row per clip, with `Datetime`, `Temperature`, `Windspeed`, `precipitation`, `Humidity`, `weathercode`, `weather_desc`, `timestamp_local`, `sunrise`, `sunset`, `time_of_day` and `clip_name`
- **Good to know:**
  - Every clip gets the weather of the nearest hour. For example, a clip at 16:45 gets the 17:00 weather.
  - One fixed location is used for all recorders.
  - `--no_weather` skips this step.
  - The team says uppercase and lowercase `.wav` raw files both work now. Not re-tested.

## Step 7: Features

- **In:** the Step 4, 3 and 5 files, the Step 6 file (optional), and the list of clips Step 2 called meaningful.
- **Out:** `features.csv`, one row per meaningful clip, 34 columns:
  - The 29 model features:
    - the 10 Step 4 features
    - `confidence` (the highest BirdNET confidence in the clip)
    - 10 yes/no columns for the sentinel species
    - `Eerie_Silence` (1 if none of the 10 sentinel species was heard)
    - `hour_sin` and `hour_cos` (the hour of day)
    - `Volume_Spike_15s` (a sudden jump in loudness compared with the previous 15 seconds)
    - `Volume_Wind_Ratio`, `Temperature`, `Windspeed`, `Humidity` (weather)
  - 5 extra columns for reference: `clip_name`, `species`, `Recorder`, `Timestamp`, `Datetime`
- **Good to know:**
  - With `--no_weather`, the four weather columns are left out, leaving 25 features (30 columns in the file).
  - Step 7 removes `.wav` from clip names before joining, so the name differences between steps don't matter here.
  - The sentinel species: Buff-rumped Warbler, Black-cheeked Ant-Tanager, Black-hooded Antshrike, Great Tinamou, Short-billed Pigeon, Yellow-throated Toucan, Blue-black Grosbeak, Rufous Piha, Band-tailed Barbthroat and Scarlet Macaw.

## Step 8: HP model

Not available yet. The step only prints a message and skips.

## Clip names

Agreed format (approved by Qian): `Audio_Moth_N_YYYYMMDD_HHMMSS.wav`, for example `Audio_Moth_5_20250319_163237.wav`.

- The date and time are the clip's start time, in local Costa Rica time (UTC-6).
- Every step should use this name, with `.wav`, and treat `.wav` and `.WAV` the same.

Current state: Steps 2, 3, 5 and 6 already use the name with `.wav`. Step 4 writes it without `.wav` and is still to be changed. Step 7 removes `.wav` from every table before joining, so nothing breaks in the meantime.
