# Feature Pipeline — Progress Summary

**Author:** Qian Fu  
**Date:** July 20, 2026  
**Project:** KWF Human Presence Model

---

## Step 1 — Download both Original and aug clips 

Jacky ran audio-space augmentation (pitch shift, time stretch, Gaussian noise) on all positive clips. Each original clip produced ~40 augmented variants.

File naming convention:
- Original: `Audio_Moth_1_20250318_171418.wav`
- Augmented: `Audio_Moth_1_20250318_171418_aug_0.wav`

| Device | Original Clips | Augmented Clips | Total |
|---|---:|---:|---:|
| Audio_Moth_1 | 901 | 35,000 | 35,901 |
| Audio_Moth_2 | 1,201 | 50,000 | 51,201 |
| Audio_Moth_3 | 500 | 20,000 | 20,500 |
| Audio_Moth_4 | 2,376 | 95,000 | 97,376 |
| Audio_Moth_5 | 2,822 | 115,000 | 117,822 |
| Audio_Moth_6 | 3,071 | 130,711 | 133,782 |
| **Total** | **10,871** | **445,711** | **456,582** |

---

## Step 2 — Train / Test Split

Because augmentation was performed before splitting, a standard random split would cause data leakage — the same original clip could appear in both train and test sets under different augmented names. To prevent this, a **group-based split** was used: each original clip and all its augmented variants are always assigned to the same partition. The **test set contains only original clips** — no augmented clips ever enter the test set.

**Parameters:**
- Method: Group-based stratified split
- Ratio: 80 / 20
- Random seed: 42
- Group key: root clip name (strip `_aug_N` suffix)
- Test set: original clips only

| Device | Train — Orig | Train — Aug | Train Total | Test — Orig Only |
|---|---:|---:|---:|---:|
| Audio_Moth_1 | 720 | 28,194 | 28,914 | 181 |
| Audio_Moth_2 | 958 | 40,071 | 41,029 | 243 |
| Audio_Moth_3 | 399 | 16,146 | 16,545 | 101 |
| Audio_Moth_4 | 1,898 | 75,894 | 77,792 | 478 |
| Audio_Moth_5 | 2,257 | 91,876 | 94,133 | 565 |
| Audio_Moth_6 | 2,456 | 104,553 | 107,009 | 615 |
| **Total** | **8,688** | **356,734** | **365,422** | **2,183** |

Files are linked using OS hardlinks (not copied) — no extra disk space consumed.  
Script: `feature_pipeline/scripts/train_test_split.py`

---

## Step 3 — Acoustic Feature Extraction

Ten acoustic features are extracted from every clip using **librosa 0.11.0**. Each feature is the mean across all frames of the clip, producing one scalar per feature per clip. Extraction runs in parallel across 8 CPU workers.

**Parameters:**

| Parameter | Value |
|---|---|
| Sample rate | 22,050 Hz |
| FFT window (n_fft) | 2,048 |
| Hop length | 512 |
| MFCC coefficients | 13 |
| Workers | 8 (parallel) |

**Features extracted:**

| Column | Description |
|---|---|
| `RMS_Energy` | Root mean square energy — overall loudness |
| `Spectral_Contrast` | Peak vs. valley difference across frequency bands |
| `Spectral_Flatness` | How noise-like vs. tonal the signal is |
| `Spectral_Bandwidth` | Weighted spread of frequencies around the centroid |
| `Spectral_Rolloff_85` | Frequency below which 85% of energy falls |
| `Onset_Strength` | Rate and intensity of new sound events |
| `MFCC_8` | Mel-frequency cepstral coefficient 8 |
| `MFCC_9` | Mel-frequency cepstral coefficient 9 |
| `MFCC_12` | Mel-frequency cepstral coefficient 12 |
| `MFCC_13` | Mel-frequency cepstral coefficient 13 |

**Output files:**

| File | Rows | Columns |
|---|---|---|
| `feature_pipeline/features/train_acoustic_features.csv` | 365,422 | 11 (clip_name + 10 features) |
| `feature_pipeline/features/test_acoustic_features.csv` | 2,183 | 11 (clip_name + 10 features) |

Join key: `clip_name` (file stem without `.wav` extension). For augmented clips, strip `_aug_N` to look up the parent clip's non-acoustic features from the original dataset CSV.  
Script: `feature_pipeline/scripts/extract_acoustic_features.py`

---

## Decision — HP Model Feature Set and Threshold (October 8, 2026)

### Human Activity Score dropped

`Human Activity Score` is YAMNet's human-sound confidence. YAMNet was removed from the project for low recall: it flagged 735 clips out of 631,321, against 10,000+ clips with real human activity in the simulation table (see `AED_Model/docs/DIAGNOSIS_LABELS.md`). The inference pipeline never computed it, so the old 30-feature MLP cannot run in deployment.

The HP model will be **retrained on 29 features**, all produced by `inference_pipeline/` so training and runtime values match:

| Group | Count | Features |
|---|---:|---|
| Acoustic | 10 | `RMS_Energy`, `Spectral_Contrast`, `Spectral_Flatness`, `Spectral_Bandwidth`, `Spectral_Rolloff_85`, `Onset_Strength`, `MFCC_8`, `MFCC_9`, `MFCC_12`, `MFCC_13` |
| Sentinel species | 10 | Binary BirdNET flags (see `SENTINEL_SPECIES` in `feature_engineering.py`) |
| BirdNET | 1 | `confidence` (max confidence in clip) |
| Engineered | 5 | `hour_sin`, `hour_cos`, `Eerie_Silence`, `Volume_Wind_Ratio`, `Volume_Spike_15s` |
| Weather | 3 | `Temperature`, `Humidity`, `Windspeed` |

### Threshold — no official value until retraining

Earlier values (0.30 for the 30-feature MLP, 0.55–0.85 for XGBoost runs, ~0.38 in `run_pipeline.py`) are **retired**. They were tuned on the test set, so they and their F1 scores are optimistic, and they belong to models that will not be deployed.

### Rules for retraining

1. **Split by time block or by recorder**, not randomly by clip. Neighbouring 3-second clips from the same event must not land on both sides, and `Volume_Spike_15s` already uses neighbouring clips.
2. **Hold out three sets:** train / validation / test. Choose the threshold on validation only; report final metrics once on test.
3. **Pick the threshold from operating needs**, e.g. a minimum recall on human activity while keeping false alerts per recorder per day manageable — not just best F1.
4. **Save the model and scaler** as `hp_model.pkl` and `hp_scaler.pkl`, together with the feature order and the chosen threshold.

### Weather features — undecided, train both versions

Whether to keep weather depends on how the system will be deployed: Open-Meteo needs an internet connection at inference time. Until that is decided, **train two models** on the same split and compare them on the same validation and test sets:

| Version | Features | Difference |
|---|---:|---|
| With weather | 29 | Full feature set above |
| Without weather | 25 | Drops `Temperature`, `Humidity`, `Windspeed`, and `Volume_Wind_Ratio` (computed from `Windspeed`) |

Report ROC-AUC, plus precision and recall at each model's chosen threshold, side by side. If the 25-feature model is close, dropping weather simplifies deployment.

Before building weather features for the 29-feature version, fix two issues in `inference_pipeline/scripts/weather_info.py`; otherwise the comparison understates weather:
- **Timezone:** AudioMoth filenames are local Costa Rica time (UTC−6). BirdNET detections peak at 05:00–06:00 filename time (dawn chorus), not 11:00–12:00. The script currently treats filenames as UTC, so weather is matched 6 hours off.
- **Location:** `LATITUDE`/`LONGITUDE` in `config.py` (9.7489, −83.7534) are about 110 km from the recorders. The simulation GPS points in `simulations.csv` put them at about 8.66, −83.65.

### Volume_Spike_15s — new definition

The old notebook and the pipeline computed this differently, and both counted rows rather than time, so after AED filtering or at a gap between recordings the "previous clips" could be minutes old. Retraining uses the pipeline definition in `inference_pipeline/scripts/feature_engineering.py`:

- **Window:** mean `RMS_Energy` of the same recorder's clips in the **previous 15 seconds by timestamp**, current clip excluded (normally the 5 preceding clips).
- **Value:** `RMS_Energy − window mean`, floored at 0. No clips in the window (start of a recording) → 0.
- **Order:** compute on **all clips, before AED filtering**, then keep the meaningful clips. Acoustic features must therefore be extracted for every clip.
