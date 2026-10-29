# Runtime feature validation

I checked that the runtime pipeline (`audio_split_rename.py` and `extract_acoustic_features.py`, called the same way as `run_pipeline.py` steps 1 and 4) gives the same acoustic features as the reference test_acoustic_features.csv, using one day of audio from each of the six recorders.

**Conclusion:** on all six recorders, every one of the 944 reference clips expected for the days tested was produced and matched the reference on all 10 acoustic features, with the largest relative difference 1.73e-04 (MFCC_13), well under the 1e-3 tolerance.

## All six recorders

| Recorder | Date | Clips checked | Reference clips matched / expected | Largest difference (feature) | Result |
|---|---|---|---|---|---|
| Audio_Moth_1 | 2025-03-19 | 28,798 | 159 / 159 | 1.43e-05 (Spectral_Contrast) | pass |
| Audio_Moth_2 | 2025-03-19 | 28,798 | 166 / 166 | 6.01e-05 (MFCC_13) | pass |
| Audio_Moth_3 | 2025-03-19 | 26,896 | 101 / 101 | 8.03e-06 (Spectral_Contrast) | pass |
| Audio_Moth_4 | 2025-03-20 | 28,798 | 178 / 178 | 1.73e-04 (MFCC_13) | pass |
| Audio_Moth_5 | 2025-03-20 | 28,798 | 169 / 169 | 2.48e-05 (Spectral_Contrast) | pass |
| Audio_Moth_6 | 2025-03-20 | 28,798 | 171 / 171 | 2.59e-05 (Spectral_Contrast) | pass |

- Matched 944 of 2,183 reference clips in test_acoustic_features.csv across 6 recorders; 944 were expected for the recorders and dates tested.
- 6 of 6 runs pass; 170,886 clips checked, 0 with the wrong length or sample rate.
- Largest difference overall: 1.73e-04 (MFCC_13, Audio_Moth_4).

A run passes when every clip is 144,000 samples at 48 kHz, every reference clip for that recorder and date is among the pipeline's clips, and every feature's largest relative difference (|pipeline − reference| / |reference|) is under 1e-3. "Clips checked" is the number of 3-second clips cut from the day's two recordings; Audio_Moth_3 has fewer because its morning recording starts at 08:00:49 instead of about 06:25.

Runs: Audio_Moth_4 on 2026-10-22 with commit `c90fc80` (details below; its summary.json was rebuilt from that run's outputs with `2854adc`), Audio_Moth_6 on 2026-10-24, Audio_Moth_5, Audio_Moth_2 and Audio_Moth_1 on 2026-10-28, and Audio_Moth_3 on 2026-10-29, all with commit `2854adc`. The code in `inference_pipeline/scripts` is the same in both commits.

## Limitations

- **6 of the 19 recorder-days in the reference.** The reference has 2,183 clips from 19 recorder-days (17–21 March 2025). These runs cover 944 of them; the other 1,239 come from 13 recorder-days that were not run.
- **Daytime clips only.** The reference clips for the days tested start between 07:26 and 17:57, and for the 20 March recorders only between 15:56 and 17:57. The other 169,942 clips, including all night-time audio, were checked for length and sample rate, but their feature values were not compared.
- **Acoustic features only.** The reference has just the 10 acoustic features, so metadata, weather, BirdNET and sentinel values were not checked.
- **One environment.** All runs used Windows 11, Python 3.13.14, librosa 1.0.0, numpy 2.3.5, scipy 1.17.0 and soundfile 0.14.0. Other versions were not tested.

## Reproducing

Put one recorder/day folder from the "Audio Moth Data" Drive folder (for example `Audio_Moth_3_031925`, both recordings) in `<raw_audio>` and run, from the repo root:

```
python inference_pipeline/validation/validate_features.py --raw_audio <raw_audio> --reference <reference_csv> --output <output_amN>
```

Each run writes validation_report.md and summary.json to `<output_amN>`, plus the clips and CSVs (about the size of the raw audio; delete them after checking). Then combine the runs:

```
python inference_pipeline/validation/combine_reports.py <output_am1> <output_am2> <output_am3> <output_am4> <output_am5> <output_am6> --output <combined.md>
```

The per-recorder reports are not committed; the Audio_Moth_4 one is below as an example.

## Audio_Moth_4 in detail (20 March 2025)

- Generated: 2026-10-22 16:42, repo commit `c90fc80`
- Raw audio: `<raw_audio>` (Audio_Moth_4_032025, both recordings from the "Audio Moth Data" Drive folder)
- Reference: `<reference_csv>` (test_acoustic_features.csv from the shared Drive folder, 2,183 clips)
- Pipeline outputs: `<output>/outputs`

### 1. Clip lengths

28,798 clips; expected 144000 samples at 48000 Hz.

All clips are exactly the expected length and sample rate.

### 2. Match with the reference

Reference already uses the pipeline column names; no renaming.

- Pipeline clips: 28,798
- Reference clips: 2,183
- **Matched on clip_name: 178**
- Reference clips not produced by the pipeline: 2,005 (e.g. Audio_Moth_1_20250318_171427, Audio_Moth_1_20250318_171436, Audio_Moth_1_20250318_171451, Audio_Moth_1_20250318_171533, Audio_Moth_1_20250318_171539)
- Of those, inside the recorded audio (a pipeline clip from the same recorder starts < 3 s away): 0

### 3. Acoustic features

Relative difference = |pipeline − reference| / |reference|; under 0.001 counts as a match.

| Feature | Median rel. diff | Max rel. diff | Clips < 1e-3 | NaN | Worst clip | Result |
|---|---|---|---|---|---|---|
| RMS_Energy | 0.00e+00 | 1.17e-07 | 100.0% | 0 | Audio_Moth_4_20250320_173216 | match |
| Spectral_Contrast | 4.78e-06 | 3.16e-05 | 100.0% | 0 | Audio_Moth_4_20250320_173058 | match |
| Spectral_Flatness | 5.88e-07 | 2.79e-06 | 100.0% | 0 | Audio_Moth_4_20250320_173149 | match |
| Spectral_Bandwidth | 4.88e-10 | 2.16e-09 | 100.0% | 0 | Audio_Moth_4_20250320_171907 | match |
| Spectral_Rolloff_85 | 0.00e+00 | 1.06e-05 | 100.0% | 0 | Audio_Moth_4_20250320_163425 | match |
| Onset_Strength | 0.00e+00 | 1.38e-07 | 100.0% | 0 | Audio_Moth_4_20250320_163516 | match |
| MFCC_8 | 0.00e+00 | 2.26e-07 | 100.0% | 0 | Audio_Moth_4_20250320_164549 | match |
| MFCC_9 | 0.00e+00 | 1.64e-07 | 100.0% | 0 | Audio_Moth_4_20250320_163310 | match |
| MFCC_12 | 0.00e+00 | 2.32e-07 | 100.0% | 0 | Audio_Moth_4_20250320_163113 | match |
| MFCC_13 | 1.88e-07 | 1.73e-04 | 100.0% | 0 | Audio_Moth_4_20250320_164322 | match |

Reference Spectral_Rolloff_85 max is 9,812 Hz (≤ 11,025 Hz), consistent with 22050 Hz.

### 4. Recorder coverage

Clock hours (local time) with at least one clip; 05–08 means 05:00–08:59.

| Recorder | Clips | Matched to reference | First clip | Last clip | Hours covered (local) |
|---|---|---|---|---|---|
| Audio_Moth_4 | 28798 | 178 | 2025-03-20 06:25:43 | 2025-03-21 06:25:39 | 2025-03-20: 06–23<br>2025-03-21: 00–06 |
