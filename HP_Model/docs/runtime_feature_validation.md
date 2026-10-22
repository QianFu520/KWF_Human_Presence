# Runtime feature validation

- Generated: 2026-10-22 16:42, repo commit `c90fc80`
- Raw audio: `C:\Users\patri\kwf_val\raw`
- Reference: `C:\Users\patri\Downloads\test_acoustic_features.csv` (2,183 clips)
- Pipeline outputs: `C:\Users\patri\kwf_val\report\outputs`

## 1. Clip lengths

28,798 clips; expected 144000 samples at 48000 Hz.

All clips are exactly the expected length and sample rate.

## 2. Match with the reference

Reference already uses the pipeline column names; no renaming.

- Pipeline clips: 28,798
- Reference clips: 2,183
- **Matched on clip_name: 178**
- Reference clips not produced by the pipeline: 2,005 (e.g. Audio_Moth_1_20250318_171427, Audio_Moth_1_20250318_171436, Audio_Moth_1_20250318_171451, Audio_Moth_1_20250318_171533, Audio_Moth_1_20250318_171539)
- Of those, inside the recorded audio (a pipeline clip from the same recorder starts < 3 s away): 0

## 3. Acoustic features

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

## 4. Recorder coverage

Clock hours (local time) with at least one clip; 05–08 means 05:00–08:59.

| Recorder | Clips | Matched to reference | First clip | Last clip | Hours covered (local) |
|---|---|---|---|---|---|
| Audio_Moth_4 | 28798 | 178 | 2025-03-20 06:25:43 | 2025-03-21 06:25:39 | 2025-03-20: 06–23<br>2025-03-21: 00–06 |
