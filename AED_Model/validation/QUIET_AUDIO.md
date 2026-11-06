# Why AED scores quiet audio as background

Status: 2026-11-05. Includes Coleman's AM3 results and Ben's 12 AM1 setup
clips (measured). Still open: listening to the dropped off-trail clips and
which AM1 clips hold people (see [Open](#open)).

## Finding

AED does not react to loudness: its input is a mel spectrogram in dB relative
to each clip's own maximum, so turning a clip down changes nothing. What it
reacts to is **how far the sound stands above the clip's noise floor**. When a
real sound is faint relative to the background, the normalised spectrogram is
dominated by broadband noise; it looks flat, and flat is what AED learned to
call background. Quiet recorders and distant activity therefore get rejected.

The AM1 setup clips show a second route to the same result: loud, close,
broadband sounds (footsteps, handling) also give a flat spectrum and are
rejected, even though they stand far above the noise. The common factor is
**spectral flatness**, not level: AED treats broadband, noise-like spectra as
background, whether they come from a quiet recording or a nearby footstep.

Likely origin: v3's 11,213 `not_meaningful` training clips are 2,903
`background_flatness` + 94 `background_energy` clips (rule-based: flat
spectrum, low energy) plus 8,216 clips labelled by earlier AED versions that
were themselves trained from those rules. "Flat spectrum = background" is
built into the labels.

## Evidence

Scripts: `quiet_audio_probe.py`, `quiet_snr_probe.py`. Per-clip outputs are
on Drive (`AED validation results/quiet_audio_20261105/`).

### 1. Signal-to-background ratio drives the score (controlled test)

34 held-out on-trail clips (all kept by AED) were turned down and mixed into
held-out background clips, then scored with v3. Turning the clip down on its
own never changes the score.

| Gain on the walk clip | Median SNR vs background | Dropped at 0.5 | Dropped, attenuated only (no background) |
|---|---|---|---|
| 0 dB | +9.9 dB | 18% | 0% |
| −12 dB | −2.1 dB | 29% | 0% |
| −18 dB | −8.1 dB | 53% | 0% |
| −24 dB | −14.1 dB | 74% | 0% |
| −42 dB | −32.1 dB | 100% | 0% |

At 0 dB gain the 6 dropped pairs all had the walk clip below the background
(SNR −6 to −26 dB); the kept pairs had a median SNR of +11 dB.

### 2. The quiet off-trail walk looks like background, not like on-trail walks

Medians (held-out clips; background and on-trail sampled 150, some unreadable
from Drive):

| Set | Clips | AED p | Level (dBFS) | Spectral flatness | Mel median vs max | Energy > 4 kHz |
|---|---|---|---|---|---|---|
| Off-trail walk, dropped | 34 | 0.93 | −51.7 | 0.47 | −19.6 dB | 37% |
| Off-trail walk, kept | 65 | 0.05 | −51.6 | 0.46 | −23.1 dB | 37% |
| On-trail walks | 148 | 0.00 | −31.9 | 0.06 | −33.7 dB | 95% |
| Background | 138 | 1.00 | −47.5 | 0.33 | −24.1 dB | 65% |

- The whole off-trail walk is about 20 dB quieter than the on-trail walks and
  even flatter than the labelled background.
- Inside the walk, dropped and kept clips have the same level. What separates
  them is flatness and how much the spectrogram stands out of its floor:
  Spearman correlation with AED's score is +0.57 for flatness, +0.44 for the
  mel median (flatter, less contrast), −0.38 for loudest-frame contrast. The
  clips that tip over are the ones where nothing stands out of the noise.

### 3. Independent results from the team

**Coleman, HP retraining** (`hp-retraining`, `HP_Model/retraining/REPORT.md`,
"AM3 (quiet recorder)"): on the full re-cut data, AM3 is 18–20 dB quieter than
the other recorders at every hour (median RMS −49.6 dBFS vs −28.7 to −31.7),
with flatter spectra, and **AED rejects 25% of AM3's clips against 0.5–1.3%
on the other recorders**. AM3's gain setting cannot be read (no CONFIG.TXT,
re-cut clips have no header). He also found AED's score adds nothing as an HP
feature (ranked last of 24).

**Ben, pipeline step 2** (`step1-4-testing-notebooks`,
`inference_pipeline/testing/notebooks/02_tinycnn_filter.ipynb`): on 1,200 AM1
clips AED flagged 12 as not_meaningful, all in the first 16 minutes after AM1
started recording on 17 March (09:31–09:47, the setup period), with scores
0.56–1.00. In a manual listen, **9 of the 12 had clear footsteps or human
voices**.

Measured on the 12 clips (`quiet_audio_probe.py` descriptors, v3 score
reproduced: 0.56–1.00), medians:

| Set | Level (dBFS) | Peak (dBFS) | Loudest-frame contrast | Flatness |
|---|---|---|---|---|
| AM1 setup clips (12) | −31.7 | −4.1 | 13.3 dB | 0.40 |
| On-trail walks (kept) | −31.9 | −16.5 | 4.7 dB | 0.06 |
| Off-trail walk, dropped | −51.7 | −31.1 | 3.8 dB | 0.47 |
| Background | −47.5 | −30.8 | 3.5 dB | 0.33 |

These clips are **not quiet**: normal level, peaks close to full scale (−0.2
dBFS on two clips) and loudest frames 13 dB above the rest, about three times
the contrast of any other set. What they share with the rejected quiet clips
is a broadband, flat spectrum (0.40, between background and the off-trail
walk, far from the 0.06 of the on-trail walks that AED keeps). Near-full-scale
peaks also suggest the recorder was being handled. So AED rejects them for
their spectral shape, not their level.

Coleman's AM3 result fits the low-SNR route; Ben's AM1 clips fit the
broadband-transient route. Both end at a flat spectrum.

### 4. The quiet recorder is AM3 (held-out sample)

Held-out meaningful clips, sampled up to 60 per recorder:

| Recorder | Clips | Median level (dBFS) | Median flatness | Meaningful dropped |
|---|---|---|---|---|
| Audio_Moth_2 | 58 | −29.0 | 0.04 | 0% |
| Audio_Moth_3 | 32 | −45.7 | 0.19 | 6% |
| Audio_Moth_4 | 59 | −30.2 | 0.03 | 0% |
| Audio_Moth_5 | 56 | −31.0 | 0.06 | 0% |

AM3 records about 15 dB lower than the others with a much flatter spectrum,
consistent with Coleman's 25% rejection there (his figure covers all of AM3's
clips; this table is a 32-clip held-out sample). Whether that is the device's
gain setting, its placement, or a microphone fault is not known; AM1 and AM6
have no held-out clips here.

## What it means

- Since nothing is dropped any more, the score still reaches HP. Read it as
  "how much the clip stands out of its own noise floor", not "no human here".
  Low-SNR human activity (distant, quiet, or on a low-gain recorder) gets a
  background-like score.
- Per-recorder levels differ by ~15 dB, so a single cutoff treats recorders
  differently. Recorder gain/level is worth checking in the field metadata.
- Options if this matters for HP (none needed for the current pipeline):
  add quiet human clips to AED's `meaningful` training data; stop using the
  flatness rule as a background label; or give HP the recorder's noise level
  alongside the AED score.

## Limits

- The SNR test mixes real clips; it shows the mechanism, not field rates.
- The comparison with v3's own training clips failed: Google Drive would not
  download clips that were not already cached (9 and 3 clips read), so the
  training-label argument rests on the label counts above.
- Nobody has listened to the 34 dropped off-trail clips yet, so some may hold
  no audible activity.

## Open

- Ben's 12 AM1 setup clips (`Audio_Moth_1_20250317_` + 093130, 093639, 093824,
  094151, 094203, 094630, 094657, 094736, 094739, 094742, 094745, 094751) are
  measured above; per-clip values in `am1_setup_clips.csv` on Drive. Ask Ben
  which 9 had footsteps or voices, to compare those with the other 3, and
  whether the recorder was being handled.
- Listen to the dropped off-trail clips.
