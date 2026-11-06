# Why AED scores quiet audio as background

Status: 2026-11-05, first pass. Still to add: Ben's AM1 setup clips, Coleman's
AM3 numbers, and listening to the dropped off-trail clips (see [Open](#open)).

## Finding

AED does not react to loudness: its input is a mel spectrogram in dB relative
to each clip's own maximum, so turning a clip down changes nothing. What it
reacts to is **how far the sound stands above the clip's noise floor**. When a
real sound is faint relative to the background, the normalised spectrogram is
dominated by broadband noise; it looks flat, and flat is what AED learned to
call background. Quiet recorders and distant activity therefore get rejected.

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

### 3. The quiet recorder is AM3

Held-out meaningful clips, sampled up to 60 per recorder:

| Recorder | Clips | Median level (dBFS) | Median flatness | Meaningful dropped |
|---|---|---|---|---|
| Audio_Moth_2 | 58 | −29.0 | 0.04 | 0% |
| Audio_Moth_3 | 32 | −45.7 | 0.19 | 6% |
| Audio_Moth_4 | 59 | −30.2 | 0.03 | 0% |
| Audio_Moth_5 | 56 | −31.0 | 0.06 | 0% |

AM3 records about 15 dB lower than the others with a much flatter spectrum,
consistent with Coleman's 25% rejection there. Whether that is the device's
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

- Ben's AM1 setup clips (real footsteps and voices rated as background): run
  `python validation/quiet_audio_probe.py --out <dir> --extra am1_setup=<folder>`.
- Coleman's AM3 numbers: compare with the per-recorder table.
- Listen to the dropped off-trail clips.
