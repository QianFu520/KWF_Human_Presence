# Why AED scores quiet audio as background

Status: 2026-11-06. Includes Coleman Bryant's full-dataset AED results
(all 616,381 clips, six recorders) and Ben's 12 AM1 setup clips (measured and
listened to). Still open: listening to the dropped
off-trail clips (see [Open](#open)).

## Finding

AED does not react to loudness: its input is a mel spectrogram in dB relative
to each clip's own maximum, so turning a clip down changes nothing. What it
reacts to is **how far the sound stands above the clip's noise floor**. When a
real sound is faint relative to the background, the normalised spectrogram is
dominated by broadband noise; it looks flat, and flat is what AED learned to
call background. Quiet recorders and distant activity therefore get rejected:
on the full dataset AED marks 25% of the quiet AM3 recorder's clips as
background, against 0.5–1.3% on the other five (Coleman, section 3).

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

### 3. AM3 on the full dataset (Coleman Bryant)

Coleman ran AED (v3, threshold 0.5) on all **616,381 clips from the six
recorders** for the HP retraining and shared the results on 2026-11-06 (also
in `hp-retraining`, `HP_Model/retraining/REPORT.md`, "AM3 (quiet recorder)"):

- **AED marks 3.2% of all clips not meaningful: 0.5–1.3% on five recorders,
  25% on AM3.**
- AM3 is **18–20 dB quieter** than the other recorders at every hour (median
  RMS −49.6 dBFS vs −28.7 to −31.7), with flatter spectra; probably its gain
  setting or microphone. The gain cannot be read (no CONFIG.TXT; re-cut clips
  have no header).
- AM3's rejections cluster in its quietest periods: **74–99% before dawn
  (03:00–04:00)** and 52–81% from 08:00 to 11:00. Rejected AM3 clips are about
  **7 dB quieter** than accepted ones.
- Real events: AED rejects about 1–2.5% of event clips overall, depending on
  the label window. **Every rejected off-trail presence clip comes from AM3**,
  where AED drops about half (48%) of the clips around its off-trail event:
  the same walk as section 2 (34 of 99 held-out clips there; the label
  windows differ).
- AED's score added nothing as an HP feature (ranked last of 24).

Level and flatness: across AM3's day, the rejected clips are quieter (7 dB),
because quiet periods are mostly noise floor. Within the off-trail walk
(section 2), dropped and kept clips have the same level and differ in
flatness. Both fit the finding: AED never sees level directly, and quietness
matters because it flattens the normalised spectrum.

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

Cheryl listened to all 12 (2026-11-05): **10 have people's voices** (093130
very faint; 093639 through 094742), and 094745 and 094751 do not. AED
rejected all 10 voice clips (scores 0.56–1.00; the faint one, 093130, is the
least confident at 0.56). The two clips without voices are the flattest of the
12 (0.47 and 0.46, against a median of 0.38 for the voice clips), which fits
the same mechanism: voices make a clip less flat, but not enough to pass.

Coleman's AM3 results fit the low-SNR route; Ben's AM1 clips fit the
broadband-transient route. Both end at a flat spectrum.

### 4. Per-recorder check on a held-out sample

A small sample, consistent with Coleman's full-dataset numbers in section 3
(which are the headline). Held-out meaningful clips, up to 60 per recorder:

| Recorder | Clips | Median level (dBFS) | Median flatness | Meaningful dropped |
|---|---|---|---|---|
| Audio_Moth_2 | 58 | −29.0 | 0.04 | 0% |
| Audio_Moth_3 | 32 | −45.7 | 0.19 | 6% |
| Audio_Moth_4 | 59 | −30.2 | 0.03 | 0% |
| Audio_Moth_5 | 56 | −31.0 | 0.06 | 0% |

The sample shows AM3 about 15 dB lower and much flatter. Its 6% rejection
(2 of 32 clips) is too small a sample to estimate the rate; use Coleman's 25%.
AM1 and AM6 have no held-out clips here.

## What it means

- Since nothing is dropped any more, the score still reaches HP. Read it as
  "how much the clip stands out of its own noise floor", not "no human here".
  Low-SNR human activity (distant, quiet, or on a low-gain recorder) gets a
  background-like score.
- Per-recorder levels differ by up to 18–20 dB, so a single cutoff treats
  recorders differently (25% rejected on AM3, ≤ 1.3% elsewhere). Recorder gain/level is worth checking in the field metadata.
- Options if this matters for HP (none needed for the current pipeline):
  add quiet human clips to AED's `meaningful` training data; stop using the
  flatness rule as a background label; or give HP the recorder's noise level
  alongside the AED score.

## Open

- Listen to the dropped off-trail clips.
