# HP model retraining: report (Nov 1, 2026)

Coleman Bryant, Team Jacamar. Branch `hp-retraining`. All numbers below come from the
scripts in `HP_Model/retraining/`; data, training sets, model files and bundles are kept
outside the repo.

**Summary.** All six recorders are processed (616,381 clips). The candidate model is the
**23-feature set: no hour features, no weather**. On test it reaches PR-AUC 0.43
(within-hour 0.45), with recall 0.28 at precision 0.71.

The 25-, 27- and 29-feature models all score higher on test, but each can tell **when** a clip
was recorded:
- **25** has the hour features.
- **27** and **29** have weather, which identifies the day and hour (the "session").

Most staged events happened at 16:00–18:00, and the training data has almost no quiet audio
from those hours or sessions. **The case for 23 rests on deployment, not on test scores** (see
§3). Recommendation: no weather for now.

**23 is the candidate for integration testing, not a field-ready model.** It gives Patrick a
real model to run in the pipeline. Its detection is still weak:
- recall 0.28 at about 10 false-alert episodes per recorder per day on test
- near-zero recall on vehicle (0.02) and off-trail presence (0.002)

---

## 1. AM1 cut

AM1's raw recordings (8 WAVs on Drive, 48 kHz, 16-bit, mono, continuous) were cut with
`cut_am1.py` into 3 s clips on the same grid as the AM2–AM6 re-cut: 144,000 samples,
`Audio_Moth_1_YYYYMMDD_HHMMSS.wav`, final partial clip dropped. The files join with only
3 s gaps (18:00:00 → 18:00:03, 06:25:42 → 06:25:45).

| Day (2025) | AM1 clips |
|---|---:|
| Mar 17 (from 09:31:12) | 17,375 |
| Mar 18 | 28,798 |
| Mar 19 | 28,798 |
| Mar 20 | 28,798 |
| Mar 21 (to 06:25:42) | 7,714 |
| **Total** | **111,483** |

Check: all 901 old AM1 clip names (from the old train/test feature CSVs) exist in the new cut,
so the time grid matches. Raw headers say "medium gain".

All recorders: AM1 111,483 · AM2 95,754 · AM3 55,694 · AM4 118,665 · AM5 117,969 ·
AM6 116,816 = **616,381 clips**. The full pipeline (`--no_weather`) ran on all of them with no
read errors. Weather was added afterwards with `add_weather.py`: 103 of 103 hours, every clip
matched, no missing values.

## 2. Training set

| Item | Definition |
|---|---|
| Events | `simulations.csv`: 76 staged events, 61 with audio. Grouped (`build_event_split.py`): same recorder with overlapping windows, or different recorders within 10 min → 26 groups. |
| Split | Event groups → train / val / test (60/20/20 of positive clips). Val and test each contain trail presence, vehicle, gunshot and chainsaw. |
| Positives (w10) | Clips on the event's own recorder within ±10 min of the event. |
| Negatives | Clips outside every event's ±30 min buffer (on **all** recorders) and outside trim exclusions (first/last 30 min of each recording segment; gaps > 1 min start a new segment). |
| Negative split | 20-minute clock-time blocks **shared by all recorders**. Blocks overlapping an event buffer follow that event's split; the rest are split 60/20/20 by a seeded search (seed 42) that balances clip counts and hour-of-day coverage. A 1-min guard drops negatives at split boundaries. (`plan_negative_split.py`) |
| AED | AED-rejected clips are **kept** (`features_all.csv`). Results are reported on all clips and on AED-passed clips only. |
| Missing values | None (0 NaN / inf in all 29 features). |

| Split | Clips | Positives | Negatives |
|---|---:|---:|---:|
| Train | 306,067 | 7,340 | 298,727 |
| Val | 99,961 | 2,217 | 97,744 |
| Test | 100,613 | 2,969 | 97,644 |

**Feature sets** (base lists from `hp_features.py`):

| Set | Features | Hour features | Weather | Role |
|---|---|:-:|:-:|---|
| **23** | `HP_FEATURES_25` minus `hour_sin`, `hour_cos` | – | – | **Candidate** |
| 23rel | 23, with `RMS_Energy` and `Volume_Spike_15s` divided by each recorder's median train RMS | – | – | Diagnostic (AM3) |
| 25 | `HP_FEATURES_25` | ✓ | – | Reference: time-of-day shortcut |
| 27 | `HP_FEATURES_29` minus `hour_sin`, `hour_cos` (Qian's approved 27) | – | ✓ | Reference: weather/session shortcut |
| 29 | `HP_FEATURES_29` | ✓ | ✓ | Reference: time-of-day and session shortcuts |

(Until Nov 1 the 23-feature set was mislabelled "27" in this work. It has been relabelled; the
model and its numbers are unchanged.)

**Model:** StandardScaler + `HistGradientBoostingClassifier(class_weight="balanced")`, early
stopping off. sklearn's early stopping uses a random clip-level split, which would leak
neighbouring clips. Regularisation was tuned on validation only (`tune_val.py`): max_depth
{3, 4, 6, None} × min_samples_leaf {20, 500, 5000} × l2 {0, 10} × max_iter {20, 50, 100, 200},
best validation PR-AUC per set. **Configs were locked before test was scored.**

| Set | max_depth | min_samples_leaf | l2 | max_iter |
|---|---|---:|---:|---:|
| 23 | None | 500 | 0 | 200 |
| 23rel | None | 20 | 10 | 200 |
| 25 | 6 | 5000 | 10 | 50 |
| 27 | 4 | 500 | 10 | 200 |
| 29 | 6 | 5000 | 0 | 200 |

**Threshold:** chosen on validation as the threshold with the most recall at ≤ 5 false-alert
**episodes** per recorder per day. An episode is a run of false-positive clips on one recorder,
less than 30 s apart, capped at 5 min, counted per recorder-day of negative audio.
**Test was scored once per model.**

**Metrics.** PR-AUC is the area under the precision-recall curve; ROC-AUC is the area under
the ROC curve. **Within-hour PR-AUC** is computed separately in each hour of day that has
both positives and negatives, then averaged weighted by positives. That way, hour of day
alone can't separate the classes. It is the main comparison between models. It removes the
**hour** shortcut but **not the session shortcut**: no negative shares a day and hour with a
positive, so anything that identifies the day (weather) still helps.

## 3. Results

### Key finding: the time/session confound

More than half the events (**34 of 61**) were staged at **16:00–18:00**. The ±30 min event
buffers remove nearly all negatives at those hours: validation has **0** negatives at 16:00
and 408 at 17:00, against 1,655 positives at 16:00–17:59. So `hour_sin`/`hour_cos` can separate
the classes without hearing anything. On val, the 25-feature model gives *negative* clips at
17:00 a median score of 0.96, and its top two features are `hour_cos` and `hour_sin`.

Weather is hourly and identical for all recorders, so it identifies the session (day and hour).
The 27-feature model has no hour features, but its top three features are wind speed,
temperature and humidity (permutation importance 0.55, 0.39, 0.36, against ≤ 0.03 for any
acoustic feature). It reaches val within-hour PR-AUC 0.99.

**Real activity won't follow the staging schedule.** A model that leans on time of day or session
could miss activity at other hours and on other days.

### Overall (all clips)

| Set | Role | Val PR-AUC | **Val within-hour** | Test PR-AUC | **Test within-hour** | Test ROC-AUC | Test precision | Test recall | Test F1 | Test FA clips / rec / day | Test FA episodes / rec / day |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **23** | candidate | 0.423 | **0.714** | 0.426 | **0.450** | 0.910 | 0.708 | 0.275 | 0.396 | 99 | 10.6 |
| 23rel | diagnostic | 0.439 | 0.736 | 0.423 | 0.442 | 0.913 | 0.776 | 0.248 | 0.376 | 63 | 9.4 |
| 25 | reference (hour) | 0.712 | 0.622 | 0.687 | 0.484 | 0.986 | 0.481 | 0.693 | 0.568 | 655 | 10.6 |
| 27 | reference (session) | 0.899 | 0.988* | 0.659 | 0.682* | 0.979 | 0.681 | 0.544 | 0.605 | 223 | 5.9 |
| 29 | reference (hour + session) | 0.896 | 0.924* | 0.871 | 0.919* | 0.997 | 0.696 | 1.000 | 0.821 | 382 | 8.6 |

\* weather identifies the session, so within-hour PR-AUC does not remove its shortcut.
Thresholds: 23 0.9882 · 23rel 0.9894 · 25 0.9484 · 27 0.6430 · 29 0.1930.
Within-hour hours: val 13, 14, 17; test 08, 17 (the only hours with both classes).

### Why 23 is the candidate: deployment, not test scores

**On test, 23 is not the best model.** 25 edges it within-hour (0.48 vs 0.45), and the
weather models score higher still (27: 0.68, 29: 0.92). The case for 23 rests on how the model
will be used:

1. **No time-of-day shortcut.** 25 and 29 lean mainly on `hour_sin`/`hour_cos`, which encode
   the staging schedule. Real poaching won't follow it, so those scores won't carry over to the
   field.
2. **No session shortcut and no internet.** 27 and 29 rely on weather, which identifies the
   session here. In the field it would need an Open-Meteo connection at inference time.
3. **What 23 scores is what it hears.** Its top features are acoustic (Spectral_Rolloff_85,
   MFCC_9, RMS_Energy, Spectral_Contrast). Its test numbers are low, but they are the closest
   thing here to field performance.

The within-hour comparison doesn't settle it either way. On val, 23 beats 25 (0.71 vs 0.62); on
test, 25 is slightly ahead (0.48 vs 0.45). Test's within-hour score rests on only two hours
(08:00 and 17:00).

### Test, AED-passed only

| Set | PR-AUC | Within-hour | Precision | Recall | F1 | FA episodes / rec / day |
|---|---:|---:|---:|---:|---:|---:|
| 23 | 0.448 | 0.473 | 0.748 | 0.287 | 0.415 | 11.3 |
| 23rel | 0.444 | 0.466 | 0.824 | 0.259 | 0.395 | 10.0 |
| 25 | 0.705 | 0.496 | 0.518 | 0.679 | 0.588 | 11.0 |
| 27 | 0.733 | 0.789* | 0.779 | 0.524 | 0.626 | 6.1 |
| 29 | 0.962 | 0.978* | 0.747 | 1.000 | 0.855 | 8.8 |

Every model does a little better on AED-passed clips.

### Test per recorder

| Recorder | Positives | 23 PR-AUC | 23 recall | 23 FA episodes / day | 25 PR-AUC | 27 PR-AUC | 29 PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| AM1 | 401 | 0.577 | 0.037 | 0.0 | 0.588 | 0.642 | 0.956 |
| AM2 | 0 | – | – | 2.0 | – | – | – |
| AM3 | 0 | – | – | 0.0 | – | – | – |
| AM4 | 882 | 0.537 | 0.459 | 20.9 | 0.714 | 0.656 | 0.927 |
| AM5 | 1,265 | 0.382 | 0.213 | 16.4 | 0.798 | 0.822 | 0.913 |
| AM6 | 421 | 0.473 | 0.297 | 15.4 | 0.813 | 0.381 | 0.792 |

AM2 and AM3 have no test events (event split), so only their false alerts are measured. Full
per-recorder tables (within-hour, precision, false alerts for every model) are in
`final_w10/test_summary.csv` and `final_w10_27/test_summary.csv`.

### Test per event type (positives of that type vs all test negatives)

| Event type | Positives | 23 recall | 23 PR-AUC | 25 recall | 25 PR-AUC | 27 recall | 27 PR-AUC | 29 recall | 29 PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Presence on trail | 1,949 | 0.385 | 0.489 | 0.788 | 0.727 | 0.558 | 0.613 | 1.000 | 0.843 |
| Gunshot | 1,604 | 0.325 | 0.413 | 0.746 | 0.645 | 0.550 | 0.598 | 1.000 | 0.805 |
| Chainsaw | 1,549 | 0.173 | 0.210 | 0.481 | 0.452 | 0.420 | 0.474 | 1.000 | 0.727 |
| Vehicle | 801 | 0.020 | 0.039 | 0.499 | 0.161 | 0.502 | 0.196 | 1.000 | 0.504 |
| Presence off trail | 401 | 0.002 | 0.015 | 0.000 | 0.055 | 0.010 | 0.093 | 1.000 | 0.293 |

29's recall of 1.000 on every type, at 382 false-alert clips per recorder per day, is the
shortcut, not detection.

### Weather conclusion

**No weather for now.** 27 scores better than 23 on test (within-hour 0.68 vs 0.45), but this
data can't separate a real benefit from session identification:
- **No shared sessions:** no negative shares a day and hour with a positive, so anything that
  identifies the session separates the classes.
- **Where the gain comes from:** 27's top features are wind speed, temperature and humidity, not
  acoustic features.
- **It only partly carries over:** 27's val within-hour score of 0.99 drops to 0.68 on test,
  where sessions differ from training.

Staying with the 25-family features (no hour features in the candidate) also means no internet
is needed in the field. **Re-test weather once clean simultaneous negatives exist** (quiet
control recordings at the staging hours, or recorders too far away to hear an event; see Open
issues).

### AM3 (quiet recorder) and the relative-loudness diagnostic

- AM3 is about **18–20 dB quieter** than the other recorders at every hour (median RMS
  −49.6 dBFS vs −28.7 to −31.7), with flatter spectra. AED rejects 25% of its clips, against
  0.5–1.3% elsewhere. Its gain setting can't be read locally: the re-cut clips have no header,
  and the AM1 Drive zips have no CONFIG.TXT.
- AM3's on-trail and gunshot events are in **val** (416 positive clips); its off-trail event is
  in train (222). It has no test positives.
- With hour features, AM3 looks perfect on val (PR-AUC 0.99), but only because its val positives
  are all at 16–17h.
- Without hour features, 23 has AM3 val PR-AUC **0.05**. Recorder-relative loudness (23rel)
  raises it to **0.12**, but AM3 recall at ≤ 10 episodes per day stays **0**. Overall, 23rel is
  about equal to 23 (test PR-AUC 0.42 vs 0.43; fewer false alerts, slightly lower recall).
- Conclusion: relative loudness helps AM3 a little, not enough to detect its events. Not adopted
  for the candidate.

### Bundles

Saved outside the repo in `C:\BirdNET ONNX\hp_audit\bundles\`, in Patrick's proposed format
(joblib dict) plus extra keys. scikit-learn 1.8.0, joblib 1.5.3. All four pass
`check_bundles.py`, which also requires `len(features) == int(feature_set)`.

| File | feature_set | Features | Threshold | Role | git_commit |
|---|---|---:|---:|---|---|
| `hp-23-20261101.joblib` | 23 | 23 | 0.9882 | **candidate** | a8e21c2 |
| `hp-25-20261101.joblib` | 25 | 25 | 0.9484 | reference: time-of-day shortcuts | a8e21c2 |
| `hp-27-20261101.joblib` | 27 | 27 | 0.6430 | reference: weather/session shortcuts | fa328af |
| `hp-29-20261101.joblib` | 29 | 29 | 0.1930 | reference: time-of-day and session shortcuts | a8e21c2 |

Reference bundles say so in `trained_on`.

The 23 bundle is the model first saved as `hp-27-20261101.joblib`. It was relabelled (same model
and threshold, identical predictions) and the old file removed; its `trained_on` notes the
relabel. Patrick's draft format lists feature sets "25" and "29"; "23" and "27" still need to
be allowed.

## 4. Open issues

1. **Threshold transfer.** Val → test false alerts rise for most models (set for ≤ 5 episodes
   per recorder per day; test gives 8.6–10.6 for 23, 25 and 29, and 5.9 for 27). 23's
   false-alert clips go from 7.7 per day on val to 99 on test. Day-to-day variation is large, so
   a threshold from a single split is fragile. Recalibrate on more days before deployment.
2. **Weak event types.** The candidate barely detects **vehicle** (test recall 0.02) and
   **off-trail presence** (0.002; one event, on AM5, in test). AM3's events are missed (val
   recall 0).
3. **The confound limits every metric here.** Within-hour PR-AUC removes the hour shortcut but
   not the session (day) shortcut, and rests on 2–3 hours. Leave-one-day-out CV **wouldn't remove
   the confound** for 4 of 5 days (Mar 17–20 all have 16:00–18:00 events; Mar 21 has none), so
   it's a later check, not the fix.
4. **Feature-set names in the bundle format.** Patrick's loader needs to accept "23" and "27".
   Add a 23 list (and the approved 27) to `hp_features.py` so training and inference share them.
5. **Recommendations:**
   - (a) In future field sessions, **record quiet control periods at the staging hours**
     (16:00–18:00 and the morning slots), so negatives exist at the same times as events.
   - (b) **Ask Qian for recorder coordinates.** Recorders too far away to hear an event could
     then supply negatives simultaneous with it. That is the only way, with this data, to
     separate detection from time of day and session.
6. Not done yet: other label windows (per_type, tight), leave-one-day-out CV, an MLP comparison.
