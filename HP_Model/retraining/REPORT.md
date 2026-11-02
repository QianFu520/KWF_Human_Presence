# HP model retraining: report (Nov 1, 2026)

Coleman Bryant, Team Jacamar. Branch `hp-retraining`. All numbers below come from the
scripts in `HP_Model/retraining/`; data, training sets, model files and bundles are kept
outside the repo.

**Summary.** All six recorders are processed (616,381 clips). The candidate model is the
no-hour, no-weather feature set ("27", 23 features in practice; see Open issues). On test it
reaches PR-AUC 0.43 (within-hour 0.45), with recall 0.28 at precision 0.71. The 25- and
29-feature models score higher, but mainly because they can tell **when** a clip was recorded:
most staged events happened at 16:00–18:00, and the training data has almost no quiet audio
from those hours. Weather adds nothing beyond identifying the session. **Recommendation: no
weather.**

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

**Feature sets** (names from `hp_features.py`):

| Set | Features | Role |
|---|---|---|
| 27 | `HP_FEATURES_25` minus `hour_sin`, `hour_cos` (23 features) | **Candidate** |
| 27rel | 27, with `RMS_Energy` and `Volume_Spike_15s` divided by each recorder's median train RMS | Diagnostic (AM3) |
| 25 | `HP_FEATURES_25` | Reference: schedule-inflated upper bound |
| 29 | `HP_FEATURES_29` (25 + Temperature, Humidity, Windspeed, Volume_Wind_Ratio) | Reference: schedule- and session-inflated upper bound |

**Model:** StandardScaler + `HistGradientBoostingClassifier(class_weight="balanced")`, early
stopping off. sklearn's early stopping uses a random clip-level split, which would leak
neighbouring clips. Regularisation was tuned on validation only (`tune_val.py`): max_depth ×
min_samples_leaf × l2 × max_iter, best validation PR-AUC per set. **The configs were locked
before test was scored.**

| Set | max_depth | min_samples_leaf | l2 | max_iter |
|---|---|---:|---:|---:|
| 27 | None | 500 | 0 | 200 |
| 27rel | None | 20 | 10 | 200 |
| 25 | 6 | 5000 | 10 | 50 |
| 29 | 6 | 5000 | 0 | 200 |

**Threshold:** chosen on validation as the threshold with the most recall at ≤ 5 false-alert
**episodes** per recorder per day. An episode is a run of false-positive clips on one recorder,
less than 30 s apart, capped at 5 min, counted per recorder-day of negative audio.
**Test was scored once.**

**Metrics.** PR-AUC is the area under the precision-recall curve; ROC-AUC is the area under
the ROC curve. **Within-hour PR-AUC** is computed separately in each hour of day that has
both positives and negatives, then averaged weighted by positives. That way, hour of day
alone can't separate the classes. **It is the main comparison between 25, 27 and 29.**

## 3. Results

### Key finding: the time/session confound

More than half the events (**34 of 61**) were staged at **16:00–18:00**. The ±30 min event
buffers remove nearly all negatives at those hours: validation has **0** negatives at 16:00
and 408 at 17:00, against 1,655 positives at 16:00–17:59. So `hour_sin`/`hour_cos` can separate
the classes without hearing anything. On val, the 25-feature model gives *negative* clips at
17:00 a median score of 0.96, and its top two features are `hour_cos` and `hour_sin`.

Weather is hourly and identical for all recorders, so it identifies the session (day and hour).
Because the buffers ensure no negative shares a session with a positive, even within-hour PR-AUC
can't remove that: the 29-feature model scores 1.00 within-hour at some hours and reaches train
PR-AUC 1.000.

**Real activity won't follow the staging schedule.** A model that leans on time of day could
miss activity at other hours, which is why 25 and 29 are reported only as upper bounds.

### Overall (all clips)

| Set | Role | Val PR-AUC | **Val within-hour** | Test PR-AUC | **Test within-hour** | Test ROC-AUC | Test precision | Test recall | Test F1 | Test FA clips / rec / day | Test FA episodes / rec / day |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **27** | candidate | 0.423 | **0.714** | 0.426 | **0.450** | 0.910 | 0.708 | 0.275 | 0.396 | 99 | 10.6 |
| 27rel | diagnostic | 0.439 | 0.736 | 0.423 | 0.442 | 0.913 | 0.776 | 0.248 | 0.376 | 63 | 9.4 |
| 25 | reference | 0.712 | 0.622 | 0.687 | 0.484 | 0.986 | 0.481 | 0.693 | 0.568 | 655 | 10.6 |
| 29 | reference | 0.896 | 0.924* | 0.871 | 0.919* | 0.997 | 0.696 | 1.000 | 0.821 | 382 | 8.6 |

\* inflated: weather identifies the session.
Thresholds: 27 0.9882 · 27rel 0.9894 · 25 0.9484 · 29 0.1930.
Within-hour hours: val 13, 14, 17; test 08, 17 (the only hours with both classes).

- **Within-hour, 27 vs 25 is mixed.** On val, 27 beats 25 (0.71 vs 0.62). On test, 25 is
  slightly ahead (0.48 vs 0.45). Test's within-hour score rests on only two hours (08:00 and
  17:00), so neither set shows a clear acoustic advantage. The large overall-PR-AUC gap
  (0.69 vs 0.43) is mostly the schedule.
- **The threshold doesn't transfer well.** It was set for ≤ 5 episodes per recorder per day on
  val, but test gives 8.6–10.6 episodes for every model. 27's false-alert clips go from 7.7 per
  day on val to 99 on test.
- **AED-passed clips only** are slightly better for every model (27: PR-AUC 0.448, within-hour
  0.473, precision 0.748, recall 0.287).

### Test, AED-passed only

| Set | PR-AUC | Within-hour | Precision | Recall | F1 | FA episodes / rec / day |
|---|---:|---:|---:|---:|---:|---:|
| 27 | 0.448 | 0.473 | 0.748 | 0.287 | 0.415 | 11.3 |
| 27rel | 0.444 | 0.466 | 0.824 | 0.259 | 0.395 | 10.0 |
| 25 | 0.705 | 0.496 | 0.518 | 0.679 | 0.588 | 11.0 |
| 29 | 0.962 | 0.978* | 0.747 | 1.000 | 0.855 | 8.8 |

### Test per recorder (27, candidate)

| Recorder | Positives | PR-AUC | Within-hour | Precision | Recall | FA episodes / rec / day |
|---|---:|---:|---:|---:|---:|---:|
| AM1 | 401 | 0.577 | 0.835 | 1.000 | 0.037 | 0.0 |
| AM2 | 0 | – | – | – | – | 2.0 |
| AM3 | 0 | – | – | – | – | 0.0 |
| AM4 | 882 | 0.537 | 0.677 | 0.797 | 0.459 | 20.9 |
| AM5 | 1,265 | 0.382 | 0.495 | 0.622 | 0.213 | 16.4 |
| AM6 | 421 | 0.473 | – | 0.648 | 0.297 | 15.4 |

AM2 and AM3 have no test events (event split), so only their false alerts are measured.
Per-recorder within-hour values rest on few hours ("–" = no hour with both classes).
Reference models per recorder are in `final_w10/test_summary.csv`.

### Test per event type (positives of that type vs all test negatives)

| Event type | Positives | 27 recall | 27 PR-AUC | 25 recall | 25 PR-AUC | 29 recall | 29 PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Presence on trail | 1,949 | 0.385 | 0.489 | 0.788 | 0.727 | 1.000 | 0.843 |
| Gunshot | 1,604 | 0.325 | 0.413 | 0.746 | 0.645 | 1.000 | 0.805 |
| Chainsaw | 1,549 | 0.173 | 0.210 | 0.481 | 0.452 | 1.000 | 0.727 |
| Vehicle | 801 | 0.020 | 0.039 | 0.499 | 0.161 | 1.000 | 0.504 |
| Presence off trail | 401 | 0.002 | 0.015 | 0.000 | 0.055 | 1.000 | 0.293 |

29's recall of 1.000 on every type at 382 false-alert clips per recorder per day is the session
shortcut, not detection.

### Weather conclusion

There is **no evidence that weather helps detection beyond identifying the session**. Its gains
appear exactly where weather can fingerprint the day and hour, and it reaches perfect train
scores. **Recommend no weather** (25-family features): no internet needed in the field, and
nothing measurable is lost.

### AM3 (quiet recorder) and the relative-loudness diagnostic

- AM3 is about **18–20 dB quieter** than the other recorders at every hour (median RMS
  −49.6 dBFS vs −28.7 to −31.7), with flatter spectra. AED rejects 25% of its clips, against
  0.5–1.3% elsewhere. Its gain setting can't be read locally: the re-cut clips have no header,
  and the AM1 Drive zips have no CONFIG.TXT.
- AM3's on-trail and gunshot events are in **val** (416 positive clips); its off-trail event is
  in train (222). It has no test positives.
- With hour features, AM3 looks perfect on val (PR-AUC 0.99), but only because its val
  positives are all at 16–17h.
- Without hour features, 27 has AM3 val PR-AUC **0.05**. Recorder-relative loudness (27rel)
  raises it to **0.12**, but AM3 recall at ≤ 10 episodes per day stays **0**. Overall, 27rel is
  about equal to 27 (test PR-AUC 0.42 vs 0.43; fewer false alerts, slightly lower recall).
- Conclusion: relative loudness helps AM3 a little, not enough to detect its events. Not
  adopted for the candidate.

### Bundles

Saved outside the repo in `C:\BirdNET ONNX\hp_audit\bundles\`, in Patrick's proposed format
(joblib dict) plus extra keys. scikit-learn 1.8.0, joblib 1.5.3. All pass `check_bundles.py`.

| File | feature_set | Features | Threshold | Role |
|---|---|---:|---:|---|
| `hp-27-20261101.joblib` | 27 | 23 | 0.9882 | candidate |
| `hp-25-20261101.joblib` | 25 | 25 | 0.9484 | reference (trained_on notes time-of-day shortcuts) |
| `hp-29-20261101.joblib` | 29 | 29 | 0.1930 | reference (trained_on notes time-of-day and weather/session shortcuts) |

Keys: model, scaler (fitted on a DataFrame, so `feature_names_in_` is set), features,
feature_set, threshold, version, trained_on, created, sklearn_version, val_metrics (precision,
recall, false-alert episodes per recorder per day), threshold_rule, git_commit (a8e21c2).

## 4. Open issues

1. **"27" naming.** The original plan defined 27 as **29 minus hour features** (keeps weather, 27
   features). The candidate built here is **25 minus hour features** (23 features), as requested
   on Nov 1, but it is labelled `feature_set: "27"`. Before Patrick's loader allows "27", agree
   on the name (e.g. "23") or the definition.
2. **The confound limits every metric here.** Within-hour PR-AUC removes the hour shortcut but
   not the session (day) shortcut, and rests on 2–3 hours. Leave-one-day-out CV **wouldn't
   remove the confound** for 4 of 5 days (Mar 17–20 all have 16:00–18:00 events; Mar 21 has
   none), so it's a later check, not the fix.
3. **Threshold transfer.** Val → test false alerts roughly double (≤ 5 → 8.6–10.6 episodes per
   recorder per day). Day-to-day variation is large; a single split's threshold is fragile.
4. **Weak types.** The candidate barely detects vehicle (recall 0.02) and off-trail presence
   (0.002, one event on AM5 in test). AM3's events are missed.
5. **Recommendations:**
   - (a) In future field sessions, **record quiet control periods at the staging hours**
     (16:00–18:00 and the morning slots), so negatives exist at the same times as events.
   - (b) **Ask Qian for recorder coordinates.** Recorders too far away to hear an event could
     then supply negatives simultaneous with it. That is the only way, with this data, to
     separate detection from time of day.
6. Not done yet: other label windows (per_type, tight), leave-one-day-out CV, an MLP comparison.
