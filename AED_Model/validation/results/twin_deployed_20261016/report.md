# Digital Twin on field recordings: measured against ground truth

AED gate threshold 0.75, BirdNET skipped (twin defaults). Twin backend: `../../digital_twin_mvp/backend`.

## Recall per event type

`recall` is what the dashboard reports. `chance_recall` is what detections at the background rate would score by luck; `density_ratio` is detections per minute inside the windows relative to outside (1.0 = no different from background). Read recall against both.

```
                    type  events  detected  recall  chance_recall  det_per_min  density_ratio  no_stage1_trigger  aed_rejected
                CHAINSAW       2         2   1.000          1.000        1.000          0.291                  0             0
                 GUNSHOT       4         4   1.000          0.992        2.343          1.677                  0             0
HUMAN_PRESENCE_OFF_TRAIL       2         1   0.500          0.984        0.198          0.100                  1             0
 HUMAN_PRESENCE_ON_TRAIL       6         6   1.000          1.000        2.723          1.709                  0             0
                 VEHICLE       2         2   1.000          0.998        5.419          2.868                  0             0
```

Staged events often overlap (e.g. a gunshot during a chainsaw run inside a human-presence walk), so a detection can count for several types at once.

## Recordings

```
                   recording  duration_h  detections  detections_outside_events  det_per_min_outside  stage1_triggers_outside  aed_keep_rate_outside wall_min  twin_recall  twin_precision  events  matched
Audio_Moth_2_20250318_062543      11.571         536                        521                0.773                      522                  0.998     None        1.000           0.028       1        1
Audio_Moth_3_20250319_080049       9.986         428                        392                0.683                      413                  0.949     None        0.667           0.084       3        2
Audio_Moth_5_20250317_112812       6.530        1451                       1390                3.739                     1444                  0.963     None        1.000           0.042       3        3
Audio_Moth_5_20250319_062544      10.072        1105                       1005                1.738                     1010                  0.995     None        1.000           0.090       3        3
Audio_Moth_5_20250321_062545       7.395         921                        833                1.983                      833                  1.000     None        1.000           0.096       6        6
```

Precision is low by design here: AED flags birds and other wildlife, and every detection outside a staged human event counts as a false positive.

## Events

```
                   recording                     type   start_s  duration_s  detections  detected  chance_hit  det_per_min  density_ratio  stage1_triggers  aed_kept                    outcome
Audio_Moth_2_20250318_062543  HUMAN_PRESENCE_ON_TRAIL 37476.000    1197.000          15      True       1.000        0.750          0.971               15        15                   detected
Audio_Moth_3_20250319_080049                  GUNSHOT 32922.000     297.000          13      True       0.967        2.600          3.808               13        13                   detected
Audio_Moth_3_20250319_080049 HUMAN_PRESENCE_OFF_TRAIL    66.000     297.000           0     False       0.967        0.000          0.000                0         0 missed: no stage-1 trigger
Audio_Moth_3_20250319_080049  HUMAN_PRESENCE_ON_TRAIL 32274.000    1197.000          36      True       1.000        1.800          2.637               36        36                   detected
Audio_Moth_5_20250317_112812                 CHAINSAW 19665.000     297.000           9      True       1.000        1.800          0.481               11         9                   detected
Audio_Moth_5_20250317_112812                  GUNSHOT 19545.000     300.000          11      True       1.000        2.178          0.583               11        11                   detected
Audio_Moth_5_20250317_112812  HUMAN_PRESENCE_ON_TRAIL 18876.000    1197.000          61      True       1.000        3.050          0.816               65        61                   detected
Audio_Moth_5_20250319_062544  HUMAN_PRESENCE_ON_TRAIL  8283.000    1200.000          93      True       1.000        4.638          2.669               93        93                   detected
Audio_Moth_5_20250319_062544  HUMAN_PRESENCE_ON_TRAIL 35904.000     354.000           7      True       1.000        1.176          0.677                7         7                   detected
Audio_Moth_5_20250319_062544                  VEHICLE  8937.000     297.000          19      True       1.000        3.800          2.187               19        19                   detected
Audio_Moth_5_20250321_062545                 CHAINSAW  8109.000     297.000           1      True       1.000        0.200          0.101                1         1                   detected
Audio_Moth_5_20250321_062545                  GUNSHOT  8154.000     300.000           3      True       1.000        0.594          0.300                3         3                   detected
Audio_Moth_5_20250321_062545                  GUNSHOT 26271.000     297.000          20      True       1.000        4.000          2.017               21        20                   detected
Audio_Moth_5_20250321_062545 HUMAN_PRESENCE_OFF_TRAIL  8052.000     300.000           2      True       1.000        0.396          0.200                2         2                   detected
Audio_Moth_5_20250321_062545  HUMAN_PRESENCE_ON_TRAIL 25611.000    1008.000          83      True       1.000        4.926          2.484               84        83                   detected
Audio_Moth_5_20250321_062545                  VEHICLE 26460.000     159.000          19      True       0.995        7.037          3.549               20        19                   detected
```
