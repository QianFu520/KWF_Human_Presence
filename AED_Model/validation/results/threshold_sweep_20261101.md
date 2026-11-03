# Threshold sweep

Run: `validation/results/20261016_044454` (4027 labelled clips). Manifest: `validation/field/manifest_heldout.csv`.
Rule: `not_meaningful` if `not_meaningful_prob >= threshold`. Cells: share of clips given the expected label (kept for meaningful groups, dropped for background).

## Totals

```
 threshold meaningful kept background dropped
      0.50     3407 / 3441          586 / 586
      0.70     3415 / 3441          585 / 586
      0.80     3421 / 3441          585 / 586
      0.90     3423 / 3441          585 / 586
      0.95     3426 / 3441          585 / 586
      0.99     3432 / 3441          585 / 586
```

## By group

```
                   group       expected  clips  events    0.5    0.7    0.8    0.9   0.95   0.99
              BACKGROUND not_meaningful    586       0 100.0%  99.8%  99.8%  99.8%  99.8%  99.8%
         BIRDNET_SPECIES     meaningful    500       0 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
                CHAINSAW     meaningful    198       2 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
                 GUNSHOT     meaningful    198       2 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
HUMAN_PRESENCE_OFF_TRAIL     meaningful     99       1  65.7%  73.7%  79.8%  81.8%  84.8%  90.9%
 HUMAN_PRESENCE_ON_TRAIL     meaningful   2248       8 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
                 VEHICLE     meaningful    198       2 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
```

## By staged event (15 events; share of clips kept)

Events whose result changes with the threshold: 1 of 15.

```
                                                event  clips    0.5    0.7    0.8    0.9   0.95   0.99
                Chainsaw_Audio_Moth_2_20250319_171452     99 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
                Chainsaw_Audio_Moth_5_20250319_163504     99 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
                 Gunshot_Audio_Moth_3_20250319_170931     99 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
                 Gunshot_Audio_Moth_4_20250318_162640     99 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
Human Presence off Trail_Audio_Moth_3_20250319_080155     99  65.7%  73.7%  79.8%  81.8%  84.8%  90.9%
 Human Presence on Trail_Audio_Moth_2_20250318_165019    399 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
 Human Presence on Trail_Audio_Moth_2_20250319_074131    300 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
 Human Presence on Trail_Audio_Moth_2_20250319_170352    301 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
 Human Presence on Trail_Audio_Moth_3_20250319_165843    300 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
 Human Presence on Trail_Audio_Moth_5_20250318_160431    399 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
 Human Presence on Trail_Audio_Moth_5_20250319_084347    301 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
 Human Presence on Trail_Audio_Moth_5_20250319_162408    118 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
 Human Presence on Trail_Audio_Moth_5_20250319_163237    130 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
                 Vehicle_Audio_Moth_2_20250319_075243     99 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
                 Vehicle_Audio_Moth_5_20250319_085441     99 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
```
