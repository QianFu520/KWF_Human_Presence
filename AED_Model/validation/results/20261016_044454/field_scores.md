# Field scores

Run: `validation/results/20261016_044454`  
Scores from: `p_api`  
Clips in manifest: 4027; scored: 4027; errors: 0

## API vs model
- clips compared: 4027
- max |p_api - p_model|: 4.97e-05 (API rounds to 4 decimals)
- label disagreements at 0.5: 0

## Results by group
clip_correct = share of clips given the expected label; event_recall = share of events with at least one clip kept as meaningful.

```
                   group       expected  clips  clip_correct@0.5  clip_correct@0.7  clip_correct@0.95  events  event_recall@0.5  event_recall@0.7  event_recall@0.95
              BACKGROUND not_meaningful    586             1.000             0.998              0.998     NaN               NaN               NaN                NaN
         BIRDNET_SPECIES     meaningful    500             1.000             1.000              1.000     NaN               NaN               NaN                NaN
                CHAINSAW     meaningful    198             1.000             1.000              1.000     2.0               1.0               1.0                1.0
                 GUNSHOT     meaningful    198             1.000             1.000              1.000     2.0               1.0               1.0                1.0
HUMAN_PRESENCE_OFF_TRAIL     meaningful     99             0.657             0.737              0.848     1.0               1.0               1.0                1.0
 HUMAN_PRESENCE_ON_TRAIL     meaningful   2248             1.000             1.000              1.000     8.0               1.0               1.0                1.0
                 VEHICLE     meaningful    198             1.000             1.000              1.000     2.0               1.0               1.0                1.0
```

## Misses at 0.5 (meaningful clips scored not_meaningful), lowest-confidence first

```
                       clip_name                    group      p
Audio_Moth_3_20250319_080316.wav HUMAN_PRESENCE_OFF_TRAIL 0.9998
Audio_Moth_3_20250319_080534.wav HUMAN_PRESENCE_OFF_TRAIL 0.9997
Audio_Moth_3_20250319_080319.wav HUMAN_PRESENCE_OFF_TRAIL 0.9991
Audio_Moth_3_20250319_080313.wav HUMAN_PRESENCE_OFF_TRAIL 0.9990
Audio_Moth_3_20250319_080422.wav HUMAN_PRESENCE_OFF_TRAIL 0.9987
Audio_Moth_3_20250319_080410.wav HUMAN_PRESENCE_OFF_TRAIL 0.9953
Audio_Moth_3_20250319_080646.wav HUMAN_PRESENCE_OFF_TRAIL 0.9945
Audio_Moth_3_20250319_080634.wav HUMAN_PRESENCE_OFF_TRAIL 0.9924
Audio_Moth_3_20250319_080322.wav HUMAN_PRESENCE_OFF_TRAIL 0.9904
Audio_Moth_3_20250319_080310.wav HUMAN_PRESENCE_OFF_TRAIL 0.9807
Audio_Moth_3_20250319_080622.wav HUMAN_PRESENCE_OFF_TRAIL 0.9767
Audio_Moth_3_20250319_080301.wav HUMAN_PRESENCE_OFF_TRAIL 0.9766
Audio_Moth_3_20250319_080631.wav HUMAN_PRESENCE_OFF_TRAIL 0.9738
Audio_Moth_3_20250319_080404.wav HUMAN_PRESENCE_OFF_TRAIL 0.9660
Audio_Moth_3_20250319_080637.wav HUMAN_PRESENCE_OFF_TRAIL 0.9635
Audio_Moth_3_20250319_080537.wav HUMAN_PRESENCE_OFF_TRAIL 0.9444
Audio_Moth_3_20250319_080304.wav HUMAN_PRESENCE_OFF_TRAIL 0.9391
Audio_Moth_3_20250319_080640.wav HUMAN_PRESENCE_OFF_TRAIL 0.9187
Audio_Moth_3_20250319_080455.wav HUMAN_PRESENCE_OFF_TRAIL 0.8967
Audio_Moth_3_20250319_080531.wav HUMAN_PRESENCE_OFF_TRAIL 0.8532
```
