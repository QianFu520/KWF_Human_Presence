# Validation run 20261022_072418 (UTC)

Criteria status: PROPOSED. Not yet approved by the team lead. Thresholds were written down before the live run and are taken from numbers already recorded in this repo (Phase 4, verification_data/). Nothing here was tuned to the live result.

| ID | Layer | Gate | Result | Detail |
|---|---|---|---|---|
| C0 | artifact_identity | yes | PASS | model bytes 18878769 (want 18878769); labels sha256_lf 72b910ab9eae575b (want 72b910ab9eae575b) |
| C1 | identity | yes | PASS | FP16 section: 14 checks, 0 failed (FP32 section not judged) |
| C2 | service_vs_recorded_fp16 | yes | PASS | exit code 0 |
| C3 | service_vs_reference | yes | PASS | 100.000 >= 99.8 |
| C4 | service_vs_reference | yes | PASS | 100.000 >= 99.9 |
| C5 | service_vs_reference | no | below threshold (not gated) | 97.179 >= 98.6 |
| C6 | service_vs_reference | yes | PASS | 100.000 >= 100.0 |
| C8 | interface | yes | PASS | 10/10 bad files not scored |
| C9 | interface | no | FAIL (known issue V-001, not gated) | 0/10 rejections visible to the caller |

**Overall: PASS**

See run_info.json for model/label hashes, versions and git commit.
