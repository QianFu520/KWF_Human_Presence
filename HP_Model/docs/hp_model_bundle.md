# HP model bundle format

Status: agreed with Coleman; his `hp-23`, `hp-25`, `hp-27` and `hp-29-20261101` bundles use it. `inference_pipeline/scripts/hp_inference.py` loads it for pipeline step 8 (`run_pipeline.py --bundle`) and, later, the HP service.

## Why one file

Step 8 and the HP service need four things that only work together: the model, the scaler fitted with it, the exact feature list and order, and the threshold chosen for it. If they ship as separate files, a model from one run can end up next to a scaler or threshold from another, and nothing would notice. One file keeps them together and lets the loader check them against each other.

This replaces rule 4 in the "Decision" section of `feature_pipeline_progress.md` (separate `hp_model.pkl` and `hp_scaler.pkl` files).

## Format

One file written with `joblib.dump`, containing a dict.

**Required keys**

| Key | Type | Meaning |
|---|---|---|
| `model` | fitted scikit-learn classifier | Has `predict_proba`; `classes_` is `[0, 1]`, where 1 = human presence. Coleman's bundles use `HistGradientBoostingClassifier` |
| `scaler` | fitted `StandardScaler` | Fitted on a DataFrame with exactly `features`, in that order, so `feature_names_in_` equals `features` |
| `features` | list of str | Exactly the `hp_features.py` list for `feature_set`: same names, same order |
| `feature_set` | str | One of `"23"`, `"25"`, `"27"`, `"29"` (a string, not a number) |
| `threshold` | float | A clip gets label 1 when its probability is ≥ `threshold` |
| `version` | str | `hp-<feature_set>-<YYYYMMDD>`, e.g. `hp-23-20261101` |
| `trained_on` | str | Data, split and label window |
| `created` | str | Date the bundle was written, `YYYY-MM-DD` |

**Optional keys**

| Key | Type | Meaning |
|---|---|---|
| `role` | str | `"candidate"` (meant for deployment) or `"reference"` (for comparison only). The loader warns, but still loads, if it is missing, so older bundles keep working |
| `sklearn_version` | str | scikit-learn version used for training. The loader warns if it differs from the installed one |
| `val_metrics` | dict | Validation metrics at the threshold: `precision`, `recall`, `fa_episodes_per_rec_day` |
| `threshold_rule` | str | How the threshold was chosen |
| `git_commit` | str | Commit of the training code |

**Feature sets**

| `feature_set` | List in `hp_features.py` | Weather | Hour of day | Pipeline run |
|---|---|---|---|---|
| `"23"` | `HP_FEATURES_23` | no | no | `--no_weather` |
| `"25"` | `HP_FEATURES_25` | no | yes | `--no_weather` |
| `"27"` | `HP_FEATURES_27` | yes | no | with weather |
| `"29"` | `HP_FEATURES_29` | yes | yes | with weather |

## Current bundles

- **`hp-23-20261101` is the integration candidate** (Qian, Nov 5; `role` `"candidate"`): 23 features, no hour and no weather. It is for integration testing only and is **not field-ready**.
- **`hp-25`, `hp-27` and `hp-29-20261101` are reference bundles** (`role` `"reference"`). They rely on time or weather shortcuts, so they are for comparison only, not for deployment. Step 8 logs a warning when one is used.
- **The threshold comes from the bundle.** Nothing in the pipeline hard-codes it, and it may be re-tuned during integration.
- **Bundles live in the team Drive folder, never in git.** Point the pipeline at a local copy: `--bundle <bundle_path>`.

## Rules

1. **Fit the scaler on a DataFrame in `hp_features` order.** Select the columns with `select_features(df, HP_FEATURES_23)` (or the list for your set) and fit on that DataFrame, so the scaler records `feature_names_in_`. A scaler fitted on a plain NumPy array has no column names, and the loader rejects it because the order can't be checked.
2. **Choose the threshold on the validation set only.** The test set is used once, for the final metrics, never for picking the threshold.
3. **Name versions `hp-<feature_set>-YYYYMMDD`,** using the training date. Add a letter (`hp-23-20261101b`) for a second bundle on the same day. Name the file after the version: `hp-23-20261101.joblib`.
4. **Train with the library versions the pipeline uses:** scikit-learn 1.8.0 and joblib 1.5.3 (`inference_pipeline/requirements.txt`), and record `sklearn_version`. A pickle from another scikit-learn version may not load, or may behave differently.
5. **Don't commit bundles to the repo.** Share them through the team Drive folder.

## How it is loaded

`load_bundle(<bundle_path>)` in `hp_inference.py` stops with a clear error (`BundleError`) unless:

1. All required keys are present (the error names any that are missing).
2. `feature_set` is `"23"`, `"25"`, `"27"` or `"29"`, and `features` equals the matching `hp_features.py` list exactly.
3. `check_scaler_alignment(scaler, features)` passes.
4. `model` has `predict_proba` and `[int(c) for c in model.classes_] == [0, 1]`.
5. `threshold` is a number between 0 and 1.

It logs the version, feature set, threshold and role, and warns (without failing) on a missing `role` or a different `sklearn_version`.

`predict(bundle, df)` selects `features` with `select_features`, scales them, takes `predict_proba(...)[:, 1]` as the probability and sets `label = 1` when probability ≥ `threshold`. It returns one row per clip: `clip_name` (with `.wav`, added if missing), `probability` and `label`, plus `not_meaningful_prob` and `predicted_label` from AED when the input has them.

- **Pipeline step 8:** `run_pipeline.py --bundle <bundle_path>` loads and checks the bundle before step 1 and writes `outputs/predictions.csv`. A 23- or 25-feature bundle needs `--no_weather`; a 27- or 29-feature bundle needs weather. Otherwise the run stops at once with a message saying which to use. A `"reference"` bundle still runs, but step 8 logs a warning that it is not for deployment. Without `--bundle`, step 8 logs a warning and is skipped.
- **HP service (planned):** loads the bundle once at startup from the `HP_BUNDLE` environment variable, with the same checks, and reports `version`, `feature_set`, `threshold` and `role` in `/health`.
