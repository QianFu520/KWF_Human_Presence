"""
Reproduce TinyCNN v3's train/validation split, so field metrics can be reported
on clips the model never trained on.

The code below is notebooks/05_train.ipynb cell 1, applied to labels_progress.csv
as it stood when v3 was trained (the model_inference_v3 labels were produced by v3
afterwards, so they are dropped). The result is checked against the counts saved in
the v3 checkpoint (n_train 101,820, n_val 28,333) and refuses to continue if they
differ.

Caveat: v3 was picked over v4 on this validation split, so it is held out from
training but not from model selection.
"""
import numpy as np
import pandas as pd

V3_N_TRAIN, V3_N_VAL = 101820, 28333


def v3_split(labels: pd.DataFrame) -> pd.Series:
    """Return clip_name -> 'train' / 'val' for every clip v3 saw."""
    lab = labels[labels.meaningful_source != "model_inference_v3"]
    t = lab[lab["meaningful"] != "unknown"].copy()
    t["recorder_date"] = t["Recorder"] + "_" + t["clip_name"].str.extract(r"_(\d{8})_")[0]
    rng = np.random.RandomState(42)

    m = t[t["meaningful"] == "meaningful"].copy()
    # pandas 2.x unique() returned a numpy object array; keep that type so the shuffle matches
    groups = np.array(m["recorder_date"].unique(), dtype=object)
    rng.shuffle(groups)
    val_groups = set(groups[int(0.8 * len(groups)):])
    m["split"] = np.where(m["recorder_date"].isin(val_groups), "val", "train")

    n = t[t["meaningful"] == "not_meaningful"].copy()
    idx = rng.permutation(len(n))
    n["split"] = "val"
    n.iloc[idx[: int(0.8 * len(n))], n.columns.get_loc("split")] = "train"

    split = pd.concat([m, n]).set_index("clip_name")["split"]
    counts = split.value_counts()
    if counts.get("train") != V3_N_TRAIN or counts.get("val") != V3_N_VAL:
        raise RuntimeError(f"split does not match the v3 checkpoint: {counts.to_dict()} "
                           f"vs train {V3_N_TRAIN}, val {V3_N_VAL}")
    return split
