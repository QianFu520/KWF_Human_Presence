"""
add_bundle_role.py: one-off (Nov 13 2026). Add a "role" key to the four Nov 1 HP bundles and
change nothing else: "candidate" for hp-23, "reference" for hp-25, hp-27 and hp-29.

The originals are first copied to <bundles>\\pre_role\\ (never overwritten or deleted). Each new
bundle is rebuilt from that copy, saved under the same file name with joblib, then checked:
  * keys are the original keys plus "role"; every other value is unchanged (model and scaler
    compared by joblib.hash)
  * predict_proba on 1,000 validation rows (seed 42) is identical before and after, using the
    bundle's own features (w10 training set for 23/25, the weather training set for 27/29)

Usage (PowerShell):
  python add_bundle_role.py "C:\\BirdNET ONNX\\hp_audit\\bundles" `
      --data "C:\\BirdNET ONNX\\hp_audit\\training\\training_set_w10.csv" `
      --data-weather "C:\\BirdNET ONNX\\hp_audit\\training\\training_set_w10_weather.csv"
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROLES = {"hp-23-20261101": "candidate", "hp-25-20261101": "reference", "hp-27-20261101": "reference",
         "hp-29-20261101": "reference"}
USES_WEATHER = {"27", "29"}


def val_rows(path: Path, feats: list[str]) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=["split"] + feats)
    return d[d.split == "val"].sample(1000, random_state=42)[feats]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundles", type=Path)
    ap.add_argument("--data", required=True, type=Path, help="training set without weather (23, 25)")
    ap.add_argument("--data-weather", required=True, type=Path, help="training set with weather (27, 29)")
    args = ap.parse_args()
    assert joblib.__version__ == "1.5.3", f"joblib {joblib.__version__}; bundles are saved with 1.5.3"

    pre = args.bundles / "pre_role"
    pre.mkdir(exist_ok=True)
    for stem in ROLES:  # copy every original first, before any bundle is rewritten
        src, dst = args.bundles / f"{stem}.joblib", pre / f"{stem}.joblib"
        if dst.exists():
            assert "role" not in joblib.load(dst), f"{dst} already has a role key"
        else:
            assert "role" not in joblib.load(src), f"{src} already has a role key; nothing to do"
            shutil.copy2(src, dst)
            assert filecmp.cmp(src, dst, shallow=False), f"copy of {src} differs"
        print(f"original kept: {dst}")

    for stem, role in ROLES.items():
        old = joblib.load(pre / f"{stem}.joblib")
        joblib.dump({**old, "role": role}, args.bundles / f"{stem}.joblib")
        new = joblib.load(args.bundles / f"{stem}.joblib")

        assert list(new) == list(old) + ["role"] and new["role"] == role
        for k in old:
            same = joblib.hash(old[k]) == joblib.hash(new[k]) if k in ("model", "scaler") else old[k] == new[k]
            assert same, f"{stem}: {k} changed"
        feats = old["features"]
        x = val_rows(args.data_weather if old["feature_set"] in USES_WEATHER else args.data, feats)
        p_old = old["model"].predict_proba(old["scaler"].transform(x))[:, 1]
        p_new = new["model"].predict_proba(new["scaler"].transform(x))[:, 1]
        assert np.array_equal(p_old, p_new), f"{stem}: predict_proba differs"
        print(f"OK {stem}.joblib: role={role}; other {len(old)} keys unchanged; predict_proba identical on "
              f"{len(x)} val rows (max |diff| {np.abs(p_old - p_new).max():.1e}, mean score {p_new.mean():.4f})")


if __name__ == "__main__":
    sys.exit(main())
