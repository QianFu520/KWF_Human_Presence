"""Check that each pruned BirdNET ONNX model is the right file and is paired
with its own 493-species label file.

Run from the repo root:
    python scripts/check_model_pairing.py
    python scripts/check_model_pairing.py --models-dir "D:/somewhere/models"

Needs: onnx, onnxruntime, numpy (same pins as src/bridge/requirements.txt).

What it proves:
  - file size is what the docs say
  - input is (N, 144000) float32, output is (N, 493)
  - the FP16 file really holds float16 weights, the FP32 file holds none
  - label file: right name, 493 unique lines, matches the mapping CSV
  - allow-list species are all inside the label set
  - the model loads and runs, and returns 493 finite values
What it does NOT prove: that label ORDER is right. Only a known-answer test
on real clips does that (see src/bridge/verify_bridge_parity.py).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

EXPECT = {
    "fp16": {"mb": 18.00, "has_fp16": True},
    "fp32": {"mb": 35.82, "has_fp16": False},
}
N_CLASSES = 493
WINDOW = 144_000
BANNED_COUNT = 6362

failures = 0


def report(ok: bool, text: str) -> None:
    global failures
    if not ok:
        failures += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")


def dims(t) -> list:
    return [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]


def check_variant(models_dir: Path, tag: str, allow: set[str]) -> None:
    stem = f"birdnet_{tag}_pruned493"
    model_path = models_dir / f"{stem}.onnx"
    labels_path = models_dir / f"{stem}.labels.txt"
    mapping_path = models_dir / f"{stem}.mapping.csv"
    print(f"\n== {stem} ==")

    if not model_path.is_file():
        report(False, f"model file missing: {model_path}")
        return
    report(labels_path.is_file(), f"own label file present: {labels_path.name}")
    if not labels_path.is_file():
        return

    mb = model_path.stat().st_size / (1024 * 1024)
    want = EXPECT[tag]["mb"]
    report(abs(mb - want) < 0.1, f"size {mb:.2f} MB (expected about {want:.2f})")

    m = onnx.load(str(model_path))
    inp, out = m.graph.input[0], m.graph.output[0]
    report(dims(inp)[-1] == WINDOW, f"input '{inp.name}' shape {dims(inp)} (last dim must be {WINDOW})")
    report(inp.type.tensor_type.elem_type == TensorProto.FLOAT, "input type is float32")
    report(dims(out)[-1] == N_CLASSES, f"output '{out.name}' shape {dims(out)} (last dim must be {N_CLASSES})")
    report(out.type.tensor_type.elem_type == TensorProto.FLOAT, "output type is float32")

    n16 = sum(t.data_type == TensorProto.FLOAT16 for t in m.graph.initializer)
    if EXPECT[tag]["has_fp16"]:
        report(n16 > 0, f"{n16} float16 weight tensors (must be > 0 for the FP16 file)")
    else:
        report(n16 == 0, f"{n16} float16 weight tensors (must be 0 for the FP32 file)")

    labels = labels_path.read_text(encoding="utf-8-sig").splitlines()
    labels = [x.strip() for x in labels if x.strip()]
    report(len(labels) == N_CLASSES, f"{len(labels)} labels (must be {N_CLASSES})")
    report(len(labels) != BANNED_COUNT and labels_path.name != "labels.txt", "not the banned labels.txt")
    report(len(set(labels)) == len(labels), "all labels unique")
    report(all("_" in x for x in labels), "every label looks like 'Scientific name_Common name'")

    if mapping_path.is_file():
        with mapping_path.open(encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        same = [r["species"] for r in rows] == labels
        idx = [int(r["new_index"]) for r in rows] == list(range(len(rows)))
        report(same and idx, "mapping CSV matches label file line by line, indices 0..492")
    else:
        report(False, f"mapping CSV missing: {mapping_path.name}")

    if allow:
        missing = allow - set(labels)
        report(not missing, f"allow-list: {len(allow) - len(missing)}/{len(allow)} species inside label set")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
    y = sess.run(None, {inp.name: np.zeros((1, WINDOW), dtype=np.float32)})[0]
    report(y.shape == (1, N_CLASSES) and bool(np.isfinite(y).all()),
           f"runs on silence: output shape {y.shape}, all values finite")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()
    models_dir = Path(args.models_dir)

    allow_path = models_dir / "geomodel_allowlist_218.txt"
    allow = set()
    if allow_path.is_file():
        allow = {x.strip() for x in allow_path.read_text(encoding="utf-8-sig").splitlines() if x.strip()}

    for tag in ("fp16", "fp32"):
        check_variant(models_dir, tag, allow)

    print("\n" + ("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
