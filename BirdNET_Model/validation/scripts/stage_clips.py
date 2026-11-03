"""Copy the clips named in a CSV into one flat folder.

verify_bridge_parity.py expects a flat folder (it does not search subfolders).
Only --source is scanned, so point it at one recorder folder, not the whole set.
"""
import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--source", type=Path, required=True, help="folder holding the clips (searched recursively)")
ap.add_argument("--csv", type=Path, required=True, help="CSV that names the clips")
ap.add_argument("--column", default="clip", help="CSV column with clip file names")
ap.add_argument("--out", type=Path, required=True, help="flat output folder")
ap.add_argument("--manifest", type=Path, default=None,
                help="optional CSV (clip,bytes,sha256); staged files are checked against it")
a = ap.parse_args()

with open(a.csv, encoding="utf-8", newline="") as f:
    need = [r[a.column] for r in csv.DictReader(f)]

index = {}
for p in a.source.rglob("*.wav"):
    index.setdefault(p.name, p)

a.out.mkdir(parents=True, exist_ok=True)
missing = [n for n in need if n not in index]
for n in need:
    if n in index and not (a.out / n).exists():
        shutil.copy2(index[n], a.out / n)

print(f"requested {len(need)}, found {len(need) - len(missing)}, missing {len(missing)}")
if missing:
    print("first missing:", missing[:5])
    sys.exit(1)

if a.manifest:
    bad = []
    with open(a.manifest, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            p = a.out / r["clip"]
            ok = p.is_file() and p.stat().st_size == int(r["bytes"]) \
                and hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"]
            if not ok:
                bad.append(r["clip"])
    print(f"manifest check: {len(bad)} file(s) differ from {a.manifest.name}")
    if bad:
        print("first differing:", bad[:5])
        sys.exit(1)
