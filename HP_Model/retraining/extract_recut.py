"""Extract the AM2-AM6 re-cut zips into one folder per recorder.

Clips go to <out>/Audio_Moth_N/<clip name>. Clips that already exist are
skipped, so the three zips that repeat another zip's clips are deduped by
filename. Reports wrong-size entries and unique clip counts per recorder.

Usage:
  python extract_recut.py --src "C:\\BirdNET ONNX\\zips\\recut" --out "C:\\BirdNET ONNX\\clips"
"""

import argparse
import re
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

CLIP_BYTES = 288044
NAME_RE = re.compile(r"^(Audio_Moth_\d)_\d{8}_\d{6}\.wav$", re.IGNORECASE)


def replace_with_retry(tmp, dest, tries=20):
    # Antivirus/indexer can briefly lock a freshly written file on Windows.
    for i in range(tries):
        try:
            tmp.replace(dest)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.25)


def extract_zip(zip_path, out_root, bad):
    written = skipped = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name  # never trust paths inside the zip
            m = NAME_RE.match(name)
            if not m:
                bad.append((zip_path.name, info.filename, "unexpected name"))
                continue
            if info.file_size != CLIP_BYTES:
                bad.append((zip_path.name, info.filename, f"size {info.file_size}"))
                continue
            dest = out_root / m.group(1) / name
            if dest.exists():
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            with z.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            replace_with_retry(tmp, dest)
            written += 1
    print(f"{zip_path.name}: wrote {written:,}, skipped {skipped:,} existing", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="folder of re-cut zips")
    ap.add_argument("--out", required=True, type=Path, help="clips root folder")
    args = ap.parse_args()

    zips = sorted(args.src.glob("*.zip"))
    if not zips:
        sys.exit(f"No zips in {args.src}")

    bad = []
    for zp in zips:
        extract_zip(zp, args.out, bad)

    for rec, *_ in bad[:20]:
        print("BAD:", rec, *_)
    print(f"{len(bad)} bad entries")

    counts = Counter()
    for rec_dir in sorted(args.out.glob("Audio_Moth_[2-6]")):
        counts[rec_dir.name] = sum(1 for _ in rec_dir.glob("*.wav"))
    for rec, n in sorted(counts.items()):
        print(f"{rec}: {n:,} unique clips")
    print(f"Total: {sum(counts.values()):,}")


if __name__ == "__main__":
    main()
