"""Cut raw AudioMoth 1 recordings into 3 s clips matching the AM2-AM6 re-cut.

Each clip: 48 kHz, mono, 16-bit, 144,000 samples (288,044 bytes),
named Audio_Moth_1_YYYYMMDD_HHMMSS.wav. Clip times step by 3 s from the
recording's start time (first 15 characters of the raw filename, which is
local Costa Rica time). The final partial clip is dropped.

Usage:
  python cut_am1.py --src "C:\\BirdNET ONNX\\raw\\AM1" --out "C:\\BirdNET ONNX\\clips\\Audio_Moth_1" --dry-run
"""

import argparse
import sys
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path

RATE = 48000
CHANNELS = 1
SAMPWIDTH = 2
CLIP_SECONDS = 3
CLIP_FRAMES = RATE * CLIP_SECONDS
PREFIX = "Audio_Moth_1_"


def start_time(path):
    # Drive appends "-001"/"-002"; the time is the first 15 characters.
    return datetime.strptime(path.stem[:15], "%Y%m%d_%H%M%S")


def check_format(path, w):
    fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth())
    if fmt != (RATE, CHANNELS, SAMPWIDTH):
        sys.exit(f"{path.name}: expected 48000 Hz mono 16-bit, got {fmt[0]} Hz, "
                 f"{fmt[1]} ch, {fmt[2] * 8}-bit. Refusing to cut.")


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


def cut_file(path, out_dir, dry_run):
    t0 = start_time(path)
    with wave.open(str(path), "rb") as w:
        check_format(path, w)
        n_frames = w.getnframes()
        n_clips = n_frames // CLIP_FRAMES
        hours = n_frames / RATE / 3600
        t_end = t0 + timedelta(seconds=n_frames / RATE)
        print(f"{path.name}: {hours:.2f} h, {t0:%m-%d %H:%M:%S} -> {t_end:%m-%d %H:%M:%S}, "
              f"{n_clips:,} clips")
        if dry_run:
            return n_clips

        written = skipped = 0
        for i in range(n_clips):
            frames = w.readframes(CLIP_FRAMES)
            name = f"{PREFIX}{t0 + timedelta(seconds=i * CLIP_SECONDS):%Y%m%d_%H%M%S}.wav"
            dest = out_dir / name
            if dest.exists():
                skipped += 1
                continue
            tmp = dest.with_suffix(".tmp")
            with wave.open(str(tmp), "wb") as c:
                c.setnchannels(CHANNELS)
                c.setsampwidth(SAMPWIDTH)
                c.setframerate(RATE)
                c.writeframes(frames)
            replace_with_retry(tmp, dest)
            written += 1
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1:,}/{n_clips:,}", flush=True)
        print(f"  wrote {written:,}, skipped {skipped:,} existing")
    return n_clips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="folder of raw AM1 WAVs")
    ap.add_argument("--out", required=True, type=Path, help="output clip folder")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    files = sorted((p for p in args.src.iterdir() if p.is_file() and p.suffix.lower() == ".wav"),
                   key=start_time)
    if not files:
        sys.exit(f"No WAV files in {args.src}")
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)

    total = sum(cut_file(p, args.out, args.dry_run) for p in files)
    print(f"{len(files)} files, {total:,} clips{' (dry run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
