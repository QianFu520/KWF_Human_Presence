#!/usr/bin/env python3
"""Build the AED sample-audio set used by validate_aed.py.

Two parts, both written as 3.0 s, 48 kHz, mono, 16-bit WAVs of exactly
144,000 samples (the same contract the KWF pipeline's splitter produces):

1. esc50/   Real labelled recordings from the public ESC-50 dataset
            (https://github.com/karolpiczak/ESC-50, CC BY-NC 3.0; the
            ESC-10 subset is CC BY 3.0). Each 5 s ESC-50 clip is resampled
            44.1 kHz -> 48 kHz and cut to the 3 s window with the most
            energy, so the sound event is inside what AED actually scores.
            Expected labels follow the project's definition of "meaningful"
            (LABELING_STRATEGY.md): birds, other animals and human activity
            are meaningful; rain, wind, water and storms are background.
            Insects/crickets are ambiguous (insect-only chorus counts as
            background, insects with other calls count as meaningful), so
            they are labelled "ambiguous" and excluded from metrics.

2. synthetic/  Generated signals with known structure (noise, tones, chirps,
               silence, a late tone burst, wrong-format files, a corrupt
               file). Used for contract and behaviour checks, not accuracy.

ESC-50 is NOT AudioMoth field audio: different microphones, places and
recording conditions. Metrics on it test whether the service behaves
sensibly on real sound, not deployment accuracy.

Usage:
    python make_sample_set.py --out sample_audio [--per_class 8] [--seed 0]
Writes sample_audio/manifest.csv (clip_name, path, expected_label, group,
source, note).
"""
import argparse
import io
import os
import urllib.request

import numpy as np
import pandas as pd
import soundfile as sf

SR = 48000
N = 144000
ESC_RAW = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"

# ESC-50 category -> (expected_label, group)
ESC_MAP = {
    # birds / other animals
    "chirping_birds": ("meaningful", "bird"),
    "crow": ("meaningful", "bird"),
    "frog": ("meaningful", "other_animal"),
    # human activity (the class the project exists to catch)
    "chainsaw": ("meaningful", "human_activity"),
    "hand_saw": ("meaningful", "human_activity"),
    "footsteps": ("meaningful", "human_activity"),
    "engine": ("meaningful", "human_activity"),
    "fireworks": ("meaningful", "human_activity"),  # closest ESC-50 class to a gunshot
    "coughing": ("meaningful", "human_activity"),
    "laughing": ("meaningful", "human_activity"),
    # background
    "rain": ("not_meaningful", "background_rain"),
    "wind": ("not_meaningful", "background_wind"),
    "sea_waves": ("not_meaningful", "background_water"),
    "thunderstorm": ("not_meaningful", "background_storm"),
    # ambiguous under the project definition
    "insects": ("ambiguous", "insects"),
    "crickets": ("ambiguous", "insects"),
}


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def best_window(y, sr, n):
    """Start index of the n-sample window with the highest energy (hop 0.1 s)."""
    if len(y) <= n:
        return 0
    e = np.convolve(y.astype(np.float64) ** 2, np.ones(n), mode="valid")
    hop = int(0.1 * sr)
    starts = np.arange(0, len(e), hop)
    return int(starts[np.argmax(e[starts])])


def to_contract(y, sr):
    """Resample to 48 kHz mono and return exactly N samples around the loudest 3 s."""
    import librosa

    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    s = best_window(y, SR, N)
    out = y[s:s + N]
    if len(out) < N:
        out = np.pad(out, (0, N - len(out)))
    return out, s / SR


def build_esc50(out_dir, per_class, seed):
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(io.BytesIO(fetch(f"{ESC_RAW}/meta/esc50.csv")))
    rng = np.random.RandomState(seed)
    rows = []
    for cat, (label, group) in ESC_MAP.items():
        sub = meta[meta["category"] == cat].sort_values("filename")
        pick = sub.iloc[rng.permutation(len(sub))[:per_class]]
        for _, r in pick.iterrows():
            raw = fetch(f"{ESC_RAW}/audio/{r['filename']}")
            y, sr = sf.read(io.BytesIO(raw))
            clip, start_s = to_contract(np.asarray(y), sr)
            name = f"esc50_{cat}_{r['filename']}"
            path = os.path.join(out_dir, name)
            sf.write(path, clip, SR, subtype="PCM_16")
            rows.append({
                "clip_name": name, "path": path, "expected_label": label,
                "group": group, "source": f"ESC-50 {r['filename']} ({cat}, esc10={r['esc10']})",
                "note": f"loudest 3 s window starts at {start_s:.1f} s of the 5 s original",
            })
    return rows


def build_synthetic(out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SR
    rows = []

    def w(name, y, label="", group="synthetic", note="", sr=SR, subtype="PCM_16"):
        path = os.path.join(out_dir, name)
        sf.write(path, np.asarray(y, dtype=np.float32), sr, subtype=subtype)
        rows.append({"clip_name": name, "path": path, "expected_label": label,
                     "group": group, "source": "synthetic", "note": note})

    chirp = np.zeros(N)
    for st in (0.3, 0.9, 1.6, 2.2):
        idx = (t >= st) & (t < st + 0.25)
        tt = t[idx] - st
        chirp[idx] = 0.3 * np.sin(2 * np.pi * (3000 + 8000 * tt) * tt) * np.hanning(idx.sum())
    w("syn_chirps_clean.wav", chirp, "meaningful", note="four 0.25 s up-chirps, no background")
    w("syn_chirps_in_noise.wav", chirp + 0.05 * rng.standard_normal(N), "meaningful",
      note="same chirps over white noise")
    w("syn_white_noise.wav", 0.1 * rng.standard_normal(N), "not_meaningful",
      note="broadband noise, flat spectrum (rain-like)")
    w("syn_tone_1khz.wav", 0.2 * np.sin(2 * np.pi * 1000 * t), "", note="steady tone; no expectation")
    w("syn_silence.wav", np.zeros(N), "", note="digital silence; no expectation")
    late = 0.05 * rng.standard_normal(N)
    idx = t >= 2.8
    late[idx] += 0.3 * np.sin(2 * np.pi * 5000 * t[idx])
    w("syn_late_burst_2p8s.wav", late, "", note="5 kHz burst starting at 2.8 s, inside the cropped tail")
    # format edge cases
    w("syn_short_1s_noise.wav", 0.1 * rng.standard_normal(SR), "", note="1 s long; gets zero-padded")
    w("syn_long_10s.wav", np.concatenate([chirp, 0.1 * rng.standard_normal(7 * SR)]), "",
      note="10 s long; only first 3 s scored")
    t2 = np.arange(3 * 44100) / 44100
    w("syn_stereo_44k.wav", np.stack([0.2 * np.sin(2 * np.pi * 2000 * t2)] * 2, axis=1), "",
      sr=44100, note="stereo, 44.1 kHz; tests resample + downmix")
    corrupt = os.path.join(out_dir, "syn_corrupt.wav")
    with open(corrupt, "wb") as f:
        f.write(b"RIFF not really audio")
    rows.append({"clip_name": "syn_corrupt.wav", "path": corrupt, "expected_label": "",
                 "group": "synthetic", "source": "synthetic", "note": "undecodable file"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sample_audio")
    ap.add_argument("--per_class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_esc50", action="store_true", help="synthetic only (no network)")
    args = ap.parse_args()

    rows = build_synthetic(os.path.join(args.out, "synthetic"), args.seed)
    if not args.no_esc50:
        rows += build_esc50(os.path.join(args.out, "esc50"), args.per_class, args.seed)
    man = pd.DataFrame(rows)
    man["path"] = man["path"].apply(lambda p: os.path.relpath(p, args.out))
    man.to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    print(man.fillna("").groupby(["expected_label", "group"]).size())
    print(f"\nWrote {len(man)} clips + manifest.csv to {args.out}")


if __name__ == "__main__":
    main()
