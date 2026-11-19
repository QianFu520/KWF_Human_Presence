"""
Does AED's score depend on how far the sound stands above the noise floor?

Controlled test on real field clips: each held-out on-trail clip (kept by AED)
is attenuated by 0 to -40 dB and mixed into a held-out background clip from
the same recorder set, then scored. The mel is relative to each clip's max,
so plain attenuation alone changes nothing (gain invariance, check F2); only
the ratio to the background noise matters. Also reports level, flatness and
AED rejection per recorder on the held-out clips.

Usage (from AED_Model/, AED_AUDIO_DIR set):
    python validation/quiet_snr_probe.py --out validation/results/quiet_audio_<date>
"""
import argparse
import os
import sys

import librosa
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import aed_inference as cli  # noqa: E402
import paths  # noqa: E402

SR, N = 48000, 144000
GAINS_DB = [0, -6, -12, -18, -24, -30, -36, -42]


def mel_of(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128, n_fft=1024, hop_length=512, fmin=50, fmax=16000)
    return librosa.power_to_db(m, ref=np.max)[:, :256].astype(np.float32)


def load(p):
    y, _ = librosa.load(p, sr=SR, mono=True, duration=3.0)
    return np.pad(y, (0, max(0, N - len(y))))[:N]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cpu")
    model, _ = cli.load_model(os.path.join(os.path.dirname(HERE), "outputs/models/tinycnn_v3.pth"), device=dev)

    def score(y):
        return float(cli.predict(model, torch.from_numpy(mel_of(y))[None, None], dev)[0])

    man = pd.read_csv(os.path.join(HERE, "field/manifest_heldout.csv"), keep_default_na=False)
    rng = np.random.default_rng(a.seed)
    sig = man[man.group == "HUMAN_PRESENCE_ON_TRAIL"].sample(a.n, random_state=a.seed)
    bg = man[man.group == "BACKGROUND"].sample(a.n, random_state=a.seed + 1)

    rows = []
    for s, b in zip(sig.itertuples(), bg.itertuples()):
        try:
            ys, yb = load(paths.expand(s.path)), load(paths.expand(b.path))
        except Exception as e:  # noqa: BLE001
            print("skip", s.clip_name, repr(e)[:60])
            continue
        rs, rb = np.sqrt(np.mean(ys ** 2)) + 1e-12, np.sqrt(np.mean(yb ** 2)) + 1e-12
        for g in GAINS_DB:
            yy = ys * 10 ** (g / 20)
            snr = 20 * np.log10(np.sqrt(np.mean(yy ** 2)) / rb)
            rows.append({"signal": s.clip_name, "background": b.clip_name, "gain_db": g, "snr_db": snr,
                         "p_mix": score(yy + yb), "p_attenuated_only": score(yy)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "snr_mix.csv"), index=False)
    t = df.groupby("gain_db").agg(pairs=("p_mix", "size"), median_snr_db=("snr_db", "median"),
                                  dropped_at_05=("p_mix", lambda x: (x >= 0.5).mean()),
                                  median_p=("p_mix", "median"),
                                  attenuated_only_dropped=("p_attenuated_only", lambda x: (x >= 0.5).mean()))
    t = t.sort_index(ascending=False).round(3)
    t.to_csv(os.path.join(a.out, "snr_mix_summary.csv"))
    print(t.to_string())

    # per recorder, held-out meaningful + background clips that are readable
    per = []
    sub = pd.concat([d.sample(min(len(d), 60), random_state=a.seed) for _, d in man.groupby("recorder")])
    for r in sub.itertuples():
        try:
            y = load(paths.expand(r.path))
        except Exception:  # noqa: BLE001
            continue
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
        per.append({"recorder": r.recorder, "group": r.group,
                    "rms_dbfs": 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12),
                    "flatness": float(np.mean(librosa.feature.spectral_flatness(S=S))), "p": score(y)})
    pr = pd.DataFrame(per)
    pr.to_csv(os.path.join(a.out, "per_recorder_clips.csv"), index=False)
    s = pr[pr.group != "BACKGROUND"].groupby("recorder").agg(
        clips=("p", "size"), median_rms_dbfs=("rms_dbfs", "median"), median_flatness=("flatness", "median"),
        meaningful_dropped=("p", lambda x: (x >= 0.5).mean())).round(3)
    s.to_csv(os.path.join(a.out, "per_recorder_summary.csv"))
    print(s.to_string())


if __name__ == "__main__":
    main()
