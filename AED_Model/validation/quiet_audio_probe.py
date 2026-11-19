"""
Why does AED (TinyCNN v3) score quiet human activity as background?

Computes simple acoustic descriptors for clip sets and compares them with the
model's score:

  offtrail_kept / offtrail_dropped  the quiet off-trail walk (Audio_Moth_3,
                                    19 March 08:01), split at 0.5
  ontrail       held-out on-trail walk clips (all kept)
  background    held-out rule-labelled background clips (all dropped)
  train_bg      clips v3 trained on as not_meaningful
  train_mean    clips v3 trained on as meaningful
  extra:<name>  any folder given with --extra name=folder (e.g. Ben's AM1 clips)

Descriptors (on the waveform and on the exact training mel, 128x256 dB re max):
  rms_dbfs        overall level
  peak_dbfs       loudest sample
  crest_db        peak / rms: how transient the clip is
  frame_contrast  loudest frame vs median frame energy (dB): an event that
                  stands out of its own background scores high
  mel_median_db   median mel cell relative to the clip's max (0 dB)
  flatness        spectral flatness (1 = noise-like, 0 = tonal)
  low/mid/high    share of energy < 1 kHz, 1-4 kHz, 4-16 kHz

Note: the mel is in dB relative to each clip's own maximum, so the model
cannot see absolute loudness (gain-invariant, check F2). Quietness can only
matter through how the clip's content stands out of its noise floor.

Usage (from AED_Model/, AED_AUDIO_DIR set):
    python validation/quiet_audio_probe.py --out validation/results/quiet_audio_<date> \
        [--extra am1_setup=/path/to/clips]
"""
import argparse
import glob
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
import config  # noqa: E402
import paths  # noqa: E402

SR = 48000
OFF_TRAIL = "Human Presence off Trail_Audio_Moth_3_20250319_080155"


def describe(path):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=3.0)
    y = np.pad(y, (0, max(0, SR * 3 - len(y))))
    rms = np.sqrt(np.mean(y ** 2)) + 1e-12
    peak = np.max(np.abs(y)) + 1e-12
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    tot = S.sum() + 1e-20
    frame_e = S.sum(axis=0) + 1e-20
    mel = cli.preprocess_clip(path)
    return {
        "rms_dbfs": 20 * np.log10(rms), "peak_dbfs": 20 * np.log10(peak),
        "crest_db": 20 * np.log10(peak / rms),
        "frame_contrast": 10 * np.log10(frame_e.max() / np.median(frame_e)),
        "mel_median_db": float(np.median(mel)),
        "flatness": float(np.mean(librosa.feature.spectral_flatness(S=np.sqrt(S)))),
        "low": S[f < 1000].sum() / tot, "mid": S[(f >= 1000) & (f < 4000)].sum() / tot,
        "high": S[(f >= 4000) & (f <= 16000)].sum() / tot,
    }, mel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150, help="clips sampled per reference set")
    ap.add_argument("--extra", action="append", default=[], help="name=folder of extra clips")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    man = pd.read_csv(os.path.join(HERE, "field/manifest_heldout.csv"), keep_default_na=False)
    sets = {"offtrail": man[man.event_id == OFF_TRAIL],
            "ontrail": man[man.group == "HUMAN_PRESENCE_ON_TRAIL"].sample(a.n, random_state=a.seed),
            "background": man[man.group == "BACKGROUND"].sample(a.n, random_state=a.seed)}
    rows = [(k, r.clip_name, paths.expand(r.path)) for k, d in sets.items() for r in d.itertuples()]

    lab = pd.read_csv(config.LABELS_PROGRESS_PATH, usecols=["clip_name", "audio_path", "meaningful", "meaningful_source"])
    split = pd.read_csv(os.path.join(HERE, "field/v3_split.csv"))
    tr = lab.merge(split[split.split == "train"], on="clip_name")
    seg = os.path.join(config.AUDIO_DIR, "")
    for k, cond in [("train_bg", tr.meaningful == "not_meaningful"), ("train_mean", tr.meaningful == "meaningful")]:
        d = tr[cond].sample(a.n, random_state=a.seed)
        rows += [(k, r.clip_name, seg + r.audio_path.split("Segmented_Foldered/")[-1].split("Segmented Foldered/")[-1])
                 for r in d.itertuples()]
    for spec in a.extra:
        name, folder = spec.split("=", 1)
        for p in sorted(glob.glob(os.path.join(folder, "**", "*.[wW][aA][vV]"), recursive=True)):
            rows.append((f"extra:{name}", os.path.basename(p), p))

    model, _ = cli.load_model(os.path.join(os.path.dirname(HERE), "outputs/models/tinycnn_v3.pth"),
                              device=torch.device("cpu"))
    out = []
    for k, name, p in rows:
        try:
            feats, mel = describe(p)
        except Exception as e:  # noqa: BLE001
            print("skip", name, repr(e)[:80])
            continue
        prob = float(cli.predict(model, torch.from_numpy(mel)[None, None], torch.device("cpu"))[0])
        out.append({"set": k, "clip_name": name, "p": prob, **feats})
    df = pd.DataFrame(out)
    df.loc[df.set == "offtrail", "set"] = np.where(df.loc[df.set == "offtrail", "p"] >= 0.5,
                                                    "offtrail_dropped", "offtrail_kept")
    df.to_csv(os.path.join(a.out, "clips.csv"), index=False)
    cols = ["p", "rms_dbfs", "peak_dbfs", "crest_db", "frame_contrast", "mel_median_db", "flatness", "low", "mid", "high"]
    summ = df.groupby("set")[cols].median().round(3)
    summ.insert(0, "n", df.groupby("set").size())
    summ.to_csv(os.path.join(a.out, "summary.csv"))
    corr = df[df.set.str.startswith("offtrail")][cols].corr(method="spearman")["p"].round(2)
    print(summ.to_string()); print("\nSpearman with p inside the off-trail walk:\n", corr.to_string())
    corr.to_csv(os.path.join(a.out, "offtrail_spearman.csv"))


if __name__ == "__main__":
    main()
