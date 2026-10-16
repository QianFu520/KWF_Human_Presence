#!/usr/bin/env python3
"""AED service validation harness.

Checks that the deployed AED component (TinyCNN v3) is the right artifact,
preprocesses audio exactly as in training, serves a correct FastAPI contract,
and gives the same answer through every inference path. Optionally scores a
labelled manifest and checks the Digital Twin's vendored copy of AED.

Sections (each check is PASS / FAIL / WARN / INFO):
  A  Environment            versions, platform
  B  Model artifact          checksum, metadata, strict load, ONNX equivalence,
                             which file each consumer points at
  C  Preprocessing parity    training notebook code vs CLI vs API vs Digital Twin
  D  API contract            endpoints, request/response schema, errors,
                             determinism, latency
  E  Cross-path consistency  CLI vs API vs ONNX vs Digital Twin gate, per clip
  F  Behaviour probes        known edge behaviours, reported not failed
  G  Metrics                 confusion matrix / precision / recall per group
                             from a manifest with expected labels
  H  Digital Twin workflow   end-to-end run_node_audio_workflow on a long file,
                             gate outputs re-derived independently

Nothing here retrains or modifies the model.

Usage (from AED_Model/):
    python validation/validate_aed.py \
        --aed_root . \
        --clips validation/sample_audio \
        --manifest validation/sample_audio/manifest.csv \
        [--twin_backend /path/to/digital_twin_mvp/backend] \
        [--api_url http://localhost:8000]      # live server instead of in-process
        [--out validation/results]

Exit code 0 when no check FAILs, 1 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import glob
import hashlib
import importlib
import io
import json
import os
import platform
import re
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EXPECTED_V3_SHA256 = "913a78a468ff42e2fbd250c3d43ad658de79fd1ed1aa320c56c7ecba57f83cdb"
EXPECTED_PARAMS = 547_937
N_MELS, N_FRAMES, SR, N_SAMPLES = 128, 256, 48000, 144000
CONSISTENCY_TOL = 1e-4   # prob difference allowed between inference paths
MEL_TOL_DB = 1e-3        # dB difference allowed between preprocessing implementations
PEAK_EPS_DB = 1e-4       # float32 rounding in power_to_db can put the peak a few 1e-6 dB off 0
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------
class Checks:
    def __init__(self):
        self.rows = []

    def add(self, cid, section, name, status, detail=""):
        self.rows.append({"id": cid, "section": section, "check": name,
                          "status": status, "detail": str(detail)})
        mark = {"PASS": "ok ", "FAIL": "XX ", "WARN": "!! ", "INFO": "-- "}[status]
        print(f"  {mark}{cid:<5} {name}: {detail}")

    def guard(self, cid, section, name, fn):
        """Run fn() -> (status, detail); any exception is a FAIL with the error."""
        try:
            status, detail = fn()
        except Exception as e:  # noqa: BLE001
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        self.add(cid, section, name, status, detail)
        return status

    def df(self):
        return pd.DataFrame(self.rows)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cli_score_clips(cli, clips, model_path, threshold, output_csv):
    """aed_inference.score_folder() minus folder discovery, for an explicit clip list."""
    import torch
    from torch.utils.data import DataLoader

    model, checkpoint = cli.load_model(model_path, device=torch.device("cpu"))
    loader = DataLoader(cli.ClipFolderDataset(clips[["audio_path", "clip_name"]]), batch_size=64,
                        shuffle=False, num_workers=0, collate_fn=cli.collate_skip_none)
    rows = []
    for batch in loader:
        if batch is None:
            continue
        x, names, paths = batch
        for name, path, prob in zip(names, paths, cli.predict(model, x, torch.device("cpu"))):
            rows.append({"clip_name": name, "audio_path": path, "not_meaningful_prob": float(prob),
                         "predicted_label": "not_meaningful" if prob >= threshold else "meaningful"})
    df = pd.DataFrame(rows, columns=["clip_name", "audio_path", "not_meaningful_prob", "predicted_label"])
    df.to_csv(output_csv, index=False)
    return df


def find_clips(clips_dir, manifest):
    if manifest is not None:
        base = os.path.dirname(os.path.abspath(manifest.attrs["path"]))
        from paths import expand  # {AUDIO_DIR}/... -> config.AUDIO_DIR/...
        paths = [expand(p) for p in manifest["path"]]
        paths = [p if os.path.isabs(p) else os.path.join(base, p) for p in paths]
        return pd.DataFrame({"clip_name": manifest["clip_name"], "audio_path": paths})
    paths = sorted(p for p in glob.glob(os.path.join(clips_dir, "**", "*"), recursive=True)
                   if p.lower().endswith(AUDIO_EXT))
    return pd.DataFrame({"clip_name": [os.path.basename(p) for p in paths], "audio_path": paths})


# ---------------------------------------------------------------------------
# loaders for each implementation under test
# ---------------------------------------------------------------------------
def training_mel_fn(notebook_path):
    """Rebuild compute_mel exactly as written in a training/labelling notebook."""
    import librosa

    nb = json.load(open(notebook_path))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and "def compute_mel" in src:
            tree = ast.parse(src)
            keep = [n for n in tree.body
                    if (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets))
                    or (isinstance(n, ast.FunctionDef) and n.name == "compute_mel")]
            ns = {"librosa": librosa, "np": np}
            exec(compile(ast.Module(body=keep, type_ignores=[]), notebook_path, "exec"), ns)
            return ns["compute_mel"]
    raise RuntimeError(f"no compute_mel in {notebook_path}")


def import_upstream(aed_root, model_path, threshold):
    """Import aed_inference (CLI module) and api.main exactly as shipped."""
    aed_root = os.path.abspath(aed_root)
    if aed_root not in sys.path:
        sys.path.insert(0, aed_root)
    os.environ["MODEL_PATH"] = os.path.abspath(model_path)
    os.environ["THRESHOLD"] = str(threshold)
    cli = importlib.import_module("aed_inference")
    api = importlib.import_module("api.main")
    return cli, api


def import_twin(twin_backend):
    twin_backend = os.path.abspath(twin_backend)
    if twin_backend not in sys.path:
        sys.path.insert(0, twin_backend)
    inf = importlib.import_module("app.services.aed.inference")
    gate = importlib.import_module("app.services.aed.gate")
    common = importlib.import_module("app.audio_workflow.audio_event_common")
    return inf, gate, common


def twin_clip(common, path):
    """The Digital Twin's own clip loader, for a 3 s file centred at 1.5 s."""
    clip, sr, _, _ = common.extract_centered_clip_from_file(path, 1.5, clip_s=3.0, target_sr=SR)
    return clip, sr


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aed_root", default=".", help="AED_Model folder")
    ap.add_argument("--model", default=None, help="checkpoint (default <aed_root>/outputs/models/tinycnn_v3.pth)")
    ap.add_argument("--onnx", default=None, help="ONNX export (default <aed_root>/outputs/models/tinycnn_v3.onnx)")
    ap.add_argument("--clips", default=None, help="folder of clips (ignored when --manifest is given)")
    ap.add_argument("--manifest", default=None, help="CSV: clip_name,path,expected_label[,group]")
    ap.add_argument("--threshold", type=float, default=0.5, help="not_meaningful_prob cutoff (upstream default 0.5)")
    ap.add_argument("--twin_backend", default=None, help="digital_twin_mvp/backend folder (enables sections C/E twin + H)")
    ap.add_argument("--twin_threshold", type=float, default=0.3,
                    help="Digital Twin gate threshold on confidence=1-p (twin default 0.3)")
    ap.add_argument("--api_url", default=None, help="test a running server instead of in-process")
    ap.add_argument("--out", default="validation/results")
    ap.add_argument("--skip_workflow", action="store_true", help="skip section H")
    args = ap.parse_args()

    aed_root = os.path.abspath(args.aed_root)
    model_path = os.path.abspath(args.model or os.path.join(aed_root, "outputs/models/tinycnn_v3.pth"))
    onnx_path = os.path.abspath(args.onnx or os.path.join(aed_root, "outputs/models/tinycnn_v3.onnx"))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(os.path.join(args.out, stamp))
    os.makedirs(out_dir, exist_ok=True)

    manifest = None
    if args.manifest:
        manifest = pd.read_csv(args.manifest, keep_default_na=False)
        manifest.attrs["path"] = args.manifest
    args._manifest = manifest
    clips = find_clips(args.clips, manifest)

    C = Checks()
    import torch
    import librosa
    import soundfile as sf

    # ------------------------------------------------------------------ A
    print("\n[A] Environment")
    env = {"timestamp": stamp, "python": sys.version.split()[0], "platform": platform.platform(),
           "machine": platform.machine(), "torch": torch.__version__, "librosa": librosa.__version__,
           "numpy": np.__version__, "soundfile": sf.__version__}
    for mod in ("onnxruntime", "fastapi", "pandas"):
        try:
            env[mod] = importlib.import_module(mod).__version__
        except Exception:
            env[mod] = "not installed"
    pinned = {"torch": "2.2.2", "librosa": "0.11.0", "numpy": "1.26.4"}
    off = {k: env[k] for k, v in pinned.items() if not env[k].startswith(v)}
    C.add("A1", "Environment", "Library versions", "INFO", ", ".join(f"{k} {v}" for k, v in env.items() if k not in ("timestamp", "platform")))
    C.add("A2", "Environment", "Matches AED_Model/requirements.txt pins",
          "PASS" if not off else "WARN", "all pinned versions" if not off else f"differs: {off} (pins {pinned})")

    # ------------------------------------------------------------------ B
    print("\n[B] Model artifact")
    ck = torch.load(model_path, map_location="cpu", weights_only=True)
    sha = sha256(model_path)
    C.add("B1", "Artifact", "Checkpoint SHA-256",
          "PASS" if sha == EXPECTED_V3_SHA256 else "FAIL", f"{os.path.basename(model_path)} {sha[:16]}…")
    C.add("B2", "Artifact", "Checkpoint metadata says v3",
          "PASS" if "v3" in str(ck.get("notes", "")) else "FAIL",
          f"notes='{ck.get('notes')}', val_acc={ck.get('val_acc')}, nm_precision={ck.get('not_meaningful_precision')}, "
          f"nm_recall={ck.get('not_meaningful_recall')}, n_train={ck.get('n_train')}, n_val={ck.get('n_val')}")

    cli, api = import_upstream(aed_root, model_path, args.threshold)
    model = cli.TinyCNN()
    missing = model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    C.add("B3", "Artifact", "Strict load into TinyCNN + parameter count",
          "PASS" if n_params == EXPECTED_PARAMS else "FAIL", f"{n_params:,} parameters, strict load ok ({missing})")

    def b4():
        txt = open(os.path.join(aed_root, "api", "main.py")).read()
        m = re.search(r'MODEL_PATH",\s*"([^"]+)"', txt)
        default = m.group(1) if m else "?"
        pipe = os.path.join(os.path.dirname(aed_root), "inference_pipeline", "run_pipeline.py")
        pipe_ref = re.findall(r"tinycnn_v\d+\.pth", open(pipe).read()) if os.path.exists(pipe) else []
        ok = default.endswith("tinycnn_v3.pth") and all(r == "tinycnn_v3.pth" for r in pipe_ref)
        return ("PASS" if ok else "FAIL", f"API default MODEL_PATH='{default}'; run_pipeline.py refs={sorted(set(pipe_ref))}")
    C.guard("B4", "Artifact", "Consumers point at v3", b4)

    def b5():
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        i, o = sess.get_inputs()[0], sess.get_outputs()[0]
        sig = f"in {i.name}{i.shape} -> out {o.name}{o.shape}"
        ok_sig = i.name == "mel_spectrogram" and list(i.shape[1:]) == [1, N_MELS, N_FRAMES] and o.name == "not_meaningful_prob"
        rng = np.random.default_rng(0)
        x = (rng.uniform(-80, 0, size=(32, 1, N_MELS, N_FRAMES))).astype(np.float32)
        with torch.no_grad():
            pt = 1 - torch.sigmoid(model(torch.from_numpy(x)).squeeze(1)).numpy()
        ox = sess.run(None, {i.name: x})[0]
        d = float(np.max(np.abs(pt - ox)))
        return ("PASS" if ok_sig and d <= 1e-5 else "FAIL", f"{sig}; max |pt-onnx| on 32 random inputs = {d:.2e}")
    C.guard("B5", "Artifact", "ONNX export = same model", b5)

    twin = None
    if args.twin_backend:
        def b6():
            tw = os.path.join(os.path.abspath(args.twin_backend), "app/services/aed/models/tinycnn_v3.pth")
            s = sha256(tw)
            return ("PASS" if s == sha else "FAIL", f"Digital Twin bundled checkpoint {s[:16]}… "
                    f"{'identical to' if s == sha else 'DIFFERS from'} upstream")
        C.guard("B6", "Artifact", "Digital Twin bundles the same checkpoint", b6)
        try:
            twin = import_twin(args.twin_backend)
            os.environ.pop("AED_MODEL_PATH", None)
            C.add("B7", "Artifact", "Digital Twin checkpoint resolution", "PASS"
                  if os.path.basename(twin[0].resolve_checkpoint_path() or "") == "tinycnn_v3.pth" else "FAIL",
                  f"resolve_checkpoint_path() -> {twin[0].resolve_checkpoint_path()}")
        except Exception as e:  # noqa: BLE001
            C.add("B7", "Artifact", "Digital Twin AED import", "FAIL", f"{type(e).__name__}: {e}")
            twin = None

    # ------------------------------------------------------------------ C
    print("\n[C] Preprocessing parity")
    nb_dir = os.path.join(aed_root, "notebooks")
    train_fn = training_mel_fn(os.path.join(nb_dir, "05_train.ipynb"))
    label_fn = training_mel_fn(os.path.join(nb_dir, "06_inference_labeling.ipynb"))
    impls = {"train_05": lambda p: train_fn(p), "labeling_06": lambda p: label_fn(p),
             "cli": lambda p: cli.preprocess_clip(p),
             "api": lambda p: api.compute_mel(open(p, "rb").read()).numpy()[0, 0]}
    if twin:
        impls["twin"] = lambda p: twin[0].clip_to_mel(*twin_clip(twin[2], p))

    per_clip = []
    mels_cache = {}
    max_diff = {k: 0.0 for k in impls if k != "train_05"}
    contract_bad, unreadable, tail_peak, native_sr = [], [], [], {}
    for _, r in clips.iterrows():
        p = r["audio_path"]
        row = {"clip_name": r["clip_name"]}
        try:
            ref = impls["train_05"](p)
        except Exception as e:  # noqa: BLE001
            unreadable.append(r["clip_name"])
            row["preprocess_error"] = f"{type(e).__name__}"
            per_clip.append(row)
            continue
        mels_cache[r["clip_name"]] = ref
        if ref.shape != (N_MELS, N_FRAMES) or ref.dtype != np.float32 or not np.isfinite(ref).all() or ref.max() > PEAK_EPS_DB:
            contract_bad.append(r["clip_name"])
        if ref.max() < -PEAK_EPS_DB:  # loudest frame was in the 26 frames cropped off the end
            tail_peak.append(r["clip_name"])
        try:
            native_sr[r["clip_name"]] = sf.info(p).samplerate
        except Exception:  # noqa: BLE001
            native_sr[r["clip_name"]] = None
        for k, fn in impls.items():
            if k == "train_05":
                continue
            try:
                m = fn(p)
                d = float(np.max(np.abs(m - ref))) if m.shape == ref.shape else float("inf")
            except Exception as e:  # noqa: BLE001
                d = float("inf")
                row[f"{k}_error"] = f"{type(e).__name__}: {e}"
            row[f"mel_diff_{k}_vs_train"] = d
            max_diff[k] = max(max_diff[k], d)
        per_clip.append(row)
    n_ok = len(mels_cache)
    C.add("C1", "Preprocessing", "Mel contract (128x256, float32, finite, peak 0 dB)",
          "PASS" if not contract_bad else "FAIL",
          f"{n_ok} readable clips; violations: {contract_bad or 'none'}; unreadable: {unreadable or 'none'}")
    C.add("C1b", "Preprocessing", "Loudest moment falls in the cropped tail (last 0.27 s)", "INFO",
          f"{len(tail_peak)}/{n_ok} clips: {tail_peak}")
    for k, d in max_diff.items():
        if k == "twin":
            continue
        cid = {"labeling_06": "C2", "cli": "C3", "api": "C4", "twin": "C5"}[k]
        name = {"labeling_06": "Labelling notebook (06) = training (05)",
                "cli": "CLI preprocess_clip = training",
                "api": "FastAPI compute_mel = training",
                "twin": "Digital Twin clip loader + clip_to_mel = training"}[k]
        st = "PASS" if d == 0 else ("PASS" if d <= MEL_TOL_DB else "FAIL")
        exact = "bit-identical" if d == 0 else f"max diff {d:.2e} dB (tol {MEL_TOL_DB})"
        C.add(cid, "Preprocessing", name, st, f"{exact} over {n_ok} clips")
    if "twin" in impls:
        pcd = pd.DataFrame(per_clip)
        pcd["sr"] = pcd["clip_name"].map(native_sr)
        col = "mel_diff_twin_vs_train"
        at48 = pcd[(pcd["sr"] == SR) & pcd[col].notna()]
        other = pcd[(pcd["sr"] != SR) & pcd[col].notna()]
        d48 = float(at48[col].max()) if len(at48) else float("nan")
        C.add("C5", "Preprocessing", "Digital Twin clip loader + clip_to_mel = training (48 kHz input)",
              "PASS" if d48 <= MEL_TOL_DB else "FAIL",
              f"max diff {d48:.2e} dB over {len(at48)} clips (twin peak-normalises audio; dB is relative to clip max)")
        if len(other):
            C.add("C6", "Preprocessing", "Digital Twin = training for non-48 kHz input",
                  "PASS" if other[col].max() <= MEL_TOL_DB else "WARN",
                  "; ".join(f"{r.clip_name} ({int(r.sr)} Hz): max diff {getattr(r, col):.2f} dB" for r in other.itertuples())
                  + ". Twin resamples with scipy resample_poly, training with librosa (soxr); see E4 for the effect on scores")

    # ------------------------------------------------------------------ D
    print("\n[D] API contract")
    if args.api_url:
        import httpx
        client = httpx.Client(base_url=args.api_url, timeout=60)
        api_desc = f"live server {args.api_url}"
        ctx = None
    else:
        from fastapi.testclient import TestClient
        ctx = TestClient(api.app)
        client = ctx.__enter__()
        api_desc = "in-process (FastAPI TestClient, same app object uvicorn serves)"
    C.add("D0", "API", "Target", "INFO", api_desc)

    def post(path, name=None):
        name = name or os.path.basename(path)
        with open(path, "rb") as f:
            return client.post("/predict", files={"file": (name, f, "audio/wav")})

    def d1():
        r = client.get("/health")
        return ("PASS" if r.status_code == 200 and r.json() == {"status": "ok"} else "FAIL", f"{r.status_code} {r.json()}")
    C.guard("D1", "API", "GET /health", d1)

    def d2():
        spec = client.get("/openapi.json").json()
        paths = sorted(spec["paths"])
        pr = spec["paths"]["/predict"]["post"]
        body = json.dumps(pr.get("requestBody", {}))
        resp = spec["components"]["schemas"].get("PredictResponse", {}).get("properties", {})
        ok = paths == ["/health", "/predict"] and "multipart/form-data" in body and set(resp) == {"label", "not_meaningful_prob"}
        return ("PASS" if ok else "FAIL", f"paths={paths}; request multipart/form-data with 'file'; response fields={sorted(resp)}")
    C.guard("D2", "API", "OpenAPI schema", d2)

    api_scores, api_lat, schema_bad = {}, [], []
    for _, r in clips.iterrows():
        if r["clip_name"] not in mels_cache:
            continue
        name = r["clip_name"] if r["clip_name"].lower().endswith(".wav") else r["clip_name"] + ".wav"
        t0 = time.perf_counter()
        resp = post(r["audio_path"], name)
        api_lat.append((time.perf_counter() - t0) * 1000)
        js = resp.json()
        ok = (resp.status_code == 200 and set(js) == {"label", "not_meaningful_prob"}
              and js["label"] in ("meaningful", "not_meaningful") and 0.0 <= js["not_meaningful_prob"] <= 1.0
              and round(js["not_meaningful_prob"], 4) == js["not_meaningful_prob"]
              and (js["label"] == "not_meaningful") == (js["not_meaningful_prob"] > args.threshold))
        if not ok:
            schema_bad.append((r["clip_name"], resp.status_code, js))
        api_scores[r["clip_name"]] = js
    C.add("D3", "API", "POST /predict on every readable clip",
          "PASS" if not schema_bad and api_scores else "FAIL",
          f"{len(api_scores)} requests, 200 + valid schema + label consistent with threshold "
          f"{args.threshold}; problems: {schema_bad[:3] or 'none'}")

    def d4():
        any_clip = clips[clips.clip_name.isin(mels_cache)].iloc[0]["audio_path"]
        cases = {
            "non-.wav filename": (client.post("/predict", files={"file": ("x.flac", b"abc", "audio/flac")}), 400),
            "corrupt .wav": (client.post("/predict", files={"file": ("c.wav", b"RIFF junk", "audio/wav")}), 422),
            "empty .wav": (client.post("/predict", files={"file": ("e.wav", b"", "audio/wav")}), 422),
            "no file field": (client.post("/predict"), 422),
            "upper-case .WAV": (post(any_clip, "CLIP.WAV"), 200),
        }
        bad = {k: r.status_code for k, (r, want) in cases.items() if r.status_code != want}
        return ("PASS" if not bad else "FAIL",
                "; ".join(f"{k} -> {r.status_code}" for k, (r, _) in cases.items()))
    C.guard("D4", "API", "Error handling", d4)

    def d5():
        any_clip = clips[clips.clip_name.isin(mels_cache)].iloc[0]["audio_path"]
        outs = {json.dumps(post(any_clip).json(), sort_keys=True) for _ in range(5)}
        return ("PASS" if len(outs) == 1 else "FAIL", f"5 identical requests -> {len(outs)} distinct response(s)")
    C.guard("D5", "API", "Determinism", d5)

    lat = np.array(api_lat)
    C.add("D6", "API", "Latency per request (this machine)", "INFO",
          f"p50 {np.percentile(lat, 50):.1f} ms, p95 {np.percentile(lat, 95):.1f} ms, max {lat.max():.1f} ms over {len(lat)} requests"
          if len(lat) else "n/a")
    if ctx is not None:
        ctx.__exit__(None, None, None)

    # ------------------------------------------------------------------ E
    print("\n[E] Cross-path consistency")
    readable = clips[clips.clip_name.isin(mels_cache)].reset_index(drop=True)
    tmp_csv = os.path.join(out_dir, "cli_predictions.csv")
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        if args.manifest:
            # Score exactly the manifest's clips. score_folder() only scans a folder, and
            # manifest clips can live anywhere (e.g. on Drive), so run the same steps it
            # runs after discovery: ClipFolderDataset -> collate_skip_none -> predict.
            cli_df = cli_score_clips(cli, readable, model_path, args.threshold, tmp_csv)
        else:
            cli_df = cli.score_folder(folder_path=args.clips, model_path=model_path, output_csv=tmp_csv,
                                      threshold=args.threshold, device=torch.device("cpu"))
    cli_by_path = dict(zip(map(os.path.abspath, cli_df["audio_path"]), cli_df["not_meaningful_prob"]))

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    gate = None
    if twin:
        gate = twin[1].AEDMeaningfulGate(threshold=args.twin_threshold)

    rows_by_name = {r["clip_name"]: r for r in per_clip}
    worst = {"api": 0.0, "onnx": 0.0, "twin": 0.0}
    label_mismatch = []
    compared = 0
    for _, r in readable.iterrows():
        n, p = r["clip_name"], os.path.abspath(r["audio_path"])
        row = rows_by_name[n]
        mel = mels_cache[n]
        x = mel[None, None]
        row["p_cli"] = float(cli_by_path.get(p, np.nan))
        row["p_api"] = float(api_scores[n]["not_meaningful_prob"])
        row["label_api"] = api_scores[n]["label"]
        row["p_onnx"] = float(sess.run(None, {"mel_spectrogram": x})[0][0])
        if gate is not None:
            clip_audio, csr = twin_clip(twin[2], p)
            g = gate.confirm(clip_audio, csr, n, 1.5)
            row["p_twin"] = float(g.raw["not_meaningful_prob"])
            row["twin_confidence"] = g.confidence
            row["twin_pass"] = bool(g.is_birdcall)
            row["twin_label"] = g.label
        row["label_cli"] = "not_meaningful" if row["p_cli"] >= args.threshold else "meaningful"
        if np.isnan(row["p_cli"]):  # no CLI score: E1 fails; never count it as agreement
            continue
        compared += 1
        for k in ("api", "onnx", "twin"):
            if f"p_{k}" in row:
                worst[k] = max(worst[k], abs(row[f"p_{k}"] - row["p_cli"]))
        if row["label_cli"] != row["label_api"]:
            label_mismatch.append(n)
    C.add("E1", "Consistency", "CLI scored every readable clip",
          "PASS" if readable.audio_path.map(lambda q: os.path.abspath(q) in cli_by_path).all() else "FAIL",
          f"{len(cli_df)} rows in CLI CSV for {len(readable)} readable clips")
    C.add("E2", "Consistency", "API vs CLI probability", "PASS" if compared and worst["api"] <= CONSISTENCY_TOL else "FAIL",
          f"max |diff| {worst['api']:.2e} over {compared} clips (API rounds to 4 dp; tol {CONSISTENCY_TOL})")
    C.add("E3", "Consistency", "ONNX vs CLI probability", "PASS" if compared and worst["onnx"] <= CONSISTENCY_TOL else "FAIL",
          f"max |diff| {worst['onnx']:.2e} over {compared} clips")
    if gate is not None:
        C.add("E4", "Consistency", "Digital Twin gate vs CLI probability",
              "PASS" if compared and worst["twin"] <= CONSISTENCY_TOL else "FAIL",
              f"max |diff| {worst['twin']:.2e} over {compared} clips")
    C.add("E5", "Consistency", "CLI vs API labels at the same threshold",
          "PASS" if not label_mismatch else "WARN",
          f"{len(label_mismatch)} disagreements {label_mismatch[:5]} (CLI uses >=, API uses >)")

    # ------------------------------------------------------------------ F
    print("\n[F] Behaviour probes")
    t = np.arange(N_SAMPLES) / SR
    rng = np.random.default_rng(1)
    base = 0.05 * rng.standard_normal(N_SAMPLES)

    def score_wave(y):
        buf = io.BytesIO()
        sf.write(buf, np.asarray(y, dtype=np.float32), SR, format="WAV")
        m = api.compute_mel(buf.getvalue())
        with torch.no_grad():
            return float(1 - torch.sigmoid(model(m).squeeze(1)).item())

    onsets = [0.5, 1.5, 2.5, 2.6, 2.7, 2.8]
    tail = {}
    for st in onsets:
        y = base.copy()
        idx = (t >= st) & (t < st + 0.2)
        y[idx] += 0.3 * np.sin(2 * np.pi * 5000 * t[idx])
        tail[st] = score_wave(y)
    noise_only = score_wave(base)
    C.add("F1", "Probes", "End-of-clip blind spot (0.2 s 5 kHz burst in noise, by onset)",
          "WARN" if tail[2.8] > args.threshold and noise_only > args.threshold else "INFO",
          "; ".join(f"{k:.1f}s→{v:.3f}" for k, v in tail.items()) + f"; noise alone→{noise_only:.3f}. "
          f"Frames kept cover {N_FRAMES * 512 / SR:.3f} s of 3.0 s")
    gains = {g: score_wave(base * g / 0.05) for g in (0.001, 0.05, 0.5)}
    C.add("F2", "Probes", "Gain invariance", "PASS" if max(gains.values()) - min(gains.values()) < 1e-3 else "WARN",
          "; ".join(f"amp {g}→{v:.4f}" for g, v in gains.items()))
    full = 0.1 * rng.standard_normal(N_SAMPLES)
    pad = np.concatenate([full[:SR], np.zeros(N_SAMPLES - SR)])
    C.add("F3", "Probes", "Zero-padding changes the decision", "INFO",
          f"3 s noise → {score_wave(full):.3f}; first 1 s of same noise, zero-padded → {score_wave(pad):.3f}")
    C.add("F4", "Probes", "Digital silence", "INFO", f"{score_wave(np.zeros(N_SAMPLES)):.3f}")

    # ------------------------------------------------------------------ G
    metrics = {}
    if manifest is not None and "expected_label" in manifest:
        print("\n[G] Metrics against expected labels")
        pc = pd.DataFrame(per_clip)
        lab = manifest[["clip_name", "expected_label"] + [c for c in ("group", "source") if c in manifest]]
        pc = pc.merge(lab, on="clip_name", how="left")
        ev = pc[pc["expected_label"].isin(["meaningful", "not_meaningful"]) & pc["p_cli"].notna()].copy()

        def summarise(df, thr, col="p_cli"):
            pred_nm = df[col] >= thr
            true_nm = df["expected_label"] == "not_meaningful"
            tp, fp = int((pred_nm & true_nm).sum()), int((pred_nm & ~true_nm).sum())
            fn, tn = int((~pred_nm & true_nm).sum()), int((~pred_nm & ~true_nm).sum())
            div = lambda a, b: (a / b) if b else float("nan")  # noqa: E731
            return {"threshold": thr, "n": len(df), "tn_meaningful_kept": tn, "fp_meaningful_dropped": fp,
                    "fn_background_kept": fn, "tp_background_dropped": tp,
                    "accuracy": div(tp + tn, len(df)),
                    "not_meaningful_precision": div(tp, tp + fp), "not_meaningful_recall": div(tp, tp + fn),
                    "meaningful_recall": div(tn, tn + fp)}

        metrics["upstream_threshold"] = summarise(ev, args.threshold)
        metrics["twin_effective_threshold"] = summarise(ev, 1 - args.twin_threshold)
        metrics["sweep"] = [summarise(ev, th) for th in (0.3, 0.5, 0.7, 0.9, 0.95, 0.99)]
        grp = []
        for (g, exp), d in ev.groupby(["group", "expected_label"]):
            grp.append({"group": g, "expected": exp, "n": len(d),
                        "correct_at_threshold": int(((d["p_cli"] >= args.threshold) == (d["expected_label"] == "not_meaningful")).sum()),
                        "median_p": float(d["p_cli"].median()), "min_p": float(d["p_cli"].min()), "max_p": float(d["p_cli"].max())})
        metrics["by_group"] = grp
        amb = pc[pc["expected_label"] == "ambiguous"]
        if len(amb):
            metrics["ambiguous"] = {"n": len(amb), "predicted_not_meaningful": int((amb["p_cli"] >= args.threshold).sum())}
        u = metrics["upstream_threshold"]
        C.add("G1", "Metrics", f"Confusion at threshold {args.threshold} ({u['n']} labelled clips)", "INFO",
              f"meaningful kept {u['tn_meaningful_kept']}, meaningful dropped {u['fp_meaningful_dropped']}, "
              f"background kept {u['fn_background_kept']}, background dropped {u['tp_background_dropped']}; "
              f"accuracy {u['accuracy']:.3f}, meaningful recall {u['meaningful_recall']:.3f}, "
              f"not_meaningful precision {u['not_meaningful_precision']:.3f}, recall {u['not_meaningful_recall']:.3f}")
        ha = ev[ev.get("group", pd.Series(dtype=str)) == "human_activity"] if "group" in ev else ev.iloc[0:0]
        if len(ha):
            dropped = ha[ha["p_cli"] >= args.threshold]["clip_name"].tolist()
            C.add("G2", "Metrics", "Human-activity clips kept", "PASS" if not dropped else "WARN",
                  f"{len(ha) - len(dropped)}/{len(ha)} kept; dropped: {dropped or 'none'}")
        for g in grp:
            C.add("G3", "Metrics", f"group {g['group']}", "INFO",
                  f"{g['correct_at_threshold']}/{g['n']} as expected ({g['expected']}); median p {g['median_p']:.3f}")
        if "p_twin" in ev:
            tw = metrics["twin_effective_threshold"]
            C.add("G4", "Metrics", f"Same clips through the Digital Twin gate (confidence >= {args.twin_threshold})", "INFO",
                  f"meaningful kept {tw['tn_meaningful_kept']}/{tw['tn_meaningful_kept'] + tw['fp_meaningful_dropped']}, "
                  f"background dropped {tw['tp_background_dropped']}/{tw['tp_background_dropped'] + tw['fn_background_kept']}")

    # ------------------------------------------------------------------ H
    if twin and not args.skip_workflow:
        print("\n[H] Digital Twin workflow end-to-end")
        C.guard("H1", "Twin workflow", "run_node_audio_workflow on a long file",
                lambda: twin_workflow(twin, args, readable, out_dir, model, api, metrics))

    # ------------------------------------------------------------------ write
    checks = C.df()
    checks.to_csv(os.path.join(out_dir, "checks.csv"), index=False)
    pd.DataFrame(per_clip).to_csv(os.path.join(out_dir, "per_clip.csv"), index=False)
    json.dump({"environment": env, "model": {"path": model_path, "sha256": sha,
               "metadata": {k: v for k, v in ck.items() if k != "model_state_dict"}},
               "metrics": metrics}, open(os.path.join(out_dir, "summary.json"), "w"), indent=2, default=str)
    write_report(out_dir, checks, env, metrics, args)
    n_fail = int((checks.status == "FAIL").sum())
    n_warn = int((checks.status == "WARN").sum())
    verdict = "PASS" if n_fail == 0 else "FAIL"
    print(f"\nRESULT: {verdict} — {int((checks.status == 'PASS').sum())} pass, {n_fail} fail, {n_warn} warn, "
          f"{int((checks.status == 'INFO').sum())} info")
    print(f"Report: {os.path.join(out_dir, 'report.md')}")
    sys.exit(0 if n_fail == 0 else 1)


def twin_workflow(twin, args, readable, out_dir, model, api, metrics):
    """Concatenate labelled clips into one long recording, run the twin's own
    workflow on it, and re-derive each stage-2 score independently."""
    import soundfile as sf
    import torch

    wf = importlib.import_module("app.audio_workflow.node_audio_workflow")
    gap = np.zeros(SR * 2, dtype=np.float32)  # 2 s of silence between clips
    pieces, spans, cursor = [], [], 0.0
    for _, r in readable.iterrows():
        y, sr = sf.read(r["audio_path"], dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR or len(y) != N_SAMPLES:
            continue
        pieces += [y, gap]
        spans.append((r["clip_name"], cursor, cursor + 3.0))
        cursor += 5.0
    long_path = os.path.join(out_dir, "twin_long_input.wav")
    sf.write(long_path, np.concatenate(pieces), SR, subtype="PCM_16")

    wf_out = os.path.join(out_dir, "twin_workflow")
    res = wf.run_node_audio_workflow(input_audio=long_path, out_dir=wf_out, node_id="validation",
                                     run_id="aed_validation", tinycnn_threshold=args.twin_threshold,
                                     skip_birdnet=True)
    tl = res.get("stage_timelines") or {}
    gate_events = tl.get("stage_2_tinycnn_birdcall") or []
    n_trig = len(tl.get("stage_1_event_candidates") or [])

    # independent re-derivation: read the same window with soundfile, upstream preprocessing
    worst, statuses, rows = 0.0, set(), []
    gate_events = [e.to_dict() if hasattr(e, "to_dict") else e for e in gate_events]
    for ev in gate_events:
        g = ev.get("shaman_ii") or {}
        statuses.add(g.get("model_status"))
        start = float(ev["clip_start_s"])
        y, _ = sf.read(long_path, start=int(start * SR), frames=N_SAMPLES, dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, y, SR, format="WAV")
        with torch.no_grad():
            p_ref = float(1 - torch.sigmoid(model(api.compute_mel(buf.getvalue())).squeeze(1)).item())
        p_tw = float((g.get("raw") or {}).get("not_meaningful_prob", np.nan))
        worst = max(worst, abs(p_ref - p_tw))
        centre = float(ev["trigger_time_s"])
        src = next((n for n, a, b in spans if a - 1.0 <= centre <= b + 1.0), None)
        rows.append({"trigger_s": centre, "source_clip": src, "p_twin": p_tw, "p_reference": p_ref,
                     "confidence": g.get("confidence"), "passed": ev.get("event_type") == "birdcall_stage2_confirmed"})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "twin_workflow_events.csv"), index=False)
    reached = {r["source_clip"] for r in rows if r["source_clip"]}
    passed = {r["source_clip"] for r in rows if r["source_clip"] and r["passed"]}
    man = getattr(args, "_manifest", None)
    by_group = []
    if man is not None and "group" in man:
        g_of = dict(zip(man["clip_name"], man["group"]))
        e_of = dict(zip(man["clip_name"], man["expected_label"]))
        tab = pd.DataFrame({"clip_name": [n for n, _, _ in spans]})
        tab["group"] = tab["clip_name"].map(g_of)
        tab["expected"] = tab["clip_name"].map(e_of)
        tab["reached_aed"] = tab["clip_name"].isin(reached)
        tab["passed_aed"] = tab["clip_name"].isin(passed)
        tab.to_csv(os.path.join(out_dir, "twin_workflow_clips.csv"), index=False)
        for (g, e), d in tab.groupby(["group", "expected"]):
            by_group.append({"group": g, "expected": e, "n": len(d), "reached_aed": int(d.reached_aed.sum()),
                             "passed_aed": int(d.passed_aed.sum())})
        metrics["twin_workflow_by_group"] = by_group
    metrics["twin_workflow"] = {"clips_in_file": len(spans), "stage1_triggers": n_trig,
                                "stage2_scored": len(gate_events), "clips_reaching_aed": len(reached),
                                "max_abs_diff_vs_reference": worst, "model_status": sorted(map(str, statuses))}
    ok = statuses == {"aed_tinycnn"} and worst <= CONSISTENCY_TOL and len(gate_events) > 0
    return ("PASS" if ok else "FAIL",
            f"{len(spans)} clips in a {cursor:.0f} s file; stage 1 raised {n_trig} triggers; AED scored {len(gate_events)} "
            f"(model_status {sorted(map(str, statuses))}); {len(reached)}/{len(spans)} source clips reached AED; "
            f"max |twin - independent re-score| {worst:.2e}")


def write_report(out_dir, checks, env, metrics, args):
    lines = [f"# AED validation report — {env['timestamp']}", ""]
    n = checks.status.value_counts().to_dict()
    verdict = "PASS" if n.get("FAIL", 0) == 0 else "FAIL"
    lines += [f"**Result: {verdict}** — {n.get('PASS', 0)} pass, {n.get('FAIL', 0)} fail, "
              f"{n.get('WARN', 0)} warn, {n.get('INFO', 0)} info.", "",
              f"Command: `{' '.join(sys.argv)}`", "",
              f"Environment: Python {env['python']}, torch {env['torch']}, librosa {env['librosa']}, "
              f"numpy {env['numpy']}, {env['platform']}", "", "| ID | Check | Status | Detail |", "| --- | --- | --- | --- |"]
    for _, r in checks.iterrows():
        lines.append(f"| {r['id']} | {r['check']} | {r['status']} | {r['detail'].replace('|', '/')} |")
    if metrics.get("sweep"):
        lines += ["", "## Threshold sweep (labelled clips, CLI scores)", "",
                  "| threshold | meaningful kept | meaningful dropped | background dropped | background kept | accuracy |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for s in metrics["sweep"]:
            lines.append(f"| {s['threshold']} | {s['tn_meaningful_kept']} | {s['fp_meaningful_dropped']} | "
                         f"{s['tp_background_dropped']} | {s['fn_background_kept']} | {s['accuracy']:.3f} |")
    if metrics.get("twin_workflow_by_group"):
        lines += ["", "## Digital Twin workflow: which clips reached AED", "",
                  "Stage 1 (the twin's bird-like prefilter) runs before AED; clips it does not trigger on are never scored.", "",
                  "| group | expected | clips | reached AED (stage 1 triggered) | passed AED gate |", "| --- | --- | --- | --- | --- |"]
        for g in metrics["twin_workflow_by_group"]:
            lines.append(f"| {g['group']} | {g['expected']} | {g['n']} | {g['reached_aed']} | {g['passed_aed']} |")
    if metrics.get("by_group"):
        lines += ["", "## By group", "", "| group | expected | n | as expected | median p | min p | max p |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for g in metrics["by_group"]:
            lines.append(f"| {g['group']} | {g['expected']} | {g['n']} | {g['correct_at_threshold']} | "
                         f"{g['median_p']:.3f} | {g['min_p']:.3f} | {g['max_p']:.3f} |")
    lines += ["", "Files: `checks.csv` (all checks), `per_clip.csv` (every clip, every path), "
              "`summary.json` (environment, model metadata, metrics), `cli_predictions.csv`."]
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
