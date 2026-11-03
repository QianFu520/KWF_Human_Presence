"""Interface and preprocessing checks for the running API (FP16 model).

INTERFACE (what happens to input the API must not score)
  - real wrong-length clips you supply with --wrong-length-dir
  - stereo copies made from good clips
  - one mixed job (good + wrong length + stereo) to see what the caller gets back
  Gated rule C8: such files are never scored.
  Reported rule C9 (known issue V-001, not gated): the caller is told the file was rejected.

PREPROCESSING (what the Digital Twin does to audio before BirdNET)
  Twin code (audio_event_common.py): audio / (peak + 1e-8), then soundfile.write
  to a temporary 16-bit WAV. Two cases are measured against the unchanged clips:
    peak_norm   peak-normalized, written as float32 (isolates normalization)
    twin_chain  peak-normalized, then 16-bit PCM (the Twin's actual chain)
  A 16-bit-only case is not run: the source clips are already 16-bit PCM.
  These are reported numbers, not pass/fail: the repo has no baseline for them.

Writes in --out-dir: interface_and_preprocessing.json, interface_cases.csv,
preprocessing_per_clip.csv. Generated audio goes to a temp folder and is deleted.
Exit code 0 only if every gated criterion passes.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src" / "bridge"))

import httpx  # noqa: E402
import service_vs_reference as svr  # noqa: E402
import verify_bridge_parity as vbp  # noqa: E402

VARIANT = svr.VARIANT
WINDOW = 144_000
POLL_S, TIMEOUT_S = 0.3, 900.0


# ---------------------------------------------------------------- job helper
def submit(client: httpx.Client, paths: list) -> dict:
    """Run one job. Never raises on a bad job: records what the caller would see."""
    body = {"paths": [str(p) for p in paths], "variant": VARIANT, "min_confidence": 0.0,
            "apply_allow_list": True, "collect_detections": True}
    r = client.post("/jobs", json=body)
    if r.status_code >= 400:
        return {"submit_http_status": r.status_code, "status": "rejected_at_submit",
                "error": r.text[:200], "windows": 0, "detections": [], "completed": 0, "total": len(paths)}
    job_id = r.json()["job_id"]
    t0 = time.perf_counter()
    while True:
        st = client.get(f"/jobs/{job_id}").json()
        if st["status"] in ("done", "error", "cancelled"):
            break
        if time.perf_counter() - t0 > TIMEOUT_S:
            raise SystemExit(f"FATAL: job {job_id} did not finish in {TIMEOUT_S}s")
        time.sleep(POLL_S)
    m = st.get("metrics") or {}
    st["windows"] = m.get("windows", 0)
    return st


def dets_by_clip(state: dict) -> dict:
    out: dict = {}
    for d in state.get("detections") or []:
        out.setdefault(d["data_id"], []).append((d["species"], d["confidence"], d.get("in_allow_list", True)))
    return out


def classify(state: dict) -> tuple[str, bool]:
    """-> (outcome, caller_can_see_problem)."""
    if state["status"] == "rejected_at_submit":
        return "rejected_at_submit", True
    if state["status"] == "error" or state.get("error"):
        return "job_error", True
    if state["windows"] > 0:
        return "scored", False
    return "skipped_silently", False        # done, 0 windows, no error (V-001)


# ---------------------------------------------------------------- interface
def header(path: Path) -> dict:
    try:
        i = sf.info(str(path))
        return {"frames": i.frames, "samplerate": i.samplerate, "channels": i.channels,
                "subtype": i.subtype, "multiple_of_144000": i.frames % WINDOW == 0}
    except Exception as e:                                   # noqa: BLE001
        return {"header_error": repr(e)}


def run_interface(client, good: list, wrong_dir: Path | None, work: Path) -> tuple[list, dict]:
    rows = []

    def one(case: str, path: Path):
        st = submit(client, [path])
        outcome, visible = classify(st)
        rows.append({"case": case, "file": path.name, **header(path), "status": st["status"],
                     "completed": st.get("completed"), "windows_scored": st["windows"],
                     "outcome": outcome, "caller_can_see_problem": visible,
                     "error": st.get("error") or ""})

    wrong = sorted(wrong_dir.glob("*.wav")) if wrong_dir and wrong_dir.is_dir() else []
    for p in wrong:
        one("wrong_length_real", p)

    pick = sorted({0, len(good) // 4, len(good) // 2, 3 * len(good) // 4, len(good) - 1})
    stereo = []
    for i in pick:
        data, sr = sf.read(str(good[i]), dtype="float32")
        p = work / f"stereo_{good[i].name}"
        sf.write(str(p), np.stack([data, data], axis=1), sr, subtype="PCM_16")
        stereo.append(p)
        one("stereo_made", p)

    mixed = {}
    if wrong and stereo:
        st = submit(client, [good[0], wrong[0], stereo[0]])
        by = dets_by_clip(st)
        mixed = {"files": [good[0].name, wrong[0].name, stereo[0].name], "status": st["status"],
                 "completed": st.get("completed"), "total": st.get("total"),
                 "windows_scored": st["windows"], "good_clip_scored": good[0].name in by,
                 "error": st.get("error") or ""}
    return rows, mixed


# ------------------------------------------------------------ preprocessing
def make_case(case: str, src: list, dest: Path) -> list:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for p in src:
        x, sr = sf.read(str(p), dtype="float32")
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 0:
            x = (x / (peak + 1e-8)).astype(np.float32)          # Twin: audio / (peak + 1e-8)
        sub = "FLOAT" if case == "peak_norm" else "PCM_16"
        sf.write(str(dest / p.name), x, sr, subtype=sub)
        out.append(dest / p.name)
    return out


def summarize_case(case: str, names: list, base: dict, got: dict, gains: list) -> tuple[dict, list]:
    per, shifts = [], []
    top1_changed = det_changed = dropped = 0
    for n in names:
        b, g = svr.clip_view(base.get(n, [])), svr.clip_view(got.get(n, []))
        bsp = b["top1"]
        bconf = b["conf"].get(bsp)
        gconf = g["conf"].get(bsp)
        shift = abs(gconf - bconf) if (gconf is not None and bconf is not None) else None
        if shift is None:
            dropped += 1
        else:
            shifts.append(shift)
        top1_changed += g["top1"] != bsp
        det_changed += g["det_set"] != b["det_set"]
        per.append({"case": case, "clip": n, "top1_base": bsp, "top1_case": g["top1"],
                    "conf_base": bconf, "conf_case": gconf, "abs_shift": shift,
                    "det_set_changed": g["det_set"] != b["det_set"]})
    n = len(names)
    srt = sorted(shifts)
    s = {"case": case, "n_clips": n, "top1_changed": top1_changed,
         "top1_changed_pct": 100.0 * top1_changed / n,
         "det_set_changed": det_changed, "base_species_not_returned": dropped,
         "conf_shift_mean": statistics.fmean(shifts) if shifts else None,
         "conf_shift_p95": srt[int(0.95 * (len(srt) - 1))] if srt else None,
         "conf_shift_max": srt[-1] if srt else None,
         "within_0.01_pct": 100.0 * sum(x <= 0.01 for x in shifts) / n,
         "within_0.05_pct": 100.0 * sum(x <= 0.05 for x in shifts) / n}
    if gains:
        s["gain_min"], s["gain_median"], s["gain_max"] = min(gains), statistics.median(gains), max(gains)
    return s, per


def run_preprocessing(client, good: list, work: Path) -> tuple[list, list, dict]:
    names = [p.name for p in good]
    gains = []
    for p in good:
        x, _ = sf.read(str(p), dtype="float32")
        pk = float(np.max(np.abs(x))) if x.size else 0.0
        gains.append(1.0 / (pk + 1e-8) if pk > 0 else 1.0)
    base_state = submit(client, good)
    if base_state["status"] != "done":
        raise SystemExit(f"FATAL: baseline job ended as {base_state['status']}: {base_state.get('error')}")
    base = dets_by_clip(base_state)
    summaries, per_all = [], []
    for case in ("peak_norm", "twin_chain"):
        d = work / case
        paths = make_case(case, good, d)
        st = submit(client, paths)
        if st["status"] != "done":
            raise SystemExit(f"FATAL: {case} job ended as {st['status']}: {st.get('error')}")
        s, per = summarize_case(case, names, base, dets_by_clip(st), gains)
        summaries.append(s)
        per_all += per
        shutil.rmtree(d, ignore_errors=True)
    return summaries, per_all, {"gain_for_peak_norm": {"min": min(gains), "median": statistics.median(gains), "max": max(gains)}}


# --------------------------------------------------------------- criteria
def judge(rows: list, mixed: dict, criteria: dict, wrong_run: bool) -> list:
    meta = {c["id"]: c for c in criteria["criteria"] if c.get("layer") == "interface"}
    bad = [r for r in rows if r["outcome"] == "scored"]
    hidden = [r for r in rows if not r["caller_can_see_problem"]]
    verdicts = []
    if "C8" in meta:
        verdicts.append({"id": "C8", "gate": meta["C8"]["gate"], "pass": not bad and bool(rows),
                         "detail": f"{len(rows) - len(bad)}/{len(rows)} bad files not scored"
                                   + ("" if wrong_run else " (wrong-length set not supplied: stereo only)")})
    if "C9" in meta:
        verdicts.append({"id": "C9", "gate": meta["C9"]["gate"], "pass": not hidden and bool(rows),
                         "known_issue": meta["C9"].get("known_issue"),
                         "detail": f"{len(rows) - len(hidden)}/{len(rows)} rejections visible to the caller"})
    return verdicts


# ------------------------------------------------------------------- main
def run(client, args) -> int:
    criteria = json.loads(args.criteria.read_text(encoding="utf-8"))
    root = vbp.resolve_audio_root(args.audio_root)
    with open(vbp.DETAIL_CSV, encoding="utf-8", newline="") as f:
        names = [r["clip"] for r in csv.DictReader(f)]
    good = [Path(p) for p in vbp.resolve_paths(names, root)]
    if args.sample:
        good = good[::max(1, len(good) // args.sample)][:args.sample]

    work = Path(tempfile.mkdtemp(prefix="bn_val_", dir=args.work_dir))
    try:
        rows, mixed = run_interface(client, good, args.wrong_length_dir, work)
        summaries, per_all, extra = run_preprocessing(client, good, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    verdicts = judge(rows, mixed, criteria, bool(args.wrong_length_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "interface_and_preprocessing.json").write_text(json.dumps(
        {"variant": VARIANT, "interface": rows, "mixed_job": mixed, "preprocessing": summaries,
         **extra, "criteria": verdicts}, indent=2, default=str), encoding="utf-8")
    for fname, data in (("interface_cases.csv", rows), ("preprocessing_per_clip.csv", per_all)):
        if data:
            with open(args.out_dir / fname, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)

    print(f"Interface and preprocessing | {VARIANT} | {len(good)} good clips\n")
    print("INTERFACE (what the API does with input it should not score)")
    for r in rows:
        print(f"  {r['case']:<18}{r['file'][:36]:<38}frames={r.get('frames')} ch={r.get('channels')} "
              f"-> {r['outcome']}  (caller can see it: {r['caller_can_see_problem']})")
    if mixed:
        print(f"  mixed job (good + wrong length + stereo): status={mixed['status']} completed={mixed['completed']}/"
              f"{mixed['total']} windows_scored={mixed['windows_scored']} good_clip_scored={mixed['good_clip_scored']}")
    print("\nPREPROCESSING (Twin transforms vs unchanged clips; reported, not pass/fail)")
    for s in summaries:
        print(f"  {s['case']:<11} top-1 changed {s['top1_changed']}/{s['n_clips']}  det-set changed {s['det_set_changed']}  "
              f"conf shift mean {s['conf_shift_mean']:.4f} p95 {s['conf_shift_p95']:.4f} max {s['conf_shift_max']:.4f}  "
              f"within 0.01: {s['within_0.01_pct']:.1f}%  within 0.05: {s['within_0.05_pct']:.1f}%")
    g = extra["gain_for_peak_norm"]
    print(f"  gain applied by peak normalization: min {g['min']:.2f}  median {g['median']:.2f}  max {g['max']:.2f}")
    print("\nCriteria")
    for v in verdicts:
        tag = "PASS" if v["pass"] else ("FAIL" if v["gate"] else f"FAIL (known issue {v.get('known_issue')}, not gated)")
        print(f"  {v['id']}: {v['detail']} -> {tag}")
    failed = [v for v in verdicts if v["gate"] and not v["pass"]]
    print(f"\nOVERALL: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--audio-root", type=Path, default=None)
    ap.add_argument("--wrong-length-dir", type=Path, default=None,
                    help="folder with real wrong-length WAVs (e.g. a few from filtered_clips_6)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--criteria", type=Path, default=svr.CRITERIA_PATH)
    ap.add_argument("--sample", type=int, default=0, help="use N evenly spaced clips instead of all (faster)")
    ap.add_argument("--work-dir", type=Path, default=None, help="where temporary audio is written")
    a = ap.parse_args()
    with httpx.Client(base_url=a.host, timeout=30.0) as client:
        health = client.get("/health").json()
        if not health.get("ready") or VARIANT not in health.get("variants_loaded", []):
            raise SystemExit(f"FATAL: service not ready or {VARIANT} not loaded: {health}")
        return run(client, a)


if __name__ == "__main__":
    sys.exit(main())
