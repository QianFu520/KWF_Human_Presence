"""Score the running service (FastAPI) against the ORIGINAL BirdNET output.

Ground truth is the original BirdNET v2.4 (birdnet package v0.1.7) result
recorded in verification_data/phase4_detail_Audio_Moth_6.csv (ref_top1,
ref_conf), plus the recorded FP16 detection sets. This answers
"did compression change what the model says?". It is not accuracy against
human labels.

Reuses src/bridge/verify_bridge_parity.py for audio-root checks and the job
runner. Does not modify it.

    python validation/scripts/service_vs_reference.py \
        --audio-root C:/path/to/am6_709 --out-dir validation/results/<run>

Writes in --out-dir:
    service_vs_reference_per_clip.csv   one row per clip
    service_vs_reference.json           metrics, criteria verdicts, service metrics
Exit code 0 only if every gated criterion passes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "bridge"))

import httpx  # noqa: E402
import verify_bridge_parity as vbp  # noqa: E402

VARIANT = "fp16_pruned493"
DET_THRESHOLD = 0.25          # detection threshold used in Phase 4 and by the service default
CRITERIA_PATH = REPO / "validation" / "config" / "criteria.json"
LAYER = "service_vs_reference"


def parse_set(cell: str) -> frozenset:
    """Recorded detection sets are species joined by ';'. Empty cell = empty set."""
    cell = (cell or "").strip()
    return frozenset(s for s in cell.split(";") if s)


def clip_view(dets: list) -> dict:
    """dets: list of (species, confidence, in_allow_list) for one clip."""
    allowed = [d for d in dets if d[2]]
    top1 = max(allowed, key=lambda d: d[1])[0] if allowed else None
    det_set = frozenset(s for s, c, a in dets if a and c >= DET_THRESHOLD)
    by_species = {s: c for s, c, _ in dets}
    return {"top1": top1, "det_set": det_set, "conf": by_species}


def score(rows: list, dets_by_clip: dict) -> tuple[dict, list]:
    per_clip = []
    for r in rows:
        v = clip_view(dets_by_clip.get(r["clip"], []))
        ref_conf = float(r["ref_conf"])
        svc_conf = v["conf"].get(r["ref_top1"])
        dev = abs(svc_conf - ref_conf) if svc_conf is not None else float("inf")
        rec16 = parse_set(r["fp16_det_set"])
        per_clip.append({
            "clip": r["clip"],
            "ref_top1": r["ref_top1"],
            "ref_conf": ref_conf,
            "svc_top1_allow": v["top1"] or "",
            "svc_conf_for_ref": "" if svc_conf is None else svc_conf,
            "abs_dev": "" if svc_conf is None else dev,
            "top1_match": v["top1"] == r["ref_top1"],
            "svc_det_set": ";".join(sorted(v["det_set"])),
            "det_eq_recorded_fp16": v["det_set"] == rec16,
        })
    n = len(per_clip)

    def pct(k: int) -> float:
        return 100.0 * k / n if n else 0.0

    devs = [(p["abs_dev"], p["clip"]) for p in per_clip if p["abs_dev"] != ""]
    worst = max(devs) if devs else (None, None)
    metrics = {
        "n_clips": n,
        "top1_agree_n": sum(p["top1_match"] for p in per_clip),
        "conf_within_0.01_n": sum(1 for d, _ in devs if d <= 0.01),
        "conf_within_0.05_n": sum(1 for d, _ in devs if d <= 0.05),
        "ref_species_not_returned_n": n - len(devs),
        "max_abs_dev": worst[0],
        "max_abs_dev_clip": worst[1],
        "det_set_vs_recorded_fp16_n": sum(p["det_eq_recorded_fp16"] for p in per_clip),
    }
    metrics["top1_agree_pct"] = pct(metrics["top1_agree_n"])
    metrics["conf_within_0.01_pct"] = pct(metrics["conf_within_0.01_n"])
    metrics["conf_within_0.05_pct"] = pct(metrics["conf_within_0.05_n"])
    metrics["det_set_vs_recorded_fp16_pct"] = pct(metrics["det_set_vs_recorded_fp16_n"])
    return metrics, per_clip


def judge(metrics: dict, criteria: dict) -> list:
    out = []
    for c in criteria["criteria"]:
        if c.get("layer") != LAYER:
            continue
        got = metrics[c["metric"]]
        ok = got >= c["value"] if c["op"] == ">=" else got <= c["value"]
        out.append({"id": c["id"], "metric": c["metric"], "op": c["op"], "threshold": c["value"],
                    "value": got, "gate": c["gate"], "pass": bool(ok)})
    return out


def fetch_detections(client: httpx.Client, paths: list) -> tuple[dict, dict]:
    state = vbp.run_job(client, VARIANT, paths)
    by_clip: dict = {}
    for d in state.get("detections") or []:
        by_clip.setdefault(d["data_id"], []).append(
            (d["species"], d["confidence"], d.get("in_allow_list", True)))
    return by_clip, state.get("metrics") or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--audio-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--criteria", type=Path, default=CRITERIA_PATH)
    a = ap.parse_args()

    criteria = json.loads(a.criteria.read_text(encoding="utf-8"))
    audio_root = vbp.resolve_audio_root(a.audio_root)
    with open(vbp.DETAIL_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    names = [r["clip"] for r in rows]
    paths = vbp.resolve_paths(names, audio_root)
    order = {Path(p).name: p for p in paths}

    with httpx.Client(base_url=a.host, timeout=30.0) as client:
        health = client.get("/health").json()
        if not health.get("ready") or VARIANT not in health.get("variants_loaded", []):
            raise SystemExit(f"FATAL: service not ready or {VARIANT} not loaded: {health}")
        dets_by_clip, svc_metrics = fetch_detections(client, [order[n] for n in names])

    metrics, per_clip = score(rows, dets_by_clip)
    verdicts = judge(metrics, criteria)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "service_vs_reference_per_clip.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_clip[0]))
        w.writeheader()
        w.writerows(per_clip)
    result = {"variant": VARIANT, "metrics": metrics, "criteria": verdicts,
              "service_metrics": svc_metrics, "detection_threshold": DET_THRESHOLD}
    (a.out_dir / "service_vs_reference.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Service vs ORIGINAL BirdNET reference | {VARIANT} | {metrics['n_clips']} clips\n")
    print(f"  top-1 agrees with reference:        {metrics['top1_agree_n']}/{metrics['n_clips']}  ({metrics['top1_agree_pct']:.3f}%)")
    print(f"  confidence within 0.05:             {metrics['conf_within_0.05_n']}/{metrics['n_clips']}  ({metrics['conf_within_0.05_pct']:.3f}%)")
    print(f"  confidence within 0.01:             {metrics['conf_within_0.01_n']}/{metrics['n_clips']}  ({metrics['conf_within_0.01_pct']:.3f}%)")
    print(f"  max |service - reference|:          {metrics['max_abs_dev']}  ({metrics['max_abs_dev_clip']})")
    print(f"  reference species not returned:     {metrics['ref_species_not_returned_n']}")
    print(f"  det set == recorded FP16 (>=0.25):  {metrics['det_set_vs_recorded_fp16_n']}/{metrics['n_clips']}")
    print("\n  Criteria (config/criteria.json):")
    for v in verdicts:
        tag = "PASS" if v["pass"] else ("FAIL" if v["gate"] else "BELOW (not gated)")
        print(f"    {v['id']} {v['metric']} {v['op']} {v['threshold']}: {v['value']:.3f}  -> {tag}")
    failed = [v for v in verdicts if v["gate"] and not v["pass"]]
    print(f"\nOVERALL: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
