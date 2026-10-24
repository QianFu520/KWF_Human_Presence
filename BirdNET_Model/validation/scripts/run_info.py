"""Record exactly what a validation run used, so it can be reproduced and audited.

Writes run_info.json (and audio_manifest.csv) into --out-dir:
  - UTC time, OS, Python, package versions
  - git commit and whether the working tree had uncommitted changes
  - SHA-256 and size of every model, label and allow-list file the service loaded
  - SHA-256 of the criteria file and the reference CSVs
  - the service's /health and /models answers
  - per-file SHA-256 of the test audio (names only, no paths)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGES = ["numpy", "onnx", "onnxruntime", "httpx", "fastapi", "uvicorn",
            "pydantic", "soundfile", "scipy", "psutil"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lf(path: Path) -> str:
    """Hash of a text file with CRLF converted to LF (line endings differ between OS checkouts)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_info(path: Path) -> dict:
    if not path.is_file():
        return {"name": path.name, "error": "not found"}
    info = {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
    if path.suffix in (".txt", ".csv", ".json"):
        info["sha256_lf"] = sha256_lf(path)
    return info


def git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def service_files(models_config: Path | None) -> dict:
    if not models_config or not models_config.is_file():
        return {}
    cfg = json.loads(models_config.read_text(encoding="utf-8"))
    base = REPO / "src" / "bridge"       # relative paths in models.json are relative to src/bridge
    out = {}
    for variant, spec in cfg.get("variants", {}).items():
        out[variant] = {}
        for key in ("model", "labels", "allow_list"):
            if spec.get(key):
                p = Path(spec[key])
                out[variant][key] = file_info(p if p.is_absolute() else (base / p).resolve())
    return out


def audio_manifest(root: Path, names: list[str], out_csv: Path) -> dict:
    rows = []
    for n in sorted(names):
        p = root / n
        rows.append((n, p.stat().st_size, sha256(p)))
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "bytes", "sha256"])
        w.writerows(rows)
    agg = hashlib.sha256("\n".join(f"{n},{s},{h}" for n, s, h in rows).encode()).hexdigest()
    return {"clips": len(rows), "total_bytes": sum(r[1] for r in rows),
            "distinct_sizes": sorted({r[1] for r in rows}), "manifest_sha256": agg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--models-config", type=Path, default=None)
    ap.add_argument("--audio-root", type=Path, default=None)
    ap.add_argument("--criteria", type=Path, default=REPO / "validation" / "config" / "criteria.json")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = REPO / "verification_data" / "phase4_detail_Audio_Moth_6.csv"
    info = {
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": versions(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_uncommitted_changes": bool(git("status", "--porcelain")),
        "criteria": file_info(a.criteria),
        "reference_files": {
            "phase4_detail_Audio_Moth_6.csv": file_info(detail_csv),
            "birdnet_species_108k_fp16_verification.csv": file_info(
                REPO / "verification_data" / "birdnet_species_108k_fp16_verification.csv"),
        },
        "service_files": service_files(a.models_config),
    }
    try:
        import httpx
        with httpx.Client(base_url=a.host, timeout=10.0) as c:
            info["service_health"] = c.get("/health").json()
            info["service_models"] = c.get("/models").json()
    except Exception as e:                       # service may be down; record that fact
        info["service_error"] = repr(e)
    if a.audio_root and a.audio_root.is_dir():
        with open(detail_csv, encoding="utf-8", newline="") as f:
            names = [r["clip"] for r in csv.DictReader(f)]
        info["audio"] = audio_manifest(a.audio_root, names, a.out_dir / "audio_manifest.csv")
    (a.out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Wrote {a.out_dir / 'run_info.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
