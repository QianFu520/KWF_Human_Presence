"""Run the full BirdNET service validation and save every output.

    python validation/scripts/run_validation.py \
        --audio-root C:/path/to/am6_709 \
        --models-dir C:/path/to/models \
        --models-config C:/path/to/models.json

The service must already be running (see validation/README.md). Creates
validation/results/<UTC timestamp>/ containing:
    run_info.json, audio_manifest.csv       what was used
    01_identity.txt                         scripts/check_model_pairing.py
    02_service_vs_recorded_fp16.txt         src/bridge/verify_bridge_parity.py
    03_service_vs_reference.txt/.json/.csv  validation/scripts/service_vs_reference.py
    04_interface_and_preprocessing.txt/.json/.csv  validation/scripts/interface_and_preprocessing.py
    SUMMARY.md                              verdict per criterion
Exit code 0 only if every gated criterion passes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run_step(name: str, cmd: list, out_txt: Path, show_tail: bool = True) -> int:
    print(f"\n>>> {name}\n    {' '.join(map(str, cmd))}")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    text = p.stdout + (("\n--- stderr ---\n" + p.stderr) if p.stderr.strip() else "")
    out_txt.write_text(text, encoding="utf-8")
    if show_tail:
        tail = "\n".join(p.stdout.strip().splitlines()[-6:])
        print(tail if tail else "(no stdout)")
    print(f"    exit code {p.returncode}  ->  {out_txt.name}")
    return p.returncode


def fp16_identity_verdict(txt: Path) -> tuple[bool, str]:
    """check_model_pairing.py checks FP16 and FP32. Judge the FP16 section only."""
    lines = txt.read_text(encoding="utf-8").splitlines()
    sect, inside = [], False
    for ln in lines:
        if ln.startswith("== "):
            inside = "fp16_pruned493" in ln
            continue
        if inside:
            if ln.startswith("ALL CHECKS") or "CHECK(S) FAILED" in ln:
                break
            sect.append(ln)
    checks = [x for x in sect if "[PASS]" in x or "[FAIL]" in x]
    fails = [x for x in checks if "[FAIL]" in x]
    if not checks:
        return False, "FP16 section not found in output"
    return not fails, f"FP16 section: {len(checks)} checks, {len(fails)} failed (FP32 section not judged)"


def identity_verdict(run_info: Path, expected: dict) -> tuple[bool, str]:
    """C0: documented size and label content hash for the FP16 files the service loaded."""
    try:
        files = json.loads(run_info.read_text(encoding="utf-8"))["service_files"]["fp16_pruned493"]
    except (OSError, KeyError, ValueError):
        return False, "fp16_pruned493 not found in run_info.json"
    size_ok = files["model"].get("bytes") == expected["fp16_model_bytes"]
    h = files["labels"].get("sha256_lf", "")
    hash_ok = h.startswith(expected["fp16_labels_sha256_lf_prefix"])
    return size_ok and hash_ok, (f"model bytes {files['model'].get('bytes')} (want {expected['fp16_model_bytes']}); "
                                 f"labels sha256_lf {h[:16]} (want {expected['fp16_labels_sha256_lf_prefix']})")


def redact_home(folder: Path) -> None:
    """Replace the user's home directory with ~ in the saved results, so the
    committed files do not carry a personal path."""
    home = str(Path.home())
    forms = {home, home.replace("\\", "\\\\"), home.replace("\\", "/")}
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix in {".txt", ".json", ".md", ".csv"}:
            t = f.read_text(encoding="utf-8", errors="replace")
            new = t
            for h in forms:
                new = new.replace(h, "~")
            if new != t:
                f.write_text(new, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, required=True, help="folder with the .onnx and .labels.txt files")
    ap.add_argument("--models-config", type=Path, required=True, help="the models.json the service was started with")
    ap.add_argument("--wrong-length-dir", type=Path, default=None,
                    help="folder with real wrong-length WAVs (a few from filtered_clips_6)")
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--results-root", type=Path, default=REPO / "validation" / "results")
    a = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = a.results_root / ts
    out.mkdir(parents=True)
    py = sys.executable
    print(f"Results folder: {out}")

    rc = {}
    rc["info"] = run_step("Record run info", [py, HERE / "run_info.py", "--out-dir", out, "--host", a.host,
                          "--models-config", a.models_config, "--audio-root", a.audio_root], out / "00_run_info.txt")
    rc["C1"] = run_step("C1 identity: model and labels", [py, REPO / "scripts" / "check_model_pairing.py",
                        "--models-dir", a.models_dir], out / "01_identity.txt", show_tail=False)
    print("    " + fp16_identity_verdict(out / "01_identity.txt")[1])
    rc["C2"] = run_step("C2 service vs recorded FP16", [py, REPO / "src" / "bridge" / "verify_bridge_parity.py",
                        "--audio-root", a.audio_root, "--variant", "fp16_pruned493", "--host", a.host],
                        out / "02_service_vs_recorded_fp16.txt")
    rc["ref"] = run_step("C3-C6 service vs original reference", [py, HERE / "service_vs_reference.py",
                         "--audio-root", a.audio_root, "--host", a.host, "--out-dir", out],
                         out / "03_service_vs_reference.txt")

    cmd = [py, HERE / "interface_and_preprocessing.py", "--audio-root", a.audio_root, "--host", a.host, "--out-dir", out]
    if a.wrong_length_dir:
        cmd += ["--wrong-length-dir", a.wrong_length_dir]
    rc["iface"] = run_step("C8-C9 interface and preprocessing effects", cmd, out / "04_interface_and_preprocessing.txt")

    criteria = json.loads((REPO / "validation" / "config" / "criteria.json").read_text(encoding="utf-8"))
    ip_json = out / "interface_and_preprocessing.json"
    iface = {v["id"]: v for v in json.loads(ip_json.read_text(encoding="utf-8"))["criteria"]} if ip_json.is_file() else {}
    ref_json = out / "service_vs_reference.json"
    ref = {v["id"]: v for v in json.loads(ref_json.read_text(encoding="utf-8"))["criteria"]} if ref_json.is_file() else {}
    lines = [f"# Validation run {ts} (UTC)", "",
             f"Criteria status: {criteria['status']}", "",
             "| ID | Layer | Gate | Result | Detail |", "|---|---|---|---|---|"]
    all_ok = True
    for c in criteria["criteria"]:
        cid = c["id"]
        if cid == "C0":
            ok, detail = identity_verdict(out / "run_info.json", c["expected"])
        elif cid == "C1":
            ok, detail = fp16_identity_verdict(out / "01_identity.txt")
        elif cid == "C2":
            ok = rc[cid] == 0
            detail = f"exit code {rc[cid]}"
        elif cid in ref:
            ok = ref[cid]["pass"]
            detail = f"{ref[cid]['value']:.3f} {ref[cid]['op']} {ref[cid]['threshold']}"
        elif cid in iface:
            ok = iface[cid]["pass"]
            detail = iface[cid]["detail"]
        else:
            ok, detail = False, "no result (step failed to run)"
        if c["gate"]:
            all_ok &= ok
        if ok:
            result = "PASS"
        elif c["gate"]:
            result = "FAIL"
        elif c.get("known_issue"):
            result = f"FAIL (known issue {c['known_issue']}, not gated)"
        else:
            result = "below threshold (not gated)"
        lines.append(f"| {cid} | {c['layer']} | {'yes' if c['gate'] else 'no'} | {result} | {detail} |")
    lines += ["", f"**Overall: {'PASS' if all_ok else 'FAIL'}**", "",
              "See run_info.json for model/label hashes, versions and git commit."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    redact_home(out)
    print("\n" + "\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
