"""
Integration tests for the AED step (Step 2) and its handoff to BirdNET (Step 3).

Runs the pipeline's own step2_filter and step3_birdnet on a few generated
3 s / 48 kHz clips (tones, chirps, noise; no field audio needed), and checks:
  - the pipeline loads the validated AED checkpoint and the same code as AED_Model/
  - Step 2 output schema: one row per clip, probability in [0, 1],
    label = not_meaningful if prob >= threshold
  - Step 2 -> Step 3: both files describe the same clips, with the same
    clip_name format, so later steps can join them

Run with: python -m pytest inference_pipeline/tests/test_aed_handoff.py -v
"""

import ast
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

PIPELINE_DIR = Path(__file__).resolve().parent.parent
REPO = PIPELINE_DIR.parent
sys.argv = sys.argv[:1]  # run_pipeline parses arguments only in main(), but keep pytest's out of the way

import aed_inference  # noqa: E402  (inference_pipeline/scripts/, via conftest)
import run_pipeline  # noqa: E402

SR, N = 48000, 144000
V3_SHA256 = "913a78a468ff42e2fbd250c3d43ad658de79fd1ed1aa320c56c7ecba57f83cdb"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def clips_dir(tmp_path_factory):
    """Six clips named like the splitter writes them, in a nested folder like the real layout."""
    d = tmp_path_factory.mktemp("clips")
    sub = d / "Audio_Moth_5"
    sub.mkdir()
    t = np.arange(N) / SR
    rng = np.random.default_rng(0)
    signals = {
        "093000": 0.3 * np.sin(2 * np.pi * 3000 * t),
        "093003": 0.3 * np.sin(2 * np.pi * (2000 + 1500 * t) * t),
        "093006": 0.05 * rng.standard_normal(N),
        "093009": 0.001 * rng.standard_normal(N),
        "093012": np.where((t % 0.5) < 0.05, 0.4, 0.0) * np.sin(2 * np.pi * 4000 * t),
        "093015": np.zeros(N),
    }
    for hhmmss, y in signals.items():
        sf.write(sub / f"Audio_Moth_5_20250319_{hhmmss}.wav", y.astype(np.float32), SR, subtype="PCM_16")
    return d


@pytest.fixture(scope="module")
def step_outputs(clips_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("outputs")
    meaningful, cnn_csv = run_pipeline.step2_filter(clips_dir, out)
    birdnet_csv = run_pipeline.step3_birdnet(clips_dir, meaningful, out)
    return meaningful, pd.read_csv(cnn_csv), pd.read_csv(birdnet_csv)


def test_pipeline_uses_validated_checkpoint():
    assert sha256(run_pipeline.MODELS_DIR / "tinycnn_v3.pth") == V3_SHA256


def test_pipeline_checkpoint_matches_aed_model():
    aed_copy = REPO / "AED_Model" / "outputs" / "models" / "tinycnn_v3.pth"
    if not aed_copy.exists():
        pytest.skip("AED_Model checkpoint not in this checkout")
    assert sha256(aed_copy) == sha256(run_pipeline.MODELS_DIR / "tinycnn_v3.pth")


def module_constants(path):
    """Module-level UPPER_CASE assignments with literal values (mel parameters and the like)."""
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():
                try:
                    out[name] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


def test_pipeline_aed_code_matches_aed_model():
    """The pipeline keeps its own copy of aed_inference.py; it must not drift from AED_Model/."""
    src = REPO / "AED_Model" / "aed_inference.py"
    if not src.exists():
        pytest.skip("AED_Model/aed_inference.py not in this checkout")
    code = src.read_text()
    # the import lines differ by design; preprocessing, prediction and constants must not
    for name in ("preprocess_clip", "predict"):
        assert inspect.getsource(getattr(aed_inference, name)) in code, f"{name} differs from AED_Model"
    assert module_constants(aed_inference.__file__) == module_constants(src)


def test_step2_schema(step_outputs, clips_dir):
    _, aed, _ = step_outputs
    n_clips = len(list(clips_dir.rglob("*.wav")))
    assert list(aed.columns) == ["clip_name", "audio_path", "not_meaningful_prob", "predicted_label",
                                 "threshold", "model_version"]
    assert len(aed) == n_clips and aed.clip_name.is_unique
    assert aed.clip_name.str.endswith(".wav").all()
    assert aed.not_meaningful_prob.between(0, 1).all()
    assert (aed.threshold == 0.5).all()
    expected = np.where(aed.not_meaningful_prob >= aed.threshold, "not_meaningful", "meaningful")
    assert (aed.predicted_label == expected).all()


def test_step2_returns_meaningful_list(step_outputs):
    meaningful, aed, _ = step_outputs
    assert sorted(meaningful) == sorted(aed.loc[aed.predicted_label == "meaningful", "clip_name"])


def test_step2_to_step3_same_clips(step_outputs):
    """BirdNET runs on every clip (mark, don't drop), so both files must cover the same clip names."""
    _, aed, birdnet = step_outputs
    assert birdnet.clip_name.is_unique
    assert set(aed.clip_name) == set(birdnet.clip_name)
    joined = aed.merge(birdnet, on="clip_name", how="inner", validate="one_to_one")
    assert len(joined) == len(aed)
