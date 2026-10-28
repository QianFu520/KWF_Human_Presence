"""Shared paths for the testing notebooks.

PIPELINE_DIR: the inference_pipeline folder (found from this file's location; override with KWF_PIPELINE_DIR).
KWF_TEST_DIR: folder with test data and outputs outside the repo (default ~/kwf_test; override with KWF_TEST_DIR).
"""
import os
from pathlib import Path

PIPELINE_DIR = Path(os.environ.get("KWF_PIPELINE_DIR", Path(__file__).resolve().parents[2]))
KWF_TEST_DIR = Path(os.environ.get("KWF_TEST_DIR", Path.home() / "kwf_test"))
