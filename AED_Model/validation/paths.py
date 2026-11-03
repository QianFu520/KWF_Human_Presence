"""
Portable data paths for manifests and ground-truth logs.

Committed files store audio paths with a placeholder instead of one person's
Drive mount, and every script expands it from AED_Model/config.py:

  {AUDIO_DIR}  config.AUDIO_DIR                    .../Audio Moth Data/Segmented Foldered (3 s clips)
  {RAW_AUDIO}  the folder above config.AUDIO_DIR   .../Audio Moth Data (raw recordings)

Set AUDIO_DIR in config.py to your own copy and the same files work unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402  (AED_Model/config.py)

_AUDIO = config.AUDIO_DIR.rstrip("/")
_RAW = os.path.dirname(_AUDIO)
# most specific first: AUDIO_DIR sits inside RAW_AUDIO
ROOTS = [("{AUDIO_DIR}", _AUDIO), ("{RAW_AUDIO}", _RAW)]


def expand(path):
    """Placeholder path -> real path on this machine (other paths unchanged)."""
    for key, root in ROOTS:
        if path.startswith(key):
            return root + path[len(key):]
    return path


def portable(path):
    """Real path under a known root -> placeholder path (other paths unchanged)."""
    for key, root in ROOTS:
        if path == root or path.startswith(root + "/"):
            return key + path[len(root):]
    return path
