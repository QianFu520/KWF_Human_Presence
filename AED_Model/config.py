"""Configuration for rainforest-audio-detection project.
All paths and constants live here so scripts don't hardcode them.
"""

import os

# Project root — the folder this config file lives in
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Input datasets (not in git — see .gitignore)
EVENT_TABLE_PATH = os.path.join(DATA_DIR, "event_table.csv")
CLIP_CATALOG_PATH = os.path.join(DATA_DIR, "clip_catalog.csv")

# Raw audio lives outside the repo (too large to store with the project).
# Point AED_AUDIO_DIR at your copy of "Audio Moth Data/Segmented Foldered", e.g. the
# Google Drive for desktop mount:
#   export AED_AUDIO_DIR="$HOME/Library/CloudStorage/GoogleDrive-<you>/My Drive/Audio Moth Data/Segmented Foldered"
# Without it, audio is expected in data/raw_audio/ (git-ignored).
AUDIO_DIR = os.path.expanduser(os.environ.get("AED_AUDIO_DIR", os.path.join(DATA_DIR, "raw_audio")))

# Outputs directory (regenerable artifacts, git-ignored)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Labeling pipeline intermediate state
LABELS_PROGRESS_PATH = os.path.join(OUTPUTS_DIR, "labels_progress.csv")
