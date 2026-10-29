import sys
from pathlib import Path

# Same import setup as run_pipeline.py: config.py lives in inference_pipeline/,
# the step modules in inference_pipeline/scripts/.
PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "scripts"))
