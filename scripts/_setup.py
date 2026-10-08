"""Shared start-up for the scripts: make `src/` importable and run from the repo root."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)  # all paths in configs are relative to the repo root
