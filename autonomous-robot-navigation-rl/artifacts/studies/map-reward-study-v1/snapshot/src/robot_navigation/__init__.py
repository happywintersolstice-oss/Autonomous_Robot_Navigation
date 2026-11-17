"""Fixed-map navigation: environment, policies, recorded experiments and replay."""
import os
from pathlib import Path

# Keep Matplotlib's writable cache inside this isolated project on Windows.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))

__version__ = "0.1.0"
