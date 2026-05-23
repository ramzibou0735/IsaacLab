from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"

sys.path.insert(0, str(PYTHON_ROOT))

project = "rlPx4ControllerTorch"
author = "OpenAI Codex"
copyright = "2026, OpenAI Codex"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]
autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
