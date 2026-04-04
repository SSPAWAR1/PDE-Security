
---

## `config.py`

```python
"""
Global project configuration.

This file contains cross-project settings that are shared across experiments.
Experiment-specific settings should live in `configs/exp*.py`.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure output directory exists when imported
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------

DEFAULT_RANDOM_SEED = 42

# Shared transpilation defaults
DEFAULT_BASIS_GATES = ["rz", "sx", "x", "cx"]
DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_LAYOUT_METHOD = "sabre"
DEFAULT_ROUTING_METHOD = "sabre"

# Verification defaults
DEFAULT_VERIFY_TRANSPILATION = True
DEFAULT_TVD_THRESHOLD = 0.05

# Plotting / export defaults
DEFAULT_DPI = 300

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def outputs_path(filename: str) -> Path:
    """
    Return a path inside the outputs directory.
    """
    return OUTPUTS_DIR / filename
