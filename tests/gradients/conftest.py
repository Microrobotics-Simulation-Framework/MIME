import sys
from pathlib import Path

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def x64():
    """The gradient gates run in float64 (round 3, item 4); a test of float32 sets its own. Restored after each
    module."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


#: The materials tests' record fixtures (``tests/materials/records.py``), which gate 6's extrapolated read takes; appended,
#: so that no module of this folder's is shadowed.
MATERIALS = str(Path(__file__).resolve().parents[1] / "materials")
if MATERIALS not in sys.path:
    sys.path.append(MATERIALS)
