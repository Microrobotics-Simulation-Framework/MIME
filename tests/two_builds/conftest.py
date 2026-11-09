import sys
from pathlib import Path

import jax
import pytest

#: The registry walk's cases (``tests/saved/cases.py``), which the false-positive arm walks; appended, so that no module
#: of this folder's is shadowed.
SAVED = str(Path(__file__).resolve().parents[1] / "saved")
if SAVED not in sys.path:
    sys.path.append(SAVED)


@pytest.fixture(autouse=True, scope="module")
def x64():
    """The two-builds check runs in float64 (in float32 its trace form stops at about 3e-8 of a value), and so do the
    seam's gradients. A test parametrised by ``precision`` sets its own. Scoped to each module and restored after it."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)
