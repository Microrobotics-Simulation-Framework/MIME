"""The solve-site registry's tests build the FVM's nodes with ``tests/flow``'s cases and the near field's with
``tests/near_field``'s: their folders go on the path here, since pytest puts a test's own folder there and no other."""

import sys
from pathlib import Path

import jax
import pytest

for folder in ("flow", "near_field"):
    held = str(Path(__file__).resolve().parents[1] / folder)
    if held not in sys.path:
        sys.path.append(held)


@pytest.fixture(params=["float32", "float64"])
def dtype(request: pytest.FixtureRequest):
    """Each test in both dtypes, the run's dtype set for it and restored after."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    yield request.param
    jax.config.update("jax_enable_x64", held)


@pytest.fixture
def float64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield "float64"
    jax.config.update("jax_enable_x64", held)
