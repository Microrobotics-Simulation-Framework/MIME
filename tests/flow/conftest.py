import jax
import pytest


@pytest.fixture(params=["float32", "float64"])
def dtype(request: pytest.FixtureRequest):
    """Each test in both dtypes, the run's dtype set for it and restored after: the FVM follows the run's dtype."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    yield request.param
    jax.config.update("jax_enable_x64", held)


@pytest.fixture
def float64():
    """Float64 only: the gradient gate, the central difference and the golden step."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield "float64"
    jax.config.update("jax_enable_x64", held)
