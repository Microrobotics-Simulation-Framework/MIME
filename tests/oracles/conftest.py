import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def x64():
    """The build's bitwise claims are tested in float64, as the canonical order was first measured. Scoped to each
    module and restored after it, so no other test runs in the mode it did not ask for."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)
