"""Regression guard for the jax_enable_x64 test-isolation leak.

Several verification modules used to flip ``jax_enable_x64`` on at *import*
time via a module-level ``jax.config.update``. pytest runs that during
collection, so x64 leaked on for the whole session — making results
order-dependent (notably under pytest-xdist) and producing spurious
``float64 -> float32`` cast FutureWarnings in float32 nodes. The fix
(see ``tests/conftest.py``) scopes x64 to ``@pytest.mark.x64`` tests with
deterministic teardown. These tests pin that contract so a future
module-level enable is caught immediately.
"""

import jax
import pytest


def test_x64_off_by_default():
    """An unmarked test must see JAX's float32 default.

    If any test module re-introduces a module-level
    ``jax.config.update("jax_enable_x64", True)``, collection turns x64 on
    globally and this assertion fails — flagging the leak at its source.
    """
    assert jax.config.jax_enable_x64 is False


@pytest.mark.x64
def test_x64_on_when_marked():
    """The ``x64`` marker enables double precision for the marked test."""
    assert jax.config.jax_enable_x64 is True


def test_x64_restored_after_marked_test():
    """x64 is torn down after a marked test (this unmarked test follows it
    in file order and must see the default again)."""
    assert jax.config.jax_enable_x64 is False
