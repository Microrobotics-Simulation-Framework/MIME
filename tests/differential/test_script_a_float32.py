"""Script A in float32 against the frozen float32 reference, with the differences declared for float32 before the
run (``script_a_expected_differences.json``: the f32 reference moved by the emulated difference between the two
schemes; the commands round alike in float32, so theirs is absent). Each line rounds on its own, so the tolerances are
at least ten times the reference's own float32-float64 gap in each kind of quantity. What it bounds is
``test_script_a.py``'s."""

import jax
import pytest
from script_a_runs import DECLARED, GATED, REFERENCE, compare, quantities, reported, run, table


@pytest.fixture(autouse=True, scope="module")
def x64():
    """This module runs in float32: it overrides the folder's float64 fixture, and restores the mode after."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", held)


def test_script_a_in_float32_matches_its_float32_reference_with_the_declared_differences():
    history = run(10.0)
    assert history.position.dtype.name == "float32"
    measured = quantities(history, 10.0)
    tolerances = DECLARED["tolerances"]["f32"]
    assert set(tolerances) == set(GATED)
    rows = compare(measured, REFERENCE["f32"], DECLARED["predicted"]["f32_10hz"], tolerances)
    print("\nfloat32, 10 Hz\n" + table(rows) + "\n" + reported(measured))
    assert [row.quantity for row in rows if not row.holds] == []
