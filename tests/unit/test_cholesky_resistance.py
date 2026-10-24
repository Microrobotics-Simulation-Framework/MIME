"""``cholesky_resistance`` builds ``L`` whole-array (one gather and two selects). The 0.2 line's loop of 21 single-entry
writes stays here as the reference (``reference_cholesky_resistance``, not library code): the new function equals it
on entries as the packaged network emits them, in both dtypes, in value and in gradient, and its lowering holds none of
the operations the rewrite exists to remove. The control: the same guard, run on the reference, fails."""

import re

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.core import jnp
from mime.data import asset, asset_bytes
from mime.surrogates import DIAGONAL_FLOOR, TRIANGLE, _cholesky_mlp, cholesky_resistance, read_cholesky_mlp

NETWORK = read_cholesky_mlp(asset_bytes(asset("dejongh_mlp_v2")))

#: How many standard deviations about the stored mean the network's outputs are drawn over: its outputs are the stored
#: statistics' normalised values, which a trained network keeps within a few of them.
STANDARD_DEVIATIONS = 4.0

DTYPES = [pytest.param(np.float32, id="float32"), pytest.param(np.float64, id="float64")]


def reference_cholesky_resistance(entries):
    """The loop the whole-array form replaced: one single-entry write per entry of ``L``'s lower triangle."""
    factor = jnp.zeros((6, 6), dtype=jnp.asarray(entries).dtype)
    for index, (i, j) in enumerate(TRIANGLE):
        value = entries[index]
        factor = factor.at[i, j].set(jax.nn.softplus(value) + DIAGONAL_FLOOR if i == j else value)
    return factor @ factor.T


@pytest.fixture(params=DTYPES, scope="module")
def dtype(request):
    """The dtype under test, the process in the mode that makes it the default float. Module-scoped, as Hypothesis
    requires of a fixture its tests draw under."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param is np.float64)
    yield request.param
    jax.config.update("jax_enable_x64", held)


def network_entries(draw_normalised, dtype):
    return (NETWORK.entry_mean + NETWORK.entry_std * np.asarray(draw_normalised)).astype(dtype)


normalised_entries = st.lists(st.floats(-STANDARD_DEVIATIONS, STANDARD_DEVIATIONS), min_size=21, max_size=21)
resistance_weights = st.lists(st.floats(-1.0, 1.0), min_size=36, max_size=36)

#: The functions as a compiled step holds them.
new_value, old_value = jax.jit(cholesky_resistance), jax.jit(reference_cholesky_resistance)


def weighted_gradient(function):
    """The gradient with respect to the entries of a weighted sum of the resistance's cells, as a compiled step has it."""
    return jax.jit(jax.grad(lambda entries, weights: jnp.sum(weights * function(entries))))


new_gradient, old_gradient = weighted_gradient(cholesky_resistance), weighted_gradient(reference_cholesky_resistance)


@settings(deadline=None, max_examples=250)
@given(draw=normalised_entries)
def test_the_whole_array_form_equals_the_loop_in_value(dtype, draw):
    entries = jnp.asarray(network_entries(draw, dtype))
    new, old = np.asarray(new_value(entries)), np.asarray(old_value(entries))
    assert new.dtype == old.dtype == dtype
    np.testing.assert_array_equal(new, old)
    np.testing.assert_array_equal(np.asarray(cholesky_resistance(entries)), np.asarray(reference_cholesky_resistance(entries)))


@settings(deadline=None, max_examples=250)
@given(draw=normalised_entries, drawn_weights=resistance_weights)
def test_the_whole_array_form_equals_the_loop_in_gradient(dtype, draw, drawn_weights):
    entries = jnp.asarray(network_entries(draw, dtype))
    weights = jnp.asarray(np.asarray(drawn_weights, dtype=dtype).reshape(6, 6))
    new, old = np.asarray(new_gradient(entries, weights)), np.asarray(old_gradient(entries, weights))
    assert new.dtype == old.dtype == dtype
    np.testing.assert_array_equal(new, old)


def test_the_comparison_sees_a_wrong_index_map(dtype, monkeypatch):
    """The control for the two properties above: with the index map transposed, the function no longer equals the loop
    at the network's own mean."""
    entries = jnp.asarray(network_entries(np.zeros(21), dtype))
    monkeypatch.setattr(_cholesky_mlp, "_ENTRY_AT", _cholesky_mlp._ENTRY_AT.T.copy())
    assert not np.array_equal(np.asarray(cholesky_resistance(entries)), np.asarray(reference_cholesky_resistance(entries)))


def operations(lowering, name):
    return len(re.findall(rf'"?stablehlo\.{name}"?\(', lowering))


def removed_operations(function, dtype):
    """What a lowering of ``function`` holds of the operations the whole-array form exists to remove: its scatters and
    dynamic update slices, and gathers beyond the one."""
    lowering = jax.jit(function).lower(jax.ShapeDtypeStruct((21,), dtype)).as_text()
    return {"scatter": operations(lowering, "scatter"), "dynamic_update_slice": operations(lowering, "dynamic_update_slice"),
            "extra gather": max(operations(lowering, "gather") - 1, 0)}


def test_the_lowering_holds_no_scatter_or_update_slice_and_at_most_one_gather(dtype):
    held = removed_operations(cholesky_resistance, dtype)
    assert held == {"scatter": 0, "dynamic_update_slice": 0, "extra gather": 0}, held
    lowering = jax.jit(cholesky_resistance).lower(jax.ShapeDtypeStruct((21,), dtype)).as_text()
    assert operations(lowering, "gather") == 1


def test_the_guard_fails_on_the_loop(dtype):
    """The control, on every run: the same guard on the reference counts its 21 scatters."""
    held = removed_operations(reference_cholesky_resistance, dtype)
    assert held["scatter"] + held["dynamic_update_slice"] >= len(TRIANGLE), held


def test_it_runs_in_float32_then_float64_then_float32_in_one_process():
    """The constant index map is shared by every call: an ``int64`` one, indexing under float32, was cached by JAX as the
    ``int32`` it became and then failed under x64 (the test suites switch modes within a process)."""
    held = jax.config.jax_enable_x64
    try:
        for x64 in (False, True, False):
            jax.config.update("jax_enable_x64", x64)
            wanted = np.float64 if x64 else np.float32
            entries = jnp.asarray(network_entries(np.zeros(21), wanted))
            assert cholesky_resistance(entries).dtype == wanted
    finally:
        jax.config.update("jax_enable_x64", held)
