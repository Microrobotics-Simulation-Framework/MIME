"""FieldSuperpositionNode — linear superposition of N magnetic sources.

Pins the combiner the dual-RPM controller relies on: net field is the sum of
the source fields; net gradient is the sum of the source gradients; two
synchronised dipoles can cancel the net gradient while the net field is
preserved.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from mime.nodes.actuation.field_superposition import FieldSuperpositionNode


def test_metadata_is_consistent():
    node = FieldSuperpositionNode("sp", 1e-3, n_sources=2)
    assert node.validate_mime_consistency() == []


def test_rejects_zero_sources():
    with pytest.raises(ValueError):
        FieldSuperpositionNode("sp", 1e-3, n_sources=0)


def test_input_output_spec_scales_with_n_sources():
    node = FieldSuperpositionNode("sp", 1e-3, n_sources=3)
    inputs = node.boundary_input_spec()
    assert set(inputs) == {
        "field_vector_0", "field_vector_1", "field_vector_2",
        "field_gradient_0", "field_gradient_1", "field_gradient_2",
    }
    outputs = node.boundary_flux_spec()
    assert set(outputs) == {"field_vector", "field_gradient"}
    assert outputs["field_vector"].shape == (3,)
    assert outputs["field_gradient"].shape == (3, 3)


def test_fields_add():
    node = FieldSuperpositionNode("sp", 1e-3, n_sources=3)
    bi = {
        "field_vector_0": jnp.array([1.0, 0.0, 0.0]),
        "field_vector_1": jnp.array([0.0, 2.0, 0.0]),
        "field_vector_2": jnp.array([0.0, 0.0, 3.0]),
    }
    out = node.update(node.initial_state(), bi, 1e-3)
    assert jnp.allclose(out["field_vector"], jnp.array([1.0, 2.0, 3.0]))


def test_opposite_gradients_cancel_while_field_survives():
    # The dual-RPM gradient-cancellation case: two dipoles whose gradients are
    # equal and opposite (net force-free) but whose fields add (net steering).
    node = FieldSuperpositionNode("sp", 1e-3, n_sources=2)
    g = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    bi = {
        "field_vector_0": jnp.array([0.01, 0.0, 0.0]),
        "field_gradient_0": g,
        "field_vector_1": jnp.array([0.0, 0.01, 0.0]),
        "field_gradient_1": -g,
    }
    out = node.update(node.initial_state(), bi, 1e-3)
    assert jnp.allclose(out["field_gradient"], jnp.zeros((3, 3)))
    assert jnp.allclose(out["field_vector"], jnp.array([0.01, 0.01, 0.0]))


def test_unwired_source_contributes_zero():
    # An unconnected slot uses its zero default — a 2-source node fed only
    # source 0 returns exactly source 0's contribution.
    node = FieldSuperpositionNode("sp", 1e-3, n_sources=2)
    bi = {"field_vector_0": jnp.array([5.0, 0.0, 0.0])}
    out = node.update(node.initial_state(), bi, 1e-3)
    assert jnp.allclose(out["field_vector"], jnp.array([5.0, 0.0, 0.0]))


def test_update_is_jittable():
    node = FieldSuperpositionNode("sp", 1e-3, n_sources=2)
    bi = {
        "field_vector_0": jnp.array([1.0, 1.0, 1.0]),
        "field_vector_1": jnp.array([2.0, 2.0, 2.0]),
        "field_gradient_0": jnp.eye(3),
        "field_gradient_1": jnp.eye(3),
    }
    jitted = jax.jit(lambda s, b: node.update(s, b, 1e-3))
    out = jitted(node.initial_state(), bi)
    assert jnp.allclose(out["field_vector"], jnp.array([3.0, 3.0, 3.0]))
    assert jnp.allclose(out["field_gradient"], 2.0 * jnp.eye(3))
