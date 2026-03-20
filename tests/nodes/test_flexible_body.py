"""Tests for FlexibleBodyNode."""

import pytest
import jax
import jax.numpy as jnp

from mime.nodes.robot.flexible_body import FlexibleBodyNode, build_beam_stiffness_matrix


class TestFlexibleBodyBasic:
    def test_initial_state_zeros(self):
        node = FlexibleBodyNode("beam", 0.001, n_nodes=20)
        state = node.initial_state()
        assert state["deflection"].shape == (20,)
        assert jnp.allclose(state["deflection"], jnp.zeros(20))

    def test_no_forcing_no_deflection(self):
        node = FlexibleBodyNode("beam", 0.001, n_nodes=10)
        state = node.initial_state()
        bi = {"actuation_moment": 0.0}
        new = node.update(state, bi, 0.001)
        assert jnp.allclose(new["deflection"], jnp.zeros(10), atol=1e-20)

    def test_moment_causes_deflection(self):
        node = FlexibleBodyNode("beam", 0.001, n_nodes=10,
                                bending_stiffness_nm2=1e-18)
        state = node.initial_state()
        bi = {"actuation_moment": 1e-15}  # Small but nonzero moment
        new = node.update(state, bi, 0.001)
        # Some nodes should deflect (not all zero)
        assert jnp.max(jnp.abs(new["deflection"])) > 0

    def test_clamped_end_stays_zero(self):
        """Node 0 (clamped end) should always be zero."""
        node = FlexibleBodyNode("beam", 0.001, n_nodes=10,
                                bending_stiffness_nm2=1e-18)
        state = node.initial_state()
        bi = {"actuation_moment": 1e-15}
        for _ in range(10):
            state = node.update(state, bi, 0.001)
        assert jnp.abs(state["deflection"][0]) < 1e-20

    def test_deflection_finite_after_many_steps(self):
        node = FlexibleBodyNode("beam", 0.001, n_nodes=10,
                                bending_stiffness_nm2=1e-20)
        state = node.initial_state()
        bi = {"actuation_moment": 1e-16}
        for _ in range(100):
            state = node.update(state, bi, 0.001)
        assert jnp.isfinite(state["deflection"]).all()


class TestFlexibleBodyMetadata:
    def test_meta_set(self):
        assert FlexibleBodyNode.meta is not None
        assert FlexibleBodyNode.meta.algorithm_id == "MIME-NODE-006"

    def test_validate_consistency(self):
        node = FlexibleBodyNode("beam", 0.001)
        errors = node.validate_mime_consistency()
        assert errors == [], f"Errors: {errors}"


class TestStiffnessMatrix:
    def test_shape(self):
        S = build_beam_stiffness_matrix(10, 1e-5, 1e-20)
        assert S.shape == (10, 10)

    def test_zero_stiffness(self):
        S = build_beam_stiffness_matrix(10, 1e-5, 0.0)
        assert jnp.allclose(S, jnp.zeros((10, 10)))


class TestFlexibleBodyJAX:
    def test_jit_traceable(self):
        node = FlexibleBodyNode("beam", 0.001, n_nodes=10)
        state = node.initial_state()
        bi = {"actuation_moment": jnp.array(1e-16)}
        jitted = jax.jit(node.update)
        new = jitted(state, bi, 0.001)
        assert jnp.isfinite(new["deflection"]).all()
