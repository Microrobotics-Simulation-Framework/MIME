"""Tests for ExternalMagneticFieldNode."""

import pytest
import jax
import jax.numpy as jnp

from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode


class TestExternalMagneticFieldBasic:
    def test_initial_state_zero_field(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        state = node.initial_state()
        assert jnp.allclose(state["field_vector"], jnp.zeros(3))
        assert state["field_vector"].shape == (3,)
        assert state["field_gradient"].shape == (3, 3)

    def test_update_produces_rotating_field(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        state = node.initial_state()
        bi = {"frequency_hz": 10.0, "field_strength_mt": 20.0}
        new_state = node.update(state, bi, 0.001)
        B = new_state["field_vector"]
        # Field magnitude should be 20 mT = 0.02 T
        assert jnp.abs(jnp.linalg.norm(B) - 0.02) < 1e-6

    def test_field_rotates_in_xy(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        state = node.initial_state()
        bi = {"frequency_hz": 10.0, "field_strength_mt": 10.0}
        # Step to t=0
        s1 = node.update(state, bi, 0.0)
        # At t~0, B should be along x: [B_0, 0, 0]
        # Actually t = 0 + dt = 0, so cos(0) = 1, sin(0) = 0
        # But dt=0 is degenerate — step with small dt
        s1 = node.update(state, bi, 1e-6)
        B = s1["field_vector"]
        assert jnp.abs(B[2]) < 1e-10  # z-component always zero

    def test_field_z_component_always_zero(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        state = node.initial_state()
        bi = {"frequency_hz": 50.0, "field_strength_mt": 30.0}
        for _ in range(100):
            state = node.update(state, bi, 0.001)
            assert jnp.abs(state["field_vector"][2]) < 1e-10

    def test_field_magnitude_constant(self):
        """Field magnitude should stay constant regardless of time."""
        node = ExternalMagneticFieldNode("field", 0.001)
        state = node.initial_state()
        bi = {"frequency_hz": 25.0, "field_strength_mt": 15.0}
        for _ in range(50):
            state = node.update(state, bi, 0.001)
            mag = jnp.linalg.norm(state["field_vector"])
            assert jnp.abs(mag - 0.015) < 1e-8

    def test_gradient_is_zero_uniform_mode(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        state = node.initial_state()
        bi = {"frequency_hz": 10.0, "field_strength_mt": 10.0}
        new_state = node.update(state, bi, 0.001)
        assert jnp.allclose(new_state["field_gradient"], jnp.zeros((3, 3)))

    def test_boundary_input_spec(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        spec = node.boundary_input_spec()
        assert "frequency_hz" in spec
        assert "field_strength_mt" in spec

    def test_compute_boundary_fluxes(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        state = node.initial_state()
        bi = {"frequency_hz": 10.0, "field_strength_mt": 10.0}
        state = node.update(state, bi, 0.001)
        fluxes = node.compute_boundary_fluxes(state, bi, 0.001)
        assert "field_vector" in fluxes
        assert "field_gradient" in fluxes

    def test_time_accumulates(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        state = node.initial_state()
        bi = {"frequency_hz": 10.0, "field_strength_mt": 10.0}
        for _ in range(10):
            state = node.update(state, bi, 0.001)
        assert jnp.abs(state["sim_time"] - 0.01) < 1e-6  # float32 tolerance


class TestExternalMagneticFieldMetadata:
    def test_meta_set(self):
        assert ExternalMagneticFieldNode.meta is not None
        assert ExternalMagneticFieldNode.meta.algorithm_id == "MIME-NODE-001"

    def test_mime_meta_set(self):
        assert ExternalMagneticFieldNode.mime_meta is not None
        assert ExternalMagneticFieldNode.mime_meta.role.value == "external_apparatus"

    def test_validate_consistency(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        errors = node.validate_mime_consistency()
        assert errors == [], f"Consistency errors: {errors}"

    def test_commandable_fields(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        cf = node.commandable_fields()
        assert "frequency_hz" in cf
        assert "field_strength_mt" in cf

    def test_requires_halo_false(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        assert node.requires_halo is False


class TestExternalMagneticFieldJAX:
    def test_jit_traceable(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        state = node.initial_state()
        bi = {"frequency_hz": jnp.array(10.0), "field_strength_mt": jnp.array(10.0)}
        jitted = jax.jit(node.update)
        new_state = jitted(state, bi, 0.001)
        assert new_state["field_vector"].shape == (3,)

    def test_grad_wrt_field_strength(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        state = node.initial_state()

        def field_magnitude(strength_mt):
            bi = {"frequency_hz": jnp.array(10.0), "field_strength_mt": strength_mt}
            s = node.update(state, bi, 0.001)
            return jnp.linalg.norm(s["field_vector"])

        grad_fn = jax.grad(field_magnitude)
        g = grad_fn(jnp.array(10.0))
        # d|B|/d(strength_mt) = 1e-3 (mT to T conversion)
        assert jnp.abs(g - 1e-3) < 1e-6

    def test_vmap_over_frequencies(self):
        node = ExternalMagneticFieldNode("field", 0.001)
        state = node.initial_state()
        freqs = jnp.array([1.0, 10.0, 50.0, 100.0])

        def run_at_freq(f):
            bi = {"frequency_hz": f, "field_strength_mt": jnp.array(10.0)}
            return node.update(state, bi, 0.01)["field_vector"]

        results = jax.vmap(run_at_freq)(freqs)
        assert results.shape == (4, 3)
        # All should have same magnitude
        mags = jnp.linalg.norm(results, axis=1)
        assert jnp.allclose(mags, 0.01, atol=1e-8)
