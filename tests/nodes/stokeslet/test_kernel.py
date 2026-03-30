"""Tests for the regularised Stokeslet tensor kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.stokeslet.kernel import stokeslet_tensor


class TestStokesletTensor:
    def test_symmetry(self):
        """S_jk(x, x0) should be symmetric in j,k."""
        x = jnp.array([1.0, 2.0, 3.0])
        x0 = jnp.array([0.5, 0.1, -0.3])
        S = stokeslet_tensor(x, x0, epsilon=0.1)
        np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_no_singularity_at_coincident_points(self):
        """S at x = x0 should be finite (regularised)."""
        x = jnp.array([1.0, 2.0, 3.0])
        S = stokeslet_tensor(x, x, epsilon=0.1)
        assert jnp.all(jnp.isfinite(S))
        # At x = x0: S_jk = delta_jk * 2*eps^2 / eps^3 = 2*delta_jk / eps
        expected_diag = 2.0 / 0.1
        np.testing.assert_allclose(jnp.diag(S), expected_diag, rtol=1e-6)

    def test_off_diagonal_at_coincident(self):
        """Off-diagonal elements should be zero at x = x0."""
        x = jnp.array([0.0, 0.0, 0.0])
        S = stokeslet_tensor(x, x, epsilon=0.5)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(float(S[i, j])) < 1e-12

    def test_decay_with_distance(self):
        """S should decay as ~1/r for large r."""
        x0 = jnp.zeros(3)
        eps = 0.01
        S_near = stokeslet_tensor(jnp.array([1.0, 0.0, 0.0]), x0, eps)
        S_far = stokeslet_tensor(jnp.array([10.0, 0.0, 0.0]), x0, eps)
        # Diagonal elements should scale roughly as 1/r
        ratio = float(S_near[0, 0] / S_far[0, 0])
        assert 8.0 < ratio < 12.0  # should be ~10

    def test_vmap_over_targets(self):
        """vmap over target points should produce correct shapes."""
        targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        x0 = jnp.zeros(3)
        S_batch = jax.vmap(stokeslet_tensor, in_axes=(0, None, None))(
            targets, x0, 0.1,
        )
        assert S_batch.shape == (3, 3, 3)

    def test_double_vmap(self):
        """Double vmap (targets × sources) should produce (M, N, 3, 3)."""
        targets = jnp.ones((5, 3))
        sources = jnp.zeros((7, 3))
        S_all = jax.vmap(
            jax.vmap(stokeslet_tensor, in_axes=(None, 0, None)),
            in_axes=(0, None, None),
        )(targets, sources, 0.1)
        assert S_all.shape == (5, 7, 3, 3)

    def test_jit_compatible(self):
        """Kernel should be JIT-compilable."""
        x = jnp.array([1.0, 0.0, 0.0])
        x0 = jnp.zeros(3)
        S_jit = jax.jit(stokeslet_tensor)(x, x0, 0.1)
        S_ref = stokeslet_tensor(x, x0, 0.1)
        np.testing.assert_allclose(S_jit, S_ref, atol=1e-12)
