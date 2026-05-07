"""T1 — Rhie-Chow checkerboard suppression.

A correctly-functioning Rhie-Chow correction is the only mechanism on a
collocated grid that lets the projection step damp the pressure
checkerboard mode. Naive linear face interpolation has the
checkerboard mode in its null space (avg(+1,-1) = 0) and so cell-
centred Green-Gauss gradient is identically zero — making the mode
invisible to standard pressure-correction. This test verifies:

  * Cell-centred ``grad(p_check)`` is identically zero (only true when
    boundaries are periodic — otherwise extrapolation contributes).
  * RMS of the *naive* face flux (= ``avg(u) · Sf - D · avg(grad p) · Sf``)
    is zero whereas Rhie-Chow's RMS is nonzero — i.e. RC produces the
    face-flux signal the projection step needs to damp the mode.
  * One PISO pressure-correction step on this RC flux drives the
    checkerboard p mode to numerical zero (factor ≪ 1e-3).

Reference: Rhie & Chow (1983) AIAA J. 21(11) 1525-1532. Moukalled,
Mangani & Darwish (2016) §15.6.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.fvm import make_cartesian_mesh_2d
from mime.nodes.environment.fvm.operators import (
    grad_green_gauss, face_velocity_rhie_chow,
    momentum_diagonal_uniform_cartesian, divergence_face_flux,
)
from mime.nodes.environment.fvm.pressure import make_pressure_solver


@pytest.mark.gpu
def test_rhie_chow_suppresses_checkerboard():
    N = 32
    L = 1.0
    mesh = make_cartesian_mesh_2d(
        N, N, L, L, periodic_x=True, periodic_y=True,
    )

    u = jnp.zeros((mesh.N_cells, 2), dtype=mesh.V.dtype)
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    p_check = jnp.asarray(((-1.0) ** (ii + jj)).reshape(-1),
                          dtype=mesh.V.dtype)

    a_p = momentum_diagonal_uniform_cartesian(
        mesh, nu=0.01, rho=1.0,
        F_face=jnp.zeros(mesh.N_faces, dtype=mesh.V.dtype),
        dt=None,
    )
    grad_p_cell = grad_green_gauss(p_check, mesh)
    # Cell-centred Green-Gauss gradient kills checkerboard on uniform
    # periodic mesh.
    assert float(jnp.max(jnp.abs(grad_p_cell))) < 1e-5

    D_bar = jnp.mean(mesh.V / a_p)
    naive_face_u = -D_bar * 0.5 * (
        grad_p_cell[mesh.owner] + grad_p_cell[mesh.neighbour]
    )
    F_naive = jnp.einsum("fd,fd->f", naive_face_u, mesh.Sf)
    rms_naive = float(jnp.sqrt(jnp.mean(F_naive ** 2)))

    rc_face_u = face_velocity_rhie_chow(u, p_check, grad_p_cell, a_p, mesh)
    F_rc = jnp.einsum("fd,fd->f", rc_face_u, mesh.Sf)
    rms_rc = float(jnp.sqrt(jnp.mean(F_rc ** 2)))

    # Naive RMS should be effectively zero (no checkerboard signal),
    # Rhie-Chow should produce a substantial flux ~ checkerboard ampl.
    assert rms_naive < 1e-5, f"naive flux unexpectedly nonzero: {rms_naive}"
    assert rms_rc > 1e-3, f"Rhie-Chow flux too small: {rms_rc}"

    # One pressure-correction step should kill the checkerboard mode.
    pres_solver = make_pressure_solver(mesh, bc=("periodic", "periodic"))
    div_F_rc = divergence_face_flux(F_rc, mesh)
    p_prime = pres_solver(div_F_rc / D_bar)
    p_new = p_check + p_prime - jnp.mean(p_prime)
    rms_after = float(jnp.sqrt(jnp.mean(p_new ** 2)))
    rms_before = float(jnp.sqrt(jnp.mean(p_check ** 2)))
    # Damping factor should be < 1e-3 (in fact ~1e-7 in float32).
    assert rms_after / rms_before < 1e-3, (
        f"checkerboard not damped: before={rms_before:g}, "
        f"after={rms_after:g}"
    )
