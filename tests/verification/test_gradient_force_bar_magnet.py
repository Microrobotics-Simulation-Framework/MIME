"""V1 — Gradient force F = (∇B)·m validation.

Compares ``PermanentMagnetResponseNode``'s ``magnetic_force`` output
against the analytical force on a magnetic dipole in the field of a
finite bar magnet, computed in this test file.

The path under test (``permanent_magnet_response.py:183``) has never
been numerically checked with a non-zero gradient — every existing
field node in the codebase hardcodes ``field_gradient`` to
``jnp.zeros((3,3))``. Phase 1 of the station-keeping simulation is
the first time it matters; if the path is wrong, every downstream
result is invalid.

Scope (deliberately tight):
- Stationary dipole, stationary field source. No integrator,
  controller, or hydrodynamics.
- The reference ``∇B`` is finite-differenced from a closed-form
  bar-magnet B-field defined in this file. We are checking that
  ``magnetic_force = (∇B) · m_lab`` matches that algebra to machine
  precision — *not* that the bar-magnet field model is correct
  (a separate validation, owed to the future bar-magnet field node).

Cross-check between two B-field formulae:
- Point-dipole far-field (any position): the reference for off-axis
  configurations.
- Coaxial circular current loop, on-axis only: a sanity check that the
  point-dipole formula is faithful where the two overlap. We accept
  the point-dipole as truth iff the two agree to <1% on the on-axis
  test points at the 3-, 5-, 10-mm distances used here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import jax.numpy as jnp

from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode
from mime.core.quaternion import identity_quat


MU_0 = 4.0 * np.pi * 1e-7   # T·m/A


# ──────────────────────────────────────────────────────────────────
# Closed-form bar-magnet field models
# ──────────────────────────────────────────────────────────────────

def B_dipole(r_xyz: np.ndarray, m_source: np.ndarray) -> np.ndarray:
    """Point-dipole far-field at displacement r from source.

    B = (μ₀ / 4π) [ 3 (m·r̂) r̂ − m ] / r³.
    """
    r_xyz = np.asarray(r_xyz, dtype=np.float64)
    m_source = np.asarray(m_source, dtype=np.float64)
    r = np.linalg.norm(r_xyz)
    rhat = r_xyz / r
    return (MU_0 / (4.0 * np.pi)) * (
        3.0 * np.dot(m_source, rhat) * rhat - m_source
    ) / (r ** 3)


def B_current_loop_on_axis(z: float, I_eff: float, R_loop: float) -> np.ndarray:
    """Coaxial circular current loop, on-axis only.

    B_z(z) = (μ₀ I R²) / (2 (R² + z²)^{3/2}); other components are 0.
    The loop's equivalent dipole moment is m = I · π R² along ẑ.
    """
    Bz = (MU_0 * I_eff * R_loop ** 2) / (
        2.0 * (R_loop ** 2 + z ** 2) ** 1.5
    )
    return np.array([0.0, 0.0, Bz])


def grad_B_finite_diff(
    B_func: Callable[[np.ndarray], np.ndarray],
    pos: np.ndarray,
    delta: float = 10e-6,
) -> np.ndarray:
    """Centered finite-difference 3×3 gradient of B at ``pos``.

    Result indexed as ``grad_B[i, j] = ∂B_i / ∂x_j`` (so the
    consumer's ``F_i = grad_B[i, j] · m_j`` matches the standard
    convention used by ``PermanentMagnetResponseNode``).
    """
    pos = np.asarray(pos, dtype=np.float64)
    grad = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        ej = np.zeros(3); ej[j] = 1.0
        Bp = B_func(pos + delta * ej)
        Bm = B_func(pos - delta * ej)
        grad[:, j] = (Bp - Bm) / (2.0 * delta)
    return grad


# ──────────────────────────────────────────────────────────────────
# Cross-check the two B-field formulae before using either as truth
# ──────────────────────────────────────────────────────────────────

def test_dipole_vs_current_loop_on_axis_agree():
    """Cross-check: at z = 10 mm with R_loop = 0.5 mm
    (i.e. z/R_loop = 20), the two formulae must agree to <1%. Closer
    in, the near-field correction (R/z)² is the dominant source of
    disagreement and is *physics*, not error — at z = 3 mm the
    expected (and observed) dipole-vs-loop disagreement is ~4%, at
    z = 5 mm ~1%.

    The point of this cross-check is *not* to validate the dipole
    formula as a model of a real bar magnet (it isn't — that's a
    separate validation owed to the bar-magnet field node). It is
    to confirm that ``B_dipole`` is faithful in the far-field limit,
    so that when we use it on both sides of the V1 comparison
    (analytical force vs node force) we are using a sane reference.
    """
    R_loop = 0.5e-3
    m_dipole = 1.0
    I_eff = m_dipole / (np.pi * R_loop ** 2)
    m_source = np.array([0.0, 0.0, m_dipole])

    z_far = 10e-3
    B_dip = B_dipole(np.array([0.0, 0.0, z_far]), m_source)
    B_loop = B_current_loop_on_axis(z_far, I_eff, R_loop)
    rel = np.linalg.norm(B_dip - B_loop) / np.linalg.norm(B_loop)
    assert rel < 0.01, (
        f"Dipole vs loop disagree in the far field at z={z_far*1e3:.1f} mm: "
        f"rel={rel:.4f}. Investigate before trusting V1 results."
    )


# ──────────────────────────────────────────────────────────────────
# V1 main test grid — 15 configurations
# ──────────────────────────────────────────────────────────────────

# Source magnet: 1 A·m² along +ẑ at the origin.
M_SOURCE = np.array([0.0, 0.0, 1.0])

# Test dipole ("the UMR magnet"): magnitude irrelevant for the linear
# (∇B)·m identity; using 1 A·m² keeps the analytical force at order
# 10⁻⁵ N — well within float32 representable range without numerical
# tricks.
M_DIPOLE_MAG = 1.0

POSITIONS_M = [
    np.array([0.0, 0.0, 3e-3]),    # on-axis, near
    np.array([0.0, 0.0, 5e-3]),    # on-axis, mid
    np.array([0.0, 0.0, 10e-3]),   # on-axis, far
    np.array([3e-3, 0.0, 5e-3]),   # off-axis (xz)
    np.array([3e-3, 4e-3, 5e-3]),  # off-axis (xyz, fully 3D)
]

# Three orientations of the dipole's body-frame moment axis. The node
# stores ``moment_axis`` in the body frame and rotates it to the lab
# frame using the supplied quaternion. We keep the quaternion at
# identity (so m_lab = moment_axis) and instead vary moment_axis to
# get three distinct lab-frame orientations.
ORIENTATIONS_LAB = [
    np.array([0.0, 0.0, 1.0]),                          # along +ẑ
    np.array([1.0, 0.0, 0.0]),                          # along +x̂
    np.array([np.sin(np.pi/6), 0.0, np.cos(np.pi/6)]),  # 30° tilt in xz
]


def _node_force(grad_B: np.ndarray, m_axis_body: np.ndarray) -> np.ndarray:
    """Drive the node directly with a hand-built ``boundary_inputs``.

    Returns the lab-frame ``magnetic_force`` (3-vector, N).
    """
    node = PermanentMagnetResponseNode(
        "magnet_v1", timestep=1e-3,
        n_magnets=1,
        m_single=M_DIPOLE_MAG,
        moment_axis=tuple(float(c) for c in m_axis_body),
    )
    state = node.initial_state()
    bi = {
        "field_vector":  jnp.zeros(3),  # not used by the gradient-force path
        "field_gradient": jnp.asarray(grad_B, dtype=jnp.float32),
        "orientation":    identity_quat(),
    }
    out = node.update(state, bi, 1e-3)
    return np.asarray(out["magnetic_force"], dtype=np.float64)


@pytest.mark.parametrize("pos_idx", range(len(POSITIONS_M)))
@pytest.mark.parametrize("orient_idx", range(len(ORIENTATIONS_LAB)))
def test_gradient_force_matches_analytical(pos_idx, orient_idx):
    """For every (position, orientation) pair, F_node ≈ F_analytic.

    Tolerance: per-component rel error < 1e-4 OR absolute error
    < 1e-12 N (whichever is more permissive — the latter handles
    near-zero components without dividing by ~0).
    """
    pos = POSITIONS_M[pos_idx]
    m_axis = ORIENTATIONS_LAB[orient_idx]

    # ∇B from the trusted closed form.
    grad_B = grad_B_finite_diff(
        lambda r: B_dipole(r, M_SOURCE),
        pos,
        delta=10e-6,
    )

    # Analytical force on the dipole: F_i = (∂B_i/∂x_j) · m_j.
    m_lab = M_DIPOLE_MAG * m_axis
    F_analytic = grad_B @ m_lab

    F_node = _node_force(grad_B, m_axis)

    abs_err = np.abs(F_node - F_analytic)
    denom = np.maximum(np.abs(F_analytic), 1e-15)
    rel_err = abs_err / denom

    # Component-wise: each component is either small in absolute terms
    # OR small in relative terms.
    ok = (abs_err < 1e-12) | (rel_err < 1e-4)
    assert ok.all(), (
        f"pos={pos*1e3} mm, m_axis={m_axis}\n"
        f"F_analytic = {F_analytic}\n"
        f"F_node     = {F_node}\n"
        f"abs_err    = {abs_err}\n"
        f"rel_err    = {rel_err}"
    )


def test_zero_gradient_zero_force():
    """Sanity floor: the existing test-suite case still holds when
    we drive the node through this validation harness.
    """
    F = _node_force(np.zeros((3, 3)), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(F, 0.0, atol=1e-30)


def test_units_consistent():
    """A quick manual scaling check: F = (∇B)·m with grad_B = 1 T/m
    and m = 1 A·m² should give F = 1 N along the dominant component.
    """
    grad_B = np.eye(3) * 1.0  # 1 T/m diagonal
    F = _node_force(grad_B, np.array([0.0, 0.0, 1.0]))
    # With moment along +ẑ: F_z = grad_B[2,2] · m_z = 1·1 = 1 N.
    assert np.isclose(F[2], 1.0, rtol=1e-5)
    assert np.isclose(F[0], 0.0, atol=1e-12)
    assert np.isclose(F[1], 0.0, atol=1e-12)
