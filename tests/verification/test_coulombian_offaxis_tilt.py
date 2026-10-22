"""coulombian_poles off-axis field-tilt correction (TASK B).

The lateral-offset field *tilt* drives the rotating-magnet step-out instability.
The pure-dipole and (scalar-corrected) current_loop models leave the off-axis
field *direction* equal to the dipole's — they cannot represent the tilt
correction. ``coulombian_poles`` now uses the true two-point-monopole (Gilbert)
field off-axis, whose direction tilts with the finite pole separation.

These tests pin: (a) far-field recovery of the dipole (on- and off-axis), and
(b) a genuine off-axis *direction* deviation from the pure dipole that grows as
the target approaches the magnet, scaling ~ (L/2d)² (finite-length).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.actuation.permanent_magnet import (
    _b_coulombian_poles, _b_point_dipole,
)

pytestmark = pytest.mark.x64

_M = jnp.array([0.0, 0.0, 1.0])      # moment along +z, |m| = 1 A·m²
_POS = jnp.zeros(3)
_R = 0.0175                          # ar4 magnet radius (m)
_L = 0.02                            # ar4 magnet length (m)


def _dir(B):
    B = np.asarray(B)
    return B / np.linalg.norm(B)


def _angle_to_dipole(target):
    Bc = _b_coulombian_poles(target, _POS, _M, jnp.asarray(_R), jnp.asarray(_L))
    Bd = _b_point_dipole(target, _POS, _M, jnp.asarray(_R), jnp.asarray(_L))
    cos = float(np.clip(np.dot(_dir(Bc), _dir(Bd)), -1.0, 1.0))
    return np.degrees(np.arccos(cos))


def test_far_field_recovers_dipole():
    # 50 magnet-lengths away, off-axis: coulombian ≈ dipole (dir + magnitude).
    target = jnp.array([20.0 * _L, 0.0, 30.0 * _L])
    Bc = np.asarray(_b_coulombian_poles(target, _POS, _M, jnp.asarray(_R), jnp.asarray(_L)))
    Bd = np.asarray(_b_point_dipole(target, _POS, _M, jnp.asarray(_R), jnp.asarray(_L)))
    assert np.allclose(Bc, Bd, rtol=1e-3)
    # on-axis far field also recovers the dipole
    on_ax = jnp.array([0.0, 0.0, 40.0 * _L])
    Bc_ax = np.asarray(_b_coulombian_poles(on_ax, _POS, _M, jnp.asarray(_R), jnp.asarray(_L)))
    Bd_ax = np.asarray(_b_point_dipole(on_ax, _POS, _M, jnp.asarray(_R), jnp.asarray(_L)))
    assert np.allclose(Bc_ax, Bd_ax, rtol=5e-3)


def test_offaxis_direction_deviates_and_decays():
    # 45° off-axis line at increasing distance d (in units of magnet length L).
    s = np.sqrt(0.5)
    ang = {}
    for d_over_L in (2.0, 4.0, 8.0):
        d = d_over_L * _L
        target = jnp.array([s * d, 0.0, s * d])
        ang[d_over_L] = _angle_to_dipole(target)

    # (a) genuine tilt deviation from the pure dipole at close range
    assert ang[2.0] > 1e-3
    # (b) the deviation decays as the target recedes ...
    assert ang[2.0] > ang[4.0] > ang[8.0]
    # (c) ... roughly quadratically (~(L/2d)²): each doubling of d shrinks the
    #     angle by ~4× (allow a generous band for the 45°-line geometry)
    assert 2.5 < ang[2.0] / ang[4.0] < 6.0
    assert 2.5 < ang[4.0] / ang[8.0] < 6.0
