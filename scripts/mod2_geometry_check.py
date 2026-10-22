#!/usr/bin/env python3
"""MOD-2 geometry de-risk: find a dual-RPM placement that CANCELS the net field
gradient ∇B at the body while MATCHING |B| of the single-RPM setup.

D1 second-half (gradient vs gradient-cancelled step-out) needs a 'gradient-cancelled'
field at matched magnitude. A symmetric pair of RPMs at ±standoff, each with half the
single moment, gives net B = single (matched) and net ∇B ≈ 0 by mirror symmetry. This
script verifies that numerically with the real PermanentMagnetNode field model.

Run: python scripts/mod2_geometry_check.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import jax.numpy as jnp
from mime.nodes.actuation.permanent_magnet import PermanentMagnetNode


def field_at(node, magnet_pos, target=(0., 0., 0.), quat=(1., 0., 0., 0.)):
    pose = jnp.array([*magnet_pos, *quat])
    st = node.initial_state()
    out = node.update(st, {"magnet_pose_world": pose,
                           "target_position_world": jnp.array(target)}, node.delta_t)
    return np.asarray(out["field_vector"]), np.asarray(out["field_gradient"])


def main():
    dt = 5e-4
    standoff = 0.15
    m_single = 18.89
    common = dict(magnetization_axis_in_body=(1, 0, 0), magnet_radius_m=17.5e-3,
                  magnet_length_m=20e-3, field_model="point_dipole",
                  earth_field_world_t=(0, 0, 0))
    # single RPM above the pipe at +z standoff
    n1 = PermanentMagnetNode("m", dt, dipole_moment_a_m2=m_single, **common)
    B_s, G_s = field_at(n1, (0, 0, standoff))
    # dual: two half-moment RPMs at ±z standoff, same orientation
    n2 = PermanentMagnetNode("m2", dt, dipole_moment_a_m2=m_single / 2, **common)
    B_a, G_a = field_at(n2, (0, 0, +standoff))
    B_b, G_b = field_at(n2, (0, 0, -standoff))
    B_d, G_d = B_a + B_b, G_a + G_b

    print("=" * 66)
    print("MOD-2 gradient-cancellation geometry check (±z half-moment pair)")
    print("=" * 66)
    print(f"  single RPM (m={m_single}) at +{standoff*100:.0f} cm:")
    print(f"    |B| = {np.linalg.norm(B_s)*1e3:.4f} mT   |∇B| = {np.linalg.norm(G_s):.4e} T/m")
    print(f"  dual RPM (m={m_single/2} each) at ±{standoff*100:.0f} cm:")
    print(f"    |B| = {np.linalg.norm(B_d)*1e3:.4f} mT   |∇B| = {np.linalg.norm(G_d):.4e} T/m")
    print("-" * 66)
    bmatch = np.linalg.norm(B_d) / np.linalg.norm(B_s)
    gcancel = np.linalg.norm(G_d) / np.linalg.norm(G_s)
    print(f"  |B| match (dual/single)        = {bmatch:.4f}   (target 1.000)")
    print(f"  ∇B cancellation (dual/single)  = {gcancel:.4e}   (target ≪ 1)")
    # also check a few moment orientations (rotating field) — cancellation must hold
    print("  ∇B cancellation across moment orientations (rotating field):")
    for ang in (0, 30, 60, 90):
        a = np.radians(ang)
        q = (np.cos(a/2), 0.0, 0.0, np.sin(a/2))   # rotate moment about world-z
        _, Ga = field_at(n2, (0, 0, +standoff), quat=q)
        _, Gb = field_at(n2, (0, 0, -standoff), quat=q)
        _, Gs = field_at(n1, (0, 0, standoff), quat=q)
        print(f"    moment @ {ang:2d}°: ∇B_dual/∇B_single = "
              f"{np.linalg.norm(Ga+Gb)/np.linalg.norm(Gs):.4e}")
    print("=" * 66)
    verdict = "PASS" if (abs(bmatch-1) < 0.02 and gcancel < 0.05) else "REVIEW"
    print(f"  VERDICT: {verdict}  (PASS = |B| within 2% and ∇B < 5% of single)")


if __name__ == "__main__":
    main()
