"""SI-native confined BEM — physical drag validated against de Jongh swim speed.

The confined Liron–Shahar wall table is **dimensionless** (body radius ≡ 1, R_cyl ≡
the confinement ratio). ``StokesletFluidNode(..., length_scale=R_body)`` makes the
node **SI-native**: it normalises the body by ``length_scale`` for the table-based
assembly but keeps the velocity BC and the output force/torque/traction in SI — so
the rest of the graph (arm, magnet, FVM, body) is plain SI and never sees the
non-dimensionalisation.

This pins that the SI scaling is correct by the one number with an experimental
anchor: the **force-free swim speed** of the de Jongh FL-9 screw at 10 Hz, which the
paper reports at ~3 mm/s. We build the SI confined resistance, extract the force-free
axial speed, and check it lands in the de Jongh band — the same check
``scene/_compare_mlp_bem.py`` applies to the MLP surrogate.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.environment.stokeslet.dejongh_geometry import dejongh_fl_mesh
from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
from mime.nodes.environment.stokeslet.interface import create_interface_mesh
from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table

pytestmark = [pytest.mark.x64, pytest.mark.slow]

# de Jongh FL-9 in a 1/4" tube: R_body = 1.56 mm, R_vessel = 3.175 mm ⇒ ratio 2.035.
_R_BODY_M = 1.56e-3
_MU = 1e-3
_TABLE = (Path(__file__).resolve().parents[2]
          / "data" / "dejongh_benchmark" / "wall_tables" / "wall_R2.035.npz")


def _si_mesh():
    """FL-9 surface mesh in metres (the generator is mm-numeric), z-centred."""
    m = dejongh_fl_mesh(9, n_theta=24, n_zeta=36)
    P = (np.asarray(m.points) - np.asarray(m.points).mean(0)) * 1e-3   # mm → m
    return dataclasses.replace(m, points=P, weights=np.asarray(m.weights) * 1e-6)


@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
def test_si_confined_bem_recovers_dejongh_swim_speed():
    mesh = _si_mesh()
    table = load_wall_table(str(_TABLE))
    bem = StokesletFluidNode(
        "bem", 5e-4, mu=_MU, body_mesh=mesh,
        interface_mesh=create_interface_mesh(radius=1.8, n_refine=1),
        wall_table=table, R_cyl=float(table.R_cyl), length_scale=_R_BODY_M)

    nb = mesh.n_points

    def drag(U, om):
        st = bem.initial_state()
        ns = bem.update(st, {"body_velocity": jnp.asarray(U, float),
                             "body_angular_velocity": jnp.asarray(om, float),
                             "body_orientation": jnp.array([1.0, 0, 0, 0]),
                             "background_flow": jnp.zeros((nb, 3))}, 5e-4)
        return np.asarray(ns["drag_force"]), np.asarray(ns["drag_torque"])

    # SI 6×6 resistance from unit translations / rotations.
    R = np.zeros((6, 6))
    for i in range(3):
        F, T = drag(np.eye(3)[i], np.zeros(3)); R[:3, i] = F; R[3:, i] = T
    for i in range(3):
        F, T = drag(np.zeros(3), np.eye(3)[i]); R[:3, 3 + i] = F; R[3:, 3 + i] = T

    # translation resistance is physical SI for a mm body (Stokes 6πμa ~ 3e-5).
    assert 1e-5 < abs(R[2, 2]) < 5e-4, R[2, 2]

    # force-free swim about the helical (z) axis at 10 Hz: F_z = 0 ⇒ U_z = -(R_FW/R_FU)·ω
    w = 2 * np.pi * 10.0
    U_z_mm_s = -(R[2, 5] / R[2, 2]) * w * 1e3
    print(f"\n[SI confined BEM] R_FU_zz={R[2,2]:.3e} N·s/m, "
          f"force-free swim={U_z_mm_s:.3f} mm/s (de Jongh ~3 mm/s @10Hz)")
    # within the de Jongh band — the SI normalisation/scale-back is correct.
    assert 1.5 < U_z_mm_s < 5.0
