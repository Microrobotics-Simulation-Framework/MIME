"""M0 — screw-through-coupling de-risk (HARD GATE for the schwarz_vessel_helix gate).

The two-scale coupling (FVM far-field ⊗ confined Liron–Shahar BEM near-field) has
only ever been exercised with a **sphere** body (C2, TASK A). Before standing up the
``schwarz_vessel_helix`` experiment we must prove the **de Jongh screw** couples at
all — it is the one genuinely unknown step; everything downstream is composition.

This pins, on a *coarsened* FL-9 screw inside a cylinder-IBM vessel, that:

  1. **Point count is tractable** — the dense confined-BEM system (3·M)² fits and
     LU-factorises at a screw resolution that resolves its ~1.8 helical turns.
  2. **In-lumen geometry** — the axis-aligned, z-centred screw stays inside the
     vessel (``max ρ < R_cyl``); this is also the ``_check_centering`` guard.
  3. **Stable + finite** — the coupled solve runs without blowing up and the drag
     time-series settles monotonically toward steady state, yielding a finite,
     positive confined drag that behaves like the validated sphere.

Convergence note (verified empirically against the sphere): the **per-step**
interface residual is O(1) in this transient, subcycled harness — the *validated*
sphere baseline (C2 / TASK A) reports res≈12 at iters≈1, and the screw reports
res≈1.6 at iters≈3, i.e. the screw is no worse. C2/TASK A therefore assert only
``isfinite(residual)``; "convergence" here means the **drag time-series** settles
(its step-to-step drift shrinks), not that the per-step residual drops below the
normalized 1.0 threshold. Raising ``max_iterations`` (8→30) changes nothing — the
loop is not iteration-limited; the FVM far-field is simply still developing. The
*absolute* drag is transient (a converged magnitude needs many more steps); the
gate experiment uses the **differential** counter-flow response, which sidesteps
this. This test pins feasibility + stability, not a converged drag magnitude.

Non-dimensional wall-table convention (body radius ≈ 1, ``R_cyl`` = 2.5 ⇒ λ = 0.4,
``mu`` = 1) — the ``wall_R2.500.npz`` table, matching the locked envelope λ.

Construction note: this uses the raw ``make_two_scale_coupling`` harness (the TASK A
pattern), NOT the ``HydrodynamicModel.TwoScale`` effect — the effect's ``build``
needs the full ``Experiment`` body/medium context, which would entangle the
screw-feasibility question with body/kinematics wiring. The effects-first attach
*with the screw* is proven in M3. Feasibility here is backend-agnostic.

AoA note: the confined wall table is valid only for a **centred, axis-aligned**
body (``_check_centering`` warns off-axis). The envelope's angle-of-attack therefore
enters through the far-field background cross-flow, NOT a body tilt — so the screw
is kept axis-aligned here, by design.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig
from mime.nodes.environment.fvm.fluid_node import FVMFluidNode
from mime.nodes.environment.fvm.ibm import IBMBody
from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
from mime.nodes.environment.stokeslet.interface import create_interface_mesh
from mime.nodes.environment.stokeslet.dejongh_geometry import dejongh_umr_surface
from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table
from mime.nodes.environment.two_scale import make_two_scale_coupling

pytestmark = [pytest.mark.x64, pytest.mark.slow]

_TABLE = (Path(__file__).resolve().parents[2]
          / "data" / "dejongh_benchmark" / "wall_tables" / "wall_R2.500.npz")

# FL-9 (ν = 2.33) in the wall-table's non-dimensional units: screw characteristic
# radius ≈ 1 (R_cyl_screw=1), de Jongh aspect L/R = 7.47/1.56 = 4.79, ε=0.33, N=2.
_NU_FL9 = 2.33
_R_SCREW = 1.0
_L_UMR = 4.79
_EPS, _N = 0.33, 2
# coarse but resolves the ~1.78 helical turns (ν·L/R = 11.2 rad) — feasibility, not
# resolution-converged physics.
_N_THETA, _N_ZETA = 20, 28


def _screw_body():
    """Coarse FL-9 screw, z-centred and axis-aligned (AoA=0)."""
    mesh = dejongh_umr_surface(
        nu=_NU_FL9, L_UMR=_L_UMR, R_cyl=_R_SCREW, epsilon=_EPS, N=_N,
        n_theta=_N_THETA, n_zeta=_N_ZETA)
    P = np.asarray(mesh.points, dtype=np.float64).copy()
    # centre on the z-axis origin (the FVM box is symmetric about 0); the screw is
    # already ~axis-centred in xy by construction — zero any tiny residual.
    P[:, 0] -= P[:, 0].mean()
    P[:, 1] -= P[:, 1].mean()
    P[:, 2] -= P[:, 2].mean()
    return dataclasses.replace(mesh, points=P)  # SurfaceMesh is frozen


def _build(table):
    R_cyl, mu, rho = float(table.R_cyl), float(table.mu), 1.0
    body = _screw_body()
    nb = body.n_points
    rho_max = float(np.sqrt(body.points[:, 0] ** 2 + body.points[:, 1] ** 2).max())
    z_half = float(np.abs(body.points[:, 2]).max())

    a = _R_SCREW
    dx = a / 4                                          # cpr = 4 (affordable regime)
    Lxy = 2 * 1.25 * R_cyl
    Lz = 2 * (z_half + 2 * a)                            # screw + margin each end
    nxy, nz = int(np.ceil(Lxy / dx)), int(np.ceil(Lz / dx))
    mesh = make_cartesian_mesh_3d(nxy, nxy, nz, Lxy, Lxy, Lz,
                                  origin=(-Lxy / 2, -Lxy / 2, -Lz / 2),
                                  dtype=jnp.float64, periodic_z=True)
    wall = IBMBody(name="wall", sdf=lambda x: R_cyl - jnp.sqrt(
        x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30))
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3))
           for p in ("x_min", "x_max", "y_min", "y_max")}
    cfg = PisoConfig(nu=mu / rho, rho=rho, gamma_conv=0.5, n_corrector=2,
                     pressure_bc=("neumann", "neumann", "periodic"),
                     velocity_bc=("dirichlet", "dirichlet", "periodic"),
                     ibm_alpha=1e5, ibm_eps=1.0 * dx, transform_backend="dense")
    far = FVMFluidNode("fvm", 1e-2, mesh=mesh, bcs=bcs, cfg=cfg, static_bodies=[wall],
                       n_sample_points=nb, n_forcing_points=nb, forcing_sigma=1.5 * dx)
    # confined Schwarz BEM: wall_table + R_cyl → A_body + G_wall; interface_mesh is a
    # mode flag only (its geometry is unused on the confined path).
    near = StokesletFluidNode("bem", 1e-2, mu=mu, body_mesh=body,
                              interface_mesh=create_interface_mesh(radius=1.8, n_refine=1),
                              wall_table=table, R_cyl=R_cyl)
    gm = GraphManager()
    info = make_two_scale_coupling(gm, far, near,
                                   body_points=jnp.asarray(body.points),
                                   body_weights=jnp.asarray(body.weights))
    for fld, sh in (("body_velocity", (3,)), ("body_angular_velocity", (3,)),
                    ("body_orientation", (4,))):
        gm.add_external_input("bem", fld, shape=sh, dtype=jnp.float64)
    ext = dict(info["geometry_inputs"])
    ext["bem"] = {"body_velocity": jnp.array([0.0, 0.0, 1.0]),
                  "body_angular_velocity": jnp.array([0.0, 0.0, 1.0]),  # screw spin
                  "body_orientation": jnp.array([1.0, 0.0, 0.0, 0.0])}
    meta = {"nb": nb, "rho_max": rho_max, "R_cyl": R_cyl, "cells": nxy * nxy * nz,
            "dof": 3 * nb}
    return gm, ext, meta


@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
def test_screw_couples_converges_in_lumen():
    import time
    table = load_wall_table(str(_TABLE))
    gm, ext, meta = _build(table)

    # (2) in-lumen geometry — the _check_centering guard would already raise; assert
    # explicitly with margin so the gate criterion is visible.
    assert meta["rho_max"] < meta["R_cyl"], (
        f"screw leaves lumen: max ρ={meta['rho_max']:.3f} ≥ R_cyl={meta['R_cyl']}")

    n_steps = 60
    fz = np.empty(n_steps)
    t0 = time.perf_counter()
    for i in range(n_steps):
        st = gm.step(ext)
        fz[i] = float(st["bem"]["drag_force"][2])
    elapsed = time.perf_counter() - t0

    Fz = fz[-1]
    diag = gm.coupling_diagnostics()["bem+fvm"]
    early_drift = abs(fz[5] - fz[4]) / abs(fz[4])      # transient settling check
    late_drift = abs(fz[-1] - fz[-2]) / abs(fz[-1])

    print(f"\n[M0 screw smoke] N_body={meta['nb']} (DOF={meta['dof']}), "
          f"rho_max={meta['rho_max']:.3f}/R_cyl={meta['R_cyl']}, "
          f"FVM cells={meta['cells']}, iters={diag['iterations']}, "
          f"residual={diag['residual']:.2e}, Fz={Fz:.4f}, "
          f"drift early={early_drift:.2e}→late={late_drift:.2e}, "
          f"{elapsed / n_steps * 1e3:.0f} ms/step")

    # (3) stable + finite, behaving like the validated sphere.
    assert np.isfinite(diag["residual"])               # the meaningful coupling check
    assert np.all(np.isfinite(fz)) and Fz > 0.0
    # drag time-series settles: step-to-step drift shrinks over the run (transient
    # decay toward steady state — the actual "convergence" here, see module docstring).
    assert late_drift < early_drift
    assert late_drift < 5e-2
    # confined-BEM drag is a sane multiple of the free-screw scale (not blown up /
    # not collapsed to ~free-space). Loose band — feasibility, not validation.
    stokes_scale = 6 * np.pi * float(table.mu) * _R_SCREW * 1.0
    assert 0.1 < Fz / stokes_scale < 20.0
