"""M1 — flow-rate-driven far-field (finite-Re, Poiseuille + Womersley).

The schwarz_vessel_helix gate must prove the thing ar4_helical_drive lacks: a
**vessel flow rate** acting on the confined swimmer at **finite Re** (no longer
strictly Stokes). M0 proved the screw couples; here we drive the FVM far-field with
a controllable vessel inflow and pin that the swimmer feels it.

Set-up: the M0 coupled screw (confined wall-table BEM ⊗ Cartesian+IBM FVM far-field),
**held** (body velocity = 0), in a steady Poiseuille flow of mean velocity ``U``
(flow rate ``Q = U·πR²``) along the vessel axis. The flow-induced axial reaction
drag ``Fz`` is the **differential** counter-flow signal the envelope sweep uses —
it sidesteps the self-wake absolute-transient issue (M0 docstring).

Pins:
  1. **Zero-flow asymptote** — ``Fz(Q=0) ≈ 0`` (held body, no flow → zero BEM RHS →
     zero drag, exactly).
  2. **Responds to Q, ~linearly** — ``Fz(2U) ≈ 2·Fz(U)`` within tolerance, i.e. the
     coupled response is in the locally-linear (small-Re) regime.
  3. **Womersley path runs** — the analytical pulsatile lift produces a finite
     cycle-mean differential of the same sign/order as the steady Poiseuille point.

Sign convention (verified): a **held** body in a +z flow is equivalent to a body
moving in −z, so by the BEM reaction convention (TASK A: +z *motion* → +drag) the
flow-induced ``Fz`` is **negative**. The magnitude is the full screw-in-flow drag
(the screw sees a background of order the centreline velocity), not a small
perturbation — that is the point: the swimmer genuinely feels the vessel flow.

Re is reported (diameter-based, ``Re = U·2R/ν``); kept ≲ 25 (local-Stokes-valid).
Non-dimensional wall-table units (body radius ≈ 1, R_cyl=2.5, ν=μ/ρ=1).

The velocity sampled into the BEM background includes the lift
(``fluid_node.py`` adds ``u_lift`` back before sampling), so the counter-flow
reaches the swimmer through the coupling.
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
from mime.nodes.environment.fvm.lifting import (
    make_poiseuille_lift, make_womersley_lift_analytical)
from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
from mime.nodes.environment.stokeslet.interface import create_interface_mesh
from mime.nodes.environment.stokeslet.dejongh_geometry import dejongh_umr_surface
from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table
from mime.nodes.environment.two_scale import make_two_scale_coupling

pytestmark = [pytest.mark.x64, pytest.mark.slow]

_TABLE = (Path(__file__).resolve().parents[2]
          / "data" / "dejongh_benchmark" / "wall_tables" / "wall_R2.500.npz")

# coarser than M0 (held body, several solves) — feasibility/response, not resolution.
_NU_FL9, _R_SCREW, _L_UMR, _EPS, _N = 2.33, 1.0, 4.79, 0.33, 2
_N_THETA, _N_ZETA = 16, 22


def _screw_body():
    m = dejongh_umr_surface(nu=_NU_FL9, L_UMR=_L_UMR, R_cyl=_R_SCREW, epsilon=_EPS,
                            N=_N, n_theta=_N_THETA, n_zeta=_N_ZETA)
    P = np.asarray(m.points, np.float64).copy()
    P -= P.mean(axis=0)
    return dataclasses.replace(m, points=P)


_DX = _R_SCREW / 4


def _make_mesh(body, R_cyl):
    z_half = float(np.abs(body.points[:, 2]).max())
    Lxy = 2 * 1.25 * R_cyl
    Lz = 2 * (z_half + 2 * _R_SCREW)
    nxy, nz = int(np.ceil(Lxy / _DX)), int(np.ceil(Lz / _DX))
    return make_cartesian_mesh_3d(nxy, nxy, nz, Lxy, Lxy, Lz,
                                  origin=(-Lxy / 2, -Lxy / 2, -Lz / 2),
                                  dtype=jnp.float64, periodic_z=True)


def _build(table, body, mesh, lift):
    """Coupled held screw on a given mesh with a given far-field lift (None → quiescent)."""
    R_cyl, mu, rho, dx, nb = float(table.R_cyl), float(table.mu), 1.0, _DX, body.n_points
    wall = IBMBody(name="wall", sdf=lambda x: R_cyl - jnp.sqrt(
        x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30))
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3))
           for p in ("x_min", "x_max", "y_min", "y_max")}
    cfg = PisoConfig(nu=mu / rho, rho=rho, gamma_conv=0.5, n_corrector=2,
                     pressure_bc=("neumann", "neumann", "periodic"),
                     velocity_bc=("dirichlet", "dirichlet", "periodic"),
                     ibm_alpha=1e5, ibm_eps=1.0 * dx, transform_backend="dense")
    far = FVMFluidNode("fvm", 1e-2, mesh=mesh, bcs=bcs, cfg=cfg, static_bodies=[wall],
                       lifting=lift, n_sample_points=nb, n_forcing_points=nb,
                       forcing_sigma=1.5 * dx)
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
    ext["bem"] = {"body_velocity": jnp.zeros(3),            # HELD body
                  "body_angular_velocity": jnp.zeros(3),
                  "body_orientation": jnp.array([1.0, 0.0, 0.0, 0.0])}
    return gm, ext


def _drag_z(gm, ext, n=45):
    for _ in range(n):
        st = gm.step(ext)
    return float(st["bem"]["drag_force"][2])


@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
def test_held_screw_axial_drag_responds_to_flow_rate():
    table = load_wall_table(str(_TABLE))
    R_cyl, nu = float(table.R_cyl), float(table.mu) / 1.0
    body = _screw_body()

    U = 1.0                                            # Re = U·2R/ν = 5  (≲25, valid)
    Re = U * 2 * R_cyl / nu
    Q = U * np.pi * R_cyl ** 2
    mesh = _make_mesh(body, R_cyl)

    # quiescent (Q=0): zero-flow asymptote
    Fz0 = _drag_z(*_build(table, body, mesh, None))
    # Poiseuille at U and 2U (same mesh geometry; lift differs)
    Fz1 = _drag_z(*_build(table, body, mesh,
                          make_poiseuille_lift(mesh, R_pipe=R_cyl, U_mean=U, axis=2)))
    Fz2 = _drag_z(*_build(table, body, mesh,
                          make_poiseuille_lift(mesh, R_pipe=R_cyl, U_mean=2 * U, axis=2)))

    print(f"\n[M1 flow response] Re={Re:.1f}, Q={Q:.3f} (U={U}), "
          f"Fz(0)={Fz0:.4f}, Fz(U)={Fz1:.4f}, Fz(2U)={Fz2:.4f}, "
          f"ratio Fz(2U)/Fz(U)={Fz2 / Fz1:.3f}")

    scale = abs(Fz1)
    # (1) zero-flow asymptote: held body with no flow → ~no drag
    assert abs(Fz0) < 0.1 * scale
    # (2a) responds: flow induces a clearly nonzero axial drag
    assert scale > 0.0 and abs(Fz1 - Fz0) > 0.3 * scale
    # (2b) sign: held-in-+z-flow ≡ moving-in-−z ⇒ Fz < 0 (see module docstring);
    #      both flow rates give the same (negative) sign.
    assert Fz1 < 0.0 and Fz2 < 0.0
    # (2c) ~linear in Q (locally-linear small-Re regime): doubling U doubles Fz
    assert abs((Fz2 - Fz0) / (Fz1 - Fz0) - 2.0) < 0.35


@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
def test_womersley_pulsatile_path_runs_and_recovers_poiseuille_sign():
    """Pulsatile (Womersley) far-field runs and its cycle-mean differential drag is
    finite, co-directed, and the same order as the steady Poiseuille point."""
    table = load_wall_table(str(_TABLE))
    R_cyl, nu = float(table.R_cyl), float(table.mu) / 1.0
    body = _screw_body()
    U_dc = 1.0
    mesh = _make_mesh(body, R_cyl)

    # steady reference
    lift_p = make_poiseuille_lift(mesh, R_pipe=R_cyl, U_mean=U_dc, axis=2)
    Fz_steady = _drag_z(*_build(table, body, mesh, lift_p))

    # Womersley: small oscillation about the same DC. The node dt is 1e-2; pick ω so
    # one period ≈ 20 steps (cheap) — the *path* runs and stays bounded; AC accuracy
    # is test_fvm_womersley's job, not this gate's.
    dt, steps_per_period = 1e-2, 20
    omega = 2 * np.pi / (steps_per_period * dt)
    Wo = R_cyl * np.sqrt(omega / nu)
    lift_w = make_womersley_lift_analytical(
        mesh, R_pipe=R_cyl, U_mean_dc=U_dc, U_mean_amp=0.3 * U_dc,
        omega=omega, nu=nu, axis=2)
    gmw, extw = _build(table, body, mesh, lift_w)

    n_steps = 4 * steps_per_period
    fz = []
    for k in range(n_steps):
        st = gmw.step(extw)
        if k >= n_steps - steps_per_period:            # cycle-mean over the last period
            fz.append(float(st["bem"]["drag_force"][2]))
    Fz_cycle_mean = float(np.mean(fz))

    print(f"\n[M1 womersley] Wo={Wo:.1f}, Fz_steady={Fz_steady:.4f}, "
          f"Fz_womersley_cyclemean={Fz_cycle_mean:.4f}, "
          f"ratio={Fz_cycle_mean / Fz_steady:.3f}")

    assert np.all(np.isfinite(fz))
    assert Fz_cycle_mean < 0.0                         # same sign as steady (negative)
    # same order as the steady Poiseuille DC (cycle-mean of an oscillation about it).
    assert 0.4 < Fz_cycle_mean / Fz_steady < 1.6
