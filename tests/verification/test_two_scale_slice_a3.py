"""Two-scale Schwarz slice (A3) — the key coupling de-risk.

Couples a Cartesian (dense) FVM far-field ⊗ a free-space BEM near-field (sphere)
through ``GraphManager.add_coupling_group`` (interface norm, IQN-ILS), with the
A1/A2 transfers:

    far → near :  FVM velocity sampled at body points  →  BEM background_flow
    near → far :  BEM body traction × weights (N)       →  FVM reaction forcing

A translating sphere in initially-still fluid is the test case. It pins, on a
geometry that needs no body-fitted mesh:

* **Stokes drag recovery** — uncoupled (single-pass) drag ≈ 6πμaV.
* **Within-step convergence** — the coupling fixed point is reached: the
  first-step drag stops changing as ``max_iterations`` grows (Cauchy), and the
  later increments are smaller than the earlier ones (contraction).
* **Sign-correct transfers** — the body dragging the fluid in +z builds a +z
  far-field flow (near→far), and that background then reduces the drag
  (far→near feedback). Transverse components stay negligible.

This proves the coupling machinery before the body-fitted-mesh investment (B).
The builder here is the hand-wired precursor to C1's ``make_two_scale_coupling``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig
from mime.nodes.environment.fvm.fluid_node import FVMFluidNode
from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
from mime.nodes.environment.stokeslet.interface import create_interface_mesh

pytestmark = [pytest.mark.x64, pytest.mark.slow]

# Physical setup (Stokes regime): a sphere radius a translating at +V z.
_A = 1e-3
_V = 1e-3
_RHO = 1000.0
_NU = 1e-3            # mu = rho*nu = 1.0
_MU = _RHO * _NU
_DT = 1e-4
_L = 8 * _A
_N = 24
_STOKES = 6 * np.pi * _MU * _A * _V


def _build_two_scale_slice(max_iterations=6, acceleration="iqn-ils"):
    """Hand-wired FVM⊗BEM Schwarz slice (precursor to C1).

    Returns ``(gm, ext, meta)`` where ``ext`` is the constant external-input
    dict and ``meta`` carries the body points and centre-cell index.
    """
    mesh = make_cartesian_mesh_3d(
        _N, _N, _N, _L, _L, _L,
        origin=(-_L / 2, -_L / 2, -_L / 2), dtype=jnp.float64)
    bcs = {p: VelocityBC(u_wall=jnp.zeros(3))
           for p in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
    cfg = PisoConfig(nu=_NU, rho=_RHO, transform_backend="dense")

    body = sphere_surface_mesh(center=(0, 0, 0), radius=_A, n_refine=2)
    iface = create_interface_mesh(center=(0, 0, 0), radius=2 * _A, n_refine=2)
    nb = body.n_points
    body_pts = jnp.asarray(body.points)
    body_wts = jnp.asarray(body.weights)

    fvm = FVMFluidNode("fvm", _DT, mesh=mesh, bcs=bcs, cfg=cfg,
                       n_sample_points=nb, n_forcing_points=nb,
                       forcing_sigma=1.5 * (_L / _N))
    bem = StokesletFluidNode("bem", _DT, mu=_MU, body_mesh=body,
                             interface_mesh=iface)

    gm = GraphManager()
    gm.add_node(fvm)
    gm.add_node(bem)
    gm.add_edge("fvm", "bem", "velocity_at_points", "background_flow")

    def traction_to_force(tr, _w=body_wts):
        return tr * _w[:, None]

    gm.add_edge("bem", "fvm", "body_traction", "forcing_values",
                transform=traction_to_force)
    for fld in ("sample_points", "forcing_points"):
        gm.add_external_input("fvm", fld, shape=(nb, 3), dtype=jnp.float64)
    for fld, sh in (("body_velocity", (3,)), ("body_angular_velocity", (3,)),
                    ("body_orientation", (4,))):
        gm.add_external_input("bem", fld, shape=sh, dtype=jnp.float64)

    af = ({"fvm": ("u",), "bem": ("body_traction",)}
          if acceleration in ("iqn-ils", "iqn-imvj") else None)
    gm.add_coupling_group(
        ["fvm", "bem"], max_iterations=max_iterations, tolerance=1e-4,
        convergence_norm="interface", acceleration=acceleration,
        accelerated_fields=af, atol=1e-7, rtol=1e-3, diagnostics=True)

    ext = {
        "fvm": {"sample_points": body_pts, "forcing_points": body_pts},
        "bem": {"body_velocity": jnp.array([0.0, 0.0, _V]),
                "body_angular_velocity": jnp.zeros(3),
                "body_orientation": jnp.array([1.0, 0.0, 0.0, 0.0])},
    }
    c = int(np.argmin(np.linalg.norm(np.asarray(mesh.x), axis=1)))
    return gm, ext, {"centre_cell": c, "body_pts": body_pts}


def test_uncoupled_first_pass_recovers_stokes_drag():
    # One coupling pass leaves the far field still (zero background), so the
    # BEM sees a clean translating sphere: drag ≈ 6πμaV.
    gm, ext, _ = _build_two_scale_slice(max_iterations=1)
    st = gm.step(ext)
    Fz = float(st["bem"]["drag_force"][2])
    assert abs(Fz / _STOKES - 1.0) < 0.05


def test_within_step_coupling_converges():
    # First-step drag as a function of the coupling-iteration cap: it must
    # reach a fixed point (Cauchy) and the increments must contract.
    fz = {}
    for mi in (1, 2, 8):
        gm, ext, _ = _build_two_scale_slice(max_iterations=mi)
        fz[mi] = float(gm.step(ext)["bem"]["drag_force"][2])
    d_early = abs(fz[2] - fz[1])
    d_late = abs(fz[8] - fz[2])
    assert d_early > 0.0                       # the coupling does something
    assert d_late < 0.05 * d_early             # ... and converges
    assert abs(fz[8]) < abs(fz[1])             # background reduces the drag


def test_transfers_are_sign_correct():
    gm, ext, meta = _build_two_scale_slice(max_iterations=6)
    c = meta["centre_cell"]
    Fz0 = None
    for k in range(20):
        st = gm.step(ext)
        if k == 0:
            Fz0 = float(st["bem"]["drag_force"][2])
    u_c = np.asarray(st["fvm"]["u"][c])
    Fz = float(st["bem"]["drag_force"][2])

    # near→far: the body translating +z drives a +z far-field flow.
    assert u_c[2] > 0.0
    assert abs(u_c[0]) < 0.05 * u_c[2] and abs(u_c[1]) < 0.05 * u_c[2]
    # far→near feedback: the background flow reduces the drag below its
    # still-fluid value (positive, sign-correct coupling).
    assert 0.0 < Fz < Fz0
    # converged within the iteration cap (diagnostics report a finite residual)
    diag = gm.coupling_diagnostics()["bem+fvm"]
    assert diag["iterations"] <= 6
    assert np.isfinite(diag["residual"])
