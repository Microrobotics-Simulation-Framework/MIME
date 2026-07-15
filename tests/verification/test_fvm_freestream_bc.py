"""Uniform-translation preservation — the cheapest possible falsification of the solve.

An **empty box**: no IBM body, no sphere, no immersed geometry of any kind. The
domain is initialised to a uniform ``u = U ẑ`` and every boundary is told the same
thing. The exact solution is ``u = U ẑ`` for all time, ``p = const`` — a rigid
uniform translation, with zero convection, zero viscous stress and zero pressure
gradient. **Any deviation is pure solver/BC error.** There is no sphere, no IBM,
and no resolution excuse available.

Why this test exists
--------------------
``scripts/fvm_validation/a2b_analytical_stokes.py`` validates
``surface_integral_force`` on a *prescribed* analytical field, which isolates the
extraction formula and deliberately bypasses the IBM/PISO solve — so it cannot
detect a failure in the solve or its boundary conditions. This test bypasses the
**sphere** instead, exercising only the solve + BCs. The two are complementary,
and this half was missing.

It is deliberately **not** marked ``slow`` or ``gpu``. Those markers mean "does
not run in CI" (``addopts = "-m 'not slow'"``), and every existing FVM drag
assertion is behind one of them. A physics check that never runs is documentation.

What it caught
--------------
The Dirichlet arm currently **fails**: ``piso.py`` computes ``diff_specs =
velocity_diffusion_specs(mesh, bcs, mu=mu)`` and never uses it, building the
Helmholtz solver as ``make_helmholtz_solver(mesh, bc=cfg.velocity_bc)`` — the BC
*type string* only, never ``u_wall``. The solver's own docstring says Dirichlet
means "cell-centred **zero** at the physical face", so PISO silently enforces
``u = 0`` at every Dirichlet wall regardless of what ``VelocityBC.u_wall`` asks
for. (``simple.py`` does thread ``diff_specs`` through; PISO alone drops it.)

Measured on an empty box: an **8.8 % freestream deficit** and **reversed flow**
near the walls. That is the entire origin of the 22.6 % "boundary layer" error
previously attributed to the sphere in EP-0c's driven-inflow rig — whose recorded
``U_far`` matches this **no-sphere** value to 0.2 %.

The periodic arm passes to ~1e-5, which is what makes the Dirichlet arm's failure
attributable to the BCs rather than to the scheme, the mesh, or the timestep.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from maddening.core.compliance.validation import (
    verification_benchmark, BenchmarkType,
)

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso, initial_state

pytestmark = pytest.mark.x64

# Small box: this is a BC-enforcement bug, so it shows immediately and needs no
# production geometry. Kept CI-cheap on purpose.
A, U = 0.05, 0.1
N_XY, N_Z = 12, 20
L_XY, L_Z = 12 * A, 20 * A
NU = U * 2 * A / 100.0          # Re = 100 on the notional body scale
DT = 0.01
N_STEPS = 60

# The exact solution is uniform translation; these are generous.
PERIODIC_TOL = 1e-3             # fraction of U
DIRICHLET_TOL = 1e-2            # fraction of U


def _uniform_bc(mesh, patch, axis=2):
    """Freestream Dirichlet: u = U*e_axis at the wall (NOT no-slip)."""
    p = mesh.patch(patch)
    u = jnp.zeros((p.owner.size, mesh.dim), dtype=mesh.V.dtype)
    u = u.at[:, axis].set(U)
    F = (U * p.Sf[:, axis]).astype(mesh.V.dtype)
    return VelocityBC(u_wall=u, F_through=F)


def _build(periodic: bool):
    mesh = make_cartesian_mesh_3d(
        N_XY, N_XY, N_Z, L_XY, L_XY, L_Z,
        origin=(-L_XY / 2, -L_XY / 2, -L_Z / 2),
        periodic_x=periodic, periodic_y=periodic, periodic_z=periodic,
    )
    dx = mesh.cartesian_spacing[0]
    if periodic:
        bcs = {}
        cfg = PisoConfig(nu=NU, rho=1.0, gamma_conv=1.0, n_corrector=2,
                         pressure_bc="periodic", velocity_bc="periodic",
                         ibm_alpha=1e5, ibm_eps=1.0 * dx)
    else:
        bcs = {p: _uniform_bc(mesh, p) for p in
               ("x_min", "x_max", "y_min", "y_max", "z_min")}
        bcs["z_max"] = VelocityBC(u_wall=None, F_through=None)   # zero-gradient out
        cfg = PisoConfig(nu=NU, rho=1.0, gamma_conv=1.0, n_corrector=2,
                         pressure_bc="neumann", velocity_bc="dirichlet",
                         ibm_alpha=1e5, ibm_eps=1.0 * dx)
    return mesh, bcs, cfg


def _run(periodic: bool) -> np.ndarray:
    """Return u_z after N_STEPS from an exact uniform-translation start."""
    mesh, bcs, cfg = _build(periodic)
    st = dict(initial_state(mesh))
    st["u"] = jnp.zeros_like(st["u"]).at[:, 2].set(U)
    st["F"] = (U * mesh.Sf[:, 2]).astype(mesh.V.dtype)
    st = run_piso(mesh, bcs, cfg, n_steps=N_STEPS, dt=DT,
                  body_force_fn=None, ibm_bodies=[], initial=st)
    return np.asarray(st["u"][:, 2])


@verification_benchmark(
    benchmark_id="MIME-VER-030",
    description=(
        "FVM/PISO preserves uniform translation in an EMPTY box (no IBM body). "
        "Initialised to u = U*z_hat with all boundaries set to the same uniform "
        "freestream, the exact solution is u = U*z_hat for all time (zero "
        "convection, zero viscous stress, zero pressure gradient). Run in both "
        "fully-periodic and inhomogeneous-Dirichlet BC modes. Any deviation is "
        "solver/BC error with no sphere, no IBM and no resolution excuse. "
        "Complements a2b_analytical_stokes.py, which prescribes the field and so "
        "cannot test the solve."
    ),
    node_type="FVMFluidNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "max|u_z - U| / U < 1e-3 (periodic) and < 1e-2 (inhomogeneous Dirichlet) "
        "after 60 steps from an exact uniform-translation initial condition."
    ),
    references=(),
)
def test_ver030_uniform_translation_periodic():
    """The control. Periodic BCs cannot get uniform translation wrong."""
    uz = _run(periodic=True)
    err = np.abs(uz - U).max() / U
    assert err < PERIODIC_TOL, (
        f"periodic box failed to preserve uniform translation: "
        f"max|u_z - U|/U = {err:.3e}. If THIS fails, the defect is in the "
        f"scheme/mesh/timestep, not the BCs."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN BUG: PISO drops u_wall. piso.py computes diff_specs = "
        "velocity_diffusion_specs(mesh, bcs, mu=mu) and never uses it; the "
        "Helmholtz solver is built from cfg.velocity_bc (the BC TYPE STRING) "
        "only, so Dirichlet means 'cell-centred zero at the physical face' and "
        "u=0 is enforced at every Dirichlet wall regardless of VelocityBC.u_wall. "
        "simple.py threads diff_specs through; PISO alone drops it. Measured: "
        "~8.8% freestream deficit and reversed flow, with no body present. This "
        "is the true origin of the 22.6% 'sub-grid boundary layer' error in "
        "EP-0c's driven-inflow rig. FIX: thread diff_specs into the Helmholtz "
        "solve, or use the lifting decomposition the node docstring prescribes "
        "(u_wall=0 + lift). strict=True: when fixed this XPASSes and CI will "
        "demand this marker be removed."
    ),
)
def test_ver030_uniform_translation_dirichlet():
    """The bug. Inhomogeneous-Dirichlet BCs are silently replaced by u = 0."""
    uz = _run(periodic=False)
    err = np.abs(uz - U).max() / U
    assert err < DIRICHLET_TOL, (
        f"inhomogeneous-Dirichlet box failed to preserve uniform translation: "
        f"max|u_z - U|/U = {err:.3e}, min u_z = {uz.min():.6f} (U = {U}). "
        f"A negative min u_z is reversed flow in a pure translation."
    )
