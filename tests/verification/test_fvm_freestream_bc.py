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
``piso.py`` computed ``diff_specs = velocity_diffusion_specs(mesh, bcs, mu=mu)``
and never used it, building the Helmholtz solver as
``make_helmholtz_solver(mesh, bc=cfg.velocity_bc)`` — the BC *type string* only,
never ``u_wall``. The solver's own docstring says Dirichlet means "cell-centred
**zero** at the physical face", so PISO silently enforced ``u = 0`` at every
Dirichlet wall regardless of what ``VelocityBC.u_wall`` asked for.
(``simple.py`` always threaded ``diff_specs`` through; PISO alone dropped it.)

Measured on this empty box: an **8.8 % freestream deficit** (``U_far`` = 0.0911
against U = 0.1) and **reversed flow** near the walls. The tell was that
``u_wall = U ẑ`` and ``u_wall = None`` produced *bit-identical* fields — the
argument had no effect at all. That is the origin of the 22.6 % "boundary layer"
error previously attributed to the sphere in EP-0c's driven-inflow rig — whose
recorded ``U_far`` matches this **no-sphere** value to 0.2 %.

Fixed by :func:`~mime.nodes.environment.fvm.boundary.velocity_helmholtz_bc_source`,
which adds the difference between the physical and basis boundary fluxes onto the
Helmholtz RHS. The Dirichlet arm now holds uniform translation to ~1e-5, the same
order as the periodic arm.

A second, independent defect this test's own rig used to carry
--------------------------------------------------------------
The rig originally set ``z_max = VelocityBC(u_wall=None, F_through=None)``,
commented "zero-gradient out". But ``divergence_face_flux`` **skips** any patch
absent from ``boundary_F``, and a skipped patch contributes zero flux — i.e. it
is **impermeable**, not an outlet. So the box was fed inflow at ``z_min`` with no
way out, and uniform translation *violates discrete continuity* under those BCs:
the projection is obliged to stall the flow (``u_z → 0`` at ``z_max``), which is
correct behaviour for an ill-posed rig. That stall, not the ``u_wall`` bug, was
what pinned ``max|u_z - U|/U`` near 1.0.

The two were separable by varying ``F_through`` alone (all with the ``u_wall``
fix in place)::

    z_max = (u_wall=None,  F_through=None)   ->  max err 1.011   [closed]
    z_max = (u_wall=None,  F_through=U·Sf)   ->  max err 7.7e-06 [outlet]
    z_max = (u_wall=U ẑ,  F_through=None)   ->  max err 1.019   [closed]
    z_max = (u_wall=U ẑ,  F_through=U·Sf)   ->  max err 9.4e-06 [outlet]

So ``z_max`` now carries the same uniform BC as every other patch — which is what
this test's premise ("every boundary is told the same thing") always claimed, and
makes the rig well-posed. Note ``VelocityBC``'s docstring still advertises
``u_wall=None`` as giving "zero-gradient extrapolation" at an outlet; no
extrapolation exists in ``divergence_face_flux``, so an outlet must be given an
explicit ``F_through``.

The periodic arm passes to ~1e-5, which is what makes the Dirichlet arm's
behaviour attributable to the BCs rather than to the scheme, the mesh, or the
timestep.
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
        # Every patch — z_max included — is told the same freestream. z_max
        # MUST carry an explicit F_through: divergence_face_flux skips patches
        # absent from boundary_F, and a skipped patch is impermeable, which
        # would close the box and make uniform translation violate discrete
        # continuity. See the module docstring.
        bcs = {p: _uniform_bc(mesh, p) for p in
               ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
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


def test_ver030_uniform_translation_dirichlet():
    """Inhomogeneous-Dirichlet BCs are enforced, not silently replaced by u = 0.

    Regression for the dropped-``u_wall`` bug: before the fix this measured an
    8.8 % freestream deficit (max u_z = 0.0911) and reversed flow, and was
    bit-identical to the same run with ``u_wall=None``.
    """
    uz = _run(periodic=False)
    err = np.abs(uz - U).max() / U
    assert err < DIRICHLET_TOL, (
        f"inhomogeneous-Dirichlet box failed to preserve uniform translation: "
        f"max|u_z - U|/U = {err:.3e}, min u_z = {uz.min():.6f} (U = {U}). "
        f"A negative min u_z is reversed flow in a pure translation."
    )
