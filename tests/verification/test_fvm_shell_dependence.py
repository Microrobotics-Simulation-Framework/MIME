"""Pins the KNOWN shell-dependence of ``surface_integral_force``.

In exact Stokes ``div σ = 0``, so ``∮ σ·n dA`` is identical over every surface
enclosing the body: the extracted drag **must not depend on the integration
shell**. It does. This test does not assert the defect is absent — it pins how
big it is, so it cannot silently worsen while nobody is looking, and so the
plateau that two independent extractors agree on stays put.

``surface_integral_force``'s own docstring has always prescribed this check
("try (1,3) and (0.5,4)... result should be robust if the shell sits in clean
fluid"). It appears never to have been run against a solved IBM field:
``scripts/fvm_validation/a2b_analytical_stokes.py`` feeds the extractor a
*prescribed* analytical Stokes field, which validates the quadrature but cannot
see the IBM band contaminate ``(u, p)``. That is the gap this closes.

What is pinned (measured at cpr = 4, λ = 0.2, Re = 0.1, ibm_eps = 1 dx):

  * the OUTER PLATEAU is tight — shells (3,6) and (4,8) agree to 0.3 %, and
    that plateau (~209) matches the independent momentum-sink extractor
    ``compute_ibm_forces_piso_step`` (210.4) to 1 %;
  * the INNER shells are badly contaminated — (0.5,2.5) reads ~1.9x the
    plateau. That shell lies *inside* the diffuse band, where the Brinkman
    sink is active and ``div σ ≠ 0``, so the surface-integral premise does
    not hold there at all;
  * the defect lives in the VISCOUS term (``2μ ε·n`` from a Green-Gauss
    gradient that averages across the band), not the pressure term.

See ``surface_integral_force``'s docstring for the full decomposition.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from maddening.core.compliance.validation import (
    verification_benchmark, BenchmarkType,
)

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.ibm import IBMBody, surface_integral_force
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso, initial_state

pytestmark = pytest.mark.slow

# Geometry matches the diagnostic rig the numbers below were measured on.
A = 0.1                 # sphere radius
R_PIPE = 0.5            # λ = a/R = 0.2
LX = LY = 1.2
LZ = 1.0
RHO = 1000.0
NU = 1e-3
MU = NU * RHO
RE = 0.1
U = RE * NU / (2 * A)   # sphere translation speed
NX = 48                 # cpr = a/dx = 4
DT = 2.0
N_STEPS = 400

# Shells in units of dx. ibm_eps = 1.0*dx, so (0.5, 2.5) starts INSIDE the band.
SHELL_INNER_BAND = (0.5, 2.5)
SHELL_PLATEAU = [(3.0, 6.0), (4.0, 8.0)]

# Pinned tolerances — deliberately loose enough not to be flaky, tight enough
# that a real change in the defect trips them.
PLATEAU_SPREAD_MAX = 0.05      # measured 0.3%
INNER_RATIO_MIN = 1.5          # measured ~1.9x
INNER_RATIO_MAX = 2.5


def _build():
    nz = int(round(NX * LZ / LX))
    # Centre the sphere in z. The outermost shell reaches r = 3a = 0.3, and the
    # sphere SDF does NOT wrap periodically, so an off-centre sphere lets the
    # domain edge clip the shell and depress the reading (measured: 199 vs 210
    # at z0 = -0.25 LZ). The shell must fit inside the box on every side.
    mesh = make_cartesian_mesh_3d(
        NX, NX, nz, LX, LY, LZ,
        origin=(-LX / 2, -LY / 2, -LZ / 2), periodic_z=True,
    )
    dx = float(mesh.cartesian_spacing[0])
    wall = IBMBody(
        name="pipe_wall",
        sdf=lambda x: R_PIPE - jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30),
    )
    # Static sphere at the origin translating at +U ẑ, imposed via u_body.
    sphere = IBMBody(
        name="sphere",
        sdf=lambda x: jnp.sqrt(jnp.sum(x ** 2, axis=-1) + 1e-30) - A,
        u_body=lambda x: jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, U], dtype=x.dtype), x.shape,
        ),
        extract_force=True,
        ref_point=jnp.zeros(3),
    )
    bcs = {}
    for nm in ("x_min", "x_max", "y_min", "y_max"):
        nbf = int(mesh.patch(nm).owner.size)
        bcs[nm] = VelocityBC(u_wall=jnp.zeros((nbf, 3)), F_through=jnp.zeros((nbf,)))
    cfg = PisoConfig(
        nu=NU, rho=RHO, gamma_conv=1.0, n_corrector=2,
        pressure_bc=("neumann", "neumann", "periodic"),
        velocity_bc=("dirichlet", "dirichlet", "periodic"),
        ibm_alpha=1e5, ibm_eps=1.0 * dx,
    )
    return mesh, bcs, cfg, dx, [wall, sphere]


def _cd_at_shells(state, mesh, dx, shells):
    sdf = lambda x: jnp.sqrt(jnp.sum(x ** 2, axis=-1) + 1e-30) - A
    q = 0.5 * RHO * U ** 2 * np.pi * A ** 2
    out = []
    for si, so in shells:
        F, _ = surface_integral_force(
            state["u"], state["p"], mesh, sdf, mu=MU, dx=dx,
            shell_inner=si, shell_outer=so, ref_point=None,
        )
        out.append(abs(float(F[2])) / q)
    return out


@verification_benchmark(
    benchmark_id="MIME-VER-031",
    description=(
        "Pins the known shell-dependence of surface_integral_force on a SOLVED "
        "IBM/PISO field (confined Stokes sphere, lambda=0.2, Re=0.1, cpr=4). In "
        "exact Stokes the traction integral is identical over every enclosing "
        "surface, so shell-dependence is pure error measurable without knowing "
        "the right answer. Pins (a) the outer plateau is tight (<5%), and (b) a "
        "shell lying inside the diffuse IBM band — where the Brinkman sink makes "
        "div(sigma) != 0 and the premise is void — reads 1.5-2.5x the plateau. "
        "Complements a2b_analytical_stokes.py, which prescribes the field and so "
        "cannot see IBM contamination."
    ),
    node_type="FVMFluidNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "Outer band-free shells (3,6) and (4,8) agree to <5% of their mean; the "
        "in-band shell (0.5,2.5) reads 1.5-2.5x the plateau. Both pin a "
        "characterised defect against silent regression."
    ),
    references=(),
)
def test_ver031_surface_integral_shell_dependence():
    mesh, bcs, cfg, dx, bodies = _build()
    state = run_piso(
        mesh, bcs, cfg, n_steps=N_STEPS, dt=DT,
        ibm_bodies=bodies, initial=initial_state(mesh),
    )
    state["u"].block_until_ready()

    plateau = _cd_at_shells(state, mesh, dx, SHELL_PLATEAU)
    inner = _cd_at_shells(state, mesh, dx, [SHELL_INNER_BAND])[0]

    spread = (max(plateau) - min(plateau)) / float(np.mean(plateau))
    assert spread < PLATEAU_SPREAD_MAX, (
        f"outer plateau widened: shells {SHELL_PLATEAU} give Cd = {plateau}, "
        f"spread {100*spread:.1f}% > {100*PLATEAU_SPREAD_MAX:.0f}%. The two "
        f"outermost band-free shells enclose the same body in near-Stokes flow "
        f"and must agree; if they no longer do, the extraction or the far field "
        f"has changed."
    )

    ratio = inner / float(np.mean(plateau))
    assert INNER_RATIO_MIN < ratio < INNER_RATIO_MAX, (
        f"in-band shell {SHELL_INNER_BAND} reads Cd = {inner:.1f} vs plateau "
        f"{np.mean(plateau):.1f} (ratio {ratio:.2f}), outside the pinned band "
        f"[{INNER_RATIO_MIN}, {INNER_RATIO_MAX}]. This shell lies inside the "
        f"diffuse IBM band where div(sigma) != 0, so it is expected to be "
        f"badly wrong — this pins HOW wrong. A ratio near 1.0 would mean the "
        f"contamination changed character; investigate before celebrating."
    )
