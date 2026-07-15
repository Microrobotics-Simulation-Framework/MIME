"""MIME-VER-032 — the IBM penalty must be consistent with the viscous solve.

THE DEFECT (default path, ``ibm_implicit_viscous=False``)
--------------------------------------------------------
:func:`~mime.nodes.environment.fvm.piso.make_piso_step` applies the Brinkman
penalty as a closed-form pre-step and *then* runs the implicit viscous
(Helmholtz) solve. Nothing holds no-slip **during** that solve, so velocity
diffuses freely back into the body over the step: the body leaks to a depth
``√(ν dt)``. The drag deficit is therefore ``√(ν dt)/a`` — HALF-ORDER in dt,
and measured at ``1.216·√(ν dt)/a`` on the confined Stokes sphere (λ = 0.2,
Re = 0.1, cpr = 4, closed tube), i.e. **51 % low at dt = 2.0**::

    dt        0.0156  0.0312  0.0625  0.125  0.25   0.5    1.0    2.0   10.0
    Cd        565.0   556.1   541.3   517.3  481.6  432.1  367.8  290.2  111.2

The limiting length is the BODY radius ``a``, not ``dx`` and not ``ibm_eps``:
at equal dt, ``eps = 0.025`` and ``eps = 0.05`` give matching deficits. So the
term does NOT shrink under mesh refinement, and a cpr sweep at fixed dt is not
a convergence study — it measures the leak. Worse, the ``eps`` error is
POSITIVE and this one NEGATIVE, so a curve at fixed nonzero dt can reach a
reference by CANCELLATION while the physics gets worse.

THE FIX (``ibm_implicit_viscous=True``)
---------------------------------------
Fold the penalty into the viscous solve as a Brinkman drag,
``(I − ν dt ∇² + α dt I_body) u* = u_pred + α dt I_body u_body``. The leak
length becomes the Brinkman screening depth ``√(ν/α)`` — set by the penalty and
**independent of dt**. Equivalently: the coupled scheme's penalty/viscous ratio
is ``α I/ν``, which has no dt in it, whereas the split scheme's pre-step
hardness is ``exp(−α I dt)``, which does.

WHAT IS TESTED HERE, AND WHY THIS METRIC
----------------------------------------
``state["u_pre_ibm"]`` is post-viscous-solve, post-projection, and *pre* the
final Brinkman. It carries the **same meaning in both schemes**, so it is an
apples-to-apples probe of whether no-slip survived the viscous solve — and it
needs only ONE step, no steady state, so this runs in the CI lane.

The metric is the indicator-weighted mean slip ``|u_pre_ibm − u_body|/U`` over
the body = the fraction of the body's momentum lost during the viscous stage.
Its *absolute* value is not the leak: a residual ~0.33 remains as dt → 0 in
BOTH schemes, because the diffuse band legitimately does not enforce u_body
where the indicator is small. That floor is the ``eps`` geometry, a different
term. What is diagnostic is the metric's **dt-sensitivity**, which is the leak
and nothing else.

The split arm is a REQUIRED CONTROL, not decoration. Without it, the coupled
arm passing would be uninformative — it could mean the metric is simply blind
to the thing being tested. The control proves the metric discriminates.

Measured (cpr = 3, eps = 1 dx, α = 1e5, one step from rest, 64× dt range):

    dt        √(ν dt)/a   SPLIT slip/U   COUPLED slip/U
    2.0       0.447       0.5489         0.3385
    0.5       0.224       0.3980         0.3339
    0.125     0.112       0.3430         0.3314
    0.03125   0.056       0.3319         0.3303
              change:     65 %           2.5 %

Note the split arm CONVERGES TO the coupled arm as dt → 0 (0.3319 vs 0.3385,
2 % apart). That is the consistency statement: the coupled scheme at dt = 2.0
delivers what the split scheme only reaches in the dt → 0 limit.

Confirmed downstream on the full confined-Stokes-sphere drag at cpr = 4
(closed tube, eps = 0.025), all true fixed points (resid 1e-8..1e-11):

    dt        2.0      0.5      0.125    0.03125   swing over 64x
    split     290.16   432.14   517.28   556.13      91.7 %
    coupled   551.66   557.58   561.76   567.21       2.8 %   (33x less)

A ~2.8 % residual dt-dependence survives in the coupled path. It is NOT this
leak — that was half-order and 92 % — and it does not follow the √dt law. Its
mechanism was not established here; the leading suspect is the well-known
Rhie-Chow dt-dependence of the converged solution (``D_bar = dt/ρ``), a
separate issue. Untested.

NOTE ON THE REFERENCE. Bohlin (1960) gives Cd = 403 here, and the fix moves the
number AWAY from it (290 → 552). That is expected and is not a regression: the
leak is a NEGATIVE error and the ``eps = 1 dx`` diffuse band is a POSITIVE one
(~+44 %), so the pre-fix agreement was partly cancellation between two wrong
things. Judging this fix by proximity to 403 at some convenient dt is exactly
the trap; judge it by dt-independence.
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
from mime.nodes.environment.fvm.ibm import IBMBody, smoothed_indicator
from mime.nodes.environment.fvm.piso import (
    PisoConfig, make_piso_step, initial_state,
)
from mime.nodes.environment.fvm.pressure import (
    make_helmholtz_solver, make_helmholtz_apply,
    make_helmholtz_solver_fft, make_helmholtz_apply_fft,
    make_helmholtz_solver_iterative, make_helmholtz_apply_iterative,
)

A = 0.1                  # sphere radius
RHO = 1000.0
NU = 1e-3
U = 5e-4                 # sphere translation speed (Re = 0.1)
NX = 36                  # cpr = 3
LX = 1.2

DT_HI, DT_LO = 2.0, 0.03125            # 64x apart

# Pinned tolerances: loose enough not to be flaky, tight enough to trip on a
# real change. Measured 2.5% (coupled) vs 65% (split).
COUPLED_DT_SENSITIVITY_MAX = 0.10
SPLIT_DT_SENSITIVITY_MIN = 0.35
LIMIT_AGREEMENT_MAX = 0.10             # measured 2%


def _build(dt, implicit):
    mesh = make_cartesian_mesh_3d(NX, NX, NX, LX, LX, LX, origin=(-LX / 2,) * 3)
    dx = float(mesh.cartesian_spacing[0])
    sphere = IBMBody(
        name="sphere",
        sdf=lambda x: jnp.sqrt(jnp.sum(x ** 2, axis=-1) + 1e-30) - A,
        u_body=lambda x: jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, U], dtype=x.dtype), x.shape),
        extract_force=True,
        ref_point=jnp.zeros(3),
    )
    bcs = {}
    for nm in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        nbf = int(mesh.patch(nm).owner.size)
        bcs[nm] = VelocityBC(u_wall=jnp.zeros((nbf, 3)),
                             F_through=jnp.zeros((nbf,)))
    cfg = PisoConfig(
        nu=NU, rho=RHO, gamma_conv=1.0, n_corrector=2,
        pressure_bc="neumann", velocity_bc="dirichlet",
        ibm_alpha=1e5, ibm_eps=dx,
        ibm_implicit_viscous=implicit, ibm_cg_iters=60,
    )
    return mesh, bcs, cfg, sphere, dx


def _body_slip(dt, implicit):
    """Indicator-weighted mean |u_pre_ibm − u_body| / U after ONE step from rest.

    The fraction of the body's momentum that failed to survive the viscous
    stage. The fluid starts at rest and the sphere translates at U: a uniform-U
    initial field would instead be annihilated by the projection (it is not
    admissible in a closed box) and would measure only that transient.
    """
    mesh, bcs, cfg, sphere, dx = _build(dt, implicit)
    step = jax.jit(lambda s: make_piso_step(mesh, bcs, cfg,
                                            ibm_bodies=[sphere])(s, dt))
    state = step(initial_state(mesh))
    I = np.asarray(smoothed_indicator(sphere.sdf(mesh.x), dx))
    w = I * np.asarray(mesh.V)
    ub = np.zeros((mesh.N_cells, 3))
    ub[:, 2] = U
    slip = np.linalg.norm(np.asarray(state["u_pre_ibm"]) - ub, axis=-1)
    return float((slip * w).sum() / w.sum()) / U


@verification_benchmark(
    benchmark_id="MIME-VER-032",
    description=(
        "The IBM no-slip constraint must be consistent with the implicit "
        "viscous solve. With the penalty applied OUTSIDE the Helmholtz solve "
        "(the default split scheme) nothing holds no-slip during that solve, "
        "so the body leaks by diffusion to depth sqrt(nu*dt) and the confined "
        "Stokes sphere drag is half-order in dt (51% low at dt=2.0, deficit "
        "1.216*sqrt(nu*dt)/a). PisoConfig.ibm_implicit_viscous couples the "
        "penalty into the viscous solve, making the leak length the Brinkman "
        "screening depth sqrt(nu/alpha), which is dt-independent. Measured on "
        "the per-step body momentum loss over a 64x dt range: split varies "
        "65%, coupled 2.5%, and split converges to coupled as dt->0."
    ),
    node_type="FVMFluidNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "Over a 64x dt range at fixed mesh/eps/alpha, the coupled path's "
        "per-step body momentum loss varies <10% (measured 2.5%) while the "
        "split path varies >35% (measured 65%), and the two agree to <10% in "
        "the dt->0 limit (measured 2%). The split arm is a required control "
        "proving the metric can detect the leak at all."
    ),
    references=(),
)
def test_ver032_coupled_ibm_removes_the_dt_leak():
    coupled_hi = _body_slip(DT_HI, implicit=True)
    coupled_lo = _body_slip(DT_LO, implicit=True)
    split_hi = _body_slip(DT_HI, implicit=False)
    split_lo = _body_slip(DT_LO, implicit=False)

    coupled_sens = abs(coupled_hi - coupled_lo) / coupled_lo
    split_sens = abs(split_hi - split_lo) / split_lo

    # CONTROL FIRST. If the split arm does NOT show the leak, this metric is
    # not measuring the thing under test and the coupled arm's result below is
    # uninformative -- an invariant metric is not a null result.
    assert split_sens > SPLIT_DT_SENSITIVITY_MIN, (
        f"CONTROL FAILED: the split (default) path's per-step body momentum "
        f"loss varied only {100*split_sens:.1f}% over a {DT_HI/DT_LO:.0f}x dt "
        f"range ({split_hi:.4f} at dt={DT_HI} -> {split_lo:.4f} at dt={DT_LO}), "
        f"but the sqrt(nu*dt) leak should make it vary >{100*SPLIT_DT_SENSITIVITY_MIN:.0f}% "
        f"(measured 65%). Either the split scheme changed -- in which case this "
        f"whole benchmark needs rewriting -- or this metric has gone blind to "
        f"the leak, and the coupled assertion below proves nothing."
    )

    assert coupled_sens < COUPLED_DT_SENSITIVITY_MAX, (
        f"the coupled path (ibm_implicit_viscous=True) is dt-sensitive: "
        f"per-step body momentum loss {coupled_hi:.4f} at dt={DT_HI} vs "
        f"{coupled_lo:.4f} at dt={DT_LO} ({100*coupled_sens:.1f}% > "
        f"{100*COUPLED_DT_SENSITIVITY_MAX:.0f}%). Its penalty/viscous ratio is "
        f"alpha*I/nu, which contains no dt, so this should be flat. A leak has "
        f"returned -- check that the penalty is still inside the solve and that "
        f"ibm_cg_iters is enough to converge it."
    )

    # Consistency: the coupled scheme at LARGE dt must deliver what the split
    # scheme only reaches as dt -> 0. Without this, "flat in dt" could just mean
    # "flatly wrong".
    agree = abs(coupled_hi - split_lo) / split_lo
    assert agree < LIMIT_AGREEMENT_MAX, (
        f"the coupled path at dt={DT_HI} ({coupled_hi:.4f}) disagrees with the "
        f"split path's dt->0 limit ({split_lo:.4f} at dt={DT_LO}) by "
        f"{100*agree:.1f}%. Both discretise the same steady problem, so they "
        f"must agree there; a flat-but-different coupled answer means the "
        f"coupled scheme changed the body, not just the leak."
    )


@pytest.mark.parametrize("backend", ["dense", "fft", "iterative"])
def test_ver032_helmholtz_apply_inverts_its_solver(backend):
    """``apply(solver(b, α), α) == b`` — the forward operator the coupled IBM
    solve builds its residual from must be the SAME H_s the preconditioner
    inverts, or the Krylov solve is solving a different system than intended.
    """
    mesh = make_cartesian_mesh_3d(12, 12, 12, 1.2, 1.2, 1.2, origin=(-0.6,) * 3)
    mk = {
        "dense": (make_helmholtz_solver, make_helmholtz_apply),
        "fft": (make_helmholtz_solver_fft, make_helmholtz_apply_fft),
        "iterative": (make_helmholtz_solver_iterative,
                      make_helmholtz_apply_iterative),
    }[backend]
    solver, apply = mk[0](mesh, bc="dirichlet"), mk[1](mesh, bc="dirichlet")
    b = jax.random.normal(jax.random.PRNGKey(0), (mesh.N_cells, 3))
    for alpha in (2e-3, 1.0):
        rt = apply(solver(b, alpha), alpha)
        err = float(jnp.max(jnp.abs(rt - b))) / float(jnp.max(jnp.abs(b)))
        # The dense basis matrices are float32 (mesh dtype), so that path is
        # only ~1e-7 self-consistent even under x64; fft is exact to ~1e-15.
        assert err < 1e-5, (
            f"{backend} apply(solve(b, {alpha})) != b: rel err {err:.2e}. The "
            f"forward and inverse Helmholtz operators have diverged."
        )
