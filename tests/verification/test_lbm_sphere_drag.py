"""MIME-VER-012 — drag on a sphere translating in a tube (IBLBMFluidNode).

The physics check on the geometry-hook / translation refactor of
``IBLBMFluidNode`` (MIME-NODE-010 v1.1.0): the node's *first* non-UMR body and
its *first* translating body, so the drag it reports needs an independent
reference rather than a self-consistency guard.

Reference
---------
Stokes drag with the Faxén / Haberman-Sayre wall correction for a sphere on
the axis of a cylindrical tube:

    F = 6*pi*mu*a*U / K(lambda),      lambda = a / R_tube

    K = 1 - 2.10444*L + 2.08877*L^3 - 0.94813*L^5
          - 1.372*L^6 + 3.87*L^8 - 4.19*L^10

READ THE ERROR BUDGET in ``test_translating_sphere_drag_vs_haberman_sayre``
before tightening the tolerance: the dominant term is a **domain** effect, not
a solver error, and Bouzidi does not remove it.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.validation import (
    verification_benchmark, BenchmarkType,
)

from mime.nodes.environment.lbm.body import make_sphere_body_factory
from mime.nodes.environment.lbm.fluid_node import IBLBMFluidNode


def haberman_sayre_K(lam: float) -> float:
    """Faxén / Haberman-Sayre wall-correction factor; ``F = 6*pi*mu*a*U / K``.

    Series valid for ``lambda = a/R_tube`` up to ~0.3-0.6 depending on source;
    used here at 0.27.
    """
    return (1.0
            - 2.10444 * lam
            + 2.08877 * lam ** 3
            - 0.94813 * lam ** 5
            - 1.372 * lam ** 6
            + 3.87 * lam ** 8
            - 4.19 * lam ** 10)


def _sphere_in_tube(nx, ny, nz, R_pipe, R_sph, tau, use_bouzidi=False):
    return IBLBMFluidNode(
        name="lbm", timestep=1.0, nx=nx, ny=ny, nz=nz, tau=tau,
        vessel_radius_lu=R_pipe,
        body_factory=make_sphere_body_factory("sphere", radius=R_sph),
        use_bouzidi=use_bouzidi,
        body_reference_position_lu=(nx / 2.0, ny / 2.0, nz / 2.0),
        boundary_links_margin=3.0,
    )


# -- Stokes linearity (fast) -------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-012",
    description=(
        "Sphere drag in IBLBMFluidNode is linear in the body velocity "
        "(Stokes regime), and the reported load is the reaction on the fluid"
    ),
    node_type="IBLBMFluidNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "F(2U)/F(U) within 1% of 2.0 at Re ~ 0.2; drag axis-aligned with the "
        "body velocity to within 2%"
    ),
    references=("Ladd1994",),
)
def test_sphere_drag_is_linear_in_velocity():
    """Stokes flow is linear in U, so the drag must be too.

    This is the *sharp* physics check on the translation path: it is exact
    (not resolution-limited), because holding the sphere's position fixed and
    varying only ``body_velocity`` keeps the geometry — and therefore every
    discretisation error — identical between the two runs. Any deviation from
    the factor 2 is a genuine nonlinearity, not a staircase artefact.

    (The LBM equilibrium carries O(u^2) terms, so linearity is not algebraically
    exact; at U ~ 2e-3 those are O(1e-5) relative and invisible at 1%.)
    """
    nx = ny = 32
    nz = 48
    node = _sphere_in_tube(nx, ny, nz, R_pipe=14.0, R_sph=4.0, tau=0.9)
    pos = jnp.array([nx / 2.0, ny / 2.0, nz / 2.0])
    step = jax.jit(lambda st, bi: node.update(st, bi, 1.0))

    def drag_at(U):
        state = node.initial_state()
        bi = {"body_position": pos,
              "body_velocity": jnp.array([0.0, 0.0, U])}
        for _ in range(150):
            state = step(state, bi)
        return np.asarray(state["drag_force"])

    U = 0.002
    F1 = drag_at(U)
    F2 = drag_at(2 * U)

    ratio = F2[2] / F1[2]
    assert ratio == pytest.approx(2.0, rel=0.01), (
        f"Stokes drag must be linear in U: F(2U)/F(U) = {ratio:.4f}, "
        f"F(U)={F1}, F(2U)={F2}"
    )

    # The node reports the *reaction* (+R*U, force on the fluid) — the
    # documented IBLBM convention that HydrodynamicModel.LBM flips via
    # native_drag_sign=-1 (FLUID_NODE_CONTRACT.md).
    assert F1[2] > 0.0, f"reaction on the fluid should be +z, got {F1}"
    # Axisymmetric body on the tube axis: no transverse load.
    assert abs(F1[0]) < 0.02 * F1[2]
    assert abs(F1[1]) < 0.02 * F1[2]


# -- Quantitative drag vs Haberman-Sayre (slow) ------------------------------

@pytest.mark.slow
@verification_benchmark(
    benchmark_id="MIME-VER-015",
    description=(
        "Drag on a sphere translating along the axis of a tube vs Stokes "
        "drag with the Faxen / Haberman-Sayre wall correction"
    ),
    node_type="IBLBMFluidNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "Period-averaged drag within 35% of 6*pi*mu*a*U/K(lambda) at "
        "a = 6 cells, lambda = 0.27, Re = 0.18. MEASURED 0.739x (a 26% "
        "under-prediction). The tolerance encodes a KNOWN, MEASURED error "
        "budget — roughly half of it a periodic-domain co-flow that is not a "
        "solver error at all (re-referencing to the measured relative "
        "velocity gives 0.863x), the rest discretisation at a = 6 cells. "
        "It is NOT a statement of solver agreement. Read the docstring's "
        "error budget before tightening."
    ),
    references=("HabermanSayre1958", "Ladd1994", "Bouzidi2001"),
)
def test_translating_sphere_drag_vs_haberman_sayre():
    """Sphere translating along a tube axis; drag vs Stokes + Haberman-Sayre.

    Configuration: 48x48x96 lattice, R_tube = 22, a = 6 (lambda = 0.273),
    tau = 0.9 (nu = 0.1333), U = 0.002 => Re = U*2a/nu = 0.18 (Stokes),
    Ma = sqrt(3)*U = 0.003. Run 5000 steps ~ 1.4 viscous times
    (R_tube^2/nu = 3630), then average the drag over the last 500 steps.

    **Why average over 500 steps.** 500 steps is exactly the time for the
    sphere to translate one lattice cell (U = 0.002). The drag settles into a
    periodic ripple locked to that cell crossing — the classic moving-
    bounce-back "lattice ratchet" — of about +/-1.5% (measured 3.1% peak-to-
    peak with simple BB, 1.25% with Bouzidi, which resolves the sub-cell wall
    position). Averaging over one crossing period is the moving-boundary
    analogue of averaging the UMR torque over one rotation period.

    **Error budget — measured, not assumed.** The result is 0.739x the
    Haberman-Sayre prediction. The 26% decomposes as follows (all figures
    measured on this configuration):

    1. **Periodic-domain co-flow (~12 points — the largest single term, and
       NOT a solver error).** Haberman-Sayre is the *sealed* tube: zero net
       flux, fluid at rest far from the sphere. A periodic LBM tube cannot
       reproduce that. The sphere continuously injects +z momentum into the
       fluid, and the only sink is wall shear — which requires a mean flow to
       exist. The tube therefore carries a steady co-flow *in the direction of
       travel*, so the true relative velocity is below U and the drag is
       correspondingly lower. Measured far-field mean u_z = +14.0% of U;
       balancing the injected momentum against Poiseuille wall friction,
       u_mean ~ F/(8*pi*mu*L_z), predicts the same order and the same 1/L_z
       scaling. **Re-referencing the drag to the measured relative velocity
       U - u_far gives 0.863x** — i.e. this term alone accounts for about half
       the gap. (Note the sign: it is a CO-flow, not a backflow.)
    2. **Coarse body + wall staircase (~14 points).** a = 6 cells, and the
       tube wall is voxelised at R = 22 with an effective halfway-bounce-back
       radius ~0.5 cell smaller, shifting the effective lambda.

    Evidence that (2) is *not* staircase-dominated: **Bouzidi interpolated
    bounce-back — which resolves the sub-cell wall position and would largely
    fix a staircase error — gives essentially the same mean ratio** (0.726 vs
    0.739, 1.8% apart). It halves the ripple but not the bias.

    Evidence that the **translation path itself is essentially unbiased**:
    holding the mask fixed and imposing the same surface velocity (at Re->0
    Stokes flow is quasi-steady, so the instantaneous drag must match) gives
    0.742 vs the translating 0.739 — **0.4% apart, with the ripple collapsing
    from 3.1% to 0.22%**. So sweeping the body across the lattice, despite the
    absence of a fresh-node refill scheme, costs a ripple but no measurable
    mean-drag bias at this U.

    This test is therefore a *regime/scale* check — "the drag is the right
    physical quantity to tens of percent, with the right sign, linearity and
    confinement enhancement". The *sharp* check on the physics is
    ``test_sphere_drag_is_linear_in_velocity``.

    Do NOT tighten the tolerance without first removing term (1) — a much
    longer tube, or re-referencing the drag to the measured relative velocity.
    """
    nx = ny = 48
    nz = 96
    R_pipe, R_sph, tau, U = 22.0, 6.0, 0.9, 0.002
    n_steps, n_avg = 5000, 500

    nu = (tau - 0.5) / 3.0
    mu = 1.0 * nu                      # rho_lattice = 1
    lam = R_sph / R_pipe
    Re = U * 2 * R_sph / nu
    assert Re < 1.0, f"config must stay in the Stokes regime, got Re={Re}"

    F_expected = 6.0 * math.pi * mu * R_sph * U / haberman_sayre_K(lam)

    node = _sphere_in_tube(nx, ny, nz, R_pipe, R_sph, tau)
    state = node.initial_state()
    step = jax.jit(lambda st, bi: node.update(st, bi, 1.0))

    z0 = nz / 2.0
    fzs = []
    for i in range(n_steps):
        bi = {
            "body_position": jnp.array([nx / 2.0, ny / 2.0, z0 + U * i]),
            "body_velocity": jnp.array([0.0, 0.0, U]),
        }
        state = step(state, bi)
        if i >= n_steps - n_avg:
            fzs.append(float(state["drag_force"][2]))

    F_mean = float(np.mean(fzs))
    ratio = F_mean / F_expected

    # The ripple is the cell-crossing ratchet; it must stay small, or the
    # moving-boundary treatment (no fresh-node refill) has broken down.
    ripple = (max(fzs) - min(fzs)) / abs(F_mean)
    assert ripple < 0.10, (
        f"cell-crossing drag ripple {ripple:.1%} > 10% — the moving-boundary "
        f"path may have destabilised (fzs range {min(fzs):.5f}..{max(fzs):.5f})"
    )

    assert abs(ratio - 1.0) < 0.35, (
        f"drag {F_mean:.6f} vs Haberman-Sayre {F_expected:.6f} "
        f"(ratio {ratio:.4f}) — outside the documented 35% budget. "
        f"See this test's error budget before adjusting the tolerance."
    )
    # Guard the *sign* and order of magnitude explicitly: confinement at
    # lambda=0.27 must roughly double the unbounded Stokes drag.
    F_unbounded = 6.0 * math.pi * mu * R_sph * U
    assert F_mean > F_unbounded, (
        f"confined drag {F_mean:.6f} must exceed unbounded Stokes "
        f"{F_unbounded:.6f} (Haberman-Sayre enhancement is "
        f"{1/haberman_sayre_K(lam):.2f}x)"
    )


@pytest.mark.slow
def test_translating_body_mass_offset_is_bounded_not_a_leak():
    """A body sweeping the lattice must not *accumulate* mass error.

    This node has no fresh-node refill scheme (algorithm guide, Limitation #2):
    nodes the body vacates become fluid carrying whatever populations they
    collided to while "solid". The question that matters for the motivating
    use case is not whether the mass error is zero but whether it **grows**: a
    Re ~ 360 sphere run needs ~1e5 steps, so a per-step leak would compound
    into a wrecked simulation, whereas a bounded startup offset is benign.

    Measured behaviour (32x32x64, a=5, R_tube=14, U=0.004): the relative mass
    error rises to ~8e-4 over the first ~500 steps and then **saturates** —
    it is 8.01e-4 at step 1000 and 7.93e-4 at step 6000, having swept 24
    cells. It is therefore a bounded offset, and it very slowly *decays*.

    Two observations pin down what it is (and is not):

    * It is **not a moving-mask artefact**: holding the mask fixed with the
      same imposed wall velocity gives the same 8.00e-4 offset.
    * It **scales with the wall velocity** (3.4e-4 at U=0.002, 8.0e-4 at
      U=0.004), i.e. it is the initial density adjustment to switching on a
      moving boundary (the Ladd correction injects momentum at boundary
      links), not the body sweeping the lattice.

    So this test asserts the two properties that carry the physics: the offset
    stays small, and it does not grow once established.
    """
    nx = ny = 32
    nz = 64
    node = _sphere_in_tube(nx, ny, nz, R_pipe=14.0, R_sph=5.0, tau=0.9)
    state = node.initial_state()
    step = jax.jit(lambda st, bi: node.update(st, bi, 1.0))

    m0 = float(jnp.sum(state["f"]))
    U = 0.004
    z0 = nz / 2.0 - 8.0

    drift = {}
    for i in range(3000):   # sweeps 12 cells
        bi = {
            "body_position": jnp.array([nx / 2.0, ny / 2.0, z0 + U * i]),
            "body_velocity": jnp.array([0.0, 0.0, U]),
        }
        state = step(state, bi)
        if (i + 1) in (1000, 3000):
            drift[i + 1] = abs(float(jnp.sum(state["f"])) - m0) / m0

    # Bounded.
    assert drift[3000] < 2e-3, (
        f"mass offset {drift[3000]:.2e} larger than the documented ~8e-4 "
        f"startup offset — the moving-boundary path may have changed"
    )
    # And, the property that actually matters: not accumulating. Over steps
    # 1000->3000 the body sweeps 8 further cells; a per-cell leak would show
    # up here as growth.
    assert drift[3000] <= drift[1000] * 1.05, (
        f"mass error is ACCUMULATING, not a bounded offset: "
        f"{drift[1000]:.3e} at step 1000 -> {drift[3000]:.3e} at step 3000. "
        f"A fresh-node refill scheme would be required for long runs."
    )
