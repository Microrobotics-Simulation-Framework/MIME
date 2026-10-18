"""Verification benchmarks for ``SuperparamagneticResponseNode`` (MIME-NODE-009).

Six benchmarks:

- **MIME-VER-144** (analytical): the node reproduces the *measured* Landers 2025
  VSM datum — 72 memu at 30 mT for a 1.69 mm capsule — and, by exact volume
  scaling, the 1.4 mm variant the paper's own CFD uses. This is the anchor.
- **MIME-VER-145** (analytical): the Langevin low-field limit. ``L(x) -> x/3``
  exactly, so ``dm/dB|_0 = m_sat/(3 b0)``, verified against AD of the curve
  itself. This benchmark also *pins the tension*: the three quoted Landers
  numbers over-determine a 2-parameter Langevin and are mutually inconsistent,
  and the size of that inconsistency (-5.7 % / +6.8 % depending on which pair you
  pin) is asserted rather than papered over.
- **MIME-VER-146** (analytical): the force identity. For a Maxwell-consistent
  (symmetric) gradient, ``grad_B @ m == m(|B|) * grad|B|`` to machine precision.
  This is a real check of the whole chain, not a tautology: it holds *only* if
  the gradient is curl-free, and it establishes that the gradient force is
  conservative with potential ``integral_0^|B| m(b) db``.
- **MIME-VER-147** (analytical): the torque vanishes *structurally*. The node
  emits exact zeros; this asserts the emitted moment really is field-parallel to
  machine precision, so the zero is earned rather than hardcoded over a
  discrepancy.
- **MIME-VER-148** (cross-code): the two paths agree, and MIME-NODE-002 does not.
  An intrinsic ``chi = 1.98`` through ``chi_eff = chi/(1 + N chi)`` reproduces the
  measured moment; ``MagneticResponseNode``'s hardwired demag *ceiling* overshoots
  it by 2.5x. This is the benchmark that justifies the node existing at all.
- **MIME-VER-149** (regression): Earnshaw consistency. A field-aligned moment
  seeks field maxima, and ``|B|^2`` is subharmonic in free space (MIME-VER-143),
  so the gradient force can have no stable equilibrium. Asserted here as the
  positive-definiteness of ``div F`` in the linear regime.

Plus a ``GraphManager`` coupling test, which also pins the deliberate absence of
the ``orientation`` boundary input.

All benchmarks run in float64 (``pytest.mark.x64``): several claims here are
machine-precision claims about quantities of order 1e-5 A.m^2 and 1e-6 N, which
are meaningless at float32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.validation import (
    verification_benchmark, BenchmarkType,
)

from mime.nodes.robot.magnetic_response import MagneticResponseNode
from mime.nodes.robot.superparamagnetic_response import (
    DemagLimitedLinearCurve,
    LANDERS_B_MEASURED_T,
    LANDERS_CAPSULE_DIAMETER_M,
    LANDERS_CFD_DIAMETER_M,
    LANDERS_CHORD_SLOPE_AM2_PER_T,
    LANDERS_M_AT_30MT_AM2,
    LANDERS_M_SAT_AM2,
    LangevinCurve,
    MU_0,
    SuperparamagneticResponseNode,
    langevin,
    sphere_volume_m3,
)

pytestmark = pytest.mark.x64

DT = 1.0e-3


def _landers_node(diameter_m: float = LANDERS_CAPSULE_DIAMETER_M):
    return SuperparamagneticResponseNode(
        "capsule", DT,
        curve=LangevinCurve.landers_capsule(),
        volume_m3=sphere_volume_m3(diameter_m),
    )


def _langevin_lambert_cf(x: float, depth: int = 60) -> float:
    """INDEPENDENT oracle for L(x), via Lambert's continued fraction::

        L(x) = coth(x) - 1/x = x / (3 + x^2 / (5 + x^2 / (7 + x^2 / (9 + ...))))

    Structurally different from the implementation (which uses a Maclaurin series
    below |x| = 0.1 and the direct ``coth(x) - 1/x`` above), so agreement is
    evidence rather than tautology. Crucially it has **no cancellation anywhere**
    — every term is a sum of positives — which makes it trustworthy in exactly
    the small-x regime where the direct form falls apart (at x = 1e-4 the direct
    float64 form is already 4.3e-8 wrong; this is accurate to ~1e-16).

    Evaluated backwards from ``depth``; converges geometrically for the |x| <~ 10
    range used here.
    """
    acc = 2.0 * depth + 3.0
    for k in range(depth, -1, -1):
        acc = (2.0 * k + 3.0) + x * x / acc
    return x / acc


def _symmetric_traceless_gradient(rng, scale=0.5):
    """A Maxwell-consistent grad_B: symmetric (curl-free) and traceless (div-free)."""
    A = rng.normal(size=(3, 3))
    G = 0.5 * (A + A.T)
    G = G - np.eye(3) * np.trace(G) / 3.0
    return jnp.asarray(G * scale / np.abs(G).max())


# ── MIME-VER-144 ────────────────────────────────────────────────────────────

@verification_benchmark(
    benchmark_id="MIME-VER-144",
    description=(
        "Landers et al. (2025) VSM anchor: a 1.69 mm superparamagnetic capsule "
        "has a moment of 72 memu (7.2e-5 A.m^2) at 30 mT applied field. Exact "
        "sphere volume scaling then reproduces the 1.4 mm variant used by the "
        "paper's own CFD."
    ),
    node_type="SuperparamagneticResponseNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "|m(30 mT)| == 7.2e-5 A.m^2 to 1e-9 relative for d = 1.69 mm; volume "
        "scaling to the 1.4 mm CFD variant exact to 1e-12 relative (ratio "
        "0.5685) in the moment, the saturation AND the EMITTED FORCE out of "
        "update() — the path the graph consumes, not merely the moment_am2() "
        "helper; high-field asymptote m -> m_sat(1 - b0/B) to 1e-9 relative — "
        "note m never reaches m_sat = 2.25e-4 at any finite field"
    ),
    references=("Landers2025",),
)
def test_landers_capsule_reproduces_the_measured_moment():
    node = _landers_node()

    # The measured datum, straight out of update().
    out = node.update(
        node.initial_state(),
        {"field_vector": jnp.array([0.0, 0.0, LANDERS_B_MEASURED_T]),
         "field_gradient": jnp.zeros((3, 3))},
        DT,
    )
    m = float(jnp.linalg.norm(out["magnetization"]))
    assert m == pytest.approx(LANDERS_M_AT_30MT_AM2, rel=1e-9), (
        f"m(30 mT) = {m / 1e-6:.4f} memu, expected 72.0 memu")

    # Saturation is the b0/B -> 0 ASYMPTOTE, not a value reached at any finite
    # field: L(x) -> 1 - 1/x, so m(B) -> m_sat (1 - b0/B). A superparamagnet
    # never truly saturates — at 100 T (3400x the operating point) the Landers
    # curve is still 0.03 % short. Asserting m(100 T) == m_sat would be wrong.
    assert node.m_sat_am2() == pytest.approx(LANDERS_M_SAT_AM2, rel=1e-12)
    b0 = LangevinCurve.landers_capsule().b0_t
    for B in (1.0, 100.0, 1.0e4):
        assert float(node.moment_am2(B)) == pytest.approx(
            LANDERS_M_SAT_AM2 * (1.0 - b0 / B), rel=1e-9), (
            f"Langevin high-field asymptote m -> m_sat(1 - b0/B) broken at {B} T")
    assert float(node.moment_am2(1.0e4)) == pytest.approx(
        LANDERS_M_SAT_AM2, rel=1e-5)   # only NOW is it within 1e-5

    # The capsule is NOT saturated at 30 mT — it sits in the near-linear regime,
    # which is the premise of the whole measured-curve approach.
    assert m / LANDERS_M_SAT_AM2 == pytest.approx(0.32, rel=1e-6)

    # Exact volume scaling gives the paper's own CFD size for free.
    cfd = _landers_node(LANDERS_CFD_DIAMETER_M)
    ratio = (LANDERS_CFD_DIAMETER_M / LANDERS_CAPSULE_DIAMETER_M) ** 3
    assert float(cfd.moment_am2(LANDERS_B_MEASURED_T)) == pytest.approx(
        ratio * LANDERS_M_AT_30MT_AM2, rel=1e-12)
    assert cfd.m_sat_am2() == pytest.approx(ratio * LANDERS_M_SAT_AM2, rel=1e-12)
    # ...and it is a 43 % swing in the force. The source never reconciles this.
    assert ratio == pytest.approx(0.5685, rel=1e-3)

    # ── THE SCALING MUST HOLD ON THE PATH THAT ACTUALLY FEEDS THE BODY.
    # update() and the moment_am2() helper apply volume_scale through two
    # SEPARATE expressions (self.params["volume_scale"] vs self.volume_scale),
    # so asserting the helper does not protect the emitted force — the graph
    # consumes update()'s output, not the helper's. Drive BOTH diameters through
    # update() with a real gradient and pin the force itself.
    GRAD = 0.5                                   # T/m, the Landers-scale figure
    G = jnp.diag(jnp.array([0.0, 0.0, GRAD]))
    B_vec = jnp.array([0.0, 0.0, LANDERS_B_MEASURED_T])

    def _emitted(node):
        return node.update(node.initial_state(),
                           {"field_vector": B_vec, "field_gradient": G}, DT)

    out_ref, out_cfd = _emitted(_landers_node()), _emitted(cfd)

    # The 1.69 mm arm: F_z = m * dBz/dz, hand-computed from the measured datum.
    assert float(out_ref["magnetic_force"][2]) == pytest.approx(
        GRAD * LANDERS_M_AT_30MT_AM2, rel=1e-9)

    # The 1.4 mm arm: moment AND force both scale exactly by the volume ratio.
    assert float(jnp.linalg.norm(out_cfd["magnetization"])) == pytest.approx(
        ratio * LANDERS_M_AT_30MT_AM2, rel=1e-12)
    assert float(out_cfd["magnetic_force"][2]) == pytest.approx(
        GRAD * ratio * LANDERS_M_AT_30MT_AM2, rel=1e-12)
    assert float(out_cfd["magnetic_force"][2]) == pytest.approx(
        ratio * float(out_ref["magnetic_force"][2]), rel=1e-12)

    # Stated as the physical claim MIME-ANO-105 makes: the size ambiguity is a
    # 43 % force swing, and it is the EMITTED force that swings.
    f_ratio = (float(out_cfd["magnetic_force"][2])
               / float(out_ref["magnetic_force"][2]))
    assert f_ratio == pytest.approx(0.5685, rel=1e-3), (
        f"the 1.4-vs-1.69 mm force ratio is {f_ratio:.4f}, not the 0.5685 that "
        f"MIME-ANO-105 quotes as the bound on every Landers-derived claim")


# ── MIME-VER-145 ────────────────────────────────────────────────────────────

@verification_benchmark(
    benchmark_id="MIME-VER-145",
    description=(
        "Langevin low-field limit L(x) -> x/3, giving dm/dB|_0 = m_sat/(3 b0); "
        "and the documented tension between the three quoted Landers numbers, "
        "which over-determine a two-parameter Langevin and are mutually "
        "inconsistent by ~6 %."
    ),
    node_type="SuperparamagneticResponseNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "L(x) matches an independent Lambert continued-fraction oracle to 1e-12 "
        "relative over x in [1e-8, 10] spanning both branches and the crossover; "
        "the low-field deviation of L(x)/x from 1/3 follows x^2/15 to 1e-3 "
        "relative; AD tangent matches m_sat/(3 b0) to 1e-9; slope-pinned fit "
        "gives -5.65 % at 30 mT and point-pinned fit gives +6.81 % on the "
        "tangent — both asserted, neither hidden"
    ),
    references=("Bean1959", "Landers2025"),
)
def test_langevin_low_field_limit_and_the_landers_tension():
    # ── L(x) against an INDEPENDENT oracle (Lambert's continued fraction),
    # across the series branch, the crossover, and the direct branch.
    for x in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 0.0999, 0.1, 0.1001,
              0.5, 1.0, 3.0, 10.0):
        assert float(langevin(jnp.asarray(x))) == pytest.approx(
            _langevin_lambert_cf(x), rel=1e-12), f"L(x) wrong at x={x}"

    # The limit proper: L(x)/x -> 1/3. Note the approach is QUADRATIC —
    # L(x)/x = 1/3 - x^2/45 + ... — so the deviation from 1/3 at finite x is real
    # physics, not error (6.7e-10 relative at x = 1e-4, 6.7e-8 at x = 1e-3). A
    # test that demanded 1/3 to 1e-10 at x = 1e-3 would be asserting something
    # false about the Langevin function.
    for x in (1e-8, 1e-6):
        assert float(langevin(jnp.asarray(x))) / x == pytest.approx(
            1.0 / 3.0, rel=1e-11)
    for x in (1e-4, 1e-3, 1e-2):
        dev = abs(float(langevin(jnp.asarray(x))) / x - 1.0 / 3.0) / (1.0 / 3.0)
        assert dev == pytest.approx(x ** 2 / 15.0, rel=1e-3), (
            f"low-field deviation should be x^2/15 at x={x}, got {dev:.3e}")

    # THE SERIES BRANCH IS LOAD-BEARING, not defensive decoration: the direct
    # form coth(x) - 1/x cancels catastrophically at small x. At x = 1e-4 it is
    # already ~4e-8 wrong in float64 — 5 orders worse than the series — which is
    # what the |x| < 0.1 cutoff exists to avoid.
    x = 1e-4
    direct_f64 = 1.0 / np.tanh(x) - 1.0 / x
    oracle = _langevin_lambert_cf(x)
    assert abs(direct_f64 - oracle) / oracle > 1e-9, (
        "the direct form is unexpectedly accurate at x=1e-4; if this fails the "
        "series branch may no longer be needed — re-derive the cutoff")
    assert abs(float(langevin(jnp.asarray(x))) - oracle) / oracle < 1e-14

    # No seam at the series/direct crossover. Test each branch against the
    # ORACLE rather than against each other: L genuinely changes by L'(0.1)*2*eps
    # ~ 6.7e-10 across the gap, so demanding L(0.1-eps) == L(0.1+eps) to 1e-10
    # would be asserting that L is locally constant, which it is not.
    eps = 1e-9
    for x in (0.1 - eps, 0.1, 0.1 + eps):
        assert float(langevin(jnp.asarray(x))) == pytest.approx(
            _langevin_lambert_cf(x), rel=1e-12), f"seam at the crossover, x={x}"

    # The step across the crossover is exactly the slope, not a discontinuity.
    jump = float(langevin(jnp.asarray(0.1 + eps))) - float(langevin(jnp.asarray(0.1 - eps)))
    slope = float(jax.grad(langevin)(0.1))
    assert jump == pytest.approx(slope * 2 * eps, rel=1e-5), (
        f"crossover jump {jump:.3e} is not the local slope {slope * 2 * eps:.3e}")

    # ── The analytic tangent m_sat/(3 b0) matches AD of the curve.
    curve = LangevinCurve.landers_capsule()
    analytic = curve.m_sat_am2 / (3.0 * curve.b0_t)
    assert curve.initial_susceptibility_am2_per_t() == pytest.approx(
        analytic, rel=1e-9)

    # ── THE TENSION. 72 memu == 2.4 memu/mT * 30 mT EXACTLY, so the quoted
    # "low-field slope" is a chord through the 30 mT point, not a tangent.
    assert LANDERS_CHORD_SLOPE_AM2_PER_T * LANDERS_B_MEASURED_T == \
        pytest.approx(LANDERS_M_AT_30MT_AM2, rel=1e-12)

    # L is strictly concave on x > 0, so m(30 mT) < tangent * 30 mT = 72 memu
    # for ANY finite b0: no Langevin with finite saturation fits all three.
    V = sphere_volume_m3(LANDERS_CAPSULE_DIAMETER_M)

    # Fit A — pin (m_sat, chord slope read as a tangent): misses the 30 mT point low.
    fit_a = LangevinCurve.from_saturation_and_slope(
        LANDERS_M_SAT_AM2, LANDERS_CHORD_SLOPE_AM2_PER_T, V)
    assert fit_a.b0_t == pytest.approx(31.25e-3, rel=1e-9)
    m_a = float(fit_a.moment_am2(LANDERS_B_MEASURED_T))
    assert m_a == pytest.approx(67.93e-6, rel=1e-3)
    err_a = (m_a - LANDERS_M_AT_30MT_AM2) / LANDERS_M_AT_30MT_AM2
    assert err_a == pytest.approx(-0.0565, abs=1e-3), (
        f"slope-pinned fit is {100 * err_a:+.2f} % at 30 mT, expected -5.65 %")

    # Fit B — pin (m_sat, 30 mT point): exact there, overshoots the tangent.
    fit_b = LangevinCurve.landers_capsule()
    assert fit_b.b0_t == pytest.approx(29.258e-3, rel=1e-3)
    assert float(fit_b.moment_am2(LANDERS_B_MEASURED_T)) == pytest.approx(
        LANDERS_M_AT_30MT_AM2, rel=1e-9)
    tangent_b = fit_b.initial_susceptibility_am2_per_t()
    err_b = (tangent_b - LANDERS_CHORD_SLOPE_AM2_PER_T) / LANDERS_CHORD_SLOPE_AM2_PER_T
    assert err_b == pytest.approx(0.0681, abs=1e-3), (
        f"point-pinned fit tangent is {100 * err_b:+.2f} %, expected +6.81 %")

    # landers_capsule() is Fit B: the operating point is exact, the tangent is not.
    assert fit_b.b0_t != pytest.approx(fit_a.b0_t, rel=1e-3)

    # The concavity that makes the two irreconcilable: the chord always
    # under-runs the tangent, approaching it only as b0 -> inf (m_sat -> inf).
    for b0_mt in (10.0, 20.0, 31.25, 50.0, 100.0, 1000.0):
        c = LangevinCurve(3.0 * LANDERS_CHORD_SLOPE_AM2_PER_T * b0_mt * 1e-3,
                          b0_mt * 1e-3, V)
        assert float(c.moment_am2(LANDERS_B_MEASURED_T)) < LANDERS_M_AT_30MT_AM2


# ── MIME-VER-146 ────────────────────────────────────────────────────────────

@verification_benchmark(
    benchmark_id="MIME-VER-146",
    description=(
        "Gradient-force identity for a field-aligned moment: for a "
        "Maxwell-consistent (curl-free, hence symmetric) gradient, "
        "F = grad_B @ m is identically m(|B|) * grad|B|, establishing that the "
        "force is conservative with potential integral_0^|B| m(b) db. Also pins "
        "the index convention against an asymmetric hand-built gradient."
    ),
    node_type="SuperparamagneticResponseNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "|grad_B @ m - m(|B|) grad|B|| / |F| < 1e-12 over 200 random "
        "symmetric-traceless gradients; hand-computed force matches to 1e-12; "
        "the identity FAILS for an asymmetric gradient (the check has teeth)"
    ),
    references=("Abbott2020", "Landers2025"),
)
def test_force_equals_moment_times_grad_field_magnitude():
    node = _landers_node()
    rng = np.random.default_rng(20250715)

    for _ in range(200):
        B = jnp.asarray(rng.normal(size=3) * 0.03)
        G = _symmetric_traceless_gradient(rng)

        out = node.update(node.initial_state(),
                          {"field_vector": B, "field_gradient": G}, DT)
        F = out["magnetic_force"]

        # Independent form: F = m(|B|) * grad|B|, with grad|B| = grad_B^T @ B_hat.
        b_mag = jnp.linalg.norm(B)
        b_hat = B / b_mag
        F_ref = node.moment_am2(b_mag) * (G.T @ b_hat)

        rel = float(jnp.linalg.norm(F - F_ref) / jnp.linalg.norm(F_ref))
        assert rel < 1e-12, f"force identity broken: rel err {rel:.3e}"

    # The identity has teeth: it holds BECAUSE the gradient is symmetric.
    # An asymmetric (unphysical, curl-carrying) gradient breaks it, which is
    # exactly the silent hazard the node's metadata warns about.
    B = jnp.array([0.0, 0.0, 0.030])
    G_asym = jnp.array([[0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0]])
    out = node.update(node.initial_state(),
                      {"field_vector": B, "field_gradient": G_asym}, DT)
    F_asym = out["magnetic_force"]
    F_ref_asym = node.moment_am2(0.030) * (G_asym.T @ jnp.array([0.0, 0.0, 1.0]))
    assert float(jnp.linalg.norm(F_asym - F_ref_asym)) > 1e-9

    # Hand computation pinning the jacrev index convention grad_B[i][j] = dB_i/dx_j.
    # m points along +z; only dBx/dz is non-zero; so F = (m * dBx/dz, 0, 0).
    m = float(node.moment_am2(0.030))
    np.testing.assert_allclose(np.asarray(F_asym), np.array([m, 0.0, 0.0]),
                               rtol=1e-12)

    # Linear-regime cross-check against the textbook chi V grad(|B|^2) / (2 mu_0):
    # the one-half belongs to the ENERGY, and the force still comes out as
    # (m.grad)B. Use a small field where the Langevin is linear.
    curve = LangevinCurve.landers_capsule()
    chi_eff_V = curve.initial_susceptibility_am2_per_t() * MU_0   # = chi_eff * V
    B_small = jnp.array([0.0, 0.0, 1e-4])
    G = _symmetric_traceless_gradient(np.random.default_rng(5))
    out = node.update(node.initial_state(),
                      {"field_vector": B_small, "field_gradient": G}, DT)
    # grad(|B|^2) = 2 * grad_B^T @ B  for symmetric grad_B.
    F_textbook = chi_eff_V / (2.0 * MU_0) * (2.0 * (G.T @ B_small))
    rel = float(jnp.linalg.norm(out["magnetic_force"] - F_textbook)
                / jnp.linalg.norm(F_textbook))
    assert rel < 1e-6, f"linear-regime textbook force mismatch: {rel:.3e}"


# ── MIME-VER-147 ────────────────────────────────────────────────────────────

@verification_benchmark(
    benchmark_id="MIME-VER-147",
    description=(
        "The magnetic torque on a field-aligned superparamagnetic moment "
        "vanishes STRUCTURALLY: m = m(|B|) B_hat is parallel to B by "
        "construction, so m x B == 0 identically rather than approximately. The "
        "node emits exact zeros; this asserts the emitted moment really is "
        "field-parallel to machine precision, so the zero is earned."
    ),
    node_type="SuperparamagneticResponseNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "magnetic_torque == 0.0 EXACTLY (bitwise) for all inputs; and "
        "|cross(m, B)| / (|m||B|) < 1e-14, i.e. the emitted moment is parallel "
        "so the zero is not covering a discrepancy"
    ),
    references=("Landers2025",),
)
def test_torque_vanishes_structurally_for_a_field_aligned_moment():
    node = _landers_node()
    rng = np.random.default_rng(31415)

    for _ in range(200):
        B = jnp.asarray(rng.normal(size=3) * rng.uniform(1e-4, 0.1))
        G = _symmetric_traceless_gradient(rng, scale=rng.uniform(0.01, 1.0))
        out = node.update(node.initial_state(),
                          {"field_vector": B, "field_gradient": G}, DT)

        # (a) The emitted torque is EXACTLY zero — not small, zero.
        assert np.all(np.asarray(out["magnetic_torque"]) == 0.0)

        # (b) The zero is justified: the moment really is parallel to B, so
        # m x B is zero up to roundoff. Were it not, the exact zero would be
        # hiding a real torque.
        m = out["magnetization"]
        sin_angle = float(jnp.linalg.norm(jnp.cross(m, B))
                          / (jnp.linalg.norm(m) * jnp.linalg.norm(B)))
        assert sin_angle < 1e-14, f"moment is not field-parallel: {sin_angle:.3e}"

    # The consequence, stated as a test: this node cannot reorient a body. A
    # strong uniform field produces a large moment and NO torque whatsoever.
    out = node.update(node.initial_state(),
                      {"field_vector": jnp.array([0.05, 0.0, 0.0]),
                       "field_gradient": jnp.zeros((3, 3))}, DT)
    assert float(jnp.linalg.norm(out["magnetization"])) > 1e-5
    assert np.all(np.asarray(out["magnetic_torque"]) == 0.0)

    # The zero is stronger than the point-dipole argument. One might worry that
    # the DISTRIBUTED force over a body in a GRADIENT exerts a net torque about
    # the centre of mass even where the local m x B vanishes. For a sphere with
    # uniform field-aligned M it does not:
    #
    #   tau = int r x (M.grad)B(r) dV = (V R^2/5) eps_ijk M_l d_l d_j B_k = 0
    #
    # since d_j B_k is SYMMETRIC in (j,k) for a curl-free field while eps_ijk is
    # ANTISYMMETRIC in (j,k). Verified here on random Maxwell-consistent second
    # derivative tensors: the emitted zero follows from Maxwell + sphericity, not
    # merely from idealising the bead as a point dipole.
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
    for _ in range(20):
        # d_l d_j B_k, symmetric in (j,k) because B = -grad(psi) => d_j B_k is a
        # Hessian of psi and hence symmetric.
        T = rng.normal(size=(3, 3, 3))
        T = 0.5 * (T + T.transpose(0, 2, 1))
        M = rng.normal(size=3)
        tau = np.einsum("ijk,l,ljk->i", eps, M, T)
        assert np.abs(tau).max() < 1e-14, (
            "distributed-force torque on a sphere should vanish identically by "
            "the eps/Hessian symmetry contraction")


# ── MIME-VER-148 ────────────────────────────────────────────────────────────

@verification_benchmark(
    benchmark_id="MIME-VER-148",
    description=(
        "Two-path consistency, and why MIME-NODE-002 cannot be used here. The "
        "gated first-principles path chi_eff = chi/(1 + N chi) with an intrinsic "
        "chi = 1.98 and N = 1/3 reproduces the measured Landers moment, "
        "confirming the measurement is consistent with an ordinary material. "
        "MagneticResponseNode's hardwired demagnetization CEILING (chi_eff = 3, "
        "the chi -> inf limit) overshoots the same measurement by 2.5x."
    ),
    node_type="SuperparamagneticResponseNode",
    benchmark_type=BenchmarkType.CROSS_CODE,
    acceptance_criteria=(
        "chi = 1.9816 through chi_eff = chi/(1+N chi) reproduces 7.2e-5 A.m^2 at "
        "30 mT to 1e-6 relative; MagneticResponseNode gives 1.81e-4 A.m^2, a "
        "ratio of 2.51 +/- 0.01; chi_eff -> 1/N = 3 as chi -> inf"
    ),
    references=("Landers2025", "Osborn1945"),
)
def test_first_principles_path_agrees_and_the_demag_ceiling_does_not():
    V = sphere_volume_m3(LANDERS_CAPSULE_DIAMETER_M)
    N_SPHERE = 1.0 / 3.0
    B = jnp.array([0.0, 0.0, LANDERS_B_MEASURED_T])

    # ── The intrinsic chi that reproduces the measurement.
    # chi_eff = m / (V * |B| / mu_0); chi = chi_eff / (1 - N chi_eff).
    chi_eff_needed = LANDERS_M_AT_30MT_AM2 / (V * LANDERS_B_MEASURED_T / MU_0)
    chi_intrinsic = chi_eff_needed / (1.0 - N_SPHERE * chi_eff_needed)
    assert chi_eff_needed == pytest.approx(1.1933, rel=1e-3)
    assert chi_intrinsic == pytest.approx(1.9816, rel=1e-3)

    curve = DemagLimitedLinearCurve(
        intrinsic_susceptibility=chi_intrinsic,
        demag_factor=N_SPHERE,
        m_sat_am2=LANDERS_M_SAT_AM2,
        reference_volume_m3=V,
    )
    assert curve.effective_susceptibility == pytest.approx(chi_eff_needed, rel=1e-9)

    fp_node = SuperparamagneticResponseNode("fp", DT, curve=curve)
    out = fp_node.update(fp_node.initial_state(),
                         {"field_vector": B, "field_gradient": jnp.zeros((3, 3))},
                         DT)
    m_fp = float(jnp.linalg.norm(out["magnetization"]))
    assert m_fp == pytest.approx(LANDERS_M_AT_30MT_AM2, rel=1e-6), (
        "the gated first-principles path does not reproduce the measurement")

    # It sits comfortably BELOW the ceiling — the measurement is consistent with
    # an ordinary dense iron-oxide loading, not with a pathological material.
    assert curve.effective_susceptibility < 3.0
    assert chi_intrinsic < 3.0

    # chi_eff -> 1/N as chi -> inf: that limit IS MagneticResponseNode's assumption.
    ceiling = DemagLimitedLinearCurve(1e12, N_SPHERE, LANDERS_M_SAT_AM2, V)
    assert ceiling.effective_susceptibility == pytest.approx(3.0, rel=1e-9)

    # ── MIME-NODE-002 on the same problem. n_axi = n_rad = 1/3 is the sphere,
    # the ONLY choice that avoids inventing anisotropy (and hence torque).
    legacy = MagneticResponseNode(
        "legacy", DT, volume_m3=V, n_axi=N_SPHERE, n_rad=N_SPHERE, m_sat=0.0)
    legacy_out = legacy.update(
        legacy.initial_state(),
        {"field_vector": B, "field_gradient": jnp.zeros((3, 3)),
         "orientation": jnp.array([1.0, 0.0, 0.0, 0.0])},
        DT)
    # NOTE MagneticResponseNode's "magnetization" is A/m (it multiplies by volume
    # inside the force/torque), so recover the moment as m * V.
    m_legacy = float(jnp.linalg.norm(legacy_out["magnetization"])) * V
    assert m_legacy == pytest.approx(1.81e-4, rel=1e-2)
    assert m_legacy / LANDERS_M_AT_30MT_AM2 == pytest.approx(2.51, abs=0.01), (
        f"MagneticResponseNode gives {m_legacy / 1e-6:.1f} memu vs the measured "
        f"72.0 memu — the 2.5x overshoot that motivates MIME-NODE-009")

    # And it cannot be fixed by detuning n: the constraint n_axi + 2 n_rad = 1
    # pins the sphere, and any other split is anisotropic and produces a
    # SPURIOUS TORQUE on what is physically an isotropic bead.
    aniso = MagneticResponseNode(
        "aniso", DT, volume_m3=V, n_axi=0.6, m_sat=0.0)   # n_rad = 0.2
    aniso_out = aniso.update(
        aniso.initial_state(),
        {"field_vector": jnp.array([0.02, 0.0, 0.022]),   # off-axis
         "field_gradient": jnp.zeros((3, 3)),
         "orientation": jnp.array([1.0, 0.0, 0.0, 0.0])},
        DT)
    assert float(jnp.linalg.norm(aniso_out["magnetic_torque"])) > 0.0, (
        "detuning the demag factors invents a torque on an isotropic bead")


# ── MIME-VER-149 ────────────────────────────────────────────────────────────

@verification_benchmark(
    benchmark_id="MIME-VER-149",
    description=(
        "Earnshaw consistency for the gradient-pull force. A field-aligned "
        "moment seeks field maxima; |B|^2 is subharmonic in free space "
        "(laplacian(|B|^2) = 2 ||grad_B||_F^2 >= 0, MIME-VER-143), so no stable "
        "trap can exist. In the linear regime the force is F = (chi_eff V / "
        "2 mu_0) grad(|B|^2), whose divergence is therefore non-negative — a "
        "3-D force field that can never be everywhere-inward around a point."
    ),
    node_type="SuperparamagneticResponseNode",
    benchmark_type=BenchmarkType.REGRESSION,
    acceptance_criteria=(
        "div F >= 0 for a linear-regime moment in a Maxwell-consistent field, "
        "for 100 random symmetric-traceless gradients; div F == 0 only when "
        "grad_B == 0"
    ),
    references=("Landers2025",),
)
def test_gradient_force_admits_no_stable_trap():
    # Work in the linear regime, where m = k|B| and the force is
    # F = (k / 2|B|) grad(|B|^2) with k constant, so div F = k * div(grad|B|... )
    # -- concretely: F = k * grad_B^T @ B, and for symmetric traceless grad_B in
    # a locally-linear field B(x) = B0 + G x, div F = k * ||G||_F^2 >= 0.
    curve = LangevinCurve.landers_capsule()
    k = curve.initial_susceptibility_am2_per_t()
    node = SuperparamagneticResponseNode("bead", DT, curve=curve)
    rng = np.random.default_rng(1618)

    for _ in range(100):
        G = _symmetric_traceless_gradient(rng, scale=rng.uniform(0.05, 1.0))
        B0 = jnp.asarray(rng.normal(size=3) * 1e-5)      # small: linear regime

        def force_at(x):
            B = B0 + G @ x
            return node.update(node.initial_state(),
                               {"field_vector": B, "field_gradient": G},
                               DT)["magnetic_force"]

        div_F = float(jnp.trace(jax.jacrev(force_at)(jnp.zeros(3))))

        # Analytically div F = k ||G||_F^2 for a linear moment and a symmetric
        # traceless G. Non-negative => |B|^2 subharmonic => no stable trap.
        expected = k * float(jnp.sum(G * G))
        assert div_F >= 0.0, f"div F = {div_F:.3e} < 0 would permit a trap"
        assert div_F == pytest.approx(expected, rel=1e-4)

    # A zero gradient is the only way to get div F == 0.
    def force_uniform(x):
        return node.update(node.initial_state(),
                           {"field_vector": jnp.array([0.0, 0.0, 1e-5]),
                            "field_gradient": jnp.zeros((3, 3))},
                           DT)["magnetic_force"]
    assert float(jnp.trace(jax.jacrev(force_uniform)(jnp.zeros(3)))) == \
        pytest.approx(0.0, abs=1e-30)


# ── Graph coupling ──────────────────────────────────────────────────────────

def test_couples_to_a_field_source_and_a_rigid_body_in_a_graph():
    """End-to-end wiring: an external field node -> this node -> a rigid body,
    with the magnetic load delivered on additive edges."""
    from maddening.core.graph_manager import GraphManager
    from mime.nodes.actuation.external_magnetic_field import (
        ExternalMagneticFieldNode,
    )
    from mime.nodes.robot.rigid_body import RigidBodyNode

    gm = GraphManager()
    field = ExternalMagneticFieldNode("field", DT)
    bead = _landers_node()
    # fluid_viscosity_pa_s has no default since MIME 0.3.0; 8.5e-4 was the default.
    body = RigidBodyNode("body", DT, mass=8.0e-6, fluid_viscosity_pa_s=8.5e-4)

    gm.add_node(field)
    gm.add_node(bead)
    gm.add_node(body)
    gm.add_edge("field", "capsule", "field_vector", "field_vector")
    gm.add_edge("field", "capsule", "field_gradient", "field_gradient")
    gm.add_edge("capsule", "body", "magnetic_force", "magnetic_force", additive=True)
    gm.add_edge("capsule", "body", "magnetic_torque", "magnetic_torque", additive=True)

    issues = [i for i in gm.validate() if i.startswith("ERROR")]
    assert issues == [], issues

    gm.step()
    assert np.all(np.isfinite(np.asarray(gm.get_node_state("capsule")["magnetic_force"])))


def test_orientation_edge_is_a_harmless_no_op():
    """This node deliberately declares NO ``orientation`` boundary input (a
    field-aligned isotropic moment has no frame to rotate into). The existing
    ``_MagneticEffect.build()`` hardwires a body.orientation -> response
    edge regardless, so pin that the edge validates clean and is ignored —
    which is what makes omitting the dead input safe."""
    from maddening.core.graph_manager import GraphManager
    from mime.nodes.robot.rigid_body import RigidBodyNode

    gm = GraphManager()
    bead = _landers_node()
    # fluid_viscosity_pa_s has no default since MIME 0.3.0; 8.5e-4 was the default.
    body = RigidBodyNode("body", DT, mass=8.0e-6, fluid_viscosity_pa_s=8.5e-4)
    gm.add_node(bead)
    gm.add_node(body)
    # Exactly the edge _MagneticEffect.build() adds unconditionally.
    gm.add_edge("body", "capsule", "orientation", "orientation")
    gm.add_edge("capsule", "body", "magnetic_force", "magnetic_force", additive=True)

    issues = [i for i in gm.validate() if i.startswith("ERROR")]
    assert issues == [], issues

    gm.step()   # must not raise: the undeclared target field is simply unread
    assert np.all(np.isfinite(np.asarray(gm.get_node_state("capsule")["magnetic_force"])))
