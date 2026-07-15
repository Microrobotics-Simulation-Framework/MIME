"""Verification benchmarks for ``CoilArrayFieldNode`` (MIME-NODE-103).

Four benchmarks:

- **MIME-VER-140** (cross-code): the actuation matrix $A_b(p)$ and the gradient
  5-vector $G_5(p, i)$ must reproduce ETH's ``mag_manip`` 3.0.0 reference
  library on the *real* calibrated Navion_2 (3-coil) and CardioMag (8-coil)
  models, at 50 in-workspace positions each. This pins our independent
  reimplementation of MPEM against the upstream implementation. The golden data
  lives in ``tests/data/emns_mpem_oracle.npz``; ``mag_manip`` itself is not a
  MIME dependency (its wheel is built against the numpy 1.x ABI and hard-fails
  on numpy >= 2).
- **MIME-VER-141** (analytical): every shipped ETH calibration is order 1, and
  an order-1 exterior multipole term *is* a point dipole. This benchmark asserts
  the general-order Legendre path reduces exactly to the closed-form dipole —
  cross-checked against MIME's own independently-validated
  ``permanent_magnet._b_point_dipole`` (MIME-NODE-101, MIME-VER-110).
- **MIME-VER-142** (analytical): Maxwell consistency. Because $B = -\\nabla\\psi$
  with $\\psi$ harmonic, $\\nabla\\cdot B = 0$ and $\\nabla\\times B = 0$ hold
  *structurally* — they are never enforced numerically. This benchmark asserts
  they hold to machine precision, which is the property that makes MPEM
  preferable to a data-driven field fit (whose gradient, and hence force, can
  violate Maxwell even when the field looks accurate).
- **MIME-VER-143** (analytical): Earnshaw's theorem in its rigorous pointwise
  form. Each Cartesian component of a curl-free, divergence-free field is
  harmonic, giving the exact identity
  $\\nabla^2(|B|^2) = 2\\lVert\\nabla B\\rVert_F^2 \\geq 0$ — i.e. $|B|^2$ is
  *subharmonic* and so attains no interior maximum. That is the mechanism behind
  Earnshaw's theorem, and why a field-aligned (paramagnetic) moment, which seeks
  field maxima, can never be stably trapped.

  Note what this benchmark deliberately does **not** claim. An earlier draft
  asserted that a single-sided planar array can *only* pull toward its coils and
  can never push away. That is false: Earnshaw forbids a field *maximum*, which
  is not the same as fixing the sign of $\\nabla|B|$ along any axis. Measured on
  the calibrated Navion_2, the outward direction does occur — in ~0.2 % of random
  (position, current) samples, at ordinary field strengths (12–22 mT),
  concentrated within ~15 mm of the coil-facing workspace boundary. The inward
  pull is a strong empirical tendency, pinned by
  ``test_planar_array_inward_pull_is_a_tendency_not_an_invariant`` rather than
  dressed up as a theorem.

Plus JAX-traceability (``jit``, ``vmap``, ``grad``) and a ``GraphManager``
coupling test.

All benchmarks run in float64 (``pytest.mark.x64``): the machine-precision
claims above are meaningless at float32, where ``B_Coeff`` ~ 1e-4 and
``B`` ~ 1e-3 T sit near the low-magnitude end of the dynamic range.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.validation import (
    verification_benchmark, BenchmarkType,
)

from mime.nodes.actuation.coil_array_field import (
    BUNDLED_CALIBRATIONS,
    CoilArrayFieldNode,
    MpemCalibration,
    _b_total_world,
    _source_field,
    field_actuation_matrix,
    gradient5,
    h_erf,
    load_mpem_calibration,
)
from mime.nodes.actuation.permanent_magnet import _b_point_dipole, MU_0

# x64 everywhere in this module — see module docstring. The conftest fixture
# scopes this so it cannot leak into the rest of the session.
pytestmark = pytest.mark.x64

_ORACLE_PATH = Path(__file__).resolve().parents[1] / "data" / "emns_mpem_oracle.npz"

# Oracle key prefix -> bundled calibration name. All three shipped systems are
# covered, including OctoMag — the system Landers et al. (2025) actually used,
# and the worst-conditioned of the three.
_ORACLE_SYSTEMS = {"navion": "navion_2", "cmag": "cardiomag", "octomag": "octomag"}

# Cross-code agreement floor. Two independent implementations of the same
# algebra sum the same near-cancelling source contributions in different orders,
# so they cannot agree better than the conditioning allows:
#
#     achievable agreement  ~  eps_f64 / field_cancellation_ratio
#
# Measured against mag_manip 3.0.0 (float64, eps ~2.2e-16):
#
#     model      field_cancel   predicted floor   measured
#     cardiomag     2.5e-1          8.8e-16       8.6e-16
#     navion_2      8.0e-3          2.8e-14       4.0e-14
#     octomag       6.9e-5          3.2e-12       1.5e-12
#
# The prediction tracks measurement to within ~1.5x across a 3600x spread, so
# the tolerance is derived from the conditioning rather than pinned flat. A flat
# bound would either fail OctoMag for arithmetic it cannot avoid, or be so loose
# it stopped testing CardioMag at all. _CROSS_CODE_SLACK is the safety factor
# over the predicted floor.
_EPS_F64 = float(np.finfo(np.float64).eps)
_CROSS_CODE_SLACK = 10.0


def _oracle():
    if not _ORACLE_PATH.exists():
        pytest.skip(f"MPEM oracle not found at {_ORACLE_PATH}")
    return np.load(_ORACLE_PATH, allow_pickle=True)


def _cal(name: str) -> MpemCalibration:
    try:
        return load_mpem_calibration(name)
    except FileNotFoundError as exc:  # data/ is not installed as package data
        pytest.skip(str(exc))


def _grad_b(p, currents, cal: MpemCalibration) -> np.ndarray:
    """Full 3x3 ∇B at p for a current vector, zero background."""
    pos = jnp.asarray(cal.source_position)
    d = jnp.asarray(cal.source_direction)
    b = jnp.asarray(cal.b_coeff)
    zero = jnp.zeros(3)
    fn = lambda t: _b_total_world(t, jnp.asarray(currents), pos, d, b, zero)
    return np.asarray(jax.jacrev(fn)(jnp.asarray(p)))


# ---------------------------------------------------------------------------
# MIME-VER-140 — cross-code vs mag_manip
# ---------------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-140",
    description=(
        "CoilArrayFieldNode MPEM actuation matrix A_b(p) and gradient 5-vector "
        "G5(p,i) vs the ETH mag_manip 3.0.0 reference library, on the real "
        "calibrated Navion_2 (3-coil) and CardioMag (8-coil) models at 50 "
        "in-workspace positions each. Pins MIME's independent MPEM "
        "reimplementation against upstream."
    ),
    node_type="CoilArrayFieldNode",
    benchmark_type=BenchmarkType.CROSS_CODE,
    acceptance_criteria=(
        "Peak-normalised error, DIMENSIONLESS, defined as "
        "max_elem|mine - ref| / max_elem|ref| — the largest absolute deviation "
        "as a fraction of the largest element of the reference. NOT an "
        "element-wise relative error, which is undefined for the near-zero "
        "elements both A_b and G5 legitimately contain. Applied to BOTH the "
        "field actuation matrix A_b(p) [T/A] and the gradient 5-vector G5(p,i) "
        "[T/m], over 50 positions per system, for all three shipped systems "
        "(Navion_2, CardioMag, OctoMag). The bound is CONDITIONING-DERIVED, "
        "10 * eps_f64 / field_cancellation_ratio, not a flat constant: two "
        "independent implementations summing near-cancelling terms in different "
        "orders cannot agree better than the conditioning permits. Resulting "
        "bounds ~9e-15 (CardioMag), ~3e-13 (Navion_2), ~3e-11 (OctoMag)."
    ),
    references=("Petruska2017", "Charreyron2021"),
)
def test_ver140_mpem_vs_mag_manip_oracle():
    """Both the FIELD and the GRADIENT are pinned against upstream.

    The gradient is the load-bearing one: force is ``F = (m.grad)B``, so ``G5``
    accuracy — not field accuracy — is what a gradient-pull simulation rests on.
    It is checked here on equal terms with the field.

    The tolerance is derived per-system from the fit's conditioning. See
    ``_CROSS_CODE_SLACK`` above for the measured predicted-vs-actual table; the
    short version is that OctoMag's floor is ~3.2e-12 through no fault of either
    implementation, while CardioMag's is ~9e-16, and a single flat threshold
    cannot serve both honestly.
    """
    oracle = _oracle()

    for prefix, cal_name in _ORACLE_SYSTEMS.items():
        cal = _cal(cal_name)
        ps = oracle[f"{prefix}_ps"]
        curs = oracle[f"{prefix}_curs"]
        A_ref = oracle[f"{prefix}_A"]
        G_ref = oracle[f"{prefix}_G"]

        assert cal.n_coils == A_ref.shape[2], (
            f"{cal_name}: loaded {cal.n_coils} coils but oracle A is "
            f"{A_ref.shape}"
        )

        cancel = cal.conditioning()["field_cancellation_ratio"]
        tol = _CROSS_CODE_SLACK * _EPS_F64 / cancel

        # FIELD: actuation matrix A_b(p) [T/A]
        A_mine = np.stack([np.asarray(field_actuation_matrix(p, cal)) for p in ps])
        a_scale = max(np.abs(A_ref).max(), 1e-30)          # peak |A_ref|, T/A
        a_abs = np.abs(A_mine - A_ref).max()               # T/A
        a_err = a_abs / a_scale                            # dimensionless
        assert a_err < tol, (
            f"{cal_name}: actuation matrix disagrees with mag_manip beyond the "
            f"arithmetic floor. peak-normalised {a_err:.3e} > tol {tol:.3e} "
            f"(max abs {a_abs:.3e} T/A vs peak {a_scale:.3e} T/A; "
            f"field_cancellation_ratio {cancel:.3e})"
        )

        # GRADIENT: 5-vector G5(p, i) [T/m] — the quantity force depends on
        G_mine = np.stack([
            gradient5(jnp.asarray(_grad_b(p, c, cal)))
            for p, c in zip(ps, curs)
        ])
        g_scale = max(np.abs(G_ref).max(), 1e-30)          # peak |G_ref|, T/m
        g_abs = np.abs(G_mine - G_ref).max()               # T/m
        g_err = g_abs / g_scale                            # dimensionless
        assert g_err < tol, (
            f"{cal_name}: gradient5 disagrees with mag_manip beyond the "
            f"arithmetic floor. peak-normalised {g_err:.3e} > tol {tol:.3e} "
            f"(max abs {g_abs:.3e} T/m vs peak {g_scale:.3e} T/m; "
            f"field_cancellation_ratio {cancel:.3e})"
        )

        # Guard the guard: a conditioning-derived tolerance must not silently
        # become vacuous. Even the worst-conditioned system must agree far
        # better than its own physical field magnitude.
        assert tol < 1e-9, (
            f"{cal_name}: derived tolerance {tol:.3e} is too loose to be a test"
        )


# ---------------------------------------------------------------------------
# MIME-VER-141 — order-1 reduces to the point dipole
# ---------------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-141",
    description=(
        "CoilArrayFieldNode order-1 MPEM source reduces exactly to the "
        "closed-form point-dipole field, cross-checked against MIME's "
        "independently-validated permanent_magnet._b_point_dipole "
        "(MIME-NODE-101). All shipped ETH calibrations are order 1, so this "
        "is the path they exercise."
    ),
    node_type="CoilArrayFieldNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "|b_mpem - b_dipole| / |b_dipole| < 1e-12 at 200 random positions "
        "around a single order-1 source, and all bundled calibrations report "
        "max_order == 1."
    ),
    references=("Petruska2017", "Jackson1998"),
)
def test_ver141_order1_reduces_to_point_dipole():
    rng = np.random.default_rng(20260715)

    src_pos = np.array([0.02, -0.01, 0.03])
    src_dir = np.array([0.3554832487, 0.8171745943, -0.4537150454])
    src_dir = src_dir / np.linalg.norm(src_dir)
    b1 = -7.771057508e-4  # a real Navion B_Coeff value [T*m^3/A]

    for _ in range(200):
        # Sample well away from the source: MPEM is an exterior expansion.
        p = src_pos + rng.normal(size=3) * 0.05
        while np.linalg.norm(p - src_pos) < 0.02:
            p = src_pos + rng.normal(size=3) * 0.05

        b_mpem = np.asarray(_source_field(
            jnp.asarray(p), jnp.asarray(src_pos), jnp.asarray(src_dir),
            jnp.asarray([b1]),
        ))

        # Closed form: b = B1/r^3 * (3(d.rhat)rhat - d)
        r = p - src_pos
        rn = np.linalg.norm(r)
        rh = r / rn
        b_closed = b1 / rn ** 3 * (3.0 * np.dot(src_dir, rh) * rh - src_dir)

        rel = np.linalg.norm(b_mpem - b_closed) / np.linalg.norm(b_closed)
        assert rel < 1e-12, f"closed form mismatch at p={p}: rel={rel:.3e}"

        # And against MIME's own dipole node helper. _b_point_dipole takes a
        # moment m [A*m^2] with the mu_0/4pi factor applied internally, so the
        # equivalent moment is m = B1 * 4pi/mu_0 along the source direction.
        m_equiv = b1 * 4.0 * np.pi / MU_0 * src_dir
        b_mime = np.asarray(_b_point_dipole(
            jnp.asarray(p), jnp.asarray(src_pos), jnp.asarray(m_equiv),
        ))
        rel_mime = np.linalg.norm(b_mpem - b_mime) / np.linalg.norm(b_mime)
        assert rel_mime < 1e-12, (
            f"disagrees with permanent_magnet._b_point_dipole at p={p}: "
            f"rel={rel_mime:.3e}"
        )

    # The premise: every shipped calibration really is order 1.
    for name in BUNDLED_CALIBRATIONS:
        cal = _cal(name)
        assert cal.max_order == 1, (
            f"{name} has max_order={cal.max_order}; MIME-VER-141's premise "
            f"(all shipped ETH calibrations are order 1) no longer holds"
        )


# ---------------------------------------------------------------------------
# MIME-VER-142 — Maxwell consistency (structural)
# ---------------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-142",
    description=(
        "CoilArrayFieldNode field is curl-free and divergence-free by "
        "construction (B = -grad(psi), psi harmonic), so grad B is symmetric "
        "and trace-free to machine precision without any constraint being "
        "imposed. Checked on all three bundled calibrations (Navion_2, "
        "OctoMag, CardioMag) at random in-workspace positions with random "
        "currents."
    ),
    node_type="CoilArrayFieldNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "|div B| < 1e-9 T/m, |curl B| < 1e-9 T/m and |grad B - grad B^T| < "
        "1e-9 T/m at 30 random in-workspace positions per system, for all "
        "three bundled calibrations."
    ),
    references=("Petruska2017",),
)
def test_ver142_maxwell_structural():
    rng = np.random.default_rng(4242)

    for name in BUNDLED_CALIBRATIONS:
        cal = _cal(name)
        lo, hi = cal.workspace[:, 0], cal.workspace[:, 1]

        for _ in range(30):
            p = lo + rng.random(3) * (hi - lo)
            currents = rng.normal(size=cal.n_coils) * 5.0
            J = _grad_b(p, currents, cal)

            div = abs(np.trace(J))
            curl = np.array([
                J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1],
            ])
            sym = np.abs(J - J.T).max()

            assert div < 1e-9, f"{name}: div B = {div:.3e} T/m at p={p}"
            assert np.abs(curl).max() < 1e-9, (
                f"{name}: curl B = {curl} at p={p}"
            )
            assert sym < 1e-9, f"{name}: grad B asymmetry {sym:.3e} at p={p}"


# ---------------------------------------------------------------------------
# MIME-VER-143 — Earnshaw: a planar array can only pull inward
# ---------------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-143",
    description=(
        "Earnshaw's theorem, in its rigorous pointwise form. For a curl-free, "
        "divergence-free field every Cartesian component is harmonic, which "
        "gives the exact identity grad^2(|B|^2) = 2*||grad B||_F^2 >= 0: |B|^2 "
        "is SUBHARMONIC and therefore attains no interior maximum. This is the "
        "mechanism behind Earnshaw's theorem, and it is why a field-aligned "
        "(paramagnetic) moment, which seeks field maxima, can never be stably "
        "trapped by any current-free eMNS field. Verified on all three bundled "
        "calibrations at random in-workspace positions."
    ),
    node_type="CoilArrayFieldNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "|grad^2(|B|^2) - 2*||grad B||_F^2| / (2*||grad B||_F^2) < 1e-6 at 20 "
        "random in-workspace positions per system for all three bundled "
        "calibrations, and grad^2(|B|^2) >= 0 at every sample."
    ),
    references=("Gervasoni2024", "Boehler2023", "Jackson1998"),
)
def test_ver143_earnshaw_field_magnitude_is_subharmonic():
    rng = np.random.default_rng(1729)

    for name in BUNDLED_CALIBRATIONS:
        cal = _cal(name)
        pos = jnp.asarray(cal.source_position)
        d = jnp.asarray(cal.source_direction)
        b = jnp.asarray(cal.b_coeff)
        zero = jnp.zeros(3)
        lo, hi = cal.workspace[:, 0], cal.workspace[:, 1]

        for _ in range(20):
            p = jnp.asarray(lo + rng.random(3) * (hi - lo))
            currents = jnp.asarray(rng.normal(size=cal.n_coils) * 10.0)

            def b2(t):  # |B(t)|^2
                B = _b_total_world(t, currents, pos, d, b, zero)
                return jnp.dot(B, B)

            lap = float(jnp.trace(jax.hessian(b2)(p)))
            J = np.asarray(jax.jacrev(
                lambda t: _b_total_world(t, currents, pos, d, b, zero))(p))
            rhs = 2.0 * float(np.sum(J * J))  # 2*||grad B||_F^2

            assert lap >= 0.0, (
                f"{name}: |B|^2 is not subharmonic at p={np.asarray(p)}: "
                f"laplacian = {lap:.3e} < 0. A field maximum (stable trap) "
                f"would be possible — this contradicts Maxwell."
            )
            rel = abs(lap - rhs) / max(abs(rhs), 1e-30)
            assert rel < 1e-6, (
                f"{name}: grad^2(|B|^2) = {lap:.6e} but 2||grad B||_F^2 = "
                f"{rhs:.6e} at p={np.asarray(p)} (rel {rel:.3e}). The identity "
                f"holds only if each field component is harmonic."
            )


def test_planar_array_inward_pull_is_a_tendency_not_an_invariant():
    """Characterise (do not assert) the single-Navion inward-pull bias.

    A field-aligned moment feels ``F = |m| grad|B|``. A single-sided planar
    array pulls it toward the coil plane (+y) in the overwhelming majority of
    the workspace — the practical fact that makes a *double* Navion necessary
    for bidirectional gradient force (Landers 2025).

    It is **not** an invariant, and this test exists to pin the honest number
    against future drift. Earnshaw forbids a field *maximum*; it says nothing
    about the sign of ``grad|B|`` in any fixed direction. Empirically the
    outward direction does occur, at ordinary field strengths, concentrated
    near the coil-facing workspace boundary where the fit is least constrained.

    See the corresponding ``hazard_hints`` entry on ``CoilArrayFieldNode``.
    """
    cal = _cal("navion_2")
    rng = np.random.default_rng(1729)

    # Premise, read off the calibration rather than hard-coded: coils at +y.
    src_y = cal.source_position[..., 1]
    real = np.zeros_like(src_y, dtype=bool)
    for k, n_s in enumerate(cal.sources_per_coil):
        real[k, :n_s] = True
    assert src_y[real].min() > 0.2, (
        f"Navion sources expected at y ~ 0.25-0.37 m (the coil plane); got "
        f"min y = {src_y[real].min():.3f}"
    )

    lo, hi = cal.workspace[:, 0], cal.workspace[:, 1]
    n, n_out, out_y = 0, 0, []

    for _ in range(2000):
        p = lo + rng.random(3) * (hi - lo)
        currents = rng.normal(size=cal.n_coils) * 10.0
        B = np.asarray(field_actuation_matrix(p, cal)) @ currents
        nb = np.linalg.norm(B)
        if nb < 1e-9:
            continue
        F = _grad_b(p, currents, cal) @ (B / nb)  # = grad|B|, per unit moment
        n += 1
        if F[1] <= 0.0:
            n_out += 1
            out_y.append(p[1])

    frac_out = n_out / n
    assert frac_out < 0.02, (
        f"Inward pull should dominate: got {100*frac_out:.2f}% outward over "
        f"{n} samples (expected ~0.2%). The Navion calibration or the field "
        f"model has changed materially."
    )
    if out_y:
        # The exceptions are a boundary effect, not scattered through the bulk.
        y_max = cal.workspace[1, 1]
        assert min(out_y) > y_max - 0.03, (
            f"Outward-force samples were expected only near the coil-facing "
            f"boundary (y > {y_max - 0.03:.3f}); found one at y = "
            f"{min(out_y):.3f}. The exception mechanism has changed."
        )


# ---------------------------------------------------------------------------
# Standard node contract: traceability, saturation, graph coupling
# ---------------------------------------------------------------------------

def test_jit_vmap_grad_traceable():
    cal = _cal("navion_2")
    node = CoilArrayFieldNode("coils", 1e-3, calibration=cal)
    state = node.initial_state()
    p = jnp.array([0.0, 0.0, 0.0])
    i = jnp.array([1.0, -2.0, 3.0])

    # jit
    out = jax.jit(lambda s, bi: node.update(s, bi, 1e-3))(
        state, {"coil_currents": i, "target_position_world": p})
    assert out["field_vector"].shape == (3,)
    assert out["field_gradient"].shape == (3, 3)
    assert np.all(np.isfinite(np.asarray(out["field_vector"])))
    assert np.all(np.isfinite(np.asarray(out["field_gradient"])))

    # vmap over target positions
    targets = jnp.stack([p, p + 0.01, p - 0.01])
    fields = jax.vmap(
        lambda t: node.update(state, {"coil_currents": i,
                                      "target_position_world": t}, 1e-3)["field_vector"]
    )(targets)
    assert fields.shape == (3, 3)
    assert np.all(np.isfinite(np.asarray(fields)))

    # grad w.r.t. currents (differentiable actuation — needed for control)
    def obj(currents):
        return jnp.sum(node.update(
            state, {"coil_currents": currents,
                    "target_position_world": p}, 1e-3)["field_vector"] ** 2)

    g = jax.grad(obj)(i)
    assert g.shape == (3,)
    assert np.all(np.isfinite(np.asarray(g)))


def test_zero_current_gives_zero_field():
    """No Offset block is applied: zero current must give exactly zero field."""
    for name in BUNDLED_CALIBRATIONS:
        cal = _cal(name)
        node = CoilArrayFieldNode("coils", 1e-3, calibration=cal)
        out = node.update(node.initial_state(),
                          {"coil_currents": jnp.zeros(cal.n_coils),
                           "target_position_world": jnp.zeros(3)}, 1e-3)
        assert np.abs(np.asarray(out["field_vector"])).max() == 0.0, name
        assert np.abs(np.asarray(out["field_gradient"])).max() == 0.0, name


def test_background_field_superposes():
    cal = _cal("navion_2")
    bg = (1.0e-5, -2.0e-5, 3.0e-5)
    plain = CoilArrayFieldNode("a", 1e-3, calibration=cal)
    with_bg = CoilArrayFieldNode("b", 1e-3, calibration=cal,
                                 background_field_world_t=bg)
    bi = {"coil_currents": jnp.array([1.0, 2.0, -1.0]),
          "target_position_world": jnp.zeros(3)}
    b0 = np.asarray(plain.update(plain.initial_state(), bi, 1e-3)["field_vector"])
    b1 = np.asarray(with_bg.update(with_bg.initial_state(), bi, 1e-3)["field_vector"])
    assert np.allclose(b1 - b0, np.asarray(bg), rtol=0, atol=1e-15)
    # A uniform background must not perturb the gradient.
    g0 = np.asarray(plain.update(plain.initial_state(), bi, 1e-3)["field_gradient"])
    g1 = np.asarray(with_bg.update(with_bg.initial_state(), bi, 1e-3)["field_gradient"])
    assert np.abs(g1 - g0).max() < 1e-18


def test_saturation_matches_linear_at_low_current():
    """h'(0) = 1, so S-MPEM reduces to the linear model as i -> 0."""
    cal = _cal("cardiomag")
    beta = tuple([0.05] * cal.n_coils)
    lin = CoilArrayFieldNode("lin", 1e-3, calibration=cal)
    sat = CoilArrayFieldNode("sat", 1e-3, calibration=cal, saturation_beta=beta)

    tiny = jnp.full((cal.n_coils,), 1e-4)
    bi = {"coil_currents": tiny, "target_position_world": jnp.zeros(3)}
    bl = np.asarray(lin.update(lin.initial_state(), bi, 1e-3)["field_vector"])
    bs = np.asarray(sat.update(sat.initial_state(), bi, 1e-3)["field_vector"])
    assert np.allclose(bs, bl, rtol=1e-8)

    # At high current the saturated model must fall strictly below linear.
    big = jnp.full((cal.n_coils,), 30.0)
    bi_big = {"coil_currents": big, "target_position_world": jnp.zeros(3)}
    bl_big = np.linalg.norm(np.asarray(
        lin.update(lin.initial_state(), bi_big, 1e-3)["field_vector"]))
    bs_big = np.linalg.norm(np.asarray(
        sat.update(sat.initial_state(), bi_big, 1e-3)["field_vector"]))
    assert bs_big < bl_big

    # h_erf sanity: h(0) = 0, h is odd, h'(0) = 1
    b = jnp.array([0.05])
    assert abs(float(h_erf(jnp.array([0.0]), b)[0])) < 1e-18
    assert np.isclose(float(h_erf(jnp.array([-1.0]), b)[0]),
                      -float(h_erf(jnp.array([1.0]), b)[0]), rtol=1e-12)
    slope = float(jax.grad(lambda x: h_erf(jnp.array([x]), b)[0])(0.0))
    assert np.isclose(slope, 1.0, rtol=1e-9)


def test_metadata_consistency_all_systems():
    for name in BUNDLED_CALIBRATIONS:
        cal = _cal(name)
        node = CoilArrayFieldNode("coils", 1e-3, calibration=cal,
                                  current_limit_a=35.0)
        assert node.validate_mime_consistency() == []
        assert node.commandable_fields() == ["coil_currents"]
        assert node.within_current_limits(np.zeros(cal.n_coils))
        assert not node.within_current_limits(
            np.full(cal.n_coils, 100.0))


def test_calibration_rejects_bad_input():
    cal = _cal("navion_2")
    with pytest.raises(TypeError, match="MpemCalibration"):
        CoilArrayFieldNode("x", 1e-3, calibration="navion_2")
    with pytest.raises(ValueError, match="saturation_beta"):
        CoilArrayFieldNode("x", 1e-3, calibration=cal, saturation_beta=(0.1, 0.2))
    with pytest.raises(ValueError, match="positive"):
        CoilArrayFieldNode("x", 1e-3, calibration=cal,
                           saturation_beta=(0.1, 0.2, -0.3))
    with pytest.raises(FileNotFoundError):
        load_mpem_calibration("no_such_system")


def test_conditioning_predicts_measured_float32_error():
    """Pin the conditioning metric AND its predictive power.

    ``ε_f32 / field_cancellation_ratio`` is claimed to predict the measured
    single-precision error. The measured values (see the module comment in
    coil_array_field.py) are:

        model      field_cancel  predicted  meas. FIELD  meas. GRAD
        navion_2      8.0e-3      1.5e-5      1.1e-5      1.1e-5
        cardiomag     2.5e-1      4.8e-7      3.6e-7      5.0e-7
        octomag       6.9e-5      1.7e-3      5.2e-4      1.2e-3

    This pins the ratios and the resulting classification. If a future change
    makes the prediction stop tracking measurement, the premise of the guard is
    gone and this test should fail rather than the guard quietly misfiring.
    """
    expected = {
        # name        needs_x64,  field_cancellation_ratio bounds
        "navion_2":  (False, 3e-3, 3e-2),
        "cardiomag": (False, 1e-1, 5e-1),
        "octomag":   (True,  1e-5, 3e-4),
    }
    for name, (need64, lo, hi) in expected.items():
        cond = _cal(name).conditioning()
        ratio = cond["field_cancellation_ratio"]
        assert lo <= ratio <= hi, (
            f"{name}: field_cancellation_ratio={ratio:.4g} outside the pinned "
            f"range [{lo:.1g}, {hi:.1g}] — the calibration or the metric changed"
        )
        assert cond["needs_float64"] is need64, (
            f"{name}: needs_float64={cond['needs_float64']}, expected {need64} "
            f"(predicted f32 err {cond['predicted_float32_rel_err']:.3g}, "
            f"budget {1e-4:.0e})"
        )

    # Ordering: octomag must cancel HARDER than both well-conditioned models.
    # This is the corrected result. An earlier parametric moment-space metric
    # (|sum m_j| / sum |m_j|) got this BACKWARDS — it ranked navion_2 as the
    # worse canceller — because it ignored the 1/r^3 weighting and so counted a
    # source 245 km away the same as one 3 cm away. Field space is what matters.
    ratios = {n: _cal(n).conditioning()["field_cancellation_ratio"]
              for n in BUNDLED_CALIBRATIONS}
    assert ratios["octomag"] < ratios["navion_2"] < ratios["cardiomag"], (
        f"expected octomag to cancel hardest and cardiomag least; got {ratios}. "
        f"This ordering is the whole basis for keying the guard on field-space "
        f"cancellation."
    )


def test_ill_conditioned_calibration_raises_in_float32():
    """The OctoMag fit must REFUSE to construct without x64.

    Silent precision loss is the failure mode being guarded. A warning is too
    easy to lose in a long run, so this raises. Construction is the right place:
    it is the first point at which both the calibration and the active dtype are
    known, and it fires once rather than every step.
    """
    prev = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)

        with pytest.raises(RuntimeError, match="too ill-conditioned for float32"):
            CoilArrayFieldNode(
                "coils", 1e-3, calibration=load_mpem_calibration("octomag"))

        # The escape hatch must work, for callers who have established the
        # error is acceptable.
        node = CoilArrayFieldNode(
            "coils", 1e-3, calibration=load_mpem_calibration("octomag"),
            allow_float32=True)
        assert node.n_coils == 8

        # Well-conditioned fits must remain usable in float32 — forcing x64 on
        # everyone would be a real cost for no benefit.
        for name in ("navion_2", "cardiomag"):
            CoilArrayFieldNode("c", 1e-3, calibration=load_mpem_calibration(name))
    finally:
        jax.config.update("jax_enable_x64", prev)

    # Under x64, even the ill-conditioned fit constructs cleanly.
    CoilArrayFieldNode("coils", 1e-3, calibration=_cal("octomag"))


def test_field_sign_convention():
    """Pin the field_sign convention and, crucially, what it does and does not affect.

    ETH's measured Navion data disagrees with ETH's own model by a global sign
    (r = -0.9942 raw, +0.9942 flipped). Which side is flipped is unresolved: a
    sign-symmetric +/-35 A excitation cannot distinguish calibration polarity
    from the rig's convention, and ETH's validation metric is exactly invariant
    under B -> -B, so neither side would have caught it.

    The default is +1 (mag_manip parity — the convention MIME-VER-140 pins to
    4e-14, and the one ETH uses to closed-loop control the hardware daily).

    The load-bearing physics, asserted below: a global field flip does NOT
    change the gradient-pull force on a FIELD-ALIGNED body, because m ∝ B makes
    F = grad(m.B) ∝ grad|B|^2, which is quadratic and hence sign-invariant. It
    DOES flip the torque on a FIXED moment. So the anchor experiment (a
    superparamagnetic capsule) is immune, and torque-mode work is not.
    """
    cal = _cal("navion_2")
    bi = {"coil_currents": jnp.array([5.0, -3.0, 2.0]),
          "target_position_world": jnp.zeros(3)}

    pos_node = CoilArrayFieldNode("a", 1e-3, calibration=cal)
    neg_node = CoilArrayFieldNode("b", 1e-3, calibration=cal, field_sign=-1.0)
    assert pos_node.params["field_sign"] == 1.0, "default must be mag_manip parity"

    out_p = pos_node.update(pos_node.initial_state(), bi, 1e-3)
    out_n = neg_node.update(neg_node.initial_state(), bi, 1e-3)
    Bp, Gp = np.asarray(out_p["field_vector"]), np.asarray(out_p["field_gradient"])
    Bn, Gn = np.asarray(out_n["field_vector"]), np.asarray(out_n["field_gradient"])

    # Exact negation of both field and gradient — not an approximation.
    assert np.array_equal(Bn, -Bp), "field_sign=-1 must exactly negate B"
    assert np.array_equal(Gn, -Gp), "field_sign=-1 must exactly negate grad B"

    # THE INVARIANCE: field-aligned moment => F = |m| grad|B|, sign-invariant.
    # (grad B @ b_hat maps to (-grad B) @ (-b_hat) = grad B @ b_hat.)
    Fp = Gp @ (Bp / np.linalg.norm(Bp))
    Fn = Gn @ (Bn / np.linalg.norm(Bn))
    assert np.array_equal(Fp, Fn), (
        "the gradient-pull force on a FIELD-ALIGNED body must be invariant "
        "under a global field flip; if this fails, the rationale for defaulting "
        "field_sign=+1 (that the anchor experiment is immune) is void"
    )

    # THE ASYMMETRY: a FIXED moment's torque does flip.
    m_fixed = np.array([1e-4, 0.0, 0.0])          # A.m^2, body-fixed
    assert np.allclose(np.cross(m_fixed, Bn), -np.cross(m_fixed, Bp)), (
        "torque on a fixed moment must flip with field_sign"
    )
    # ...and so does its force, before it aligns.
    assert np.allclose(Gn @ m_fixed, -(Gp @ m_fixed))

    # It is a convention, not a gain.
    for bad in (2.0, 0.0, -0.5):
        with pytest.raises(ValueError, match="field_sign must be"):
            CoilArrayFieldNode("x", 1e-3, calibration=cal, field_sign=bad)

    # The background field is physical and must NOT be flipped by the convention.
    bg = (1e-5, -2e-5, 3e-5)
    n_bg = CoilArrayFieldNode("c", 1e-3, calibration=cal, field_sign=-1.0,
                              background_field_world_t=bg)
    B_bg = np.asarray(
        n_bg.update(n_bg.initial_state(), bi, 1e-3)["field_vector"])
    assert np.allclose(B_bg - Bn, np.asarray(bg), rtol=0, atol=1e-15), (
        "background_field_world_t is a real physical field and must be added "
        "after field_sign, not scaled by it"
    )


def test_in_workspace_helper():
    cal = _cal("navion_2")
    centre = cal.workspace.mean(axis=1)
    assert cal.in_workspace(centre)
    outside = cal.workspace[:, 1] + 1.0
    assert not cal.in_workspace(outside)
