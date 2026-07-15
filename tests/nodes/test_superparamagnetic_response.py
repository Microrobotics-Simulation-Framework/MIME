"""Unit tests for ``SuperparamagneticResponseNode`` (MIME-NODE-009).

Covers the node's public surface: construction and validation, the curve
classes, the Langevin series/direct crossover, the safe-norm guards at |B| = 0,
volume scaling, the small-body diagnostic, and JAX traceability.

The *physics* claims (Landers reproduction, the force identity, the structural
torque, the two-path consistency) live in
``tests/verification/test_superparamagnetic_response.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.robot.superparamagnetic_response import (
    DemagLimitedLinearCurve,
    InterpolatedCurve,
    LANDERS_B_MEASURED_T,
    LANDERS_CAPSULE_DIAMETER_M,
    LANDERS_CFD_DIAMETER_M,
    LANDERS_CHORD_SLOPE_AM2_PER_T,
    LANDERS_M_AT_30MT_AM2,
    LANDERS_M_SAT_AM2,
    LangevinCurve,
    LinearSaturationCurve,
    MU_0,
    SuperparamagneticResponseNode,
    langevin,
    small_body_field_variation_ratio,
    sphere_equivalent_diameter_m,
    sphere_volume_m3,
)


DT = 1.0e-3


def _node(**kw) -> SuperparamagneticResponseNode:
    kw.setdefault("curve", LangevinCurve.landers_capsule())
    return SuperparamagneticResponseNode("bead", DT, **kw)


# ── Geometry helpers ────────────────────────────────────────────────────────

def test_sphere_volume_matches_the_landers_number():
    # The brief pins V(1.69 mm) = 2.527e-9 m^3.
    assert sphere_volume_m3(LANDERS_CAPSULE_DIAMETER_M) == pytest.approx(
        2.527e-9, rel=1e-3)


def test_sphere_volume_and_diameter_roundtrip():
    for d in (1.4e-3, 1.69e-3, 5e-6):
        assert sphere_equivalent_diameter_m(sphere_volume_m3(d)) == \
            pytest.approx(d, rel=1e-12)


# ── The Langevin function ───────────────────────────────────────────────────

def test_langevin_is_zero_and_finite_at_zero():
    assert float(langevin(jnp.asarray(0.0))) == 0.0
    assert np.isfinite(float(langevin(jnp.asarray(0.0))))


def test_langevin_gradient_at_zero_is_one_third_not_nan():
    """L'(0) = 1/3 exactly. The naive coth(x) - 1/x gives NaN here, and a
    single jnp.where does not save the gradient — hence the double-where."""
    g = float(jax.grad(langevin)(0.0))
    assert np.isfinite(g)
    assert g == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_langevin_no_nan_anywhere_including_the_crossover():
    x = jnp.asarray(np.concatenate([
        np.zeros(1), np.logspace(-12, 3, 400), [0.1, 0.0999999, 0.1000001],
    ]))
    y = langevin(x)
    assert np.all(np.isfinite(np.asarray(y)))
    g = jax.vmap(jax.grad(langevin))(x)
    assert np.all(np.isfinite(np.asarray(g))), "gradient NaN — check double-where"


def test_langevin_is_monotone_and_bounded():
    x = jnp.asarray(np.logspace(-6, 2, 500))
    y = np.asarray(langevin(x))
    assert np.all(np.diff(y) > 0.0)
    assert np.all((y > 0.0) & (y < 1.0))
    assert float(langevin(jnp.asarray(500.0))) == pytest.approx(1.0, abs=1e-2)


def test_langevin_is_odd():
    x = jnp.asarray([0.03, 0.3, 3.0])
    np.testing.assert_allclose(
        np.asarray(langevin(x)), -np.asarray(langevin(-x)), rtol=1e-6)


# ── Curves ──────────────────────────────────────────────────────────────────

def test_langevin_curve_rejects_bad_parameters():
    with pytest.raises(ValueError, match="m_sat_am2 must be positive"):
        LangevinCurve(0.0, 0.03, 1e-9)
    with pytest.raises(ValueError, match="b0_t must be positive"):
        LangevinCurve(1e-4, 0.0, 1e-9)
    with pytest.raises(ValueError, match="reference_volume_m3 must be positive"):
        LangevinCurve(1e-4, 0.03, 0.0)


def test_langevin_from_saturation_and_slope_is_the_closed_form():
    c = LangevinCurve.from_saturation_and_slope(
        LANDERS_M_SAT_AM2, LANDERS_CHORD_SLOPE_AM2_PER_T, 1e-9)
    assert c.b0_t == pytest.approx(
        LANDERS_M_SAT_AM2 / (3.0 * LANDERS_CHORD_SLOPE_AM2_PER_T))
    assert c.b0_t == pytest.approx(31.25e-3, rel=1e-9)
    # The tangent it was fitted to comes back out.
    assert c.initial_susceptibility_am2_per_t() == pytest.approx(
        LANDERS_CHORD_SLOPE_AM2_PER_T, rel=1e-4)


def test_langevin_from_saturation_and_point_hits_the_point():
    c = LangevinCurve.from_saturation_and_point(
        LANDERS_M_SAT_AM2, LANDERS_B_MEASURED_T, LANDERS_M_AT_30MT_AM2, 1e-9)
    # rel=1e-6, not tighter: this session is float32, so evaluating the curve
    # costs ~1e-7 relative. The FIT itself is float64 — pinned below.
    assert float(c.moment_am2(LANDERS_B_MEASURED_T)) == pytest.approx(
        LANDERS_M_AT_30MT_AM2, rel=1e-6)


def test_langevin_fit_is_float64_regardless_of_the_ambient_jax_dtype():
    """The fitted b0 is host float64 (see _langevin_np). It must not depend on
    whether the caller has JAX_ENABLE_X64 set — the same curve object would
    otherwise differ by ~1e-7 between sessions. This test runs in the default
    float32 session and still demands the float64 answer."""
    c = LangevinCurve.from_saturation_and_point(
        LANDERS_M_SAT_AM2, LANDERS_B_MEASURED_T, LANDERS_M_AT_30MT_AM2, 1e-9)
    assert c.b0_t == pytest.approx(0.02925814984272942, rel=1e-12)
    # And the fit residual, evaluated in float64 on the host, is at bisection
    # precision rather than float32 precision.
    from mime.nodes.robot.superparamagnetic_response import _langevin_np
    m = LANDERS_M_SAT_AM2 * _langevin_np(LANDERS_B_MEASURED_T / c.b0_t)
    assert m == pytest.approx(LANDERS_M_AT_30MT_AM2, rel=1e-12)


def test_langevin_from_saturation_and_point_rejects_unreachable_points():
    with pytest.raises(ValueError, match="must lie strictly between"):
        LangevinCurve.from_saturation_and_point(1e-4, 0.03, 2e-4, 1e-9)   # m > m_sat
    with pytest.raises(ValueError, match="must lie strictly between"):
        LangevinCurve.from_saturation_and_point(1e-4, 0.03, 0.0, 1e-9)


def test_linear_saturation_curve_clamps_at_the_knee():
    c = LinearSaturationCurve(LANDERS_CHORD_SLOPE_AM2_PER_T,
                              LANDERS_M_SAT_AM2, 1e-9)
    assert c.knee_t == pytest.approx(93.75e-3, rel=1e-9)
    # rel=1e-6: float32 session. Linear below the knee, so this reproduces the
    # chord exactly in exact arithmetic — which is precisely the ~6% disagreement
    # with the Langevin documented in LangevinCurve.landers_capsule.
    assert float(c.moment_am2(0.030)) == pytest.approx(
        LANDERS_M_AT_30MT_AM2, rel=1e-6)
    assert float(c.moment_am2(1.0)) == pytest.approx(LANDERS_M_SAT_AM2)  # clamped


def test_interpolated_curve_reproduces_its_table_and_clamps_beyond_it():
    b = np.array([0.0, 0.01, 0.03, 0.1, 1.0])
    m = np.array([0.0, 24.0, 72.0, 150.0, 225.0]) * 1e-6
    c = InterpolatedCurve(b, m, 1e-9)
    for bi, mi in zip(b, m):
        assert float(c.moment_am2(bi)) == pytest.approx(mi, rel=1e-6)
    assert float(c.moment_am2(0.02)) == pytest.approx(48.0e-6, rel=1e-6)  # midpoint
    assert float(c.moment_am2(5.0)) == pytest.approx(225e-6)              # clamped
    assert c.m_sat_am2 == pytest.approx(225e-6)


def test_interpolated_curve_rejects_bad_tables():
    with pytest.raises(ValueError, match="strictly increasing"):
        InterpolatedCurve([0.0, 0.03, 0.02], [0.0, 1e-5, 2e-5], 1e-9)
    with pytest.raises(ValueError, match="same length"):
        InterpolatedCurve([0.0, 0.03], [0.0], 1e-9)
    with pytest.raises(ValueError, match="at least 2"):
        InterpolatedCurve([0.0], [0.0], 1e-9)
    with pytest.raises(ValueError, match="b_table_t must be non-negative"):
        InterpolatedCurve([-0.03, 0.03], [0.0, 1e-5], 1e-9)


def test_interpolated_curve_rejects_an_off_origin_table():
    """REGRESSION: a table starting above zero field would manufacture
    REMANENCE. jnp.interp clamps BELOW the first knot as well as above it, so
    [5 mT, ...] -> [12 memu, ...] silently reports 12 memu at |B| -> 0 — a
    permanent moment on the one node whose defining premise is that it has none,
    and a jump-discontinuous force across a field null (update() zeroes the
    moment at |B| = 0 exactly). A realistic VSM sweep starts away from zero, so
    this is the shape of table a user would actually bring."""
    with pytest.raises(ValueError, match="must start at exactly 0.0 T"):
        InterpolatedCurve([5e-3, 30e-3, 1.0], [12e-6, 72e-6, 225e-6], 1e-9)


def test_interpolated_curve_rejects_a_remanent_moment_at_zero_field():
    """m(0) = 0 by definition for a superparamagnet. A table anchored at the
    right field but the wrong moment is a ferromagnet or a VSM baseline offset —
    either way, not this node."""
    with pytest.raises(ValueError, match="must start at exactly 0.0 A.m\\^2"):
        InterpolatedCurve([0.0, 30e-3, 1.0], [12e-6, 72e-6, 225e-6], 1e-9)


def test_interpolated_curve_rejects_negative_moments():
    with pytest.raises(ValueError, match="m_table_am2 must be non-negative"):
        InterpolatedCurve([0.0, 30e-3, 1.0], [0.0, -72e-6, 225e-6], 1e-9)


def test_interpolated_curve_accepts_the_origin_anchored_form_the_error_suggests():
    """The rejection above must leave a usable path: prepending (0, 0) is what
    the error message tells the caller to do, so pin that it works and that the
    first segment extrapolates linearly to the origin."""
    b = np.concatenate([[0.0], np.array([5e-3, 30e-3, 1.0])])
    m = np.concatenate([[0.0], np.array([12e-6, 72e-6, 225e-6])])
    c = InterpolatedCurve(b, m, 1e-9)
    assert float(c.moment_am2(jnp.asarray(0.0))) == 0.0
    # Linear on [0, 5 mT] through the origin: half-way up is half the moment.
    assert float(c.moment_am2(2.5e-3)) == pytest.approx(6e-6, rel=1e-6)
    # ...and the tangent at the origin is now well-defined rather than 0.
    assert c.initial_susceptibility_am2_per_t() == pytest.approx(
        12e-6 / 5e-3, rel=1e-4)


def test_demag_limited_curve_is_gated_by_having_no_defaults():
    """The first-principles path cannot be reached by accident: every argument
    is required, so you must consciously state chi, N and a saturation."""
    with pytest.raises(TypeError):
        DemagLimitedLinearCurve()                      # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DemagLimitedLinearCurve(1.98)                  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DemagLimitedLinearCurve(1.98, 1.0 / 3.0)       # type: ignore[call-arg]


def test_demag_limited_curve_rejects_bad_parameters():
    V = sphere_volume_m3(LANDERS_CAPSULE_DIAMETER_M)
    with pytest.raises(ValueError, match="intrinsic_susceptibility must be"):
        DemagLimitedLinearCurve(-1.0, 1 / 3, LANDERS_M_SAT_AM2, V)
    with pytest.raises(ValueError, match=r"demag_factor must lie in \(0, 1\)"):
        DemagLimitedLinearCurve(1.98, 0.0, LANDERS_M_SAT_AM2, V)
    with pytest.raises(ValueError, match=r"demag_factor must lie in \(0, 1\)"):
        DemagLimitedLinearCurve(1.98, 1.0, LANDERS_M_SAT_AM2, V)


def test_demag_limited_curve_clamps_at_the_knee():
    """Mirrors test_linear_saturation_curve_clamps_at_the_knee for the GATED
    first-principles path, whose clamp had no coverage: every other test touches
    this curve at or below 30 mT, far under its 93.75 mT knee, so the clamp
    branch never executed and could be deleted wholesale unnoticed. The knee sits
    just below the node's declared |B| ceiling of 0.1 T, i.e. inside the
    validated envelope."""
    V = sphere_volume_m3(LANDERS_CAPSULE_DIAMETER_M)
    c = DemagLimitedLinearCurve(1.9816, 1.0 / 3.0, LANDERS_M_SAT_AM2, V)
    assert c.knee_t == pytest.approx(0.09375, rel=1e-3)
    # Linear right up to the knee...
    assert float(c.moment_am2(0.030)) == pytest.approx(
        LANDERS_M_AT_30MT_AM2, rel=1e-4)
    assert float(c.moment_am2(c.knee_t)) == pytest.approx(
        LANDERS_M_SAT_AM2, rel=1e-5)
    # ...and clamped hard above it. Unclamped, m(0.15 T) would be 1.6x m_sat and
    # m(1.0 T) would be 10.7x — this is the assertion that has teeth.
    assert float(c.moment_am2(0.15)) == pytest.approx(LANDERS_M_SAT_AM2, rel=1e-6)
    assert float(c.moment_am2(1.0)) == pytest.approx(LANDERS_M_SAT_AM2, rel=1e-6)


def test_demag_ceiling_is_one_over_n():
    """chi_eff -> 1/N as chi -> inf. That limit is what MagneticResponseNode
    hardwires (= 3 for a sphere)."""
    V = sphere_volume_m3(LANDERS_CAPSULE_DIAMETER_M)
    hi = DemagLimitedLinearCurve(1e9, 1.0 / 3.0, LANDERS_M_SAT_AM2, V)
    assert hi.effective_susceptibility == pytest.approx(3.0, rel=1e-6)


def _all_curves():
    """One of every MagnetizationCurve subclass, at the Landers volume."""
    V = sphere_volume_m3(LANDERS_CAPSULE_DIAMETER_M)
    return [
        LangevinCurve.landers_capsule(),
        LinearSaturationCurve(LANDERS_CHORD_SLOPE_AM2_PER_T, LANDERS_M_SAT_AM2, V),
        InterpolatedCurve([0.0, 0.03, 1.0], [0.0, 7.2e-5, 2.25e-4], V),
        DemagLimitedLinearCurve(1.98, 1.0 / 3.0, LANDERS_M_SAT_AM2, V),
    ]


@pytest.mark.parametrize("curve", _all_curves(), ids=lambda c: c.kind)
def test_every_curve_is_zero_at_zero_field(curve):
    """NO REMANENCE — the defining premise of the node, asserted on EVERY curve
    rather than on a hand-picked list. update() takes b_hat = 0 at B = 0 and so
    relies on m(0) = 0 to make that the correct limit instead of a jump."""
    assert float(curve.moment_am2(jnp.asarray(0.0))) == pytest.approx(0.0, abs=1e-30)


@pytest.mark.parametrize("curve", _all_curves(), ids=lambda c: c.kind)
def test_force_is_continuous_across_a_field_null_for_every_curve(curve):
    """The force must not JUMP as |B| passes through zero. update() zeroes the
    moment at |B| = 0 exactly, so any curve with m(0) != 0 would emit a step —
    which is what an off-origin InterpolatedCurve table used to do (6 uN of
    phantom force at |B| = 1e-9 T, ~17% of the real 30 mT operating force)."""
    n = SuperparamagneticResponseNode("b", DT, curve=curve)
    G = jnp.eye(3) * 0.5

    def force_z(b):
        return float(n.update(n.initial_state(),
                              {"field_vector": jnp.array([0.0, 0.0, b]),
                               "field_gradient": G}, DT)["magnetic_force"][2])

    assert force_z(0.0) == pytest.approx(0.0, abs=1e-30)

    # Approaching the null from above the force must decay LINEARLY to zero, not
    # step. In the low-field limit m -> chi*|B|, so |F| -> grad*chi*|B|: bound it
    # against each curve's OWN tangent rather than a fixed number (a fixed
    # threshold would be wrong — F(1e-4 T) ~ 1.2e-7 N here is correct physics,
    # not remanence). With remanence |F| -> grad*m(0) = const instead, which
    # breaks this bound by orders of magnitude as b shrinks.
    chi = n.initial_susceptibility_am2_per_t()
    for b in (1e-4, 1e-6, 1e-8):
        bound = 2.0 * 0.5 * chi * b
        assert abs(force_z(b)) < bound, (
            f"{curve.kind}: |F| = {abs(force_z(b)):.3e} N at |B| = {b} T "
            f"exceeds the linear-response bound {bound:.3e} N — the force does "
            f"not vanish with the field, i.e. the curve has remanence")


# ── Node construction ───────────────────────────────────────────────────────

def test_validate_mime_consistency_is_clean():
    assert _node().validate_mime_consistency() == []


def test_curve_is_required_and_keyword_only():
    with pytest.raises(TypeError):
        SuperparamagneticResponseNode("bead", DT)                  # no curve
    with pytest.raises(TypeError):
        SuperparamagneticResponseNode(
            "bead", DT, LangevinCurve.landers_capsule())           # positional


def test_curve_must_be_a_magnetization_curve():
    with pytest.raises(TypeError, match="curve must be a MagnetizationCurve"):
        SuperparamagneticResponseNode("bead", DT, curve=3.0)       # type: ignore[arg-type]


def test_volume_defaults_to_the_curves_reference_volume():
    n = _node()
    assert n.volume_scale == pytest.approx(1.0)
    assert n.params["volume_m3"] == pytest.approx(
        sphere_volume_m3(LANDERS_CAPSULE_DIAMETER_M))


def test_volume_must_be_positive():
    with pytest.raises(ValueError, match="volume_m3 must be positive"):
        _node(volume_m3=0.0)


def test_params_are_json_serialisable_for_graph_json():
    import json
    n = _node()
    json.dumps({k: v for k, v in n.params.items()
                if k in ("volume_m3", "reference_volume_m3", "volume_scale",
                         "curve_kind", "curve_description")})


def test_node_has_no_orientation_boundary_input():
    """Deliberate: a field-aligned isotropic moment has no body-frame quantity
    to rotate, so the input would be dead. See the class docstring."""
    assert "orientation" not in _node().boundary_input_spec()
    assert set(_node().boundary_input_spec()) == {"field_vector", "field_gradient"}


def test_state_and_flux_field_names_match_the_siblings():
    n = _node()
    assert set(n.initial_state()) == {
        "magnetization", "magnetic_torque", "magnetic_force"}
    fluxes = n.compute_boundary_fluxes(n.initial_state(), {}, DT)
    assert set(fluxes) == {"magnetic_torque", "magnetic_force"}


# ── update() ────────────────────────────────────────────────────────────────

def test_update_at_zero_field_is_zero_not_nan():
    n = _node()
    out = n.update(n.initial_state(),
                   {"field_vector": jnp.zeros(3),
                    "field_gradient": jnp.eye(3) * 0.5}, DT)
    for k, v in out.items():
        assert np.all(np.isfinite(np.asarray(v))), f"{k} is not finite at B=0"
        np.testing.assert_allclose(np.asarray(v), 0.0, atol=1e-30)


def test_update_with_no_boundary_inputs_uses_zero_defaults():
    n = _node()
    out = n.update(n.initial_state(), {}, DT)
    for v in out.values():
        np.testing.assert_allclose(np.asarray(v), 0.0, atol=1e-30)


def test_uniform_field_gives_zero_force():
    """No gradient, no force — the defining property of gradient pulling."""
    n = _node()
    out = n.update(n.initial_state(),
                   {"field_vector": jnp.array([0.0, 0.0, 0.030]),
                    "field_gradient": jnp.zeros((3, 3))}, DT)
    np.testing.assert_allclose(np.asarray(out["magnetic_force"]), 0.0, atol=1e-30)
    # ...but there IS a moment.
    assert float(jnp.linalg.norm(out["magnetization"])) > 0.0


def test_moment_is_parallel_to_the_field_for_arbitrary_directions():
    n = _node()
    rng = np.random.default_rng(7)
    for _ in range(20):
        B = jnp.asarray(rng.normal(size=3) * 0.03)
        out = n.update(n.initial_state(),
                       {"field_vector": B,
                        "field_gradient": jnp.zeros((3, 3))}, DT)
        m = out["magnetization"]
        cross = jnp.cross(m, B)
        scale = float(jnp.linalg.norm(m) * jnp.linalg.norm(B))
        assert float(jnp.linalg.norm(cross)) / scale < 1e-6


def test_torque_is_exactly_zero_for_arbitrary_inputs():
    n = _node()
    rng = np.random.default_rng(11)
    for _ in range(20):
        out = n.update(n.initial_state(),
                       {"field_vector": jnp.asarray(rng.normal(size=3) * 0.05),
                        "field_gradient": jnp.asarray(rng.normal(size=(3, 3)))},
                       DT)
        assert np.all(np.asarray(out["magnetic_torque"]) == 0.0)


def test_moment_magnitude_follows_the_curve():
    n = _node()
    for b in (0.001, 0.01, 0.03, 0.1):
        out = n.update(n.initial_state(),
                       {"field_vector": jnp.array([0.0, 0.0, b]),
                        "field_gradient": jnp.zeros((3, 3))}, DT)
        assert float(jnp.linalg.norm(out["magnetization"])) == pytest.approx(
            float(n.moment_am2(b)), rel=1e-5)


def test_force_uses_the_jacrev_gradient_convention():
    """F_i = m_j dB_i/dx_j, i.e. F = grad_B @ m with grad_B[i][j] = dB_i/dx_j.
    A hand-built asymmetric gradient pins the index order — a transpose bug
    would pass a symmetric-gradient test."""
    n = _node()
    B = jnp.array([0.0, 0.0, 0.030])
    grad_B = jnp.array([[0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0]])          # only dBx/dz is non-zero
    out = n.update(n.initial_state(),
                   {"field_vector": B, "field_gradient": grad_B}, DT)
    m = float(n.moment_am2(0.030))
    # m points along +z, so F_x = m_z * dBx/dz = m, and F_y = F_z = 0.
    np.testing.assert_allclose(np.asarray(out["magnetic_force"]),
                               np.array([m, 0.0, 0.0]), rtol=1e-5)


# ── Volume scaling ──────────────────────────────────────────────────────────

def test_volume_scaling_is_exact_and_linear():
    curve = LangevinCurve.landers_capsule()
    big = SuperparamagneticResponseNode(
        "big", DT, curve=curve, volume_m3=sphere_volume_m3(LANDERS_CAPSULE_DIAMETER_M))
    small = SuperparamagneticResponseNode(
        "small", DT, curve=curve, volume_m3=sphere_volume_m3(LANDERS_CFD_DIAMETER_M))
    ratio = (LANDERS_CFD_DIAMETER_M / LANDERS_CAPSULE_DIAMETER_M) ** 3
    assert small.volume_scale == pytest.approx(ratio, rel=1e-12)
    for b in (0.005, 0.03, 0.2):
        assert float(small.moment_am2(b)) == pytest.approx(
            ratio * float(big.moment_am2(b)), rel=1e-6)


def test_the_landers_size_ambiguity_is_a_43_percent_force_swing():
    """The main text says 1.69 mm; the paper's own CFD says 1.4 mm. Neither is
    blessed — this pins how much it matters."""
    ratio = (LANDERS_CFD_DIAMETER_M / LANDERS_CAPSULE_DIAMETER_M) ** 3
    assert ratio == pytest.approx(0.5685, rel=1e-3)


# ── Small-body diagnostic ───────────────────────────────────────────────────

def test_small_body_ratio_at_the_landers_operating_point():
    """d = 1.69 mm, |B| = 30 mT, |grad B| = 0.5 T/m -> ~2.8 %."""
    B = jnp.array([0.0, 0.0, 0.030])
    grad_B = jnp.array([[0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.5]])          # dBz/dz = 0.5 T/m
    r = small_body_field_variation_ratio(B, grad_B, LANDERS_CAPSULE_DIAMETER_M)
    assert float(r) == pytest.approx(0.5 * 1.69e-3 / 0.030, rel=1e-6)
    assert float(r) == pytest.approx(0.0282, abs=1e-3)


def test_node_small_body_ratio_uses_its_own_diameter():
    n = _node()
    assert n.diameter_m == pytest.approx(LANDERS_CAPSULE_DIAMETER_M, rel=1e-9)
    B = jnp.array([0.0, 0.0, 0.030])
    grad_B = jnp.diag(jnp.array([0.0, 0.0, 0.5]))
    assert float(n.small_body_ratio(B, grad_B)) == pytest.approx(0.0282, abs=1e-3)


def test_node_diameter_follows_volume_m3_not_the_curves_reference_volume():
    """The default node has volume_m3 == reference_volume_m3, so the test above
    CANNOT tell the two apart. Build a node where they differ: diameter_m must
    track THIS body (1.4 mm), not the curve's reference body (1.69 mm).

    This matters beyond naming — diameter_m is the sole input to
    small_body_ratio(), the caller-facing guard on the node's own central
    assumption. Sourcing it from the reference volume would over-report
    delta|B|/|B| by 21% on exactly the CFD arm that volume_m3 exists to express."""
    n = _node(volume_m3=sphere_volume_m3(LANDERS_CFD_DIAMETER_M))
    assert n.diameter_m == pytest.approx(LANDERS_CFD_DIAMETER_M, rel=1e-9)
    assert n.diameter_m != pytest.approx(LANDERS_CAPSULE_DIAMETER_M, rel=1e-3)

    # ...and the diagnostic must follow the SMALLER body.
    B = jnp.array([0.0, 0.0, 0.030])
    grad_B = jnp.diag(jnp.array([0.0, 0.0, 0.5]))
    assert float(n.small_body_ratio(B, grad_B)) == pytest.approx(
        0.5 * LANDERS_CFD_DIAMETER_M / 0.030, rel=1e-5)


def test_node_initial_susceptibility_is_volume_scaled():
    """The node's scaled accessors must be uniform: moment_am2 and m_sat_am2 are
    volume-scaled, so the tangent must be too. Reaching through to
    node.curve.initial_susceptibility_am2_per_t() gives the REFERENCE body."""
    curve = LangevinCurve.landers_capsule()
    n = SuperparamagneticResponseNode(
        "small", DT, curve=curve, volume_m3=sphere_volume_m3(LANDERS_CFD_DIAMETER_M))
    ratio = (LANDERS_CFD_DIAMETER_M / LANDERS_CAPSULE_DIAMETER_M) ** 3
    assert n.initial_susceptibility_am2_per_t() == pytest.approx(
        ratio * curve.initial_susceptibility_am2_per_t(), rel=1e-6)
    # The unscaled curve value is a DIFFERENT number — that is the trap.
    assert n.initial_susceptibility_am2_per_t() != pytest.approx(
        curve.initial_susceptibility_am2_per_t(), rel=1e-3)


def test_small_body_ratio_scales_linearly_in_d_and_gradient_and_inversely_in_b():
    B = jnp.array([0.0, 0.0, 0.030])
    G = jnp.diag(jnp.array([0.0, 0.0, 0.5]))
    base = float(small_body_field_variation_ratio(B, G, 1.69e-3))
    assert float(small_body_field_variation_ratio(B, G, 2 * 1.69e-3)) == \
        pytest.approx(2 * base, rel=1e-6)
    assert float(small_body_field_variation_ratio(B, 2 * G, 1.69e-3)) == \
        pytest.approx(2 * base, rel=1e-6)
    assert float(small_body_field_variation_ratio(2 * B, G, 1.69e-3)) == \
        pytest.approx(base / 2, rel=1e-6)


def test_small_body_ratio_ignores_pure_rotation_of_the_field():
    """The curve is a function of |B|, so a gradient that only ROTATES B does
    not break the small-body assumption. Only grad|B| counts."""
    B = jnp.array([0.0, 0.0, 0.030])
    # dBx/dz != 0 rotates B without changing |B| to first order.
    G = jnp.array([[0.0, 0.0, 0.5],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
    assert float(small_body_field_variation_ratio(B, G, 1.69e-3)) == \
        pytest.approx(0.0, abs=1e-12)


def test_small_body_ratio_is_inf_not_zero_at_a_field_null():
    """The diagnostic must not report its most REASSURING value at the one place
    the model breaks worst. The ratio diverges as |B| -> 0 (0.845 at 1 mT, 84.5
    at 10 uT), so returning 0 at the null would tell a caller
    sweeping a workspace 'transfers cleanly' at exactly the voxel that must be
    flagged — and an eMNS null can be EXACTLY zero by symmetry in a computed
    field. NaN is not acceptable either: it is not a number a comparison can use.
    """
    r = small_body_field_variation_ratio(jnp.zeros(3), jnp.eye(3), 1.69e-3)
    assert np.isinf(float(r))
    assert not np.isnan(float(r))
    assert float(r) > 0.0
    # It must exceed any sane ceiling, which is the whole point of the value.
    assert float(r) > 0.2


def test_small_body_ratio_diverges_approaching_a_field_null():
    """The inf at exactly zero is the LIMIT of the neighbourhood, not a special
    case bolted on: the ratio really does blow up as |B| -> 0 at fixed gradient."""
    G = jnp.diag(jnp.array([0.0, 0.0, 0.5]))
    prev = 0.0
    for b in (1e-3, 1e-5, 1e-7):
        r = float(small_body_field_variation_ratio(
            jnp.array([0.0, 0.0, b]), G, 1.69e-3))
        assert r > prev
        prev = r
    assert float(small_body_field_variation_ratio(
        jnp.array([0.0, 0.0, 1e-3]), G, 1.69e-3)) == pytest.approx(0.845, rel=1e-3)


def test_small_body_ratio_at_the_navion_double_gradient_exceeds_the_rule_of_thumb():
    """The anchor's own pull-against-flow mode (1 T/m double-gradient) puts the
    1.69 mm capsule at 5.6%, ABOVE the 0.05 rule of thumb — twice the 2.8%
    headline figure quoted for 0.5 T/m. The node's validated_regimes ceiling is
    0.07 precisely to admit this; pin the number so the two cannot drift apart."""
    B = jnp.array([0.0, 0.0, 0.030])
    r_single = float(small_body_field_variation_ratio(
        B, jnp.diag(jnp.array([0.0, 0.0, 0.25])), LANDERS_CAPSULE_DIAMETER_M))
    r_double = float(small_body_field_variation_ratio(
        B, jnp.diag(jnp.array([0.0, 0.0, 1.0])), LANDERS_CAPSULE_DIAMETER_M))
    assert r_single == pytest.approx(0.0141, abs=1e-3)
    assert r_double == pytest.approx(0.0563, abs=1e-3)
    assert r_single < 0.05 < r_double < 0.07
    # And the declared 2 mm body at the declared 1 T/m ceiling: 6.7%, still < 0.07.
    r_max = float(small_body_field_variation_ratio(
        B, jnp.diag(jnp.array([0.0, 0.0, 1.0])), 2.0e-3))
    assert r_max == pytest.approx(0.0667, abs=1e-3)
    assert r_max < 0.07, "the declared regime box must be self-consistent"


# ── Traceability ────────────────────────────────────────────────────────────

def test_update_is_jittable():
    n = _node()
    bi = {"field_vector": jnp.array([0.0, 0.0, 0.030]),
          "field_gradient": jnp.eye(3) * 0.5}
    out = jax.jit(lambda s, b: n.update(s, b, DT))(n.initial_state(), bi)
    assert np.all(np.isfinite(np.asarray(out["magnetic_force"])))


def test_update_is_vmappable_over_a_batch_of_fields():
    n = _node()
    B = jnp.asarray(np.random.default_rng(3).normal(size=(16, 3)) * 0.03)
    G = jnp.broadcast_to(jnp.eye(3) * 0.5, (16, 3, 3))
    f = jax.vmap(lambda b, g: n.update(
        n.initial_state(), {"field_vector": b, "field_gradient": g}, DT))
    out = f(B, G)
    assert out["magnetic_force"].shape == (16, 3)
    assert np.all(np.isfinite(np.asarray(out["magnetic_force"])))


def test_force_is_differentiable_wrt_the_field():
    n = _node()
    G = jnp.eye(3) * 0.5

    def fx(B):
        return n.update(n.initial_state(),
                        {"field_vector": B, "field_gradient": G},
                        DT)["magnetic_force"][0]

    g = jax.grad(fx)(jnp.array([0.001, 0.002, 0.030]))
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.linalg.norm(g)) > 0.0


def test_gradient_is_finite_at_zero_field():
    """The norm has a cone point at B = 0; the safe-norm double-where keeps AD
    finite there (returning 0, which is a choice — the true gradient of |B| at
    the origin is undefined)."""
    n = _node()
    G = jnp.eye(3) * 0.5

    def fx(B):
        return n.update(n.initial_state(),
                        {"field_vector": B, "field_gradient": G},
                        DT)["magnetic_force"][0]

    g = jax.grad(fx)(jnp.zeros(3))
    assert np.all(np.isfinite(np.asarray(g))), "NaN gradient at B=0"


def test_all_curves_are_traceable_under_jit():
    V = sphere_volume_m3(LANDERS_CAPSULE_DIAMETER_M)
    for curve in (
        LangevinCurve.landers_capsule(),
        LinearSaturationCurve(LANDERS_CHORD_SLOPE_AM2_PER_T, LANDERS_M_SAT_AM2, V),
        InterpolatedCurve([0.0, 0.03, 1.0], [0.0, 7.2e-5, 2.25e-4], V),
        DemagLimitedLinearCurve(1.98, 1.0 / 3.0, LANDERS_M_SAT_AM2, V),
    ):
        n = SuperparamagneticResponseNode("b", DT, curve=curve)
        out = jax.jit(lambda s, b: n.update(s, b, DT))(
            n.initial_state(),
            {"field_vector": jnp.array([0.0, 0.0, 0.030]),
             "field_gradient": jnp.eye(3) * 0.5})
        assert np.all(np.isfinite(np.asarray(out["magnetic_force"]))), curve.kind
