"""The shape conformance suite (``mime.testing.check_shape``) over every registered shape, the cylinder's normal
within its axis band, and the screw on its focal set.

**The property.** For each registered shape and each of its run-time capabilities (signed distance, outward normal,
nearest point, and the screw's tangent), Hypothesis draws probes at physical scales (a shape of 10 um to 1 cm, turned
at random and moved up to three sizes, a point off, near or on one of its declared sets, in float64 or float32, a
direction and a parameter moved with it: a scalar, or the centre, the axis or the orientation), and autodiff's
derivative along the curve must equal the float64 difference on the point's side within ``SHAPE_TOLERANCE`` of the
derivative's scale plus the rounding floor; on a set, autodiff must be finite, and both one-sided slopes are reported.

**The measured agreement**, by dtype (``MEASURED``: the largest ``|AD - FD| / S`` where the rounding floor is below a
tenth of the tolerance in float64 and below the tolerance in float32, over the stress runs recorded in the pull
request), is each dtype's own: its tolerance lies
thirty times above it or more and its planted wrong tangent ten times above the tolerance or more
(``test_the_tolerances_keep_their_recorded_margins``). Near a set the float32 comparison resolves its floor, not its
tolerance: about 1e-3 of the scale a tenth of the size from the set, 1e-1 a thousandth from it.

**The examples** (fixed, so they hold on every run): a probe near each declared set of each shape, for each moved
parameter, each capability and each dtype, a thousandth of the size from the set in float64 and a tenth in float32,
where each can discriminate (asserted, not assumed). Autodiff equals the difference at each, so a derivative lost in
any parameter fails there; and **the control** (decision, audit finding A3-S.11), a wrong tangent, autodiff's scaled
by ``1 + PLANTED_TANGENT[dtype]`` through a ``custom_jvp``, fails at each.

**The cylinder's in-band derivative** (design 3.19 items 6 and 18.2): within its axis band the cylinder's normal is
zero, and its derivative in the point, through a ``stop_gradient``, is the projection across the axis, recorded as an
artefact (``artefacts/cylinder_in_band_jacobian.json``: float64, a point 1e-16 m off the axis, inside the 4096-ulp
band). The control writes the in-band branch as the constant zero, whose Jacobian is zero, and fails.

**The screw's focal set**, where its seam ends: on it (the review's settings, FL-9's and the most twisted, each at a
crest and a trough) the distance is exact with the normal its derivative; near it, off the crest's ray, the distance
is the section's minimum to rounding; and the declared orders are measured along the ray. The control is the screw
as it was, its Newton step solved with the singular Hessian from an unpolished root, which fails each."""

import functools
import json
import math
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize_scalar

import mime.testing._shape_suite as suite
from mime.core import AXIS_ULPS, CYLINDER, SCREW, Cylinder, Screw, jnp, switch
from mime.testing import (
    PLANTED_TANGENT,
    SHAPE_CASES,
    SHAPE_TOLERANCE,
    ShapeCheck,
    ShapeProbe,
    check_shape,
    shape_probes,
    uncased_shapes,
)

PAIRS = [(shape, name) for shape, case in sorted(SHAPE_CASES.items()) for name in case.capabilities]
#: The largest agreement the stress runs measured where the floor was negligible, by dtype: below a tenth of the
#: tolerance in float64, below the tolerance in float32, whose floor near a set is never smaller (6,400 and 6,762
#: probes; the second run measured 9.5e-11 and 1.3e-6).
MEASURED = {"float64": 1.3e-10, "float32": 1.5e-6}
ARTEFACT = Path(__file__).parent / "artefacts" / "cylinder_in_band_jacobian.json"


def test_every_registered_shape_has_a_case_placing_each_of_its_switches():
    assert uncased_shapes() == []


def test_control_a_shape_with_no_case_or_a_switch_its_case_does_not_place_is_named(monkeypatch):
    held = dict(SHAPE_CASES)
    monkeypatch.delitem(suite.SHAPE_CASES, "sphere")
    monkeypatch.setitem(suite.SHAPE_CASES, "capsule", replace(held["capsule"], sets=held["capsule"].sets[:1]))
    assert uncased_shapes() == ["shape capsule: its switch capsule.end.low has no set in its case",
                                "shape capsule: its switch capsule.end.high has no set in its case",
                                "shape sphere: no case in the shape conformance suite"]


@pytest.mark.parametrize(("shape", "name"), PAIRS)
@settings(deadline=None)  # a draw's first evaluation compiles the shape's checks for its dtype and starts
@given(data=st.data())
def test_autodiff_equals_the_difference_on_the_points_side(shape, name, data):
    found = check_shape(data.draw(shape_probes(SHAPE_CASES[shape])), name)
    if found.one_sided is not None:
        print(f"\n{found.line()}: one-sided slopes {found.one_sided[0]} and {found.one_sided[1]}, autodiff "
              f"{found.autodiff}")
    if not found.discriminating:  # not a pass, not a failure: counted, and its share bounded by its own test below
        event("non-discriminating")
        print(f"\n{found.line()}")
    assert found.problems == (), f"{found.line()}: " + "; ".join(found.problems)


#: The share of difference probes the suite's draws may leave non-discriminating (no step brings the difference's
#: truncation under ``TRUNCATION_SHARE`` of the allowance without it). Measured: 0 of 5,262 over 20 stress runs of
#: the draws below (400 each, the rest on a set); 1 of 750 in the screw's jittered neighbourhood of its focus, the
#: worst place found. The bound is above the measured share by the room a rare draw needs: 5 probes of 272.
NON_DISCRIMINATING_SHARE = 0.02


def test_the_share_of_probes_whose_difference_cannot_discriminate_is_bounded_over_the_suites_draws():
    """Every probe the property draws is checked, and the non-discriminating ones counted (they pass nothing): their
    share of the probes that take a difference stays below ``NON_DISCRIMINATING_SHARE``."""
    checked: list[ShapeCheck] = []

    @settings(deadline=None, max_examples=400, database=None)
    @given(data=st.data())
    def draw(data):
        shape, name = data.draw(st.sampled_from(PAIRS))
        checked.append(check_shape(data.draw(shape_probes(SHAPE_CASES[shape])), name))

    draw()
    differenced = [found for found in checked if found.difference is not None]
    blind = [found for found in differenced if not found.discriminating]
    print(f"\n{len(blind)} of {len(differenced)} difference probes non-discriminating "
          f"(bound {NON_DISCRIMINATING_SHARE:.1%}); {len(checked) - len(differenced)} on a set"
          + "".join(f"\n  {found.line()}" for found in blind))
    assert len(blind) <= NON_DISCRIMINATING_SHARE * len(differenced)


#: Each case's shape as an example places it: a size each shape is used at, turned off every coordinate axis.
TYPICAL = {
    "sphere": {"centre": (0.0, 0.0, 0.0), "radius": 1e-3},
    "cylinder": {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "radius": 3.175e-3},
    "capsule": {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "radius": 5e-4, "height": 2e-3},
    "slab": {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "half_thickness": 5e-4},
    "screw": {"centre": (0.0, 0.0, 0.0), "orientation": (1.0, 0.0, 0.0, 0.0), "radius": 1.56e-3,
                      "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33, "starts": 2},
}
TURN = np.array([0.9, 0.3, -0.2, 0.25]) / np.linalg.norm([0.9, 0.3, -0.2, 0.25])
DIRECTION = np.array([0.48, -0.6, 0.64])
DTYPES = ("float64", "float32")
#: How far an example lies from its set, in the shape's size, by dtype: where autodiff's floor lies below the planted
#: tangent. A thousandth in float64; a tenth in float32, about two hundred of its axis bands and the edge of its near
#: class, where that floor is about 1e-3 of the scale (a thousandth from a set it is about 1e-1).
EXAMPLE_OFFSET = {"float64": 1e-3, "float32": 0.1}
#: A moved parameter's rate in an example (per metre the point moves, over the size for a dimensionless one or a
#: turn), and the two uniform numbers that give a centre's direction or a turn's axis: off every axis.
RATE, TURNING = 1.0, np.array([0.31, 0.77])
#: The draws that place an example on a set; the second places the screw's focus at a crest where the first places it
#: at a trough.
DRAWS = (np.array([0.37, 0.61, 0.53]), np.array([0.62, 0.21, 0.18]))


@functools.cache
def examples(dtype: str) -> tuple[tuple[ShapeProbe, str, str, ShapeCheck], ...]:
    """A probe near each declared set of each shape, for each parameter the case moves and each capability, in
    ``dtype``, ``EXAMPLE_OFFSET[dtype]`` of the size from the set on its positive side, along a direction off every
    axis with the parameter moved at ``RATE``, of the sign along which the float64 difference finds the steeper
    derivative (so that no parameter's part cancels the point's); with the parameter and its unplanted check."""
    found = []
    for shape, case in sorted(SHAPE_CASES.items()):
        own = TYPICAL[shape]
        size = case.size(own)
        rotation = suite._rotation(TURN)  # noqa: SLF001 - the suite's own rotation, as its probes turn
        shift = size * np.array([0.7, -1.3, 2.1])
        world = case.turned(own, TURN, shift)
        offset = EXAMPLE_OFFSET[dtype] * size
        for chosen in case.sets:
            for draw in DRAWS[:2 if chosen.owner == "screw.focus" else 1]:
                point = rotation @ chosen.place(own, offset, draw) + shift
                for name in case.moved:
                    curves = [ShapeProbe(shape, world, point, DIRECTION / np.linalg.norm(DIRECTION),
                                         {name: suite.moved_rate(case, world, name, sign * RATE, TURNING, size)},
                                         chosen.owner, "near", offset, dtype, size) for sign in (1.0, -1.0)]
                    for capability in case.capabilities:
                        checks = [(probe, check_shape(probe, capability)) for probe in curves]
                        probe, check = max(checks, key=lambda held: _steepness(held[1]))
                        found.append((probe, capability, name, check))
    return tuple(found)


def _steepness(check: ShapeCheck) -> float:
    return float(np.max(np.abs(check.difference))) / check.scale if check.difference is not None else 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_autodiff_equals_the_difference_at_an_example_near_each_set_for_each_moved_parameter(dtype):
    """The property at fixed points, so that each parameter's derivative is compared on every run: a derivative lost
    in a centre, an axis, an orientation or any scalar (a ``stop_gradient`` on it) fails here."""
    failed = [f"{check.line()} (moving {name}): " + "; ".join(check.problems)
              for _, _, name, check in examples(dtype) if not check.passed]
    assert failed == [], "\n".join(failed)


#: The examples whose derivative along the curve is zero wherever a difference is taken, so that no scaling of the
#: tangent can show: the slab's normal is its axis on one side of the mid-plane and its opposite on the other, so it
#: moves with neither the point, the half-thickness nor the centre (only the axis turns it).
CONSTANT = {("slab", "normal", "half_thickness"), ("slab", "normal", "centre")}


@pytest.mark.parametrize("dtype", DTYPES)
def test_control_a_planted_wrong_tangent_fails_at_every_example_near_each_set(dtype):
    """Each example can discriminate (the planted tangent above the tolerance and the floor there, so a miss is a
    failure and not a null result), and each fails planted, on every run (nothing is drawn)."""
    caught, missed, constant, margins = [], [], set(), []
    for probe, capability, name, clean in examples(dtype):
        steep = float(np.max(np.abs(clean.autodiff))) / clean.scale
        if steep <= SHAPE_TOLERANCE[dtype]:  # zero, to rounding
            constant.add((probe.shape, capability, name))
            continue
        resolution = (SHAPE_TOLERANCE[dtype] * clean.scale + clean.floor + clean.truncation) \
            / float(np.max(np.abs(clean.autodiff)))
        assert clean.discriminating, clean.line()
        assert PLANTED_TANGENT[dtype] > resolution, \
            f"{clean.line()} (moving {name}): the comparison resolves {resolution:.3g} here, so the planted tangent " \
            f"cannot show"
        margins.append((PLANTED_TANGENT[dtype] / resolution, f"{clean.line()} (moving {name})"))
        planted = check_shape(probe, capability, planted=PLANTED_TANGENT[dtype])
        (caught if planted.problems else missed).append((probe.shape, probe.owner, capability, name))
    smallest = min(margins)
    print(f"\n{dtype}: the planted tangent (x {1 + PLANTED_TANGENT[dtype]}) is caught at {len(caught)} of "
          f"{len(caught) + len(missed)} examples, at least {smallest[0]:.3g} times what the comparison resolves there "
          f"({smallest[1]}); {sorted(constant)} have no derivative to scale")
    assert missed == [], f"missed at {missed}"
    assert constant == CONSTANT
    sets = {(shape, s.owner, capability) for shape, case in SHAPE_CASES.items() for s in case.sets
            for capability in case.capabilities}
    assert {held[:3] for held in caught} == sets


#: The draw that failed CI run 40722059209: the screw's nearest point, 1.25e-4 m from its focus, with the wavenumber
#: moved by a thousand per unit step. The step the policy gave there, 1e-7 m, left the Richardson value 1.09e-4 off
#: (an h^4 error, 2.5e-6 of the scale, against a tolerance of 1e-6); autodiff was right. Printed to the digits that
#: reproduce the run's 2.53e-06.
CAUSTIC_DRAW = ShapeProbe(
    "screw",
    {"centre": (0.0, 0.0, 0.0), "orientation": (0.0, 0.0, 1.0, 0.0), "radius": 0.001, "length": 0.005,
     "wavenumber": 0.0, "modulation": 0.09375, "starts": 4},
    np.array([-0.0025, 0.00188423, 0.00078047]), np.array([0.0, 0.0, -1.0]), {"wavenumber": 1000.0},
    "screw.focus", "off", -0.000125, "float64", 0.001)


def test_the_draw_whose_step_left_the_difference_off_by_its_truncation_passes_with_the_step_halved():
    """An explicit example of the property (CI's draw, fixed, so it holds on every run): the step is halved until the
    truncation is a small part of the allowance, and the difference agrees with autodiff to the tolerance."""
    found = check_shape(CAUSTIC_DRAW, "nearest")
    policy = check_shape(CAUSTIC_DRAW, "nearest", adapt=False)
    assert found.passed, found.line()
    assert found.step < policy.step, (found.line(), policy.line())
    assert found.agreement <= suite.SHAPE_TOLERANCE["float64"], found.line()  # the draw's failure is gone, not allowed
    assert found.truncation <= suite.TRUNCATION_SHARE * (suite.SHAPE_TOLERANCE["float64"] * found.scale + found.floor)
    assert check_shape(CAUSTIC_DRAW, "nearest", planted=PLANTED_TANGENT["float64"]).problems != ()


def test_control_the_old_step_policy_without_the_truncation_allowance_fails_that_draw_on_every_run():
    """The criterion as it was: the difference at the policy's step, taken as exact. It misses autodiff by 2.5 times
    its allowance here, so the control does not hinge on a draw's rounding."""
    old = check_shape(CAUSTIC_DRAW, "nearest", adapt=False, allow_truncation=False)
    assert old.problems != ()
    error = float(np.max(np.abs(old.autodiff - old.difference)))
    assert error >= 2.0 * (SHAPE_TOLERANCE["float64"] * old.scale + old.floor), old.line()


def test_control_the_policys_step_with_the_truncation_allowed_passes_that_draw_without_being_able_to_discriminate():
    """The pass this change replaces: the truncation (2.5 tolerances here) added to the allowance. It is flagged
    non-discriminating, is not a pass, and the failure message names it."""
    found = check_shape(CAUSTIC_DRAW, "nearest", adapt=False)
    assert found.problems == () and not found.discriminating and not found.passed
    assert "NON-DISCRIMINATING" in found.line()


def test_autodiff_is_right_at_that_draw_the_difference_taken_a_hundred_times_finer_agrees_with_it():
    """The diagnosis, as a check on the reference and not on the allowance: at a step of 1e-9 m the truncation is
    below the tolerance by itself (its allowance off), and the rounding floor is still far below it."""
    finer = check_shape(CAUSTIC_DRAW, "nearest", allow_truncation=False, step=1e-9)
    assert finer.problems == (), finer.line()
    assert finer.floor < SHAPE_TOLERANCE["float64"] * finer.scale, finer.line()


#: Where the policy's step left the truncation as large as the derivative (the review of this change, on jittered
#: neighbourhoods of the draw above: the point 1e-6 to 1e-1 of the size off it, the wavenumber moved up to a thousand
#: per unit step, the modulation varied): point, modulation, wavenumber's rate, direction, the truncation at the
#: policy's step in the derivative's scale. Each passed the old allowance-widened criterion with the planted tangent
#: missed (1e-4 at all; 100% wrong too at the first).
JITTERED_DRAWS = (
    ([-0.0024992270021371673, 0.001883941494082644, 0.0007804891344175437], 0.1008364059758812, -1000.0,
     [-0.1089130294145968, 0.24417357729594252, 0.9635959818171916], 0.75),
    ([-0.002499864778224864, 0.0018835630030840079, 0.0007810261156102593], 0.10121848854776402, -1000.0,
     [-0.935326893940821, -0.1649576879645736, 0.3129737411547466], 10.0),
    ([-0.0025006994627756853, 0.0018828108748256075, 0.00078181682377662], 0.09782676792446526, -1000.0,
     [-0.8015903171331984, 0.4872705834605664, 0.34643952137753803], 3.1e-4),
    ([-0.0025007545231113875, 0.0018854209480450525, 0.000780502970824506], 0.10001592221693645, -1000.0,
     [0.7091856009854599, -0.5578490690145884, 0.43111506532995214], 0.22),
)


#: A draw no step makes discriminating (the review's, wavenumber rate -895): the point is 4.6e-9 m from the focus, so
#: the step is capped at a sixteenth of that, 2.9e-10 m, where the rounding of the difference already equals the
#: tolerance, and the truncation (2.4e-6 of the scale, an h^4 error on a curve that bends on the length of the
#: distance) stays above the tolerance. The policy's step with the truncation allowed passed it, and a tangent 1e-4 too
#: large was caught by luck of a small truncation, not by the comparison being able to.
UNRESOLVED_DRAW = (
    [-0.0025000308263572434, 0.0018842208573051997, 0.0007804831308396414], 0.10044079204788545, -894.8630890070912,
    [0.19275961104551234, 0.9800190832157013, 0.049054346215592196], 0.13)


def test_a_draw_no_step_makes_discriminating_is_reported_and_is_not_a_pass():
    """Every run: the clean check has no problem (autodiff and the difference agree within the allowance with its
    truncation), and is flagged non-discriminating: ``passed`` is false and the message names it."""
    found = check_shape(jittered(UNRESOLVED_DRAW), "nearest")
    assert found.problems == (), found.line()
    assert not found.discriminating and not found.passed
    assert "NON-DISCRIMINATING" in found.line()
    assert found.truncation > SHAPE_TOLERANCE["float64"] * found.scale


def jittered(draw) -> ShapeProbe:
    point, modulation, rate, direction, _ = draw
    return replace(CAUSTIC_DRAW, point=np.array(point), direction=np.array(direction), moved={"wavenumber": rate},
                   parameters={**CAUSTIC_DRAW.parameters, "modulation": modulation})


@pytest.mark.parametrize("draw", JITTERED_DRAWS, ids=[f"truncation-{draw[4]}" for draw in JITTERED_DRAWS])
def test_the_planted_tangent_fails_at_each_jittered_draw_and_the_clean_probe_passes(draw):
    """Every run (nothing is drawn): the step is halved until the difference can discriminate, the clean probe passes
    there, and a tangent 1e-4 too large, or 100% too large, fails."""
    probe = jittered(draw)
    clean = check_shape(probe, "nearest")
    assert clean.passed, clean.line()
    assert clean.truncation / clean.scale <= SHAPE_TOLERANCE["float64"], clean.line()
    for planted in (PLANTED_TANGENT["float64"], 1.0):
        assert check_shape(probe, "nearest", planted=planted).problems != (), f"{clean.line()}: planted {planted} missed"


@pytest.mark.parametrize("draw", JITTERED_DRAWS, ids=[f"truncation-{draw[4]}" for draw in JITTERED_DRAWS])
def test_control_the_policys_step_with_the_truncation_allowed_misses_the_planted_tangent_at_each_jittered_draw(draw):
    """The control for the test above: the criterion this change replaces misses the 1e-4 tangent at each (so the
    draws are ones the halving is needed for), and says the probe is non-discriminating."""
    probe = jittered(draw)
    old = check_shape(probe, "nearest", adapt=False, planted=PLANTED_TANGENT["float64"])
    assert old.problems == () and not old.discriminating, old.line()


def test_control_a_step_too_coarse_for_the_difference_to_discriminate_is_reported_and_is_not_a_pass():
    """A fixed step ten times the policy's at the CI draw leaves the truncation larger than the allowance without it:
    the check is flagged, names itself in its line, and ``passed`` is false."""
    coarse = check_shape(CAUSTIC_DRAW, "nearest", step=1e-6)
    assert not coarse.discriminating and not coarse.passed
    assert "NON-DISCRIMINATING" in coarse.line()


def test_the_tolerances_keep_their_recorded_margins():
    """Each dtype's tolerance thirty times above its measured agreement or more, and its planted tangent ten times
    above its tolerance or more."""
    for dtype, measured in MEASURED.items():
        assert SHAPE_TOLERANCE[dtype] >= 30 * measured, dtype
        assert PLANTED_TANGENT[dtype] >= 10 * SHAPE_TOLERANCE[dtype], dtype


# -- the cylinder's normal within its axis band ---------------------------------------------------------------------

RADIUS = 3.175e-3


def in_band_jacobian(cylinder, axis, offset: float = 1e-16) -> np.ndarray:
    """The Jacobian, in the point, of ``cylinder``'s outward normal at a point ``offset`` metres off its axis (along
    it 1 cm from its centre), in float64."""
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    across = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
    across /= np.linalg.norm(across)
    parameters = {"centre": jnp.zeros(3, dtype=jnp.float64), "axis": jnp.asarray(axis), "radius": jnp.asarray(RADIUS)}
    point = jnp.asarray(0.01 * axis + offset * across)
    return np.asarray(jax.jacfwd(lambda at: cylinder.outward_normal(parameters, at))(point))


def in_band_problem(jacobian: np.ndarray, axis) -> str | None:
    """Why ``jacobian`` is not the projection across ``axis``, ``I - a a^T``, to rounding; or ``None``."""
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    expected = np.eye(3) - np.outer(a, a)
    apart = float(np.max(np.abs(jacobian - expected)))
    return None if apart <= 8 * np.finfo(np.float64).eps else \
        f"the in-band Jacobian {jacobian.tolist()} is {apart:.3g} from the projection across the axis"


def test_the_cylinders_normal_within_its_axis_band_has_the_projection_across_the_axis_as_its_jacobian():
    assert 1e-16 < AXIS_ULPS * np.finfo(np.float64).eps * RADIUS, "the point lies inside the band"
    found = in_band_jacobian(CYLINDER, (1.0, 0.0, 0.0))
    recorded = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    assert found.tolist() == recorded["jacobian"] == np.diag([0.0, 1.0, 1.0]).tolist()
    for axis in ((0.0, 0.6, 0.8), (0.48, -0.6, 0.64), (-1.0, 1e-3, 2e-3)):
        assert in_band_problem(in_band_jacobian(CYLINDER, axis), axis) is None, axis


class ConstantInBand(Cylinder):
    """The control: the cylinder's normal with its in-band branch written as the constant zero."""

    def outward_normal(self, parameters, point):
        radial, length, off = self._radial(parameters, point)
        return switch.where(off, radial / switch.domain_guard(off, length, 1.0), jnp.zeros_like(radial),
                            owner="cylinder.axis", resolution=0.0, indicator=length)


def test_control_the_normal_with_its_in_band_branch_the_constant_zero_fails():
    for axis in ((1.0, 0.0, 0.0), (0.48, -0.6, 0.64)):
        found = in_band_jacobian(ConstantInBand(), axis)
        assert np.all(found == 0.0)
        assert in_band_problem(found, axis) is not None


def test_off_the_band_both_normals_have_the_curvature_across_the_axis():
    """Off the band the control's normal and the cylinder's are one expression: the control differs only inside."""
    for axis in ((1.0, 0.0, 0.0), (0.48, -0.6, 0.64)):
        assert np.array_equal(in_band_jacobian(CYLINDER, axis, 1e-4), in_band_jacobian(ConstantInBand(), axis, 1e-4))
        assert math.isclose(float(np.linalg.norm(in_band_jacobian(CYLINDER, axis, 1e-4), 2)), 1e4, rel_tol=1e-6)


# -- the screw's focal set --------------------------------------------------------------------------------------------

#: Where the screw's focal set was found (the review's settings, untwisted: the distance NaN at 2 starts and
#: modulation 0.5, its gradient 2.9e13 off at 0.33 and 2.5e16 off at 3 starts), FL-9's, and the most twisted and
#: modulated screw the kind accepts: starts, modulation, wavenumber.
FOCAL = [(2, 0.5, 0.0), (2, 0.33, 0.0), (3, 0.33, 0.0), (2, 0.33, 2.33), (4, 0.5, 4.0)]
FOCAL_CASES = [(*setting, trough) for setting in FOCAL for trough in (False, True)]


def focal_case(starts, modulation, nu, trough, out=0.0):
    """FL-9's screw with these settings, a point ``out`` metres from its focal set at a crest (or a trough) towards
    the surface (negative: away from it, on the seam), the ray's outward direction there, and the exact distance along
    the ray (the crest's or the trough's own point is the nearest on the set's surface side)."""
    parameters = {**TYPICAL["screw"], "starts": starts, "modulation": modulation, "wavenumber": nu}
    radius = parameters["radius"]
    at = suite.screw_focus(parameters, trough)
    assert at is not None and at <= suite.FARTHEST_FOCUS
    point = suite._screw_focus_place(parameters, 0.0, np.array([0.37, 0.61 if trough else 0.21, 0.0]))  # noqa: SLF001
    ray = np.array([0.0, point[1], point[2]]) / math.hypot(point[1], point[2])
    point = point + (-out if trough else out) * ray
    surface = (1.0 - modulation if trough else 1.0 + modulation) * radius
    exact = (at * radius - out - surface) if trough else -(surface - at * radius - out)
    return parameters, point, ray, exact


#: How far the normal on the focal set may lie from the ray: there the nearest point moves as the cube root of the
#: point's offset, so the point's own rounding, a few units in the last place of its scale, turns it by about
#: eps^(1/3) (measured: up to 3e-5).
HOLDER = 1e-4


def held(parameters):
    return {key: (jnp.asarray(value) if isinstance(value, (float, tuple)) else value)
            for key, value in parameters.items()}


def focal_problems(parameters, point, ray, exact, screw=SCREW) -> list[str]:
    """Why ``screw``'s distance at a point on its focal set (inside the focus's band) is not its exact value with the
    normal as its derivative (the envelope's, at a stationary root) and the ray's direction as the normal, all finite,
    with the normal's derivative that of its nearest point held (zero for an untwisted screw): the step not taken."""
    numbers, at = held(parameters), jnp.asarray(point)
    value = float(screw.signed_distance(numbers, at))
    gradient = np.asarray(jax.grad(lambda p: screw.signed_distance(numbers, p))(at))
    normal = np.asarray(screw.outward_normal(numbers, at))
    found = []
    if not abs(value - exact) <= 64 * np.finfo(np.float64).eps * parameters["radius"]:
        found.append(f"the distance is {value!r} m, and {exact!r} m exactly")
    if not np.max(np.abs(gradient - normal)) <= 1e-12:
        found.append(f"its gradient is {gradient.tolist()}, and its normal {normal.tolist()}")
    if not np.max(np.abs(normal - ray)) <= HOLDER:
        found.append(f"its normal is {normal.tolist()}, and the ray's direction {ray.tolist()}")
    if parameters["wavenumber"] == 0.0:  # untwisted, the normal held with its nearest point does not move
        jacobian = np.asarray(jax.jacfwd(lambda p: screw.outward_normal(numbers, p))(at))
        if not np.all(jacobian == 0.0):
            found.append(f"its normal's Jacobian is {jacobian.tolist()}, and its nearest point held gives zero")
    return found


@pytest.mark.parametrize(("starts", "modulation", "nu", "trough"), FOCAL_CASES)
def test_on_the_screws_focal_set_the_distance_is_exact_with_the_normal_its_derivative(starts, modulation, nu, trough):
    """At a point on the focal set (the closed form's centre of curvature, inside the focus's band), the step is not
    taken: the distance is the crest's (or the trough's) to rounding and its derivative the normal, all finite."""
    assert focal_problems(*focal_case(starts, modulation, nu, trough)) == []


class AsItWas(Screw):
    """The control: the screw as it was before its focus was declared and its root polished: the root find's root,
    and one Newton step from it solved with the Hessian wherever the point is."""

    def _polish(self, residual, theta, tau):
        return theta, tau

    def _newton_step(self, parameters, residual, theta, tau, dtype):
        gradient, hessian = self._derivatives(residual, theta[None], tau[None])[:2]
        return self._solve(hessian, gradient)[:, 0]


@pytest.mark.parametrize(("starts", "modulation", "nu", "trough"), FOCAL_CASES)
def test_control_the_screw_as_it_was_fails_on_its_focal_set(starts, modulation, nu, trough):
    """The step solved with the singular Hessian there gives a NaN, or a distance or derivative off by orders of
    magnitude, at every case, on every run (nothing is drawn)."""
    assert focal_problems(*focal_case(starts, modulation, nu, trough), screw=AsItWas()) != []


def section_distance(point, parameters) -> float:
    """The untwisted screw's distance from ``point``, its nearest point found in the point's own cross-section (the
    surface is the section moved along the axis): a scan of the section's angle, then Brent's method."""
    radius, epsilon, starts = parameters["radius"], parameters["modulation"], parameters["starts"]
    y, z = point[1] / radius, point[2] / radius

    def squared(theta):
        rho = 1.0 + epsilon * np.sin(starts * theta)
        return (rho * np.cos(theta) - y) ** 2 + (rho * np.sin(theta) - z) ** 2

    angles = np.linspace(0.0, 2.0 * math.pi, 200001)
    start = angles[np.argmin(squared(angles))]
    found = minimize_scalar(squared, bracket=(start - 1e-4, start, start + 1e-4), tol=1e-15)
    return math.sqrt(found.fun) * radius


#: Points about the review's focal sets, untwisted, off the ray through the crest: across it, at the set's radius and
#: further out (in the radius). There the root find's fixed lift stalled in the valley with no curvature, and the Newton
#: step from the stalled root overshot.
ACROSS = (1e-10, 1e-8, 1e-7, 1e-6)
OUT = (0.0, 1e-6, 1e-5)


def off_ray_errors(starts, modulation, screw):
    parameters = {**TYPICAL["screw"], "starts": starts, "modulation": modulation, "wavenumber": 0.0}
    radius, numbers = parameters["radius"], held(parameters)
    centre = suite._screw_focus_place(parameters, 0.0, np.array([0.37, 0.21, 0.0]))  # noqa: SLF001
    ray = np.array([0.0, centre[1], centre[2]]) / math.hypot(centre[1], centre[2])
    across = np.cross([1.0, 0.0, 0.0], ray)
    points = [centre + radius * (out * ray + a * across) for out in OUT for a in ACROSS]
    found = np.asarray(jax.vmap(lambda p: screw.signed_distance(numbers, p))(jnp.asarray(np.array(points))))
    return [abs(abs(value) - section_distance(point, parameters)) / radius for value, point in zip(found, points,
                                                                                                    strict=True)]


@pytest.mark.parametrize(("starts", "modulation"), [(2, 0.5), (2, 0.33), (3, 0.33)])
def test_off_its_ray_near_the_focal_set_the_screws_distance_is_the_sections_minimum(starts, modulation):
    """Within a millionth of the radius of the focal set, off the crest's ray: the distance to rounding (it was off by
    up to 0.6 R, enough to put a point inside the body outside it)."""
    errors = off_ray_errors(starts, modulation, SCREW)
    assert max(errors) <= 64 * np.finfo(np.float64).eps, errors


@pytest.mark.parametrize(("starts", "modulation"), [(2, 0.5), (2, 0.33), (3, 0.33)])
def test_control_off_its_ray_the_screw_as_it_was_misses_the_sections_minimum(starts, modulation):
    """The control, on every run: the screw as it was is off by far more than rounding at some of those points."""
    assert max(off_ray_errors(starts, modulation, AsItWas())) > 1e-6


@pytest.mark.parametrize(("starts", "modulation", "nu", "trough"), FOCAL_CASES)
def test_the_screws_distance_is_once_differentiable_at_its_focal_set_and_its_normal_continuous(starts, modulation, nu,
                                                                                               trough):
    """The declared orders, measured along the ray: towards the surface the normal's Jacobian grows as one over the
    distance from the set (a tenfold step nearer, tenfold larger: no derivative there, order 0), while the distance's
    derivative stays the ray's direction; on the seam's side, where the two nearest points part, it tends to it as the
    square root of that distance (continuous at the set: order 1)."""
    radius = TYPICAL["screw"]["radius"]
    jacobians, parted = [], []
    for out in (1e-3, 1e-4, 1e-5):
        parameters, point, ray, _ = focal_case(starts, modulation, nu, trough, out * radius)
        numbers = held(parameters)
        jacobian = np.asarray(jax.jacfwd(lambda p: SCREW.outward_normal(numbers, p))(jnp.asarray(point)))
        jacobians.append(float(np.linalg.norm(jacobian, 2)) * radius)
        gradient = np.asarray(jax.grad(lambda p: SCREW.signed_distance(numbers, p))(jnp.asarray(point)))
        assert np.max(np.abs(gradient - ray)) <= 1e-9, (out, gradient, ray)
        parameters, point, ray, _ = focal_case(starts, modulation, nu, trough, -out * radius)
        gradient = np.asarray(jax.grad(lambda p: SCREW.signed_distance(numbers, p))(jnp.asarray(point)))
        parted.append(float(np.linalg.norm(gradient - ray)))
    print(f"\nthe normal's Jacobian times the radius {jacobians}; the distance's derivative from the ray's on the seam "
          f"{parted}, a thousandth to a hundred-thousandth of the radius from the set")
    assert all(8.0 <= jacobians[k + 1] / jacobians[k] <= 12.5 for k in range(2)), jacobians
    assert all(0.2 <= parted[k + 1] / parted[k] <= 0.5 for k in range(2)), parted
