"""The shape conformance suite (``mime.testing.check_shape``) over every registered shape, and the cylinder's normal
within its axis band.

**The property.** For each registered shape and each of its run-time capabilities (signed distance, outward normal,
nearest point, and the screw's tangent), Hypothesis draws probes at physical scales (a shape of 10 um to 1 cm, turned
at random and moved up to three sizes, a point off, near or on one of its declared sets, in float64 or float32, a
direction and a parameter moved with it), and autodiff's derivative along the curve must equal the float64 difference
on the point's side within ``SHAPE_TOLERANCE`` of the derivative's scale plus the rounding floor; on a set, autodiff
must be finite, and both one-sided slopes are reported.

**The measured agreement** (two stress runs of 2026-10-28, 6,400 probes each over every shape and capability, 400
draws per pair, no failure in the second: the largest ``|AD - FD| / S`` where the rounding floor is below a tenth of
the tolerance): ``MEASURED``, both at the screw's normal off its seam. The float64 tolerance lies thousands of times
above it and the float32 tolerance tens of times; the planted wrong tangent is a hundred times the float64 tolerance
(``test_the_tolerances_keep_their_recorded_margins``).

**The control** (decision, audit finding A3-S.11): a wrong tangent, autodiff's scaled by ``1 + PLANTED_TANGENT``
through a ``custom_jvp``, fails at an explicit example near each declared set of each shape, for each capability; the
same examples pass unplanted. The examples are fixed, so the control fails on every run; its rate over drawn probes
is reported, not relied on.

**The cylinder's in-band derivative** (design 3.19 items 6 and 18.2): within its axis band the cylinder's normal is
zero, and its derivative in the point, through a ``stop_gradient``, is the projection across the axis, recorded as an
artefact (``artefacts/cylinder_in_band_jacobian.json``: float64, a point 1e-16 m off the axis, inside the 4096-ulp
band). The control writes the in-band branch as the constant zero, whose Jacobian is zero, and fails."""

import json
import math
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import mime.testing._shape_suite as suite
from mime.core import AXIS_ULPS, CYLINDER, Cylinder, jnp, switch
from mime.testing import (
    PLANTED_TANGENT,
    SHAPE_CASES,
    SHAPE_TOLERANCE,
    ShapeProbe,
    check_shape,
    shape_probes,
    uncased_shapes,
)

PAIRS = [(shape, name) for shape, case in sorted(SHAPE_CASES.items()) for name in case.capabilities]
#: The largest agreement the stress runs measured where the floor was negligible, by dtype.
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
    assert found.problems == (), f"{found.line()}: " + "; ".join(found.problems)


#: Each case's shape as an example places it: a size each shape is used at, turned off every coordinate axis.
TYPICAL = {
    "sphere": {"centre": (0.0, 0.0, 0.0), "radius": 1e-3},
    "cylinder": {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "radius": 3.175e-3},
    "capsule": {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "radius": 5e-4, "height": 2e-3},
    "slab": {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "half_thickness": 5e-4},
    "dejongh_screw": {"centre": (0.0, 0.0, 0.0), "orientation": (1.0, 0.0, 0.0, 0.0), "radius": 1.56e-3,
                      "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33, "starts": 2},
}
TURN = np.array([0.9, 0.3, -0.2, 0.25]) / np.linalg.norm([0.9, 0.3, -0.2, 0.25])
DIRECTION = np.array([0.48, -0.6, 0.64])


def examples() -> list[tuple[ShapeProbe, str]]:
    """A probe a thousandth of the shape's size from each declared set, on its positive side, in float64, along a
    direction off every axis with the case's first parameter moved: where the floor is negligible and every
    capability's derivative along the curve is of its scale."""
    found = []
    for shape, case in sorted(SHAPE_CASES.items()):
        own = TYPICAL[shape]
        size = case.size(own)
        rotation = suite._rotation(TURN)  # noqa: SLF001 - the suite's own rotation, as its probes turn
        shift = size * np.array([0.7, -1.3, 2.1])
        first = case.moved[0]
        moved = {first: 0.1 / size if first in case.plain else 0.1}
        for chosen in case.sets:
            point = rotation @ chosen.place(own, 1e-3 * size, np.array([0.37, 0.61, 0.53])) + shift
            probe = ShapeProbe(shape, case.turned(own, TURN, shift), point, DIRECTION / np.linalg.norm(DIRECTION),
                               moved, chosen.owner, "near", 1e-3 * size, "float64", size)
            found += [(probe, name) for name in case.capabilities]
    return found


#: Capabilities whose derivative is zero wherever a difference is taken, so that no scaling of the tangent can show:
#: the slab's normal is its axis on one side of the mid-plane and its opposite on the other.
CONSTANT = {("slab", "normal")}


def test_control_a_planted_wrong_tangent_fails_at_every_example_near_each_set():
    caught, missed, constant = [], [], set()
    for probe, name in examples():
        clean = check_shape(probe, name)
        assert clean.problems == (), f"{clean.line()}: " + "; ".join(clean.problems)
        size = float(np.max(np.abs(clean.autodiff))) / clean.scale
        if size <= SHAPE_TOLERANCE["float64"]:  # zero, to rounding
            constant.add((probe.shape, name))
            continue
        assert size >= 0.05, f"{clean.line()}: the derivative is {size:.3g} of its scale, too small to show the tangent"
        planted = check_shape(probe, name, planted=PLANTED_TANGENT)
        (caught if planted.problems else missed).append(planted.line())
    print(f"\nthe planted tangent (x {1 + PLANTED_TANGENT}) is caught at {len(caught)} of {len(caught) + len(missed)} "
          f"examples; {sorted(constant)} have no derivative to scale")
    assert missed == [], "missed at: " + "; ".join(missed)
    assert constant == CONSTANT
    assert {line.split(" ")[0] for line in caught} == set(SHAPE_CASES)


def test_the_tolerances_keep_their_recorded_margins():
    """The tolerances thirty times above the measured agreement or more, and the planted tangent ten times above the
    float64 tolerance or more."""
    for dtype, measured in MEASURED.items():
        assert SHAPE_TOLERANCE[dtype] >= 30 * measured, dtype
    assert PLANTED_TANGENT >= 10 * SHAPE_TOLERANCE["float64"]


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
