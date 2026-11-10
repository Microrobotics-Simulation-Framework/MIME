"""The MRS default spacing's step below a rounding past its bound. The family's spacing is the largest to three
significant figures at which the kind gives its count; where the rounding lands above the boundary the rule steps one
unit of the third figure down. Drag 0.5.0 sizes that unit from the boundary, so the result is the boundary rounded
down to three figures; drag 0.3.0 and 0.4.0 size it from the rounded value, which within half a unit below a power of
ten is that power, and step ten times too far (0.99 mm, not 0.999 mm), kept so released runs reproduce.

**The property** runs the family's own path (:meth:`PointForceDrag._family_spacing`, its bisection included) on a kind
whose boundary is drawn, across nine decades, and compares with the boundary rounded down by exact decimal arithmetic.
The draw is a three-figure integer ``n`` and a fraction of the last unit, so the exact answer is ``n`` units; ``n`` is
999 half the time, which with a fraction past a half is the band where the rounding lands on a power of ten.

**The control**, deterministic (a fixed grid in the band, so it fails on every run): the released step misses the
exact rounding-down at every grid point in the band, and takes the power of ten's unit there."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from hypothesis import given
from hypothesis import strategies as st
from mrs_cases import mrs, uncoupled
from script_a import fl9

from mime.interactions import Contact, Drag
from mime.interactions.hydrodynamic import UncoupledDrag, _step_down
from mime.registry import Hull
from mime.scene import Scene, Tube, Z
from mime.things import Part
from mime.units import kg, kg_m2, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

#: Nine decades of the boundary's unit, 1e-9 m to 1e-1 m a unit of the third figure: a micrometre to a metre.
DECADES = range(-9, 0)
FRAME = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
BAND = (0.5, 1.0)  # the fraction of the last unit, with n = 999, where the rounding lands on a power of ten


def sized_for(boundary: float, entry: Any) -> float:
    """``entry``'s family spacing for a kind that gives its count (the entry's compact samples) at every spacing up to
    ``boundary`` and none beyond it, found by the rule's own bisection from a reach 3.7 times the boundary (so the bisection has work to do)."""
    kind = SimpleNamespace(name="stub", patches=("p",), radial_reach=lambda settings: 3.7 * boundary,
                           sample_counts=lambda settings, spacing, patch: (entry.compact_samples,) if spacing <= boundary
                           else (0,))
    hull = Hull("body", "shell", kind, {}, (0.0, 0.0, 0.0), FRAME, {})  # type: ignore[arg-type]
    with mock.patch("mime.interactions.hydrodynamic.implements", lambda held, capability: capability == "radial_reach"):
        return float(entry._family_spacing(hull))


def rounded_down(digits: int, decade: int) -> float:
    """``digits`` units of ``10**decade``, as the float the rule's ``float(f"{...:.3g}")`` gives."""
    return float(Decimal(digits).scaleb(decade))


def exact_rounding_down(boundary: Decimal) -> float:
    """``boundary`` rounded down to three significant figures, in exact decimal arithmetic."""
    unit_exponent = boundary.adjusted() - 2
    return float(boundary.quantize(Decimal(1).scaleb(unit_exponent), rounding=ROUND_DOWN))


def drawn_boundary(digits: int, fraction: float, decade: int) -> Decimal:
    return (Decimal(digits) + Decimal(fraction)).scaleb(decade)


@pytest.mark.parametrize("decade", DECADES)
@given(digits=st.one_of(st.just(999), st.integers(100, 999)), fraction=st.floats(1e-6, 1.0 - 1e-6))
def test_drag_0_5_0_default_spacing_is_the_boundary_rounded_down_to_three_figures(decade, digits, fraction):
    boundary = drawn_boundary(digits, fraction, decade)
    assert sized_for(float(boundary), mrs(mm(0.6))) == exact_rounding_down(boundary)


@pytest.mark.parametrize("decade", DECADES)
@given(digits=st.integers(100, 998), fraction=st.floats(1e-6, 1.0 - 1e-6))
def test_the_released_step_is_the_same_outside_the_band(decade, digits, fraction):
    """Below 999 the rounded value's unit and the boundary's are one: drag 0.4.0 and 0.5.0 agree."""
    boundary = drawn_boundary(digits, fraction, decade)
    assert sized_for(float(boundary), uncoupled(mm(0.6))) == sized_for(float(boundary), mrs(mm(0.6)))


@pytest.mark.parametrize("decade", DECADES)
def test_control_the_released_step_misses_the_exact_rounding_in_the_band_at_every_point(decade):
    grid = [BAND[0] + (BAND[1] - BAND[0]) * k / 20 for k in range(1, 20)]
    for fraction in grid:
        boundary = drawn_boundary(999, fraction, decade)
        released = sized_for(float(boundary), uncoupled(mm(0.6)))
        assert released != exact_rounding_down(boundary)
        assert released == pytest.approx(rounded_down(99, decade + 1), rel=1e-12), "ten units under the power of ten"
        assert sized_for(float(boundary), mrs(mm(0.6))) == exact_rounding_down(boundary) == \
            pytest.approx(rounded_down(999, decade), rel=1e-12)


@pytest.mark.parametrize("decade", DECADES)
@given(digits=st.one_of(st.just(999), st.integers(100, 999)), fraction=st.floats(1e-6, 1.0 - 1e-6))
def test_the_step_the_confined_path_shares_is_exact_from_the_bound(decade, digits, fraction):
    """The confined path's bound is ``approach / CONFINED_GAP``; its rounding past the bound takes the same step."""
    bound = drawn_boundary(digits, fraction, decade)
    rounded = float(f"{float(bound):.3g}")
    if rounded > float(bound):
        assert _step_down(rounded, float(bound), True) == exact_rounding_down(bound)


def held_off(law: Any, margin_mm: float) -> Any:
    """FL-9 in a tube, a contact holding it ``margin_mm`` off the inner surface, and ``law`` with its spacing left to
    the rule (the confined path)."""
    from mime import Body, Experiment, Medium, State, Structure
    from mime.dynamics import RigidInertial

    e = Experiment("held_off", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    e.add_structure(Structure("tube", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(3.175), wall=mm(0.5),
                                                    length=mm(100.0))))
    e.add_medium(Medium("water", fills="tube.lumen"))
    e.declare("water.viscosity", mPa_s(1.0), reason="water")
    e.add_body(Body("body", dynamics=RigidInertial(), parts={"shell": Part(fl9())}, initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    e.declare("body.mass", kg(1e-5), reason="a millimetre body's order")
    e.declare("body.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
    e.attach(law(wall_kind="tube"), body="body", medium="water")
    e.attach(Contact.Projection(margin=mm(margin_mm)), body="body", surface="tube.inner_surface")
    return e


def COARSE(**settings: Any) -> Any:  # noqa: N802 - a rung's alias, as Drag.MRS reads
    """Drag 0.5.0's mrs rung with lubrication's hand-over off: its default spacing follows the closest approach."""
    return Drag.MRS(lubrication=False, **settings)


def planned_spacing_mm(law: Any, margin_mm: float) -> float:
    taken = {d.path: d.value for d in held_off(law, margin_mm).plan().profile.defaults}
    return float(taken["drag:body.spacing"].value) * 1e3


def test_the_confined_spacing_a_closest_approach_needs_steps_from_its_bound_on_0_5_0_and_as_released_on_0_4_0():
    """A margin of 0.054978 mm is a bound of 0.09996 mm (a closest approach over CONFINED_GAP, 0.55), whose three
    figures, 0.100 mm, lie past it: 0.5.0 takes 0.0999 mm and 0.4.0, as released, 0.099 mm. A margin whose bound is not
    in the band (0.0549 mm, a bound of 0.09982 mm) gives 0.0998 mm on both."""
    assert planned_spacing_mm(COARSE, 0.054978) == pytest.approx(0.0999, rel=1e-12)
    assert planned_spacing_mm(UncoupledDrag.MRS, 0.054978) == pytest.approx(0.099, rel=1e-12)
    assert planned_spacing_mm(COARSE, 0.0549) == planned_spacing_mm(UncoupledDrag.MRS, 0.0549) == \
        pytest.approx(0.0998, rel=1e-12)
