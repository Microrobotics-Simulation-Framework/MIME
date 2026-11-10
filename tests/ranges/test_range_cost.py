"""The cost of a choice over declared ranges, refused before it is paid (``range.cost``): a choice evaluates its measure
at each of its rule's points in each plan the ranges ask for (the experiment's own and each corner's), and past
``MAX_EVALUATIONS`` in all it is refused before its first evaluation, naming the numbers its rule samples, with a
route (fewer ranged numbers, the rest a study's axes).

On the first consumer: FL-9 at 1.5 mm with its radius, length and modulation ranged asks for 729 points (each sampled
at nine) in 9 plans, 6561 evaluations, which took about ten minutes of planning before the limit (the review of the
declared ranges); it is refused with no force point counted. On the planted entry (cheap, no solver) the limit is
met exactly: 9 points in 3 plans (one range) and in 5 (two), refused one evaluation below, every run (no draw)."""

from __future__ import annotations

import dataclasses

import pytest
import ranged_planted
from mrs_cases import free_space, mrs
from ranged_rigs import planted
from script_a import fl9

import mime.interactions.hydrodynamic as hydrodynamic
import mime.registry._ranges as ranges
from mime.build import BuildRefused
from mime.registry._ranges import MAX_EVALUATIONS
from mime.units import kg, kg_m2, mm, mPa_s

SCREW = "umr.parts.shell."


def screw_with(*numbers: str):
    """FL-9 at 1.5 mm in free space with each of ``numbers`` ranged a few percent about its own value."""
    screw = fl9()
    experiment = free_space("fl9", {"umr": screw}, mrs(mm(1.5)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)))
    for number in numbers:
        own = getattr(screw, number)
        low, high = (dataclasses.replace(own, value=float(own.value) * factor) for factor in (0.97, 1.03))
        experiment.declare_range(SCREW + number, low, high, reason="a design range")
    return experiment


def test_three_sampled_numbers_of_the_screw_are_refused_before_any_force_point_is_counted(
        monkeypatch: pytest.MonkeyPatch):
    counted: list[object] = []
    counting = hydrodynamic.force_points
    monkeypatch.setattr(hydrodynamic, "force_points", lambda *args: counted.append(args) or counting(*args))
    with pytest.raises(BuildRefused) as refused:
        screw_with("radius", "length", "modulation").plan()
    (cost,) = [r for r in refused.value.refusals if r.code == "range.cost"]
    assert counted == [], "refused before the first evaluation"
    assert "729 points in all, in each of the 9 plans" in cost.message and "6561 evaluations" in cost.message
    assert f"over the {MAX_EVALUATIONS} a build makes" in cost.message
    for number in ("radius", "length", "modulation"):
        assert SCREW + number in cost.action
    assert "Study(exp).sweep(grid({...}), duration=...)" in cost.action
    assert [r.code for r in refused.value.refusals] == ["range.cost"], "the user's, not the entry's defect"


def test_the_screws_default_spacing_is_refused_by_the_same_limit_before_it_is_chosen():
    screw = fl9()
    experiment = free_space("fl9", {"umr": screw}, hydrodynamic.Drag.MRS(), mass=kg(1e-7),
                            inertia=kg_m2((1e-14, 2e-14, 3e-14)))
    for number in ("radius", "length", "modulation"):
        own = getattr(screw, number)
        experiment.declare_range(SCREW + number, *(dataclasses.replace(own, value=float(own.value) * f)
                                                   for f in (0.97, 1.03)), reason="a design range")
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    assert "range.cost" in [r.code for r in refused.value.refusals]
    assert any("choice 'default spacing'" in r.message for r in refused.value.refusals if r.code == "range.cost")


@pytest.mark.parametrize(("extra", "plans"), [(False, 3), (True, 5)], ids=["one-range", "two-ranges"])
def test_the_limit_is_the_rules_points_times_the_plans_met_exactly_and_refused_one_below(
        monkeypatch: pytest.MonkeyPatch, extra: bool, plans: int):
    """The planted choice samples the radius at nine points; the plans are the experiment's own and each corner of
    every declared range, the viscosity's as well as the radius's."""
    calls: list[object] = []
    peaked = ranged_planted.peaked
    monkeypatch.setattr(ranged_planted, "peaked", lambda hull: calls.append(hull) or peaked(hull))

    def experiment():
        made = planted("peak-sampled")
        if extra:
            made.declare_range("water.viscosity", mPa_s(0.9), mPa_s(1.1), reason="water's range")
        return made

    monkeypatch.setattr(ranges, "MAX_EVALUATIONS", 9 * plans)
    experiment().plan()
    assert calls, "at the limit the choice is made"
    calls.clear()
    monkeypatch.setattr(ranges, "MAX_EVALUATIONS", 9 * plans - 1)
    with pytest.raises(BuildRefused) as refused:
        experiment().plan()
    assert [r.code for r in refused.value.refusals] == ["range.cost"]
    assert f"in each of the {plans} plans" in refused.value.refusals[0].message
    assert f"{9 * plans} evaluations" in refused.value.refusals[0].message
    assert calls == [], "refused before the first evaluation"
