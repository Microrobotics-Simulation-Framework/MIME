"""The rules a choice over a declared range keeps, and the checks a study makes of a ranged experiment before its first
sweep, on the planted entry (``ranged_planted.RangedShares``, no solver of its own: a build of it is cheap).

- **A choice declares its rule.** Monotone (judged at the corners) or sampled (and inside, at a stated count); one that
  declares neither is refused as the entry's defect. A choice that peaks inside the range, declared monotone, gives
  less at its corners than the member at its peak needs, and that member is refused (the control the coordinator asked
  for: corners alone miss it); declared sampled at nine points it is found.
- **A split inside a range is the entry's.** A rung that chooses from the member's own value splits a sweep across the
  range, and the plan reports it so (exploratory and research); the strict profile's rule refuses it.
- **The independence check.** The planted entry that reads the range's width past ``choose_over`` is refused; the
  clean one is not.
- **The trace form.** The planted entry that bakes a Python float from the radius into its step is refused.
- **Cached.** The second plan of an unchanged experiment reuses both verdicts.

Every control here fails on every run: each is deterministic (no draw), and each test asserts its control's refusal by
code."""

from __future__ import annotations

import pytest
from ranged_rigs import RADIUS, planted

from mime.build import BuildRefused
from mime.registry import Monotone, Sampled
from mime.registry._ranges import RangeChoice, points_over
from mime.study import Study, grid
from mime.study._ranges import forget, range_check
from mime.study._study import RANGE_SPLIT
from mime.units import mm, ms

INSIDE = [mm(r) for r in (1.0, 1.05, 1.1, 1.15)]


@pytest.fixture(autouse=True)
def fresh_checks():
    forget()
    yield
    forget()


def sweep(mode: str, values=INSIDE, **sweep_settings):
    return Study(planted(mode)).sweep(grid({RADIUS: list(values)}), duration=ms(2), **sweep_settings)


def test_a_monotone_rule_is_judged_at_the_corners_and_a_sampled_one_inside_as_well():
    ranges = {"radius": (1.0, 2.0), "length": (3.0, 5.0)}
    assert len(points_over(ranges, Monotone("a basis"))) == 4
    sampled = points_over(ranges, Sampled(5, "a basis"))
    assert len(sampled) == 25 and {p["radius"] for p in sampled} == {1.0, 1.25, 1.5, 1.75, 2.0}


def test_a_choice_that_declares_no_rule_is_refused_as_the_entrys_defect():
    with pytest.raises(BuildRefused) as refused:
        planted("unruled").plan()
    assert [r.code for r in refused.value.refusals] == ["entry.defect"]
    assert "neither Monotone(basis) nor Sampled(count, basis)" in refused.value.refusals[0].message


def test_a_peak_inside_the_range_declared_monotone_is_refused_at_the_peak_and_found_when_sampled():
    corners = planted("peak", 1.0).plan()  # at a corner the corners' choice is the value's own
    (choice,) = corners.range_choices
    assert (choice.chosen, choice.own, choice.short) == (50.0, 50.0, False)
    with pytest.raises(BuildRefused) as refused:
        planted("peak", 1.075).plan()
    assert [r.code for r in refused.value.refusals] == ["entry.defect"]
    assert "misses this value" in refused.value.refusals[0].message and "own value needs 106" in \
        refused.value.refusals[0].message
    (found,) = planted("peak-sampled", 1.075).plan().range_choices
    assert (found.chosen, found.own, found.short) == (106.0, 106.0, False)
    assert planted("peak-sampled", 1.0).plan().range_choices[0].chosen == 106.0, "the peak, from every member"


def test_a_choice_falls_short_only_where_the_value_needs_more_than_it_gives():
    rule = Monotone("a basis")
    assert RangeChoice("n", "o", rule, ("p",), {"p": 1.0}, 10.0, 11.0).short
    assert not RangeChoice("n", "o", rule, ("p",), {"p": 1.0}, 10.0, 10.0).short
    assert RangeChoice("s", "o", rule, ("p",), {"p": 1.0}, 0.2, 0.1, largest=False).short


def test_a_sweep_inside_the_range_is_one_group_and_one_compile():
    plan = sweep("clean").plan()
    assert len(plan.groups) == 1 and plan.compiles == 1, plan.text(memory=False)
    result = sweep("clean").run()
    assert result.compiles == (1,)


def test_a_rung_that_chooses_from_the_members_own_value_splits_the_sweep_reported_as_its_defect():
    plan = sweep("own").plan()
    assert len(plan.groups) == 4, plan.text(memory=False)
    for group in plan.groups[1:]:
        (reason,) = group.reasons
        assert reason.startswith(f"inside the declared range of {RADIUS}, an entry chose from the member's own value")
        assert "the entry's defect" in reason
    assert "inside the declared range" in plan.text(memory=False)


def test_the_strict_profiles_rule_refuses_a_split_inside_a_range_and_the_others_report_it():
    assert RANGE_SPLIT == {"exploratory": "reported", "research": "reported", "strict": "refused"}
    from mime.study._study import range_splits

    plan = sweep("own").plan()
    settings, _, _ = planted("own").declared_as()
    from dataclasses import replace

    strict = replace(settings, profile="strict")
    found = range_splits(strict, plan.axes, plan.points, list(plan.groups), {RADIUS})
    assert [r.code for r in found] == ["study.range-split"] * 3
    assert "the strict profile refuses a split inside a declared range" in found[0].message
    assert range_splits(replace(settings, profile="research"), plan.axes, plan.points, list(plan.groups),
                        {RADIUS}) == []


def test_a_point_outside_the_range_is_built_for_its_own_value_a_group_of_its_own():
    plan = sweep("clean", values=[mm(1.0), mm(1.1), mm(1.3)]).plan()
    assert [len(g.members) for g in plan.groups] == [2, 1]
    assert plan.points[2].declared.outside == (RADIUS,)
    assert plan.groups[1].reasons[0] == (f"its points lie outside the declared range of {RADIUS}, and are built for "
                                         f"their own value")


def test_the_independence_check_refuses_an_entry_that_reads_the_range_past_its_choices():
    assert range_check(planted("clean")).refusals == ()
    with pytest.raises(BuildRefused) as refused:
        sweep("unreported").plan()
    (defect,) = refused.value.refusals
    assert defect.code == "entry.defect" and "reads the range other than through a choice it reports" in defect.message
    for at in ("the experiment's own value", f"{RADIUS} 0.001 m", f"{RADIUS} 0.00115 m"):
        assert f"at {at}, " in defect.message, at


def test_the_trace_form_refuses_an_entry_that_bakes_the_ranged_value_into_its_step():
    found = range_check(planted("clean"))
    assert found.traced == {RADIUS: "quiet"}
    with pytest.raises(BuildRefused) as refused:
        sweep("bake").plan()
    assert [r.code for r in refused.value.refusals] == ["entry.defect"]
    assert "changes a constant of the compiled step" in refused.value.refusals[0].message


def test_an_unchanged_experiments_next_plan_reuses_the_checks_verdicts():
    first = range_check(planted("clean"))
    again = range_check(planted("clean"))
    assert not first.reused and first.seconds > 0.0
    assert again.reused and again.seconds == 0.0 and again.refusals == first.refusals
    other = planted("clean", ranged=(1.0, 1.2))
    assert not range_check(other).reused, "another range is another content"


def test_a_difference_or_a_bake_no_entry_owns_is_refused_as_mimes_defect(monkeypatch: pytest.MonkeyPatch):
    """Where the difference lies in no node an entry built (the graph's edges, the traced step as a whole), the refusal
    is MIME's: the same plants with their nodes' owner unknown."""
    import mime.study._ranges as ranges

    monkeypatch.setattr(ranges, "_owner_class", lambda plans, node: None)
    assert [r.code for r in range_check(planted("unreported")).refusals] == ["study.range-dependent"]
    assert [r.code for r in range_check(planted("bake")).refusals] == ["study.range-bake"]
