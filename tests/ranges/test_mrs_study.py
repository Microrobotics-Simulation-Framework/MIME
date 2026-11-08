"""Gate 4 on the first consumer: a sweep of the pulled sphere's radius across its declared range ([1.0, 1.15] mm, at
1.0, 1.05, 1.1 and 1.15 mm) is one group that compiles once, each member on the range's 91 force points; a point
outside it (1.2 mm) is built for its own value, a group of its own, the split reason naming the range. The range's
checks find nothing on it (the independence check at the base and both corners, the trace form at a move inside), and
the first plan reports what they cost. An override inside the range runs as data on the one compiled step and is the
experiment at that value run alone, bit for bit; outside it, it is refused, naming the range. S2b's gate 1 with the range
in the base: each member against ``run(override=...)`` of its point on a fresh build of the base, within the members'
tolerance. A member made an experiment of its own (``as_experiment``) carries the range where it lay inside it, and runs
as the member ran; the study folder's saved base holds the range."""

from __future__ import annotations

import numpy as np
import pytest
from ranged_rigs import RADIUS, moving, pulled

from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.study import Study, grid
from mime.study._ranges import forget
from mime.units import mm, ms

RADII = (1.0, 1.05, 1.1, 1.15, 1.2)
DURATION = ms(3)
#: S2b's gate 1 line for members against their own runs: a member batched with others and the same point run alone
#: differ by the batch's rounding only, a few units in the last place of each field's own scale.
MEMBER_EPS = 64


@pytest.fixture(scope="module")
def swept():
    forget()
    study = Study(pulled()).sweep(grid({RADIUS: [mm(r) for r in RADII]}), duration=DURATION)
    plan = study.plan()
    return plan, study.run()


def test_a_sweep_across_the_range_is_one_group_on_the_ranges_force_points_and_compiles_once(swept):
    plan, result = swept
    assert [len(group.members) for group in plan.groups] == [4, 1], plan.text(memory=False)
    assert result.compiles == (1, 1)
    inside, outside = plan.groups
    assert inside.built is not None and inside.built.graph.get_node("drag:ball").sample_count == 91
    assert outside.built is not None and outside.built.graph.get_node("drag:ball").sample_count == 91
    assert outside.built.graph.get_node("drag:ball").constants()["counts"] == ((51, 101),), "its own sampling"
    assert outside.reasons[0] == f"its points lie outside the declared range of {RADIUS}, and are built for their own value"
    assert plan.range_check is not None and plan.range_check.refusals == () and not plan.range_check.reused
    assert plan.range_check.traced == {RADIUS: "quiet"} and plan.range_check.differences == ()
    assert "the declared ranges checked before the sweep" in plan.text(memory=False)


def test_each_member_inside_the_range_is_its_point_run_on_the_built_base(swept):
    plan, result = swept
    for index in range(4):
        alone = pulled().build().run(DURATION, history=True, override={RADIUS: mm(RADII[index])})
        mine = np.asarray(result.member(index).final["ball"]["pose"], dtype=np.float64)
        theirs = np.asarray(alone.final["ball"]["pose"], dtype=np.float64)
        eps = float(np.finfo(np.asarray(alone.final["ball"]["pose"]).dtype).eps)
        scale = float(np.max(np.abs(theirs)))
        assert float(np.max(np.abs(mine - theirs))) <= MEMBER_EPS * eps * scale, index


def test_an_override_inside_the_range_is_the_experiment_at_that_value_run_alone_and_outside_it_is_refused():
    built = pulled().build()
    overridden = built.run(DURATION, history=True, override={RADIUS: mm(1.1)})
    alone = pulled(1.1).build().run(DURATION, history=True)
    for field in ("pose", "twist"):
        assert np.array_equal(np.asarray(overridden.final["ball"][field]), np.asarray(alone.final["ball"][field])), \
            field
    with pytest.raises(BuildRefused) as refused:
        built.run(DURATION, override={RADIUS: mm(1.2)})
    assert [r.code for r in refused.value.refusals] == ["override.outside-range"]
    assert "outside its declared range" in refused.value.refusals[0].message


def test_a_member_made_an_experiment_carries_the_range_and_one_outside_it_does_not(tmp_path):
    forget()
    study = Study(moving()).sweep(grid({RADIUS: [mm(1.1), mm(1.2)]}), duration=DURATION)
    result = study.run_to(tmp_path / "study")
    from mime.build import RangeDeclaration

    inside = result.member(0).as_experiment("inside")
    outside = result.member(1).as_experiment("outside")
    assert [str(d.path) for d in inside.declared_as()[1] if isinstance(d, RangeDeclaration)] == [RADIUS]
    assert not any(isinstance(d, RangeDeclaration) for d in outside.declared_as()[1])
    again = inside.build().run(DURATION, history=True)
    member = result.member(0)
    eps = float(np.finfo(np.asarray(again.final["ball"]["pose"]).dtype).eps)
    mine, theirs = (np.asarray(x, dtype=np.float64) for x in (member.final["ball"]["pose"], again.final["ball"]["pose"]))
    assert float(np.max(np.abs(mine - theirs))) <= MEMBER_EPS * eps * float(np.max(np.abs(theirs)))
    from mime.results._study_folder import BASE_DIR

    base = Experiment.load(tmp_path / "study" / BASE_DIR)
    assert [str(d.path) for d in base.declared_as()[1] if isinstance(d, RangeDeclaration)] == [RADIUS]
