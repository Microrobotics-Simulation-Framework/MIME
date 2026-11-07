"""One reason per swept axis in the research profile (the maintainer's decision on #187's review, m4): a research study
whose axis has no reason is refused naming the axis; the same study with a reason runs, and the reason is in the
folder's manifest and in each member's record and on the value each member declares; a reason for something that is no
axis is refused. The control: the exploratory profile runs a study with no reason, each value declared with the generic
study reason, which the manifest says."""

from __future__ import annotations

import json

import pytest
from profile_cases import Paperwork, settling_ball

from mime import Study
from mime.build import BuildRefused
from mime.results._study_folder import STUDY_FILE
from mime.study import grid
from mime.study._axes import STUDY_REASON
from mime.units import kg_per_m3, mm, ms

AXIS = "ball.parts.shell.density"
REASON = "steel grades span 7750 to 8100 kg/m^3"
SEALED = Paperwork(profile="research", acceptance="water is Newtonian at a settling ball's shear rates",
                   density="a steel ball's density", spacing=mm(0.2, reason="converged at half this spacing"),
                   blob=0.5, quadrature_refinement=4,
                   sources={"drag:ball.blob": "the regularised method's usual blob",
                            "drag:ball.quadrature_refinement": "the quadrature's spacing half the blob"})
POINTS = grid({AXIS: [kg_per_m3(7850.0), kg_per_m3(8100.0)]})


def study(paperwork, **options):
    return Study(settling_ball(paperwork)).sweep(POINTS, duration=ms(8.0), record=["swim_speed:ball.speed"], **options)


def test_a_research_axis_without_a_reason_is_refused_naming_the_axis_and_with_one_it_runs(tmp_path):
    with pytest.raises(BuildRefused) as refused:
        study(SEALED).plan()
    assert refused.value.codes() == ["study.axis-reason"]
    assert AXIS in refused.value.refusals[0].message and "one string per axis, not one per point" in refused.value.refusals[0].action
    with pytest.raises(BuildRefused) as blank:
        study(SEALED, reasons={AXIS: "  "}).plan()
    assert blank.value.codes() == ["study.axis-reason"]
    held = study(SEALED, reasons={AXIS: REASON})
    plan = held.plan()
    declared = [d for d in plan.points[1].declared.declarations if str(getattr(d, "path", "")) == AXIS]
    assert len(declared) == 1 and declared[0].reason == REASON, "the value each member declares carries its axis's reason"
    out = tmp_path / "study"
    result = held.run_to(out)
    document = json.loads((out / STUDY_FILE).read_text())
    assert document["axes"][0]["reason"] == REASON and document["experiment"]["profile"] == "research"
    assert [o["reason"] for m in document["members"] for o in m["overrides"]] == [REASON, REASON]
    assert result.record["axes"][0]["reason"] == REASON


def test_a_reason_for_something_that_is_no_axis_is_refused():
    with pytest.raises(BuildRefused) as refused:
        study(SEALED, reasons={AXIS: REASON, "ball.radius": "a typo"}).plan()
    assert refused.value.codes() == ["study.reason-no-axis"] and "ball.radius" in refused.value.refusals[0].message


def test_the_control_an_exploratory_study_without_a_reason_runs_stamped_with_the_generic_one(tmp_path):
    held = study(Paperwork())
    out = tmp_path / "study"
    result = held.run_to(out)
    assert len(result) == 2
    document = json.loads((out / STUDY_FILE).read_text())
    assert document["axes"][0]["reason"] == STUDY_REASON and document["experiment"]["profile"] == "exploratory"
    stamp = result.member(0).layout.stamp
    assert stamp is not None and stamp.profile == "exploratory" and "profile" in result.member(0).text()
