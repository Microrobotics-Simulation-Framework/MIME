"""The frame proposals, each checked: up is declared once, on the scene, and gravity derived from it; there are no
default axes, and a missing up is refused naming it; directions refer to a thing's rather than repeat it; the report
has one orientation line per thing, a reference written as what it refers to. Nodes never hard-coding an axis is
``test_rotated_world.py``, and the build writing the stage is ``test_stage.py``."""

from dataclasses import replace

import pytest
from script_a import script_a

from mime import Experiment, Scene
from mime.build import BuildRefused
from mime.scene import Tube, X, Z
from mime.units import m_per_s2, mm, ms, standard_gravity


def refusals(experiment: Experiment) -> dict[str, str]:
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    return {r.code: f"{r.message} | {r.action}" for r in refused.value.refusals}


def test_up_is_declared_on_the_scene_and_gravity_derived_along_minus_up():
    plans = Experiment("frame", scene=Scene.new(up=(0, 2, 0), units="m"), gravity=m_per_s2(9.0), dt=ms(20)).plan()
    assert plans.frame.up == (0.0, 1.0, 0.0) and plans.frame.gravity == (-0.0, -9.0, -0.0)
    assert -Scene.new(up=Z, units="m").up == (0.0, 0.0, -1.0)


def test_an_up_changed_after_its_declaration_is_refused_at_the_build():
    """``Scene.new`` refuses an up other than +Y or +Z; the build holds a scene changed after it to the same rule,
    since the stage's ``upAxis`` names only those two."""
    scene = Scene.new(up=Z, units="m")
    scene.declared_up = X
    found = refusals(Experiment("frame", scene=scene, gravity=standard_gravity(), dt=ms(1)))
    assert "neither +Y nor +Z" in found["frame.up-not-y-or-z"]


def test_up_on_the_experiment_is_refused_alone_and_beside_a_scene():
    alone = refusals(Experiment("frame", up=(0, 0, 1), gravity=standard_gravity(), dt=ms(1)))
    assert set(alone) == {"frame.up-on-experiment"}
    assert "declare it on the scene" in alone["frame.up-on-experiment"]
    twice = refusals(Experiment("frame", scene=Scene.new(up=Z, units="m"), up=(0, 0, 1), gravity=standard_gravity(),
                                dt=ms(1)))
    assert set(twice) == {"frame.up-twice"} and "up is declared twice" in twice["frame.up-twice"]


def test_a_missing_up_is_refused_naming_up_and_no_axis_has_a_default():
    for experiment in (Experiment("frame", gravity=standard_gravity(), dt=ms(1)),
                       Experiment("frame", scene=Scene.new(up=None, units="m"), gravity=standard_gravity(), dt=ms(1))):
        found = refusals(experiment)
        assert "no up direction, and MIME assumes no axis" in found["frame.up-missing"]
    with pytest.raises(TypeError, match="'up'"):
        Scene.new(units="m")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="'axis'"):
        Tube(inner_radius=mm(3), wall=mm(1), length=mm(9))  # type: ignore[call-arg]
    assert "scene.units" in refusals(Experiment("frame", scene=Scene.new(up=Z, units="furlong"),
                                                gravity=standard_gravity(), dt=ms(1)))


def test_a_direction_refers_to_the_tubes_axis_and_follows_it():
    def axes(tube_axis):
        experiment = script_a()
        held = experiment._declarations[0].thing
        experiment._declarations[0] = replace(experiment._declarations[0], thing=replace(
            held, geometry=replace(held.geometry, axis=tube_axis)))
        plans = experiment.plan()
        field = next(p for p in plans.entries if p.owner == "field.magnetic")
        speed = next(p for p in plans.entries if p.owner == "swim_speed:umr")
        return field.settings["rotation_axis"], speed.settings["along"]

    assert axes(+X) == ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert axes(-Z) == ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0))
    fields = script_a().document()["things"]["field"]["fields"]["magnetic"]["settings"]
    assert fields["rotation_axis"] == "tube.axis"


def test_the_report_has_one_orientation_line_per_thing_with_references_as_references():
    report = script_a().build().report
    lines = {line.thing: line.text() for line in report.orientation}
    assert set(lines) == {"tube", "water", "umr", "field"}
    assert lines["tube"] == "tube (structure at /World/tube): axis (1, 0, 0)"
    assert lines["water"] == "water (medium at /World/water): fills tube.lumen"
    assert lines["umr"] == ("umr (body at /World/umr): initial orientation (w, x, y, z) (1, 0, 0, 0); swim_speed:umr "
                            "along = tube.axis (1, 0, 0)")
    assert lines["field"] == ("field (apparatus at /World/field): field.magnetic rotation_axis = tube.axis (1, 0, 0); "
                              "field.magnetic initial_direction (0, 1, 0)")
    text = report.text(details=True)
    assert "scene: up (0, 0, 1) from the scene, so gravity (0, 0, -9.81) m/s^2" in text
    assert "orientation, one line per thing:\n  tube (structure at /World/tube): axis (1, 0, 0)" in text
