"""Each refusal the study layer constructs, met where it is raised: the code, and the action it names."""

from __future__ import annotations

import pytest
from control_cases import spun_magnet
from script_a import reproduction_arm

from mime import Study
from mime.build import BuildRefused
from mime.build._refusals import MimeError
from mime.results import StudyResult
from mime.study import DT, PRECISION, grid
from mime.study._override import parse_override, replayed
from mime.units import A_m2, Hz, kg, mm, ms, s


def codes(error: pytest.ExceptionInfo[MimeError]) -> list[str]:
    return error.value.codes()


def test_the_paths_and_points_a_sweep_refuses():
    exp = reproduction_arm()
    misspelt = Study(exp).sweep(grid({"umr.mas": [kg(8e-5)]}), duration=s(0.01)).plan()
    with pytest.raises(MimeError) as path:
        Study(exp).sweep(grid({"nothing.here": [kg(8e-5)]}), duration=s(0.01)).plan()
    with pytest.raises(MimeError) as name:
        exp.handle("feild")
    with pytest.raises(MimeError) as values:
        Study(exp).sweep([{"umr.mass": kg(8e-5)}, {"umr.inertia": kg(8e-5)}], duration=s(0.01))
    with pytest.raises(MimeError) as twice:
        position = exp.handle("umr").initial.position  # type: ignore[union-attr]
        Study(exp).sweep(grid({position: [mm((0.0, 0.0, -1.0))], "umr.initial.position": [mm((0.0, 0.0, -1.0))]}),
                         duration=s(0.01)).plan()
    with pytest.raises(MimeError) as batch:
        Study(exp).sweep(grid({"umr.mass": [kg(8e-5)]}), duration=s(0.01), batch=0)
    with pytest.raises(MimeError) as no_sweep:
        Study(exp).plan()
    with pytest.raises(MimeError) as precision:
        Study(exp).sweep(grid({PRECISION: ["float16"]}), duration=s(0.01)).plan()
    assert misspelt.refused and misspelt.refused[0].refusals, misspelt.text()  # a value's path is its plan's to judge
    assert codes(path) == ["study.unknown-path"] and codes(name) == ["call.unknown-name"]
    assert codes(values) == ["study.axis-values"] and codes(twice) == ["study.axis-twice"]
    assert codes(batch) == ["study.batch"] and codes(no_sweep) == ["study.no-sweep"]
    assert codes(precision) == ["study.precision"]


def test_a_group_of_settings_not_given_cannot_have_its_settings_set():
    import dataclasses

    from mime.build import Attachment
    from mime.experiment import Experiment

    settings, declarations, _ = reproduction_arm().declared_as()
    held = [Attachment(dataclasses.replace(d.law, lubrication=None), d.bindings)
            if isinstance(d, Attachment) and d.law.name == "drag" else d for d in declarations]
    with pytest.raises(MimeError) as unset:
        Study(Experiment.declared_from(settings, held)).sweep(
            grid({"drag:umr.lubrication.epsilon": [mm(0.1)]}), duration=s(0.01)).plan()
    assert codes(unset) == ["study.unset-group"]


def test_reading_a_result_wrongly_is_refused():
    result = Study(reproduction_arm()).sweep(grid({DT: [s(5e-4), s(2.5e-4)]}), duration=s(0.005),
                                             record=["swim_speed:umr.speed"]).run()
    with pytest.raises(MimeError) as member:
        result.member(2)
    with pytest.raises(MimeError) as observation:
        _ = result["nothing:here"]
    with pytest.raises(MimeError) as series:
        _ = result["swim_speed:umr"].speed
    assert codes(member) == ["study.no-member"] and codes(observation) == ["study.no-observation"]
    assert codes(series) == ["study.series-time-axes"]
    empty = StudyResult("x", ["a"], {"a": []}, [], [], [])
    with pytest.raises(MimeError) as none:
        _ = empty["swim_speed:umr"]
    assert codes(none) == ["study.no-observation"]


def test_the_overrides_a_run_refuses():
    exp = spun_magnet()
    with pytest.raises(BuildRefused) as callback:
        exp.build().run(ms(30), override={"spin.magnetic_moment": A_m2(20.0)})
    built = reproduction_arm().build()
    built.run(s(0.005))
    with pytest.raises(BuildRefused) as started:
        built.run(s(0.005), override={"umr.initial.position": mm((0.0, 0.0, -0.9))})
    built._declared = (None, None)  # noqa: SLF001 - a build made without its declarations
    with pytest.raises(BuildRefused) as undeclared:
        built.run(s(0.005), override={"umr.mass": kg(8e-5)})
    with pytest.raises(BuildRefused) as unreplayable:
        replayed({"path": "umr.mass", "value": "8e-05 kg", "si": None, "unit": None})
    with pytest.raises(BuildRefused) as written:
        parse_override("field.commands.frequency")
    assert codes(callback) == ["override.callback"] and codes(started) == ["override.initial-after-start"]
    assert codes(undeclared) == ["override.no-declarations"] and codes(unreplayable) == ["override.unreplayable"]
    assert codes(written) == ["override.written"]
    assert parse_override("field.commands.frequency=20Hz")[1] == Hz(20.0)
    assert parse_override("umr.initial.position=0,0,-1 mm")[1] == mm((0.0, 0.0, -1.0))
