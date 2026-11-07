"""``drives.Ramp`` and a study's protocols, ``FromRest`` and ``Ramp``, with what refuses them.

A ramp writes each step its value at the step's middle (the ramp's mean over the step where it neither starts nor
ends inside it), held, then moved at its rate, then held where it ends, in either direction; its numbers are leaves, so
ramps of other numbers share one compiled program. The protocol ``Ramp`` declares on each point a ramp starting from
the point's value of its command, and the result names the quantity it gives (``ramp up``, ``ramp down``), which
``step_out`` (from rest) refuses, as ``ramp_crossings`` refuses a study from rest. The build reads a ramp's turning rate
as the largest it reaches."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np
import pytest
from range_cases import rotating_screw
from script_a import reproduction_arm

from mime import Study, drives
from mime.build import ArgumentRefused, BuildRefused, ThingDeclaration
from mime.build._watch import WatchPlan
from mime.observations import PhaseLag
from mime.scene import X
from mime.study import FromRest, Ramp, grid, ramp_crossings, step_out
from mime.study._protocol import commanded
from mime.things import Apparatus
from mime.units import Hz, mT, ms, s

DT = 5e-4
LAG = "phase_lag:umr"


def ramped(drive: drives.Ramp, *, steps: int = 12) -> np.ndarray:
    """The values ``drive`` wrote into script A's frequency over ``steps`` steps."""
    history = reproduction_arm(frequency=drive).build().run(s(steps * DT), history=True).history
    assert history is not None
    return np.asarray(history["drive:field.frequency"]["value"], dtype=np.float64)


@pytest.mark.parametrize(("start", "to"), [(10.0, 10.6), (10.6, 10.0)], ids=["up", "down"])
def test_a_ramp_writes_its_value_at_each_steps_middle_held_moved_and_held(start, to):
    drive = drives.Ramp(Hz(start), to=Hz(to), rate=Hz(200.0) / s(1.0), hold=ms(1.0))
    found = ramped(drive)
    middle = (np.arange(1, 13) - 0.5) * DT
    expected = start + np.sign(to - start) * 200.0 * np.clip(middle - 1e-3, 0.0, abs(to - start) / 200.0)
    print(f"\n{found}")
    assert np.allclose(found, expected, rtol=0.0, atol=1e-6 * start)
    assert np.allclose(commanded(drive, 12, DT), expected, rtol=0.0, atol=1e-12)
    assert found[0] == pytest.approx(start, abs=1e-6 * start) and found[-1] == pytest.approx(to, abs=1e-6 * start)


@pytest.mark.parametrize(("settings", "said"), [
    ({"rate": Hz(0.0) / s(1.0)}, "a rate above zero"),
    ({"rate": Hz(-1.0) / s(1.0)}, "a rate above zero"),
    ({"hold": s(-1.0)}, "not a time from 0 on")])
def test_a_ramp_needs_a_rate_above_zero_and_a_hold_from_zero(settings, said):
    drive = drives.Ramp(Hz(10.0), to=Hz(20.0), **{"rate": Hz(1.0) / s(1.0), **settings})
    with pytest.raises(BuildRefused) as refused:
        reproduction_arm(frequency=drive).build()
    assert refused.value.codes() == ["setting.value"] and said in refused.value.refusals[0].message


def test_a_ramps_rate_is_in_the_commands_units_per_second():
    with pytest.raises(BuildRefused) as refused:
        reproduction_arm(frequency=drives.Ramp(Hz(10.0), to=Hz(20.0), rate=Hz(1.0))).build()
    assert refused.value.codes() == ["setting.units"]


def test_the_turning_rate_the_build_reads_of_a_ramp_is_the_largest_it_reaches():
    """The post-run watch's turning rate, read from the field's drive: a ramp's is the largest it reaches."""
    exp = rotating_screw()
    held = cast(Any, exp)
    swapped = []
    for declaration in held._declarations:
        thing = getattr(declaration, "thing", None)
        if isinstance(thing, Apparatus) and thing.name == "field":
            ramp = drives.Ramp(Hz(5.0), to=Hz(8.0), rate=Hz(1.0) / s(1.0))
            declaration = ThingDeclaration(dataclasses.replace(thing, commands={**thing.commands, "frequency": ramp}))
        swapped.append(declaration)
    watch = next(w for w in held._declared(held._settings, swapped).plan().watches
                 if isinstance(w, WatchPlan) and w.turning_rate is not None)
    assert watch.turning_rate == pytest.approx(2.0 * np.pi * 8.0)
    assert watch.turning_from == "field.frequency is ramped to 8 Hz at the most, a rate of 2 pi x 8 rad/s"


def protocol_study(protocol: object, values=(Hz(20.0), Hz(30.0)), **sweep) -> Study:
    experiment = reproduction_arm()
    experiment.observe(PhaseLag("umr", field="field.magnetic", about=X))
    field = experiment.handle("field")
    return Study(experiment).sweep(grid({field.commands.frequency: list(values)}),  # type: ignore[union-attr]
                                   duration=s(0.02), protocol=protocol, **sweep)


def test_the_ramp_protocol_starts_each_points_ramp_at_its_value_in_one_compiled_program():
    protocol = Ramp("field.commands.frequency", to=Hz(5.0), rate=Hz(100.0) / s(1.0), hold=ms(5.0))
    study = protocol_study(protocol, record=[f"{LAG}.spin"])
    plan = study.plan()
    assert len(plan.groups) == 1 and plan.groups[0].compiles == 1
    assert "protocol: a ramp of field.commands.frequency" in plan.text(memory=False)
    starts = [point.ramp.start for point in plan.points if point.ramp is not None]
    assert starts == [Hz(20.0), Hz(30.0)]
    result = study.run()
    assert result.protocol == protocol and [drive.start for drive in result.ramps if drive] == starts
    values = [np.asarray(result.member(i).final["drive:field.frequency"]["value"]) for i in range(2)]
    assert values[0] == pytest.approx(20.0 - 100.0 * (0.02 - DT / 2 - 5e-3))
    assert values[1] == pytest.approx(30.0 - 100.0 * (0.02 - DT / 2 - 5e-3))
    crossings = ramp_crossings(result, LAG, window=ms(5.0))
    assert [c.quantity for c in crossings] == ["ramp down", "ramp down"]


def test_from_rest_is_the_default_and_adds_nothing_to_the_plan():
    study = protocol_study(None)
    plan = study.plan()
    assert isinstance(plan.protocol, FromRest) and "protocol" not in plan.text(memory=False)
    assert all(point.ramp is None for point in plan.points)


def test_each_reading_refuses_the_other_quantity():
    rest = protocol_study(FromRest()).run()
    with pytest.raises(ArgumentRefused) as refused:
        ramp_crossings(rest, LAG)
    assert refused.value.codes() == ["study.protocol-quantity"]
    ramp = protocol_study(Ramp("field.commands.frequency", to=Hz(5.0), rate=Hz(1.0) / s(1.0)),
                          record=[f"{LAG}.spin"]).run()
    with pytest.raises(ArgumentRefused) as refused:
        step_out(ramp, LAG, "field.commands.frequency")
    assert refused.value.codes() == ["study.protocol-quantity"]


def test_a_ramps_crossings_need_the_spin_at_every_step():
    result = protocol_study(Ramp("field.commands.frequency", to=Hz(5.0), rate=Hz(1.0) / s(1.0)),
                            record=[f"{LAG}.spin"], every=ms(1.0)).run()
    with pytest.raises(ArgumentRefused) as refused:
        ramp_crossings(result, LAG, window=ms(5.0))
    assert refused.value.codes() == ["study.series-every-step"]


@pytest.mark.parametrize("command", ["field.frequency", "field.commands.speed", "nothing.commands.frequency"])
def test_the_ramp_protocol_refuses_a_command_it_cannot_ramp(command):
    with pytest.raises((ArgumentRefused, BuildRefused)) as refused:
        protocol_study(Ramp(command, to=Hz(5.0), rate=Hz(1.0) / s(1.0))).plan()
    assert "study.protocol-command" in refused.value.codes()


def test_the_ramp_protocol_refuses_a_command_already_driven_by_a_ramp():
    experiment = reproduction_arm(frequency=drives.Ramp(Hz(10.0), to=Hz(20.0), rate=Hz(1.0) / s(1.0)))
    field = experiment.handle("field")
    study = Study(experiment).sweep(grid({field.commands.amplitude: [mT(1.0), mT(1.2)]}),  # type: ignore[union-attr]
                                    duration=s(0.01),
                                    protocol=Ramp("field.commands.frequency", to=Hz(5.0), rate=Hz(1.0) / s(1.0)))
    with pytest.raises(ArgumentRefused) as refused:
        study.plan()
    assert refused.value.codes() == ["study.protocol-command"]
    assert "gives it the drive Ramp" in refused.value.refusals[0].message
