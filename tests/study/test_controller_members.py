"""Members of an experiment whose controller is called on the host between chunks run one after another on the group's
one compiled step, each from its own initial state with its own law made from its own plan, and the plan says so. The
spun magnet's ``Ticker`` writes ``k`` at its tick ``k``, so a law carried from one member to the next would write other
values to the second: each member is compared with its own experiment's run, bit for bit (the same compiled scan, its
leaves passed as arguments), at the usable CPU and OpenBLAS thread counts the test runs at."""

from __future__ import annotations

import numpy as np
from control_cases import spun_magnet

from mime import Study
from mime.experiment import Experiment
from mime.study import grid
from mime.units import A_m2, ms

MOMENTS = [A_m2(m) for m in (14.0, 18.89, 22.0)]


def test_members_run_one_after_another_each_with_its_own_law():
    study = Study(spun_magnet()).sweep(grid({"spin.magnetic_moment": MOMENTS}), duration=ms(30))
    plan = study.plan()
    (group,) = plan.groups
    assert not group.batched and "one after another" in plan.text()
    result = study.run()
    for index, point in enumerate(plan.points):
        own = Experiment.declared_from(point.declared.settings, point.declared.declarations).build().run(ms(30))
        member = result.member(index)
        assert all(np.array_equal(own.final[n][f], member.final[n][f]) for n in own.final for f in own.final[n]), index
    angles = {round(float(result.member(i).final["spin.motor"]["angle"]), 9) for i in range(3)}
    assert len(angles) == 1, angles  # every member's law wrote the same ticks: none continued another's


def test_the_control_a_law_carried_over_writes_other_ticks():
    """Two runs of one build, the second continuing the first's law: its writes differ from a fresh member's."""
    built = spun_magnet().build()
    first = built.run(ms(30))
    second = built.run(ms(30))
    assert not np.array_equal(first.final["spin.motor"]["angle"], second.final["spin.motor"]["angle"])
