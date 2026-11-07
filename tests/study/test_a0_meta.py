"""A0 with its controller inside the compiled step at its own rate, a third of the base rate: ``_meta`` holds the
multi-rate counter that decides the controller's ticks, so each member must carry its own, seeded from the built graph's
and carried through every chunk. Two members over the motor's velocity, 30 ms in chunks of two steps, each against its
own run within gate 1's tolerance (float32). Its builds and compiles take most of its time (on the allowlist)."""

from __future__ import annotations

import math

from compare import TOLERANCE_EPS, misses, worst
from script_a0 import a0

from mime import Study
from mime.study import grid
from mime.units import ms, rad_per_s

COMMAND = "rpm.commands.motor.velocity"


def test_each_member_carries_its_own_multi_rate_counter():
    speeds = [rad_per_s(2 * math.pi * f) for f in (3.0, 3.5)]
    study = Study(a0(rate=ms(1.5))).sweep(grid({COMMAND: speeds}), duration=ms(30))
    plan = study.plan()
    assert plan.groups[0].batched and plan.groups[0].chunks == (2,) * 30
    result = study.run()
    reference = a0(rate=ms(1.5)).build().run(ms(30), history=True, override={COMMAND: speeds[1]})
    where, miss = worst(misses(reference, result.member(1)))
    assert miss <= TOLERANCE_EPS["float32"], (where, miss)
