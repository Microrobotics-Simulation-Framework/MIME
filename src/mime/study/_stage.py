"""A study's stage samples: the bound fields each member keeps for the ghosted comparison, and the one time base they
share.

``stage_every`` is a time that is a whole number of every member's base step: the members of a study may run at other
steps (a ``dt`` axis), and one overlay layer has one ``timeCodesPerSecond``, so its sample index must mean the same
simulated time in every member. A value that is not is refused with the nearest one that is."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.results._stage import StageRecord, sampled
from mime.units import TIME, Quantity, si

if TYPE_CHECKING:
    from mime.experiment import BuiltExperiment


def _lcm(a: Fraction, b: Fraction) -> Fraction:
    """The least common multiple of two positive rationals: the smallest time that is a whole number of each."""
    return Fraction(math.lcm(a.numerator, b.numerator), math.gcd(a.denominator, b.denominator))


def common_step(steps: Sequence[float]) -> float:
    """The smallest time that is a whole number of every base step of ``steps`` (s)."""
    held = [Fraction(step).limit_denominator(10**12) for step in steps]
    out = held[0]
    for each in held[1:]:
        out = _lcm(out, each)
    return float(out)


def check_stage_every(stage_every: Quantity | None, steps: Sequence[float]) -> None:
    """Refuses (``study.stage-every``) a ``stage_every`` that is not a whole number of every base step in ``steps`` (the
    members' ``dt``), naming the nearest one that is."""
    if stage_every is None:
        return
    seconds = float(cast(float, si(stage_every, TIME, what="stage_every=")))
    if not steps:
        return
    if all(seconds >= dt * (1 - 1e-9) and math.isclose(seconds / dt, round(seconds / dt), rel_tol=1e-9) for dt in steps):
        return
    unit = common_step(steps)
    nearest = max(1, round(seconds / unit)) * unit
    raise ArgumentRefused([Refusal(
        "study.stage-every", RefusalClass.USER,
        f"stage_every= is {seconds!r} s, which is not a whole number of every member's base step "
        f"({', '.join(sorted({f'{dt!r}' for dt in steps}))} s): the overlay's samples are on one time base, and a sample "
        f"index must be the same simulated time in every member",
        f"use the nearest time that is: stage_every=s({nearest!r})",
        why="the overlay layer has one timeCodesPerSecond")])


def stage_meta(built: BuiltExperiment) -> dict[str, Any]:
    """What places a member's samples on a stage: the binding's entries, the stage's up axis and metres per unit, the
    base step, the experiment's name and its input hash."""
    settings = built.recording_settings()
    return {"binding": built.binding().record()["entries"], "up": list(settings.up),
            "meters_per_unit": settings.meters_per_unit, "dt": settings.dt, "experiment": settings.experiment,
            "input_hash": settings.input_hash}


def stage_record(built: BuiltExperiment, initial: Mapping[str, Mapping[str, Any]], kept: Mapping[str, Mapping[str, Any]],
                 steps: int, every: int) -> StageRecord:
    """One member's samples: the bound fields at step 0 (its initial state) and every ``every`` steps of ``kept`` (its
    bound fields at every step)."""
    fields = built.binding().fields()
    taken, values = sampled(initial, kept, fields, steps, every)
    return StageRecord(taken, values, stage_meta(built))


__all__ = ["check_stage_every", "common_step", "stage_meta", "stage_record"]
