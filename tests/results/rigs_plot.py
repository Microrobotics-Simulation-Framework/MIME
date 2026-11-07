"""Hand-built results for the plot and comparison tests: runs from arrays with no graph (the speed set by the test), a
study of them, and the same data as a measured result."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mime.registry import OutputKind, Reduction
from mime.results import FlagLayout, FlagTrack, Layout, ObservationLayout, OutputLayout, RunResult, Samples, StudyResult
from mime.results._result import MemberOf
from mime.units import DIMENSIONLESS, FREQUENCY, VELOCITY, Hz

SPEED = OutputLayout("probe:p", "speed", "probe:p", "speed", OutputKind.PER_STEP, VELOCITY, (), "a speed")
MEAN = OutputLayout("probe:p", "mean", "probe:p", "mean", OutputKind.AT_END, VELOCITY, (), "its mean",
                    reduces="speed", rule=Reduction.MEAN, after=None)
LAG = OutputLayout("phase_lag:p", "lag", "phase_lag:p", "lag", OutputKind.PER_STEP, DIMENSIONLESS, (),
                   "the angle from the field to the body's moment")
FIRED = FlagLayout("law:p", "law 0.1.0", "law:p", "fired", "the law left its range")
LIMITED = FlagLayout("law:p", "law 0.1.0", "probe:p", "speed", "the speed left its validity range", "speed_limit", (2.5,))


def run_of(speeds: Sequence[float], dt: float, *, raised: Sequence[bool] | None = None, limited: bool = False,
           dtype: type = np.float64, lag: Sequence[float] | None = None, at: float | None = None) -> RunResult:
    """A run of ``len(speeds)`` steps of ``dt`` whose recorded speed is ``speeds`` (one per step), its flag raised where
    ``raised`` says; ``limited`` adds a flag with a numeric limit on the speed's own field, ``lag`` records a ``PhaseLag``'s lag (rad) beside
    it, and ``at`` makes the run a member of a study at that frequency (Hz): the coordinates the comparison reads."""
    speed = np.asarray(speeds, dtype=dtype)
    steps = np.arange(1, len(speed) + 1, dtype=np.int64)
    fired = np.zeros(len(speed), dtype=bool) if raised is None else np.asarray(raised, dtype=bool)
    flags = (FIRED, LIMITED) if limited else (FIRED,)
    observations = (ObservationLayout("probe:p", "probe 0.1.0", (SPEED, MEAN)),
                    *((ObservationLayout("phase_lag:p", "phase_lag 0.1.0", (LAG,)),) if lag is not None else ()))
    layout = Layout("synthetic", dt, observations, flags)
    final = {"probe:p": {"speed": speed[-1:].reshape(()), "mean": np.asarray(speed.mean(), dtype=dtype)},
             "law:p": {"fired": np.asarray(float(fired[-1]), dtype=dtype)}}
    tracks = {FIRED.path: FlagTrack(fired, bool(fired[-1]))}
    if limited:
        tracks[LIMITED.path] = FlagTrack(np.abs(speed) > 2.5, bool(abs(speed[-1]) > 2.5))
    recorded = {SPEED.path: Samples(steps, speed)}
    if lag is not None:
        final["phase_lag:p"] = {"lag": np.asarray(lag[-1], dtype=dtype)}
        recorded[LAG.path] = Samples(steps, np.asarray(lag, dtype=dtype))
    run = RunResult(len(speed), final, layout=layout, recorded=recorded, tracks=tracks, every=1)
    if at is not None:
        run.member_of = MemberOf("synthetic", 0, 0, (("field.commands.frequency", Hz(at)),), None, {})
    return run


def study_of(runs: Sequence[RunResult], frequencies: Sequence[float]) -> StudyResult:
    return StudyResult("synthetic", ["field.commands.frequency"], {"field.commands.frequency": [Hz(f) for f in frequencies]},
                       list(runs), [0] * len(runs), list(range(len(runs))))


__all__ = ["FREQUENCY", "run_of", "study_of"]
