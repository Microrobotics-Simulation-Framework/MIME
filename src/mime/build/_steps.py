"""The rule a time holds to be a run's length: a whole number of base steps, at least one. A run's ``steps_in`` and the
build's check of a declared duration both read it here, so the two cannot disagree about which lengths are runs."""

from __future__ import annotations

import math


def whole_steps(seconds: float | tuple[float, ...], dt: float) -> int | None:
    """The number of base steps ``seconds`` is, where it is a whole number of at least one (within a relative 1e-9, the
    rounding of a duration written as a decimal), else ``None``: a time that is not one number, not finite or not
    above zero is not a run's length."""
    if isinstance(seconds, tuple):
        return None
    ratio = seconds / dt
    steps = round(ratio) if math.isfinite(ratio) else 0
    return steps if steps >= 1 and math.isclose(ratio, steps, rel_tol=1e-9) else None


def whole_runs_either_side(seconds: float | tuple[float, ...], dt: float) -> tuple[float, float]:
    """The lengths, in seconds, of the runs of whole steps just below and just above ``seconds`` (a run of one step at
    least), the two that a refusal offers."""
    ratio = math.nan if isinstance(seconds, tuple) else seconds / dt
    return (max(1, math.floor(ratio)) * dt, max(1, math.ceil(ratio)) * dt) if math.isfinite(ratio) else (dt, dt)


#: How many times the bound's steps a period ``dt="auto"`` searches for a step that also divides a declared duration.
_AUTO_SEARCH = 64


def auto_step(largest: float, period: float | None, duration: float | None) -> float:
    """``dt="auto"``'s base step: the largest at most ``largest`` that divides ``period`` (a periodic drive's) and a
    declared ``duration`` into whole steps; where none within the search divides both, the period's own (and the
    duration's refusal names the step)."""
    spans = [span for span in (period, duration) if span]
    if not spans:
        return largest
    first = spans[0]
    least = max(1, math.ceil(first / largest - 1e-12))
    if len(spans) == 1:
        return first / least
    for count in range(least, _AUTO_SEARCH * least + 1):
        if whole_steps(spans[1], first / count) is not None:
            return first / count
    return first / least
