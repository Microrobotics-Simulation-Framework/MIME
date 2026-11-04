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
