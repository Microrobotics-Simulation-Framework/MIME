"""The aliasing warning: a recording whose samples are as far apart as half the period of a declared drive's frequency
(or farther) cannot show that frequency, and shows a slower motion in its place. The run log's 10 ms recording of a
rocking body did exactly this near a 50 Hz drive. A warning and never a refusal: a recording of a slow quantity under a
fast drive (a mean drift) is a sound one, and the drive's rate is a declaration, not a measurement of what the recorded
quantity does. In-graph outputs (``PhaseLag``, the readings at the end) are computed every step and do not alias."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from mime.errors._frames import warn_at_script

#: How close to half the period a spacing may be and still count as at it (a spacing written as ``ms(10)`` against a
#: 50 Hz drive is half the period to the last bit or one off).
AT_HALF_PERIOD = 1e-9


def aliasing_text(rates: Iterable[tuple[float, str]], spacing: float, what: str) -> str | None:
    """The warning for ``what`` sampled every ``spacing`` seconds under the drives ``rates`` (rad/s, with what each was
    computed from), or ``None`` where every drive is sampled more than twice a period. The fastest drive is named.
    Sampling at exactly twice the frequency is not enough to show the waveform (a sine sampled at its zeros reads
    zero), so a spacing equal to half the period is warned about."""
    fastest = max(rates, default=None)
    if fastest is None or not spacing > 0.0 or not math.isfinite(spacing):
        return None
    hertz = fastest[0] / (2.0 * math.pi)
    half_period = 0.5 / hertz
    if not hertz > 0.0 or spacing < half_period * (1.0 - AT_HALF_PERIOD):
        return None
    return (f"{what} is one sample every {spacing:g} s, and {fastest[1]}: a frequency of {hertz:g} Hz, whose period "
            f"{1.0 / hertz:g} s is no more than twice that spacing. Samples this far apart cannot show that frequency, "
            f"and show a slower motion in its place (aliasing). Record at most {half_period / 2.0:g} s apart "
            f"(every=s({half_period / 2.0:g}) or less), or read a quantity the graph reduces every step. A warning, "
            f"not a refusal: a quantity that does not move at the drive's frequency is sampled soundly at any spacing")


def warn_aliasing(rates: Iterable[tuple[float, str]], spacing_steps: int, dt: float, what: str) -> None:
    """Warn (at the script's line) if ``what``, sampled every ``spacing_steps`` base steps of ``dt``, is aliased."""
    message = aliasing_text(rates, spacing_steps * dt, what)
    if message is not None:
        warn_at_script(message, UserWarning)


def largest(per_point: Sequence[Iterable[tuple[float, str]]]) -> list[tuple[float, str]]:
    """The rates of every point of a study, flat: the fastest of them is the one that is warned about."""
    return [rate for rates in per_point for rate in rates]


__all__ = ["AT_HALF_PERIOD", "aliasing_text", "largest", "warn_aliasing"]
