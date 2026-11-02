"""A duration on the command line: ``10s``, ``500ms``, ``250us``, ``2min``, ``1h``, or a bare number, which is seconds.

The units are ``mime.units``' own (``s``, ``ms``, ``us``): their factors, taken from the units. A minute and an hour are
not units of ``mime.units`` (it holds SI's and the prefixed ones a model declares), so they are 60 s and 3600 s here,
and only here.

**A written number is read exactly and converted once.** ``700ms`` is the decimal 0.7 s, rounded to a float once, so it is
the very float ``--duration 0.7`` is, and the run folder's ``run.json`` records the same duration whichever way it was
written; multiplying the float 700 by the float 1e-3 rounds twice and gives 0.7000000000000001."""

from __future__ import annotations

import decimal
import re
from decimal import Decimal
from typing import cast

from mime.units import Unit, ms, s, us


def _factor(unit: Unit) -> Decimal:
    """The seconds in one of ``unit``, as the decimal the unit writes (``1e-3`` for ``ms``), not the float's long expansion."""
    return Decimal(repr(cast(float, unit(1.0).value)))


#: The suffixes and the seconds each stands for.
_SECONDS: dict[str, Decimal] = {"us": _factor(us), "ms": _factor(ms), "s": _factor(s), "min": Decimal(60),
                                "h": Decimal(3600)}
#: Exact over any number a person writes, and neither overflowing nor underflowing before the float does.
_EXACT = decimal.Context(prec=80, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN, traps=[])

#: A number and a suffix: ``10s``, ``2.5 min``, ``1e-3s``.
_WRITTEN = re.compile(r"(?P<number>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<unit>[A-Za-z]+)")

#: The forms, as a refusal and the help name them.
FORMS = "10s, 500ms, 250us, 2min, 1h, or a bare number, which is seconds"


def seconds(text: str) -> float | None:
    """``text`` as seconds, or ``None`` when it is not a duration. A bare number (anything ``float`` reads) is
    seconds, as it was before units."""
    try:
        return float(text)
    except ValueError:
        pass
    found = _WRITTEN.fullmatch(text.strip())
    if found is None:
        return None
    factor = _SECONDS.get(found["unit"])
    if factor is None:
        return None
    return float(_EXACT.multiply(Decimal(found["number"]), factor))
