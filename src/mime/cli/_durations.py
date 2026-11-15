"""A duration on the command line: ``10s``, ``500ms``, ``250us``, ``2min`` or ``1h``. A bare number is refused, as the
Python API refuses one (``--duration 3`` once ran three seconds, 1000 times what ``3ms`` runs): units are never
assumed.

The units are ``mime.units``' own (``s``, ``ms``, ``us``): their factors, taken from the units. A minute and an hour are
not units of ``mime.units`` (it holds SI's and the prefixed ones a model declares), so they are 60 s and 3600 s here,
and only here.

**A written number is read exactly and converted once.** ``700ms`` is the decimal 0.7 s, rounded to a float once, so it is
the very float ``0.7s`` is, and the run folder's ``run.json`` records the same duration whichever way it was written;
multiplying the float 700 by the float 1e-3 rounds twice and gives 0.7000000000000001."""

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

_PLAIN_NUMBER = re.compile(r"\d+\.?\d*|\.\d+")

#: A number and a suffix: ``10s``, ``2.5 min``, ``1e-3s``.
_WRITTEN = re.compile(r"(?P<number>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<unit>[A-Za-z]+)")

#: The forms, as a refusal and the help name them.
FORMS = "10s, 500ms, 250us, 2min, 1h: a number and its unit as one word (3s), or in quotes (\"3 s\")"


def is_bare_number(text: str) -> bool:
    """Whether ``text`` is a number with no unit (anything ``float`` reads, ``inf`` and ``nan`` included)."""
    try:
        float(text)
    except ValueError:
        return False
    return True


def unit_examples(text: str) -> str:
    """The bare number ``text`` written with a unit two ways, as a refusal shows it (``3s for 3 seconds, 3ms for 3
    milliseconds``); two forms of the stock number where it is not a plain one (``inf``)."""
    number = text.strip()
    if _PLAIN_NUMBER.fullmatch(number) is None:
        number = "10"
    return f"{number}s for {number} seconds, {number}ms for {number} milliseconds"


def seconds(text: str) -> float | None:
    """``text`` as seconds, or ``None`` when it is not a duration: a bare number is not one."""
    if is_bare_number(text):
        return None
    found = _WRITTEN.fullmatch(text.strip())
    if found is None:
        return None
    factor = _SECONDS.get(found["unit"])
    if factor is None:
        return None
    return float(_EXACT.multiply(Decimal(found["number"]), factor))
