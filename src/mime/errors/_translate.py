"""Errors from below, translated into MIME's terms, and the one rule for the rest: an error MIME raised without a
code is a gap in MIME's codes (``error.uncoded``, to report), and any other is unrecognised, with the script's own line
where there is one. Each keeps the class of the error it translates (:func:`standing_for`), so that a handler written
for that error still catches it."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from types import TracebackType

from mime.build._refusals import (
    ISSUES,
    ArgumentRefused,
    InternalError,
    MimeError,
    Refusal,
    RefusalClass,
    RunFailed,
    UnrecognisedError,
)
from mime.errors._frames import Place, raised_by_mime, script_places

#: The note JAX appends to an error whose frames it filtered: it names JAX's own switch, which MIME's short traceback
#: replaces with its own.
_JAX_NOTE = re.compile(r"\n*-{10,}\n*For simplicity, JAX has removed its internal frames.*", re.DOTALL)
#: How many lines of an error's own message a translation quotes.
QUOTED_LINES = 3


def first_lines(error: BaseException, count: int = QUOTED_LINES) -> str:
    """``error``'s type and the first lines of its message, without the note JAX appends."""
    text = _JAX_NOTE.sub("", str(error)).strip()
    held = [line for line in text.splitlines() if line.strip()][:count]
    return f"{type(error).__name__}: {' '.join(line.strip() for line in held)}" if held else type(error).__name__


def script_place(error: BaseException) -> Place | None:
    """The innermost of the script's own frames the error passed through: the line it was raised at, or the call
    into the library that raised it."""
    places = script_places(error.__traceback__)
    return places[-1] if places else None


def raised_at(error: BaseException) -> str | None:
    """The script's own line ``error`` was raised at, or passed through last, as ``file, line N``: a lab's entry's own
    line, for an error its entry raised inside MIME."""
    place = script_place(error)
    return place.text() if place is not None else None


def place_of(traceback: TracebackType | None) -> str | None:
    places = script_places(traceback)
    return places[-1].text() if places else None


#: The error types JAX raises when Python asks a traced value for something only a concrete one has.
_TRACER_ERRORS = ("TracerBoolConversionError", "ConcretizationTypeError", "TracerIntegerConversionError",
                  "TracerArrayConversionError")


def traced_value_error(error: BaseException) -> bool:
    """Whether ``error`` is JAX's refusal of a Python branch, ``int()``, ``float()`` or NumPy call on a traced value."""
    import jax

    held = tuple(getattr(jax.errors, name) for name in _TRACER_ERRORS if hasattr(jax.errors, name))
    return isinstance(error, held)


_OUT_OF_MEMORY = re.compile(r"RESOURCE_EXHAUSTED|out of memory|Out of memory|OOM when allocating")


def _jaxs(error: BaseException) -> bool:
    """Whether ``error``'s class is JAX's or XLA's: a device's error reaches Python as one of theirs."""
    return any(cls.__module__.split(".", 1)[0] in ("jax", "jaxlib") for cls in type(error).__mro__)


def out_of_memory(error: BaseException) -> MimeError | None:
    """A device out of memory, as XLA reports it. The host's own ``MemoryError`` (an array of the script's too large
    for the machine) is not the device's, and keeps its class as an unrecognised error."""
    if isinstance(error, MemoryError) or not (_jaxs(error) and _OUT_OF_MEMORY.search(str(error))):
        return None
    place = script_place(error)
    return RunFailed([Refusal(
        "run.out-of-memory", RefusalClass.ENVIRONMENT, f"the device ran out of memory ({first_lines(error, 1)})",
        "keep less: name fewer series in run(..., record=[...]), drop history=True, run a shorter duration per "
        "run() call, or give a boundary-element drag a coarser spacing=; or run on a device with more memory",
        why="a run holds the whole state of every recorded step, and a dense solve's matrix, in the device's memory",
        where=place.text() if place is not None else None)])


def _unit_function(dimension: object) -> str | None:
    """The name of the SI unit of ``dimension`` in ``mime.units`` (``s`` for a time), where it has one."""
    from mime import units

    held = (name for name, value in vars(units).items() if isinstance(value, units.Unit) and value.factor == 1.0
            and value.dimension == dimension and not name.startswith("_"))
    return next(held, None)


def unit_error(error: BaseException) -> MimeError | None:
    """A value given without its units, or in units of another dimension, where a call takes a quantity: the user's,
    with the value written in its unit."""
    from mime.units import UnitError

    if not isinstance(error, UnitError):
        return None
    dimension, given = getattr(error, "dimension", None), getattr(error, "given", None)
    unit = _unit_function(dimension) if dimension is not None else None
    if unit is not None and isinstance(given, (int, float)) and not isinstance(given, bool):
        action = f"write it with its unit: {unit}({given!r}), with `from mime.units import {unit}`"
    elif unit is not None:
        action = f"write it as a quantity of {dimension}: {unit}(...), with `from mime.units import {unit}`"
    else:
        action = "write it as a quantity in its unit, from mime.units: ms(2), um(50), mT(1.2)"
    return ArgumentRefused([Refusal(
        "call.units", RefusalClass.USER, first_lines(error, 2).removeprefix("UnitError: "), action,
        why="MIME takes every value with its units and never assumes them", where=raised_at(error))])


#: The translations tried in order on an error from below; the first to answer gives the MIME error.
TRANSLATIONS: Sequence[Callable[[BaseException], MimeError | None]] = (unit_error, out_of_memory)

#: The footer of an error MIME raised with no code: its message is MIME's own, and may say what to change.
UNCODED = "MIME gave this error no code yet: report it"


def translated(error: BaseException) -> MimeError:
    """``error``, which reached a public call with no code, as a MIME error: a known error from below in MIME's terms;
    else, raised by MIME itself, a gap in MIME's codes (``error.uncoded``); else unrecognised (``error.unrecognised``),
    never called MIME's defect, at the script's own line where it passed through one. A warning raised as an error
    is the script's filter's, and unrecognised."""
    for translation in TRANSLATIONS:
        found = translation(error)
        if found is not None:
            return found
    where = raised_at(error)
    if raised_by_mime(error) and not isinstance(error, Warning):
        return InternalError([Refusal(
            "error.uncoded", RefusalClass.INTERNAL, f"MIME raised an error with no code: {first_lines(error)}",
            f"read its message, which may say what to change; then report it at {ISSUES}, so that it gets a code",
            why="every error MIME gives a script has a code, so one without is a gap in MIME's errors, not a fault "
                "of yours or of the run",
            where=where)])
    return UnrecognisedError([Refusal(
        "error.unrecognised", RefusalClass.UNRECOGNISED, first_lines(error),
        "read it at the line of yours it names; to see every frame below it, run again with MIME_TRACEBACK=full",
        why="it was raised below MIME (by your code, JAX, MADDENING or another library), and MIME has no "
            "translation for it",
        where=where)])


__all__ = ["QUOTED_LINES", "TRANSLATIONS", "UNCODED", "first_lines", "out_of_memory", "place_of", "raised_at",
           "script_place", "traced_value_error", "translated", "unit_error"]
