"""An OpenMaterica record's table, as MIME's table evaluator (``mime.core.tables``) reads it during a step.

The record's table form (its record format's section 6) maps member for member onto the evaluator's one-axis table:
``axis`` and ``axis_unit`` onto the axis's name and unit, ``x``, ``edges`` and ``source_range`` as they are, ``y`` as
the values, ``scheme`` (``linear`` or ``loglog``) as the scheme. The record's own checks are OpenMaterica's, which
read it first; the evaluator's description checks the form again when it is made, so a table that breaks it is
refused here too. The experiment's extrapolation of the field (``Record.extrapolation``) becomes the evaluator's when
it is ``constant``; a ``linear`` one is the consumer's law, computed beside the evaluator from the value at the
readable range's nearest end, and waits for ``MaterialAt``'s first consumer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from mime.core.tables import Axis, ConstantExtrapolation, Scheme, Side, Table, TableInvalid
from mime.materials._records import Extrapolation

#: The members of a 0.1.0 table that OpenMaterica's writer may write.
_MEMBERS = frozenset({"axis", "axis_unit", "x", "y", "scheme", "edges", "source_range"})


def evaluator_table(table: Mapping[str, object], *, side: Side = "right",
                    extrapolation: Extrapolation | None = None) -> Table:
    """The evaluator's table for a record's ``table`` (as OpenMaterica's read returns it), its derivative at an
    interior knot taken from ``side``, under the experiment's ``extrapolation`` of the field, if any. Refuses
    (:class:`~mime.core.tables.TableInvalid`) a table holding ``points``, which OpenMaterica 0.1.0 refuses too
    (``points_unsupported``), one whose axis unit is not stated, one that breaks the form, and a ``linear``
    extrapolation."""
    if "points" in table:
        raise TableInvalid("the table holds points, which a 0.1.0 reader refuses (points_unsupported)")
    unknown = set(table) - _MEMBERS
    if unknown:
        raise TableInvalid(f"the table holds members its form does not name: {sorted(unknown)}")
    if "edges" not in table:
        raise TableInvalid("a table's edges are required, an empty list where it has none")
    scheme = table.get("scheme")
    if scheme not in ("linear", "loglog"):
        raise TableInvalid(f"a record's table is linear or loglog, and this one's scheme is {scheme!r}")
    held_range = table.get("source_range")
    axis = Axis(name=cast(str, table.get("axis")), unit=cast(str, table.get("axis_unit")),
                x=tuple(cast(list[float], table.get("x", ()))), edges=tuple(cast(list[float], table.get("edges", ()))),
                source_range=None if held_range is None else cast(tuple[float, float],
                                                                  tuple(cast(list[float], held_range))))
    if extrapolation is not None and extrapolation.method != "constant":
        raise TableInvalid(f"the evaluator extrapolates by a constant only; a {extrapolation.method} extrapolation "
                           f"is the consumer's law, computed beside it from the end value, which waits for "
                           f"MaterialAt's first consumer")
    return Table(axes=(axis,), y=table.get("y"), scheme=cast(Scheme, scheme), side=side,
                 extrapolation=None if extrapolation is None else ConstantExtrapolation(extrapolation.basis))


__all__ = ["evaluator_table"]
