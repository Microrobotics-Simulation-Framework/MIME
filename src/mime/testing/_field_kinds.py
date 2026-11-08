"""SF-12, the solved-field contract's conformance suite: the cases every field kind a medium carries passes, run on
its declarations (``mime.fields.SolvedField``) and a small rig of it, **each case applicable by what the kind
declares** and every case it skips listed with the declaration that excluded it, never a silent subset (F0's SF-12;
round 4, item 10; R4-B-09 option 0).

The cases (``FieldCase``): its **declarations** (SF-1 to SF-11 written, "not declared" only on the three open rows,
what it provides and accepts in the vocabulary and on the kind); a **reading** at points, with its quantity's shape and
units (against an oracle in SI); **outside** (a kind that declares it takes no delivery takes none: its node's inputs
are its drives' ports); its **ledger** (route 2: the step identity to the ledger's floor; route 1: every step
dissipated); **steady** (a declared steady solve is the time-stepped fixed point: a step from it moves it by no more
than the solve's tolerance); the **affine** response (none is built before F2b). A kind that declares a capability
whose case its rig does not give fails, as a planted absent case must. **The standing check as a test**: nothing here
reads a solver, a projector or a fluid's operator; a non-fluid kind passes it with its own rig."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from mime.core import ledger as books
from mime.fields._solved import SolvedField
from mime.vocabulary import lookup


class FieldCase(StrEnum):
    DECLARATIONS = "declarations"
    READING = "reading"
    OUTSIDE = "outside"
    LEDGER = "ledger"
    STEADY = "steady"
    AFFINE = "affine"


@dataclass(frozen=True)
class FieldRig:
    """What the suite needs of one kind, built small by its author: ``node`` (the kind's node), ``params`` and ``state``
    (its step's constants and a state a few steps in), ``reading(params, state, points) -> (n, *shape)`` for each
    quantity it provides, by interface name; ``points`` (m) and ``expected`` (the reading there from an independent
    oracle, SI) within ``tolerance`` of the oracle's scale; ``metered()``, a run's final state with the ledger attached
    (for a kind with a ledger route); ``steady()``, the declared steady solve's move under one time step from it over
    its scale, with its line (for a kind declaring one)."""

    node: Any
    params: Mapping[str, Any]
    state: Mapping[str, Any]
    reading: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any], Any], Any]]
    points: Any
    expected: Mapping[str, Any]
    tolerance: float
    metered: Callable[[], Mapping[str, Any]] | None = None
    steady: Callable[[], tuple[float, float]] | None = None


@dataclass(frozen=True)
class CaseResult:
    case: FieldCase
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class FieldConformance:
    """The suite's run on one kind: each case run, and each skipped with its reason."""

    kind: str
    results: tuple[CaseResult, ...]
    skipped: Mapping[FieldCase, str] = field(default_factory=dict[FieldCase, str])

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def failures(self) -> list[str]:
        return [f"{r.case}: {r.detail}" for r in self.results if not r.passed]


def _declarations(kind: Any, declared: SolvedField, dtype: str) -> CaseResult:
    found = list(declared.problems())
    for name in declared.provides:
        if lookup(name) is None:
            found.append(f"it provides {name!r}, which the vocabulary does not name")
    accepts = set(getattr(type(kind), "accepts", {}) or {})
    missing = [name for name in declared.accepts if name not in accepts]
    if missing:
        found.append(f"it declares it accepts {missing}, which the kind does not (accepts = {sorted(accepts)})")
    if dtype not in declared.dtypes:
        found.append(f"it was run in {dtype}, which it does not declare")
    return CaseResult(FieldCase.DECLARATIONS, not found, "; ".join(found))


def _reading(declared: SolvedField, rig: FieldRig) -> CaseResult:
    found: list[str] = []
    points = np.asarray(rig.points)
    for name in declared.provides:
        if name not in rig.reading:
            found.append(f"it provides {name} and its rig gives no reading of it")
            continue
        interface = lookup(name)
        if interface is None:
            found.append(f"it provides {name!r}, which the vocabulary does not name")
            continue
        value = np.asarray(rig.reading[name](rig.params, rig.state, rig.points), np.float64)
        shape = (points.shape[0], *(interface.shape or ()))
        if value.shape != shape:
            found.append(f"{name} read at {points.shape[0]} points has the shape {value.shape}, not {shape}")
            continue
        if not np.all(np.isfinite(value)):
            found.append(f"{name}'s reading is not finite")
            continue
        expected = np.asarray(rig.expected[name], np.float64)
        scale = float(np.max(np.abs(expected)))
        error = float(np.max(np.abs(value - expected)))
        if not error <= rig.tolerance * scale:
            found.append(f"{name}'s reading is off its oracle in SI ({interface.units}) by {error:.3g}, over "
                         f"{rig.tolerance:g} of its scale {scale:.3g}: the wrong units or the wrong field")
    return CaseResult(FieldCase.READING, not found, "; ".join(found))


def _outside(kind: Any, rig: FieldRig) -> CaseResult:
    ports = {port.field for port in (getattr(type(kind), "accepts", {}) or {}).values()}
    inputs = set(rig.node.boundary_input_spec())
    extra = sorted(inputs - ports)
    return CaseResult(FieldCase.OUTSIDE, not extra,
                      f"it declares it is outside, and its node takes {extra}, which are not its drives' ports" if extra
                      else "")


def _ledger(declared: SolvedField, rig: FieldRig, dtype: str) -> CaseResult:
    if rig.metered is None:
        return CaseResult(FieldCase.LEDGER, False, f"it declares ledger route {declared.ledger_route} and its rig gives "
                                                   f"no metered run")
    state = rig.metered()
    r = books.RECORDS
    eps = float(np.finfo(dtype).eps)
    if declared.ledger_route == 2:
        worst = float(np.asarray(state[f"{r}identity_worst"]))
        ok = worst <= books.IDENTITY_ULPS * eps and float(np.asarray(state[f"{r}unattributed_worst"])) == 0.0
        return CaseResult(FieldCase.LEDGER, ok, "" if ok else f"its step identity's worst step is {worst:.3g} of its "
                                                               f"scale, over {books.IDENTITY_ULPS} eps")
    if f"{r}route_one_negative_steps" not in state:
        return CaseResult(FieldCase.LEDGER, False, "it declares route 1 and its records hold no route-1 check")
    negative = float(np.asarray(state[f"{r}route_one_negative_steps"]))
    worst = float(np.asarray(state[f"{r}route_one_worst"]))
    ok = negative == 0.0
    return CaseResult(FieldCase.LEDGER, ok, "" if ok else f"{negative:g} steps gave energy (route 1's dissipation at "
                                                           f"worst {worst:.3g} of its scale)")


def _steady(rig: FieldRig) -> CaseResult:
    if rig.steady is None:
        return CaseResult(FieldCase.STEADY, False, "it declares a steady solve and its rig gives no steady case")
    moved, line = rig.steady()
    return CaseResult(FieldCase.STEADY, moved <= line, "" if moved <= line else
                      f"a step from its steady solution moves it by {moved:.3g} of its scale, over {line:g}")


def field_conformance(kind: Any, declared: SolvedField, rig: FieldRig, *, dtype: str = "float64",
                      cases: Sequence[FieldCase] | None = None) -> FieldConformance:
    """SF-12 on ``kind`` (an instance of a field kind's entry), its ``declared`` contract and its ``rig``: each case
    its declarations make applicable, or each of ``cases`` (one not applicable is skipped with its reason; one left
    out of ``cases`` is listed as left out)."""
    chosen = tuple(FieldCase) if cases is None else tuple(cases)
    skipped: dict[FieldCase, str] = {c: "left out of the cases asked for" for c in FieldCase if c not in chosen}
    results: list[CaseResult] = []
    for case in chosen:
        if case is FieldCase.DECLARATIONS:
            results.append(_declarations(kind, declared, dtype))
        elif case is FieldCase.READING:
            results.append(_reading(declared, rig))
        elif case is FieldCase.OUTSIDE:
            if declared.outside:
                results.append(_outside(kind, rig))
            else:
                skipped[case] = "it takes deliveries (not outside); its exchange law is C5's"
        elif case is FieldCase.LEDGER:
            if declared.ledger_route is None:
                skipped[case] = f"no ledger route: {declared.ledger_reason}"
            else:
                results.append(_ledger(declared, rig, dtype))
        elif case is FieldCase.STEADY:
            if declared.steady_solve:
                results.append(_steady(rig))
            else:
                skipped[case] = "it declares no steady solve"
        elif case is FieldCase.AFFINE:
            if declared.affine_response:
                results.append(CaseResult(case, False, "it declares an affine response, and no affine case is built "
                                                       "before F2b"))
            else:
                skipped[case] = "it declares no affine response (none is built before F2b)"
    return FieldConformance(type(kind).__name__, tuple(results), skipped)


__all__ = ["CaseResult", "FieldCase", "FieldConformance", "FieldRig", "field_conformance"]
