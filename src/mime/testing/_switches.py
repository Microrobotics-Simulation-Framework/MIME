"""The switch harness: does each switch measure as its declared order, at the outputs of whatever evaluates it?

A row places a point a quarter of a step ``delta`` from its owner's switch (in the indicator's units, on the side it
names), forces the owner's branch each way there (``mime.core.switch.forcing``, in a step traced afresh), and takes
the **forced-branch difference** of each output group: the largest change of any output between the two branches. It
does so at ``delta`` and at ``delta / 10``. A kink's difference shrinks with the step, about tenfold; a jump's does
not. Each output is judged on its own, so a jump in one is not hidden by a larger kink in another: an output whose
difference at ``delta / 10`` is at most the kink limit times its difference at ``delta``, or below the rounding floor,
measures continuous, and a group measures continuous when all of its outputs do.

A row passes when every group it declares an order of 0 or more for measures continuous (an order of 1 or more is
checked as at least 0: the harness tells a jump from anything continuous), every group it declares nothing for is
unchanged by the branch, the owner's indicator at each placed point is that point's offset (so a sign or a units
error fails), and the forced setting reached the owner in every evaluation (a step reused from a trace made before
the setting opened reaches nothing). A declared jump always passes; one measured continuous is reported.

**The rounding floor** is judged on the inputs' scale, not on the difference's: ``FLOOR_ULPS`` units in the last place
of the largest output of the group in either branch.

**Domain guards.** Where a guard's condition is false its fallback is discarded, so replacing every fallback with
other finite values (twice and half the original, ``replacing_fallbacks``) must change no output and no derivative
(:func:`fallback_problems`)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import jax
import numpy as np

from mime.core import switch

#: The rounding floor, in units in the last place of a group's largest output.
FLOOR_ULPS = 64
#: How far an indicator may be from its placed offset, relative to the offset: far above its rounding, far below a
#: sign's or a unit's error.
INDICATOR_TOLERANCE = 1e-2
#: The two steps a row is measured at, as fractions of its ``delta``.
STEPS = (1.0, 0.1)
#: The largest ratio of the finer step's difference to the coarser's that measures continuous: the geometric mean of
#: the ratios measured on drag's clamp, a kink (0.106), and its end, a jump (1.00), in float64, to two digits; checked
#: on an analytic kink and an analytic jump (``tests/saved/test_switch_harness.py``).
KINK_LIMIT = 0.33

type Outputs = Mapping[str, Any]


@dataclass(frozen=True)
class SwitchRow:
    """One owner's row: its ``declaration``; ``evaluate(offset)``, the outputs by group of what evaluates the switch
    at a point ``offset`` from it in the indicator's units, traced afresh at every call; ``delta``; the ``side`` the
    point is placed on; and ``held``, other owners' branches held while this one is forced, to place the point where
    the owner's branch reaches the outputs."""

    declaration: switch.Declaration
    evaluate: Callable[[float], Outputs]
    delta: float
    side: int = 1
    held: Mapping[str, bool] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class Measure:
    """One output group's forced-branch differences at the two steps (each the largest over the group's outputs), its
    rounding floor, the largest ratio of an output that is not below the floor at the finer step, and whether every
    output measured continuous."""

    coarse: float
    fine: float
    floor: float
    ratio: float
    continuous: bool


@dataclass
class Classification:
    owner: str
    measures: dict[str, Measure] = field(default_factory=dict[str, Measure])
    indicators: list[tuple[float, float]] = field(default_factory=list[tuple[float, float]])
    problems: list[str] = field(default_factory=list[str])

    def line(self) -> str:
        return f"{self.owner}: " + "; ".join(
            f"{group} {measure.coarse:.3g} then {measure.fine:.3g} (ratio {measure.ratio:.3g}, floor "
            f"{measure.floor:.2g}): {'continuous' if measure.continuous else 'not continuous'}"
            for group, measure in sorted(self.measures.items()))


def _arrays(outputs: Outputs) -> dict[str, np.ndarray]:
    return {group: np.asarray(value) for group, value in outputs.items()}


def differences(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each output's change between two forced branches: none where it is equal in both (the same infinity, both
    NaN), and without bound where it is finite in one and not the other."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    finite = np.isfinite(a) & np.isfinite(b)
    with np.errstate(invalid="ignore"):
        return np.where(same, 0.0, np.where(finite, np.abs(a - b), np.inf))


def rounding_floor(a: np.ndarray, b: np.ndarray, epsilon: float) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    finite = np.concatenate([a[np.isfinite(a)].ravel(), b[np.isfinite(b)].ravel(), np.zeros(1)])
    return FLOOR_ULPS * epsilon * float(np.max(np.abs(finite)))


def measure(coarse: np.ndarray, fine: np.ndarray, floor: float, kink_limit: float) -> Measure:
    """A group's measure from its outputs' differences at the two steps: an output is continuous if its difference
    at the finer step is below the floor, or at most ``kink_limit`` times a finite one at the coarser step."""
    shrunk = (fine <= floor) | (np.isfinite(coarse) & (fine <= kink_limit * coarse))
    read = fine > floor
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(read, fine / np.where(coarse > 0, coarse, np.nan), np.nan)
    ratio = float(np.nanmax(ratios)) if np.isfinite(ratios).any() else (math.inf if read.any() else math.nan)
    return Measure(float(coarse.max(initial=0.0)), float(fine.max(initial=0.0)), floor, ratio, bool(shrunk.all()))


def classify(row: SwitchRow, kink_limit: float) -> Classification:
    """``row`` measured at its two steps and judged against its declaration."""
    owner = row.declaration.owner
    found = Classification(owner)
    measured: dict[str, list[tuple[np.ndarray, float]]] = {}
    for step in STEPS:
        offset = row.side * row.delta * step / 4
        with switch.collecting() as rows:
            row.evaluate(offset)
        indicators = [float(np.asarray(r.indicator).ravel()[np.argmin(np.abs(np.asarray(r.indicator).ravel()))])
                      for r in rows if r.owner == owner]
        if not indicators:
            found.problems.append(f"{owner}: no evaluation at offset {offset:g} recorded the switch")
        for indicator in indicators:
            found.indicators.append((offset, indicator))
            if not abs(indicator - offset) <= INDICATOR_TOLERANCE * abs(offset):
                found.problems.append(f"{owner}: its indicator is {indicator:.6g} at a point placed {offset:.6g} from "
                                      f"the switch: a sign or units error, or a misplaced point")
        branches: list[dict[str, np.ndarray]] = []
        for branch in (True, False):
            with switch.forcing({**row.held, owner: branch}) as reached:
                branches.append(_arrays(row.evaluate(offset)))
            if owner not in reached.owners:
                found.problems.append(f"{owner}: the forced branch never reached it, so the step that ran was traced "
                                      f"before the setting opened: evaluate a step traced afresh")
        for group in branches[0]:
            a, b = branches[0][group], branches[1][group]
            measured.setdefault(group, []).append((differences(a, b), rounding_floor(a, b, _epsilon(a))))
    orders = row.declaration.orders
    for group, ((coarse, coarse_floor), (fine, fine_floor)) in measured.items():
        held = measure(coarse, fine, max(coarse_floor, fine_floor), kink_limit)
        found.measures[group] = held
        unchanged = held.coarse <= held.floor and held.fine <= held.floor
        if group not in orders and not unchanged:
            found.problems.append(f"{owner}: its branch changes {group} ({held.coarse:.3g}, then {held.fine:.3g}), for "
                                  f"which it declares no order")
        elif group in orders and orders[group] >= 0 and not held.continuous:
            found.problems.append(f"{owner}: declared order {orders[group]} in {group}, and its forced-branch "
                                  f"difference is {held.coarse:.3g} at delta and {held.fine:.3g} at delta / 10 (ratio "
                                  f"{held.ratio:.3g}, kink limit {kink_limit}): it is a jump")
    for group in set(orders) - set(measured):
        found.problems.append(f"{owner}: declares an order for {group}, which what evaluates it does not output")
    return found


def _epsilon(values: np.ndarray) -> float:
    """The unit roundoff of the dtype ``values`` were computed in; none for an integer output, which is exact."""
    return float(np.finfo(values.dtype).eps) if np.issubdtype(values.dtype, np.floating) else 0.0


def fallback_problems(evaluate: Callable[[], Outputs], derivative: Callable[[], Outputs], where: str,
                      scales: Sequence[float] = (2.0, 0.5)) -> list[str]:
    """Every output and derivative of ``evaluate`` and ``derivative`` (each traced afresh at every call) that
    changes when every domain guard's fallback is multiplied by each of ``scales``, at a point ``where`` names, at
    which some guard's condition is false; and a point at which no guard was reached, or a zero fallback, which
    multiplying cannot change, is refused as a test that cannot discriminate."""
    found: list[str] = []
    reference = {"value": _arrays(evaluate()), "derivative": _arrays(_flat(derivative()))}
    for scale in scales:
        with switch.replacing_fallbacks(scale) as reached:
            replaced = {"value": _arrays(evaluate()), "derivative": _arrays(_flat(derivative()))}
        if reached.guards == 0:
            found.append(f"{where}: no domain guard was reached, so replacing fallbacks tests nothing")
        if reached.zero_fallbacks:
            found.append(f"{where}: {reached.zero_fallbacks} domain guard(s) with a zero fallback, which scaling "
                         f"cannot change")
        for kind, outputs in reference.items():
            for group, value in outputs.items():
                change = float(differences(value, replaced[kind][group]).max(initial=0.0))
                if change > 0:
                    found.append(f"{where}: with every fallback times {scale}, the {kind} of {group} changes by "
                                 f"{change:.3g}: a fallback reaches an output")
    return found


def _flat(tree: Any) -> dict[str, Any]:
    tree_util: Any = jax.tree_util
    leaves: list[tuple[Any, Any]] = tree_util.tree_flatten_with_path(tree)[0]
    return {str(tree_util.keystr(path)): leaf for path, leaf in leaves}


__all__ = ["FLOOR_ULPS", "INDICATOR_TOLERANCE", "KINK_LIMIT", "STEPS", "Classification", "Measure", "SwitchRow",
           "classify", "differences", "fallback_problems", "measure", "rounding_floor"]
