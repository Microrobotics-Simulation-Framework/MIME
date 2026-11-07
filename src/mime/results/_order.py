"""The observed order of convergence of an output over three levels of a step or a spacing, and the value it
extrapolates to (Richardson), refused where the levels are not in the asymptotic range.

Three levels ``h1 > h2 > h3`` with values ``f1, f2, f3`` fit ``f(h) = f* + C h^p``. The order ``p`` solves
``(f1 - f2) / (f2 - f3) = (h1^p - h2^p) / (h2^p - h3^p)``, exact for levels of any ratios (for a constant ratio ``r`` it
is ``log((f1 - f2) / (f2 - f3)) / log r``), and ``f* = f3 - C h3^p`` is the extrapolated value. Three levels fit three
unknowns and leave nothing to check the model with, so the range is judged on what the levels themselves say, in four
tests, each a refusal (``study.order``) that names what it saw:

1. the levels are three distinct positive numbers, and the values finite;
2. the two refinement ratios ``h1 / h2`` and ``h2 / h3`` agree within :data:`RATIO_AGREEMENT` of the smaller, since a
   scheme refined by different factors at the two steps is not the one sequence the fit assumes;
3. the successive differences have one sign and neither is zero: a value that goes past the limit and back, or does
   not move, is not converging from one side;
4. the order found lies in :data:`ORDER_RANGE`, and the ratio of the successive differences ``(f1 - f2) / (f2 - f3)``
   is within :data:`ORDER_AGREEMENT` of the ratio a scheme of the formal order ``n`` gives at these levels,
   ``(h1^n - h2^n) / (h2^n - h3^n)`` (``r^n`` for a constant ratio ``r``). The formal order is ``expected`` where the
   caller gives it, and otherwise the whole number nearest the order found, at least 1: a scheme's formal order is
   whole unless it is stated. So ``1 + 0.9 h + 5 h^2`` at ``h = 0.4, 0.2, 0.1``, a first-order scheme whose
   second-order term still dominates, is refused: its differences' ratio 3.25 is 19 % from the 4 of the second order
   it reads nearest (order 1.70). A contamination that moves the ratio by less than the bound is not seen
   (``1 + 0.9 h + 5 h^3`` reads 1.88, its ratio 8 % from 4): three levels cannot show it, and a fourth level can."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.units import Quantity

#: How far the two refinement ratios may differ, as a fraction of the smaller.
RATIO_AGREEMENT = 0.1
#: The orders taken as a convergence rate: below, the values hardly move with the step; above, they are rounding.
ORDER_RANGE = (0.1, 10.0)
#: How far the ratio of the successive differences may be from the formal order's ratio at the levels, as a fraction
#: of the formal order's.
ORDER_AGREEMENT = 0.1


@dataclass(frozen=True)
class Order:
    """The observed order of an output over three levels: ``order``, the ``extrapolated`` value (a quantity of the
    output's dimension), the refinement ratio (the geometric mean of the two), the ``levels`` and ``values`` coarse to
    fine, and the distance of the finest value from the extrapolated one."""

    order: float
    extrapolated: Quantity
    ratio: float
    levels: NDArray[np.float64]
    values: NDArray[np.float64]
    finest_miss: Quantity

    def text(self) -> str:
        return (f"observed order {self.order:.4g} over levels {', '.join(f'{h:.6g}' for h in self.levels)} (ratio "
                f"{self.ratio:.4g}); extrapolated value {float(np.asarray(self.extrapolated.value))!r}, the finest level "
                f"{abs(float(np.asarray(self.finest_miss.value))):.3g} from it")


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> ArgumentRefused:
    return ArgumentRefused([Refusal(code, category, message, action,
                                    why="three levels fit three unknowns, so the range is judged on the levels")])


def observed_order(levels: Quantity, values: Quantity, *, expected: float | None = None) -> Order:
    """The order and extrapolated value of ``values`` over ``levels`` (three of each, any order); see the module's
    four tests."""
    h = np.atleast_1d(np.asarray(levels.value, dtype=np.float64))
    f = np.asarray(values.value, dtype=np.float64)
    if h.shape != (3,) or f.shape != (3,):
        raise _refuse("study.order", RefusalClass.USER, f"the order is read from three levels of a scalar output, and this has levels of shape {h.shape} "
                      f"and values of shape {f.shape}",
                      "run three levels of one axis, `grid({DT: [s(4e-3), s(2e-3), s(1e-3)]})`, and an output with one number each")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(f)) and np.all(h > 0)) or len(set(h.tolist())) != 3:
        raise _refuse("study.order", RefusalClass.USER, f"the levels {h.tolist()} are not three distinct positive numbers, or a value "
                      f"({f.tolist()}) is not finite", "refine by one factor each time: `grid({DT: [s(4e-3), s(2e-3), s(1e-3)]})`")
    ranked = np.argsort(-h)
    h, f = h[ranked], f[ranked]
    first, second = float(h[0] / h[1]), float(h[1] / h[2])
    if abs(first - second) / min(first, second) > RATIO_AGREEMENT:
        raise _refuse("study.order", RefusalClass.USER, f"the refinement ratios {first:.4g} (from {h[0]:.6g} to {h[1]:.6g}) and {second:.4g} (from "
                      f"{h[1]:.6g} to {h[2]:.6g}) differ by more than {RATIO_AGREEMENT:.0%} of the smaller",
                      "refine by one factor each time: `grid({DT: [s(4e-3), s(2e-3), s(1e-3)]})`")
    d12, d23 = float(f[0] - f[1]), float(f[1] - f[2])
    if d12 == 0.0 or d23 == 0.0 or (d12 > 0) != (d23 > 0):
        raise _refuse("study.order", RefusalClass.USER, f"the values {f.tolist()} do not approach their limit from one side (successive differences "
                      f"{d12:.6g} and {d23:.6g}): not in the asymptotic range",
                      "refine further, `grid({DT: [s(4e-3), s(2e-3), s(1e-3), s(5e-4)]})`, or the output is not smooth in this step")
    target = d12 / d23

    def gap(p: float) -> float:
        return float((h[0] ** p - h[1] ** p) / (h[1] ** p - h[2] ** p)) - target

    low, high = ORDER_RANGE
    if gap(low) * gap(high) > 0 or gap(low) > 0:
        raise _refuse("study.order", RefusalClass.USER, f"the successive differences ({d12:.6g}, {d23:.6g}) give an order outside {low} to {high}",
                      "refine further, `grid({DT: [s(4e-3), s(2e-3), s(1e-3), s(5e-4)]})`, or the output is not converging at a power of the step")
    # The gap increases with p (for equal ratios it is r^p - target), so bisection finds its one root.
    rising = gap(high) > 0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if (gap(middle) > 0) == rising:
            high = middle
        else:
            low = middle
        if high - low <= 4 * np.finfo(float).eps * high:
            break
    order = 0.5 * (low + high)
    formal = float(expected) if expected is not None else float(max(1, round(order)))
    model = gap(formal) + target
    if abs(target - model) > ORDER_AGREEMENT * model:
        given = "the formal order given" if expected is not None else "the whole order nearest it"
        raise _refuse("study.order", RefusalClass.USER,
                      f"the ratio of the successive differences, {target:.4g} (order {order:.4g}), is more than "
                      f"{ORDER_AGREEMENT:.0%} from the {model:.4g} that {given}, {formal:g}, gives at these levels: not in "
                      f"the asymptotic range of a scheme of formal order {formal:g}",
                      "refine further, `grid({DT: [s(4e-3), s(2e-3), s(1e-3), s(5e-4)]})`; or, for a scheme whose formal "
                      "order is not whole, state it: `result.order(..., expected=1.5)`")
    coefficient = d23 / float(h[1] ** order - h[2] ** order)
    limit = float(f[2]) - coefficient * float(h[2] ** order)
    return Order(order, Quantity(limit, values.dimension), math.sqrt(first * second), h, f,
                 Quantity(float(f[2]) - limit, values.dimension))


__all__ = ["ORDER_AGREEMENT", "ORDER_RANGE", "RATIO_AGREEMENT", "Order", "observed_order"]
