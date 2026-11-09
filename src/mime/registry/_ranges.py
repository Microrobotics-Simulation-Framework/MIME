"""A choice an entry makes over a declared range (``exp.declare_range``): a count, a spacing, a scheme, made once for
every value inside the range rather than from the value itself, so that every value inside it builds one program.

An entry makes each such choice through :func:`choose_over`, the one way the range reaches it, and declares the
choice's rule over the range: ``ChoiceRule.Monotone`` (its measure moves one way in each ranged setting, so the
range's corners hold its extreme) or ``ChoiceRule.Sampled`` (not known to be: the range is also sampled inside at a
stated count).
A choice with neither is refused as the entry's defect. Each choice is recorded with its price against the value's
own (the plan's price report), and judged at the value itself: a value whose own choice is more demanding than the
range's (``RangeChoice.short``) means the rule missed it, and is refused as the entry's defect. The range's
independence check (the study's, ``mime.study._ranges``) pins each recorded choice and builds without the range: the
two builds are one program only where the range reaches the build through these choices alone.

**The cost.** A choice evaluates its measure at each of its rule's points, and the build makes it in every plan the
ranges ask for: the experiment's own and one at each corner of its ranges (``2 ** n`` for ``n`` ranges). The measure
is the entry's to price; the MRS drag's (``Drag.MRS``'s force points on FL-9 at 1.5 mm) cost about 0.16 s an
evaluation, measured on a shared laptop: its radius and length sampled over two ranges, 405 evaluations, planned in 66
s. Past :data:`MAX_EVALUATIONS` evaluations in all a choice is refused before it evaluates one (``range.cost``), naming
the numbers its rule samples."""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Callable, Generator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mime.registry._entries import Hull

#: The most evaluations of its measure a choice may ask for: its rule's points times the plans the ranges ask for (the
#: experiment's own and each corner's). The MRS drag's force points on FL-9 at 1.5 mm cost about 0.16 s an evaluation
#: (two of its numbers sampled over two ranges, 405 evaluations: 66 s), so the limit is about three minutes of planning
#: there; three of its numbers sampled over three ranges (729 points, 9 plans) took about ten minutes before this
#: limit refused them. It admits a choice sampled in one number over six ranges (585) and one monotone in three (520).
MAX_EVALUATIONS = 1024


class ChoiceRule:
    """The rule a choice over a declared range is made by, as its entry declares it: ``ChoiceRule.Monotone(basis)`` or
    ``ChoiceRule.Sampled(count, basis)``."""

    @dataclass(frozen=True)
    class Monotone:
        """The choice's measure moves one way in each ranged setting (``basis``: why, and what tests it), so that the
        range's corners hold its extreme and the choice is judged at them alone."""

        basis: str

    @dataclass(frozen=True)
    class Sampled:
        """The choice's measure is not known to move one way (``basis``: why): it is judged at ``count`` points along
        each ranged setting, the ends among them, every combination of them."""

        count: int
        basis: str


type Rule = ChoiceRule.Monotone | ChoiceRule.Sampled


@dataclass(frozen=True)
class RangeChoice:
    """A choice made over a range: what it chooses (``what``, its measure's unit ``unit``), whose (``owner``, the
    entry's id), the rule it declared, the ranged settings it was made over (``over``, the paths) and the point it took
    (``at``, each ranged setting's value there, SI), its measure there (``chosen``) and at the value itself (``own``),
    whether a larger measure is the more demanding (``largest``), and what the range's choice costs against the value's
    own, in words (``cost``)."""

    what: str
    owner: str
    rule: Rule
    over: tuple[str, ...]
    at: Mapping[str, float]
    chosen: float
    own: float
    largest: bool = True
    unit: str = ""
    cost: str = ""

    @property
    def moved(self) -> bool:
        return self.chosen != self.own

    @property
    def short(self) -> bool:
        """Whether the value itself would choose more than the range's choice gives: the rule missed it."""
        return self.own > self.chosen if self.largest else self.own < self.chosen

    def text(self, units: Mapping[str, str] | None = None) -> str:
        """The choice in words, each point's value in its range's unit (``units``, by path; SI where none)."""
        unit = f" {self.unit}" if self.unit else ""
        at = ", ".join(f"{path} {value:.6g}{f' {units[path]}' if units and path in units else ''}"
                       for path, value in self.at.items())
        cost = f"; {self.cost}" if self.cost else ""
        return (f"{self.owner}'s {self.what} {_number(self.chosen, self.unit)}{unit}, chosen at {at}, against "
                f"{_number(self.own, self.unit)}{unit} at the experiment's own value{cost}")


def _number(value: float, unit: str) -> str:
    if not unit:
        return f"{value:g}"
    return f"{value * 1e3:.4g}" if unit == "mm" else f"{value:.4g}"


#: The choices recorded while a plan is made, which the planner reads into its price report (``None``: none recorded).
_RECORDED: ContextVar[list[RangeChoice] | None] = ContextVar("range_choices", default=None)
#: The points the independence check pins each choice to, by its owner and what it chooses (``None``: none pinned).
_PINNED: ContextVar[Mapping[tuple[str, str], Mapping[str, float]] | None] = ContextVar("range_pins", default=None)
#: The plans the experiment's declared ranges ask for, which every choice is made in: its own and each corner's.
_PLANS: ContextVar[int] = ContextVar("range_plans", default=1)


@contextlib.contextmanager
def recording() -> Generator[list[RangeChoice]]:
    """The choices :func:`choose_over` makes inside the block, in order."""
    held: list[RangeChoice] = []
    token = _RECORDED.set(held)
    plans = _PLANS.set(1)
    try:
        yield held
    finally:
        _PLANS.reset(plans)
        _RECORDED.reset(token)


def planned_over(ranges: int) -> None:
    """Inside :func:`recording`: the experiment declares ``ranges`` ranges, so each choice is made in ``1 + 2 **
    ranges`` plans (its own and each corner's; one where it declares none), and its cost is judged on all of them."""
    _PLANS.set(1 + 2 ** ranges if ranges else 1)


@contextlib.contextmanager
def pinned(choices: Mapping[tuple[str, str], Mapping[str, float]]) -> Generator[None]:
    """Inside the block each choice of ``choices`` (by its owner and what it chooses) is taken at its point, with no
    range to choose over: what the independence check builds."""
    token = _PINNED.set(choices)
    try:
        yield
    finally:
        _PINNED.reset(token)


def points_over(ranges: Mapping[str, tuple[float, float]], rule: Rule) -> list[dict[str, float]]:
    """The points a choice under ``rule`` is judged at: every corner of ``ranges`` (each setting at its low or high
    end) for a monotone one; every combination of ``count`` evenly spaced values of each setting for a sampled one."""
    names = sorted(ranges)
    if isinstance(rule, ChoiceRule.Monotone):
        axes = [(ranges[name][0], ranges[name][1]) for name in names]
    else:
        axes = [tuple(ranges[name][0] + (ranges[name][1] - ranges[name][0]) * i / (rule.count - 1)
                      for i in range(rule.count)) for name in names]
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*axes)]


def _rule_problem(rule: object) -> str | None:
    if isinstance(rule, ChoiceRule.Monotone):
        return None if rule.basis.strip() else "its monotone rule gives no basis"
    if isinstance(rule, ChoiceRule.Sampled):
        if not isinstance(rule.count, int) or rule.count < 2:  # pyright: ignore[reportUnnecessaryIsInstance]
            return f"its sampled rule's count is {rule.count!r}, not a whole number of at least 2 (the ends)"
        return None if rule.basis.strip() else "its sampled rule gives no basis"
    return (f"it declares {rule!r} as its rule over the range, neither ChoiceRule.Monotone(basis) nor "
            f"ChoiceRule.Sampled(count, basis): a choice made over a range declares how its extreme is found")


def choose_over(owner: str, hull: Hull, what: str, rule: Rule, measure: Callable[[Hull], float], *,
                largest: bool = True, unit: str = "", cost: Callable[[float, float], str] | None = None) -> Hull:
    """The hull ``owner``'s choice ``what`` is made at: ``hull`` itself where none of its settings is ranged; else the
    point of its ranged settings (``Hull.ranges``) where ``measure`` is largest (or smallest, ``largest=False``), among
    the points ``rule`` names, with every other setting as it is. The choice is recorded with its price (``cost`` of
    the measure there and at ``hull``, in words). ``rule`` is ``ChoiceRule.Monotone`` or ``ChoiceRule.Sampled``, and
    anything else is the entry's defect, raised here. A choice asking for more than :data:`MAX_EVALUATIONS`
    evaluations over the plans the ranges ask for is refused (``range.cost``) before its first."""
    pins = _PINNED.get()
    pin = pins.get((owner, what)) if pins is not None else None
    if pin is not None:
        return hull.at(pin)
    if not hull.ranges:
        return hull
    problem = _rule_problem(rule)
    if problem is not None:
        raise TypeError(f"{owner}'s choice {what!r} over a declared range: {problem}")
    points = points_over(hull.ranges, rule)
    paths = {name: f"{hull.body}.parts.{hull.part}.{name}" for name in hull.ranges}
    plans = _PLANS.get()
    if len(points) * plans > MAX_EVALUATIONS:
        raise _cost_refused(owner, what, rule, [paths[name] for name in sorted(hull.ranges)], len(points), plans)
    found = [(measure(hull.at(point)), point) for point in points]
    best = (max if largest else min)(found, key=lambda pair: pair[0])
    own = measure(hull)
    choice = RangeChoice(what, owner, rule, tuple(paths[name] for name in sorted(hull.ranges)),
                         {paths[name]: value for name, value in sorted(best[1].items())}, float(best[0]), float(own),
                         largest, unit, cost(float(best[0]), float(own)) if cost is not None else "")
    held = _RECORDED.get()
    if held is not None and not any(c.owner == owner and c.what == what for c in held):
        held.append(choice)
    return hull.at(best[1])


def _cost_refused(owner: str, what: str, rule: Rule, over: list[str], points: int, plans: int) -> Exception:
    from mime.build import BuildRefused, Refusal, RefusalClass

    how = (f"its rule samples each at {rule.count} points, {points} points in all"
           if isinstance(rule, ChoiceRule.Sampled) else f"its rule judges the {points} corners of their ranges")
    return BuildRefused([Refusal(
        "range.cost", RefusalClass.USER,
        f"{owner}'s choice {what!r} is made over the declared ranges of {', '.join(over)}, and {how}, in each of the "
        f"{plans} plans the experiment's ranges ask for (its own and each corner's): {points * plans} evaluations, "
        f"over the {MAX_EVALUATIONS} a build makes. Each evaluation is the entry's measure of a body: the MRS drag's "
        f"force points on FL-9 at 1.5 mm cost about 0.16 s each (405 evaluations planned in 66 s on a shared "
        f"laptop), and three of its numbers sampled over three ranges (6561 evaluations) took about ten minutes",
        f"range fewer of the numbers {owner}'s rule samples (here {', '.join(over)}), or declare fewer ranges, and "
        f"sweep the other values as a study's axes, each value its own build: Study(exp).sweep(grid({{...}}), "
        f"duration=...)",
        why="the build makes each choice over the range in its own plan and in every corner's, so its cost is its "
            "points times those plans, and it is refused before the first is evaluated")])


def pins_of(choices: Any) -> dict[tuple[str, str], dict[str, float]]:
    """Each recorded choice's point, by its owner and what it chooses, with its settings named as the hull holds them:
    what :func:`pinned` takes."""
    return {(c.owner, c.what): {path.rsplit(".", 1)[-1]: value for path, value in c.at.items()} for c in choices}


__all__ = ["MAX_EVALUATIONS", "ChoiceRule", "RangeChoice", "Rule", "choose_over", "planned_over", "pinned",
           "pins_of", "points_over", "recording"]
