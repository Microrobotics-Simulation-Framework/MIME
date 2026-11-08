"""A choice an entry makes over a declared range (``exp.declare_range``): a count, a spacing, a scheme, made once for
every value inside the range rather than from the value itself, so that every value inside it builds one program.

An entry makes each such choice through :func:`choose_over`, the one way the range reaches it, and declares the
choice's rule over the range: :class:`Monotone` (its measure moves one way in each ranged setting, so the range's
corners hold its extreme) or :class:`Sampled` (not known to be: the range is also sampled inside at a stated count).
A choice with neither is refused as the entry's defect. Each choice is recorded with its price against the value's
own (the plan's price report), and judged at the value itself: a value whose own choice is more demanding than the
range's (``RangeChoice.short``) means the rule missed it, and is refused as the entry's defect. The range's
independence check (``mime.testing.range_independence``) pins each recorded choice and builds without the range: the
two builds are one program only where the range reaches the build through these choices alone."""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Callable, Generator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mime.registry._entries import Hull, Interaction, Observation

#: The most points a sampled rule may evaluate its measure at, over every ranged setting of its choice together.
SAMPLED_POINTS = 4096


@dataclass(frozen=True)
class Monotone:
    """The choice's measure moves one way in each ranged setting (``basis``: why, and what tests it), so that the
    range's corners hold its extreme and the choice is judged at them alone."""

    basis: str


@dataclass(frozen=True)
class Sampled:
    """The choice's measure is not known to move one way (``basis``: why): it is judged at ``count`` points along each
    ranged setting, the ends among them, every combination of them."""

    count: int
    basis: str


type ChoiceRule = Monotone | Sampled


@dataclass(frozen=True)
class RangeChoice:
    """A choice made over a range: what it chooses (``what``, its measure's unit ``unit``), whose (``owner``, the
    entry's id), the rule it declared, the ranged settings it was made over (``over``, the paths) and the point it took
    (``at``, each ranged setting's value there, SI), its measure there (``chosen``) and at the value itself (``own``),
    whether a larger measure is the more demanding (``largest``), and what the range's choice costs against the value's
    own, in words (``cost``)."""

    what: str
    owner: str
    rule: ChoiceRule
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


@contextlib.contextmanager
def recording() -> Generator[list[RangeChoice]]:
    """The choices :func:`choose_over` makes inside the block, in order."""
    held: list[RangeChoice] = []
    token = _RECORDED.set(held)
    try:
        yield held
    finally:
        _RECORDED.reset(token)


@contextlib.contextmanager
def pinned(choices: Mapping[tuple[str, str], Mapping[str, float]]) -> Generator[None]:
    """Inside the block each choice of ``choices`` (by its owner and what it chooses) is taken at its point, with no
    range to choose over: what the independence check builds."""
    token = _PINNED.set(choices)
    try:
        yield
    finally:
        _PINNED.reset(token)


def points_over(ranges: Mapping[str, tuple[float, float]], rule: ChoiceRule) -> list[dict[str, float]]:
    """The points a choice under ``rule`` is judged at: every corner of ``ranges`` (each setting at its low or high
    end) for a monotone one; every combination of ``count`` evenly spaced values of each setting for a sampled one."""
    names = sorted(ranges)
    if isinstance(rule, Monotone):
        axes = [(ranges[name][0], ranges[name][1]) for name in names]
    else:
        axes = [tuple(ranges[name][0] + (ranges[name][1] - ranges[name][0]) * i / (rule.count - 1)
                      for i in range(rule.count)) for name in names]
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*axes)]


def _rule_problem(rule: object) -> str | None:
    if isinstance(rule, Monotone):
        return None if rule.basis.strip() else "its monotone rule gives no basis"
    if isinstance(rule, Sampled):
        if not isinstance(rule.count, int) or rule.count < 2:  # pyright: ignore[reportUnnecessaryIsInstance]
            return f"its sampled rule's count is {rule.count!r}, not a whole number of at least 2 (the ends)"
        return None if rule.basis.strip() else "its sampled rule gives no basis"
    return (f"it declares {rule!r} as its rule over the range, neither Monotone(basis) nor Sampled(count, basis): a "
            f"choice made over a range declares how its extreme is found")


def owner_of(entry: Interaction | Observation, bindings: Mapping[str, str]) -> str:
    """The id of ``entry`` bound so, as a choice names its owner."""
    from mime.build import entry_id

    return entry_id(type(entry), bindings)


def choose_over(owner: str, hull: Hull, what: str, rule: ChoiceRule, measure: Callable[[Hull], float], *,
                largest: bool = True, unit: str = "", cost: Callable[[float, float], str] | None = None) -> Hull:
    """The hull ``owner``'s choice ``what`` is made at: ``hull`` itself where none of its settings is ranged; else the
    point of its ranged settings (``Hull.ranges``) where ``measure`` is largest (or smallest, ``largest=False``), among
    the points ``rule`` names, with every other setting as it is. The choice is recorded with its price (``cost`` of
    the measure there and at ``hull``, in words). ``rule`` is :class:`Monotone` or :class:`Sampled`, and anything else is
    the entry's defect, raised here."""
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
    if isinstance(rule, Sampled) and len(points) > SAMPLED_POINTS:
        raise ValueError(f"{owner}'s choice {what!r} samples {len(points)} points of the range, over the "
                         f"{SAMPLED_POINTS} a sampled rule may ask for")
    found = [(measure(hull.at(point)), point) for point in points]
    best = (max if largest else min)(found, key=lambda pair: pair[0])
    own = measure(hull)
    paths = {name: f"{hull.body}.parts.{hull.part}.{name}" for name in hull.ranges}
    choice = RangeChoice(what, owner, rule, tuple(paths[name] for name in sorted(hull.ranges)),
                         {paths[name]: value for name, value in sorted(best[1].items())}, float(best[0]), float(own),
                         largest, unit, cost(float(best[0]), float(own)) if cost is not None else "")
    held = _RECORDED.get()
    if held is not None and not any(c.owner == owner and c.what == what for c in held):
        held.append(choice)
    return hull.at(best[1])


def pins_of(choices: Any) -> dict[tuple[str, str], dict[str, float]]:
    """Each recorded choice's point, by its owner and what it chooses, with its settings named as the hull holds them:
    what :func:`pinned` takes."""
    return {(c.owner, c.what): {path.rsplit(".", 1)[-1]: value for path, value in c.at.items()} for c in choices}


__all__ = ["SAMPLED_POINTS", "ChoiceRule", "Monotone", "RangeChoice", "Sampled", "choose_over", "owner_of", "pinned",
           "pins_of", "points_over", "recording"]
