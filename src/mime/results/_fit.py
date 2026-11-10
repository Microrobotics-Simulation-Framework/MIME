"""A fit's result: each iterate as it was judged (its values, its residual and loss, the derivative of the residual in
each axis, every validity line with its margin, whether it was taken and why), the Fisher information at the start
with the declared noise, and where the fit ended and why."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, override

import numpy as np
from numpy.typing import NDArray

from mime.gradients import GradientStatus

#: Where a validity line is judged: when the iterate is planned (a refusal), from its run (a flag it raised, a watch
#: read at its end), or by its box (the declared range and the leaves' bounds the iterates are clipped to).
LINE_KINDS = ("plan", "run", "box")


@dataclass(frozen=True)
class LineMargin:
    """One validity line at one iterate: its ``name`` (a refusal's code, a flag's path, a watch's), where it is judged
    (``kind``, one of :data:`LINE_KINDS`), the iterate's ``value`` of what the line bounds and the line's ``limit``
    (NaN where the line has no number: a refusal, a flag without a measure), and the ``margin`` (in the value's units:
    how far inside the line the iterate is, at or above zero inside, below it past; NaN where the line could not
    be judged, ``meaning`` then saying why), ``past`` where it is past."""

    name: str
    kind: str
    value: float
    limit: float
    margin: float
    meaning: str = ""

    @property
    def past(self) -> bool:
        return bool(self.margin < 0.0)

    @property
    def judged(self) -> bool:
        """Whether the line could be judged at the iterate (a watch whose number is not computed cannot)."""
        return not np.isnan(self.margin)

    def text(self) -> str:
        if not self.judged:
            return f"{self.kind} line {self.name}: not judged ({self.meaning})"
        if np.isfinite(self.limit):
            where = f"{self.value:.6g} against {self.limit:.6g}, margin {self.margin:.3g}"
        else:
            where = "raised" if self.past else "not raised"
        return f"{self.kind} line {self.name}: {where}{' (past)' if self.past else ''}"


@dataclass(frozen=True)
class FitIterate:
    """One iterate: its ``values`` (by axis path, SI), the residual over the declared noise (``scaled``) and the loss
    (half its squared norm), the derivative of the scaled residual in each axis, every validity line with its margin,
    whether the fit took it, and why it did or did not. ``computed`` holds the lines the iterate's plan computed at
    build (by entry; prose, which their runtime forms, the flags, number): empty where the iterate's nodes were
    judged at what the start's were."""

    index: int
    values: Mapping[str, NDArray[np.float64]]
    loss: float
    scaled: NDArray[np.float64]
    jacobian: Mapping[str, NDArray[np.float64]]
    lines: tuple[LineMargin, ...]
    taken: bool
    why: str
    computed: Mapping[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])

    @property
    def past(self) -> tuple[LineMargin, ...]:
        return tuple(line for line in self.lines if line.past)

    @property
    def unjudged(self) -> tuple[LineMargin, ...]:
        """The lines that could not be judged here: never counted inside."""
        return tuple(line for line in self.lines if not line.judged)


@dataclass(frozen=True)
class FitResult:
    """A fit: the axes it fitted (``wrt``), the values it started from and ended at (SI), the loss there, where it
    ended (``ended``: ``converged``; ``at a line`` with the line's name in ``why`` (a refusal's code, a flag's path, a
    Reynolds line, ``box:<axis>`` where the range pins the step); ``unjudged`` where an iterate had a line that could not
    be judged; ``iterations``; ``refused`` on the partial result of a refused fit), every
    iterate it judged, the Fisher information of the scaled residual at the start (MADDENING's ``FIMReport``, with the
    declared noise) and the axes it names insensitive there, and the path's status as a study's derivative composes
    it."""

    wrt: tuple[str, ...]
    start: Mapping[str, NDArray[np.float64]]
    values: Mapping[str, NDArray[np.float64]]
    loss: float
    ended: str
    why: str
    iterates: tuple[FitIterate, ...]
    fim: Any
    insensitive: tuple[str, ...]
    status: GradientStatus
    path: Mapping[str, GradientStatus] = field(default_factory=dict[str, GradientStatus])

    @property
    def taken(self) -> tuple[FitIterate, ...]:
        return tuple(iterate for iterate in self.iterates if iterate.taken)

    def text(self) -> str:
        values = ", ".join(f"{path} = {np.array2string(np.asarray(value), precision=10)}"
                           for path, value in self.values.items())
        lines = [f"fit of {', '.join(self.wrt)}: ended {self.ended} ({self.why}) at {values}, loss {self.loss:.6g}, "
                 f"{len(self.taken)} of {len(self.iterates)} iterates taken; path {self.status.value}"]
        if self.insensitive:
            lines.append(f"  insensitive at the start (the Fisher information with the declared noise): "
                         f"{', '.join(self.insensitive)}")
        unscaled = tuple(getattr(self.fim, "zero_scaled", ()))
        if unscaled:
            lines.append(f"  not resolved by the Fisher information (a start of exactly 0.0 has no relative scale): "
                         f"{', '.join(unscaled)}")
        first: Mapping[str, tuple[str, ...]] = self.iterates[0].computed if self.iterates else {}
        for iterate in self.iterates:
            shown = ", ".join(f"{np.array2string(np.asarray(value), precision=8)}" for value in iterate.values.values())
            lines.append(f"  iterate {iterate.index}: {shown}, loss {iterate.loss:.6g}, "
                         f"{'taken' if iterate.taken else 'not taken'}: {iterate.why}")
            lines += [f"    {line.text()}" for line in iterate.lines
                      if line.past or not line.judged or np.isfinite(line.limit)]
            lines += [f"    plan line of {owner}: {text}" for owner, held in iterate.computed.items() for text in held
                      if text not in first.get(owner, ())]
        return "\n".join(lines)

    @override
    def __str__(self) -> str:
        return self.text()


__all__ = ["LINE_KINDS", "FitIterate", "FitResult", "LineMargin"]
