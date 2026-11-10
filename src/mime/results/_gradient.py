"""A differentiated study's result: for each member, the objective and its derivative in each axis, with what its run
raised (the flags, a node's stop, a table read extrapolated by the experiment's declaration), the status of the path
the derivative took through the experiment's entries, and each iterative solve's report, forward and in the
derivative solves, with the warning of a solve that did not converge and the recheck of it with the cause removed."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import override

import numpy as np
from numpy.typing import NDArray

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.gradients import GradientStatus
from mime.nodes.flow.fvm._solve import REPORT_FIELDS

#: The columns of a report whose counts are a solve that stopped short of its tolerance: what a warning counts.
SHORT = ("at_cap", "stagnated", "not_finite")


def _count(report: NDArray[np.float64] | None, names: tuple[str, ...]) -> int:
    if report is None:
        return 0
    held = np.asarray(report, dtype=np.float64).reshape(-1, len(REPORT_FIELDS))
    return int(round(float(sum(held[:, REPORT_FIELDS.index(name)].sum() for name in names))))


def _short(report: NDArray[np.float64] | None) -> int:
    return _count(report, SHORT)


@dataclass(frozen=True)
class SolveReport:
    """One node's iterative solves over a member's run: ``forward``, the primal solves' reports summed over the run;
    ``steps``, each step's primal report (one row a step); and ``backward``, the derivative solves' (the transpose
    solves' in reverse mode, a row a step where the node keeps its report a step and one summed row otherwise; the
    tangent solves' in forward mode, summed over the run and over the directions), each in :data:`REPORT_FIELDS`'
    order. The counts add where reports are summed, so each keeps its meaning; the residual column of a sum is a sum."""

    node: str
    forward: NDArray[np.float64]
    steps: NDArray[np.float64] | None
    backward: NDArray[np.float64] | None
    mode: str

    @property
    def forward_short(self) -> int:
        """The primal solves that stopped short of their tolerance (at the cap, stagnated or not finite)."""
        return _short(self.forward)

    @property
    def backward_short(self) -> int:
        """The derivative solves that stopped short of their tolerance."""
        return _short(self.backward)

    @property
    def not_finite(self) -> int:
        """The solves, primal and derivative, whose residual was not finite: no cap or restart removes that."""
        return _count(self.forward, ("not_finite",)) + _count(self.backward, ("not_finite",))

    def flagged_steps(self) -> tuple[int, ...]:
        """The steps (from 1) whose primal solve stopped short, where the run kept each step's report."""
        if self.steps is None:
            return ()
        rows = np.asarray(self.steps, dtype=np.float64).reshape(-1, len(REPORT_FIELDS))
        short = sum(rows[:, REPORT_FIELDS.index(name)] for name in SHORT)
        return tuple(int(i) + 1 for i in np.flatnonzero(short > 0))

    def text(self) -> str:
        iterations = int(round(float(np.asarray(self.forward)[0])))
        lines = [f"{self.node}: {iterations} primal iterations over the run, {self.forward_short} solve"
                 f"{'s' if self.forward_short != 1 else ''} stopped short"]
        if self.backward is not None:
            held = np.asarray(self.backward, dtype=np.float64).reshape(-1, len(REPORT_FIELDS))
            which = "transpose" if self.mode == "reverse" else "tangent"
            lines.append(f"  {which} solves: {int(round(float(held[:, 0].sum())))} iterations, {self.backward_short} "
                         f"stopped short")
        return "\n".join(lines)


@dataclass(frozen=True)
class Recheck:
    """A member's derivative computed again with the cause of each flagged solve removed: ``how``; ``lifted``, the
    nodes re-solved with their cap lifted and their stall restarts raised, at the member's own tolerance (a solve that
    did not reach its cap or stall is unchanged, so only the solves that stopped short move); ``not_rechecked``, each
    node flagged that was not re-solved, with the reason (a non-finite residual, which neither removes; a node whose
    solve has no recheck); the objective and the derivative the recheck gave, their changes from the member's own
    (relative to the member's), and its own solves' reports, which say whether it converged. A recheck of no node
    (``made`` false) gives the member's own values and NaN changes."""

    how: str
    lifted: tuple[str, ...]
    not_rechecked: Mapping[str, str]
    value: float
    gradient: Mapping[str, NDArray[np.float64]]
    value_change: float
    gradient_change: Mapping[str, float]
    solves: tuple[SolveReport, ...]

    @property
    def made(self) -> bool:
        """Whether any flagged solve was re-solved."""
        return bool(self.lifted)

    @property
    def converged(self) -> bool:
        """Whether a recheck was made and none of its own solves stopped short."""
        return self.made and all(report.forward_short == 0 and report.backward_short == 0 for report in self.solves)


@dataclass(frozen=True)
class RaisedFlag:
    """A flag a member's differentiated run raised: its path (``node.field``), what it means, how many steps raised it
    and the first and the last (from 1)."""

    path: str
    meaning: str
    count: int
    first: int
    last: int


@dataclass(frozen=True)
class MemberGradient:
    """One member's derivative: its point (``index``) and its values of the study's axes, the objective, its derivative
    in each axis of ``wrt`` (by path, an array of the axis value's shape, in the objective's units per the axis's SI
    unit, ``units``), the flags its run raised, the refusals its run met (a node's stop), the table reads its plan
    extrapolated, each solving node's report, the recheck where a solve stopped short, and the warnings."""

    index: int
    values: Mapping[str, str]
    value: float
    gradient: Mapping[str, NDArray[np.float64]]
    units: Mapping[str, str]
    flags: tuple[RaisedFlag, ...] = ()
    refusals: tuple[Refusal, ...] = ()
    extrapolations: tuple[str, ...] = ()
    solves: tuple[SolveReport, ...] = ()
    recheck: Recheck | None = None
    warnings: tuple[str, ...] = ()

    @property
    def finite(self) -> bool:
        return bool(np.isfinite(self.value)) and all(bool(np.all(np.isfinite(g))) for g in self.gradient.values())

    @property
    def clean(self) -> bool:
        """Whether its run raised nothing, met no refusal, extrapolated no read and warned of nothing."""
        return not (self.flags or self.refusals or self.extrapolations or self.warnings)

    def text(self) -> str:
        where = ", ".join(f"{path} = {value}" for path, value in self.values.items())
        lines = [f"member {self.index}{f' ({where})' if where else ''}: objective {self.value:.10g}"]
        for path, value in self.gradient.items():
            shown = f"{float(value):.10g}" if np.ndim(value) == 0 else np.array2string(np.asarray(value), precision=10)
            lines.append(f"  d/d {path}: {shown} ({self.units.get(path, '')})")
        lines += [f"  flag {flag.path}: raised at {flag.count} step{'s' if flag.count != 1 else ''} ({flag.first} to "
                  f"{flag.last}): {flag.meaning}" for flag in self.flags]
        lines += [f"  refused [{refusal.code}]: {refusal.message}" for refusal in self.refusals]
        lines += [f"  extrapolated: {text}" for text in self.extrapolations]
        lines += [f"  {line}" for report in self.solves for line in report.text().splitlines()]
        lines += [f"  warning: {text}" for text in self.warnings]
        if self.recheck is not None:
            recheck = self.recheck
            if recheck.made:
                changes = ", ".join(f"{path} {change:.3g}" for path, change in recheck.gradient_change.items())
                lines.append(f"  recheck ({recheck.how}): objective changed {recheck.value_change:.3g}, the "
                             f"derivative {changes} (relative); its own solves "
                             f"{'converged' if recheck.converged else 'stopped short as well'}")
            lines += [f"  recheck not made for {node}: {reason}" for node, reason in recheck.not_rechecked.items()]
        return "\n".join(lines)


@dataclass(frozen=True)
class GradientResult:
    """A study's derivative: the axes it is taken in (``wrt``, by path), the mode (``forward`` or ``reverse``), the
    status of its path composed from the entries the derivative passes through (``path``, entry id to status) and
    each acknowledgement it needed (``acknowledged``), and each member's :class:`MemberGradient`, by point (``None``
    for a point the study dropped)."""

    wrt: tuple[str, ...]
    mode: str
    status: GradientStatus
    path: Mapping[str, GradientStatus]
    members: tuple[MemberGradient | None, ...]
    acknowledged: Mapping[str, str] = field(default_factory=dict[str, str])

    def member(self, index: int) -> MemberGradient:
        """Point ``index``'s derivative."""
        held = self.members[index] if -len(self.members) <= index < len(self.members) else None
        if held is None:
            ran = [str(i) for i, member in enumerate(self.members) if member is not None]
            raise ArgumentRefused([Refusal(
                "study.no-member", RefusalClass.USER,
                f"the study has no derivative at point {index}: it has {len(self.members)} points, and one it dropped "
                f"(drop_refused=True) has none",
                f"take a point it differentiated, `result.member(i)` for i in {', '.join(ran) or 'none'}")])
        return held

    @property
    def warnings(self) -> int:
        """The members with a warning: a solve that stopped short, forward or in its derivative."""
        return sum(1 for member in self.members if member is not None and member.warnings)

    def text(self) -> str:
        path = ", ".join(f"{owner} {status.value}" for owner, status in sorted(self.path.items()))
        lines = [f"the derivative of the objective in {', '.join(self.wrt)}, {self.mode} mode; its path's status is "
                 f"{self.status.value} ({path or 'no entry'})"]
        lines += [f"  acknowledged {owner}: {reason}" for owner, reason in sorted(self.acknowledged.items())]
        for member in self.members:
            if member is not None:
                lines += member.text().splitlines()
        return "\n".join(lines)

    @override
    def __str__(self) -> str:
        return self.text()


__all__ = ["SHORT", "GradientResult", "MemberGradient", "RaisedFlag", "Recheck", "SolveReport"]
