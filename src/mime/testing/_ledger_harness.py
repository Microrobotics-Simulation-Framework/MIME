"""The energy ledger as a gate's tool: one check that a metered run's books close, the ratio flags a node declares, and
the order check of a scheme's numerical terms. General: a dynamics, a field kind or a scheme brings only its
declarations (``MimeNode.LEDGER_FLAGS``, a scheme's ``TermOrder`` per term, ``mime.core.ledger``); nothing here reads a
solver, a grid or a fluid.

1. :func:`ledger_closes`: a metered receiver's records (or every metered receiver of a run) hold the step identity
   within its floor at every step (on route 1, the part of its step it books by formula), no work in an unattributed
   input (or some, where the gate fed it there on purpose), no gift or indefinite step of a law, every step dissipated
   on route 1, no jump between steps that no producer owns, and a balance within its rounding (route 2: route 1
   leaves the scheme's unbooked dissipation in its balance, judged by its sign each step). ``step_identity`` and
   SF-12's ledger case judge with it.
2. **Ratio flags**: :func:`flag_readings` reads each active flag of a node on one metered step (traced, inside the
   gate's scan; :func:`run_flags` is that scan), :func:`judge_flags` holds each flag's worst reading over a run against
   its line in the run's dtype. A flag declared ``pending`` is listed apart, never counted as a check; a flag that is
   not pending and has no plant caught every run is coverage claimed, not shown, and :func:`flag_problems` names it.
3. :func:`term_orders`: each numerical term against its declared order, from a rig's sums at its setting, with dt
   halved and with the tolerance a hundred times looser (each apart): a term of order ``p`` in dt, summed over a fixed
   time, falls by ``2^(p - 1)`` as dt halves; one of order ``q`` in the tolerance, summed over a fixed count of steps,
   by ``100^q``. It passes where the order its fall shows rounds to the declared one (within
   :data:`ORDER_HALF_WIDTH`), or reaches the declared order's floor where the order is a floor (``TermOrder.at_least``:
   a residual a solve's stop bounds). A term declaring no order fails, naming it. **The fall is read from each term
   summed over the rig's drawn cases**, each weighing by its own size: on one drawn case a term whose leading
   coefficient cancels shows an order its scheme does not have (a ratio of two small numbers), and a single case is
   no judgement of an order."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np
from maddening.testing.verification import VerificationResult

from mime.core import ledger
from mime.core.ledger import LedgerFlag, TermOrder

if TYPE_CHECKING:
    from mime.results import EnergyAccount

#: How far each argument is refined: dt halved; the tolerance a hundred times looser. A solve's stop bounds its residual
#: by the tolerance, which it lands under by a varying share, so halving the tolerance often leaves the stop where it
#: was (the flow's N_Krylov fell by 1.1 to 7.3 over a halving, 200 seeds a dtype): a hundredfold step, #225's, resolves
#: it (20 to 1700; 36 to 460 on the order rig's drawn ensembles, 1500 a dtype).
REFINEMENT = {"dt": 2.0, "tolerance": 100.0}
#: A term passes where the order its fall shows is within this of its declared order: the ratio rounds to the declared
#: order and to no neighbour. For a second order in dt, 1.41 to 2.83; for an order of at least one in the tolerance, a
#: fall of at least 10 (#225's), so a residual falling as the tolerance's square root (a stop on a squared norm, order
#: one half) passes: no plant of that order was measured. **On one drawn case this band false-alarms**: the flow's
#: ``N_conv`` is the residual of products that cancel (its sum is 0.02 % to 79 % of their magnitudes), so where its
#: leading coefficient nearly vanishes one swirl falls at an apparent order of 1 or 3.7 (ratios 1.01 to 6.59 over 400
#: swirls a dtype, 2 out of the band, and the third-order plant missed on 1): the band is judged on the sum over
#: drawn cases (:func:`term_orders`), never widened to pass a single case. Three swirls summed, at the order rig's
#: step of 0.00125: 1.97 to 2.22 over 1500 fresh ensembles a dtype, none out, each wrong order (1 or 3) caught on every
#: one, and an order shift of ``N_conv`` caught on every ensemble from 0.49 up and 0.57 down (https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42794715500).
ORDER_HALF_WIDTH = 0.5
_NAME = "ledger closes"


def _records_of(state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in state.items() if key.startswith(ledger.RECORDS)}


def _record_failures(records: Mapping[str, np.ndarray], *, fed_unattributed: bool, grouped: bool) -> list[str]:
    """What a metered receiver's records say failed, each with its amount (empty where its books close)."""
    r = ledger.RECORDS
    if f"{r}work" not in records:
        return ["it holds no ledger records: the receiver is not metered"]
    eps = float(np.finfo(records[f"{r}work"].dtype).eps)
    floor = ledger.IDENTITY_ULPS * eps
    found = [f"its record {key} is not finite" for key, value in records.items()
             if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value))]
    route_one = f"{r}route_one_step" in records
    worst = float(records[f"{r}identity_worst"])
    if worst > floor:
        what = "the part of its step it books by formula" if route_one else "the step identity's residual"
        found.append(f"{what} is {worst / floor:.3g} of its floor at its worst step")
    between = float(records[f"{r}between_worst"])
    if not grouped and between > floor:
        found.append(f"its storage jumped between steps by {between / floor:.3g} times its rounding floor, which no "
                     f"declared producer owns")
    unattributed = abs(float(records[f"{r}unattributed_worst"]))
    if fed_unattributed and not unattributed > 0.0:
        found.append("work fed into its unattributed input was not booked as unattributed work: it was booked under a "
                     "law or lost")
    elif not fed_unattributed and unattributed != 0.0:
        found.append(f"work arrived unattributed, {unattributed:.3g} J at its largest step (exactly zero where every "
                     f"delivery is a law's)")
    gifts = np.asarray(records[f"{r}gift_steps"], dtype=np.float64)
    if np.sum(gifts) > 0.0:
        found.append(f"a dissipative law gave energy: {int(np.sum(gifts))} law-steps, by row {gifts.tolist()}")
    indefinite = np.asarray(records[f"{r}indefinite_steps"], dtype=np.float64)
    if np.sum(indefinite) > 0.0:
        found.append(f"an operator checked definite at run time was not: {int(np.sum(indefinite))} law-steps")
    if route_one:
        negative = float(records[f"{r}route_one_negative_steps"])
        if negative != 0.0:
            found.append(f"{negative:g} steps gave energy (route 1: each step must dissipate; at worst "
                         f"{float(records[f'{r}route_one_worst']):.3g} of its scale)")
    else:
        balance = abs(float(np.asarray(ledger.balance(records, grouped=grouped))))
        rounding = float(np.asarray(ledger.balance_floor(records)))
        if not balance <= rounding:
            found.append(f"its books' balance is {balance:.3g} J, over their rounding {rounding:.3g} J")
    return found


def ledger_closes(books: Any, *, fed_unattributed: bool = False, grouped: bool = False) -> VerificationResult:
    """Whether a metered run's books close at every step: ``books`` a metered receiver's state or records (a node's
    final state, ``result.final[name]``), or a run's result (each metered receiver in it, a coupling group's members
    judged as members, and the account's findings). ``fed_unattributed``: the gate fed work into the receiver's
    unattributed input on purpose, which must then come out booked as unattributed (``step_identity``'s case);
    ``grouped``: a receiver whose jump between steps its coupling group owns."""
    final = getattr(books, "final", None)
    if final is None:
        found = _record_failures(_records_of(cast(Mapping[str, Any], books)), fed_unattributed=fed_unattributed,
                                 grouped=grouped)
        return VerificationResult(_NAME, "FAIL" if found else "PASS", detail="; ".join(found))
    account = cast("EnergyAccount | None", books.energy)
    groups: Mapping[str, str] = account.groups if account is not None else {}
    members = {member for group in groups for member in group.split("+")}
    found: list[str] = []
    metered = 0
    for name, state in cast(Mapping[str, Mapping[str, Any]], final).items():
        records = _records_of(state)
        if records:
            metered += 1
            found += [f"{name}: {failure}" for failure in _record_failures(
                records, fed_unattributed=fed_unattributed, grouped=grouped or name in members)]
    if not metered:
        found.append("the run holds no metered receiver: build it with EnergyLedger() observed")
    if account is not None:
        found += [f"the run's finding: {finding}" for finding in account.findings]
    return VerificationResult(_NAME, "FAIL" if found else "PASS", detail="; ".join(found))


# -- the ratio flags ------------------------------------------------------------------------------------------------

def declared_flags(node: Any) -> tuple[LedgerFlag, ...]:
    held: Any = node.__class__
    return tuple(cast(tuple[LedgerFlag, ...], getattr(held, "LEDGER_FLAGS", ())))


def active_flags(node: Any) -> tuple[LedgerFlag, ...]:
    return tuple(flag for flag in declared_flags(node) if flag.active)


def flag_problems(flags: Sequence[LedgerFlag]) -> list[str]:
    """What is wrong with a node's flag declarations: a name twice; a flag not pending with no reading, a line missing
    or not positive in a dtype, or no plant caught every run (coverage claimed, not shown); evidence with no link; a
    pending flag with no reason or carrying evidence (a measured flag is made active, not left pending)."""
    found: list[str] = []
    names = [flag.name for flag in flags]
    found += [f"the flag {name!r} is declared twice" for name in sorted({n for n in names if names.count(n) > 1})]
    for flag in flags:
        for item in flag.evidence:
            if not item.evidence.startswith("https://"):
                found.append(f"{flag.name}: the plant {item.plant!r} has no link to the run that measured it")
        if flag.pending:
            if flag.evidence:
                found.append(f"{flag.name} is pending and carries evidence: a flag measured with its plants is active")
            continue
        if flag.reading is None:
            found.append(f"{flag.name} is active and gives no reading")
        for dtype in ("float32", "float64"):
            line = flag.line.get(dtype)
            if line is None or not np.isfinite(line) or line <= 0.0:
                found.append(f"{flag.name} has no measured line in {dtype}")
        for dtype in ("float32", "float64"):
            if not flag.shown(dtype):
                found.append(f"{flag.name} is declared a check and no plant of it is caught every run with no false "
                             f"alarm in {dtype}: coverage claimed, not shown (declare it pending, with the reason)")
    return found


def flag_readings(node: Any, p: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any],
                  flags: Sequence[LedgerFlag] | None = None) -> dict[str, Any]:
    """Each active flag's reading (or each of ``flags``': a flag being measured) of one metered step of ``node``, from
    the states either side of it (traced)."""
    chosen = active_flags(node) if flags is None else flags
    return {flag.name: cast(Callable[..., Any], flag.reading)(node, p, before, after) for flag in chosen}


def run_flags(node: Any, step: Callable[[Mapping[str, Any]], Mapping[str, Any]], state: Mapping[str, Any],
              steps: int, p: Mapping[str, Any] | None = None,
              flags: Sequence[LedgerFlag] | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """``steps`` of ``step`` (a metered node's step with its inputs fed, ``state -> state``) from ``state`` under one
    compiled scan, each active flag (or each of ``flags``) read at each step: the final state, and each flag's
    readings, a row a step."""
    params = node._step_constants(None) if p is None else p  # noqa: SLF001 - the step's own constants

    def body(s: Mapping[str, Any], _: Any) -> tuple[Mapping[str, Any], dict[str, Any]]:
        ended = step(s)
        return ended, flag_readings(node, params, s, ended, flags)

    scan = cast(Any, jax).lax.scan

    def run(s: Mapping[str, Any]) -> Any:
        return scan(body, s, None, length=steps)

    ended, readings = cast(Any, jax).jit(run)(dict(state))
    return dict(ended), {name: np.asarray(values, dtype=np.float64) for name, values in readings.items()}


@dataclass(frozen=True)
class FlagReading:
    """One active flag over a run: its worst reading (the largest magnitude after its ``settle`` steps; NaN if any
    reading was not a number), its line in the run's dtype, and the step it was worst at (from the run's start)."""

    flag: str
    worst: float
    line: float
    step: int

    @property
    def passed(self) -> bool:
        return bool(self.worst <= self.line)


@dataclass(frozen=True)
class FlagReport:
    """A run's flags: each active flag's reading (the checks), each pending flag with its reason, and each silent one
    (it read exactly zero at every judged step: the run formed nothing it reads, as convection's flag where the node's
    convection is off), and each with no line in the run's dtype (a failure). Neither a pending nor a silent flag is a
    check, and a run with no check does not pass."""

    node: str
    readings: tuple[FlagReading, ...]
    pending: Mapping[str, str] = field(default_factory=dict[str, str])
    silent: tuple[str, ...] = ()
    unlined: tuple[str, ...] = ()

    @property
    def checks(self) -> int:
        return len(self.readings)

    @property
    def passed(self) -> bool:
        return bool(self.readings) and not self.unlined and all(reading.passed for reading in self.readings)

    def failures(self) -> list[str]:
        found = [] if self.readings else [f"no flag of {self.node} read anything: nothing was checked (silent: "
                                          f"{list(self.silent)}, pending: {sorted(self.pending)})"]
        found += [f"{name} has no measured line in the run's dtype: it judged nothing" for name in self.unlined]
        return found + [f"{r.flag} read {r.worst:.3g} at step {r.step}, past its line {r.line:.3g}"
                        for r in self.readings if not r.passed]


def judge_flags(node: Any, readings: Mapping[str, Any], dtype: str,
                flags: Sequence[LedgerFlag] | None = None) -> FlagReport:
    """Each active flag of ``node`` (or each of ``flags``) judged on its readings over a run (a row a step, from the
    run's start, as :func:`run_flags` gives them) against its line in ``dtype``; the pending ones listed with their
    reasons; one that read exactly zero at every judged step listed as silent. A flag whose readings are missing fails,
    as one whose reading is not a number does, and one with no line in ``dtype``."""
    held: list[FlagReading] = []
    silent: list[str] = []
    unlined: list[str] = []
    for flag in active_flags(node) if flags is None else flags:
        if dtype not in flag.line:
            unlined.append(flag.name)
            continue
        values = np.abs(np.asarray(readings.get(flag.name, np.full(1, np.nan)), dtype=np.float64))[flag.settle:]
        if values.size == 0 or not np.all(np.isfinite(values)):
            held.append(FlagReading(flag.name, float("nan"), flag.line[dtype], -1))
            continue
        if not np.any(values):
            silent.append(flag.name)
            continue
        at = int(np.argmax(values))
        held.append(FlagReading(flag.name, float(values[at]), flag.line[dtype], at + flag.settle))
    pending = {flag.name: flag.pending for flag in declared_flags(node) if flag.pending}
    return FlagReport(str(getattr(node, "name", type(node).__name__)), tuple(held), pending, tuple(silent),
                      tuple(unlined))


# -- the order check ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderCheck:
    """One term against its declared order in one argument (``"dt"`` or ``"tolerance"``): its sum at the coarser
    setting over its sum at the finer (``ratio``), the order that ratio shows (``observed``), and the band of ratios
    whose order rounds to the declared one (or reaches its floor, for an order declared ``at_least``)."""

    term: str
    argument: str
    order: int
    ratio: float
    observed: float
    band: tuple[float, float]

    @property
    def passed(self) -> bool:
        return bool(self.band[0] <= self.ratio <= self.band[1])


@dataclass(frozen=True)
class OrderReport:
    """Each declared order checked; each term with no order in an argument listed with its declaration's reason; and
    each term that declares no order at all (a failure)."""

    checks: tuple[OrderCheck, ...]
    skipped: Mapping[str, str] = field(default_factory=dict[str, str])
    undeclared: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.undeclared and all(check.passed for check in self.checks)

    def failures(self) -> list[str]:
        found = [f"{term} declares no order" for term in self.undeclared]
        return found + [f"{c.term}'s sum fell by {c.ratio:.4g} over its {c.argument}'s refinement, order {c.observed:.3g}, "
                        f"outside [{c.band[0]:.3g}, {c.band[1]:.3g}] for order {c.order}" for c in self.checks
                        if not c.passed]


def observed_order(argument: str, ratio: float) -> float:
    """The order a term's fall over its argument's refinement shows: ``1 + log2`` of it in dt (summed over a fixed
    time, the steps double as dt halves), its log in the tolerance's own factor (a fixed count of steps)."""
    shift = 1.0 if argument == "dt" else 0.0
    return shift + math.log(ratio) / math.log(REFINEMENT[argument]) if ratio > 0.0 else float("-inf")


def ratio_band(argument: str, order: int, at_least: bool) -> tuple[float, float]:
    """The ratios whose order is within :data:`ORDER_HALF_WIDTH` of ``order`` (at least its floor, for an order
    declared ``at_least``)."""
    shift = 1.0 if argument == "dt" else 0.0
    factor = REFINEMENT[argument]
    low = factor ** (order - ORDER_HALF_WIDTH - shift)
    return low, float("inf") if at_least else factor ** (order + ORDER_HALF_WIDTH - shift)


Sums = Callable[[float, float], Mapping[str, float]]


def term_orders(sums: Sums | Sequence[Sums], terms: Sequence[str], orders: Mapping[str, TermOrder]) -> OrderReport:
    """Each of ``terms`` against its order in ``orders`` (a scheme's declarations): ``sums(dt_scale,
    tolerance_scale)`` is each term's magnitude summed over a fixed time at the rig's step times ``dt_scale`` and its
    tolerance times ``tolerance_scale``, called at the rig's setting, with dt halved and with the tolerance
    :data:`REFINEMENT` times looser, each where some term declares an order in it. ``sums`` is one per drawn case
    (a sequence) or one for the whole rig: each term's fall is read from its sum over every case, so a case whose
    term cancels weighs by its own small size (:data:`ORDER_HALF_WIDTH`)."""
    cases: tuple[Sums, ...] = (sums,) if callable(sums) else tuple(sums)

    def summed(dt_scale: float, tolerance_scale: float) -> dict[str, float]:
        each = [case(dt_scale, tolerance_scale) for case in cases]
        return {term: math.fsum(float(one[term]) for one in each) for term in terms if term in orders}

    undeclared = tuple(term for term in terms if term not in orders)
    base = summed(1.0, 1.0)
    other: dict[str, Mapping[str, float]] = {}
    checks: list[OrderCheck] = []
    skipped: dict[str, str] = {}
    for term in terms:
        order = orders.get(term)
        if order is None:
            continue
        for argument, declared in (("dt", order.dt), ("tolerance", order.tolerance)):
            if declared is None:
                skipped[f"{term} ({argument})"] = order.reason or "it declares no order in it"
                continue
            if argument not in other:
                other[argument] = summed(1.0 / REFINEMENT["dt"], 1.0) if argument == "dt" else \
                    summed(1.0, REFINEMENT["tolerance"])
            coarse, fine = (base, other[argument]) if argument == "dt" else (other[argument], base)
            ratio = float(coarse[term]) / float(fine[term])
            checks.append(OrderCheck(term, argument, declared, ratio, observed_order(argument, ratio),
                                     ratio_band(argument, declared, order.at_least)))
    return OrderReport(tuple(checks), skipped, undeclared)


__all__ = ["ORDER_HALF_WIDTH", "REFINEMENT", "FlagReading", "FlagReport", "OrderCheck", "OrderReport", "Sums", "active_flags",
           "declared_flags", "flag_problems", "flag_readings", "judge_flags", "ledger_closes", "observed_order",
           "ratio_band", "run_flags", "term_orders"]
