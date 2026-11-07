"""The energy ledger's account of a run (``EnergyLedger``), read from the metered receivers' own records at the run's
end: per role, in plain words, what is stored, lost, supplied (named by what supplies it) and held in each potential.

**Supplied, as the account states it, is taken from the dissipation side**: what was lost plus the change in what is
stored, each potential's included. The work the laws supplied from outside paired in the receivers' steps (the paired
supply) stays inside the step identity's check, in the detailed account beside the scheme's own numerical terms,
which are never a loss: a load read at a step's start and paired with the step's turn departs from its limit at first
order in the step (on a body wobbling fast against its step the paired supply ran 35 % above its limit while the
dissipation side was within 2 % of it). Where several things outside supply, the dissipation side is shared among them
in proportion to what each one's laws paired.

**A potential's change** is taken from the run's two ends with the law's own parameters, as the run had them, at the
configuration its receiver kept at the run's start and the one it ended at; its residual, the law's paired work plus
that change, is zero to rounding where the law's force is evaluated where it is paired (gravity without a couple),
and holds the step's own first-order error otherwise (a couple paired with the step's turn): reported, not judged."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mime.core import ledger

if TYPE_CHECKING:
    from mime.core import MimeNode


@dataclass(frozen=True)
class LawEnergy:
    """One law on one receiver over the run, in joules: its class, its paired work, what it took from the motion
    (``lost``, a dissipative law's), what supplies it from outside, and, for a potential law, its energy's change and
    the residual of its paired work against it with that residual's rounding floor; for a law held in its receiver's
    step, its host's projection's term (``host_term``, the scheme's, apart); and the steps its class's run-time checks
    counted (``gift_steps``, a dissipative law's work positive beyond rounding; ``indefinite_steps``, an operator
    checked at the state that failed)."""

    law: str
    receiver: str
    energy_class: str
    work: float
    lost: float = 0.0
    supplied_by: tuple[str, ...] = ()
    potential_change: float | None = None
    residual: float | None = None
    residual_floor: float | None = None
    host_term: float | None = None
    gift_steps: int = 0
    indefinite_steps: int = 0

    def line(self) -> str:
        parts = [f"{self.law} on {self.receiver}: {self.energy_class}, paired work {self.work:+.4g} J"]
        if self.energy_class == ledger.DISSIPATIVE:
            parts.append(f"lost {self.lost:+.4g} J")
        if self.host_term is not None:
            parts.append(f"its host's projection's term {self.host_term:+.4g} J (the scheme's, apart)")
        if self.potential_change is not None and self.residual is not None:
            parts.append(f"its energy's change {self.potential_change:+.4g} J, residual {self.residual:+.3g} J "
                         f"(rounding {self.residual_floor:.3g} J)")
        if self.supplied_by:
            parts.append(f"supplied by {', '.join(self.supplied_by)}")
        if self.gift_steps:
            parts.append(f"positive in {self.gift_steps} steps")
        if self.indefinite_steps:
            parts.append(f"indefinite in {self.indefinite_steps} steps")
        return ", ".join(parts)


@dataclass(frozen=True)
class EnergyAccount:
    """A run's energy, per role (the build's ids), in joules, from the build's start to the run's end.

    ``stored`` is each receiver's change in what it stores; ``lost`` each dissipative law's loss; ``potentials`` each
    potential law's energy's change; ``supplied`` what each thing outside supplied, taken from the dissipation side.
    The detailed account: ``laws`` (each law on each receiver), ``numerical`` (each receiver's scheme's own numerical
    terms, never a loss), ``paired_supply`` (the work each thing outside's laws paired in the receivers' steps),
    ``unattributed`` (the work each receiver's unattributed inputs paired: zero, exactly, where every delivery is a
    law's), ``identity_worst`` (each receiver's step identity's worst step over its rounding floor: above one is a
    finding), and ``balance``, the receivers' books summed over the run, with ``balance_floor``, its rounding."""

    stored: Mapping[str, float]
    lost: Mapping[str, float]
    potentials: Mapping[str, float]
    supplied: Mapping[str, float]
    laws: tuple[LawEnergy, ...]
    numerical: Mapping[str, float]
    paired_supply: Mapping[str, float]
    unattributed: Mapping[str, float]
    identity_worst: Mapping[str, float]
    balance: float
    balance_floor: float
    steps: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple[str, ...])

    @property
    def supplied_total(self) -> float:
        """What was supplied, from the dissipation side: the losses plus the change in what is stored, each
        potential's included."""
        return float(sum(self.lost.values()) + sum(self.stored.values()) + sum(self.potentials.values()))

    def efficiency(self, useful: float) -> float:
        """``useful`` work (J, the output a study names: a swimmer's towing work at its mean speed) as a share of what
        was supplied, the losses plus the change in what is stored (:attr:`supplied_total`)."""
        total = self.supplied_total
        return float(useful / total) if total else float("nan")

    def text(self, *, details: bool = False) -> str:
        """The account in plain words; with ``details``, each law on each receiver, the scheme's numerical terms, the
        paired supply and the checks besides."""

        def joules(held: Mapping[str, float]) -> str:
            return ", ".join(f"{name} {value:+.4g} J" for name, value in held.items()) or "none"

        lines = [f"energy over {self.steps} steps from the build's start:",
                 f"  stored: {joules(self.stored)}",
                 f"  lost: {joules(self.lost)}",
                 f"  supplied: {joules(self.supplied) if self.supplied else f'{self.supplied_total:+.4g} J'} "
                 f"(what was lost plus the change in what is stored)"]
        if self.potentials:
            lines.append(f"  potentials, each one's change: {joules(self.potentials)}")
        lines += [f"  {note}" for note in self.notes]
        if not details:
            return "\n".join([*lines, "  each law, the scheme's own terms and the checks: text(details=True)"])
        lines += [f"  scheme's own numerical terms (apart, never a loss): {joules(self.numerical)}",
                  f"  paired supply (the work supplied laws paired in the steps): {joules(self.paired_supply)}"]
        lines += [f"  {law.line()}" for law in self.laws]
        worst = ", ".join(f"{name} {value:.3g}" for name, value in self.identity_worst.items())
        lines += [f"  unattributed: {joules(self.unattributed)}",
                  f"  the step identity's worst step over its rounding floor: {worst}",
                  f"  balance {self.balance:+.3g} J, rounding {self.balance_floor:.3g} J"]
        return "\n".join(lines)


def _float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _laws_nodes(graph: Any, law: str) -> list[MimeNode]:
    """The law's nodes that state an energy (a potential law's): its node named by its id, or by its id and a local
    name."""
    from mime.core import MimeNode

    found: list[MimeNode] = []
    for name in cast(list[str], graph.node_names):
        node: object = graph.get_node(name)
        if (name == law or name.startswith(f"{law}.")) and isinstance(node, MimeNode) and \
                type(node).energy is not MimeNode.energy:
            found.append(node)
    return found


def _potential_change(graph: Any, law: str, receiver: Any, final: Mapping[str, Any], params: Mapping[str, Any],
                      dtype: Any) -> tuple[float, float] | None:
    """A potential law's energy's change over the run, from its own parameters as the run had them, at its receiver's
    configuration at the run's start (kept in its records) and at its end; and the energies' magnitude at the two ends,
    which their difference's rounding is judged on. ``None`` where the law states no energy."""
    from mime.core import jnp

    start = final.get(f"{ledger.RECORDS}configuration_start")
    end = receiver.configuration(receiver.constants(params.get(receiver.name)), final)
    if start is None or end is None:
        return None
    changes: list[float] = []
    sizes: list[float] = []
    for node in _laws_nodes(graph, law):
        p = node.constants(params.get(node.name))
        try:
            ends = [node.energy(p, jnp.asarray(at, dtype=dtype)) for at in (start, end)]
        except NotImplementedError:
            continue
        changes.append(_float(ends[1]) - _float(ends[0]))
        sizes.append(abs(_float(ends[0])) + abs(_float(ends[1])))
    return (sum(changes), sum(sizes)) if changes else None


def energy_account(graph: Any, final: Mapping[str, Mapping[str, Any]], params: Mapping[str, Any] | None = None,
                   *, steps: int = 0) -> EnergyAccount | None:
    """The account of a run of ``graph`` (built with ``EnergyLedger`` attached) that ended at ``final``, each law's
    parameters as the run had them (``params``, the parameter tree's nodes; the graph's own where not given); ``None``
    for a graph with no metered receiver."""
    nodes = dict(params or {})
    stored: dict[str, float] = {}
    lost: dict[str, float] = {}
    potentials: dict[str, float] = {}
    numerical: dict[str, float] = {}
    paired: dict[str, float] = {}
    unattributed: dict[str, float] = {}
    worst: dict[str, float] = {}
    lines: list[LawEnergy] = []
    notes: list[str] = []
    balance = floor = 0.0
    for name in cast(list[str], graph.node_names):
        receiver = graph.get_node(name)
        held = ledger.deliveries(dict(getattr(receiver, "params", {}) or {}))
        if not held:
            continue
        records = final[name]
        dtype = np.asarray(records[f"{ledger.RECORDS}work"]).dtype
        eps = float(np.finfo(dtype).eps)
        stored[name] = _float(ledger.stored(records))
        numerical[name] = _float(np.sum(np.asarray(records[f"{ledger.RECORDS}terms"])))
        unattributed[name] = _float(records[f"{ledger.RECORDS}unattributed"])
        worst[name] = _float(records[f"{ledger.RECORDS}identity_worst"]) / (ledger.IDENTITY_ULPS * eps)
        balance += _float(ledger.balance(records))
        floor += _float(ledger.balance_floor(records))
        scale = _float(records[f"{ledger.RECORDS}scale"])
        taken = np.asarray(ledger.losses(records, held), dtype=np.float64)
        for row, law in enumerate(ledger.laws(held)):
            first = next(d for d in held if d.law == law)
            work = _float(records[f"{ledger.RECORDS}work"][row])
            line = LawEnergy(law, name, first.energy_class, work, float(taken[row]),
                             tuple(first.supplied_by.split(",")) if first.supplied_by else (),
                             gift_steps=int(_float(records[f"{ledger.RECORDS}gift_steps"][row])),
                             indefinite_steps=int(_float(records[f"{ledger.RECORDS}indefinite_steps"][row])))
            if first.kind == ledger.HOSTED:
                line = _replaced(line, host_term=_float(records[f"{ledger.RECORDS}hosted"][row, 1]))
                numerical[f"{law} (its host's projection)"] = line.host_term or 0.0
            if first.energy_class == ledger.DISSIPATIVE:
                lost[law] = lost.get(law, 0.0) + line.lost
            if first.energy_class == ledger.POTENTIAL:
                found = _potential_change(graph, law, receiver, records, nodes, dtype)
                if found is None:
                    notes.append(f"{law} states no energy on its node, so its change is not in the account")
                else:
                    change, size = found
                    potentials[law] = potentials.get(law, 0.0) + change
                    line = _replaced(line, potential_change=change, residual=work + change,
                                     residual_floor=ledger.IDENTITY_ULPS * eps * (scale + size))
            for producer in line.supplied_by:
                paired[producer] = paired.get(producer, 0.0) + work / len(line.supplied_by)
            if first.energy_class == ledger.UNMETERED and not line.supplied_by:
                notes.append(f"{law} on {name} declares no class and is supplied by nothing outside: its paired work, "
                             f"{work:+.4g} J, is in neither what was lost nor what was supplied")
            lines.append(line)
    if not stored:
        return None
    total = sum(lost.values()) + sum(stored.values()) + sum(potentials.values())
    whole = sum(paired.values())
    supplied = {producer: total * share / whole for producer, share in paired.items()} if whole else {}
    return EnergyAccount(stored, lost, potentials, supplied, tuple(lines), numerical, paired, unattributed, worst,
                         balance, floor, steps, tuple(notes))


def _replaced(line: LawEnergy, **changed: Any) -> LawEnergy:
    from dataclasses import replace

    return replace(line, **changed)


__all__ = ["EnergyAccount", "LawEnergy", "energy_account"]
