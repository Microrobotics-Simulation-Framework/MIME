"""The energy ledger's account of a run (``EnergyLedger``), read from the metered receivers' own records at the run's
end: per role, in plain words, what is stored, lost, supplied (named by what supplies it) and held in each potential.

**Supplied, as the account states it, is taken from the dissipation side**: what was lost plus the change in what is
stored, each potential's included. It is an estimate under the ledger's exact identity, not a bound: it differs from
the paired supply by the scheme's own terms (1 % on script A, 28 % on a case at a coarse step), and which of the two is
nearer the continuum value is not established at that step. The work the laws supplied from outside paired in the receivers' steps (the paired
supply) stays inside the step identity's check, in the detailed account beside the scheme's own numerical terms,
which are never a loss: a load read at a step's start and paired with the step's turn departs from its limit at first
order in the step (on a body wobbling fast against its step the paired supply ran 35 % above its limit while the
dissipation side was within 2 % of it). Where several things outside supply, the dissipation side is shared among them
in proportion to what each one's laws paired.

**A potential's change** is taken from the run's two ends with the law's own parameters, as the run had them, at the
configuration its receiver kept at the run's start and the one it ended at; its residual, the law's paired work plus
that change, is zero to rounding where the law's force is evaluated where it is paired (gravity without a couple),
and holds the step's own first-order error otherwise (a couple paired with the step's turn): reported, not judged.

**Traced.** Every sum the account holds (``stored``, ``lost``, ``potentials``, ``supplied``, :attr:`EnergyAccount.
supplied_total` and :meth:`EnergyAccount.efficiency`) is formed from the records and the laws' own parameters by
arithmetic JAX traces, with no conversion to NumPy on the way, so that ``energy_account`` called inside a function
being differentiated (``jax.grad`` of a scan's end, parameters traced) is differentiable, the potentials' change
included. Under a trace the values are traced scalars; outside one they are Python floats, as before. What a trace
cannot hold is text (:meth:`EnergyAccount.text` formats numbers), the step counts (carried traced, not converted to
integers), and a shared ``supplied`` over a sum of paired work that is exactly zero (zero, where outside a trace it is
empty)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from mime.core import ledger

if TYPE_CHECKING:
    from mime.core import MimeNode


#: A quantity of the account in joules: a Python float, or the traced scalar it is while a function is being
#: differentiated or compiled.
Scalar = float | Array
#: A step count: an integer, or traced.
Count = int | Array


_TRACER: type = cast(Any, jax).core.Tracer


def _traced(value: Any) -> bool:
    return isinstance(value, _TRACER)


def _scalar(value: Any) -> Scalar:
    """``value`` as a float (float64), or itself where it is traced (no NumPy on a traced path)."""
    return cast(Array, value) if _traced(value) else float(np.asarray(value, dtype=np.float64))


def _count(value: Any) -> Count:
    return cast(Array, value) if _traced(value) else int(float(np.asarray(value, dtype=np.float64)))


def _sum(value: Any) -> Scalar:
    """The sum of the elements of ``value``, traced or not."""
    return jnp.sum(value) if _traced(value) else float(np.sum(np.asarray(value)))


def _taken(value: Any) -> Any:
    """A vector of joules, as float64 where concrete."""
    return value if _traced(value) else np.asarray(value, dtype=np.float64)


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
    work: Scalar
    lost: Scalar = 0.0
    supplied_by: tuple[str, ...] = ()
    potential_change: Scalar | None = None
    residual: Scalar | None = None
    residual_floor: Scalar | None = None
    host_term: Scalar | None = None
    gift_steps: Count = 0
    indefinite_steps: Count = 0

    def line(self) -> str:
        parts = [f"{self.law} on {self.receiver}: {self.energy_class}, paired work {self.work:+.4g} J"]
        if self.energy_class == ledger.DISSIPATIVE:
            parts.append(f"lost {self.lost:+.4g} J")
        if self.host_term is not None:
            parts.append(f"its host's projection's term {self.host_term:+.4g} J (the scheme's, apart)")
        if self.potential_change is not None and self.residual is not None:
            unjudged = ("; beyond it: the step's own first-order pairing, reported, not judged"
                        if abs(self.residual) > abs(self.residual_floor or 0.0) else "")
            parts.append(f"its energy's change {self.potential_change:+.4g} J, residual {self.residual:+.3g} J "
                         f"(rounding {self.residual_floor:.3g} J{unjudged})")
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
    finding; the jump of a receiver's storage between steps that no declared producer explains is judged in it, apart
    in ``between_worst``), and ``balance``, the receivers' books summed over the run, with ``balance_floor``, its
    rounding. ``findings`` are the checks that failed, in words, by receiver and with the amount: work that arrived
    unattributed, a step identity over its floor. They are what a run's text says before anything else."""

    stored: Mapping[str, Scalar]
    lost: Mapping[str, Scalar]
    potentials: Mapping[str, Scalar]
    supplied: Mapping[str, Scalar]
    laws: tuple[LawEnergy, ...]
    numerical: Mapping[str, Scalar]
    paired_supply: Mapping[str, Scalar]
    unattributed: Mapping[str, Scalar]
    identity_worst: Mapping[str, Scalar]
    balance: Scalar
    balance_floor: Scalar
    steps: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple[str, ...])
    #: What each receiver's storage jumped between steps, summed (J): where its frames carry a delivered inertia, the
    #: exchange's own (under ``exchanges``); inside a coupling group, the group's residual (under ``groups``); rounding
    #: elsewhere, and nothing else: a jump no declared producer explains is judged (``between_worst``).
    between: Mapping[str, Scalar] = field(default_factory=dict[str, Scalar])
    #: The worst jump of each receiver's own state's storage between steps, over the step identity's rounding floor
    #: (above one is a failure of the step identity, counted in ``identity_worst``). A member of a coupling group is
    #: not judged here: its jump is the group's.
    between_worst: Mapping[str, Scalar] = field(default_factory=dict[str, Scalar])
    #: The checks that failed, in words (see above); empty on a run whose books close.
    findings: tuple[str, ...] = field(default_factory=tuple[str, ...])
    #: Each workless law's paired work summed over its receivers, with the jump between steps of a storage its
    #: delivered inertia moves (J): the exchange's own discretisation error, reported, not judged.
    exchanges: Mapping[str, Scalar] = field(default_factory=dict[str, Scalar])
    #: Each coupling group by its members (``a+b``): MADDENING's verdict at the run's last step, and the group's
    #: residual, its members' jumps between steps summed.
    groups: Mapping[str, str] = field(default_factory=dict[str, str])
    #: Each node the build left unmetered though it receives deliveries (a flow the build does not yet meter), with
    #: its reason: what the account leaves out.
    unmetered: Mapping[str, str] = field(default_factory=dict[str, str])

    @property
    def supplied_total(self) -> Scalar:
        """What was supplied, from the dissipation side: the losses plus the change in what is stored, each
        potential's included. Differentiable: traced where the account was formed under a trace."""
        return _scalar(sum(self.lost.values()) + sum(self.stored.values()) + sum(self.potentials.values()))

    def efficiency(self, useful: Scalar) -> Scalar:
        """``useful`` work (J, the output a study names: a swimmer's towing work at its mean speed) as a share of what
        was supplied, the losses plus the change in what is stored (:attr:`supplied_total`). Differentiable in
        ``useful`` and in everything the account was formed from; ``nan`` where nothing was supplied."""
        total = self.supplied_total
        if _traced(total) or _traced(useful):
            return jnp.where(total == 0, jnp.nan, useful / jnp.where(total == 0, 1.0, total))
        return float(useful / total) if total else float("nan")

    def text(self, *, details: bool = False) -> str:
        """The account in plain words; with ``details``, each law on each receiver, the scheme's numerical terms, the
        paired supply and the checks besides."""

        def joules(held: Mapping[str, Scalar]) -> str:
            return ", ".join(f"{name} {value:+.4g} J" for name, value in held.items()) or "none"

        left_out = [f"  not metered, so not in the account: {name}: {reason}"
                    for name, reason in self.unmetered.items()]
        if not self.stored:
            return "\n".join([f"energy over {self.steps} steps from the build's start: no receiver is metered",
                              *left_out])
        lines = [f"energy over {self.steps} steps from the build's start:",
                 f"  stored: {joules(self.stored)}",
                 f"  lost: {joules(self.lost)}",
                 f"  supplied: {joules(self.supplied) if self.supplied else f'{self.supplied_total:+.4g} J'} "
                 f"(what was lost plus the change in what is stored)", *left_out]
        if self.potentials:
            lines.append(f"  potentials, each one's change: {joules(self.potentials)}")
        lines += [f"  FINDING: {finding}" for finding in self.findings]
        lines += [f"  {note}" for note in self.notes]
        if not details:
            return "\n".join([*lines, "  each law, the scheme's own terms and the checks: text(details=True)"])
        lines += [f"  scheme's own numerical terms (apart, never a loss): {joules(self.numerical)}",
                  f"  paired supply (the work supplied laws paired in the steps): {joules(self.paired_supply)}"]
        lines += [f"  {law.line()}" for law in self.laws]
        worst = ", ".join(f"{name} {value:.3g}" for name, value in self.identity_worst.items())
        lines += [f"  unattributed: {joules(self.unattributed)}",
                  f"  the step identity's worst step over its rounding floor: {worst}",
                  f"  the storage's jumps between steps: {joules(self.between)}; the worst one no declared producer "
                  f"owns, over the rounding floor: "
                  f"{', '.join(f'{name} {value:.3g}' for name, value in self.between_worst.items())}"]
        lines += [f"  {law}: workless, summed over its receivers {value:+.3g} J (its own discretisation error)"
                  for law, value in self.exchanges.items()]
        lines += [f"  coupling group {members}: {verdict}" for members, verdict in self.groups.items()]
        lines.append(f"  balance {self.balance:+.3g} J, rounding {self.balance_floor:.3g} J")
        return "\n".join(lines)


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
                      dtype: Any, hosted: bool = False) -> tuple[Scalar, Scalar] | None:
    """A potential law's energy's change over the run, from its own parameters as the run had them, at its receiver's
    configuration at the run's start (kept in its records) and at its end; and the energies' magnitude at the two ends,
    which their difference's rounding is judged on. A law the receiver hosts (a dynamic arm's weight) states its energy
    on the receiver's node. ``None`` where the law states no energy."""
    from mime.core import jnp

    start = final.get(f"{ledger.RECORDS}configuration_start")
    end = receiver.configuration(receiver.constants(params.get(receiver.name)), final)
    if start is None or end is None:
        return None
    changes: list[Scalar] = []
    sizes: list[Scalar] = []
    for node in [receiver] if hosted else _laws_nodes(graph, law):
        p = node.constants(params.get(node.name))
        try:
            ends = [node.energy(p, jnp.asarray(at, dtype=dtype)) for at in (start, end)]
        except NotImplementedError:
            continue
        changes.append(_scalar(ends[1]) - _scalar(ends[0]))
        sizes.append(abs(_scalar(ends[0])) + abs(_scalar(ends[1])))
    return (_scalar(sum(changes)), _scalar(sum(sizes))) if changes else None


#: The input of a field's law that reads the field itself, whose provider states the pair's energy (``energy_at``).
_FIELD_INPUT = "flux_density"


def _field_change(graph: Any, law: str, receiver: Any, final: Mapping[str, Mapping[str, Any]],
                  params: Mapping[str, Any], dtype: Any) -> tuple[Scalar, Scalar] | str | None:
    """The change over the run of the energy a field's law and its pair derive from, stated once for the pair by the
    field's provider (``energy_at``: ``-m . B`` at the body's magnet's centroid), between the run's two ends: the body's
    configuration at its start (its records) and its end, and the frames the field's source is mounted on at the start
    (the records of the segment that carries it) and at the end. Taken for the law that reads the field itself (the
    torque); ``None`` for a law that reads none, and the reason where the source's start is not recorded."""
    from mime.core import jnp

    edges = [e for e in graph.edges if e.target_node == law and e.target_field == _FIELD_INPUT]
    if len(edges) != 1:
        return None
    (edge,) = edges
    provider = graph.get_node(edge.source_node)
    if not hasattr(provider, "energy_at") or not edge.source_field.startswith(f"{_FIELD_INPUT}_"):
        return None
    slot = edge.source_field.removeprefix(f"{_FIELD_INPUT}_")
    pp = provider.constants(params.get(provider.name))
    mounts = [e for e in graph.edges if e.target_node == provider.name and e.target_field == "mount_poses"]
    frames: list[Any] = [None, None]
    if mounts:
        segment = graph.get_node(mounts[0].source_node)
        start = final.get(segment.name, {}).get(f"{ledger.RECORDS}configuration_start")
        if start is None or not hasattr(segment, "frame_poses_at"):
            return (f"its source's start is not recorded ({segment.name} is not metered), so its pair's energy's "
                    f"change is not in the account")
        sp = segment.constants(params.get(segment.name))
        frames = [segment.frame_poses_at(sp, jnp.asarray(start, dtype=dtype)),
                  jnp.asarray(final[segment.name]["frame_poses"], dtype=dtype)]
    own = graph.get_node(law)
    moment = own.constants(params.get(own.name))["moment"]
    body_start = final[receiver.name].get(f"{ledger.RECORDS}configuration_start")
    body_end = receiver.configuration(receiver.constants(params.get(receiver.name)), final[receiver.name])
    if body_start is None or body_end is None:
        return None
    ends = [provider.energy_at(pp, at_frames, slot, jnp.asarray(pose, dtype=dtype), moment)
            for at_frames, pose in zip(frames, (body_start, body_end), strict=True)]
    return _scalar(ends[1]) - _scalar(ends[0]), abs(_scalar(ends[0])) + abs(_scalar(ends[1]))


def energy_account(graph: Any, final: Mapping[str, Mapping[str, Any]], params: Mapping[str, Any] | None = None,
                   *, steps: int = 0, unmetered: Mapping[str, str] | None = None) -> EnergyAccount | None:
    """The account of a run of ``graph`` (built with ``EnergyLedger`` attached) that ended at ``final``, each law's
    parameters as the run had them (``params``, the parameter tree's nodes; the graph's own where not given), naming
    what the build left unmetered (``unmetered``, each node with its reason); ``None`` for a graph with no metered
    receiver and nothing unmetered."""
    nodes = dict(params or {})
    stored: dict[str, Scalar] = {}
    lost: dict[str, Scalar] = {}
    potentials: dict[str, Scalar] = {}
    numerical: dict[str, Scalar] = {}
    paired: dict[str, Scalar] = {}
    unattributed: dict[str, Scalar] = {}
    worst: dict[str, Scalar] = {}
    lines: list[LawEnergy] = []
    notes: list[str] = []
    balance: Scalar = 0.0
    floor: Scalar = 0.0
    between: dict[str, Scalar] = {}
    between_worst: dict[str, Scalar] = {}
    exchanges: dict[str, Scalar] = {}
    findings: list[str] = []
    paired_fields: set[str] = set()
    field_laws: set[str] = set()
    verdicts = group_verdicts(graph)
    grouped = {member for members in verdicts for member in members.split("+")}
    for name in cast(list[str], graph.node_names):
        receiver = graph.get_node(name)
        held = ledger.deliveries(dict(getattr(receiver, "params", {}) or {}))
        if not held:
            continue
        records = final[name]
        dtype = records[f"{ledger.RECORDS}work"].dtype
        eps = float(np.finfo(dtype).eps)
        stored[name] = _scalar(ledger.stored(records))
        numerical[name] = _sum(records[f"{ledger.RECORDS}terms"])
        unattributed[name] = _scalar(records[f"{ledger.RECORDS}unattributed"])
        worst[name] = _scalar(records[f"{ledger.RECORDS}identity_worst"]) / (ledger.IDENTITY_ULPS * eps)
        between_worst[name] = _scalar(ledger.between_floors(records))
        if name not in grouped:
            worst[name] = jnp.maximum(worst[name], between_worst[name]) if _traced(worst[name]) or _traced(
                between_worst[name]) else max(worst[name], between_worst[name])
        balance += _scalar(ledger.balance(records, grouped=name in grouped))
        floor += _scalar(ledger.balance_floor(records))
        between[name] = _scalar(records[f"{ledger.RECORDS}between"])
        carrying = next((d.law for d in held if d.kind == ledger.INERTIA), None)
        if carrying is not None:
            # The jump the delivered inertia's exchange owns: the part of the storage that reads the delivery.
            exchanges[carrying] = exchanges.get(carrying, 0.0) + between[name] \
                - _scalar(records[f"{ledger.RECORDS}between_own"])
        if not _traced(worst[name]) and not _traced(unattributed[name]):
            findings += _findings(name, records, worst[name], between_worst[name], unattributed[name],
                                  grouped=name in grouped)
        scale = _scalar(records[f"{ledger.RECORDS}scale"])
        taken = _taken(ledger.losses(records, held))
        for row, law in enumerate(ledger.laws(held)):
            first = next(d for d in held if d.law == law)
            work = _scalar(records[f"{ledger.RECORDS}work"][row])
            line = LawEnergy(law, name, first.energy_class, work, _scalar(taken[row]),
                             tuple(first.supplied_by.split(",")) if first.supplied_by else (),
                             gift_steps=_count(records[f"{ledger.RECORDS}gift_steps"][row]),
                             indefinite_steps=_count(records[f"{ledger.RECORDS}indefinite_steps"][row]))
            if first.kind == ledger.HOSTED:
                host = _scalar(records[f"{ledger.RECORDS}hosted"][row, 1])
                line = _replaced(line, host_term=host)
                numerical[f"{law} (its host's projection)"] = host
            if first.energy_class == ledger.DISSIPATIVE:
                lost[law] = lost.get(law, 0.0) + line.lost
            if first.energy_class == ledger.WORKLESS:
                exchanges[law] = exchanges.get(law, 0.0) + work
            if first.energy_class == ledger.POTENTIAL and first.supplied_by:
                notes.append(f"{law} on {name} reads {', '.join(line.supplied_by)}, outside the components: its "
                             f"paired work is supplied by it, and its energy's change is not the components'")
            elif first.energy_class == ledger.POTENTIAL and law not in potentials and (
                    first.kind == ledger.HOSTED or any(e.source_node == law and e.target_node == name
                                                       for e in graph.edges)):
                found: tuple[Scalar, Scalar] | str | None = _potential_change(
                    graph, law, receiver, records, nodes, dtype, hosted=first.kind == ledger.HOSTED)
                if found is None and first.kind != ledger.HOSTED:
                    found = _field_change(graph, law, receiver, final, nodes, dtype)
                    if found is None:
                        paired_fields.add(law)
                    elif not isinstance(found, str):
                        field_laws.add(law)
                if isinstance(found, str):
                    notes.append(f"{law}: {found}")
                elif found is not None:
                    change, size = found
                    potentials[law] = potentials.get(law, 0.0) + change
                    line = _replaced(line, potential_change=change, residual=work + change,
                                     residual_floor=ledger.IDENTITY_ULPS * eps * (scale + size))
            for producer in line.supplied_by:
                paired[producer] = paired.get(producer, 0.0) + work / len(line.supplied_by)
            if first.energy_class == ledger.UNMETERED and not line.supplied_by:
                notes.append(f"{law} on {name} declares no class and is supplied by nothing outside: its paired work, "
                             f"{work:+.4g} J, is booked under its name and is neither a loss nor a supply: it is in "
                             f"{name}'s stored change, and where the law has another half on another receiver, the "
                             f"two halves' sum is the coupling's first-order error, reported, not judged")
            lines.append(line)
    left_out = dict(unmetered or {})
    if not stored:
        return EnergyAccount({}, {}, {}, {}, (), {}, {}, {}, {}, 0.0, 0.0, steps, unmetered=left_out) if left_out \
            else None
    for law in sorted(paired_fields - set(potentials)):
        if field_laws:
            notes.append(f"{law}'s energy is its pair's, stated once by its field and taken under "
                         f"{', '.join(sorted(field_laws))}")
        else:
            notes.append(f"{law} states no energy on its node, so its change is not in the account")
    groups: dict[str, str] = {}
    for members, state in verdicts.items():
        residual = sum(between.get(member, 0.0) for member in members.split("+"))
        groups[members] = f"MADDENING's verdict at the run's last step: {state}; its residual, its members' " \
                          f"storage jumps between steps summed, {residual:+.3g} J"
    total = sum(lost.values()) + sum(stored.values()) + sum(potentials.values())
    supplied = _shared(total, paired)
    return EnergyAccount(stored, lost, potentials, supplied, tuple(lines), numerical, paired, unattributed, worst,
                         balance, floor, steps, tuple(notes), between, between_worst, tuple(findings), exchanges,
                         groups, left_out)


def _findings(name: str, records: Mapping[str, Any], identity: Scalar, between_floors: Scalar, summed: Scalar,
              *, grouped: bool) -> list[str]:
    """The checks a metered receiver failed over the run, in words, with the amount. Work that arrived in an
    unattributed input (above zero: exactly zero where every delivery is a law's, whatever the rounding floor) is
    named first, the step identity over its floor after it, with the jump between steps no declared producer owns
    where that is what failed it."""
    found: list[str] = []
    worst = abs(float(np.asarray(records[f"{ledger.RECORDS}unattributed_worst"], dtype=np.float64)))
    if worst > 0.0:
        found.append(f"work arrived on {name} that no law or writer owns: {float(cast(float, summed)):+.4g} J summed "
                     f"over the run, {worst:.3g} J at the largest step (the books' rounding is "
                     f"{float(cast(float, _scalar(ledger.balance_floor(records)))):.3g} J, and this is exactly zero "
                     f"where every delivery is a law's). The identity and the books close with it, under no name: an "
                     f"edge the build did not declare, or a controller called on the host (mode 'callback'), whose "
                     f"writes it cannot attribute")
    if float(cast(float, identity)) > 1.0:
        owned = " by the jump of its storage between steps, which no declared producer owns" \
            if not grouped and float(cast(float, between_floors)) > 1.0 else ""
        found.append(f"the step identity of {name} failed{owned}: its worst step is "
                     f"{float(cast(float, identity)):.3g} times its rounding floor")
    return found


def group_verdicts(graph: Any) -> dict[str, str]:
    """Each coupling group's verdict at the last step a run of ``graph`` took, in words, by its members (``a+b``):
    MADDENING's own (``converged``, and the passes it took); a group that stopped at its cap is said so. The run
    reports it (``RunResult.groups``) in every profile, with or without the energy ledger: energy cannot tell a capped
    group from a converged one."""
    found: dict[str, str] = {}
    for members, verdict in _diagnostics(graph).items():
        passes = int(verdict.get("iterations", 0))
        found[members] = ("converged" if verdict.get("converged", True) else
                          f"not converged at its last step after {passes} passes: it stopped at its cap")
    return found


def _diagnostics(graph: Any) -> dict[str, dict[str, Any]]:
    """MADDENING's coupling verdicts at the last step a run took, by group (``a+b``); none for a graph without groups
    or one that has not run."""
    try:
        found = graph.coupling_diagnostics()
    except Exception:  # noqa: BLE001  (a graph with no group, or none run, has no verdict to read)
        return {}
    return {str(key): {k: v for k, v in dict(value).items() if k in ("converged", "iterations", "residual")}
            for key, value in dict(found or {}).items()}


def _shared(total: Scalar, paired: Mapping[str, Scalar]) -> dict[str, Scalar]:
    """``total`` shared among the things outside in proportion to the work each one's laws paired. **A share is a
    ratio of paired works, so it is unbounded where they nearly cancel**: two things outside supplying work of opposite
    sign, or a sum near zero, give shares far outside 0 to 1 (script A has one thing outside, whose share is 1). Where
    the paired work sums to zero exactly the account shares nothing: empty outside a trace, zero under one."""
    whole = sum(paired.values())
    if _traced(whole) or _traced(total):
        zero = whole == 0
        return {producer: jnp.where(zero, 0.0, total * share / jnp.where(zero, 1.0, whole))
                for producer, share in paired.items()}
    return {producer: total * share / whole for producer, share in paired.items()} if whole else {}


def _replaced(line: LawEnergy, **changed: Any) -> LawEnergy:
    from dataclasses import replace

    return replace(line, **changed)


__all__ = ["EnergyAccount", "LawEnergy", "energy_account", "group_verdicts"]
