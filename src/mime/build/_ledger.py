"""The energy ledger's wiring, which the build does only when an observation that meters energy is attached
(``EnergyLedger``): without one, every program is built as before, edge for edge and node for node.

With the ledger attached:

- **outside is worked out per thing** (F0-1 M3): a thing (an apparatus, a body, a medium's flow), taken from each
  entry's plan, none of whose nodes receives a delivery or a reaction from the components is outside, and the work a
  law reading it does on the components is supplied, named by that thing. Nothing is declared;
- **each law delivers into its own input on its receiver.** A receiver is a dynamics' node that writes its step
  identity (``MimeNode.LEDGER_TERMS``) and names its port kind (``Dynamics.port``). Every delivery edge into it is moved
  from the input it sums into (``applied_wrench``) to the law's own (``mime.core.ledger.law_input``), the summed input
  kept for anything unattributed; each delivery is paired as its port kind says of the interface it is delivered into;
  a confinement is held in the receiver's step, under the confining law's id. The receiver is constructed again from
  its own arguments with its ``ledger``: no MADDENING change. Within each input the laws are held in the order of
  their ids, the order the edges into a summed input are added in, so the receiver sums them as the graph did;
- **the boundary is reported**, the build report's ``energy`` lines: each metered receiver and each law's class (or
  its reason for being unmetered), what each supplied law reads from outside, what is outside, and each receiver the
  ledger cannot meter;
- **an entry that declares nothing** (a law with no class that reads nothing from outside, a dynamics receiving
  deliveries with no port kind) and states no reason (``unmetered``) is logged under exploratory and refused under
  research (``ledger.unmetered``); strict, a1's, refuses every unmetered entry, reason or not: :func:`refuses_unmetered`
  is where it does."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from maddening.core.node import SimulationNode

from mime.build._builders import DELIVERY, FIELD, REACTION, STATE, EdgeDecl
from mime.build._plan import EntryPlan, PlanSet, Section
from mime.build._refusals import Refusal, RefusalClass, entry_defect
from mime.core import ledger
from mime.registry import Dynamics, Interaction, Observation
from mime.validity import Profile
from mime.vocabulary import CONFINEMENT, Definiteness, port_kind

#: The sections whose entries produce: what may be outside.
_PRODUCERS = (Section.DYNAMICS, Section.PROVIDER, Section.DRIVE)


@dataclasses.dataclass(frozen=True)
class EnergyBoundary:
    """What the ledger's wiring found: each metered receiver's deliveries in the order its records keep them, the
    things outside, each law that reads one with what it reads, and the report's lines."""

    receivers: Mapping[str, tuple[ledger.LedgerDelivery, ...]]
    outside: tuple[str, ...]
    supplied_by: Mapping[str, tuple[str, ...]]
    lines: tuple[str, ...]


def ledger_attached(plans: PlanSet) -> bool:
    """Whether an observation that meters energy (``Observation.meters_energy``) is attached."""
    return any(plan.section is Section.OBSERVATION and isinstance(plan.entry, Observation)
               and type(plan.entry).meters_energy for plan in plans.entries)


def refuses_unmetered(profile: Profile, reason: str) -> bool:
    """Whether a build under ``profile`` refuses an entry the ledger cannot meter, given its stated ``reason`` (empty
    for none): exploratory logs it, research needs a reason. Strict (a1's) refuses every unmetered entry, reason or
    not, here."""
    return profile is not Profile.EXPLORATORY and not reason


def _owners(plans: PlanSet) -> dict[str, EntryPlan]:
    return {name: plan for plan in plans.entries for name in plan.node_names()}


def _thing(owners: Mapping[str, EntryPlan], node: str) -> str | None:
    """The thing a node belongs to, as its entry's plan states it: a producer's (a dynamics, a provider, a drive),
    never a law's or an observation's."""
    plan = owners.get(node)
    return plan.thing if plan is not None and plan.section in _PRODUCERS else None


def _class_line(law: Interaction | None) -> str:
    if law is None or law.energy_class is None:
        return ledger.UNMETERED
    return law.energy_class.value


def _unmetered_reason(entry: object) -> str:
    return str(getattr(entry, "unmetered", "") or "")


def attach(plans: PlanSet, built: dict[str, SimulationNode],
           edges: list[EdgeDecl]) -> tuple[dict[str, SimulationNode], list[EdgeDecl], EnergyBoundary, list[Refusal]]:
    """The graph's nodes and edges with the ledger attached, the boundary it found, and its refusals (an unmetered
    entry under research, a port kind a metered dynamics names that does not pair what it is delivered)."""
    owners = _owners(plans)
    laws = {plan.owner: plan for plan in plans.entries if plan.section is Section.INTERACTION}
    producing = {plan.thing for plan in plans.entries if plan.section in _PRODUCERS and plan.thing}
    receiving = {thing for e in edges if e.carries in (DELIVERY, REACTION)
                 for thing in (_thing(owners, e.target[0]),) if thing is not None}
    outside = tuple(sorted(producing - receiving))
    supplied_by: dict[str, tuple[str, ...]] = {}
    for by in laws:
        read = {thing for e in edges if e.by == by and e.carries in (STATE, FIELD)
                for thing in (_thing(owners, e.source[0]),) if thing is not None and thing in outside}
        if read:
            supplied_by[by] = tuple(sorted(read))
    profile = plans.profile.profile
    refusals: list[Refusal] = []
    lines: list[str] = []
    receivers: dict[str, tuple[ledger.LedgerDelivery, ...]] = {}
    moved: dict[int, tuple[str, str]] = {}
    for plan in plans.entries:
        if plan.section is not Section.DYNAMICS or not isinstance(plan.entry, Dynamics):
            continue
        dynamics = plan.entry
        for name in plan.node_names():
            into = sorted({(e.target[1], e.by) for e in edges if e.carries in (DELIVERY, REACTION)
                           and e.target[0] == name})
            if not into and not plan.confinements:
                continue
            node = built[name]
            if not dynamics.port or getattr(type(node), "LEDGER_TERMS", None) is None:
                reason = _unmetered_reason(dynamics)
                lines.append(f"{name}: receives deliveries ({', '.join(by for _, by in into)}) and is not metered: "
                             + (reason or "its dynamics declares no port kind and states no reason"))
                if refuses_unmetered(profile, reason):
                    refusals.append(_unmetered_refusal(plan.owner, f"the dynamics of {plan.owner} "
                                                       f"({type(dynamics).label()})", "a port kind (Dynamics.port)"))
                continue
            held, problems = _deliveries(plan, dynamics, name, into, laws, supplied_by)
            refusals += [entry_defect(type(dynamics), problem, context=f", metering {name}") for problem in problems]
            if problems:
                continue
            index_of = {(d.field, d.law): index for index, d in enumerate(held) if d.kind != ledger.HOSTED}
            for position, edge in enumerate(edges):
                if edge.carries in (DELIVERY, REACTION) and edge.target[0] == name:
                    moved[position] = (name, ledger.law_input(edge.target[1], index_of[edge.target[1], edge.by]))
            built[name] = type(node)(node.name, float(node.delta_t), **dict(getattr(node, "params")),
                                     **{ledger.LEDGER: tuple(d.encoded() for d in held)})
            receivers[name] = held
            lines += _receiver_lines(name, dynamics, held, laws)
    for by, plan in laws.items():
        law = plan.entry
        if not isinstance(law, Interaction) or law.energy_class is not None or by in supplied_by:
            continue
        reason = _unmetered_reason(law)
        if not reason:
            lines.append(f"{by}: declares no energy class, reads nothing from outside and states no reason")
        if refuses_unmetered(profile, reason):
            refusals.append(_unmetered_refusal(by, f"{by} ({type(law).label()})",
                                               "an energy class (Interaction.energy_class)"))
    lines.append(f"outside: {', '.join(outside) or 'nothing'} (receives no delivery from the components)")
    rewired = [dataclasses.replace(e, target=moved[position]) if position in moved else e
               for position, e in enumerate(edges)]
    return built, rewired, EnergyBoundary(receivers, outside, supplied_by, tuple(lines)), refusals


def _deliveries(plan: EntryPlan, dynamics: Dynamics, name: str, into: Sequence[tuple[str, str]],
                laws: Mapping[str, EntryPlan], supplied_by: Mapping[str, tuple[str, ...]]
                ) -> tuple[tuple[ledger.LedgerDelivery, ...], list[str]]:
    """A metered receiver's deliveries, each law's in the order of its id within each input, its hosted law last; and
    what keeps its port kind from pairing them."""
    kind = port_kind(dynamics.port)
    if kind is None:
        return (), [f"it names the port kind {dynamics.port!r}, which is not registered (mime.vocabulary.port_kinds)"]
    local = name.removeprefix(plan.owner).removeprefix(".")
    interface_of = {port.field: interface for interface, port in dynamics.accepts.items() if port.node == local}
    held: list[ledger.LedgerDelivery] = []
    problems: list[str] = []
    for field, by in into:
        interface = interface_of.get(field)
        paired = kind.deliveries.get(interface) if interface is not None else None
        if paired is None:
            problems.append(f"its port kind {kind.name!r} does not say how {field!r} ({interface}) is paired, and "
                            f"{by} delivers into it")
            continue
        law = laws[by].entry if by in laws else None
        interaction = law if isinstance(law, Interaction) else None
        check = ledger.CHECK_DEFINITE if interaction is not None and \
            interaction.definiteness() is Definiteness.STATE else ""
        held.append(ledger.LedgerDelivery(field, by, _class_line(interaction), paired.value, check,
                                          ",".join(supplied_by.get(by, ()))))
    for confinement in plan.confinements:
        if CONFINEMENT.name not in kind.deliveries:
            problems.append(f"its port kind {kind.name!r} takes no confinement, and {confinement.id} confines it")
            continue
        law = laws[confinement.id].entry if confinement.id in laws else None
        held.append(ledger.LedgerDelivery("confinement", confinement.id,
                                          _class_line(law if isinstance(law, Interaction) else None), ledger.HOSTED))
    return tuple(held), problems


def _receiver_lines(name: str, dynamics: Dynamics, held: Sequence[ledger.LedgerDelivery],
                    laws: Mapping[str, EntryPlan]) -> list[str]:
    lines = [f"{name}: metered ({dynamics.port} port), its step identity and each law's paired work recorded in its "
             f"step"]
    for law in ledger.laws(held):
        mine = [d for d in held if d.law == law]
        first = mine[0]
        how = "held in its step" if first.kind == ledger.HOSTED else ", ".join(d.field for d in mine)
        notes = [f"supplied by {first.supplied_by.replace(',', ', ')}"] if first.supplied_by else []
        if first.check == ledger.CHECK_DEFINITE:
            notes.append("its resistance checked definite at each step")
        entry = laws[law].entry if law in laws else None
        if first.energy_class == ledger.UNMETERED and _unmetered_reason(entry):
            notes.append(f"unmetered: {_unmetered_reason(entry)}")
        lines.append(f"  {law}: {first.energy_class} ({how})" + "".join(f"; {note}" for note in notes))
    return lines


def _unmetered_refusal(owner: str, what: str, declares: str) -> Refusal:
    return Refusal("ledger.unmetered", RefusalClass.USER,
                   f"{what} declares no {declares} and states no reason for being unmetered, and the research profile "
                   f"needs one where the energy ledger is attached",
                   f"build under the exploratory profile, which logs it (Experiment(..., profile=\"exploratory\")), "
                   f"or without exp.observe(EnergyLedger()), or attach in place of {owner} an entry that declares "
                   f"one; its author states its reason in its class's unmetered = \"...\"")


def misattributed(graph_edges: Sequence[tuple[str, str, str, str]], receivers: Mapping[str, Any],
                  owner_of: Any) -> list[str]:
    """Each delivery the graph holds into a metered receiver's input that is not its law's own: an edge into a law's
    own input from a node another entry owns (two laws' loads swapped between their inputs), or an edge from a law
    the receiver holds into another input of its kind (its own, or the summed one, where it would arrive
    unattributed). ``receivers`` maps each metered node to its deliveries; ``owner_of`` gives a node's entry id."""
    found: list[str] = []
    for source, source_field, target, field in graph_edges:
        held = receivers.get(target)
        if not held:
            continue
        own = ledger.own_input(field)
        owner = owner_of(source)
        if own is not None:
            base, index = own
            expected = held[index].law if index < len(held) else None
            if expected != owner or held[index].field != base:
                found.append(f"{source}.{source_field} -> {target}.{field}: {target}'s input for {expected} "
                             f"({held[index].field if expected else 'none'}) takes a delivery from {owner}")
        elif any(d.law == owner and d.field == field for d in held):
            found.append(f"{source}.{source_field} -> {target}.{field}: {owner}'s delivery arrives in the summed "
                         f"input, where its work is unattributed")
    return found


__all__ = ["EnergyBoundary", "attach", "ledger_attached", "misattributed", "refuses_unmetered"]
