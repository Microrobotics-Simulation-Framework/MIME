"""The energy ledger's wiring, which the build does only when an observation that meters energy is attached
(``EnergyLedger``): without one, every program is built as before, edge for edge and node for node.

With the ledger attached:

- **outside is worked out per thing**: a thing (a body, a medium's flow, a field), taken from each entry's
  plan, none of whose nodes receives a delivery or a reaction from the components is outside, and the work a law
  reading it does on the components is supplied, named by that thing. **An apparatus is taken by its mount tree**: each
  chain segment is outside unless it receives a delivery or a reaction, and each field it carries unless the laws
  reading it deliver their reactions into a segment of it (a magnet on a dynamic arm, whose field's change the arm's
  paired reaction pays, is not outside; a magnet on a kinematic one is), so that an apparatus holding a dynamic and a
  kinematic segment is outside in its kinematic part alone. A drive and a controller receive nothing, and are outside;
  a command they write into a metered receiver is supplied work, named by its writer. **A law is a source outside the
  components where it declares no class, no read it declares is of a component's state, and it delivers only loads its
  receivers pair** (a body force driving a metered flow, whose node computes its waveform from its own parameters):
  what it delivers is supplied work, **named by the law itself**, also where it reads a unit outside, and the law is
  listed outside (a class-less law that reads a component is supplied by the units outside it reads, as any law is).
  Every read the law declares counts, the points a field is read at among them (a field read at a body's pose reads the
  body); and only a load, whose work does not depend on the components where its node reads none of them (a resistance
  is paired with the receiver's own solution; a delivery into a receiver the ledger does not meter is paired as
  nothing). The rule is for laws that declare no class, and only those: a class says how a law's work is booked, so a
  law that declares one keeps it whatever its node reads (gravity without a couple reads nothing, and is a potential,
  its energy's change the components'). Nothing is declared;
- **each law delivers into its own input on its receiver.** A receiver is a dynamics' node that writes its step
  identity (``MimeNode.LEDGER_TERMS``) and names its port kinds (``Dynamics.port_kinds``). Every delivery edge into it is moved
  from the input it sums into (``applied_wrench``) to the law's own (``mime.core.ledger.law_input``), the summed input
  kept for anything unattributed; each delivery is paired as the one port kind naming the interface it is delivered
  into says;
  a confinement is held in the receiver's step, under the confining law's id, and so is each law the receiver hosts
  itself (``MimeNode.hosted_laws``: a dynamic arm's weight, damping, stops and friction). What one chain segment hands
  the segment it is mounted on (its stator's reaction, its rotor's momentum and inertia) is booked under the output
  that carries it (``rpm.motor.reaction``), the same name its own books give their half of it; a command into an input
  the receiver pairs (``MimeNode.PAIRED_COMMANDS``) is moved likewise, under its writer. The receiver is constructed
  again from its own arguments with its ``ledger``: no MADDENING change. Within each input the laws are held in the
  order of their ids, the order the edges into a summed input are added in, so the receiver sums them as the graph
  did;
- **the boundary is reported**, the build report's ``energy`` lines: each metered receiver and each law's class (or
  its reason for being unmetered), what each supplied law reads from outside, what is outside, and each receiver the
  ledger cannot meter;
- **an entry that declares nothing** (a law with no class that reads nothing from outside, a source among them, whose
  work is booked as supplied all the same; a dynamics receiving deliveries with no port kind) and states no reason
  (``unmetered``) is logged under exploratory and refused under research (``ledger.unmetered``); strict, a1's, refuses
  every unmetered entry, reason or not: :func:`refuses_unmetered` is where it does."""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

from maddening.core.node import SimulationNode

from mime.build._control import Mode
from mime.build._builders import COMMAND, DELIVERY, FIELD, REACTION, STATE, EdgeDecl
from mime.build._stages import writers_through
from mime.build._plan import EntryPlan, PlanSet, Section
from mime.build._refusals import Refusal, RefusalClass, entry_defect
from mime.core import ledger
from mime.registry import Dynamics, Interaction, Observation
from mime.validity import Profile
from mime.things import ThingKind
from mime.vocabulary import CONFINEMENT, Definiteness, DeliveryKind, port_kind

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
    #: Every delivery the wiring moved into a law's own input, as ``(source node, source field, receiver, own
    #: input)``: what the graph must hold into the receivers' own inputs, once each, and nothing else. Read from the
    #: edges the builders declared, before they were moved, so that the post-build check compares the graph as built
    #: with the delivery each law made, not with the rule that places it.
    wiring: tuple[tuple[str, str, str, str], ...] = ()
    #: Each node that receives deliveries and is not metered, with its dynamics' reason (or that it states none): what
    #: the run's account names as left out of it.
    unmetered: Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])


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


#: The sections whose entries write commands: what is always outside, each its own unit.
_WRITERS = (Section.DRIVE, Section.CONTROLLER)


def _unit(plans: PlanSet, plan: EntryPlan | None) -> str | None:
    """The unit ``outside`` is judged on that a producer's plan belongs to: its thing (a body, a field's coils, a
    medium's flow), but on an apparatus with a chain each segment, each field it carries and each drive of it by its
    own id (its mount tree), and the controller by its id; ``None`` for a law's or an observation's."""
    if plan is None:
        return None
    if plan.section is Section.CONTROLLER:
        return plan.owner
    if plan.section not in _PRODUCERS:
        return None
    if plan.thing is not None and plan.thing in _chained(plans):
        return plan.owner
    return plan.thing


def _chained(plans: PlanSet) -> set[str]:
    """The apparatus with a chain: those taken by their mount tree, each segment, field and drive of it by its own id.
    An apparatus with no chain (a field's coils) is one thing, as a body is."""
    return {plan.thing for plan in plans.entries if plan.section is Section.DYNAMICS and plan.thing is not None
            and plans.things.get(plan.thing) is ThingKind.APPARATUS}


def _class_line(law: Interaction | None) -> str:
    if law is None or law.energy_class is None:
        return ledger.UNMETERED
    return law.energy_class.value


def _unmetered_reason(entry: object) -> str:
    return str(getattr(entry, "unmetered", "") or "")


def boundary(plans: PlanSet, edges: Sequence[EdgeDecl]) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """What is outside, by unit (:func:`_unit`), and each law that reads it with the units it reads. A unit is outside
    when none of its nodes receives a delivery or a reaction from the components; a field an apparatus carries, when
    no law reading it delivers into a segment of that apparatus (the reaction its field's change is paid by); a drive
    and a controller always; and a law that declares no class, reads no component's state and delivers only loads its
    receivers pair (:func:`_sources`), which supplies its own deliveries."""
    owners = _owners(plans)
    chained = _chained(plans)
    units = {name: unit for name, plan in owners.items() for unit in (_unit(plans, plan),) if unit is not None}
    producing = set(units.values())
    receiving = {units[e.target[0]] for e in edges if e.carries in (DELIVERY, REACTION) and e.target[0] in units}
    # A body a law holds in its step (a confinement) takes that law's work from the components as a delivery.
    receiving |= {unit for plan in plans.entries if plan.section is Section.DYNAMICS and plan.confinements
                  for unit in (_unit(plans, plan),) if unit is not None}
    for plan in plans.entries:
        if plan.section is not Section.PROVIDER or plan.thing not in chained:
            continue
        mine = set(plan.node_names())
        readers = {e.by for e in edges if e.source[0] in mine and e.carries in (STATE, FIELD)}
        paid = any(e.by in readers and e.carries == DELIVERY and owners.get(e.target[0]) is not None
                   and owners[e.target[0]].thing == plan.thing and owners[e.target[0]].section is Section.DYNAMICS
                   for e in edges)
        if paid:
            receiving.add(plan.owner)
        else:
            receiving.discard(plan.owner)
    writers = {unit for plan in plans.entries if plan.section in _WRITERS
               for unit in (_unit(plans, plan),) if unit is not None}
    found = (producing - receiving) | (writers & producing)
    sources = _sources(plans, edges, units, found)
    outside = tuple(sorted(found | sources))
    supplied_by: dict[str, tuple[str, ...]] = {law: (law,) for law in sorted(sources)}
    for plan in plans.entries:
        if plan.section is not Section.INTERACTION or plan.owner in sources:
            continue
        read = {units[e.source[0]] for e in edges if e.by == plan.owner and e.carries in (STATE, FIELD)
                and e.source[0] in units and units[e.source[0]] in outside}
        if read:
            supplied_by[plan.owner] = tuple(sorted(read))
    return outside, supplied_by


def _sources(plans: PlanSet, edges: Sequence[EdgeDecl], units: Mapping[str, str], outside: set[str]) -> set[str]:
    """The laws that are sources outside the components: each declares no class, has nodes of its own, every read it
    declares is from itself or a unit outside, and every delivery it makes is a load its receiver pairs as one
    (:func:`_paired_as`).

    - **Every read it declares**: each state or field edge the law asked for, into its own nodes or not, so that the
      points a field is read at count (a field read at a body's pose, ``at=``, reads the body). A read from a node
      that is no unit (another law's) is a component's.
    - **Loads only**: a node that reads nothing supplies work that does not depend on the components only where it
      delivers a load. A force the step solves for is paired as ``-U^T R U``, the receiver's own state through the
      pairing itself, as a flow handed in and a delivered inertia are; and a delivery into a receiver whose port kind
      does not pair it (one the ledger does not meter) is paired as nothing.
    - **Only a law that declares no class**: a class says how a law's work is booked (a potential's as its energy's
      change, which is the components'), so gravity without a couple, whose node reads nothing, stays a potential."""
    owners = _owners(plans)
    found: set[str] = set()
    for plan in plans.entries:
        law = plan.entry
        if plan.section is not Section.INTERACTION or not isinstance(law, Interaction) or law.energy_class is not None:
            continue
        mine = set(plan.node_names())
        reads = [e for e in edges if e.carries in (STATE, FIELD) and (e.by == plan.owner or e.target[0] in mine)
                 and e.source[0] not in mine]
        delivered = [e for e in edges if e.by == plan.owner and e.carries == DELIVERY and e.target[0] not in mine]
        if mine and delivered and all(units.get(e.source[0]) in outside for e in reads) \
                and all(_paired_as(owners, e.target[0], e.target[1]) is DeliveryKind.LOAD for e in delivered):
            found.add(plan.owner)
    return found


def _paired_as(owners: Mapping[str, EntryPlan], node: str, field: str) -> DeliveryKind | None:
    """What a delivery into ``node``'s input ``field`` is, as its receiver's port kind pairs it; ``None`` where the
    node is no dynamics', its dynamics names no port kind, or the kind does not pair that input."""
    plan = owners.get(node)
    dynamics = plan.entry if plan is not None and plan.section is Section.DYNAMICS else None
    if plan is None or not isinstance(dynamics, Dynamics) or not dynamics.port_kinds:
        return None
    local = node.removeprefix(plan.owner).removeprefix(".")
    interface = next((name for name, port in dynamics.capabilities().accepts.items()
                      if port.node == local and port.field == field), None)
    return _pairing(dynamics, interface) if interface is not None else None


def _pairing(dynamics: Dynamics, interface: str) -> DeliveryKind | None:
    """How the one port kind of ``dynamics`` whose deliveries name ``interface`` pairs a delivery into it, or ``None``
    where none does (the registry refuses two kinds pairing one interface)."""
    kinds = [kind for name in dynamics.port_kinds for kind in (port_kind(name),) if kind is not None]
    return next((kind.deliveries[interface] for kind in kinds if interface in kind.deliveries), None)


def _law_of(edge: EdgeDecl, laws: Mapping[str, EntryPlan], owners: Mapping[str, EntryPlan],
            through: Mapping[tuple[str, str], str] | None = None) -> str:
    """The law a delivery edge into a receiver is booked under: the delivering interaction's id; for what a chain
    segment hands the segment it is mounted on, the output that carries it (``rpm.motor.reaction``); for a command,
    its writer's id, looked for through a port's stage (``through``, by the stage's output: the writer of its input);
    for a part of a law delivered from the receiver's own field (a resolved body's displaced share), the part's own name
    (``drag:ball.displaced``)."""
    if edge.carries == COMMAND and through is not None and edge.source in through:
        return through[edge.source]
    if edge.by in laws or edge.carries == COMMAND or edge.by.rpartition(".")[0] in laws:
        return edge.by
    plan = owners.get(edge.source[0])
    if plan is not None and plan.section is Section.DYNAMICS:
        return f"{edge.source[0]}.{edge.source[1]}"
    return edge.by


def attach(plans: PlanSet, built: dict[str, SimulationNode],
           edges: list[EdgeDecl]) -> tuple[dict[str, SimulationNode], list[EdgeDecl], EnergyBoundary, list[Refusal]]:
    """The graph's nodes and edges with the ledger attached, the boundary it found, and its refusals (an unmetered
    entry under research, a port kind a metered dynamics names that does not pair what it is delivered)."""
    owners = _owners(plans)
    laws = {plan.owner: plan for plan in plans.entries if plan.section is Section.INTERACTION}
    through = writers_through(edges)
    outside, supplied_by = boundary(plans, edges)
    units = {name: unit for name, plan in owners.items() for unit in (_unit(plans, plan),) if unit is not None}
    profile = plans.profile.profile
    refusals: list[Refusal] = []
    lines: list[str] = []
    receivers: dict[str, tuple[ledger.LedgerDelivery, ...]] = {}
    unmetered: dict[str, str] = {}
    moved: dict[int, tuple[str, str]] = {}
    for plan in plans.entries:
        if plan.section is not Section.DYNAMICS or not isinstance(plan.entry, Dynamics):
            continue
        dynamics = plan.entry
        for name in plan.node_names():
            node = built[name]
            commands = set(cast(tuple[str, ...], getattr(type(node), "PAIRED_COMMANDS", ())))
            into = sorted({(e.target[1], _law_of(e, laws, owners, through), e.carries) for e in edges
                           if e.target[0] == name and (e.carries in (DELIVERY, REACTION)
                                                       or (e.carries == COMMAND and e.target[1] in commands))})
            # Laws held in the step by registration (a flow's openings' laws) are metered as its deliveries are.
            held_laws = bool(plan.registrations) and bool(dynamics.port_kinds)
            if not any(carries != COMMAND for *_, carries in into) and not plan.confinements and not held_laws:
                continue
            if not dynamics.port_kinds or getattr(type(node), "LEDGER_TERMS", None) is None:
                reason = _unmetered_reason(dynamics)
                unmetered[name] = reason or "its dynamics declares no port kind and states no reason"
                lines.append(f"{name}: receives deliveries ({', '.join(dict.fromkeys(law for _, law, _ in into))}) "
                             f"and is not metered: {unmetered[name]}")
                if refuses_unmetered(profile, reason):
                    refusals.append(_unmetered_refusal(plan.owner, f"the dynamics of {plan.owner} "
                                                       f"({type(dynamics).label()})", "a port kind (Dynamics.port_kinds)"))
                continue
            host = next((e.source[0] for e in edges if e.target[0] == name and e.target[1] == "mount_poses"), None)
            host_unit = units.get(host) if host is not None else None
            held, problems = _deliveries(plan, dynamics, name, into, laws, owners, built, supplied_by, units, outside,
                                         host_unit if host_unit in outside else None)
            refusals += [entry_defect(type(dynamics), problem, context=f", metering {name}") for problem in problems]
            if problems:
                continue
            index_of = {(d.field, d.law): index for index, d in enumerate(held) if d.kind != ledger.HOSTED}
            for position, edge in enumerate(edges):
                if edge.target[0] == name and (edge.carries in (DELIVERY, REACTION)
                                               or (edge.carries == COMMAND and edge.target[1] in commands)):
                    moved[position] = (name, ledger.law_input(edge.target[1],
                                                              index_of[edge.target[1], _law_of(edge, laws, owners, through)]))
            built[name] = type(node)(node.name, float(node.delta_t), **dict(getattr(node, "params")),
                                     **{ledger.LEDGER: tuple(d.encoded() for d in held)})
            receivers[name] = held
            lines += _receiver_lines(name, dynamics, held, laws)
    for by, plan in laws.items():
        law = plan.entry
        if not isinstance(law, Interaction) or law.energy_class is not None:
            continue
        reason = _unmetered_reason(law)
        source = supplied_by.get(by) == (by,)
        if by in supplied_by and (reason or not source):
            continue
        # A source that states no reason for declaring no class is still a declaration missing: logged, or refused.
        lines.append(f"{by}: declares no energy class and states no reason; it reads no component's state and "
                     f"delivers only loads, so it is outside and supplies its own work" if source else
                     f"{by}: declares no energy class and reads nothing from outside: not metered: {reason}" if reason
                     else f"{by}: declares no energy class, reads nothing from outside and states no reason")
        if refuses_unmetered(profile, reason):
            refusals.append(_unmetered_refusal(by, f"{by} ({type(law).label()})",
                                               "an energy class (Interaction.energy_class)"))
    called = plans.controller is not None and plans.controller.mode not in (None, Mode.GRAPH)
    named = [*outside, *(["controller (called on the host)"] if called else [])]
    lines.append(f"outside: {', '.join(named) or 'nothing'} (receives no delivery from the components)")
    if called:
        lines.append("a controller called on the host is no node of the graph: what it writes into a metered receiver "
                     "arrives in an unattributed input, under no writer's name, and the run reports the work as "
                     "unattributed (mode 'graph' books it under the controller)")
    rewired = [dataclasses.replace(e, target=moved[position]) if position in moved else e
               for position, e in enumerate(edges)]
    wiring = tuple((edges[position].source[0], edges[position].source[1], *moved[position])
                   for position in sorted(moved))
    return built, rewired, EnergyBoundary(receivers, outside, supplied_by, tuple(lines), wiring, unmetered), refusals


def _deliveries(plan: EntryPlan, dynamics: Dynamics, name: str, into: Sequence[tuple[str, str, str]],
                laws: Mapping[str, EntryPlan], owners: Mapping[str, EntryPlan], built: Mapping[str, SimulationNode],
                supplied_by: Mapping[str, tuple[str, ...]], units: Mapping[str, str], outside: Sequence[str],
                host: str | None) -> tuple[tuple[ledger.LedgerDelivery, ...], list[str]]:
    """A metered receiver's deliveries, each law's in the order of its id within each input, then its hosted laws (a
    confinement's, then its own); and what keeps its port kind from pairing them. ``host`` is the unit of the segment
    it is mounted on where that is outside, which then supplies what the receiver's own laws exchange with it."""
    unknown = [name for name in dynamics.port_kinds if port_kind(name) is None]
    if unknown:
        return (), [f"it names the port kinds {unknown}, which are not registered (mime.vocabulary.port_kinds)"]
    named = " and ".join(repr(name) for name in dynamics.port_kinds)
    local = name.removeprefix(plan.owner).removeprefix(".")
    accepts = dynamics.capabilities().accepts
    interface_of = {port.field: interface for interface, port in accepts.items() if port.node == local}
    held: list[ledger.LedgerDelivery] = []
    problems: list[str] = []
    for field, by, carries in into:
        if carries == COMMAND:
            writer = units.get(by, by)
            held.append(ledger.LedgerDelivery(field, by, ledger.UNMETERED, ledger.LOAD, "",
                                              writer if writer in outside else ""))
            continue
        interface = interface_of.get(field)
        paired = _pairing(dynamics, interface) if interface is not None else None
        if paired is None:
            problems.append(f"its port kinds {named} do not say how {field!r} ({interface}) is paired, and "
                            f"{by} delivers into it")
            continue
        law = laws[by].entry if by in laws else None
        interaction = law if isinstance(law, Interaction) else None
        # A law whose operator's definiteness depends on the state has it checked where it delivers that operator,
        # the one its receiver solves with, and not on the loads it delivers beside it (the MRS rung's background load).
        check = ledger.CHECK_DEFINITE if interaction is not None and paired.value == ledger.SOLVED and \
            interaction.definiteness() is Definiteness.STATE else ""
        if by in laws and laws[by].members or interaction is not None and interaction.judged_in_sum():
            check = ledger.CHECK_CLUSTER  # judged in a sum by the law's node: a near field's, an exchange's
        energy_class, supplier = _class_line(interaction), ",".join(supplied_by.get(by, ()))
        if interaction is None:
            source, _, output = by.rpartition(".")
            energy_class = getattr(type(built[source]), "DELIVERY_CLASSES", {}).get(output, ledger.UNMETERED) \
                if source in built else ledger.UNMETERED
            supplier = units.get(source, "") if units.get(source) in outside else ""
        held.append(ledger.LedgerDelivery(field, by, energy_class, paired.value, check, supplier))
    for confinement in plan.confinements:
        if _pairing(dynamics, CONFINEMENT.name) is None:
            problems.append(f"its port kinds {named} take no confinement, and {confinement.id} confines it")
            continue
        law = laws[confinement.id].entry if confinement.id in laws else None
        held.append(ledger.LedgerDelivery("confinement", confinement.id,
                                          _class_line(law if isinstance(law, Interaction) else None), ledger.HOSTED))
    own = built[name]
    # A law the receiver holds by its registration (a flow's opening's) that declares no class is a source: it reads
    # no component's state and supplies its own work, as a class-less law's own node does (the source rule).
    registered = {registration.id for registration in plan.registrations}
    for role, law_id, energy_class in getattr(own, "hosted_laws", lambda: ())():
        exchanged = host if role in getattr(type(own), "EXCHANGED", ()) else None
        source = law_id if law_id in registered and energy_class == ledger.UNMETERED else None
        # A law held here that is judged in its sum (a resolved body's drag, its work split between the fluid and the
        # body) is judged by its host, at one velocity, not on this row.
        hosted = laws[law_id].entry if law_id in laws else None
        check = ledger.CHECK_CLUSTER if isinstance(hosted, Interaction) and hosted.judged_in_sum() else ""
        held.append(ledger.LedgerDelivery(role, law_id, energy_class, ledger.HOSTED, check,
                                          exchanged or source or ""))
    return tuple(held), problems


def _receiver_lines(name: str, dynamics: Dynamics, held: Sequence[ledger.LedgerDelivery],
                    laws: Mapping[str, EntryPlan]) -> list[str]:
    lines = [f"{name}: metered ({' and '.join(dynamics.port_kinds)} port{'s' if len(dynamics.port_kinds) > 1 else ''}), "
             f"its step identity and each law's paired work recorded in its step"]
    for law in ledger.laws(held):
        mine = [d for d in held if d.law == law]
        first = mine[0]
        how = "held in its step" if first.kind == ledger.HOSTED else ", ".join(d.field for d in mine)
        notes = [f"supplied by {first.supplied_by.replace(',', ', ')}"] if first.supplied_by else []
        if first.check == ledger.CHECK_DEFINITE:
            notes.append("its resistance checked definite at each step")
        entry = laws[law].entry if law in laws else None
        if first.check == ledger.CHECK_CLUSTER:
            own = entry.judged_in_sum() if isinstance(entry, Interaction) else ""
            notes.append(own or "its cross loads judged in its cluster's sum by its node, the grand resistance's "
                                "quadratic form at the step's starting twists, not on this row (one body drags another)")
        if first.energy_class == ledger.UNMETERED and _unmetered_reason(entry) and first.supplied_by != law:
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
                  owner_of: Any, wiring: Sequence[tuple[str, str, str, str]]) -> list[str]:
    """Each way the graph's edges into a metered receiver differ from the deliveries the wiring made (``wiring``: each
    as ``(source node, source field, receiver, own input)``). Into a law's own input: an edge the wiring did not make
    there (a delivery in another law's input, another field's, or the same law's other input, **whatever the shapes**);
    a delivery made and missing (an edge dropped); a delivery there more than once (an edge duplicated). Into any
    other input of the receiver: an edge from a law the receiver holds, where its work would arrive unattributed or
    not at all. ``receivers`` maps each metered node to its deliveries; ``owner_of`` gives a node's entry id."""
    found: list[str] = []
    expected = Counter(wiring)
    seen: Counter[tuple[str, str, str, str]] = Counter()
    for edge in graph_edges:
        source, source_field, target, field = edge
        held = receivers.get(target)
        if not held:
            continue
        owner = owner_of(source)
        if ledger.own_input(field) is not None:
            seen[edge] += 1
        elif any(d.law == owner or d.law.startswith(f"{owner}.") for d in held):
            found.append(f"{source}.{source_field} -> {target}.{field}: {owner}'s delivery arrives in {field}, which "
                         f"takes what no law owns, where its work is unattributed or missing")
    for edge in sorted(set(expected) | set(seen)):
        source, source_field, target, field = edge
        held = receivers.get(target)
        if not held or seen[edge] == expected[edge]:
            continue
        index = ledger.own_input(field)
        law = held[index[1]].law if index is not None and index[1] < len(held) else None
        stands = f"the input of {law}" if law else "an input of no law"
        delivery = f"{owner_of(source)}'s delivery {source}.{source_field} -> {target}.{field} ({stands})"
        if not expected[edge]:
            found.append(f"{delivery} is not one the wiring made")
        elif not seen[edge]:
            found.append(f"{delivery} is missing from the graph")
        else:
            found.append(f"{delivery} is in the graph {seen[edge]} times, not {expected[edge]}")
    return found


__all__ = ["EnergyBoundary", "attach", "boundary", "ledger_attached", "misattributed", "refuses_unmetered"]
