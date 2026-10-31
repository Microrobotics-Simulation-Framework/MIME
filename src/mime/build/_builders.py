"""The restricted builders: all an entry sees of the graph. They name and time its nodes, wire only what its plan
declared, and set every edge's ``additive`` flag from the kind of the target it delivers into."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, cast, override

from maddening.core.node import SimulationNode

from mime.build._plan import EntryPlan, ResolvedMount, ResolvedRead, ResolvedRegistration, ResolvedTarget
from mime.vocabulary import FRAME_POSES, LINK_WRENCH

#: What an edge carries, which decides whether its reading the previous step is planned: a dynamics' published
#: ``STATE`` (a law, provider, dynamics or observation reading the thing bound to it), a provider's ``FIELD``, a
#: ``DELIVERY`` into a target, a ``COMMAND`` into an apparatus, or a chain segment's ``REACTION`` on the segment it is
#: mounted on (what its stator passed over the step before).
STATE, FIELD, DELIVERY, COMMAND, REACTION = "state", "field", "delivery", "command", "reaction"


class EntryBuildError(Exception):
    """An entry used its builder against its own plan; the build refuses it as a defect of that entry."""


@dataclass(frozen=True)
class EdgeDecl:
    """An edge an entry asked for, held until phase 2 adds every edge in the canonical order. ``by`` is the id of
    the entry that delivers it; ``seq`` keeps one entry's edges in the order it asked for them. ``units`` label both
    ends: MIME converts no units on an edge, so what leaves the source is what the target takes. ``carries`` is one
    of :data:`STATE`, :data:`FIELD`, :data:`DELIVERY`, :data:`COMMAND` and :data:`REACTION`."""

    by: str
    source: tuple[str, str]
    target: tuple[str, str]
    additive: bool
    units: str
    seq: int
    carries: str = STATE


@dataclass(frozen=True)
class PortRef:
    node: str
    field: str


class NodeRef:
    """One of the entry's own nodes, once added. ``node["load"]`` is its port ``load``."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __getitem__(self, field: str) -> PortRef:
        return PortRef(self.name, field)


@dataclass(frozen=True)
class ReadRef:
    key: tuple[str, ...]
    read: ResolvedRead
    by: str


@dataclass(frozen=True)
class TargetRef:
    key: tuple[str, ...]
    target: ResolvedTarget


class RestrictedBuilder:
    def __init__(self, plan: EntryPlan, timestep: float) -> None:
        self._plan = plan
        self._timestep = timestep
        self.nodes: list[tuple[str, SimulationNode]] = []
        self.edges: list[EdgeDecl] = []
        self._connected: Counter[tuple[str, ...]] = Counter()
        self._points: set[tuple[str, str | None]] = set()

    def _add(self, node: SimulationNode, local: str) -> NodeRef:
        if local not in self._plan.nodes:
            raise EntryBuildError(f"it adds node {local!r}, and its plan declares {self._plan.nodes!r}")
        if any(held == local for held, _ in self.nodes):
            raise EntryBuildError(f"it adds node {local!r} twice")
        if not isinstance(cast(object, node), SimulationNode):  # an entry's own code, checked like a script's
            raise EntryBuildError(f"node {local!r} is a {type(node).__name__}, not a MADDENING SimulationNode")
        name = self._plan.node_name(local)
        if node.name != name:
            raise EntryBuildError(f"it named a node {node.name!r}; construct it with "
                                  f"ctx.node_name({local!r}) = {name!r}")
        if float(node.delta_t) != self._timestep:
            raise EntryBuildError(f"node {name!r} has timestep {node.delta_t!r}; construct it with ctx.timestep = "
                                  f"{self._timestep!r}")
        self.nodes.append((local, node))
        return NodeRef(name)

    def _own(self, port: object) -> bool:
        return isinstance(port, PortRef) and port.node in self._plan.node_names()

    def _edge(self, by: str, source: tuple[str, str], target: tuple[str, str], additive: bool, units: str,
              carries: str) -> None:
        self.edges.append(EdgeDecl(by, source, target, additive, units, len(self.edges), carries))

    def connect(self, source: ReadRef | PortRef, *, into: PortRef | TargetRef) -> None:
        """A read into an input of one of the entry's own nodes, an output of one of its own nodes into a target, or,
        for an interaction, one of its own nodes' outputs into another's input (:meth:`InteractionBuilder.connect`).
        The edge's ``additive`` flag comes from the target's kind and its units labels from the vocabulary; an entry
        sets neither."""
        if isinstance(source, ReadRef):
            if not isinstance(into, PortRef) or not self._own(into):
                raise EntryBuildError(f"read {source.key[-1]!r} goes into {into!r}, not an input of its own nodes")
            read = source.read
            if read.points is not None and (read.node, read.slot) not in self._points:
                self._points.add((read.node, read.slot))
                self._edge(source.by, (read.points.node, read.points.field), (read.node, f"points_{read.slot}"), False,
                           read.points.units, STATE)
            self._edge(source.by, (read.node, read.field), (into.node, into.field), False, read.units,
                       STATE if read.slot is None else FIELD)
            self._connected[source.key] += 1
        elif isinstance(into, TargetRef):
            if not self._own(source):
                raise EntryBuildError(f"target {into.key[-1]!r} is delivered from {source!r}, not an output of its own "
                                      f"nodes")
            target = into.target
            self._edge(target.by, (source.node, source.field), (target.node, target.field), target.kind.additive,
                       target.units, DELIVERY)
            self._connected[into.key] += 1
        else:
            raise EntryBuildError(f"connect({source!r}, into={into!r}): connect a read into an own input, or an own "
                                  f"output into a target")

    def _expect(self, keys: list[tuple[str, ...]]) -> list[str]:
        problems = [f"it connects {key[-1]!r} {self._connected[key]} times, not once" for key in keys
                    if self._connected[key] != 1]
        added = [local for local, _ in self.nodes]
        problems += [f"it declares node {local!r} and never adds it"
                     for local in self._plan.nodes if local not in added]
        return problems

    def finish(self) -> list[str]:
        return self._expect([])

    def _mounted(self, key: tuple[str, ...], mount: ResolvedMount, into: PortRef) -> None:
        """The edge the build derives for a mount: every frame pose of the segment ``mount`` names, into ``into``,
        an input of one of the entry's own nodes, which reads the mount frame's row. It carries the segment's
        published state."""
        if not self._own(into):
            raise EntryBuildError(f"its mount goes into {into!r}, not an input of its own nodes")
        self._edge(self._plan.owner, (mount.node, mount.field), (into.node, into.field), False, FRAME_POSES.units,
                   STATE)
        self._connected[key] += 1


class _ReaderBuilder(RestrictedBuilder):
    def read(self, label: str) -> ReadRef:
        """The read its plan declared under ``label``."""
        if label not in self._plan.reads:
            raise EntryBuildError(f"it reads {label!r}, and its plan declares {sorted(self._plan.reads)}")
        return ReadRef(("read", label), self._plan.reads[label], self._plan.owner)

    @override
    def finish(self) -> list[str]:
        return self._expect([("read", label) for label in self._plan.reads])


class InteractionBuilder(_ReaderBuilder):
    """An interaction's builder: its response nodes, its reads into them, their outputs into its targets and into
    one another, and, for a reaction on a field's source, the poses of the frames of the segment it is delivered to."""

    def add_response_node(self, node: SimulationNode, local: str = "") -> NodeRef:
        return self._add(node, local)

    def target(self, label: str) -> TargetRef:
        """The delivery its plan declared under ``label``."""
        if label not in self._plan.delivers:
            raise EntryBuildError(f"it delivers {label!r}, and its plan declares {sorted(self._plan.delivers)}")
        return TargetRef(("target", label), self._plan.delivers[label])

    def delivers(self, label: str) -> bool:
        """Whether its plan declared a delivery under ``label``."""
        return label in self._plan.delivers

    @override
    def connect(self, source: ReadRef | PortRef, *, into: PortRef | TargetRef) -> None:
        """As :meth:`RestrictedBuilder.connect`, and an output of one of its own nodes into an input of another of its
        own, added after it: what the law delivers, into the node that makes its reaction."""
        if isinstance(source, PortRef) and isinstance(into, PortRef):
            added = {node.name: node for _, node in self.nodes}
            if source.node not in added or into.node not in added or source.node == into.node:
                raise EntryBuildError(f"connect({source!r}, into={into!r}): an output of one of its own nodes goes "
                                      f"into an input of another of its own, both added")
            spec = cast(dict[str, Any], added[into.node].boundary_input_spec()).get(into.field)
            if spec is None:
                raise EntryBuildError(f"connect({source!r}, into={into!r}): node {into.node!r} has no input "
                                      f"{into.field!r}")
            self._edge(self._plan.owner, (source.node, source.field), (into.node, into.field), False,
                       str(getattr(spec, "expected_units", "") or ""), DELIVERY)
            return
        super().connect(source, into=into)

    def mount(self, label: str, into: PortRef) -> None:
        """The frame poses of the segment the reaction delivered under ``label`` goes to, into ``into``, an input of
        one of its own nodes: its row ``target(label).target.mount.index`` is the frame the source is mounted on."""
        mount = self.target(label).target.mount
        if mount is None:
            raise EntryBuildError(f"its delivery {label!r} is no reaction on a mounted source")
        self._mounted(("mount", label), mount, into)

    @override
    def finish(self) -> list[str]:
        return self._expect([("read", label) for label in self._plan.reads]
                            + [("target", label) for label in self._plan.delivers]
                            + [("mount", label) for label, target in self._plan.delivers.items()
                               if target.mount is not None])


class ObservationBuilder(_ReaderBuilder):
    """An observation's builder: its nodes and its reads. It has no way to deliver into the physics."""

    def add_node(self, node: SimulationNode, local: str = "") -> NodeRef:
        return self._add(node, local)

    @override
    def connect(self, source: ReadRef | PortRef, *, into: PortRef | TargetRef) -> None:
        if not isinstance(source, ReadRef):
            raise EntryBuildError("an observation only reads")
        super().connect(source, into=into)


class DynamicsBuilder(RestrictedBuilder):
    """A dynamics' builder: its nodes, and for each registration addressed to it the registered thing's state into
    its inputs and its outputs into the registered thing's targets, delivered under the registering id. A chain
    segment after the first takes the frame poses of the segment it mounts on with :meth:`mount`."""

    def add_node(self, node: SimulationNode, local: str = "") -> NodeRef:
        return self._add(node, local)

    def mount(self, into: PortRef) -> None:
        """The frame poses of the segment this one mounts on (``ctx.mount``), into ``into``."""
        mount = self._plan.mount
        if mount is None:
            raise EntryBuildError("it takes a mount, and it is its chain's first segment, mounted on nothing")
        self._mounted(("mount",), mount, into)

    def reaction(self, output: PortRef) -> bool:
        """``output`` of one of its own nodes, the wrench its stator passes to the frame it mounts on (a row per frame
        of that segment), into that segment's loads where it takes them, a dynamic segment's: whether it is
        delivered. Read at the start of each step, it is what the stator passed over the step before."""
        mount = self._plan.mount
        if not self._own(output) or mount is None:
            raise EntryBuildError(f"reaction({output!r}): an output of its own nodes, from a segment mounted on "
                                  f"another")
        if self._connected[("reaction",)]:
            raise EntryBuildError("it delivers its reaction twice")
        self._connected[("reaction",)] += 1
        if mount.loads is None:
            return False
        self._edge(self._plan.owner, (output.node, output.field), mount.loads, True, LINK_WRENCH.units, REACTION)
        return True

    def _registration(self, registration: ResolvedRegistration) -> ResolvedRegistration:
        if registration not in self._plan.registrations:
            raise EntryBuildError(f"registration {registration.id} is not addressed to it")
        return registration

    def read(self, registration: ResolvedRegistration, label: str) -> ReadRef:
        reads = self._registration(registration).reads
        if label not in reads:
            raise EntryBuildError(f"registration {registration.id} reads {sorted(reads)}, not {label!r}")
        return ReadRef(("read", registration.id, label), reads[label], registration.id)

    def target(self, registration: ResolvedRegistration, label: str) -> TargetRef:
        returns = self._registration(registration).returns
        if label not in returns:
            raise EntryBuildError(f"registration {registration.id} returns {sorted(returns)}, not {label!r}")
        return TargetRef(("target", registration.id, label), returns[label])

    @override
    def finish(self) -> list[str]:
        keys = [("read", r.id, label) for r in self._plan.registrations for label in r.reads]
        keys += [("target", r.id, label) for r in self._plan.registrations for label in r.returns]
        keys += [("mount",)] if self._plan.mount is not None else []
        return self._expect(keys)


class ProviderBuilder(RestrictedBuilder):
    """A field provider's builder: its one node, with a points input and a field output per reader slot, which the
    readers' builders wire, and an input per command, which its drives wire. A part it reads that is mounted on a
    segment of its apparatus's chain takes that segment's frame poses with :meth:`part`."""

    def add_node(self, node: SimulationNode, local: str = "") -> NodeRef:
        return self._add(node, local)

    def part(self, setting: str, into: PortRef) -> None:
        """The frame poses of the segment the part named by ``setting`` is mounted on (``ctx.parts``), into
        ``into``."""
        held = self._plan.parts.get(setting)
        if held is None or held.mount is None:
            raise EntryBuildError(f"it reads the mount of {setting!r}, which names no part mounted on a segment")
        self._mounted(("part", setting), held.mount, into)

    @override
    def finish(self) -> list[str]:
        return self._expect([("part", setting) for setting, held in self._plan.parts.items()
                             if held.mount is not None])


class DriveBuilder(RestrictedBuilder):
    """A drive's builder: its one node, and its output into the command it drives, by an exclusive edge."""

    def add_node(self, node: SimulationNode, local: str = "") -> NodeRef:
        return self._add(node, local)

    def command(self, output: PortRef) -> None:
        """``output`` of its own node into the command the drive drives."""
        command = self._plan.command
        if command is None or not self._own(output):
            raise EntryBuildError(f"command({output!r}): a drive writes an output of its own node into its command")
        self._edge(self._plan.owner, (output.node, output.field), (command.node, command.field), False, command.units,
                   COMMAND)
        self._connected[("command",)] += 1

    @override
    def finish(self) -> list[str]:
        return self._expect([("command",)])
