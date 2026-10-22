"""Phase 1: every declaration resolved and checked as data, with no graph, and every refusal collected."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from mime import vocabulary
from mime.build._declarations import (
    Attachment,
    Declaration,
    ObservationDeclaration,
    RunSettings,
    ThingDeclaration,
)
from mime.build._frame import WorldFrame, world_frame
from mime.build._refusals import MIME_DEFECT, BuildRefused, Refusal, RefusalClass
from mime.build._report import BuildReport, RunStructure
from mime.registry import (
    CapabilityRecord,
    Delivery,
    Dynamics,
    Entry,
    EntryKind,
    FieldProvider,
    GroupRequest,
    Interaction,
    Observation,
    Plan,
    PlanContext,
    Port,
    Read,
    Registered,
    Registration,
    RoleKind,
    registered,
    registration_of,
)
from mime.things import Apparatus, Body, FieldPath, Handle, Thing, ThingKind, TypedPath, did_you_mean
from mime.units import FREQUENCY, TIME, Quantity, UnitError, si
from mime.vocabulary import ReadKind, TargetKind

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LOCAL = re.compile(r"[a-z_][a-z0-9_]*")
_RATE_RTOL = 1e-9
_DYNAMICS_ORDER = {ThingKind.MEDIUM: 0, ThingKind.BODY: 1, ThingKind.APPARATUS: 2, ThingKind.STRUCTURE: 3}


class Section(Enum):
    """The canonical order's sections, in the order their nodes are added."""

    PROVIDER = 0
    INTERACTION = 1
    DYNAMICS = 2
    OBSERVATION = 3


@dataclass(frozen=True)
class ResolvedTarget:
    """A target port of a built dynamics node, and who delivers into it (``by``). ``units`` are the vocabulary's,
    which every edge into the port carries as its label."""

    node: str
    field: str
    kind: TargetKind
    thing: str
    interface: str
    units: str
    by: str = ""


@dataclass(frozen=True)
class ResolvedRead:
    """Where a read comes from. A state read names the publishing node's field. A field read names the provider's
    output port for its reader slot, and ``points`` the state read its points come from. ``units`` are the
    vocabulary's, which the edge carrying the read is labelled with."""

    node: str
    field: str
    interface: str
    units: str
    points: ResolvedRead | None = None
    slot: str | None = None


@dataclass(frozen=True)
class Slot:
    """A reader slot of a field provider: ``points_<name>`` in, ``<prefix>_<name>`` out for each interface read."""

    name: str
    reader: str
    interfaces: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedRegistration:
    """A registration addressed to a dynamics: the registering interaction's id, the record's kind, the registered
    thing, and by label the subject's state to read and the subject's targets to deliver into."""

    id: str
    kind: str
    subject: str
    reads: Mapping[str, ResolvedRead]
    returns: Mapping[str, ResolvedTarget]


@dataclass
class EntryPlan:
    """One declared entry, resolved: who it is, what it was bound to, which nodes it will add and at what rate."""

    section: Section
    owner: str
    entry: Entry
    registered: Registered
    nodes: tuple[str, ...]
    divider: int
    declared: int
    thing: str | None = None
    thing_kind: ThingKind | None = None
    bindings: dict[str, str] = field(default_factory=dict[str, str])
    plan: Plan | None = None
    record: CapabilityRecord | None = None
    reads: dict[str, ResolvedRead] = field(default_factory=dict[str, ResolvedRead])
    delivers: dict[str, ResolvedTarget] = field(default_factory=dict[str, ResolvedTarget])
    registrations: list[ResolvedRegistration] = field(default_factory=list[ResolvedRegistration])
    slots: list[Slot] = field(default_factory=list[Slot])
    initial: dict[str, Any] = field(default_factory=dict[str, Any])
    seeds: dict[str, int] = field(default_factory=dict[str, int])

    def node_name(self, local: str = "") -> str:
        return self.owner if not local else f"{self.owner}.{local}"

    def node_names(self) -> list[str]:
        return [self.node_name(local) for local in self.nodes]

    def reader_cls(self) -> type[Interaction] | type[Observation]:
        """The class of an interaction's or observation's plan."""
        return cast(type[Interaction] | type[Observation], self.registered.cls)

    def describe(self) -> str:
        rung = getattr(self.entry, "rung", None)
        return self.registered.cls.label() + (f" (rung {rung})" if rung else "")

    def canonical_key(self) -> tuple[int, int, str, str]:
        if self.section is Section.DYNAMICS and self.thing_kind is not None:
            return (self.section.value, _DYNAMICS_ORDER[self.thing_kind], self.thing or "", self.owner)
        return (self.section.value, 0, "", self.owner)


@dataclass(frozen=True)
class GroupPlan:
    """A coupling group: the union of overlapping solve-together requests, with the settings they agree on."""

    nodes: tuple[str, ...]
    tolerance: float
    max_iterations: int
    requested_by: tuple[str, ...]


@dataclass(frozen=True)
class PlanSet:
    """Phase 1's result: everything phase 2 constructs, in canonical order and in declaration order."""

    settings: RunSettings
    dt: float
    frame: WorldFrame
    entries: tuple[EntryPlan, ...]
    declared: tuple[EntryPlan, ...]
    groups: tuple[GroupPlan, ...]
    targets: Mapping[tuple[str, str], ResolvedTarget]
    vocabulary: Mapping[str, str]
    run: RunStructure = RunStructure()
    things: Mapping[str, ThingKind] = field(default_factory=dict[str, ThingKind])

    def owner_of(self, node: str) -> str | None:
        for plan in self.entries:
            if node in plan.node_names():
                return plan.owner
        return None


def _class_record(cls: type[Entry]) -> CapabilityRecord:
    if issubclass(cls, FieldProvider):
        return CapabilityRecord(cls.label(), frozenset(cls.flags),
                                {name: Port(prefix) for name, prefix in cls.publishes.items()})
    dynamics = cast(type[Dynamics], cls)
    return CapabilityRecord(dynamics.label(), frozenset(dynamics.flags), dict(dynamics.publishes),
                            dict(dynamics.accepts), frozenset(dynamics.registration_kinds))


def _offers(role: RoleKind) -> EntryKind:
    """The kind of entry whose capability record a role's support predicate is asked of."""
    return EntryKind.FIELD_PROVIDER if role is RoleKind.FIELD else EntryKind.DYNAMICS


def _qualifying(kind: EntryKind, predicate: Callable[[CapabilityRecord], bool]) -> str:
    names = [r.cls.label() for r in registered(kind) if predicate(_class_record(r.cls))]
    return ", ".join(names) if names else "none registered"


def _seconds(value: float) -> str:
    return f"{value!r} s"


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def binding_name(value: Any) -> str:
    """The name a binding refers to, as written: a handle's name, a typed path's dotted string, or the string."""
    if isinstance(value, Handle):
        return value.name
    if isinstance(value, TypedPath):
        return str(value)
    return value if isinstance(value, str) else repr(value)


def entry_id(cls: type[Interaction] | type[Observation], bindings: Mapping[str, Any]) -> str:
    """An interaction's or observation's id: its registered name and the names bound to its id roles, never an
    attachment counter, so the canonical order cannot depend on the order of the script's calls."""
    roles = cls.roles
    id_roles = cls.id_roles
    if id_roles is None:
        id_roles = tuple(r for r, k in roles.items() if k is RoleKind.BODY) or tuple(roles)
    names = [binding_name(bindings[r]) if r in bindings and bindings[r] is not None else "?" for r in id_roles]
    return f"{cls.name}:{'-'.join(names)}"


class _Planner:
    def __init__(self, settings: RunSettings, declarations: list[Declaration]) -> None:
        self.settings = settings
        self.declarations = declarations
        self.refusals: list[Refusal] = []
        self.things: dict[str, Thing] = {}
        self.dynamics: dict[str, EntryPlan] = {}
        self.providers: dict[str, EntryPlan] = {}
        self.readers: list[EntryPlan] = []
        self.declared: list[EntryPlan] = []
        self.dt = 0.0
        self.frame: WorldFrame | None = None

    def refuse(self, code: str, category: RefusalClass, message: str, action: str) -> None:
        self.refusals.append(Refusal(code, category, message, action))

    def defect(self, cls: type[Entry], message: str) -> None:
        self.refuse("entry.defect", RefusalClass.INTERNAL, f"{cls.label()} ({cls.__module__}): {message}",
                    f"a defect in {cls.label()}, defined in {cls.__module__}: report it to that package. "
                    f"If that package is MIME: {MIME_DEFECT}")

    def run(self) -> PlanSet:
        self.plan_settings()
        for index, declaration in enumerate(self.declarations):
            if isinstance(declaration, ThingDeclaration):
                self.plan_thing(declaration.thing, index)
        for index, declaration in enumerate(self.declarations):
            if isinstance(declaration, Attachment):
                self.plan_reader(declaration.law, dict(declaration.bindings), index, Section.INTERACTION)
            elif isinstance(declaration, ObservationDeclaration):
                observation = declaration.observation
                self.plan_reader(observation, observation.bindings(), index, Section.OBSERVATION)
        self.check_ids()
        self.collect_registrations_and_slots()
        targets = self.collect_targets()
        self.check_exclusive()
        groups = self.union_groups()
        self.check_rates()
        self.derive_seeds()
        if self.refusals or self.frame is None:
            raise BuildRefused(self.refusals, BuildReport(self.settings.experiment, self.dt or None,
                                                          vocabulary=vocabulary.versions(),
                                                          refusals=tuple(self.refusals)))
        entries = tuple(sorted(self.declared, key=EntryPlan.canonical_key))
        return PlanSet(self.settings, self.dt, self.frame, entries, tuple(self.declared), groups, targets,
                       vocabulary.versions(), self.run_structure(),
                       {name: thing.kind for name, thing in sorted(self.things.items())})

    # -- run settings -----------------------------------------------------------------------------------------

    def plan_settings(self) -> None:
        name = self.settings.experiment
        if not (isinstance(name, str) and _IDENTIFIER.fullmatch(name)):
            self.refuse("experiment.name", RefusalClass.USER, f"the experiment name {name!r} is not an identifier",
                        "name the experiment with letters, digits and underscores, starting with a letter")
        frame, problems = world_frame(self.settings.up, self.settings.gravity)
        for code, message, action in problems:
            self.refuse(code, RefusalClass.USER, message, action)
        self.frame = frame
        dt = self.settings.dt
        if dt is None:
            self.refuse("run.dt-missing", RefusalClass.USER,
                        "dt, the base step, has no default: every node's timestep is a whole multiple of it",
                        "pass dt as a time to Experiment, such as dt=ms(1)")
            return
        try:
            value = si(dt, TIME, what="dt")
        except UnitError as error:
            self.refuse("run.dt-units", RefusalClass.USER, str(error), "write dt as a time, such as ms(1) or s(0.02)")
            return
        if isinstance(value, tuple) or not (math.isfinite(value) and value > 0.0):
            self.refuse("run.dt-invalid", RefusalClass.USER, f"dt = {dt} is not a finite time above zero",
                        "pass a positive dt, such as ms(1)")
            return
        self.dt = value
        if not isinstance(self.settings.seed, int) or isinstance(self.settings.seed, bool):
            self.refuse("run.seed", RefusalClass.USER, f"seed={self.settings.seed!r} is not an integer",
                        "pass an integer seed")

    def divider(self, rate: Any, what: str) -> int:
        """``rate`` as a whole multiple of ``dt`` (a period, or a frequency whose period is one), or a refusal."""
        if rate is None or not self.dt:
            return 1
        try:
            if isinstance(rate, Quantity) and rate.dimension == FREQUENCY:
                frequency = si(rate, FREQUENCY, what=f"the rate of {what}")
                period = 1.0 / frequency if isinstance(frequency, float) and frequency > 0 else math.nan
            else:
                period = si(rate, TIME, what=f"the rate of {what}")
        except UnitError as error:
            self.refuse("run.rate-units", RefusalClass.USER, str(error),
                        "write a rate as a period, such as ms(2), or a frequency, such as Hz(500)")
            return 1
        ratio = period / self.dt if isinstance(period, float) else math.nan
        whole = round(ratio) if math.isfinite(ratio) else 0
        if whole >= 1 and abs(ratio - whole) <= _RATE_RTOL * ratio:
            return whole
        below, above = (max(1, math.floor(ratio)), math.ceil(ratio)) if math.isfinite(ratio) else (1, 1)
        nearest = sorted({below, max(above, 1)})
        self.refuse("run.rate-not-multiple", RefusalClass.USER,
                    f"the rate of {what} is a period of {_seconds(period) if isinstance(period, float) else rate}, "
                    f"not a whole multiple of the base step dt = {_seconds(self.dt)}",
                    "use a period that is a whole multiple of dt; the nearest are "
                    + " and ".join(_seconds(k * self.dt) for k in nearest))
        return 1

    # -- things, their dynamics and providers -----------------------------------------------------------------

    def plan_thing(self, thing: Thing, index: int) -> None:
        name = cast(object, thing.name)  # a script is not type-checked
        if not (isinstance(name, str) and _IDENTIFIER.fullmatch(name)):
            self.refuse("thing.name", RefusalClass.USER, f"the {thing.kind.value} name {name!r} is not an identifier",
                        "name things with letters, digits and underscores, starting with a letter")
            return
        if name in self.things:
            self.refuse("thing.duplicate", RefusalClass.USER,
                        f"two things are named {name!r} (a {self.things[name].kind.value} and a {thing.kind.value})",
                        "give every thing its own name")
            return
        self.things[name] = thing
        if thing.dynamics is not None:
            self.plan_dynamics(thing, thing.dynamics, index)
        if isinstance(thing, Apparatus):
            for field_name, provider in thing.fields.items():
                self.plan_provider(thing.name, field_name, provider, index)

    def registered_as(self, entry: Entry, kind: EntryKind, where: str) -> Registered | None:
        record = registration_of(entry)
        if record is None or record.kind is not kind:
            self.refuse("entry.unregistered", RefusalClass.USER,
                        f"{where}: {type(entry).__qualname__} is not a registered {kind.value}",
                        f"pass a registered {kind.value}, or register it with "
                        f"mime.registry.register_{kind.name.lower()}")
            return None
        return record

    def plan_dynamics(self, thing: Thing, dynamics: Dynamics, index: int) -> None:
        where = f"{thing.kind.value} {thing.name!r}"
        record = self.registered_as(dynamics, EntryKind.DYNAMICS, where)
        if record is None:
            return
        cls = cast(type[Dynamics], record.cls)
        if thing.kind not in cls.applies_to:
            applies = ", ".join(sorted(k.value for k in cls.applies_to))
            others = [cast(type[Dynamics], r.cls) for r in registered(EntryKind.DYNAMICS)]
            self.refuse("dynamics.wrong-thing", RefusalClass.USER,
                        f"{where} has dynamics {cls.label()}, which applies to {applies} only",
                        f"choose a dynamics for a {thing.kind.value}: " + (", ".join(
                            other.label() for other in others if thing.kind in other.applies_to)
                            or "none registered"))
            return
        try:
            capabilities = dynamics.capabilities()
        except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
            self.defect(cls, f"capabilities() raised {error!r}")
            return
        owner = f"{thing.name}.flow" if thing.kind is ThingKind.MEDIUM else thing.name
        plan = EntryPlan(Section.DYNAMICS, owner, dynamics, record, tuple(cls.nodes),
                         self.divider(dynamics.rate, f"{where}'s dynamics"), index, thing=thing.name,
                         thing_kind=thing.kind, record=capabilities)
        if not self.check_locals(cls, plan.nodes):
            return
        if isinstance(thing, Body):
            plan.initial = self.initial_state(thing, cls)
        self.dynamics[thing.name] = plan
        self.declared.append(plan)

    def initial_state(self, thing: Body, cls: type[Dynamics]) -> dict[str, Any]:
        wanted = cls.initial_state
        given = dict(thing.initial)
        values: dict[str, Any] = {}
        for key, value in given.items():
            if key not in wanted:
                self.refuse("state.unknown", RefusalClass.USER,
                            f"body {thing.name!r}: its dynamics {cls.label()} takes no initial {key!r}",
                            did_you_mean(key, wanted))
                continue
            try:
                values[key] = si(value, wanted[key], what=f"body {thing.name!r}'s initial {key}")
            except UnitError as error:
                self.refuse("state.units", RefusalClass.USER, str(error), f"write {key} in units of {wanted[key]}")
            except TypeError as error:
                self.refuse("state.value", RefusalClass.USER, f"body {thing.name!r}'s initial {key}: {error}",
                            f"write {key} as a quantity of {wanted[key]}")
        missing = [key for key in wanted if key not in given]
        if missing:
            self.refuse("state.missing", RefusalClass.USER,
                        f"body {thing.name!r}: its dynamics {cls.label()} needs an initial "
                        f"{', '.join(missing)}, and MIME assumes none",
                        f"pass initial=State({', '.join(f'{k}=...' for k in missing)})")
        return values

    def check_locals(self, cls: type, nodes: tuple[str, ...]) -> bool:
        if len(set(nodes)) != len(nodes) or not all(n == "" or _LOCAL.fullmatch(n) for n in nodes):
            self.defect(cls, f"its nodes {nodes!r} are not distinct lower-case local names")
            return False
        return True

    def plan_provider(self, apparatus: str, field_name: str, provider: FieldProvider, index: int) -> None:
        where = f"field {field_name!r} of apparatus {apparatus!r}"
        if not _IDENTIFIER.fullmatch(field_name):
            self.refuse("thing.field-name", RefusalClass.USER, f"{where}: the name is not an identifier",
                        "name fields with letters, digits and underscores")
            return
        record = self.registered_as(provider, EntryKind.FIELD_PROVIDER, where)
        if record is None:
            return
        try:
            capabilities = provider.capabilities()
        except Exception as error:  # noqa: BLE001
            self.defect(record.cls, f"capabilities() raised {error!r}")
            return
        path = f"{apparatus}.{field_name}"
        plan = EntryPlan(Section.PROVIDER, path, provider, record, ("",), 1, index, thing=apparatus,
                         thing_kind=ThingKind.APPARATUS, record=capabilities)
        self.providers[path] = plan
        self.declared.append(plan)

    # -- bindings -------------------------------------------------------------------------------------------

    def resolve(self, where: str, role: str, kind: RoleKind, value: Any) -> str | None:
        if kind is RoleKind.FIELD:
            path = str(value) if isinstance(value, (FieldPath, str)) else None
            if isinstance(value, Handle) or path is None:
                self.refuse("binding.wrong-kind", RefusalClass.USER,
                            f"{where}: {role}= binds a field, written apparatus.field, not {value!r}",
                            f"bind {role}= one of: {', '.join(sorted(self.providers)) or 'no field is declared'}")
                return None
            if path not in self.providers:
                self.refuse("binding.unknown", RefusalClass.USER, f"{where}: {role}={path!r} names no field",
                            did_you_mean(path, self.providers))
                return None
            return path
        if isinstance(value, TypedPath):
            self.refuse("binding.wrong-kind", RefusalClass.USER,
                        f"{where}: {role}= binds a {kind.value}, not the path {value}",
                        f"bind {role}= one of: {', '.join(self.names_of(kind)) or f'no {kind.value} is declared'}")
            return None
        name = binding_name(value)
        thing = self.things.get(name)
        if thing is None:
            self.refuse("binding.unknown", RefusalClass.USER, f"{where}: {role}={name!r} names no thing",
                        did_you_mean(name, self.names_of(kind)))
            return None
        if thing.kind is not kind.thing_kind:
            self.refuse("binding.wrong-kind", RefusalClass.USER,
                        f"{where}: {role}={name!r} is a {thing.kind.value}, and {role}= binds a {kind.value}",
                        f"bind {role}= one of: {', '.join(self.names_of(kind)) or f'no {kind.value} is declared'}")
            return None
        return name

    def names_of(self, kind: RoleKind) -> list[str]:
        if kind is RoleKind.FIELD:
            return sorted(self.providers)
        return sorted(n for n, t in self.things.items() if t.kind is kind.thing_kind)

    def record_of(self, kind: RoleKind, name: str) -> CapabilityRecord | None:
        if kind is RoleKind.FIELD:
            return self.providers[name].record
        plan = self.dynamics.get(name)
        return plan.record if plan else None

    # -- interactions and observations --------------------------------------------------------------------

    def plan_reader(self, entry: Interaction | Observation, given: dict[str, Any], index: int,
                    section: Section) -> None:
        kind = EntryKind.INTERACTION if section is Section.INTERACTION else EntryKind.OBSERVATION
        record = self.registered_as(entry, kind, f"{kind.value} #{index + 1}")
        if record is None:
            return
        cls = cast(type[Interaction] | type[Observation], record.cls)
        owner = entry_id(cls, given)
        where = f"{kind.value} {owner}"
        bindings: dict[str, str] = {}
        complete = True
        for role in given:
            if role not in cls.roles:
                self.refuse("binding.unknown-role", RefusalClass.USER, f"{where} has no role {role!r}",
                            did_you_mean(role, cls.roles))
                complete = False
        for role, role_kind in cls.roles.items():
            value = given.get(role)
            if value is None:
                candidates = ", ".join(self.names_of(role_kind)) or f"no {role_kind.value} is declared"
                self.refuse("binding.missing", RefusalClass.USER,
                            f"{where} needs {role}=, and MIME binds nothing by default",
                            f"bind {role}= one of: {candidates}")
                complete = False
                continue
            name = self.resolve(where, role, role_kind, value)
            if name is None:
                complete = False
            else:
                bindings[role] = name
        rungs = getattr(cls, "rungs", ())
        rung = getattr(entry, "rung", None) if rungs else None
        if rungs and rung not in rungs:
            valid = ", ".join(repr(r) for r in rungs)
            if rung is None:
                self.refuse("rung.missing", RefusalClass.USER, f"{where} has no default rung",
                            f"write the rung: rung= one of {valid}")
            else:
                self.refuse("rung.unknown", RefusalClass.USER, f"{where}: rung={rung!r} is not one of its rungs",
                            did_you_mean(str(rung), rungs))
            complete = False
        if not complete:
            return
        records = {role: self.record_of(cls.roles[role], name) for role, name in bindings.items()}
        if section is Section.INTERACTION and not self.supported(cast(type[Interaction], cls), where, rung, records):
            return
        divider = self.divider(getattr(entry, "rate", None), where) if section is Section.OBSERVATION else 1
        up, gravity = (self.frame.up, self.frame.gravity) if self.frame else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        context = PlanContext(self.dt, up, gravity, dict(bindings), records)
        try:
            plan = cast(object, entry.plan(context))  # an entry's own code, checked like a script's
        except Exception as error:  # noqa: BLE001
            self.defect(cls, f"plan() raised {error!r}")
            return
        if not isinstance(plan, Plan):
            self.defect(cls, f"plan() returned {type(plan).__name__}, not a mime.registry.Plan")
            return
        if not self.check_locals(cls, tuple(plan.nodes)):
            return
        entry_plan = EntryPlan(section, owner, entry, record, tuple(plan.nodes), divider, index, bindings=bindings,
                               plan=plan)
        if section is Section.OBSERVATION and (plan.delivers or plan.registrations or plan.groups):
            self.defect(cls, "an observation reads only: it delivers into nothing, registers nothing and asks "
                             "for no coupling group")
            return
        self.plan_reads(entry_plan, where)
        for label, delivery in plan.delivers.items():
            target = self.target(entry_plan, where, delivery)
            if target is not None:
                entry_plan.delivers[label] = target
        for registration in plan.registrations:
            self.plan_registration(entry_plan, where, registration)
        for request in plan.groups:
            self.check_request(entry_plan, request)
        self.readers.append(entry_plan)
        self.declared.append(entry_plan)

    def supported(self, cls: type[Interaction], where: str, rung: str | None,
                  records: Mapping[str, CapabilityRecord | None]) -> bool:
        rows = [row for row in cls.support if row.rung in (None, rung)]
        if not cls.support or any(all(test(records.get(role)) for role, test in row.when.items()) for row in rows):
            return True
        bound = "; ".join(f"{role}: {rec.describe() if rec else 'no dynamics'}" for role, rec in records.items())
        options: list[str] = []
        for row in rows:
            needs = "; ".join(f"{role} needs {test} (qualifying: {_qualifying(_offers(cls.roles[role]), test)})"
                              for role, test in row.when.items())
            options.append(f"{'rung ' + row.rung + ': ' if row.rung else ''}{needs}")
        self.refuse("support.unsupported", RefusalClass.USER,
                    f"{where}{f' with rung {rung!r}' if rung else ''} does not support what it is bound to ({bound})",
                    "the supported constructions are: " + " | ".join(options or ["none for this rung"]))
        return False

    def interface(self, cls: type[Entry], name: str, kinds: tuple[type, ...], what: str) -> vocabulary.Interface | None:
        entry = vocabulary.lookup(name)
        if entry is None:
            self.defect(cls, f"it {what} {name!r}, which is not in the vocabulary")
            return None
        if entry.reserved:
            self.refuse("vocabulary.reserved", RefusalClass.USER,
                        f"{cls.label()} {what} {name!r}, which is reserved, not built: its kind "
                        f"({entry.kind.value}), units ({entry.units}), frame ({entry.frame}) and sign are fixed and "
                        f"nothing builds it yet",
                        f"choose an entry that does not use {name!r} until it is built")
            return None
        if not isinstance(entry.kind, kinds):
            self.defect(cls, f"it {what} {name!r}, which is a {entry.kind.value} interface")
            return None
        return entry

    def state_read(self, plan: EntryPlan, where: str, read: Read) -> ResolvedRead | None:
        cls = plan.reader_cls()
        role_kind = cls.roles.get(read.role)
        if role_kind is None or role_kind is RoleKind.FIELD:
            self.defect(cls, f"it reads state through role {read.role!r}, which is not one of its thing roles")
            return None
        entry = self.interface(cls, read.interface, (ReadKind,), "reads")
        if entry is None:
            return None
        if entry.kind is not ReadKind.STATE:
            self.defect(cls, f"it reads the field {read.interface!r} as state; read it through a field role")
            return None
        thing = plan.bindings[read.role]
        dynamics = self.dynamics.get(thing)
        port = dynamics.record.publishes.get(read.interface) if dynamics and dynamics.record else None
        if dynamics is None or port is None:
            has = f"its dynamics {dynamics.describe()}" if dynamics else "it has no dynamics, so it"
            self.refuse("read.not-published", RefusalClass.USER,
                        f"{where} reads {read.interface} from {role_kind.value} {thing!r}, and {has} "
                        f"does not publish it",
                        f"give {thing!r} a dynamics that publishes {read.interface}: "
                        + _qualifying(EntryKind.DYNAMICS, lambda r: read.interface in r.publishes))
            return None
        return ResolvedRead(dynamics.node_name(port.node), port.field, read.interface, entry.units)

    def plan_reads(self, plan: EntryPlan, where: str) -> None:
        cls = plan.reader_cls()
        assert plan.plan is not None
        sources: dict[tuple[str, ResolvedRead], str] = {}
        for label in sorted(plan.plan.reads):
            read = plan.plan.reads[label]
            if cls.roles.get(read.role) is not RoleKind.FIELD:
                if read.at is not None:
                    self.defect(cls, f"read {label!r}: only a field is read at points")
                    continue
                resolved = self.state_read(plan, where, read)
                if resolved is not None:
                    plan.reads[label] = resolved
                continue
            entry = self.interface(cls, read.interface, (ReadKind,), "reads")
            if entry is None:
                continue
            if entry.kind is not ReadKind.FIELD:
                self.defect(cls, f"it reads the state {read.interface!r} through field role {read.role!r}")
                continue
            if read.at is None:
                self.defect(cls, f"read {label!r} of field {read.interface!r} gives no points (at=)")
                continue
            path = plan.bindings[read.role]
            provider = self.providers[path]
            prefix = provider.record.publishes.get(read.interface) if provider.record else None
            if prefix is None:
                self.refuse("read.not-published", RefusalClass.USER,
                            f"{where} reads {read.interface} from field {path!r}, whose provider "
                            f"{provider.describe()} does not publish it",
                            f"bind a field whose provider publishes {read.interface}: "
                            + _qualifying(EntryKind.FIELD_PROVIDER,
                                          lambda r, name=read.interface: name in r.publishes))
                continue
            if plan.section is Section.OBSERVATION:
                self.refuse("observation.field-read", RefusalClass.USER,
                            f"{where} reads the field {path!r}; an observation reads a field through its own "
                            f"instance of the provider, which is not built yet",
                            "observe the state the field acts on instead, until observations' own providers land")
                continue
            points = self.state_read(plan, where, read.at)
            if points is None:
                continue
            key = (path, points)
            slot = sources.setdefault(key, _sanitize(plan.owner) + (f"_{len(sources)}" if sources else ""))
            plan.reads[label] = ResolvedRead(provider.owner, f"{prefix.field}_{slot}", read.interface, entry.units,
                                             points, slot)

    def target(self, plan: EntryPlan, where: str, delivery: Delivery) -> ResolvedTarget | None:
        cls = plan.reader_cls()
        role_kind = cls.roles.get(delivery.role)
        if role_kind is None or role_kind is RoleKind.FIELD:
            self.defect(cls, f"it delivers through role {delivery.role!r}, which is not one of its thing roles")
            return None
        entry = self.interface(cls, delivery.target, (TargetKind,), "delivers into")
        if entry is None:
            return None
        kind = entry.kind
        if not isinstance(kind, TargetKind):
            self.defect(cls, f"it delivers into {delivery.target!r}, which is a {kind.value} interface")
            return None
        if kind is TargetKind.REGISTRATION:
            self.defect(cls, f"it delivers into the registration {delivery.target!r}; plan a Registration instead")
            return None
        if kind is TargetKind.INEQUALITY_CONSTRAINT:
            self.refuse("target.not-built", RefusalClass.USER,
                        f"{where} delivers into {delivery.target}, an inequality constraint: the dynamics takes it "
                        f"at construction, which is not built yet",
                        "remove the interaction until inequality-constraint targets are built")
            return None
        thing = plan.bindings[delivery.role]
        dynamics = self.dynamics.get(thing)
        port = dynamics.record.accepts.get(delivery.target) if dynamics and dynamics.record else None
        if dynamics is None or port is None:
            has = f"its dynamics {dynamics.describe()}" if dynamics else "it has no dynamics, so it"
            self.refuse("target.not-accepted", RefusalClass.USER,
                        f"{where} delivers into {delivery.target} of {role_kind.value} {thing!r}, and {has} does "
                        f"not accept it",
                        f"give {thing!r} a dynamics that accepts {delivery.target}: "
                        + _qualifying(EntryKind.DYNAMICS, lambda r: delivery.target in r.accepts))
            return None
        return ResolvedTarget(dynamics.node_name(port.node), port.field, kind, thing, delivery.target, entry.units,
                              plan.owner)

    def plan_registration(self, plan: EntryPlan, where: str, registration: Registration) -> None:
        cls = plan.reader_cls()
        for role in (registration.role, registration.subject):
            if cls.roles.get(role) in (None, RoleKind.FIELD):
                self.defect(cls, f"its registration names role {role!r}, which is not one of its thing roles")
                return
        entry = self.interface(cls, registration.target, (TargetKind,), "registers through")
        if entry is None:
            return
        if entry.kind is not TargetKind.REGISTRATION:
            self.defect(cls, f"its registration target {registration.target!r} is a {entry.kind.value} target")
            return
        receiver = plan.bindings[registration.role]
        dynamics = self.dynamics.get(receiver)
        record = dynamics.record if dynamics else None
        if dynamics is None or record is None or registration.target not in record.accepts \
                or registration.kind not in record.registration_kinds:
            has = f"its dynamics {dynamics.describe()}" if dynamics else "it has no dynamics, so it"
            self.refuse("registration.not-accepted", RefusalClass.USER,
                        f"{where} registers {plan.bindings[registration.subject]!r} with {receiver!r} as "
                        f"{registration.kind}, and {has} does not accept that registration",
                        f"give {receiver!r} a dynamics that accepts {registration.kind}: "
                        + _qualifying(EntryKind.DYNAMICS, lambda r: registration.kind in r.registration_kinds))
            return
        reads = {label: self.state_read(plan, where, Read(registration.subject, interface))
                 for label, interface in registration.reads.items()}
        returns = {label: self.target(plan, where, Delivery(registration.subject, target))
                   for label, target in registration.returns.items()}
        resolved_reads = {label: read for label, read in reads.items() if read is not None}
        resolved_returns = {label: target for label, target in returns.items() if target is not None}
        if len(resolved_reads) != len(reads) or len(resolved_returns) != len(returns):
            return
        subject = plan.bindings[registration.subject]
        dynamics.registrations.append(
            ResolvedRegistration(plan.owner, registration.kind, subject, resolved_reads, resolved_returns))

    def check_request(self, plan: EntryPlan, request: GroupRequest) -> None:
        cls = plan.reader_cls()
        tolerance, cap = cast(object, request.tolerance), cast(object, request.max_iterations)  # an entry's own values
        if not (isinstance(tolerance, float) and math.isfinite(tolerance) and tolerance > 0.0):
            self.defect(cls, f"its coupling-group request gives tolerance {tolerance!r}, not a float above zero")
        if not (isinstance(cap, int) and not isinstance(cap, bool) and cap >= 1):
            self.defect(cls, f"its coupling-group request gives max_iterations {cap!r}, not an integer >= 1")
        for role in request.roles:
            if cls.roles.get(role) in (None, RoleKind.FIELD):
                self.defect(cls, f"its coupling-group request names role {role!r}, not one of its thing roles")
            elif plan.bindings[role] not in self.dynamics:
                self.refuse("group.no-dynamics", RefusalClass.USER,
                            f"{plan.owner} asks to solve {plan.bindings[role]!r} together with it, and "
                            f"{plan.bindings[role]!r} has no dynamics",
                            f"give {plan.bindings[role]!r} a dynamics, or bind another thing")

    # -- across every plan ----------------------------------------------------------------------------------

    def check_ids(self) -> None:
        seen: dict[str, EntryPlan] = {}
        for plan in self.readers:
            if plan.owner in seen:
                self.refuse("id.duplicate", RefusalClass.USER,
                            f"two declarations share the id {plan.owner!r}: an id is the registered name and the "
                            f"things bound to it, so the same entry is bound to the same things twice",
                            "remove the duplicate declaration")
            seen[plan.owner] = plan

    def collect_registrations_and_slots(self) -> None:
        for dynamics in self.dynamics.values():
            dynamics.registrations.sort(key=lambda r: r.id)
        slots: dict[str, dict[str, Slot]] = {}
        for plan in self.readers:
            for read in plan.reads.values():
                if read.slot is None:
                    continue
                held = slots.setdefault(read.node, {}).get(read.slot)
                interfaces = (*(held.interfaces if held else ()), read.interface)
                if held and held.reader != plan.owner:
                    self.refuse("slot.collision", RefusalClass.INTERNAL,
                                f"field {read.node!r}: readers {held.reader} and {plan.owner} map to one slot name "
                                f"{read.slot!r}", MIME_DEFECT)
                slots[read.node][read.slot] = Slot(read.slot, plan.owner, tuple(dict.fromkeys(interfaces)))
        for path, provider in list(self.providers.items()):
            provider.slots = [slots[path][name] for name in sorted(slots.get(path, {}))]
            if not provider.slots:
                # A field nothing reads builds no node: there is no reader slot to give it.
                self.declared.remove(provider)
                del self.providers[path]

    def collect_targets(self) -> dict[tuple[str, str], ResolvedTarget]:
        targets: dict[tuple[str, str], ResolvedTarget] = {}
        for plan in self.dynamics.values():
            assert plan.record is not None
            for name, port in plan.record.accepts.items():
                entry = vocabulary.lookup(name)
                if entry is None or not isinstance(entry.kind, TargetKind):
                    self.defect(plan.registered.cls, f"it accepts {name!r}, which is not a target in the vocabulary")
                    continue
                if entry.reserved:
                    continue
                targets[(plan.node_name(port.node), port.field)] = ResolvedTarget(
                    plan.node_name(port.node), port.field, entry.kind, plan.thing or "", name, entry.units)
        return targets

    def deliveries(self) -> list[ResolvedTarget]:
        found = [t for plan in self.readers for t in plan.delivers.values()]
        found += [t for plan in self.dynamics.values() for r in plan.registrations for t in r.returns.values()]
        return found

    def check_exclusive(self) -> None:
        by_port: dict[tuple[str, str], list[ResolvedTarget]] = {}
        for target in self.deliveries():
            if target.kind.exclusive:
                by_port.setdefault((target.node, target.field), []).append(target)
        for found in by_port.values():
            if len(found) > 1:
                first = found[0]
                self.refuse("target.exclusive", RefusalClass.USER,
                            f"{first.interface} of {first.thing!r} is an {first.kind.value} and takes one delivery; "
                            f"{', '.join(sorted(t.by for t in found))} all deliver into it",
                            "keep one of them")

    def union_groups(self) -> tuple[GroupPlan, ...]:
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            while parent.setdefault(node, node) != node:
                node = parent[node]
            return node

        requests: list[tuple[EntryPlan, GroupRequest, list[str]]] = []
        for plan in self.readers:
            assert plan.plan is not None
            for request in plan.plan.groups:
                members = [n for role in request.roles if plan.bindings.get(role) in self.dynamics
                           for n in self.dynamics[plan.bindings[role]].node_names()]
                members += plan.node_names() if request.own_nodes else []
                if not members:
                    continue
                for node in members[1:]:
                    parent[find(node)] = find(members[0])
                requests.append((plan, request, members))
        merged: dict[str, list[tuple[EntryPlan, GroupRequest, list[str]]]] = {}
        for item in requests:
            merged.setdefault(find(item[2][0]), []).append(item)
        dividers = {n: p.divider for p in self.declared for n in p.node_names()}
        groups: list[GroupPlan] = []
        for items in merged.values():
            nodes = sorted({n for _, _, members in items for n in members})
            settings = {(r.tolerance, r.max_iterations) for _, r, _ in items}
            by = tuple(sorted(f"{plan.owner}: {request.reason}" for plan, request, _ in items))
            if len(settings) > 1:
                self.refuse("group.settings", RefusalClass.USER,
                            f"the requests merged into one coupling group over {', '.join(nodes)} ask for different "
                            f"tolerances or caps: " + "; ".join(
                                f"{plan.owner}: tolerance {r.tolerance!r}, max_iterations {r.max_iterations}"
                                for plan, r, _ in items),
                            "make the merged requests agree")
                continue
            if len({dividers.get(n, 1) for n in nodes}) > 1:
                self.refuse("group.rates", RefusalClass.USER,
                            f"the coupling group over {', '.join(nodes)} has members at different rates: "
                            + ", ".join(f"{n} every {dividers.get(n, 1)} steps" for n in nodes),
                            "give every member of a coupling group the same rate")
                continue
            ((tolerance, cap),) = settings
            groups.append(GroupPlan(tuple(nodes), tolerance, cap, by))
        return tuple(groups)

    def check_rates(self) -> None:
        if not self.declared or not self.dt:
            return
        dividers = {n: p.divider for p in self.declared for n in p.node_names()}
        if dividers and min(dividers.values()) > 1:
            self.refuse("run.no-node-at-dt", RefusalClass.USER,
                        f"no node runs at the base step dt = {_seconds(self.dt)} (the fastest runs every "
                        f"{min(dividers.values())} steps); MADDENING would take the greatest common divisor of the "
                        f"rates as its base step instead of dt",
                        f"run some entry at dt, or raise dt to {_seconds(min(dividers.values()) * self.dt)}")

    def run_structure(self) -> RunStructure:
        """Nothing an experiment declares yet runs on the host between compiled calls (a controller in modes 2 to
        4, a per-chunk hook, an observation outside the step, a study's variant loop), so every run is one compiled
        scan; each of those adds itself here when it lands."""
        return RunStructure()

    def derive_seeds(self) -> None:
        for plan in self.readers:
            assert plan.plan is not None
            for stream in plan.plan.streams:
                plan.seeds[stream] = derive_seed(self.settings.seed, plan.owner, plan.bindings, stream)


def derive_seed(seed: int, owner: str, bindings: Mapping[str, str], stream: str) -> int:
    """A stream's seed from the experiment's seed and the stream's identity (its entry's id, bindings and name), so
    attaching or removing another entry leaves it unchanged."""
    identity = "\x1f".join([str(seed), owner, *(f"{k}={v}" for k, v in sorted(bindings.items())), stream])
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")


def plan_experiment(settings: RunSettings, declarations: list[Declaration]) -> PlanSet:
    """Phase 1. Raises :class:`BuildRefused` with every reason found."""
    return _Planner(settings, declarations).run()
