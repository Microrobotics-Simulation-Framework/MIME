"""The experiment's profile at build time: the defaults taken, the sources of its settings, what its profile's rules
made of each finding, and what is left pending, with the lines to paste.

Every setting of the experiment has a path (``tube.inner_radius``, ``umr.parts.shell.radius``, ``umr.parts.shell.at``,
``field.magnetic.rotation_axis``, ``field.frequency`` for a command's value, ``drag:umr.spacing``, ``controller.gain``,
``controller.rate``, ``dt``, ``gravity``, ``conditions.temperature``), its declared role and its source: the reason its
quantity carries (``mm(3.175, reason=...)``) or one ``exp.source(path, reason=...)`` gives. Each role is a finding class
in the profile rules (design #191 item 3): a physical value with no source (a convention taken for one included), a
numerical value with none (its convergence evidence until a1's evidence records), a numerical default taken, a default
its entry declares nothing of, and a pending reason anywhere. Every site applies its class's rule through one function,
:func:`mime.validity.decide`, at build time: a profile changes what is refused, waived or listed, never a value or a
graph. A reason written where no site judges it is refused as an input error naming the slot (``source.no-site``).

What the profile made of an experiment is recorded in a canonical order (by target, then code), so that the order a
script declares things in never reaches the saved form's hashed record."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from mime.build._refusals import RefusalClass
from mime.materials import read_target
from mime.registry import (
    DefaultKind,
    Entry,
    Interaction,
    Observation,
    Role,
    SettingGroup,
    SettingRole,
    declared_roles,
    registration_of,
    setting_names,
)
from mime.things import Apparatus, Body, Medium, Structure, TypedPath, did_you_mean
from mime.units import Condition, Quantity, unit_symbol
from mime.validity import (
    PENDING,
    REASON,
    Action,
    Basis,
    Chosen,
    Decision,
    Finding,
    Judge,
    PendingItem,
    Profile,
    Stamp,
    Waiver,
    basis_problem,
    basis_text,
    decide,
    is_pending,
    pending_text,
)

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type Problem = tuple[str, str, str]

#: The finding codes the build gives (finding_classes.txt): a reason left pending, a numerical default taken, a default
#: its entry declares nothing of, a physical value with no source, a numerical value with no evidence.
BASIS_PENDING = "basis_pending"
DEFAULT_NUMERICAL = "default_numerical"
DEFAULT_UNDECLARED = "default_undeclared"
SOURCE_MISSING = "source_missing"
EVIDENCE_MISSING = "evidence_missing"

#: The SI unit each dimension's quantity is written in on a line to paste, by its symbol: a constructor of mime.units.
_CONSTRUCTORS = {"m": "m", "s": "s", "kg": "kg", "Hz": "Hz", "N": "N", "A": "A", "K": "K", "T": "T", "m/s": "m_per_s",
                 "m/s^2": "m_per_s2", "1": "dimensionless", "m^3": "m3", "kg/m^3": "kg_per_m3", "kg m^2": "kg_m2",
                 "A m^2": "A_m2", "Pa s": "Pa_s", "N m": "N_m", "N m s": "N_m_s"}
_UNITS = "mime.units"
_ACCEPTANCE = ("mime.materials", "Acceptance")
_NOMINAL = ("mime.materials", "Nominal")


def _number(value: object) -> str:
    if isinstance(value, tuple):
        return "(" + ", ".join(repr(float(c)) for c in cast(tuple[float, ...], value)) + ")"
    return repr(value)


def _unit_of(value: Quantity) -> str | None:
    symbol = unit_symbol(value.dimension)
    return "rad_per_s" if symbol == "Hz" and value.angle == "radian" else _CONSTRUCTORS.get(symbol)


def quantity_line(value: Quantity, reason: str | None = REASON) -> str:
    """``value`` as a script writes it, in SI, with ``reason`` in its slot (``kg_per_m3(1410.0, reason=...)``), or
    with none."""
    unit = _unit_of(value)
    tail = f", reason={reason}" if reason is not None else ""
    if unit is None:
        return f"Quantity({_number(value.value)}, parse_unit({unit_symbol(value.dimension)!r}){tail})"
    return f"{unit}({_number(value.value)}{tail})"


def quantity_imports(value: Quantity) -> tuple[tuple[str, str], ...]:
    """What :func:`quantity_line`'s line needs imported."""
    unit = _unit_of(value)
    return ((_UNITS, unit),) if unit is not None else ((_UNITS, "Quantity"), (_UNITS, "parse_unit"))


def _quote(text: str) -> str:
    return json.dumps(text)


def quantity_doc(value: Quantity, reason: Json = REASON) -> Json:
    magnitude = value.value
    held: Json = [float(c) for c in magnitude] if isinstance(magnitude, tuple) else float(cast(float, magnitude))
    return {"value": held, "unit": unit_symbol(value.dimension), "reason": reason}


@dataclass(frozen=True)
class DefaultTaken:
    """A default the build took for a setting the experiment did not give: its path, its kind, the value, its rule and
    basis in words, the entry whose setting it is (``drag 0.3.0``; ``None`` for a run setting) and what was measured of
    it. The saved form records its kind, value and entry, never its words, so a reworded basis changes no lock."""

    path: str
    kind: DefaultKind
    value: object
    rule: str
    basis: str
    entry: str | None = None
    measured: str = ""

    def text(self) -> str:
        shown = str(self.value) if isinstance(self.value, Quantity) else repr(self.value)
        held = f"{self.path} = {shown}, a {self.kind.value} default ({self.rule}; {self.basis})"
        return held + (f"; measured: {self.measured}" if self.measured else "")


@dataclass(frozen=True)
class Site:
    """One setting of the experiment: its path, its declared role (``None``: it declares none), the value as written,
    its written default, where the value is written (for the line to paste), its name there, how a document fragment
    reaches it (a function of its settings), and its entry (``drag 0.3.0``)."""

    path: str
    role: SettingRole | None
    value: object
    written: object
    where: str
    setting: str
    document: Callable[[dict[str, Json]], Json] | None = None
    entry: str | None = None


@dataclass(frozen=True)
class ProfileOutcome:
    """What the profile made of the experiment: the profile, every waiver the build rests on with a pending basis
    (generated, or the experiment's own), the defaults taken, the items left pending and those listed, every setting's
    source by path, and each value given as the experimenter's own choice, by path, with its declared role (a declared
    change of its role). Each in a canonical order."""

    profile: Profile
    waivers: tuple[Waiver, ...] = ()
    defaults: tuple[DefaultTaken, ...] = ()
    items: tuple[PendingItem, ...] = ()
    sources: Mapping[str, Basis] = field(default_factory=dict[str, Basis])
    changes: Mapping[str, str] = field(default_factory=dict[str, str])

    @property
    def pending(self) -> tuple[PendingItem, ...]:
        return tuple(item for item in self.items if not item.listed)

    @property
    def listed(self) -> tuple[PendingItem, ...]:
        return tuple(item for item in self.items if item.listed)

    def stamp(self) -> Stamp:
        return Stamp(self.profile.value, tuple(item.line() for item in self.pending),
                     tuple(f"{item.target}: {item.measured}" for item in self.pending if item.measured))

    def text(self) -> str:
        """Each item in plain words with its lines to paste (``built.report.pending()``)."""
        return pending_text(self.profile.value, self.items)

    def report_lines(self) -> list[str]:
        lines = [self.stamp().text()]
        lines += [f"  pending: {item.line()}" for item in self.pending]
        lines += [f"  listed: {item.line()}" for item in self.listed]
        lines += [f"  default: {taken.text()}" for taken in self.defaults]
        lines += [f"  declared change: {path}: declared {role}, given here as the experimenter's choice "
                  f"({basis_text(self.sources[path]) if path in self.sources else 'chosen'})"
                  for path, role in self.changes.items()]
        return lines


def _fragment(*path: str) -> Callable[[dict[str, Json]], Json]:
    """A function wrapping a mapping of settings into a document fragment at ``path`` (``things``, ``tube``,
    ``geometry``): what the formatter prints, merged into the document by key."""
    def wrap(settings: dict[str, Json]) -> Json:
        held: Json = {"settings": cast(Json, settings)}
        for key in reversed(path):
            held = {key: held}
        return held
    return wrap


def _bare_fragment(*path: str) -> Callable[[dict[str, Json]], Json]:
    """As :func:`_fragment`, for a slot the document writes directly under ``path`` (a part's ``at``, the controller's
    ``rate``), not among an entry's settings."""
    def wrap(settings: dict[str, Json]) -> Json:
        held: Json = cast(Json, settings)
        for key in reversed(path):
            held = {key: held}
        return held
    return wrap


def _listed_fragment(key: str, name_key: str, name: str, *outer: str) -> Callable[[dict[str, Json]], Json]:
    """As :func:`_fragment`, for an entry kept in a list (an interaction, an observation, a chain segment), named by
    ``name_key``."""
    def wrap(settings: dict[str, Json]) -> Json:
        held: Json = {key: [{name_key: name, "settings": cast(Json, settings)}]}
        for part in reversed(outer):
            held = {part: held}
        return held
    return wrap


def _entry_sites(entry: object, prefix: str, where: str, document: Callable[[dict[str, Json]], Json] | None,
                 value_path: str | None = None) -> list[Site]:
    """Every setting of a registered entry as a site, a group's own settings below it. ``value_path`` names a setting
    whose path is the prefix itself (a constant drive's value is its command's)."""
    record = registration_of(cast(Entry, entry))
    if record is None:
        return []
    found: list[Site] = []
    owner = f"{record.name} {record.version}"
    for name in setting_names(record.cls):
        value = getattr(entry, name, None)
        path = prefix if name == value_path else f"{prefix}.{name}"
        role = record.roles.get(name)
        written = _written(record.cls, name)
        keyed = _keyed(value) if role is not None and role.role is Role.PHYSICAL else None
        if keyed:
            # a physical setting keyed by name (an arm's armature by joint): each quantity is a site of its own, with
            # the setting's role, its reason judged there; a name of a record that carries its own source (a
            # registered actuator) is no site
            found += [Site(f"{path}.{key}", role, held, dataclasses.MISSING, f"{where}'s {name}=", f"{name}[{key!r}]",
                           None, owner) for key, held in keyed if isinstance(held, Quantity)]
            continue
        found.append(Site(path, role, value, written, where, name, document, owner))
        if isinstance(value, SettingGroup):
            group_roles = declared_roles(type(value))
            for inner in setting_names(type(value)):
                found.append(Site(f"{path}.{inner}", group_roles.get(inner, role), getattr(value, inner, None),
                                  _written(type(value), inner), f"{where}'s {name}=", inner, None, owner))
    return found


def _keyed(value: object) -> list[tuple[str, object]] | None:
    """A setting held as pairs of a name and a quantity or a record's name (``(("joint_6", kg_m2(...)), ...)``), as
    its pairs; ``None`` for any other setting, or one holding none."""
    if not isinstance(value, tuple) or not value:
        return None
    pairs = cast(tuple[object, ...], value)
    if not all(isinstance(pair, tuple) and len(cast(tuple[object, ...], pair)) == 2
               and isinstance(cast(tuple[object, ...], pair)[0], str)
               and isinstance(cast(tuple[object, ...], pair)[1], (Quantity, str)) for pair in pairs):
        return None
    return list(cast(tuple[tuple[str, object], ...], pairs))


def _written(cls: type, name: str) -> object:
    held = next((f for f in dataclasses.fields(cls) if f.name == name), None)
    if held is None:
        return dataclasses.MISSING
    if held.default is not dataclasses.MISSING:
        return cast(object, held.default)
    if held.default_factory is not dataclasses.MISSING:
        return cast(object, held.default_factory())
    return dataclasses.MISSING


#: A part's placement: a fact about the device (where a magnet sits in its shell), required where it is used.
_PART_AT = SettingRole(Role.PHYSICAL)


def _part_sites(owner: str, label: str, parts: Mapping[str, Any]) -> list[Site]:
    """Each part's solid's settings, and its placement ``at``."""
    found: list[Site] = []
    for key, part in sorted(parts.items()):
        found += _entry_sites(part.geometry, f"{owner}.parts.{key}", f"in {label} {owner!r}'s part {key!r}",
                              _fragment("things", owner, "parts", key, "solid"))
        found.append(Site(f"{owner}.parts.{key}.at", _PART_AT, part.at, None, f"in {label} {owner!r}'s part {key!r}'s "
                          f"Part(...)", "at", _bare_fragment("things", owner, "parts", key)))
    return found


def experiment_sites(things: Sequence[object], readers: Sequence[tuple[str, str, object]],
                     controller: object | None) -> list[Site]:
    """Every setting of the experiment's things, interactions, observations and controller (``readers``: each
    interaction and observation with its section and id; ``controller``: its declaration), by path."""
    found: list[Site] = []
    for thing in things:
        if isinstance(thing, Structure):
            if thing.geometry is not None:
                found += _entry_sites(thing.geometry, thing.name, f"in structure {thing.name!r}'s "
                                      f"{type(thing.geometry).__name__}(...)", _fragment("things", thing.name,
                                                                                         "geometry"))
            if thing.dynamics is not None:
                found += _entry_sites(thing.dynamics, f"{thing.name}.motion", f"in structure {thing.name!r}'s "
                                      f"dynamics", _fragment("things", thing.name, "dynamics", "motion"))
        elif isinstance(thing, Medium):
            if thing.flow is not None:
                found += _entry_sites(thing.flow, f"{thing.name}.flow", f"in medium {thing.name!r}'s flow",
                                      _fragment("things", thing.name, "dynamics", "flow"))
        elif isinstance(thing, Body):
            found += _entry_sites(thing.dynamics, f"{thing.name}.motion", f"in body {thing.name!r}'s dynamics",
                                  _fragment("things", thing.name, "dynamics", "motion"))
            found += _part_sites(thing.name, "body", thing.parts)
            for key, value in thing.initial.items():
                found.append(Site(f"{thing.name}.initial.{key}", SettingRole(Role.DESIGN), value, dataclasses.MISSING,
                                  f"in body {thing.name!r}'s State(...)", key, None))
        elif isinstance(thing, Apparatus):
            for name, provider in sorted(thing.fields.items()):
                found += _entry_sites(provider, f"{thing.name}.{name}", f"in apparatus {thing.name!r}'s field "
                                      f"{name!r}", _fragment("things", thing.name, "fields", name))
            for name, drive in sorted(thing.commands.items()):
                found += _entry_sites(drive, f"{thing.name}.{name}", f"in apparatus {thing.name!r}'s commands",
                                      _fragment("things", thing.name, "commands", name), value_path="value")
            for name, segment in thing.chain.items():
                found += _entry_sites(segment, f"{thing.name}.{name}", f"in apparatus {thing.name!r}'s segment "
                                      f"{name!r}", _listed_fragment("chain", "segment", name, "things", thing.name))
            found += _part_sites(thing.name, "apparatus", thing.parts)
    for section, owner, entry in readers:
        found += _entry_sites(entry, owner, f"in {section} {owner}'s {type(entry).__name__}(...)",
                              _listed_fragment(f"{section}s", "id", owner))
    if controller is not None:
        entry = getattr(controller, "controller", None)
        found += _entry_sites(entry, "controller", f"in the controller's {type(entry).__name__}(...)",
                              _fragment("controller"))
        taken = {site.path for site in found}
        for name, role in (("rate", SettingRole(Role.DESIGN)), ("latency", SettingRole(Role.PHYSICAL))):
            if f"controller.{name}" not in taken:
                found.append(Site(f"controller.{name}", role, getattr(controller, name, None), None,
                                  "in exp.set_controller(...)", name, _bare_fragment("controller")))
    return found


def _reason_of(value: object) -> Basis | None:
    if isinstance(value, Quantity):
        return value.reason
    if isinstance(value, Condition):
        return value.value.reason
    return None


def _judged(value: object) -> Iterator[int]:
    """The quantities a site judges the reason of: its value's, or a condition's value's."""
    if isinstance(value, Quantity):
        yield id(value)
    elif isinstance(value, Condition):
        yield id(value.value)


def _same(value: object, written: object) -> bool:
    """Whether ``value`` is its setting's written default: of one type and equal, a tuple's subclass (a signed axis)
    counting as a tuple, as the document reads it."""
    if written is dataclasses.MISSING or written is None:
        return False
    kinds: tuple[type, type] = (type(value), type(written))
    tuples = all(issubclass(kind, tuple) for kind in kinds)
    return (tuples or kinds[0] is kinds[1]) and value == written


def reasons_without_a_site(roots: Sequence[tuple[str, object]], judged: set[int]) -> list[str]:
    """Where a quantity carries a reason that no site judges (a nominal's value, a condition's tolerance), as a path
    from its root (``material 'n45'.nominal['magnetic.remanence'].value``): a reason there is neither judged nor
    saved, so it is refused as an input error naming the slot."""
    found: list[str] = []
    seen: set[int] = set()

    def walk(value: object, where: str, depth: int) -> None:
        if isinstance(value, Quantity):
            if value.reason is not None and id(value) not in judged:
                found.append(where)
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool, type)) or depth > 16 or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, Mapping):
            for key, held in cast(Mapping[object, object], value).items():
                walk(held, f"{where}[{key!r}]", depth + 1)
        elif isinstance(value, (tuple, list)):
            for index, held in enumerate(cast(Sequence[object], value)):
                walk(held, f"{where}[{index}]", depth + 1)
        elif dataclasses.is_dataclass(value):
            for held in dataclasses.fields(value):
                walk(getattr(value, held.name, None), f"{where}.{held.name}", depth + 1)

    for name, root in roots:
        walk(root, name, 0)
    return found


class ProfileWork:
    """The profile's work while an experiment is planned: the judge, the defaults taken, the waivers and items, and
    what to refuse. The planner feeds it as it goes, and :meth:`finish` judges every setting. Every site applies its
    class's rule through :meth:`apply`, and so through :func:`mime.validity.decide`."""

    def __init__(self, profile: Profile) -> None:
        self.judge = Judge(profile)
        self.defaults: dict[str, DefaultTaken] = {}
        self.waivers: list[Waiver] = []
        self.items: list[PendingItem] = []
        self.problems: list[tuple[str, RefusalClass, str, str]] = []
        self.products: list[tuple[str, object]] = []
        self.sources: dict[str, Basis] = {}
        self.changes: dict[str, str] = {}
        for problem in self.judge.problems():
            self.refuse("profile.tables", RefusalClass.INTERNAL,
                        f"the profile tables hold a row no site can apply: {problem}",
                        "report it: a profile rule generates or accepts a waiver and nothing else, lists only what a "
                        "waiver, a source or evidence covers, and names a class with codes")

    @property
    def profile(self) -> Profile:
        return self.judge.profile

    def refuse(self, code: str, category: RefusalClass, message: str, action: str) -> None:
        self.problems.append((code, category, message, action))

    def take_default(self, taken: DefaultTaken) -> None:
        self.defaults[taken.path] = taken

    def _add(self, item: PendingItem) -> None:
        if item not in self.items:
            self.items.append(item)

    def _rule_text(self, decision: Decision) -> str:
        name = self.judge.class_of(decision.finding.code)
        action = decision.action.value if decision.action is not None else "nothing"
        if decision.from_rule:
            return f"the {self.profile.value} profile's rule for {name}: {action}"
        return f"refused with no rule for it in the {self.profile.value} profile, as before profiles"

    def record(self, decision: Decision, item: PendingItem, *, own: bool = False) -> None:
        """What a decision leaves on the record: a ``generate`` rule's waiver, with ``item`` pending and stamped
        (``own``: the waiver accepts the experiment's own pending reason, as it is), or anything else it made refused,
        since profiles act on waivers alone; a ``list`` rule's item, listed."""
        if decision.action is Action.GENERATE:
            product = decision.product
            self.products.append((decision.finding.target, product))
            if isinstance(product, Waiver):
                self.waivers.append(replace(product, generated=False) if own else product)
                self._add(item)
            else:
                self.refuse("profile.not-a-waiver", RefusalClass.INTERNAL,
                            f"the {self.profile.value} profile's rule for {decision.finding.target} generated a "
                            f"{type(product).__name__}, and profiles act on waivers alone",
                            "report it: a profile rule generates or accepts a waiver and nothing else")
        elif decision.action is Action.LIST:
            self._add(replace(item, listed=True))

    def apply(self, finding: Finding, item: PendingItem, baseline: Action | None, refusal: Problem, *,
              own: bool = False) -> bool:
        """``finding``'s class's rule applied (:func:`mime.validity.decide`), with ``baseline`` what the site did before
        profiles: recorded (:meth:`record`), and where it stops the finding (``basis``, ``refuse``), refused as
        ``refusal`` says, naming the rule. Whether the site goes on with it."""
        decision = decide(self.judge, finding, baseline)
        self.record(decision, item, own=own)
        if not decision.proceeds and decision.action is not Action.GENERATE:
            code, message, action = refusal
            self.refuse(code, RefusalClass.USER, f"{message} ({self._rule_text(decision)})", action)
        return decision.proceeds

    def pending(self, code: str, target: str, what: str, item: PendingItem) -> bool:
        """A reason left pending (``target``), judged by the profile's rule for its class: recorded with its item and
        stamped where the rule generates, refused by name (``code``) where it refuses or has no rule. Whether the build
        goes on with it."""
        return self.apply(Finding(target, BASIS_PENDING, what), item, Action.REFUSE,
                          (code, f"{target}: {what}, and a pending reason is a placeholder, not a reason",
                           f"give it its reason: {paste_command(item.python)}"), own=True)

    # -- the experiment's reads, declarations and settings ------------------------------------------------------

    def reads(self, materials: Mapping[str, Any], outcomes: Sequence[Any], every: Sequence[Any] = ()) -> None:
        """Each material's acceptances with a pending reason, and each read the build rests on with what the profile
        decided of its unknown reasons, as one item per read: its acceptance's covers and the reasons waived together,
        so its pasted line is one acceptance. ``outcomes`` are the reads' ``ReadOutcome``s; ``every`` every read the
        build made, whose rules' products are held to being waivers even where the build does not rest on it."""
        resting = {(outcome.material, outcome.field) for outcome in outcomes}
        for outcome in every:
            if (outcome.material, outcome.field) not in resting:
                for decision in outcome.decisions:
                    if decision.action is Action.GENERATE and not isinstance(decision.product, Waiver):
                        self.record(decision, acceptance_item(outcome.material, outcome.field, (), ""))
        decided: dict[tuple[str, str], tuple[list[Decision], tuple[str, ...]]] = {}
        for outcome in outcomes:
            if outcome.decisions:
                decided[(outcome.material, outcome.field)] = (list(outcome.decisions), tuple(outcome.reasons))
        accepted = {(name, field_name): acceptance for name, material in materials.items()
                    for field_name, acceptance in material.accept.items()}
        for key in sorted({*decided, *(k for k, a in accepted.items() if is_pending(getattr(a, "reason", None)))}):
            material, field_name = key
            acceptance = accepted.get(key)
            covers = tuple(acceptance.covers) if acceptance is not None else ()
            decisions, texts = decided.get(key, ([], ()))
            waived = [d for d in decisions if d.action is Action.GENERATE]
            listed = [d for d in decisions if d.action is Action.LIST]
            codes = [code for d in waived for code in getattr(d.product, "reasons", (d.finding.code,))]
            pending_acceptance = acceptance is not None and is_pending(getattr(acceptance, "reason", None))
            words: list[str] = []
            if pending_acceptance:
                words.append("accepted by the experiment with its reason pending")
            if waived:
                words.append(f"OpenMaterica reads it unknown ({'; '.join(texts)}); the {self.profile.value} profile "
                             f"waives {', '.join(codes)} with a pending reason")
            kind = ("accepted and waived, its reason pending" if pending_acceptance and waived else
                    "accepted, its reason pending" if pending_acceptance else "waived, its reason pending")
            item = acceptance_item(material, field_name, [*covers, *(c for c in codes if c not in covers)],
                                   "; ".join(words), kind=kind,
                                   codes=tuple(sorted({*codes, *([BASIS_PENDING] if pending_acceptance else [])})))
            if pending_acceptance:
                self.pending("material.accept-pending", f"material {material!r}'s acceptance of {field_name}",
                             "its reason is pending", item)
            for decision in waived:
                self.record(decision, item)
            if listed:
                listed_codes = [d.finding.code for d in listed]
                held = acceptance_item(material, field_name, [*covers, *listed_codes],
                                       f"OpenMaterica reads it unknown ({'; '.join(texts)}); the {self.profile.value} "
                                       f"profile lets it through, listed", kind="unknown, listed",
                                       codes=tuple(sorted(listed_codes)))
                for decision in listed:
                    self.record(decision, held)

    def nominals(self, materials: Mapping[str, Any]) -> None:
        """Each material's nominal value with a pending reason, which behaves as a pending acceptance does."""
        for name, material in sorted(materials.items()):
            for field_name, nominal in sorted(material.nominal.items()):
                if is_pending(getattr(nominal, "reason", None)) and isinstance(nominal.value, Quantity):
                    self.pending("material.nominal-pending", f"material {name!r}'s nominal {field_name}",
                                 "its reason is pending", nominal_item(name, field_name, nominal.value))

    def finish(self, sites: Sequence[Site], given: Sequence[tuple[object, object]], run: Sequence[Site],
               roots: Sequence[tuple[str, object]] = (), declared: Sequence[object] = ()) -> ProfileOutcome:
        """Every setting judged: ``given`` the experiment's ``exp.source`` declarations (path, reason), ``run`` the
        run settings' sites, ``roots`` everything the experiment declares by name (for a reason no site judges) and
        ``declared`` the quantities ``exp.declare`` judges. Refusals go to :attr:`problems`."""
        everything = [*run, *sites]
        paths = {site.path: site for site in everything}
        stated: dict[str, Basis] = {}
        for written, reason in given:
            path = str(written) if isinstance(written, (TypedPath, str)) else repr(written)
            problem = basis_problem(reason)
            if path not in paths:
                self.refuse("source.unknown", RefusalClass.USER,
                            f"exp.source names {path!r}, which is no setting of this experiment",
                            f"name one of its settings: exp.source(\"<setting>\", reason=\"...\"). "
                            f"{did_you_mean(path, sorted(paths))}")
            elif path in stated:
                self.refuse("source.duplicate", RefusalClass.USER, f"exp.source gives {path} a source twice",
                            f"keep one of the two exp.source({path!r}, reason=...) lines")
            elif problem is not None:
                self.refuse("source.reason", RefusalClass.USER, f"exp.source({path!r}): {problem}",
                            f"write exp.source({path!r}, reason=\"where the value comes from\")")
            else:
                stated[path] = cast(Basis, reason)
        for site in everything:
            carried = _reason_of(site.value)
            if carried is not None and site.path in stated:
                self.refuse("source.duplicate", RefusalClass.USER,
                            f"{site.path} is given a source twice, on its quantity and by exp.source",
                            f"give it one: on the quantity, mm(..., reason=\"...\"), or by exp.source({site.path!r}, "
                            f"reason=\"...\"), not both")
                continue
            basis = carried if carried is not None else stated.get(site.path)
            if carried is not None and (problem := basis_problem(carried)) is not None:
                self.refuse("source.reason", RefusalClass.USER, f"{site.path}'s reason: {problem}",
                            "write its reason as a text, chosen(\"...\") or pending()")
                continue
            if basis is not None:
                self.sources[site.path] = basis
            self._judge_site(site, basis, from_quantity=carried is not None)
        judged = {held for site in everything for held in _judged(site.value)} | {id(value) for value in declared}
        for where in reasons_without_a_site(roots, judged):
            self.refuse("source.no-site", RefusalClass.USER,
                        f"{where} carries a reason, and nothing there is judged or saved: a reason belongs to a "
                        f"setting's value, a declared value, a condition's value, an acceptance or a nominal",
                        "take the reason off it and give it where it belongs: on the setting's own quantity or by "
                        "exp.source(path, reason=...), in degC(..., plus_minus=..., reason=...) for a condition, in "
                        "Nominal(..., reason=...) for a nominal value")
        return ProfileOutcome(
            self.profile, tuple(sorted(self.waivers, key=lambda w: (w.target, w.reasons, w.generated))),
            tuple(self.defaults[p] for p in sorted(self.defaults)),
            tuple(sorted(self.items, key=lambda item: (*item.key(), item.listed))),
            dict(sorted(self.sources.items())), dict(sorted(self.changes.items())))

    def _judge_site(self, site: Site, basis: Basis | None, *, from_quantity: bool) -> None:
        role = site.role
        taken = self.defaults.get(site.path)
        if taken is None and basis is None and site.value is not None and _same(site.value, site.written):
            if role is None:
                taken = DefaultTaken(site.path, DefaultKind.UNDECLARED, site.value, "the setting's written default",
                                     "its entry declares no role or default kind for it", site.entry)
                self.take_default(taken)
                self._undeclared_default(site, taken)
                return
            if role.default in (DefaultKind.CONVENTION, DefaultKind.NUMERICAL) and role.by is None:
                taken = DefaultTaken(site.path, role.default, site.value, role.rule, role.basis, site.entry,
                                     role.measured)
                self.take_default(taken)
        computed = role is not None and role.by is not None
        if taken is not None and taken.kind is DefaultKind.NUMERICAL and (basis is None or computed):
            # a computed default's value is in no document, so no reason given for it (a choice, a source, a pending
            # one) states or chooses it: a choice of a value nobody wrote is no choice. Write the value.
            self._numerical_default(site, taken)
            return
        if basis is not None and is_pending(basis):
            item = (setting_item(site, f"{site.path}'s reason is pending", reason_on_value=True) if from_quantity
                    else source_item(site.path, f"{site.path}'s source is pending"))
            self.pending("source.pending", site.path, "its reason is pending", item)
            return
        if role is None:
            return
        held = role.role
        if isinstance(basis, Chosen):
            if held is not Role.DESIGN:
                self.changes[site.path] = held.value
            return
        if basis is not None:
            return
        # a convention taken, its written value or absent (None: a segment at its frame's origin, no servo lag)
        convention = role.default is DefaultKind.CONVENTION and (
            taken is not None or site.value is None or _same(site.value, site.written))
        if convention and held is Role.PHYSICAL:
            what = f"a convention taken for a physical value ({role.basis}), with no source: does it hold here?"
            self._listed(site, SOURCE_MISSING, what, "a convention, no source",
                         ("setting.source-required", f"{site.path}: {what}", _source_line(site.path)))
            return
        if taken is not None or site.value is None:
            return
        if held is Role.PHYSICAL:
            what = "a physical value with no source: where does it come from?"
            self._listed(site, SOURCE_MISSING, what, "no source",
                         ("setting.source-required", f"{site.path}: {what}", _source_line(site.path)))
        elif held is Role.NUMERICAL:
            what = "a numerical value with no reason: its convergence evidence, until a1's evidence records"
            self._listed(site, EVIDENCE_MISSING, what, "no convergence evidence",
                         ("setting.evidence-required", f"{site.path}: {what}", _source_line(site.path)))

    def _numerical_default(self, site: Site, taken: DefaultTaken) -> None:
        shown = str(taken.value) if isinstance(taken.value, Quantity) else repr(taken.value)
        what = (f"{shown}, its numerical default ({taken.rule}; {taken.basis}); research needs its value stated with "
                f"its convergence evidence")
        item = default_item(site, taken, what)
        line = f"write it: {paste_command(item.python)}" + (f", and {paste_command(item.also)}" if item.also else "")
        self.apply(Finding(site.path, DEFAULT_NUMERICAL, what), item, Action.REFUSE,
                   ("setting.numerical-default", f"{site.path} is not given, and its default is numerical "
                                                 f"({taken.rule}): its value needs stating, with its convergence "
                                                 f"evidence as its reason", line))

    def _undeclared_default(self, site: Site, taken: DefaultTaken) -> None:
        shown = str(taken.value) if isinstance(taken.value, Quantity) else repr(taken.value)
        what = (f"{shown}, the setting's written default, and its entry ({site.entry}) declares no role or default "
                f"kind for it, so nothing says what the default claims")
        item = source_item(site.path, what, kind="an undeclared default", codes=(DEFAULT_UNDECLARED,))
        self.apply(Finding(site.path, DEFAULT_UNDECLARED, what), item, None,
                   ("setting.undeclared-default", f"{site.path} is not given, and its entry ({site.entry}) declares no "
                                                  f"role or default kind for it: its written default {shown} was "
                                                  f"taken", f"declare the value with its source: "
                                                                f"exp.source({_quote(site.path)}, reason=\"...\") (or "
                                                                f"write it where the setting is written, with its "
                                                                f"reason), or declare the setting's role and default "
                                                                f"kind in the entry's setting_roles"))

    def _listed(self, site: Site, code: str, what: str, kind: str, refusal: Problem) -> None:
        item = source_item(site.path, what, kind=kind, codes=(code,))
        self.apply(Finding(site.path, code, what), item, None, refusal)


def paste_command(line: str) -> str:
    """An item's line to paste, its reason a placeholder to fill: code, which the actions' lint reads as such."""
    return line.replace(REASON, '"..."')


def _source_line(path: str) -> str:
    return f"give it its source: exp.source({_quote(path)}, reason=\"...\")"


# -- the items: what is pending, with the lines to paste ------------------------------------------------------


def acceptance_item(material: str, field_name: str, covers: Sequence[str], what: str,
                    kind: str = "waived, its reason pending", codes: tuple[str, ...] = ()) -> PendingItem:
    """An acceptance of a read, with ``covers`` and its reason's slot."""
    held = ", ".join(_quote(code) for code in covers)
    return PendingItem(read_target(material, field_name), what, f"in material {material!r}'s Record(...)",
                       f"accept={{{_quote(field_name)}: Acceptance(covers=({held},), reason={REASON})}}",
                       {"materials": {material: {"accept": {field_name: {"covers": list(covers), "reason": REASON}}}}},
                       kind=kind, codes=codes, imports=(_ACCEPTANCE,))


def nominal_item(material: str, field_name: str, value: Quantity) -> PendingItem:
    """A nominal value of a read, with its reason's slot."""
    return PendingItem(read_target(material, field_name), "a nominal value declared with its reason pending",
                       f"in material {material!r}'s Record(...)",
                       f"nominal={{{_quote(field_name)}: Nominal({quantity_line(value, None)}, reason={REASON})}}",
                       {"materials": {material: {"nominal": {field_name: {
                           "value": float(cast(float, value.value)), "unit": unit_symbol(value.dimension),
                           "reason": REASON}}}}},
                       kind="its nominal's reason pending", codes=(BASIS_PENDING,),
                       imports=(_NOMINAL, *quantity_imports(value)))


def declared_item(path: str, value: Quantity, what: str) -> PendingItem:
    """A value declared in the experiment, in place of its ``exp.declare(...)``."""
    return PendingItem(path, what, "in place of its exp.declare(...)",
                       f"exp.declare({_quote(path)}, {quantity_line(value, None)}, reason={REASON})",
                       {"declared": {path: quantity_doc(value)}}, kind="its reason pending", codes=(BASIS_PENDING,),
                       imports=quantity_imports(value))


def source_item(path: str, what: str, *, kind: str = "its source pending",
                codes: tuple[str, ...] = (BASIS_PENDING,)) -> PendingItem:
    """A setting's source by ``exp.source``."""
    return PendingItem(path, what, "anywhere after the experiment is made",
                       f"exp.source({_quote(path)}, reason={REASON})", {"sources": {path: REASON}}, kind=kind,
                       codes=codes)


def setting_item(site: Site, what: str, *, reason_on_value: bool) -> PendingItem:
    """A setting written with its reason on its value, where the value is a quantity; by ``exp.source`` beside it
    otherwise."""
    value = site.value
    if reason_on_value and isinstance(value, Quantity) and site.document is not None:
        return PendingItem(site.path, what, site.where, f"{site.setting}={quantity_line(value)}",
                           site.document({site.setting: quantity_doc(value)}), kind="its reason pending",
                           codes=(BASIS_PENDING,), imports=quantity_imports(value))
    return source_item(site.path, what)


def default_item(site: Site, taken: DefaultTaken, what: str) -> PendingItem:
    """A numerical default taken: its value written in the setting with its reason, where the setting is a quantity
    computed by a rule; ``exp.source`` alone for one whose written default is the value (the document does not write a
    setting equal to its default), with the value beside it for one computed that is not a quantity."""
    value = taken.value
    kind = "a numerical default"
    if site.document is None or site.role is None or site.role.by is None:
        return replace(source_item(site.path, what, kind=kind, codes=(DEFAULT_NUMERICAL,)), measured=taken.measured)
    if isinstance(value, Quantity):
        return PendingItem(site.path, what, site.where, f"{site.setting}={quantity_line(value)}",
                           site.document({site.setting: quantity_doc(value)}), kind=kind, codes=(DEFAULT_NUMERICAL,),
                           imports=quantity_imports(value), measured=taken.measured)
    fragment = cast(dict[str, Json], site.document({site.setting: cast(Json, value)}))
    return PendingItem(site.path, what, site.where, f"{site.setting}={value!r}",
                       {**fragment, "sources": {site.path: REASON}}, kind=kind,
                       also=f"exp.source({_quote(site.path)}, reason={REASON})", codes=(DEFAULT_NUMERICAL,),
                       measured=taken.measured)


#: Gravity's role when the experiment states none: standard gravity, a convention.
GRAVITY_CONVENTION = SettingRole(Role.PHYSICAL, DefaultKind.CONVENTION, "standard gravity, along -up",
                                 "the experiment states no gravity")


def run_sites(settings: Any, conventional_gravity: object) -> list[Site]:
    """The run settings as sites: ``dt`` (numerical, required), ``gravity`` (physical, standard by convention when not
    given), ``seed`` (design, zero by convention) and each condition (physical)."""
    conventional = settings.gravity is conventional_gravity
    found = [Site("dt", SettingRole(Role.NUMERICAL), settings.dt, dataclasses.MISSING, "in Experiment(...)", "dt",
                  lambda held: {"run": held}),
             Site("seed", SettingRole(Role.DESIGN), settings.seed, dataclasses.MISSING, "in Experiment(...)", "seed",
                  None),
             Site("gravity", GRAVITY_CONVENTION if conventional else SettingRole(Role.PHYSICAL), settings.gravity,
                  dataclasses.MISSING, "in Experiment(...)", "gravity", lambda held: cast(Json, held))]
    conditions = cast(object, settings.conditions)
    if isinstance(conditions, Mapping):
        for name, condition in sorted(cast(Mapping[object, object], conditions).items(), key=str):
            found.append(Site(f"conditions.{name}", SettingRole(Role.PHYSICAL), condition, dataclasses.MISSING,
                              "in Experiment(conditions=...)", str(name), None))
    return found


def reader_sites(declarations: Sequence[object], entry_id: Callable[[Any, Mapping[str, Any]], str]
                 ) -> list[tuple[str, str, object]]:
    """Each interaction and observation declared, with its section and its id."""
    from mime.build._declarations import Attachment, ObservationDeclaration

    found: list[tuple[str, str, object]] = []
    for declaration in declarations:
        if isinstance(declaration, Attachment) and isinstance(cast(object, declaration.law), Interaction):
            found.append(("interaction", entry_id(type(declaration.law), declaration.bindings), declaration.law))
        elif isinstance(declaration, ObservationDeclaration) and isinstance(cast(object, declaration.observation),
                                                                          Observation):
            observation = declaration.observation
            found.append(("observation", entry_id(type(observation), observation.bindings()), observation))
    return found


__all__ = ["BASIS_PENDING", "DEFAULT_NUMERICAL", "DEFAULT_UNDECLARED", "EVIDENCE_MISSING", "GRAVITY_CONVENTION",
           "PENDING", "SOURCE_MISSING", "DefaultTaken", "ProfileOutcome", "ProfileWork", "Site", "acceptance_item",
           "declared_item", "default_item", "experiment_sites", "nominal_item", "quantity_doc", "quantity_imports",
           "quantity_line", "reader_sites", "reasons_without_a_site", "run_sites", "setting_item", "source_item"]
