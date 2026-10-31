"""The experiment's profile at build time: the defaults taken, the sources of its settings, what its profile's rules
made of each finding, and what is left pending, with the lines to paste.

Every setting of the experiment has a path (``tube.inner_radius``, ``umr.parts.shell.radius``,
``field.magnetic.rotation_axis``, ``field.frequency`` for a command's value, ``drag:umr.spacing``, ``controller.gain``,
``dt``, ``gravity``, ``conditions.temperature``), its declared role and its source: the reason its quantity carries
(``mm(3.175, reason=...)``) or one ``exp.source(path, reason=...)`` gives. Each role is a finding class in the profile
rules (design #191 item 3): a physical value with no source, a numerical value with none (its convergence evidence
until a1's evidence records), a numerical default taken, and a pending reason anywhere. The profile acts on each by its
class alone, at build time: a profile changes what is refused, waived or listed, never a value or a graph."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
    Finding,
    Judge,
    PendingItem,
    Profile,
    Stamp,
    Waiver,
    basis_problem,
    basis_text,
    is_pending,
    pending_text,
)

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type Problem = tuple[str, str, str]

#: The finding codes the build gives (finding_classes.txt): a reason left pending, a numerical default taken, a
#: physical value with no source, a numerical value with no evidence.
BASIS_PENDING = "basis_pending"
DEFAULT_NUMERICAL = "default_numerical"
SOURCE_MISSING = "source_missing"
EVIDENCE_MISSING = "evidence_missing"

#: The SI unit each dimension's quantity is written in on a line to paste, by its symbol: a constructor of mime.units.
_CONSTRUCTORS = {"m": "m", "s": "s", "kg": "kg", "Hz": "Hz", "N": "N", "A": "A", "K": "K", "T": "T", "m/s": "m_per_s",
                 "m/s^2": "m_per_s2", "1": "dimensionless", "m^3": "m3", "kg/m^3": "kg_per_m3", "kg m^2": "kg_m2",
                 "A m^2": "A_m2", "Pa s": "Pa_s", "N m": "N_m", "N m s": "N_m_s"}


def _number(value: object) -> str:
    if isinstance(value, tuple):
        return "(" + ", ".join(repr(float(c)) for c in cast(tuple[float, ...], value)) + ")"
    return repr(value)


def quantity_line(value: Quantity, reason: str | None = REASON) -> str:
    """``value`` as a script writes it, in SI, with ``reason`` in its slot (``kg_per_m3(1410.0, reason=...)``), or
    with none."""
    symbol = unit_symbol(value.dimension)
    unit = "rad_per_s" if symbol == "Hz" and value.angle == "radian" else _CONSTRUCTORS.get(symbol)
    tail = f", reason={reason}" if reason is not None else ""
    if unit is None:
        return f"Quantity({_number(value.value)}, parse_unit({symbol!r}){tail})"
    return f"{unit}({_number(value.value)}{tail})"


def _quote(text: str) -> str:
    return json.dumps(text)


def quantity_doc(value: Quantity, reason: Json = REASON) -> Json:
    magnitude = value.value
    held: Json = [float(c) for c in magnitude] if isinstance(magnitude, tuple) else float(cast(float, magnitude))
    return {"value": held, "unit": unit_symbol(value.dimension), "reason": reason}


@dataclass(frozen=True)
class DefaultTaken:
    """A default the build took for a setting the experiment did not give: its path, its kind, the value, and its rule
    and basis in words."""

    path: str
    kind: DefaultKind
    value: object
    rule: str
    basis: str

    def text(self) -> str:
        shown = str(self.value) if isinstance(self.value, Quantity) else repr(self.value)
        return f"{self.path} = {shown}, a {self.kind.value} default ({self.rule}; {self.basis})"


@dataclass(frozen=True)
class Site:
    """One setting of the experiment: its path, its declared role (``None``: it declares none), the value as written,
    its written default, where the value is written (for the line to paste), its name there, and how a document
    fragment reaches it (a function of its settings)."""

    path: str
    role: SettingRole | None
    value: object
    written: object
    where: str
    setting: str
    document: Callable[[dict[str, Json]], Json] | None = None


@dataclass(frozen=True)
class ProfileOutcome:
    """What the profile made of the experiment: the profile, every waiver the build rests on with a pending basis
    (generated, or the experiment's own), the defaults taken, the items left pending and those listed, every setting's
    source by path, and each value given as the experimenter's own choice (a declared change of its role)."""

    profile: Profile
    waivers: tuple[Waiver, ...] = ()
    defaults: tuple[DefaultTaken, ...] = ()
    items: tuple[PendingItem, ...] = ()
    sources: Mapping[str, Basis] = field(default_factory=dict[str, Basis])
    changes: tuple[str, ...] = ()

    @property
    def pending(self) -> tuple[PendingItem, ...]:
        return tuple(item for item in self.items if not item.listed)

    @property
    def listed(self) -> tuple[PendingItem, ...]:
        return tuple(item for item in self.items if item.listed)

    def stamp(self) -> Stamp:
        return Stamp(self.profile.value, tuple(item.line() for item in self.pending))

    def text(self) -> str:
        """Each item in plain words with its lines to paste (``built.report.pending()``)."""
        return pending_text(self.profile.value, self.items)

    def report_lines(self) -> list[str]:
        lines = [self.stamp().text()]
        lines += [f"  pending: {item.line()}" for item in self.pending]
        lines += [f"  listed: {item.line()}" for item in self.listed]
        lines += [f"  default: {taken.text()}" for taken in self.defaults]
        lines += [f"  declared change: {change}" for change in self.changes]
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
    for name in setting_names(record.cls):
        value = getattr(entry, name, None)
        path = prefix if name == value_path else f"{prefix}.{name}"
        role = record.roles.get(name)
        written = _written(record.cls, name)
        found.append(Site(path, role, value, written, where, name, document))
        if isinstance(value, SettingGroup):
            group_roles = declared_roles(type(value))
            for inner in setting_names(type(value)):
                found.append(Site(f"{path}.{inner}", group_roles.get(inner, role), getattr(value, inner, None),
                                  _written(type(value), inner), f"{where}'s {name}=", inner, None))
    return found


def _written(cls: type, name: str) -> object:
    held = next((f for f in dataclasses.fields(cls) if f.name == name), None)
    if held is None:
        return dataclasses.MISSING
    if held.default is not dataclasses.MISSING:
        return cast(object, held.default)
    if held.default_factory is not dataclasses.MISSING:
        return cast(object, held.default_factory())
    return dataclasses.MISSING


def experiment_sites(things: Sequence[object], readers: Sequence[tuple[str, str, object]],
                     controller: object | None) -> list[Site]:
    """Every setting of the experiment's things, interactions, observations and controller (``readers``: each
    interaction and observation with its section and id), by path."""
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
            for key, part in sorted(thing.parts.items()):
                found += _entry_sites(part.geometry, f"{thing.name}.parts.{key}", f"in body {thing.name!r}'s part "
                                      f"{key!r}", _fragment("things", thing.name, "parts", key, "solid"))
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
            for key, part in sorted(thing.parts.items()):
                found += _entry_sites(part.geometry, f"{thing.name}.parts.{key}", f"in apparatus {thing.name!r}'s part "
                                      f"{key!r}", _fragment("things", thing.name, "parts", key, "solid"))
    for section, owner, entry in readers:
        found += _entry_sites(entry, owner, f"in {section} {owner}'s {type(entry).__name__}(...)",
                              _listed_fragment(f"{section}s", "id", owner))
    if controller is not None:
        found += _entry_sites(controller, "controller", f"in the controller's {type(controller).__name__}(...)",
                              _fragment("controller"))
    return found


def _reason_of(value: object) -> Basis | None:
    if isinstance(value, Quantity):
        return value.reason
    if isinstance(value, Condition):
        return value.value.reason
    return None


def _same(value: object, written: object) -> bool:
    """Whether ``value`` is its setting's written default: of one type and equal, a tuple's subclass (a signed axis)
    counting as a tuple, as the document reads it."""
    if written is dataclasses.MISSING:
        return False
    kinds: tuple[type, type] = (type(value), type(written))
    tuples = all(issubclass(kind, tuple) for kind in kinds)
    return (tuples or kinds[0] is kinds[1]) and value == written


class ProfileWork:
    """The profile's work while an experiment is planned: the judge, the defaults taken, the waivers and items, and
    what to refuse. The planner feeds it as it goes, and :meth:`finish` judges every setting."""

    def __init__(self, profile: Profile) -> None:
        self.judge = Judge(profile)
        self.defaults: dict[str, DefaultTaken] = {}
        self.waivers: list[Waiver] = []
        self.items: list[PendingItem] = []
        self.problems: list[tuple[str, RefusalClass, str, str]] = []
        self.products: list[tuple[str, object]] = []
        self.sources: dict[str, Basis] = {}
        self.changes: list[str] = []

    @property
    def profile(self) -> Profile:
        return self.judge.profile

    def refuse(self, code: str, category: RefusalClass, message: str, action: str) -> None:
        self.problems.append((code, category, message, action))

    def take_default(self, taken: DefaultTaken) -> None:
        self.defaults[taken.path] = taken

    def pending(self, code: str, target: str, what: str, item: PendingItem) -> bool:
        """A reason left pending (``target``), judged by the profile's rule for its class: recorded with its item and
        stamped where the rule generates, refused by name (``code``) where it refuses. Whether the build goes on with
        it."""
        rule = self.judge.rule(BASIS_PENDING)
        if rule is not None and rule.action is Action.REFUSE:
            self.refuse(code, RefusalClass.USER, f"{target}: {what}, and the {self.profile.value} profile refuses a pending reason: a "
                              f"placeholder is not a reason",
                        f"give it its reason: {item.python.replace(REASON, '\"...\"')}")
            return False
        self.waivers.append(Waiver(target, (BASIS_PENDING,), PENDING))
        self.items.append(item)
        return True

    def generated(self, product: object, item: PendingItem, where: str) -> None:
        """A product of a ``generate`` rule: a waiver, recorded with its item; anything else is a defect of the rule,
        refused, since profiles act on waivers alone."""
        self.products.append((where, product))
        if isinstance(product, Waiver):
            self.waivers.append(product)
            self.items.append(item)
        else:
            self.refuse("profile.not-a-waiver", RefusalClass.INTERNAL, f"the {self.profile.value} profile's rule for {where} generated a "
                                                f"{type(product).__name__}, and profiles act on waivers alone",
                        "report it: a profile rule generates or accepts a waiver and nothing else")

    # -- the experiment's reads, declarations and settings ------------------------------------------------------

    def read_products(self, material: str, field_name: str, products: Sequence[object], covers: Sequence[str],
                      texts: Sequence[str]) -> None:
        """The waivers the profile generated for a read's reasons (``covers`` the read's own acceptance already
        covers, to which a pasted acceptance adds them)."""
        codes = [code for made in products for code in getattr(made, "reasons", ())]
        item = acceptance_item(material, field_name, [*covers, *codes],
                               f"OpenMaterica reads it unknown ({'; '.join(texts)}); the {self.profile.value} profile "
                               f"waives {', '.join(codes) or 'its reasons'} with a pending reason")
        for made in products:
            self.generated(made, item, read_target(material, field_name))

    def finish(self, sites: Sequence[Site], given: Sequence[tuple[object, object]], run: Sequence[Site],
               ) -> ProfileOutcome:
        """Every setting judged: ``given`` the experiment's ``exp.source`` declarations (path, reason), ``run`` the
        run settings' sites. Refusals go to :attr:`problems`."""
        everything = [*run, *sites]
        paths = {site.path: site for site in everything}
        stated: dict[str, Basis] = {}
        for written, reason in given:
            path = str(written) if isinstance(written, (TypedPath, str)) else repr(written)
            problem = basis_problem(reason)
            if path not in paths:
                self.refuse("source.unknown", RefusalClass.USER, f"exp.source names {path!r}, which is no setting of this experiment",
                            did_you_mean(path, sorted(paths)))
            elif path in stated:
                self.refuse("source.duplicate", RefusalClass.USER, f"exp.source gives {path} a source twice",
                            "give each setting one source")
            elif problem is not None:
                self.refuse("source.reason", RefusalClass.USER, f"exp.source({path!r}): {problem}",
                            f"write exp.source({path!r}, reason=\"where the value comes from\")")
            else:
                stated[path] = cast(Basis, reason)
        for site in everything:
            carried = _reason_of(site.value)
            if carried is not None and site.path in stated:
                self.refuse("source.duplicate", RefusalClass.USER, f"{site.path} is given a source twice, on its quantity and by "
                                                f"exp.source", "give it one: on the quantity, or by exp.source")
                continue
            basis = carried if carried is not None else stated.get(site.path)
            if carried is not None and (problem := basis_problem(carried)) is not None:
                self.refuse("source.reason", RefusalClass.USER, f"{site.path}'s reason: {problem}",
                            f"write its reason as a text, chosen(\"...\") or pending()")
                continue
            if basis is not None:
                self.sources[site.path] = basis
            self._judge_site(site, basis, from_quantity=carried is not None)
        return ProfileOutcome(self.profile, tuple(self.waivers), tuple(self.defaults[p] for p in sorted(self.defaults)),
                              tuple(self.items), dict(sorted(self.sources.items())), tuple(self.changes))

    def _judge_site(self, site: Site, basis: Basis | None, *, from_quantity: bool) -> None:
        if basis is not None and is_pending(basis):
            item = (setting_item(site, f"{site.path}'s reason is pending", reason_on_value=True) if from_quantity
                    else source_item(site.path, f"{site.path}'s source is pending"))
            self.pending("source.pending", site.path, "its reason is pending", item)
            return
        role = site.role
        if role is None:
            return
        held = role.role
        if isinstance(basis, Chosen):
            if held is not Role.DESIGN:
                self.changes.append(f"{site.path}: declared {held.value}, given here as the experimenter's choice "
                                    f"({basis_text(basis)})")
            return
        taken = self.defaults.get(site.path)
        if taken is None and basis is None and site.value is not None and role.default in (
                DefaultKind.CONVENTION, DefaultKind.NUMERICAL) and role.by is None and _same(site.value, site.written):
            taken = DefaultTaken(site.path, role.default, site.value, role.rule, role.basis)
            self.take_default(taken)
        if taken is not None and taken.kind is DefaultKind.NUMERICAL and (basis is None or role.by is not None):
            # a computed default's value is in no document, so a source given for it states nothing: write the value
            self._numerical_default(site, taken)
        elif taken is None and basis is None and held is Role.PHYSICAL and site.value is not None:
            self._listed(site, SOURCE_MISSING, "a physical value with no source: where does it come from?",
                         "no source")
        elif taken is None and basis is None and held is Role.NUMERICAL and site.value is not None:
            self._listed(site, EVIDENCE_MISSING, "a numerical value with no reason: its convergence evidence, until "
                                                 "a1's evidence records", "no convergence evidence")

    def _numerical_default(self, site: Site, taken: DefaultTaken) -> None:
        shown = str(taken.value) if isinstance(taken.value, Quantity) else repr(taken.value)
        what = (f"{shown}, its numerical default ({taken.rule}; {taken.basis}); research needs its value stated with "
                f"its convergence evidence")
        rule = self.judge.rule(DEFAULT_NUMERICAL)
        item = default_item(site, taken, what)
        if rule is None:
            return
        if rule.action is Action.REFUSE:
            self.refuse("setting.numerical-default", RefusalClass.USER,
                        f"{site.path} is not given, and its default is numerical ({taken.rule}): the "
                        f"{self.profile.value} profile needs the value stated, with its convergence evidence as its "
                        f"reason", f"write it: {item.python.replace(REASON, '\"...\"')}"
                                   + (f", and {item.also.replace(REASON, '\"...\"')}" if item.also else ""))
        elif rule.action is Action.GENERATE:
            self.generated(rule.make(Finding(site.path, DEFAULT_NUMERICAL, what)), item, site.path)

    def _listed(self, site: Site, code: str, what: str, kind: str) -> None:
        rule = self.judge.rule(code)
        if rule is not None and rule.action is Action.LIST:
            self.items.append(source_item(site.path, what, listed=True, kind=kind))


# -- the items: what is pending, with the lines to paste ------------------------------------------------------


def acceptance_item(material: str, field_name: str, covers: Sequence[str], what: str,
                    kind: str = "waived, its reason pending") -> PendingItem:
    """An acceptance of a read, with ``covers`` and its reason's slot."""
    held = ", ".join(_quote(code) for code in covers)
    return PendingItem(read_target(material, field_name), what, f"in material {material!r}'s Record(...)",
                       f"accept={{{_quote(field_name)}: Acceptance(covers=({held},), reason={REASON})}}",
                       {"materials": {material: {"accept": {field_name: {"covers": list(covers), "reason": REASON}}}}},
                       kind=kind)


def declared_item(path: str, value: Quantity, what: str) -> PendingItem:
    """A value declared in the experiment, in place of its ``exp.declare(...)``."""
    return PendingItem(path, what, "in place of its exp.declare(...)",
                       f"exp.declare({_quote(path)}, {quantity_line(value, None)}, reason={REASON})",
                       {"declared": {path: quantity_doc(value)}}, kind="its reason pending")


def source_item(path: str, what: str, *, listed: bool = False, kind: str = "its source pending") -> PendingItem:
    """A setting's source by ``exp.source``."""
    return PendingItem(path, what, "anywhere after the experiment is made",
                       f"exp.source({_quote(path)}, reason={REASON})", {"sources": {path: REASON}}, listed, kind)


def setting_item(site: Site, what: str, *, reason_on_value: bool) -> PendingItem:
    """A setting written with its reason on its value, where the value is a quantity; by ``exp.source`` beside it
    otherwise."""
    value = site.value
    if reason_on_value and isinstance(value, Quantity) and site.document is not None:
        return PendingItem(site.path, what, site.where, f"{site.setting}={quantity_line(value)}",
                           site.document({site.setting: quantity_doc(value)}), kind="its reason pending")
    return source_item(site.path, what)


def default_item(site: Site, taken: DefaultTaken, what: str) -> PendingItem:
    """A numerical default taken: its value written in the setting with its reason, where the setting is a quantity
    computed by a rule; ``exp.source`` alone for one whose written default is the value (the document does not write a
    setting equal to its default), with the value beside it for one computed that is not a quantity."""
    value = taken.value
    kind = "a numerical default"
    if site.document is None or site.role is None or site.role.by is None:
        return source_item(site.path, what, kind=kind)
    if isinstance(value, Quantity):
        return PendingItem(site.path, what, site.where, f"{site.setting}={quantity_line(value)}",
                           site.document({site.setting: quantity_doc(value)}), kind=kind)
    fragment = cast(dict[str, Json], site.document({site.setting: cast(Json, value)}))
    return PendingItem(site.path, what, site.where, f"{site.setting}={value!r}",
                       {**fragment, "sources": {site.path: REASON}}, kind=kind,
                       also=f"exp.source({_quote(site.path)}, reason={REASON})")


def run_sites(settings: Any, conventional_gravity: object) -> list[Site]:
    """The run settings as sites: ``dt`` (numerical, required), ``gravity`` (physical, standard by convention),
    ``seed`` (design, zero by convention) and each condition (physical)."""
    found = [Site("dt", SettingRole(Role.NUMERICAL), settings.dt, dataclasses.MISSING, "in Experiment(...)", "dt",
                  lambda held: {"run": held}),
             Site("seed", SettingRole(Role.DESIGN), settings.seed, dataclasses.MISSING, "in Experiment(...)", "seed",
                  None)]
    if settings.gravity is not conventional_gravity:
        found.append(Site("gravity", SettingRole(Role.PHYSICAL), settings.gravity, dataclasses.MISSING,
                          "in Experiment(...)", "gravity", lambda held: cast(Json, held)))
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


__all__ = ["BASIS_PENDING", "DEFAULT_NUMERICAL", "EVIDENCE_MISSING", "SOURCE_MISSING", "DefaultTaken",
           "ProfileOutcome", "ProfileWork", "Site", "acceptance_item", "declared_item", "default_item",
           "experiment_sites", "quantity_doc", "quantity_line", "reader_sites", "run_sites", "setting_item",
           "source_item"]
