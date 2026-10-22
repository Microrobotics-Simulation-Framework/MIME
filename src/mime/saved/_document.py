"""The authored document: what an experiment declares, as data, written as YAML and read back with PyYAML's safe
loader. Its hash is taken over canonical JSON, so formatting never changes it."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import yaml

from mime.build import Attachment, Declaration, RunSettings, ThingDeclaration, binding_name
from mime.build._derived import sha256_of
from mime.registry import Dynamics, Entry, FieldProvider, Interaction, Observation, registration_of
from mime.saved._schema import (
    DOCUMENT_SCHEMA,
    BoundEntryDoc,
    DynamicsDoc,
    EntryDoc,
    ExperimentDocument,
    QuantityDoc,
    SettingValue,
    ThingDoc,
)
from mime.things import Apparatus, Body, Medium, Structure, Thing, ThingKind
from mime.units import Quantity, parse_unit, unit_symbol


class UnsavableError(ValueError):
    """A declaration the saved form cannot hold, named in the experiment's terms."""


def encode_setting(value: object, where: str) -> SettingValue:
    """A setting as the document writes it: a number, string, boolean or null; a list of settings (a tuple saves as a
    list and loads as a tuple); a quantity as its SI value and unit. Anything else, code included, cannot be saved."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsavableError(f"{where} is {value!r}; the document holds finite numbers only")
        return value
    if isinstance(value, Quantity):
        magnitude = value.value
        quantity: QuantityDoc = {"value": list(magnitude) if isinstance(magnitude, tuple) else magnitude,
                                 "unit": unit_symbol(value.dimension)}
        return quantity
    if isinstance(value, (list, tuple)):
        return [encode_setting(item, f"{where}[{i}]") for i, item in enumerate(cast(Sequence[object], value))]
    raise UnsavableError(f"{where} is a {type(value).__name__}; a document holds numbers, strings, booleans, null, "
                         f"lists and quantities, and never code")


def decode_setting(value: SettingValue) -> object:
    """The value :func:`encode_setting` wrote: lists come back as tuples, quantities as :class:`Quantity`."""
    if isinstance(value, list):
        return tuple(decode_setting(item) for item in value)
    if isinstance(value, dict):
        magnitude = value["value"]
        return Quantity(tuple(float(v) for v in magnitude) if isinstance(magnitude, list) else float(magnitude),
                        parse_unit(value["unit"]))
    return value


def _binding_fields(entry: Entry) -> frozenset[str]:
    return frozenset(entry.roles) if isinstance(entry, Observation) else frozenset()


def _default(field: dataclasses.Field[Any]) -> object:
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING


def _noted(value: object, where: str) -> SettingValue:
    try:
        return encode_setting(value, where)
    except UnsavableError:
        return f"(cannot be saved: a {type(value).__name__})"


def all_settings(entry: Entry, where: str) -> dict[str, SettingValue]:
    """Every setting of ``entry``, its defaults included: the resolved record's view. A setting the document cannot
    hold is noted as such here; the document refuses it, and so does ``save``."""
    skip = _binding_fields(entry)
    return {f.name: _noted(getattr(entry, f.name), f"{where}, setting {f.name!r}")
            for f in dataclasses.fields(entry) if f.name not in skip}


def written_settings(entry: Entry, where: str) -> dict[str, SettingValue]:
    """The settings of ``entry`` that differ from their defaults: the document's view. A default is written into the
    resolved record instead, so a later version that changes it shows there."""
    skip = _binding_fields(entry)
    written: dict[str, SettingValue] = {}
    for f in dataclasses.fields(entry):
        value = getattr(entry, f.name)
        default = _default(f)
        if f.name in skip or (default is not dataclasses.MISSING and type(default) is type(value)
                              and default == value):
            continue
        written[f.name] = encode_setting(value, f"{where}, setting {f.name!r}")
    return written


def _entry_doc(entry: Entry, where: str) -> EntryDoc:
    record = registration_of(entry)
    if record is None:
        raise UnsavableError(f"{where}: {type(entry).__qualname__} is not a registered entry")
    doc: EntryDoc = {"name": record.name, "version": record.version}
    settings = written_settings(entry, where)
    if settings:
        doc["settings"] = settings
    return doc


def _bound_doc(entry: Interaction | Observation, entry_id: str, bindings: Mapping[str, object],
               where: str) -> BoundEntryDoc:
    plain = _entry_doc(entry, where)
    doc: BoundEntryDoc = {"id": entry_id, "name": plain["name"], "version": plain["version"],
                          "binds": {role: binding_name(value) for role, value in sorted(bindings.items())}}
    if "settings" in plain:
        doc["settings"] = plain["settings"]
    return doc


def _thing_doc(thing: Thing) -> ThingDoc:
    where = f"{thing.kind.value} {thing.name!r}"
    doc: ThingDoc = {"kind": thing.kind.value}
    if isinstance(thing, Medium) and thing.flow is not None:
        doc["dynamics"] = DynamicsDoc(flow=_entry_doc(thing.flow, f"{where}'s flow"))
    elif isinstance(thing, (Body, Structure)) and thing.dynamics is not None:
        doc["dynamics"] = DynamicsDoc(motion=_entry_doc(thing.dynamics, f"{where}'s motion"))
    if isinstance(thing, Body):
        doc["initial"] = {key: encode_setting(value, f"{where}'s initial {key}")
                          for key, value in thing.initial.items()}
    if isinstance(thing, Apparatus):
        doc["fields"] = {name: _entry_doc(provider, f"{where}'s field {name!r}")
                         for name, provider in sorted(thing.fields.items())}
    return doc


def document_of(settings: RunSettings, declarations: Sequence[Declaration]) -> ExperimentDocument:
    """The document of a declared experiment. Things are keyed by name and interactions and observations sorted by
    id, so the order a script declares them in changes nothing, as it changes nothing in the build."""
    from mime import __version__
    from mime.build import entry_id

    things: dict[str, ThingDoc] = {}
    interactions: list[BoundEntryDoc] = []
    observations: list[BoundEntryDoc] = []
    for declaration in declarations:
        if isinstance(declaration, ThingDeclaration):
            things[declaration.thing.name] = _thing_doc(declaration.thing)
        elif isinstance(declaration, Attachment):
            law = declaration.law
            owner = entry_id(type(law), declaration.bindings)
            interactions.append(_bound_doc(law, owner, declaration.bindings, f"interaction {owner}"))
        else:
            observation = declaration.observation
            owner = entry_id(type(observation), observation.bindings())
            observations.append(_bound_doc(observation, owner, observation.bindings(), f"observation {owner}"))
    try:
        up = [float(c) for c in cast(Sequence[Any], settings.up)]
    except (TypeError, ValueError) as error:
        raise UnsavableError(f"up={settings.up!r} is not three numbers") from error
    dt, gravity = encode_setting(settings.dt, "dt"), encode_setting(settings.gravity, "gravity")
    if not (isinstance(dt, dict) and isinstance(gravity, dict)):
        raise UnsavableError("dt and gravity are quantities, such as ms(1) and standard_gravity()")
    return {
        "schema": DOCUMENT_SCHEMA,
        "mime": __version__,
        "experiment": settings.experiment,
        "run": {"dt": dt, "seed": settings.seed},
        "frame": {"up": up, "gravity": gravity},
        "things": dict(sorted(things.items())),
        "interactions": sorted(interactions, key=lambda d: d["id"]),
        "observations": sorted(observations, key=lambda d: d["id"]),
        "materials": {},
        "scene": None,
        "controller": None,
    }


def document_hash(document: ExperimentDocument) -> str:
    return sha256_of(document)


class _Dumper(yaml.SafeDumper):
    """Lists of plain values in flow style (``[0.0, 0.0, 1.0]``), mappings in block style."""


def _represent_list(dumper: yaml.SafeDumper, data: list[object]) -> yaml.Node:
    flow = all(not isinstance(item, (list, dict)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


def dump_document(document: ExperimentDocument) -> str:
    return yaml.dump(document, Dumper=_Dumper, sort_keys=False, default_flow_style=False, allow_unicode=True,
                     width=120)


class _Loader(yaml.SafeLoader):
    """PyYAML's safe loader, which constructs no Python object a tag names, refusing a mapping that repeats a key
    (the safe loader keeps the last one without a word)."""


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[object, object]:
    keys: set[object] = set()
    for key_node, _ in cast(list[tuple[yaml.Node, yaml.Node]], node.value):
        key = cast(object, cast(Any, loader).construct_object(key_node, deep=True))
        if key in keys:
            raise yaml.constructor.ConstructorError(None, None, f"the key {key!r} appears twice",
                                                    key_node.start_mark)
        keys.add(key)
    return cast(dict[object, object], loader.construct_mapping(node, deep=True))


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def parse_yaml(text: str) -> object:
    """The data in ``text``, built with the safe loader: no tag can make it construct a Python object or run code.
    Raises ``yaml.YAMLError``."""
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - _Loader is a SafeLoader


def thing_from(name: str, doc: ThingDoc, entries: Mapping[tuple[str, str], Entry]) -> Thing:
    """A thing from its document entry, with its entries already constructed and keyed by ``(name, where)``."""
    kind = ThingKind(doc["kind"])
    if kind is ThingKind.MEDIUM:
        return Medium(name, flow=cast(Dynamics | None, entries.get((name, "flow"))))
    if kind is ThingKind.BODY:
        initial = {key: decode_setting(value) for key, value in doc.get("initial", {}).items()}
        from mime.things import State

        return Body(name, dynamics=cast(Dynamics, entries[(name, "motion")]), initial=State(**initial))
    if kind is ThingKind.APPARATUS:
        return Apparatus(name, fields={field: cast(FieldProvider, entries[(name, f"field:{field}")])
                                       for field in doc.get("fields", {})})
    return Structure(name, dynamics=cast(Dynamics | None, entries.get((name, "motion"))))
