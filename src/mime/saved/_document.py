"""The authored document: what an experiment declares, as data, written as YAML and read back with PyYAML's safe
loader. Its hash is taken over canonical JSON, so formatting never changes it."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import yaml

from mime.build import Attachment, Declaration, RunSettings, ThingDeclaration, ValueDeclaration, binding_name
from mime.build._derived import sha256_of
from mime.data import Asset
from mime.registry import (
    Drive,
    Dynamics,
    Entry,
    FieldProvider,
    Geometry,
    Interaction,
    Observation,
    SettingGroup,
    registration_of,
    setting_group_named,
)
from mime.saved._schema import (
    DOCUMENT_SCHEMA,
    AssetDoc,
    BoundEntryDoc,
    DeclaredDoc,
    DynamicsDoc,
    EntryDoc,
    ExperimentDocument,
    GroupDoc,
    GroupedValue,
    MagnetDoc,
    QuantityDoc,
    ReferenceDoc,
    SceneDoc,
    SettingValue,
    ThingDoc,
)
from mime.scene import LAYERS, Scene, sha256_hex
from mime.things import (
    Apparatus,
    Body,
    Medium,
    PermanentMagnet,
    PrimPath,
    Structure,
    Thing,
    ThingKind,
    TypedPath,
    prim_path_of,
)
from mime.units import Quantity, parse_unit, unit_symbol

#: Where the saved form writes the stage and the layers it references, below the document's folder.
SCENE_FOLDER = "scene"


class UnsavableError(ValueError):
    """A declaration the saved form cannot hold, named in the experiment's terms."""


def encode_setting(value: object, where: str) -> SettingValue:
    """A setting as the document writes it: a number, string, boolean or null; a list of settings (a tuple saves as a
    list and loads as a tuple); a quantity as its SI value and unit; a typed path (``tube.axis``) as its string, which
    the build reads as the same path; an asset by its name and sha256; a group of settings by its group name, with
    those settings that differ from their defaults. Anything else, code included, cannot be saved."""
    if isinstance(value, SettingGroup):
        group: GroupDoc = {"group": value.group, "settings": {
            name: _grouped(setting, f"{where}.{name}") for name, setting in _written(value).items()}}
        return group
    if isinstance(value, (list, tuple)):
        return [encode_setting(item, f"{where}[{i}]") for i, item in enumerate(cast(Sequence[object], value))]
    return _plain(value, where)


def _grouped(value: object, where: str) -> GroupedValue:
    if isinstance(value, SettingGroup):
        raise UnsavableError(f"{where} is a group of settings inside a group; a group holds no group")
    if isinstance(value, (list, tuple)):
        return [_grouped(item, f"{where}[{i}]") for i, item in enumerate(cast(Sequence[object], value))]
    return _plain(value, where)


def _plain(value: object, where: str) -> None | bool | int | float | str | QuantityDoc | AssetDoc:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, TypedPath):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsavableError(f"{where} is {value!r}; the document holds finite numbers only")
        return value
    if isinstance(value, Quantity):
        magnitude = value.value
        quantity: QuantityDoc = {"value": list(magnitude) if isinstance(magnitude, tuple) else magnitude,
                                 "unit": unit_symbol(value.dimension)}
        return quantity
    if isinstance(value, Asset):
        held: AssetDoc = {"asset": value.name, "sha256": value.sha256}
        return held
    raise UnsavableError(f"{where} is a {type(value).__name__}; a document holds numbers, strings, booleans, null, "
                         f"lists, quantities, assets and groups of settings, and never code")


def _written(group: SettingGroup) -> dict[str, object]:
    """A group's settings that differ from their defaults."""
    return {f.name: getattr(group, f.name) for f in dataclasses.fields(group)
            if _default(f) is dataclasses.MISSING or getattr(group, f.name) != _default(f)}


def decode_setting(value: SettingValue | GroupedValue) -> object:
    """The value :func:`encode_setting` wrote: lists come back as tuples, quantities as :class:`Quantity`, assets as
    :class:`~mime.data.Asset` and groups as their registered class. ``KeyError`` names a group no installed package
    registers."""
    if isinstance(value, list):
        return tuple(decode_setting(item) for item in value)
    if isinstance(value, dict):
        if "group" in value:
            cls = setting_group_named(value["group"])
            return cast(Any, cls)(**{name: decode_setting(item) for name, item in value["settings"].items()})
        if "asset" in value:
            return Asset(value["asset"], value["sha256"])
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


def _magnet_doc(role: object, where: str) -> MagnetDoc:
    if not isinstance(role, PermanentMagnet):
        raise UnsavableError(f"{where} is {role!r}, not a magnet's role")
    doc: MagnetDoc = {"model": "permanent_magnet"}
    if role.direction is not None:
        direction = encode_setting(role.direction, f"{where}'s direction")
        if not (isinstance(direction, list) and all(isinstance(c, (int, float)) for c in direction)):
            raise UnsavableError(f"{where}'s direction is {role.direction!r}, not three numbers")
        doc["direction"] = [float(cast(float, c)) for c in direction]
    return doc


def _thing_doc(thing: Thing) -> ThingDoc:
    where = f"{thing.kind.value} {thing.name!r}"
    doc: ThingDoc = {"kind": thing.kind.value, "prim": prim_path_of(thing)}
    if isinstance(thing, Medium) and thing.flow is not None:
        doc["dynamics"] = DynamicsDoc(flow=_entry_doc(thing.flow, f"{where}'s flow"))
    elif isinstance(thing, (Body, Structure)) and thing.dynamics is not None:
        doc["dynamics"] = DynamicsDoc(motion=_entry_doc(thing.dynamics, f"{where}'s motion"))
    if isinstance(thing, Body):
        doc["initial"] = {key: encode_setting(value, f"{where}'s initial {key}")
                          for key, value in thing.initial.items()}
        if thing.magnets:
            doc["magnets"] = {key: _magnet_doc(cast(object, role), f"{where}'s magnet {key!r}")
                              for key, role in sorted(thing.magnets.items())}
    if isinstance(thing, Apparatus):
        doc["fields"] = {name: _entry_doc(provider, f"{where}'s field {name!r}")
                         for name, provider in sorted(thing.fields.items())}
        if thing.commands:
            doc["commands"] = {name: _entry_doc(cast(Entry, drive), f"{where}'s command {name!r}")
                               for name, drive in sorted(thing.commands.items())}
    if isinstance(thing, Structure) and thing.geometry is not None:
        doc["geometry"] = _entry_doc(thing.geometry, f"{where}'s geometry")
    if isinstance(thing, Medium) and thing.fills is not None:
        fills = encode_setting(thing.fills, f"{where}'s fills")
        if not isinstance(fills, str):
            raise UnsavableError(f"{where} fills {thing.fills!r}, not a region such as tube.lumen")
        doc["fills"] = fills
    return doc


def _declared_doc(declaration: ValueDeclaration) -> tuple[str, DeclaredDoc]:
    path = str(declaration.path)
    if not isinstance(declaration.value, Quantity) or not isinstance(declaration.reason, str):
        raise UnsavableError(f"the declared value {path} is {declaration.value!r} with the reason "
                             f"{declaration.reason!r}: a quantity and a string")
    value = declaration.value.value
    return path, {"value": list(value) if isinstance(value, tuple) else value,
                  "unit": unit_symbol(declaration.value.dimension), "reason": declaration.reason}


def document_of(settings: RunSettings, declarations: Sequence[Declaration]) -> ExperimentDocument:
    """The document of a declared experiment. Things are keyed by name and interactions and observations sorted by
    id, so the order a script declares them in changes nothing, as it changes nothing in the build."""
    from mime import __version__
    from mime.build import entry_id

    things: dict[str, ThingDoc] = {}
    interactions: list[BoundEntryDoc] = []
    observations: list[BoundEntryDoc] = []
    declared: dict[str, DeclaredDoc] = {}
    for declaration in declarations:
        if isinstance(declaration, ValueDeclaration):
            path, doc = _declared_doc(declaration)
            declared[path] = doc
        elif isinstance(declaration, ThingDeclaration):
            things[declaration.thing.name] = _thing_doc(declaration.thing)
        elif isinstance(declaration, Attachment):
            law = declaration.law
            owner = entry_id(type(law), declaration.bindings)
            interactions.append(_bound_doc(law, owner, declaration.bindings, f"interaction {owner}"))
        else:
            observation = declaration.observation
            owner = entry_id(type(observation), observation.bindings())
            observations.append(_bound_doc(observation, owner, observation.bindings(), f"observation {owner}"))
    dt, gravity = encode_setting(settings.dt, "dt"), encode_setting(settings.gravity, "gravity")
    if not (isinstance(dt, dict) and "unit" in dt and isinstance(gravity, dict) and "unit" in gravity):
        raise UnsavableError("dt and gravity are quantities, such as ms(1) and standard_gravity()")
    return {
        "schema": DOCUMENT_SCHEMA,
        "mime": __version__,
        "experiment": settings.experiment,
        "run": {"dt": dt, "seed": settings.seed},
        "scene": scene_doc(settings.scene),
        "gravity": gravity,
        "things": dict(sorted(things.items())),
        "interactions": sorted(interactions, key=lambda d: d["id"]),
        "observations": sorted(observations, key=lambda d: d["id"]),
        "declared": dict(sorted(declared.items())),
        "materials": {},
        "controller": None,
    }


def scene_doc(scene: object) -> SceneDoc:
    """The scene as the document holds it: up, units, and each reference's prim, its file below the document's
    folder (where ``save`` writes it), and its pin (the file's SHA-256 when the script gave none)."""
    if not isinstance(scene, Scene):
        raise UnsavableError(f"scene={scene!r} is not a Scene")
    try:
        up = [float(c) for c in cast(Sequence[Any], scene.declared_up)]
    except (TypeError, ValueError) as error:
        raise UnsavableError(f"the scene's up={scene.declared_up!r} is not three numbers") from error
    if not isinstance(scene.units, str):
        raise UnsavableError(f"the scene's units={scene.units!r} is not a unit's name")
    references: list[ReferenceDoc] = []
    for reference in scene.references:
        try:
            pin = reference.asset.sha256 or sha256_hex(reference.asset.read())
        except OSError as error:
            raise UnsavableError(f"the asset referenced at {reference.at} cannot be read to pin it: {error}") from error
        references.append({"at": reference.at, "layer": f"{SCENE_FOLDER}/{LAYERS}/{reference.asset.path.name}",
                           "sha256": pin})
    return {"up": up, "units": scene.units, "references": sorted(references, key=lambda r: r["at"])}


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


def thing_from(name: str, doc: ThingDoc, entries: Mapping[tuple[str, str], Entry], bound: bool = False) -> Thing:
    """A thing from its document entry, with its entries already constructed and keyed by ``(name, where)``;
    ``bound`` when its prim is one a referenced asset brings in."""
    kind = ThingKind(doc["kind"])
    if kind is ThingKind.MEDIUM:
        return Medium(name, fills=doc.get("fills"), flow=cast(Dynamics | None, entries.get((name, "flow"))))
    if kind is ThingKind.BODY:
        initial = {key: decode_setting(value) for key, value in doc.get("initial", {}).items()}
        from mime.things import State

        magnets = {key: PermanentMagnet(cast(tuple[float, float, float], tuple(magnet["direction"]))
                                        if "direction" in magnet else None)
                   for key, magnet in doc.get("magnets", {}).items()}
        return Body(name, dynamics=cast(Dynamics, entries[(name, "motion")]), initial=State(**initial),
                    prim=PrimPath(doc["prim"]) if bound else None, magnets=magnets)
    if kind is ThingKind.APPARATUS:
        return Apparatus(name, fields={field: cast(FieldProvider, entries[(name, f"field:{field}")])
                                       for field in doc.get("fields", {})},
                         commands={command: cast(Drive, entries[(name, f"command:{command}")])
                                   for command in doc.get("commands", {})})
    return Structure(name, geometry=cast(Geometry | None, entries.get((name, "geometry"))),
                     dynamics=cast(Dynamics | None, entries.get((name, "motion"))))
