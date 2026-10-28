"""The authored document: what an experiment declares, as data, written as YAML and read back with PyYAML's safe
loader. Its hash is taken over canonical JSON, so formatting never changes it."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from mime.build import Attachment, ControllerDeclaration, Declaration, RunSettings, ThingDeclaration, \
    ValueDeclaration, binding_name, mode_of
from mime.build._derived import sha256_of
from mime.data import Asset
from mime.materials import RECORDS, Acceptance, Extrapolation, Nominal, Record, load_record, record_file
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
    AcceptanceDoc,
    AssetDoc,
    FileDoc,
    BoundEntryDoc,
    ConditionDoc,
    ControllerDoc,
    DeclaredDoc,
    DynamicsDoc,
    EntryDoc,
    ExperimentDocument,
    GroupDoc,
    GroupedValue,
    MagnetDoc,
    MaterialDoc,
    PartDoc,
    QuantityDoc,
    ReferenceDoc,
    SceneDoc,
    SegmentDoc,
    SettingValue,
    ThingDoc,
)
from mime.scene import LAYERS, Scene, sha256_hex
from mime.scene import Asset as FileAsset
from mime.scene import asset as file_asset
from mime.things import (
    Apparatus,
    Body,
    Medium,
    Part,
    PermanentMagnet,
    PrimPath,
    Structure,
    Thing,
    ThingKind,
    TypedPath,
    prim_path_of,
)
from mime.units import Condition, Quantity, parse_unit, unit_symbol

#: Where the saved form writes the stage and the layers it references, below the document's folder.
SCENE_FOLDER = "scene"
#: Where it writes the files an entry reads, such as a robot's description.
DESCRIPTIONS = "descriptions"


class UnsavableError(ValueError):
    """A declaration the saved form cannot hold, named in the experiment's terms."""


def encode_setting(value: object, where: str) -> SettingValue:
    """A setting as the document writes it: a number, string, boolean or null; a list of settings (a tuple saves as a
    list and loads as a tuple); a quantity as its SI value and unit; a typed path (``tube.axis``) as its string, which
    the build reads as the same path; an asset by its name and sha256; a file an entry reads by its path below the
    folder and its sha256; a group of settings by its group name, with those settings that differ from their defaults.
    Anything else, code included, cannot be saved."""
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


def description_path(file_name: str, sha256: str) -> str:
    """Where ``save`` writes a description below the folder: its name behind a prefix of its pin, so that two files of
    one name in different folders keep their own copies."""
    prefix = f"{sha256[:16]}-"
    return f"{DESCRIPTIONS}/{file_name if file_name.startswith(prefix) else prefix + file_name}"


def file_doc(held: FileAsset, where: str) -> FileDoc:
    """A file an entry reads, as the document holds it: its path below the folder, where ``save`` writes it, and its
    pin, the SHA-256 of its bytes when the script gave none."""
    try:
        pin = held.sha256 or sha256_hex(held.read())
    except OSError as error:
        raise UnsavableError(f"{where} names the file {held.path}, which cannot be read to pin it: {error}") from error
    return {"file": description_path(held.path.name, pin), "sha256": pin}


def _plain(value: object, where: str) -> None | bool | int | float | str | QuantityDoc | AssetDoc | FileDoc:
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
    if isinstance(value, FileAsset):
        return file_doc(value, where)
    raise UnsavableError(f"{where} is a {type(value).__name__}; a document holds numbers, strings, booleans, null, "
                         f"lists, quantities, assets and groups of settings, and never code")


def _written(group: SettingGroup) -> dict[str, object]:
    """A group's settings that differ from their defaults."""
    return {f.name: getattr(group, f.name) for f in dataclasses.fields(group)
            if _default(f) is dataclasses.MISSING or getattr(group, f.name) != _default(f)}


def decode_setting(value: SettingValue | GroupedValue, folder: Path | None = None) -> object:
    """The value :func:`encode_setting` wrote: lists come back as tuples, quantities as :class:`Quantity`, assets as
    :class:`~mime.data.Asset`, files as file assets read from below ``folder`` (the working directory when none is
    given) and held to their pins, and groups as their registered class. ``KeyError`` names a group no installed
    package registers."""
    if isinstance(value, list):
        return tuple(decode_setting(item, folder) for item in value)
    if isinstance(value, dict):
        if "group" in value:
            cls = setting_group_named(value["group"])
            return cast(Any, cls)(**{name: decode_setting(item, folder) for name, item in value["settings"].items()})
        if "asset" in value:
            return Asset(value["asset"], value["sha256"])
        if "file" in value:
            return file_asset((folder if folder is not None else Path.cwd()) / value["file"], sha256=value["sha256"])
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
        if f.name in skip or (default is not dataclasses.MISSING and _same_kind(default, value) and default == value):
            continue
        written[f.name] = encode_setting(value, f"{where}, setting {f.name!r}")
    return written


def _same_kind(default: object, value: object) -> bool:
    """Whether ``value`` is of its default's kind, so that being equal it is the default: of one type (so ``1`` is not
    ``True`` or ``1.0``), or both tuples, since a tuple's subclass (a signed axis, ``Z``) saves and loads as a tuple."""
    return type(default) is type(value) or (isinstance(default, tuple) and isinstance(value, tuple))


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
        if thing.parts:
            doc["parts"] = {key: _part_doc(cast(object, part), f"{where}'s part {key!r}")
                            for key, part in sorted(thing.parts.items())}
    if isinstance(thing, Apparatus):
        doc["fields"] = {name: _entry_doc(provider, f"{where}'s field {name!r}")
                         for name, provider in sorted(thing.fields.items())}
        if thing.chain:
            doc["chain"] = [_segment_doc(name, cast(Entry, segment), f"{where}'s segment {name!r}")
                            for name, segment in thing.chain.items()]
        if thing.magnets:
            doc["magnets"] = {key: _magnet_doc(cast(object, role), f"{where}'s magnet {key!r}")
                              for key, role in sorted(thing.magnets.items())}
        if thing.parts:
            doc["parts"] = {key: _part_doc(cast(object, part), f"{where}'s part {key!r}")
                            for key, part in sorted(thing.parts.items())}
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
    if isinstance(thing, Medium) and thing.material is not None:
        doc["material"] = _material_name(cast(object, thing.material), f"{where}'s material")
    return doc


def _segment_doc(name: str, segment: Entry, where: str) -> SegmentDoc:
    plain = _entry_doc(segment, where)
    doc: SegmentDoc = {"segment": name, "name": plain["name"], "version": plain["version"]}
    if "settings" in plain:
        doc["settings"] = plain["settings"]
    return doc


def _material_name(material: object, where: str) -> str:
    if not isinstance(material, Record):
        raise UnsavableError(f"{where} is {material!r}, not a mime.materials.Record")
    return material.name


def _part_doc(part: object, where: str) -> PartDoc:
    if not isinstance(part, Part):
        raise UnsavableError(f"{where} is {part!r}, not a Part")
    doc: PartDoc = {"solid": _entry_doc(cast(Entry, part.geometry), f"{where}'s solid")}
    if part.at is not None:
        at = encode_setting(part.at, f"{where}'s at")
        if not (isinstance(at, dict) and "unit" in at):
            raise UnsavableError(f"{where}'s at={part.at!r} is not a point (three lengths)")
        doc["at"] = at
    if part.material is not None:
        doc["material"] = _material_name(cast(object, part.material), f"{where}'s material")
    if part.inside is not None:
        doc["inside"] = str(part.inside)
    if part.mount is not None:
        doc["mount"] = str(part.mount)
    return doc


def materials_of(declarations: Sequence[Declaration]) -> dict[str, Record]:
    """Every material the experiment's things name, by its name; two different materials under one name cannot be
    saved."""
    found: dict[str, Record] = {}
    for declaration in declarations:
        if not isinstance(declaration, ThingDeclaration):
            continue
        thing = declaration.thing
        held: list[object] = []
        if isinstance(thing, Medium) and thing.material is not None:
            held.append(thing.material)
        if isinstance(thing, Body):
            held += [part.material for part in thing.parts.values()
                     if isinstance(cast(object, part), Part) and part.material is not None]
        for material in held:
            if not isinstance(material, Record):
                raise UnsavableError(f"{thing.kind.value} {thing.name!r}'s material is {material!r}, not a Record")
            if found.get(material.name, material) is not material:
                raise UnsavableError(f"two different materials are named {material.name!r}")
            found[material.name] = material
    return found


def material_doc(material: Record) -> MaterialDoc:
    """``material`` as the document holds it: its record's reference (the file below the folder, the file's SHA-256,
    the record's id, version and content hash, read from the file) and the experiment's own terms for reading it.
    Never a value from the record."""
    loaded, problems = load_record(material)
    if loaded is None or problems:
        raise UnsavableError("; ".join(message for _, message, _ in problems))
    doc: MaterialDoc = {"record": {"file": f"{RECORDS}/{loaded.file}", "sha256": loaded.sha256, "id": loaded.id,
                                   "version": loaded.version, "content_hash": loaded.content_hash}}
    if material.require:
        doc["require"] = {name: list(tokens) for name, tokens in sorted(material.require.items())}
    if material.accept:
        doc["accept"] = {name: AcceptanceDoc(covers=list(held.covers), reason=held.reason)
                         for name, held in sorted(material.accept.items())}
    if material.nominal:
        nominal: dict[str, Any] = {}
        for name, held in sorted(material.nominal.items()):
            value = held.value.value
            if isinstance(value, tuple):
                raise UnsavableError(f"material {material.name!r}'s nominal {name} is {held.value}, not one number")
            nominal[name] = {"value": value, "unit": unit_symbol(held.value.dimension), "reason": held.reason}
        doc["nominal"] = nominal
    if material.extrapolation:
        doc["extrapolation"] = {name: {"method": held.method, "basis": held.basis, "reason": held.reason}
                                for name, held in sorted(material.extrapolation.items())}
    return doc


def conditions_doc(conditions: object) -> dict[str, ConditionDoc]:
    """The experiment's operating conditions, each its value and tolerance in SI with their unit."""
    if conditions is None:
        return {}
    if not isinstance(conditions, Mapping):
        raise UnsavableError(f"conditions={conditions!r} is not a mapping of names to conditions")
    found: dict[str, ConditionDoc] = {}
    for name, condition in sorted(cast(Mapping[str, object], conditions).items()):
        if not isinstance(condition, Condition) or isinstance(condition.value.value, tuple) or isinstance(
                condition.plus_minus.value, tuple):
            raise UnsavableError(f"the condition {name!r} is {condition!r}, not one value with its tolerance")
        found[name] = {"value": condition.value.value, "plus_minus": condition.plus_minus.value,
                       "unit": unit_symbol(condition.value.dimension)}
    return found


def conditions_from(conditions: Mapping[str, ConditionDoc]) -> dict[str, Condition]:
    """The conditions :func:`conditions_doc` wrote. Raises ``UnitError``, a ``ValueError``, for a unit MIME does not
    write."""
    found: dict[str, Condition] = {}
    for name, held in conditions.items():
        dimension = parse_unit(held["unit"])
        found[name] = Condition(Quantity(float(held["value"]), dimension), Quantity(float(held["plus_minus"]),
                                                                                    dimension))
    return found


def materials_from(materials: Mapping[str, MaterialDoc], folder: Path) -> dict[str, Record]:
    """The materials a document names, each record file read from below ``folder`` and held to its pin. Raises
    ``UnitError``, a ``ValueError``, for a nominal's unit MIME does not write."""
    found: dict[str, Record] = {}
    for name, held in materials.items():
        nominal = {key: Nominal(Quantity(float(value["value"]), parse_unit(value["unit"])), value["reason"])
                   for key, value in held.get("nominal", {}).items()}
        found[name] = Record(
            name, record_file(folder / held["record"]["file"], sha256=held["record"]["sha256"]),
            require={key: tuple(tokens) for key, tokens in held.get("require", {}).items()},
            accept={key: Acceptance(tuple(value["covers"]), value["reason"])
                    for key, value in held.get("accept", {}).items()},
            nominal=nominal,
            extrapolation={key: Extrapolation(value["method"], value["basis"], value["reason"])
                           for key, value in held.get("extrapolation", {}).items()})
    return found


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
    controller: ControllerDoc | None = None
    for declaration in declarations:
        if isinstance(declaration, ControllerDeclaration):
            if controller is not None:
                raise UnsavableError("set_controller is called more than once, and an experiment has one controller")
            controller = _controller_doc(declaration)
        elif isinstance(declaration, ValueDeclaration):
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
        "conditions": conditions_doc(settings.conditions),
        "things": dict(sorted(things.items())),
        "interactions": sorted(interactions, key=lambda d: d["id"]),
        "observations": sorted(observations, key=lambda d: d["id"]),
        "declared": dict(sorted(declared.items())),
        "materials": {name: material_doc(held) for name, held in sorted(materials_of(declarations).items())},
        "controller": controller,
    }


def _controller_doc(declaration: ControllerDeclaration) -> ControllerDoc:
    """The controller as the document holds it: the entry and its settings, the mode by word, the paths bound to its
    labels, and its rate and latency when declared."""
    plain = _entry_doc(declaration.controller, "the controller")
    mode = mode_of(declaration.mode)
    if mode is None:
        raise UnsavableError(f"the controller's mode={declaration.mode!r} names no mode")
    paths: dict[str, dict[str, str]] = {}
    for kind, given in (("reads", declaration.reads), ("writes", declaration.writes)):
        held = cast(object, given)  # a script is not type-checked
        if not isinstance(held, Mapping):
            raise UnsavableError(f"the controller's {kind}={held!r} is not a mapping of labels to paths")
        paths[kind] = {str(label): binding_name(path) for label, path in sorted(cast(Mapping[object, object],
                                                                                     held).items(), key=str)}
    doc: ControllerDoc = {"name": plain["name"], "version": plain["version"], "mode": cast(Any, mode.word),
                          "reads": paths["reads"], "writes": paths["writes"]}
    if "settings" in plain:
        doc["settings"] = plain["settings"]
    if declaration.rate is not None:
        doc["rate"] = _quantity_doc(declaration.rate, "the controller's rate")
    if declaration.latency is not None:
        doc["latency"] = _quantity_doc(declaration.latency, "the controller's latency")
    return doc


def _quantity_doc(value: object, where: str) -> QuantityDoc:
    written = encode_setting(value, where)
    if not (isinstance(written, dict) and "unit" in written):
        raise UnsavableError(f"{where}={value!r} is not a quantity")
    return written


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


def thing_from(name: str, doc: ThingDoc, entries: Mapping[tuple[str, str], Entry], bound: bool = False,
               materials: Mapping[str, Record] | None = None) -> Thing:
    """A thing from its document entry, with its entries already constructed and keyed by ``(name, where)`` (a
    part's solid by ``(name, "part:<key>")``), and the materials it names by name; ``bound`` when its prim is one a
    referenced asset brings in."""
    kind = ThingKind(doc["kind"])
    held = materials or {}
    if kind is ThingKind.MEDIUM:
        material = held[doc["material"]] if "material" in doc else None
        return Medium(name, fills=doc.get("fills"), flow=cast(Dynamics | None, entries.get((name, "flow"))),
                      material=material)
    if kind is ThingKind.BODY:
        initial = {key: decode_setting(value) for key, value in doc.get("initial", {}).items()}
        from mime.things import State

        magnets = {key: PermanentMagnet(cast(tuple[float, float, float], tuple(magnet["direction"]))
                                        if "direction" in magnet else None)
                   for key, magnet in doc.get("magnets", {}).items()}
        parts = {key: Part(cast(Any, entries[(name, f"part:{key}")]),
                           material=held[part["material"]] if "material" in part else None,
                           at=cast(Quantity, decode_setting(part["at"])) if "at" in part else None,
                           inside=part.get("inside"), mount=part.get("mount"))
                 for key, part in doc.get("parts", {}).items()}
        return Body(name, dynamics=cast(Dynamics, entries[(name, "motion")]), initial=State(**initial),
                    prim=PrimPath(doc["prim"]) if bound else None, magnets=magnets, parts=parts)
    if kind is ThingKind.APPARATUS:
        return Apparatus(name, fields={field: cast(FieldProvider, entries[(name, f"field:{field}")])
                                       for field in doc.get("fields", {})},
                         commands={command: cast(Drive, entries[(name, f"command:{command}")])
                                   for command in doc.get("commands", {})},
                         chain={segment["segment"]: cast(Any, entries[(name, f"segment:{segment['segment']}")])
                                for segment in doc.get("chain", [])},
                         parts={key: Part(cast(Any, entries[(name, f"part:{key}")]),
                                          at=cast(Quantity, decode_setting(part["at"])) if "at" in part else None,
                                          mount=part.get("mount")) for key, part in doc.get("parts", {}).items()},
                         magnets={key: PermanentMagnet(cast(tuple[float, float, float], tuple(magnet["direction"]))
                                                       if "direction" in magnet else None)
                                  for key, magnet in doc.get("magnets", {}).items()})
    return Structure(name, geometry=cast(Geometry | None, entries.get((name, "geometry"))),
                     dynamics=cast(Dynamics | None, entries.get((name, "motion"))))
