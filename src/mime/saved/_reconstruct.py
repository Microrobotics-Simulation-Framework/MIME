"""A document back into declarations: each entry resolved among installed packages and constructed from its saved
settings, so that the experiment it describes can be built again without the script that first declared it."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from mime.build import (
    Attachment,
    Declaration,
    ObservationDeclaration,
    Refusal,
    RefusalClass,
    RunSettings,
    ThingDeclaration,
    ValueDeclaration,
    entry_id,
)
from mime.registry import Entry, Interaction, Observation
from mime.saved._document import decode_setting, thing_from
from mime.saved._folder import LoadRefused
from mime.saved._resolve import import_namespaces, resolve_entry
from mime.saved._schema import BoundEntryDoc, EntryDoc, EntryRecord, ExperimentDocument, SettingValue, ThingDoc
from mime.scene import Scene, asset, within
from mime.things import WORLD, ThingKind

#: The dynamics aspects each kind of thing takes, and whether it takes an initial state or fields.
_ASPECTS: dict[ThingKind, frozenset[str]] = {
    ThingKind.STRUCTURE: frozenset({"motion"}),
    ThingKind.MEDIUM: frozenset({"flow"}),
    ThingKind.BODY: frozenset({"motion"}),
    ThingKind.APPARATUS: frozenset(),
}


class _Reader:
    def __init__(self, locked: Sequence[EntryRecord], referenced: Sequence[str] = ()) -> None:
        self.locked = locked
        self.referenced = tuple(referenced)
        self.refusals: list[Refusal] = []

    def refuse(self, code: str, message: str, action: str) -> None:
        self.refusals.append(Refusal(code, RefusalClass.USER, message, action))

    def entry(self, kind: str, doc: EntryDoc | BoundEntryDoc, where: str,
              bindings: Mapping[str, str] | None = None) -> Entry | None:
        registered, refusal = resolve_entry(kind, doc["name"], doc["version"], where, self.locked)
        if refusal is not None or registered is None:
            self.refusals += [refusal] if refusal else []
            return None
        cls = registered.cls
        written: Mapping[str, SettingValue] = doc.get("settings", {})
        try:
            settings: dict[str, object] = {key: decode_setting(value) for key, value in written.items()}
        except ValueError as error:
            self.refuse("load.units", f"{where}: {error}", "write units as experiment.save() writes them")
            return None
        except (KeyError, TypeError) as error:
            self.refuse("load.setting", f"{where}: a group of settings cannot be read back: {error}",
                        "install the package that registers the group, or fix the group in the document")
            return None
        fields = {f.name for f in dataclasses.fields(cls)}
        roles = set(cls.roles) if issubclass(cls, Observation) else set[str]()
        unknown = sorted(set(settings) - (fields - roles))
        if unknown:
            self.refuse("load.setting-unknown", f"{where}: {registered.name} {registered.version} has no setting "
                                                f"{', '.join(repr(k) for k in unknown)}",
                        f"its settings are: {', '.join(sorted(fields - roles)) or 'none'}")
            return None
        try:
            return cast(Any, cls)(**settings, **(bindings or {}))
        except Exception as error:  # noqa: BLE001 - a setting the entry refuses is a refusal naming it
            self.refuse("load.setting", f"{where}: {registered.name} {registered.version} refuses its saved "
                                        f"settings: {error!r}", "fix the setting in the document")
            return None

    def thing(self, name: str, doc: ThingDoc) -> Declaration | None:
        kind = ThingKind(doc["kind"])
        where = f"{kind.value} {name!r}"
        dynamics = doc.get("dynamics", {})
        wrong = [aspect for aspect in dynamics if aspect not in _ASPECTS[kind]]
        if wrong:
            self.refuse("load.document-thing", f"{where} has {', '.join(wrong)} dynamics, which a {kind.value} does "
                                               f"not take", "remove them from the document")
            return None
        if (("initial" in doc) != (kind is ThingKind.BODY)
                or (("fields" in doc or "commands" in doc) and kind is not ThingKind.APPARATUS)
                or ("geometry" in doc and kind is not ThingKind.STRUCTURE)
                or ("fills" in doc and kind is not ThingKind.MEDIUM)):
            self.refuse("load.document-thing", f"{where}: only a body has an initial state, only an apparatus has "
                                               f"fields and commands, only a structure a geometry, and only a medium "
                                               f"fills a region", "write the thing as experiment.save() writes it")
            return None
        if kind is ThingKind.BODY and "motion" not in dynamics:
            self.refuse("load.document-thing", f"{where} has no motion dynamics", "give the body its dynamics")
            return None
        if "magnets" in doc and kind is not ThingKind.BODY:
            self.refuse("load.document-thing", f"{where}: only a body's parts take a magnet's role",
                        "write the thing as experiment.save() writes it")
            return None
        bound = any(within(doc["prim"], at) for at in self.referenced)
        if (bound and kind is not ThingKind.BODY) or (not bound and doc["prim"] != f"{WORLD}/{name}"):
            self.refuse("load.document-thing", f"{where} has the prim {doc['prim']}: a body may bind a prim a "
                                               f"referenced asset brings in, and every other thing is written at "
                                               f"{WORLD}/{name}", "write the thing as experiment.save() writes it")
            return None
        entries: dict[tuple[str, str], Entry] = {}
        for aspect, entry_doc in dynamics.items():
            built = self.entry("dynamics", entry_doc, f"{where}'s {aspect}")
            if built is None:
                return None
            entries[(name, aspect)] = built
        for field_name, provider_doc in doc.get("fields", {}).items():
            built = self.entry("field provider", provider_doc, f"{where}'s field {field_name!r}")
            if built is None:
                return None
            entries[(name, f"field:{field_name}")] = built
        for command, drive_doc in doc.get("commands", {}).items():
            built = self.entry("drive", drive_doc, f"{where}'s command {command!r}")
            if built is None:
                return None
            entries[(name, f"command:{command}")] = built
        if "geometry" in doc:
            built = self.entry("geometry", doc["geometry"], f"{where}'s geometry")
            if built is None:
                return None
            entries[(name, "geometry")] = built
        try:
            return ThingDeclaration(thing_from(name, doc, entries, bound))
        except ValueError as error:
            self.refuse("load.units", f"{where}: {error}", "write units as experiment.save() writes them")
            return None

    def bound(self, kind: str, doc: BoundEntryDoc) -> Declaration | None:
        where = f"{kind} {doc['id']}"
        observation = kind == "observation"
        built = self.entry(kind, doc, where, doc["binds"] if observation else None)
        if built is None:
            return None
        derived = entry_id(cast(type[Interaction] | type[Observation], type(built)), doc["binds"])
        if derived != doc["id"]:
            self.refuse("load.id", f"{where}: its registered name and bindings give the id {derived!r}; an id is "
                                   f"part of the model, so a renamed one is a different experiment",
                        f"write id: {derived}")
            return None
        if observation:
            return ObservationDeclaration(cast(Observation, built))
        return Attachment(cast(Interaction, built), dict(doc["binds"]))


def entry_names(document: ExperimentDocument) -> list[str]:
    names = [doc["name"] for doc in [*document["interactions"], *document["observations"]]]
    for thing in document["things"].values():
        names += [entry["name"] for entry in thing.get("dynamics", {}).values()]
        names += [entry["name"] for entry in thing.get("fields", {}).values()]
        names += [entry["name"] for entry in thing.get("commands", {}).values()]
        names += [thing["geometry"]["name"]] if "geometry" in thing else []
    return names


def scene_from(document: ExperimentDocument, folder: Path) -> Scene:
    """The scene ``document`` declares, its references read from below ``folder`` (the document's own) and held to
    their pins."""
    held = document["scene"]
    scene = Scene.new(up=tuple(held["up"]), units=held["units"])
    for reference in held["references"]:
        scene.reference(asset(folder / reference["layer"], sha256=reference["sha256"]), at=reference["at"])
    return scene


def declarations_from(document: ExperimentDocument, locked: Sequence[EntryRecord] = (),
                      folder: Path | None = None) -> tuple[RunSettings, list[Declaration]]:
    """The run settings and declarations ``document`` describes, its scene's layers read from below ``folder``
    (the working directory when none is given). Resolves its entries among installed packages and never installs
    anything; raises :class:`LoadRefused` with every reason, each with the line to install."""
    reader = _Reader(locked, [reference["at"] for reference in document["scene"]["references"]])
    reader.refusals += import_namespaces(entry_names(document), locked)
    declarations: list[Declaration] = []
    for name, thing in document["things"].items():
        declarations.append(cast(Declaration, reader.thing(name, thing)))
    for doc in document["interactions"]:
        declarations.append(cast(Declaration, reader.bound("interaction", doc)))
    for doc in document["observations"]:
        declarations.append(cast(Declaration, reader.bound("observation", doc)))
    for path, declared in document["declared"].items():
        try:
            value = decode_setting({"value": declared["value"], "unit": declared["unit"]})
        except ValueError as error:
            reader.refuse("load.units", f"the declared value {path}: {error}", "write units as experiment.save() "
                                                                                "writes them")
            continue
        declarations.append(ValueDeclaration(path, value, declared["reason"]))
    run = document["run"]
    try:
        dt, gravity = decode_setting(run["dt"]), decode_setting(document["gravity"])
    except ValueError as error:
        reader.refuse("load.units", f"the run settings: {error}", "write units as experiment.save() writes them")
        dt = gravity = None
    if reader.refusals:
        raise LoadRefused(reader.refusals)
    scene = scene_from(document, folder if folder is not None else Path.cwd())
    return RunSettings(document["experiment"], scene, gravity, dt, run["seed"]), declarations
