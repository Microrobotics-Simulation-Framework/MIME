"""What a build writes beside the graph: the document, the resolved record, the entries and their distributions,
the seeds, the environment, and the hashes. Computed at every build; written to a folder by ``save``."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax

from mime.build import Declaration, PlanSet, Refusal, RefusalClass, RunSettings
from mime.build._derived import HASH_RECIPE, DerivedGraph, sha256_of
from mime.build._plan import EntryPlan
from mime.data import Asset, asset_source
from mime.materials import RECORDS, LoadedRecord
from mime.registry import SettingGroup
from mime.saved._document import (
    DESCRIPTIONS,
    SCENE_FOLDER,
    UnsavableError,
    all_settings,
    document_hash,
    document_of,
    encode_setting,
)
from mime.saved._resolve import entry_records
from mime.saved._schema import (
    ConditionDoc,
    DescriptionRecord,
    EntryRecord,
    EnvironmentRecord,
    ExperimentDocument,
    GeometryRecord,
    GroupRecord,
    HashRecord,
    Json,
    Lock,
    MagnetRecord,
    PartRecord,
    ReadRecord,
    RecordDoc,
    RecordFileRecord,
    ReferenceRecord,
    RegistrationRecord,
    ResolvedEntry,
    ResolvedMaterials,
    ResolvedRecord,
    ResolvedScene,
    SettingValue,
    SlotRecord,
    StageRecord,
    ValueRecord,
)
from mime.core.multibody import DescriptionClassification, ElementGroup
from mime.scene import STAGE_FILE, ComposedLayer, sha256_hex
from mime.scene import Asset as FileAsset
from mime.vocabulary import MIME_NAMESPACE, namespace_of


def environment() -> EnvironmentRecord:
    """What this process builds under: the versions that decide what a build computes, the platform, the device,
    the precision and the PRNG's implementation and flag (the values drawn for a key depend on both)."""
    from mime import __version__

    config = cast(Any, jax.config)
    return {
        "mime": __version__,
        "maddening": importlib.metadata.version("maddening"),
        "openmaterica": importlib.metadata.version("openmaterica"),
        "usd_core": importlib.metadata.version("usd-core"),
        "jax": importlib.metadata.version("jax"),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "device": jax.default_backend(),
        "x64": bool(config.jax_enable_x64),
        "prng": f"{config.jax_default_prng_impl} (jax_threefry_partitionable={config.jax_threefry_partitionable})",
    }


def _floats(value: object) -> list[float]:
    return [float(v) for v in cast(Sequence[float], value)]


def _resolved_entry(plan: EntryPlan) -> ResolvedEntry:
    record = plan.registered
    initial: dict[str, SettingValue] = {key: encode_setting(value, f"{plan.owner}'s initial {key}")
                                        for key, value in sorted(plan.initial.items())}
    return {
        "kind": record.kind.value,
        "name": record.name,
        "version": record.version,
        "differentiability": record.cls.differentiability.value,
        "binds": dict(sorted(plan.bindings.items())),
        "settings": all_settings(plan.entry, plan.owner),
        "initial": initial,
        "nodes": plan.node_names(),
        "every": plan.divider,
    }


def _namespaces_used(plans: PlanSet) -> set[str]:
    """MIME's, and those of every entry the experiment uses and every interface it reads or delivers into: the
    vocabulary versions a load must agree with, and no namespace this process merely happens to hold."""
    used = {MIME_NAMESPACE}
    for plan in plans.entries:
        used.add(plan.registered.namespace)
        interfaces = [read.interface for read in plan.reads.values()]
        interfaces += [target.interface for target in plan.delivers.values()]
        for registration in plan.registrations:
            interfaces += [read.interface for read in registration.reads.values()]
            interfaces += [target.interface for target in registration.returns.values()]
        used |= {namespace_of(name) for name in interfaces}
    used |= {geometry.registered.namespace for geometry in plans.geometries.values()}
    return used


def resolved_scene(plans: PlanSet) -> ResolvedScene:
    magnets: dict[str, dict[str, MagnetRecord]] = {}
    for magnet in plans.scene.magnets:
        magnets.setdefault(magnet.body, {})[magnet.key] = {
            "part": magnet.part, "model": magnet.model, "in_part": list(magnet.in_part),
            "in_body": list(magnet.in_body), "source": magnet.source}
    return {"up": list(plans.scene.up), "units": plans.scene.units, "meters_per_unit": plans.scene.meters_per_unit,
            "prims": dict(plans.scene.prims), "bound": sorted(plans.scene.bound), "magnets": magnets}


def resolved_record(plans: PlanSet) -> ResolvedRecord:
    things = {name: kind.value for name, kind in plans.things.items()}
    registrations: dict[str, list[RegistrationRecord]] = {
        p.owner: [{"id": r.id, "kind": r.kind, "subject": r.subject} for r in p.registrations]
        for p in plans.entries if p.registrations}
    slots: dict[str, list[SlotRecord]] = {
        p.owner: [{"name": s.name, "reader": s.reader, "interfaces": list(s.interfaces)} for s in p.slots]
        for p in plans.entries if p.slots}
    groups: list[GroupRecord] = [{"nodes": list(g.nodes), "tolerance": g.tolerance, "max_iterations": g.max_iterations,
                                  "requested_by": list(g.requested_by)} for g in plans.groups]
    values: dict[str, ValueRecord] = {
        v.path: {"value": list(v.value) if isinstance(v.value, tuple) else v.value, "unit": v.unit,
                 "source": v.source, "reason": v.reason, "read_by": list(v.read_by), "details": list(v.details)}
        for v in plans.values}
    geometry: dict[str, GeometryRecord] = {
        name: {"name": g.registered.name, "version": g.registered.version,
               "settings": all_settings(g.geometry, f"structure {name!r}'s geometry")}
        for name, g in plans.geometries.items()}
    return {
        "vocabulary": {name: version for name, version in plans.vocabulary.items()
                       if name in _namespaces_used(plans)},
        "run": {"experiment": plans.settings.experiment, "dt": plans.dt, "seed": plans.settings.seed,
                "up": _floats(plans.frame.up), "gravity": _floats(plans.frame.gravity),
                "structure": plans.run.text()},
        "things": dict(sorted(things.items())),
        "entries": {p.owner: _resolved_entry(p) for p in sorted(plans.entries, key=lambda p: p.owner)},
        "order": list(plans.order),
        "registrations": registrations,
        "slots": slots,
        "groups": groups,
        "values": values,
        "geometry": geometry,
        "fills": dict(plans.fills),
        "assets": {held.name: {"source": asset_source(held), "sha256": held.sha256}
                   for held in sorted(_assets(plans), key=lambda a: (a.name, a.sha256))},
        "scene": resolved_scene(plans),
        "materials": resolved_materials(plans),
        "descriptions": {plan.owner: description_record(plan.segment.description) for plan in plans.entries
                         if plan.segment is not None and plan.segment.description is not None},
    }


def description_record(held: DescriptionClassification) -> DescriptionRecord:
    """A description as read: the file and its pin, every element by group and kind, what was read past and why, and
    the links read as massless."""
    def by_kind(group: ElementGroup) -> dict[str, list[str]]:
        return {kind: list(paths) for kind, paths in sorted(held.categories(group).items())}

    return {"file": f"{DESCRIPTIONS}/{held.source.file_name}", "sha256": held.source.sha256,
            "used": by_kind(ElementGroup.READ), "unused": by_kind(ElementGroup.UNUSED),
            "refused": by_kind(ElementGroup.REFUSED), "acknowledged": dict(held.acknowledged),
            "massless": [gap.link for gap in held.inertia_gaps]}


def record_doc(loaded: LoadedRecord) -> RecordDoc:
    return {"file": f"{RECORDS}/{loaded.file}", "sha256": loaded.sha256, "id": loaded.id, "version": loaded.version,
            "content_hash": loaded.content_hash}


def resolved_materials(plans: PlanSet) -> ResolvedMaterials:
    """The conditions, every record's pins, every read the values rest on (the request as sent, OpenMaterica's
    verdict, what the build did), and each body's parts."""
    held = plans.materials
    conditions: dict[str, ConditionDoc] = {c.name: {"value": c.value, "plus_minus": c.plus_minus, "unit": c.unit}
                                           for c in held.conditions}
    reads: list[ReadRecord] = []
    for outcome in held.reads:
        taken: dict[str, float | list[float]] = {}
        if outcome.value is not None:
            taken["value"] = outcome.value
        if outcome.bounds is not None:
            taken["bounds"] = list(outcome.bounds)
        reads.append(cast(ReadRecord, {
            "material": outcome.material, "field": outcome.field,
            "request": cast(Json, json.loads(json.dumps(outcome.request))), "status": outcome.status,
            "reasons": list(outcome.reasons), "use": outcome.use, **taken}))
    parts: dict[str, dict[str, PartRecord]] = {}
    for part in held.parts:
        parts.setdefault(part.body, {})[part.name] = {
            "solid": {"name": part.registered.name, "version": part.registered.version,
                      "settings": all_settings(part.solid, f"body {part.body!r}'s part {part.name!r}")},
            "volume": part.volume, "centre": list(part.centre), "rotation": [list(row) for row in part.rotation],
            "inertia_per_mass": [list(row) for row in part.inertia_per_mass], "inside": part.inside,
            "material": part.material, "magnet": part.magnet}
    return {"conditions": conditions, "records": {r.material: record_doc(r) for r in held.records}, "reads": reads,
            "parts": parts}


def _files(plans: PlanSet) -> list[tuple[str, bytes]]:
    """Every file an entry of the experiment reads among its settings (a robot's description), as its path below the
    folder and its bytes, read once, which ``save`` writes there."""
    found: dict[str, bytes] = {}
    for value in (getattr(plan.entry, f.name) for plan in plans.entries for f in dataclasses.fields(plan.entry)):
        if isinstance(value, FileAsset):
            found[f"{DESCRIPTIONS}/{value.path.name}"] = value.read()
    return sorted(found.items())


def _assets(plans: PlanSet) -> set[Asset]:
    """Every asset an entry of the experiment names among its settings, groups of settings included."""
    found: set[Asset] = set()
    for value in (getattr(plan.entry, f.name) for plan in plans.entries for f in dataclasses.fields(plan.entry)):
        held = [getattr(value, f.name) for f in dataclasses.fields(value)] if isinstance(value, SettingGroup) else [value]
        found |= {item for item in held if isinstance(item, Asset)}
    return found


def input_hash(document: ExperimentDocument, resolved: ResolvedRecord, entries: Sequence[EntryRecord]) -> str:
    """The exact identity a load must reproduce: the document, everything the build resolved from it, and the
    registered names with their versions (not their distributions, which the environment record reports)."""
    names = sorted(f"{e['kind']}:{e['name']}@{e['version']}" for e in entries)
    return sha256_of({"recipe": HASH_RECIPE, "document": document, "resolved": resolved, "entries": names})


@dataclass(frozen=True)
class SavedRecord:
    """The saved form as a build computes it, before ``save`` writes it with the reference outputs. ``unsavable``
    holds every reason it cannot be written, each a refusal of ``save``. ``stage`` is the stage's text and
    ``layers`` the asset layers it references, as the build read them."""

    document: ExperimentDocument | None
    resolved: ResolvedRecord
    entries: tuple[EntryRecord, ...]
    seeds: dict[str, dict[str, int]]
    environment: EnvironmentRecord
    derived: DerivedGraph
    unsavable: tuple[Refusal, ...]
    stage: str = ""
    layers: tuple[ComposedLayer, ...] = ()
    records: tuple[LoadedRecord, ...] = ()
    files: tuple[tuple[str, bytes], ...] = ()

    @property
    def stage_hash(self) -> str:
        return sha256_hex(self.stage.encode("utf-8"))

    def record_files(self) -> dict[str, RecordFileRecord]:
        return {r.material: {"file": f"{RECORDS}/{r.file}", "sha256": r.sha256} for r in self.records}

    def stage_record(self) -> StageRecord:
        return {"file": STAGE_FILE, "sha256": self.stage_hash,
                "layers": {f"{SCENE_FOLDER}/{layer.file}": layer.sha256 for layer in self.layers}}

    def hashes(self) -> HashRecord:
        if self.document is None:
            raise ValueError("this experiment cannot be saved:\n" + "\n".join(str(r) for r in self.unsavable))
        return {"recipe": HASH_RECIPE, "document": document_hash(self.document),
                "input": input_hash(self.document, self.resolved, self.entries),
                "derived": self.derived.physics_hash, "observations": self.derived.observations_hash}

    def lock(self, reference: ReferenceRecord) -> Lock:
        return {
            "schema": "mime.lock/1",
            "environment": self.environment,
            "entries": list(self.entries),
            "seeds": self.seeds,
            "hashes": self.hashes(),
            "resolved": self.resolved,
            "derived": {"physics": self.derived.physics, "observations": self.derived.observations},
            "stage": self.stage_record(),
            "records": self.record_files(),
            "reference": reference,
        }


def record_of(settings: RunSettings, declarations: Sequence[Declaration], plans: PlanSet,
              derived: DerivedGraph, stage: str = "") -> SavedRecord:
    unsavable: list[Refusal] = []
    document: ExperimentDocument | None = None
    try:
        document = document_of(settings, declarations)
    except UnsavableError as error:
        unsavable.append(Refusal("save.unsavable", RefusalClass.USER, str(error),
                                 "write the setting as a number, string, boolean, null, list or quantity"))
    entries, problems = entry_records([*(p.registered for p in plans.entries),
                                       *(g.registered for g in plans.geometries.values())])
    unsavable += [Refusal("save.not-loadable", RefusalClass.USER, problem,
                          "register the entry from an installed package with a mime.entries entry point")
                  for problem in problems]
    seeds = {p.owner: dict(sorted(p.seeds.items())) for p in plans.entries if p.seeds}
    try:
        files = tuple(_files(plans))
    except OSError as error:
        unsavable.append(Refusal("save.unsavable", RefusalClass.USER, f"a file an entry reads cannot be read: {error}",
                                 "keep the file where the experiment names it until it is saved"))
        files = ()
    return SavedRecord(document, resolved_record(plans), tuple(entries), seeds, environment(), derived,
                       tuple(unsavable), stage, plans.scene.layers, plans.materials.records, files)


__all__ = ["SavedRecord", "environment", "input_hash", "record_of", "resolved_record"]
