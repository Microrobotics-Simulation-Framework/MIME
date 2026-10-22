"""What a build writes beside the graph: the document, the resolved record, the entries and their distributions,
the seeds, the environment, and the hashes. Computed at every build; written to a folder by ``save``."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax

from mime.build import Declaration, PlanSet, Refusal, RefusalClass, RunSettings
from mime.build._derived import HASH_RECIPE, DerivedGraph, sha256_of
from mime.build._plan import EntryPlan
from mime.saved._document import UnsavableError, all_settings, document_hash, document_of, encode_setting
from mime.saved._resolve import entry_records
from mime.saved._schema import (
    EntryRecord,
    EnvironmentRecord,
    ExperimentDocument,
    GroupRecord,
    HashRecord,
    Lock,
    ReferenceRecord,
    RegistrationRecord,
    ResolvedEntry,
    ResolvedRecord,
    SettingValue,
    SlotRecord,
)
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
    return used


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
    return {
        "vocabulary": {name: version for name, version in plans.vocabulary.items()
                       if name in _namespaces_used(plans)},
        "run": {"experiment": plans.settings.experiment, "dt": plans.dt, "seed": plans.settings.seed,
                "up": _floats(plans.frame.up), "gravity": _floats(plans.frame.gravity),
                "structure": plans.run.text()},
        "things": dict(sorted(things.items())),
        "entries": {p.owner: _resolved_entry(p) for p in sorted(plans.entries, key=lambda p: p.owner)},
        "order": [name for p in plans.entries for name in p.node_names()],
        "registrations": registrations,
        "slots": slots,
        "groups": groups,
    }


def input_hash(document: ExperimentDocument, resolved: ResolvedRecord, entries: Sequence[EntryRecord]) -> str:
    """The exact identity a load must reproduce: the document, everything the build resolved from it, and the
    registered names with their versions (not their distributions, which the environment record reports)."""
    names = sorted(f"{e['kind']}:{e['name']}@{e['version']}" for e in entries)
    return sha256_of({"recipe": HASH_RECIPE, "document": document, "resolved": resolved, "entries": names})


@dataclass(frozen=True)
class SavedRecord:
    """The saved form as a build computes it, before ``save`` writes it with the reference outputs. ``unsavable``
    holds every reason it cannot be written, each a refusal of ``save``."""

    document: ExperimentDocument | None
    resolved: ResolvedRecord
    entries: tuple[EntryRecord, ...]
    seeds: dict[str, dict[str, int]]
    environment: EnvironmentRecord
    derived: DerivedGraph
    unsavable: tuple[Refusal, ...]

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
            "reference": reference,
        }


def record_of(settings: RunSettings, declarations: Sequence[Declaration], plans: PlanSet,
              derived: DerivedGraph) -> SavedRecord:
    unsavable: list[Refusal] = []
    document: ExperimentDocument | None = None
    try:
        document = document_of(settings, declarations)
    except UnsavableError as error:
        unsavable.append(Refusal("save.unsavable", RefusalClass.USER, str(error),
                                 "write the setting as a number, string, boolean, null, list or quantity"))
    entries, problems = entry_records(p.registered for p in plans.entries)
    unsavable += [Refusal("save.not-loadable", RefusalClass.USER, problem,
                          "register the entry from an installed package with a mime.entries entry point")
                  for problem in problems]
    seeds = {p.owner: dict(sorted(p.seeds.items())) for p in plans.entries if p.seeds}
    return SavedRecord(document, resolved_record(plans), tuple(entries), seeds, environment(), derived,
                       tuple(unsavable))


__all__ = ["SavedRecord", "environment", "input_hash", "record_of", "resolved_record"]
