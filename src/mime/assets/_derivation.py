"""Derived assets and their derivation records (design #181).

An asset's truth comes from one of three places: an *authored* asset (a CAD mesh, a URDF) is true by declaration, a
*measured* one (digitised data, a calibration) is a fact with an uncertainty, and a *derived* one (a wall table, a
surrogate's weights, its training data) is a cached computation, rebuildable and checkable by recomputing. A reference
(``AssetRef``) is the same for all three: bytes are bytes. A derived asset adds a **derivation record**: its builder
and the commit it ran at, its inputs by hash (other assets among them, each under the role it plays), its parameters,
its checks with their results, and its validity domain.

**Staleness is mechanical.** An input is named by the role it plays (``wall:tube``, the table the tube's wall kind
reads) and the hash it had. ``stale_inputs`` compares each with the asset that plays the role now: a surrogate built
through a table whose hash has since changed is stale, which review alone found late (MIME-ANO-117).

**A builder** (``Builder``) produces an asset's bytes and the checks it ran from parameters and inputs; ``derive`` runs
it and writes the record, and refuses (``asset.derivation-check``) when any check failed, so a failed check produces
no asset. A table builder is the simplest builder (an interpolation, no training); a surrogate builder is a sweep, a
fit and a check. Registered builders, ``mime assets build`` and rebuilds are the general version (dev2, after
``Study.sweep``); this interface is what they register.

The records MIME ships are package data, ``mime.data``'s ``derivations/<asset>.json`` (``shipped_derivations``)."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import Protocol, cast

from mime.assets._reference import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass

#: A value a record holds: what JSON writes.
Json = None | bool | int | float | str | list["Json"] | dict[str, "Json"]

#: The folder of ``mime.data`` holding the shipped records, one ``<asset>.json`` each.
DERIVATIONS_FOLDER = "derivations"
#: The record's keys, in the order they are written.
RECORD_KEYS = ("asset", "sha256", "builder", "builder_version", "commit", "inputs", "parameters", "checks", "domain",
               "notes")


@dataclass(frozen=True)
class CheckResult:
    """A check a build ran: its ``name``, the value ``measured`` (``None`` where it is not a number), the
    ``tolerance`` it was held to, as written before the build, and whether it ``passed``."""

    name: str
    measured: float | None
    tolerance: str
    passed: bool

    def as_json(self) -> dict[str, Json]:
        measured = self.measured if self.measured is None or math.isfinite(self.measured) else None
        return {"name": self.name, "measured": measured, "tolerance": self.tolerance, "passed": self.passed}


@dataclass(frozen=True)
class DerivationInput:
    """An input of a derivation: the ``role`` it plays (``wall:tube``), the asset's ``name`` and its ``sha256``."""

    role: str
    name: str
    sha256: str


@dataclass(frozen=True)
class Produced:
    """What a builder produced: the asset's bytes and the checks it ran on them."""

    content: bytes
    checks: tuple[CheckResult, ...]


class Builder(Protocol):
    """What produces a derived asset: its ``name`` and ``version``, and ``produce``, from parameters and the input
    assets by role. The checks it returns are the build's."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def produce(self, parameters: Mapping[str, Json], inputs: Mapping[str, AssetRef]) -> Produced: ...


@dataclass(frozen=True)
class DerivationRecord:
    """How a derived asset was made: the ``asset`` and the ``sha256`` of its bytes; the ``builder``, its
    ``builder_version`` and the ``commit`` it ran at; its ``inputs`` by role and hash; its ``parameters``; its
    ``checks``; its validity ``domain``; and ``notes``, what a reader needs that is none of these."""

    asset: str
    sha256: str
    builder: str
    builder_version: str
    commit: str
    inputs: tuple[DerivationInput, ...]
    parameters: Mapping[str, Json]
    checks: tuple[CheckResult, ...]
    domain: Mapping[str, Json]
    notes: tuple[str, ...] = field(default=())

    def failed(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def input(self, role: str) -> DerivationInput | None:
        """The input playing ``role``, or ``None`` when the derivation has none."""
        found = [each for each in self.inputs if each.role == role]
        return found[0] if found else None

    def as_json(self) -> dict[str, Json]:
        return {"asset": self.asset, "sha256": self.sha256, "builder": self.builder,
                "builder_version": self.builder_version, "commit": self.commit,
                "inputs": [{"role": each.role, "name": each.name, "sha256": each.sha256} for each in self.inputs],
                "parameters": dict(self.parameters), "checks": [check.as_json() for check in self.checks],
                "domain": dict(self.domain), "notes": list(self.notes)}

    def text(self) -> str:
        return json.dumps(self.as_json(), indent=1, sort_keys=False, allow_nan=False) + "\n"

    @staticmethod
    def from_json(data: Mapping[str, object]) -> DerivationRecord:
        """The record ``data`` describes; a key missing or unknown, or a value of another shape, raises ``ValueError``."""
        if tuple(data) != RECORD_KEYS:
            raise ValueError(f"a derivation record's keys are {list(RECORD_KEYS)}, in that order, not {list(data)}")
        inputs = cast(list[Mapping[str, object]], data["inputs"])
        checks = cast(list[Mapping[str, object]], data["checks"])
        for each in inputs:
            if set(each) != {"role", "name", "sha256"}:
                raise ValueError(f"an input's keys are role, name and sha256, not {sorted(each)}")
        for each in checks:
            if set(each) != {"name", "measured", "tolerance", "passed"}:
                raise ValueError(f"a check's keys are name, measured, tolerance and passed, not {sorted(each)}")
        return DerivationRecord(
            str(data["asset"]), str(data["sha256"]), str(data["builder"]), str(data["builder_version"]),
            str(data["commit"]),
            tuple(DerivationInput(str(each["role"]), str(each["name"]), str(each["sha256"])) for each in inputs),
            MappingProxyType(dict(cast(Mapping[str, Json], data["parameters"]))),
            tuple(CheckResult(str(each["name"]), None if each["measured"] is None else float(cast(float, each["measured"])),
                        str(each["tolerance"]), bool(each["passed"])) for each in checks),
            MappingProxyType(dict(cast(Mapping[str, Json], data["domain"]))),
            tuple(str(each) for each in cast(list[object], data["notes"])))


def stale_inputs(record: DerivationRecord, current: Mapping[str, AssetRef]) -> list[str]:
    """Why ``record`` is stale, one sentence per input: each input whose role names, in ``current``, an asset with
    another hash than the one it was built from. A role ``current`` does not name is not judged."""
    found: list[str] = []
    for each in record.inputs:
        now = current.get(each.role)
        if now is not None and now.sha256 != each.sha256:
            found.append(f"{record.asset} was built from {each.name!r} (sha256 {each.sha256[:12]}...) as its "
                         f"{each.role!r}, which is now {now.name!r} (sha256 {now.sha256[:12]}...)")
    return found


def derive(asset: str, builder: Builder, parameters: Mapping[str, Json], inputs: Mapping[str, AssetRef], *,
           commit: str, domain: Mapping[str, Json], notes: Sequence[str] = ()) -> tuple[bytes, DerivationRecord]:
    """Run ``builder`` and return the asset's bytes with its record; refuse (``asset.derivation-check``) when a check
    failed, naming each, so that a failed check produces no asset."""
    produced = builder.produce(parameters, inputs)
    failed = [check for check in produced.checks if not check.passed]
    if failed:
        listed = "; ".join(f"{check.name}: {check.measured!r} against {check.tolerance}" for check in failed[:12])
        raise BuildRefused([Refusal(
            "asset.derivation-check", RefusalClass.USER,
            f"{len(failed)} of the {len(produced.checks)} checks of {asset!r} failed, so it is not produced: {listed}",
            "change the parameters or the inputs that the failed checks name, and build it again")])
    record = DerivationRecord(asset, hashlib.sha256(produced.content).hexdigest(), builder.name, builder.version,
                              commit, tuple(DerivationInput(role, ref.name, ref.sha256)
                                            for role, ref in sorted(inputs.items())),
                              MappingProxyType(dict(parameters)), produced.checks, MappingProxyType(dict(domain)),
                              tuple(notes))
    return produced.content, record


@cache
def shipped_derivations() -> Mapping[str, DerivationRecord]:
    """Every derivation record MIME ships, by the asset it records."""
    found: dict[str, DerivationRecord] = {}
    folder = importlib.resources.files("mime.data") / DERIVATIONS_FOLDER
    for file in sorted(folder.iterdir(), key=lambda entry: entry.name):
        if file.name.endswith(".json"):
            record = DerivationRecord.from_json(json.loads(file.read_text(encoding="utf-8")))
            found[record.asset] = record
    return MappingProxyType(found)


def derivation_of(asset: str) -> DerivationRecord:
    """The shipped record of ``asset``; ``KeyError`` naming those shipped when there is none."""
    held = shipped_derivations().get(asset)
    if held is None:
        raise KeyError(f"no derivation record is shipped for {asset!r}; those shipped are "
                       f"{sorted(shipped_derivations())}")
    return held


__all__ = ["DERIVATIONS_FOLDER", "RECORD_KEYS", "Builder", "CheckResult", "DerivationInput", "DerivationRecord", "Json",
           "Produced", "derivation_of", "derive", "shipped_derivations", "stale_inputs"]
