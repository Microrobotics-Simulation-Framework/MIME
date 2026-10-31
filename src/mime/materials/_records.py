"""Materials an experiment reads from OpenMaterica records, and the record files they come from.

There is no curated library yet, so an experiment names its record files: a file on disk, or a copy MIME ships in
``mime.data``. Each file is pinned by the SHA-256 of its bytes, which the build checks before anything reads it, and
the record inside it by its id, version and content hash, which every read checks. The saved form writes the files
beside the document and holds these pins; the document never holds a value copied out of a record."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import materica

from mime.data import AssetError, PACKAGED, asset, asset_bytes
from mime.units import Quantity
from mime.validity import Basis, basis_problem

if TYPE_CHECKING:
    from materica.core.record import RecordStore

type Problem = tuple[str, str, str]

#: Where the saved form writes each record file, below the document's folder.
RECORDS = "records"


@dataclass(frozen=True)
class RecordFile:
    """A record file the experiment names: a file on disk (``path``), or a copy MIME ships (``packaged``, a name in
    ``mime.data``), pinned by the SHA-256 of its bytes. With ``sha256`` given the build refuses other bytes; without
    it the build pins the file as it finds it, and the saved form records the pin. A packaged copy is always pinned
    by ``mime.data``."""

    path: Path | None = None
    packaged: str | None = None
    sha256: str | None = None

    @property
    def file_name(self) -> str:
        """The file's name, under which the saved form writes it."""
        if self.packaged is not None:
            held = PACKAGED.get(self.packaged)
            return Path(held.file).name if held is not None else f"{self.packaged}.json"
        return self.path.name if self.path is not None else ""

    def describe(self) -> str:
        return f"mime.data's {self.packaged!r}" if self.packaged is not None else str(self.path)

    def read(self) -> bytes:
        """The file's bytes; ``OSError`` or ``mime.data.AssetError`` when they cannot be read as pinned."""
        if self.packaged is not None:
            return asset_bytes(asset(self.packaged))
        if self.path is None:
            raise OSError("a record file names neither a path nor a packaged record")
        return self.path.read_bytes()


def record_file(path: str | os.PathLike[str], *, sha256: str | None = None) -> RecordFile:
    """A record file on disk, pinned by ``sha256`` when given."""
    return RecordFile(path=Path(path), sha256=sha256)


def packaged_record(name: str) -> RecordFile:
    """A copy of a record MIME ships in ``mime.data`` (``openmaterica_example_water``), pinned there."""
    held = PACKAGED.get(name)
    if held is None or not held.file.endswith(".json"):
        records = sorted(n for n, a in PACKAGED.items() if a.file.endswith(".json"))
        raise AssetError(f"no packaged record is named {name!r}; the packaged records are {records}")
    return RecordFile(packaged=name, sha256=held.sha256)


def _frozen[V](mapping: Mapping[str, V] | None) -> Mapping[str, V]:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class Acceptance:
    """The experiment's acceptance of a read OpenMaterica answers ``unknown``: the reasons it covers, each a reason
    code or ``code:condition`` (``condition_unstated:shear_rate``), and why: a waiver, its reason a text or
    ``pending()`` (which exploratory stamps on every output and research refuses by name, as
    ``material.accept-pending``). A reason it does not name is waived only where the profile generates a waiver for
    its class, and otherwise still refuses the build, so a record that changes is never accepted for a reason nobody
    saw."""

    covers: tuple[str, ...]
    reason: Basis

    def __post_init__(self) -> None:
        object.__setattr__(self, "covers", tuple(self.covers))


@dataclass(frozen=True)
class Nominal:
    """The value an experiment uses from a field that holds an interval, with its reason: the build reads the
    bounds and refuses a value outside them. Its reason is a text or ``pending()``, as an acceptance's is (which
    exploratory stamps on every output and research refuses by name, as ``material.nominal-pending``); the value
    carries none of its own."""

    value: Quantity
    reason: Basis


@dataclass(frozen=True)
class Extrapolation:
    """How the experiment extends a table past its readable range, with its basis and reason: ``constant`` keeps the
    end value; ``linear`` continues the table's end segment, computed by MIME from the table the read returns."""

    method: Literal["constant", "linear"]
    basis: str
    reason: str


@dataclass(frozen=True, eq=False)
class Record:
    """A material read from an OpenMaterica record, named ``name`` in the experiment, from ``file``.

    Every read is at the experiment's conditions. ``require`` gives, for token conditions, the tokens the experiment
    accepts in every value read from it (``{"test_method": ("reference_formulation",)}``). By record field
    (``fluid.dynamic_viscosity``): ``accept`` holds the acceptance of a read OpenMaterica answers ``unknown``;
    ``nominal`` the value used from an interval; ``extrapolation`` how a table is extended past its range."""

    name: str
    file: RecordFile
    require: Mapping[str, Sequence[str]] = field(default_factory=dict[str, Sequence[str]])
    accept: Mapping[str, Acceptance] = field(default_factory=dict[str, Acceptance])
    nominal: Mapping[str, Nominal] = field(default_factory=dict[str, Nominal])
    extrapolation: Mapping[str, Extrapolation] = field(default_factory=dict[str, Extrapolation])

    def __post_init__(self) -> None:
        object.__setattr__(self, "require", _frozen({k: tuple(v) for k, v in dict(self.require or {}).items()}))
        object.__setattr__(self, "accept", _frozen(self.accept))
        object.__setattr__(self, "nominal", _frozen(self.nominal))
        object.__setattr__(self, "extrapolation", _frozen(self.extrapolation))


_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: The problem code of a malformed entry of each member, written out so that the refusal scan reads each one.
_MEMBER_CODES = {"accept": "material.accept", "nominal": "material.nominal", "extrapolation": "material.extrapolation"}


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def record_problems(record: Record) -> list[Problem]:
    """Every way ``record``'s declaration is malformed, checked here since a script is not type-checked: its name, its
    file, and each requirement, acceptance, nominal and extrapolation with its reason."""
    found: list[Problem] = []
    held = cast(object, record)
    name, file = cast(object, record.name), cast(object, record.file)
    where = f"material {name!r}"
    if not (isinstance(name, str) and _NAME.fullmatch(name)):
        found.append(("material.name", f"{where}: the name is not an identifier",
                      "name materials with letters, digits and underscores, starting with a letter"))
    if not isinstance(file, RecordFile):
        found.append(("material.file", f"{where}'s file is {file!r}, not a record file",
                      "name it with packaged_record(...) or record_file(...)"))
    for condition, tokens in cast(Mapping[object, object], getattr(held, "require", {})).items():
        if not (isinstance(condition, str) and isinstance(tokens, tuple)
                and all(isinstance(t, str) for t in cast(tuple[object, ...], tokens))):
            found.append(("material.require", f"{where}: require[{condition!r}] = {tokens!r} is not a list of tokens",
                          "write require={\"test_method\": [\"reference_formulation\"]}"))
    entries: list[tuple[str, type, object, object]] = [
        ("accept", Acceptance, key, value) for key, value in cast(Mapping[object, object], record.accept).items()]
    entries += [("nominal", Nominal, key, value) for key, value in cast(Mapping[object, object],
                                                                        record.nominal).items()]
    entries += [("extrapolation", Extrapolation, key, value)
                for key, value in cast(Mapping[object, object], record.extrapolation).items()]
    for member, kind, key, value in entries:
        reason = getattr(value, "reason", None)
        given = basis_problem(reason) is None if isinstance(value, (Acceptance, Nominal)) else _text(reason)
        if not isinstance(value, kind) or not isinstance(key, str) or not given:
            found.append((_MEMBER_CODES[member], f"{where}: {member}[{key!r}] is {value!r}, not a {kind.__name__} "
                                                 f"with its reason, keyed by the record field",
                          f"write {member}={{\"<section>.<field>\": {kind.__name__}(..., reason=\"...\")}}"))
    for key, value in cast(Mapping[object, object], record.accept).items():
        covers = getattr(value, "covers", ())
        if isinstance(value, Acceptance) and not (covers and all(_text(c) for c in cast(tuple[object, ...], covers))):
            found.append(("material.accept", f"{where}: the acceptance of {key!r} covers no reason",
                          "name each reason it covers: a code, or code:condition"))
    for key, value in cast(Mapping[object, object], record.nominal).items():
        if isinstance(value, Nominal) and not isinstance(cast(object, value.value), Quantity):
            found.append(("material.nominal", f"{where}: the nominal of {key!r} is {value.value!r}, not a quantity",
                          "write the nominal with its units, such as T(1.345)"))
    for key, value in cast(Mapping[object, object], record.extrapolation).items():
        if isinstance(value, Extrapolation) and (value.method not in ("constant", "linear") or not _text(value.basis)):
            found.append(("material.extrapolation", f"{where}: the extrapolation of {key!r} is {value.method!r} with "
                                                    f"the basis {value.basis!r}",
                          "write Extrapolation(\"constant\" or \"linear\", basis=\"...\", reason=\"...\")"))
    return found


@dataclass(frozen=True)
class LoadedRecord:
    """A record file as the build read it: the material it is for, its file name and the SHA-256 of its bytes, the
    record's id, version and content hash as the file states them, and the bytes."""

    material: str
    file: str
    sha256: str
    id: str
    version: str
    content_hash: str
    content: bytes

    def pin(self) -> dict[str, str]:
        return {"id": self.id, "version": self.version, "content_hash": self.content_hash}

    def describe(self) -> str:
        return f"{self.id} {self.version} ({self.content_hash[:19]}...)"


def load_record(record: Record) -> tuple[LoadedRecord | None, list[Problem]]:
    """``record``'s file read and checked against its pin, with the identity it states; or every reason not."""
    where = f"material {record.name!r}'s record {record.file.describe()}"
    try:
        content = record.file.read()
    except (OSError, AssetError) as error:
        return None, [("record.file", f"cannot read {where}: {error}",
                       "name a record file that exists, or a packaged record")]
    digest = hashlib.sha256(content).hexdigest()
    if record.file.sha256 is not None and digest != record.file.sha256:
        return None, [("record.changed", f"{where} is pinned at sha256 {record.file.sha256} and its bytes hash to "
                                         f"{digest}: the file changed after it was pinned",
                       "restore the pinned file, or pin the new one with its hash")]
    try:
        data = cast(object, json.loads(content.decode("utf-8")))
    except (UnicodeDecodeError, ValueError) as error:
        return None, [("record.unreadable", f"{where} is not a JSON record: {error}", "name an OpenMaterica record")]
    held = cast(dict[str, object], data) if isinstance(data, dict) else {}
    identity = [held.get(key) for key in ("id", "version", "content_hash")]
    if not all(isinstance(value, str) for value in identity):
        return None, [("record.unreadable", f"{where} states no string id, version and content_hash",
                       "name an OpenMaterica record, which states all three")]
    id_, version, content_hash = (cast(str, value) for value in identity)
    return LoadedRecord(record.name, record.file.file_name, digest, id_, version, content_hash, content), []


def record_store(loaded: Iterable[LoadedRecord]) -> RecordStore:
    """OpenMaterica's store of the records as the build read them: the exact bytes it pinned, handed to OpenMaterica's
    own loader through a folder of this build's own, so no file can change between the pin and the read."""
    with tempfile.TemporaryDirectory(prefix="mime-records-") as folder:
        paths: list[Path] = []
        for index, record in enumerate(loaded):
            path = Path(folder) / f"{index}-{record.file}"
            path.write_bytes(record.content)
            paths.append(path)
        return materica.load_records(paths)


__all__ = ["RECORDS", "Acceptance", "Extrapolation", "LoadedRecord", "Nominal", "Record", "RecordFile",
           "load_record", "packaged_record", "record_file", "record_problems", "record_store"]
