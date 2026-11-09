"""The saved form on disk, and the checks a load and ``verify`` make of it.

A folder holds the document (``experiment.mime.yaml``), the lock (``mime.lock``, JSON), the reference outputs
(``reference.npz``), the scene (``scene/scene.usda``, with the asset layers it references in ``scene/layers/``) and
the record files the experiment's materials are read from (``records/``), the names provisional until the
experiment folder. Reading them runs no code: the document goes through the safe
loader, the lock through ``json``, the arrays through NumPy with pickling off, and the scene's files are hashed, not
opened.

**"The same result" means bit for bit within one environment**: equal on every field of the lock's environment but the
runner's information (the versions, platform, device and precision, and the GPU, the CPU, the matmul precision, XLA's
flags and the runner's record). Across environments a load and ``verify`` report differences within tolerances; they never claim bits.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from packaging.version import InvalidVersion, Version

from mime._lock_json import read_lock
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.build._derived import array_record, sha256_of
from mime.build._refusals import MIME_DEFECT
from mime.materials import RECORDS
from mime.saved._document import DESCRIPTIONS, SCENE_FOLDER, dump_document, parse_yaml
from mime.saved._record import SavedRecord, environment
from mime.saved._scales import OWN, RESOLUTION_FACTOR, SCALED_ATOL, SCALED_FROM, FieldUnits, judge, tolerance_for
from mime.saved._schema import DOCUMENT_SCHEMA, EntryRecord, ExperimentDocument, Json, Lock, ReferenceProgram, \
    ReferenceRecord, conforms
from mime.saved._versions import (
    BASED_ON_ADDED,
    COUNT_KEYS,
    CURRENT,
    DOCUMENT_SCHEMAS,
    DURATION_ADDED,
    LOCK_SCHEMAS,
    NEW_RUNG,
    OLD_RUNG,
    PREVIOUS_PROFILE,
    PROGRAM_ADDED,
    TERMS_ADDED,
    added_after,
    based_on_held,
    duration_held,
    newer_form_refusal,
    physics_in_form,
    program_keys_held,
    resolved_in_form,
    terms_keys_held,
    upgraded_document,
    upgraded_lock,
)
from mime.scene import LAYERS, STAGE_FILE, STAGE_WRITER, STAGE_WRITERS, sha256_hex, stage_derived_findings, stage_edits

#: The resolved record's key for the profile's record, which MIME alone writes.
_PROFILE_RECORD = "profile"

#: The families whose ids a newer MIME may write at a version above this one's.
DOCUMENT_FAMILY, LOCK_FAMILY = "document", "lock"
DOCUMENT_FILE = "experiment.mime.yaml"
LOCK_FILE = "mime.lock"
REFERENCE_FILE = "reference.npz"

#: How many steps the reference outputs cover. The tolerance a save writes is its precision's
#: (``mime.saved._scales.tolerance_for``), over each block's largest magnitude; a lock of the ninth form or earlier holds
#: ``rtol`` 1e-6 and ``atol`` 1e-9, element by element (``PREVIOUS_RTOL``, ``PREVIOUS_ATOL``).
DEFAULT_STEPS = 8
PREVIOUS_RTOL = 1e-6
PREVIOUS_ATOL = 1e-9

_PYTHON_TAG = "tag:yaml.org,2002:python/"

#: A distribution name as PEP 503 (and PEP 508) write one, and the characters a version is made of: both are printed
#: into a ``pip install`` command, so what the lock holds is held to them before anything reads it.
_DISTRIBUTION = re.compile(r"[A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]")
_VERSION_CHARACTERS = re.compile(r"[A-Za-z0-9.+!_-]+")


class LoadRefused(BuildRefused):
    """Every reason a saved experiment could not be loaded, each with what to do about it."""

    headline = "the load is refused"


def history_key(node: str, field_name: str) -> str:
    return f"history/{node}/{field_name}"


def static_key(path: str) -> str:
    return f"static/{path}"


def arrays_hash(arrays: Mapping[str, np.ndarray]) -> str:
    return sha256_of({key: array_record(value) for key, value in sorted(arrays.items())})


def lock_text(lock: Lock | Mapping[str, object]) -> str:
    """The lock as ``save`` writes it, and as ``mime push`` writes the lock it packs into a published bundle."""
    return json.dumps(lock, indent=1, sort_keys=True, allow_nan=False) + "\n"


def write_folder(folder: Path, record: SavedRecord, history: Mapping[str, Mapping[str, np.ndarray]],
                 steps: int, program: ReferenceProgram | None = None,
                 resolution: Mapping[str, Mapping[str, float | None]] | None = None) -> Lock:
    """Write the document, the lock and the reference outputs (``history``, every field over ``steps`` steps,
    with the static arrays beside them, run by ``program``: the history scan of whole control periods where it is
    ``None``) into ``folder``, with each field's resolution by block (``mime.saved._scales.resolution``, by its label
    ``node.field``)."""
    if record.unsavable or record.document is None:
        raise BuildRefused(record.unsavable)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {history_key(node, name): np.asarray(value) for node, fields in history.items()
              for name, value in fields.items()}
    arrays.update({static_key(path): value for path, value in record.derived.static.items()})
    np.savez(folder / REFERENCE_FILE, allow_pickle=False, **arrays)
    reference: ReferenceRecord = {"file": REFERENCE_FILE, "sha256": arrays_hash(arrays), "steps": steps,
                                  "rtol": tolerance_for(environment()["x64"]), "atol": SCALED_ATOL}
    if program is not None:
        reference = {**reference, "program": program}
    if resolution is not None:
        reference = {**reference, "resolution": {label: dict(sorted(held.items()))
                                                 for label, held in sorted(resolution.items()) if held},
                     "resolution_factor": RESOLUTION_FACTOR}
    lock = record.lock(reference)
    stage = lock["stage"]
    (folder / stage["file"]).parent.mkdir(parents=True, exist_ok=True)
    (folder / stage["file"]).write_bytes(record.stage.encode("utf-8"))
    for layer in record.layers:
        target = folder / SCENE_FOLDER / layer.file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(layer.content)
    for held in record.records:
        target = folder / RECORDS / held.file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(held.content)
    for path, content in record.files:
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (folder / DOCUMENT_FILE).write_text(dump_document(record.document), encoding="utf-8", newline="\n")
    (folder / LOCK_FILE).write_text(lock_text(lock), encoding="utf-8", newline="\n")
    return lock


#: What to do about terms a document holds in a form ``save`` never writes: it was edited after it was saved.
_EDITED_TERMS_ACTION = ("save the experiment again from its script, built.save(folder), or put the document's terms "
                        "back as experiment.save() wrote them: a document is changed by its script, not by hand")


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


def _newer(data: object, family: str, reads: tuple[str, ...], path: Path) -> Refusal | None:
    """``load.newer-form`` when the file ``data`` was read from states a form of ``family`` above the newest of
    ``reads``."""
    schema = cast("dict[str, object]", data).get("schema") if isinstance(data, dict) else None
    return newer_form_refusal(schema, family, reads, str(path))


def read_document(path: Path) -> tuple[ExperimentDocument | None, list[Refusal]]:
    """The document at ``path``, checked against its schema, or every reason it is refused. A document of an earlier
    schema is read under the current one (``mime.saved._versions``)."""
    document, refusals, _, _ = read_document_version(path)
    return document, refusals


def read_document_version(path: Path) -> tuple[ExperimentDocument | None, list[Refusal], int, tuple[str, ...]]:
    """As :func:`read_document`, with the version the document was written in and the ids of the drags that named the
    rung by its old name, which a load reads as naming ``mrs``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        return None, [_refuse("load.missing-file", RefusalClass.USER, f"cannot read the document {path}: {error}",
                              f"pass a folder holding {DOCUMENT_FILE}, as built.save(folder) writes it")], CURRENT, ()
    except UnicodeDecodeError as error:
        return None, [_refuse("load.document-encoding", RefusalClass.USER,
                              f"{path} is not text in UTF-8: byte {error.start} cannot be read ({error.reason})",
                              "save the document as UTF-8 text in your editor, or save the experiment again: "
                              "built.save(folder)")], CURRENT, ()
    try:
        data = parse_yaml(text)
    except yaml.YAMLError as error:
        if _PYTHON_TAG in str(error):
            return None, [_refuse("load.code-in-document", RefusalClass.USER,
                                  f"{path} holds a Python tag: a document never holds code, and loading constructs "
                                  f"no Python object a tag names ({error})",
                                  "remove the tag: write settings as numbers, strings, booleans, lists and quantities, "
                                  "or save the experiment again with built.save(folder)")], CURRENT, ()
        return None, [_refuse("load.document-yaml", RefusalClass.USER, f"{path} is not a readable document: {error}",
                              "fix the YAML, or save the experiment again: built.save(folder)")], CURRENT, ()
    newer = _newer(data, DOCUMENT_FAMILY, tuple(DOCUMENT_SCHEMAS.values()), path)
    if newer is not None:
        return None, [newer], CURRENT, ()
    upgraded, version, renamed = upgraded_document(data)
    document, problems = conforms(upgraded, ExperimentDocument)
    added = added_after(data, version)
    terms = [key for key in added if key in terms_keys_held(data)]
    if terms:
        problems = [f"a {DOCUMENT_SCHEMAS[version]} document holds {', '.join(terms)}, which version {TERMS_ADDED} of "
                    f"the saved form added: a document of an earlier version declares no licence, terms or permission",
                    *problems]
    if duration_held(data) and version < DURATION_ADDED:
        problems = [f"a {DOCUMENT_SCHEMAS[version]} document holds run.duration, which version {DURATION_ADDED} of the "
                    f"saved form added: a document of an earlier version declares no run length", *problems]
    if based_on_held(data) and version < BASED_ON_ADDED:
        problems = [f"a {DOCUMENT_SCHEMAS[version]} document holds based_on, which version {BASED_ON_ADDED} of the "
                    f"saved form added: a document of an earlier version was made from no other experiment", *problems]
    return document, [_refuse("load.document-shape", RefusalClass.USER, f"{path}: {problem}",
                              "write the document as experiment.save() writes it")
                      for problem in problems], version, renamed


@dataclass(frozen=True)
class SavedForm:
    """A saved form as read: its folder, document, lock and reference outputs, and the stage's text where its bytes
    are not the ones the lock pins (``None`` where they are), which the load compares with the stage its rebuild
    writes."""

    folder: Path
    document: ExperimentDocument
    lock: Lock
    reference: dict[str, np.ndarray]
    stage_found: str | None = None
    #: The version of the saved form the folder was written in, whose lock a load compares in that form, and the ids of
    #: the drags that named the rung by its old name, which the document is read as naming ``mrs``.
    version: int = CURRENT
    renamed: tuple[str, ...] = ()
    #: What reading the document's terms found to say, never to refuse: a licence naming an id the installed packaging
    #: does not list, read as written (``mime.saved._terms.read_check``).
    notes: tuple[str, ...] = ()

    @property
    def previous(self) -> bool:
        return self.version != CURRENT

    @property
    def predates_terms(self) -> bool:
        """Whether the form is one before licences and terms, which declares none."""
        return self.version < TERMS_ADDED


def layout_problem(name: str) -> str | None:
    """Why ``name``, a path below the folder written as ``/`` writes it, is not where a saved folder keeps a file, or
    ``None``: the document, the lock, the reference outputs, ``scene/scene.usda`` with its layers in
    ``scene/layers/``, and the files of ``records/`` and ``descriptions/``, each one level down. Nothing else is
    carried, so a bundle cannot bring a ``.git`` directory, an ``.env`` or an editor's file, and a pack cannot
    take one."""
    parts = name.split("/")
    held = (name in (DOCUMENT_FILE, LOCK_FILE, REFERENCE_FILE, STAGE_FILE)
            or (len(parts) == 3 and parts[:2] == [SCENE_FOLDER, LAYERS])
            or (len(parts) == 2 and parts[0] in (RECORDS, DESCRIPTIONS)))
    return None if held else (f"is not a file a saved experiment holds ({DOCUMENT_FILE}, {LOCK_FILE}, "
                              f"{REFERENCE_FILE}, {STAGE_FILE}, {SCENE_FOLDER}/{LAYERS}/*, {RECORDS}/* and "
                              f"{DESCRIPTIONS}/*)")


def _file_values(value: object, found: set[str]) -> None:
    """Every ``file`` the document or lock names below ``descriptions/``."""
    if isinstance(value, dict):
        for key, each in cast("dict[object, object]", value).items():
            if key == "file" and isinstance(each, str) and each.startswith(f"{DESCRIPTIONS}/"):
                found.add(each)
            else:
                _file_values(each, found)
    elif isinstance(value, list):
        for each in cast("list[object]", value):
            _file_values(each, found)


def named_files(form: SavedForm) -> set[str]:
    """Every file below the folder the saved form names: the document and the lock, the lock's reference outputs, stage,
    layers and record files, and the description files the document names. A folder holds these and nothing else."""
    lock = form.lock
    described: set[str] = set()
    _file_values(form.document, described)
    _file_values(lock["resolved"]["descriptions"], described)
    return {DOCUMENT_FILE, LOCK_FILE, lock["reference"]["file"], lock["stage"]["file"], *lock["stage"]["layers"],
            *(held["file"] for held in lock["records"].values()), *described}


def lock_text_problems(lock: Lock) -> list[str]:
    """What in the lock is not safe to read or to print: a file it names that is not where a saved folder keeps one
    (a path to somewhere else), a distribution name that is not a PEP 503 name, a version that is not PEP 440. Each as
    a sentence; the offending text is quoted with ``repr``, which escapes control characters."""
    problems: list[str] = []
    named = [lock["reference"]["file"], lock["stage"]["file"], *lock["stage"]["layers"],
             *(held["file"] for held in lock["records"].values()), *lock.get("referenced_files", {})]
    for file in named:
        why = layout_problem(file)
        if why is not None:
            problems.append(f"the lock names the file {file!r}, which {why}")
    asset_names = {held["name"] for held in lock.get("assets", [])}
    for file, name in sorted(lock.get("referenced_files", {}).items()):
        if file in (DOCUMENT_FILE, LOCK_FILE, lock["reference"]["file"], lock["stage"]["file"]):
            problems.append(f"the lock says {file!r} travels as a reference, and MIME writes that file")
        if name not in asset_names:
            problems.append(f"the lock says {file!r} travels as the asset {name!r}, which its assets do not name")
    for entry in lock["entries"]:
        name, version = entry["distribution"], entry["distribution_version"]
        if not _DISTRIBUTION.fullmatch(name):
            problems.append(f"the distribution {name!r} is not a name PEP 503 allows (letters, digits, '.', '_' and '-')")
        try:
            Version(version)
            valid = _VERSION_CHARACTERS.fullmatch(version) is not None
        except InvalidVersion:
            valid = False
        if not valid:
            problems.append(f"the version {version!r} of {name!r} is not a version PEP 440 allows")
    return problems


def read_folder(folder: Path, *, absent_referenced: bool = False) -> SavedForm:
    """The saved form in ``folder``, every file checked; raises :class:`LoadRefused` with every reason. A file the
    lock says travels as a reference and that is not in the folder is refused as ``asset.missing``, with the command
    that fetches and places it, unless ``absent_referenced`` (an unpacked bundle, before its references are placed);
    one that is there is held to its pin as every file is."""
    from mime.saved._terms import read_check

    document, refusals, version, renamed = read_document_version(folder / DOCUMENT_FILE)
    notes: tuple[str, ...] = ()
    if document is not None and version >= TERMS_ADDED:
        found, notes = read_check(document)
        refusals += [replace(each, message=f"{folder / DOCUMENT_FILE}: {each.message}", action=_EDITED_TERMS_ACTION)
                     for each in found]
    lock: Lock | None = None
    try:
        data: object = read_lock(folder / LOCK_FILE)
    except (OSError, ValueError) as error:
        refusals.append(_refuse("load.lock", RefusalClass.USER, f"cannot read the lock {folder / LOCK_FILE}: {error}",
                                "save the experiment again, built.save(folder): the document is checked against the "
                                "lock its build "
                                "wrote"))
    else:
        newer = _newer(data, LOCK_FAMILY, tuple(LOCK_SCHEMAS.values()), folder / LOCK_FILE)
        if newer is not None:
            refusals.append(newer)
        else:
            data, lock_version = upgraded_lock(data)
            if document is not None and lock_version != version:
                refusals.append(_refuse("load.lock-shape", RefusalClass.USER,
                                        f"{folder}: the document and the lock are of different versions of the saved "
                                        f"form (the document is version {version}, the lock version {lock_version})",
                                        "save the experiment again from your script, built.save(folder), which writes "
                                        "both"))
            lock, problems = conforms(data, Lock)
            added = terms_keys_held(data) if lock_version < TERMS_ADDED else []
            if added:
                problems = [f"a {LOCK_SCHEMAS[lock_version]} lock holds {', '.join(added)}, which version "
                            f"{TERMS_ADDED} of the saved form added", *problems]
            added = program_keys_held(data) if lock_version < PROGRAM_ADDED else []
            if added:
                problems = [f"a {LOCK_SCHEMAS[lock_version]} lock holds {', '.join(added)}, which version "
                            f"{PROGRAM_ADDED} of the saved form added: a lock of an earlier version names no program "
                            f"and records no count of CPUs", *problems]
            if lock is not None:
                problems = lock_text_problems(lock)
                lock = lock if not problems else None
            refusals += [_refuse("load.lock-shape", RefusalClass.USER, f"{folder / LOCK_FILE}: {problem}",
                                 "save the experiment again from your script, built.save(folder), which writes a new "
                                 "lock; this folder cannot be loaded to do it, its lock being the problem")
                         for problem in problems]
    reference: dict[str, np.ndarray] = {}
    if lock is not None:
        try:
            with np.load(folder / lock["reference"]["file"], allow_pickle=False) as held:
                reference = {key: np.asarray(held[key]) for key in held.files}
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:  # a missing, truncated or corrupt file
            refusals.append(_refuse("load.reference", RefusalClass.USER,
                                    f"cannot read the reference outputs {lock['reference']['file']}: {error}",
                                    "save the experiment again: Experiment.load(...) or your script, then "
                                    "built.save(folder)"))
        else:
            if arrays_hash(reference) != lock["reference"]["sha256"]:
                refusals.append(_refuse("load.reference-changed", RefusalClass.USER,
                                        f"the reference outputs in {lock['reference']['file']} are not the ones "
                                        f"the lock records", "save the experiment again: Experiment.load(...) or your "
                                                             "script, then built.save(folder)"))
    stage_found: str | None = None
    if lock is not None:
        absent = absent_referenced_files(folder, lock)
        if not absent_referenced:
            refusals += [missing_referenced(folder, lock, file) for file in absent]
        found_refusals, stage_found = scene_file_refusals(folder, lock, skip=frozenset(absent))
        refusals += found_refusals
        refusals += record_file_refusals(folder, lock, skip=frozenset(absent))
    if refusals or document is None or lock is None:
        raise LoadRefused(refusals)
    return SavedForm(folder, document, lock, reference, stage_found, version, renamed, notes)


def absent_referenced_files(folder: Path, lock: Lock) -> list[str]:
    """The files the lock says travel as references that are not in ``folder``, sorted."""
    return sorted(file for file in lock.get("referenced_files", {}) if not (folder / file).is_file())


def missing_referenced(folder: Path, lock: Lock, file: str) -> Refusal:
    """``asset.missing`` for a file that travels as a reference and is not in the folder: the command that fetches it
    into the cache and places it."""
    from mime.assets import size_text
    from mime.assets._commands import fetch_command

    name = lock.get("referenced_files", {})[file]
    held = next((each for each in lock.get("assets", []) if each["name"] == name), None)
    size = f" ({size_text(held['size'])})" if held is not None else ""
    return _refuse("asset.missing", RefusalClass.ENVIRONMENT,
                   f"the experiment needs {file}, the asset {name!r}{size}, which travels as a reference and is not "
                   f"in the folder", f"Run: {fetch_command(folder=folder)}")


def scene_file_refusals(folder: Path, lock: Lock, *, skip: frozenset[str] = frozenset()) -> tuple[list[Refusal],
                                                                                                  str | None]:
    """The stage and every layer it references, each held to the hash the lock records: an asset file changed
    after its lock was written is refused, and so is a file that cannot be read. A stage whose bytes changed is
    returned as text, for the load to compare with the stage the rebuild writes (``stage_change_refusals``): the build
    writes the stage from the document, so an edit is judged against what it writes. A layer in ``skip`` (one that
    travels as a reference, not yet placed) is not read."""
    stage = lock["stage"]
    refusals: list[Refusal] = []
    stage_found: str | None = None
    files = [(stage["file"], stage["sha256"], "load.stage-changed"),
             *((layer, digest, "load.asset-changed") for layer, digest in sorted(stage["layers"].items())
               if layer not in skip)]
    for file, digest, code in files:
        try:
            content = (folder / file).read_bytes()
        except OSError as error:
            refusals.append(_refuse(code, RefusalClass.USER, f"cannot read {file}: {error}",
                                    "save the experiment again, or restore the file"))
            continue
        found = sha256_hex(content)
        if found == digest:
            continue
        if code == "load.stage-changed":
            stage_found = content.decode("utf-8", errors="replace")
            continue
        refusals.append(_refuse(code, RefusalClass.USER,
                                f"the asset layer {file} changed after its lock was written: sha256 {found}, and "
                                f"the lock records {digest}",
                                "restore the file the lock pins, or build and save the experiment again"))
    return refusals, stage_found


def stage_writers(lock: Lock) -> tuple[int, ...]:
    """The stage writers a lock's stage may have been written by: the one it records, or, for a lock written before
    the writer was recorded, the first writer and the current one (the current writer wrote stages before it recorded
    itself, on a development line)."""
    held = lock["stage"].get("writer")
    return (held,) if held is not None else (1, STAGE_WRITER)


def stage_change_refusals(form: SavedForm, record: SavedRecord, compare: bool,
                          rebuilt: str | None = None) -> tuple[list[Refusal], list[str]]:
    """A saved stage whose bytes are not the ones its lock pins, judged against the stage the rebuild writes from the
    same document (``compare``; without it, as after a changed document, any change is refused), by the stage writer
    that wrote it (``rebuilt``; the record's own stage when not given). An edit confined to
    derived prims (a kind's drawn mesh, a region's primitive) is reported, one line each: they are output only, read
    by nothing, and every build writes them again; every profile of 0.3.0 reports it, and ``strict`` may refuse such
    an edit here. Any other edit is refused."""
    found = form.stage_found
    if found is None:
        return [], []
    stage = form.lock["stage"]
    refused = [_refuse("load.stage-changed", RefusalClass.USER,
                       f"the stage {stage['file']} changed after its lock was written: sha256 "
                       f"{sha256_hex(found.encode('utf-8'))}, and the lock records {stage['sha256']}",
                       "restore the file the lock pins, or build and save the experiment again: built.save(folder)")]
    if not compare:
        return refused, []
    try:
        edits = stage_edits(found, record.stage if rebuilt is None else rebuilt)
    except Exception:  # noqa: BLE001 - a stage USD cannot read is a changed stage, refused as one
        return refused, []
    if edits.outside:
        return refused, []
    findings = {finding.path: finding.message for finding in stage_derived_findings(found)}
    return [], [f"{findings.get(path, f'{path} differs from the derived prim the build writes')}; it is output "
                f"only, and the build writes it again" for path in edits.derived]


def record_file_refusals(folder: Path, lock: Lock, *, skip: frozenset[str] = frozenset()) -> list[Refusal]:
    """Every record file the lock pins, held to its hash: a record edited after its lock was written, by so much as
    one byte, is refused. A file in ``skip`` (one that travels as a reference, not yet placed) is not read."""
    refusals: list[Refusal] = []
    for material, held in sorted(lock["records"].items()):
        if held["file"] in skip:
            continue
        try:
            found = sha256_hex((folder / held["file"]).read_bytes())
        except OSError as error:
            refusals.append(_refuse("load.record-changed", RefusalClass.USER,
                                    f"cannot read material {material!r}'s record {held['file']}: {error}",
                                    "save the experiment again, built.save(folder), or restore the file"))
            continue
        if found != held["sha256"]:
            refusals.append(_refuse("load.record-changed", RefusalClass.USER,
                                    f"material {material!r}'s record {held['file']} changed after its lock was "
                                    f"written: sha256 {found}, and the lock records {held['sha256']}",
                                    "restore the record the lock pins, or build and save the experiment again: "
                                    "built.save(folder)"))
    return refusals


def _short(value: object) -> str:
    return json.dumps(value, default=repr)[:120]


def differences(saved: object, found: object, path: str = "", limit: int = 12) -> list[str]:
    """Where two JSON values differ, each as ``path: saved -> found``, at most ``limit`` of them."""
    out: list[str] = []

    def walk(a: object, b: object, at: str) -> None:
        left, right = a, b
        if len(out) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            held_a, held_b = cast(dict[str, object], a), cast(dict[str, object], b)
            for key in sorted(set(held_a) | set(held_b)):
                walk(held_a.get(key, "(absent)"), held_b.get(key, "(absent)"), f"{at}.{key}" if at else key)
        elif isinstance(a, list) and isinstance(b, list) and len(cast(list[object], a)) == len(cast(list[object], b)):
            for i, (x, y) in enumerate(zip(cast(list[object], a), cast(list[object], b), strict=True)):
                walk(x, y, f"{at}[{i}]")
        elif left != right or type(left) is not type(right):
            out.append(f"{at or '(root)'}: {_short(left)} -> {_short(right)}")

    walk(saved, found, path)
    return out


def _number(value: Json) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers_beyond(saved: Json, found: Json, rtol: float, atol: float, at: str, out: list[str], *,
                    scaled: bool = False) -> None:
    """Each number of the derived graph's physics beyond the tolerance: element by element (a form before the tenth),
    or, ``scaled``, each list of numbers against its own largest magnitude and each number against its own."""
    if isinstance(saved, dict) and isinstance(found, dict):
        for key in sorted(set(saved) & set(found)):
            _numbers_beyond(saved[key], found[key], rtol, atol, f"{at}.{key}", out, scaled=scaled)
    elif isinstance(saved, list) and isinstance(found, list) and len(saved) == len(found):
        if scaled and saved and all(_number(a) for a in saved) and all(_number(b) for b in found):
            held, rebuilt = np.asarray(saved, dtype=np.float64), np.asarray(found, dtype=np.float64)
            if not judge(held, rebuilt, OWN, rtol, atol, steps=False).within:
                out.append(f"{at}: max |difference| {float(np.max(np.abs(held - rebuilt))):.3e}")
            return
        for i, (a, b) in enumerate(zip(saved, found, strict=True)):
            _numbers_beyond(a, b, rtol, atol, f"{at}[{i}]", out, scaled=scaled)
    elif (_number(saved) and _number(found)
          and not math.isclose(cast(float, saved), cast(float, found), rel_tol=rtol, abs_tol=atol)):
        out.append(f"{at}: {saved!r} -> {found!r}")


def _distribution_versions(entries: Sequence[EntryRecord]) -> list[str]:
    out: list[str] = []
    for record in entries:
        try:
            installed = importlib.metadata.version(record["distribution"])
        except importlib.metadata.PackageNotFoundError:
            installed = "not installed"
        if installed != record["distribution_version"]:
            out.append(f"{record['distribution']}: built under {record['distribution_version']}, loaded under "
                       f"{installed}")
    return sorted(set(out))


#: The environment's fields a load only stores: a runner's information does not decide a result.
INFORMATION_ONLY = frozenset({"runner_info"})
#: The environment's records of a device, whose fields are read from the system and are ``None`` where it did not say.
READ_FROM_THE_SYSTEM = frozenset({"gpu", "cpu"})


def _differences_of(key: str, saved: object, here: object) -> list[str]:
    """How one field of the environment differs. A device's field either side could not read is never equal to
    anything, ``None`` included: two reads that failed are no evidence of one device."""
    if key in READ_FROM_THE_SYSTEM and isinstance(saved, dict) and isinstance(here, dict):
        held = cast(dict[str, object], saved)
        now = cast(dict[str, object], here)
        return [f"{key}.{name}: built under {held[name]!r}, loaded under {now[name]!r}"
                + (" (unreadable, so not the same)" if held[name] is None or now[name] is None else "")
                for name in sorted(now) if held.get(name) is None or now[name] is None or held[name] != now[name]]
    if key in COUNT_KEYS and (saved is None or here is None):
        return [f"{key}: built under {saved!r}, loaded under {here!r} (unreadable, so not the same)"]
    if saved != here:
        return [f"{key}: built under {saved!r}, loaded under {here!r}"]
    return []


def environment_differences(lock: Lock) -> list[str]:
    """Every way this process differs from the one the lock was written in, the runner's information aside. A lock of a
    form before version 8 records neither count (of usable CPUs, of BLAS threads), and is compared without them, as it
    was. A count either side could not read is never equal to anything, as a device's field is not."""
    here = cast(dict[str, object], environment())
    saved = cast(dict[str, object], lock["environment"])
    found = [line for key in sorted(here) if key not in INFORMATION_ONLY and (key in saved or key not in COUNT_KEYS)
             for line in _differences_of(key, saved[key], here[key])]
    return found + _distribution_versions(lock["entries"])


@dataclass(frozen=True)
class LoadReport:
    """What a load found: whether this is the environment the lock was written in, every difference from it, and
    how the rebuilt graph compares with the saved one."""

    folder: str
    same_environment: bool
    environment: tuple[str, ...]
    derived: str
    derived_differences: tuple[str, ...] = ()
    derived_prims_edited: tuple[str, ...] = ()
    #: The saved form's version (``mime.document/11``), and the profile the experiment is read under.
    schema: str = DOCUMENT_SCHEMA
    profile: str = PREVIOUS_PROFILE
    version: int = CURRENT
    #: The ids of the drags that named the rung by its old name, which the document is read as naming ``mrs``.
    renamed: tuple[str, ...] = ()
    #: What reading the folder found to say and not to refuse (``SavedForm.notes``).
    notes: tuple[str, ...] = ()
    #: The stage writer whose stage the folder holds, held to its lock (``mime.scene.STAGE_WRITER`` or an earlier one).
    stage_writer: int = STAGE_WRITER

    @property
    def previous(self) -> bool:
        return self.version != CURRENT

    @property
    def predates_terms(self) -> bool:
        """Whether the form is one before licences and terms, which declares none."""
        return self.version < TERMS_ADDED

    def form_line(self) -> str:
        """The saved form's version and the profile it is read under, with why for a previous form."""
        if not self.previous:
            return f"saved form: {self.schema}, built and read under the {self.profile} profile"
        if self.version == 1:
            line = (f"saved form: {DOCUMENT_SCHEMAS[1]}, the form before profiles, read under the "
                    f"{self.profile} profile: it was built before profiles, and {PREVIOUS_PROFILE} accepts every "
                    f"build that form accepted, stamping what is pending (mime pending lists it)")
        else:
            line = f"saved form: {DOCUMENT_SCHEMAS[self.version]}, read as version {CURRENT}"
        if self.renamed:
            line += (f"; the rung {OLD_RUNG} of {', '.join(self.renamed)} is read as {NEW_RUNG}: the rung was renamed "
                     f"for its method, and is the same")
        if self.predates_terms:
            line += "; it declares no licence, terms or permission, being of a form before them"
        return line

    def text(self) -> str:
        lines = [f"load report: {self.folder}", self.form_line(), "input hash: reproduced exactly", *self.notes]
        if self.stage_writer != STAGE_WRITER:
            lines.append(f"stage: written by stage writer {self.stage_writer}, rebuilt by it and held to its lock; "
                         f"saving the experiment again writes stage writer {STAGE_WRITER}'s, which draws every chain "
                         f"frame, visual and vector a live session or a recording moves")
        if self.derived_prims_edited:
            lines.append("stage: derived prims edited after it was written (reported, not refused)")
            lines += [f"  {line}" for line in self.derived_prims_edited]
        if self.same_environment:
            lines.append("environment: the one the lock was written in")
        else:
            lines.append("environment: built under another; results are not guaranteed to match: run verify")
            lines += [f"  {line}" for line in self.environment]
        lines.append(f"derived graph: {self.derived}")
        lines += [f"  {line}" for line in self.derived_differences]
        return "\n".join(lines)


def _is_development_release(version: str) -> bool:
    try:
        return Version(version).is_devrelease
    except InvalidVersion:
        return False


def other_release_refusal(form: SavedForm, running: str | None = None) -> Refusal | None:
    """``load.other-release`` when the MIME that saved ``form`` is another version than the one running and either is
    a development release, which makes no compatibility promise (design #232): MIME's version is in the input hash, so
    the hash check would refuse the folder too, and say the document was changed. This is that refusal in its own words,
    both versions and what to do; a lock that records a publication is a folder unpacked from a published record
    (``mime pull``), and the refusal says so and names the record's page. ``None`` where the versions agree, or
    neither is a development release."""
    saved = form.lock["environment"]["mime"]
    current = running if running is not None else environment()["mime"]
    if saved == current or not (_is_development_release(saved) or _is_development_release(current)):
        return None
    folder = str(form.folder)
    publication = form.lock.get("publication")
    message = (f"the folder {folder} was saved by MIME {saved}, and this is MIME {current}: development releases do "
               "not load each other's folders (MIME's version is part of a saved experiment's input hash, and a "
               "development release makes no promise that a saved form reads in the next)")
    if publication is None:
        action = (f"run the script that built the experiment under this MIME {current} and save it again, "
                  f"`built.save({folder!r})`; or load the folder as it is under the MIME that saved it, "
                  f"`pip install mime-engine=={saved}`")
    else:
        from mime.push._service import SERVICES

        service = SERVICES.get(publication["service"])
        page = f"{service.site}/records/{publication['record']}" if service is not None else \
            f"the record {publication['record']} of {publication['service']}"
        message += (f"; it came from `mime pull`, the published record {publication['doi']} (version "
                    f"{publication['version']})")
        action = (f"the record's page is {page}: ask its author for a record saved by MIME {current}, or load it as it "
                  f"is under the MIME that saved it, `pip install mime-engine=={saved}`; with the script that built it, "
                  f"run the script under this MIME and save it again, `built.save({folder!r})`")
    return _refuse("load.other-release", RefusalClass.USER, message, action)


#: Whether ``Experiment.based_on`` exists (U15, which it does): a loaded experiment that was edited cannot be built or
#: saved, and the action names the call that keeps the edit in a new experiment; without it, what else works.
_BASED_ON_EXISTS = True


def _edited_after_load_action(folder: Path) -> str:
    """What to do with a loaded experiment that was edited: only what works. A loaded experiment is checked against its
    lock at every build, so the edited one can be neither built nor saved (a save builds first)."""
    if _BASED_ON_EXISTS:
        return (f"to keep the edit, start a new experiment from the folder, Experiment.based_on({str(folder)!r}, "
                f"name=\"<a new name>\"), make the edit there and save it as a new folder; to drop it, load the folder "
                f"again, Experiment.load({str(folder)!r})")
    return (f"this experiment cannot be built or saved while it differs from its lock. To drop the edit, load the "
            f"folder again, Experiment.load({str(folder)!r}). To keep it, make the same edit in the script that built "
            f"the experiment and save from there, built.save(\"<new folder>\")")


def check_load(form: SavedForm, record: SavedRecord, stages: Mapping[int, str] | None = None, *,
               edited_after_load: bool = False) -> LoadReport:
    """The three load checks, against the rebuilt experiment's ``record``. A folder saved by another development
    release is refused first, as ``load.other-release`` (:func:`other_release_refusal`), before the hash that would
    refuse it for the version in it. ``edited_after_load`` is true for an experiment that was loaded and has been
    built since, where a document that differs is one edited after it was loaded. The input hash must match exactly; the
    environment is compared and reported; the derived graph must match exactly within one environment, and
    otherwise is compared within the lock's tolerance and reported. A stage edited after its lock was written is
    refused, or reported where only its derived prims were edited (:func:`stage_change_refusals`). ``stages`` is the
    stage as each writer the lock's stage may be of (:func:`stage_writers`) rebuilds it; the rebuilt stage must be one
    of them, byte for byte, within one environment, and a stage of a writer this MIME does not have is refused.
    Raises :class:`LoadRefused`."""
    other_release = other_release_refusal(form)
    if other_release is not None:
        raise LoadRefused([other_release])
    lock = form.lock
    stages = dict(stages) if stages else {STAGE_WRITER: record.stage}
    writer = next((held for held, text in sorted(stages.items()) if sha256_hex(text.encode("utf-8")) ==
                   lock["stage"]["sha256"]), None)
    rebuilt = stages[writer] if writer is not None else stages[min(stages)]
    if record.document is None:
        # the rebuilt experiment's document cannot be written (terms the build judges as save would): every reason,
        # each by its code
        raise LoadRefused(list(record.unsavable))
    saved_hashes, hashes = lock["hashes"], record.hashes(version=form.version)
    refusals, derived_prims = stage_change_refusals(form, record, hashes["input"] == saved_hashes["input"], rebuilt)
    named = lock["stage"].get("writer")
    if named is not None and named not in STAGE_WRITERS:
        refusals.append(_refuse("load.stage-writer", RefusalClass.ENVIRONMENT,
                                f"the stage was written by stage writer {named}, and this MIME writes versions "
                                f"{', '.join(map(str, STAGE_WRITERS))}",
                                f"install the MIME that saved it, `pip install "
                                f"mime-engine=={lock['environment']['mime']}`, or build the experiment from its script "
                                "and save it again: `built.save(folder)`"))
    if hashes["input"] != saved_hashes["input"]:
        edited = hashes["document"] != saved_hashes["document"]
        rebuilt = resolved_in_form(record.resolved, form.version)
        saved = resolved_in_form(lock["resolved"], form.version)
        found = differences(saved, rebuilt, "resolved")
        found += differences(sorted(f"{e['kind']}:{e['name']}@{e['version']}" for e in lock["entries"]),
                             sorted(f"{e['kind']}:{e['name']}@{e['version']}" for e in record.entries), "entries")
        detail = "; ".join(found) or "the document itself"
        own = bool(found) and all(line.startswith(f"resolved.{_PROFILE_RECORD}") for line in found) and \
            lock["environment"]["mime"] == environment()["mime"]
        if edited and edited_after_load:
            from mime._notebook import WHERE, in_notebook

            refusals.append(_refuse("load.input-hash", RefusalClass.USER,
                                    f"this experiment was loaded from {form.folder} and edited after it was loaded, so "
                                    f"it no longer reproduces the lock it was loaded with: {detail}"
                                    + (f" ({WHERE}: a line or cell run after the load changed it)"
                                       if in_notebook() else ""),
                                    _edited_after_load_action(form.folder)))
        elif edited:
            refusals.append(_refuse("load.input-hash", RefusalClass.USER,
                                    f"the document was changed after its lock was written: {detail}",
                                    "build the experiment and save it again, built.save(folder), which writes a new "
                                    "lock; to build the "
                                    "edited document as it is, "
                                    "Experiment.from_document(\"<folder>/experiment.mime.yaml\")"))
        elif own:
            # the profile's record is MIME's own, under the MIME that wrote the lock: a difference is MIME's defect,
            # never the installed entries'
            refusals.append(_refuse("load.input-hash", RefusalClass.INTERNAL,
                                    f"MIME rebuilt its own record of the profile differently from the build that wrote "
                                    f"the lock, under the same MIME version: {detail}",
                                    f"report it: {MIME_DEFECT}"))
        else:
            refusals.append(_refuse("load.input-hash", RefusalClass.ENVIRONMENT,
                                    f"the installed entries resolve this document differently from the build that "
                                    f"wrote the lock, under the same names and versions: {detail}",
                                    "install the versions the lock names (`pip install <package>==<version>` for each "
                                    "of its entries), or report the "
                                    "entry that changed what it resolves without a new version"))
    refusals += asset_refusals(lock, record)
    env = environment_differences(lock)
    same = not env
    derived_differences: list[str] = []
    if refusals:
        # Another input builds another graph; the derived hash guards the same input building something else.
        raise LoadRefused(refusals)
    if writer is None:
        tried = ", ".join(f"{sha256_hex(text.encode('utf-8'))} by stage writer {held}"
                          for held, text in sorted(stages.items()))
        if same:
            refusals.append(_refuse("load.stage-hash", RefusalClass.ENVIRONMENT,
                                    f"in the environment the lock was written in, the rebuilt stage hashes to "
                                    f"{tried}, and the lock records {lock['stage']['sha256']}",
                                    "report it: the stage writer wrote other bytes from the same declarations; `mime "
                                    "verify <folder>` repeats it"))
        else:
            env = [*env, f"stage: written as {lock['stage']['sha256']}, rebuilt as {tried}"]
    if (hashes["derived"], hashes["observations"]) == (saved_hashes["derived"], saved_hashes["observations"]):
        derived = "the same graph, hash for hash"
    else:
        derived_differences = differences(lock["derived"]["physics"], physics_in_form(record.derived.physics, form.version),
                                        "physics")
        derived_differences += differences(lock["derived"]["observations"], record.derived.observations,
                                           "observations")
        if same:
            refusals.append(_refuse("load.derived-hash", RefusalClass.ENVIRONMENT,
                                    "in the environment the lock was written in, the rebuilt graph differs from the "
                                    "saved one: " + "; ".join(derived_differences),
                                    "an entry builds something other than what its version built: report it to the "
                                    "package that registers "
                                    "it; `mime registry` lists each entry's version"))
        derived = "differs from the saved graph; compared within the lock's tolerance"
        derived_differences = []
        rtol, atol = lock["reference"]["rtol"], lock["reference"]["atol"]
        scaled = form.version >= SCALED_FROM
        _numbers_beyond(lock["derived"]["physics"], physics_in_form(record.derived.physics, form.version), rtol, atol,
                        "physics", derived_differences, scaled=scaled)
        for path, array in sorted(record.derived.static.items()):
            held = form.reference.get(static_key(path))
            if held is None or held.shape != array.shape:
                derived_differences.append(f"static array {path}: missing or of another shape")
            elif not (judge(held, array, OWN, rtol, atol, steps=False).within if scaled
                      else np.allclose(held, array, rtol=rtol, atol=atol)):
                derived_differences.append(f"static array {path}: max |difference| "
                                           f"{float(np.max(np.abs(held.astype(np.float64) - array))):.3e}")
    if refusals:
        raise LoadRefused(refusals)
    return LoadReport(str(form.folder), same, tuple(env), derived, tuple(derived_differences), tuple(derived_prims),
                      form.document["schema"], form.document["profile"], form.version, form.renamed, form.notes,
                      writer if writer is not None else STAGE_WRITER)


def asset_refusals(lock: Lock, record: SavedRecord) -> list[Refusal]:
    """The lock's ``assets`` key against the remote assets the rebuilt experiment reads (``load.assets``): an asset it
    reads that the lock does not name, by name, since ``mime assets fetch <folder>`` and a run on Modal fetch only what
    the lock names; and one the lock names under another hash than the one read now. A lock may name more assets than
    the entries read (a bundle's own files): those are fetched and not judged here."""
    saved = {held["name"]: held for held in lock.get("assets", [])}
    read = {ref.name: ref for ref in record.remote_assets}
    refusals: list[Refusal] = []
    for name, ref in read.items():
        held = saved.get(name)
        if held is None:
            refusals.append(_refuse("load.assets", RefusalClass.USER,
                                    f"the experiment reads the asset {name!r} ({ref.size} bytes), and its lock does "
                                    f"not name it, so neither `mime assets fetch <folder>` nor a run on Modal would "
                                    f"fetch it", "build the experiment and save it again, built.save(folder), which "
                                                 "writes the lock's assets"))
        elif held["sha256"] != ref.sha256:
            refusals.append(_refuse("load.assets", RefusalClass.ENVIRONMENT,
                                    f"the lock names the asset {name!r} with sha256 {held['sha256']}, and the "
                                    f"installed MIME reads it with sha256 {ref.sha256}",
                                    "install the MIME the lock was written with (`pip install "
                                    "mime-engine==<version>`), or build and save the "
                                    "experiment again: built.save(folder)"))
    return refusals


#: What a moved line says of a field the rerun lacks or gave in another shape (no numbers, so no tolerance holds it).
SHAPE_MOVED = "missing or of another shape"


#: What ``verify`` says of a reference made on a GPU with XLA's deterministic mode off, and what a save says where it
#: writes one: once, never on the CPU, never in deterministic mode.
DETERMINISTIC_OFF = ("deterministic mode is off: this reference was made on a GPU without XLA's deterministic mode, so it "
                     "verifies within tolerance, not bit for bit; for bit-for-bit reproducibility, rerun with "
                     "--deterministic")
SAVED_DETERMINISTIC_OFF = ("deterministic mode is off: this reference was written on a GPU without XLA's deterministic "
                           "mode, so it verifies within tolerance, not bit for bit; for bit-for-bit reproducibility, rerun "
                           "with --deterministic (`mime run <folder> --deterministic`, or `mime.env.deterministic_mode()` "
                           "before anything computes)")


class NondeterministicReference(UserWarning):
    """A reference written on a GPU with XLA's deterministic mode off (:data:`SAVED_DETERMINISTIC_OFF`)."""


def nondeterministic_gpu(environment: Mapping[str, object]) -> bool:
    """Whether a reference written under ``environment`` (a lock's) came from a GPU with XLA's deterministic mode off:
    two such runs of one program differ in their last bits, so no run repeats it bit for bit."""
    return environment.get("device") == "gpu" and not environment.get("deterministic", False)


@dataclass(frozen=True)
class VerifyReport:
    """A rerun of the first ``steps`` steps against the reference outputs: bit for bit in the environment the lock was
    written in, unless its reference was made on a GPU with deterministic mode off (``bits``), and otherwise within the
    lock's tolerance: from the tenth form each field block by block, on its largest magnitude or its measured
    resolution (``scaled``, ``mime.saved._scales``), before it element by element."""

    steps: int
    same_environment: bool
    rtol: float
    atol: float
    environment: str
    moved: tuple[str, ...] = field(default_factory=tuple)
    #: Every field the rerun was compared on (``node.field``, sorted); a line of ``moved`` starts with its field's label.
    compared: tuple[str, ...] = field(default_factory=tuple)
    #: Whether the fields were held bit for bit; ``None`` reads as ``same_environment``.
    bits: bool | None = None
    #: Whether each field was judged block by block (the tenth form on), not element by element.
    scaled: bool = False
    #: The warning of a reference made on a GPU with deterministic mode off (:data:`DETERMINISTIC_OFF`), or ``None``.
    warning: str | None = None
    #: The factor over each block's measured resolution its floor is (``reference.resolution_factor``), where judged so.
    factor: float = RESOLUTION_FACTOR

    @property
    def bit_for_bit(self) -> bool:
        return self.same_environment if self.bits is None else self.bits

    @property
    def tolerance(self) -> str:
        """What the comparison held the fields to, in words."""
        if self.bit_for_bit:
            return "bit for bit"
        if self.scaled:
            return (f"within rtol {self.rtol} of each block's largest magnitude, or {self.factor:g} times its measured "
                    f"resolution" + (f", atol {self.atol}" if self.atol else ""))
        return f"within rtol {self.rtol}, atol {self.atol}"

    @property
    def reproduced(self) -> bool:
        return not self.moved

    def text(self) -> str:
        if self.reproduced:
            lines = [f"verify: reproduced {self.steps} steps {self.tolerance} on {self.environment}"]
        else:
            lines = [f"verify: {len(self.moved)} reference output(s) moved over {self.steps} steps on {self.environment}"]
            lines += [f"  {line}" for line in self.moved]
        return "\n".join(lines + ([self.warning] if self.warning else []))


def moved_line(label: str, saved: np.ndarray, found: np.ndarray, *, bits: bool, scaled: bool, rtol: float,
               atol: float, field: FieldUnits = OWN, resolutions: Mapping[str, object] | None = None,
               factor: float = RESOLUTION_FACTOR, forces: tuple[np.ndarray, np.ndarray] | None = None,
               steps: bool = True) -> str | None:
    """The line of a field that moved, ``None`` where it did not: bit for bit, block by block (``scaled``: on each
    block's largest magnitude or ``factor`` times its resolution, ``resolutions`` by block), or element by element.
    ``saved`` and ``found`` are of one shape, the first axis the steps where ``steps``."""
    if bits:
        if np.array_equal(saved, found):
            return None
    elif scaled:
        judged = judge(saved, found, field, rtol, atol, resolutions=resolutions, factor=factor, forces=forces,
                       steps=steps)
        block = judged.beyond
        if block is None:
            return None
        said = "it differs" if judged.exact else block.words(rtol, factor)
        held = " on the rows whose force acts" if field.forces is not None and forces is not None else ""
        return f"{label}: {said}{held}" + (f", first at step {judged.first}" if steps else "")
    elif np.allclose(saved, found, rtol=rtol, atol=atol):
        return None
    delta = np.abs(saved.astype(np.float64) - found.astype(np.float64))
    if not steps:
        return f"{label}: max |difference| {float(np.max(delta)):.3e}"
    step = int(np.argmax(np.any(delta.reshape(delta.shape[0], -1) > 0, axis=1)))
    return f"{label}: max |difference| {float(np.max(delta)):.3e}, first at step {step}"


def recorded_resolution(lock: Lock) -> tuple[Mapping[str, Mapping[str, object]], float]:
    """The resolution each field's blocks were measured to at save, by label, and the floor's factor over it, as the
    lock records them; a lock that records none gives every block no floor."""
    reference = cast(Mapping[str, object], lock["reference"])
    held = reference.get("resolution")
    factor = reference.get("resolution_factor")
    return (cast(Mapping[str, Mapping[str, object]], held) if isinstance(held, dict) else {},
            float(factor) if isinstance(factor, (int, float)) and not isinstance(factor, bool) else RESOLUTION_FACTOR)


def verify_history(form: SavedForm, history: Mapping[str, Mapping[str, np.ndarray]],
                   fields: Mapping[str, FieldUnits] | None = None) -> VerifyReport:
    """The rerun's ``history`` against the reference outputs. ``fields`` are the fields' declared units and pairings
    by label (``node.field``, ``BuiltExperiment.field_units``): a field without one is one block."""
    lock = form.lock
    reference = lock["reference"]
    steps, rtol, atol = reference["steps"], reference["rtol"], reference["atol"]
    same = not environment_differences(lock)
    made = cast(Mapping[str, object], lock["environment"])
    bits = same and not nondeterministic_gpu(made)
    scaled = form.version >= SCALED_FROM
    held_fields = fields or {}
    resolutions, factor = recorded_resolution(lock)
    here = environment()
    moved: list[str] = []
    compared: list[str] = []
    found = {history_key(node, name): np.asarray(value) for node, named in history.items()
             for name, value in named.items()}
    saved = {key: value for key, value in form.reference.items() if key.startswith("history/")}

    def label_of(key: str) -> str:
        return key.removeprefix("history/").replace("/", ".", 1)

    for key in sorted(set(saved) | set(found)):
        a, b = saved.get(key), found.get(key)
        label = label_of(key)
        compared.append(label)
        if a is None or b is None or a.shape != b.shape:
            moved.append(f"{label}: {SHAPE_MOVED}")
            continue
        units = held_fields.get(label, OWN)
        forces = None
        if units.forces is not None:
            node, _, name = units.forces.partition(".")
            pair = saved.get(history_key(node, name)), found.get(history_key(node, name))
            if pair[0] is not None and pair[1] is not None and pair[0].shape == pair[1].shape:
                forces = (pair[0], pair[1])
        line = moved_line(label, a, b, bits=bits, scaled=scaled, rtol=rtol, atol=atol, field=units,
                          resolutions=resolutions.get(label), factor=factor, forces=forces)
        if line is not None:
            moved.append(line)
    device = f"{here['device']}, {'float64' if here['x64'] else 'float32'}"
    return VerifyReport(steps, same, rtol, atol, device, tuple(moved), tuple(compared), bits=bits, scaled=scaled,
                        warning=DETERMINISTIC_OFF if nondeterministic_gpu(made) else None, factor=factor)


__all__ = ["DEFAULT_STEPS", "DETERMINISTIC_OFF", "DOCUMENT_FILE", "LOCK_FILE", "REFERENCE_FILE", "SAVED_DETERMINISTIC_OFF",
           "LoadRefused", "LoadReport", "NondeterministicReference", "lock_text", "moved_line", "nondeterministic_gpu",
           "recorded_resolution",
           "SavedForm", "VerifyReport", "absent_referenced_files", "asset_refusals", "check_load", "differences",
           "environment_differences", "missing_referenced",
           "layout_problem", "lock_text_problems", "named_files", "read_document", "read_document_version",
           "read_folder",
           "scene_file_refusals", "stage_change_refusals", "stage_writers", "verify_history", "write_folder"]
