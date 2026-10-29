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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from packaging.version import InvalidVersion, Version

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.build._derived import array_record, sha256_of
from mime.materials import RECORDS
from mime.saved._document import DESCRIPTIONS, SCENE_FOLDER, dump_document, parse_yaml
from mime.saved._record import SavedRecord, environment
from mime.saved._schema import EntryRecord, ExperimentDocument, Json, Lock, ReferenceRecord, conforms
from mime.scene import LAYERS, STAGE_FILE, sha256_hex, stage_derived_findings, stage_edits

DOCUMENT_FILE = "experiment.mime.yaml"
LOCK_FILE = "mime.lock"
REFERENCE_FILE = "reference.npz"

#: How many steps the reference outputs cover, and the tolerance ``verify`` compares them with across environments.
DEFAULT_STEPS = 8
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9

_PYTHON_TAG = "tag:yaml.org,2002:python/"

#: A distribution name as PEP 503 (and PEP 508) write one, and the characters a version is made of: both are printed
#: into a ``pip install`` command, so what the lock holds is held to them before anything reads it.
_DISTRIBUTION = re.compile(r"[A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]")
_VERSION_CHARACTERS = re.compile(r"[A-Za-z0-9.+!_-]+")


class LoadRefused(BuildRefused):
    """Every reason a saved experiment could not be loaded, each with what to do about it."""


def history_key(node: str, field_name: str) -> str:
    return f"history/{node}/{field_name}"


def static_key(path: str) -> str:
    return f"static/{path}"


def arrays_hash(arrays: Mapping[str, np.ndarray]) -> str:
    return sha256_of({key: array_record(value) for key, value in sorted(arrays.items())})


def write_folder(folder: Path, record: SavedRecord, history: Mapping[str, Mapping[str, np.ndarray]],
                 steps: int) -> Lock:
    """Write the document, the lock and the reference outputs (``history``, every field over ``steps`` steps,
    with the static arrays beside them) into ``folder``."""
    if record.unsavable or record.document is None:
        raise BuildRefused(record.unsavable)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {history_key(node, name): np.asarray(value) for node, fields in history.items()
              for name, value in fields.items()}
    arrays.update({static_key(path): value for path, value in record.derived.static.items()})
    np.savez(folder / REFERENCE_FILE, allow_pickle=False, **arrays)
    reference: ReferenceRecord = {"file": REFERENCE_FILE, "sha256": arrays_hash(arrays), "steps": steps,
                                  "rtol": DEFAULT_RTOL, "atol": DEFAULT_ATOL}
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
    (folder / DOCUMENT_FILE).write_text(dump_document(record.document), encoding="utf-8")
    (folder / LOCK_FILE).write_text(json.dumps(lock, indent=1, sort_keys=True, allow_nan=False) + "\n",
                                    encoding="utf-8")
    return lock


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


def read_document(path: Path) -> tuple[ExperimentDocument | None, list[Refusal]]:
    """The document at ``path``, checked against its schema, or every reason it is refused."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        return None, [_refuse("load.missing-file", RefusalClass.USER, f"cannot read the document {path}: {error}",
                              f"pass a folder holding {DOCUMENT_FILE}")]
    try:
        data = parse_yaml(text)
    except yaml.YAMLError as error:
        if _PYTHON_TAG in str(error):
            return None, [_refuse("load.code-in-document", RefusalClass.USER,
                                  f"{path} holds a Python tag: a document never holds code, and loading constructs "
                                  f"no Python object a tag names ({error})",
                                  "remove the tag: write settings as numbers, strings, booleans, lists and "
                                  "quantities")]
        return None, [_refuse("load.document-yaml", RefusalClass.USER, f"{path} is not a readable document: {error}",
                              "fix the YAML, or save the experiment again")]
    document, problems = conforms(data, ExperimentDocument)
    return document, [_refuse("load.document-shape", RefusalClass.USER, f"{path}: {problem}",
                              "write the document as experiment.save() writes it")
                      for problem in problems]


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
             *(held["file"] for held in lock["records"].values())]
    for file in named:
        why = layout_problem(file)
        if why is not None:
            problems.append(f"the lock names the file {file!r}, which {why}")
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


def read_folder(folder: Path) -> SavedForm:
    """The saved form in ``folder``, every file checked; raises :class:`LoadRefused` with every reason."""
    document, refusals = read_document(folder / DOCUMENT_FILE)
    lock: Lock | None = None
    try:
        data: object = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        refusals.append(_refuse("load.lock", RefusalClass.USER, f"cannot read the lock {folder / LOCK_FILE}: {error}",
                                "save the experiment again: the document is checked against the lock its build "
                                "wrote"))
    else:
        lock, problems = conforms(data, Lock)
        if lock is not None:
            problems = lock_text_problems(lock)
            lock = lock if not problems else None
        refusals += [_refuse("load.lock-shape", RefusalClass.USER, f"{folder / LOCK_FILE}: {problem}",
                             "save the experiment again") for problem in problems]
    reference: dict[str, np.ndarray] = {}
    if lock is not None:
        try:
            with np.load(folder / lock["reference"]["file"], allow_pickle=False) as held:
                reference = {key: np.asarray(held[key]) for key in held.files}
        except (OSError, ValueError) as error:
            refusals.append(_refuse("load.reference", RefusalClass.USER,
                                    f"cannot read the reference outputs {lock['reference']['file']}: {error}",
                                    "save the experiment again"))
        else:
            if arrays_hash(reference) != lock["reference"]["sha256"]:
                refusals.append(_refuse("load.reference-changed", RefusalClass.USER,
                                        f"the reference outputs in {lock['reference']['file']} are not the ones "
                                        f"the lock records", "save the experiment again"))
    stage_found: str | None = None
    if lock is not None:
        found_refusals, stage_found = scene_file_refusals(folder, lock)
        refusals += found_refusals
        refusals += record_file_refusals(folder, lock)
    if refusals or document is None or lock is None:
        raise LoadRefused(refusals)
    return SavedForm(folder, document, lock, reference, stage_found)


def scene_file_refusals(folder: Path, lock: Lock) -> tuple[list[Refusal], str | None]:
    """The stage and every layer it references, each held to the hash the lock records: an asset file changed
    after its lock was written is refused, and so is a file that cannot be read. A stage whose bytes changed is
    returned as text, for the load to compare with the stage the rebuild writes (``stage_change_refusals``): the build
    writes the stage from the document, so an edit is judged against what it writes."""
    stage = lock["stage"]
    refusals: list[Refusal] = []
    stage_found: str | None = None
    files = [(stage["file"], stage["sha256"], "load.stage-changed"),
             *((layer, digest, "load.asset-changed") for layer, digest in sorted(stage["layers"].items()))]
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


def stage_change_refusals(form: SavedForm, record: SavedRecord, compare: bool) -> tuple[list[Refusal], list[str]]:
    """A saved stage whose bytes are not the ones its lock pins, judged against the stage the rebuild writes from the
    same document (``compare``; without it, as after a changed document, any change is refused). An edit confined to
    derived prims (a kind's drawn mesh, a region's primitive) is reported, one line each: they are output only, read
    by nothing, and every build writes them again. Profiles arrive later, and a ``strict`` profile refuses such an
    edit here. Any other edit is refused."""
    found = form.stage_found
    if found is None:
        return [], []
    stage = form.lock["stage"]
    refused = [_refuse("load.stage-changed", RefusalClass.USER,
                       f"the stage {stage['file']} changed after its lock was written: sha256 "
                       f"{sha256_hex(found.encode('utf-8'))}, and the lock records {stage['sha256']}",
                       "restore the file the lock pins, or build and save the experiment again")]
    if not compare:
        return refused, []
    try:
        edits = stage_edits(found, record.stage)
    except Exception:  # noqa: BLE001 - a stage USD cannot read is a changed stage, refused as one
        return refused, []
    if edits.outside:
        return refused, []
    findings = {finding.path: finding.message for finding in stage_derived_findings(found)}
    return [], [f"{findings.get(path, f'{path} differs from the derived prim the build writes')}; it is output "
                f"only, and the build writes it again" for path in edits.derived]


def record_file_refusals(folder: Path, lock: Lock) -> list[Refusal]:
    """Every record file the lock pins, held to its hash: a record edited after its lock was written, by so much as
    one byte, is refused."""
    refusals: list[Refusal] = []
    for material, held in sorted(lock["records"].items()):
        try:
            found = sha256_hex((folder / held["file"]).read_bytes())
        except OSError as error:
            refusals.append(_refuse("load.record-changed", RefusalClass.USER,
                                    f"cannot read material {material!r}'s record {held['file']}: {error}",
                                    "save the experiment again, or restore the file"))
            continue
        if found != held["sha256"]:
            refusals.append(_refuse("load.record-changed", RefusalClass.USER,
                                    f"material {material!r}'s record {held['file']} changed after its lock was "
                                    f"written: sha256 {found}, and the lock records {held['sha256']}",
                                    "restore the record the lock pins, or build and save the experiment again"))
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


def _numbers_beyond(saved: Json, found: Json, rtol: float, atol: float, at: str, out: list[str]) -> None:
    if isinstance(saved, dict) and isinstance(found, dict):
        for key in sorted(set(saved) & set(found)):
            _numbers_beyond(saved[key], found[key], rtol, atol, f"{at}.{key}", out)
    elif isinstance(saved, list) and isinstance(found, list) and len(saved) == len(found):
        for i, (a, b) in enumerate(zip(saved, found, strict=True)):
            _numbers_beyond(a, b, rtol, atol, f"{at}[{i}]", out)
    elif (isinstance(saved, (int, float)) and isinstance(found, (int, float)) and not isinstance(saved, bool)
          and not isinstance(found, bool) and not math.isclose(saved, found, rel_tol=rtol, abs_tol=atol)):
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
    if saved != here:
        return [f"{key}: built under {saved!r}, loaded under {here!r}"]
    return []


def environment_differences(lock: Lock) -> list[str]:
    """Every way this process differs from the one the lock was written in, the runner's information aside."""
    here = cast(dict[str, object], environment())
    saved = cast(dict[str, object], lock["environment"])
    found = [line for key in sorted(here) if key not in INFORMATION_ONLY
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

    def text(self) -> str:
        lines = [f"load report: {self.folder}", "input hash: reproduced exactly"]
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


def check_load(form: SavedForm, record: SavedRecord) -> LoadReport:
    """The three load checks, against the rebuilt experiment's ``record``. The input hash must match exactly; the
    environment is compared and reported; the derived graph must match exactly within one environment, and
    otherwise is compared within the lock's tolerance and reported. A stage edited after its lock was written is
    refused, or reported where only its derived prims were edited (:func:`stage_change_refusals`). Raises
    :class:`LoadRefused`."""
    lock = form.lock
    saved_hashes, hashes = lock["hashes"], record.hashes()
    refusals, derived_prims = stage_change_refusals(form, record, hashes["input"] == saved_hashes["input"])
    if hashes["input"] != saved_hashes["input"]:
        edited = hashes["document"] != saved_hashes["document"]
        found = differences(lock["resolved"], record.resolved, "resolved")
        found += differences(sorted(f"{e['kind']}:{e['name']}@{e['version']}" for e in lock["entries"]),
                             sorted(f"{e['kind']}:{e['name']}@{e['version']}" for e in record.entries), "entries")
        detail = "; ".join(found) or "the document itself"
        if edited:
            refusals.append(_refuse("load.input-hash", RefusalClass.USER,
                                    f"the document was changed after its lock was written: {detail}",
                                    "build the experiment and save it again, which writes a new lock"))
        else:
            refusals.append(_refuse("load.input-hash", RefusalClass.ENVIRONMENT,
                                    f"the installed entries resolve this document differently from the build that "
                                    f"wrote the lock, under the same names and versions: {detail}",
                                    "install the versions the lock names, or report the entry that changed what it "
                                    "resolves without a new version"))
    env = environment_differences(lock)
    same = not env
    derived_differences: list[str] = []
    if refusals:
        # Another input builds another graph; the derived hash guards the same input building something else.
        raise LoadRefused(refusals)
    if record.stage_hash != lock["stage"]["sha256"]:
        if same:
            refusals.append(_refuse("load.stage-hash", RefusalClass.ENVIRONMENT,
                                    f"in the environment the lock was written in, the rebuilt stage hashes to "
                                    f"{record.stage_hash}, and the lock records {lock['stage']['sha256']}",
                                    "report it: the stage writer wrote other bytes from the same declarations"))
        else:
            env = [*env, f"stage: written as {lock['stage']['sha256']}, rebuilt as {record.stage_hash}"]
    if (hashes["derived"], hashes["observations"]) == (saved_hashes["derived"], saved_hashes["observations"]):
        derived = "the same graph, hash for hash"
    else:
        derived_differences = differences(lock["derived"]["physics"], record.derived.physics, "physics")
        derived_differences += differences(lock["derived"]["observations"], record.derived.observations,
                                           "observations")
        if same:
            refusals.append(_refuse("load.derived-hash", RefusalClass.ENVIRONMENT,
                                    "in the environment the lock was written in, the rebuilt graph differs from the "
                                    "saved one: " + "; ".join(derived_differences),
                                    "an entry builds something other than what its version built: report it to "
                                    "the package that registers it"))
        derived = "differs from the saved graph; compared within the lock's tolerance"
        derived_differences = []
        rtol, atol = lock["reference"]["rtol"], lock["reference"]["atol"]
        _numbers_beyond(lock["derived"]["physics"], record.derived.physics, rtol, atol, "physics",
                        derived_differences)
        for path, array in sorted(record.derived.static.items()):
            held = form.reference.get(static_key(path))
            if held is None or held.shape != array.shape:
                derived_differences.append(f"static array {path}: missing or of another shape")
            elif not np.allclose(held, array, rtol=rtol, atol=atol):
                derived_differences.append(f"static array {path}: max |difference| "
                                           f"{float(np.max(np.abs(held - array))):.3e}")
    if refusals:
        raise LoadRefused(refusals)
    return LoadReport(str(form.folder), same, tuple(env), derived, tuple(derived_differences), tuple(derived_prims))


@dataclass(frozen=True)
class VerifyReport:
    """A rerun of the first ``steps`` steps against the reference outputs. Within the environment the lock was written
    in every field must match bit for bit; across environments, within the lock's tolerance."""

    steps: int
    same_environment: bool
    rtol: float
    atol: float
    environment: str
    moved: tuple[str, ...] = field(default_factory=tuple)

    @property
    def reproduced(self) -> bool:
        return not self.moved

    def text(self) -> str:
        if self.reproduced:
            how = "bit for bit" if self.same_environment else f"within rtol {self.rtol}, atol {self.atol}"
            return f"verify: reproduced {self.steps} steps {how} on {self.environment}"
        lines = [f"verify: {len(self.moved)} reference output(s) moved over {self.steps} steps on {self.environment}"]
        return "\n".join(lines + [f"  {line}" for line in self.moved])


def verify_history(form: SavedForm, history: Mapping[str, Mapping[str, np.ndarray]]) -> VerifyReport:
    lock = form.lock
    reference = lock["reference"]
    steps, rtol, atol = reference["steps"], reference["rtol"], reference["atol"]
    same = not environment_differences(lock)
    here = environment()
    moved: list[str] = []
    found = {history_key(node, name): np.asarray(value) for node, fields in history.items()
             for name, value in fields.items()}
    saved = {key: value for key, value in form.reference.items() if key.startswith("history/")}
    for key in sorted(set(saved) | set(found)):
        a, b = saved.get(key), found.get(key)
        label = key.removeprefix("history/").replace("/", ".", 1)
        if a is None or b is None or a.shape != b.shape:
            moved.append(f"{label}: missing or of another shape")
            continue
        equal = np.array_equal(a, b) if same else np.allclose(a, b, rtol=rtol, atol=atol)
        if not equal:
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            step = int(np.argmax(np.any(delta.reshape(delta.shape[0], -1) > 0, axis=1)))
            moved.append(f"{label}: max |difference| {float(np.max(delta)):.3e}, first at step {step}")
    device = f"{here['device']}, {'float64' if here['x64'] else 'float32'}"
    return VerifyReport(steps, same, rtol, atol, device, tuple(moved))


__all__ = ["DEFAULT_STEPS", "DOCUMENT_FILE", "LOCK_FILE", "REFERENCE_FILE", "LoadRefused", "LoadReport",
           "SavedForm", "VerifyReport", "check_load", "differences", "environment_differences", "layout_problem",
           "lock_text_problems", "named_files", "read_document", "read_folder", "scene_file_refusals",
           "stage_change_refusals", "verify_history", "write_folder"]
