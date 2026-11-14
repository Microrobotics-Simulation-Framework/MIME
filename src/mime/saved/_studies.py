"""A study container (``.mimestudy``): one study folder as one file, to share it and check it on the other side.

It is the second kind of container (:mod:`mime.saved._container`), format ``mime.study/1``, its files under ``study/``.
Its layout check is the study folder's: ``study.json``, ``study.mime.yaml``, ``compare.usda``, ``figures/<name>.json``,
``groups/<k>/outputs.npz`` and ``report.json``, and, inside the study's saved folders (``experiment/``, the base, and
``groups/<k>/experiment/`` past group 0), the saved form's own layout. Its closed manifest names the study (its folder's
name), every carried file's sha256 and size, each file that travels as a reference with its own saved folder's asset,
the NOTICE's hash, the asset records of every lock it holds, and the container's own hash. It carries no record of a
publication: a study container names no DOI of its own.

``pack`` judges the study folder before it carries anything: the study reads (its layout, its closed manifest, each
group file against the sha256 it records), every member is done (``study.incomplete`` names the rest), the base's input
hash is the one the study and its spec record (``study.input-hash``), each figure's data is the data it was made from
(``plot.data-changed``), and each saved folder loads and is judged as ``mime pack`` judges a bundle's (its strays, its
terms, its references, a referenced file's bytes carried at another path, in any saved folder of the study); then every
file is scanned as a bundle's are, and each refusal names the study container. ``unpack`` checks every name, member and
hash as a bundle's are, then the same study checks on the folder written, so a container whose manifest was rewritten to
agree with a changed file is refused by the study's own records. A **publishing folder** holds study folders and the
record's ``mime.toml``: ``pack_studies`` writes one container per study, and lists what else it holds, which no container
carries.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import warnings
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from typing_extensions import ReadOnly, TypedDict

from mime._terminal import plain
from mime.assets import folder_assets, known_bytes, size_text
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.build._refusals import ArgumentRefused, MimeError
from mime.results._study_folder import (
    BASE_DIR,
    COMPARE_FILE,
    DONE,
    FIGURES_DIR,
    GROUP_OUTPUTS,
    GROUP_REPORT,
    GROUPS_DIR,
    SPEC_FILE,
    STUDY_FILE,
)
from mime.saved import _bundle
from mime.saved._bundle import BUNDLE, SHARING_FILES, ByRequestWarning, referenced_members, shipped_asset_problems
from mime.saved._container import (
    BundleFile,
    ContainerKind,
    ReferencedFile,
    collisions,
    container_hash,
    destination_problem,
    files_of,
    member_kind_problems,
    member_problem,
    members_listed,
    notice_of,
    read_manifest,
    refuse,
    sha256_of,
    staging_of,
    stated_format,
    unknown_format,
    unreadable,
    write_container,
    write_members,
)
from mime.saved._folder import SavedForm, layout_problem, named_files, read_folder
from mime.saved._schema import Lock, RemoteAssetRecord
from mime.saved._terms import INLINE_LIMIT_BYTES, FolderTerms, folder_terms, lock_records, place_referenced

STUDY_SUFFIX = ".mimestudy"
STUDY_CONTAINER = "mime.study/1"
STUDY_PREFIX = "study/"
#: The block's container (P4's), reserved: its suffix and format are known, and a block is refused by name until this
#: MIME reads one.
BLOCK_SUFFIX = ".mimeblock"
BLOCK_CONTAINER = "mime.block/1"
_GROUP = re.compile(r"0|[1-9][0-9]*")
_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_FAMILY = re.compile(r"mime\.(?P<family>[a-z_]+)/[1-9][0-9]*")
#: How the study container's refusals name where a file is.
IN_STUDY = "in the study container"


class StudyManifest(TypedDict, closed=True):
    """What a study container says it holds: its format, the study's name, the files carried and those that travel as
    references (study-relative paths, each reference with its own saved folder's asset), the NOTICE's sha256 and size,
    the asset records of every lock it holds, and ``container_hash``, the sha256 of the manifest without it."""

    format: ReadOnly[Literal["mime.study/1"]]
    name: ReadOnly[str]
    files: ReadOnly[dict[str, BundleFile]]
    referenced: ReadOnly[dict[str, ReferencedFile]]
    notice: ReadOnly[BundleFile]
    assets: ReadOnly[list[RemoteAssetRecord]]
    container_hash: ReadOnly[str]


def saved_folder_of(name: str) -> tuple[str, str] | None:
    """The study's saved folder a path below a study folder is in (``experiment``, or ``groups/<k>/experiment`` past
    group 0) and the path below it; ``None`` for a path in no saved folder."""
    parts = name.split("/")
    if parts[0] == BASE_DIR and len(parts) > 1:
        return BASE_DIR, "/".join(parts[1:])
    if len(parts) > 3 and parts[0] == GROUPS_DIR and _GROUP.fullmatch(parts[1]) and parts[1] != "0" \
            and parts[2] == BASE_DIR:
        return "/".join(parts[:3]), "/".join(parts[3:])
    return None


_KEPT = (f"{STUDY_FILE}, {SPEC_FILE}, {COMPARE_FILE}, {FIGURES_DIR}/<name>.json, {GROUPS_DIR}/<k>/{GROUP_OUTPUTS} and "
         f"{GROUP_REPORT}, and the saved folders {BASE_DIR}/ and {GROUPS_DIR}/<k>/{BASE_DIR}/")


def study_layout_problem(name: str) -> str | None:
    """Why ``name``, a path below a study folder written as ``/`` writes it, is not where one keeps a file, or ``None``:
    its own files, and inside each of its saved folders the saved form's own layout."""
    inside = saved_folder_of(name)
    if inside is not None:
        return layout_problem(inside[1])
    parts = name.split("/")
    if name in (STUDY_FILE, SPEC_FILE, COMPARE_FILE):
        return None
    if len(parts) == 2 and parts[0] == FIGURES_DIR and parts[1].endswith(".json"):
        return None
    if len(parts) == 3 and parts[0] == GROUPS_DIR and _GROUP.fullmatch(parts[1]) and parts[2] in (GROUP_OUTPUTS,
                                                                                                  GROUP_REPORT):
        return None
    return f"is not where a study folder keeps a file (it keeps {_KEPT})"


STUDY = ContainerKind("study", STUDY_SUFFIX, "study", (STUDY_CONTAINER,), STUDY_PREFIX, "container_hash",
                      study_layout_problem, "a study container", "the study container")
BLOCK = ContainerKind("block", BLOCK_SUFFIX, "block", (BLOCK_CONTAINER,), "block/", "container_hash",
                      lambda name: "is a block's, which this MIME does not read yet", "a block", "the block")
#: Every kind of container this MIME knows, by suffix and format family.
KINDS = (BUNDLE, STUDY, BLOCK)


def kind_of_format(held: object) -> ContainerKind | None:
    """The kind whose format family ``held`` (a manifest's ``format``) is of, or ``None``."""
    match = _FAMILY.fullmatch(held) if isinstance(held, str) else None
    return next((kind for kind in KINDS if match is not None and kind.family == match["family"]), None)


def kind_of_name(name: str) -> ContainerKind | None:
    """The kind whose suffix ``name`` ends in, or ``None``."""
    return next((kind for kind in KINDS if name.endswith(kind.suffix)), None)


def container_kind(path: str | os.PathLike[str], *, named: str | None = None) -> ContainerKind:
    """The kind of the container at ``path``, as its manifest's format says; ``named`` is the name it came by (the
    file's own by default, or a record's file name for a download). A name whose suffix is another kind's is refused
    (``container.kind``), so that neither kind is ever taken for the other, and so is a block, which this MIME does not
    read yet. A manifest that cannot be read here is judged by the kind its name says (a bundle's where it says none),
    whose checks refuse it with their reasons."""
    file = Path(path)
    name = named if named is not None else file.name
    by_name = kind_of_name(name)
    try:
        with zipfile.ZipFile(file) as archive:
            stated = stated_format(archive)
    except zipfile.BadZipFile as error:
        raise unreadable(error, by_name or BUNDLE) from error
    by_format = kind_of_format(stated)
    kind = by_format or by_name or BUNDLE
    if by_format is not None and by_name is not None and by_name is not by_format:
        stem = name.removesuffix(by_name.suffix)
        raise BuildRefused([refuse(
            "container.kind", RefusalClass.USER,
            f"{plain(name)} is named as {by_name.article} ({by_name.suffix}) and holds {by_format.article} "
            f"({plain(str(stated))})",
            f"rename it and pull it again, `mime pull {plain(stem)}{by_format.suffix}`, or fetch it again from its "
            f"source: MIME reads a container by what its manifest says it is, and refuses one whose name says "
            f"otherwise")])
    if kind is BLOCK:
        raise BuildRefused([refuse(
            "container.kind", RefusalClass.USER,
            f"{plain(name)} is a block ({plain(str(stated or BLOCK_CONTAINER))}), which this MIME does not read yet",
            "read it with a MIME release that reads blocks (`pip install -U mime-engine`)")])
    return kind


# --- the study folder's own checks ----------------------------------------------------------------------------------


def saved_folders(document: Mapping[str, Any]) -> list[str]:
    """The study's saved folders, as its manifest names them: the base, then each group's own build past group 0. A name
    that is not one a study folder keeps a saved folder at is refused (``study.manifest``)."""
    found = [BASE_DIR]
    for group in cast(list[dict[str, Any]], document["groups"]):
        held = group.get("experiment")
        if held is None:
            continue
        if not isinstance(held, str) or member_problem(held) is not None or saved_folder_of(f"{held}/x") != (held, "x") \
                or held != f"{GROUPS_DIR}/{group['index']}/{BASE_DIR}":
            raise BuildRefused([refuse(
                "study.manifest", RefusalClass.USER,
                f"{STUDY_FILE} names {plain(repr(held))} as group {plain(str(group['index']))}'s own build, which is not "
                f"where a study folder keeps one ({GROUPS_DIR}/<k>/{BASE_DIR})",
                "write the folder with `Study.run_to(folder)`")])
        found.append(held)
    return found


def incomplete_problem(document: Mapping[str, Any], where: str) -> Refusal | None:
    """``study.incomplete`` for a study whose manifest holds members not done, naming them; ``None`` for a study whose
    every member is done."""
    rows = cast(list[dict[str, Any]], document["members"])
    pending = [int(row["point"]) for row in rows if row["status"] != DONE]
    if not pending:
        return None
    shown = ", ".join(map(str, pending[:8])) + (", ..." if len(pending) > 8 else "")
    plural = "s" if len(pending) != 1 else ""
    return refuse("study.incomplete", RefusalClass.USER,
                  f"{where}: {len(pending)} of its {len(rows)} members {'are' if plural else 'is'} not run (point{plural} "
                  f"{shown}): a study that was stopped is shared only once it is whole",
                  f"finish it: run it again into the same folder, which resumes it, `mime study run {where}/{SPEC_FILE} "
                  f"--out {where}`; then pack it")


def input_hash_problems(root: Path, document: Mapping[str, Any], base_lock: Lock, where: str) -> list[Refusal]:
    """``study.input-hash`` where the base's lock records another input hash than the study's manifest or its spec:
    the study was not run on the base it holds."""
    from mime.study._spec import read_spec

    found = str(base_lock["hashes"]["input"])
    held = [(STUDY_FILE, document["experiment"]["input_hash"])]
    held.append((SPEC_FILE, read_spec(root / SPEC_FILE)["base"].get("input_hash")))
    return [refuse("study.input-hash", RefusalClass.USER,
                   f"{where}: {name} records the base experiment's input hash {plain(str(recorded))}, and the base's "
                   f"lock ({BASE_DIR}/mime.lock) records {plain(found)}: the study was not run on the base it holds",
                   f"run the study again on its base, `mime study run {where}/{SPEC_FILE} --out <a new folder>`: a "
                   f"study folder holds the base it ran on, byte for byte")
            for name, recorded in held if recorded != found]


def figure_problems(root: Path) -> list[Refusal]:
    """Each figure's recipe in ``figures/`` read and its data held to the sha256 it records (``plot.data-changed``,
    ``plot.recipe``), with nothing drawn. Plotly's JSON of a figure (``<name>.plotly.json``) is presentation, and only
    scanned."""
    from mime.results._figure import read_recipe

    folder = root / FIGURES_DIR
    names = sorted(path.name.removesuffix(".json") for path in folder.glob("*.json")
                   if not path.name.endswith(".plotly.json")) if folder.is_dir() else []
    found: list[Refusal] = []
    for name in names:
        try:
            read_recipe(root, name)
        except ArgumentRefused as refused:
            found += refused.refusals
    return found


def study_problems(root: Path, document: Mapping[str, Any], base_lock: Lock, where: str) -> list[Refusal]:
    """What a study folder's own records refuse beyond what reading it does: a member not done, a base its records do
    not name, a figure's data changed."""
    found = [each for each in [incomplete_problem(document, where)] if each is not None]
    found += input_hash_problems(root, document, base_lock, where)
    return found + figure_problems(root)


def _within(where: str, refusals: Sequence[Refusal]) -> list[Refusal]:
    """``refusals`` of a saved folder of a study, each message saying where it is."""
    return [dataclasses.replace(each, message=f"{IN_STUDY}, {where}/: {each.message}") for each in refusals]


def _unique(refusals: Sequence[Refusal]) -> list[Refusal]:
    seen: set[tuple[str, str]] = set()
    found: list[Refusal] = []
    for each in refusals:
        if (each.code, each.message) not in seen:
            seen.add((each.code, each.message))
            found.append(each)
    return found


# --- packing ---------------------------------------------------------------------------------------------------------


def _stray_files(files: Sequence[tuple[str, Path]], forms: Mapping[str, SavedForm]) -> list[Refusal]:
    """A refusal for each file below the study that is not one it keeps: inside a saved folder, one the saved form does
    not name (as a bundle refuses it); and each pair of names one platform reads as one."""
    found: list[Refusal] = []
    named = {where: named_files(form) for where, form in forms.items()}
    for member, _ in files:
        inside = saved_folder_of(member)
        problem = STUDY.name_problem(member)
        if problem is None and inside is not None:
            where, below = inside
            problem = None if below in named.get(where, set()) else (
                f"is not listed in its lock" if where in named else "is in a saved folder the study does not name")
        if problem is not None:
            found.append(refuse("bundle.stray-file", RefusalClass.USER,
                                f"{IN_STUDY}, {plain(member)} is not a file the study folder holds: it {problem}",
                                "a study container carries what the study's run and its saved experiments wrote and "
                                "nothing else (an .env, a .git directory and an editor's file stay behind): remove the "
                                "file, or keep the folder somewhere it is not, then `mime pack <folder>` again"))
    found += [refuse("bundle.name-collision", RefusalClass.USER,
                     f"{IN_STUDY}, {plain(one)} and {plain(other)} are one name on a case-insensitive filesystem",
                     "rename one, so that the study container unpacks to the same folder on macOS and Windows as on "
                     "Linux; then `mime pack <folder>`")
              for one, other in collisions([member for member, _ in files])]
    return found


def _sent(judged: FolderTerms) -> set[str]:
    """The sha256s of a saved folder's files that never travel inside a container: those that travel as references and
    those that may not be redistributed."""
    return {file.sha256 for file in judged.files if not file.verdict.carried or file.referenced is not None}


def _cross_copies(members: Mapping[str, bytes], judged: Mapping[str, FolderTerms]) -> list[Refusal]:
    """``bundle.referenced-copy`` for each carried file whose bytes are those of a file that travels as a reference in
    another of the study's saved folders (a copy in one saved folder is that folder's own check)."""
    found: list[Refusal] = []
    for member, data in sorted(members.items()):
        inside = saved_folder_of(member)
        digest = sha256_of(data)
        for where, terms in judged.items():
            if inside is not None and inside[0] == where:
                continue
            original = next((file for file in terms.files if file.sha256 == digest), None)
            if original is not None and digest in _sent(terms):
                found.append(refuse(
                    "bundle.referenced-copy", RefusalClass.USER,
                    f"{IN_STUDY}, {plain(member)} holds the same bytes as {plain(where)}/{plain(original.path)} "
                    f"(sha256 {digest}), which travels as a reference ({original.verdict.words()}), so the container "
                    f"would carry those bytes at a second path",
                    "keep one copy, in the saved folder whose terms reference it, and reference it from the other; then "
                    "save the experiment again, built.save(folder), and `mime pack <folder>`"))
                break
    return found


def _own_payload_problems(members: Mapping[str, bytes], locks: Sequence[Lock], sent: set[str]) -> list[Refusal]:
    """``bundle.referenced-copy`` for each of the study's own files (its manifest, outputs, reports, figures) whose bytes
    a record in reach of any of its locks says may not travel."""
    known = known_bytes([(ref, "in a lock of the study") for lock in locks for ref in lock_records(lock)])
    found: list[Refusal] = []
    for member, data in sorted(members.items()):
        digest = sha256_of(data)
        held = known.found(digest)
        if held is not None and digest not in sent:
            found.append(refuse(
                "bundle.referenced-copy", RefusalClass.USER,
                f"{IN_STUDY}, {plain(member)} holds the bytes of {plain(held[1])} (sha256 {digest}), which may not be "
                f"redistributed: a container never carries them",
                "keep the file out of the study folder; then `mime pack <folder>`"))
    return found


def _assets(root: Path, folders: Sequence[str]) -> list[RemoteAssetRecord]:
    """The asset records of every lock of the study, in the order of its saved folders and each lock's, each once."""
    found: list[RemoteAssetRecord] = []
    for where in folders:
        for ref in folder_assets(root / where):
            record = ref.to_record()
            if record not in found:
                found.append(record)
    return found


def study_name(root: Path) -> str:
    """The name a study container gives its study: its folder's, written as a file name."""
    return _NAME_SAFE.sub("_", root.name) or "study"


def _too_large(member: str, size: int) -> Refusal:
    if saved_folder_of(member) is not None:
        action = ("give its terms an origin, asset(path, terms=Terms(..., origin=(\"https://...\",))), and save the "
                  "experiment again, built.save(folder): it then travels as a reference that `mime assets fetch "
                  "<folder>` fetches on request")
    else:
        action = ("keep less of the run: record fewer outputs (record=[...]) or less often (every=...), and run the "
                  "study again into a new folder; a study's own output has no origin to travel as a reference from")
    return refuse("bundle.file-too-large", RefusalClass.USER,
                  f"{IN_STUDY}, {plain(member)} has {size_text(size)}, over the {size_text(INLINE_LIMIT_BYTES)} a "
                  f"container carries", action)


def pack_study(folder: str | os.PathLike[str], *, out: str | os.PathLike[str] | None = None) -> Path:
    """Write the study container of the study folder ``folder`` (by default ``<folder>.mimestudy`` beside it) and return
    its path: refused, with every reason at once, as the module says. The same folder makes the same bytes."""
    from mime.experiment import Experiment
    from mime.saved._card import by_request_warnings
    from mime.saved._study_card import study_notice_text
    from mime.results._study_read import read_study

    root = Path(os.path.abspath(folder))
    document = read_study(root).record
    target = Path(out) if out is not None else root.with_name(root.name + STUDY_SUFFIX)
    if root.resolve() in (target.resolve(), *target.resolve().parents):
        raise BuildRefused([refuse("bundle.output-inside", RefusalClass.USER,
                                   "the study container would be written inside the folder it packs",
                                   "write it beside the folder (--out)")])
    folders = saved_folders(document)
    for where in folders:
        Experiment.load(root / where)
    forms = {where: read_folder(root / where) for where in folders}
    refusals = study_problems(root, document, forms[BASE_DIR].lock, root.name)
    stored = _bundle.held_tokens()
    files = [(member, path) for member, path in files_of(root, STUDY) if member not in SHARING_FILES]
    strays = _stray_files(files, forms)
    if strays:
        raise BuildRefused([*refusals, *strays])
    judged = {where: folder_terms(root / where, form.document, form.lock, named_files(form), form.version)
              for where, form in forms.items()}
    unlisted = _within_all(judged, lambda where, terms: terms.unlisted_refusals("mime pack"))
    if unlisted:
        raise BuildRefused([*refusals, *unlisted])
    referenced: dict[str, ReferencedFile] = {}
    for where, terms in judged.items():
        found, problems = referenced_members(terms, forms[where].lock)
        referenced |= {f"{where}/{path}": held for path, held in found.items()}
        refusals += _within(where, [*problems, *shipped_asset_problems(forms[where].lock)])
    members: dict[str, bytes] = {}
    for member, path in files:
        if member in referenced:
            continue
        size = path.stat().st_size
        if size > INLINE_LIMIT_BYTES:
            refusals.append(_too_large(member, size))
            continue
        carried = _bundle.scanned(f"{member} ({IN_STUDY})", path, stored, refusals)
        if carried is not None:
            members[member] = carried
    for where, terms in judged.items():
        own = {member.removeprefix(f"{where}/"): data for member, data in members.items()
               if (saved_folder_of(member) or ("", ""))[0] == where}
        refusals += _within(where, _bundle.payload_problems(own, terms, forms[where].lock))
    sent = {digest for terms in judged.values() for digest in _sent(terms)}
    refusals += _own_payload_problems({member: data for member, data in members.items() if saved_folder_of(member) is None},
                                      [form.lock for form in forms.values()], sent)
    refusals += _cross_copies(members, judged)
    notice = study_notice_text(root, document, judged, forms).encode("utf-8")
    refusals += _bundle.scan_text(f"the NOTICE ({IN_STUDY})", notice, stored)
    refusals = _unique(refusals)
    if refusals:
        raise BuildRefused(refusals)
    manifest: dict[str, object] = {
        "format": STUDY_CONTAINER, "name": study_name(root),
        "files": {member: {"sha256": sha256_of(data), "size": len(data)} for member, data in sorted(members.items())},
        "referenced": dict(sorted(referenced.items())),
        "notice": {"sha256": sha256_of(notice), "size": len(notice)},
        "assets": _assets(root, folders)}
    manifest["container_hash"] = container_hash(manifest, "container_hash")
    text = (json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")
    refusals = _bundle.scan_text(f"the manifest ({IN_STUDY})", text, stored)
    if refusals:
        raise BuildRefused(refusals)
    path = write_container(target, text, notice, members, STUDY)
    for where, terms in judged.items():
        for line in by_request_warnings(terms):
            warnings.warn(f"{IN_STUDY}, {where}/: {line}", ByRequestWarning, stacklevel=2)
    return path


def _within_all(judged: Mapping[str, FolderTerms],
                each: Callable[[str, FolderTerms], Sequence[Refusal]]) -> list[Refusal]:
    return [refusal for where, terms in judged.items() for refusal in _within(where, each(where, terms))]


def is_study_folder(root: Path) -> bool:
    return (root / STUDY_FILE).is_file()


def publishing_entries(folder: str | os.PathLike[str]) -> tuple[list[Path], list[str]]:
    """The study folders a publishing folder holds, sorted by name, and its other entries (its ``mime.toml`` aside),
    which no container carries and nothing refuses."""
    root = Path(os.path.abspath(folder))
    studies: list[Path] = []
    others: list[str] = []
    for child in sorted(root.iterdir(), key=lambda each: each.name):
        if child.is_dir() and not child.is_symlink() and is_study_folder(child):
            studies.append(child)
        elif child.name not in SHARING_FILES:
            others.append(child.name + ("/" if child.is_dir() else ""))
    return studies, others


def study_folders(folder: str | os.PathLike[str]) -> list[Path]:
    """The study folders of ``folder``: a publishing folder's, or a study folder itself."""
    root = Path(os.path.abspath(folder))
    return publishing_entries(root)[0] if is_publishing_folder(root) else [root]


def is_publishing_folder(root: Path) -> bool:
    """Whether ``root`` is a publishing folder: neither a study nor a saved folder, and holding a study folder."""
    from mime.saved._folder import DOCUMENT_FILE

    return root.is_dir() and not is_study_folder(root) and not (root / DOCUMENT_FILE).is_file() \
        and bool(publishing_entries(root)[0])


def pack_studies(folder: str | os.PathLike[str], *, out: str | os.PathLike[str] | None = None) -> tuple[Path, ...]:
    """Write one study container per study folder the publishing folder ``folder`` holds, as ``<study>.mimestudy`` in
    ``out`` (by default the folder's parent), and return their paths, sorted. Every study is judged first and a refusal
    of any is every study's reasons at once, each naming its study: no container is written unless all are. What else
    the folder holds is left out (:func:`publishing_entries` lists it)."""
    root = Path(os.path.abspath(folder))
    studies, _ = publishing_entries(root)
    if not studies:
        raise BuildRefused([refuse("study.not-a-study-folder", RefusalClass.USER,
                                   f"{root} holds no study folder (a folder with {STUDY_FILE})",
                                   "pack a study folder, a saved folder, or a folder of study folders: `mime pack "
                                   "<folder>`")])
    directory = Path(os.path.abspath(out)) if out is not None else root.parent
    if root.resolve() in (directory.resolve(), *directory.resolve().parents):
        raise BuildRefused([refuse("bundle.output-inside", RefusalClass.USER,
                                   "the study containers would be written inside the folder they pack",
                                   "write them beside the folder (--out <folder>)")])
    clashing = collisions([study.name for study in studies])
    if clashing:
        raise BuildRefused([refuse("bundle.name-collision", RefusalClass.USER,
                                   f"the study folders {plain(one)} and {plain(other)} are one name on a "
                                   f"case-insensitive filesystem", "rename one; then `mime pack <folder>`")
                            for one, other in clashing])
    directory.mkdir(parents=True, exist_ok=True)
    staging = staging_of(directory / root.name)
    staging.mkdir()
    try:
        written: list[Path] = []
        refusals: list[Refusal] = []
        for study in studies:
            try:
                written.append(pack_study(study, out=staging / (study.name + STUDY_SUFFIX)))
            except MimeError as refused:
                refusals += [dataclasses.replace(each, message=f"{plain(study.name)}: {each.message}")
                             for each in refused.refusals]
        if refusals:
            raise BuildRefused(refusals)
        placed = [directory / path.name for path in written]
        for path, target in zip(written, placed, strict=True):
            path.replace(target)
        return tuple(placed)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def pack(folder: str | os.PathLike[str], *, out: str | os.PathLike[str] | None = None,
         own_locations: frozenset[str] = frozenset()) -> Path:
    """Write the container of ``folder``: a study folder's ``.mimestudy`` (:func:`pack_study`), or a saved folder's
    bundle (:func:`mime.saved._bundle.pack`), by default beside the folder; its path. ``own_locations`` is the bundle's,
    for ``mime push``."""
    if is_study_folder(Path(os.path.abspath(folder))):
        return pack_study(folder, out=out)
    return _bundle.pack(folder, out=out, own_locations=own_locations)


# --- unpacking -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UnpackedStudy:
    """What an unpacked study container was: its manifest, the folder it was written to and its ``NOTICE``; each saved
    folder's lock; and the files that travel as references, those placed from the asset cache and those still to be
    fetched (``mime assets fetch <saved folder>`` fetches and places them), as paths below the study folder."""

    manifest: StudyManifest
    folder: Path
    notice: str
    locks: Mapping[str, Lock]
    placed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


def _carried_reference_problems(manifest: StudyManifest) -> list[Refusal]:
    """What no study container ``pack`` writes holds, judged from the manifest before anything is written: a reference
    in no saved folder, a carried member with a reference's bytes, and two names, one carried and one a reference, one
    name on a case-insensitive filesystem."""
    referenced = manifest["referenced"]
    action = ("this is not a study container `mime pack` wrote; do not unpack it: ask its sender to pack the folder "
              "again with `mime pack <folder>`")
    carried = {held["sha256"]: name for name, held in manifest["files"].items()}
    problems = [refuse("bundle.unsafe", RefusalClass.USER,
                       f"the reference {path!r} {STUDY.name_problem(path) or 'is in no saved folder of the study'}",
                       action)
                for path in sorted(referenced) if STUDY.name_problem(path) is not None or saved_folder_of(path) is None]
    problems += [refuse("bundle.unsafe", RefusalClass.USER,
                        f"the study container carries the bytes of the reference {path!r} as "
                        f"{carried[held['sha256']]!r}", action)
                 for path, held in sorted(referenced.items()) if held["sha256"] in carried]
    problems += [refuse("bundle.unsafe", RefusalClass.USER,
                        f"{one!r} and {other!r}, one carried and one a reference, are one name on a case-insensitive "
                        f"filesystem", action)
                 for one, other in collisions([*manifest["files"], *referenced])
                 if (one in referenced) != (other in referenced)]
    return problems


def _folder_problems(manifest: StudyManifest, forms: Mapping[str, SavedForm]) -> list[Refusal]:
    """Each saved folder's files against what its saved form names (those carried and those that travel as references
    together), and each reference against its lock: its asset holding the same bytes, and the lock's references the
    manifest's."""
    action = ("this is not a study container `mime pack` wrote; do not unpack it: ask its sender to pack the folder "
              "again with `mime pack <folder>`")
    every = {*manifest["files"], *manifest["referenced"]}
    problems: list[Refusal] = []
    inside = {name for name in every if saved_folder_of(name) is not None}
    expected = {f"{where}/{name}" for where, form in forms.items() for name in named_files(form)}
    if inside != expected:
        extra, missing = sorted(inside - expected), sorted(expected - inside)
        problems.append(refuse("bundle.unsafe", RefusalClass.USER,
                               f"the study container's files in its saved folders are not the ones their saved "
                               f"experiments name: {len(extra)} they do not name "
                               f"({', '.join(repr(each) for each in extra[:3])}), {len(missing)} they name and the "
                               f"container lacks", action))
    for where, form in forms.items():
        listed = form.lock.get("referenced_files", {})
        assets = {held["name"]: held for held in form.lock.get("assets", [])}
        mine = {name.removeprefix(f"{where}/"): held for name, held in manifest["referenced"].items()
                if (saved_folder_of(name) or ("", ""))[0] == where}
        for path, held in sorted(mine.items()):
            asset = assets.get(held["name"])
            if listed.get(path) != held["name"] or asset is None or \
                    (asset["sha256"], asset["size"]) != (held["sha256"], held["size"]):
                problems.append(refuse("bundle.manifest", RefusalClass.USER,
                                       f"the manifest's reference {plain(where)}/{plain(path)} is not its lock's",
                                       action))
        if set(listed) != set(mine):
            problems.append(refuse("bundle.manifest", RefusalClass.USER,
                                   f"the manifest's references in {plain(where)}/ are not the files its lock says "
                                   f"travel as references", action))
    return problems


def unpack_study(container: str | os.PathLike[str], into: str | os.PathLike[str]) -> UnpackedStudy:
    """Check the study container and write its study folder to ``into``, which must not exist (or be an empty folder).
    Every member is checked as a bundle's are (its name against the study layout, its kind, its hash, written by this
    code's own opens into a hidden folder beside ``into``); then the study written there is read as ``StudyResult.read``
    reads one (its layout, its closed manifest, each group file's sha256), held to the study's own checks (every member
    done, the base's input hash, each figure's data), and each saved folder read as a saved form whose files are exactly
    the ones it names, its references its lock's; only then is it renamed to ``into``. Each file that travels as a
    reference is placed from the asset cache where it is cached. Raises ``BuildRefused`` (or the study reader's
    ``ArgumentRefused``)."""
    from mime.results._study_read import read_study

    refused = destination_problem(into)
    if refused is not None:
        raise refused
    destination = Path(os.path.abspath(into))
    staging = staging_of(destination)
    try:
        with zipfile.ZipFile(container) as archive:
            later = unknown_format(archive, STUDY)
            if later is not None:
                raise later
            refusals = member_kind_problems(archive, STUDY)
            if refusals:
                raise BuildRefused(refusals)
            manifest = cast(StudyManifest, read_manifest(archive, STUDY, lambda held: StudyManifest))
            unlisted = members_listed(archive, manifest["files"], STUDY)
            if unlisted is not None:
                raise unlisted
            notice, problems = notice_of(archive, manifest["notice"], STUDY)
            unsafe = _carried_reference_problems(manifest)
            if unsafe:
                raise BuildRefused([*problems, *unsafe])
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.mkdir()
            problems += write_members(archive, manifest["files"], staging, STUDY)
            if problems:
                raise BuildRefused(problems)
        document = read_study(staging).record
        folders = saved_folders(document)
        forms = {where: read_folder(staging / where, absent_referenced=True) for where in folders}
        problems = study_problems(staging, document, forms[BASE_DIR].lock, manifest["name"])
        problems += _folder_problems(manifest, forms)
        if problems:
            raise BuildRefused(problems)
        if manifest["assets"] != _assets(staging, folders):
            raise BuildRefused([refuse("bundle.manifest", RefusalClass.USER,
                                       "the manifest's asset records are not those of the study's locks",
                                       "this is not a study container `mime pack` wrote: ask its sender to pack the "
                                       "folder again with `mime pack <folder>`")])
        placed: list[str] = []
        missing: list[str] = []
        for where in folders:
            held = place_referenced(staging / where)
            placed += [f"{where}/{path}" for path in held.placed]
            missing += [f"{where}/{path}" for path, _, _ in held.missing]
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
    except zipfile.BadZipFile as error:
        raise unreadable(error, STUDY) from error
    except RecursionError:
        raise BuildRefused([refuse("bundle.manifest", RefusalClass.USER,
                                   "a file of the study container is nested too deeply to be read",
                                   "this is not a study container `mime pack` wrote: ask its sender to pack the folder "
                                   "again with `mime pack <folder>`")]) from None
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return UnpackedStudy(manifest, Path(into), notice, {where: form.lock for where, form in forms.items()},
                         tuple(placed), tuple(missing))


# --- verifying ------------------------------------------------------------------------------------------------------


#: MIME's words after a study's verification: what it is, and what it is not.
ACCEPTANCE = ("An acceptance: the study's files are the ones its run wrote, and its experiments reproduce here. It bounds "
              "nothing about physics, and it does not run the study again: `mime study run <folder>/study.mime.yaml "
              "--out <a new folder>` does.")


@dataclass(frozen=True)
class StudyVerified:
    """A study folder verified: its folder, and each of its saved folders with whether it reproduced and its
    verification's text (and its load report's, where the load reported anything)."""

    folder: Path
    folders: tuple[tuple[str, bool, str], ...]

    @property
    def reproduced(self) -> bool:
        return all(reproduced for _, reproduced, _ in self.folders)

    def lines(self) -> list[str]:
        head = (f"\N{CHECK MARK} study {plain(self.folder.name)}: its files are its run's, and its "
                f"{len(self.folders)} saved experiment{'s' if len(self.folders) != 1 else ''} reproduce"
                if self.reproduced else f"\N{BALLOT X} study {plain(self.folder.name)}: "
                f"{sum(1 for _, held, _ in self.folders if not held)} of its saved experiments did not reproduce")
        lines = [head]
        for where, reproduced, text in self.folders:
            lines.append(f"  - {plain(where)}/: {'reproduced' if reproduced else 'did not reproduce'}")
            if not reproduced:
                lines += [f"    {line}" for line in text.splitlines()]
        return lines


def verify_study(folder: str | os.PathLike[str]) -> StudyVerified:
    """Verify the study folder ``folder``: it is read as ``StudyResult.read`` reads one, held to the study's own
    checks (every member done, the base's input hash, each figure's data), and each of its saved folders (the base, then
    each group's own build) loaded, rebuilt and rerun against its reference outputs in the precision its lock records,
    as ``mime verify`` verifies a saved folder. A study its checks refuse raises, and nothing is run."""
    from mime.env import apply_lock_precision
    from mime.experiment import Experiment
    from mime.results._study_read import read_study

    root = Path(os.path.abspath(folder))
    document = read_study(root).record
    folders = saved_folders(document)
    forms = {where: read_folder(root / where) for where in folders}
    refusals = study_problems(root, document, forms[BASE_DIR].lock, root.name)
    if refusals:
        raise BuildRefused(refusals)
    found: list[tuple[str, bool, str]] = []
    for where in folders:
        apply_lock_precision(root / where)
        experiment = Experiment.load(root / where)
        report = experiment.build().verify()
        load = experiment.load_report.text() if experiment.load_report is not None else ""
        found.append((where, report.reproduced, "\n".join(text for text in (load, report.text()) if text)))
    return StudyVerified(root, tuple(found))


def study_container_name(container: str | os.PathLike[str]) -> str:
    """The study's name a study container's manifest gives, written as a file name: where ``pull`` unpacks it."""
    try:
        with zipfile.ZipFile(container) as archive:
            name = cast(StudyManifest, read_manifest(archive, STUDY, lambda held: StudyManifest))["name"]
    except zipfile.BadZipFile as error:
        raise unreadable(error, STUDY) from error
    return name if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) else "study"


__all__ = ["ACCEPTANCE", "BLOCK", "BLOCK_CONTAINER", "BLOCK_SUFFIX", "IN_STUDY", "KINDS", "STUDY", "STUDY_CONTAINER", "STUDY_PREFIX",
           "STUDY_SUFFIX", "StudyManifest", "StudyVerified", "UnpackedStudy", "container_kind", "figure_problems", "incomplete_problem",
           "input_hash_problems", "is_publishing_folder", "is_study_folder", "kind_of_format", "kind_of_name", "pack",
           "pack_studies", "pack_study", "publishing_entries", "saved_folder_of", "saved_folders", "study_container_name", "study_folders",
           "study_layout_problem", "study_name", "study_problems", "unpack_study", "verify_study"]
