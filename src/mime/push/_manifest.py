"""``mime.toml``: one manifest for every kind of shared folder, beside what it describes and outside it.

``format`` (``mime.toml/1``), exactly one kind table and ``[record]``. **Everything outside ``[record]`` defines the thing;
``[record]`` is metadata**, never in a container, so editing it is a metadata edit of the record and never a new version
(Zenodo's metadata is editable and its files immutable). A bundle's definition is ``[bundle] licence``, at a saved folder's
root; a study's is ``[study] licence``, at a study folder's root; each equals the licence its experiment declares, which
the document already holds, so ``mime pack`` carries no ``mime.toml`` and neither carries nor refuses it. A block's tables
(``[block]``, ``[requires]``, ``[sockets]``, ``[[assets]]``, ``[licences.*]``, ``[[permissions]]``, ``[[validity]]``,
``[cases]``) are reserved: this MIME refuses them by name (``manifest.kind``).

**``sharing.yaml`` is no longer read.** A folder holding only it is refused with the route (``manifest.missing``: ``mime
push <folder> --setup`` writes ``mime.toml`` from it); one holding both is refused (``manifest.both``), naming the one to
delete. MIME never deletes a person's file.
"""

from __future__ import annotations

import datetime
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from mime._shell import quoted
from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._sharing import DEFAULT_LICENCE, RECORD_KEYS, Problem, Reader, Sharing, licence_of, record_of
from mime.push._toml import key_lines, value_text
from mime.saved._versions import newer_form_refusal

MANIFEST_FILE = "mime.toml"
MANIFEST_FORMAT = "mime.toml/1"
MANIFEST_FAMILY = "toml"
#: The record's metadata before ``mime.toml``, which this MIME no longer reads.
OLD_FILE = "sharing.yaml"
BUNDLE, STUDY = "bundle", "study"
KINDS = (BUNDLE, STUDY)
#: A block's tables (P4's), reserved: refused by name until this MIME reads a block.
BLOCK_TABLES = ("block", "requires", "sockets", "assets", "licences", "permissions", "validity", "cases")
_TOP = ("format", *KINDS, "record")
#: What each kind's folder is, said where a kind table does not match its folder.
_FOLDER_WORDS = {BUNDLE: "a saved folder (Experiment.save writes one)",
                 STUDY: "a study folder (Study.run_to writes one), or a publishing folder of study folders"}


@dataclass(frozen=True)
class Manifest:
    """``mime.toml`` as read: where it is, its kind, and ``[record]`` with the kind table's licence."""

    path: Path
    kind: str
    record: Sharing


def manifest_path(folder: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(folder)) / MANIFEST_FILE


def folder_kind(root: Path) -> str | None:
    """``study`` for a study folder or a publishing folder (one that holds study folders), ``bundle`` for a saved folder,
    ``None`` for another."""
    from mime.results._study_folder import STUDY_FILE
    from mime.saved._folder import DOCUMENT_FILE
    from mime.saved._studies import is_publishing_folder

    if (root / STUDY_FILE).is_file():
        return STUDY
    if (root / DOCUMENT_FILE).is_file():
        return BUNDLE
    return STUDY if is_publishing_folder(root) else None


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> BuildRefused:
    return BuildRefused([Refusal(code, category, message, action)])


def present(root: Path, *, new: bool = False, old_ignored: bool = False) -> Path:
    """``mime.toml`` of ``root``, where it is the folder's one manifest: refused where the folder holds ``sharing.yaml``
    (alone: ``manifest.missing`` with the route; beside it: ``manifest.both``) or none (``manifest.missing``). ``new``
    asks for the path a new one is written at, which only ``mime.toml`` beside ``sharing.yaml`` refuses; ``old_ignored``
    reads ``mime.toml`` beside ``sharing.yaml`` (``--setup`` checking the file it has just written from it)."""
    path, old = root / MANIFEST_FILE, root / OLD_FILE
    if path.is_file() and old.is_file() and not old_ignored:
        raise _refuse(
            "manifest.both", RefusalClass.USER,
            f"{root.name} holds both {MANIFEST_FILE} and {OLD_FILE}, which may disagree, and MIME reads only "
            f"{MANIFEST_FILE}",
            f"delete {old} once {path} holds what you want (`mime check {quoted(str(root))}` checks it): MIME merges "
            f"nothing and never deletes your file")
    if new or path.is_file():
        return path
    if old.is_file():
        raise _refuse(
            "manifest.missing", RefusalClass.USER,
            f"{root.name} holds {OLD_FILE}, which this MIME no longer reads: a record's metadata is kept in "
            f"{MANIFEST_FILE} now",
            f"`mime push {quoted(str(root))} --setup` writes {MANIFEST_FILE} from it, a field at a time in a terminal "
            f"(`mime push {quoted(str(root))} --to <service> --init` writes it without questions); then delete "
            f"{OLD_FILE} yourself: MIME never deletes your file")
    raise _refuse(
        "manifest.missing", RefusalClass.USER,
        f"{root.name} has no {MANIFEST_FILE}: the record's title, creators and licence are kept there",
        f"write it from what the folder knows, `mime push {quoted(str(root))} --to <service> --init`, then name its "
        f"creators; or `mime push {quoted(str(root))} --setup`, a field at a time")


def read_toml(path: Path) -> tuple[str, dict[str, object]]:
    """The text of the TOML file ``path`` and what it reads as; refused where it does not read (``manifest.read``, or
    ``defaults.file`` for a person's defaults)."""
    code = "manifest.read" if path.name == MANIFEST_FILE else "defaults.file"
    try:
        text = path.read_bytes().decode("utf-8")
        return text, tomllib.loads(text)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        why = f": {error}" if isinstance(error, tomllib.TOMLDecodeError) else f" ({type(error).__name__})"
        raise _refuse(code, RefusalClass.USER, f"{path} is not TOML that reads{plain(why)}",
                      f"correct it by hand (TOML's form: https://toml.io), then run the command again") from None


def kind_of(held: Mapping[str, object], path: Path, root: Path) -> str:
    """The kind ``mime.toml`` (``held``) names, the folder's own; refused (``manifest.kind``) where it names none, more
    than one, a block, or one its folder is not."""
    named = [kind for kind in KINDS if kind in held]
    reserved = [table for table in BLOCK_TABLES if table in held]
    if reserved:
        raise _refuse("manifest.kind", RefusalClass.USER,
                      f"{path} holds a block's table{'s' if len(reserved) > 1 else ''} "
                      f"({', '.join(f'[{each}]' for each in reserved)}), which this MIME does not read yet",
                      f"remove those tables, then `mime check {quoted(str(root))}`; a block is checked by the MIME "
                      f"release that reads blocks")
    if len(named) != 1:
        raise _refuse("manifest.kind", RefusalClass.USER,
                      f"{path} names {'no kind' if not named else 'more than one kind, ' + ', '.join(named)}: it "
                      f"holds exactly one kind table, [bundle] for a saved folder or [study] for a study folder or a "
                      f"publishing folder of them",
                      f"keep the one table its folder is, `[{folder_kind(root) or BUNDLE}]`, and its licence")
    (kind,) = named
    found = folder_kind(root)
    if found != kind:
        raise _refuse("manifest.kind", RefusalClass.USER,
                      f"{path} names [{kind}], and {root.name} is " +
                      (_FOLDER_WORDS[found] if found else "neither a saved folder nor a study folder"),
                      f"name its own kind, `[{found}]`" if found else
                      f"put it beside what it describes, {_FOLDER_WORDS[kind]}, then `mime check <that folder>`")
    return kind


def manifest_problems(folder: str | os.PathLike[str], *,
                      old_ignored: bool = False) -> tuple[Manifest | None, list[Problem]]:
    """``mime.toml`` of ``folder`` read, and every problem of its keys (each key's check), the manifest ``None`` where
    there is one. What makes it unreadable is refused at once: no ``mime.toml``, ``sharing.yaml`` alone or beside it
    (:func:`present`), TOML that does not read (``manifest.read``), a newer format (``load.newer-form``), a kind its folder
    is not (``manifest.kind``). ``old_ignored`` as :func:`present`'s."""
    root = Path(os.path.abspath(folder))
    path = present(root, old_ignored=old_ignored)
    _, held = read_toml(path)
    if (newer := newer_form_refusal(held.get("format"), MANIFEST_FAMILY, (MANIFEST_FORMAT,), path.name)) is not None:
        raise BuildRefused([newer])
    kind = kind_of(held, path, root)
    top = Reader(path, table="")
    top.mapping(held, "", _TOP)
    if held.get("format") != MANIFEST_FORMAT:
        top.problem("format", f"is {plain(repr(held.get('format')))}, and this MIME reads {MANIFEST_FORMAT}",
                    f'format = "{MANIFEST_FORMAT}"')
    table = Reader(path, table=kind)
    definition = table.mapping(held[kind], "", ("licence",))
    licence = licence_of(table, definition) if definition is not None else None
    reader = Reader(path)
    record = record_of(reader, held.get("record"), licence)
    problems = [*top.problems, *table.problems, *reader.problems]
    return (Manifest(path, kind, record) if record is not None and not problems else None), problems


def read_manifest(folder: str | os.PathLike[str]) -> Manifest:
    """``mime.toml`` of ``folder``, checked: refused with every reason at once."""
    manifest, problems = manifest_problems(folder)
    if manifest is None:
        raise BuildRefused([each.refusal for each in problems])
    return manifest


def read_record(folder: str | os.PathLike[str]) -> Sharing:
    """``[record]`` of a saved, study or publishing folder's ``mime.toml``, with its kind table's licence: what push
    reads."""
    return read_manifest(folder).record


def toml_shaped(value: object) -> object:
    """``value`` (read from YAML) as TOML holds it: a ``None`` left out (TOML has no null), a day as its ISO text."""
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, list):
        return [toml_shaped(each) for each in cast(list[object], value) if each is not None]
    if isinstance(value, dict):
        return {str(key): toml_shaped(each) for key, each in cast(dict[object, object], value).items()
                if each is not None}
    return value


def toml_problems(value: object, where: str = "") -> list[str]:
    """What TOML cannot hold of ``value`` as YAML read it, each as ``<where> <what>``, so that nothing is dropped or
    changed in the move: a null inside an array (a mapping's null is a key left out, which holds nothing), a key that is
    not text (YAML reads ``1``, ``yes`` and a date as a number, a boolean and a date: TOML's keys are text, and the keys
    ``1`` and ``'1'`` would be one key), and a value TOML has no form for (:func:`toml_holdable`)."""
    found: list[str] = []
    if isinstance(value, list):
        for index, each in enumerate(cast(list[object], value)):
            if each is None:
                found.append(f"{where}[{index}] is null, which TOML cannot hold in an array")
            else:
                found += toml_problems(each, f"{where}[{index}]")
    elif isinstance(value, dict):
        for key, each in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                found.append(f"{where or 'the top'} holds the key {plain(repr(key))}, which YAML reads as "
                             f"{type(key).__name__} and TOML's keys cannot be (quote it, `'{plain(str(key))}'`)")
            else:
                found += toml_problems(each, f"{where}.{plain(key)}" if where else plain(key))
    elif value is not None and not toml_holdable(value):
        found.append(f"{where} is a value TOML cannot hold (a YAML tag's binary, set or pairs, or text UTF-8 cannot "
                     f"write)")
    return found


def old_values(root: Path) -> dict[str, object]:
    """What ``sharing.yaml`` beside the folder holds, by key (``licence`` among them), as ``mime.toml`` holds it: the
    values ``--setup`` and ``--init`` write ``mime.toml`` from. Its own checks are ``mime.toml``'s, after. **Nothing is
    dropped silently**: a key outside ``[record]``'s and ``licence``, and what :func:`toml_problems` finds, refuse the
    move (``manifest.read``) before anything is written, each named."""
    path = root / OLD_FILE
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise _refuse("manifest.read", RefusalClass.USER,
                      f"{path} is not YAML that reads ({type(error).__name__}), so {MANIFEST_FILE} cannot be written "
                      f"from it",
                      f"correct it by hand, or move it aside and write {MANIFEST_FILE} afresh, `mime push "
                      f"{quoted(str(root))} --setup`") from None
    held = cast(dict[object, object], loaded) if isinstance(loaded, dict) else {}
    odd = toml_problems(held)
    odd += [f"{plain(str(key))} is not a key sharing.yaml held, and {MANIFEST_FILE} has no place for it"
            for key in held if isinstance(key, str) and key not in ("format", "licence", *RECORD_KEYS)]
    if odd:
        raise _refuse("manifest.read", RefusalClass.USER,
                      f"{path} holds what {MANIFEST_FILE} cannot carry over, so nothing is written: " + "; ".join(odd),
                      f"correct it by hand (delete the key, quote the key, fill the null), or move it aside and write "
                      f"{MANIFEST_FILE} afresh, `mime push {quoted(str(root))} --setup`")
    return {str(key): toml_shaped(value) for key, value in held.items() if key != "format" and value is not None}


def toml_holdable(value: object) -> bool:
    """Whether TOML holds ``value`` (text, a number, a boolean, a day, and arrays and tables of them: no null), every text
    in it writable as UTF-8."""
    if isinstance(value, list):
        return all(toml_holdable(each) for each in cast(list[object], value))
    if isinstance(value, dict):
        return all(toml_holdable(each) for each in cast(dict[str, object], value).values())
    if isinstance(value, str):
        return not any(0xD800 <= ord(each) <= 0xDFFF for each in value)
    return isinstance(value, (int, float, datetime.date))


#: Each key of ``[record]`` explained where ``--init`` writes it, and its line where it holds no value.
_EXPLAINED: Mapping[str, tuple[tuple[str, ...], str | None]] = {
    "title": ((), None),
    "description": (("A paragraph shown above the card, which is the record's description; empty for the card alone.",),
                    None),
    "creators": (("Each person who made it, as the record names them: family and given names, an ORCID iD where they",
                  "have one, an affiliation (with its ROR identifier where you know it). Everyone named must have",
                  "agreed to be named. One a line, for example:",
                  '  { family = "Carberry", given = "Josiah", orcid = "0000-0002-1825-0097", '
                  'affiliation = "Brown University" },'), None),
    "contributors": (("Others credited, each with a DataCite contributor type (DataCurator, Supervisor,",
                      "HostingInstitution, ...). The rights holder of each third-party file is added by push from the",
                      "files' terms."), None),
    "keywords": (("Searchable words; push adds mime-experiment.",), None),
    "related": (("Papers and other works: a DOI or an https:// address, and DataCite's relation (IsSupplementTo for a",
                 "paper it supports, References, IsDerivedFrom, ...). push adds the MIME release it requires."), None),
    "also_cite": (("Works whoever uses this is asked to cite as well (a paper, once published: a metadata edit): a DOI",
                   "or an https:// address, and how it is cited. The card ends with them, and push states each as",
                   "IsSupplementTo unless related names it."), None),
    "uploader": (("The person whose token pushes, and their institution: a declared permission's grantee must be one",
                  "of them."), '# uploader = { name = "<family name>, <given name>", institution = "<your institution>" }'),
    "funding": (("Who funded the work: each funder's name (with its ROR identifier where you know it), and the",
                 "award's number and title. For example:",
                 '  { funder = "European Commission", ror = "00k4n6c32", award = "101001234", '
                 'title = "<the award\'s title>" },'), None),
    "conference": (("A conference it was shown at (Zenodo's meeting field): title, acronym, dates, place, url,",
                    "session, part."),
                   '# conference = { title = "<the conference\'s name>", acronym = "<...>", dates = "3-7 May 2027", '
                   'place = "<city>" }'),
    "fields": (("Any other field of the record, in the records API's own names and shapes, checked before anything",
                "is sent: subjects, dates, languages, references, additional_titles, additional_descriptions, and",
                "Zenodo's own journal:journal, imprint:imprint, thesis:university and thesis:type. For example:",
                '  fields = { languages = [{ id = "eng" }], dates = [{ date = "2026-10", type = { id = "collected" } }] }'),
               None),
    "access": (("Who may read the record and its files once published, where that is not everyone (public, or",
                "restricted), and an embargo's end and reason."), None),
}
_EMPTY: Mapping[str, object] = {"description": "", "creators": [], "contributors": [], "keywords": [], "related": [],
                                "also_cite": [], "funding": [], "fields": {}}


def manifest_text(kind: str, name: str, licence: str | None, values: Mapping[str, object] | None = None, *,
                  fallback: str = DEFAULT_LICENCE) -> str:
    """The ``mime.toml`` ``--init`` writes for a folder of ``kind``: the folder's name as the title, the licence its
    experiment declares (``fallback`` where it declares none), no creator yet, each key explained where it stands.
    ``values`` (a person's defaults, ``sharing.yaml``'s values, or what ``--setup`` asked) fills a key where it stands."""
    held: Mapping[str, object] = values or {}
    whose = "experiment" if kind == BUNDLE else "study's base experiment"
    declared = (f"The {whose} declares {licence}." if licence else
                f"The {whose} declares none: {fallback} is the suggestion, and push asks the experiment to declare "
                f"it too.")
    lines = [
        "# The record's manifest for `mime push`: what is shared ([" + kind + "]) and the record's metadata ([record]).",
        "# [record] is metadata: a change there is a metadata edit of the record, never a new version. `mime pack`",
        "# neither carries nor refuses this file. Edit it by hand or with `mime push <folder> --setup`; `mime check",
        "# <folder>` checks it.",
        f'format = "{MANIFEST_FORMAT}"', "",
        f"# The {whose}'s own licence, which covers its files: it must equal the licence the experiment declares",
        f"# (exp.licence(...), bound to its files when it is saved). {declared}",
        f"[{kind}]", f"licence = {value_text(held.get('licence', licence or fallback))}", "", "[record]"]
    for key, (explained, absent) in _EXPLAINED.items():
        if key == "access" and key not in held:
            continue
        lines += [f"# {line}" for line in explained]
        value = held.get(key, name if key == "title" else _EMPTY.get(key))
        lines += key_lines(key, value) if value is not None else [absent or f"# {key} = ..."]
    return "\n".join([*lines, ""])


__all__ = ["BLOCK_TABLES", "BUNDLE", "KINDS", "MANIFEST_FILE", "MANIFEST_FORMAT", "OLD_FILE", "STUDY", "Manifest",
           "folder_kind", "kind_of", "manifest_path", "manifest_problems", "manifest_text", "old_values", "present",
           "read_manifest", "read_record", "read_toml", "toml_holdable", "toml_problems", "toml_shaped"]
