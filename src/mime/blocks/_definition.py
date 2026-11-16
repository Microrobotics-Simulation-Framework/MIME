"""A block's definition: ``mime.toml`` of a block folder, less ``[record]``.

``[block]`` names it (``id``, ``version``) and licenses its content (``licence``); ``[requires]`` names the distributions
its fragment's entries come from, with PEP 440 ranges; ``[sockets]`` what a user binds; ``[[assets]]`` the records of
the files it reads, whole, never bytes; ``[licences.<id>]`` its custom licences; ``[[permissions]]`` permissions for
its assets' files; ``[[validity]]`` the lines its author states; ``[cases]`` what its cases show. Every key's problem is
gathered before any is refused (``manifest.key``, as ``[record]``'s are), and a socket's is ``block.socket``.

**The content hash** names what the block is: the sha256 of a canonical JSON of the definition (parsed, so comments and
order do not count, each asset's locations and request left out: a new home for a file is not a new block) and the
sha256 of each content file (``block.mime.yaml``, ``records/*``, ``descriptions/*``). The cases and the assets' bytes
are outside it: the cases test the content, the assets are named by sha256 in it.

**The tier** is computed from the licence, never written: open or conditional is ``copied`` (the expanded declarations
go into the user's document), anything else ``referenced``."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from mime._manifest_file import BLOCK_TABLES, MANIFEST_FAMILY, MANIFEST_FILE, MANIFEST_FORMAT
from mime._terminal import plain
from mime.assets import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._sharing import Problem, Reader, Sharing, as_text, record_of
from mime.saved._versions import newer_form_refusal
from mime.terms import (
    CUSTOM_STATUSES,
    CustomLicence,
    Evidence,
    Permission,
    Scope,
    Verdict,
    ids_of,
    licence_problem,
    licence_verdict,
    permission_problems,
)

FRAGMENT_FILE = "block.mime.yaml"
RECORDS = "records"
DESCRIPTIONS = "descriptions"
ASSETS = "assets"
CASES = "cases"
#: The distribution MIME is, which every block requires.
MIME_DISTRIBUTION = "mime-engine"
#: The kinds of thing a socket takes: MIME's thing kinds.
SOCKET_KINDS = ("structure", "medium", "body", "apparatus")
#: A socket that would place a block on a frame of the user's: refused until a mount on another apparatus's frame is
#: built (the apparatus lane's AP6), which is the one change that lifts it.
FRAME_KIND = "frame"
FRAME_ROUTE = ("a frame socket waits for the mount of one apparatus on another's frame (the apparatus lane's AP6): "
               "until then a block's things are placed in the world, and nothing in an experiment names a frame of "
               "the user's a block could bind")
#: ``exp.use``'s own keywords, which a socket cannot be named.
RESERVED_SOCKETS = ("name", "accept_licences", "yes", "force")
_ID = re.compile(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*")
_SOCKET = re.compile(r"[a-z][a-z0-9_]*")
#: A distribution's name as PEP 508 writes it.
_DISTRIBUTION = re.compile(r"[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: The block's id, version and licence as they are written, for a problem's line.
LINES = {"id": 'id = "<namespace>.<name>"', "version": 'version = "0.1.0"', "licence": 'licence = "CC-BY-4.0"',
         "mime-engine": 'mime-engine = ">=0.3.0.dev1,<0.4"', "shows": 'shows = "<what the cases show, a paragraph>"'}
def id_problem(text: str) -> str | None:
    """Why ``text`` is not a block's id, ``<namespace>.<name>``, or ``None``."""
    if _ID.fullmatch(text):
        return None
    return (f"{plain(repr(text))} is not <namespace>.<name> (lower-case letters, digits and '_', each starting with a "
            f"letter)")


def version_problem(text: str) -> str | None:
    """Why ``text`` is not a PEP 440 version, or ``None``."""
    try:
        Version(text)
    except InvalidVersion:
        return f"{plain(repr(text))} is not a PEP 440 version"
    return None


#: The content hash's recipe, so that a later one is never compared with this one.
CONTENT_RECIPE = "mime.block-content/1"

type Tier = Literal["copied", "referenced"]


@dataclass(frozen=True)
class Validity:
    line: str
    reason: str


@dataclass(frozen=True)
class Definition:
    """A block's definition as read: where it is (its folder and ``mime.toml``), its id, version and licence, its
    requirements (each distribution's name and specifiers), its sockets (each name and its kind), its asset records,
    custom licences, permissions and validity lines, what its cases show, its ``[record]`` where it reads (``None``
    where it has none or it does not read), and ``held``, the definition's tables as parsed (``[record]`` left out):
    what the content hash covers."""

    folder: Path
    id: str
    version: str
    licence: str
    requires: tuple[tuple[str, str], ...]
    sockets: Mapping[str, str]
    assets: tuple[AssetRef, ...]
    licences: Mapping[str, CustomLicence]
    permissions: tuple[Permission, ...]
    validity: tuple[Validity, ...]
    shows: str | None
    record: Sharing | None
    held: Mapping[str, object]

    @property
    def path(self) -> Path:
        return self.folder / MANIFEST_FILE

    @property
    def words(self) -> str:
        """The block as a message names it: ``block labx.demo_probe 1.2.0``."""
        return f"block {self.id} {self.version}"

    @property
    def name(self) -> str:
        """The id's name part, which names a use by default."""
        return self.id.split(".", 1)[1]

    @property
    def verdict(self) -> Verdict:
        return licence_verdict(self.licence, self.licences)

    @property
    def tier(self) -> Tier:
        return "copied" if self.verdict.tier in ("open", "conditional") else "referenced"

    def content_files(self) -> dict[str, str]:
        """Each content file below the folder, by its path as ``/`` writes it, with the sha256 of its bytes."""
        return content_files(self.folder)

    def content_hash(self) -> str:
        return content_hash(self.held, self.content_files())


def content_files(folder: Path) -> dict[str, str]:
    """``block.mime.yaml`` and every file under ``records/`` and ``descriptions/``, by path, with their sha256s."""
    found: dict[str, str] = {}
    candidates = [folder / FRAGMENT_FILE]
    for below in (RECORDS, DESCRIPTIONS):
        if (folder / below).is_dir():
            candidates += sorted(path for path in (folder / below).rglob("*") if path.is_file())
    for path in candidates:
        if path.is_file():
            found[path.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(found.items()))


def _identity(held: Mapping[str, object]) -> dict[str, object]:
    """The definition as the content hash reads it: each asset record without its locations and request, which are
    where the file is and who to ask, not what it is."""
    found = dict(held)
    assets = found.get("assets")
    if isinstance(assets, list):
        found["assets"] = [{key: value for key, value in cast(dict[str, object], each).items()
                            if key not in ("locations", "request")} if isinstance(each, dict) else each
                           for each in cast(list[object], assets)]
    return found


def content_hash(held: Mapping[str, object], files: Mapping[str, str]) -> str:
    """The sha256 of the definition's tables (``held``, ``[record]`` left out) and the content files' sha256s."""
    body = {"recipe": CONTENT_RECIPE, "definition": _identity(held), "files": dict(sorted(files.items()))}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_block_folder(root: Path) -> bool:
    """Whether ``root`` is a block folder: one holding ``block.mime.yaml``."""
    return (root / FRAGMENT_FILE).is_file()


def _table(reader: Reader, value: object) -> dict[str, object] | None:
    """``value`` as a table of keys of the author's choosing, or ``None`` with its problem."""
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    reader.problem("the table", f"is {type(value).__name__}, not a table of keys")
    return None


def _requires(reader: Reader, value: object) -> tuple[tuple[str, str], ...]:
    held = _table(reader, value)
    if held is None:
        return ()
    found: list[tuple[str, str]] = []
    for name, spec in held.items():
        if name == "blocks":
            reader.problem("blocks", "names blocks a block requires, which waits for assemblies (a1): a block in "
                                     "dev2 requires distributions alone", "remove [requires.blocks]")
            continue
        if not _DISTRIBUTION.fullmatch(name):
            reader.problem(plain(name), "is not a distribution's name as PEP 508 writes it (letters, digits, '.', '_' "
                                        "and '-')")
            continue
        if not isinstance(spec, str):
            reader.problem(plain(name), f"is {type(spec).__name__}, not a text of PEP 440 version specifiers",
                           f'{plain(name)} = ">=1.0,<2"')
            continue
        if ";" in spec or "@" in spec:
            reader.problem(plain(name), "holds a marker or an address: a block requires a distribution by its "
                                        "versions alone", f'{plain(name)} = ">=1.0,<2"')
            continue
        try:
            SpecifierSet(spec)
        except InvalidSpecifier:
            reader.problem(plain(name), f"{plain(repr(spec))} is not a set of PEP 440 version specifiers",
                           f'{plain(name)} = ">=1.0,<2"')
            continue
        found.append((name, spec))
    names = [canonicalize_name(name) for name, _ in found]
    if len(set(names)) != len(names):
        reader.problem("the table", "names one distribution twice (PEP 503 reads names case- and separator-blind)")
    if not any(canonicalize_name(name) == MIME_DISTRIBUTION for name in held):
        reader.problem(MIME_DISTRIBUTION, "is missing: every block requires the MIME releases it is written for",
                       LINES["mime-engine"])
    return tuple(found)


def _socket_problem(refusal: Refusal) -> Problem:
    return Problem(refusal, None, table="sockets")


def _sockets(reader: Reader, value: object, path: Path) -> tuple[dict[str, str], list[Problem]]:
    held = _table(reader, value)
    found: dict[str, str] = {}
    problems: list[Problem] = []
    where = f"{path.name}, [sockets]"
    for name, socket in (held or {}).items():
        if not _SOCKET.fullmatch(name) or name in RESERVED_SOCKETS:
            problems.append(_socket_problem(Refusal(
                "block.socket", RefusalClass.USER,
                f"{where} {plain(name)}: is not a socket's name (lower-case letters, digits and '_', starting with a "
                f"letter, and none of exp.use's own keywords: {', '.join(RESERVED_SOCKETS)})",
                f"rename it in {path}, `<another name> = {{ kind = \"...\" }}`, and in the fragment where it is "
                f"used")))
            continue
        table = reader.mapping(socket, f"{name}", ("kind", "of"))
        if table is None:
            continue
        kind = table.get("kind")
        if kind == FRAME_KIND:
            problems.append(_socket_problem(Refusal(
                "block.socket", RefusalClass.USER, f"{where} {plain(name)}: is a frame socket, refused in this "
                                                   f"release: {FRAME_ROUTE}",
                f"remove `{plain(name)} = {{ kind = \"frame\", ... }}` from {path}; place the block's things in the "
                f"world")))
            continue
        if kind not in SOCKET_KINDS:
            problems.append(_socket_problem(Refusal(
                "block.socket", RefusalClass.USER, f"{where} {plain(name)}: its kind is {plain(repr(kind))}, and a "
                                                   f"socket's kind is one of {', '.join(SOCKET_KINDS)}",
                f"write it `{plain(name)} = {{ kind = \"body\" }}` in {path}")))
            continue
        if "of" in table:
            problems.append(_socket_problem(Refusal(
                "block.socket", RefusalClass.USER, f"{where} {plain(name)}: names `of`, which only a frame socket takes",
                f"write it `{plain(name)} = {{ kind = \"{kind}\" }}` in {path}")))
            continue
        found[name] = cast(str, kind)
    return found, problems


def _assets(reader: Reader, values: Sequence[object]) -> tuple[AssetRef, ...]:
    found: list[AssetRef] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            reader.problem(f"[{index}]", f"is {type(value).__name__}, not a table of an asset's record")
            continue
        try:
            ref = AssetRef.from_record(cast(dict[str, object], value))
        except BuildRefused as refused:
            for each in refused.refusals:
                reader.problem(f"[{index}]", plain(each.message))
            continue
        if ref.name in {each.name for each in found}:
            reader.problem(f"[{index}]", f"names the asset {plain(ref.name)} a second time")
            continue
        found.append(ref)
    return tuple(found)


def _licences(reader: Reader, value: object) -> dict[str, CustomLicence]:
    held = _table(reader, value)
    found: dict[str, CustomLicence] = {}
    for key, table in (held or {}).items():
        entry = reader.mapping(table, plain(key), ("title", "url", "redistribution"))
        if entry is None:
            continue
        redistribution = entry.get("redistribution", "unknown")
        licence = CustomLicence(key, cast(str, entry.get("title", "")), cast(str, entry.get("url", "")),
                                cast("Literal['not_permitted', 'unknown']", redistribution))
        problem = licence.problem() if redistribution in CUSTOM_STATUSES or not isinstance(redistribution, str) \
            else f"the custom licence {key}'s redistribution is one of {', '.join(CUSTOM_STATUSES)}"
        if problem is not None:
            reader.problem(plain(key), plain(problem))
            continue
        found[key] = licence
    return found


def _permissions(reader: Reader, values: Sequence[object], assets: Sequence[AssetRef]) -> tuple[Permission, ...]:
    found: list[Permission] = []
    known = {ref.sha256 for ref in assets}
    for index, value in enumerate(values):
        entry = reader.mapping(value, f"[{index}]", ("grantor", "grantee", "files", "scope", "date", "evidence"))
        if entry is None:
            continue
        scope, evidence = entry.get("scope"), entry.get("evidence")
        files = entry.get("files")
        if not isinstance(scope, dict) or not isinstance(evidence, dict) or not isinstance(files, list):
            reader.problem(f"[{index}]", "holds scope = { redistribute, modify, commercial }, evidence = { kind, "
                                         "pointer } and files = [<sha256>, ...]")
            continue
        held_scope = cast(dict[str, object], scope)
        held_evidence = cast(dict[str, object], evidence)
        permission = Permission(cast(str, entry.get("grantor")), cast(str, entry.get("grantee")),
                                tuple(cast(list[object], files)),
                                Scope(cast(bool, held_scope.get("redistribute")), cast(bool, held_scope.get("modify")),
                                      cast(bool, held_scope.get("commercial"))),
                                cast(str, as_text(entry.get("date"))),
                                Evidence(cast("Literal['public_statement', 'written_permission']",
                                              held_evidence.get("kind")), cast(str, held_evidence.get("pointer"))))
        named = [str(each) for each in cast(list[object], files)]
        problems = permission_problems(permission, named)
        problems += [f"the permission names the file {each}, which is no sha256 of the block's [[assets]]"
                     for each in named if _SHA256.fullmatch(each) and each not in known]
        for problem in problems:
            reader.problem(f"[{index}]", plain(problem))
        if not problems:
            found.append(permission)
    return tuple(found)


def _validity(reader: Reader, values: Sequence[object]) -> tuple[Validity, ...]:
    found: list[Validity] = []
    for index, value in enumerate(values):
        entry = reader.mapping(value, f"[{index}]", ("line", "reason"))
        if entry is None:
            continue
        line = reader.text(entry, "line", f"[{index}].", required=True)
        reason = reader.text(entry, "reason", f"[{index}].", required=True)
        for key, text in (("line", line), ("reason", reason)):
            if text is not None and not text.isprintable():
                reader.problem(f"[{index}].{key}", "is not one printable line of text")
                break
        else:
            if line is not None and reason is not None:
                found.append(Validity(line, reason))
    return tuple(found)


def _reader(path: Path, table: str) -> Reader:
    return Reader(path, table=table)


def read_definition(folder: str | os.PathLike[str], *, record: bool = True) -> tuple[Definition | None, list[Problem]]:
    """The definition of the block folder ``folder``, and every problem of its keys; the definition ``None`` where
    one is found. What makes the file unreadable is refused at once: no ``mime.toml``, TOML that does not read
    (``manifest.read``), a newer format (``load.newer-form``). ``record``: read ``[record]`` too (its problems are
    push's, which ``mime check`` lists and ``exp.use`` never asks for)."""
    from mime.push._manifest import read_toml

    root = Path(os.path.abspath(folder))
    path = root / MANIFEST_FILE
    if not path.is_file():
        raise BuildRefused([Refusal(
            "block.format", RefusalClass.USER, f"{root} holds no {MANIFEST_FILE}, so it is not a block folder: a block "
                                               f"is its {MANIFEST_FILE} ([block], [requires], [sockets], ...) and its "
                                               f"{FRAGMENT_FILE}",
            f"use a block folder, or write {MANIFEST_FILE} beside {FRAGMENT_FILE} (`mime check <folder>` checks "
            f"it)")])
    _, raw = read_toml(path)
    if (newer := newer_form_refusal(raw.get("format"), MANIFEST_FAMILY, (MANIFEST_FORMAT,), path.name)) is not None:
        raise BuildRefused([newer])
    top = Reader(path, table="")
    top.mapping(raw, "", ("format", *BLOCK_TABLES, "record"))
    if raw.get("format") != MANIFEST_FORMAT:
        top.problem("format", f"is {plain(repr(raw.get('format')))}, and this MIME reads {MANIFEST_FORMAT}",
                    f'format = "{MANIFEST_FORMAT}"')
    for other in ("bundle", "study"):
        if other in raw:
            top.problem(f"[{other}]", f"is another kind's table, and {root.name} is a block folder (it holds "
                                      f"{FRAGMENT_FILE}): a manifest names one kind", f"remove [{other}]")
    held = {key: as_text(value) for key, value in raw.items() if key != "record"}
    block = Reader(path, table="block")
    table = block.mapping(raw.get("block", {}), "", ("id", "version", "licence"))
    identifier = version = licence = None
    licences_reader = _reader(path, "licences")
    licences = _licences(licences_reader, raw.get("licences", {}))
    if table is not None:
        identifier = block.text(table, "id", "", required=True)
        if identifier is not None and (problem := id_problem(identifier)) is not None:
            block.problem("id", problem, LINES["id"])
            identifier = None
        version = block.text(table, "version", "", required=True)
        if version is not None and (problem := version_problem(version)) is not None:
            block.problem("version", problem, LINES["version"])
            version = None
        licence = block.text(table, "licence", "", required=True)
        if licence is not None and (problem := licence_problem(licence)) is not None:
            block.problem("licence", plain(problem), LINES["licence"])
            licence = None
        elif licence is not None:
            undefined = [each for each in ids_of(licence) if each.startswith("LicenseRef-") and each not in licences]
            if undefined:
                block.problem("licence", f"names {', '.join(undefined)}, which no [licences.<id>] defines",
                              f'[licences.{undefined[0]}] title = "<its title>" url = "https://..."')
                licence = None
    requires_reader = _reader(path, "requires")
    requires = _requires(requires_reader, raw.get("requires", {}))
    sockets_reader = _reader(path, "sockets")
    sockets, socket_problems = _sockets(sockets_reader, raw.get("sockets", {}), path)
    assets_reader = _reader(path, "assets")
    assets = _assets(assets_reader, assets_reader.items(raw, "assets"))
    permissions_reader = _reader(path, "permissions")
    permissions = _permissions(permissions_reader, permissions_reader.items(raw, "permissions"), assets)
    validity_reader = _reader(path, "validity")
    validity = _validity(validity_reader, validity_reader.items(raw, "validity"))
    cases_reader = _reader(path, "cases")
    cases = cases_reader.mapping(raw.get("cases", {}), "", ("shows",))
    shows = cases_reader.text(cases, "shows", "") if cases is not None else None
    if shows is not None and any(not line.isprintable() for line in shows.splitlines()):
        cases_reader.problem("shows", "is not printable text", LINES["shows"])
        shows = None
    record_reader = Reader(path)
    sharing = record_of(record_reader, raw.get("record"), licence) if record else None
    readers = (top, block, requires_reader, sockets_reader, assets_reader, licences_reader, permissions_reader,
               validity_reader, cases_reader)
    problems = [each for reader in readers for each in reader.problems] + socket_problems
    record_problems = record_reader.problems if record else []
    if problems or identifier is None or version is None or licence is None:
        return None, [*problems, *record_problems]
    definition = Definition(root, identifier, version, licence, requires, sockets, assets, licences, permissions,
                            validity, shows, sharing, held)
    return definition, list(record_problems)


def definition_of(folder: str | os.PathLike[str]) -> Definition:
    """The definition of ``folder``, checked: refused with every reason at once (``[record]``'s problems aside: a block
    is used, shown and checked whatever its record says, and pushed only once it reads)."""
    definition, problems = read_definition(folder, record=False)
    if definition is None:
        raise BuildRefused([each.refusal for each in problems])
    return definition


__all__ = ["ASSETS", "CASES", "CONTENT_RECIPE", "DESCRIPTIONS", "FRAGMENT_FILE", "FRAME_KIND", "FRAME_ROUTE",
           "MIME_DISTRIBUTION", "RECORDS", "RESERVED_SOCKETS", "SOCKET_KINDS", "Definition", "Tier", "Validity", "content_files", "content_hash", "definition_of",
           "id_problem", "is_block_folder", "read_definition", "version_problem"]
