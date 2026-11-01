"""What a saved folder may share: the document's licence, terms and permissions, each file's verdict, and the files
that travel as references (designs #198, #199).

**In the document** (version 4): the experiment's own ``licence``, each custom licence it names (``licences``), each
third-party file's ``terms`` by its path below the folder with the sha256 they were declared for, and each declared
``permission`` with its files by sha256. :func:`encode` checks their form when the experiment is saved and refuses
what is incomplete, by code: a licence that is not an SPDX expression (``terms.licence``), a custom licence without
its title or link (``terms.custom-licence``), terms for a file the experiment does not hold or for another file's
bytes, and two copies of one file's bytes declared with different terms (``terms.file``), an origin that is not a
location (``terms.origin``), and a permission lacking a part or bound to bytes no file has (``terms.permission``). A
load checks the same form when it reads the document (:func:`read_check`), so a hand edit is refused by its code; an
id the installed packaging does not list is read as written there, and reported, and ``pack`` and the card refuse it
(``terms.unlisted-licence``) until packaging is updated. The experiment's own licence may be absent: ``save`` and
``pack`` never refuse that, and every report of the folder says so.

**Each file's verdict** (:func:`folder_terms`): a file with terms is a third party's, judged by its licence and by
every permission that names its bytes; a record file is OpenMaterica's, judged by its record's own licence of its
values, as given; every other file (the files MIME writes, and those the author brings with no terms) is the author's
own, under the experiment's licence, and is always carried: ``permitted`` only where that licence is on MIME's lists,
and otherwise ``unknown`` (or ``not_permitted``, for a custom licence that says so), in the tier ``carried``; with no
licence declared, a reuser is granted none. **A file that may not be redistributed travels as a reference, whatever
its size**: its bytes stay out of a bundle, and the lock names the asset that holds them (``referenced_files``, and
the asset in ``assets``), built from its origin, one asset for one sha256. So does a file over the size a bundle
carries that has an origin. ``mime pull``, ``unpack`` and ``mime assets fetch <folder>`` place a referenced file from
the asset cache (:func:`place_referenced`), in the folder's real path; a file it cannot place is reported, with why,
and a folder still missing one is refused as ``asset.missing`` with the command.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from mime.assets import AssetRef, is_cached, location_problem
from mime.assets import file_path as cached_file
from mime.build import Refusal, RefusalClass
from mime.saved._schema import CustomLicenceDoc, ExperimentDocument, Lock, PermissionDoc, TermsDoc
from mime.terms import (
    CUSTOM_PREFIX,
    NO_ASSERTION,
    CustomLicence,
    Evidence,
    Permission,
    Role,
    Scope,
    Terms,
    Verdict,
    freest,
    ids_of,
    licence_problem,
    licence_verdict,
    permission_problems,
    permission_verdict,
    unlisted_ids,
    unlisted_words,
)

#: The most one file may weigh and still travel inside a bundle. A choice, not a measurement: a bundle is read and
#: hashed member by member, and one a person is to download, check and keep next to their experiment should stay
#: small. A larger file is an asset (its reference travels, with its sha256, size, licence and locations) and is
#: fetched on request.
INLINE_LIMIT_BYTES = 50_000_000

type Owner = Literal["yours", "third party", "record"]
_ASSET_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_CHUNK = 1 << 20

#: What to do about each kind of terms refusal.
_TERMS_LICENCE_ACTION = ("write an SPDX licence expression in its canonical spelling (licence=\"CC-BY-4.0\"), or "
                         "declare a licence of your own with its title and link: CustomLicence(\"LicenseRef-<name>\", "
                         "title=..., url=\"https://...\")")
_TERMS_FILE_ACTION = ("declare a third-party file's terms where the file is declared, asset(path, terms=Terms(...)), "
                      "then save the experiment again: built.save(folder)")
_TERMS_PERMISSION_ACTION = ("declare the permission with every part, its files by the sha256 of the bytes it covers: "
                            "exp.permission(Permission(grantor=..., grantee=..., files=(...), scope=Scope(...), "
                            "date=\"YYYY-MM-DD\", evidence=Evidence(...))), then save: built.save(folder)")
_TERMS_COPY_ACTION = ("declare the same terms on each copy, asset(path, terms=Terms(...)), or keep one copy and "
                      "reference it at each prim; then save the experiment again: built.save(folder)")
#: What decides the verdict of the author's own files where the experiment declares no licence.
NO_LICENCE_BY = "the author's own, no licence declared: a reuser is granted no licence"


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


@dataclass(frozen=True)
class Declared:
    """What a script or a loaded document declares of its terms, before :func:`encode` checks it: the experiment's
    licences (more than one is refused), each file's terms by path with the pin of its bytes and whether they were read
    from a saved document, and each permission with its files pinned. A licence read from a document may name an id
    the installed packaging does not list (the packaging that saved it did): it is kept as written, and reported."""

    licences: tuple[object, ...]
    terms: tuple[tuple[str, str, object, bool], ...]
    permissions: tuple[tuple[object, tuple[str, ...]], ...]
    #: Every file the experiment saves that terms may name (its layers and descriptions), by path, with its pin;
    #: ``None`` where they are not known yet (a document read before its build): the checks against them wait for it.
    files: Mapping[str, str] | None
    #: Whether the experiment's licence was read from a saved document.
    licence_read: bool = False


@dataclass(frozen=True)
class Encoded:
    """The document's terms keys as :func:`encode` writes them; each is absent from the document where it is empty.
    ``unlisted`` names each licence read from a document that names an id the installed packaging does not list."""

    licence: str | None
    licences: dict[str, CustomLicenceDoc]
    terms: dict[str, TermsDoc]
    permissions: list[PermissionDoc]
    unlisted: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _licence_text(given: object, where: str, custom: dict[str, CustomLicence], refusals: list[Refusal],
                  *, own: bool = False, read: bool = False,
                  unlisted: list[tuple[str, tuple[str, ...]]] | None = None) -> str | None:
    """``given`` as the expression the document holds, its custom licence noted in ``custom``; ``None`` (with the
    refusal added) when it is not a licence MIME stores. A licence ``read`` from a document may name ids the installed
    packaging does not list: they are noted in ``unlisted``, by ``where``."""
    if isinstance(given, CustomLicence):
        problem = given.problem()
        if problem is not None:
            refusals.append(_refuse("terms.custom-licence", RefusalClass.USER, f"{where}: {problem}",
                                    _TERMS_LICENCE_ACTION))
            return None
        held = custom.get(given.id)
        if held is not None and held != given:
            refusals.append(_refuse(
                "terms.custom-licence", RefusalClass.USER, f"two different custom licences are named {given.id}",
                "declare it once and use the same one wherever it is named: CustomLicence(\"LicenseRef-<name>\", "
                "title=..., url=\"https://...\")"))
            return None
        custom[given.id] = given
        return given.id
    problem = licence_problem(given, unlisted=read)
    if problem is not None:
        refusals.append(_refuse("terms.licence", RefusalClass.USER, f"{where}: {problem}", _TERMS_LICENCE_ACTION))
        return None
    text = cast(str, given)
    found = unlisted_ids(text)
    if found and unlisted is not None:
        unlisted.append((where, found))
    if own and text == NO_ASSERTION:
        refusals.append(_refuse(
            "terms.licence", RefusalClass.USER, f"{where} is {NO_ASSERTION}, which says nobody stated one",
            "declare the experiment's licence, exp.licence(\"CC-BY-4.0\"), or declare none"))
        return None
    named = [each for each in ids_of(text) if each.startswith(CUSTOM_PREFIX)]
    if named:
        refusals.append(_refuse(
            "terms.licence", RefusalClass.USER,
            f"{where} names {', '.join(named)} as text: a licence of your own is declared with its title and link",
            _TERMS_LICENCE_ACTION))
        return None
    return text


def _terms_doc(path: str, pin: str, given: object, custom: dict[str, CustomLicence], refusals: list[Refusal], *,
               read: bool = False, unlisted: list[tuple[str, tuple[str, ...]]] | None = None) -> TermsDoc | None:
    where = f"the terms of {path}"
    if not isinstance(given, Terms):
        refusals.append(_refuse("terms.file", RefusalClass.USER, f"{where} are {given!r}, not a mime.terms.Terms",
                                _TERMS_FILE_ACTION))
        return None
    found = len(refusals)
    licence = _licence_text(given.licence, f"the licence in {where}", custom, refusals, read=read, unlisted=unlisted)
    credit = cast(object, given.credit)
    if not isinstance(credit, str) or not credit.strip():
        refusals.append(_refuse(
            "terms.file", RefusalClass.USER, f"{where} give no credit: name whom its licence asks to be credited",
            "write credit=\"<the rights holder, as its licence asks>\" in its Terms"))
    role = cast(object, given.role)
    if role not in ("results", "presentation"):
        refusals.append(_refuse(
            "terms.file", RefusalClass.USER,
            f"{where} give the role {role!r}: a file is one results depend on (\"results\") or one that only draws "
            f"the scene (\"presentation\")",
            "write role=\"results\" or role=\"presentation\" in its Terms"))
    origin = cast(object, given.origin)
    if not isinstance(origin, tuple) or not all(isinstance(each, str) for each in cast(tuple[object, ...], origin)):
        refusals.append(_refuse(
            "terms.origin", RefusalClass.USER, f"{where} give an origin of {origin!r}, not a tuple of locations",
            "write origin=(\"https://...\",) in its Terms, its locations primary first"))
        origin = ()
    for location in cast(tuple[str, ...], origin):
        problem = location_problem(location)
        if problem is not None:
            refusals.append(_refuse(
                "terms.origin", RefusalClass.USER, f"{where} give the origin {location!r}: {problem}",
                "give each location of the file's exact bytes as an https:// URL pinned to a commit, or a doi: URL: "
                "origin=(\"https://...\",)"))
    if len(refusals) > found or licence is None:
        return None
    doc: TermsDoc = {"sha256": pin, "licence": licence, "credit": given.credit, "role": cast(Role, role)}
    if origin:
        doc = {**doc, "origin": list(cast(tuple[str, ...], origin))}
    return doc


def _permission_doc(given: object, files: tuple[str, ...], known: set[str] | None,
                    refusals: list[Refusal]) -> PermissionDoc | None:
    problems = permission_problems(given, files)
    stale = sorted(set(files) - known) if not problems and known is not None else []
    problems += [f"the permission names the bytes {each}, which no file the experiment holds has: a changed file "
                 f"inherits nothing" for each in stale]
    refusals += [_refuse("terms.permission", RefusalClass.USER, problem, _TERMS_PERMISSION_ACTION)
                 for problem in problems]
    if problems:
        return None
    held = cast(Permission, given)
    return {"grantor": held.grantor, "grantee": held.grantee, "files": sorted(set(files)),
            "scope": {"redistribute": held.scope.redistribute, "modify": held.scope.modify,
                      "commercial": held.scope.commercial},
            "date": held.date, "evidence": {"kind": held.evidence.kind, "pointer": held.evidence.pointer}}


def encode(declared: Declared) -> tuple[Encoded, list[Refusal]]:
    """The document's terms keys, and every refusal of their form, at once."""
    refusals: list[Refusal] = []
    custom: dict[str, CustomLicence] = {}
    unlisted: list[tuple[str, tuple[str, ...]]] = []
    licence: str | None = None
    if len(declared.licences) > 1:
        refusals.append(_refuse(
            "terms.licence", RefusalClass.USER,
            f"exp.licence is called {len(declared.licences)} times, and an experiment has one licence of its own",
            "call exp.licence(...) once, with an SPDX expression: exp.licence(\"CC-BY-4.0\")"))
    elif declared.licences:
        licence = _licence_text(declared.licences[0], "the experiment's licence", custom, refusals, own=True,
                                read=declared.licence_read, unlisted=unlisted)
    terms: dict[str, TermsDoc] = {}
    seen: dict[str, object] = {}
    copies: dict[str, tuple[str, object]] = {}
    for path, pin, given, read in declared.terms:
        if declared.files is not None and path not in declared.files:
            refusals.append(_refuse(
                "terms.file", RefusalClass.USER,
                f"terms are declared for {path}, which is not a layer or a description the experiment saves",
                _TERMS_FILE_ACTION))
            continue
        if declared.files is not None and pin != declared.files[path]:
            refusals.append(_refuse(
                "terms.file", RefusalClass.USER,
                f"the terms of {path} were declared for the bytes {pin}, and the file has {declared.files[path]}: a "
                f"changed file inherits nothing", _TERMS_FILE_ACTION))
            continue
        if path in seen and seen[path] != given:
            refusals.append(_refuse("terms.file", RefusalClass.USER, f"{path} is declared with two different terms",
                                    _TERMS_FILE_ACTION))
            continue
        seen[path] = given
        original = copies.setdefault(pin, (path, given))
        if original[0] != path and original[1] != given:
            refusals.append(_refuse(
                "terms.file", RefusalClass.USER,
                f"{original[0]} and {path} hold the same bytes (sha256 {pin}) and are declared with different terms: "
                f"each copy of a file is declared with its terms, and one file's bytes have one set of terms",
                _TERMS_COPY_ACTION))
            continue
        doc = _terms_doc(path, pin, given, custom, refusals, read=read, unlisted=unlisted)
        if doc is not None:
            terms[path] = doc
    known = set(declared.files.values()) if declared.files is not None else None
    permissions = [doc for given, files in declared.permissions
                   if (doc := _permission_doc(given, files, known, refusals)) is not None]
    permissions.sort(key=lambda doc: json.dumps(doc, sort_keys=True))
    licences: dict[str, CustomLicenceDoc] = {
        key: {"title": held.title, "url": held.url, "redistribution": cast(Literal["not_permitted", "unknown"],
                                                                         held.redistribution)}
        for key, held in sorted(custom.items())}
    return Encoded(licence, licences, dict(sorted(terms.items())), permissions, tuple(unlisted)), refusals


def write_into(document: ExperimentDocument, encoded: Encoded) -> None:
    """Put the terms keys into ``document``, each only where it holds something."""
    if encoded.licence is not None:
        document["licence"] = encoded.licence
    if encoded.licences:
        document["licences"] = encoded.licences
    if encoded.terms:
        document["terms"] = encoded.terms
    if encoded.permissions:
        document["permissions"] = encoded.permissions


def custom_licences(document: ExperimentDocument) -> dict[str, CustomLicence]:
    return {key: CustomLicence(key, held["title"], held["url"], held["redistribution"])
            for key, held in document.get("licences", {}).items()}


def _given(expression: str, custom: Mapping[str, CustomLicence]) -> str | CustomLicence:
    return custom.get(expression, expression)


def experiment_licence(document: ExperimentDocument) -> str | CustomLicence | None:
    """The experiment's own licence as a script declares it, or ``None`` where it declares none."""
    held = document.get("licence")
    return None if held is None else _given(held, custom_licences(document))


def declared_terms(document: ExperimentDocument) -> list[tuple[str, str, Terms]]:
    """Each file's terms as a script declares them, by path and pin."""
    custom = custom_licences(document)
    return [(path, held["sha256"], Terms(_given(held["licence"], custom), held["credit"], held["role"],
                                         tuple(held.get("origin", []))))
            for path, held in document.get("terms", {}).items()]


def declared_permissions(document: ExperimentDocument) -> list[Permission]:
    return [Permission(held["grantor"], held["grantee"], tuple(held["files"]),
                       Scope(held["scope"]["redistribute"], held["scope"]["modify"], held["scope"]["commercial"]),
                       held["date"], Evidence(held["evidence"]["kind"], held["evidence"]["pointer"]))
            for held in document.get("permissions", [])]


def unlisted_note(where: str, found: Sequence[str]) -> str:
    """The load's note for a licence that names an id the installed packaging does not list."""
    return (f"terms: {where} names {unlisted_words(found)}; the load reads it as written and judges it unknown, and "
            f"`mime pack` and `mime card` refuse it until packaging lists it")


def read_check(document: ExperimentDocument) -> tuple[list[Refusal], tuple[str, ...]]:
    """The document's licence, terms and permissions checked for their form as the document is read, before its build
    (what a hand edit can break): every refusal, by the code ``save`` gives it, and a note for each licence that names
    an id the installed packaging does not list, which is read as written. The checks against the files themselves
    (a file's pin, a permission's bytes) are the build's."""
    licence = experiment_licence(document)
    declared = Declared((licence,) if licence is not None else (),
                        tuple((path, pin, terms, True) for path, pin, terms in declared_terms(document)),
                        tuple((held, cast(tuple[str, ...], held.files)) for held in declared_permissions(document)),
                        None, licence_read=True)
    encoded, refusals = encode(declared)
    return refusals, tuple(unlisted_note(where, found) for where, found in encoded.unlisted)


# --- each file's verdict ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FileTerms:
    """One file of a saved folder: its path, the sha256 and size of its bytes, whose it is, its licence (``None`` for
    the author's own where the experiment declares none), its verdict, and for a third party's its credit, role and
    origin, the permissions that name its bytes (by their index in the document), and the asset its bytes travel as
    when they travel as a reference."""

    path: str
    sha256: str
    size: int
    owner: Owner
    licence: str | None
    verdict: Verdict
    credit: str | None = None
    role: Role | None = None
    origin: tuple[str, ...] = ()
    permissions: tuple[int, ...] = ()
    referenced: str | None = None


def _record_licence(data: bytes) -> str:
    """The licence a record file states of its values (``licence.values``), as given; ``NOASSERTION`` where it states
    none MIME reads."""
    try:
        held: object = json.loads(data)
    except ValueError:
        return NO_ASSERTION
    licence = cast(dict[str, object], held).get("licence") if isinstance(held, dict) else None
    values = cast(dict[str, object], licence).get("values") if isinstance(licence, dict) else None
    return values if isinstance(values, str) and licence_problem(values) is None else NO_ASSERTION


def own_verdict(licence: str | None, custom: Mapping[str, CustomLicence]) -> Verdict:
    """The verdict of the author's own files, always carried: the experiment's licence's where it is on MIME's open or
    conditional list (with its conditions); otherwise its status (``unknown``, or ``not_permitted`` for a custom
    licence that says so) in the tier ``carried``; and ``unknown`` where the experiment declares no licence, since a
    reuser is then granted none."""
    if licence is None:
        return Verdict("unknown", "carried", (), NO_LICENCE_BY)
    held = licence_verdict(licence, custom)
    if held.tier == "referenced":
        return Verdict(held.status, "carried", (), "the author's own")
    return Verdict("permitted", held.tier, held.conditions, "the author's own")


def third_party_verdict(doc: TermsDoc, sha256: str, permissions: Sequence[PermissionDoc],
                        custom: Mapping[str, CustomLicence]) -> tuple[Verdict, tuple[int, ...]]:
    """A third party's file judged by its licence and by every permission that names its bytes, and those
    permissions' indices: a permission binds by sha256, so a changed file inherits nothing."""
    covering = tuple(index for index, held in enumerate(permissions) if sha256 in held["files"])
    granted = [permission_verdict(Scope(**permissions[index]["scope"])) for index in covering]
    verdict = freest([licence_verdict(doc["licence"], custom), *granted])
    return verdict, covering


def file_verdicts(document: ExperimentDocument, files: Mapping[str, tuple[str, int]],
                  records: Mapping[str, str]) -> list[FileTerms]:
    """Every file of ``files`` (path to sha256 and size), each with its verdict; ``records`` maps a record file's path
    to the licence it states."""
    custom = custom_licences(document)
    licence = document.get("licence")
    terms = document.get("terms", {})
    permissions = document.get("permissions", [])
    found: list[FileTerms] = []
    for path, (sha256, size) in sorted(files.items()):
        held = terms.get(path)
        if held is not None:
            verdict, covering = third_party_verdict(held, sha256, permissions, custom)
            found.append(FileTerms(path, sha256, size, "third party", held["licence"], verdict, held["credit"],
                                   held["role"], tuple(held.get("origin", [])), covering))
        elif path in records:
            found.append(FileTerms(path, sha256, size, "record", records[path],
                                   _with_by(licence_verdict(records[path], custom), "its record's licence")))
        else:
            found.append(FileTerms(path, sha256, size, "yours", licence, own_verdict(licence, custom)))
    return found


def _with_by(verdict: Verdict, by: str) -> Verdict:
    return Verdict(verdict.status, verdict.tier, verdict.conditions, by)


def asset_name(path: str, taken: set[str], sha256: str) -> str:
    """The name of the asset a referenced file travels as: its file name, written as an asset's name, and followed by
    its hash's first digits where another asset of the lock holds the name."""
    name = _ASSET_NAME_UNSAFE.sub("_", path.rpartition("/")[2]).lstrip("._-") or "file"
    return f"{name}-{sha256[:12]}" if name in taken else name


def must_reference(file: FileTerms, limit: int) -> bool:
    """Whether a file's bytes travel as a reference: it may not be redistributed, whatever its size; or it is over
    ``limit`` and its terms give an origin."""
    return not file.verdict.carried or (file.size > limit and bool(file.origin))


def referenced_assets(document: ExperimentDocument, files: Mapping[str, tuple[str, int]],
                      taken: Sequence[str]) -> list[tuple[str, AssetRef]]:
    """The files of ``files`` whose bytes travel as references, each with the asset that holds them, built from its
    origin: what ``save`` writes into the lock (``referenced_files`` and ``assets``). A file that must travel as a
    reference and gives no origin is left out here: ``pack`` refuses it, naming what to declare. Copies of one file's
    bytes (which ``save`` holds to one set of terms) travel as one asset."""
    names = set(taken)
    held: dict[str, AssetRef] = {}
    found: list[tuple[str, AssetRef]] = []
    for file in file_verdicts(document, {path: each for path, each in files.items() if path in
                                         document.get("terms", {})}, {}):
        if must_reference(file, INLINE_LIMIT_BYTES) and file.origin:
            ref = held.get(file.sha256)
            if ref is None:
                name = asset_name(file.path, names, file.sha256)
                names.add(name)
                ref = AssetRef(name, file.sha256, file.size, file.licence or NO_ASSERTION, file.origin, unlisted=True)
                held[file.sha256] = ref
            found.append((file.path, ref))
    return found


def sha256_and_size(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest, path.stat().st_size


@dataclass(frozen=True)
class FolderTerms:
    """Every file a saved folder names, each with its verdict, in path order; the experiment's own licence (``None``
    where it declares none), the custom licences, the permissions, and the saved form's version."""

    files: tuple[FileTerms, ...]
    licence: str | None
    custom: Mapping[str, CustomLicence]
    permissions: tuple[PermissionDoc, ...]
    version: int

    def referenced(self) -> tuple[FileTerms, ...]:
        return tuple(file for file in self.files if file.referenced is not None)

    def unlisted_refusals(self, command: str) -> list[Refusal]:
        """``terms.unlisted-licence`` for each licence of the folder that names an id the installed packaging does not
        list: ``command`` (``mime pack``, ``mime card``) would otherwise state a verdict this packaging cannot give."""
        named = [("the experiment's licence", self.licence)] if self.licence is not None else []
        named += [(file.path, file.licence) for file in self.files if file.owner == "third party" and file.licence]
        return [_refuse("terms.unlisted-licence", RefusalClass.ENVIRONMENT,
                        f"{where}: {licence!r} names {unlisted_words(found)}",
                        f"update packaging (pip install -U packaging), then `{command} <folder>` again")
                for where, licence in named if (found := unlisted_ids(licence))]


def folder_terms(folder: Path, document: ExperimentDocument, lock: Lock, named: set[str], version: int) -> FolderTerms:
    """Each of the ``named`` files of a saved folder with its verdict. A file that is in the folder is hashed there; a
    file that travels as a reference and is not there yet is described by the asset the lock names for it. A file the
    lock says travels as a reference is marked with its asset's name."""
    referenced = lock.get("referenced_files", {})
    assets = {held["name"]: held for held in lock.get("assets", [])}
    files: dict[str, tuple[str, int]] = {}
    records: dict[str, str] = {}
    record_paths = {held["file"] for held in lock["records"].values()}
    for path in sorted(named):
        where = folder / path
        if where.is_file():
            files[path] = sha256_and_size(where)
            if path in record_paths:
                records[path] = _record_licence(where.read_bytes())
        elif path in referenced and referenced[path] in assets:
            held = assets[referenced[path]]
            files[path] = (held["sha256"], held["size"])
    judged = [FileTerms(f.path, f.sha256, f.size, f.owner, f.licence, f.verdict, f.credit, f.role, f.origin,
                        f.permissions, referenced.get(f.path)) for f in file_verdicts(document, files, records)]
    return FolderTerms(tuple(judged), document.get("licence"), custom_licences(document),
                       tuple(document.get("permissions", [])), version)


# --- placing a referenced file -------------------------------------------------------------------------------------


#: Why :func:`place_referenced` left a file unplaced.
NOT_CACHED = "the asset is not in the cache"
OTHER_FILE = "another file is in its place"
NO_ASSET = "the lock's assets do not name the asset"
OUTSIDE = "its path is not a place inside the folder a saved form keeps a file"
CORRUPT = "the cached copy is not the file's bytes (mime assets fetch hashes it and fetches it again)"


@dataclass(frozen=True)
class Placed:
    """What :func:`place_referenced` did: the files it placed from the cache, those already in place, and those it
    could not place, each with the asset's name and why (:data:`NOT_CACHED`, :data:`OTHER_FILE`, :data:`NO_ASSET`,
    :data:`OUTSIDE`, :data:`CORRUPT`)."""

    placed: tuple[str, ...] = ()
    present: tuple[str, ...] = ()
    missing: tuple[tuple[str, str, str], ...] = ()


def _lock_of(folder: Path) -> dict[str, object]:
    from mime.saved._folder import LOCK_FILE

    try:
        held: object = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return cast(dict[str, object], held) if isinstance(held, dict) else {}


def _copy_checked(source: Path, target: Path, sha256: str) -> bool:
    """Copy ``source`` to ``target`` through a temporary file beside it, hashing while copying; the copy is renamed
    into place only when its sha256 is ``sha256``, and never over a file that appeared meanwhile."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}.partial")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reading, temporary.open("xb") as writing:
            while chunk := reading.read(_CHUNK):
                digest.update(chunk)
                writing.write(chunk)
        if digest.hexdigest() != sha256:
            return False
        try:
            os.link(temporary, target)
        except FileExistsError:
            return False
        except OSError:
            if target.exists():
                return False
            temporary.replace(target)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def place_referenced(folder: str | os.PathLike[str]) -> Placed:
    """Put each file the folder's lock says travels as a reference at its path, from the asset cache: copied and
    hashed, never over a file already there, never outside the folder. A file that is not cached is left for ``mime
    assets fetch <folder>``, which stores it and calls this again; every file it does not place is reported missing,
    with why. The folder is judged by its real path, so that a folder reached through a link (macOS's ``/tmp`` and
    ``/var``) is the folder its files are in. Reads the lock as JSON and runs nothing."""
    from mime.saved._bundle import name_problem

    root = Path(os.path.realpath(folder))
    lock = _lock_of(root)
    listed = lock.get("referenced_files", {})
    held_assets = lock.get("assets", [])
    if not isinstance(listed, dict) or not isinstance(held_assets, list):
        return Placed()
    refs: dict[str, AssetRef] = {}
    for record in cast(list[object], held_assets):
        if isinstance(record, dict):
            try:
                ref = AssetRef.from_record(cast(dict[str, object], record))
            except Exception:  # noqa: BLE001 - a malformed record is the load's to refuse, with its reason
                continue
            refs[ref.name] = ref
    placed: list[str] = []
    present: list[str] = []
    missing: list[tuple[str, str, str]] = []
    for path, name in sorted(cast(dict[object, object], listed).items(), key=str):
        if not isinstance(path, str) or not isinstance(name, str):
            continue
        target = root / path
        ref = refs.get(name)
        if name_problem(path) is not None or root not in target.resolve().parents:
            missing.append((path, name, OUTSIDE))
        elif ref is None:
            missing.append((path, name, NO_ASSET))
        elif target.is_file():
            if sha256_and_size(target)[0] == ref.sha256:
                present.append(path)
            else:
                missing.append((path, name, OTHER_FILE))
        elif not is_cached(ref):
            missing.append((path, name, NOT_CACHED))
        elif _copy_checked(cached_file(ref), target, ref.sha256):
            placed.append(path)
        else:
            missing.append((path, name, OTHER_FILE if target.exists() else CORRUPT))
    return Placed(tuple(placed), tuple(present), tuple(missing))


def unplaced_refusals(folder: str | os.PathLike[str], placed: Placed) -> list[Refusal]:
    """``asset.missing`` for each file :func:`place_referenced` left unplaced in ``folder``, with why and what to do:
    what ``mime assets fetch <folder>`` refuses once it has fetched every asset the lock names."""
    from mime.assets._commands import fetch_command

    return [_refuse("asset.missing", RefusalClass.ENVIRONMENT,
                    f"{path}, which travels as the reference {name}, is not placed in the folder: {why}",
                    "the folder's lock names a place outside it: save the experiment again from its script, "
                    "built.save(folder)" if why == OUTSIDE else
                    f"move the file in its place away, or fetch the asset, then `{fetch_command(folder=folder)}` "
                    f"again")
            for path, name, why in placed.missing]


__all__ = ["CORRUPT", "INLINE_LIMIT_BYTES", "NOT_CACHED", "NO_ASSET", "NO_LICENCE_BY", "OTHER_FILE", "OUTSIDE",
           "Declared", "Encoded", "FileTerms", "FolderTerms", "Placed", "asset_name", "custom_licences",
           "declared_permissions", "declared_terms", "encode", "experiment_licence", "file_verdicts", "folder_terms",
           "must_reference", "own_verdict", "place_referenced", "read_check", "referenced_assets", "sha256_and_size",
           "third_party_verdict", "unlisted_note", "unplaced_refusals", "write_into"]
