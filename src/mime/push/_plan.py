"""The plan push shows before anything goes online (design §2.2), computed offline after the folder loads: every
file of the record with its licence and why it is there, every file inside the bundle with its tier and verdict, every
file that travels as a reference and is never uploaded, each declared permission in #198's words, and the record's
metadata. Every refusal is found here, at once, before a request.

**What the record holds** (§2.1): the bundle ``pack`` makes of the folder, the ``NOTICE`` it carries (uploaded beside
it), and each asset of the folder's lock that may be redistributed and has no location on Zenodo yet (#171 item 4),
read from the cache and hashed in full first. A file that may not be redistributed, whatever its size, is never a
record file: ``pack`` carries it as a reference, and so does the record.

**The metadata** is the records API's: resource type ``dataset``; the title, creators, contributors, keywords and
related works from ``sharing.yaml``; the rights holder (its credit) of each third-party file the record holds, and each
permission's grantor, as a ``rightsholder`` contributor (a file that travels as a reference is not in the record: its
rights holder is credited beside it in the NOTICE and the card, and is no contributor of the record); the card
(``mime card``'s, the same function) as the description, in HTML; every licence of a file in the record or the bundle
as a rights entry, an SPDX id as the vocabulary's id (lower case, as the
sandbox's ``cc-by-4.0`` and ``mit`` are) and a custom ``LicenseRef-`` licence as an entry with its title and link and
no ``id``, which the sandbox refuses for a ``LicenseRef-`` (a 400 naming no field); and the MIME release it was saved
with as ``requires`` and ``iscompiledby``.
"""

from __future__ import annotations

import datetime
import hashlib
import html
import json
import os
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mime._terminal import plain
from mime.assets import AssetRef, folder_assets, locate, size_text
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._client import Json
from mime.push._owners import FIELD_OWNERS, OWN_RELATED, field_owner
from mime.push._service import Service
from mime.push._sharing import Person, Sharing, sharing_path
from mime.push._state import DraftState
from mime.saved._bundle import MANIFEST_MEMBER, NOTICE_MEMBER, read_manifest
from mime.saved._card import availability, permission_words
from mime.saved._schema import Lock, PermissionDoc
from mime.saved._terms import FileTerms, FolderTerms
from mime.terms import CUSTOM_PREFIX, NO_ASSERTION, Verdict, ids_of, licence_verdict

#: Zenodo's limits (its quickstart): 50 GB in a record, and in any one file, and 100 files.
MOST_BYTES = 50_000_000_000
MOST_FILES = 100
#: The publisher of a record whose DOI Zenodo mints: DataCite requires one to register the DOI, and the sandbox
#: refuses a draft without it ("Missing publisher field required for DOI registration", the sandbox lane, 2026-11-01).
PUBLISHER = "Zenodo"
#: The fields below ``metadata`` that DataCite needs to register a DOI, each of which MIME always sends (a test holds
#: the payload to them, and the fake holds a draft to them, as the sandbox does).
DOI_FIELDS = ("publisher", "publication_date", "creators", "title", "resource_type")


#: The keyword every record push makes carries: the uncurated index of MIME's experiments (design §4).
KEYWORD = "mime-experiment"
NOTICE_KEY = "NOTICE"
_CHUNK = 1 << 20
_ZENODO_LOCATION = re.compile(r"doi:10\.(?:5281|5072)/zenodo\.\d+/|https://(?:sandbox\.)?zenodo\.org/")
_SANDBOX_LOCATION = re.compile(r"doi:10\.5072/|https://sandbox\.zenodo\.org/")
_EXTENSION = re.compile(r"(\.[A-Za-z0-9]{1,8})+$")
_PYPI = "https://pypi.org/project/mime-engine/{}/"


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


def file_hashes(path: Path) -> tuple[str, str, int]:
    """The md5 (the checksum Zenodo states) and sha256 (MIME's own) of a file, in one read, and its size."""
    md5, sha256, size = hashlib.md5(usedforsecurity=False), hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return md5.hexdigest(), sha256.hexdigest(), size


@dataclass(frozen=True)
class RecordFile:
    """A file the record holds: its key there, where its bytes are read from, their size, md5 and sha256, its
    licence, and why it is in the record."""

    key: str
    path: Path
    size: int
    md5: str
    sha256: str
    licence: str | None
    what: str


@dataclass(frozen=True)
class AssetChoice:
    """An asset of the folder's lock, and what the record does with it: ``upload`` (its bytes become a record file)
    or not, with why."""

    ref: AssetRef
    verdict: Verdict
    upload: bool
    why: str


def on_zenodo(ref: AssetRef) -> bool:
    return any(_ZENODO_LOCATION.match(each) for each in ref.locations)


def asset_choices(refs: Sequence[AssetRef], judged: FolderTerms, lock: Lock) -> list[AssetChoice]:
    """Each asset with what the record does with it (#198's tiers, #171 item 4): one that holds a file that may not be
    redistributed, or whose own licence does not let it be, travels as a reference and is never uploaded; one that
    may be redistributed is uploaded where it has no location on Zenodo yet."""
    travelling = {file.referenced: file for file in judged.files if file.referenced is not None}
    listed = set(lock.get("referenced_files", {}).values())
    found: list[AssetChoice] = []
    for ref in refs:
        file = travelling.get(ref.name)
        verdict = file.verdict if file is not None else licence_verdict(ref.licence, judged.custom)
        if ref.name in listed and file is None:
            verdict = Verdict("unknown", "referenced", (), "a referenced file push cannot judge")
        if not verdict.carried:
            found.append(AssetChoice(ref, verdict, False, f"{verdict.words()}: travels as a reference, never "
                                                          f"uploaded"))
        elif on_zenodo(ref):
            found.append(AssetChoice(ref, verdict, False, "already on Zenodo: the record references it"))
        else:
            found.append(AssetChoice(ref, verdict, True, f"{verdict.words()}; no Zenodo location yet, so uploaded "
                                                         f"(#171)"))
    return found


def record_key(ref: AssetRef, path: str | None) -> str:
    """The record's file name for an asset: its name, with its file's extension (record file names collide across
    the shipped tables, asset names do not)."""
    source = path or (ref.locations[0].rsplit("/", 1)[-1] if ref.locations else "")
    found = _EXTENSION.search(source.rsplit("/", 1)[-1])
    extension = found.group(0) if found is not None else ""
    return ref.name if ref.name.endswith(extension) else ref.name + extension


@dataclass(frozen=True)
class Plan:
    """What a push will do, computed offline: the service, the experiment, the record files in upload order (the
    bundle, the NOTICE, then the uploaded assets), the folder's judged terms, the lock's assets with what the record
    does with each, the experiment's licence, ``sharing.yaml`` as read, the metadata payload and the MIME release."""

    service: Service
    experiment: str
    folder: Path
    files: tuple[RecordFile, ...]
    judged: FolderTerms
    assets: tuple[AssetChoice, ...]
    licence: str
    sharing: Sharing
    payload: Json
    mime_version: str

    @property
    def size(self) -> int:
        return sum(file.size for file in self.files)

    def _licences(self) -> list[str]:
        return record_licences(self.judged, self.assets, self.licence)

    def owner(self, field: str) -> str:
        """Who owns the metadata field at ``field``: :func:`mime.push._owners.field_owner`, for this sharing.yaml."""
        return field_owner(field, len(self.sharing.contributors))

    def lines(self, state: DraftState | None) -> list[str]:
        """The plan as push prints it, before any request."""
        draft = (f"the draft {state.record} MIME made earlier ({state.page}), updated: its metadata is written again "
                 f"from sharing.yaml (a change made on the site stops the push, unless --overwrite-site-edits) and "
                 f"only files that differ are sent" if state is not None and state.record else
                 "a new draft, its DOI reserved")
        lines = [f"Plan for {plain(self.experiment)!r} on {self.service.words}: {draft}",
                 f"  Record files ({len(self.files)}, {size_text(self.size)}):"]
        width = max(len(plain(file.key)) for file in self.files)
        lines += [f"    {plain(file.key):<{width}}  {size_text(file.size):>8}  {plain(file.licence or 'generated'):<14}"
                  f"  {file.what}" for file in self.files]
        inside = [file for file in self.judged.files if file.verdict.carried and file.referenced is None]
        lines.append(f"  Inside the bundle ({len(inside)} files):")
        lines += [f"    {plain(file.path)}  {size_text(file.size)}  {plain(file.licence or 'no licence')}  "
                  f"{_whose(file)}; {file.verdict.words()}" for file in inside]
        held = [file for file in self.judged.files if file.referenced is not None or not file.verdict.carried]
        held_assets = [choice for choice in self.assets if not choice.upload and
                       choice.ref.name not in {file.referenced for file in held}]
        lines.append(f"  Referenced, not uploaded ({len(held) + len(held_assets)}):" +
                     ("" if held or held_assets else " none"))
        lines += [f"    {plain(file.path)}  {size_text(file.size)}  licence {plain(file.licence or 'none')}, "
                  f"{file.verdict.words()}, role {file.role or 'results'}, from "
                  f"{plain(file.origin[0]) if file.origin else 'no origin'}, sha256 {file.sha256}" for file in held]
        lines += [f"    the asset {plain(choice.ref.name)}  {size_text(choice.ref.size)}  licence "
                  f"{plain(choice.ref.licence)}: {choice.why}, at {plain(choice.ref.locations[0])}"
                  for choice in held_assets if choice.ref.locations]
        lines += [f"  The experiment's licence: {plain(self.licence)}, bound to its files (sharing.yaml states the "
                  f"same)",
                  f"  Licences on the record: {', '.join(plain(each) for each in self._licences())} (the "
                  f"NOTICE maps each file to its licence; Zenodo has no licence per file)",
                  f"  Creators: {'; '.join(person.words() for person in self.sharing.creators)}",
                  f"  Contributors: {'; '.join(_contributor_words(self)) or 'none'}",
                  f"  Related: {'; '.join(_related_words(self))}",
                  f"  Keywords: {', '.join(plain(each) for each in _keywords(self.sharing))}",
                  *_built_lines(self)]
        permissions = self.judged.permissions
        if permissions:
            lines.append(f"  Permissions you declare ({len(permissions)}); the draft is private, and each becomes "
                         f"public when the record is published:")
            for index, held_permission in enumerate(permissions, start=1):
                lines += ["    " + line for line in _declaring(index, held_permission, self.judged)]
        lines += [f"  Availability: {availability(self.judged)}",
                  f"Next: on {self.service.name}, the draft is made or updated, its DOI reserved, and each record file "
                  f"that differs from the draft's is uploaded and its md5 checked; nothing is published (publishing "
                  f"comes with a later MIME)."]
        return lines


def _built_lines(plan: Plan) -> list[str]:
    """The metadata the plan names beyond the credit: one line each for the title, the description, the publisher, the
    date, the resource type, the access and push's identifier of the draft."""
    held = _mapping(plan.payload.get("metadata"))
    description = str(held.get("description", ""))
    paragraph = "sharing.yaml's paragraph, then " if plan.sharing.description else ""
    shown = _mapping(plan.payload.get("access"))
    return [f"  Title: {plain(str(held.get('title', '')))}",
            f"  Description: {paragraph}the card, as HTML ({len(description):,} characters)",
            f"  Publisher: {plain(str(held.get('publisher', '')))} (DataCite needs one to register the DOI)",
            f"  Publication date: {plain(str(held.get('publication_date', '')))}",
            f"  Resource type: {plain(str(_mapping(held.get('resource_type')).get('id', '')))}",
            f"  Access once published: the record {plain(str(shown.get('record', '')))}, its files "
            f"{plain(str(shown.get('files', '')))} (a draft is private)",
            "  Alternate identifier: a urn:uuid push makes for the draft, by which it knows the draft is its own"]


def _mapping(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _whose(file: FileTerms) -> str:
    return {"yours": "yours", "third party": "a third party's", "record": "OpenMaterica's record"}[file.owner]


def _declaring(index: int, held: PermissionDoc, judged: FolderTerms) -> list[str]:
    """#198 item 6's words for one declaration, then the facts the card states of it."""
    paths = [file.path for file in judged.files if file.sha256 in held["files"]]
    count = len(held["files"])
    scope = held["scope"]
    words = ", ".join(("redistribute" if scope["redistribute"] else "no redistribution",
                       "modify" if scope["modify"] else "no modification",
                       "commercial use" if scope["commercial"] else "non-commercial"))
    files = f"these {count} files" if count != 1 else "this file"
    return [f"{index}. You are declaring publicly, under your name, that {plain(held['grantor'])} permitted you to "
            f"redistribute {files} (scope: {words}; evidence: "
            f"{held['evidence']['kind'].replace('_', ' ')}, dated {plain(held['date'])}).",
            *("   " + line for line in permission_words(index, held, paths)[1:])]


def record_licences(judged: FolderTerms, choices: Sequence[AssetChoice], licence: str) -> list[str]:
    """Every licence id of a file in the record or the bundle, and the experiment's: the record lists each, so as not
    to mislead."""
    found = {identifier for file in judged.files if file.verdict.carried and file.referenced is None
             and file.licence is not None for identifier in ids_of(file.licence)}
    found |= {identifier for choice in choices if choice.upload for identifier in ids_of(choice.ref.licence)}
    found.update(ids_of(licence))
    found.discard(NO_ASSERTION)
    return sorted(found)


def _rights_holders(judged: FolderTerms) -> list[str]:
    """The rights holder of each third-party file the record holds (in the bundle), and each permission's grantor: a
    file that travels as a reference is not in the record, so its rights holder is credited beside it (the NOTICE, the
    card) and is not a contributor of the record."""
    names = [file.credit for file in judged.files if file.owner == "third party" and file.credit
             and file.verdict.carried and file.referenced is None]
    names += [held["grantor"] for held in judged.permissions]
    return list(dict.fromkeys(names))


def _contributor_words(plan: Plan) -> list[str]:
    return [person.words() for person in plan.sharing.contributors] + \
        [f"{plain(name)} (RightsHolder, from a file's terms)" for name in _rights_holders(plan.judged)]


def _related_words(plan: Plan) -> list[str]:
    return [f"Requires and IsCompiledBy mime-engine {plain(plan.mime_version)}"] + \
        [f"{related.relation} {plain(related.identifier)}" for related in plan.sharing.related]


def _keywords(sharing: Sharing) -> list[str]:
    return list(dict.fromkeys([KEYWORD, *sharing.keywords]))


# --- the metadata --------------------------------------------------------------------------------------------------


def _person(person: Person) -> Json:
    if person.organisation is not None:
        held: Json = {"type": "organizational", "name": person.organisation}
    else:
        held = {"type": "personal", "family_name": person.family}
        if person.given:
            held["given_name"] = person.given
        if person.orcid:
            held["identifiers"] = [{"scheme": "orcid", "identifier": person.orcid}]
    entry: Json = {"person_or_org": held}
    if person.affiliation:
        entry["affiliations"] = [{"id": person.ror, "name": person.affiliation} if person.ror
                                 else {"name": person.affiliation}]
    if person.role:
        entry["role"] = {"id": person.role.lower()}
    return entry


def rights_entries(identifiers: Sequence[str], judged: FolderTerms) -> list[Json]:
    """The record's rights: an SPDX id as the vocabulary's id; a ``LicenseRef-`` licence as an entry with its title,
    its link and what it allows, and no ``id`` (the sandbox refuses a ``LicenseRef-`` id)."""
    found: list[Json] = []
    for identifier in identifiers:
        custom = judged.custom.get(identifier)
        if identifier.startswith(CUSTOM_PREFIX) and custom is not None:
            found.append({"title": {"en": custom.title},
                          "description": {"en": f"{identifier}: a licence of its own, whose redistribution is "
                                                f"{custom.redistribution.replace('_', ' ')} (each file's licence is "
                                                f"in the NOTICE)"},
                          "link": custom.url})
        elif not identifier.startswith(CUSTOM_PREFIX):
            found.append({"id": identifier.lower()})
    return found


def card_html(markdown: str) -> str:
    """The card's Markdown as the HTML of a record's description: its headings, lists, tables, code blocks and
    paragraphs, with ``**bold**`` and ``code``; every character of the text escaped first. Deterministic."""
    out: list[str] = []
    paragraph: list[str] = []
    listing: list[str] = []
    table: list[list[str]] = []
    code: list[str] | None = None

    def inline(text: str) -> str:
        escaped = html.escape(text, quote=False)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)

    def flush() -> None:
        if paragraph:
            out.append("<p>" + "<br>".join(inline(line) for line in paragraph) + "</p>")
            paragraph.clear()
        if listing:
            out.append("<ul>" + "".join(f"<li>{inline(item)}</li>" for item in listing) + "</ul>")
            listing.clear()
        if table:
            head, *rows = table
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(cell)}</th>" for cell in head) + "</tr></thead>"
                       "<tbody>" + "".join("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>"
                                           for row in rows) + "</tbody></table>")
            table.clear()

    for line in markdown.splitlines():
        if code is not None:
            if line.startswith("```"):
                out.append("<pre><code>" + html.escape("\n".join(code), quote=False) + "</code></pre>")
                code = None
            else:
                code.append(line)
        elif line.startswith("```"):
            flush()
            code = []
        elif line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif line.startswith("- "):
            if paragraph or table:
                flush()
            listing.append(line[2:])
        elif line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not all(set(cell) <= {"-", ":"} for cell in cells):
                if paragraph or listing:
                    flush()
                table.append(cells)
        elif not line.strip():
            flush()
        else:
            if listing or table:
                flush()
            paragraph.append(line)
    flush()
    return "\n".join(out) + "\n"


def _holder(name: str) -> Json:
    return {"person_or_org": {"type": "organizational", "name": name}, "role": {"id": "rightsholder"}}


def _related(identifier: str, scheme: str, relation: str) -> Json:
    return {"identifier": identifier, "scheme": scheme, "relation_type": {"id": relation}}


def payload_of(plan_parts: tuple[Sharing, FolderTerms, Sequence[str], str, str], today: datetime.date) -> Json:
    """The draft's body for the records API, from ``sharing.yaml``, the folder's terms, the record's licences, the
    description's HTML and the MIME release."""
    sharing, judged, licences, description, version = plan_parts
    contributors: list[Json] = [_person(person) for person in sharing.contributors]
    contributors += [_holder(name) for name in _rights_holders(judged)]
    related: list[Json] = [{"identifier": _PYPI.format(version), "scheme": "url", "relation_type": {"id": relation},
                            "resource_type": {"id": "software"}} for relation in ("requires", "iscompiledby")]
    related += [_related(each.identifier, each.scheme, each.relation.lower()) for each in sharing.related]
    metadata: Json = {
        "resource_type": {"id": "dataset"},
        "title": sharing.title,
        "publication_date": today.isoformat(),
        "publisher": PUBLISHER,
        "description": description,
        "creators": [_person(person) for person in sharing.creators],
        "rights": rights_entries(licences, judged),
        "subjects": [{"subject": each} for each in _keywords(sharing)],
        "related_identifiers": related,
    }
    if contributors:
        metadata["contributors"] = contributors
    return {"metadata": metadata, "access": {"record": "public", "files": "public"}, "files": {"enabled": True}}


# --- making the plan -----------------------------------------------------------------------------------------------


def make_plan(folder: str | os.PathLike[str], service: Service, sharing: Sharing, staged: Path, *,
              today: datetime.date | None = None) -> Plan:
    """Pack the folder into ``staged`` and work out everything push will send, refusing with every reason found."""
    from mime.credentials import held_tokens
    from mime.saved import card_text, pack
    from mime.saved._bundle import scan_text
    from mime.saved._card import judged_folder

    root = Path(os.path.abspath(folder))
    refusals: list[Refusal] = []
    judged, document, lock = judged_folder(root)
    name = document["experiment"]
    licence = judged.licence
    if licence is None:
        refusals.append(_refuse(
            "push.licence", RefusalClass.USER,
            f"the experiment {plain(name)} declares no licence, and a record is never published without one: its "
            f"own files would be its author's, with no licence granted to anyone",
            f"declare it where the experiment is built, exp.licence({sharing.licence!r}) (sharing.yaml states "
            f"{plain(sharing.licence)}), then save it again, built.save(folder)"))
    elif licence != sharing.licence:
        refusals.append(_refuse(
            "push.licence", RefusalClass.USER,
            f"sharing.yaml states the licence {plain(sharing.licence)}, and the experiment declares {plain(licence)}, "
            f"which its files carry: the record would say one and its NOTICE the other",
            f"write `licence: {plain(licence)}` in {sharing_path(root)}, or declare the other where the experiment is "
            f"built, exp.licence({sharing.licence!r}), and save it again, built.save(folder)"))
    if judged.permissions:
        refusals += _grantee_problems(judged.permissions, sharing, root)
    refs = folder_assets(root)
    if service.production:
        refusals += [_refuse(
            "push.sandbox-location", RefusalClass.USER,
            f"the lock's asset {plain(ref.name)} names a sandbox location, {plain(location)}, which the sandbox can "
            f"clean at any time and which no production record may point to",
            "save the experiment again from its script, built.save(folder), so the lock is written as MIME ships its "
            "assets; then `mime push <folder> --to zenodo`")
            for ref in refs for location in ref.locations if _SANDBOX_LOCATION.match(location)]
    try:
        bundle = pack(root, out=staged / f"{re.sub(r'[^A-Za-z0-9._-]', '_', name) or 'experiment'}.mimebundle")
    except BuildRefused as refused:
        if not refusals:
            raise
        raise BuildRefused([*refusals, *refused.refusals]) from None
    if refusals:
        raise BuildRefused(refusals)
    with zipfile.ZipFile(bundle) as archive:
        read_manifest(archive)
        notice = archive.read(NOTICE_MEMBER)
        assert MANIFEST_MEMBER in archive.namelist()
    notice_path = staged / NOTICE_KEY
    notice_path.write_bytes(notice)
    version = lock["environment"]["mime"]
    files = [_record_file(bundle.name, bundle, judged.licence, "the bundle, packed from the folder now"),
             _record_file(NOTICE_KEY, notice_path, None,
                          "each file's licence, credit, verdict and declared permission (the bundle carries it too)")]
    choices = asset_choices(refs, judged, lock)
    by_asset = {asset: path for path, asset in lock.get("referenced_files", {}).items()}
    for choice in choices:
        if not choice.upload:
            continue
        placed = by_asset.get(choice.ref.name)
        source = root / placed if placed is not None and (root / placed).is_file() else locate(choice.ref, folder=root)
        key = record_key(choice.ref, placed)
        if key in {file.key for file in files}:
            key = f"{choice.ref.sha256[:12]}-{key}"
        held = _record_file(key, source, choice.ref.licence, f"the asset {plain(choice.ref.name)}: {choice.why}")
        if (held.sha256, held.size) != (choice.ref.sha256, choice.ref.size):
            refusals.append(_refuse(
                "push.checksum", RefusalClass.ENVIRONMENT,
                f"the copy of the asset {plain(choice.ref.name)} at {source} is not its bytes (sha256 {held.sha256}, "
                f"and the lock names {choice.ref.sha256})", f"hash the cache again, `mime assets check`, and fetch "
                                                            f"it again, `mime assets fetch {plain(choice.ref.name)}`"))
        files.append(held)
    refusals += [_refuse("push.too-many-files", RefusalClass.USER,
                         f"the record would hold {len(files)} files, and Zenodo holds at most {MOST_FILES}",
                         "move assets to an asset record of their own (`mime push --asset`, which comes later), or "
                         "reference them with an origin, Terms(..., origin=(...)), and save again")
                 ] if len(files) > MOST_FILES else []
    total = sum(file.size for file in files)
    if total > MOST_BYTES or any(file.size > MOST_BYTES for file in files):
        refusals.append(_refuse(
            "push.too-large", RefusalClass.USER,
            f"the record would hold {size_text(total)}, and Zenodo holds at most {size_text(MOST_BYTES)} in a record "
            f"and in any one file", "move assets to an asset record of their own (`mime push --asset`, which comes "
                                    "later), or reference them with an origin, Terms(..., origin=(...)), and save "
                                    "again"))
    description = card_html(card_text(root))
    if sharing.description:
        description = f"<p>{html.escape(sharing.description, quote=False)}</p>\n" + description
    own = licence or sharing.licence
    payload = payload_of((sharing, judged, record_licences(judged, choices, own), description, version),
                         today or datetime.datetime.now(datetime.UTC).date())
    refusals += scan_text("the record's metadata", json.dumps(payload, sort_keys=True).encode("utf-8"),
                          held_tokens())
    if refusals:
        raise BuildRefused(refusals)
    return Plan(service, name, root, tuple(files), judged, tuple(choices), own, sharing, payload, version)


def _record_file(key: str, path: Path, licence: str | None, what: str) -> RecordFile:
    md5, sha256, size = file_hashes(path)
    return RecordFile(key, path, size, md5, sha256, licence, what)


def _grantee_problems(permissions: Sequence[PermissionDoc], sharing: Sharing, root: Path) -> list[Refusal]:
    """A declared permission's grantee is the uploader or their institution (#198 item 6), whom sharing.yaml names."""
    if sharing.uploader is None:
        return [_refuse("push.permission", RefusalClass.USER,
                        f"the experiment declares {len(permissions)} permission{'s' if len(permissions) != 1 else ''}, "
                        f"and sharing.yaml names no uploader to hold its grantee to",
                        f"name yourself, `uploader: {{name: \"<family name>, <given name>\", institution: <your "
                        f"institution>}}` in {sharing_path(root)}")]
    allowed = {sharing.uploader.name, *([sharing.uploader.institution] if sharing.uploader.institution else [])}
    return [_refuse("push.permission", RefusalClass.USER,
                    f"permission {index} names the grantee {plain(held['grantee'])}, who is neither the uploader "
                    f"({plain(sharing.uploader.name)}) nor their institution: the declaration is the uploader's own",
                    f"declare the permission to yourself or your institution, exp.permission(Permission(grantee=...)),"
                    f" and save again; or name the right uploader in {sharing_path(root)}")
            for index, held in enumerate(permissions, start=1) if held["grantee"] not in allowed]


__all__ = ["DOI_FIELDS", "FIELD_OWNERS", "KEYWORD", "MOST_BYTES", "MOST_FILES", "NOTICE_KEY", "OWN_RELATED",
           "PUBLISHER", "AssetChoice", "field_owner", "Plan", "RecordFile",
           "asset_choices", "card_html", "file_hashes", "make_plan", "on_zenodo", "payload_of", "record_key",
           "record_licences", "rights_entries"]
