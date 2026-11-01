"""``sharing.yaml``: the record's credit and description, beside the saved form and outside it.

Credit lives here, not in the document (design #199 question 1): Zenodo keeps a record's metadata editable and its files
immutable, so a misspelt name is a metadata edit, never a new version. ``mime push <folder> --to <service> --init``
writes it from what the folder knows (the experiment's name and licence), and the author edits it by hand. It is read
with YAML's safe loader against a closed schema: a key missing, unknown or of another type is refused by its path
(``push.sharing``), and a creator's or contributor's ORCID is held to its form and its check digit (``push.creators``).
``pack`` neither carries nor refuses the file.

The experiment's own licence is bound to its files (``exp.licence(...)``, in the document and the input hash); the file
states it again, pre-filled ``CC-BY-4.0`` where the experiment declares none (question 2), and push holds the two
equal, so the licence a record shows is always the one its files carry and the author has written it twice.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import yaml

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._fields import Funding, conference_problems, field_problems, funding_problems
from mime.terms import licence_problem

SHARING_FILE = "sharing.yaml"
SHARING_FORMAT = "mime.sharing/1"
#: The licence ``--init`` writes where the experiment declares none: the coordinator's lean, agreed (#199 question 2).
DEFAULT_LICENCE = "CC-BY-4.0"
#: DataCite's contributor types (DataCite Metadata Schema 4.5, contributorType), which Zenodo's contributor roles are.
CONTRIBUTOR_TYPES = ("ContactPerson", "DataCollector", "DataCurator", "DataManager", "Distributor", "Editor",
                     "HostingInstitution", "Producer", "ProjectLeader", "ProjectManager", "ProjectMember",
                     "RegistrationAgency", "RegistrationAuthority", "RelatedPerson", "Researcher", "ResearchGroup",
                     "RightsHolder", "Sponsor", "Supervisor", "WorkPackageLeader", "Other")
#: DataCite's relation types (relatedIdentifier's relationType) a record may state of another work.
RELATION_TYPES = ("IsCitedBy", "Cites", "IsSupplementTo", "IsSupplementedBy", "IsContinuedBy", "Continues",
                  "IsDescribedBy", "Describes", "HasMetadata", "IsMetadataFor", "HasVersion", "IsVersionOf",
                  "IsNewVersionOf", "IsPreviousVersionOf", "IsPartOf", "HasPart", "IsPublishedIn", "IsReferencedBy",
                  "References", "IsDocumentedBy", "Documents", "IsCompiledBy", "Compiles", "IsVariantFormOf",
                  "IsOriginalFormOf", "IsIdenticalTo", "IsReviewedBy", "Reviews", "IsDerivedFrom", "IsSourceOf",
                  "IsRequiredBy", "Requires", "IsObsoletedBy", "Obsoletes")
_ORCID = re.compile(r"(?:https://orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])")
_ROR = re.compile(r"(?:https://ror\.org/)?(0[a-z0-9]{6}\d{2})")
_DOI = re.compile(r"10\.\d{4,9}/\S+")
_TOP = ("format", "title", "description", "licence", "creators", "contributors", "keywords", "related", "uploader",
        "funding", "conference", "fields")
_PERSON = ("family", "given", "organisation", "orcid", "affiliation", "ror")
_LINE = {
    "format": f"format: {SHARING_FORMAT}",
    "title": "title: <the record's title>",
    "description": "description: <a paragraph shown above the card, or nothing>",
    "licence": f"licence: {DEFAULT_LICENCE}",
    "creators": "creators: [{family: <family name>, given: <given name>, orcid: <0000-0000-0000-0000>}]",
    "contributors": "contributors: [{family: <family name>, given: <given name>, type: DataCurator}]",
    "keywords": "keywords: [<a keyword>]",
    "related": "related: [{identifier: <10.xxxx/... or https://...>, relation: IsSupplementTo}]",
    "uploader": "uploader: {name: \"<family name>, <given name>\", institution: <your institution>}",
    "funding": "funding: [{funder: <the funder's name>, ror: <its ROR id>, award: <number>, title: <its title>}]",
    "conference": "conference: {title: <the conference's name>, acronym: <...>, dates: <...>, place: <...>}",
    "fields": "fields: {languages: [{id: eng}]}",
}


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> Refusal:
    return Refusal(code, category, message, action)


def orcid_problem(text: str) -> str | None:
    """Why ``text`` is not an ORCID iD, or ``None``: sixteen characters in four groups (or its ``https://orcid.org/``
    form), the last a check digit by ISO 7064 MOD 11-2 (ORCID's own documentation of the iD's structure)."""
    found = _ORCID.fullmatch(text)
    if found is None:
        return "an ORCID iD is four groups of four, 0000-0000-0000-000X, the last character a digit or X"
    digits = found.group(1).replace("-", "")
    total = 0
    for character in digits[:-1]:
        total = (total + int(character)) * 2
    result = (12 - total % 11) % 11
    expected = "X" if result == 10 else str(result)
    if digits[-1] != expected:
        return (f"the ORCID iD's check digit is {digits[-1]}, and its first fifteen digits give {expected}: a digit "
                f"is mistyped")
    return None


def bare_orcid(text: str) -> str:
    found = _ORCID.fullmatch(text)
    return found.group(1) if found is not None else text


@dataclass(frozen=True)
class Person:
    """A creator or a contributor: a person (``family``, with ``given`` where they have one) or an ``organisation``,
    an ORCID iD for a person where they have one, an affiliation and its ROR identifier where given; ``role`` is a
    contributor's DataCite type (``None`` for a creator)."""

    family: str | None = None
    given: str | None = None
    organisation: str | None = None
    orcid: str | None = None
    affiliation: str | None = None
    ror: str | None = None
    role: str | None = None

    def name(self) -> str:
        if self.organisation is not None:
            return self.organisation
        return f"{self.family}, {self.given}" if self.given else str(self.family)

    def words(self) -> str:
        """The person as the plan and the card name them."""
        parts = [plain(self.name())]
        if self.orcid:
            parts.append(f"ORCID {self.orcid}")
        if self.affiliation:
            parts.append(plain(self.affiliation) + (f", ROR {self.ror}" if self.ror else ""))
        if self.role:
            parts.append(self.role)
        return parts[0] + (f" ({'; '.join(parts[1:])})" if len(parts) > 1 else "")


@dataclass(frozen=True)
class Related:
    """Another work the record relates to: a DOI or an ``https://`` address, and DataCite's relation."""

    identifier: str
    relation: str

    @property
    def scheme(self) -> str:
        return "doi" if _DOI.fullmatch(self.identifier) else "url"


@dataclass(frozen=True)
class Uploader:
    """The person whose token pushes, and their institution: a declared permission's grantee is one of the two."""

    name: str
    institution: str | None = None


@dataclass(frozen=True)
class Sharing:
    """``sharing.yaml`` as read: the record's title and its description's first paragraph, the licence the author states
    (held equal to the experiment's), the creators, the contributors, keywords, related works, the uploader, the
    funding, a conference, and any other field of the record (:mod:`mime.push._fields`)."""

    title: str
    description: str
    licence: str
    creators: tuple[Person, ...]
    contributors: tuple[Person, ...] = ()
    keywords: tuple[str, ...] = ()
    related: tuple[Related, ...] = ()
    uploader: Uploader | None = None
    funding: tuple[Funding, ...] = ()
    #: Zenodo's meeting field, as the records API holds it.
    conference: Mapping[str, str] | None = None
    #: Any other field of the record, checked against its shape (:data:`mime.push._fields.SHAPES`).
    fields: Mapping[str, object] = field(default_factory=lambda: cast(Mapping[str, object], {}))


class _Reader:
    """The checks of one file, gathering every refusal before any is raised."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.refusals: list[Refusal] = []

    def problem(self, where: str, problem: str, line: str | None = None) -> None:
        key = where.split(".")[0].split("[")[0]
        self.refusals.append(_refuse(
            "push.sharing", RefusalClass.USER, f"{self.path.name}, {where}: {problem}",
            f"write it as `{line or _LINE.get(key, _LINE['title'])}` in {self.path}, then `mime push <folder> --to "
            f"<service>` again"))

    def text(self, held: Mapping[str, object], key: str, where: str, *, required: bool = False,
             blank: bool = False) -> str | None:
        value = held.get(key)
        if value is None:
            if required:
                self.problem(f"{where}{key}", "is missing")
            return None
        if not isinstance(value, str) or (not blank and not value.strip()):
            self.problem(f"{where}{key}", f"is {type(value).__name__} {plain(repr(value))[:60]}, not text")
            return None
        return value.strip()

    def items(self, held: Mapping[str, object], key: str) -> list[object]:
        value = held.get(key, [])
        if value is None:
            return []
        if not isinstance(value, list):
            self.problem(key, f"is {type(value).__name__}, not a list")
            return []
        return cast(list[object], value)

    def mapping(self, value: object, where: str, allowed: Sequence[str]) -> Mapping[str, object] | None:
        if not isinstance(value, dict):
            self.problem(where, f"is {type(value).__name__}, not a mapping of keys")
            return None
        held = cast(dict[object, object], value)
        for key in held:
            if not isinstance(key, str) or key not in allowed:
                self.problem(f"{where}.{plain(str(key))}" if where else plain(str(key)),
                             f"is not a key it holds (it holds {', '.join(allowed)})")
        return cast(Mapping[str, object], held)

    def person(self, value: object, where: str, *, contributor: bool) -> Person | None:
        held = self.mapping(value, where, (*_PERSON, "type") if contributor else _PERSON)
        if held is None:
            return None
        family = self.text(held, "family", f"{where}.")
        given = self.text(held, "given", f"{where}.")
        organisation = self.text(held, "organisation", f"{where}.")
        if (family is None) == (organisation is None):
            self.problem(where, "names a person (family, and given where they have one) or an organisation, one of "
                                "the two")
        orcid = self.text(held, "orcid", f"{where}.")
        if orcid is not None:
            problem = orcid_problem(orcid)
            if problem is not None:
                self.refusals.append(_refuse(
                    "push.creators", RefusalClass.USER, f"{self.path.name}, {where}.orcid: {plain(orcid)}: {problem}",
                    f"copy the iD from the person's ORCID record, `orcid: 0000-0000-0000-0000` in {self.path}"))
            elif organisation is not None:
                self.problem(f"{where}.orcid", "is an ORCID iD, which a person holds and an organisation does not")
            orcid = bare_orcid(orcid)
        ror = self.text(held, "ror", f"{where}.")
        if ror is not None and _ROR.fullmatch(ror) is None:
            self.problem(f"{where}.ror", f"{plain(ror)} is not a ROR identifier (0 and eight letters and digits, as "
                                         "https://ror.org/ lists it)", "ror: 05xxxxx00")
        affiliation = self.text(held, "affiliation", f"{where}.")
        if ror is not None and affiliation is None:
            self.problem(f"{where}.ror", "is given without the affiliation it identifies", "affiliation: <its name>")
        role = None
        if contributor:
            role = self.text(held, "type", f"{where}.", required=True)
            if role is not None and role not in CONTRIBUTOR_TYPES:
                self.problem(f"{where}.type", f"{plain(role)} is not one of DataCite's contributor types "
                                              f"({', '.join(CONTRIBUTOR_TYPES)})", "type: DataCurator")
        found = _ROR.fullmatch(ror) if ror is not None else None
        return Person(family, given, organisation, orcid, affiliation, found.group(1) if found else None, role)


def sharing_path(folder: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(folder)) / SHARING_FILE


def read_sharing(folder: str | os.PathLike[str]) -> Sharing:
    """The ``sharing.yaml`` of a saved folder, checked: refused with every reason at once (``push.sharing-missing``,
    ``push.sharing``, ``push.creators``)."""
    path = sharing_path(folder)
    if not path.is_file():
        raise BuildRefused([_refuse(
            "push.sharing-missing", RefusalClass.USER,
            f"{path.parent.name} has no {SHARING_FILE} beside its saved form: the record's title, creators and "
            f"licence are kept there", f"write it from what the folder knows, `mime push {path.parent} --to "
                                       f"<service> --init`, then edit its creators")])
    reader = _Reader(path)
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as error:
        reader.problem("the file", f"is not YAML that reads ({type(error).__name__})", _LINE["title"])
        raise BuildRefused(reader.refusals) from None
    held: Mapping[str, object] | None = reader.mapping(loaded, "", _TOP) if loaded is not None else {}
    if held is None:
        raise BuildRefused(reader.refusals)
    if held.get("format") != SHARING_FORMAT:
        reader.problem("format", f"is {plain(repr(held.get('format')))}, and this MIME reads {SHARING_FORMAT}")
    title = reader.text(held, "title", "", required=True)
    description = reader.text(held, "description", "", blank=True) or ""
    licence = reader.text(held, "licence", "", required=True)
    if licence is not None and (problem := licence_problem(licence)) is not None:
        reader.problem("licence", plain(problem))
    creators = [reader.person(each, f"creators[{index}]", contributor=False)
                for index, each in enumerate(reader.items(held, "creators"))]
    if held.get("creators") in (None, []):
        reader.refusals.append(_refuse(
            "push.creators", RefusalClass.USER, f"{path.name} names no creator: a record names the people who made it",
            f"name each, `{_LINE['creators']}` in {path}"))
    contributors = [reader.person(each, f"contributors[{index}]", contributor=True)
                    for index, each in enumerate(reader.items(held, "contributors"))]
    keywords = [reader.text({"keyword": each}, "keyword", f"keywords[{index}].")
                for index, each in enumerate(reader.items(held, "keywords"))]
    related: list[Related] = []
    for index, each in enumerate(reader.items(held, "related")):
        entry = reader.mapping(each, f"related[{index}]", ("identifier", "relation"))
        if entry is None:
            continue
        identifier = reader.text(entry, "identifier", f"related[{index}].", required=True)
        relation = reader.text(entry, "relation", f"related[{index}].", required=True)
        if identifier is not None and not (_DOI.fullmatch(identifier) or (identifier.startswith("https://")
                                                                          and " " not in identifier)):
            reader.problem(f"related[{index}].identifier", f"{plain(identifier)} is neither a DOI (10.xxxx/...) nor "
                                                           f"an https:// address")
        if relation is not None and relation not in RELATION_TYPES:
            reader.problem(f"related[{index}].relation", f"{plain(relation)} is not one of DataCite's relation types "
                                                         f"(IsSupplementTo, References, Cites, IsDerivedFrom, ...)")
        if identifier is not None and relation is not None:
            related.append(Related(identifier, relation))
    uploader = None
    if held.get("uploader") is not None:
        entry = reader.mapping(held["uploader"], "uploader", ("name", "institution"))
        name = reader.text(entry, "name", "uploader.", required=True) if entry is not None else None
        if entry is not None and name is not None:
            uploader = Uploader(name, reader.text(entry, "institution", "uploader."))
    funding, problems = funding_problems(held.get("funding"))
    conference, found = conference_problems(held.get("conference"))
    problems += found
    fields: object = held.get("fields") or {}
    problems += field_problems(fields)
    for problem in problems:
        where, _, what = problem.partition(": ")
        reader.problem(where, what)
    if reader.refusals:
        raise BuildRefused(reader.refusals)
    assert title is not None and licence is not None
    return Sharing(title, description, licence, tuple(p for p in creators if p is not None),
                   tuple(p for p in contributors if p is not None), tuple(k for k in keywords if k is not None),
                   tuple(related), uploader, tuple(funding), conference, cast(Mapping[str, object], fields))


def scaffold(name: str, licence: str | None) -> str:
    """The text ``--init`` writes: the experiment's name as the title, its licence (``CC-BY-4.0`` where it declares
    none), no creator yet, each key explained where it stands."""
    title = yaml.safe_dump(name, default_style='"', width=10_000).strip()
    return "\n".join([
        "# The record's metadata for `mime push`: credit and description, kept beside the saved form and never in",
        "# it (a change here is a metadata edit of the record, not a new version). `mime pack` neither carries nor",
        "# refuses this file. Edit it by hand; push reads it with YAML's safe loader against a closed schema.",
        f"format: {SHARING_FORMAT}",
        f"title: {title}",
        "# A paragraph shown above the card, which is the record's description; empty for the card alone.",
        "description: \"\"",
        "# The experiment's own licence, which covers its files: it must equal the licence the experiment declares",
        "# (exp.licence(...), bound to its files when it is saved). " +
        (f"The experiment declares {licence}." if licence else
         f"The experiment declares none: {DEFAULT_LICENCE} is MIME's suggestion, and push asks the experiment to "
         f"declare it too."),
        f"licence: {licence or DEFAULT_LICENCE}",
        "# Each person who made the experiment, as the record names them: family and given names, an ORCID iD where",
        "# they have one, an affiliation (with its ROR identifier where you know it). Everyone named must have agreed",
        "# to be named. For example:",
        "#   - {family: Carberry, given: Josiah, orcid: 0000-0002-1825-0097, affiliation: Brown University}",
        "creators: []",
        "# Others credited, each with a DataCite contributor type (DataCurator, Supervisor, HostingInstitution, ...).",
        "# The rights holder of each third-party file the record holds is added by push from the files' terms.",
        "contributors: []",
        "# Searchable words; push adds mime-experiment.",
        "keywords: []",
        "# Papers and other works: a DOI or an https:// address, and DataCite's relation (IsSupplementTo for a paper",
        "# the experiment supports, References, IsDerivedFrom, ...). push adds the MIME release it requires.",
        "related: []",
        "# The person whose token pushes, and their institution: a declared permission's grantee must be one of them.",
        "# uploader: {name: \"<family name>, <given name>\", institution: <your institution>}",
        "# Who funded the work: each funder's name (with its ROR identifier where you know it), and the award's number",
        "# and title. For example:",
        "#   - {funder: European Commission, ror: 00k4n6c32, award: \"101001234\", title: <the award's title>}",
        "funding: []",
        "# A conference the experiment was shown at (Zenodo's meeting field): title, acronym, dates, place, url,",
        "# session, part. For example:",
        "# conference: {title: <the conference's name>, acronym: <...>, dates: 3-7 May 2027, place: <city>}",
        "# Any other field of the record, in the records API's own names and shapes, checked before anything is sent:",
        "# subjects, dates, languages, references, additional_titles, additional_descriptions, and Zenodo's own",
        "# journal:journal, imprint:imprint, thesis:university and thesis:type. For example:",
        "#   fields: {languages: [{id: eng}], dates: [{date: 2026-10, type: {id: collected}}]}",
        "fields: {}",
        ""])


__all__ = ["CONTRIBUTOR_TYPES", "DEFAULT_LICENCE", "RELATION_TYPES", "SHARING_FILE", "SHARING_FORMAT", "Person",
           "Related", "Sharing", "Uploader", "bare_orcid", "orcid_problem", "read_sharing", "scaffold", "sharing_path"]
