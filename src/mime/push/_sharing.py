"""``[record]``: the record's credit and description, in ``mime.toml`` beside the saved form and outside it.

Credit lives here, not in the document (design #199 question 1): Zenodo keeps a record's metadata editable and its files
immutable, so a misspelt name is a metadata edit, never a new version. ``[record]`` is read against a closed schema: a key
missing, unknown or of another type is refused by its path (``manifest.key``), and a creator's or contributor's ORCID is
held to its form and its check digit (``push.creators``). The checks are each key's, whatever the kind of the folder
(:mod:`mime.push._manifest` reads the file and its kind table).

The experiment's own licence is bound to its files (``exp.licence(...)``, in the document and the input hash); the kind
table states it again (``[bundle] licence``), pre-filled ``CC-BY-4.0`` where the experiment declares none (question 2), and
push holds the two equal, so the licence a record shows is always the one its files carry and the author has written it
twice.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from mime._terminal import plain
from mime.build import Refusal, RefusalClass
from mime.push._fields import (
    ROR_FORM,
    Funding,
    access_problems,
    conference_problems,
    field_problems,
    funding_problems,
    ror_identifier,
)
from mime.terms import licence_problem

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
#: The relation push states of each work ``also_cite`` names, where ``related`` names it under none of its own.
ALSO_CITE_RELATION = "IsSupplementTo"
#: An ORCID iD: ASCII digits alone, each group's and the check character's (``\d`` and ``int`` would take any script's
#: digits, a full-width one a CJK input method types among them, which read as digits on a terminal).
_ORCID = re.compile(r"(?:https://orcid\.org/)?([0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X])")
_DOI = re.compile(r"10\.\d{4,9}/\S+")
#: ``[record]``'s keys, in the order ``--init`` writes them and ``--setup`` asks them.
RECORD_KEYS = ("title", "description", "creators", "contributors", "keywords", "related", "also_cite", "uploader",
               "funding", "conference", "fields", "access")
_PERSON = ("family", "given", "organisation", "orcid", "affiliation", "ror")
#: The line each key is written as, said where its value is refused.
LINES = {
    "title": 'title = "<the record\'s title>"',
    "description": 'description = "<a paragraph shown above the card, or nothing>"',
    "licence": f'licence = "{DEFAULT_LICENCE}"',
    "creators": 'creators = [{ family = "<family name>", given = "<given name>", orcid = "0000-0000-0000-0000" }]',
    "contributors": 'contributors = [{ family = "<family name>", given = "<given name>", type = "DataCurator" }]',
    "keywords": 'keywords = ["<a keyword>"]',
    "related": 'related = [{ identifier = "<10.xxxx/... or https://...>", relation = "IsSupplementTo" }]',
    "also_cite": 'also_cite = [{ identifier = "<10.xxxx/... or https://...>", text = "<the work, as it is cited>" }]',
    "uploader": 'uploader = { name = "<family name>, <given name>", institution = "<your institution>" }',
    "funding": 'funding = [{ funder = "<the funder\'s name>", ror = "<its ROR id>", award = "<number>", '
               'title = "<its title>" }]',
    "conference": 'conference = { title = "<the conference\'s name>", acronym = "<...>", dates = "<...>", '
                  'place = "<...>" }',
    "fields": 'fields = { languages = [{ id = "eng" }] }',
    "access": 'access = { files = "restricted", embargo = { until = "2030-01-01", reason = "<why>" } }',
}


def orcid_problem(text: str) -> str | None:
    """Why ``text`` is not an ORCID iD, or ``None``: sixteen characters in four groups (or its ``https://orcid.org/``
    form), the last a check digit by ISO 7064 MOD 11-2 (ORCID's own documentation of the iD's structure)."""
    found = _ORCID.fullmatch(text)
    if found is None:
        return ("an ORCID iD is four groups of four, 0000-0000-0000-000X, the last character a digit or X, each digit "
                "an ASCII one (0 to 9)")
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
class AlsoCite:
    """A work the author asks whoever uses the record to cite as well (a paper, often added after publishing): a DOI or
    an ``https://`` address, and how it is cited, where given."""

    identifier: str
    text: str | None = None

    def words(self) -> str:
        return f"{plain(self.text)} ({plain(self.identifier)})" if self.text else plain(self.identifier)


@dataclass(frozen=True)
class Uploader:
    """The person whose token pushes, and their institution: a declared permission's grantee is one of the two."""

    name: str
    institution: str | None = None


@dataclass(frozen=True)
class Sharing:
    """``[record]`` as read, with the licence the kind table states (held equal to the experiment's): the record's title
    and its description's first paragraph, the creators, the contributors, keywords, related works, the uploader, the
    funding, a conference, any other field of the record (:mod:`mime.push._fields`), who may read it where that is not
    everyone, and the works whoever uses it is asked to cite as well."""

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
    #: Who may read the record and its files once published, and an embargo; empty for public.
    access: Mapping[str, object] = field(default_factory=lambda: cast(Mapping[str, object], {}))
    #: The works whoever uses the record is asked to cite as well.
    also_cite: tuple[AlsoCite, ...] = ()

    def related_with_citations(self) -> tuple[Related, ...]:
        """The related works push states: ``related``, then each work ``also_cite`` names that ``related`` does not, as
        :data:`ALSO_CITE_RELATION`."""
        named = {each.identifier for each in self.related}
        return (*self.related, *(Related(each.identifier, ALSO_CITE_RELATION) for each in self.also_cite
                                 if each.identifier not in named))


def as_text(value: object) -> object:
    """``value`` with every TOML date and time in it as its ISO text, as the record holds a day."""
    if isinstance(value, datetime.date | datetime.time):
        return value.isoformat()
    if isinstance(value, list):
        return [as_text(each) for each in cast(list[object], value)]
    if isinstance(value, dict):
        return {key: as_text(each) for key, each in cast(dict[str, object], value).items()}
    return value


@dataclass(frozen=True)
class Problem:
    """A problem of a manifest: its refusal, the line to write (``None`` where there is none to give) and the table it
    goes in (``""`` for the top), and whether it is a value push still needs (``needed``: none written yet) rather than
    one that is wrong."""

    refusal: Refusal
    line: str | None = None
    needed: bool = False
    table: str = "record"


class Reader:
    """The checks of one file, gathering every problem before any is refused: a key's problem is ``manifest.key``, or
    ``defaults.file`` in a person's ``defaults``; ``table`` is the table its keys are in (``""`` for the top)."""

    def __init__(self, path: Path, *, defaults: bool = False, table: str = "record") -> None:
        self.path = path
        self.defaults = defaults
        self.table = table
        self.problems: list[Problem] = []

    @property
    def refusals(self) -> list[Refusal]:
        return [each.refusal for each in self.problems]

    def where(self, key: str) -> str:
        return f"[{self.table}] {key}".rstrip() if self.table else key

    def problem(self, where: str, problem: str, line: str | None = None, *, needed: bool = False) -> None:
        key = where.split(".")[0].split("[")[0]
        written = line or LINES.get(key)
        place = f" under [{self.table}]" if self.table else ""
        action = f"write it as `{written}`{place} in {self.path}" if written else f"correct it in {self.path}"
        code = "defaults.file" if self.defaults else "manifest.key"
        self.problems.append(Problem(Refusal(code, RefusalClass.USER,
                                             f"{self.path.name}, {self.where(where)}: {problem}", action),
                                     written, needed, self.table))

    def text(self, held: Mapping[str, object], key: str, where: str, *, required: bool = False,
             blank: bool = False) -> str | None:
        value = held.get(key)
        if value is None:
            if required:
                self.problem(f"{where}{key}", "is missing", needed=not where)
            return None
        if not isinstance(value, str) or (not blank and not value.strip()):
            self.problem(f"{where}{key}", f"is {type(value).__name__} {plain(repr(value))[:60]}, not text")
            return None
        return value.strip()

    def items(self, held: Mapping[str, object], key: str) -> list[object]:
        value = held.get(key, [])
        if not isinstance(value, list):
            self.problem(key, f"is {type(value).__name__}, not a list")
            return []
        return cast(list[object], value)

    def mapping(self, value: object, where: str, allowed: Sequence[str]) -> Mapping[str, object] | None:
        if not isinstance(value, dict):
            self.problem(where, f"is {type(value).__name__}, not a table of keys")
            return None
        held = cast(dict[str, object], value)
        for key in held:
            if key not in allowed:
                self.problem(f"{where}.{plain(key)}" if where else plain(key),
                             f"is not a key it holds (it holds {', '.join(allowed)})")
        return held

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
                self.problems.append(Problem(Refusal(
                    "push.creators", RefusalClass.USER,
                    f"{self.path.name}, {self.where(where)}.orcid: {plain(orcid)}: {problem}",
                    f"copy the iD from the person's ORCID record, `orcid = \"0000-0000-0000-0000\"` in {self.path}"),
                    'orcid = "0000-0000-0000-0000"', table=self.table))
            elif organisation is not None:
                self.problem(f"{where}.orcid", "is an ORCID iD, which a person holds and an organisation does not")
            orcid = bare_orcid(orcid)
        ror = self.text(held, "ror", f"{where}.")
        identifier = ror_identifier(ror) if ror is not None else None
        if ror is not None and identifier is None:
            self.problem(f"{where}.ror", f"{plain(ror)} is not a ROR identifier ({ROR_FORM})", 'ror = "05xxxxx00"')
        affiliation = self.text(held, "affiliation", f"{where}.")
        if ror is not None and affiliation is None:
            self.problem(f"{where}.ror", "is given without the affiliation it identifies",
                         'affiliation = "<its name>"')
        role = None
        if contributor:
            role = self.text(held, "type", f"{where}.", required=True)
            if role is not None and role not in CONTRIBUTOR_TYPES:
                self.problem(f"{where}.type", f"{plain(role)} is not one of DataCite's contributor types "
                                              f"({', '.join(CONTRIBUTOR_TYPES)})", 'type = "DataCurator"')
        return Person(family, given, organisation, orcid, affiliation, identifier, role)

    def identified(self, value: object, where: str, other: str) -> tuple[str | None, str | None]:
        """A related work's or a work to cite's ``identifier`` (a DOI or an ``https://`` address) and its ``other``
        key (``relation``, ``text``), each read; the identifier ``None`` where it is not one."""
        entry = self.mapping(value, where, ("identifier", other))
        if entry is None:
            return None, None
        identifier = self.text(entry, "identifier", f"{where}.", required=True)
        if identifier is not None and (problem := identifier_problem(identifier)) is not None:
            self.problem(f"{where}.identifier", problem)
            identifier = None
        return identifier, self.text(entry, other, f"{where}.", required=other == "relation")


def people(reader: Reader, held: Mapping[str, object], key: str) -> list[Person | None]:
    """``creators`` or ``contributors`` (``key``) read, each by :meth:`Reader.person`."""
    return [reader.person(each, f"{key}[{index}]", contributor=key == "contributors")
            for index, each in enumerate(reader.items(held, key))]


def keyword_list(reader: Reader, held: Mapping[str, object]) -> list[str | None]:
    """``keywords`` read: each a word or a phrase."""
    return [reader.text({"keyword": each}, "keyword", f"keywords[{index}].")
            for index, each in enumerate(reader.items(held, "keywords"))]


def uploader_of(reader: Reader, held: Mapping[str, object]) -> Uploader | None:
    """``uploader`` read: the person whose token pushes, and their institution."""
    if held.get("uploader") is None:
        return None
    entry = reader.mapping(held["uploader"], "uploader", ("name", "institution"))
    name = reader.text(entry, "name", "uploader.", required=True) if entry is not None else None
    if entry is None or name is None:
        return None
    return Uploader(name, reader.text(entry, "institution", "uploader."))


def funding_of(reader: Reader, held: Mapping[str, object]) -> list[Funding]:
    """``funding`` read: each funder and its award."""
    funding, problems = funding_problems(held.get("funding"))
    for problem in problems:
        where, _, what = problem.partition(": ")
        reader.problem(where, what)
    return funding


def ror_problem(text: str) -> str | None:
    """Why ``text`` is not a ROR identifier, or ``None``."""
    return None if ror_identifier(text) is not None else f"{plain(text)} is not a ROR identifier ({ROR_FORM})"


def identifier_problem(text: str) -> str | None:
    """Why ``text`` is not a related work's identifier (a DOI or an ``https://`` address), or ``None``."""
    if _DOI.fullmatch(text) or (text.startswith("https://") and " " not in text):
        return None
    return f"{plain(text)} is neither a DOI (10.xxxx/...) nor an https:// address"


def record_of(reader: Reader, value: object, licence: str | None) -> Sharing | None:
    """``[record]`` (``value``) read, with the licence the kind table states, every problem gathered in ``reader``;
    ``None`` where one is found."""
    if value is None:
        reader.problem("the table", "is missing: a record's title and creators are kept there", "[record]",
                       needed=True)
        return None
    held = reader.mapping(as_text(value), "", RECORD_KEYS)
    if held is None:
        return None
    title = reader.text(held, "title", "", required=True)
    description = reader.text(held, "description", "", blank=True) or ""
    creators = people(reader, held, "creators")
    if not held.get("creators"):
        reader.problems.append(Problem(Refusal(
            "push.creators", RefusalClass.USER,
            f"{reader.path.name} names no creator: a record names the people who made it",
            f"name each, `{LINES['creators']}` under [record] in {reader.path}"), LINES["creators"], needed=True))
    contributors = people(reader, held, "contributors")
    keywords = keyword_list(reader, held)
    related: list[Related] = []
    for index, each in enumerate(reader.items(held, "related")):
        identifier, relation = reader.identified(each, f"related[{index}]", "relation")
        if relation is not None and relation not in RELATION_TYPES:
            reader.problem(f"related[{index}].relation", f"{plain(relation)} is not one of DataCite's relation types "
                                                         f"(IsSupplementTo, References, Cites, IsDerivedFrom, ...)")
        elif identifier is not None and relation is not None:
            related.append(Related(identifier, relation))
    also_cite: list[AlsoCite] = []
    for index, each in enumerate(reader.items(held, "also_cite")):
        identifier, text = reader.identified(each, f"also_cite[{index}]", "text")
        if identifier is not None:
            also_cite.append(AlsoCite(identifier, text))
    uploader = uploader_of(reader, held)
    funding = funding_of(reader, held)
    conference, problems = conference_problems(held.get("conference"))
    fields: object = held.get("fields", {})
    problems += field_problems(fields)
    access, found_access = access_problems(held.get("access"))
    problems += found_access
    for problem in problems:
        where, _, what = problem.partition(": ")
        reader.problem(where, what)
    if reader.problems or title is None or licence is None:
        return None
    return Sharing(title, description, licence, tuple(p for p in creators if p is not None),
                   tuple(p for p in contributors if p is not None), tuple(k for k in keywords if k is not None),
                   tuple(related), uploader, tuple(funding), conference, cast(Mapping[str, object], fields), access,
                   tuple(also_cite))


def licence_of(reader: Reader, held: Mapping[str, object]) -> str | None:
    """A kind table's ``licence``, read: an SPDX expression the installed packaging knows."""
    licence = reader.text(held, "licence", "", required=True)
    if licence is not None and (problem := licence_problem(licence)) is not None:
        reader.problem("licence", plain(problem))
        return None
    return licence


__all__ = ["ALSO_CITE_RELATION", "CONTRIBUTOR_TYPES", "DEFAULT_LICENCE", "LINES", "RECORD_KEYS", "RELATION_TYPES",
           "AlsoCite", "Person", "Problem", "Reader", "Related", "Sharing", "Uploader", "as_text", "bare_orcid",
           "funding_of", "identifier_problem", "keyword_list", "licence_of", "orcid_problem", "people", "record_of",
           "ror_problem", "uploader_of"]
