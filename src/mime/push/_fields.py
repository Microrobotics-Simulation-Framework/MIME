"""The record's metadata beyond the credit: funding and a conference, first class in ``sharing.yaml``, and any other
field of the record the author writes, in the records API's own names and shapes, held to those shapes before
anything is sent (design #199 (c)).

**Funding** (``funding:``, a list): each funder by name, with its ROR identifier where the author knows it (Zenodo's
funders are ROR's), and the award, its number and title (and an address). Sent as InvenioRDM's ``funding``: the funder
by its ROR id where one is given and by name where not, and the award as one of the record's own (its title and
number), never as an id of Zenodo's awards vocabulary, which MIME cannot check offline and which, missed, refuses the
whole draft.

**A conference** (``conference:``): Zenodo's meeting field (its custom field ``meeting:meeting``): title, acronym,
dates, place, address, session and part.

**Access** (``access:``): who may read the record and its files once it is published, and an embargo, as the service's
draft page sets them. Left out, the record and its files are public. An edit made on the draft's page is settled as any
other site edit is, and kept here; so an embargo set there is never written over unasked, and what the plan says is what
is published.

**Any other field** (``fields:``): a field of InvenioRDM's metadata MIME does not build (subjects, dates, languages,
references, additional titles and descriptions) or a custom field of Zenodo's (``journal:journal``,
``imprint:imprint``, ``thesis:university``, ``thesis:type``), written as the records API holds it. Each is checked
against its shape here, every problem said by its path at once (``push.sharing``); a field MIME builds, or one this
table does not hold, is refused by name. Every field here is the author's: a refusal the service gives of one says to
correct it in ``sharing.yaml``. The shapes are the records API's as documented and as Zenodo's deposit form shows them,
recalled, not probed: the sandbox session checks each.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from mime._terminal import plain

type Check = Callable[[object, str, list[str]], None]

#: Zenodo's date types (its deposit form's), for ``fields.dates[].type.id``.
DATE_TYPES = ("accepted", "available", "collected", "copyrighted", "created", "issued", "other", "submitted",
              "updated", "valid", "withdrawn")
_DATE = re.compile(r"\d{4}(-\d{2}(-\d{2})?)?(/\d{4}(-\d{2}(-\d{2})?)?)?")
_LANGUAGE = re.compile(r"[a-z]{3}")
_ROR = re.compile(r"(?:https://ror\.org/)?(0[a-z0-9]{6}\d{2})")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
#: Who may read a record or its files.
ACCESS_LEVELS = ("public", "restricted")


def _kind(value: object) -> str:
    return type(value).__name__


def _text(value: object, where: str, problems: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{where}: is {_kind(value)}, not text")


def _matching(pattern: re.Pattern[str], words: str) -> Check:
    def check(value: object, where: str, problems: list[str]) -> None:
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            problems.append(f"{where}: {plain(repr(value))[:60]} is not {words}")
    return check


def _one_of(values: tuple[str, ...]) -> Check:
    def check(value: object, where: str, problems: list[str]) -> None:
        if value not in values:
            problems.append(f"{where}: {plain(repr(value))[:60]} is not one of {', '.join(values)}")
    return check


def _mapping(keys: Mapping[str, tuple[Check, bool]]) -> Check:
    """A mapping of ``keys``, each with its check and whether it is required; no other key."""
    def check(value: object, where: str, problems: list[str]) -> None:
        if not isinstance(value, dict):
            problems.append(f"{where}: is {_kind(value)}, not a mapping of {', '.join(keys)}")
            return
        held = cast(dict[object, object], value)
        for key in held:
            if key not in keys:
                problems.append(f"{where}.{plain(str(key))}: is not a key it holds (it holds {', '.join(keys)})")
        for key, (inner, required) in keys.items():
            if key in held:
                inner(held[key], f"{where}.{key}", problems)
            elif required:
                problems.append(f"{where}.{key}: is missing")
    return check


def _listing(item: Check) -> Check:
    def check(value: object, where: str, problems: list[str]) -> None:
        items = _items(value)
        if not items:
            problems.append(f"{where}: is {_kind(value)}, not a list of one or more")
            return
        for index, each in enumerate(items):
            item(each, f"{where}[{index}]", problems)
    return check


def _vocabulary(inner: Check = _text) -> Check:
    return _mapping({"id": (inner, True)})


def _items(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _keyed(value: object) -> bool:
    return isinstance(value, dict) and "id" in cast(dict[str, object], value)


def _subject(value: object, where: str, problems: list[str]) -> None:
    if _keyed(value):
        _mapping({"id": (_text, True)})(value, where, problems)
    else:
        _mapping({"subject": (_text, True), "scheme": (_text, False)})(value, where, problems)


_LANG = _vocabulary(_matching(_LANGUAGE, "a language's three-letter code, as Zenodo's languages hold it (eng)"))
#: Each field ``fields:`` may hold, with its shape. A name with a colon is a custom field of Zenodo's.
SHAPES: Mapping[str, Check] = {
    "subjects": _listing(_subject),
    "dates": _listing(_mapping({"date": (_matching(_DATE, "a date, 2026, 2026-11 or 2026-11-01, or a range a/b"),
                                         True),
                                "type": (_vocabulary(_one_of(DATE_TYPES)), True), "description": (_text, False)})),
    "languages": _listing(_LANG),
    "references": _listing(_mapping({"reference": (_text, True), "identifier": (_text, False),
                                     "scheme": (_text, False)})),
    "additional_titles": _listing(_mapping({"title": (_text, True), "type": (_vocabulary(), True),
                                            "lang": (_LANG, False)})),
    "additional_descriptions": _listing(_mapping({"description": (_text, True), "type": (_vocabulary(), True),
                                                  "lang": (_LANG, False)})),
    "journal:journal": _mapping({"title": (_text, True), "issn": (_text, False), "volume": (_text, False),
                                 "issue": (_text, False), "pages": (_text, False)}),
    "imprint:imprint": _mapping({"title": (_text, True), "isbn": (_text, False), "place": (_text, False),
                                 "pages": (_text, False), "edition": (_text, False)}),
    "thesis:university": _text,
    "thesis:type": _text,
}
#: Fields of the record MIME builds, or that ``sharing.yaml`` holds under keys of its own: refused in ``fields:``.
BUILT = {"title": "title", "creators": "creators", "contributors": "contributors", "related_identifiers": "related",
         "funding": "funding", "meeting:meeting": "conference", "rights": "the experiment's licence",
         "description": "description", "publisher": "MIME", "publication_date": "MIME", "resource_type": "MIME",
         "identifiers": "MIME", "version": "MIME"}
#: The conference's keys in ``sharing.yaml``, and Zenodo's meeting field's.
CONFERENCE = {"title": "title", "acronym": "acronym", "dates": "dates", "place": "place", "url": "url",
              "session": "session", "part": "session_part"}


def field_problems(fields: object) -> list[str]:
    """Every problem of ``fields:``, each by its path."""
    if not isinstance(fields, dict):
        return [f"fields: is {_kind(fields)}, not a mapping of field names"]
    problems: list[str] = []
    for name, value in cast(dict[object, object], fields).items():
        where = f"fields.{plain(str(name))}"
        if name in BUILT:
            owner = BUILT[str(name)]
            problems.append(f"{where}: is a field MIME builds" if owner == "MIME" else
                            f"{where}: is written as sharing.yaml's `{owner}`" if owner != "the experiment's licence"
                            else f"{where}: is {owner}, declared where the experiment is built")
        elif name not in SHAPES:
            problems.append(f"{where}: is not a field push sends (it sends {', '.join(SHAPES)})")
        else:
            SHAPES[str(name)](value, where, problems)
    return problems


@dataclass(frozen=True)
class Funding:
    """A funder (by name, with its ROR identifier where given) and, where given, its award's number, title and
    address."""

    funder: str
    ror: str | None = None
    award: str | None = None
    title: str | None = None
    url: str | None = None

    def words(self) -> str:
        parts = [plain(self.funder) + (f" (ROR {self.ror})" if self.ror else "")]
        if self.award:
            parts.append(f"award {plain(self.award)}" + (f", {plain(self.title)}" if self.title else ""))
        return ": ".join(parts)

    def entry(self) -> dict[str, object]:
        """The records API's ``funding`` entry."""
        held: dict[str, object] = {"funder": {"id": self.ror} if self.ror else {"name": self.funder}}
        if self.award:
            award: dict[str, object] = {"title": {"en": self.title or self.award}, "number": self.award}
            if self.url:
                award["identifiers"] = [{"scheme": "url", "identifier": self.url}]
            held["award"] = award
        return held


_FUNDING_KEYS = ("funder", "ror", "award", "title", "url")


def funding_problems(value: object) -> tuple[list[Funding], list[str]]:
    """``funding:`` read: each funder, and every problem by its path."""
    found: list[Funding] = []
    problems: list[str] = []
    if value is None:
        return found, problems
    if not isinstance(value, list):
        return found, [f"funding: is {_kind(value)}, not a list"]
    for index, each in enumerate(cast(list[object], value)):
        where = f"funding[{index}]"
        if not isinstance(each, dict):
            problems.append(f"{where}: is {_kind(each)}, not a mapping of {', '.join(_FUNDING_KEYS)}")
            continue
        held = cast(dict[object, object], each)
        problems += [f"{where}.{plain(str(key))}: is not a key it holds (it holds {', '.join(_FUNDING_KEYS)})"
                     for key in held if key not in _FUNDING_KEYS]
        texts: dict[str, str | None] = {}
        for key in _FUNDING_KEYS:
            got = held.get(key)
            if got is not None and (not isinstance(got, (str, int)) or not str(got).strip()):
                problems.append(f"{where}.{key}: is {_kind(got)}, not text")
                got = None
            texts[key] = str(got).strip() if got is not None else None
        if texts["funder"] is None:
            problems.append(f"{where}.funder: is missing (the funder's name)")
            continue
        ror = texts["ror"]
        matched = _ROR.fullmatch(ror) if ror is not None else None
        if ror is not None and matched is None:
            problems.append(f"{where}.ror: {plain(ror)} is not a ROR identifier (0 and eight letters and digits)")
        if texts["award"] is None and (texts["title"] or texts["url"]):
            problems.append(f"{where}: names an award's title or address without its number, award: <number>")
        if texts["url"] is not None and not texts["url"].startswith("https://"):
            problems.append(f"{where}.url: is not an https:// address")
        found.append(Funding(texts["funder"], matched.group(1) if matched else None, texts["award"], texts["title"],
                             texts["url"]))
    return found, problems


def access_problems(value: object) -> tuple[dict[str, object], list[str]]:
    """``access:`` read: who may read the record and its files where it is not everyone, an embargo's end and reason,
    and every problem by its path."""
    if value is None:
        return {}, []
    if not isinstance(value, dict):
        return {}, [f"access: is {_kind(value)}, not a mapping of record, files, embargo"]
    held = cast(dict[object, object], value)
    problems = [f"access.{plain(str(key))}: is not a key it holds (it holds record, files, embargo)"
                for key in held if key not in ("record", "files", "embargo")]
    found: dict[str, object] = {}
    for key in ("record", "files"):
        if key in held:
            _one_of(ACCESS_LEVELS)(held[key], f"access.{key}", problems)
            found[key] = held[key]
    embargo = held.get("embargo")
    if embargo is not None:
        until = cast(dict[str, object], embargo).get("until") if isinstance(embargo, dict) else None
        terms: dict[str, object] = dict(cast(dict[str, object], embargo)) if isinstance(embargo, dict) else {}
        if isinstance(until, datetime.date):  # YAML reads an unquoted day as a date
            terms["until"] = until.isoformat()
        _mapping({"until": (_matching(_DAY, "a day, 2030-01-01"), True), "reason": (_text, False)})(
            terms if isinstance(embargo, dict) else embargo, "access.embargo", problems)
        if "restricted" not in (found.get("record"), found.get("files")):
            problems.append("access.embargo: holds back what is restricted, and neither the record nor its files are")
        found["embargo"] = terms
    return found, problems


def access_entry(access: Mapping[str, object]) -> dict[str, object]:
    """The records API's ``access`` from ``access:`` as read: public where it says nothing."""
    found: dict[str, object] = {"record": access.get("record", "public"), "files": access.get("files", "public")}
    embargo = access.get("embargo")
    if isinstance(embargo, dict):
        found["embargo"] = {"active": True, **cast(dict[str, object], embargo)}
    return found


def access_words(access: Mapping[str, object]) -> str:
    """An ``access`` of the records API in words."""
    embargo = access.get("embargo")
    terms: dict[str, object] = cast(dict[str, object], embargo) if isinstance(embargo, dict) else {}
    until = f", embargoed until {plain(str(terms.get('until')))}" if terms.get("active", bool(terms)) else ""
    return f"the record {plain(str(access.get('record', '')))}, its files {plain(str(access.get('files', '')))}{until}"


def conference_problems(value: object) -> tuple[dict[str, str] | None, list[str]]:
    """``conference:`` read: Zenodo's meeting field, and every problem by its path."""
    if value is None:
        return None, []
    if not isinstance(value, dict):
        return None, [f"conference: is {_kind(value)}, not a mapping of {', '.join(CONFERENCE)}"]
    held = cast(dict[object, object], value)
    problems = [f"conference.{plain(str(key))}: is not a key it holds (it holds {', '.join(CONFERENCE)})"
                for key in held if key not in CONFERENCE]
    meeting: dict[str, str] = {}
    for key, name in CONFERENCE.items():
        got = held.get(key)
        if got is None:
            continue
        if not isinstance(got, (str, int)) or not str(got).strip():
            problems.append(f"conference.{key}: is {_kind(got)}, not text")
            continue
        meeting[name] = str(got).strip()
    if "title" not in meeting:
        problems.append("conference.title: is missing (the conference's name)")
    if "url" in meeting and not meeting["url"].startswith("https://"):
        problems.append("conference.url: is not an https:// address")
    return meeting, problems


__all__ = ["BUILT", "CONFERENCE", "DATE_TYPES", "SHAPES", "Funding", "conference_problems", "field_problems",
           "funding_problems"]
