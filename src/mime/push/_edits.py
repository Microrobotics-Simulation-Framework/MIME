"""Edits made on the site, never lost (design #199 (d)): each field of a draft's metadata edited on the service's site
since push last wrote it is shown against what push would write, and the author chooses, field by field, to keep it
(written back into ``sharing.yaml``, so the folder stays the source of truth) or to write push's over it. Files stay
MIME's: a file added or changed on the site is refused (``push.draft-changed``), never kept.

A field can be kept where ``sharing.yaml`` holds it: the title, the creators, the contributors (those push adds as
rights holders from the files' terms stay push's), the keywords, the related works (push's own two stay push's), the
funding, the conference, any field of ``fields:``, the record's access (who may read it and its files, an embargo), and
the description's paragraph above the card. The licences are the files' (declared where the experiment is built), and
the rest is what MIME builds: an edit of one of those is written over, or the push stops.

**What the service adds is not an edit.** The service fills in what it is sent (a licence's title, a person's display
name, a vocabulary entry's title), so two values are compared as the author writes them (:func:`reduced`): a
vocabulary entry by its ``id`` alone, a person by their family and given names (not the display name the service
makes of them).

**The write-back is checked.** The key's lines in ``sharing.yaml`` are replaced, atomically (its other lines and
comments kept, line endings included; a comment inside the replaced lines goes with them), the file is read again, and
the payload made from it must hold the site's value and every other key as it was; if it does not, the file is put back
byte for byte and the push stops, saying what to copy in by hand.

**A question shows where two values differ**: a long value is cut around the first difference (:func:`apart`), so its
two lines never read the same.
"""

from __future__ import annotations

import html
import os
import re
import secrets
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._fields import ACCESS_LEVELS, CONFERENCE
from mime.push._plan import KEYWORD, Plan, with_sharing
from mime.push._sharing import CONTRIBUTOR_TYPES, RELATION_TYPES, read_sharing, sharing_path
from mime.push._state import ACCESS, CUSTOM, access_of

type Json = dict[str, object]

#: The ``sharing.yaml`` key that holds each field a person may keep.
SHARING_KEYS: Mapping[str, str] = {
    "title": "title", "creators": "creators", "contributors": "contributors", "subjects": "keywords",
    "related_identifiers": "related", "funding": "funding", f"{CUSTOM}meeting:meeting": "conference",
    "description": "description", "dates": "fields", "languages": "fields", "references": "fields",
    "additional_titles": "fields", "additional_descriptions": "fields", f"{CUSTOM}journal:journal": "fields",
    f"{CUSTOM}imprint:imprint": "fields", f"{CUSTOM}thesis:university": "fields", f"{CUSTOM}thesis:type": "fields",
    ACCESS: "access"}
#: The longest value a question quotes of a field.
SHOWN = 160


def _map(value: object) -> Json:
    return cast(Json, value) if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def reduced(value: object, *, top: bool = True) -> object:
    """``value`` as the author writes it: a vocabulary entry by its ``id`` alone, a person without the display name the
    service makes of their family and given names, and empty parts left out."""
    if isinstance(value, list):
        return [reduced(each, top=False) for each in cast(list[object], value)]
    if not isinstance(value, dict):
        return value
    held = cast(Json, value)
    if not top and "id" in held and "person_or_org" not in held and set(held) - {"id"} <= {
            "title", "props", "name", "scheme", "subject", "description", "icon", "type", "acronym"}:
        return {"id": held["id"]}
    found: Json = {}
    for key, inner in sorted(held.items()):
        if key == "name" and "family_name" in held:
            continue
        if inner in (None, "", [], {}):
            continue
        found[key] = reduced(inner, top=False)
    return found


def _as_written(value: object) -> object:
    return reduced(value, top=False) if _nested(value) else value


def _nested(value: object) -> bool:
    return isinstance(value, list | dict)


def field_value(holder: Mapping[str, object], name: str) -> object:
    """The field ``name`` of a draft or a payload (``custom_fields.<name>`` among its custom fields; ``access`` as an
    author sets it)."""
    if name == ACCESS:
        return access_of(holder.get(ACCESS))
    if name.startswith(CUSTOM):
        return _map(holder.get("custom_fields")).get(name.removeprefix(CUSTOM))
    return _map(holder.get("metadata")).get(name)


def differs(name: str, site: Mapping[str, object], ours: Mapping[str, object]) -> bool:
    """Whether push would write over the site's value of ``name``: the two differ as the author writes them."""
    return reduced(field_value(site, name)) != reduced(field_value(ours, name))


def words_of(value: object) -> str:
    """A field's value on one line, whole."""
    if value in (None, "", [], {}):
        return "nothing"
    import json

    return plain(value if isinstance(value, str) else json.dumps(reduced(value), sort_keys=True, ensure_ascii=False))


def shown(value: object) -> str:
    text = words_of(value)
    return text if len(text) <= SHOWN else text[:SHOWN] + "..."


def as_yaml(value: object) -> str:
    """A value as sharing.yaml holds it, on one line, whole."""
    return plain(yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=10_000,
                                sort_keys=False).strip().removesuffix("\n...").strip())


def apart(site: str, ours: str) -> tuple[str, str]:
    """Two values as a question shows them: each whole where both fit, and otherwise each cut around the first place
    they differ, so the two lines never read the same."""
    if len(site) <= SHOWN and len(ours) <= SHOWN:
        return site, ours
    first = next((index for index, (one, other) in enumerate(zip(site, ours, strict=False)) if one != other),
                 min(len(site), len(ours)))
    start = max(0, first - SHOWN // 4)

    def cut(text: str) -> str:
        return ("..." if start else "") + text[start:start + SHOWN] + ("..." if len(text) > start + SHOWN else "")

    return cut(site), cut(ours)


def written_value(plan: Plan, key: str) -> object:
    """The value of ``key`` in ``sharing.yaml`` as it is written now, or ``None``; the access a record has where
    ``sharing.yaml`` says nothing of it, public."""
    try:
        held: object = yaml.safe_load(sharing_path(plan.folder).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        held = None
    found = cast(Json, held).get(key) if isinstance(held, dict) else None
    return {"record": "public", "files": "public"} if found is None and key == ACCESS else found


# --- the site's value, as sharing.yaml writes it ---------------------------------------------------------------------


def _type_of(role: object) -> str | None:
    word = str(_map(role).get("id", "")).lower()
    return next((each for each in CONTRIBUTOR_TYPES if each.lower() == word), None)


def _person_back(entry: object, contributor: bool) -> Json | None:
    held = _map(entry)
    person = _map(held.get("person_or_org"))
    found: Json = {}
    if person.get("type") == "organizational":
        found["organisation"] = person.get("name")
    else:
        found["family"] = person.get("family_name")
        if person.get("given_name"):
            found["given"] = person.get("given_name")
        orcid = next((_map(each).get("identifier") for each in _list(person.get("identifiers"))
                      if _map(each).get("scheme") == "orcid"), None)
        if orcid:
            found["orcid"] = orcid
    affiliation = _map(next(iter(_list(held.get("affiliations"))), None))
    if affiliation.get("name"):
        found["affiliation"] = affiliation.get("name")
        if affiliation.get("id"):
            found["ror"] = affiliation.get("id")
    if contributor:
        role = _type_of(held.get("role"))
        if role is None:
            return None
        found["type"] = role
    return found if all(isinstance(value, str) and value for value in found.values()) else None


def _related_back(entry: object) -> Json | None:
    held = _map(entry)
    word = str(_map(held.get("relation_type")).get("id", "")).lower()
    relation = next((each for each in RELATION_TYPES if each.lower() == word), None)
    identifier = held.get("identifier")
    return {"identifier": identifier, "relation": relation} if relation and isinstance(identifier, str) else None


def _funding_back(entry: object) -> Json | None:
    held = _map(entry)
    funder = _map(held.get("funder"))
    award = _map(held.get("award"))
    found: Json = {"funder": funder.get("name") or funder.get("id")}
    if funder.get("id"):
        found["ror"] = funder["id"]
    if award:
        found["award"] = award.get("number") or award.get("id")
        title = _map(award.get("title")).get("en")
        if title:
            found["title"] = title
        url = next((_map(each).get("identifier") for each in _list(award.get("identifiers"))
                    if _map(each).get("scheme") == "url"), None)
        if url:
            found["url"] = url
    return found if all(isinstance(value, str) and value for value in found.values()) else None


def _all(items: Sequence[Json | None]) -> list[Json] | None:
    return None if any(each is None for each in items) else [cast(Json, each) for each in items]


def _access_back(value: object) -> dict[str, object] | None:
    """The site's access as ``sharing.yaml`` writes it (public is left out), or ``None`` for a shape push does not
    write."""
    held = _map(value)
    found: Json = {key: held[key] for key in ("record", "files") if held.get(key, "public") != "public"}
    if any(each not in ACCESS_LEVELS for each in found.values()):
        return None
    if "embargo" in held:
        terms = _map(held["embargo"])
        if not isinstance(terms.get("until"), str):
            return None
        found["embargo"] = terms
    return {ACCESS: found or None}


def kept(name: str, site: Mapping[str, object], plan: Plan, card: str | None = None) -> dict[str, object] | None:
    """The ``sharing.yaml`` keys, with their new values, that keep the site's value of the field ``name``; ``None``
    where ``sharing.yaml`` cannot hold it (a field MIME builds, the licences, a value of a shape push does not write).
    ``card`` is the card the site's description holds, where it is not the plan's: a new version's copy of the published
    record."""
    if name not in SHARING_KEYS:
        return None
    value = field_value(site, name)
    sharing = plan.sharing
    fields = dict(sharing.fields)
    if name == "title":
        return {"title": value} if isinstance(value, str) and value.strip() else None
    if name in ("creators", "contributors"):
        contributor = name == "contributors"
        entries = _list(value)
        if contributor:
            ours = _list(_map(plan.payload.get("metadata")).get("contributors"))[len(sharing.contributors):]
            added = [reduced(each) for each in ours]
            entries = [each for each in entries if reduced(each) not in added]
        people = _all([_person_back(each, contributor) for each in entries])
        return {name: people} if people is not None and (people or contributor) else None
    if name == "subjects":
        entries = [_map(each) for each in _list(value)]
        words = [str(each["subject"]) for each in entries if "subject" in each and "id" not in each
                 and each["subject"] != KEYWORD]
        own = [each for each in _list(fields.get("subjects")) if "subject" in _map(each)]
        words = [word for word in words if {"subject": word} not in [reduced(each) for each in own]]
        listed = [{"id": each["id"]} for each in entries if "id" in each]
        updates: dict[str, object] = {"keywords": words}
        if listed or "subjects" in fields:
            if listed or own:
                fields["subjects"] = [*own, *listed]
            else:
                fields.pop("subjects", None)
            updates["fields"] = fields
        return updates
    if name == "related_identifiers":
        entries = [each for each in _list(value) if "pypi.org/project/mime-engine/" not in
                   str(_map(each).get("identifier"))]
        related = _all([_related_back(each) for each in entries])
        return {"related": related} if related is not None else None
    if name == "funding":
        funding = _all([_funding_back(each) for each in _list(value)])
        return {"funding": funding} if funding is not None else None
    if name == f"{CUSTOM}meeting:meeting":
        meeting = _map(value)
        back = {key: meeting[field] for key, field in CONFERENCE.items() if isinstance(meeting.get(field), str)}
        return {"conference": back or None} if "title" in back or not meeting else None
    if name == "description":
        return _paragraph(value, plan.card if card is None else card)
    if name == ACCESS:
        return _access_back(value)
    short = name.removeprefix(CUSTOM)
    if value in (None, "", [], {}):
        fields.pop(short, None)
    else:
        fields[short] = _as_written(value)
    return {"fields": fields}


def _paragraph(value: object, card: str) -> dict[str, object] | None:
    """The description's paragraph above the card, where the site's description is a paragraph and then the card as
    MIME wrote it (or the card alone)."""
    if not card or not isinstance(value, str) or not value.endswith(card):
        return None
    head = value.removesuffix(card)
    if not head:
        return {"description": ""}
    found = re.fullmatch(r"<p>(.*)</p>\n", head, flags=re.DOTALL)
    return {"description": html.unescape(found.group(1))} if found is not None and "<" not in found.group(1) else None


# --- writing sharing.yaml ------------------------------------------------------------------------------------------


def _block(lines: list[str], key: str) -> tuple[int, int] | None:
    """The lines of the top-level ``key`` (its line and those that continue it), or ``None``."""
    start = next((index for index, line in enumerate(lines) if re.match(rf"{re.escape(key)}\s*:", line)), None)
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and (lines[end][:1] in (" ", "\t", "-") or not lines[end].strip()):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return start, end


def _dumped(key: str, value: object) -> str:
    """``key: value`` as YAML, in the person's own letters; where YAML would not read those back (it folds a U+0085 or a
    U+2028 it wrote as it stands), escaped."""
    plainly = yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=True, width=10_000, default_flow_style=False)
    if yaml.safe_load(plainly) == {key: value}:
        return plainly
    return yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=False, width=10_000, default_flow_style=False)


def written_text(text: str, updates: Mapping[str, object]) -> str:
    """``sharing.yaml``'s ``text`` with each key of ``updates`` written as its value (its lines replaced, or added at
    the end), every other line as it was, in the file's own line endings. Lines are YAML's, ended by a line feed: a
    U+2028 or a form feed inside a value is no line's end."""
    ending = "\r" if "\r\n" in text else ""
    lines = text.removesuffix("\n").split("\n") if text else []
    for key, value in updates.items():
        dumped = [line + ending for line in _dumped(key, value).rstrip("\n").split("\n")]
        found = _block(lines, key)
        if found is None:
            lines += dumped
        else:
            lines[found[0]:found[1]] = dumped
    return "\n".join(lines) + "\n"


def _replaced(path: Path, data: bytes) -> None:
    """``path`` holding ``data``, whole or not at all."""
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    try:
        temporary.write_bytes(data)
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _others(text: str, updates: Mapping[str, object]) -> Json:
    held: object = yaml.safe_load(text)
    return {key: value for key, value in cast(Json, held).items() if key not in updates} if isinstance(held, dict) else {}


def _holds(name: str, site: Mapping[str, object], made: Plan, card: str | None) -> bool:
    """Whether the plan made from ``sharing.yaml`` holds the site's ``name``. A description is held by its paragraph:
    the card below it is the card of the version being made, which a published version's is not."""
    if name != "description" or card is None:
        return not differs(name, site, made.payload)
    paragraph = _paragraph(field_value(site, name), card)
    return paragraph is not None and paragraph["description"] == made.sharing.description


def write_back(plan: Plan, updates: Mapping[str, object], names: Sequence[str], site: Mapping[str, object],
               card: str | None = None) -> Plan:
    """Write ``updates`` into ``sharing.yaml``, read it again, and return the plan made from it, checked to hold the
    site's value of each field ``names`` lists and every other key as it was; otherwise ``sharing.yaml`` is put back as
    it was, byte for byte, and the push stops (``push.sharing``), with the lines to copy in by hand. ``card`` as
    :func:`kept`'s."""
    path = sharing_path(plan.folder)
    lines = plain(yaml.safe_dump(dict(updates), sort_keys=False, allow_unicode=True, width=10_000).strip())
    by_hand = (f"copy the site's value in by hand, as these lines, in {path}: {lines}; then `mime push <folder> --to "
               f"{plan.service.name}` again")
    keeping = f"push wrote the site's {', '.join(names)} into it to keep {'them' if len(names) != 1 else 'it'}"
    before = path.read_bytes()
    text = before.decode("utf-8")
    after = written_text(text, updates)
    unwritten = None if os.access(path, os.W_OK) else "it is read-only"
    if unwritten is None:
        try:
            _replaced(path, after.encode("utf-8"))
        except OSError as error:
            unwritten = type(error).__name__
    if unwritten is not None:
        raise BuildRefused([Refusal("push.sharing", RefusalClass.USER,
                                    f"{path}: push could not write the site's {', '.join(names)} into it, to keep "
                                    f"{'them' if len(names) != 1 else 'it'} ({unwritten}), and sent nothing", by_hand)])
    problem: str | None = None
    made = plan
    try:
        made = with_sharing(plan, read_sharing(plan.folder))
        wrong = [name for name in names if not _holds(name, site, made, card)]
        if wrong:
            problem = f"read again, it does not give the site's {', '.join(wrong)}"
        elif _others(after, updates) != _others(text, updates):
            problem = "read again, another of its keys is not what it was"
    except BuildRefused as refused:
        problem = "; ".join(each.message for each in refused.refusals)
    if problem is None:
        return made
    _replaced(path, before)
    raise BuildRefused([Refusal("push.sharing", RefusalClass.USER,
                                f"{path}: {keeping}, and {problem}, so it put the file back as it was and sent nothing",
                                by_hand)])


__all__ = ["SHARING_KEYS", "apart", "as_yaml", "differs", "field_value", "kept", "reduced", "shown", "words_of",
           "write_back", "written_text", "written_value"]
