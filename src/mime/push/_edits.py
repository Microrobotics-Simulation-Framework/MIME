"""Edits made on the site, never lost (design #199 (d)): each field of a draft's metadata edited on the service's site
since push last wrote it is shown against what push would write, and the author chooses, field by field, to keep it
(written back into ``sharing.yaml``, so the folder stays the source of truth) or to write push's over it. Files stay
MIME's: a file added or changed on the site is refused (``push.draft-changed``), never kept.

A field can be kept where ``sharing.yaml`` holds it: the title, the creators, the contributors (those push adds as
rights holders from the files' terms stay push's), the keywords, the related works (push's own two stay push's), the
funding, the conference, any field of ``fields:``, and the description's paragraph above the card. The licences are the
files' (declared where the experiment is built), and the rest is what MIME builds: an edit of one of those is written
over, or the push stops.

**What the service adds is not an edit.** The service fills in what it is sent (a licence's title, a person's display
name, a vocabulary entry's title), so two values are compared as the author writes them (:func:`reduced`): a
vocabulary entry by its ``id`` alone, a person by their family and given names (not the display name the service
makes of them).

**The write-back is checked.** The key's lines in ``sharing.yaml`` are replaced (its other lines and comments kept), the
file is read again, and the payload made from it must hold the site's value; if it does not, the file is put back as it
was and the push stops, saying what to copy in by hand.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from typing import cast

import yaml

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._fields import CONFERENCE
from mime.push._plan import KEYWORD, Plan, with_sharing
from mime.push._sharing import CONTRIBUTOR_TYPES, RELATION_TYPES, read_sharing, sharing_path
from mime.push._state import CUSTOM

type Json = dict[str, object]

#: The ``sharing.yaml`` key that holds each field a person may keep.
SHARING_KEYS: Mapping[str, str] = {
    "title": "title", "creators": "creators", "contributors": "contributors", "subjects": "keywords",
    "related_identifiers": "related", "funding": "funding", f"{CUSTOM}meeting:meeting": "conference",
    "description": "description", "dates": "fields", "languages": "fields", "references": "fields",
    "additional_titles": "fields", "additional_descriptions": "fields", f"{CUSTOM}journal:journal": "fields",
    f"{CUSTOM}imprint:imprint": "fields", f"{CUSTOM}thesis:university": "fields", f"{CUSTOM}thesis:type": "fields"}
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
    """The field ``name`` of a draft or a payload (``custom_fields.<name>`` among its custom fields)."""
    if name.startswith(CUSTOM):
        return _map(holder.get("custom_fields")).get(name.removeprefix(CUSTOM))
    return _map(holder.get("metadata")).get(name)


def differs(name: str, site: Mapping[str, object], ours: Mapping[str, object]) -> bool:
    """Whether push would write over the site's value of ``name``: the two differ as the author writes them."""
    return reduced(field_value(site, name)) != reduced(field_value(ours, name))


def shown(value: object) -> str:
    if value in (None, "", [], {}):
        return "nothing"
    import json

    text = plain(value if isinstance(value, str) else json.dumps(reduced(value), sort_keys=True, ensure_ascii=False))
    return text if len(text) <= SHOWN else text[:SHOWN] + "..."


def as_yaml(value: object) -> str:
    """A value as sharing.yaml holds it, on one line, quoted where it is long."""
    text = plain(yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=10_000,
                                sort_keys=False).strip().removesuffix("\n...").strip())
    return text if len(text) <= SHOWN else text[:SHOWN] + "..."


def written_value(plan: Plan, key: str) -> object:
    """The value of ``key`` in ``sharing.yaml`` as it is written now, or ``None``."""
    try:
        held: object = yaml.safe_load(sharing_path(plan.folder).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return cast(Json, held).get(key) if isinstance(held, dict) else None


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


def kept(name: str, site: Mapping[str, object], plan: Plan) -> dict[str, object] | None:
    """The ``sharing.yaml`` keys, with their new values, that keep the site's value of the field ``name``; ``None``
    where ``sharing.yaml`` cannot hold it (a field MIME builds, the licences, a value of a shape push does not
    write)."""
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
        return _paragraph(value, plan)
    short = name.removeprefix(CUSTOM)
    if value in (None, "", [], {}):
        fields.pop(short, None)
    else:
        fields[short] = _as_written(value)
    return {"fields": fields}


def _paragraph(value: object, plan: Plan) -> dict[str, object] | None:
    """The description's paragraph above the card, where the site's description is a paragraph and then the card as
    MIME writes it (or the card alone)."""
    if not isinstance(value, str) or not value.endswith(plan.card):
        return None
    head = value.removesuffix(plan.card)
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


def written_text(text: str, updates: Mapping[str, object]) -> str:
    """``sharing.yaml``'s ``text`` with each key of ``updates`` written as its value (its lines replaced, or added at
    the end), every other line as it was."""
    lines = text.splitlines()
    for key, value in updates.items():
        dumped = yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=True, width=10_000,
                                default_flow_style=False).rstrip("\n").splitlines()
        found = _block(lines, key)
        if found is None:
            lines += dumped
        else:
            lines[found[0]:found[1]] = dumped
    return "\n".join(lines) + "\n"


def write_back(plan: Plan, updates: Mapping[str, object], names: Sequence[str], site: Mapping[str, object]) -> Plan:
    """Write ``updates`` into ``sharing.yaml``, read it again, and return the plan made from it, checked to hold the
    site's value of each field ``names`` lists; otherwise ``sharing.yaml`` is put back as it was and the push stops
    (``push.sharing``), with the lines to copy in by hand."""
    path = sharing_path(plan.folder)
    before = path.read_text(encoding="utf-8")
    after = written_text(before, updates)
    path.write_text(after, encoding="utf-8", newline="\n")
    problem: str | None = None
    made = plan
    try:
        made = with_sharing(plan, read_sharing(plan.folder))
        wrong = [name for name in names if differs(name, site, made.payload)]
        if wrong:
            problem = f"read again, it does not give the site's {', '.join(wrong)}"
    except BuildRefused as refused:
        problem = "; ".join(each.message for each in refused.refusals)
    if problem is None:
        return made
    path.write_text(before, encoding="utf-8", newline="\n")
    lines = yaml.safe_dump(dict(updates), sort_keys=False, allow_unicode=True, width=10_000).strip()
    raise BuildRefused([Refusal(
        "push.sharing", RefusalClass.USER,
        f"{path}: push wrote the site's {', '.join(names)} into it to keep {'them' if len(names) != 1 else 'it'}, and "
        f"{problem}, so it put the file back as it was and sent nothing",
        f"copy the site's value in by hand, as these lines, in {path}: {plain(lines)}; then `mime push <folder> --to "
        f"{plan.service.name}` again")])


__all__ = ["SHARING_KEYS", "as_yaml", "differs", "field_value", "kept", "reduced", "shown", "write_back",
           "written_text", "written_value"]
