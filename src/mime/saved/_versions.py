"""The saved form before version 4, read under the current schema and held to its own lock.

**Version 1** (``mime.document/1``, ``mime.lock/1``) is the form before phase 7l-a. A document of it gains what version
2 added, with the values a build before profiles had: no setting's source given apart from its value, and the profile
**exploratory**. That is the profile it was built under in the sense that matters for loading it: exploratory accepts
every build version 1 accepted (version 1 had no pending reason, and exploratory's other rules only waive what research
would ask for, never refuse more), while research would refuse a version 1 build that relied on a default MIME now
classes as numerical (an arm tracker's tuned gains). Its outputs then carry the stamp of whatever is pending, which for
a version 1 document is only those numerical defaults, and ``mime pending`` lists what sealing it needs.

**Version 2** (``mime.document/2``, ``mime.lock/2``) is the form before the drag's rung ``bem`` was renamed ``mrs``
(the method is the method of regularised Stokeslets, not a boundary-element method). Version 3 changed three things a
hash saw: the rung's name in the document and in the resolved record, the drag node's type in the derived graph
(``BEMDragNode``, now ``MRSDragNode``), and the possessive in the sources of the values declared in the experiment,
which a message built from a quoted name had written ``body 'umr''s`` (now ``body umr's``). A version 2 document that
names ``bem`` is read as naming ``mrs``, and the load says so.

**Version 3** (``mime.document/3``, ``mime.lock/3``) is the form before licences and terms (designs #198, #199).
Version 4 adds four keys to the document (``licence``, ``licences``, ``terms``, ``permissions``) and one to the lock
(``referenced_files``), each written only where something is declared; nothing else changed. A document of an earlier
version declares no licence, no terms and no permission, and is read so: every file of it is its author's own, carried
by ``pack``, with no licence declared. One of an earlier version that holds a key version 4 added is refused, never
read as declaring it.

A previous lock's input hash was taken over the previous document and resolved record, and its derived hash over the
previous derived graph, so a load compares the rebuilt experiment in the previous form: the rebuilt document, resolved
record and derived graph are written back into the form the lock was written in (the keys version 4 added taken off;
the keys version 2 added taken off for version 1; the rung, the node type and the possessive written as version 2
wrote them, for versions 1 and 2) and hashed there. Its schema check is the current one, run on the upgraded form, so
the two cannot drift apart. A version 2 source whose name is not an identifier (a material named with spaces or
quotes) was quoted differently, and is not written back: such a document's input hash is refused, and saving it again
from the script writes the current version."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from mime.build import Refusal, RefusalClass
from mime.saved._schema import DOCUMENT_SCHEMA, LOCK_SCHEMA

#: The saved form this MIME writes.
CURRENT = 4
DOCUMENT_SCHEMAS: Mapping[int, str] = {1: "mime.document/1", 2: "mime.document/2", 3: "mime.document/3",
                                       CURRENT: DOCUMENT_SCHEMA}
LOCK_SCHEMAS: Mapping[int, str] = {1: "mime.lock/1", 2: "mime.lock/2", 3: "mime.lock/3", CURRENT: LOCK_SCHEMA}
#: The version that renamed the drag's rung: a form before it names the rung ``bem``.
RUNG_RENAMED = 3
#: The keys version 4 added, to the document and to the lock: an earlier form holds none of them, and declares no
#: licence, no terms and no permission.
TERMS_KEYS: tuple[str, ...] = ("licence", "licences", "terms", "permissions")
LOCK_TERMS_KEY = "referenced_files"

#: The profile a version 1 document is read under, and the keys version 2 added, with the values a version 1 form holds.
PREVIOUS_PROFILE = "exploratory"
_DOCUMENT_ADDED: Mapping[str, object] = {"profile": PREVIOUS_PROFILE, "sources": {}}
_RESOLVED_ADDED = "profile"
_PREVIOUS_RESOLVED_PROFILE: Mapping[str, object] = {"profile": PREVIOUS_PROFILE, "pending": [], "waivers": [],
                                                    "defaults": {}, "sources": {}, "listed": [], "changes": {}}

#: The rung of the drag as version 2 and earlier named it, and as version 3 on names it; the drag node's type likewise.
OLD_RUNG, NEW_RUNG = "bem", "mrs"
OLD_NODE, NEW_NODE = "BEMDragNode", "MRSDragNode"
_DRAG = "drag"
#: ``body umr's`` as version 3 writes a possessive in a value's source, and ``body 'umr''s`` as version 2 wrote it. Only
#: a name that is an identifier is written back: ``repr`` quotes any other name differently.
_PLAIN_POSSESSIVE = re.compile(r"\b(body|material) ([A-Za-z_][A-Za-z0-9_]*)'s\b")


def version_of(schema: object, schemas: Mapping[int, str]) -> int | None:
    """The version whose schema string is ``schema``, or ``None``."""
    return next((version for version, text in schemas.items() if text == schema), None)


#: A form's id of one family: ``mime.document/4`` is ``mime.document`` at version 4 (no sign, no leading zero).
_FORM_ID = re.compile(r"mime\.(?P<family>[a-z_]+)/(?P<version>[1-9][0-9]*)")
#: The longest version number compared as a number: a longer one is newer than any form MIME has written.
_COMPARABLE_DIGITS = 9


def newer_form_refusal(found: object, family: str, reads: Sequence[str], where: str) -> Refusal | None:
    """The refusal ``load.newer-form`` when ``found`` (the schema or format id a file states) is of ``family``
    (``document``, for ``mime.document/N``) at a version above the newest of ``reads`` (the ids this MIME reads), or
    ``None``: an id of another family, a version this MIME reads and any other text are for the file's own check to
    judge. ``where`` names the file."""
    from mime import __version__

    match = _FORM_ID.fullmatch(found) if isinstance(found, str) else None
    if match is None or match["family"] != family:
        return None
    versions = [int(text.rpartition("/")[2]) for text in reads]
    digits = match["version"]
    if len(digits) <= _COMPARABLE_DIGITS and int(digits) <= max(versions):
        return None
    return Refusal(
        "load.newer-form", RefusalClass.USER,
        f"{where} was written by a newer MIME: its form is {found}, and this MIME ({__version__}) reads "
        f"{', '.join(reads)}",
        "install the MIME that wrote it, a later one (`pip install -U mime-engine`), then load it again")


def _renamed_rungs(document: dict[str, object]) -> tuple[dict[str, object], tuple[str, ...]]:
    """``document`` with each drag that names the old rung naming the new one, and the ids of those drags."""
    found: list[str] = []
    interactions = document.get("interactions")
    if not isinstance(interactions, list):
        return document, ()
    changed: list[object] = []
    for item in cast("list[object]", interactions):
        if isinstance(item, dict):
            entry = cast("dict[str, object]", item)
            settings = entry.get("settings")
            if entry.get("name") == _DRAG and isinstance(settings, dict) and \
                    cast("dict[str, object]", settings).get("rung") == OLD_RUNG:
                item = {**entry, "settings": {**cast("dict[str, object]", settings), "rung": NEW_RUNG}}
                found.append(str(entry.get("id")))
        changed.append(cast(object, item))
    return ({**document, "interactions": changed} if found else document), tuple(found)


def terms_keys_held(data: object) -> list[str]:
    """The keys version 4 added that ``data`` (a document, or a lock) holds."""
    if not isinstance(data, dict):
        return []
    held = cast("dict[str, object]", data)
    return [key for key in (*TERMS_KEYS, LOCK_TERMS_KEY) if key in held]


def upgraded_document(data: object) -> tuple[object, int, tuple[str, ...]]:
    """``data`` as the current schema reads it, the version it was written in (the current one for a document that is
    not of an earlier version, which the schema check judges), and the ids of the drags that named the old rung. A
    document of an earlier version that holds a key a later one added (version 2's in version 1, version 4's in any
    earlier) is left as it is, for the schema check to refuse."""
    if not isinstance(data, dict):
        return data, CURRENT, ()
    held = dict(cast("dict[str, object]", data))
    version = version_of(held.get("schema"), DOCUMENT_SCHEMAS)
    if version is None or version == CURRENT:
        return held, CURRENT, ()
    if terms_keys_held(held):
        return held, version, ()
    if version == 1:
        if any(key in held for key in _DOCUMENT_ADDED):
            return held, version, ()
        held = {**held, **_DOCUMENT_ADDED}
    renamed: tuple[str, ...] = ()
    if version < RUNG_RENAMED:
        held, renamed = _renamed_rungs(held)
    return {**held, "schema": DOCUMENT_SCHEMA}, version, renamed


def upgraded_lock(data: object) -> tuple[object, int]:
    """``data`` as the current schema reads it, and the version it was written in. The lock's contents stay in the form
    it was written in: a load compares the rebuilt experiment written back into that form. A lock of an earlier
    version that holds the key version 4 added is left as it is, for the schema check to refuse."""
    if not isinstance(data, dict):
        return data, CURRENT
    held = dict(cast("dict[str, object]", data))
    version = version_of(held.get("schema"), LOCK_SCHEMAS)
    if version is None or version == CURRENT:
        return held, CURRENT
    if LOCK_TERMS_KEY in held:
        return held, version
    resolved = held.get("resolved")
    if version == 1 and isinstance(resolved, dict) and _RESOLVED_ADDED not in resolved:
        held["resolved"] = {**cast("dict[str, object]", resolved), _RESOLVED_ADDED: dict(_PREVIOUS_RESOLVED_PROFILE)}
    held["schema"] = LOCK_SCHEMA
    return held, version


def document_in_form(document: Mapping[str, Any], version: int) -> dict[str, Any]:
    """A document written back into the form of ``version``, as its lock hashed it."""
    held = dict(document)
    if version == CURRENT:
        return held
    held = {key: value for key, value in held.items() if key not in TERMS_KEYS}
    interactions = held.get("interactions")
    if version < RUNG_RENAMED and isinstance(interactions, list):
        held["interactions"] = [_with_rung(item, NEW_RUNG, OLD_RUNG) for item in cast("list[Any]", interactions)]
    if version == 1:
        held = {key: value for key, value in held.items() if key not in _DOCUMENT_ADDED}
    held["schema"] = DOCUMENT_SCHEMAS[version]
    return held


def _with_rung(entry: Any, was: str, now: str) -> Any:
    """``entry`` (a bound entry of the document, or a resolved entry) with the drag's rung ``was`` written ``now``."""
    if not isinstance(entry, dict):
        return entry
    held = cast("dict[str, Any]", entry)
    settings = held.get("settings")
    if held.get("name") == _DRAG and isinstance(settings, dict) and cast("dict[str, Any]", settings).get("rung") == was:
        return {**held, "settings": {**cast("dict[str, Any]", settings), "rung": now}}
    return held


def _quoted_possessive(text: object) -> object:
    return _PLAIN_POSSESSIVE.sub(r"\1 '\2''s", text) if isinstance(text, str) else text


def _value_in_form(record: Any) -> Any:
    """A value's record of the resolved record, its source written as version 2 wrote it."""
    if not isinstance(record, dict):
        return record
    held = cast("dict[str, Any]", record)
    return {**held, "source": _quoted_possessive(held.get("source"))}


def resolved_in_form(resolved: Mapping[str, Any], version: int) -> dict[str, Any]:
    """A resolved record written back into the form of ``version``: version 3's is the current one."""
    held = dict(resolved)
    if version >= RUNG_RENAMED:
        return held
    entries = held.get("entries")
    if isinstance(entries, dict):
        held["entries"] = {name: _with_rung(entry, NEW_RUNG, OLD_RUNG)
                           for name, entry in cast("dict[str, Any]", entries).items()}
    values = held.get("values")
    if isinstance(values, dict):
        held["values"] = {path: _value_in_form(record) for path, record in cast("dict[str, Any]", values).items()}
    if version == 1:
        held = {key: value for key, value in held.items() if key != _RESOLVED_ADDED}
    return held


def physics_in_form(physics: Mapping[str, Any], version: int) -> dict[str, Any]:
    """The derived graph's physics written back into the form of ``version``: the drag node's type as it was named."""
    held = dict(physics)
    nodes = held.get("nodes")
    if version < RUNG_RENAMED and isinstance(nodes, list):
        held["nodes"] = [{**cast("dict[str, Any]", node), "type": OLD_NODE}
                         if isinstance(node, dict) and cast("dict[str, Any]", node).get("type") == NEW_NODE else node
                         for node in cast("list[Any]", nodes)]
    return held


__all__ = ["CURRENT", "DOCUMENT_SCHEMAS", "LOCK_SCHEMAS", "LOCK_TERMS_KEY", "NEW_RUNG", "OLD_RUNG", "PREVIOUS_PROFILE",
           "RUNG_RENAMED", "TERMS_KEYS", "document_in_form", "newer_form_refusal", "physics_in_form", "resolved_in_form",
           "terms_keys_held", "upgraded_document", "upgraded_lock", "version_of"]
