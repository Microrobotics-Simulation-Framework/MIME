"""The saved form before phase 7l-a (``mime.document/1``, ``mime.lock/1``), read under the current schema and held to
its own lock.

A previous document gains what version 2 adds, with the values a build before profiles had: no setting's source given
apart from its value, and the profile **exploratory**. That is the profile it was built under in the sense that
matters for loading it: exploratory accepts every build version 1 accepted (version 1 had no pending reason, and
exploratory's other rules only waive what research would ask for, never refuse more), while research would refuse a
version 1 build that relied on a default MIME now classes as numerical (an arm tracker's tuned gains). Its outputs then
carry the stamp of whatever is pending, which for a version 1 document is only those numerical defaults, and ``mime
pending`` lists what sealing it needs.

A previous lock's input hash was taken over the previous document and resolved record, so a load compares the rebuilt
experiment in the previous form: the keys version 2 added taken off both, which a document read from version 1 holds
at those values. Its schema check is the current one, run on the upgraded form, so the two cannot drift apart."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from mime.saved._schema import DOCUMENT_SCHEMA, LOCK_SCHEMA, PREVIOUS_DOCUMENT_SCHEMA, PREVIOUS_LOCK_SCHEMA

#: The profile a previous document is read under, and the keys version 2 added, with the values a previous form holds.
PREVIOUS_PROFILE = "exploratory"
_DOCUMENT_ADDED: Mapping[str, object] = {"profile": PREVIOUS_PROFILE, "sources": {}}
_RESOLVED_ADDED = "profile"
_PREVIOUS_RESOLVED_PROFILE: Mapping[str, object] = {"profile": PREVIOUS_PROFILE, "pending": [], "waivers": [],
                                                    "defaults": {}, "sources": {}, "listed": [], "changes": {}}


def upgraded_document(data: object) -> tuple[object, bool]:
    """``data`` as the current schema reads it, and whether it was a previous document. A previous one that holds a
    key version 2 added is left as it is, for the schema check to refuse."""
    if not isinstance(data, dict) or cast(dict[str, object], data).get("schema") != PREVIOUS_DOCUMENT_SCHEMA:
        return cast(object, data), False
    held = dict(cast(dict[str, object], data))
    if any(key in held for key in _DOCUMENT_ADDED):
        return held, True
    return {**held, "schema": DOCUMENT_SCHEMA, **_DOCUMENT_ADDED}, True


def upgraded_lock(data: object) -> tuple[object, bool]:
    """``data`` as the current schema reads it, and whether it was a previous lock."""
    if not isinstance(data, dict) or cast(dict[str, object], data).get("schema") != PREVIOUS_LOCK_SCHEMA:
        return cast(object, data), False
    held = dict(cast(dict[str, object], data))
    resolved = held.get("resolved")
    if isinstance(resolved, dict) and _RESOLVED_ADDED not in resolved:
        held["resolved"] = {**cast(dict[str, object], resolved), _RESOLVED_ADDED: dict(_PREVIOUS_RESOLVED_PROFILE)}
    held["schema"] = LOCK_SCHEMA
    return held, True


def previous_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """A document in the previous form, as its lock hashed it: the keys version 2 added taken off."""
    held = {key: value for key, value in document.items() if key not in _DOCUMENT_ADDED}
    held["schema"] = PREVIOUS_DOCUMENT_SCHEMA
    return held


def previous_resolved(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """A resolved record in the previous form: its profile taken off."""
    return {key: value for key, value in resolved.items() if key != _RESOLVED_ADDED}


__all__ = ["PREVIOUS_PROFILE", "previous_document", "previous_resolved", "upgraded_document", "upgraded_lock"]
