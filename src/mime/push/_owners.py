"""Who owns each field of a record's metadata: the author in ``sharing.yaml``, the author in the experiment (its terms,
saved with it), or MIME, which builds the rest. A field the service refuses, in any answer (a 2xx that lists
``errors`` or a 4xx), is the author's to correct where they own it and MIME's defect where MIME does; a refusal that
names no field is MIME's."""

from __future__ import annotations

#: Each field below ``metadata`` by its name (the start of the path the service names, ``metadata.creators.0`` for the
#: first creator); a field not listed, or outside ``metadata`` and ``custom_fields``, is MIME's.
FIELD_OWNERS: dict[str, str] = {
    "title": "sharing", "creators": "sharing", "contributors": "sharing", "subjects": "sharing",
    "related_identifiers": "sharing", "funding": "sharing", "dates": "sharing", "languages": "sharing",
    "references": "sharing", "additional_titles": "sharing", "additional_descriptions": "sharing",
    "rights": "experiment", "resource_type": "mime", "publication_date": "mime", "publisher": "mime",
    "description": "mime", "identifiers": "mime", "version": "mime"}
#: Where custom fields sit in a record (Zenodo's meeting, journal, imprint and thesis): each is the author's, from
#: sharing.yaml's ``conference`` or ``fields``.
CUSTOM_FIELDS = "custom_fields"
#: A record's access, the author's, from sharing.yaml's ``access``.
ACCESS_FIELD = "access"
#: MIME's own related identifiers, first in every record: the release it requires and that compiled it.
OWN_RELATED = 2
#: MIME's own keyword is the first subject of every record (``mime-experiment``); the author's follow it.
OWN_SUBJECTS = 1


def field_owner(field: str, contributors: int = 0) -> str:
    """``sharing``, ``experiment`` or ``mime``: who owns the metadata field at ``field`` (``metadata.creators.0``),
    for a record whose ``sharing.yaml`` names ``contributors`` contributors. The contributors after those are the
    rights holders MIME adds from the files' terms, and MIME's, as its own keyword and related identifiers are."""
    parts = field.split(".")
    if len(parts) >= 2 and parts[0] == CUSTOM_FIELDS or parts[0] == ACCESS_FIELD:
        return "sharing"
    if len(parts) < 2 or parts[0] != "metadata":
        return "mime"
    index = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
    if parts[1] == "related_identifiers" and index is not None and index < OWN_RELATED:
        return "mime"
    if parts[1] == "subjects" and index is not None and index < OWN_SUBJECTS:
        return "mime"
    if parts[1] == "contributors" and (index >= contributors if index is not None else contributors == 0):
        return "mime"
    return FIELD_OWNERS.get(parts[1], "mime")


__all__ = ["ACCESS_FIELD", "CUSTOM_FIELDS", "FIELD_OWNERS", "OWN_RELATED", "OWN_SUBJECTS", "field_owner"]
