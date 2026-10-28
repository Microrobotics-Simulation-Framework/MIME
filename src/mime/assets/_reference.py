"""An asset reference: what a build or a saved experiment holds in place of a large file."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlsplit

from typing_extensions import ReadOnly, TypedDict

from mime.build import BuildRefused, Refusal, RefusalClass

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: Plain ``http`` is accepted for these literal hosts only (a mirror on the same machine, the tests' server): the sha256
#: guards the bytes whatever the route, but a remote location is ``https`` or a DOI.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
_UNSAFE = re.compile(r"[\\\s\x00-\x1f\x7f]")
_FORMS = "a location is an https:// URL or a doi: URL"


class AssetRecord(TypedDict, closed=True):
    """A reference as a lock holds it. Locations are in the order they are tried."""

    name: ReadOnly[str]
    sha256: ReadOnly[str]
    size: ReadOnly[int]
    licence: ReadOnly[str]
    locations: ReadOnly[list[str]]


def _invalid(name: str, problem: str) -> BuildRefused:
    return BuildRefused([Refusal(
        "asset.reference", RefusalClass.USER, f"asset reference {name!r}: {problem}",
        "write the reference as name, sha256 (64 lowercase hex digits), size in bytes, licence, and locations "
        "that are https:// URLs or doi: URLs")])


def location_problem(location: str) -> str | None:
    """Why ``location`` is not a form pooch is handed, or ``None``: an ``https://`` URL with a host, or
    ``doi:<doi>/<file>``, or ``http://`` to a loopback host (``127.0.0.1``, ``localhost`` or ``[::1]``).

    The host is read once, by ``urllib3`` (the parser ``requests`` downloads with), and the standard library's parse must
    agree with it; a URL with a backslash, whitespace, a control character, userinfo or a percent-escape in its
    authority is refused whatever its host, since parsers disagree on those."""
    if location.startswith("doi:"):
        doi, _, file = location.removeprefix("doi:").rpartition("/")
        if doi.startswith("10.") and "/" in doi and file:
            return None
        return "a doi: location is doi:10.<registrant>/<suffix>/<file>"
    if _UNSAFE.search(location):
        return "a location has no backslash, whitespace or control character"
    authority = location.partition("://")[2]
    for end in "/?#":
        authority = authority.partition(end)[0]
    if "@" in authority or "%" in authority:
        return "a location has no userinfo and no percent-escape in its host"
    from urllib3.exceptions import LocationParseError
    from urllib3.util import parse_url

    try:
        parsed = parse_url(location)
    except LocationParseError:
        return _FORMS
    host = (parsed.host or "").lower()
    standard = (urlsplit(location).hostname or "").lower()
    if not host or host.strip("[]") != standard:
        return _FORMS
    if parsed.scheme == "https":
        return None
    if parsed.scheme == "http" and host in _LOOPBACK_HOSTS:
        return None
    return _FORMS


@dataclass(frozen=True)
class AssetRef:
    """A large file named by the sha256 of its bytes, never by where it lives. ``locations`` are tried in order, so a
    new home is one more location, and no saved experiment that names the sha256 changes."""

    name: str
    sha256: str
    size: int
    licence: str
    locations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise _invalid(self.name, "a name is letters, digits, '.', '_' and '-', starting with a letter or digit")
        if not _SHA256.fullmatch(self.sha256):
            raise _invalid(self.name, f"sha256 {self.sha256!r} is not 64 lowercase hex digits")
        if self.size <= 0:
            raise _invalid(self.name, f"size {self.size} is not a positive number of bytes")
        if not self.licence:
            raise _invalid(self.name, "the licence is empty")
        if not self.locations:
            raise _invalid(self.name, "no location")
        for location in self.locations:
            problem = location_problem(location)
            if problem is not None:
                raise _invalid(self.name, f"location {location!r}: {problem}")

    def to_record(self) -> AssetRecord:
        return {"name": self.name, "sha256": self.sha256, "size": self.size, "licence": self.licence,
                "locations": list(self.locations)}

    @staticmethod
    def from_record(record: Mapping[str, object]) -> AssetRef:
        """The reference a lock's record describes; a key missing or unknown, or a value of another type, is refused."""
        name = str(record.get("name", "?"))
        if set(record) != set(AssetRecord.__annotations__):
            raise _invalid(name, f"the record's keys are {sorted(record)}, not {sorted(AssetRecord.__annotations__)}")
        size, listed = record["size"], record["locations"]
        locations = cast("list[object]", listed) if isinstance(listed, list) else None
        texts = (record["name"], record["sha256"], record["licence"])
        if (not all(isinstance(text, str) for text in texts) or not isinstance(size, int) or isinstance(size, bool)
                or locations is None or not all(isinstance(each, str) for each in locations)):
            raise _invalid(name, "the record's values are not str, str, int, str and a list of str")
        return AssetRef(str(texts[0]), str(texts[1]), size, str(texts[2]), tuple(str(each) for each in locations))
