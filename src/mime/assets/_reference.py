"""An asset record: what a build or a saved experiment holds in place of a file, wherever the file is kept.

One record type serves every asset: a wall table MIME ships, a mesh a description's manifest lists, a
scan one experiment declares. It says what the file is (its sha256 and size), under what terms (its licence, the
credit and notice it is shown with, its role), where its truth comes from (its kind, with its derivation record's hash
for a derived file), where it may be fetched from (its locations, primary first), and the adapter package a location
of a scheme MIME does not provide needs. Its tier is a verdict :mod:`mime.terms` computes from the licence, never a
value written by hand. A record also has room for what a shared part will need (its own version and citation), so a
part can be built of records without changing their shape.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from typing import Literal, cast
from urllib.parse import urlsplit

from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.terms import ROLES, CustomLicence, Role, Verdict, licence_problem, licence_verdict

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: Plain ``http`` is accepted for these literal hosts only (a mirror on the same machine, the tests' server): the sha256
#: guards the bytes whatever the route, but a remote location is ``https`` or a DOI.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
_UNSAFE = re.compile(r"[\\\s\x00-\x1f\x7f]")
_FORMS = ("a location is an https:// URL, a doi: URL, package:<source>/<path>, a mirror directory file:///<path>/, or "
          "a URL of a scheme an installed adapter provides")
#: A URL scheme as RFC 3986 writes one, lowercase.
_SCHEME = re.compile(r"[a-z][a-z0-9+.-]*")
#: A distribution's name as PEP 508 allows it.
_DISTRIBUTION = re.compile(r"[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?")
#: The name of a source of records a ``package:`` location names: an entry point's name, or ``mime`` for MIME's own.
_SOURCE = re.compile(r"[a-z0-9][a-z0-9_.-]*")
#: The schemes MIME's own adapters read; any other needs an adapter a package provides.
BUILT_IN_SCHEMES = frozenset({"https", "http", "doi", "package", "file"})
PACKAGE_PREFIX = "package:"
MIRROR_PREFIX = "file://"

type Kind = Literal["authored", "measured", "derived"]
#: Where an asset's truth comes from: declared (a CAD mesh, a description), measured (a calibration, a
#: digitised curve: a fact with an uncertainty), or derived (a wall table, a surrogate's weights: a computation its
#: derivation record can repeat).
KINDS: tuple[Kind, ...] = ("authored", "measured", "derived")


class CustomLicenceRecord(TypedDict, closed=True):
    """A custom licence a record's licence names, by its ``LicenseRef-`` id: its title, its link, and what a person
    says it allows (``unknown`` or ``not_permitted``). This is the shape a saved folder's lock holds (a test holds the
    two together): a licence's ``conditions`` are a key of a manifest file (``mime.asset_manifest/2``), written only by
    ``to_record(conditions=True)`` and read by :meth:`AssetRef.from_record`, never by the lock."""

    title: ReadOnly[str]
    url: ReadOnly[str]
    redistribution: ReadOnly[Literal["not_permitted", "unknown"]]


class AssetRecord(TypedDict, closed=True):
    """A record as a lock, a manifest or a package holds it. The first five keys are every record's; the others are
    written only where a record states them. Locations are in the order they are tried."""

    name: ReadOnly[str]
    sha256: ReadOnly[str]
    size: ReadOnly[int]
    licence: ReadOnly[str]
    locations: ReadOnly[list[str]]
    credit: NotRequired[ReadOnly[str]]
    role: NotRequired[ReadOnly[Literal["results", "presentation"]]]
    kind: NotRequired[ReadOnly[Literal["authored", "measured", "derived"]]]
    derivation: NotRequired[ReadOnly[str]]
    notice: NotRequired[ReadOnly[str]]
    licences: NotRequired[ReadOnly[dict[str, CustomLicenceRecord]]]
    version: NotRequired[ReadOnly[str]]
    citation: NotRequired[ReadOnly[str]]
    adapters: NotRequired[ReadOnly[dict[str, str]]]


#: Every record's keys, and the keys a record writes only where it states them.
REQUIRED_KEYS = ("name", "sha256", "size", "licence", "locations")
OPTIONAL_KEYS = ("credit", "role", "kind", "derivation", "notice", "licences", "version", "citation", "adapters")


def _invalid(name: str, problem: str) -> BuildRefused:
    return BuildRefused([Refusal(
        "asset.reference", RefusalClass.USER, f"asset reference {name!r}: {problem}",
        "write the reference as name, sha256 (64 lowercase hex digits), size in bytes, licence (an SPDX licence "
        "expression: licence=\"CC0-1.0\"), and locations that are https:// URLs or doi: URLs: AssetRef(name=..., "
        "sha256=..., size=..., licence=..., locations=(...))")])


#: An access key an address carries as its ``token`` parameter (a record's share link's, or a service's own form): never
#: printed, quoted or logged with the address.
_KEY_PARAMETER = re.compile(r"([?&;]token=)[^&#\s'\"]*", re.IGNORECASE)
#: What an access key is said as.
KEY_HIDDEN = "<hidden>"


def redacted(text: str) -> str:
    """``text`` with the value of every ``token`` parameter of an address in it hidden: what a line, a refusal or a log
    says of an address, which may carry an access key."""
    return _KEY_PARAMETER.sub(lambda found: found.group(1) + KEY_HIDDEN, text)


def carries_key(text: str) -> bool:
    """Whether ``text`` holds an address's ``token`` parameter, which :func:`redacted` hides."""
    return _KEY_PARAMETER.search(text) is not None


def scheme_of(location: str) -> str:
    """A location's scheme, lowercase: the text before its first ``:`` (``https``, ``doi``, ``package``, ``file``, or
    a scheme an adapter provides)."""
    return location.partition(":")[0].lower()


def _package_problem(location: str) -> str | None:
    source, _, path = location.removeprefix(PACKAGE_PREFIX).partition("/")
    if not _SOURCE.fullmatch(source) or not path:
        return "a package: location is package:<source>/<path>, the source the name of a mime.assets entry point"
    if _UNSAFE.search(path) or path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")):
        return "a package: location's path is relative, its parts neither empty, '.' nor '..', with no backslash"
    return None


def _mirror_problem(location: str) -> str | None:
    rest = location.removeprefix(MIRROR_PREFIX)
    if not rest.startswith("/") or not location.endswith("/") or _UNSAFE.search(location.replace(" ", "")):
        return "a mirror is a directory written file:///<absolute path>/, ending in '/'"
    if any(part in (".", "..") for part in rest.split("/")):
        return "a mirror's path has no '.' or '..' part"
    return None


def location_problem(location: str, adapters: Sequence[str] = ()) -> str | None:
    """Why ``location`` is not a location a record may hold, or ``None``: an ``https://`` URL with a host,
    ``doi:<doi>/<file>``, ``http://`` to a loopback host (``127.0.0.1``, ``localhost`` or ``[::1]``),
    ``package:<source>/<path>`` (bytes inside an installed package), ``file:///<path>/`` (a mirror directory), or a URL
    of another scheme where an adapter is registered for it or the record names the package that provides it
    (``adapters``, its schemes).

    An ``http(s)`` URL's host is read once, by ``urllib3``, and the standard library's parse must agree with it; a URL
    with a backslash, whitespace, a control character, userinfo or a percent-escape in its authority is refused
    whatever its host, since parsers disagree on those. No location carries userinfo: a credential comes from the
    machine's store, never from a record."""
    if location.startswith("doi:"):
        doi, _, file = location.removeprefix("doi:").rpartition("/")
        if doi.startswith("10.") and "/" in doi and file:
            return None
        return "a doi: location is doi:10.<registrant>/<suffix>/<file>"
    if location.startswith(PACKAGE_PREFIX):
        return _package_problem(location)
    if location.startswith(MIRROR_PREFIX):
        return _mirror_problem(location)
    if _UNSAFE.search(location):
        return "a location has no backslash, whitespace or control character"
    scheme, _, rest = location.partition(":")
    authority = location.partition("://")[2]
    for end in "/?#":
        authority = authority.partition(end)[0]
    if "@" in authority or "%" in authority:
        return "a location has no userinfo and no percent-escape in its host"
    if scheme in ("http", "https"):
        return _web_problem(location)
    if not _SCHEME.fullmatch(scheme) or not rest or scheme in BUILT_IN_SCHEMES:
        return _FORMS
    from mime.assets._adapters import registered_schemes

    if scheme in adapters or scheme in registered_schemes():
        return None
    return (f"no adapter for the scheme {scheme!r} is installed, and the record names no package that provides one: "
            f"{_FORMS}")


def _web_problem(location: str) -> str | None:
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


def _text_problem(value: object, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return f"its {what} is a text that is not empty"
    return None


@dataclass(frozen=True)
class AssetRef:
    """A file named by the sha256 of its bytes, never by where it lives. ``locations`` are tried in order, so a new
    home is one more location, and no saved experiment that names the sha256 changes. ``licence`` is an SPDX licence
    expression in its canonical spelling (``CC0-1.0``), or ``NOASSERTION`` where nobody has stated one
    (:mod:`mime.terms`); free text is refused. A ``LicenseRef-`` id it names is defined in ``licences``
    (:class:`~mime.terms.CustomLicence`), so its verdict can say ``not_permitted``.

    The rest is stated where it is known, and written only then: ``credit`` (whom the licence asks to be credited),
    ``role`` (``results``, or ``presentation``: no result depends on it), ``kind`` (``authored``, ``measured`` or
    ``derived``, with ``derivation`` the sha256 of its derivation record), ``notice`` (terms to show with it),
    ``version`` and ``citation`` (its own identity, as a shared part's), and ``adapters``, each scheme of a location MIME
    does not read itself with the package that provides its adapter, so ``mime pull`` can name it. ``unlisted`` reads
    a licence id the installed packaging does not list as written, as a reference read from a record is
    (:meth:`from_record`): the packaging that wrote it listed it."""

    name: str
    sha256: str
    size: int
    licence: str
    locations: tuple[str, ...]
    credit: str | None = None
    role: Role | None = None
    kind: Kind | None = None
    derivation: str | None = None
    notice: str | None = None
    licences: tuple[CustomLicence, ...] = ()
    version: str | None = None
    citation: str | None = None
    adapters: tuple[tuple[str, str], ...] = ()
    unlisted: InitVar[bool] = False

    def __post_init__(self, unlisted: bool) -> None:
        if not _NAME.fullmatch(self.name):
            raise _invalid(self.name, "a name is letters, digits, '.', '_' and '-', starting with a letter or digit")
        if not _SHA256.fullmatch(self.sha256):
            raise _invalid(self.name, f"sha256 {self.sha256!r} is not 64 lowercase hex digits")
        if self.size <= 0:
            raise _invalid(self.name, f"size {self.size} is not a positive number of bytes")
        licence = licence_problem(self.licence, unlisted=unlisted)
        if licence is not None:
            raise _invalid(self.name, licence)
        for problem in self._stated_problems():
            raise _invalid(self.name, problem)
        if not self.locations:
            raise _invalid(self.name, "no location")
        schemes = [scheme for scheme, _ in self.adapters]
        for location in self.locations:
            problem = location_problem(location, schemes)
            if problem is not None:
                raise _invalid(self.name, f"location {location!r}: {problem}")

    def _stated_problems(self) -> list[str]:
        found = [problem for value, what in ((self.credit, "credit"), (self.notice, "notice"),
                                             (self.version, "version"), (self.citation, "citation"))
                 if (problem := _text_problem(value, what)) is not None]
        if self.role is not None and self.role not in ROLES:
            found.append(f"its role is {' or '.join(ROLES)}, not {self.role!r}")
        if self.kind is not None and self.kind not in KINDS:
            found.append(f"its kind is {', '.join(KINDS)}, not {self.kind!r}")
        if self.derivation is not None and (self.kind != "derived" or not _SHA256.fullmatch(str(self.derivation))):
            found.append("its derivation is the sha256 of its derivation record, and only a derived asset has one")
        ids: set[str] = set()
        for held in self.licences:
            problem = held.problem() if isinstance(cast(object, held), CustomLicence) else "not a CustomLicence"
            if problem is not None:
                found.append(f"its custom licence: {problem}")
            elif held.id in ids:
                found.append(f"its custom licence {held.id} is defined twice")
            ids.add(held.id)
        for scheme, package in self.adapters:
            if not _SCHEME.fullmatch(scheme) or scheme in BUILT_IN_SCHEMES:
                found.append(f"its adapters name the scheme {scheme!r}, which is not one an adapter package provides")
            elif package and not _DISTRIBUTION.fullmatch(package):
                found.append(f"its adapters name the package {package!r} for {scheme}, which is not a distribution's "
                             f"name")
        return found

    @property
    def custom(self) -> Mapping[str, CustomLicence]:
        """The custom licences the record defines, by id."""
        return {held.id: held for held in self.licences}

    @property
    def verdict(self) -> Verdict:
        """The record's verdict, from its licence and the custom licences it defines (``mime.terms.licence_verdict``):
        ``open`` and ``conditional`` bytes may travel, ``referenced`` ones never."""
        return licence_verdict(self.licence, self.custom)

    @property
    def tier(self) -> str:
        return self.verdict.tier

    def to_record(self, *, conditions: bool = False) -> AssetRecord:
        """The record as a lock or a manifest holds it: the five keys every record has, then each one it states. The
        conditions of its custom licences (short lines a download shows and asks to be accepted) are written only
        with ``conditions``, which a manifest file does (``mime.asset_manifest/2``) and a saved folder's lock, whose
        form has no key for them, does not."""
        held: dict[str, object] = {"name": self.name, "sha256": self.sha256, "size": self.size,
                                   "licence": self.licence, "locations": list(self.locations)}
        for key in ("credit", "role", "kind", "derivation", "notice"):
            value = getattr(self, key)
            if value is not None:
                held[key] = value
        if self.licences:
            held["licences"] = {each.id: {"title": each.title, "url": each.url, "redistribution": each.redistribution,
                                          **({"conditions": list(each.conditions)}
                                             if conditions and each.conditions else {})}
                                for each in sorted(self.licences, key=lambda each: each.id)}
        for key in ("version", "citation"):
            value = getattr(self, key)
            if value is not None:
                held[key] = value
        if self.adapters:
            held["adapters"] = dict(sorted(self.adapters))
        return cast(AssetRecord, held)

    @staticmethod
    def from_record(record: Mapping[str, object]) -> AssetRef:
        """The reference a record describes; a key missing or unknown, or a value of another type, is refused."""
        name = str(record.get("name", "?"))
        keys = set(record)
        if not set(REQUIRED_KEYS) <= keys or not keys <= {*REQUIRED_KEYS, *OPTIONAL_KEYS}:
            raise _invalid(name, f"the record's keys are {sorted(record)}: {', '.join(REQUIRED_KEYS)}, and any of "
                                 f"{', '.join(OPTIONAL_KEYS)}")
        size, listed = record["size"], record["locations"]
        locations = cast("list[object]", listed) if isinstance(listed, list) else None
        texts = (record["name"], record["sha256"], record["licence"])
        if (not all(isinstance(text, str) for text in texts) or not isinstance(size, int) or isinstance(size, bool)
                or locations is None or not all(isinstance(each, str) for each in locations)):
            raise _invalid(name, "the record's values are not str, str, int, str and a list of str")
        stated: dict[str, str] = {}
        for key in ("credit", "role", "kind", "derivation", "notice", "version", "citation"):
            if key in record:
                value = record[key]
                if not isinstance(value, str):
                    raise _invalid(name, f"the record's {key} is not a text")
                stated[key] = value
        licences = _licences_of(name, record.get("licences", {}))
        adapters = _adapters_of(name, record.get("adapters", {}))
        return AssetRef(str(texts[0]), str(texts[1]), size, str(texts[2]), tuple(str(each) for each in locations),
                        stated.get("credit"), cast("Role | None", stated.get("role")),
                        cast("Kind | None", stated.get("kind")), stated.get("derivation"), stated.get("notice"),
                        licences, stated.get("version"), stated.get("citation"), adapters, unlisted=True)


@dataclass(frozen=True)
class BundleContainer(AssetRef):
    """The reference to the file a pull has just downloaded to read a bundle from: a container, not an asset of an
    experiment, so it states no licence (``NOASSERTION``) and its own terms are judged when it is unpacked. Only the
    download of a bundle (:func:`~mime.assets._fetch.retrieve`) makes one; a record read from a lock, a manifest or a
    locations file is always a plain :class:`AssetRef`, so no record can claim to be one. It is what
    :func:`~mime.assets._cache.locate_bundle_container` accepts without asking about a licence."""


def _licences_of(name: str, held: object) -> tuple[CustomLicence, ...]:
    shape = ("the record's licences are an object of LicenseRef- id to its title, url and redistribution, as texts, "
             "and its conditions, a list of texts, where it has them")
    if not isinstance(held, dict):
        raise _invalid(name, shape)
    found: list[CustomLicence] = []
    for key, value in cast("dict[object, object]", held).items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise _invalid(name, shape)
        entry = dict(cast("dict[str, object]", value))
        said = entry.pop("conditions", ())
        lines = cast("list[object]", said) if isinstance(said, list) else None
        if (set(entry) != {"title", "url", "redistribution"} or not all(isinstance(each, str) for each in entry.values())
                or (lines is None and said != ()) or not all(isinstance(each, str) for each in lines or ())):
            raise _invalid(name, shape)
        found.append(CustomLicence(key, str(entry["title"]), str(entry["url"]),
                                   cast("Literal['not_permitted', 'unknown']", entry["redistribution"]),
                                   tuple(str(each) for each in lines or ())))
    return tuple(found)


def _adapters_of(name: str, held: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(held, dict) or not all(isinstance(key, str) and isinstance(value, str)
                                              for key, value in cast("dict[object, object]", held).items()):
        raise _invalid(name, "the record's adapters are an object of scheme to the package that provides its adapter")
    return tuple(sorted(cast("dict[str, str]", held).items()))


__all__ = ["BUILT_IN_SCHEMES", "KEY_HIDDEN", "KINDS", "MIRROR_PREFIX", "OPTIONAL_KEYS", "PACKAGE_PREFIX",
           "REQUIRED_KEYS", "AssetRecord", "AssetRef", "CustomLicenceRecord", "Kind", "carries_key",
           "location_problem", "redacted", "scheme_of"]
