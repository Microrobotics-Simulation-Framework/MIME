"""One way to declare a file the experiment reads: ``asset(source)``, the source decided by its type, never guessed from a
string's shape.

- **A string is a record's name**, dotted and namespaced like an entry's (``asset("softtissue.fibrin_plasma")``; MIME's
  own have no namespace, ``asset("dejongh_mlp_v2")``): MIME's packaged files, a record an installed package ships, or
  one a package's resolver answers (``mime.assets`` entry point). Never a URL: a remote file needs a record with its
  sha256.
- **A path** (``Path("arm.urdf")``) **is a local file**, pinned as the build finds it or to ``sha256=``, with ``terms=``
  where it is a third party's. Its record is made when it is read: its sha256, and the licence its terms state or
  ``NOASSERTION``, so the record alone grants nothing until a licence is declared.
- **An** :class:`~mime.assets.AssetRef` **is a whole record**: a one-off file named by its sha256 with its terms and
  locations, read from the asset cache.

Each origin keeps the saved form it always had: a name as ``{asset, sha256}`` (and in the lock's ``resolved.assets``
with where its bytes come from), a file or a record as ``{file, sha256}`` with its bytes in the folder, and a record
whole in the lock's ``assets``. Each setting that takes a file says which origins it reads."""

from __future__ import annotations

import hashlib
import importlib.resources
import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal, cast

from mime._refusals import ArgumentRefused, ArgumentTypeRefused, BuildRefused, MimeError, Refusal, RefusalClass
from mime.assets._adapters import MIME_SOURCE
from mime.assets._reference import PACKAGE_PREFIX, AssetRef
from mime.assets._sources import claims_an_own_name
from mime.terms import NO_ASSERTION, CustomLicence, Terms, expression_of

type Origin = Literal["named", "file", "record"]
#: Where a location of MIME's own package data starts: ``package:mime/<path>``, a path below ``mime.data``.
_PACKAGED = f"{PACKAGE_PREFIX}{MIME_SOURCE}/"
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: A character a record's name may not hold, in a local file's name.
_NOT_IN_A_NAME = re.compile(r"[^A-Za-z0-9._-]")
#: The forms ``asset`` takes, as an action says them.
FORMS = "asset(\"<record name>\"), asset(Path(\"<file>\")) or asset(AssetRef(...))"


@dataclass(frozen=True)
class Asset:
    """A file the experiment reads, by one of three origins (:attr:`origin`): a record's ``name`` with the ``sha256`` the
    experiment is written against; a local file at ``path``, pinned to ``sha256`` or as the build finds it, with
    ``terms`` where it is a third party's (a file without them is the author's own, under the experiment's licence); or
    a whole ``record``, whose file the asset cache keeps at ``path`` under the record's name. :meth:`ref` is the record
    behind any of them."""

    name: str
    sha256: str | None = None
    path: Path | None = None
    terms: Terms | None = None
    record: AssetRef | None = None

    def __post_init__(self) -> None:
        if self.path is not None and self.name != self.path.name:
            object.__setattr__(self, "name", self.path.name)
        if self.path is None and self.record is None and not (isinstance(cast(object, self.sha256), str)
                                                              and _SHA256.fullmatch(self.sha256 or "")):
            raise ArgumentRefused([Refusal(
                "asset.source", RefusalClass.USER,
                f"Asset({self.name!r}) names a record without the sha256 the experiment is written against",
                f"write asset({self.name!r}) to take the record's sha256, or Asset({self.name!r}, \"<64 hex digits>\")")])

    @property
    def origin(self) -> Literal["named", "file", "record"]:
        """``named`` (a record's name), ``file`` (a local file) or ``record`` (a whole record)."""
        if self.record is not None:
            return "record"
        return "file" if self.path is not None else "named"

    def read(self) -> bytes:
        """The file's bytes: a local file as it is; a record's from the asset cache (never downloaded); a name's from
        MIME's package, or from the cache through its licence gate, each held to the pinned sha256."""
        if self.record is not None:
            from mime.assets._cache import named_file

            return named_file(self.record).read_bytes()
        if self.path is not None:
            return self.path.read_bytes()
        return _named_bytes(self)

    def ref(self) -> AssetRef:
        """The record behind the asset: a name's, as the experiment's pins or the sources hold it; the record declared;
        or, for a local file, one made from the file as read: its sha256 and size, and the licence, credit, role and
        origin its terms state, ``NOASSERTION`` and no location without them."""
        if self.record is not None:
            return self.record
        if self.path is None:
            from mime.assets._known import reference

            return reference(self.name)
        content = self.read()
        terms = self.terms
        licence = terms.licence if terms is not None else NO_ASSERTION
        return AssetRef(_record_name(self.name), self.sha256 or hashlib.sha256(content).hexdigest(), len(content),
                        expression_of(licence), terms.origin if terms is not None else (),
                        credit=(terms.credit or None) if terms is not None else None,
                        role=terms.role if terms is not None else None,
                        licences=(licence,) if isinstance(licence, CustomLicence) else ())


def _record_name(file_name: str) -> str:
    held = _NOT_IN_A_NAME.sub("_", file_name)
    return held if held[:1].isalnum() else f"f{held}"


def _record_terms(record: AssetRef) -> Terms:
    """The terms a file declared by its record carries: the record's licence (its custom licence where it names one
    alone), its credit, its role, and its locations as the origin its bytes travel by."""
    licence: str | CustomLicence = record.custom.get(record.licence, record.licence)
    return Terms(licence, record.credit or "", record.role or "results", record.locations)


#: Each origin, as a refusal says it.
ORIGIN_WORDS: dict[Origin, str] = {"named": "a record's name", "file": "a local file", "record": "a whole record"}


def refused_origin(held: Asset, setting: str, forms: str) -> ArgumentRefused:
    """``asset.source``: ``setting`` does not read ``held``'s origin; ``forms`` are the ones it reads."""
    return ArgumentRefused([Refusal(
        "asset.source", RefusalClass.USER,
        f"{setting} does not read {ORIGIN_WORDS[held.origin]}, asset({held.name!r})", f"write `{forms}`")])


#: A string that starts as a location: a scheme of two letters or more and a colon (``https:``, ``DOI:``, ``s3:``;
#: ``C:`` is a drive), or two slashes or backslashes (a host's share).
_LOCATION = re.compile(r"[A-Za-z][A-Za-z0-9+.-]+:|//|\\\\")
#: A path whose text starts as a URL: ``Path("https://x")`` reads ``https:/x``.
_URL_PATH = re.compile(r"[A-Za-z][A-Za-z0-9+.-]+:/")


def _location_refused(shown: object, text: str) -> ArgumentRefused:
    return ArgumentRefused([Refusal(
        "asset.source", RefusalClass.USER,
        f"asset({shown!r}) names a location; a string names a record, and a remote file needs a record with its "
        f"sha256", f"write asset(AssetRef(name=\"<name>\", sha256=\"<64 hex digits>\", size=<bytes>, "
                  f"licence=\"<SPDX id>\", locations=({text!r},))), or give a record's name")])


def _named(name: str, sha256: str | None, terms: Terms | None) -> Asset:
    if _LOCATION.match(name) or "://" in name:
        raise _location_refused(name, name)
    if terms is not None:
        raise ArgumentRefused([Refusal(
            "asset.source", RefusalClass.USER, f"asset({name!r}, terms=...) gives terms to a record, whose terms are its own",
            f"write asset({name!r}) alone; terms= is a local file's, asset(Path(...), terms=...)")])
    from mime.assets._known import known, reference

    try:
        found = reference(name)
    except BuildRefused as refused:
        if refused.code != "asset.unknown":
            raise
        file_here = Path(name).is_file()
        names = sorted(known())
        raise ArgumentRefused([Refusal(
            "asset.unknown", RefusalClass.USER, f"no record is named {name!r}"
            + (f", and {name!r} is a file here: a string names a record" if file_here else ""),
            f"write asset(Path({name!r})) for a file on disk" if file_here else
            f"for a file on disk write asset(Path({name!r})); a record is named by one of {names}")
        ]) from None
    return Asset(name, sha256 if sha256 is not None else found.sha256)


def asset(source: str | os.PathLike[str] | AssetRef, *, sha256: str | None = None, terms: Terms | None = None) -> Asset:
    """A file the experiment reads, its origin decided by the type of ``source``: a record's name (a string), a local
    file (a path: pinned as the build finds it or to ``sha256``, with ``terms`` where it is a third party's), or a whole
    record (an :class:`~mime.assets.AssetRef`, whose sha256 and terms are its own). Refused: a string that names no
    record (``asset.unknown``, with ``asset(Path(...))`` as the action where it is a file here), a string that is a
    location, terms given to a record or a name, and a sha256 that is not 64 lowercase hex digits (``asset.source``);
    any other type (``call.wrong-type``)."""
    if sha256 is not None and not (isinstance(cast(object, sha256), str) and _SHA256.fullmatch(sha256)):
        raise ArgumentRefused([Refusal(
            "asset.source", RefusalClass.USER, f"sha256={sha256!r} is not 64 lowercase hex digits",
            "write the file's SHA-256 as 64 lowercase hex digits, or leave sha256= out")])
    if isinstance(source, AssetRef):
        refused = claims_an_own_name(source)
        if refused is not None:
            raise refused
        if sha256 is not None or terms is not None:
            raise ArgumentRefused([Refusal(
                "asset.source", RefusalClass.USER,
                "a file declared by its record takes its sha256 and its terms from the record", "write asset(record) alone")])
        from mime.assets._cache import named_path

        return Asset(source.name, source.sha256, named_path(source), _record_terms(source), source)
    if isinstance(source, str):
        return _named(source, sha256, terms)
    if isinstance(cast(object, source), os.PathLike):
        path = Path(source)
        if _URL_PATH.match(str(path)):
            raise _location_refused(path, str(source))
        return Asset(path.name, sha256, path, terms)
    raise ArgumentTypeRefused([Refusal(
        "call.wrong-type", RefusalClass.USER, f"asset takes a record's name, a path or an AssetRef, not "
                                              f"{type(source).__name__}", f"write `{FORMS}`")])


# --- reading a name's bytes ----------------------------------------------------------------------------------------


def packaged_path(ref: AssetRef) -> str | None:
    """The path below ``mime.data`` of a record whose bytes MIME's own package holds (its location
    ``package:mime/<path>``), or ``None``."""
    return next((location.removeprefix(_PACKAGED) for location in ref.locations if location.startswith(_PACKAGED)), None)


def lock_source(held: Asset) -> str:
    """Where a name's bytes are read from, as the lock's ``resolved.assets`` says it: ``mime.data/<path>`` for MIME's
    own packaged file, else ``not packaged``."""
    try:
        path = packaged_path(held.ref())
    except MimeError:
        path = None
    return f"mime.data/{path}" if path is not None else "not packaged"


@cache
def _packaged_contents(path: str) -> tuple[bytes, str]:
    data = importlib.resources.files("mime.data").joinpath(*path.split("/")).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def _mismatch(held: Asset, ref: AssetRef, digest: str, path: str | None) -> str:
    if path is not None:
        return (f"the packaged asset {held.name!r} (mime.data/{path}) has sha256 {digest}, and the experiment asks "
                f"for {held.sha256}: the installed MIME ships other contents under that name")
    return (f"the record {held.name!r} names the bytes {ref.sha256}, and the experiment asks for {held.sha256}: write "
            f"the record's sha256, or the record the experiment was written against")


def named_problem(held: Asset) -> str | None:
    """Why a name's bytes cannot be read as the experiment pins them, or ``None``: no record of the name, a record of
    other bytes, or MIME's packaged file of other bytes. A record's file in the cache is judged where it is read (its
    gate's own codes: ``asset.missing``, ``asset.licence-not-accepted``)."""
    try:
        ref = held.ref()
    except MimeError as refused:
        return "; ".join(refusal.message for refusal in refused.refusals)
    path = packaged_path(ref)
    if path is None:
        return _mismatch(held, ref, ref.sha256, None) if ref.sha256 != held.sha256 else None
    try:
        _, digest = _packaged_contents(path)
    except OSError as error:
        return f"the packaged asset {held.name!r} (mime.data/{path}) cannot be read: {error}"
    return _mismatch(held, ref, digest, path) if digest != held.sha256 else None


def _named_bytes(held: Asset) -> bytes:
    problem = named_problem(held)
    if problem is not None:
        raise BuildRefused([Refusal("asset.hash-mismatch", RefusalClass.ENVIRONMENT, problem,
                                    "install the MIME, or the package, the experiment was written against, or "
                                    "declare the asset again with `asset(...)` to take these bytes")])
    ref = held.ref()
    path = packaged_path(ref)
    if path is not None:
        from mime.assets._gates import require_accepted

        require_accepted([ref])
        return _packaged_contents(path)[0]
    from mime.assets._cache import locate

    data = locate(ref).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != held.sha256:
        raise BuildRefused([Refusal(
            "asset.hash-mismatch", RefusalClass.ENVIRONMENT,
            f"the stored file of the record {held.name!r} has sha256 {digest}, not the record's {held.sha256}",
            f"fetch it again: `mime assets fetch {held.name}`")])
    return data


__all__ = ["FORMS", "ORIGIN_WORDS", "Asset", "Origin", "asset", "lock_source", "named_problem", "packaged_path",
           "refused_origin"]
