"""A manifest: a named group of ordinary asset records, with the credit and notice the group is shown with, and how a
description names each of its files.

A robot's meshes are a manifest beside its description (``visuals.json`` next to a URDF), so choosing the robot brings
its recipe; a lab's package ships manifests of its own; one experiment's scan is a record on its own. Each entry of
``assets`` is a record (``AssetRef``) as a lock holds it; the manifest's ``credit`` and ``licences`` fill each record
that does not state its own, so a manifest of eighteen meshes need not repeat them. ``files`` maps each path a
description writes (``meshes/ar4_mk5/Link_1_Aluminum.STL``) to a record by name, with its git blob where it comes from
a repository (which names it in any clone, and in Software Heritage). No tier is written: each record's is the verdict
of its licence (:mod:`mime.terms`).

The format is ``mime.asset_manifest/3``. Its addition to ``/2`` is the **record with no location** (saved form 14's):
a record may name no location, its file had by request alone, and may name where to request it (``request``). A ``/2``
or ``/1`` manifest that holds either is refused, never read as stating it, and a MIME that reads only ``/2`` refuses a
``/3`` manifest by its format, naming a later MIME, where it would otherwise refuse its record's shape. ``/2``'s
addition to ``/1`` is the **conditions** of a custom licence: each entry of ``licences`` may hold ``conditions``, a list
of short lines (``non-commercial use only``) that a download of a file under that licence shows and asks to be accepted
(:mod:`mime.assets._gates`), beside the manifest's ``notice``, which stays as the licence's own words. ``/2`` and
``/1`` are still read, ``/1`` as a manifest whose licences state no condition lines; a ``/1`` manifest that holds
``conditions`` is refused, never read as stating them. A saved folder's lock (saved form 6) holds a manifest as
``/1``, with no condition lines: its form has no key for them; its records hold what the lock's own form allows (from
saved form 14, a record with no location), and are read so (``read_manifest_data(..., lock=True)``). The earlier visual-set form (``mime.visual_set/1``) is still read: its
hand-written tier is dropped and the verdict computed, and its licence, which it named with a link alone, is read as a
custom licence whose redistribution nobody stated (unknown, so referenced). Any other format is refused
(``asset.manifest``), never misread.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib.resources.abc import Traversable
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from mime._terminal import for_terminal, plain
from mime.assets._reference import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.terms import CustomLicence, Verdict, strictest

MANIFEST_FORMAT = "mime.asset_manifest/3"
#: The form before a record could name no location: the same, every record with a location and no ``request``. Read.
SECOND_FORMAT = "mime.asset_manifest/2"
#: The first form of a manifest: as the second, with no conditions on a custom licence. Read, and what a saved folder's
#: lock holds.
FIRST_FORMAT = "mime.asset_manifest/1"
#: The earlier form of a description's meshes, before a visual set was a manifest of records.
VISUAL_SET_FORMAT = "mime.visual_set/1"
#: The manifest MIME looks for beside a description, and the name it takes after the description's own stem.
BESIDE_NAME = "visuals.json"
BESIDE_SUFFIX = ".visuals.json"
_KEYS = frozenset({"format", "name", "title", "credit", "notice", "licences", "upstream", "archives",
                   "archive_checked", "assets", "files"})


@dataclass(frozen=True)
class ManifestFile:
    """How a description names one of a manifest's files: its path as the description writes it, the record's name,
    and its git blob where it comes from a repository."""

    path: str
    asset: str
    git_blob: str | None = None


@dataclass(frozen=True)
class Manifest:
    """A named group of records: its name, title, credit and notice; the records; how a description names each
    file; where they come from upstream and which archives hold them (as read on a date); and where the manifest was
    read from (``where``), for messages."""

    name: str
    title: str
    credit: str
    notice: str
    assets: tuple[AssetRef, ...]
    files: tuple[ManifestFile, ...] = ()
    upstream: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    archives: tuple[str, ...] = ()
    archive_checked: str | None = None
    where: str = ""

    def record(self, name: str) -> AssetRef | None:
        return next((held for held in self.assets if held.name == name), None)

    def file(self, mesh: str) -> tuple[ManifestFile, AssetRef] | None:
        """The file and record a description's mesh names (``meshes/ar4_mk5/Link_1_Aluminum.STL``, or with its
        package, ``package://<package>/meshes/...``), or ``None``."""
        wanted = mesh_path(mesh)
        held = next((each for each in self.files if each.path == wanted), None)
        if held is None:
            return None
        found = self.record(held.asset)
        return (held, found) if found is not None else None

    @property
    def verdict(self) -> Verdict:
        """The manifest's verdict as a whole: its least free record's (no record is freer than its licence allows)."""
        return strictest([held.verdict for held in self.assets])

    @property
    def licence(self) -> str:
        return ", ".join(dict.fromkeys(held.licence for held in self.assets))

    @property
    def role(self) -> str:
        return ", ".join(dict.fromkeys(held.role or "results" for held in self.assets))

    def text(self, *, notice: bool = True) -> str:
        """The credit, licence, tier and notice, as a command prints them before it fetches (without the notice where
        a licence panel has just shown it). Every word is a stranger's, so it is written as plain characters
        (:mod:`mime._terminal`): an escape sequence in a title cannot rewrite the screen it is printed on."""
        links = "; ".join(f"{plain(each.id)}: {plain(each.title)} ({plain(each.url)})"
                          for each in {c.id: c for held in self.assets for c in held.licences}.values())
        return (f"{plain(self.title)}\ncredit: {plain(self.credit)}\nlicence: {plain(self.licence)}"
                + (f" ({links})" if links else "")
                + f"; tier {self.verdict.tier}, role {plain(self.role)}"
                + (f"\n{for_terminal(self.notice)}" if notice else ""))

    def to_json(self, *, lock: bool = False) -> dict[str, Any]:
        """The manifest as its file holds it (``/3``): each record whole, with no tier. With ``lock``, as a saved
        folder's lock holds it (``/1``), the licences' condition lines left out."""
        held: dict[str, Any] = {"format": FIRST_FORMAT if lock else MANIFEST_FORMAT, "name": self.name,
                                "title": self.title, "credit": self.credit, "notice": self.notice}
        if self.upstream:
            held["upstream"] = dict(self.upstream)
        held["archives"] = list(self.archives)
        if self.archive_checked is not None:
            held["archive_checked"] = self.archive_checked
        held["assets"] = [dict(each.to_record(conditions=not lock)) for each in self.assets]
        held["files"] = {each.path: {"asset": each.asset, **({"git_blob": each.git_blob} if each.git_blob else {})}
                         for each in self.files}
        return held


def mesh_path(mesh: str) -> str:
    """A description's mesh file name as a manifest lists it: without a ``package://<package>/`` or ``file://``
    prefix, or a leading ``./``."""
    held = mesh.strip()
    if held.startswith("package://"):
        held = held.removeprefix("package://").partition("/")[2]
    held = held.removeprefix("file://")
    while held.startswith("./"):
        held = held[2:]
    return held


def _refused(where: str, problem: str) -> BuildRefused:
    return BuildRefused([Refusal(
        "asset.manifest", RefusalClass.USER, f"the manifest {where}: {problem}",
        f"write it as {MANIFEST_FORMAT}: format, name, title, credit, notice, assets (records) and files (each path a "
        f"description writes, with its record's name); `mime assets sources <manifest file>` reads it")])


def _text(record: Mapping[str, object], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _refused(where, f"its {key} is a text that is not empty")
    return value


def _custom(held: object, where: str, *, conditions: bool = True) -> tuple[CustomLicence, ...]:
    if held is None:
        return ()
    shape = ("its licences are an object of LicenseRef- id to title, url and redistribution"
             + (", and conditions (a list of short lines) where it has them" if conditions
                else f" ({FIRST_FORMAT} has no conditions: write the manifest as {MANIFEST_FORMAT} to state them)"))
    if not isinstance(held, dict):
        raise _refused(where, shape)
    found: list[CustomLicence] = []
    for key, value in cast(dict[object, object], held).items():
        entry = dict(cast(dict[str, object], value)) if isinstance(value, dict) else None
        said = entry.pop("conditions", ()) if entry is not None and conditions else ()
        lines = cast(list[object], said) if isinstance(said, list) else None
        if (not isinstance(key, str) or entry is None or set(entry) != {"title", "url", "redistribution"}
                or (lines is None and said != ()) or not all(isinstance(each, str) for each in lines or ())):
            raise _refused(where, shape)
        licence = CustomLicence(key, str(entry["title"]), str(entry["url"]), cast(Any, entry["redistribution"]),
                                tuple(str(each) for each in lines or ()))
        problem = licence.problem()
        if problem is not None:
            raise _refused(where, problem)
        found.append(licence)
    return tuple(found)


def _filled(ref: AssetRef, credit: str, custom: tuple[CustomLicence, ...]) -> AssetRef:
    """A manifest's record with the manifest's credit, and each custom licence its licence names, where it states none
    of its own."""
    named = [each for each in custom if each.id in ref.licence and each.id not in ref.custom]
    return replace(ref, credit=ref.credit or credit, licences=(*ref.licences, *named))


def read_manifest_data(data: object, where: str, *, lock: bool = False) -> Manifest:
    """The manifest a JSON value holds, read from ``where``: refused (``asset.manifest``) whole where it is not one.
    ``lock`` reads one a saved folder's lock holds (``/1``), whose records hold what the lock's own form allows: the
    loader judges that, by the lock's version."""
    if not isinstance(data, dict):
        raise _refused(where, "is not a JSON object")
    record = cast(dict[str, object], data)
    stated = record.get("format")
    if stated == VISUAL_SET_FORMAT:
        return _from_visual_set(record, where)
    if stated not in (MANIFEST_FORMAT, SECOND_FORMAT, FIRST_FORMAT):
        raise _refused(where, f"its format is {stated!r}, and this MIME reads {MANIFEST_FORMAT} (and the earlier "
                              f"{SECOND_FORMAT}, {FIRST_FORMAT} and {VISUAL_SET_FORMAT}); a later one is written by a "
                              f"later MIME")
    unknown = sorted(set(record) - _KEYS)
    if unknown:
        raise _refused(where, f"it holds keys a manifest does not: {', '.join(unknown)}")
    name, title = _text(record, "name", where), _text(record, "title", where)
    credit, notice = _text(record, "credit", where), _text(record, "notice", where)
    custom = _custom(record.get("licences"), where, conditions=stated != FIRST_FORMAT)
    listed = record.get("assets")
    if not isinstance(listed, list) or not listed:
        raise _refused(where, "its assets are a list of records, at least one")
    refs: list[AssetRef] = []
    for each in cast(list[object], listed):
        if not isinstance(each, dict):
            raise _refused(where, "an entry of its assets is not a record")
        try:
            refs.append(_filled(AssetRef.from_record(cast(dict[str, object], each)), credit, custom))
        except BuildRefused as refused:
            raise _refused(where, "; ".join(str(r.message) for r in refused.refusals)) from None
    if stated != MANIFEST_FORMAT and not lock and any(ref.by_request or ref.request is not None for ref in refs):
        raise _refused(where, f"a record of it names no location or a route to request its file, which "
                              f"{MANIFEST_FORMAT} added (write the manifest as {MANIFEST_FORMAT})")
    if stated == FIRST_FORMAT and any(held.conditions for ref in refs for held in ref.licences):
        raise _refused(where, f"a record of it states conditions of a licence, which {FIRST_FORMAT} has no key for "
                              f"(write the manifest as {MANIFEST_FORMAT})")
    names = [ref.name for ref in refs]
    if len(set(names)) != len(names):
        raise _refused(where, "two of its records have one name")
    files = _files(record.get("files", {}), set(names), where)
    upstream = record.get("upstream", {})
    archives = record.get("archives", [])
    checked = record.get("archive_checked")
    if not isinstance(upstream, dict) or not all(isinstance(v, str) for v in cast(dict[str, object], upstream).values()):
        raise _refused(where, "its upstream is an object of texts")
    if not isinstance(archives, list) or not all(isinstance(v, str) for v in cast(list[object], archives)):
        raise _refused(where, "its archives are a list of texts")
    if checked is not None and not isinstance(checked, str):
        raise _refused(where, "its archive_checked is a text")
    return Manifest(name, title, credit, notice, tuple(refs), files,
                    MappingProxyType(dict(cast(dict[str, str], upstream))), tuple(cast(list[str], archives)),
                    checked, where)


def _files(held: object, names: set[str], where: str) -> tuple[ManifestFile, ...]:
    if not isinstance(held, dict):
        raise _refused(where, "its files are an object of a description's path to its record's name")
    found: list[ManifestFile] = []
    for path, value in cast(dict[object, object], held).items():
        entry = cast(dict[str, object], value) if isinstance(value, dict) else None
        if not isinstance(path, str) or entry is None or not set(entry) <= {"asset", "git_blob"} \
                or not isinstance(entry.get("asset"), str):
            raise _refused(where, "each of its files is {asset: <record name>}, and its git_blob where it has one")
        if entry["asset"] not in names:
            raise _refused(where, f"its file {path} names the record {entry['asset']!r}, which it does not hold")
        blob = entry.get("git_blob")
        found.append(ManifestFile(mesh_path(path), str(entry["asset"]), str(blob) if isinstance(blob, str) else None))
    return tuple(found)


def _from_visual_set(record: Mapping[str, object], where: str) -> Manifest:
    """An earlier visual set read as a manifest: its records built from its files, its hand-written tier dropped."""
    try:
        name, title = str(record["name"]), str(record["title"])
        licence, link = str(record["licence"]), str(record["licence_url"])
        custom = (CustomLicence(licence, f"{licence}, as the earlier visual set named it", link),) \
            if licence.startswith("LicenseRef-") else ()
        refs: list[AssetRef] = []
        files: list[ManifestFile] = []
        for each in cast(list[Mapping[str, Any]], record["files"]):
            stem = str(each["path"]).rsplit("/", 1)[-1]
            ref = AssetRef(f"{name}.{stem}", str(each["sha256"]), int(each["size"]), licence, (str(each["location"]),),
                           credit=str(record["credit"]), role=cast(Any, str(record["role"])), kind="authored",
                           licences=custom, unlisted=True)
            refs.append(ref)
            files.append(ManifestFile(str(each["path"]), ref.name, str(each["git_blob"])))
        return Manifest(name, title, str(record["credit"]), str(record["notice"]), tuple(refs), tuple(files),
                        MappingProxyType(dict(cast(dict[str, str], record["upstream"]))),
                        tuple(map(str, cast(list[object], record["archives"]))), str(record["archive_checked"]), where)
    except (KeyError, TypeError, ValueError, BuildRefused) as error:
        raise _refused(where, f"it is not a {VISUAL_SET_FORMAT} record ({error})") from None


def read_manifest(path: Path | Traversable, where: str | None = None) -> Manifest:
    """The manifest in the file ``path``; refused (``asset.manifest``) where it cannot be read as one."""
    shown = where or str(path)
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _refused(shown, f"cannot be read as JSON ({error})") from None
    return read_manifest_data(data, shown)


def is_manifest_data(data: object) -> bool:
    """Whether a JSON value says it is a manifest (of either form): what a package's folder of records holds beside
    plain lists of records."""
    return isinstance(data, dict) and cast(dict[str, object], data).get("format") in (
        MANIFEST_FORMAT, SECOND_FORMAT, FIRST_FORMAT, VISUAL_SET_FORMAT)


def beside(description: Path, chosen: str | None = None) -> Path | None:
    """The manifest beside a description file: ``chosen`` (a name next to it) where one is given, else
    ``<description stem>.visuals.json``, else ``visuals.json``; ``None`` where none is there."""
    folder = description.parent
    names = [chosen] if chosen is not None else [description.stem + BESIDE_SUFFIX, BESIDE_NAME]
    return next((folder / name for name in names if (folder / name).is_file()), None)


__all__ = ["BESIDE_NAME", "BESIDE_SUFFIX", "FIRST_FORMAT", "MANIFEST_FORMAT", "SECOND_FORMAT", "VISUAL_SET_FORMAT",
           "Manifest", "ManifestFile",
           "beside", "is_manifest_data", "mesh_path", "read_manifest", "read_manifest_data"]
