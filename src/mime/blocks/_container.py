"""A block's container, ``.mimeblock`` (``mime.block/1``): one block folder as one file, to publish it and check it on
the other side (:mod:`mime.saved._container`'s third kind).

Its files sit under ``block/``: ``mime.toml``, the definition alone (every table but ``[record]``, written canonically by
pack: keys sorted, comments dropped), ``block.mime.yaml``, ``records/`` and ``descriptions/`` (the content), ``assets/``
(the author's copies of open or conditional asset files, each one an ``[[assets]]`` record names by its sha256) and
``cases/<case>/`` (each a saved folder, carried as a bundle carries one: its strays refused, its terms judged, a file that
may not travel as a reference to its asset). Its closed manifest names the block (id, version, content hash), every file
carried with its sha256 and size, the cases' files that travel as references, the NOTICE's hash, every asset record (the
block's and its cases' locks'), and its own hash.

**Pack** refuses a block that does not read (its definition, its fragment), one with no case (``block.no-case``), a case
saved against other content (``block.case-stale``), an asset file whose bytes no record names or whose record may not
travel, and what a bundle's pack refuses of each case. **Unpack** checks every name against the layout, every member and
hash, then reads the folder written: its content hash recomputed from what it holds must be the manifest's
(``block.changed``), so that a container whose manifest was rewritten to agree with a changed file is refused by its own
content. A block is read from its container by unpacking it, once, into the asset cache (:func:`cached_folder`)."""

from __future__ import annotations

import json
import os
import shutil
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from typing_extensions import ReadOnly, TypedDict

from mime._terminal import plain
from mime.assets import AssetRef, size_text
from mime.blocks._definition import (
    ASSETS,
    CASES,
    DESCRIPTIONS,
    FRAGMENT_FILE,
    MANIFEST_FILE,
    RECORDS,
    Definition,
    content_hash,
    definition_of,
)
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._toml import key_text, value_text
from mime.saved._container import (
    BundleFile,
    ReferencedFile,
    collisions,
    container_hash,
    destination_problem,
    files_of,
    member_kind_problems,
    members_listed,
    notice_of,
    read_manifest,
    refuse,
    sha256_of,
    staging_of,
    unknown_format,
    unreadable,
    write_container,
    write_members,
)
from mime.saved._folder import layout_problem
from mime.saved._schema import RemoteAssetRecord
from mime.saved._studies import BLOCK, BLOCK_CONTAINER, BLOCK_SUFFIX
from mime.saved._terms import INLINE_LIMIT_BYTES

BLOCK_PREFIX = BLOCK.prefix
#: The folder below the asset cache a container is unpacked into, by its sha256, for a use to read.
CACHE_FOLDER = "blocks"


class BlockManifest(TypedDict, closed=True):
    """What a block container says it holds: its format, the block's id, version and content hash, the files carried
    and the cases' files that travel as references (paths below the block folder), the NOTICE's sha256 and size, every
    asset record (the block's and its cases'), and ``container_hash``, the sha256 of the manifest without it."""

    format: ReadOnly[Literal["mime.block/1"]]
    id: ReadOnly[str]
    version: ReadOnly[str]
    content: ReadOnly[str]
    files: ReadOnly[dict[str, BundleFile]]
    referenced: ReadOnly[dict[str, ReferencedFile]]
    notice: ReadOnly[BundleFile]
    assets: ReadOnly[list[RemoteAssetRecord]]
    container_hash: ReadOnly[str]


_KEPT = (f"{MANIFEST_FILE}, {FRAGMENT_FILE}, {RECORDS}/<file>, {DESCRIPTIONS}/<file>, {ASSETS}/<file>, and each "
         f"{CASES}/<case>/ a saved folder")


def case_of(name: str) -> tuple[str, str] | None:
    """The case a path below a block folder is in (``cases/<case>``) and the path below it; ``None`` for a path in no
    case."""
    parts = name.split("/")
    if len(parts) > 2 and parts[0] == CASES:
        return f"{CASES}/{parts[1]}", "/".join(parts[2:])
    return None


def block_layout_problem(name: str) -> str | None:
    """Why ``name``, a path below a block folder as ``/`` writes it, is not where a block folder keeps a file, or
    ``None``."""
    inside = case_of(name)
    if inside is not None:
        return layout_problem(inside[1])
    parts = name.split("/")
    if name in (MANIFEST_FILE, FRAGMENT_FILE) or (len(parts) == 2 and parts[0] in (RECORDS, DESCRIPTIONS, ASSETS)):
        return None
    return f"is not where a block folder keeps a file (it keeps {_KEPT})"




# --- the definition, written canonically ----------------------------------------------------------------------------


def definition_text(held: Mapping[str, object]) -> str:
    """The definition's tables (``held``, ``[record]`` left out) as TOML, canonically: keys sorted, a table of tables
    as one table each, an array of tables as one each, no comment. Read back by ``tomllib`` and held equal."""
    lines = [f"format = {value_text(held.get('format', 'mime.toml/1'))}"]
    for key in sorted(each for each in held if each != "format"):
        value = held[key]
        if isinstance(value, list):
            for entry in cast(list[Mapping[str, object]], value):
                lines += ["", f"[[{key_text(key)}]]", *(f"{key_text(name)} = {value_text(entry[name])}"
                                                         for name in sorted(entry))]
        elif isinstance(value, dict):
            table = cast(Mapping[str, object], value)
            nested = key == "licences"
            if nested:
                for name in sorted(table):
                    entry = cast(Mapping[str, object], table[name])
                    lines += ["", f"[{key_text(key)}.{key_text(name)}]",
                              *(f"{key_text(field)} = {value_text(entry[field])}" for field in sorted(entry))]
            else:
                lines += ["", f"[{key_text(key)}]", *(f"{key_text(name)} = {value_text(table[name])}"
                                                      for name in sorted(table))]
    text = "\n".join([*lines, ""])
    if tomllib.loads(text) != dict(held):
        raise ValueError("the definition does not read back as written")  # an invariant of the writer
    return text


# --- packing ---------------------------------------------------------------------------------------------------------


def _cases(root: Path) -> list[str]:
    folder = root / CASES
    return sorted(f"{CASES}/{each.name}" for each in folder.iterdir() if each.is_dir()) if folder.is_dir() else []


def block_notice(definition: Definition, content: str, cases: Sequence[str]) -> str:
    """The NOTICE a block container carries: what it is, its content's licence, each asset with its tier, its cases."""
    from mime.terms import licence_verdict

    verdict = licence_verdict(definition.licence, definition.licences)
    title = f"NOTICE: block {plain(definition.id)} {plain(definition.version)}"
    lines = [title, "=" * len(title), "",
             f"A block: a part of an experiment, data and no code, content hash {content}. Naming a rights holder here "
             f"implies no endorsement by them.", "", "Its content's licence", "---------------------",
             f"{plain(definition.licence)} ({verdict.words()}): it covers the fragment ({FRAGMENT_FILE}), the "
             f"definition ({MANIFEST_FILE}), and its records and descriptions. A user's saved experiment holds the "
             f"content under it where it is copied in ({'it is' if definition.tier == 'copied' else 'it is not'} "
             f"here: {definition.tier}).", "", "Assets", "------"]
    lines += [f"{plain(ref.name)}: sha256 {ref.sha256}, {size_text(ref.size)}, licence {plain(ref.licence)}, "
              f"{ref.tier}" + (", had by request" if ref.by_request else "") for ref in definition.assets] or ["None."]
    lines += ["", "Cases", "-----", *(plain(each) for each in cases), ""]
    return "\n".join(lines)


def pack_block(folder: str | os.PathLike[str], *, out: str | os.PathLike[str] | None = None,
               cases: bool = True) -> Path:
    """Write the block container of the block folder ``folder`` (by default ``<id>-<version>.mimeblock`` beside it) and
    return its path, refused with every reason at once (the module says what). ``cases=False`` packs a block with no
    case: what a save of an experiment using the block's folder by reference keeps in the cache, never what is
    published. The same folder makes the same bytes."""
    from mime.blocks._check import case_refusals
    from mime.blocks._fragment import read_fragment
    from mime.credentials import held_tokens
    from mime.experiment import Experiment
    from mime.saved import _bundle
    from mime.saved._bundle import referenced_members, shipped_asset_problems
    from mime.saved._folder import named_files, read_folder
    from mime.saved._terms import folder_terms

    root = Path(os.path.abspath(folder))
    definition = definition_of(root)
    read_fragment(root, definition.sockets)
    content = definition.content_hash()
    target = Path(out) if out is not None else root.with_name(f"{definition.id}-{definition.version}{BLOCK_SUFFIX}")
    if root.resolve() in (target.resolve(), *target.resolve().parents):
        raise BuildRefused([refuse("bundle.output-inside", RefusalClass.USER,
                                   "the block's container would be written inside the folder it packs",
                                   "write it beside the folder (--out)")])
    refusals = [refusal for _, refusal in case_refusals(definition, content) if cases or refusal.code != "block.no-case"]
    held = [each for each in _cases(root) if cases]
    stored = held_tokens()
    files = [(member, path) for member, path in files_of(root, BLOCK)
             if member != MANIFEST_FILE and (cases or case_of(member) is None)]
    refusals += [refuse("bundle.stray-file", RefusalClass.USER,
                        f"{plain(member)} is not a file a block folder holds: it {problem}",
                        "a block's container carries its definition, fragment, records, descriptions, assets and cases "
                        "and nothing else: remove the file, then `mime pack <folder>` again")
                 for member, _ in files if (problem := block_layout_problem(member)) is not None]
    refusals += [refuse("bundle.name-collision", RefusalClass.USER,
                        f"{plain(one)} and {plain(other)} are one name on a case-insensitive filesystem",
                        "rename one; then `mime pack <folder>`") for one, other in collisions([m for m, _ in files])]
    referenced: dict[str, ReferencedFile] = {}
    records: list[RemoteAssetRecord] = [cast(RemoteAssetRecord, dict(ref.to_record())) for ref in definition.assets]
    for case in held:
        try:
            Experiment.load(root / case)
            form = read_folder(root / case)
        except BuildRefused as refused:
            refusals += [Refusal(each.code, each.category, f"{case}/: {each.message}", each.action)
                         for each in refused.refusals]
            continue
        named = named_files(form)
        inside = [held_case[1] for member, _ in files if (held_case := case_of(member)) is not None
                  and held_case[0] == case]
        refusals += [refuse("bundle.stray-file", RefusalClass.USER,
                            f"{plain(case)}/{plain(below)} is not a file the case's saved form names",
                            "remove it, then `mime pack <folder>` again") for below in inside if below not in named]
        judged = folder_terms(root / case, form.document, form.lock, named, form.version)
        found, problems = referenced_members(judged, form.lock)
        referenced |= {f"{case}/{path}": ref for path, ref in found.items()}
        refusals += [Refusal(each.code, each.category, f"{case}/: {each.message}", each.action)
                     for each in [*problems, *shipped_asset_problems(form.lock)]]
        for record in form.lock.get("assets", []):
            if record not in records:
                records.append(record)
    known = {ref.sha256: ref for ref in definition.assets}
    members: dict[str, bytes] = {}
    for member, path in files:
        if member in referenced:
            continue
        size = path.stat().st_size
        if size > INLINE_LIMIT_BYTES:
            refusals.append(refuse("bundle.file-too-large", RefusalClass.USER,
                                   f"{plain(member)} has {size_text(size)}, over the {size_text(INLINE_LIMIT_BYTES)} a "
                                   f"container carries", "keep it as an asset with a location, `[[assets]]` "
                                                         "locations = [\"https://...\"], and out of the folder"))
            continue
        carried = _bundle.scanned(member, path, stored, refusals)
        if carried is None:
            continue
        if member.startswith(f"{ASSETS}/"):
            ref = known.get(sha256_of(carried))
            if ref is None or ref.tier == "referenced":
                refusals.append(refuse(
                    "bundle.referenced-copy", RefusalClass.USER,
                    f"{plain(member)} is " + ("no file [[assets]] names by its sha256" if ref is None else
                                              f"the asset {plain(ref.name)}'s, whose licence lets no copy travel"),
                    "remove it from the folder: a block carries only open or conditional asset files its [[assets]] "
                    "names; then `mime pack <folder>`"))
                continue
        members[member] = carried
    members[MANIFEST_FILE] = definition_text(definition.held).encode("utf-8")
    notice = block_notice(definition, content, held).encode("utf-8")
    refusals += _bundle.scan_text("the NOTICE", notice, stored)
    if refusals:
        raise BuildRefused(refusals)
    manifest: dict[str, object] = {
        "format": BLOCK_CONTAINER, "id": definition.id, "version": definition.version, "content": content,
        "files": {member: {"sha256": sha256_of(data), "size": len(data)} for member, data in sorted(members.items())},
        "referenced": dict(sorted(referenced.items())), "notice": {"sha256": sha256_of(notice), "size": len(notice)},
        "assets": records}
    manifest["container_hash"] = container_hash(manifest, "container_hash")
    text = (json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")
    problems = _bundle.scan_text("the manifest", text, stored)
    if problems:
        raise BuildRefused(problems)
    return write_container(target, text, notice, members, BLOCK)


# --- unpacking -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UnpackedBlock:
    """What an unpacked block container was: its manifest, the folder it was written to and its ``NOTICE``."""

    manifest: BlockManifest
    folder: Path
    notice: str


def changed_refusal(where: str, recorded: str, found: str) -> Refusal:
    return Refusal("block.changed", RefusalClass.USER,
                   f"{where}'s content hash is {found}, and {recorded} was recorded: its content is not the block's it "
                   f"names", "fetch the block again from its source, `mime pull <doi>`, or use the version the "
                                 "experiment was saved with")


def unpack_block(container: str | os.PathLike[str], into: str | os.PathLike[str]) -> UnpackedBlock:
    """Check the block container and write its folder to ``into`` (new, or an empty folder): every member as a bundle's
    is checked, written by this code's own opens into a hidden folder beside ``into``; then the folder's content hash,
    recomputed, held to the manifest's (``block.changed``), and each case read as a saved folder; only then renamed."""
    from mime.saved._folder import read_folder

    refused = destination_problem(into)
    if refused is not None:
        raise refused
    destination = Path(os.path.abspath(into))
    staging = staging_of(destination)
    try:
        with zipfile.ZipFile(container) as archive:
            later = unknown_format(archive, BLOCK)
            if later is not None:
                raise later
            problems = member_kind_problems(archive, BLOCK)
            if problems:
                raise BuildRefused(problems)
            manifest = cast(BlockManifest, read_manifest(archive, BLOCK, lambda held: BlockManifest))
            unlisted = members_listed(archive, manifest["files"], BLOCK)
            if unlisted is not None:
                raise unlisted
            notice, problems = notice_of(archive, manifest["notice"], BLOCK)
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.mkdir()
            problems += write_members(archive, manifest["files"], staging, BLOCK)
            if problems:
                raise BuildRefused(problems)
        definition = definition_of(staging)
        found = content_hash(definition.held, definition.content_files())
        if (definition.id, definition.version) != (manifest["id"], manifest["version"]) or found != manifest["content"]:
            raise BuildRefused([changed_refusal(f"the block {plain(manifest['id'])} {plain(manifest['version'])}",
                                                manifest["content"], found)])
        for case in _cases(staging):
            read_folder(staging / case, absent_referenced=True)
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
    except zipfile.BadZipFile as error:
        raise unreadable(error, BLOCK) from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return UnpackedBlock(manifest, destination, notice)


def cached_folder(container: str | os.PathLike[str]) -> Path:
    """The folder of the block container ``container``, unpacked once into the asset cache by the container's sha256
    (and checked as :func:`unpack_block` checks it), for a use to read."""
    import hashlib

    from mime.assets import cache_dir

    with Path(container).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    folder = cache_dir() / CACHE_FOLDER / digest
    if (folder / MANIFEST_FILE).is_file():
        return folder
    shutil.rmtree(folder, ignore_errors=True)
    return unpack_block(container, folder).folder


def stored(ref: AssetRef, path: Path) -> bool:
    """``path``'s bytes in the asset cache as ``ref``'s, where they are ``ref``'s (its full sha256) and the cache does
    not hold them yet: copied to a temporary name beside the cache's file and renamed onto it, then recorded. Whether
    the cache holds ``ref``'s bytes now."""
    import hashlib
    import tempfile

    from mime.assets._cache import cache_dir, cheap_problem, file_path, write_record

    if cheap_problem(ref) is None:
        return True
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ref.sha256:
            return False
    cache_dir().mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{ref.sha256}.", suffix=".partial", dir=cache_dir())
    os.close(descriptor)
    try:
        shutil.copyfile(path, temporary)
        os.replace(temporary, file_path(ref))
    finally:
        Path(temporary).unlink(missing_ok=True)
    write_record(ref, "copied by MIME from a block's folder or container, its sha256 checked")
    return True


def container_record(definition: Definition, container: Path, locations: Sequence[str] = ()) -> AssetRef:
    """The asset record of a block's container, as a referenced use's lock holds it: ``<id>-<version>.mimeblock``, its
    sha256 and size, the block's licence and its custom licences, and where it may be fetched (none for a container
    a save packed from a folder)."""
    import hashlib

    with container.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return AssetRef(f"{definition.id}-{definition.version}{BLOCK_SUFFIX}", digest, container.stat().st_size,
                    definition.licence, tuple(locations), licences=tuple(definition.licences.values()))


__all__ = ["BLOCK", "BLOCK_CONTAINER", "BLOCK_PREFIX", "BLOCK_SUFFIX", "BlockManifest", "UnpackedBlock",
           "block_layout_problem", "block_notice", "cached_folder", "case_of", "changed_refusal", "container_record",
           "definition_text", "pack_block", "stored", "unpack_block"]
