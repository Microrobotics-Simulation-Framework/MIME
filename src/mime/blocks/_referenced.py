"""The referenced tier: a block whose licence lets no copy of its content into its user's document.

**Saving.** The document records the use (``blocks``) and the user's own declarations, overrides among them, never the
block's (:func:`referenced_keys`); the lock's ``assets`` holds the container's record whole (``<id>-<version>.mimeblock``,
its sha256 and size, the block's licence, ``doi:<DOI>/<file>`` where it was pulled by DOI, no location otherwise). A block
used from its folder is packed into the asset cache at the save (with no case: its cases are not its content), and is
recorded with no location until it is published. Its own descriptions would be written into the user's folder by the save,
so a referenced block that names one is refused (``block.fragment``), naming the route: an asset with a location.

**Loading** (:func:`referenced_declarations`). One function finds the container, :func:`container_root`, by either route:
in a block folder enclosing the saved folder (``<block>/cases/<case>/``, a case of the block, matched by its content
hash: a block's published container holds its cases, so its sha256 can never be the one their locks recorded), else by the
lock's record (:func:`mime.assets.locate`: a file not cached is ``asset.missing``, one with no location
``asset.by-request``). Either way its content hash is held to the document's (``block.changed``) and, for a container
found by the lock, its own licence to the lock record's (``block.changed``: a lock edited to say another licence); then
**the gate runs on the block's own licence** (:func:`require_block_licence`: one with conditions is asked about once, or
refused where nobody can be asked), before any declaration is read from it. The lock's word on the licence is never what
is gated. It is expanded with the bindings and names the document records, the keys declared over it dropped, and the
document's own declarations follow. The input hash catches an expansion that comes out otherwise. What a referenced block cannot hide is said where
it is used (:data:`LOCK_HOLDS_VALUES`): the values its fragment sets reach the user's saved lock."""

from __future__ import annotations

import tempfile
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from mime._terminal import plain
from mime.assets import AssetRef
from mime.blocks._definition import Definition, definition_of, is_block_folder
from mime.blocks._fragment import Expanded, expand, read_fragment
from mime.build import BuildRefused, Declaration, Refusal, RefusalClass
from mime.saved._schema import BlockUseDoc, ExperimentDocument

from mime.saved._card import LOCK_HOLDS_VALUES


def kept_container(definition: Definition, source: Any, accept_licences: Collection[str] = ()) -> AssetRef:
    """The record of the container a referenced use is read from, its bytes kept in the asset cache: the container it
    came in (with its record's location where it was pulled by DOI), or the block folder packed with no case. Its first
    use is gated, as every load of it will be: a licence with conditions is asked about once, or accepted by name."""
    from mime.assets._gates import require_accepted
    from mime.blocks._container import BLOCK_SUFFIX, container_record, pack_block, stored

    container = cast(Path | None, source.container)
    if container is not None:
        locations = [f"doi:{source.doi}/{source.named}"] if source.doi is not None else []
        ref = container_record(definition, container, locations)
        stored(ref, container)
    else:
        with tempfile.TemporaryDirectory() as scratch:
            packed = pack_block(definition.folder, out=Path(scratch) / f"packed{BLOCK_SUFFIX}", cases=False)
            ref = container_record(definition, packed)
            stored(ref, packed)
    require_accepted([ref], accept_licences=accept_licences, accept_with="exp.use(<block>, ...)")
    return ref


def _files_named(value: object) -> list[str]:
    if isinstance(value, dict):
        held = cast(dict[str, object], value)
        if isinstance(held.get("file"), str) and isinstance(held.get("sha256"), str):
            return [cast(str, held["file"])]
        return [each for item in held.values() for each in _files_named(item)]
    if isinstance(value, list):
        return [each for item in cast(list[object], value) for each in _files_named(item)]
    return []


def own_file_refusals(definition: Definition, expanded: Expanded) -> list[Refusal]:
    """``block.fragment`` for each description of the block's own (named by a setting, with no third party's terms) a
    referenced block's fragment names: a save writes it into the user's folder, under the user's licence."""
    terms = cast(Mapping[str, object], expanded.sections.get("terms", {}))
    sections = {key: value for key, value in expanded.sections.items() if key != "terms"}
    own = sorted({path for path in _files_named(sections) if path not in terms})
    return [Refusal("block.fragment", RefusalClass.USER,
                    f"{definition.words} is used by reference and its fragment names its own file {plain(path)}, "
                    f"which a save would write into the user's folder under the user's licence",
                    "make the file an asset the block names, `[[assets]]` with its locations, read by the entry by "
                    "its name, or give it a third party's terms in the fragment (`terms:`)")
            for path in own]


def referenced_keys(blocks: Mapping[str, BlockUseDoc] | None) -> set[tuple[str, str]]:
    """What the uses by reference of ``blocks`` provide and the experiment has not declared over, by section and name:
    what the document leaves out. A third party's file's terms stay in, since the file they judge is in the folder."""
    found: set[tuple[str, str]] = set()
    for use in (blocks or {}).values():
        if use["tier"] != "referenced":
            continue
        overridden = set(use["overrides"])
        found |= {(section, name) for section, held in cast(Mapping[str, list[str]], use["names"]).items()
                  if section != "terms" for name in held if name not in overridden}
    return found


def left_out(blocks: Mapping[str, BlockUseDoc] | None, declarations: Sequence[Declaration]) -> list[Declaration]:
    """``declarations`` less those the uses by reference of ``blocks`` provide: what a document holds."""
    from mime.blocks._use import key_of

    excluded = referenced_keys(blocks)
    return [each for each in declarations if key_of(each) not in excluded] if excluded else list(declarations)


def own_record(definition: Definition) -> AssetRef:
    """The record the gate reads a block's own licence from: the block's id, version, licence and custom licences as its
    definition states them (which its content hash covers), named by its content hash. A lock's record of its container is
    never what is gated."""
    from mime.blocks._definition import FRAGMENT_FILE
    from mime.saved._studies import BLOCK_SUFFIX

    size = max(1, (definition.folder / FRAGMENT_FILE).stat().st_size)
    return AssetRef(f"{definition.id}-{definition.version}{BLOCK_SUFFIX}", definition.content_hash(), size,
                    definition.licence, (), licences=tuple(definition.licences.values()))


def require_block_licence(definition: Definition, *, accept_licences: Collection[str] = (),
                          accept_with: str | None = None) -> None:
    """The gate on the block's own licence (the first use of a block's content, as the asset gates guard a file): a
    licence with conditions no acceptance is on record for is asked about once, accepted by name in ``accept_licences``,
    or refused where nobody can be asked (``asset.licence-not-accepted``, naming ``accept_with``)."""
    from mime.assets._gates import require_accepted

    require_accepted([own_record(definition)], accept_licences=accept_licences, accept_with=accept_with)


def _enclosing(folder: Path | None, use: BlockUseDoc) -> Path | None:
    """The block folder that holds the saved folder ``folder`` as one of its cases, where it is the use's block with
    the use's content."""
    if folder is None:
        return None
    candidate = folder.resolve().parent.parent
    if not is_block_folder(candidate):
        return None
    try:
        definition = definition_of(candidate)
    except BuildRefused:
        return None
    if (definition.id, definition.version, definition.content_hash()) != (use["id"], use["version"], use["content"]):
        return None
    return candidate


def _recorded_container(key: str, use: BlockUseDoc, records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    from mime.blocks._container import BLOCK_SUFFIX

    name = f"{use['id']}-{use['version']}{BLOCK_SUFFIX}"
    found = next((record for record in records if record.get("name") == name), None)
    if found is None:
        raise BuildRefused([Refusal(
            "block.changed", RefusalClass.USER,
            f"the document uses the block {plain(key)!r} ({plain(use['id'])} {plain(use['version'])}) by reference, "
            f"and its lock holds no record of its container ({plain(name)})",
            "save the experiment again from its script, `exp.save(folder)`: the save records the container")])
    return found


def licence_problem(recorded: AssetRef, definition: Definition, words: str) -> Refusal | None:
    """``block.changed`` where the licence the lock records for a container is not the one the container itself states
    (its licence expression, or the custom licences it defines): the lock was edited, or is another block's. What the
    block states is what its content hash covers; what the lock says is nobody's but whoever wrote the lock."""
    if recorded.licence == definition.licence and set(recorded.custom) == set(definition.licences):
        return None
    return Refusal(
        "block.changed", RefusalClass.USER,
        f"{words}: the lock records its licence as {plain(recorded.licence)}, and the container states "
        f"{plain(definition.licence)}; the container's own is the one that binds",
        "save the experiment again from its script, `exp.save(folder)`, or restore the lock the experiment was saved with")


def content_problem(definition: Definition, use: BlockUseDoc, words: str) -> Refusal | None:
    """``block.changed`` where the block read is not the one the document was saved with: its id, version or content
    hash."""
    from mime.blocks._container import changed_refusal

    content = definition.content_hash()
    if content != use["content"] or (definition.id, definition.version) != (use["id"], use["version"]):
        return changed_refusal(words, use["content"], content)
    return None


def container_root(key: str, use: BlockUseDoc, records: Sequence[Mapping[str, object]], folder: Path | None,
                   accept_licences: Collection[str] = ()) -> tuple[Path, Definition]:
    """The folder of the block a referenced use is read from, and its definition: found by either route (an enclosing
    block folder, else the lock's record), held to the document's id, version and content hash (and, by the lock, to the
    licence the container states), and **gated on the block's own licence before anything is read from it**. The one
    place a referenced container's bytes are admitted; refused as the module says."""
    from mime.assets._cache import locate
    from mime.blocks._container import cached_folder

    words = f"the block {plain(use['id'])} {plain(use['version'])} (the use {plain(key)!r})"
    root = _enclosing(folder, use)
    recorded: AssetRef | None = None
    if root is None:
        recorded = AssetRef.from_record(_recorded_container(key, use, records))
        root = cached_folder(locate(recorded))
    definition = definition_of(root)
    problem = content_problem(definition, use, words)
    if problem is None and recorded is not None:
        problem = licence_problem(recorded, definition, words)
    if problem is not None:
        raise BuildRefused([problem])
    require_block_licence(definition, accept_licences=accept_licences,
                          accept_with=f"mime verify {plain(str(folder)) if folder is not None else '<the saved folder>'}")
    return root, definition


def referenced_declarations(document: ExperimentDocument, records: Sequence[Mapping[str, object]],
                            folder: Path | None, accept_licences: Collection[str] = ()) -> list[Declaration]:
    """The declarations of each block ``document`` uses by reference, read from its container (the module says
    where it is found), held to the document's content hash and names, less the keys declared over it."""
    from mime.blocks._use import key_of

    found: list[Declaration] = []
    refusals: list[Refusal] = []
    for key, use in sorted(document.get("blocks", {}).items()):
        if use["tier"] != "referenced":
            continue
        try:
            root, definition = container_root(key, use, records, folder, accept_licences)
        except BuildRefused as refused:
            refusals += refused.refusals
            continue
        words = f"the block {plain(use['id'])} {plain(use['version'])} (the use {plain(key)!r})"
        fragment = read_fragment(root, definition.sockets)
        expanded = next((held for prefix in ("", f"{key}_")
                         if (held := expand(fragment, use["sockets"], prefix)).names == use["names"]), None)
        if expanded is None:
            refusals.append(Refusal("block.changed", RefusalClass.USER,
                                    f"{words} expands to other names than the document records",
                                    "save the experiment again from its script, `exp.save(folder)`"))
            continue
        overridden = set(use["overrides"])
        found += [each for each in expanded.declarations if (key_of(each) or ("", ""))[1] not in overridden]
    if refusals:
        raise BuildRefused(refusals)
    return found


def container_records(blocks: Mapping[str, BlockUseDoc] | None, assets: Sequence[AssetRef]) -> list[AssetRef]:
    """The records of the containers the uses by reference of ``blocks`` are read from, among ``assets`` (the records
    the experiment's blocks bring): what the lock's ``assets`` holds beside the entries' own."""
    from mime.blocks._container import BLOCK_SUFFIX

    names = {f"{use['id']}-{use['version']}{BLOCK_SUFFIX}" for use in (blocks or {}).values()
             if use["tier"] == "referenced"}
    return [ref for ref in assets if ref.name in names]


__all__ = ["LOCK_HOLDS_VALUES", "container_records", "container_root", "content_problem", "kept_container",
           "left_out", "licence_problem", "own_file_refusals", "own_record", "referenced_declarations", "referenced_keys",
           "require_block_licence"]
