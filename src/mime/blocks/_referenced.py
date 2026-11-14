"""The referenced tier: a block whose licence lets no copy of its content into its user's document.

**Saving.** The document records the use (``blocks``) and the user's own declarations, overrides among them, never the
block's (:func:`referenced_keys`); the lock's ``assets`` holds the container's record whole (``<id>-<version>.mimeblock``,
its sha256 and size, the block's licence, ``doi:<DOI>/<file>`` where it was pulled by DOI, no location otherwise). A block
used from its folder is packed into the asset cache at the save (with no case: its cases are not its content), and is
recorded with no location until it is published. Its own descriptions would be written into the user's folder by the save,
so a referenced block that names one is refused (``block.fragment``), naming the route: an asset with a location.

**Loading** (:func:`referenced_declarations`). The container is found first in a block folder enclosing the saved folder
(``<block>/cases/<case>/``, a case of the block, matched by its content hash: a block's published container holds its
cases, so its sha256 can never be the one their locks recorded), else by the lock's record through the gate
(:func:`mime.assets.locate`: a licence with conditions is asked about once; a file not cached is ``asset.missing``, one
with no location ``asset.by-request``). Its content hash is held to the document's (``block.changed``); it is expanded
with the bindings and names the document records, the keys declared over it dropped, and the document's own declarations
follow. The input hash catches an expansion that comes out otherwise. What a referenced block cannot hide is said where
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
    from mime.assets import require_accepted
    from mime.blocks._container import container_record, pack_block, stored

    container = cast(Path | None, source.container)
    if container is not None:
        locations = [f"doi:{source.doi}/{source.named}"] if source.doi is not None else []
        ref = container_record(definition, container, locations)
        stored(ref, container)
    else:
        with tempfile.TemporaryDirectory() as scratch:
            packed = pack_block(definition.folder, out=Path(scratch) / "block.mimeblock", cases=False)
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


def _record_folder(use: BlockUseDoc, records: Sequence[Mapping[str, object]]) -> Path:
    from mime.assets import locate
    from mime.blocks._container import BLOCK_SUFFIX, cached_folder

    name = f"{use['id']}-{use['version']}{BLOCK_SUFFIX}"
    found = next((record for record in records if record.get("name") == name), None)
    if found is None:
        raise BuildRefused([Refusal(
            "block.changed", RefusalClass.USER,
            f"the document uses the block {plain(use['id'])} {plain(use['version'])} by reference, and its lock holds "
            f"no record of its container ({plain(name)})",
            "save the experiment again from its script, `exp.save(folder)`: the save records the container")])
    return cached_folder(locate(AssetRef.from_record(found)))


def referenced_declarations(document: ExperimentDocument, records: Sequence[Mapping[str, object]],
                            folder: Path | None) -> list[Declaration]:
    """The declarations of each block ``document`` uses by reference, read from its container (the module says
    where it is found), held to the document's content hash and names, less the keys declared over it."""
    from mime.blocks._container import changed_refusal
    from mime.blocks._use import key_of

    found: list[Declaration] = []
    refusals: list[Refusal] = []
    for key, use in sorted(document.get("blocks", {}).items()):
        if use["tier"] != "referenced":
            continue
        root = _enclosing(folder, use) or _record_folder(use, records)
        definition = definition_of(root)
        content = definition.content_hash()
        words = f"the block {plain(use['id'])} {plain(use['version'])} (the use {plain(key)!r})"
        if content != use["content"] or (definition.id, definition.version) != (use["id"], use["version"]):
            refusals.append(changed_refusal(words, use["content"], content))
            continue
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


__all__ = ["LOCK_HOLDS_VALUES", "container_records", "kept_container", "left_out", "own_file_refusals",
           "referenced_declarations", "referenced_keys"]
