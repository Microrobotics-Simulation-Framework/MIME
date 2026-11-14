"""``exp.use``: a block used in an experiment.

Every refusal at once, in this order: the block's definition and fragment are read and its content hash computed; its
``[requires]`` is held to what is installed (``block.requires``, the environment's to fix, with the pins table and the
line that installs what is missing); its sockets are bound (``block.socket``); its fragment is renamed for the use and
read into declarations by the load's own reader, which resolves each entry among installed packages and installs
nothing; each entry's distribution is held to ``[requires]`` (``block.entry``, the block author's to fix); what it
provides is held to what the experiment already declares (``block.name``). Its tier decides what the saved document
holds: a copied block's declarations themselves, so that a load never needs the block; a referenced block's container
is read by a later MIME, and refused here as ``load.referenced-block``.

The use is recorded in the experiment's ``blocks`` (:class:`~mime.saved._schema.BlockUseDoc`): the block's id, version,
content hash, tier and licence, how its sockets were bound, the names it provides, and the keys the experiment declared
over it since (``overrides``: a declaration replaced or removed)."""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from mime._shell import quoted
from mime._terminal import plain
from mime.assets import AssetRef
from mime.blocks._definition import Definition, definition_of, is_block_folder
from mime.blocks._fragment import CONTROLLER, Expanded, Fragment, entries_of, expand, read_fragment, renaming
from mime.build import BuildRefused, Declaration, Refusal, RefusalClass, ThingDeclaration
from mime.env._check import DistributionRow
from mime.experiment._redeclared import Taken, known_as
from mime.saved._schema import BlockNamesDoc, BlockUseDoc
from mime.things import Handle

if TYPE_CHECKING:
    from mime.experiment._experiment import ObservationHandle
    from mime.things import LawHandle

#: The verdicts of a requirement against what is installed.
MET, NOT_INSTALLED, OUTSIDE = "in its range", "not installed", "outside its range"


@dataclass(frozen=True)
class RequirementRow(DistributionRow):
    """A block's requirement against what is installed, as an environment's pin is drawn: met where what is installed
    is in its range."""

    @property
    @override
    def agrees(self) -> bool:
        return self.verdict == MET
#: How a refusal names what a section of ``names`` holds.
_WORDS = {"things": "the thing", "interactions": "the law", "observations": "the observation", "declared": "the value",
          "sources": "the source of", "ranges": "the range of", "controller": "the", "materials": "the material",
          "terms": "the terms of"}
#: Each key a declaration is known by (``mime.experiment._redeclared.Known.key``), as the section of ``names`` and
#: ``overrides`` that holds it.
SECTIONS = {"thing": "things", "attach": "interactions", "observe": "observations", "value": "declared",
            "source": "sources", "range": "ranges", "controller": "controller"}


class BlockRefused(BuildRefused):
    """A use refused: every reason at once, and the requirements that are not met here, which a terminal draws as a
    table (``pin_rows``, as an environment check's)."""

    headline = "the block is refused"

    def __init__(self, refusals: list[Refusal], pin_rows: tuple[DistributionRow, ...] = ()) -> None:
        self.pin_rows = pin_rows
        super().__init__(refusals)


def requirement_rows(definition: Definition) -> list[RequirementRow]:
    """Each of the block's requirements against what is installed: wanted (its specifiers), found (``None`` where it
    is not installed), and the verdict. Pre-releases are admitted, as MIME's own development releases are."""
    rows: list[RequirementRow] = []
    for name, wanted in definition.requires:
        try:
            found: str | None = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found = None
        if found is None:
            verdict = NOT_INSTALLED
        else:
            try:
                verdict = MET if SpecifierSet(wanted).contains(Version(found), prereleases=True) else OUTSIDE
            except InvalidVersion:
                verdict = OUTSIDE
        rows.append(RequirementRow(name, wanted, found, definition.words, verdict))
    return rows


def requires_install_line(rows: Sequence[DistributionRow]) -> str:
    """The one line that installs every requirement not met, quoted for a shell."""
    return "pip install " + " ".join(quoted(f"{row.name}{row.wanted}") for row in rows)


def requires_refusal(definition: Definition, unmet: Sequence[DistributionRow], what: str) -> Refusal:
    count = len(definition.requires)
    return Refusal(
        "block.requires", RefusalClass.ENVIRONMENT,
        f"{len(unmet)} of {definition.words}'s {count} requirement{'s' if count != 1 else ''} "
        f"{'is' if len(unmet) == 1 else 'are'} not met here ({', '.join(f'{plain(row.name)} {row.verdict}' for row in unmet)}), "
        f"so {what}",
        f"install what it requires, `{requires_install_line(unmet)}` (its author chose these ranges: check them first); MIME "
        f"installs nothing", why="a block's entries come from the plugins its [requires] names, at the versions it "
                                 "was made and checked with; one outside its range is not the block's")


def requires_refused(definition: Definition, what: str) -> BlockRefused | None:
    unmet = [row for row in requirement_rows(definition) if not row.agrees]
    return BlockRefused([requires_refusal(definition, unmet, what)], tuple(unmet)) if unmet else None


def entry_refusals(definition: Definition, sections: Mapping[str, Any]) -> list[Refusal]:
    """``block.entry`` for each entry the fragment names whose distribution is no row of ``[requires]``."""
    from mime.registry import EntryKind, lookup
    from mime.saved._resolve import provenance_of

    listed = {canonicalize_name(name) for name, _ in definition.requires}
    found: list[Refusal] = []
    for kind, name, version in entries_of(sections):
        registered = lookup(EntryKind(kind), name, version)
        if registered is None:
            continue
        provenance = provenance_of(registered)
        if isinstance(provenance, str):
            found.append(Refusal("block.entry", RefusalClass.USER,
                                 f"{definition.words}'s fragment names {kind} {plain(name)} {plain(version)}, which a "
                                 f"load could not find: {plain(provenance)}",
                                 "the block's author registers it from an installed package with a mime.entries entry "
                                 "point, names that package in [requires], and runs `mime check <block folder>`"))
        elif canonicalize_name(provenance.distribution) not in listed:
            found.append(Refusal(
                "block.entry", RefusalClass.USER,
                f"{definition.words}'s fragment names {kind} {plain(name)} {plain(version)}, from the distribution "
                f"{plain(provenance.distribution)}, which its [requires] does not name",
                f"the block's author adds it to [requires] in {definition.path}, `{plain(provenance.distribution)} = "
                f"\">={plain(provenance.version)}\"`: a block names every plugin its entries come from",
                why="a user installs what [requires] names, and nothing else tells them which plugin an entry needs"))
    return found


def expanded_for_check(definition: Definition, fragment: Fragment) -> Expanded:
    """The fragment expanded with each socket as its own name and no prefix: what ``mime check`` reads."""
    return expand(fragment, {name: name for name in definition.sockets}, "")


def _kind_of(handle: Handle) -> str | None:
    from mime.things import ApparatusHandle, BodyHandle, MediumHandle, StructureHandle

    for cls, kind in ((StructureHandle, "structure"), (MediumHandle, "medium"), (BodyHandle, "body"),
                      (ApparatusHandle, "apparatus")):
        if isinstance(handle, cls):
            return kind
    return None


def bound_sockets(definition: Definition, given: Mapping[str, object],
                  things: Mapping[str, str]) -> tuple[dict[str, str], list[Refusal]]:
    """Each socket's bound thing by name, from ``given`` (a handle or a name, by keyword), held to the things the
    experiment declares (``things``: each name and its kind); and ``block.socket`` for each socket unbound, unknown or
    bound to a thing of another kind."""
    bound: dict[str, str] = {}
    refusals: list[Refusal] = []
    words = definition.words
    declared = ", ".join(f"{name} ({kind})" for name, kind in sorted(definition.sockets.items())) or "none"
    for name in sorted(set(given) - set(definition.sockets)):
        refusals.append(Refusal("block.socket", RefusalClass.USER,
                                f"{words}: exp.use binds {plain(name)}=, which is no socket of the block",
                                f"bind its sockets, exp.use(..., <socket>=<a thing>): {declared}"))
    for name, kind in sorted(definition.sockets.items()):
        if name not in given:
            refusals.append(Refusal("block.socket", RefusalClass.USER,
                                    f"{words}: its socket {plain(name)} ({kind}) is not bound",
                                    f"bind it to a {kind} the experiment declares: exp.use(..., {plain(name)}=<its "
                                    f"handle or name>)"))
            continue
        value = given[name]
        if isinstance(value, Handle):
            target, stated = value.name, _kind_of(value)
        elif isinstance(value, str):
            target, stated = value, None
        else:
            refusals.append(Refusal("block.socket", RefusalClass.USER,
                                    f"{words}: {plain(name)}= is {type(value).__name__}, not a handle or a name",
                                    f"pass the handle add_{kind}(...) returned, or the {kind}'s name: "
                                    f"exp.use(..., {plain(name)}=<it>)"))
            continue
        held = things.get(target)
        if held is None:
            refusals.append(Refusal("block.socket", RefusalClass.USER,
                                    f"{words}: its socket {plain(name)} is bound to {plain(repr(target))}, which the "
                                    f"experiment does not declare",
                                    f"declare the {kind} first, exp.add_{kind}(...), then use the block"))
        elif held != kind or (stated is not None and stated != kind):
            refusals.append(Refusal("block.socket", RefusalClass.USER,
                                    f"{words}: its socket {plain(name)} takes a {kind}, and {plain(target)} is a {held}",
                                    f"bind it to a {kind}: exp.use(..., {plain(name)}=<a {kind}>)"))
        else:
            bound[name] = target
    return bound, refusals


def _section_of(key: tuple[str, str]) -> tuple[str, str] | None:
    section = SECTIONS.get(key[0])
    if section is None:
        return None
    return section, CONTROLLER if section == "controller" else key[1]


def provided_keys(names: BlockNamesDoc) -> set[tuple[str, str]]:
    return {(section, name) for section, held in cast(Mapping[str, list[str]], names).items() for name in held}


def held_keys(declarations: Collection[Declaration]) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for declaration in declarations:
        known = known_as(declaration)
        if known is not None and known.key is not None and (section := _section_of(known.key)) is not None:
            found.add(section)
    return found


def name_refusals(definition: Definition, expanded: Expanded, held: Collection[Declaration],
                  key: str, named: bool, socket_ids: Collection[str] = ()) -> list[Refusal]:
    """``block.name`` for each name the use provides that the experiment already declares, or a material or a file's
    terms it holds otherwise. ``socket_ids`` are the ids of the laws and observations the use provides whose bindings
    are all sockets: such an id comes from what the sockets are bound to, so ``name=`` cannot change it."""
    from mime.build import TermsDeclaration
    from mime.saved._document import material_doc, materials_of

    clashes = sorted(held_keys(held) & provided_keys(expanded.names))
    materials = {name: material_doc(record) for name, record in materials_of(list(held)).items()}
    clashes += sorted(("materials", name) for name, doc in expanded.sections.get("materials", {}).items()
                      if name in materials and materials[name] != doc)
    terms = {each.path: each for each in held if isinstance(each, TermsDeclaration)}
    clashes += sorted(("terms", each.path) for each in expanded.declarations
                      if isinstance(each, TermsDeclaration) and each.path in terms and terms[each.path] != each)
    action = ("name this use, exp.use(..., name=\"<name>\"), which names the block's things <name>_<thing>" if not named
              else "name it otherwise, exp.use(..., name=\"<another name>\")")
    why = ", and its id comes from the socket it is bound through, never from the use's name"
    fixed = ("`name=` cannot change that: bind the socket to another thing of yours, or leave the block's law out of "
             "this use and declare the second law yourself")
    return [Refusal("block.name", RefusalClass.USER,
                    f"{definition.words} (the use {plain(key)!r}) provides {_WORDS[section]} {plain(name)!r}, which "
                    f"the experiment already declares" + (why if name in socket_ids else ""),
                    fixed if name in socket_ids else action)
            for section, name in clashes]


def socket_only_ids(fragment: Fragment, bound: Mapping[str, str], prefix: str) -> set[str]:
    """The ids, as this use names them, of the laws and observations whose id does not depend on the use's name: those
    bound to no thing of the block's own, so that a prefix would leave them as they are."""
    other = renaming(fragment, bound, f"{prefix}_other_")
    return {new for old, new in renaming(fragment, bound, prefix).ids.items() if other.ids.get(old) == new}


def use_doc(definition: Definition, content: str, bound: Mapping[str, str], names: BlockNamesDoc,
            doi: str | None = None) -> BlockUseDoc:
    use: dict[str, Any] = {"id": definition.id, "version": definition.version, "content": content,
                           "tier": definition.tier, "licence": definition.licence}
    if doi is not None:
        use["doi"] = doi
    use |= {"sockets": dict(sorted(bound.items())), "names": names, "overrides": []}
    return cast(BlockUseDoc, use)


def with_override(blocks: Mapping[str, BlockUseDoc] | None, key: tuple[str, str]) -> dict[str, BlockUseDoc] | None:
    """``blocks`` with ``key`` (a declaration's, :func:`known_as`) among the overrides of the use that provides it, where
    one does; ``None`` where nothing changes."""
    section = _section_of(key)
    if not blocks or section is None:
        return None
    for name, use in blocks.items():
        provided = cast(Mapping[str, list[str]], use["names"]).get(section[0], [])
        if section[1] in provided and section[1] not in use["overrides"]:
            changed = cast(BlockUseDoc, {**use, "overrides": sorted([*use["overrides"], section[1]])})
            return {**blocks, name: changed}
    return None


def taken_from(blocks: Mapping[str, BlockUseDoc] | None,
               declarations: Collection[Declaration]) -> dict[int, tuple[Declaration, Taken]]:
    """Each of ``declarations`` a use of ``blocks`` provides and the experiment has not declared over, with where it
    was taken from: what an experiment made from data (a loaded one, one based on a saved one) holds of its blocks."""
    found: dict[int, tuple[Declaration, Taken]] = {}
    for name, use in (blocks or {}).items():
        provided = provided_keys(use["names"])
        overridden = set(use["overrides"])
        for declaration in declarations:
            known = known_as(declaration)
            section = _section_of(known.key) if known is not None and known.key is not None else None
            if section is not None and section in provided and section[1] not in overridden:
                found[id(declaration)] = (declaration, taken(use["id"], use["version"], name))
    return found


def taken(block: str, version: str, use: str) -> Taken:
    words = f"block {block} {version}"
    return Taken(f"it replaces the one {words} provides (the use {use!r}), recorded among the use's overrides",
                 f"provided by {words} (the use {use!r})", use)


@dataclass(frozen=True)
class BlockHandle:
    """A block's use, as ``exp.use`` returns it: its name, the block's id and version, its tier, and what it provides.
    An attribute is a thing the block provides, by the block's own name for it (``probe.transducer``), as the handle
    ``add_*`` returns; :meth:`handle` gives a law or an observation by its id in the block's fragment."""

    name: str
    block: str
    version: str
    tier: str
    things: Mapping[str, Handle]
    ids: Mapping[str, str]
    lookup: Callable[[str], Any]

    def __getattr__(self, local: str) -> Handle:
        things = object.__getattribute__(self, "things")
        if local in things:
            return things[local]
        raise AttributeError(f"the block {object.__getattribute__(self, 'block')} provides no thing named {local!r}; "
                             f"it provides {', '.join(sorted(things)) or 'none'}")

    def handle(self, local: str) -> Handle | LawHandle | ObservationHandle:
        """The handle of what the block provides under ``local``, its name in the block's fragment: a thing, or a law's
        or an observation's id as the fragment writes it."""
        if local in self.things:
            return self.things[local]
        return self.lookup(self.ids.get(local, local))


def _folder(block: str | os.PathLike[str]) -> Path:
    """The block folder ``block`` names. This release reads a block from its folder; a block's file and a DOI are
    refused, naming the route."""
    from mime.saved._studies import BLOCK_SUFFIX

    path = Path(os.fspath(block))
    if path.is_dir() and is_block_folder(path):
        return path
    if path.is_dir():
        raise BuildRefused([Refusal("block.format", RefusalClass.USER,
                                    f"{path} is a folder, and not a block's: a block folder holds mime.toml and "
                                    f"block.mime.yaml", "use a block folder, `exp.use(<block folder>, ...)`: your own, "
                                                        "or one its author shared")])
    what = "a block's file" if os.fspath(block).endswith(BLOCK_SUFFIX) else "not a block folder"
    raise BuildRefused([Refusal("block.format", RefusalClass.USER,
                                f"exp.use reads a block from its folder, and {plain(os.fspath(block))!r} is {what}",
                                "use the block's folder, `exp.use(<block folder>, ...)`; a block's file is unpacked by a "
                                "later MIME release (`pip install -U mime-engine`)")])


@dataclass(frozen=True)
class Used:
    """A use, worked out and not yet made: its key (the use's name), what the document records of it, the declarations
    it adds and the asset records it brings, where each declaration was taken from, and whether it is the same use made
    again (a notebook's cell run twice), which is kept once and adds nothing. ``prefix`` is what the block's things'
    names begin with, and ``ids`` each id of the fragment by the use's."""

    key: str
    doc: BlockUseDoc
    declarations: tuple[Declaration, ...]
    assets: tuple[AssetRef, ...]
    taken: Taken
    again: bool
    prefix: str
    ids: Mapping[str, str]

    def handle(self, things: Mapping[str, Handle], lookup: Callable[[str], Any]) -> BlockHandle:
        """The use's handle, from the experiment's handles of its things (``things``, by name) and its lookup by id."""
        provided = set(cast(Mapping[str, list[str]], self.doc["names"]).get("things", []))
        own = {name.removeprefix(self.prefix): held for name, held in things.items() if name in provided}
        return BlockHandle(self.key, self.doc["id"], self.doc["version"], self.doc["tier"], own, dict(self.ids), lookup)


def used(block: str | os.PathLike[str], *, name: str | None, sockets: Mapping[str, object],
         held: Sequence[Declaration], blocks: Mapping[str, BlockUseDoc]) -> Used:
    """The use of ``block`` in an experiment that holds ``held`` and uses ``blocks``, worked out (every refusal at
    once, :mod:`mime.blocks._use`); :meth:`mime.experiment.Experiment.use` makes it."""
    from mime.saved._reconstruct import referenced_block_refusals

    folder = _folder(block)
    definition = definition_of(folder)
    refusals: list[Refusal] = []
    try:
        fragment: Fragment | None = read_fragment(folder, definition.sockets)
    except BuildRefused as refused:
        fragment, refusals = None, list(refused.refusals)
    content = definition.content_hash()
    unmet = [row for row in requirement_rows(definition) if not row.agrees]
    if unmet:
        refusals.append(requires_refusal(definition, unmet, "it is not used"))
    things = {each.thing.name: each.thing.kind.value for each in held if isinstance(each, ThingDeclaration)}
    bound, problems = bound_sockets(definition, sockets, things)
    refusals += problems
    if name is not None and not (isinstance(cast(object, name), str) and name.isidentifier()):
        refusals.append(Refusal("block.name", RefusalClass.USER, f"name={name!r} is not a use's name",
                                "name it as a Python name is written, exp.use(..., name=\"unit_a\")"))
    key = name if name is not None else definition.name
    prefix = f"{name}_" if name is not None else ""
    from_block = taken(definition.id, definition.version, key)
    earlier = blocks.get(key)
    if earlier is not None and (earlier["id"], earlier["version"], earlier["content"], earlier["sockets"]) == (
            definition.id, definition.version, content, dict(sorted(bound.items()))) and not refusals:
        from mime.errors._frames import warn_at_script

        warn_at_script(f"exp.use(...) uses {definition.words} again as {key!r}, the same as before: it is kept once",
                       UserWarning)
        return Used(key, earlier, (), (), from_block, True, prefix, {})
    if earlier is not None:
        refusals.append(Refusal("block.name", RefusalClass.USER,
                                f"the experiment already uses a block as {key!r} ({plain(earlier['id'])} "
                                f"{plain(earlier['version'])})",
                                "name this use, exp.use(..., name=\"<name>\")" if name is None else
                                "name it otherwise, exp.use(..., name=\"<another name>\")"))
    if refusals or fragment is None:
        raise BlockRefused(refusals, tuple(unmet))
    expanded = expand(fragment, bound, prefix)
    refusals += entry_refusals(definition, expanded.sections)
    refusals += name_refusals(definition, expanded, held, key, name is not None,
                              socket_only_ids(fragment, bound, prefix))
    if refusals:
        raise BlockRefused(refusals)
    doc = use_doc(definition, content, bound, expanded.names)
    if definition.tier == "referenced":
        raise BuildRefused(referenced_block_refusals(cast(Any, {"mime": "", "blocks": {key: doc}}), using=True))
    return Used(key, doc, expanded.declarations, definition.assets, from_block, False, prefix, expanded.ids)


__all__ = ["MET", "NOT_INSTALLED", "OUTSIDE", "SECTIONS", "requires_install_line", "BlockHandle", "BlockRefused", "RequirementRow", "Used",
           "bound_sockets",
           "entry_refusals", "expanded_for_check", "held_keys", "name_refusals", "provided_keys",
           "requirement_rows", "requires_refusal", "requires_refused", "socket_only_ids", "taken", "taken_from", "use_doc", "used",
           "with_override"]
