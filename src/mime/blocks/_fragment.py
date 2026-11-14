"""A block's fragment, ``block.mime.yaml`` (``mime.fragment/1``): part of an experiment, written in the saved document's
own vocabulary and read with its safe loader.

It holds the document's sections of declarations (``things``, ``interactions``, ``observations``, ``controller``,
``declared``, ``ranges``, ``sources``, ``materials``, ``terms``), each in the document's own closed shapes, so that what
the saved form gains a fragment holds with no change here. The host's settings (``run``, ``scene``, ``gravity``,
``profile``, ``conditions``, ``based_on``, ``blocks``) and its licence are never a fragment's.

**A socket** stands where the user's thing goes: ``{socket: <name>}`` or, for a path below it, ``{socket: <name>, path:
<rest>}``, and only where a binding or a placement is written (a law's or an observation's ``binds``, a medium's
``fills``, the controller's ``reads`` and ``writes``). It is read first: each socket is replaced by its own name, so that
the fragment's names are local (an id is written as a document writes it, with the socket's name: ``labx.nudge:subject``)
and a socket anywhere else is refused by the shape check with its path. A bare name equal to a socket's is refused, so a
socket is always written as one. A socket is never a key: the keys of ``declared``, ``ranges`` and ``sources`` name the
block's own things and laws, and the user's thing is the user's to declare, so a socket's name heading one is refused
(``block.socket``), and a key that is not text (a list or a mapping) is refused (``block.fragment``).

**Expanding** a fragment for a use renames it: each socket to the name of the thing it is bound to, each of the block's
own things to ``<prefix><local>`` (no prefix unless the use is named), each id to the one its renamed bindings give,
and each path by its first part (a thing's name or an id). The renamed fragment is then read back into declarations by
the load's own reader (:func:`mime.saved._reconstruct.declarations_from`), its records and files read from below the
block's folder, so that a block's declarations are exactly what a document holding them would load as."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml
from mime._terminal import plain
from mime.blocks._definition import FRAGMENT_FILE
from mime.build import BuildRefused, Declaration, Refusal, RefusalClass
from mime.blocks._schema import FRAGMENT_SCHEMA, FragmentDoc
from mime.saved._schema import DOCUMENT_SCHEMA, BlockNamesDoc, ExperimentDocument, ThingDoc, shape_problems
from mime.saved._versions import newer_form_refusal
from mime.things import WORLD

FRAGMENT_FAMILY = "fragment"
#: The document's keys that are the host's, never a fragment's.
HOST_KEYS = ("mime", "experiment", "based_on", "blocks", "profile", "run", "scene", "gravity", "conditions", "licence",
             "licences", "permissions")
#: The keys a socket is written with.
SOCKET_KEYS = ("socket", "path")


def _refusal(path: Path, message: str, action: str, code: str = "block.fragment") -> Refusal:
    return Refusal(code, RefusalClass.USER, f"{path.name}: {message}", action)


@dataclass(frozen=True)
class Fragment:
    """A fragment as read, each socket written as its own name, and where it was read from (its block's folder)."""

    doc: FragmentDoc
    folder: Path

    @property
    def things(self) -> Mapping[str, ThingDoc]:
        return self.doc.get("things", {})


def _table(value: object) -> dict[str, object] | None:
    return cast(dict[str, object], value) if isinstance(value, dict) else None


@dataclass
class _Sockets:
    """The sockets a fragment may name, those it does, and what is wrong with how it names them."""

    declared: Mapping[str, str]
    path: Path
    used: set[str] = field(default_factory=set[str])
    refusals: list[Refusal] = field(default_factory=list[Refusal])

    def written(self, value: object, where: str) -> object:
        """``value`` with its socket written as the socket's own name, and its problem gathered."""
        original = value
        held = _table(value)
        if held is not None and "socket" in held:
            socket, below = held.get("socket"), held.get("path")
            if not set(held) <= set(SOCKET_KEYS) or not isinstance(socket, str) or not (
                    below is None or isinstance(below, str) and below):
                self.refusals.append(_refusal(
                    self.path, f"{where} is not a socket as a fragment writes one", "write `{socket: <name>}`, or "
                    "`{socket: <name>, path: <rest>}` for a path below the thing bound to it", "block.socket"))
                return original
            if socket not in self.declared:
                self.refusals.append(_refusal(
                    self.path, f"{where} names the socket {plain(repr(socket))}, which [sockets] does not declare",
                    f"declare it in mime.toml, `{plain(socket)} = {{ kind = \"body\" }}` under [sockets], or name one "
                    f"it declares: {', '.join(sorted(self.declared)) or 'none'}", "block.socket"))
                return original
            self.used.add(socket)
            return socket if below is None else f"{socket}.{below}"
        if isinstance(value, str) and value.split(".", 1)[0] in self.declared:
            head = value.split(".", 1)[0]
            rest = value.split(".", 1)[1] if "." in value else None
            line = f"{{socket: {head}}}" if rest is None else f"{{socket: {head}, path: {rest}}}"
            self.refusals.append(_refusal(
                self.path, f"{where} is {plain(repr(value))}, the socket {plain(head)} written as a bare name",
                f"write it `{plain(line)}`: a socket is always written as one, so that a misspelt thing is never taken "
                f"for it", "block.socket"))
        return original

    def unused_refusals(self, things: Mapping[str, object]) -> list[Refusal]:
        """``block.socket`` for each socket that is also a thing's name, and each declared socket never used."""
        refusals = [_refusal(self.path, f"the socket {plain(name)} is also the name of a thing of the block",
                             "rename one: a name is the block's own thing or a socket, never both", "block.socket")
                    for name in self.declared if name in things]
        return refusals + [_refusal(self.path, f"never uses the socket {plain(name)} that mime.toml declares",
                                    f"use it where the user's thing is bound (`{{socket: {plain(name)}}}`), or remove it "
                                    f"from [sockets]", "block.socket")
                           for name in sorted(set(self.declared) - self.used)]


def _with_sockets(data: dict[str, Any], sockets: _Sockets) -> dict[str, Any]:
    """``data`` with each socket written as its own name, at the places a binding or a placement is written."""
    held = copy.deepcopy(data)
    for section in ("interactions", "observations"):
        entries: object = held.get(section)
        for index, entry in enumerate(cast(list[object], entries) if isinstance(entries, list) else []):
            binds = _table(entry_binds) if (entry_binds := (_table(entry) or {}).get("binds")) is not None else None
            for role, value in list((binds or {}).items()):
                cast(dict[str, object], binds)[role] = sockets.written(value, f"{section}[{index}].binds.{role}")
    for name, thing in (_table(held.get("things")) or {}).items():
        table = _table(thing)
        if table is not None and "fills" in table:
            table["fills"] = sockets.written(table["fills"], f"things.{name}.fills")
    controller = _table(held.get("controller"))
    for kind in ("reads", "writes"):
        paths = _table(controller.get(kind)) if controller is not None else None
        for label, value in list((paths or {}).items()):
            cast(dict[str, object], paths)[label] = sockets.written(value, f"controller.{kind}.{label}")
    return held


def schema_refusal(path: Path, schema: object) -> Refusal | None:
    """``load.newer-form`` for a later fragment form, ``block.format`` for any other that is not this MIME's."""
    if (newer := newer_form_refusal(schema, FRAGMENT_FAMILY, (FRAGMENT_SCHEMA,), path.name)) is not None:
        return newer
    if schema != FRAGMENT_SCHEMA:
        return _refusal(path, f"states schema {plain(repr(schema))}, and this MIME reads {FRAGMENT_SCHEMA}",
                        f"write `schema: {FRAGMENT_SCHEMA}` first", "block.format")
    return None


def shape_refusals(path: Path, held: Mapping[str, object], written: Mapping[str, object]) -> list[Refusal]:
    """``block.fragment`` for each of the host's keys the fragment holds (``held``, as read), each way its sections
    (``written``, sockets as their names) differ from the document's shapes, and each thing not written at its own
    place in the world."""
    refusals = [_refusal(path, f"holds {key}, which is the experiment's that uses the block, never a block's",
                         f"remove {key}: the host declares its run, scene, profile, conditions and licence")
                for key in HOST_KEYS if key in held]
    refusals += [_refusal(path, plain(problem), "write it as the saved document writes it (`mime show <folder>` reads "
                                                "one; a saved experiment's experiment.mime.yaml is an example)")
                 for problem in shape_problems({key: value for key, value in written.items() if key not in HOST_KEYS},
                                               FragmentDoc)]
    refusals += [_refusal(path, f"things.{plain(name)}.prim is {plain(repr(table.get('prim')))}, and a block's thing "
                                f"is written at {WORLD}/{plain(name)}: the scene is the experiment's that uses it",
                          f"write `prim: {WORLD}/{plain(name)}`")
                 for name, thing in (_table(written.get("things")) or {}).items()
                 if (table := _table(thing)) is not None and table.get("prim") != f"{WORLD}/{name}"]
    return refusals


#: The sections whose keys name the block's own things and laws, where a socket is never written.
KEYED_SECTIONS = ("declared", "ranges", "sources")


def _key_refusals(path: Path, text: str) -> list[Refusal]:
    """``block.fragment`` for each mapping key of ``text`` that is not text (a list or a mapping, which YAML allows and
    a document never writes), at its path and line; ``block.socket`` where it is a socket written as a key of a section
    that names the block's own things."""
    found: list[Refusal] = []

    def walk(node: yaml.Node, where: str) -> None:
        if isinstance(node, yaml.MappingNode):
            for key_node, value_node in cast(list[tuple[yaml.Node, yaml.Node]], node.value):
                if not isinstance(key_node, yaml.ScalarNode):
                    line = key_node.start_mark.line + 1
                    shown = f"{where}[a key that is a {'mapping' if isinstance(key_node, yaml.MappingNode) else 'list'}]"
                    socket = (isinstance(key_node, yaml.MappingNode) and where in KEYED_SECTIONS and any(
                        isinstance(each, yaml.ScalarNode) and each.value == "socket"
                        for each, _ in cast(list[tuple[yaml.Node, yaml.Node]], key_node.value)))
                    found.append(_refusal(
                        path, f"{shown} (line {line}) is a socket written as a key: a socket stands only where a "
                              f"binding or a placement is written, and the user's thing is the user's to declare"
                        if socket else f"{shown} (line {line}) is not text: a key is a name, as a document writes one",
                        f"remove it from {where}, or write a name of the block's own as its key" if socket else
                        "write each key as a name", "block.socket" if socket else "block.fragment"))
                    continue
                walk(value_node, f"{where}.{plain(key_node.value)}" if where else plain(key_node.value))
        elif isinstance(node, yaml.SequenceNode):
            for index, each in enumerate(cast(list[yaml.Node], node.value)):
                walk(each, f"{where}[{index}]")

    root = cast("yaml.Node | None", cast(Any, yaml).compose(text, Loader=yaml.SafeLoader))
    if root is not None:
        walk(root, "")
    return found


def socket_key_refusals(path: Path, held: Mapping[str, object], sockets: Mapping[str, str]) -> list[Refusal]:
    """``block.socket`` for each key of ``declared``, ``ranges`` and ``sources`` whose first part is a socket's name
    (not an id of the fragment's own, whose first part is only its law's name): sockets appear only where a binding
    or a placement is written."""
    listed = [held.get(section) for section in ("interactions", "observations")]
    ids = [str(entry["id"]) for entries in listed if isinstance(entries, list)
           for each in cast(list[object], entries) if (entry := _table(each)) is not None and "id" in entry]
    refusals: list[Refusal] = []
    for section in KEYED_SECTIONS:
        for key in cast(Mapping[object, object], _table(held.get(section)) or {}):
            head = str(key).split(".", 1)[0]
            if head in sockets and not any(key == each or str(key).startswith(f"{each}.") for each in ids):
                refusals.append(_refusal(
                    path, f"{section}.{plain(str(key))} names the socket {plain(head)} in a key: a socket stands only where a "
                          f"binding or a placement is written (`binds`, `fills`, the controller's `reads` and "
                          f"`writes`), and the thing bound to it is the user's own",
                    f"remove it: the user declares their own thing's values, and a block declares its own "
                    f"things'", "block.socket"))
    return refusals


def read_fragment(folder: Path, sockets: Mapping[str, str]) -> Fragment:
    """The fragment of the block folder ``folder``, whose ``mime.toml`` declares ``sockets`` (each name and its kind),
    checked: refused with every reason at once (``block.fragment``, ``block.socket``, ``block.format``, or
    ``load.newer-form`` for a later form)."""
    from mime.saved._document import parse_yaml

    path = folder / FRAGMENT_FILE
    try:
        data = parse_yaml(path.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        raise BuildRefused([_refusal(path, f"is missing from {folder}", f"write the block's fragment at {path}",
                                     "block.format")]) from None
    except TypeError:
        text = path.read_bytes().decode("utf-8")
        raise BuildRefused(_key_refusals(path, text) or [_refusal(
            path, "does not read as YAML (a mapping key that is not text)", "write each key as a name")]) from None
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise BuildRefused([_refusal(path, f"does not read as YAML ({plain(type(error).__name__)}: "
                                           f"{plain(str(error))[:200]})", "correct it by hand")]) from None
    if not isinstance(data, dict):
        raise BuildRefused([_refusal(path, "is not a mapping of sections", f"start it `schema: {FRAGMENT_SCHEMA}`",
                                     "block.format")])
    held = cast(dict[str, Any], data)
    if (refused := schema_refusal(path, held.get("schema"))) is not None:
        raise BuildRefused([refused])
    sockets_read = _Sockets(sockets, path)
    written = _with_sockets(held, sockets_read)
    refusals = [*shape_refusals(path, held, written), *sockets_read.refusals, *socket_key_refusals(path, held, sockets),
                *sockets_read.unused_refusals(_table(written.get("things")) or {})]
    if refusals:
        raise BuildRefused(refusals)
    return Fragment(cast(FragmentDoc, written), folder)


# --- expanding ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Renaming:
    """How a use names a fragment's names: each socket's bound thing, each of the block's things, each id."""

    names: Mapping[str, str]
    ids: Mapping[str, str]
    own: Mapping[str, str]

    def path(self, written: str, *, sockets: bool = True) -> str:
        """``written`` (a thing's name, an id, or a path below either) as the use names it: a socket's name by the thing
        bound to it only where ``sockets`` (a binding or a placement), never in a key of ``declared``, ``ranges`` or
        ``sources``."""
        for old in sorted(self.ids, key=len, reverse=True):
            if written == old or written.startswith(f"{old}."):
                return self.ids[old] + written[len(old):]
        head, dot, rest = written.partition(".")
        names = self.names if sockets else self.own
        return names[head] + dot + rest if head in names else written


def entry_id_of(kind: str, entry: Mapping[str, Any], binds: Mapping[str, str]) -> str | None:
    """The id of an interaction or an observation with ``binds``, by its registered class; ``None`` where none is
    registered (the load's reader refuses it then)."""
    from mime.build import entry_id
    from mime.registry import EntryKind, lookup

    found = lookup(EntryKind(kind), str(entry.get("name")), str(entry.get("version")))
    return entry_id(cast(Any, found.cls), binds) if found is not None else None


def renaming(fragment: Fragment, bound: Mapping[str, str], prefix: str) -> Renaming:
    """The use's names of ``fragment``'s: each socket to ``bound``'s thing, each of the block's things to
    ``<prefix><local>``, and each id to the one its renamed bindings give."""
    own = {name: f"{prefix}{name}" for name in fragment.things}
    names = {**own, **bound}
    ids: dict[str, str] = {}
    partial = Renaming(names, {}, own)
    for section, kind in (("interactions", "interaction"), ("observations", "observation")):
        for entry in cast(Sequence[Mapping[str, Any]], fragment.doc.get(section, [])):
            binds = {role: partial.path(value) for role, value in cast(Mapping[str, str], entry["binds"]).items()}
            new = entry_id_of(kind, entry, binds)
            if new is not None:
                ids[str(entry["id"])] = new
    return Renaming(names, ids, own)


def _renamed_setting(value: object, renamed: Renaming) -> object:
    """A setting with each text that names one of the block's things or ids as the use names it (a quantity, an asset,
    a file and a group's name are left as they are)."""
    if isinstance(value, str):
        return renamed.path(value, sockets=False)
    if isinstance(value, list):
        return [_renamed_setting(each, renamed) for each in cast(list[object], value)]
    held = _table(value)
    settings = _table(held.get("settings")) if held is not None and "group" in held else None
    if held is not None and settings is not None:
        return {**held, "settings": {key: _renamed_setting(each, renamed) for key, each in settings.items()}}
    return value


def _renamed_entry(entry: Mapping[str, Any], renamed: Renaming) -> dict[str, Any]:
    held = dict(entry)
    if "settings" in held:
        held["settings"] = {key: _renamed_setting(value, renamed) for key, value in held["settings"].items()}
    return held


def _renamed_thing(thing: Mapping[str, Any], name: str, renamed: Renaming) -> dict[str, Any]:
    held = dict(thing)
    held["prim"] = f"{WORLD}/{name}"
    if "fills" in held:
        held["fills"] = renamed.path(held["fills"])
    for key in ("dynamics", "fields", "commands"):
        if key in held:
            held[key] = {role: _renamed_entry(entry, renamed) for role, entry in held[key].items()}
    if "geometry" in held:
        held["geometry"] = _renamed_entry(held["geometry"], renamed)
    if "parts" in held:
        held["parts"] = {key: {**part, "solid": _renamed_entry(part["solid"], renamed)}
                         for key, part in held["parts"].items()}
    if "chain" in held:
        held["chain"] = [_renamed_entry(segment, renamed) for segment in held["chain"]]
    return held


def renamed_doc(fragment: Fragment, renamed: Renaming) -> dict[str, Any]:
    """The fragment's sections as the use names them."""
    doc = fragment.doc
    found: dict[str, Any] = {
        "things": {renamed.own[name]: _renamed_thing(thing, renamed.own[name], renamed)
                   for name, thing in fragment.things.items()}}
    for section in ("interactions", "observations"):
        found[section] = [
            {**_renamed_entry(entry, renamed), "id": renamed.ids.get(entry["id"], entry["id"]),
             "binds": {role: renamed.path(value) for role, value in entry["binds"].items()}}
            for entry in cast(Sequence[Mapping[str, Any]], doc.get(section, []))]
    controller = doc.get("controller")
    found["controller"] = None if controller is None else {
        **_renamed_entry(controller, renamed),
        "reads": {label: renamed.path(value) for label, value in controller["reads"].items()},
        "writes": {label: renamed.path(value) for label, value in controller["writes"].items()}}
    for section in ("declared", "ranges", "sources"):
        held: Mapping[str, Any] = cast(Mapping[str, Any], cast(Mapping[str, Any], doc).get(section, {}))
        found[section] = {renamed.path(path, sockets=False): value for path, value in held.items()}
    found["materials"] = dict(cast(Mapping[str, Any], doc.get("materials", {})))
    found["terms"] = dict(cast(Mapping[str, Any], doc.get("terms", {})))
    return found


def as_document(sections: Mapping[str, Any]) -> ExperimentDocument:
    """``sections`` as a document of their own, with a host's settings no declaration reads: what the load's reader
    takes."""
    from mime import __version__

    document: dict[str, Any] = {
        "schema": DOCUMENT_SCHEMA, "mime": __version__, "experiment": "block", "profile": "exploratory",
        "run": {"dt": {"value": 1.0, "unit": "s"}, "seed": 0},
        "scene": {"up": [0.0, 0.0, 1.0], "units": "m", "references": []},
        "sources": sections.get("sources", {}), "conditions": {}, "things": sections.get("things", {}),
        "interactions": sections.get("interactions", []), "observations": sections.get("observations", []),
        "declared": sections.get("declared", {}), "materials": sections.get("materials", {}),
        "controller": sections.get("controller")}
    if sections.get("ranges"):
        document["ranges"] = sections["ranges"]
    if sections.get("terms"):
        document["terms"] = sections["terms"]
    return cast(ExperimentDocument, document)


def names_of(sections: Mapping[str, Any]) -> BlockNamesDoc:
    """What a use provides, by the names the document holds them under, each section where it provides any."""
    found: dict[str, list[str]] = {
        "things": sorted(sections.get("things", {})),
        "interactions": sorted(entry["id"] for entry in sections.get("interactions", [])),
        "observations": sorted(entry["id"] for entry in sections.get("observations", [])),
        "controller": [CONTROLLER] if sections.get("controller") is not None else [],
        "declared": sorted(sections.get("declared", {})), "ranges": sorted(sections.get("ranges", {})),
        "sources": sorted(sections.get("sources", {})), "materials": sorted(sections.get("materials", {})),
        "terms": sorted(sections.get("terms", {}))}
    return cast(BlockNamesDoc, {key: held for key, held in found.items() if held})


#: How ``names`` and ``overrides`` name the controller, which has no name of its own.
CONTROLLER = "controller"


@dataclass(frozen=True)
class Expanded:
    """A use's declarations, its sections as the document would hold them, and the names it provides."""

    declarations: tuple[Declaration, ...]
    sections: Mapping[str, Any]
    names: BlockNamesDoc
    #: Each id the fragment writes, by the id the use gives it.
    ids: Mapping[str, str] = field(default_factory=dict[str, str])


def expand(fragment: Fragment, bound: Mapping[str, str], prefix: str) -> Expanded:
    """``fragment``'s declarations for a use that binds each socket to ``bound``'s thing and names the block's things
    ``<prefix><local>``: read back by the load's own reader, which resolves each entry among installed packages and
    installs nothing (raises ``LoadRefused`` with every reason)."""
    from mime.saved._reconstruct import declarations_from, entry_names
    from mime.saved._resolve import import_namespaces

    import_namespaces(entry_names(as_document(cast(Mapping[str, Any], fragment.doc))), ())
    renamed = renaming(fragment, bound, prefix)
    sections = renamed_doc(fragment, renamed)
    _, declarations = declarations_from(as_document(sections), (), fragment.folder)
    return Expanded(tuple(declarations), sections, names_of(sections), dict(renamed.ids))


def entries_of(sections: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Every entry the sections name, as (kind, name, version), each once, in order."""
    found: list[tuple[str, str, str]] = []

    def add(kind: str, entry: Mapping[str, Any]) -> None:
        held = (kind, str(entry.get("name")), str(entry.get("version")))
        if held not in found:
            found.append(held)

    for thing in cast(Mapping[str, Mapping[str, Any]], sections.get("things", {})).values():
        medium = thing.get("kind") == "medium"
        for entry in thing.get("dynamics", {}).values():
            add("dynamics", entry)
        for entry in thing.get("fields", {}).values():
            add("dynamics" if medium else "field provider", entry)
        for entry in thing.get("commands", {}).values():
            add("drive", entry)
        if "geometry" in thing:
            add("geometry", thing["geometry"])
        for part in thing.get("parts", {}).values():
            add("geometry", part["solid"])
        for segment in thing.get("chain", []):
            add("dynamics", segment)
    for section, kind in (("interactions", "interaction"), ("observations", "observation")):
        for entry in sections.get(section, []):
            add(kind, entry)
    if sections.get("controller") is not None:
        add("controller", sections["controller"])
    return found


__all__ = ["CONTROLLER", "FRAGMENT_FAMILY", "HOST_KEYS", "SOCKET_KEYS", "Expanded", "Fragment",
           "Renaming", "as_document", "entries_of", "entry_id_of", "expand", "names_of", "read_fragment",
           "renamed_doc", "renaming"]
