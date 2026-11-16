"""A block's fragment written from a saved experiment: the things named become its sockets.

``fragment_from(<saved folder>, medium="water")`` reads the folder's document and writes ``block.mime.yaml`` from it:
every declaration but those of the things named (the user's, bound through the sockets), each binding and placement
of a named thing written as its socket (``{socket: medium}``, ``{socket: medium, path: lumen}``), and each id as its
socket's name gives it. A law or an observation that binds only named things is the user's, and left out, as is a value,
a range or a source over a named thing; a material only a named thing reads, too. What it cannot write is refused,
naming it: a thing the document does not declare, a body bound to the scene's prim (a block's things are placed in the
world), and a socket's name that is not one."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from mime._terminal import plain
from mime.blocks._definition import RESERVED_SOCKETS, SOCKET_KINDS
from mime.blocks._fragment import Renaming, entry_id_of
from mime.blocks._schema import FRAGMENT_SCHEMA
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.things._prims import WORLD


@dataclass(frozen=True)
class Drafted:
    """A fragment written from a saved experiment: its text, each socket with its kind (``[sockets]``'s rows), and the
    files below the folder it names (its records and descriptions), which a block folder holds at the same paths."""

    text: str
    sockets: Mapping[str, str]
    files: tuple[str, ...]

    def sockets_table(self) -> str:
        """``[sockets]`` as ``mime.toml`` holds it."""
        return "\n".join(["[sockets]", *(f'{name} = {{ kind = "{kind}" }}' for name, kind in self.sockets.items())])


def _refused(message: str, action: str, code: str = "block.socket") -> BuildRefused:
    return BuildRefused([Refusal(code, RefusalClass.USER, message, action)])


def _socket_of(value: str, sockets: Mapping[str, str]) -> object:
    head, dot, rest = value.partition(".")
    if head not in sockets:
        return value
    return {"socket": sockets[head]} if not dot else {"socket": sockets[head], "path": rest}


def _bound(entry: Mapping[str, Any], named: Mapping[str, str]) -> bool:
    """Whether an interaction or an observation binds a thing that is not a socket's (or binds none at all)."""
    heads = [str(value).partition(".")[0] for value in cast(Mapping[str, str], entry.get("binds", {})).values()]
    return not heads or any(head not in named for head in heads)


class _Dumper(yaml.SafeDumper):
    """Lists of plain values in flow style, mappings in block style, as the saved document is written."""


def _represent_list(dumper: yaml.SafeDumper, data: list[object]) -> yaml.Node:
    flow = all(not isinstance(item, (list, dict)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


def fragment_from(folder: str | os.PathLike[str], **sockets: str) -> Drafted:
    """The fragment of the saved experiment in ``folder``, each thing ``sockets`` names (``medium="water"``: the
    socket ``medium`` for the thing ``water``) its socket: the writer behind ``mime scaffold block --from``."""
    from mime.saved import read_folder

    document = cast(Mapping[str, Any], read_folder(Path(folder), absent_referenced=True).document)
    things = cast(Mapping[str, Mapping[str, Any]], document["things"])
    named: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for socket, thing in sockets.items():
        if not socket.isidentifier() or socket != socket.lower() or socket in RESERVED_SOCKETS or socket in things:
            raise _refused(f"{plain(socket)} is not a socket's name here (lower-case letters, digits and '_', none of "
                           f"{', '.join(RESERVED_SOCKETS)}, and no thing's name)", "name the socket otherwise")
        if thing not in things:
            raise _refused(f"the saved experiment declares no thing {plain(repr(thing))} for the socket "
                           f"{plain(socket)}", f"name one it declares: {', '.join(sorted(things))}")
        kind = str(things[thing]["kind"])
        if kind not in SOCKET_KINDS:
            raise _refused(f"{plain(thing)} is a {plain(kind)}, which no socket takes", "name another thing")
        named[thing], kinds[socket] = socket, kind
    own = {name: name for name in things if name not in named}
    for name in own:
        if things[name]["prim"] != f"{WORLD}/{name}":
            raise _refused(f"the body {plain(name)} binds the scene's prim {plain(things[name]['prim'])}, and a block's "
                           f"things are placed in the world: the scene is the experiment's that uses the block",
                           f"name it as a socket ({name}=...) or leave it out of the saved experiment", "block.fragment")
    partial = Renaming({**own, **named}, {}, own)
    kept = {section: [entry for entry in cast(Sequence[Mapping[str, Any]], document[section]) if _bound(entry, named)]
            for section in ("interactions", "observations")}
    ids: dict[str, str] = {}
    for section, kind in (("interactions", "interaction"), ("observations", "observation")):
        for entry in kept[section]:
            new = entry_id_of(kind, entry, {role: partial.path(value) for role, value in entry["binds"].items()})
            ids[entry["id"]] = new if new is not None else entry["id"]
    renamed = Renaming({**own, **named}, ids, own)
    fragment: dict[str, Any] = {"schema": FRAGMENT_SCHEMA}
    held_things: dict[str, Any] = {}
    for name in own:
        thing = dict(things[name])
        if "fills" in thing:
            thing["fills"] = _socket_of(renamed.path(thing["fills"]), {v: v for v in named.values()})
        held_things[name] = thing
    if held_things:
        fragment["things"] = held_things
    for section in ("interactions", "observations"):
        written = [{**entry, "id": renamed.ids[entry["id"]],
                    "binds": {role: _socket_of(renamed.path(value), {v: v for v in named.values()})
                              for role, value in entry["binds"].items()}} for entry in kept[section]]
        if written:
            fragment[section] = written
    controller = document.get("controller")
    if controller is not None:
        fragment["controller"] = {
            **controller,
            "reads": {label: _socket_of(renamed.path(value), {v: v for v in named.values()})
                      for label, value in controller["reads"].items()},
            "writes": {label: _socket_of(renamed.path(value), {v: v for v in named.values()})
                       for label, value in controller["writes"].items()}}
    dropped = {entry["id"] for section in ("interactions", "observations")
               for entry in cast(Sequence[Mapping[str, Any]], document[section]) if entry["id"] not in ids}
    for section in ("declared", "ranges", "sources"):
        written_paths = {renamed.path(path): value for path, value in cast(Mapping[str, Any],
                                                                            document.get(section, {})).items()
                         if path.partition(".")[0] not in named
                         and not any(path == each or path.startswith(f"{each}.") for each in dropped)}
        if written_paths:
            fragment[section] = written_paths
    used = {str(held) for thing in held_things.values() for held in
            [thing.get("material"), *(part.get("material") for part in thing.get("parts", {}).values())] if held}
    materials = {name: doc for name, doc in cast(Mapping[str, Any], document.get("materials", {})).items()
                 if name in used}
    if materials:
        fragment["materials"] = materials
    text = yaml.dump(fragment, Dumper=_Dumper, sort_keys=False, default_flow_style=False, allow_unicode=True, width=120)
    terms = {path: doc for path, doc in cast(Mapping[str, Any], document.get("terms", {})).items() if path in text}
    if terms:
        fragment["terms"] = terms
        text = yaml.dump(fragment, Dumper=_Dumper, sort_keys=False, default_flow_style=False, allow_unicode=True,
                         width=120)
    files = sorted({doc["record"]["file"] for doc in materials.values()} | set(_files_named(fragment)))
    return Drafted(text, kinds, tuple(files))


def _files_named(value: object) -> list[str]:
    """Every file a setting names (``{file: descriptions/..., sha256: ...}``) in ``value``."""
    if isinstance(value, dict):
        held = cast(dict[str, object], value)
        if isinstance(held.get("file"), str) and isinstance(held.get("sha256"), str):
            return [cast(str, held["file"])]
        return [each for item in held.values() for each in _files_named(item)]
    if isinstance(value, list):
        return [each for item in cast(list[object], value) for each in _files_named(item)]
    return []


__all__ = ["Drafted", "fragment_from"]
