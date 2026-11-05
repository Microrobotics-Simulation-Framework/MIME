"""The experiment's topology as a tree: what it declares, with no build.

The tree is built from the declarations alone, so it shows an experiment a script has not built (or cannot build yet)
and a saved folder's document alike. It lists the structures, the media and what each fills, the bodies and their
parts, the apparatus with its chain, parts, fields and drives, each law with the things it binds, each observation with
what it reads, and the controller with its reads and writes. Each entry shows the name it is declared under beside its
registered id and version.

**Determinism.** The tree is a function of the declarations' content, never of the order a script made them in: things
are listed by name within their kind, laws and observations by id (an id is made of names, never a counter), and the
bindings of a law in the order its roles are declared. A chain keeps the order it is declared in, since the first
segment is placed in the world and each later one mounts on the one before it: that order is the declaration's content.

The same structure is the text (:meth:`Tree.text`, plain indented lines) and the data (:meth:`Tree.record`): the text
is written from the record, so the two cannot disagree."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

from mime.build import (
    Attachment,
    ControllerDeclaration,
    Declaration,
    ObservationDeclaration,
    RunSettings,
    ThingDeclaration,
    binding_name,
    entry_id,
)
from mime.build._control import mode_of
from mime.registry import registration_of
from mime.things import Apparatus, Body, Medium, Part, Structure

#: The tree's data format, in the record's ``format``.
TREE_FORMAT = "mime.tree/1"

#: The section headings, in the order they are listed, with the record's key for each.
SECTIONS = (("structures", "structures"), ("media", "media"), ("bodies", "bodies"), ("apparatus", "apparatus"),
            ("laws", "laws"), ("observations", "observations"), ("controller", "controller"))


def _entry(entry: object) -> dict[str, Any]:
    """An entry's registered id and version (its rung too, where it has one). An entry that was never registered is
    shown by its class name with no version, which a build would refuse."""
    registration = registration_of(type(entry))
    held: dict[str, Any] = ({"id": type(entry).__name__, "version": None} if registration is None
                            else {"id": registration.name, "version": registration.version})
    rung = getattr(entry, "rung", None)
    if isinstance(rung, str):
        held["rung"] = rung
    return held


def _optional(entry: object | None) -> dict[str, Any] | None:
    return None if entry is None else _entry(entry)


def _names(values: object) -> list[str]:
    """The names a geometry kind declares of one sort (its regions, its surfaces or its directions), sorted."""
    if isinstance(values, Mapping):
        return sorted(str(name) for name in values)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if isinstance(values, (list, tuple, set, frozenset)):
        return sorted(str(name) for name in values)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return []


def _part(name: str, part: Part, magnets: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": name, "entry": _entry(part.geometry), "inside": part.inside, "mount": part.mount,
            "magnet": name in magnets}


def _parts(parts: Mapping[str, Part], magnets: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_part(name, parts[name], magnets) for name in sorted(parts)]


def _structure(thing: Structure) -> dict[str, Any]:
    geometry = type(thing.geometry) if thing.geometry is not None else None
    return {"name": thing.name, "entry": _optional(thing.geometry),
            "dynamics": _optional(thing.dynamics),
            "regions": _names(getattr(geometry, "regions", ())),
            "surfaces": _names(getattr(geometry, "surfaces", ())),
            "directions": _names(getattr(geometry, "directions", ()))}


def _medium(thing: Medium) -> dict[str, Any]:
    return {"name": thing.name, "fills": None if thing.fills is None else binding_name(thing.fills),
            "flow": _optional(thing.flow)}


def _body(thing: Body) -> dict[str, Any]:
    return {"name": thing.name, "entry": _entry(thing.dynamics), "parts": _parts(thing.parts, thing.magnets)}


def _apparatus(thing: Apparatus) -> dict[str, Any]:
    return {"name": thing.name,
            "chain": [{"name": name, "entry": _entry(thing.chain[name])} for name in thing.chain],
            "parts": _parts(thing.parts, thing.magnets),
            "fields": [{"name": name, "entry": _entry(thing.fields[name])} for name in sorted(thing.fields)],
            "commands": [{"name": name, "entry": _entry(thing.commands[name])} for name in sorted(thing.commands)]}


def _bindings(roles: Sequence[str], given: Mapping[str, Any]) -> list[list[str]]:
    """The names a law or an observation binds, as ``[role, name]`` pairs: its declared roles first, in the order they
    are declared; any binding it was given for another role after them, by role."""
    pairs = [[role, "?" if given.get(role) is None else binding_name(given[role])] for role in roles]
    return pairs + [[role, binding_name(given[role])] for role in sorted(given) if role not in roles]


def _law(declaration: Attachment) -> dict[str, Any]:
    cls = type(declaration.law)
    return {"id": entry_id(cls, declaration.bindings), "entry": _entry(declaration.law),
            "binds": _bindings(list(cls.roles), declaration.bindings)}


def _observation(declaration: ObservationDeclaration) -> dict[str, Any]:
    observation = declaration.observation
    cls = type(observation)
    given = observation.bindings()
    return {"id": entry_id(cls, given), "entry": _entry(observation), "reads": _bindings(list(cls.roles), given),
            "outputs": sorted(output.field for output in cls.outputs)}


def _controller(declaration: ControllerDeclaration) -> dict[str, Any]:
    mode = mode_of(declaration.mode)
    return {"entry": _entry(declaration.controller),
            "mode": mode.word if mode is not None else str(declaration.mode),
            "reads": [[label, binding_name(declaration.reads[label])] for label in sorted(declaration.reads)],
            "writes": [[label, binding_name(declaration.writes[label])] for label in sorted(declaration.writes)]}


def _sorted(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Rows by their name or id; two with the same one (a script's duplicate, which the build refuses) by their content,
    so that the order of the calls never shows."""
    return sorted(rows, key=lambda row: (row[key], json.dumps(row, sort_keys=True)))


def record_of(settings: RunSettings, declarations: Sequence[Declaration]) -> dict[str, Any]:
    """The tree as plain data: ``format``, the experiment's name and profile, then one list for each section of
    :data:`SECTIONS` (the controller one record, or ``None``)."""
    structures: list[dict[str, Any]] = []
    media: list[dict[str, Any]] = []
    bodies: list[dict[str, Any]] = []
    apparatus: list[dict[str, Any]] = []
    laws: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    controllers: list[dict[str, Any]] = []
    for declaration in declarations:
        if isinstance(declaration, ThingDeclaration):
            thing = declaration.thing
            if isinstance(thing, Structure):
                structures.append(_structure(thing))
            elif isinstance(thing, Medium):
                media.append(_medium(thing))
            elif isinstance(thing, Body):
                bodies.append(_body(thing))
            else:
                apparatus.append(_apparatus(thing))
        elif isinstance(declaration, Attachment):
            laws.append(_law(declaration))
        elif isinstance(declaration, ObservationDeclaration):
            observations.append(_observation(declaration))
        elif isinstance(declaration, ControllerDeclaration):
            controllers.append(_controller(declaration))
    controllers.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return {"format": TREE_FORMAT, "experiment": str(settings.experiment), "profile": str(settings.profile),
            "structures": _sorted(structures, "name"), "media": _sorted(media, "name"),
            "bodies": _sorted(bodies, "name"), "apparatus": _sorted(apparatus, "name"),
            "laws": _sorted(laws, "id"), "observations": _sorted(observations, "id"),
            "controller": controllers[0] if controllers else None}


def _entry_text(entry: Mapping[str, Any]) -> str:
    version = f" {entry['version']}" if entry["version"] else ""
    rung = f", rung {entry['rung']}" if entry.get("rung") else ""
    return f"{entry['id']}{version}{rung}"


def _pairs(pairs: Sequence[Sequence[str]], joiner: str = " ") -> str:
    return ", ".join(f"{first}{joiner}{second}" for first, second in pairs)


def _names_line(label: str, names: Sequence[str]) -> str | None:
    return f"{label} {', '.join(names)}" if names else None


def _part_line(part: Mapping[str, Any]) -> str:
    notes = [f"mounted on {part['mount']}"] if part["mount"] else []
    notes += [f"inside {part['inside']}"] if part["inside"] else []
    notes += ["permanent magnet"] if part["magnet"] else []
    return f"part {part['name']}  {_entry_text(part['entry'])}" + (f"  ({'; '.join(notes)})" if notes else "")


Line = tuple[int, str]


def lines_of(record: Mapping[str, Any]) -> list[Line]:
    """The tree as ``(depth, text)`` lines: the experiment at depth 0, each section's heading at depth 1, its items
    at 2 and what each holds at 3. The plain text indents by depth; a terminal draws the same lines as a tree."""
    lines: list[Line] = [(0, f"experiment {record['experiment']}  (profile {record['profile']})")]

    def section(heading: str, rows: Sequence[Mapping[str, Any]], one: Any) -> None:
        lines.append((1, heading if rows else f"{heading}: none"))
        for row in rows:
            head, details = one(row)
            lines.append((2, head))
            lines.extend((3, detail) for detail in details if detail is not None)

    def structure(row: Mapping[str, Any]) -> tuple[str, list[str | None]]:
        head = f"{row['name']}  {_entry_text(row['entry'])}" if row["entry"] else f"{row['name']}  (no geometry)"
        return head, [None if row["dynamics"] is None else f"dynamics {_entry_text(row['dynamics'])}",
                      _names_line("regions", row["regions"]), _names_line("surfaces", row["surfaces"]),
                      _names_line("directions", row["directions"])]

    def medium(row: Mapping[str, Any]) -> tuple[str, list[str | None]]:
        head = f"{row['name']}  fills {row['fills']}" if row["fills"] else f"{row['name']}  (fills nothing declared)"
        return head, ["still, no flow dynamics" if row["flow"] is None else f"flow {_entry_text(row['flow'])}"]

    def body(row: Mapping[str, Any]) -> tuple[str, list[str | None]]:
        return f"{row['name']}  {_entry_text(row['entry'])}", [_part_line(part) for part in row["parts"]]

    def apparatus(row: Mapping[str, Any]) -> tuple[str, list[str | None]]:
        details: list[str | None] = [f"chain {segment['name']}  {_entry_text(segment['entry'])}" for segment in row["chain"]]
        details += [_part_line(part) for part in row["parts"]]
        details += [f"field {each['name']}  {_entry_text(each['entry'])}" for each in row["fields"]]
        details += [f"drive {each['name']}  {_entry_text(each['entry'])}" for each in row["commands"]]
        return row["name"], details

    def law(row: Mapping[str, Any]) -> tuple[str, list[str | None]]:
        return f"{row['id']}  {_entry_text(row['entry'])}", [f"between {_pairs(row['binds'])}"]

    def observation(row: Mapping[str, Any]) -> tuple[str, list[str | None]]:
        return f"{row['id']}  {_entry_text(row['entry'])}", [
            f"reads {_pairs(row['reads'])}", _names_line("gives", row["outputs"])]

    section("structures", record["structures"], structure)
    section("media", record["media"], medium)
    section("bodies", record["bodies"], body)
    section("apparatus", record["apparatus"], apparatus)
    section("laws", record["laws"], law)
    section("observations", record["observations"], observation)
    controller = record["controller"]
    if controller is None:
        lines.append((1, "controller: none"))
    else:
        lines += [(1, "controller"), (2, f"{_entry_text(controller['entry'])}  (mode {controller['mode']})"),
                  (3, f"reads {_pairs(controller['reads'], ' <- ')}" if controller["reads"] else "reads nothing"),
                  (3, f"writes {_pairs(controller['writes'], ' -> ')}" if controller["writes"] else "writes nothing")]
    return lines


@dataclass(frozen=True)
class Tree:
    """An experiment's topology, from :meth:`mime.experiment.Experiment.tree`: ``text()`` is the plain indented text
    (also ``str()``), ``record()`` the same structure as plain data (``mime.tree/1``), ``json()`` that as JSON."""

    _record: Mapping[str, Any]

    def record(self) -> dict[str, Any]:
        """The tree as plain data, a copy."""
        return copy.deepcopy(dict(self._record))

    def lines(self) -> list[Line]:
        """The tree as ``(depth, text)`` lines, as a terminal draws it."""
        return lines_of(self._record)

    def text(self) -> str:
        """The tree as plain text, one line each, indented two spaces to a level, ending in a newline."""
        return "".join(f"{'  ' * depth}{text}\n" for depth, text in self.lines())

    def json(self) -> str:
        """The record as JSON, indented, ending in a newline."""
        return json.dumps(self._record, indent=2) + "\n"

    @override
    def __str__(self) -> str:
        return self.text()

    @override
    def __repr__(self) -> str:
        return self.text()


def tree_of(settings: RunSettings, declarations: Sequence[Declaration]) -> Tree:
    return Tree(record_of(settings, declarations))
