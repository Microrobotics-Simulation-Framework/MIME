"""The build report, and the canonical text of a built graph."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from maddening import GraphManager

from mime.build._maddening import graph_dict
from mime.build._refusals import Refusal

if TYPE_CHECKING:
    from mime.build._lags import EdgeLag
    from mime.build._plan import MaterialsPlan, ValueLine


@dataclass(frozen=True)
class NodeLine:
    position: int
    node: str
    owner: str
    section: str
    entry: str
    divider: int
    timestep: float


@dataclass(frozen=True)
class GroupLine:
    nodes: tuple[str, ...]
    tolerance: float
    max_iterations: int
    requested_by: tuple[str, ...]


@dataclass(frozen=True)
class OrientationLine:
    """A thing's line in the report: its name, kind and prim, and its declared directions, each a reference written
    as the path it refers to beside the vector it resolved to."""

    thing: str
    kind: str
    prim: str
    directions: tuple[str, ...]

    def text(self) -> str:
        return f"{self.thing} ({self.kind} at {self.prim}): {'; '.join(self.directions) or 'no declared directions'}"


@dataclass(frozen=True)
class ReportedWarning:
    """A warning MADDENING gave while checking or compiling the graph, with where it came from."""

    source: str
    message: str


@dataclass(frozen=True)
class RunStructure:
    """How a run executes, derived from what the experiment declares. A run is MADDENING's one jit-compiled step,
    scanned; ``between`` names what runs on the host between compiled calls of ``chunk`` steps (a controller in
    modes 2 to 4, a per-chunk hook, an observation outside the step, a study's variant loop). When nothing does,
    the whole run is one compiled scan."""

    chunk: int | None = None
    between: tuple[str, ...] = ()

    def text(self) -> str:
        if self.chunk is None:
            return "one compiled scan"
        return f"compiled chunks of {self.chunk} steps, with these between them: {', '.join(self.between)}"


@dataclass(frozen=True)
class BuildReport:
    """What the build did: every node in the order it was added, with the experiment name it belongs to and its rate;
    the coupling groups and who asked for them; the external inputs; derived seeds; declared validity lines; the
    vocabulary versions; how a run executes; MADDENING's warnings; the derived hash; the conditions records are read
    at, each record with its pins, each read with OpenMaterica's verdict and what the build did with it, and each
    body's parts; every value of a thing the entries read, with where it came from and what it was derived from;
    every edge with the step its value is read from; the scene's frame, one orientation line per thing and the
    stage's hash; each chain segment's description as read, every element of it not used or read past listed; the
    limits each entry's capability record states of its model; the controller, with its mode, period and latency, each
    read with its class and each write with the command it replaces the drive of; and every refusal."""

    experiment: str
    dt: float | None
    nodes: tuple[NodeLine, ...] = ()
    groups: tuple[GroupLine, ...] = ()
    external_inputs: tuple[str, ...] = ()
    seeds: Mapping[str, Mapping[str, int]] = field(default_factory=dict[str, Mapping[str, int]])
    validity: Mapping[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])
    vocabulary: Mapping[str, str] = field(default_factory=dict[str, str])
    refusals: tuple[Refusal, ...] = ()
    run: RunStructure = RunStructure()
    warnings: tuple[ReportedWarning, ...] = ()
    derived_hash: str | None = None
    values: tuple[ValueLine, ...] = ()
    lags: tuple[EdgeLag, ...] = ()
    scene: str | None = None
    orientation: tuple[OrientationLine, ...] = ()
    stage_hash: str | None = None
    materials: MaterialsPlan | None = None
    descriptions: tuple[str, ...] = ()
    limits: Mapping[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])
    controller: tuple[str, ...] = ()

    def previous_step(self) -> list[str]:
        """The edges that read the previous step, as ``source.field -> target.field``, in the order added."""
        from mime.build._lags import Lag

        return [edge.text for edge in self.lags if edge.lag is Lag.PREVIOUS_STEP]

    def materials_text(self) -> list[str]:
        """The conditions, the records and their reads, and the parts; nothing when the experiment names none."""
        held = self.materials
        if held is None or not (held.conditions or held.records or held.parts):
            return []
        lines = ["conditions every record is read at: " + ("; ".join(c.text() for c in held.conditions) or "none")]
        lines.append("records, each pinned by its file's sha256 and its content hash:")
        lines += [f"  {r.material}: {r.id} {r.version}, content hash {r.content_hash}, file {r.file} sha256 "
                  f"{r.sha256}" for r in held.records] or ["  none"]
        lines.append("reads, with OpenMaterica's verdict and what the build did with it:")
        lines += [f"  {r.material} {r.field}: {r.use}" for r in held.reads] or ["  none"]
        lines.append("parts, each a solid in its own frame, placed in its body's:")
        for part in held.parts:
            settings = ", ".join(f"{k} {v!r}" for k, v in sorted(part.settings.items()))
            notes = [f"inside {part.inside}" if part.inside else "", f"material {part.material}" if part.material
                     else "no material", "a magnet" if part.magnet else ""]
            lines.append(f"  {part.body}.{part.name}: {part.registered.cls.label()} ({settings}), {part.volume!r} m^3 "
                         f"at {tuple(part.centre)!r} m; " + ", ".join(n for n in notes if n))
        if not held.parts:
            lines.append("  none")
        return lines

    def summary(self) -> list[str]:
        """The report's first lines: the experiment, its step and run, each entry, each value read with where it came
        from, each record read with OpenMaterica's verdict, the controller, the attached rungs' validity lines, the
        warnings and refusals, and the hash."""
        lines = [f"build report: experiment {self.experiment!r}, "
                 + (f"base step {self.dt!r} s" if self.dt else "no base step") + f", run as {self.run.text()}"]
        owners: dict[str, str] = {}
        for node in self.nodes:
            owners.setdefault(node.owner, node.entry)
        lines.append(f"entries ({len(owners)}):" if owners else "entries: none")
        lines += [f"  {owner}: {entry}" for owner, entry in owners.items()]
        if self.values:
            lines.append(f"values read ({len(self.values)}):")
            lines += [f"  {v.path} = {v.value!r} {v.unit}  [{_short(v.source)}]" for v in self.values]
        held = self.materials
        if held is not None and held.reads:
            lines.append("record reads, with OpenMaterica's verdict:")
            lines += [f"  {r.material} {r.field}: {r.status}" + (", accepted by the experiment" if
                                                                  "accepted by the experiment" in r.use else "")
                      for r in held.reads]
        lines += list(self.controller[:1])
        for owner, validity in sorted(self.validity.items()):
            lines += [f"validity of {owner} (declared, not yet checked): {line}" for line in validity]
        lines.append(f"MADDENING's warnings: {len(self.warnings) or 'none'}; refusals: {len(self.refusals) or 'none'}")
        lines += [f"  {line}" for r in self.refusals for line in str(r).splitlines()]
        if self.derived_hash:
            lines.append(f"derived hash: {self.derived_hash}")
        return lines

    def text(self, *, details: bool = False) -> str:
        """The summary; with ``details``, every node, group, edge, value's derivation, record, part and orientation
        line besides."""
        if not details:
            return "\n".join([*self.summary(), "every node, edge and value's derivation: report.text(details=True)"])
        lines = [f"build report: experiment {self.experiment!r}",
                 f"base step dt: {self.dt!r} s" if self.dt else "base step dt: not set"]
        lines.append("nodes, in the order added:")
        for n in self.nodes:
            rate = "every step" if n.divider == 1 else f"every {n.divider} steps"
            lines.append(f"  {n.position:>3}  {n.node}  [{n.section} {n.owner}: {n.entry}]  {rate} ({n.timestep!r} s)")
        if not self.nodes:
            lines.append("  none")
        lines.append("coupling groups:")
        for g in self.groups:
            lines.append(f"  {{{', '.join(g.nodes)}}}: solver ift, atol 0.0, tolerance {g.tolerance!r}, "
                         f"max_iterations {g.max_iterations}; asked by {'; '.join(g.requested_by)}")
        if not self.groups:
            lines.append("  none")
        if self.scene is not None:
            lines.append(f"scene: {self.scene}")
            lines.append("orientation, one line per thing:")
            lines += [f"  {line.text()}" for line in self.orientation] or ["  no things"]
        if self.descriptions:
            lines.append("descriptions, each chain segment's as read:")
            lines += [f"  {line}" for line in self.descriptions]
        lines.append(f"external inputs: {', '.join(self.external_inputs) or 'none'}")
        lines += list(self.controller) or ["controller: none declared, so no latency to account for"]
        lines += self.materials_text()
        lines.append("values of things read:")
        for v in self.values:
            readers = ", ".join(v.read_by) or "nothing reads it"
            lines.append(f"  {v.path} = {v.value!r} {v.unit}  [{v.source}: {v.reason}]  read by {readers}")
            lines += [f"      {detail}" for detail in v.details]
        if not self.values:
            lines.append("  none")
        from mime.build._builders import REACTION
        from mime.build._lags import PLANNED, STAGGERED, Lag

        lines.append(f"edges, in the order added, with the step each reads; a previous-step read is planned when "
                     f"{PLANNED}, and refused otherwise unless a coupling group closes its loop:")
        for i, edge in enumerate(self.lags, 1):
            note = "" if edge.lag is not Lag.PREVIOUS_STEP else f", {STAGGERED}" if edge.carries == REACTION else \
                ", planned" if edge.planned else ", UNPLANNED"
            lines.append(f"  {i}. {edge.text}  {edge.lag.value}{note}")
        if not self.lags:
            lines.append("  none")
        for owner, seeds in sorted(self.seeds.items()):
            lines.append(f"seeds of {owner}: " + ", ".join(f"{k}={v}" for k, v in sorted(seeds.items())))
        for owner, validity in sorted(self.validity.items()):
            for line in validity:
                lines.append(f"validity of {owner} (declared, not yet checked): {line}")
        for owner, limits in sorted(self.limits.items()):
            for line in limits:
                lines.append(f"limits of {owner} (stated by its record): {line}")
        lines.append("vocabulary: " + ", ".join(f"{k} {v}" for k, v in self.vocabulary.items()))
        lines.append(f"run: {self.run.text()}")
        lines.append(f"MADDENING's warnings: {len(self.warnings) or 'none'}")
        lines += [f"  [{w.source}] {w.message}" for w in self.warnings]
        if self.derived_hash:
            lines.append(f"derived hash: {self.derived_hash}")
        if self.stage_hash:
            lines.append(f"stage: {self.stage_hash}, written to scene/scene.usda when saved")
        lines.append(f"refusals: {len(self.refusals) or 'none'}")
        lines += [f"  {line}" for r in self.refusals for line in str(r).splitlines()]
        return "\n".join(lines)


#: The most characters of a value's source the summary prints; the details print it whole.
SHORT_SOURCE = 100


def _short(text: str) -> str:
    return text if len(text) <= SHORT_SOURCE else text[:SHORT_SOURCE - 3].rstrip() + "..."


def graph_config(graph: GraphManager) -> dict[str, Any]:
    """MADDENING's own description of ``graph``, read without its warnings: the post-build checks report what those
    warnings say, each in the experiment's terms."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return graph_dict(graph, strict_mappings=False)


def _value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=repr)


def graph_text(graph: GraphManager, lags: Sequence[EdgeLag] = ()) -> str:
    """The canonical text of a built graph: nodes in the order added, with class and rate; edges in the order added,
    with the additive flag, units labels, mapping and transform, and the step each reads from MIME's lag accounting;
    coupling groups with every setting; external inputs."""
    read = {(e.source[0], e.source[1], e.target[0], e.target[1]): e.lag.value for e in lags}
    config = graph_config(graph)
    base = graph.timestep
    lines = [f"base step: {base!r}", "nodes:"]
    for i, node in enumerate(config["nodes"], 1):
        divider = round(node["timestep"] / base)
        lines.append(f"  {i}. {node['name']}  {node['type']}  timestep {node['timestep']!r}  every {divider}")
    lines.append("edges:")
    for i, edge in enumerate(config["edges"], 1):
        flags = ["additive" if edge.get("additive") else "replacing",
                 f"[{edge.get('source_units', '?')} -> {edge.get('target_units', '?')}]"]
        if "mapping" in edge:
            flags.append(f"mapping {_value(edge['mapping'])}")
        if "geometry" in edge:
            flags.append(f"geometry {_value(edge['geometry'])}")
        if "transform" in edge:
            flags.append(f"transform {edge['transform']}")
        lag = read.get((edge["source_node"], edge["source_field"], edge["target_node"], edge["target_field"]))
        if lag is not None:
            flags.append(f"reads {lag}")
        lines.append(f"  {i}. {edge['source_node']}.{edge['source_field']} -> "
                     f"{edge['target_node']}.{edge['target_field']}  {'  '.join(flags)}")
    lines.append("coupling groups:")
    for i, group in enumerate(config.get("coupling_groups", []), 1):
        settings = ", ".join(f"{k}={_value(v)}" for k, v in sorted(group.items()) if k != "nodes")
        lines.append(f"  {i}. {{{', '.join(group['nodes'])}}}  {settings}")
    if not config.get("coupling_groups"):
        lines.append("  none")
    lines.append("external inputs:")
    for ei in config["external_inputs"]:
        lines.append(f"  {ei['target_node']}.{ei['target_field']}  shape {tuple(ei['shape'])}  {ei['dtype']}")
    if not config["external_inputs"]:
        lines.append("  none")
    return "\n".join(lines) + "\n"
