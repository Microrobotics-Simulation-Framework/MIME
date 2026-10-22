"""The build report, and the canonical text of a built graph."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from maddening import GraphManager

from mime.build._maddening import graph_dict
from mime.build._refusals import Refusal


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
    vocabulary versions; how a run executes; MADDENING's warnings; the derived hash; and every refusal."""

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

    def text(self) -> str:
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
        lines.append(f"external inputs: {', '.join(self.external_inputs) or 'none'}")
        for owner, seeds in sorted(self.seeds.items()):
            lines.append(f"seeds of {owner}: " + ", ".join(f"{k}={v}" for k, v in sorted(seeds.items())))
        for owner, validity in sorted(self.validity.items()):
            for line in validity:
                lines.append(f"validity of {owner} (declared, not yet checked): {line}")
        lines.append("vocabulary: " + ", ".join(f"{k} {v}" for k, v in self.vocabulary.items()))
        lines.append(f"run: {self.run.text()}")
        lines.append(f"MADDENING's warnings: {len(self.warnings) or 'none'}")
        lines += [f"  [{w.source}] {w.message}" for w in self.warnings]
        if self.derived_hash:
            lines.append(f"derived hash: {self.derived_hash}")
        lines.append(f"refusals: {len(self.refusals) or 'none'}")
        lines += [f"  {line}" for r in self.refusals for line in str(r).splitlines()]
        return "\n".join(lines)


def graph_config(graph: GraphManager) -> dict[str, Any]:
    """MADDENING's own description of ``graph``, read without its warnings: the post-build checks report what those
    warnings say, each in the experiment's terms."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return graph_dict(graph, strict_mappings=False)


def _value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=repr)


def graph_text(graph: GraphManager) -> str:
    """The canonical text of a built graph: nodes in the order added, with class and rate; edges in the order added,
    with the additive flag, units labels, mapping and transform; coupling groups with every setting; external
    inputs."""
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
