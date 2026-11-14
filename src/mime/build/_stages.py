"""Phase 2 of an apparatus's ports (``mime.build._ports`` is phase 1): the stage nodes, ``ports:<apparatus>`` and
``measure:<apparatus>``, each built only where a staged port needs it, with their edges, and the energy ledger's look
through a command stage to the writer it carries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mime.build._builders import COMMAND, STATE, EdgeDecl
from mime.build._ports import StagePlan, stage_id
from mime.build._refusals import Refusal, RefusalClass
from mime.nodes.control import COMMAND_STAGE, MEASURE_STAGE, CommandStageNode, MeasureStageNode

if TYPE_CHECKING:
    from maddening.core.node import SimulationNode


def _elements(value: float | tuple[float, ...]) -> tuple[float, ...]:
    return value if isinstance(value, tuple) else (value,)


def stage_nodes(stages: Sequence[StagePlan], built: Mapping[str, SimulationNode],
                dt: float) -> tuple[list[SimulationNode], list[EdgeDecl], list[Refusal]]:
    """Each apparatus's stages, from what phase 1 resolved and the nodes built for the rest of the graph (each
    receiver's input gives a command's shape, each source's state a measurement's start), with their edges into each
    receiver and from each source; or what a port's numbers do not fit once the shapes are known."""
    from mime.build._maddening import initial_fields, inputs

    nodes: list[SimulationNode] = []
    edges: list[EdgeDecl] = []
    refusals: list[Refusal] = []
    for stage in stages:
        if stage.commands:
            name = stage_id(COMMAND_STAGE, stage.apparatus)
            shapes: dict[str, tuple[int, ...]] = {}
            leaves: dict[str, Any] = {}
            for plan in stage.commands:
                spec = inputs(built[plan.receiver[0]])[plan.receiver[1]]
                held_shape: tuple[int, ...] = cast(Any, spec).shape
                shape = tuple(int(n) for n in held_shape)
                size = int(np.prod(shape)) if shape else 1
                fits = [len(_elements(value)) in (1, size) for value in (plan.low, plan.high, plan.rest, plan.slew)
                        if value is not None]
                if not all(fits) or (plan.restated is not None and tuple(plan.restated) != shape):
                    refusals.append(Refusal(
                        "port.declaration", RefusalClass.USER,
                        f"apparatus {plan.apparatus!r}'s port on {plan.command!r}: {plan.receiver[0]}.{plan.receiver[1]}"
                        f" takes shape {shape}, and the port's restated shape, bounds, rest or slew do not fit it "
                        f"(one value, or one per element)",
                        f"give CommandPort(bounds=..., rest=..., slew=...) one value or {size} for each"))
                    continue
                shapes[plan.key] = shape
                leaves[f"rest_{plan.key}"] = plan.rest
                if plan.bounded:
                    leaves[f"low_{plan.key}"], leaves[f"high_{plan.key}"] = plan.low, plan.high
                if plan.slew is not None:
                    leaves[f"slew_{plan.key}"] = plan.slew
            ports = [plan for plan in stage.commands if plan.key in shapes]
            nodes.append(CommandStageNode(
                name, dt, commands=[p.key for p in ports], units={p.key: p.units for p in ports}, shapes=shapes,
                periods={p.key: p.period for p in ports}, latencies={p.key: p.latency for p in ports},
                bounded={p.key: p.bounded for p in ports}, slewed={p.key: p.slew is not None for p in ports},
                **leaves))
            edges += [EdgeDecl(name, (name, p.key), p.receiver, False, p.units, seq, COMMAND)
                      for seq, p in enumerate(ports)]
        if stage.measurements:
            name = stage_id(MEASURE_STAGE, stage.apparatus)
            leaves = {}
            shapes = {}
            ages: dict[str, int] = {}
            for plan in stage.measurements:
                start = np.asarray(initial_fields(built[plan.source[0]])[plan.source[1]], dtype=np.float64)
                shape = tuple(int(n) for n in start.shape)
                if plan.restated is not None and tuple(plan.restated) != shape:
                    refusals.append(Refusal(
                        "port.declaration", RefusalClass.USER,
                        f"apparatus {plan.apparatus!r}'s measurement {plan.port!r} restates its shape as "
                        f"{plan.restated}, and {plan.of} is {shape}", f"restate shape={shape}, or leave it out"))
                    continue
                shapes[plan.key] = shape
                flat = tuple(float(c) for c in start.reshape(-1))
                leaves[f"initial_{plan.key}"] = flat[0] if flat and all(c == flat[0] for c in flat) else flat
                # The source's value at the start was made before the run's first step: a state's at the step
                # before it, an observation with a clock of its own by its own age.
                ages[plan.key] = int(np.asarray(initial_fields(built[plan.age[0]])[plan.age[1]])) \
                    if plan.age is not None else 1
                leaves[f"initial_age_{plan.key}"] = ages[plan.key]
                if plan.sigma is not None:
                    leaves[f"sigma_{plan.key}"] = plan.sigma
                if plan.quantum is not None:
                    leaves[f"quantum_{plan.key}"] = plan.quantum
            ports_m = [plan for plan in stage.measurements if plan.key in shapes]
            nodes.append(MeasureStageNode(
                name, dt, ports=[p.key for p in ports_m], units={p.key: p.units for p in ports_m}, shapes=shapes,
                periods={p.key: p.period for p in ports_m}, latencies={p.key: p.latency for p in ports_m},
                aged={p.key: p.age is not None for p in ports_m}, seeds={p.key: p.seed for p in ports_m},
                noisy={p.key: p.sigma is not None for p in ports_m},
                quantised={p.key: p.quantum is not None for p in ports_m}, **leaves))
            for seq, p in enumerate(ports_m):
                edges.append(EdgeDecl(name, p.source, (name, p.key), False, p.units, 2 * seq, STATE))
                if p.age is not None:
                    edges.append(EdgeDecl(name, p.age, (name, f"age_{p.key}"), False, "1", 2 * seq + 1, STATE))
    return nodes, edges, refusals


def writers_through(edges: Sequence[EdgeDecl]) -> dict[tuple[str, str], str]:
    """The writer each command stage's output carries, by the stage's node and field: the entry whose edge writes the
    stage's input of the same name (a drive's id, the controller's), which the energy ledger books the command under,
    as it would without the stage."""
    stages = {edge.source[0] for edge in edges if edge.source[0].startswith(f"{COMMAND_STAGE}:")}
    return {edge.target: edge.by for edge in edges if edge.target[0] in stages}


__all__ = ["stage_nodes", "writers_through"]
