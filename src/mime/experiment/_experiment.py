from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mime.build import (
    Attachment,
    BuildReport,
    Declaration,
    ObservationDeclaration,
    PlanSet,
    RunSettings,
    ThingDeclaration,
    construct,
    entry_id,
    graph_text,
    plan_experiment,
    run_steps,
)
from mime.registry import Interaction, Observation
from mime.things import (
    Apparatus,
    ApparatusHandle,
    Body,
    BodyHandle,
    Handle,
    Medium,
    MediumHandle,
    Structure,
    StructureHandle,
    TypedPath,
)
from mime.units import TIME, Quantity, si


class ObservationHandle:
    """What ``observe`` returns: the observation's id."""

    __slots__ = ("id",)

    def __init__(self, id: str) -> None:
        self.id = id

    def __repr__(self) -> str:
        return f"ObservationHandle({self.id!r})"


class Experiment:
    """An experiment as the script declares it. Declaring touches no graph: ``add_*``, ``attach`` and ``observe``
    record declarations and return handles or ids, and nothing is checked until :meth:`plan` or :meth:`build`, which
    refuse with every reason at once.

    ``dt`` is the base step, with no default; an entry's own ``rate=`` is a whole multiple of it. ``up`` and
    ``gravity`` declare the world frame: there are no default axes, and gravity points along ``-up``.
    """

    def __init__(self, name: str, *, up: Any = None, gravity: Quantity | None = None, dt: Quantity | None = None,
                 seed: int = 0) -> None:
        self.name = name
        self._settings = RunSettings(name, up, gravity, dt, seed)
        self._declarations: list[Declaration] = []

    def _add(self, thing: Any, cls: type) -> None:
        if not isinstance(thing, cls):
            raise TypeError(f"expected a {cls.__name__}, got {type(thing).__name__}")
        self._declarations.append(ThingDeclaration(thing))

    def add_structure(self, structure: Structure) -> StructureHandle:
        self._add(structure, Structure)
        return StructureHandle(structure.name)

    def add_medium(self, medium: Medium) -> MediumHandle:
        self._add(medium, Medium)
        return MediumHandle(medium.name)

    def add_body(self, body: Body) -> BodyHandle:
        self._add(body, Body)
        return BodyHandle(body.name)

    def add_apparatus(self, apparatus: Apparatus) -> ApparatusHandle:
        self._add(apparatus, Apparatus)
        return ApparatusHandle(apparatus.name, apparatus.fields)

    def attach(self, law: Interaction, /, **bindings: Handle | TypedPath | str) -> str:
        """Bind an interaction to things by its roles; returns its id (its registered name and the things bound to
        it). Every role is bound explicitly: there is no default binding."""
        if not isinstance(law, Interaction):
            raise TypeError(f"attach takes an Interaction, got {type(law).__name__}")
        for role, value in bindings.items():
            if not isinstance(value, (Handle, TypedPath, str)):
                raise TypeError(f"{role}= takes a handle, a typed path or a name, not {type(value).__name__}")
        self._declarations.append(Attachment(law, dict(bindings)))
        return entry_id(type(law), bindings)

    def observe(self, observation: Observation) -> ObservationHandle:
        if not isinstance(observation, Observation):
            raise TypeError(f"observe takes an Observation, got {type(observation).__name__}")
        self._declarations.append(ObservationDeclaration(observation))
        return ObservationHandle(entry_id(type(observation), observation.bindings()))

    def plan(self) -> PlanSet:
        """Phase 1 alone: resolve and check every declaration, with no graph. Raises ``BuildRefused``."""
        return plan_experiment(self._settings, list(self._declarations))

    def build(self) -> BuiltExperiment:
        """Phases 1 and 2. Raises ``BuildRefused`` with every reason, each with its class and action line."""
        plans = self.plan()
        graph, report = construct(plans)
        return BuiltExperiment(graph, report, plans)


@dataclass(frozen=True)
class RunResult:
    """``final`` maps each node to its fields after the last step; ``history``, when asked for, stacks each field
    over the steps."""

    steps: int
    final: dict[str, dict[str, np.ndarray]]
    history: dict[str, dict[str, np.ndarray]] | None = None


class BuiltExperiment:
    """A built experiment: its report, its graph's canonical text, and runs. A run continues from where the last
    one left the graph."""

    def __init__(self, graph: Any, report: BuildReport, plans: PlanSet) -> None:
        self._graph = graph
        self.report = report
        self._plans = plans

    @property
    def graph(self) -> Any:
        """The MADDENING graph, for inspection. Editing it bypasses every check the build made."""
        return self._graph

    def snapshot(self) -> str:
        return graph_text(self._graph)

    def run(self, duration: Quantity, *, history: bool = False) -> RunResult:
        seconds = si(duration, TIME, what="the run's duration")
        dt = self._plans.dt
        ratio = seconds / dt  # type: ignore[operator]
        steps = round(ratio)
        if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
            raise ValueError(f"a run of {seconds!r} s is not a whole number of base steps of {dt!r} s; the nearest "
                             f"are {max(1, math.floor(ratio)) * dt!r} s and {max(1, math.ceil(ratio)) * dt!r} s")
        final, stacked = run_steps(self._graph, steps, history=history)
        return RunResult(steps, final, stacked)
