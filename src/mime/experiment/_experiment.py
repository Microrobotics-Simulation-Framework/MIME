from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, cast, override

import numpy as np
from maddening import GraphManager

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
from mime.saved import (
    DEFAULT_STEPS,
    ExperimentDocument,
    LoadReport,
    Lock,
    SavedForm,
    SavedRecord,
    VerifyReport,
    check_load,
    declarations_from,
    document_of,
    read_document,
    read_folder,
    record_of,
    verify_history,
    write_folder,
)
from mime.saved._folder import LoadRefused
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

type History = dict[str, dict[str, np.ndarray]]


class ObservationHandle:
    """What ``observe`` returns: the observation's id."""

    __slots__ = ("id",)

    def __init__(self, id: str) -> None:
        self.id = id

    @override
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
        self._saved: SavedForm | None = None
        self.load_report: LoadReport | None = None

    @classmethod
    def _declared(cls, settings: RunSettings, declarations: Sequence[Declaration]) -> Experiment:
        experiment = cls(settings.experiment, up=settings.up, gravity=settings.gravity, dt=settings.dt,
                         seed=settings.seed)
        experiment._declarations = list(declarations)
        return experiment

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
        if not isinstance(cast(object, law), Interaction):  # a script is not type-checked
            raise TypeError(f"attach takes an Interaction, got {type(law).__name__}")
        for role, value in bindings.items():
            if not isinstance(cast(object, value), (Handle, TypedPath, str)):
                raise TypeError(f"{role}= takes a handle, a typed path or a name, not {type(value).__name__}")
        self._declarations.append(Attachment(law, dict(bindings)))
        return entry_id(type(law), bindings)

    def observe(self, observation: Observation) -> ObservationHandle:
        if not isinstance(cast(object, observation), Observation):
            raise TypeError(f"observe takes an Observation, got {type(observation).__name__}")
        self._declarations.append(ObservationDeclaration(observation))
        return ObservationHandle(entry_id(type(observation), observation.bindings()))

    def plan(self) -> PlanSet:
        """Phase 1 alone: resolve and check every declaration, with no graph. Raises ``BuildRefused``."""
        return plan_experiment(self._settings, list(self._declarations))

    def document(self) -> ExperimentDocument:
        """What this experiment declares, as its saved document. Raises ``UnsavableError`` for a setting the
        document cannot hold."""
        return document_of(self._settings, self._declarations)

    def build(self) -> BuiltExperiment:
        """Phases 1 and 2, then the saved form's record. Raises ``BuildRefused`` with every reason, each with its
        class and action line. An experiment that was loaded is checked against its lock here, and raises
        ``LoadRefused`` when it does not reproduce it."""
        plans = self.plan()
        graph, report, derived = construct(plans)
        record = record_of(self._settings, self._declarations, plans, derived)
        load_report = check_load(self._saved, record) if self._saved is not None else None
        return BuiltExperiment(graph, report, plans, record, saved=self._saved, load_report=load_report)

    def save(self, folder: str | PathLike[str], *, steps: int = DEFAULT_STEPS) -> Lock:
        """Build, then write the document, the lock and the first ``steps`` steps' outputs into ``folder``."""
        return self.build().save(folder, steps=steps)

    @classmethod
    def from_document(cls, path: str | PathLike[str]) -> Experiment:
        """The experiment a document declares, with no lock and so with no load check: its entries resolved among
        installed packages, and no code run that the document names. Raises ``LoadRefused``."""
        document, refusals = read_document(Path(path))
        if refusals or document is None:
            raise LoadRefused(refusals)
        return cls._declared(*declarations_from(document))

    @classmethod
    def load(cls, folder: str | PathLike[str]) -> Experiment:
        """The experiment saved in ``folder``, rebuilt and checked against its lock. Reads the document and runs no
        code it names; resolves every entry among installed packages and installs nothing, refusing a missing one
        or another version with the line to install; then makes the three load checks, whose outcome is
        :attr:`load_report`. Raises ``LoadRefused`` with every reason."""
        form = read_folder(Path(folder))
        experiment = cls._declared(*declarations_from(form.document, form.lock["entries"]))
        experiment._saved = form
        experiment.load_report = experiment.build().load_report
        return experiment


@dataclass(frozen=True)
class RunResult:
    """``final`` maps each node to its fields after the last step; ``history``, when asked for, stacks each field
    over the steps."""

    steps: int
    final: History
    history: History | None = None


class BuiltExperiment:
    """A built experiment: its report, its graph's canonical text, its saved form's record, and runs. A run
    continues from where the last one left the graph."""

    def __init__(self, graph: GraphManager, report: BuildReport, plans: PlanSet, record: SavedRecord, *,
                 saved: SavedForm | None = None, load_report: LoadReport | None = None) -> None:
        self._graph = graph
        self.report = report
        self._plans = plans
        self.record = record
        self._saved = saved
        self.load_report = load_report

    @property
    def graph(self) -> GraphManager:
        """The MADDENING graph, for inspection. Editing it bypasses every check the build made."""
        return self._graph

    def snapshot(self) -> str:
        return graph_text(self._graph)

    def steps_in(self, duration: Quantity) -> int:
        seconds = si(duration, TIME, what="the run's duration")
        dt = self._plans.dt
        ratio = float(seconds) / dt if isinstance(seconds, float) else math.nan
        steps = round(ratio) if math.isfinite(ratio) else 0
        if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
            raise ValueError(f"a run of {seconds!r} s is not a whole number of base steps of {dt!r} s; the nearest "
                             f"are {max(1, math.floor(ratio)) * dt!r} s and {max(1, math.ceil(ratio)) * dt!r} s")
        return steps

    def run(self, duration: Quantity, *, history: bool = False) -> RunResult:
        """Run for ``duration``, a whole number of base steps, as the report's run structure says: today always
        one compiled scan."""
        steps = self.steps_in(duration)
        final, stacked = run_steps(self._graph, steps, history=history)
        return RunResult(steps, final, stacked)

    def _fresh_history(self, steps: int) -> History:
        """The first ``steps`` steps of a graph constructed afresh from the same plans, leaving this one alone."""
        graph, _, _ = construct(self._plans)
        _, stacked = run_steps(graph, steps, history=True)
        assert stacked is not None
        return stacked

    def save(self, folder: str | PathLike[str], *, steps: int = DEFAULT_STEPS) -> Lock:
        """Write the document, the lock and the reference outputs (every field over the first ``steps`` steps, run
        from the initial state on a fresh graph) into ``folder``. Raises ``BuildRefused`` when the experiment cannot
        be saved: a setting the document cannot hold, or an entry a load elsewhere could not find."""
        path = Path(folder)
        lock = write_folder(path, self.record, self._fresh_history(steps), steps)
        self._saved = read_folder(path)
        return lock

    def verify(self) -> VerifyReport:
        """Rerun the reference outputs' steps from the initial state and compare: bit for bit in the environment the
        lock was written in, within the lock's tolerance in another."""
        if self._saved is None:
            raise ValueError("nothing to verify against: load a saved experiment, or save this one first")
        return verify_history(self._saved, self._fresh_history(self._saved.lock["reference"]["steps"]))
