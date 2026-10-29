from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any, cast, override

import numpy as np
from maddening import GraphManager

from mime.build import (
    Attachment,
    BuildRefused,
    BuildReport,
    ControllerDeclaration,
    Declaration,
    ObservationDeclaration,
    PlanSet,
    RunSettings,
    ThingDeclaration,
    ValueDeclaration,
    construct,
    entry_id,
    graph_text,
    plan_experiment,
    stage_spec,
)
from mime.build._refusals import MIME_DEFECT, Refusal, RefusalClass
from mime.control import ChunkedRun, ChunkHook, make_law
from mime.registry import ControlLaw, Controller, Interaction, Observation
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
from mime.scene import Scene, stage_text
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
from mime.units import TIME, Condition, Quantity, si

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

    ``scene`` declares the world frame once (``Scene.new(up=+Z, units="m")``): there are no default axes, and
    gravity, of magnitude ``gravity``, points along ``-up``. ``up=`` given here is refused: up is declared on the
    scene. ``dt`` is the base step, with no default; an entry's own ``rate=`` is a whole multiple of it.
    ``conditions`` are the operating conditions every material record is read at, each with its tolerance
    (``{"temperature": degC(20, plus_minus=0.5)}``).
    """

    def __init__(self, name: str, *, scene: Scene | None = None, gravity: Quantity | None = None,
                 dt: Quantity | None = None, seed: int = 0, up: Any = None,
                 conditions: Mapping[str, Condition] | None = None) -> None:
        self.name = name
        self._settings = RunSettings(name, scene, gravity, dt, seed, up, conditions)
        self._declarations: list[Declaration] = []
        self._saved: SavedForm | None = None
        self.load_report: LoadReport | None = None

    @classmethod
    def _declared(cls, settings: RunSettings, declarations: Sequence[Declaration]) -> Experiment:
        experiment = cls(settings.experiment, scene=settings.scene, gravity=settings.gravity, dt=settings.dt,
                         seed=settings.seed, up=settings.up, conditions=settings.conditions)
        experiment._declarations = list(declarations)
        return experiment

    def _add(self, thing: Any, cls: type) -> None:
        if not isinstance(thing, cls):
            raise TypeError(f"expected a {cls.__name__}, got {type(thing).__name__}")
        self._declarations.append(ThingDeclaration(thing))

    def add_structure(self, structure: Structure) -> StructureHandle:
        self._add(structure, Structure)
        geometry = type(structure.geometry) if structure.geometry is not None else None
        if geometry is None:
            return StructureHandle(structure.name)
        return StructureHandle(structure.name, getattr(geometry, "regions", ()), getattr(geometry, "surfaces", {}),
                               getattr(geometry, "directions", ()))

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

    def declare(self, path: TypedPath | str, value: Quantity, *, reason: str) -> None:
        """State a value of a thing (``declare("umr.mass", kg(8.5e-5), reason="...")``) or of a body's part
        (``"umr.parts.shell.density"``) in the experiment, with its reason: the build report and the saved form label
        it "declared in the experiment". It takes the place of the value a record or the body's parts would give, and
        the report says which. Phase 1 checks it is a value the thing has, in its units and shape, declared once."""
        self._declarations.append(ValueDeclaration(path, value, reason))

    def observe(self, observation: Observation) -> ObservationHandle:
        if not isinstance(cast(object, observation), Observation):
            raise TypeError(f"observe takes an Observation, got {type(observation).__name__}")
        self._declarations.append(ObservationDeclaration(observation))
        return ObservationHandle(entry_id(type(observation), observation.bindings()))

    def set_controller(self, controller: Controller, *, reads: Mapping[str, TypedPath | str],
                       writes: Mapping[str, TypedPath | str], mode: str | int, rate: Quantity | None = None,
                       latency: Quantity | None = None) -> None:
        """Declare the experiment's controller: ``reads`` binds each label it reads to a thing's state
        (``"umr.pose"``, ``"rpm.arm.joint_positions"``) or an observation's output (``"swim_speed:umr.speed"``), and
        ``writes`` each label it writes to an apparatus's command (``"rpm.arm.joint_positions"``), whose drive it
        replaces: such a command takes no drive in the apparatus. ``mode`` is how it runs (``"callback"``, 2: called
        on the host once every control period, between compiled chunks); ``rate`` its control period, a whole
        multiple of ``dt`` (every step when not given); ``latency`` how long its writes take to act, a whole number of
        base steps (none when not given). The build derives every external input of the graph from ``writes``, and
        the run feeds each one."""
        if not isinstance(cast(object, controller), Controller):  # a script is not type-checked
            raise TypeError(f"set_controller takes a Controller, got {type(controller).__name__}")
        self._declarations.append(ControllerDeclaration(controller, reads, writes, mode, rate, latency))

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
        law, problems = make_law(plans)
        if problems and plans.controller is not None:
            cls = plans.controller.registered.cls
            refusals = [Refusal("entry.defect", RefusalClass.INTERNAL, f"{cls.label()} ({cls.__module__}): {problem}",
                                f"a defect in {cls.label()}, defined in {cls.__module__}: report it to that package. "
                                f"If that package is MIME: {MIME_DEFECT}") for problem in problems]
            raise BuildRefused(refusals, replace(report, refusals=tuple(refusals)))
        stage = stage_text(stage_spec(plans))
        record = record_of(self._settings, self._declarations, plans, derived, stage)
        report = replace(report, stage_hash=record.stage_hash)
        load_report = check_load(self._saved, record) if self._saved is not None else None
        return BuiltExperiment(graph, report, plans, record, saved=self._saved, load_report=load_report, law=law)

    def save(self, folder: str | PathLike[str], *, steps: int = DEFAULT_STEPS) -> Lock:
        """Build, then write the document, the lock, the scene and the first ``steps`` steps' outputs into
        ``folder``."""
        return self.build().save(folder, steps=steps)

    @classmethod
    def from_document(cls, path: str | PathLike[str]) -> Experiment:
        """The experiment a document declares, with no lock and so with no load check: its entries resolved among
        installed packages, its scene's layers read from beside it, and no code run that the document names. Raises
        ``LoadRefused``."""
        document, refusals = read_document(Path(path))
        if refusals or document is None:
            raise LoadRefused(refusals)
        return cls._declared(*declarations_from(document, folder=Path(path).parent))

    @classmethod
    def load(cls, folder: str | PathLike[str]) -> Experiment:
        """The experiment saved in ``folder``, rebuilt and checked against its lock. Reads the document and runs no
        code it names; resolves every entry among installed packages and installs nothing, refusing a missing one
        or another version with the line to install; then makes the three load checks, whose outcome is
        :attr:`load_report`. Raises ``LoadRefused`` with every reason."""
        form = read_folder(Path(folder))
        experiment = cls._declared(*declarations_from(form.document, form.lock["entries"], Path(folder)))
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


#: What :meth:`BuiltExperiment.run_to` writes into its folder: the outputs, the run's record, and the run's own saved
#: form, whose lock records the environment the run was made in.
RUN_OUTPUTS = "outputs.npz"
RUN_RECORD = "run.json"
RUN_EXPERIMENT = "experiment"


@dataclass(frozen=True)
class RunWritten:
    """A run :meth:`BuiltExperiment.run_to` wrote: its steps, the files, and the record (what ``run.json`` holds)."""

    steps: int
    outputs: Path
    record_file: Path
    experiment: Path
    record: Mapping[str, Any]


class BuiltExperiment:
    """A built experiment: its report, its graph's canonical text, its saved form's record, and runs. A run
    continues from where the last one left the graph."""

    def __init__(self, graph: GraphManager, report: BuildReport, plans: PlanSet, record: SavedRecord, *,
                 saved: SavedForm | None = None, load_report: LoadReport | None = None,
                 law: ControlLaw | None = None) -> None:
        self._graph = graph
        self.report = report
        self._plans = plans
        self.record = record
        self._saved = saved
        self.load_report = load_report
        self._law = law
        self._runs = ChunkedRun(graph, plans, law)

    @property
    def graph(self) -> GraphManager:
        """The MADDENING graph, for inspection. Editing it bypasses every check the build made."""
        return self._graph

    @property
    def plans(self) -> PlanSet:
        """Phase 1's result, for inspection: each entry as planned, with its nodes, reads, deliveries and settings
        in SI."""
        return self._plans

    def held_inputs(self) -> dict[str, dict[str, Any]]:
        """Every external input of the graph at the controller's held writes, by node and field: what a scan of the
        graph outside :meth:`run` feeds, since MADDENING fills an input nobody feeds with zeros. Empty without a
        controller."""
        callback = self._runs.callback
        return {} if callback is None or self._law is None else {k: dict(v) for k, v in
                                                                  callback.fed(self._law.held).items()}

    def snapshot(self) -> str:
        return graph_text(self._graph, self.report.lags)

    @property
    def stage_text(self) -> str:
        """The scene's stage as ``save`` writes it into ``scene/scene.usda``: the same bytes for the same
        declarations."""
        return self.record.stage

    def steps_in(self, duration: Quantity) -> int:
        seconds = si(duration, TIME, what="the run's duration")
        dt = self._plans.dt
        ratio = float(seconds) / dt if isinstance(seconds, float) else math.nan
        steps = round(ratio) if math.isfinite(ratio) else 0
        if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
            raise ValueError(f"a run of {seconds!r} s is not a whole number of base steps of {dt!r} s; the nearest "
                             f"are {max(1, math.floor(ratio)) * dt!r} s and {max(1, math.ceil(ratio)) * dt!r} s")
        return steps

    def run(self, duration: Quantity, *, history: bool = False, hook: ChunkHook | None = None,
            chunk: Quantity | None = None) -> RunResult:
        """Run for ``duration``, a whole number of base steps, as the report's run structure says: one compiled scan,
        or compiled chunks of one control period with the controller called between them. ``hook`` is called after
        every chunk with a :class:`~mime.control.Chunk`, its end and the state there; without a controller the run is
        chunked for it in chunks of ``chunk``, which changes nothing it computes. A chunked run is a whole number of
        chunks."""
        steps = self.steps_in(duration)
        size: int | None = None
        if chunk is not None:
            if hook is None or self._runs.chunk is not None:
                raise ValueError("chunk= sizes the chunks of a run with a hook and no controller; a controller's "
                                 "chunks are its control period")
            size = self.steps_in(chunk)
        elif hook is not None and self._runs.chunk is None:
            raise ValueError("a hook needs chunks: pass chunk=, a whole number of base steps")
        final, stacked = self._runs.run(steps, history=history, hook=hook, chunk=size)
        return RunResult(steps, final, stacked)

    def run_to(self, out: str | PathLike[str], duration: Quantity, *, history: bool = False,
               source: str | PathLike[str] | None = None, load_seconds: float | None = None) -> RunWritten:
        """Run for ``duration`` and write into ``out`` what ``mime run`` does: ``outputs.npz`` (every field after the
        last step, and with ``history`` at every step), ``run.json`` (the steps, the time each stage took, the
        environment this run was made in and the load report) and the run's own saved form, ``experiment/``, whose lock
        records that environment, so that a later run in it is verified bit for bit. ``source`` is the folder the
        experiment was loaded from and ``load_seconds`` the time loading and building took, both for the record. The
        precision a lock records is applied before loading (``mime.env.apply_lock_precision``), as ``mime run`` does."""
        import json
        import time

        started = time.perf_counter()
        result = self.run(duration, history=history)
        ran = time.perf_counter()
        target = Path(out)
        target.mkdir(parents=True, exist_ok=True)
        arrays = {f"final/{node}/{name}": np.asarray(value) for node, fields in result.final.items()
                  for name, value in fields.items()}
        for node, fields in (result.history or {}).items():
            arrays.update({f"history/{node}/{name}": np.asarray(value) for name, value in fields.items()})
        np.savez(target / RUN_OUTPUTS, allow_pickle=False, **arrays)
        self.save(target / RUN_EXPERIMENT)
        record: dict[str, Any] = {
            "folder": str(source) if source is not None else None, "duration": si(duration, TIME, what="the duration"),
            "steps": result.steps,
            "seconds": {"load_and_build": round(load_seconds, 3) if load_seconds is not None else None,
                        "run": round(ran - started, 3), "save": round(time.perf_counter() - ran, 3)},
            "environment": self.record.environment,
            "load_report": self.load_report.text() if self.load_report is not None else None}
        (target / RUN_RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return RunWritten(result.steps, target / RUN_OUTPUTS, target / RUN_RECORD, target / RUN_EXPERIMENT, record)

    def _fresh_history(self, steps: int) -> History:
        """The first ``steps`` steps of a graph constructed afresh from the same plans, leaving this one alone: run in
        whole control periods, and the steps past ``steps`` dropped."""
        graph, _, _ = construct(self._plans)
        runs = ChunkedRun(graph, self._plans, self._law)
        period = runs.chunk or 1
        _, stacked = runs.run(-(-steps // period) * period, history=True)
        assert stacked is not None
        return {node: {name: value[:steps] for name, value in fields.items()} for node, fields in stacked.items()}

    def save(self, folder: str | PathLike[str], *, steps: int = DEFAULT_STEPS) -> Lock:
        """Write the document, the lock, the scene (``scene/scene.usda`` and the asset layers it references, under
        ``scene/layers/``) and the reference outputs (every field over the first ``steps`` steps, run from the
        initial state on a fresh graph) into ``folder``. Raises ``BuildRefused`` when the experiment cannot
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
