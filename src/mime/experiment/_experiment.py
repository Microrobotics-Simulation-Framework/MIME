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
from mime.build._construct import report_of
from mime.build._report import NodeLine
from mime.build._plan import Section
from mime.build._refusals import (
    MIME_DEFECT,
    ArgumentRefused,
    ArgumentTypeRefused,
    Refusal,
    RefusalClass,
    RunFailed,
)
from mime.control import ChunkedRun, ChunkHook, make_law
from mime.errors import public
from mime.registry import ControlLaw, Controller, Interaction, Observation, OutputKind
from mime.results import (
    RUN_MANIFEST,
    FlagLayout,
    FlagTrack,
    Layout,
    ObservationLayout,
    OutputLayout,
    OutputRef,
    RunResult,
    Samples,
)
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
    did_you_mean,
)
from mime.units import TIME, Condition, Quantity, si

type History = dict[str, dict[str, np.ndarray]]


class ObservationHandle:
    """What ``observe`` returns: the observation's id, and each of its declared outputs by name (``swim.speed``), as
    ``run(record=[...])`` takes it. A run's result is indexed by it (``result[swim]``)."""

    __slots__ = ("id", "outputs")

    def __init__(self, id: str, outputs: Sequence[str] = ()) -> None:
        self.id = id
        self.outputs = tuple(outputs)

    def __getattr__(self, name: str) -> OutputRef:
        if name.startswith("_") or name in self.__slots__:
            raise AttributeError(name)
        if name in self.outputs:
            return OutputRef(self.id, name)
        raise AttributeError(f"the observation {self.id} has no output {name!r}; its outputs are "
                             f"{', '.join(self.outputs) or 'none'}")

    @override
    def __repr__(self) -> str:
        return f"ObservationHandle({self.id!r})"


#: How the build report's sections name a node's owner in a message.
_OWNER_KIND = {"observation": "the observation", "interaction": "the interaction", "provider": "the field",
               "drive": "the drive", "controller": "the controller"}


def _owner(nodes: Sequence[NodeLine], node: str) -> str:
    held = next((line for line in nodes if line.node == node), None)
    if held is None:
        return node
    kind = _OWNER_KIND.get(held.section)
    return f"{kind} {held.owner}" if kind else held.owner


def _floating(values: np.ndarray) -> bool:
    return np.issubdtype(values.dtype, np.inexact)


def non_finite(final: History, stacked: History | None, nodes: Sequence[NodeLine], start: int, dt: float, *,
               kept_all: bool) -> Refusal | None:
    """The refusal of a run whose state at its end is not finite numbers, naming the first step, thing and field the
    run kept where it went so, and how to find them where it kept none. A state that goes non-finite and comes back
    is not seen: the end is checked, and the steps only when it is not finite."""
    ended = sorted((node, name) for node, fields in final.items() for name, value in fields.items()
                   if _floating(np.asarray(value)) and not np.isfinite(np.asarray(value)).all())
    if not ended:
        return None
    observed = {line.node for line in nodes if line.section == "observation"}
    ended = [key for key in ended if key[0] not in observed] or ended
    why = ("most often a base step too large for the fastest physics in the graph, which an explicit step amplifies "
           "until it overflows; or a value given as zero that something divides by")
    smaller = f"a smaller dt= in Experiment(..., dt=s({dt / 10!r}))"
    if stacked is not None:
        first: dict[tuple[str, str], int] = {}
        for node, fields in stacked.items():
            for name, value in fields.items():
                held = np.asarray(value)
                if _floating(held) and len(held):
                    bad = ~np.isfinite(held.reshape(len(held), -1)).all(axis=1)
                    if bad.any():
                        first[(node, name)] = int(np.argmax(bad))
        if first:
            step = min(first.values())
            where = [key for key, at in first.items() if at == step]
            where = [key for key in where if key[0] not in observed] or where
            count = start + step + 1
            named = ", ".join(f"{_owner(nodes, node)}'s {name}" for node, name in sorted(where))
            among = "" if kept_all else " (among the fields this run recorded; history=True keeps every one)"
            return Refusal("run.non-finite", RefusalClass.USER,
                           f"the run's state stopped being finite numbers at step {count} (t = {count * dt:.6g} s), "
                           f"first in {named}{among}",
                           f"run again with {smaller}, and check the first step's inputs; run(..., history=True) "
                           f"keeps every step", why=why)
    named = ", ".join(f"{_owner(nodes, node)}'s {name}" for node, name in ended)
    return Refusal("run.non-finite", RefusalClass.USER,
                   f"at the end of the run, {named} {'is' if len(ended) == 1 else 'are'} not finite numbers; the run "
                   f"kept no step's state, so its first step is not known",
                   f"find it: run again with history=True, run(..., history=True), which keeps every step; or with "
                   f"{smaller}", why=why)


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
            raise ArgumentTypeRefused([Refusal(
                "call.wrong-type", RefusalClass.USER,
                f"add_{cls.__name__.lower()} takes a {cls.__name__}, not {type(thing).__name__}",
                f"pass one: exp.add_{cls.__name__.lower()}({cls.__name__}(...))")])
        self._declarations.append(ThingDeclaration(thing))

    @public
    def add_structure(self, structure: Structure) -> StructureHandle:
        self._add(structure, Structure)
        geometry = type(structure.geometry) if structure.geometry is not None else None
        if geometry is None:
            return StructureHandle(structure.name)
        return StructureHandle(structure.name, getattr(geometry, "regions", ()), getattr(geometry, "surfaces", {}),
                               getattr(geometry, "directions", ()))

    @public
    def add_medium(self, medium: Medium) -> MediumHandle:
        self._add(medium, Medium)
        return MediumHandle(medium.name)

    @public
    def add_body(self, body: Body) -> BodyHandle:
        self._add(body, Body)
        return BodyHandle(body.name)

    @public
    def add_apparatus(self, apparatus: Apparatus) -> ApparatusHandle:
        self._add(apparatus, Apparatus)
        return ApparatusHandle(apparatus.name, apparatus.fields)

    @public
    def attach(self, law: Interaction, /, **bindings: Handle | TypedPath | str) -> str:
        """Bind an interaction to things by its roles; returns its id (its registered name and the things bound to
        it). Every role is bound explicitly: there is no default binding."""
        if not isinstance(cast(object, law), Interaction):  # a script is not type-checked
            raise ArgumentTypeRefused([Refusal(
                "call.wrong-type", RefusalClass.USER,
                f"attach takes an Interaction, not {type(law).__name__}",
                "pass an interaction and its bindings: exp.attach(Gravity(), body=ball, medium=water)")])
        for role, value in bindings.items():
            if not isinstance(cast(object, value), (Handle, TypedPath, str)):
                raise ArgumentTypeRefused([Refusal(
                    "call.wrong-type", RefusalClass.USER,
                    f"{role}= takes a handle, a typed path or a name, not {type(value).__name__}",
                    f"bind it to a thing: {role}= the handle add_body or add_medium returned, or its name")])
        self._declarations.append(Attachment(law, dict(bindings)))
        return entry_id(type(law), bindings)

    @public
    def declare(self, path: TypedPath | str, value: Quantity, *, reason: str) -> None:
        """State a value of a thing (``declare("umr.mass", kg(8.5e-5), reason="...")``) or of a body's part
        (``"umr.parts.shell.density"``) in the experiment, with its reason: the build report and the saved form label
        it "declared in the experiment". It takes the place of the value a record or the body's parts would give, and
        the report says which. Phase 1 checks it is a value the thing has, in its units and shape, declared once."""
        self._declarations.append(ValueDeclaration(path, value, reason))

    @public
    def observe(self, observation: Observation) -> ObservationHandle:
        if not isinstance(cast(object, observation), Observation):
            raise ArgumentTypeRefused([Refusal(
                "call.wrong-type", RefusalClass.USER,
                f"observe takes an Observation, not {type(observation).__name__}",
                "pass an observation: exp.observe(SwimSpeed(body, along=X))")])
        self._declarations.append(ObservationDeclaration(observation))
        return ObservationHandle(entry_id(type(observation), observation.bindings()),
                                 tuple(output.field for output in type(observation).outputs))

    @public
    def set_controller(self, controller: Controller, *, reads: Mapping[str, TypedPath | str],
                       writes: Mapping[str, TypedPath | str], mode: str | int, rate: Quantity | None = None,
                       latency: Quantity | None = None) -> None:
        """Declare the experiment's controller: ``reads`` binds each label it reads to a thing's state
        (``"umr.pose"``, ``"rpm.arm.joint_positions"``) or an observation's output (``"swim_speed:umr.speed"``), and
        ``writes`` each label it writes to an apparatus's command (``"rpm.arm.joint_positions"``), whose drive it
        replaces: such a command takes no drive in the apparatus. ``mode`` is how it runs (``"graph"``, 1: inside the
        compiled step, as a node at its own rate, so a derivative of the run passes through it; ``"callback"``, 2:
        called on the host once every control period, between compiled chunks); ``rate`` its control period, a whole
        multiple of ``dt`` (every step when not given); ``latency`` how long its writes take to act, a whole number of
        base steps (none when not given). As a callback, the build derives every external input of the graph from
        ``writes``, and the run feeds each one; inside the compiled step, its node's edges feed its commands."""
        if not isinstance(cast(object, controller), Controller):  # a script is not type-checked
            raise ArgumentTypeRefused([Refusal(
                "call.wrong-type", RefusalClass.USER,
                f"set_controller takes a Controller, not {type(controller).__name__}",
                "pass a registered controller: exp.set_controller(ArmTracking(...), reads=..., writes=..., mode=...)")])
        self._declarations.append(ControllerDeclaration(controller, reads, writes, mode, rate, latency))

    @public
    def plan(self) -> PlanSet:
        """Phase 1 alone: resolve and check every declaration, with no graph. Raises ``BuildRefused``."""
        return plan_experiment(self._settings, list(self._declarations))

    @public
    def document(self) -> ExperimentDocument:
        """What this experiment declares, as its saved document. Raises ``UnsavableError`` for a setting the
        document cannot hold."""
        return document_of(self._settings, self._declarations)

    @public
    def build(self) -> BuiltExperiment:
        """Phases 1 and 2, then the saved form's record. Raises ``BuildRefused`` with every reason, each with its
        class and action line. An experiment that was loaded is checked against its lock here, and raises
        ``LoadRefused`` when it does not reproduce it."""
        plans = self.plan()
        law, problems = make_law(plans)
        if problems and plans.controller is not None:
            cls = plans.controller.registered.cls
            refusals = [Refusal("entry.defect", RefusalClass.INTERNAL, f"{cls.label()} ({cls.__module__}): {problem}",
                                f"a defect in {cls.label()}, defined in {cls.__module__}: report it to that package. "
                                f"If that package is MIME: {MIME_DEFECT}") for problem in problems]
            raise BuildRefused(refusals, report_of(plans, plans.order, refusals))
        graph, report, derived = construct(plans, law)
        stage = stage_text(stage_spec(plans))
        record = record_of(self._settings, self._declarations, plans, derived, stage)
        report = replace(report, stage_hash=record.stage_hash)
        load_report = check_load(self._saved, record) if self._saved is not None else None
        return BuiltExperiment(graph, report, plans, record, saved=self._saved, load_report=load_report, law=law)

    @public
    def save(self, folder: str | PathLike[str], *, steps: int = DEFAULT_STEPS) -> Lock:
        """Build, then write the document, the lock, the scene and the first ``steps`` steps' outputs into
        ``folder``."""
        return self.build().save(folder, steps=steps)

    @classmethod
    @public
    def from_document(cls, path: str | PathLike[str]) -> Experiment:
        """The experiment a document declares, with no lock and so with no load check: its entries resolved among
        installed packages, its scene's layers read from beside it, and no code run that the document names. Raises
        ``LoadRefused``."""
        document, refusals = read_document(Path(path))
        if refusals or document is None:
            raise LoadRefused(refusals)
        return cls._declared(*declarations_from(document, folder=Path(path).parent))

    @classmethod
    @public
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


#: What :meth:`BuiltExperiment.run_to` writes into its folder: the outputs, their manifest (``outputs.json``), the run's
#: record, and the run's own saved form, whose lock records the environment the run was made in.
RUN_OUTPUTS = "outputs.npz"
RUN_RECORD = "run.json"
RUN_EXPERIMENT = "experiment"


@dataclass(frozen=True)
class RunWritten:
    """A run :meth:`BuiltExperiment.run_to` wrote: its steps, the files, the record (what ``run.json`` holds) and the
    result, as :meth:`RunResult.read` reads it back from the folder."""

    steps: int
    outputs: Path
    record_file: Path
    experiment: Path
    record: Mapping[str, Any]
    manifest: Path
    result: RunResult


def run_layout(plans: PlanSet) -> Layout:
    """Each observation's declared outputs and each flag an entry raises on the rung attached, placed on the graph's
    nodes; an output's window starts at its setting's value in SI, as the build hands it to the node."""
    observations: list[ObservationLayout] = []
    flags: list[FlagLayout] = []
    for plan in plans.entries:
        cls = plan.registered.cls
        rung = getattr(plan.entry, "rung", None)
        flags += [FlagLayout(plan.owner, cls.label(), plan.node_name(flag.node), flag.field, flag.meaning)
                  for flag in cls.raises if flag.rung is None or flag.rung == rung]
        if plan.section is Section.OBSERVATION and issubclass(cls, Observation):
            observations.append(ObservationLayout(plan.owner, cls.label(), tuple(
                OutputLayout(plan.owner, output.field, plan.node_name(output.node), output.field, output.kind,
                             output.dimension, output.shape, output.description, output.reduces, output.rule,
                             float(cast(float, plan.settings[output.after])) if output.after is not None else None)
                for output in cls.outputs)))
    return Layout(plans.settings.experiment, plans.dt, tuple(observations), tuple(flags))


def _raised(values: np.ndarray) -> np.ndarray:
    """A flag's values over steps (one row each) as whether it was raised at each."""
    held = np.asarray(values)
    return (held.reshape(len(held), -1) != 0).any(axis=1)


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
        self.layout = run_layout(plans)

    @property
    def graph(self) -> GraphManager:
        """The MADDENING graph, for inspection. Editing it bypasses every check the build made."""
        return self._graph

    @property
    def plans(self) -> PlanSet:
        """Phase 1's result, for inspection: each entry as planned, with its nodes, reads, deliveries and settings
        in SI."""
        return self._plans

    @public
    def held_inputs(self) -> dict[str, dict[str, Any]]:
        """Every external input of the graph at the controller's held writes, by node and field: what a scan of the
        graph outside :meth:`run` feeds, since MADDENING fills an input nobody feeds with zeros. Empty without a
        controller."""
        callback = self._runs.callback
        return {} if callback is None or self._law is None else {k: dict(v) for k, v in
                                                                  callback.fed(self._law.held).items()}

    @public
    def snapshot(self) -> str:
        return graph_text(self._graph, self.report.lags)

    @property
    def stage_text(self) -> str:
        """The scene's stage as ``save`` writes it into ``scene/scene.usda``: the same bytes for the same
        declarations."""
        return self.record.stage

    @public
    def steps_in(self, duration: Quantity) -> int:
        seconds = si(duration, TIME, what="the run's duration")
        dt = self._plans.dt
        ratio = float(seconds) / dt if isinstance(seconds, float) else math.nan
        steps = round(ratio) if math.isfinite(ratio) else 0
        if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
            lower, upper = max(1, math.floor(ratio)) * dt, max(1, math.ceil(ratio)) * dt
            raise ArgumentRefused([Refusal(
                "run.duration", RefusalClass.USER,
                f"a run of {seconds!r} s is not a whole number of base steps of {dt!r} s",
                f"run a whole number of steps: run(s({lower!r})) or run(s({upper!r})), or change dt= in "
                "Experiment(...)",
                why="a run steps the graph by whole base steps, and MIME does not round a duration")])
        return steps

    def _recording(self, record: Sequence[OutputRef | ObservationHandle | str]) -> list[OutputLayout]:
        """The per-step outputs ``record`` names: an output (a handle's ``swim.speed``, or ``"<id>.<output>"``), or an
        observation (its handle, or its id) for every per-step output it declares."""
        per_step = {output.path: output for output in self.layout.per_step()}
        ids = {held.id: held for held in self.layout.observations}
        found: dict[str, OutputLayout] = {}
        for item in record:
            held = cast(object, item)  # a script is not type-checked
            if isinstance(held, OutputRef):
                paths = [held.path]
            elif isinstance(held, ObservationHandle) or (isinstance(held, str) and held in ids):
                id = held.id if isinstance(held, ObservationHandle) else held
                if id not in ids:
                    raise ArgumentRefused([Refusal(
                        "run.record", RefusalClass.USER,
                        f"record= names the observation {id!r}, which this experiment does not observe; it observes "
                        f"{', '.join(ids) or 'nothing'}",
                        f"name one it observes: record=[{', '.join(map(repr, ids)) or '...'}], or observe it first "
                        "with exp.observe(...)")])
                paths = [o.path for o in ids[id].outputs if o.kind is OutputKind.PER_STEP]
            elif isinstance(held, str):
                paths = [held]
            else:
                raise ArgumentTypeRefused([Refusal(
                    "run.record", RefusalClass.USER,
                    f"record= holds {held!r}, which is not an output or an observation",
                    "name outputs and observations: record=[swim.speed], or record=['swim_speed:umr.speed']")])
            for path in paths:
                held = self.layout.output(path)
                if held is not None and held.kind is OutputKind.AT_END:
                    raise ArgumentRefused([Refusal(
                        "run.record", RefusalClass.USER,
                        f"record= names {path}, an output at the end, which the graph reduces and every run gives",
                        f"drop {path!r} from record=; read it as result[...].{path.rsplit('.', 1)[-1]} after any run")])
                if path not in per_step:
                    raise ArgumentRefused([Refusal(
                        "run.record", RefusalClass.USER,
                        f"record= names {path!r}, which is no per-step output of this experiment's observations",
                        f"name a per-step output in record=[...]. {did_you_mean(path, per_step)}")])
                found[path] = per_step[path]
        return list(found.values())

    def _spacing(self, every: Quantity | None) -> int:
        if every is None:
            return 1
        seconds = si(every, TIME, what="every=")
        ratio = float(seconds) / self._plans.dt if isinstance(seconds, float) else math.nan
        steps = round(ratio) if math.isfinite(ratio) else 0
        if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
            raise ArgumentRefused([Refusal(
                "run.every", RefusalClass.USER,
                f"every= is {seconds!r} s, not a whole number of base steps of {self._plans.dt!r} s",
                "space the samples by whole steps: "
                f"every=s({max(1, round(ratio) if math.isfinite(ratio) else 1) * self._plans.dt!r})",
                why="a series keeps one sample every so many base steps")])
        return steps

    @public
    def run(self, duration: Quantity, *, history: bool = False,
            record: Sequence[OutputRef | ObservationHandle | str] = (), every: Quantity | None = None,
            hook: ChunkHook | None = None, chunk: Quantity | None = None) -> RunResult:
        """Run for ``duration``, a whole number of base steps, as the report's run structure says: one compiled scan,
        or compiled chunks of one control period with the controller called between them. ``hook`` is called after
        every chunk with a :class:`~mime.control.Chunk`, its end and the state there; without a controller the run is
        chunked for it in chunks of ``chunk``, which changes nothing it computes. A chunked run is a whole number of
        chunks.

        The result (:class:`~mime.results.RunResult`) holds every output at the end each observation declares, and
        the series of the per-step outputs ``record`` names (``record=[swim.speed]``), one sample every ``every`` (a
        whole number of base steps, every step when not given), with every flag the entries raise at every step. A
        run that records nothing scans as it did before, keeping no step's state, and knows each flag at its last step
        only. ``history`` keeps every field at every step, and with it every per-step output is recorded. ``every``
        spaces the samples kept and does not reduce the cost of recording, which is the whole state's history per
        step whatever is kept."""
        steps = self.steps_in(duration)
        recorded = self._recording(record)
        if every is not None and not (recorded or history):
            raise ArgumentRefused([Refusal(
                "run.every", RefusalClass.USER,
                "every= spaces the samples of the series record= names, and nothing is recorded",
                "name them: run(..., record=[swim.speed], every=...), or drop every=")])
        spacing = self._spacing(every)
        size: int | None = None
        if chunk is not None:
            if hook is None or self._runs.chunk is not None:
                raise ArgumentRefused([Refusal(
                    "run.chunks", RefusalClass.USER,
                    "chunk= sizes the chunks of a run with a hook and no controller; a controller's chunks are its "
                    "control period",
                    "drop chunk=, or pass hook= with it and no controller")])
            size = self.steps_in(chunk)
        elif hook is not None and self._runs.chunk is None:
            raise ArgumentRefused([Refusal(
                "run.chunks", RefusalClass.USER,
                "a hook needs chunks: it is called between them, and this run has none",
                "pass chunk=, a whole number of base steps: run(..., hook=hook, chunk=ms(10))")])
        start = self._runs.steps
        kept = history or bool(recorded)
        keep = None if history or not recorded else frozenset(
            {(output.node, output.field) for output in recorded}
            | {(flag.node, flag.field) for flag in self.layout.flags})
        final, stacked = self._runs.run(steps, history=kept, hook=hook, chunk=size, keep=keep)
        tracks: dict[str, FlagTrack] = {}
        series: dict[str, Samples] = {}
        if stacked is not None:
            counts = np.arange(start + 1, start + steps + 1, dtype=np.int64)
            chosen = counts % spacing == 0
            for output in (self.layout.per_step() if history else recorded):
                series[output.path] = Samples(counts[chosen], np.asarray(stacked[output.node][output.field])[chosen])
            for flag in self.layout.flags:
                raised = _raised(stacked[flag.node][flag.field])
                tracks[flag.path] = FlagTrack(raised, bool(raised[-1]))
        else:
            tracks = {flag.path: FlagTrack(None, bool(_raised(np.asarray(final[flag.node][flag.field])[None])[0]))
                      for flag in self.layout.flags}
        result = RunResult(steps, final, stacked if history else None, layout=self.layout, start_step=start,
                           recorded=series, tracks=tracks, every=spacing if stacked is not None else None)
        refusal = non_finite(final, stacked, self.report.nodes, start, self._plans.dt, kept_all=history)
        if refusal is not None:
            raise RunFailed([refusal], result)
        return result

    @public
    def run_to(self, out: str | PathLike[str], duration: Quantity, *, history: bool = False,
               record: Sequence[OutputRef | ObservationHandle | str] = (), every: Quantity | None = None,
               source: str | PathLike[str] | None = None, load_seconds: float | None = None) -> RunWritten:
        """Run for ``duration`` and write into ``out`` what ``mime run`` does: ``outputs.npz`` (every field after the
        last step, with ``history`` at every step, the series ``record`` names, one every ``every``, and the flags at
        every step when anything is recorded), ``outputs.json`` beside it (its manifest: every output with its unit,
        window and flags, each output at the end written out, so the folder is readable without MIME), ``run.json``
        (the steps, the time each stage took, the environment this run was made in and the load report) and the run's
        own saved form, ``experiment/``, whose lock records that environment, so that a later run in it is verified
        bit for bit. ``source`` is the folder the experiment was loaded from and ``load_seconds`` the time loading and
        building took, both for the record. The precision a lock records is applied before loading
        (``mime.env.apply_lock_precision``), as ``mime run`` does. :meth:`RunResult.read` reads the folder back."""
        import json
        import time

        started = time.perf_counter()
        result = self.run(duration, history=history, record=record, every=every)
        ran = time.perf_counter()
        target = Path(out)
        target.mkdir(parents=True, exist_ok=True)
        np.savez(target / RUN_OUTPUTS, allow_pickle=False, **result.arrays())
        (target / RUN_MANIFEST).write_text(json.dumps(result.manifest(), indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
        self.save(target / RUN_EXPERIMENT)
        written: dict[str, Any] = {
            "folder": str(source) if source is not None else None, "duration": si(duration, TIME, what="the duration"),
            "steps": result.steps, "recorded": list(result.recorded),
            "every": result.every * self._plans.dt if result.every is not None else None,
            "seconds": {"load_and_build": round(load_seconds, 3) if load_seconds is not None else None,
                        "run": round(ran - started, 3), "save": round(time.perf_counter() - ran, 3)},
            "environment": self.record.environment,
            "load_report": self.load_report.text() if self.load_report is not None else None}
        (target / RUN_RECORD).write_text(json.dumps(written, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return RunWritten(result.steps, target / RUN_OUTPUTS, target / RUN_RECORD, target / RUN_EXPERIMENT, written,
                          target / RUN_MANIFEST, RunResult.read(target))

    def _fresh_history(self, steps: int) -> History:
        """The first ``steps`` steps of a graph constructed afresh from the same plans, leaving this one alone: run in
        whole control periods, and the steps past ``steps`` dropped."""
        graph, _, _ = construct(self._plans, self._law)
        runs = ChunkedRun(graph, self._plans, self._law)
        period = runs.chunk or 1
        _, stacked = runs.run(-(-steps // period) * period, history=True)
        assert stacked is not None
        return {node: {name: value[:steps] for name, value in fields.items()} for node, fields in stacked.items()}

    @public
    def save(self, folder: str | PathLike[str], *, steps: int = DEFAULT_STEPS) -> Lock:
        """Write the document, the lock, the scene (``scene/scene.usda`` and the asset layers it references, under
        ``scene/layers/``) and the reference outputs (every field over the first ``steps`` steps, run from the
        initial state on a fresh graph) into ``folder``. Raises ``BuildRefused`` when the experiment cannot
        be saved: a setting the document cannot hold, or an entry a load elsewhere could not find."""
        path = Path(folder)
        lock = write_folder(path, self.record, self._fresh_history(steps), steps)
        self._saved = read_folder(path)
        return lock

    @public
    def verify(self) -> VerifyReport:
        """Rerun the reference outputs' steps from the initial state and compare: bit for bit in the environment the
        lock was written in, within the lock's tolerance in another."""
        if self._saved is None:
            raise ArgumentRefused([Refusal(
                "verify.nothing-saved", RefusalClass.USER,
                "nothing to verify against: this experiment was neither loaded nor saved",
                "load a saved one, Experiment.load(folder).build().verify(), or save this one first: "
                "built.save(folder)")])
        return verify_history(self._saved, self._fresh_history(self._saved.lock["reference"]["steps"]))
