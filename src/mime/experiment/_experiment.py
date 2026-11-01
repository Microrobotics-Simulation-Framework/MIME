from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
from maddening import GraphManager

from mime.build import (
    LicenceDeclaration,
    PermissionDeclaration,
    CONVENTIONAL_GRAVITY,
    Attachment,
    BuildRefused,
    BuildReport,
    ControllerDeclaration,
    Declaration,
    ObservationDeclaration,
    PlanSet,
    RunSettings,
    SourceDeclaration,
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
from mime.build._plan import EntryPlan, Section
from mime.build._refusals import (
    ArgumentRefused,
    ArgumentTypeRefused,
    Refusal,
    RefusalClass,
    RunFailed,
    entry_defect,
)
from mime.build import StageBinding, stage_binding
from mime.control import Chunk, ChunkedRun, ChunkHook, law_and_models
from mime.errors import public
from mime.registry import ControlLaw, Controller, Interaction, Observation, OutputKind
from mime.terms import CustomLicence, Permission
from mime.results import (
    REPORT_FORMAT,
    RUN_MANIFEST,
    RUN_REPORT,
    FlagLayout,
    FlagTrack,
    Layout,
    ObservationLayout,
    OutputLayout,
    OutputRef,
    RunResult,
    Samples,
    read_watches,
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
from mime.saved._folder import LoadRefused, stage_writers
from mime.scene import STAGE_WRITER, STAGE_WRITERS, Scene, stage_text
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
from mime.validity import DEFAULT_PROFILE, Basis, Profile
from mime.vocabulary import JOINT_VELOCITIES

if TYPE_CHECKING:
    from mime.live import LiveSession
    from mime.viz import RecordingSettings, StageSamples

type History = dict[str, dict[str, np.ndarray]]


def _plain(value: Any) -> Any:
    """A node's parameter as JSON holds it: a small array as lists, a larger one by its shape and dtype, anything
    else MADDENING's text of it."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in cast(Mapping[Any, Any], value).items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in cast(Sequence[Any], value)]
    held = np.asarray(value) if hasattr(value, "shape") and hasattr(value, "dtype") else None
    if held is not None:
        return held.tolist() if held.size <= 16 else f"array of shape {tuple(held.shape)}, {held.dtype}"
    return repr(value)


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


#: Where an earlier run of the same built left its state not finite: the step it was first seen not finite at, where
#: the run kept it, and the step the run ended at.
@dataclass(frozen=True)
class NonFinite:
    first: int | None
    end: int


def non_finite(final: History, stacked: History | None, nodes: Sequence[NodeLine], start: int, dt: float, *,
               kept_all: bool, already: NonFinite | None = None) -> Refusal | None:
    """The refusal of a run whose state at its end is not finite numbers, naming the first step, thing and field the
    run kept where it went so, and how to find them where it kept none. A run that started from a state already not
    finite (``already``, an earlier run of the same built that ended so) says so, and when. A state that goes
    non-finite and comes back is not seen: the end is checked, and the steps only when it is not finite. The actions
    are a script's; the CLI writes its own."""
    ended = sorted((node, name) for node, fields in final.items() for name, value in fields.items()
                   if _floating(np.asarray(value)) and not np.isfinite(np.asarray(value)).all())
    if not ended:
        return None
    smaller = dt / 10
    observed = {line.node for line in nodes if line.section == "observation"}
    ended = [key for key in ended if key[0] not in observed] or ended
    why = ("most often a base step too large for the fastest physics in the graph, which an explicit step amplifies "
           "until it overflows; or a value given as zero that something divides by")
    if already is not None:
        seen = f"first at step {already.first} (t = {already.first * dt:.6g} s)" if already.first is not None else \
            f"by step {already.end}, where the run before it ended"
        named = ", ".join(f"{_owner(nodes, node)}'s {name}" for node, name in ended)
        return Refusal("run.non-finite", RefusalClass.USER,
                       f"the state was already not finite when this run started, at step {start} ({seen}); at its "
                       f"end {named} {'is' if len(ended) == 1 else 'are'} not finite numbers",
                       f"rebuild and run again keeping every step, exp.build().run(..., history=True), since this "
                       f"built's state is no longer finite; or with a smaller dt= in "
                       f"Experiment(..., dt=s({smaller!r}))",
                       why="a run continues from where the last one left the state, and an earlier run of this built "
                           "left it not finite")
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
                           f"run again with a smaller dt= in Experiment(..., dt=s({smaller!r})), and check the "
                           f"first step's inputs", why=why)
    named = ", ".join(f"{_owner(nodes, node)}'s {name}" for node, name in ended)
    return Refusal("run.non-finite", RefusalClass.USER,
                   f"at the end of the run, {named} {'is' if len(ended) == 1 else 'are'} not finite numbers; the run "
                   f"kept no step's state, so its first step is not known",
                   f"find it: rebuild and run again keeping every step, exp.build().run(..., history=True); or "
                   f"run with a smaller dt= in Experiment(..., dt=s({smaller!r}))", why=why)


def first_non_finite(refusal: Refusal) -> int | None:
    """The step a ``run.non-finite`` refusal names as the first not finite, where it names one."""
    found = re.search(r"stopped being finite numbers at step (\d+)", refusal.message)
    return int(found.group(1)) if found else None


class Experiment:
    """An experiment as the script declares it. Declaring touches no graph: ``add_*``, ``attach`` and ``observe``
    record declarations and return handles or ids, and nothing is checked until :meth:`plan` or :meth:`build`, which
    refuse with every reason at once.

    ``scene`` declares the world frame once (``Scene.new(up=+Z, units="m")``): there are no default axes, and
    gravity, of magnitude ``gravity``, points along ``-up``; standard gravity by convention when not given (the report
    says so), ``m_per_s2(0)`` for none. ``up=`` given here is refused: up is declared on the scene. ``dt`` is the base
    step, with no default; an entry's own ``rate=`` is a whole multiple of it. ``conditions`` are the operating
    conditions every material record is read at, each with its tolerance (``{"temperature": degC(20,
    plus_minus=0.5)}``).

    ``profile`` is how much the build asks (``mime.validity.Profile``): ``"exploratory"`` (the default) waives what
    the experiment leaves unstated with a pending reason, recorded in the saved form and stamped on every output;
    ``"research"`` needs each with its reason and refuses a pending one by name; ``"strict"`` is a1's and refused
    until then. Validity limits and input errors are refused in every profile. ``built.report.pending()`` and
    ``mime pending <folder>`` list what is pending, with the lines to paste.
    """

    def __init__(self, name: str, *, scene: Scene | None = None, gravity: Quantity = CONVENTIONAL_GRAVITY,
                 dt: Quantity | None = None, seed: int = 0, up: Any = None,
                 conditions: Mapping[str, Condition] | None = None,
                 profile: str | Profile = DEFAULT_PROFILE.value) -> None:
        self.name = name
        self._settings = RunSettings(name, scene, gravity, dt, seed, up, conditions,
                                     profile.value if isinstance(profile, Profile) else profile)
        self._declarations: list[Declaration] = []
        self._saved: SavedForm | None = None
        self.load_report: LoadReport | None = None

    @classmethod
    def _declared(cls, settings: RunSettings, declarations: Sequence[Declaration]) -> Experiment:
        experiment = cls(settings.experiment, scene=settings.scene, gravity=settings.gravity, dt=settings.dt,
                         seed=settings.seed, up=settings.up, conditions=settings.conditions, profile=settings.profile)
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
    def declare(self, path: TypedPath | str, value: Quantity, *, reason: Basis | None = None) -> None:
        """State a value of a thing (``declare("umr.mass", kg(8.5e-5), reason="...")``) or of a body's part
        (``"umr.parts.shell.density"``) in the experiment, with its reason: the build report and the saved form label
        it "declared in the experiment". It takes the place of the value a record or the body's parts would give, and
        the report says which. Phase 1 checks it is a value the thing has, in its units and shape, declared once.

        With no reason (and none on the quantity, ``kg(8.5e-5, reason=...)``), the reason is pending: exploratory
        builds with it and stamps every output, and research refuses it by name (``value.reason-pending``). A reason
        given both on the quantity and here is refused."""
        self._declarations.append(ValueDeclaration(path, value, reason))

    @public
    def source(self, path: TypedPath | str, *, reason: Basis) -> None:
        """State where a setting's value comes from, apart from the value: ``exp.source("tube.inner_radius",
        reason="the tube's datasheet")``. For a value that is not a quantity (``starts``, an axis, a wall kind) or a
        source added when the experiment is sealed; a quantity may carry its own (``mm(3.175, reason=...)``), and one
        path given a source both ways is refused, as is a path that names no setting (with the candidates). A setting
        is named ``<structure>.<setting>`` (``tube.inner_radius``), ``<body>.parts.<part>.<setting>``,
        ``<apparatus>.<field>.<setting>``, ``<apparatus>.<command>`` (a command's value), ``<entry id>.<setting>``
        (``drag:umr.spacing``), ``dt``, ``gravity`` or ``conditions.<name>``."""
        self._declarations.append(SourceDeclaration(path, reason))

    @public
    def licence(self, licence: str | CustomLicence) -> None:
        """Declare the experiment's own licence: an SPDX licence expression (``exp.licence("CC-BY-4.0")``), or a
        licence of the author's own with its title and link (``CustomLicence("LicenseRef-...", title=...,
        url=...)``). It covers the experiment's own files: those MIME writes, and those the author brings without
        declaring terms (a third party's file is declared with its terms: ``asset(path, terms=Terms(...))``). The saved
        form records it. Without one the experiment declares none, which saving and ``mime pack`` never refuse, and
        every report of the folder says; ``mime push`` will not publish it until one is declared. Its form is checked
        when the experiment is saved (``terms.licence``)."""
        self._declarations.append(LicenceDeclaration(licence))

    @public
    def permission(self, permission: Permission) -> None:
        """Declare a rights holder's permission for files of the experiment (``mime.terms.Permission``: grantor,
        grantee, the files by sha256 or by their :func:`asset`, the scope, the date and the evidence). MIME checks its
        form when the experiment is saved (``terms.permission``) and binds it to the files' bytes, so a changed file
        inherits nothing; it never vets its truth. A permission that grants redistribution lets the files travel inside
        a bundle; its declaration goes into the folder's NOTICE and card, under the uploader's name when it is
        published."""
        if not isinstance(cast(object, permission), Permission):  # a script is not type-checked
            raise ArgumentTypeRefused([Refusal(
                "call.wrong-type", RefusalClass.USER,
                f"permission takes a mime.terms.Permission, not {type(permission).__name__}",
                "pass one: exp.permission(Permission(grantor=..., grantee=..., files=(...), scope=Scope(...), "
                "date=\"YYYY-MM-DD\", evidence=Evidence(...)))")])
        self._declarations.append(PermissionDeclaration(permission))

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
        law, problems, models = law_and_models(plans)
        if problems and plans.controller is not None:
            cls = plans.controller.registered.cls
            refusals = [entry_defect(cls, problem) for problem in problems]
            raise BuildRefused(refusals, report_of(plans, plans.order, refusals))
        graph, report, derived = construct(plans, law, models)
        stage = stage_text(stage_spec(plans))
        record = record_of(self._settings, self._declarations, plans, derived, stage)
        report = replace(report, stage_hash=record.stage_hash)
        load_report = None
        if self._saved is not None:
            # A saved folder's stage is rebuilt by the writer that wrote it, to be held to its lock; a save writes the
            # current writer's.
            writers = [held for held in stage_writers(self._saved.lock) if held in STAGE_WRITERS]
            if record.stage_hash == self._saved.lock["stage"]["sha256"] and STAGE_WRITER in writers:
                writers = [STAGE_WRITER]  # the current writer's stage, as saved: no other is written
            stages = {writer: stage if writer == STAGE_WRITER else stage_text(stage_spec(plans, writer=writer))
                      for writer in writers}
            load_report = check_load(self._saved, record, stages)
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
        :attr:`load_report`. Raises ``LoadRefused`` with every reason: a folder that is not there names the saved
        experiments beside it."""
        if not Path(folder).is_dir():
            from mime.env._check import missing_folder
            from mime.saved._folder import LoadRefused

            raise LoadRefused(missing_folder(Path(folder)).refusals)
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
#: The built graph's topology, :meth:`BuiltExperiment.topology`, beside a run's outputs.
RUN_TOPOLOGY = "topology.json"
TOPOLOGY_FORMAT = "mime.topology/1"


def report_summary(report: BuildReport, layout: Layout) -> dict[str, Any]:
    """:meth:`BuiltExperiment.report_summary` of ``report`` and the run layout's flags."""
    from mime.registry import DefaultKind
    from mime.validity import basis_text

    validity = [{"owner": owner, "kind": kind, "line": line}
                for kind, held in (("declared", report.validity), ("computed", report.checked),
                                   ("record", report.limits))
                for owner, lines in sorted(held.items()) for line in lines]
    flags = [{"path": flag.path, "owner": flag.owner, "entry": flag.entry, "node": flag.node, "meaning": flag.meaning,
              "limits": [None if math.isinf(v) else v for v in flag.limits] if flag.limits is not None else None}
             for flag in layout.flags]
    outcome = report.profile
    settings: list[dict[str, Any]] = []
    profile: dict[str, Any] | None = None
    if outcome is not None:
        taken = {held.path: held for held in outcome.defaults}
        for path, role in outcome.roles.items():
            items = [item for item in outcome.items if item.target == path]
            settings.append({
                "path": path, "role": role.role.value if role is not None else None,
                "default": role.default.value if role is not None else DefaultKind.UNDECLARED.value,
                "taken": taken[path].text() if path in taken else None,
                "source": basis_text(outcome.sources[path]) if path in outcome.sources else None,
                "pending": [item.line() for item in items if not item.listed],
                "listed": [item.line() for item in items if item.listed]})
        stamp = outcome.stamp()
        profile = {"profile": outcome.profile.value, "stamp": stamp.text(), "pending": list(stamp.pending),
                   "measured": list(stamp.measured), "listed": [item.line() for item in outcome.listed],
                   "defaults": [held.text() for held in outcome.defaults]}
    return {"format": REPORT_FORMAT, "experiment": report.experiment, "profile": profile, "validity": validity,
            "watches": [watch.line() for watch in report.watches], "flags": flags, "settings": settings,
            "summary": report.summary()}


def refuse_out_over(out: Path, sources: Sequence[str | PathLike[str]]) -> None:
    """Refuse a run folder that would be written over what the run reads: ``out`` the same folder as a source, inside
    one, or holding one (a run folder's ``experiment/`` is replaced as it is written, and its command log too)."""
    target = Path(out).resolve()
    for source in sources:
        held = Path(source).resolve()
        if held == target or held.is_relative_to(target) or target.is_relative_to(held):
            where = "is" if held == target else "holds" if held.is_relative_to(target) else "is inside"
            raise ArgumentRefused([Refusal(
                "run.out-over-source", RefusalClass.USER,
                f"the run would be written into {out}, which {where} {source}, what the run reads",
                "write it to a folder of its own: --out <a new folder> (or out=), outside the saved folder and the run "
                "folder it reads from",
                why="a run folder's experiment/ and command log are written afresh, which would delete or overwrite "
                    "the files being read")])


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
    #: The recording layer over the run's own stage, where ``stage_every=`` asked for one.
    recording: Path | None = None


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
        speed = speed_limit_flag(plans, plan)
        if speed is not None:
            flags.append(speed)
        flags += contact_flags(plans, plan)
        if plan.section is Section.OBSERVATION and issubclass(cls, Observation):
            observations.append(ObservationLayout(plan.owner, cls.label(), tuple(
                OutputLayout(plan.owner, output.field, plan.node_name(output.node), output.field, output.kind,
                             output.dimension, output.shape, output.description, output.reduces, output.rule,
                             float(cast(float, plan.settings[output.after])) if output.after is not None else None)
                for output in cls.outputs)))
    return Layout(plans.settings.experiment, plans.dt, tuple(observations), tuple(flags), plans.profile.stamp())


#: The name of the flag a dynamic chain's joint velocities raise past its description's speed limits.
SPEED_LIMIT = "beyond_speed_limit"


def speed_limit_flag(plans: PlanSet, plan: EntryPlan) -> FlagLayout | None:
    """The flag of a segment whose dynamics carry a tree and publish its joint velocities, raised at a step where a
    joint's speed is above its description's speed limit: read from the velocities after the run, so it adds nothing
    to the graph. ``None`` where the segment carries no tree, publishes no joint velocities, or its description limits
    no joint's speed."""
    model = plans.models.get(plan.owner)
    port = plan.record.publishes.get(JOINT_VELOCITIES.name) if plan.record is not None else None
    if model is None or port is None:
        return None
    names = model.articulation.dof_names
    declared = [model.joint(name).limits.speed for name in names]
    if all(limit is None for limit in declared):
        return None
    limits = tuple(math.inf if limit is None else limit for limit in declared)
    stated = ", ".join(f"{name} {limit:g}" for name, limit in zip(names, limits, strict=True) if math.isfinite(limit))
    free = [name for name, limit in zip(names, limits, strict=True) if not math.isfinite(limit)]
    unlimited = f"; {', '.join(free)} unlimited (no limit, or velocity=\"0\", in the description)" if free else ""
    return FlagLayout(plan.owner, plan.registered.cls.label(), plan.node_name(port.node), port.field,
                      f"a joint's speed passed its description's speed limit ({stated}; rad/s, m/s on a sliding "
                      f"joint{unlimited}), which the dynamic chain does not apply: the run went where the real "
                      f"mechanism would not. A backstop read after the run, never a refusal; a build check names a "
                      f"spinner that would drive a joint there, and its fix", SPEED_LIMIT, limits)


#: The names of the flags a body held by its hull raises: where a point is held; where the solve's sweeps left more
#: than ``CONVERGED`` of the free step's normal speeds unmet; and, for a hull whose kind has no support query, where a
#: point is held by its samples.
TOUCHING, UNCONVERGED, HELD_BY_SAMPLES = "touching", "unconverged", "held_by_samples"


def contact_flags(plans: PlanSet, plan: EntryPlan) -> list[FlagLayout]:
    """The flags of each confinement a body's dynamics holds by the hull, each read from a field its node holds for
    its own sake, so they add nothing to the graph: ``touching``, raised at a step where a point takes a force;
    ``unconverged``, where the solve's residual passes ``CONVERGED``; and ``held_by_samples``, where a point of a hull
    held by its samples (a kind with no support query) takes a force. None for a dynamics that holds no hull."""
    from mime.nodes.body._contact import CONVERGED
    from mime.registry import implements

    found: list[FlagLayout] = []
    for confinement in plan.confinements:
        if confinement.hull is None:
            continue
        owner = next((p for p in plans.entries if p.owner == confinement.id), None)
        label = owner.registered.cls.label() if owner is not None else confinement.id
        node = plan.node_name("")
        found.append(FlagLayout(confinement.id, label, node, "contact_forces",
                                f"a point of {confinement.subject}'s hull was held at {confinement.surface}, less the "
                                f"margin, by a force: the contact held the body where it touches (the points and "
                                f"forces are its state's contact_points and contact_forces)", TOUCHING))
        found.append(FlagLayout(confinement.id, label, node, "contact_residual",
                                f"the contact's solve left more than {CONVERGED:g} of the free step's normal speeds at "
                                f"{confinement.subject}'s held points unmet (contact_residual): the projection after "
                                f"the step kept the margin, and the forces read from the state are short by about "
                                f"that share", UNCONVERGED, (CONVERGED,)))
        if not implements(confinement.hull.kind, "support"):
            found.append(FlagLayout(confinement.id, label, node, "contact_forces",
                                    f"a point of {confinement.subject}'s hull was held by its samples, its kind having "
                                    f"no support query: a turning body's samples arrive at the wall off its line of "
                                    f"contact, and the inelastic hold acts as a friction and a shaking set by the "
                                    f"spacing", HELD_BY_SAMPLES))
    return found


def _not_writable(folder: Path, error: OSError) -> BaseException:
    """``save.not-writable``: the folder or a file in it could not be written, as an ``OSError`` too (the error's own
    class), so that ``except OSError`` catches it as before."""
    from mime.build._refusals import MimeError, standing_for

    refused = MimeError([Refusal(
        "save.not-writable", RefusalClass.ENVIRONMENT,
        f"cannot write the saved experiment into {str(folder)!r}: {type(error).__name__}: {error.strerror or error}",
        "save it where you can write: built.save(folder) with a folder of yours (one you own, on a disk with room)",
        why="saving writes the document, the lock, the stage and the reference outputs into the folder")])
    refused.original = error
    return standing_for(refused, error) or error


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
        self._binding: StageBinding | None = None
        self._summary: dict[str, Any] | None = None
        #: Where a run of this built left its state not finite, for the runs after it.
        self._non_finite: NonFinite | None = None

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
        if self._runs.callback is not None:  # a run starts from the declared inputs: none a live session set
            self._runs.callback.live.clear()
        kept = history or bool(recorded)
        keep = None if history or not recorded else frozenset(
            {(output.node, output.field) for output in recorded}
            | {(flag.node, flag.field) for flag in self.layout.flags})
        final, stacked = self._runs.run(steps, history=kept, hook=hook, chunk=size, keep=keep)
        result = self.result_of(start, steps, final, stacked, recorded if not history else self.layout.per_step(),
                                spacing, history=history)
        refusal = non_finite(final, stacked, self.report.nodes, start, self._plans.dt, kept_all=history,
                             already=self._non_finite)
        if refusal is not None:
            if self._non_finite is None:
                self._non_finite = NonFinite(first_non_finite(refusal), start + steps)
            raise RunFailed([refusal], result)
        return result

    def result_of(self, start: int, steps: int, final: History, stacked: History | None = None,
                  recorded: Sequence[OutputLayout] = (), spacing: int = 1, *, history: bool = False) -> RunResult:
        """The result of ``steps`` steps from step ``start`` that ended at ``final``: the series ``recorded`` names from
        ``stacked`` (every step's kept state, where the run kept it), one every ``spacing``, and each flag over the
        steps kept, or at the last step alone."""
        tracks: dict[str, FlagTrack] = {}
        series: dict[str, Samples] = {}
        if stacked is not None:
            counts = np.arange(start + 1, start + steps + 1, dtype=np.int64)
            chosen = counts % spacing == 0
            for output in recorded:
                series[output.path] = Samples(counts[chosen], np.asarray(stacked[output.node][output.field])[chosen])
            for flag in self.layout.flags:
                raised = flag.raised(stacked[flag.node][flag.field])
                tracks[flag.path] = FlagTrack(raised, bool(raised[-1]))
        else:
            tracks = {flag.path: FlagTrack(None, bool(flag.raised(np.asarray(final[flag.node][flag.field])[None])[0]))
                      for flag in self.layout.flags}
        return RunResult(steps, final, stacked if history else None, layout=self.layout, start_step=start,
                         recorded=series, tracks=tracks, every=spacing if stacked is not None else None,
                         watches=read_watches(self._plans.watches, final, stacked if history else None),
                         report=self.report_summary())

    @public
    def binding(self) -> StageBinding:
        """Which field of the state moves which prim of the stage: each body's pose, each frame of each chain segment
        (its row of the segment's frame poses), and each field a law reads at a body, a vector. Every bound prim is
        placed in the world, so a pose is its prim's transform as it stands. ``binding().record()`` is its JSON."""
        if self._binding is None:
            self._binding = stage_binding(self._plans)
        return self._binding

    @public
    def topology(self) -> dict[str, Any]:
        """The built graph in MIME's names, for a graph inspector: each node with the entry that built it (its kind,
        name and version), its owner, section and rate; each edge with MADDENING's keys, its units and the step it reads
        (MIME's lag accounting); the coupling groups; the external inputs; and the controller, with its mode, period,
        latency, reads, writes and live inputs. Plain JSON."""
        from mime.build._report import graph_config

        config = graph_config(self._graph)
        lines = {line.node: line for line in self.report.nodes}
        owners = {node: plan for plan in self._plans.entries for node in plan.node_names()}
        read = {(e.source[0], e.source[1], e.target[0], e.target[1]): e.lag.value for e in self.report.lags}
        nodes: list[dict[str, Any]] = []
        for node in config["nodes"]:
            name = str(node["name"])
            line, plan = lines.get(name), owners.get(name)
            nodes.append({
                "name": name, "type": node["type"], "timestep": node["timestep"],
                "every": line.divider if line is not None else None,
                "owner": line.owner if line is not None else None,
                "section": line.section if line is not None else None,
                "entry": {"kind": plan.registered.kind.value, "name": plan.registered.name,
                          "version": plan.registered.version, "label": line.entry if line is not None else None}
                if plan is not None else None,
                "thing": plan.thing if plan is not None else None,
                "params": {key: _plain(value) for key, value in sorted(dict(node.get("params") or {}).items())}})
        edges = [{"source_node": edge["source_node"], "source_field": edge["source_field"],
                  "target_node": edge["target_node"], "target_field": edge["target_field"],
                  "additive": bool(edge.get("additive")), "source_units": edge.get("source_units"),
                  "target_units": edge.get("target_units"),
                  "reads": read.get((edge["source_node"], edge["source_field"], edge["target_node"],
                                     edge["target_field"]))} for edge in config["edges"]]
        controller = self._plans.controller
        held = None if controller is None else {
            "entry": controller.registered.cls.label(), "mode": controller.mode.name.lower(),
            "every": controller.period, "latency": controller.latency,
            "reads": {r.label: r.path for r in controller.reads}, "writes": {w.label: w.path for w in controller.writes},
            "live": {label: {"units": live.units, "shape": list(live.shape), "meaning": live.meaning}
                     for label, live in cast(type[Controller], controller.registered.cls).live.items()}}
        return {"format": TOPOLOGY_FORMAT, "experiment": self._plans.settings.experiment, "dt": self._plans.dt,
                "nodes": nodes, "edges": edges,
                "coupling_groups": [{"nodes": list(g.nodes), "tolerance": g.tolerance,
                                     "max_iterations": g.max_iterations, "requested_by": list(g.requested_by)}
                                    for g in self.report.groups],
                "external_inputs": [{"node": e["target_node"], "field": e["target_field"], "shape": list(e["shape"]),
                                     "dtype": str(e["dtype"])} for e in config["external_inputs"]],
                "controller": held}

    @public
    def report_summary(self) -> dict[str, Any]:
        """The build report's summary for a client (``mime.report/1``, plain JSON): the validity lines (declared and
        not yet checked, computed at the build, and stated by a record), the post-run watches, the flags a run may raise
        and what each means, each setting's declared role and default kind with the default the build took and what
        the profile's stamp says of it, the profile with its stamp, and the summary's text. ``hello`` carries it, and
        every run folder holds it as ``report.json``, which ``RunResult.read`` reads."""
        if self._summary is None:
            self._summary = report_summary(self.report, self.layout)
        return json.loads(json.dumps(self._summary))

    @public
    def live(self, *, chunk: Quantity | None = None, pace: float | None = 1.0, select: Sequence[str] | None = None,
             out: str | PathLike[str] | None = None) -> LiveSession:
        """A live session of this built experiment (``mime.live``): it steps the run in chunks of one fixed size,
        ``chunk`` (about 60 per simulated second when not given, a whole number of control periods with a controller
        called back), hands each chunk's frame to its subscribers, takes live inputs (a controller's, in the callback
        mode) and records the stage on request, paced to ``pace`` times real time (``None``: as fast as it runs).
        ``select`` names the fields its frames carry (``<node>.<field>``; every bound pose and vector, every per-step
        observation output and every flag when not given). It writes a minimal run folder into ``out`` when it stops:
        the experiment, the command log, the final state, and the recording layer when recording was on."""
        from mime.live import LiveSession

        return LiveSession(self, chunk=chunk, pace=pace, select=select, out=Path(out) if out is not None else None)

    @public
    def run_to(self, out: str | PathLike[str], duration: Quantity, *, history: bool = False,
               record: Sequence[OutputRef | ObservationHandle | str] = (), every: Quantity | None = None,
               source: str | PathLike[str] | None = None, load_seconds: float | None = None,
               hook: ChunkHook | None = None, chunk: Quantity | None = None,
               stage_every: Quantity | None = None) -> RunWritten:
        """Run for ``duration`` and write into ``out`` what ``mime run`` does: ``outputs.npz`` (every field after the
        last step, with ``history`` at every step, the series ``record`` names, one every ``every``, and the flags at
        every step when anything is recorded), ``outputs.json`` beside it (its manifest: every output with its unit,
        window and flags, each output at the end written out, so the folder is readable without MIME), ``run.json``
        (the steps, the time each stage took, the environment this run was made in and the load report) and the run's
        own saved form, ``experiment/``, whose lock records that environment, so that a later run in it is verified
        bit for bit. ``source`` is the folder the experiment was loaded from and ``load_seconds`` the time loading and
        building took, both for the record. The precision a lock records is applied before loading
        (``mime.env.apply_lock_precision``), as ``mime run`` does. :meth:`RunResult.read` reads the folder back. A run
        whose state is not finite at its end is written all the same, its manifest and record saying so
        (``non_finite``), and then raises ``RunFailed`` (``run.non-finite``), what it wrote as its ``written``.

        ``hook`` and ``chunk`` stream the run as :meth:`run` does. ``stage_every`` writes the recording layer,
        ``recording.usdc``: the bound prims' poses and vectors (:meth:`binding`) over the folder's own stage, at the
        run's start and every ``stage_every`` after it, a whole number of chunks (the run is chunked for it, which
        changes nothing it computes, and keeps no step's state). ``topology.json`` is written beside the outputs, and
        ``report.json``, the build report's summary (:meth:`report_summary`), beside ``run.json``."""
        import time

        started = time.perf_counter()
        read = [source] if source is not None else [self._saved.folder] if self._saved is not None else []
        refuse_out_over(Path(out), read)
        failed: RunFailed | None = None
        samples, hook, chunk = self._stage_samples(stage_every, hook, chunk)
        try:
            result = self.run(duration, history=history, record=record, every=every, hook=hook, chunk=chunk)
        except RunFailed as error:
            if not isinstance(error.result, RunResult):
                raise
            # A state that is not finite is written as it is, and marked, then refused: on a remote machine the
            # outputs are what there is to look at.
            failed, result = error, error.result
        ran = time.perf_counter()
        target = Path(out)
        target.mkdir(parents=True, exist_ok=True)
        np.savez(target / RUN_OUTPUTS, allow_pickle=False, **result.arrays())
        manifest = result.manifest()
        refused = failed.refusals[0].message if failed is not None else None
        if refused is not None:
            manifest["non_finite"] = refused
        (target / RUN_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8", newline="\n")
        self.save(target / RUN_EXPERIMENT)
        (target / RUN_TOPOLOGY).write_text(json.dumps(self.topology(), indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8", newline="\n")
        (target / RUN_REPORT).write_text(json.dumps(self.report_summary(), indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8", newline="\n")
        recording = None
        if samples is not None:
            from mime.viz import write_recording

            recording = write_recording(target, self.binding(), samples, self.recording_settings())
        written: dict[str, Any] = {
            "folder": str(source) if source is not None else None, "duration": si(duration, TIME, what="the duration"),
            "steps": result.steps, "recorded": list(result.recorded),
            "every": result.every * self._plans.dt if result.every is not None else None,
            "seconds": {"load_and_build": round(load_seconds, 3) if load_seconds is not None else None,
                        "run": round(ran - started, 3), "save": round(time.perf_counter() - ran, 3)},
            "environment": self.record.environment,
            "load_report": self.load_report.text() if self.load_report is not None else None,
            "non_finite": refused,
            "stage_every": samples.every * self._plans.dt if samples is not None else None,
            "recording": recording.name if recording is not None else None}
        (target / RUN_RECORD).write_text(json.dumps(written, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        done = RunWritten(result.steps, target / RUN_OUTPUTS, target / RUN_RECORD, target / RUN_EXPERIMENT, written,
                          target / RUN_MANIFEST, RunResult.read(target), recording)
        if failed is not None:
            failed.written = done
            raise failed
        return done

    def recording_settings(self) -> RecordingSettings:
        """What the recording layer is written with: the stage's up axis and metres per unit, the base step, the
        experiment's name and its lock's hashes."""
        from mime.viz import RecordingSettings

        return RecordingSettings(up=self._plans.frame.up, meters_per_unit=self._plans.scene.meters_per_unit,
                                 dt=self._plans.dt, experiment=self._plans.settings.experiment,
                                 input_hash=str(self.record.hashes()["input"]))

    def _stage_samples(self, stage_every: Quantity | None, hook: ChunkHook | None,
                       chunk: Quantity | None) -> tuple[StageSamples | None, ChunkHook | None, Quantity | None]:
        """The samples ``stage_every`` asks for, and the hook and chunk the run takes them with: the user's hook called
        after each chunk as well, the chunks the control period's, ``chunk`` or ``stage_every``."""
        if stage_every is None:
            return None, hook, chunk
        from mime.units import s
        from mime.viz import StageSamples

        spacing = self._spacing(stage_every)
        size = self._runs.chunk or (self.steps_in(chunk) if chunk is not None else spacing)
        if spacing % size:
            raise ArgumentRefused([Refusal(
                "run.stage-every", RefusalClass.USER,
                f"stage_every= is {spacing} base steps, not a whole number of the run's chunks of {size} steps",
                f"record the stage every whole number of chunks: stage_every=s({max(1, spacing // size) * size * self._plans.dt!r})",
                why="the stage is recorded at the end of a chunk, where the state is on the host")])
        samples = StageSamples(self.binding(), spacing)
        samples.take(self._runs.steps, self.current_state())
        user = hook

        def taken(held: Chunk) -> None:
            if (held.steps - samples.first) % spacing == 0:
                samples.take(held.steps, held.state)
            if user is not None:
                user(held)

        return samples, taken, (chunk if chunk is not None or self._runs.chunk is not None else s(spacing * self._plans.dt))

    def current_state(self) -> History:
        """Every node's state as the graph stands, on the host."""
        from mime.build._maddening import node_state

        return {name: {field: np.asarray(value) for field, value in node_state(self._graph, name).items()}
                for name in self._graph.node_names}

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
        try:
            lock = write_folder(path, self.record, self._fresh_history(steps), steps)
        except OSError as error:
            raise _not_writable(path, error) from error
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
