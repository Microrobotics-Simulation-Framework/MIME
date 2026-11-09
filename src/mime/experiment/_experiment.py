from __future__ import annotations

import functools
import json
import math
import re
import weakref
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, override

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
from mime.build._checks import stopped_refusals
from mime.build._construct import description_warnings, report_of
from mime.core.multibody import DescriptionWarning
from mime.build._watch import ContainmentPlan
from mime.build._report import NodeLine
from mime.build._plan import EntryPlan, Section
from mime.build._steps import whole_runs_either_side, whole_steps
from mime.build._refusals import (
    ArgumentRefused,
    ArgumentTypeRefused,
    Refusal,
    RefusalClass,
    RunFailed,
    entry_defect,
)
from mime.build import StageBinding, stage_binding
from mime.control import Chunk, ChunkedRun, ChunkHook, Progress, StopRequest, law_and_models
from mime.build._maddening import (
    FlagTally,
    FlagTest,
    graph_params,
    holds_one_step_history,
    set_node_state,
    state_put_back,
    step_reads,
)
from mime.errors import public
from mime.errors._frames import warn_at_script
from mime.experiment._mermaid import Mermaid, mermaid_of
from mime.experiment._redeclared import add as add_declaration, known_as
from mime.experiment._removed import remove as remove_declaration
from mime.experiment._tree import Tree, tree_of
from mime._lineage import Change, Origin, remember_folder
from mime._notebook import in_notebook, where_called
from mime.registry import ControlLaw, Controller, Interaction, Observation, OutputKind
from mime.terms import CustomLicence, Permission
from mime.results import (
    REPORT_FORMAT,
    RUN_MANIFEST,
    RUN_REPORT,
    FlagLayout,
    FlagTrack,
    FlagWhole,
    Layout,
    ObservationLayout,
    OutputLayout,
    OutputRef,
    RunResult,
    Samples,
    Stopped,
    energy_account,
    group_verdicts,
    read_containments,
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
from mime.saved._folder import SAVED_DETERMINISTIC_OFF, LoadRefused, NondeterministicReference, nondeterministic_gpu, \
    stage_writers
from mime.saved._scales import FieldUnits, one_ulp_up, resolution, symbol_units
from mime.saved._schema import ReferenceProgram
from mime.scene import STAGE_WRITER, STAGE_WRITERS, Scene, stage_text
from mime.things import (
    Apparatus,
    ApparatusHandle,
    Body,
    BodyHandle,
    Handle,
    LawHandle,
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
    from mime.cli._display import RunDisplay, Watch
    from mime.registry import Problem
    from mime.live import LiveSession
    from mime.study._override import Overridden
    from mime.viz import RecordingSettings, StageSamples, WebPage

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
    plus_minus=0.5)}``). ``duration`` is the length of a run of this experiment, a whole number of base steps, which
    a saved experiment carries: ``mime run`` on its folder runs that long unless ``--duration`` says another
    (``built.duration`` reads it; ``built.run(duration)`` always takes its own).

    ``profile`` is how much the build asks (``mime.validity.Profile``): ``"exploratory"`` (the default) waives what
    the experiment leaves unstated with a pending reason, recorded in the saved form and stamped on every output;
    ``"research"`` needs each with its reason and refuses a pending one by name; ``"strict"`` is a later release's
    and refused until then. Validity limits and input errors are refused in every profile. ``built.report.pending()`` and
    ``mime pending <folder>`` list what is pending, with the lines to paste.

    A declaration made a second time (the same thing's name, law or observation id, value's or source's path, a second
    controller or licence), as when a notebook's cell runs twice, is kept once, with a warning, when it is the same as
    the first: its type, its settings with their units and reasons, and what it binds. A changed one is refused at the
    build in a script. In a notebook (IPython, Jupyter, Colab) the last one run replaces the earlier one, with a
    warning, which a warnings filter can remove: so each replacement is also recorded, with the declarations that
    follow it, in ``plan().replaced`` and the build report (the saved form holds the final declarations only). A
    structure, medium, body or apparatus replaced this way is found by the declarations that hold its name (laws,
    observations, values, sources, the controller, a medium's ``fills``), and the warning names each; so is a law or an
    observation that a source or the controller names. A name declared again as another kind, or held as an object and
    not a name, is kept beside the new one, and the build refuses the two, saying a line or cell has probably run
    twice. :meth:`remove` takes a declaration out, and is recorded (``plan().removed``) with no warning. A build made before a
    replacement or a removal is stale (:attr:`BuiltExperiment.stale`): refused in a notebook, warned about in a script.
    """

    def __init__(self, name: str, *, scene: Scene | None = None, gravity: Quantity = CONVENTIONAL_GRAVITY,
                 dt: Quantity | Literal["auto"] | None = None, seed: int = 0, up: Any = None,
                 conditions: Mapping[str, Condition] | None = None,
                 profile: str | Profile = DEFAULT_PROFILE.value, duration: Quantity | None = None) -> None:
        self.name = name
        self._settings = RunSettings(name, scene, gravity, dt, seed, up, conditions,
                                     profile.value if isinstance(profile, Profile) else profile, duration)
        self._declarations: list[Declaration] = []
        #: The note of each declaration that replaced an earlier one in this session (a notebook's cell run again):
        #: what a warnings filter cannot remove, shown by ``plan()`` and the build report. Not part of the saved form.
        self._replaced: list[str] = []
        #: The declarations an experiment made by :meth:`based_on` took from the saved experiment it is based on (the
        #: objects themselves), and that experiment's name: one of them declared again, changed, is replaced, in a
        #: script as in a notebook.
        self._inherited: tuple[Declaration, ...] = ()
        self._parent: str | None = None
        #: Likewise for each declaration removed (``remove``).
        self._removed: list[str] = []
        #: Every replacement and removal of this session, in order, which a build, a result and a view remember the
        #: count of (``mime._lineage``): what says one of them predates a change.
        self._changes: list[Change] = []
        #: Where each declaration held was made (a cell and its line, or a file and its line), by ``id``: what
        #: ``remove`` names when something still holds a name.
        self._sites: dict[int, str] = {}
        self._saved: SavedForm | None = None
        #: Whether :meth:`load` has checked this experiment against its lock: a later build that differs from the lock
        #: is an edit made after the load.
        self._load_checked = False
        self.load_report: LoadReport | None = None

    def declared_as(self) -> tuple[RunSettings, tuple[Declaration, ...], SavedForm | None]:
        """What this experiment declares as data: its run settings, its declarations, and the saved form it was loaded
        from (``None`` for a script's), whose pins a build keeps: what a study declares its points in."""
        return self._settings, tuple(self._declarations), self._saved

    def _declared_now(self) -> list[Declaration]:
        """The declarations held now: what an origin compares its own with (through a weak reference to this method, so
        that a result never keeps the experiment alive)."""
        return self._declarations

    @classmethod
    def declared_from(cls, settings: RunSettings, declarations: Sequence[Declaration]) -> Experiment:
        """An experiment of ``settings`` and ``declarations``, as data: what a study builds a group's point as."""
        return cls._declared(settings, declarations)

    def redeclared(self, settings: RunSettings, declarations: Sequence[Declaration]) -> None:
        """Take ``settings`` and the declarations among ``declarations`` that this experiment does not hold (by
        identity) in the place of what they replace, as declaring them one by one would: each replacement is recorded in
        ``plan().replaced``, with no warning. What a study's bridge (``RunResult.as_experiment``) makes an experiment of
        a member with."""
        import warnings

        self._settings = settings
        held = {id(declaration) for declaration in self._declarations}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for declaration in declarations:
                if id(declaration) not in held:
                    self._declare(declaration)

    @classmethod
    def _declared(cls, settings: RunSettings, declarations: Sequence[Declaration]) -> Experiment:
        experiment = cls(settings.experiment, scene=settings.scene, gravity=settings.gravity, dt=settings.dt,
                         seed=settings.seed, up=settings.up, conditions=settings.conditions, profile=settings.profile,
                         duration=settings.duration)
        experiment._settings = replace(experiment._settings, based_on=settings.based_on)
        experiment._declarations = list(declarations)
        return experiment

    def _declare(self, declaration: Declaration) -> None:
        """``declaration`` added by the rule of a declaration made a second time (``_redeclared``), with the change it
        made (a replacement) recorded and the place it was declared at kept."""
        before = len(self._replaced)
        add_declaration(self._declarations, declaration, self._replaced, self._inherited, self._parent)
        known = known_as(declaration)
        if len(self._replaced) > before and known is not None:
            self._changes.append(Change("replacement", known.noun, self._replaced[-1]))
        held = {id(other) for other in self._declarations}
        if id(declaration) in held:
            self._sites[id(declaration)] = where_called()
        self._sites = {key: place for key, place in self._sites.items() if key in held}

    @public
    def remove(self, target: Handle | ObservationHandle | TypedPath | str | None = None, /, *,
               kind: str | None = None) -> None:
        """Take a declaration out of the experiment: a thing (by its name or its handle), a law (by the id ``attach``
        returned), an observation (by its id or handle), a declared value or a source (by its path), or the
        controller (``exp.remove(kind="controller")``, which has no name). ``kind=`` (``"thing"``, ``"law"``,
        ``"observation"``, ``"value"``, ``"source"``, ``"controller"``) says which where a name is two kinds, a path
        with a declared value and a source. In a script and in a notebook alike.

        It is refused while another declaration still names what it removes (a thing is named by its laws, its
        observations, its values and sources, a medium that fills it, the controller; a law or an observation by the
        sources written under its id and the controller): the refusal lists each, with the cell or line that
        declared it. Nothing is removed with it. A removal is recorded (``plan().removed`` and the build report) and
        raises no warning: it is your own call, and is made under ``-W error`` too. Removing and declaring again under the same name is how a
        thing's kind is changed (``exp.add_body`` after ``exp.remove``)."""
        written = cast(object, target)  # a script is not type-checked
        if written is not None and not isinstance(written, (Handle, ObservationHandle, TypedPath, str)):
            raise ArgumentTypeRefused([Refusal(
                "call.wrong-type", RefusalClass.USER,
                f"remove takes a name, a handle or a typed path, not {type(written).__name__}",
                "name what to remove: exp.remove('water'), exp.remove(tube), exp.remove(kind='controller')")])
        if isinstance(written, Handle):
            kind = kind or "thing"
            name: str | None = written.name
        elif isinstance(written, ObservationHandle):
            kind = kind or "observation"
            name = written.id
        else:
            name = None if written is None else str(written)
        noun, note = remove_declaration(self._declarations, name, kind, self._place_of, self._removed)
        self._changes.append(Change("removal", noun, note))
        held = {id(other) for other in self._declarations}
        self._sites = {key: place for key, place in self._sites.items() if key in held}

    def _place_of(self, declaration: Declaration) -> str:
        if any(declaration is each for each in self._inherited):
            return f"taken from {self._parent!r}, the saved experiment this one is based on"
        return self._sites.get(id(declaration), "declared in the loaded document")

    def _add(self, thing: Any, cls: type) -> None:
        if not isinstance(thing, cls):
            raise ArgumentTypeRefused([Refusal(
                "call.wrong-type", RefusalClass.USER,
                f"add_{cls.__name__.lower()} takes a {cls.__name__}, not {type(thing).__name__}",
                f"pass one: exp.add_{cls.__name__.lower()}({cls.__name__}(...))")])
        self._declare(ThingDeclaration(thing))

    @public
    def add_structure(self, structure: Structure) -> StructureHandle:
        self._add(structure, Structure)
        return _handle_of(structure)

    @public
    def add_medium(self, medium: Medium) -> MediumHandle:
        self._add(medium, Medium)
        return _handle_of(medium)

    @public
    def add_body(self, body: Body) -> BodyHandle:
        self._add(body, Body)
        return _handle_of(body)

    @public
    def add_apparatus(self, apparatus: Apparatus) -> ApparatusHandle:
        self._add(apparatus, Apparatus)
        return _handle_of(apparatus)

    @public
    def attach(self, law: Interaction, /, **bindings: Handle | TypedPath | str) -> LawHandle:
        """Bind an interaction to things by its roles; returns its id (its registered name and the things bound to
        it), a ``str`` that also names its settings as typed paths (``drag.lubrication.epsilon``), as a study and an
        override take them. Every role is bound explicitly: there is no default binding."""
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
        self._declare(Attachment(law, dict(bindings)))
        return LawHandle(entry_id(type(law), bindings), law)

    @public
    def handle(self, name: str) -> Handle | LawHandle | ObservationHandle:
        """The handle of what this experiment declares under ``name``: a thing (``exp.handle("field")``, as
        ``add_apparatus`` returned it), a law by its id (``exp.handle("drag:umr")``, as ``attach`` returned it) or an
        observation by its id. For an experiment that arrives with no handles: one loaded, or made by
        :meth:`based_on`. An unknown name is refused, with the names it holds."""
        things = {d.thing.name: d.thing for d in self._declarations if isinstance(d, ThingDeclaration)}
        if name in things:
            return _handle_of(things[name])
        for declaration in self._declarations:
            if isinstance(declaration, Attachment) and entry_id(type(declaration.law), declaration.bindings) == name:
                return LawHandle(name, declaration.law)
            if isinstance(declaration, ObservationDeclaration):
                observation = declaration.observation
                if entry_id(type(observation), observation.bindings()) == name:
                    return ObservationHandle(name, tuple(output.field for output in type(observation).outputs))
        known = [*things, *(entry_id(type(d.law), d.bindings) for d in self._declarations if isinstance(d, Attachment)),
                 *(entry_id(type(d.observation), d.observation.bindings()) for d in self._declarations
                   if isinstance(d, ObservationDeclaration))]
        raise ArgumentRefused([Refusal(
            "call.unknown-name", RefusalClass.USER, f"this experiment declares nothing named {name!r}",
            f"name a thing, a law's id or an observation's id it declares, as `exp.handle(\"field\")` or "
            f"`exp.handle(\"drag:umr\")`. {did_you_mean(name, known)}")])

    @public
    def declare(self, path: TypedPath | str, value: Quantity, *, reason: Basis | None = None) -> None:
        """State a value of a thing (``declare("umr.mass", kg(8.5e-5), reason="...")``) or of a body's part
        (``"umr.parts.shell.density"``) in the experiment, with its reason: the build report and the saved form label
        it "declared in the experiment". It takes the place of the value a record or the body's parts would give, and
        the report says which. Phase 1 checks it is a value the thing has, in its units and shape, declared once.

        With no reason (and none on the quantity, ``kg(8.5e-5, reason=...)``), the reason is pending: exploratory
        builds with it and stamps every output, and research refuses it by name (``value.reason-pending``). A reason
        given both on the quantity and here is refused."""
        self._declare(ValueDeclaration(path, value, reason))

    @public
    def source(self, path: TypedPath | str, *, reason: Basis) -> None:
        """State where a setting's value comes from, apart from the value: ``exp.source("tube.inner_radius",
        reason="the tube's datasheet")``. For a value that is not a quantity (``starts``, an axis, a wall kind) or a
        source added when the experiment is sealed; a quantity may carry its own (``mm(3.175, reason=...)``), and one
        path given a source both ways is refused, as is a path that names no setting (with the candidates). A setting
        is named ``<structure>.<setting>`` (``tube.inner_radius``), ``<body>.parts.<part>.<setting>``,
        ``<apparatus>.<field>.<setting>``, ``<apparatus>.<command>`` (a command's value), ``<entry id>.<setting>``
        (``drag:umr.spacing``), ``dt``, ``gravity`` or ``conditions.<name>``."""
        self._declare(SourceDeclaration(path, reason))

    @public
    def licence(self, licence: str | CustomLicence) -> None:
        """Declare the experiment's own licence: an SPDX licence expression (``exp.licence("CC-BY-4.0")``), or a
        licence of the author's own with its title and link (``CustomLicence("LicenseRef-...", title=...,
        url=...)``). It covers the experiment's own files: those MIME writes, and those the author brings without
        declaring terms (a third party's file is declared with its terms: ``asset(path, terms=Terms(...))``). The saved
        form records it. Without one the experiment declares none, which saving and ``mime pack`` never refuse, and
        every report of the folder says; ``mime push`` will not publish it until one is declared. Its form is checked
        when the experiment is saved (``terms.licence``)."""
        self._declare(LicenceDeclaration(licence))

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
        self._declare(PermissionDeclaration(permission))

    @public
    def observe(self, observation: Observation) -> ObservationHandle:
        if not isinstance(cast(object, observation), Observation):
            raise ArgumentTypeRefused([Refusal(
                "call.wrong-type", RefusalClass.USER,
                f"observe takes an Observation, not {type(observation).__name__}",
                "pass an observation: exp.observe(SwimSpeed(body, along=X))")])
        self._declare(ObservationDeclaration(observation))
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
        self._declare(ControllerDeclaration(controller, reads, writes, mode, rate, latency))

    @public
    def plan(self) -> PlanSet:
        """Phase 1 alone: resolve and check every declaration, with no graph. Raises ``BuildRefused``."""
        try:
            plans = plan_experiment(self._settings, list(self._declarations))
        except BuildRefused as refused:
            if refused.report is not None:  # a refused build is where a replaced thing's trace matters most
                refused.report = replace(refused.report, replaced=tuple(self._replaced),
                                         removed=tuple(self._removed))
            raise
        return replace(plans, replaced=tuple(self._replaced), removed=tuple(self._removed))

    @public
    def tree(self) -> Tree:
        """What this experiment declares, as a tree: its structures, media and what each fills, bodies and their
        parts, apparatus with their fields and drives, each law with the things it binds, each observation with what it
        reads, and the controller with its reads and writes, each with its registered id and version. Nothing is built or
        checked, so it shows an experiment that would be refused too. It is the topology: what is made of what and what binds
        what, not the declared values (a mass, a viscosity), the settings of an entry, the run's settings or the scene.
        ``str(tree)`` is plain indented text, ``tree.record()`` the same as data. The same tree whatever the order of
        the calls that declared it."""
        return tree_of(self._settings, self._declarations)

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
        from mime.saved._pins import keeping_pins

        with keeping_pins(self._saved) as pins:
            plans = self.plan()
            law, problems, models = law_and_models(plans)
            if problems and plans.controller is not None:
                cls = plans.controller.registered.cls
                refusals = [entry_defect(cls, problem) for problem in problems]
                raise BuildRefused(refusals, report_of(plans, plans.order, refusals))
            graph, report, derived = construct(plans, law, models)
            stage = stage_text(stage_spec(plans))
            record = record_of(self._settings, self._declarations, plans, derived, stage)
            report = replace(report, stage_hash=record.stage_hash, assets=pins.lines())
        for text in report.range_warnings():
            warn_at_script(text, UserWarning)
        for text in description_warnings(plans):
            warn_at_script(text, DescriptionWarning)
        load_report = None
        if self._saved is not None:
            # A saved folder's stage is rebuilt by the writer that wrote it, to be held to its lock; a save writes the
            # current writer's.
            writers = [held for held in stage_writers(self._saved.lock) if held in STAGE_WRITERS]
            if record.stage_hash == self._saved.lock["stage"]["sha256"] and STAGE_WRITER in writers:
                writers = [STAGE_WRITER]  # the current writer's stage, as saved: no other is written
            with keeping_pins(self._saved):
                stages = {writer: stage if writer == STAGE_WRITER else stage_text(stage_spec(plans, writer=writer))
                          for writer in writers}
            load_report = check_load(self._saved, record, stages, edited_after_load=self._load_checked)
        origin = Origin(self._changes, len(self._changes), tuple(self._declarations), weakref.WeakMethod(self._declared_now))
        return BuiltExperiment(graph, report, plans, record, saved=self._saved, load_report=load_report, law=law,
                               origin=origin, declared=(tuple(self._declarations), self._settings))

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
        code it names; first refuses a folder another development release saved (``load.other-release``), which needs
        only the lock; resolves every entry among installed packages and installs nothing, refusing a missing one
        or another version with the line to install; then makes the three load checks, whose outcome is
        :attr:`load_report`. Raises ``LoadRefused`` with every reason: a folder that is not there names the saved
        experiments beside it."""
        if not Path(folder).is_dir():
            from mime.env._check import missing_folder
            from mime.saved._folder import LoadRefused

            raise LoadRefused(missing_folder(Path(folder)).refusals)
        form = read_folder(Path(folder))
        from mime.saved import _folder

        # First: the refusal needs only the lock, and a folder another release saved may no longer resolve or build.
        other_release = _folder.other_release_refusal(form)
        if other_release is not None:
            raise _folder.LoadRefused([other_release])
        experiment = cls._declared(*declarations_from(form.document, form.lock["entries"], Path(folder)))
        experiment._saved = form
        try:
            experiment.load_report = experiment.build().load_report
        except BuildRefused as refused:
            from mime.saved._versions import products_waiver_refusals

            worded = products_waiver_refusals(refused)
            if worded is refused:
                raise
            raise worded from None
        experiment._load_checked = True
        return experiment

    @classmethod
    @public
    def based_on(cls, source: str | PathLike[str], *, name: str, accept_licences: Collection[str] = (),
                 yes: bool = False, force: bool = False) -> Experiment:
        """A new experiment made from a saved one, to change and save as an experiment of its own: it holds the saved
        experiment's declarations under ``name``, with no lock, and saving it writes a new folder with a lock of its
        own. ``source`` is a saved folder, a bundle file, or what ``mime pull`` takes (a DOI, an address): a source that
        is not a folder is pulled first, as ``mime pull <source>`` pulls it, into ``./<its name>/``, and the experiment
        is based on that folder. Where that folder already holds the same experiment (the same input hash: a
        notebook's cell run again), it is used as it is, and nothing is unpacked into it, and a DOI's bundle is not
        downloaded again (the record is read, and the bundle comes from the asset cache) unless the cache was cleared
        or the record has a new version; where the folder holds anything else, it is refused as pull refuses it. Its
        assets are fetched into the asset cache through the download gates (:func:`mime.saved.pull`): a licence with conditions beyond credit is asked about where a person can answer, or
        accepted by name in ``accept_licences``; ``yes`` answers the question a large download asks, and ``force``
        overrides the stop of one that would leave the drive almost full; neither accepts a licence. The saved
        experiment is loaded as :meth:`load` loads it, and refused, with the load's reasons, where it does not load;
        its folder is read and never written.

        A declaration taken from the saved experiment, made again and changed, replaces the one taken, in a script as
        in a notebook (``exp.declare("rpm.magnetic_moment", A_m2(25.0), reason=...)``), with a warning that names what
        follows it, recorded in ``plan().replaced`` as a notebook's replacements are; a declaration made after this
        call follows the rule any experiment's does. The run settings (the step, the seed, gravity, the scene, the
        conditions, the profile and the run's length) are the saved experiment's.

        The new experiment's document records the one it was made from as ``based_on``: its name, the input hash its
        lock records, the MIME version that saved it, and the DOI of the record it was published as: the record it was
        pulled from, where ``source`` is a DOI (a lock that names another record is warned of, naming both), and
        otherwise the one its lock names (every bundle ``mime push`` publishes does); a DOI that is not a DOI's shape is
        not recorded, with a warning that says why. Only that one: an experiment made from one that was itself made
        from another records the one it was made from, whose own document records the next. ``mime card`` names it."""
        from mime.saved._based_on import based_on_of

        held = Path(source)
        folder, pulled_doi = (held, None) if held.is_dir() else _pulled(source, accept_licences, yes, force)
        parent = cls.load(folder)
        assert parent._saved is not None
        record, notes = based_on_of(parent._saved, pulled_doi)
        for note in notes:
            warn_at_script(note, UserWarning)
        settings = replace(parent._settings, experiment=name, based_on=record)
        experiment = cls._declared(settings, parent._declarations)
        experiment._inherited, experiment._parent = tuple(experiment._declarations), parent.name
        return experiment


def _handle_of(thing: Any) -> Any:
    """A thing's handle, as ``add_*`` returns it: a structure's with its geometry's regions, surfaces and directions, an
    apparatus's with its fields and commands, a body's with its dynamics' initial state."""
    if isinstance(thing, Structure):
        geometry = type(thing.geometry) if thing.geometry is not None else None
        if geometry is None:
            return StructureHandle(thing.name)
        return StructureHandle(thing.name, getattr(geometry, "regions", ()), getattr(geometry, "surfaces", {}),
                               getattr(geometry, "directions", ()))
    if isinstance(thing, Apparatus):
        return ApparatusHandle(thing.name, thing.fields, thing.commands)
    if isinstance(thing, Body):
        return BodyHandle(thing.name, getattr(type(thing.dynamics), "initial_state", {}))
    return MediumHandle(thing.name)


def _pulled(source: str | PathLike[str], accept_licences: Collection[str], yes: bool,
            force: bool) -> tuple[Path, str | None]:
    """The folder ``mime pull <source>`` unpacks ``source`` into (``./<its name>/``), or the one there already that
    holds the same experiment, and, for a DOI, the DOI of the record it was pulled from, for
    :meth:`Experiment.based_on` (:func:`mime.saved._pull.pulled_parent`). A folder there that holds another is refused
    as pull refuses it, with what to do in its place (pull's own action names ``--into``, which ``based_on`` does not
    take)."""
    from mime.saved._pull import pulled_parent

    try:
        return pulled_parent(source, accept_licences=accept_licences, yes=yes, force=force)
    except BuildRefused as refused:
        if "pull.destination" not in refused.codes():
            raise
        action = ("based_on unpacks a source that is not a folder into ./<its name>/, and uses a folder there only where "
                  "it holds the same experiment (the same input hash); this one holds another, so move or rename it, or "
                  "work in another folder; to base the new experiment on what it holds, give its path: "
                  "Experiment.based_on(\"<the folder>\", name=...)")
        raise BuildRefused([replace(each, action=action) if each.code == "pull.destination" else each
                            for each in refused.refusals]) from None


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
    #: Set when the run was ended before the steps it was asked for (``stop=``): why, and the steps asked for.
    stopped: Stopped | None = None
    #: The declarations the run came from (``mime._lineage``): what says its view predates a replacement or a removal.
    origin: Origin | None = None

    def view(self, *, offline: bool = False, caption: str | None = None) -> WebPage:
        """The run's view as a notebook shows it: ``mime.viz.display`` of its folder."""
        from mime.viz import display

        return display(self.record_file.parent, offline=offline, caption=caption)  # the folder's origin is found there


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
#: than ``CONVERGED`` of the free step's speed at the held points unmet; for a hull whose kind has no support query,
#: where a point is held by its samples; and for one that has, where the body may reach past a held point, its group's
#: nearest sample near being held, by more than ``SUPPORT_SHORT`` of the margin: the point stood shallower than that
#: sample, or its search did not converge and held the sample (by up to the samples' shortfall).
TOUCHING, UNCONVERGED, HELD_BY_SAMPLES, SUPPORT_SHORT_FLAG = ("touching", "unconverged", "held_by_samples",
                                                              "support_short")


def contact_flags(plans: PlanSet, plan: EntryPlan) -> list[FlagLayout]:
    """The flags of each confinement a body's dynamics holds by the hull, each read from a field its node holds for
    its own sake, so they add nothing to the graph: ``touching``, raised at a step where a point takes a force;
    ``unconverged``, where the solve's residual passes ``CONVERGED``; ``held_by_samples``, where a point of a hull
    held by its samples (a kind with no support query) takes a force; and, for a kind with one, ``support_short``,
    where the body may reach past a held point by more than ``SUPPORT_SHORT`` of the margin (of a thousandth of the
    hull's size at a margin of zero; ``_contact.shallower``). None for a dynamics that holds no hull."""
    from mime.nodes.body._contact import CONVERGED, SUPPORT_SHORT
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
                                f"the contact's solve left more than {CONVERGED:g} of the free step's speed at "
                                f"{confinement.subject}'s held points unmet (contact_residual): the projection after "
                                f"the step kept the margin, and the speeds at those points are off by up to that share "
                                f"(each point's own force is not fixed where rows share a rim)",
                                UNCONVERGED, (CONVERGED,)))
        if implements(confinement.hull.kind, "support"):
            size = float(cast(Any, confinement.hull.kind).enclosing_radius(confinement.hull.settings))
            bound = confinement.margin if confinement.margin > 0.0 else 1e-3 * size
            found.append(FlagLayout(confinement.id, label, node, "contact_short",
                                    f"{confinement.subject}'s body may have reached more than "
                                    f"{SUPPORT_SHORT * bound * 1e6:.3g} um past a point of its hull the contact held, "
                                    f"its group's nearest sample within the margin of the wall shrunk by it "
                                    f"(contact_short, how far): the point stood shallower than that sample (the "
                                    f"search went the wrong way), or the search did not converge within its bound "
                                    f"and held that sample (the body may then reach past the margin by up to the "
                                    f"samples' shortfall)",
                                    SUPPORT_SHORT_FLAG, (SUPPORT_SHORT * bound,)))
        else:
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


def _python_display(progress: bool | None) -> RunDisplay | None:
    """MIME's live display for a run, a ``run_to`` or a save started from Python (``mime.cli._display.python_display``),
    or ``None``. Outside IPython (``mime._notebook.in_notebook``), the default asks nothing more: rich's own test of a
    notebook begins with IPython's ``get_ipython``, so a script or a terminal goes the way it always went, and loads no
    display."""
    if progress is False or (progress is None and not in_notebook()):
        return None
    from mime.cli._display import python_display

    return python_display(progress)


def _whole_run(tally: FlagTally | None, index: int) -> FlagWhole | None:
    """The ``index``-th flag's reduction over the run's steps, or ``None`` for a run that reduced none."""
    if tally is None:
        return None
    first, last = int(tally.first[index]), int(tally.last[index])
    return FlagWhole(int(tally.count[index]), first if first >= 0 else None, last if last >= 0 else None)


class BuiltExperiment:
    """A built experiment: its report, its graph's canonical text, its saved form's record, and runs. A run
    continues from where the last one left the graph."""

    def __init__(self, graph: GraphManager, report: BuildReport, plans: PlanSet, record: SavedRecord, *,
                 saved: SavedForm | None = None, load_report: LoadReport | None = None,
                 law: ControlLaw | None = None, origin: Origin | None = None,
                 declared: tuple[tuple[Declaration, ...], RunSettings] | None = None) -> None:
        self._graph = graph
        #: The declarations and run settings it was built from, which an overridden run declares its values in.
        self._declared: tuple[tuple[Declaration, ...] | None, RunSettings | None] = declared or (None, None)
        #: Its own point, assembled as an overridden run's is (``mime.study._override.base_point``), once asked for.
        self.own_point: Any = None
        self._report = report
        self._origin = origin
        self._plans = plans
        self.record = record
        self._saved = saved
        self.load_report = load_report
        self._law = law
        self._runs = ChunkedRun(graph, plans, law)
        #: What the compiled step read when it was built, which a save holds the graph to before it takes the
        #: reference outputs from it (:meth:`_reference`).
        self._built_reads = step_reads(graph)
        self.layout = run_layout(plans)
        self._binding: StageBinding | None = None
        self._summary: dict[str, Any] | None = None
        #: Where a run of this built left its state not finite, for the runs after it.
        self._non_finite: NonFinite | None = None

    @property
    def report(self) -> BuildReport:
        """The build report. Once the experiment it was built from has had a replacement or a removal since the build,
        it says so on its first line (``STALE: ...``): ``built`` is then the physics of the declarations before."""
        stale = self.stale
        return self._report if stale is None else replace(self._report, stale=stale)

    @property
    def stale(self) -> str | None:
        """Why this build no longer matches its experiment, or ``None``: the replacements and removals made since it was
        built, and what to do (build again). A build of a loaded experiment or one made without an experiment never is."""
        return self._origin.stale("this build", "build the experiment again, then run that build") if self._origin is not None else None

    def _fresh(self) -> None:
        """Refuse in a notebook or IPython, and warn in a script, to use this build after the experiment it was built
        from had a replacement or a removal: it would run the physics of the declarations before, as if they were the
        new. A notebook's cell run again is the trap, with the one that builds not run again with it, so there the
        build is refused; a script that changes its experiment after a build did it on purpose (to run both), so
        there it is warned."""
        stale = self.stale
        if stale is None:
            return
        if in_notebook():
            raise ArgumentRefused([Refusal(
                "run.stale", RefusalClass.USER, stale,
                "build the experiment again (built = exp.build()), then run that build",
                why="a build is the declarations as they were when it was made; running it after a cell was run again "
                    "would pass the old physics for the new")])
        warn_at_script(stale, UserWarning)

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
        return graph_text(self._graph, self._report.lags)

    @public
    def to_mermaid(self, *, direction: str = "LR") -> Mermaid:
        """The compiled graph as a Mermaid flowchart: every node, edge, coupling group and external input, each node
        labelled with its name and the registered entry (and version) that built it. It is MADDENING's ``to_mermaid`` with
        MIME's names added; ``direction`` is MADDENING's (``LR``, ``RL``, ``TB`` or ``BT``). A ``str`` that a notebook
        also shows as a diagram where its front end draws Mermaid. The same text for the same experiment, whatever the
        order of the calls that declared it."""
        return mermaid_of(self._graph, self.report, direction=direction, origin=self._origin)

    @property
    def stage_text(self) -> str:
        """The scene's stage as ``save`` writes it into ``scene/scene.usda``: the same bytes for the same
        declarations."""
        return self.record.stage

    @property
    def duration(self) -> Quantity | None:
        """The run length the experiment declared (``Experiment(..., duration=...)``, or a saved experiment's), or
        ``None`` where it declared none. ``run`` always takes the length of its own."""
        from mime.units import s

        declared = self._plans.settings.duration
        if declared is None:
            return None
        seconds = si(declared, TIME, what="the experiment's duration")
        return s(seconds) if isinstance(seconds, float) else None

    @public
    def steps_in(self, duration: Quantity) -> int:
        seconds = si(duration, TIME, what="the run's duration")
        dt = self._plans.dt
        steps = whole_steps(seconds, dt)
        if steps is None:
            lower, upper = whole_runs_either_side(seconds, dt)
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
            hook: ChunkHook | None = None, chunk: Quantity | None = None, stop: StopRequest | None = None,
            progress: bool | None = None, override: Mapping[Any, object] | None = None) -> RunResult:
        """Run for ``duration``, a whole number of base steps, in compiled chunks, between which Python regains control
        (a ``KeyboardInterrupt`` arrives there, and a ``stop`` is asked). With a controller they are its control
        periods, with the controller called between them. ``hook`` is called after every chunk with a
        :class:`~mime.control.Chunk`, its end and the state there; without a controller the run is chunked for it in
        chunks of ``chunk``. A run asked for neither is split into about 64 chunks of one length, a divisor of its
        steps where one lies near; the chunks change nothing it computes, bit for bit. A run with a hook or a
        controller is a whole number of chunks. After a ``KeyboardInterrupt`` the built experiment is not run on:
        build it again.

        ``stop`` is called after every chunk but the last and returns ``None`` to go on, or a reason to end the run
        there: the result then holds the steps reached, and its ``stopped`` says why and how many were asked for
        (``mime run`` passes one for Ctrl+C).

        The result (:class:`~mime.results.RunResult`) holds every output at the end each observation declares, and
        the series of the per-step outputs ``record`` names (``record=[swim.speed]``), one sample every ``every`` (a
        whole number of base steps, every step when not given), with every flag the entries raise at every step. A
        run that records nothing keeps no step's state, and reduces each flag over its steps as it runs (how many raised
        it, the first and the last), so a flag raised at any step is known. ``history`` keeps
        every field at every step, and with it every per-step output is recorded. ``every`` spaces the samples kept
        and does not reduce the cost of recording, which is the whole state's history per step whatever is kept.

        ``progress`` shows ``mime run``'s live display. The default, ``None``, shows it in a notebook (Jupyter, Colab:
        rich's own detection), drawn through the notebook, and nowhere else; ``True`` shows it on a terminal too, as
        ``mime run`` does (on standard error, where that is a terminal not asked to be plain); ``False`` never. It is
        drawn from ``stop``'s place after each chunk, the caller's own ``stop`` asked after it and its answer kept; it
        reads where each chunk ended and the flags raised there, and changes neither the chunks nor anything the run
        computes.

        ``override`` declares values for this run alone (``{field.commands.frequency: Hz(20.0)}``, by the paths a
        study's axes take): the run's point is this experiment with them declared, planned in full, so its validity
        lines, material reads and refusals are judged again, and it runs on this build's compiled step with its own
        parameter leaves (and initial state, on a run from the start), every output stamped "overridden" with the
        values. What another compiled step would need is refused with its route: a value baked into a static or the
        structure (a rebuild: a Study, or ``Experiment.based_on``), a numerical setting, ``dt``, the precision or a
        rung (a variant). The next run without one runs the build's own values.

        A build whose experiment has had a replacement or a removal since is refused in a notebook or IPython, and warned
        about in a script (:attr:`stale`)."""
        self._fresh()
        point = self._overridden(override)
        display = _python_display(progress)
        if display is None:
            return self._run(duration, history=history, record=record, every=every, hook=hook, chunk=chunk, stop=stop,
                             point=point)
        try:
            watch = self._watch(display, duration, stop, finishing="finishing the last chunk")
            result = self._run(duration, history=history, record=record, every=every, hook=hook, chunk=chunk,
                               stop=watch.stop, point=point)
            watch.ended(result.steps)
            return result
        finally:
            display.close()

    def _watch(self, display: RunDisplay, duration: Quantity, stop: StopRequest | None, *,
               finishing: str | None = None) -> Watch:
        from mime.cli._display import Watch

        return Watch(display, self.steps_in(duration), self._plans.dt, self.raised_flags, stop, finishing=finishing)

    def _overridden(self, override: Mapping[Any, object] | None) -> Overridden | None:
        if not override:
            return None
        from mime.study._override import overridden

        return overridden(self, override)

    def _run(self, duration: Quantity, *, history: bool, record: Sequence[OutputRef | ObservationHandle | str],
             every: Quantity | None, hook: ChunkHook | None, chunk: Quantity | None,
             stop: StopRequest | None, point: Overridden | None = None) -> RunResult:
        steps = self.steps_in(duration)
        recorded = self._recording(record)
        if every is not None and not (recorded or history):
            raise ArgumentRefused([Refusal(
                "run.every", RefusalClass.USER,
                "every= spaces the samples of the series record= names, and nothing is recorded",
                "name them: run(..., record=[swim.speed], every=...), or drop every=")])
        spacing = self._spacing(every)
        if recorded or history:
            self.warn_aliasing(spacing, "the recorded series")
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
        self._refuse_past_nodes_reach(start + steps)
        if self._runs.callback is not None:  # a run starts from the declared inputs: none a live session set
            self._runs.callback.live.clear()
        kept = history or bool(recorded)
        keep = None if history or not recorded else frozenset(
            {(output.node, output.field) for output in recorded}
            | {(flag.node, flag.field) for flag in self.layout.flags}
            | {(plan.node, plan.field) for plan in self._plans.watches if isinstance(plan, ContainmentPlan)})
        if point is not None and start == 0:
            for name in point.starting:
                set_node_state(self._graph, name, point.point.state[name])
        final, stacked = self._runs.run(steps, history=kept, hook=hook, chunk=size, keep=keep,
                                        automatic=hook is None and self._runs.chunk is None,
                                        stop=self._stopped_by_nodes(stop),
                                        tests=self._flag_tests(), params=point.params if point is not None else None)
        taken = self._runs.steps - start
        why = self._runs.stopped
        result = self.result_of(start, taken, final, stacked, recorded if not history else self.layout.per_step(),
                                spacing, history=history,
                                stopped=Stopped(why, steps) if why is not None else None, tally=self._runs.tally,
                                point=point)
        self._warn_raised(result)
        refusal = non_finite(final, stacked, self._report.nodes, start, self._plans.dt, kept_all=history,
                             already=self._non_finite)
        if refusal is not None:
            if self._non_finite is None:
                self._non_finite = NonFinite(first_non_finite(refusal), start + taken)
            raise RunFailed(stopped_refusals(self._graph, final) or [refusal], result)
        # A node that stops a run with its state finite (the flow's CFL check, which holds its state from the step
        # past its line) names the step it stopped at in its state; the run ended at that chunk, and is refused.
        stopped = stopped_refusals(self._graph, final)
        if stopped:
            raise RunFailed(stopped, result)
        return result

    def _refuse_past_nodes_reach(self, end: int) -> None:
        """Refuse, before it runs, a run that would take a node past the steps it was built to keep a record of
        (``run_problem(end)``, where a node has one: the flow's per-step derivative reports), never clamped."""
        found = [(name, problem) for name in self._graph.node_names
                 if callable(check := getattr(self._graph.get_node(name), "run_problem", None))
                 and (problem := cast("Callable[[int], Problem | None]", check)(end)) is not None]
        if found:
            raise ArgumentRefused([Refusal(problem.code, RefusalClass.USER, f"node {name}: {problem.message}",
                                           problem.action) for name, problem in found])

    def _stopped_by_nodes(self, asked: StopRequest | None) -> StopRequest | None:
        """``asked``, and before it the nodes that stop a run (``STOP_FIELD``, a node's step count it stopped at, -1
        until it has): a run one of them has stopped ends at the chunk the stop lies in. Only that field is read from
        the device between chunks."""
        from mime.build._maddening import node_state

        watched = [(name, field) for name in self._graph.node_names
                   if isinstance(field := getattr(self._graph.get_node(name), "STOP_FIELD", None), str)]
        if not watched:
            return asked

        def stop(progress: Progress) -> str | None:
            for name, field in watched:
                at = int(np.asarray(node_state(self._graph, name)[field]))
                if at >= 0:
                    return f"node {name} stopped the run at its step {at}"
            return asked(progress) if asked is not None else None

        return stop

    def _warn_raised(self, result: RunResult) -> None:
        """A warning for each flag that warns (``Flag.warns``) and the run raised, with its count and where."""
        warns = {f"{plan.node_name(flag.node)}.{flag.field}" for plan in self._plans.entries
                 for flag in plan.registered.cls.raises
                 if flag.warns and (flag.rung is None or flag.rung == getattr(plan.entry, "rung", None))}
        for read in result.flags:
            if read.path in warns and read.was_raised:
                warn_at_script(f"{read.flag.owner} ({read.flag.entry}): {read.text(meaning=True)}", UserWarning)

    @property
    def declared(self) -> tuple[tuple[Declaration, ...] | None, RunSettings | None]:
        """The declarations and run settings it was built from (``None`` for a build made without an experiment)."""
        return self._declared

    @property
    def saved_form(self) -> SavedForm | None:
        """The saved form it was loaded from or saved to, whose pins a build of its points keeps; ``None`` for neither."""
        return self._saved

    @property
    def law(self) -> ControlLaw | None:
        """The controller's law the build made from its settings, or ``None``."""
        return self._law

    @property
    def steps_taken(self) -> int:
        """The base steps its runs have taken since the graph's start."""
        return self._runs.steps

    def recording(self, record: Sequence[OutputRef | ObservationHandle | str]) -> list[OutputLayout]:
        """The per-step outputs ``record`` names, as :meth:`run` resolves them."""
        return self._recording(record)

    def spacing(self, every: Quantity | None) -> int:
        """``every`` in base steps, as :meth:`run` takes it."""
        return self._spacing(every)

    def warn_aliasing(self, spacing: int, what: str) -> None:
        """Warns where ``what``, sampled every ``spacing`` base steps, is as sparse as half the period of a declared
        drive's frequency or sparser (``mime.experiment._aliasing``): never a refusal."""
        from mime.experiment._aliasing import warn_aliasing

        warn_aliasing(self._plans.turning, spacing, self._plans.dt, what)

    def flag_tests(self) -> tuple[FlagTest, ...]:
        """Each flag of the layout as a scan tests it (:meth:`_flag_tests`)."""
        return self._flag_tests()

    def _flag_tests(self) -> tuple[FlagTest, ...]:
        """Each flag of the layout as a scan tests it, for a run that keeps no step's state to reduce them over."""
        return tuple(FlagTest(flag.node, flag.field, flag.limits) for flag in self.layout.flags)

    def _poses_kept(self, stacked: History | None) -> History | None:
        """The poses of the bodies the containment watch reads, from ``stacked`` where the run kept them, and nothing
        else of it, so that a result holds no more than the watch needs until the watch is read."""
        if stacked is None:
            return None
        return {plan.node: {plan.field: stacked[plan.node][plan.field]} for plan in self._plans.watches
                if isinstance(plan, ContainmentPlan) and plan.field in stacked.get(plan.node, {})}

    def result_of(self, start: int, steps: int, final: History, stacked: History | None = None,
                  recorded: Sequence[OutputLayout] = (), spacing: int = 1, *, history: bool = False,
                  stopped: Stopped | None = None, tally: FlagTally | None = None, point: Overridden | None = None,
                  plans: PlanSet | None = None, report: Mapping[str, Any] | None = None) -> RunResult:
        """The result of ``steps`` steps from step ``start`` that ended at ``final``: the series ``recorded`` names from
        ``stacked`` (every step's kept state, where the run kept it), one every ``spacing``, and each flag over the
        steps kept; or, where none were, over all of them as the run reduced it (``tally``, in the order of the layout's
        flags), or at the last step alone. An overridden run's (``point``) carries its point's layout, watches and
        report, and its overrides; a study's member's, its own ``plans`` and ``report``."""
        if point is not None:
            from mime.study._study import member_report

            plans = point.point.plans
            report = member_report(self, point.point)
        layout = run_layout(plans) if plans is not None else self.layout
        watched = plans.watches if plans is not None else self._plans.watches
        tracks: dict[str, FlagTrack] = {}
        series: dict[str, Samples] = {}
        if stacked is not None:
            counts = np.arange(start + 1, start + steps + 1, dtype=np.int64)
            chosen = counts % spacing == 0
            for output in recorded:
                series[output.path] = Samples(counts[chosen], np.asarray(stacked[output.node][output.field])[chosen])
            for flag in layout.flags:
                raised = flag.raised(stacked[flag.node][flag.field])
                tracks[flag.path] = FlagTrack(raised, bool(raised[-1]))
        else:
            tracks = {flag.path: FlagTrack(None, bool(flag.raised(np.asarray(final[flag.node][flag.field])[None])[0]),
                                           _whole_run(tally, index))
                      for index, flag in enumerate(layout.flags)}
        result = RunResult(steps, final, stacked if history else None, layout=layout, start_step=start,
                         recorded=series, tracks=tracks, every=spacing if stacked is not None else None,
                         watches=read_watches(watched, final, stacked if history else None),
                         containment=functools.partial(read_containments, watched, final,
                                                       self._poses_kept(stacked), last_step=start + steps,
                                                       dt=self._plans.dt),
                         report=report if report is not None else self.report_summary(), stopped=stopped,
                         origin=self._origin, overrides=point.records if point is not None else (),
                         energy=functools.partial(energy_account, self._graph, final,
                                                  (point.params if point is not None else graph_params(self._graph))
                                                  .get("nodes", {}), steps=start + steps)
                         if self._report.energy else None,
                         groups=functools.partial(group_verdicts, self._graph) if self._report.groups else None)
        result.drives = (plans if plans is not None else self._plans).turning
        return result

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
        lines = {line.node: line for line in self._report.nodes}
        owners = {node: plan for plan in self._plans.entries for node in plan.node_names()}
        read = {(e.source[0], e.source[1], e.target[0], e.target[1]): e.lag.value for e in self._report.lags}
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
                                    for g in self._report.groups],
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
            self._summary = report_summary(self._report, self.layout)
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

        self._fresh()
        return LiveSession(self, chunk=chunk, pace=pace, select=select, out=Path(out) if out is not None else None)

    @public
    def run_to(self, out: str | PathLike[str], duration: Quantity, *, history: bool = False,
               record: Sequence[OutputRef | ObservationHandle | str] = (), every: Quantity | None = None,
               source: str | PathLike[str] | None = None, load_seconds: float | None = None,
               hook: ChunkHook | None = None, chunk: Quantity | None = None,
               stage_every: Quantity | None = None, stop: StopRequest | None = None,
               progress: bool | None = None, override: Mapping[Any, object] | None = None) -> RunWritten:
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

        ``hook`` and ``chunk`` stream the run as :meth:`run` does, and so does ``stop``: a run it ended is written as
        the steps it reached, ``outputs.json`` and ``run.json`` holding ``stopped`` (the reason, the steps reached
        and the steps and time asked for), and ``RunWritten.stopped`` set. ``outputs.json`` is written last, so that a
        run quit part-way through writing is not a folder that reads. ``stage_every`` writes the recording layer,
        ``recording.usdc``: the bound prims' poses and vectors (:meth:`binding`) over the folder's own stage, at the
        run's start and every ``stage_every`` after it, a whole number of chunks (the run is chunked for it and keeps
        no step's state; chunks of two steps or more change nothing it computes, and chunks of one step can change its
        last bits, as a scan of one step can differ from a longer one: measured in about half of the saved round-trip
        cases, in float32 and in float64). ``topology.json`` is written beside the outputs, and
        ``report.json``, the build report's summary (:meth:`report_summary`), beside ``run.json``.

        ``progress`` shows the live display as :meth:`run` does, and after the run, while the folder is written, a
        phase saying so: the run's saved form runs its reference steps from the initial state on the run's own graph,
        which is put back where the run left it (:meth:`save`). A run chunked by one step that keeps its steps' state
        (``record`` or ``history``, with ``stage_every``, ``chunk`` or a controller of one step) has compiled their
        program already; any other run compiles it once more.

        ``override`` runs the point :meth:`run` describes: ``run.json`` records each override (its path, its value with
        its unit, its SI value and the leaves it writes), ``report.json`` is the point's, every output is stamped
        "overridden", and ``experiment/`` is this build's own saved form, unchanged; ``mime verify --run`` replays them.

        As :meth:`run`, a stale build is refused in a notebook or IPython and warned about in a script."""
        self._fresh()
        point = self._overridden(override)
        display = _python_display(progress)
        if display is None:
            return self._run_to(out, duration, history=history, record=record, every=every, source=source,
                                load_seconds=load_seconds, hook=hook, chunk=chunk, stage_every=stage_every,
                                stop=stop, watch=None, point=point)
        try:
            watch = self._watch(display, duration, stop)
            return self._run_to(out, duration, history=history, record=record, every=every, source=source,
                                load_seconds=load_seconds, hook=hook, chunk=chunk, stage_every=stage_every,
                                stop=watch.stop, watch=watch, point=point)
        finally:
            display.close()

    def _run_to(self, out: str | PathLike[str], duration: Quantity, *, history: bool,
                record: Sequence[OutputRef | ObservationHandle | str], every: Quantity | None,
                source: str | PathLike[str] | None, load_seconds: float | None, hook: ChunkHook | None,
                chunk: Quantity | None, stage_every: Quantity | None, stop: StopRequest | None,
                watch: Watch | None, point: Overridden | None = None) -> RunWritten:
        import time

        started = time.perf_counter()
        read = [source] if source is not None else [self._saved.folder] if self._saved is not None else []
        refuse_out_over(Path(out), read)
        failed: RunFailed | None = None
        samples, hook, chunk = self._stage_samples(stage_every, hook, chunk)
        try:
            result = self._run(duration, history=history, record=record, every=every, hook=hook, chunk=chunk,
                               stop=stop, point=point)
        except RunFailed as error:
            if not isinstance(error.result, RunResult):
                raise
            # A state that is not finite is written as it is, and marked, then refused: on a remote machine the
            # outputs are what there is to look at.
            failed, result = error, error.result
        ran = time.perf_counter()
        if watch is not None:
            watch.saving(result.steps)
        target = Path(out)
        target.mkdir(parents=True, exist_ok=True)
        self._save(target / RUN_EXPERIMENT, DEFAULT_STEPS)
        np.savez(target / RUN_OUTPUTS, allow_pickle=False, **result.arrays())
        manifest = result.manifest()
        refused = failed.refusals[0].message if failed is not None else None
        if refused is not None:
            manifest["non_finite"] = refused
        (target / RUN_TOPOLOGY).write_text(json.dumps(self.topology(), indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8", newline="\n")
        summary = result.report if point is not None and result.report is not None else self.report_summary()
        (target / RUN_REPORT).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
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
        if result.stopped is not None:
            written["stopped"] = manifest["stopped"]
        if point is not None:
            written["overrides"] = [dict(record) for record in point.records]
        (target / RUN_RECORD).write_text(json.dumps(written, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        (target / RUN_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8", newline="\n")
        read_back = RunResult.read(target)
        read_back.origin = self._origin
        remember_folder(target, self._origin)
        done = RunWritten(result.steps, target / RUN_OUTPUTS, target / RUN_RECORD, target / RUN_EXPERIMENT, written,
                          target / RUN_MANIFEST, read_back, recording, result.stopped, self._origin)
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
        self.warn_aliasing(spacing, "the stage recording")
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

    def raised_flags(self) -> list[str]:
        """The paths of the flags raised at the step the graph stands at, read from the device: what a display of a
        run in progress (``mime run`` on a terminal) asks between chunks. It waits for the chunk's result and changes
        nothing the run computes."""
        from mime.build._maddening import node_state

        return [flag.path for flag in self.layout.flags
                if bool(flag.raised(np.asarray(node_state(self._graph, flag.node)[flag.field])[None])[0])]

    def _history_on(self, graph: GraphManager, steps: int, program: ReferenceProgram | None) -> History:
        """The first ``steps`` steps of ``graph`` from where it stands, run by ``program`` (``one-step``: every step its
        own compiled scan; ``None``: the history scan) in whole control periods, and the steps past ``steps``
        dropped."""
        runs = ChunkedRun(graph, self._plans, self._law)
        period = runs.chunk or 1
        _, stacked = runs.run(-(-steps // period) * period, history=True, one_step=program == "one-step")
        assert stacked is not None
        return {node: {name: value[:steps] for name, value in fields.items()} for node, fields in stacked.items()}

    def _reference(self, steps: int) -> tuple[History, History, ReferenceProgram | None]:
        """The reference outputs a save writes, the first ``steps`` steps from the initial state, the same steps run
        again from a start one unit in the last place off (``mime.saved._scales.one_ulp_up``), from which the save
        measures each field's resolution, and the program that ran both. Taken on this build's own graph, reset to its
        initial state and put back where it stood after (:func:`~mime.build._maddening.state_put_back`), so nothing is
        built: with the history scan of one step where the graph holds it (a run chunked by one step that kept its
        steps' state compiled it, so nothing is compiled either), else with the history scan of whole control periods.
        Where the compiled step reads anything the build did not give it (:func:`~mime.build._maddening.step_reads`:
        ``graph``, edited, bypasses every check the build made), on a graph built afresh, as before. ``verify`` reruns
        the lock's program on a fresh graph either way."""
        graph = self._graph
        if not step_reads(graph).unchanged_since(self._built_reads):
            fresh = construct(self._plans, self._law)[0]
            return (*self._twice_on(fresh, steps, None), None)
        program: ReferenceProgram | None = "one-step" if holds_one_step_history(graph) else None
        with state_put_back(graph):
            graph.reset_state()
            return (*self._twice_on(graph, steps, program), program)

    def _twice_on(self, graph: GraphManager, steps: int, program: ReferenceProgram | None) -> tuple[History, History]:
        """The first ``steps`` steps of ``graph`` from its initial state, where it stands, and again by ``program`` from
        a start one ulp off: every nonzero floating number of the initial state and of the small parameters the
        compiled step reads (``mime.build._maddening.params_one_ulp_up``) one ulp up. The parameters are moved too
        because a body released at rest at the origin starts from zeros, which a change of the state alone cannot move
        (measured on the tube demo: every block's divergence zero). The second run's step compiles nothing the first's
        did not; a value a node makes from moved parameters and holds among them (an MRS drag's matrix) is made again
        from them once, before it (``mime.core.MimeNode.made_again``). The parameters are put back after."""
        from mime.build._maddening import device_array, node_state, params_one_ulp_up

        start = {name: {field: np.asarray(value) for field, value in node_state(graph, name).items()}
                 for name in graph.node_names}
        history = self._history_on(graph, steps, program)
        graph.reset_state()
        for name, fields in one_ulp_up(start).items():
            set_node_state(graph, name, {field: device_array(value) for field, value in fields.items()})
        with params_one_ulp_up(graph):
            return history, self._history_on(graph, steps, program)

    @public
    def save(self, folder: str | PathLike[str], *, steps: int = DEFAULT_STEPS, progress: bool | None = None) -> Lock:
        """Write the document, the lock, the scene (``scene/scene.usda`` and the asset layers it references, under
        ``scene/layers/``) and the reference outputs (every field over the first ``steps`` steps, run from the
        initial state) into ``folder``. The reference steps are run on this build's graph, reset and then put back
        where its runs left it, so a run continues from there bit for bit: by the history scan of one step where a run
        chunked by one step compiled it (the lock names it, ``one-step``, and nothing of the step is compiled), else by
        the history scan; on a graph built afresh where :attr:`graph` was edited since the build. Raises
        ``BuildRefused`` when the experiment cannot be saved: a setting the document cannot hold, or an entry a load
        elsewhere could not find.
        ``progress`` shows the live display's phase while it saves, as :meth:`run` decides where (in a notebook, by
        default). A stale build is refused in a notebook or IPython and warned about in a script, as :meth:`run`'s
        is."""
        self._fresh()
        display = _python_display(progress)
        if display is None:
            return self._save(folder, steps)
        from mime.cli._display import SAVING

        try:
            display.phase(SAVING)
            return self._save(folder, steps)
        finally:
            display.close()

    def _save(self, folder: str | PathLike[str], steps: int) -> Lock:
        path = Path(folder)
        try:
            history, nudged, program = self._reference(steps)
            lock = write_folder(path, self.record, history, steps, program, self._resolution(history, nudged))
        except OSError as error:
            raise _not_writable(path, error) from error
        self._saved = read_folder(path)
        if nondeterministic_gpu(cast(Mapping[str, object], lock["environment"])):
            warn_at_script(SAVED_DETERMINISTIC_OFF, NondeterministicReference)
        return lock

    def declared_units(self) -> dict[tuple[str, str], str]:
        """The units MIME declares for each state field it declares them for, by ``(node, field)``: its node's symbol
        table, else its node's typed state (``mime.core.typed_units``), the edges that read it, the stage's binding,
        the observations' outputs, and a flag's one."""
        from mime.build._maddening import node_fields_of
        from mime.build._report import graph_config
        from mime.core import typed_units

        found: dict[tuple[str, str], str] = {}
        for name in self._graph.node_names:
            node, fields = self._graph.get_node(name), node_fields_of(self._graph, name)
            for field, units in symbol_units(node, fields).items():
                found[(name, field)] = units
            for field, units in typed_units(node, fields).items():
                found.setdefault((name, field), units)
        for edge in graph_config(self._graph)["edges"]:
            if edge.get("source_units"):
                found.setdefault((str(edge["source_node"]), str(edge["source_field"])), str(edge["source_units"]))
        for entry in self.binding().entries:
            found.setdefault((entry.node, entry.field), entry.unit)
        for observation in self.layout.observations:
            for output in observation.outputs:
                found.setdefault((output.node, output.field), output.unit)
        for flag in self.layout.flags:
            found.setdefault((flag.node, flag.field), "1")
        return found

    def field_units(self, history: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, FieldUnits]:
        """Each field of ``history`` by its label (``node.field``): the units MIME declares for it
        (:meth:`declared_units`), which split it into blocks, and, for held points, the forces they pair with (read
        from the node's typed state, ``mime.core.field_kinds``): what ``verify`` judges a field by where it claims no
        bits (``mime.saved._scales``)."""
        from mime.core import FieldKind, field_kinds, paired_forces

        units = self.declared_units()
        found: dict[str, FieldUnits] = {}
        for name, fields in sorted(history.items()):
            node = self._graph.get_node(name) if name in self._graph.node_names else None
            kinds = field_kinds(node, tuple(fields)) if node is not None else {}
            forces = paired_forces(kinds)
            for field in fields:
                paired = f"{name}.{forces}" if forces is not None and kinds.get(field) is FieldKind.HELD_POINTS else None
                found[f"{name}.{field}"] = FieldUnits(units.get((name, field)), paired)
        return found

    def _resolution(self, history: History, nudged: History) -> dict[str, dict[str, float | None]]:
        """Each field's resolution by block: its largest divergence over the reference steps run from a start one ulp
        off, held points' on the rows whose force acts (``mime.saved._scales.resolution``)."""
        fields = self.field_units(history)
        found: dict[str, dict[str, float | None]] = {}
        for name, held in history.items():
            for field, value in held.items():
                units = fields[f"{name}.{field}"]
                forces = None
                if units.forces is not None:
                    node, _, paired = units.forces.partition(".")
                    forces = (np.asarray(history[node][paired]), np.asarray(nudged[node][paired]))
                found[f"{name}.{field}"] = resolution(np.asarray(value), np.asarray(nudged[name][field]), units,
                                                      forces=forces)
        return found

    @public
    def verify(self) -> VerifyReport:
        """Rerun the reference outputs' steps from the initial state, on a graph built afresh, by the program the lock
        names, and compare: bit for bit in the environment the lock was written in, unless its reference was made on a
        GPU with XLA's deterministic mode off, and otherwise within the lock's tolerance, from the tenth form each field
        block by block, on each block's largest magnitude or its measured resolution (:meth:`field_units`)."""
        if self._saved is None:
            raise ArgumentRefused([Refusal(
                "verify.nothing-saved", RefusalClass.USER,
                "nothing to verify against: this experiment was neither loaded nor saved",
                "load a saved one, Experiment.load(folder).build().verify(), or save this one first: "
                "built.save(folder)")])
        reference = self._saved.lock["reference"]
        graph, _, _ = construct(self._plans, self._law)
        history = self._history_on(graph, reference["steps"], reference.get("program"))
        return verify_history(self._saved, history, self.field_units(history))
