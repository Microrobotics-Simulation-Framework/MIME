"""The chunked run: compiled scans between host calls, never a host call inside one. A controller in the callback mode
is called once every control period on the host, with what it reads from the graph's state as the period starts; its
writes are fed to the graph as external inputs, held through the steps they act on, from the step its latency puts
them at. A per-chunk hook is called after every chunk with the chunk's end and the state there."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast, override

import numpy as np
from maddening import GraphManager

from mime.build import ControllerPlan, Mode, PlanSet
from mime.build._in_graph import step_place
from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass, RunFailed, standing_for
from mime.build._maddening import Fed, NodeFields, fed_inputs, inputs, node_state, scan_chunk, \
    scan_chunk_with_history
from mime.core import jnp
from mime.registry import BuildContext, ControlLaw

type History = dict[str, dict[str, np.ndarray]]
type Arrays = Mapping[str, Any]


@dataclass(frozen=True)
class Chunk:
    """A chunk of a run, as the per-chunk hook sees it once it has run: its index in the ``run`` call (counted from
    zero in each call, where ``steps`` is cumulative), the base steps taken since the graph's start at its end, the
    simulated time there (s), and every node's state there by node and field."""

    index: int
    steps: int
    time: float
    state: History


type ChunkHook = Callable[[Chunk], None]


class ControllerError(RunFailed):
    """A controller's step raised while it ran, or returned writes that are not those it declares, or of another
    shape. One the step raised keeps its error as ``original``, and is of its class too (``except Done`` catches
    it)."""


def _to_numpy(state: NodeFields, keep: frozenset[tuple[str, str]] | None = None) -> History:
    """``state`` on the host: every field, or those ``keep`` names (node, field)."""
    return {node: {name: np.asarray(value) for name, value in fields.items() if keep is None or (node, name) in keep}
            for node, fields in state.items()}


#: How much of the graph's state a scan that keeps every step's holds on the device at once: a longer stretch is
#: scanned in blocks of as many steps as fit.
HISTORY_BLOCK_BYTES = 256 * 2**20


class ReadModels(Mapping[str, Any]):
    """The built trees a controller's context holds (``ctx.models``), by segment id, recording each one the law reads
    while it is made (``read``), so that the build report says which model a controller took."""

    def __init__(self, models: Mapping[str, Any]) -> None:
        self._models = dict(models)
        self.read: set[str] = set()

    @override
    def __getitem__(self, key: str) -> Any:
        held = self._models[key]
        self.read.add(key)
        return held

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._models)

    @override
    def __len__(self) -> int:
        return len(self._models)


def control_context(plans: PlanSet) -> BuildContext:
    """What a controller's law is made from: its settings in SI, the base step, its control period as
    ``timestep``, and the built tree of each segment that carries mass, what is mounted on it grafted on (a
    :class:`ReadModels`, which records the trees the law reads)."""
    plan = cast(ControllerPlan, plans.controller)
    return BuildContext(plan.owner, plans.dt, plans.dt * plan.period, plan.owner, settings=dict(plan.settings),
                        up=plans.frame.up, gravity=plans.frame.gravity, models=ReadModels(plans.models))


def make_law(plans: PlanSet) -> tuple[ControlLaw | None, list[str]]:
    """The controller's law made from its settings, and every problem with it: a law that cannot be made, or whose
    held writes are not its writes at their shapes. ``None`` and no problem when there is no controller."""
    law, problems, _ = law_and_models(plans)
    return law, problems


def law_and_models(plans: PlanSet) -> tuple[ControlLaw | None, list[str], tuple[str, ...]]:
    """:func:`make_law`, and the segments whose built trees (``ctx.models``) the law read while it was made, in
    order."""
    plan = plans.controller
    if plan is None:
        return None, [], ()
    context = control_context(plans)
    models = cast(ReadModels, context.models)
    try:
        law = plan.entry.law(context)
    except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
        return None, [f"law() raised {error!r}"], tuple(sorted(models.read))
    if not isinstance(cast(object, law), ControlLaw):
        return None, [f"law() returned {law!r}, not a mime.registry.ControlLaw"], tuple(sorted(models.read))
    problems = _write_problems(plan, law.held, "its held writes")
    return (law if not problems else None), problems, tuple(sorted(models.read))


def _write_problems(plan: ControllerPlan, writes: object, what: str) -> list[str]:
    if not isinstance(writes, Mapping):
        return [f"{what} are {writes!r}, not a mapping of its write labels to arrays"]
    held = cast(Mapping[object, object], writes)
    labels = {write.label for write in plan.writes}
    problems = [f"{what} hold {sorted(map(str, set(held) ^ labels))!r}: they are exactly its write labels "
                f"{sorted(labels)}"] if set(held) != labels else []
    for write in plan.writes:
        if write.label in held:
            shape = tuple(np.shape(cast(Any, held[write.label])))
            if not (len(shape) == len(write.shape) and all(a == b or b < 0 for a, b in zip(shape, write.shape,
                                                                                            strict=True))):
                problems.append(f"{what}: {write.label!r} has shape {shape}, and {write.path} takes {write.shape}")
    return problems


class Callback:
    """A controller in the callback mode, bound to one graph: its law, the state it carries from tick to tick, and the
    writes waiting to act, each with the step it starts to act on."""

    def __init__(self, plan: ControllerPlan, law: ControlLaw, graph: GraphManager, dt: float = 0.0) -> None:
        self.plan = plan
        self.law = law
        self.state: Arrays = law.state
        self.dtypes = {write.label: inputs(graph.get_node(write.node))[write.field].dtype for write in plan.writes}
        self.in_force = self.fed(law.held)
        self.waiting: deque[tuple[int, Fed]] = deque()
        self.ticks = 0
        self.dt = dt
        #: The live inputs a session has set, by label (:class:`~mime.registry.LiveInput`): read by the step under
        #: their labels beside its reads, from the next tick on. None until one is set, so a run with no session calls
        #: the step with its reads alone.
        self.live: dict[str, Any] = {}

    def fed(self, writes: Arrays) -> Fed:
        """Writes by label as the graph's external inputs, by node and field, each in its input's dtype."""
        found: dict[str, dict[str, Any]] = {}
        for write in self.plan.writes:
            found.setdefault(write.node, {})[write.field] = jnp.asarray(writes[write.label],
                                                                        dtype=self.dtypes[write.label])
        return found

    def tick(self, graph: GraphManager, step: int) -> None:
        """One call of the controller at the start of step ``step``: it reads the state as it stands there, and its
        writes start to act ``latency`` steps later."""
        reads = {read.label: node_state(graph, read.node)[read.field] for read in self.plan.reads}
        label = self.plan.registered.cls.label()
        try:
            self.state, writes = self.law.step(self.state, {**reads, **self.live} if self.live else reads)
        except Exception as error:
            from mime.errors import first_lines, script_place

            place = script_place(error)
            refused = ControllerError([Refusal(
                "control.law-raised", RefusalClass.USER,
                f"the controller {label}'s step raised at its tick {self.ticks} (t = {step * self.dt:.6g} s): "
                f"{first_lines(error)}",
                f"correct the step at the line named; its reads are {', '.join(map(repr, reads)) or 'none'}, as "
                f"set_controller(reads=...) binds them",
                why="MIME calls the step on the host once every control period, and it raised there",
                where=place.text() if place is not None else step_place(self.law.step))])
            refused.original = error
            standing = standing_for(refused, error)
            if standing is None:  # its class cannot be a base: the step's error leaves as it raised it
                raise
            raise standing from error
        problems = _write_problems(self.plan, writes, f"the writes of tick {self.ticks}")
        if problems:
            raise ControllerError([Refusal(
                "control.law-writes", RefusalClass.USER, f"the controller {label}: " + "; ".join(problems),
                f"return exactly its write labels ({', '.join(repr(w.label) for w in self.plan.writes)}), each at "
                f"the shape its command takes, as set_controller(writes=...) binds them",
                why="each write is fed to its command as an input of fixed shape",
                where=step_place(self.law.step))])
        self.waiting.append((step + self.plan.latency, self.fed(writes)))
        self.ticks += 1

    def at(self, step: int) -> Fed:
        """The writes in force at step ``step``: the last whose step has come, or those held before the first."""
        while self.waiting and self.waiting[0][0] <= step:
            self.in_force = self.waiting.popleft()[1]
        return self.in_force

    def next_change(self, step: int) -> int | None:
        """The first step after ``step`` at which other writes start to act, if any wait."""
        later = [start for start, _ in self.waiting if start > step]
        return min(later) if later else None


class ChunkedRun:
    """A built graph's runs, continued from one to the next: the steps it has taken and, with a controller in the
    callback mode, its binding. ``chunk`` is the control period in base steps, or ``None`` when no controller is
    called between chunks: none, or one inside the compiled step, whose state the graph carries."""

    def __init__(self, graph: GraphManager, plans: PlanSet, law: ControlLaw | None) -> None:
        self.graph = graph
        self.dt = plans.dt
        self.steps = 0
        controller = plans.controller
        called = controller is not None and controller.mode is Mode.CALLBACK and law is not None
        self.callback = Callback(controller, law, graph, plans.dt) \
            if called and controller is not None and law is not None else None
        self.chunk = controller.period if self.callback is not None and controller is not None else None
        self.checked = False
        self._block: int | None = None
        self._keep: frozenset[tuple[str, str]] | None = None

    def block(self) -> int:
        """The most steps one scan that keeps every step's state holds on the device: :data:`HISTORY_BLOCK_BYTES` over
        the bytes of one step's state."""
        if self._block is None:
            size = sum(np.asarray(value).nbytes for name in self.graph.node_names
                       for value in node_state(self.graph, name).values())
            self._block = max(1, HISTORY_BLOCK_BYTES // max(1, size))
        return self._block

    def run(self, steps: int, *, history: bool, hook: ChunkHook | None = None, chunk: int | None = None,
            keep: frozenset[tuple[str, str]] | None = None) -> tuple[History, History | None]:
        """``steps`` base steps from where the graph stands, as compiled chunks of ``chunk`` steps (the control period
        when a controller is called between them), the hook called after each; or as one compiled scan when neither a
        controller nor a hook asks for chunks. With ``history``, every step's state is kept, in scans of at most
        :meth:`block` steps, and of it only the fields ``keep`` names (node, field) when it is given."""
        size = self.chunk if self.chunk is not None else (chunk if hook is not None else None)
        stacked: list[History] = []
        self._keep = keep
        if size is None:
            final = self._scans(self.steps, self.steps + steps, history, stacked)
            return final, (_stacked(stacked) if history else None)
        if steps % size:
            nearest = [k * size for k in (steps // size, steps // size + 1) if k > 0]
            why = ("a controller called between compiled chunks is called once per chunk, its control period, so a "
                   "run is whole periods") if self.chunk is not None else \
                "a hook is called between compiled chunks of chunk= steps, so a run with one is whole chunks"
            raise ArgumentRefused([Refusal(
                "run.chunks", RefusalClass.USER,
                f"a run of {steps} base steps ({steps * self.dt:.6g} s) is not a whole number of chunks of {size} "
                "steps "
                f"({size * self.dt:.6g} s)",
                "run a whole number of chunks: run(s(...)) of " + " or ".join(f"{n * self.dt:.6g} s" for n in nearest),
                why=why)])
        final: History = {}
        for index in range(steps // size):
            start = self.steps
            if self.callback is not None:
                self.callback.tick(self.graph, start)
            final = self._scans(start, start + size, history, stacked)
            if hook is not None:
                hook(Chunk(index, self.steps, self.steps * self.dt, final))
        return final, (_stacked(stacked) if history else None)

    def _scans(self, start: int, end: int, history: bool, stacked: list[History]) -> History:
        """Steps ``start`` to ``end``, one compiled scan for each stretch through which the same writes act, each
        stretch's history appended to ``stacked``."""
        final: NodeFields = {}
        step = start
        while step < end:
            fed: Fed = {} if self.callback is None else self.callback.at(step)
            change = self.callback.next_change(step) if self.callback is not None else None
            stop = min(end, change) if change is not None else end
            if not self.checked:
                fed_inputs(self.graph, fed)
                self.checked = True
            if history:
                stop = min(stop, step + self.block())
                final, held = scan_chunk_with_history(self.graph, stop - step, fed)
                stacked.append(_to_numpy(held, self._keep))
            else:
                final = scan_chunk(self.graph, stop - step, fed)
            self.steps += stop - step
            step = stop
        return _to_numpy(final)


def _stacked(parts: list[History]) -> History:
    """Histories of successive scans, joined along their steps."""
    if len(parts) == 1:
        return parts[0]
    return {node: {name: np.concatenate([part[node][name] for part in parts]) for name in fields}
            for node, fields in parts[0].items()}


__all__ = ["HISTORY_BLOCK_BYTES", "Callback", "Chunk", "ChunkHook", "ChunkedRun", "ControllerError", "ReadModels",
           "control_context", "law_and_models", "make_law"]
