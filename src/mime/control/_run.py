"""The chunked run: compiled scans between host calls, never a host call inside one. A controller in the callback mode
is called once every control period on the host, with what it reads from the graph's state as the period starts; its
writes are fed to the graph as external inputs, held through the steps they act on, from the step its latency puts
them at. A per-chunk hook is called after every chunk with the chunk's end and the state there."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from maddening import GraphManager

from mime.build import ControllerPlan, PlanSet
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


class ControllerError(RuntimeError):
    """A controller's step returned writes that are not those it declares, or of another shape."""


def _to_numpy(state: NodeFields) -> History:
    return {node: {name: np.asarray(value) for name, value in fields.items()} for node, fields in state.items()}


def control_context(plans: PlanSet) -> BuildContext:
    """What a controller's law is made from: its settings in SI, the base step, and its control period as
    ``timestep``."""
    plan = cast(ControllerPlan, plans.controller)
    return BuildContext(plan.owner, plans.dt, plans.dt * plan.period, plan.owner, settings=dict(plan.settings),
                        up=plans.frame.up, gravity=plans.frame.gravity)


def make_law(plans: PlanSet) -> tuple[ControlLaw | None, list[str]]:
    """The controller's law made from its settings, and every problem with it: a law that cannot be made, or whose
    held writes are not its writes at their shapes. ``None`` and no problem when there is no controller."""
    plan = plans.controller
    if plan is None:
        return None, []
    try:
        law = plan.entry.law(control_context(plans))
    except Exception as error:  # noqa: BLE001 - an entry's defect becomes a refusal naming it
        return None, [f"law() raised {error!r}"]
    if not isinstance(cast(object, law), ControlLaw):
        return None, [f"law() returned {law!r}, not a mime.registry.ControlLaw"]
    problems = _write_problems(plan, law.held, "its held writes")
    return (law if not problems else None), problems


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

    def __init__(self, plan: ControllerPlan, law: ControlLaw, graph: GraphManager) -> None:
        self.plan = plan
        self.law = law
        self.state: Arrays = law.state
        self.dtypes = {write.label: inputs(graph.get_node(write.node))[write.field].dtype for write in plan.writes}
        self.in_force = self.fed(law.held)
        self.waiting: deque[tuple[int, Fed]] = deque()
        self.ticks = 0

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
        self.state, writes = self.law.step(self.state, reads)
        problems = _write_problems(self.plan, writes, f"the writes of tick {self.ticks}")
        if problems:
            raise ControllerError(f"the controller {self.plan.registered.cls.label()}: " + "; ".join(problems))
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
    called between chunks."""

    def __init__(self, graph: GraphManager, plans: PlanSet, law: ControlLaw | None) -> None:
        self.graph = graph
        self.dt = plans.dt
        self.steps = 0
        controller = plans.controller
        self.callback = Callback(controller, law, graph) if controller is not None and law is not None else None
        self.chunk = controller.period if self.callback is not None and controller is not None else None
        self.checked = False

    def run(self, steps: int, *, history: bool, hook: ChunkHook | None = None,
            chunk: int | None = None) -> tuple[History, History | None]:
        """``steps`` base steps from where the graph stands, as compiled chunks of ``chunk`` steps (the control period
        when a controller is called between them), the hook called after each; or as one compiled scan when neither a
        controller nor a hook asks for chunks."""
        size = self.chunk if self.chunk is not None else (chunk if hook is not None else None)
        stacked: list[History] = []
        if size is None:
            final = self._scans(self.steps, self.steps + steps, history, stacked)
            return final, (_stacked(stacked) if history else None)
        if steps % size:
            nearest = [k * size for k in (steps // size, steps // size + 1) if k > 0]
            raise ValueError(f"a run of {steps} base steps ({steps * self.dt:.6g} s) is not a whole number of chunks "
                             f"of {size} steps ({size * self.dt:.6g} s); the nearest runs are "
                             + " and ".join(f"{n} steps ({n * self.dt:.6g} s)" for n in nearest))
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
                final, held = scan_chunk_with_history(self.graph, stop - step, fed)
                stacked.append(_to_numpy(held))
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


__all__ = ["Callback", "Chunk", "ChunkHook", "ChunkedRun", "ControllerError", "control_context", "make_law"]
