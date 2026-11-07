"""MADDENING 0.4.x and JAX at MIME's boundary. Some of their type hints leave a container or a callable bare
(``dict``, ``Callable``), which a strict checker reads as unknown; each call MIME makes through one of those is made
here, once, and given the type MIME relies on. Nothing here changes what is called."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np
from jax import Array
from jax.extend.core import Jaxpr
from maddening import GraphManager
from maddening.core.edge import EdgeSpec
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.core.schedule import topological_sort
from maddening.core.static_data import StaticArray
from numpy.typing import NDArray

type NodeFields = dict[str, dict[str, Array]]


def _held(value: object) -> Any:
    return value


def add_edge(graph: GraphManager, source: tuple[str, str], target: tuple[str, str], *, additive: bool,
             units: str) -> None:
    _held(graph).add_edge(source[0], target[0], source[1], target[1], additive=additive, source_units=units,
                          target_units=units)


def add_external_input(graph: GraphManager, node: str, field: str) -> None:
    """An external input into ``node``'s input ``field``, at the shape and dtype that input declares, so that what
    the run feeds is cast to what the node computes in."""
    spec = inputs(graph.get_node(node))[field]
    _held(graph).add_external_input(node, field, shape=tuple(cast(tuple[int, ...], _held(spec).shape)),
                                    dtype=spec.dtype)


def external_inputs(graph: GraphManager) -> list[tuple[str, str]]:
    """Every external input the graph declares, as ``(node, field)``, read from its own description of itself."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        held = cast(list[dict[str, Any]], graph_dict(graph, strict_mappings=False)["external_inputs"])
    return [(spec["target_node"], spec["target_field"]) for spec in held]


def add_coupling_group(graph: GraphManager, nodes: Sequence[str], *, max_iterations: int, tolerance: float) -> None:
    """A coupling group with every setting that decides what it returns written out: ``"ift"`` is exact only at a
    converged fixed point, and ``atol`` above zero can accept a group that has not converged."""
    _held(graph).add_coupling_group(list(nodes), max_iterations=max_iterations, tolerance=tolerance, solver="ift",
                                    atol=0.0)


def graph_schedule(graph: GraphManager) -> list[str]:
    """The order a compiled graph steps its nodes in."""
    return cast(list[str], _held(graph).schedule)


def graph_edges(graph: GraphManager) -> list[tuple[str, str, str, str]]:
    """Every edge as ``(source node, source field, target node, target field)``, in the order added."""
    return [(e.source_node, e.source_field, e.target_node, e.target_field) for e in graph.edges]


def edge_transform(edge: EdgeSpec) -> object | None:
    return cast(object | None, _held(edge).transform)


def inputs(node: SimulationNode) -> dict[str, BoundaryInputSpec]:
    return dict(node.boundary_input_spec() or {})


def zero_inputs(node: SimulationNode) -> dict[str, np.ndarray]:
    """A zero for every input ``node`` declares with a concrete shape, at that shape and dtype; an input whose shape
    leaves a dimension symbolic (negative) takes its shape from what the graph delivers."""
    zeros: dict[str, np.ndarray] = {}
    for name, spec in inputs(node).items():
        shape = cast(tuple[int, ...], _held(spec).shape)
        if all(size >= 0 for size in shape):
            zeros[name] = np.zeros(shape, dtype=np.dtype(spec.dtype or np.float32))
    return zeros


def resolved_inputs(graph: GraphManager, name: str) -> dict[str, Any]:
    """The inputs node ``name`` receives from the graph's current state: every incoming edge applied, mappings
    included, so each has the shape and dtype the compiled step gives it."""
    return cast(dict[str, Any], _held(graph).resolve_boundary_inputs(name))


def node_fields_of(graph: GraphManager, name: str) -> list[str]:
    """The fields of node ``name``'s state, as it starts."""
    return sorted(cast(dict[str, Any], _held(graph.get_node(name)).initial_state()))


def node_update(node: SimulationNode) -> Callable[..., object]:
    return cast(Callable[..., object], _held(node).update)


def node_fluxes(node: SimulationNode) -> Callable[..., object]:
    return cast(Callable[..., object], _held(node).compute_boundary_fluxes)


def node_dict(node: SimulationNode) -> dict[str, Any]:
    return cast(dict[str, Any], _held(node).to_dict())


def graph_dict(graph: GraphManager, *, strict_mappings: bool) -> dict[str, Any]:
    return cast(dict[str, Any], _held(graph).to_dict(strict_mappings=strict_mappings))


def static_data(node: SimulationNode) -> dict[str, Any]:
    return cast(dict[str, Any], _held(node).static_data)


def node_constructor_arguments(node: SimulationNode) -> dict[str, Any]:
    return cast(dict[str, Any], _held(node).params)


def node_state(graph: GraphManager, name: str) -> dict[str, Array]:
    return cast(dict[str, Array], _held(graph).get_node_state(name))


def parameter_leaves(graph: GraphManager) -> dict[str, dict[str, Any]]:
    """``params["nodes"]``: the leaves the compiled step reads, by node."""
    return cast(dict[str, dict[str, Any]], _held(graph).params.get("nodes", {}))


def parameter_specs(graph: GraphManager) -> dict[str, dict[str, ParamSpec]]:
    return cast(dict[str, dict[str, ParamSpec]], _held(graph).param_specs().get("nodes", {}))


type Fed = Mapping[str, Mapping[str, Any]]


def fed_inputs(graph: GraphManager, fed: Fed | None) -> dict[str, dict[str, Any]]:
    """``fed``, every external input of ``graph`` by node and field, checked: MADDENING fills an input nobody feeds
    with zeros, so one left out, or one the graph does not declare, raises ``run.unfed-input`` here (a MIME defect,
    since MIME derives and feeds every one; an ``InternalError``)."""
    given = {(node, field) for node, fields in (fed or {}).items() for field in fields}
    declared = set(external_inputs(graph))
    if given != declared:
        unfed = ", ".join(f"{n}.{f}" for n, f in sorted(declared - given)) or "none"
        extra = ", ".join(f"{n}.{f}" for n, f in sorted(given - declared)) or "none"
        from mime.build._refusals import MIME_DEFECT, InternalError, Refusal, RefusalClass

        raise InternalError([Refusal(
            "run.unfed-input", RefusalClass.INTERNAL,
            f"a run feeds every external input of the graph, and no other: not fed: {unfed}; fed and not declared: "
            f"{extra}", MIME_DEFECT,
            why="MIME derives every external input from the controller's writes and feeds each one, and MADDENING "
                "would fill one nobody feeds with zeros")])
    return {node: dict(fields) for node, fields in (fed or {}).items()}


def run_scan(graph: GraphManager, steps: int, params: Mapping[str, Any] | None = None,
             fed: Fed | None = None) -> NodeFields:
    return cast(NodeFields, _held(graph).run_scan(steps, fed_inputs(graph, fed), params=params))


def run_scan_with_history(graph: GraphManager, steps: int, fed: Fed | None = None) -> tuple[NodeFields, NodeFields]:
    return cast(tuple[NodeFields, NodeFields], _held(graph).run_scan_with_history(steps, fed_inputs(graph, fed)))


def scan_chunk(graph: GraphManager, steps: int, fed: Fed, params: Mapping[str, Any] | None = None) -> NodeFields:
    """``steps`` steps as one compiled scan, ``fed`` held through them: a chunk of a run whose inputs
    :func:`fed_inputs` has checked once, at its start. ``params`` is a parameter tree the run takes in place of the
    graph's own (an overridden run's), an argument of the compiled scan, which it does not recompile."""
    if params is None:
        return cast(NodeFields, _held(graph).run_scan(steps, fed))
    return cast(NodeFields, _held(graph).run_scan(steps, fed, params=params))


@dataclass(frozen=True)
class FlagTest:
    """What makes a flag raised at a step, as a scan evaluates it: the node and state field that hold it and, for a flag
    read from a field against limits, those limits (one per component): raised where any component's magnitude is above
    its limit. Without limits a flag is raised where any component of its field is not zero."""

    node: str
    field: str
    limits: tuple[float, ...] | None = None


@dataclass(frozen=True)
class FlagTally:
    """What a scan reduced of each flag it was asked to test, in the order it was given them: the steps at which the
    flag was raised (``count``), and the first and the last of those, as step counts (-1 for a flag never raised): from
    the scan's start, the first step being 1, as the scan returns them, or from the graph's start once :meth:`shifted`."""

    count: NDArray[np.int64]
    first: NDArray[np.int64]
    last: NDArray[np.int64]

    @staticmethod
    def none(flags: int) -> FlagTally:
        return FlagTally(np.zeros(flags, np.int64), np.full(flags, -1, np.int64), np.full(flags, -1, np.int64))

    def shifted(self, steps: int) -> FlagTally:
        """The tally of a scan that began ``steps`` steps into the run, counted from the run's start."""
        return FlagTally(self.count, np.where(self.first >= 0, self.first + steps, -1),
                         np.where(self.last >= 0, self.last + steps, -1))

    def then(self, later: FlagTally) -> FlagTally:
        """This tally followed by the one of the steps after it."""
        first = np.where(self.first >= 0, self.first, later.first)
        last = np.where(later.last >= 0, later.last, self.last)
        return FlagTally(self.count + later.count, first, last)


def _threshold(limit: float, dtype: np.dtype[Any]) -> float:
    """The largest value ``dtype`` holds that is not above ``limit``: a ``dtype`` value is above ``limit`` exactly when it
    is above this, so a scan comparing in ``dtype`` decides as the host's comparison in float64 does."""
    held = dtype.type(limit)
    if np.isfinite(limit) and float(held) > limit:
        held = np.nextafter(held, dtype.type(-np.inf))
    return float(held)


def scan_chunk_tallied(graph: GraphManager, steps: int, fed: Fed, tests: Sequence[FlagTest],
                       params: Mapping[str, Any] | None = None) -> tuple[NodeFields, FlagTally]:
    """:func:`scan_chunk` that also reduces each flag of ``tests`` over its steps, inside the scan, alongside the state
    and with no per-step output: how many steps raised it and the first and last, counted from the scan's start. The state it returns is the step function's, the one :func:`scan_chunk` runs; only the carry
    holds more. It does what MADDENING's ``run_scan`` does around its scan, because that has nowhere to carry more."""
    held = _held(graph)
    jnp = _held(jax.numpy)
    held._recover_from_escaped_tracers()  # noqa: SLF001
    held._check_static_data_dirty()  # noqa: SLF001
    if held._dirty or held._compiled_step is None:  # noqa: SLF001
        held.compile()
    held._refuse_xla_loop_hazards("run_scan", scan=True)  # noqa: SLF001
    external = held._resolve_external_inputs(dict(fed))  # noqa: SLF001
    params = held._params_or_default(params)  # noqa: SLF001
    count = len(tests)

    def build() -> Callable[..., Any]:
        step_fn = held._build_step_fn()  # noqa: SLF001

        def raised_by(state: Mapping[str, Mapping[str, Array]], test: FlagTest) -> Array:
            values = jnp.reshape(state[test.node][test.field], (-1,))
            if test.limits is None:
                return jnp.any(values != 0)
            limits = jnp.asarray([_threshold(limit, np.dtype(values.dtype)) for limit in test.limits],
                                       dtype=values.dtype)
            return jnp.any(jnp.abs(values) > limits)

        def scan(state: Any, ext: Any, parameters: Any) -> tuple[Any, Any]:
            held._count_scan_trace()  # noqa: SLF001

            def body(carry: Any, _: Any) -> tuple[Any, None]:
                state, counted, first, last, at = carry
                state = step_fn(state, ext, parameters)
                at = at + 1
                raised = jnp.stack([raised_by(state, test) for test in tests])
                return (state, counted + raised.astype(counted.dtype), jnp.where(raised & (first < 0), at, first),
                        jnp.where(raised, at, last), at), None

            none = FlagTally.none(count)
            init = (state, jnp.asarray(none.count, jnp.int32), jnp.asarray(none.first, jnp.int32),
                    jnp.asarray(none.last, jnp.int32), jnp.asarray(0, jnp.int32))
            (final, counted, first, last, _), _ = _held(jax).lax.scan(body, init, None, length=int(steps))
            return final, (counted, first, last)

        return _held(jax).jit(scan)

    key = ("mime_run_scan_tallied", int(steps), tuple(tests))
    program = held._cached_scan(key, build)  # noqa: SLF001
    final, reduced = held._call_surfacing_strict(program, held._state, external, params)  # noqa: SLF001
    held._store_state(final)  # noqa: SLF001
    counted, first, last = (np.asarray(each, np.int64) for each in reduced)
    return cast(NodeFields, held._user_state(final)), FlagTally(counted, first, last)  # noqa: SLF001


def parameter_tree(graph: GraphManager) -> dict[str, dict[str, Array]]:
    """The leaves each node's step reads, by node, as the graph's compile takes them from its nodes' constructor
    arguments (``params_pytree``): what ``params["nodes"]`` holds after a compile, read from a graph assembled and not
    compiled."""
    return cast(dict[str, dict[str, Array]], _held(graph)._snapshot_params()["nodes"])  # noqa: SLF001


def static_dependencies(node: SimulationNode) -> dict[str, tuple[str, ...]]:
    """Which of ``node``'s parameters each of its static arrays is derived from (MADDENING's ``static_data_deps``)."""
    return cast(dict[str, tuple[str, ...]], _held(node).static_data_deps())


def unreadable_leaves(graph: GraphManager) -> dict[tuple[str, str], str]:
    """Each leaf of a compiled graph's ``params["nodes"]`` that writing the tree alone could not change the step for,
    with why: one a static is derived from, or one no operation of the compiled step takes as an input."""
    return cast(dict[tuple[str, str], str], _held(graph)._param_leaves_the_step_cannot_read())  # noqa: SLF001


def graph_params(graph: GraphManager) -> dict[str, Any]:
    """The graph's whole parameter tree, ``params``, as its compiled step reads it."""
    return cast(dict[str, Any], _held(graph).params)


def set_node_state(graph: GraphManager, name: str, state: Mapping[str, Any]) -> None:
    """Node ``name``'s state fields written, as a run from the start of an overridden initial state needs."""
    _held(graph).set_node_state(name, dict(state))


def full_state(graph: GraphManager) -> dict[str, Any]:
    """The graph's state as its step carries it: every node's fields and MADDENING's own counters (``_meta``)."""
    return cast(dict[str, Any], _held(graph)._state)  # noqa: SLF001


#: The key of MADDENING's own counters in a graph's state: the multi-rate step counter, coupling groups' warm starts.
META = "_meta"


def batched_scan(graph: GraphManager, steps: int, state: Any, params: Any, axes: tuple[tuple[str, str, int | None], ...],
                 tests: Sequence[FlagTest], kept: Sequence[tuple[str, str]], *, batched: bool = True,
                 fed: Fed | None = None) -> tuple[Any, tuple[Array, Array, Array], Any]:
    """``steps`` steps of a batch of members of one compiled graph, as one compiled scan vmapped over the members: each
    member's state (every node field and ``_meta``, stacked along a leading member axis) and those parameter leaves
    ``axes`` marks ``0`` (``(node, leaf, axis)``), stacked likewise; a leaf marked ``None`` is shared by every member and
    enters the step once, never copied per member. Each flag of ``tests`` is reduced per member inside the scan (steps
    raised, first, last), and each field of ``kept`` (``(node, field)``) is kept at every step. The state it returns is
    each member's at the end, ``_meta`` included. ``batched=False`` runs the same scan unvmapped, on one member's state
    and tree: the program a batch is compared with. Nothing is stored on the graph, which is left as it was."""
    held = _held(graph)
    jnp = _held(jax.numpy)
    held._recover_from_escaped_tracers()  # noqa: SLF001
    held._check_static_data_dirty()  # noqa: SLF001
    if held._dirty or held._compiled_step is None:  # noqa: SLF001
        held.compile()
    held._refuse_xla_loop_hazards("run_scan", scan=True)  # noqa: SLF001
    external = held._resolve_external_inputs(dict(fed or {}))  # noqa: SLF001
    count = len(tests)

    def build() -> Callable[..., Any]:
        step_fn = held._build_step_fn()  # noqa: SLF001

        def raised_by(state: Mapping[str, Mapping[str, Array]], test: FlagTest) -> Array:
            values = jnp.reshape(state[test.node][test.field], (-1,))
            if test.limits is None:
                return jnp.any(values != 0)
            limits = jnp.asarray([_threshold(limit, np.dtype(values.dtype)) for limit in test.limits],
                                 dtype=values.dtype)
            return jnp.any(jnp.abs(values) > limits)

        def scan(state: Any, ext: Any, parameters: Any) -> tuple[Any, Any, Any]:
            held._count_scan_trace()  # noqa: SLF001

            def body(carry: Any, _: Any) -> tuple[Any, Any]:
                state, counted, first, last, at = carry
                state = step_fn(state, ext, parameters)
                at = at + 1
                raised = jnp.stack([raised_by(state, test) for test in tests]) if tests else \
                    jnp.zeros((0,), dtype=bool)
                out = {node: {name: state[node][name] for name in sorted({f for n, f in kept if n == node})}
                       for node in sorted({n for n, _ in kept})}
                return (state, counted + raised.astype(counted.dtype), jnp.where(raised & (first < 0), at, first),
                        jnp.where(raised, at, last), at), out

            none = FlagTally.none(count)
            init = (state, jnp.asarray(none.count, jnp.int32), jnp.asarray(none.first, jnp.int32),
                    jnp.asarray(none.last, jnp.int32), jnp.asarray(0, jnp.int32))
            (final, counted, first, last, _), out = _held(jax).lax.scan(body, init, None, length=int(steps))
            return final, (counted, first, last), out

        if not batched:
            return _held(jax).jit(scan)
        marked = {(node, leaf): axis for node, leaf, axis in axes}
        tree = cast(Mapping[str, Mapping[str, Mapping[str, Any]]], params)
        nodes = {node: {leaf: marked.get((node, leaf)) for leaf in leaves} for node, leaves in tree["nodes"].items()}
        mappings = {key: None for key in tree.get("mappings", {})}
        external_axes = _held(jax).tree_util.tree_map(lambda _: None, external)
        return _held(jax).jit(_held(jax).vmap(scan, in_axes=(0, external_axes, {"nodes": nodes,
                                                                                "mappings": mappings})))

    structure = str(_held(jax).tree_util.tree_structure(params))
    key = ("mime_batched_scan", int(steps), batched, axes, structure, tuple(tests), tuple(kept))
    program = held._cached_scan(key, build)  # noqa: SLF001
    return cast(tuple[Any, tuple[Array, Array, Array], Any], program(state, external, params))


def ready(state: NodeFields) -> None:
    """Wait until ``state`` has been computed: a scan returns at once, with arrays still being filled, and a chunk's end
    is where its last array is."""
    _held(jax).block_until_ready(state)


def scan_chunk_with_history(graph: GraphManager, steps: int, fed: Fed,
                            params: Mapping[str, Any] | None = None) -> tuple[NodeFields, NodeFields]:
    if params is None:
        return cast(tuple[NodeFields, NodeFields], _held(graph).run_scan_with_history(steps, fed))
    return cast(tuple[NodeFields, NodeFields], _held(graph).run_scan_with_history(steps, fed, params=params))


#: How MADDENING's scan cache names the history scan of one step (``run_scan_with_history``), after the compile
#: generation it was built for.
_ONE_STEP_HISTORY = ("run_scan_with_history", 1)


def holds_one_step_history(graph: GraphManager) -> bool:
    """Whether ``graph`` holds the history scan of one step built for the step it has compiled now: what a run that
    keeps its steps' state compiled when it was chunked by one step. MADDENING keeps the cache private, keyed by the
    compile generation, which every recompile moves."""
    held = _held(graph)
    return (held._compile_generation, *_ONE_STEP_HISTORY) in held._scan_cache  # noqa: SLF001


def _content(value: object) -> str | None:
    """The SHA-256 of a NumPy array's bytes where it can be written in place (a JAX array cannot), else ``None``."""
    if isinstance(value, np.ndarray) and value.flags.writeable:
        held = cast(NDArray[Any], value)
        return hashlib.sha256(np.ascontiguousarray(held).tobytes()).hexdigest()
    return None


def _same(a: object, b: object) -> bool:
    """One leaf: the same object, or the same immutable scalar."""
    if a is b:
        return True
    return type(a) is type(b) and isinstance(a, (bool, int, float, complex, str, bytes, type(None))) and a == b


@dataclass(frozen=True, eq=False)
class StepReads:
    """What a graph's compiled step was compiled from and reads, as it stood when taken (:func:`step_reads`): the
    compile generation, whether a pending change marks the graph for a recompile (a ``node.params`` write MADDENING
    has not taken in, a static array of another shape), the node objects by name, the parameter tree, its leaves and
    every static array, each by identity, and the bytes of each NumPy array among them, which an edit in place
    changes and its identity does not (a JAX array cannot be edited in place). It does not see an attribute of a node
    written directly, which MADDENING's own step does not see until it is traced again."""

    generation: int
    dirty: bool
    nodes: tuple[tuple[str, object], ...]
    params: object
    structure: object
    leaves: tuple[object, ...]
    statics: tuple[tuple[str, str, object, object, object], ...]
    contents: tuple[str | None, ...]

    def unchanged_since(self, earlier: StepReads) -> bool:
        """Whether nothing this records differs from ``earlier``, and neither marks the graph for a recompile."""
        return (not self.dirty and not earlier.dirty and self.generation == earlier.generation
                and self.params is earlier.params and self.structure == earlier.structure
                and len(self.nodes) == len(earlier.nodes) and len(self.leaves) == len(earlier.leaves)
                and len(self.statics) == len(earlier.statics)
                and all(a[0] == b[0] and a[1] is b[1] for a, b in zip(self.nodes, earlier.nodes, strict=True))
                and all(_same(a, b) for a, b in zip(self.leaves, earlier.leaves, strict=True))
                and all(a[:2] == b[:2] and a[2] is b[2] and a[3:] == b[3:]
                        for a, b in zip(self.statics, earlier.statics, strict=True))
                and self.contents == earlier.contents)


def step_reads(graph: GraphManager) -> StepReads:
    """What ``graph``'s compiled step reads, as it stands (:class:`StepReads`). Reading it takes in a ``node.params``
    write MADDENING has not taken in yet, as the next run would."""
    held = _held(graph)
    dirty = bool(held._check_static_data_dirty()) or bool(held._dirty)  # noqa: SLF001
    params = held.params
    leaves, structure = _held(jax).tree_util.tree_flatten(params)
    statics: list[tuple[str, str, object, object, object]] = []
    for name in graph.node_names:
        for key, value in sorted(static_data(graph.get_node(name)).items()):
            if isinstance(value, StaticArray):
                statics.append((name, key, value.value, value.replication, value.shard_axis))
            else:
                statics.append((name, key, value, None, None))
    contents = tuple(_content(leaf) for leaf in leaves) + tuple(_content(each[2]) for each in statics)
    return StepReads(int(held._compile_generation), dirty,  # noqa: SLF001
                     tuple((name, graph.get_node(name)) for name in graph.node_names), params, structure,
                     tuple(cast(list[object], leaves)), tuple(statics), contents)


#: MADDENING's own record of the state it holds, beside the state itself, which a step or a reset writes:
#: whether the state holds tracers and what to go back to, the step a report describes, and the underflow check's
#: progress and the groups it has warned about.
_STATE_RECORD = ("_state_traced", "_state_before_trace", "_state_as_reported", "_underflow_check_pending",
                 "_underflow_free_checks", "_underflow_stores_since_check", "_underflow_warned")


@contextmanager
def state_put_back(graph: GraphManager) -> Generator[None]:
    """The graph's state, its internal counters (``_meta``) and MADDENING's own record of it, put back as they were when
    the block ends, however it ends: the same objects, holding what they held. What the block compiles stays compiled.
    Every array is shared, never copied: an array cannot change."""
    held = _held(graph)
    attributes = cast(dict[str, Any], vars(held))
    state = cast(dict[str, Any], held._state)  # noqa: SLF001
    fields = dict(state)
    meta = cast(dict[str, Any] | None, state.get("_meta") if isinstance(state.get("_meta"), dict) else None)
    counters = dict(meta) if meta is not None else None
    record = {name: attributes[name] for name in _STATE_RECORD if name in attributes}
    warned = set(cast(set[str], record["_underflow_warned"])) if "_underflow_warned" in record else None
    try:
        yield
    finally:
        if meta is not None and counters is not None:
            meta.clear()
            meta.update(counters)
        state.clear()
        state.update(fields)
        held._state = state  # noqa: SLF001
        attributes.update(record)
        if warned is not None:
            underflow = cast(set[str], attributes["_underflow_warned"])
            underflow.clear()
            underflow.update(warned)


def device_array(value: np.ndarray) -> Array:
    return cast(Array, _held(jax).device_put(value))


def make_jaxpr(fn: Callable[..., object], *args: object) -> Jaxpr:
    return cast(Jaxpr, _held(jax).make_jaxpr(fn)(*args))


def devices() -> list[tuple[str, str]]:
    """JAX's devices, each as its platform and its ``device_kind``."""
    return [(str(device.platform), str(device.device_kind)) for device in _held(jax).devices()]


def compute_capability() -> str | None:
    """The first device's compute capability as JAX reports it (``"8.0"`` for an A100), ``None`` where the device has
    none (a CPU)."""
    held = getattr(_held(jax).devices()[0], "compute_capability", None)
    return str(held) if held is not None else None

def make_jaxpr_with_shapes(fn: Callable[..., object], *args: object) -> tuple[Jaxpr, Any]:
    """``fn`` traced from ``args``: its jaxpr, and the shapes and dtypes of what it returns."""
    return cast(tuple[Jaxpr, Any], _held(jax).make_jaxpr(fn, return_shape=True)(*args))


def step_function(graph: GraphManager) -> Callable[..., NodeFields]:
    """The compiled graph's step as a function, ``(state, external_inputs) -> state``, to trace through: what a
    derivative of a run is taken through, as MADDENING's own example of one does. MADDENING keeps it private."""
    return cast(Callable[..., NodeFields], _held(graph)._build_step_fn())  # noqa: SLF001


def topological_order(order: Sequence[str], edges: Sequence[tuple[str, str]]) -> list[str]:
    """The order MADDENING's compile sorts nodes added in ``order`` into, given the edges between them (source,
    target) and before it gathers coupling groups: every node after the nodes it reads, and the nodes of a loop in
    the order they were added."""
    return topological_sort(list(order), [EdgeSpec(source, target, "", "") for source, target in edges])


def initial_fields(node: SimulationNode) -> dict[str, Array]:
    """``node``'s state at the start, before it is in a graph."""
    return cast(dict[str, Array], _held(node).initial_state())
