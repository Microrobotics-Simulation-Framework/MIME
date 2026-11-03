"""MADDENING 0.4.x and JAX at MIME's boundary. Some of their type hints leave a container or a callable bare
(``dict``, ``Callable``), which a strict checker reads as unknown; each call MIME makes through one of those is made
here, once, and given the type MIME relies on. Nothing here changes what is called."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
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


def scan_chunk(graph: GraphManager, steps: int, fed: Fed) -> NodeFields:
    """``steps`` steps as one compiled scan, ``fed`` held through them: a chunk of a run whose inputs
    :func:`fed_inputs` has checked once, at its start."""
    return cast(NodeFields, _held(graph).run_scan(steps, fed))


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


def scan_chunk_tallied(graph: GraphManager, steps: int, fed: Fed,
                       tests: Sequence[FlagTest]) -> tuple[NodeFields, FlagTally]:
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
    params = held._params_or_default(None)  # noqa: SLF001
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


def ready(state: NodeFields) -> None:
    """Wait until ``state`` has been computed: a scan returns at once, with arrays still being filled, and a chunk's end
    is where its last array is."""
    _held(jax).block_until_ready(state)


def scan_chunk_with_history(graph: GraphManager, steps: int, fed: Fed) -> tuple[NodeFields, NodeFields]:
    return cast(tuple[NodeFields, NodeFields], _held(graph).run_scan_with_history(steps, fed))


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
