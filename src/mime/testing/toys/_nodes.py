"""The toy entries' MADDENING nodes. Each keeps every constructor argument in ``params``, so its saved form is whole;
names its initial conditions ``initial_*``, which MADDENING takes as initial conditions rather than step parameters;
and declares the units of every input, spelt as the vocabulary spells them.

Each node's state and inputs are typed, as ``mime.nodes``'s README sets out: a closed TypedDict per node for each,
taken as a view inside MADDENING's own signatures (``typed_view``) and handed back as the ``dict`` MADDENING takes
(``node_fields``). Fields named at construction (one per registered body or reader slot) are the TypedDict's
``extra_items``; a state of such fields alone is handed back as a plain mapping of arrays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast, override

import jax.numpy
import numpy as np
from jax import Array
from jaxtyping import Float
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.static_data import StaticArray
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.build._maddening import node_constructor_arguments
from mime.core import MimeNode, NodeFields, nearest_point, node_fields, shape_named, typed_view

#: ``jax.numpy``'s hints name a device type from jaxlib's compiled core, which ships no stubs, so a strict checker
#: reads its array constructors as partly unknown; the toys call them through this name.
jnp: Any = jax.numpy

#: The toy field's phase per axis. A fixed table the field node holds as static data, so that the derived hash has a
#: static array's contents to cover: MADDENING hashes static data by shape and dtype only.
PHASE: tuple[float, float, float] = (0.0, 0.25, -0.5)

type Vector = Float[Array, "3"]
type Scalar = Float[Array, ""]  # noqa: F722 - jaxtyping's shape of a scalar, not a forward reference


def _float(value: Any) -> Array:
    return jnp.asarray(value, dtype=jnp.zeros(()).dtype)


def _zero3() -> Array:
    return _float((0.0, 0.0, 0.0))


def _norm(v: Array) -> Array:
    """``|v|``, whose derivative at ``v = 0`` is zero rather than NaN: the square root is taken only of a positive
    number, and its value is discarded at zero."""
    squared = jnp.sum(v * v)
    return jnp.where(squared > 0, jnp.sqrt(jnp.where(squared > 0, squared, 1.0)), 0.0)


def _input(units: str, shape: tuple[int, ...] = (3,), *, additive: bool = False) -> BoundaryInputSpec:
    return BoundaryInputSpec(shape=shape, dtype=jnp.zeros(()).dtype, default=_float(np.zeros(shape)),
                             coupling_type="additive" if additive else "replacive", expected_units=units)


def node_params(node: SimulationNode, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The node's constructor arguments, with the graph's live ``params`` written over them when given."""
    own = node_constructor_arguments(node)
    return dict(own) if params is None else {**own, **params}


class ToyNode(SimulationNode):
    """A toy's node: every constructor argument kept in ``params``, which is all of a node MADDENING saves."""

    def __init__(self, name: str, timestep: float, **params: object) -> None:
        held: Any = super()
        held.__init__(name, timestep, **params)


class ToyBodyState(TypedDict, closed=True):
    x: ReadOnly[Vector]
    v: ReadOnly[Vector]


class ToyBodyInputs(TypedDict, closed=True):
    load: NotRequired[ReadOnly[Vector]]
    hold_z: NotRequired[ReadOnly[Scalar]]


class ToyBodyNode(ToyNode):
    """A point mass: ``v += dt * load / mass``, then ``x += dt * v``. A delivered ``hold_z`` replaces the velocity
    along z."""

    def __init__(self, name: str, timestep: float, mass: float = 1.3,
                 initial_position: Sequence[float] = (0.0, 0.0, 0.0),
                 initial_velocity: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__(name, timestep, mass=mass, initial_position=tuple(initial_position),
                         initial_velocity=tuple(initial_velocity))

    @override
    def initial_state(self) -> NodeFields:
        p = node_params(self)
        return node_fields(ToyBodyState(x=_float(p["initial_position"]), v=_float(p["initial_velocity"])))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"load": _input("N", additive=True), "hold_z": _input("m/s", ())}

    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        p = node_params(self, params)
        held, given = typed_view(ToyBodyState, state), typed_view(ToyBodyInputs, boundary_inputs)
        load = given.get("load", jnp.zeros_like(held["v"]))
        v = held["v"] + dt * load / p["mass"]
        if "hold_z" in given:
            v = v.at[2].set(given["hold_z"])
        return node_fields(ToyBodyState(x=held["x"] + dt * v, v=v))


class ToyAnchoredBodyState(TypedDict, closed=True):
    x: ReadOnly[Vector]


class ToyAnchoredBodyInputs(TypedDict, closed=True):
    load: NotRequired[ReadOnly[Vector]]


class ToyAnchoredBodyNode(ToyNode):
    """A body that stays where it starts, whatever it is delivered."""

    def __init__(self, name: str, timestep: float, initial_position: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__(name, timestep, initial_position=tuple(initial_position))

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(ToyAnchoredBodyState(x=_float(node_params(self)["initial_position"])))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"load": _input("N", additive=True)}

    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        return node_fields(ToyAnchoredBodyState(x=typed_view(ToyAnchoredBodyState, state)["x"]))


class ToyMediumState(TypedDict, extra_items=ReadOnly[Vector]):
    """``u``, and ``force_<body>`` for each registered body."""

    u: ReadOnly[Vector]


class ToyMediumInputs(TypedDict, extra_items=ReadOnly[Vector]):
    """``pos_<body>`` for each registered body."""


class ToyMediumNode(ToyNode):
    """A 3-vector ``u`` relaxing towards ``sin(coupling * pos)`` of its registered bodies (their mean when there are
    several), publishing ``force_<body> = drag_gain * u`` for each."""

    def __init__(self, name: str, timestep: float, bodies: Sequence[str] = (), relax_rate: float = 2.1,
                 coupling: float = 1.9, drag_gain: float = 0.61,
                 initial_value: Sequence[float] = (0.1, 0.2, -0.1)) -> None:
        super().__init__(name, timestep, bodies=tuple(bodies), relax_rate=relax_rate, coupling=coupling,
                         drag_gain=drag_gain, initial_value=tuple(initial_value))

    def _bodies(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], node_params(self)["bodies"])

    def _state(self, u: Array, gain: Any) -> NodeFields:
        return {**node_fields(ToyMediumState(u=u)), **{f"force_{body}": gain * u for body in self._bodies()}}

    @override
    def initial_state(self) -> NodeFields:
        p = node_params(self)
        return self._state(_float(p["initial_value"]), p["drag_gain"])

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {f"pos_{body}": _input("m") for body in self._bodies()}

    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        p = node_params(self, params)
        held, given = typed_view(ToyMediumState, state), typed_view(ToyMediumInputs, boundary_inputs)
        zero = jnp.zeros_like(held["u"])
        drives = [jnp.sin(p["coupling"] * given.get(f"pos_{body}", zero)) for body in self._bodies()]
        drive = drives[0] if len(drives) == 1 else (sum(drives[1:], drives[0]) / len(drives) if drives else zero)
        return self._state(held["u"] + dt * p["relax_rate"] * (drive - held["u"]), p["drag_gain"])


class ToyLoad(TypedDict, closed=True):
    load: ReadOnly[Vector]


class ResponseNode[I: Mapping[str, Array], S: Mapping[str, Array]](ToyNode):
    """A response node: its state is ``law(inputs)``. ``READS`` names its one input, in ``READS_UNITS``, and
    ``OUTPUT`` its one state field; ``I`` and ``S`` type them."""

    READS = "q"
    READS_UNITS = "1"
    OUTPUT = "load"

    @override
    def initial_state(self) -> NodeFields:
        return {name: jnp.zeros_like(value)
                for name, value in node_fields(self.law(node_params(self), cast(I, {}))).items()}

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {self.READS: _input(self.READS_UNITS)}

    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        return node_fields(self.law(node_params(self, params), cast(I, boundary_inputs)))

    def law(self, p: Mapping[str, Any], given: I) -> S:
        """The response to ``given``, an absent input read as zero."""
        raise NotImplementedError


class ToySpringInputs(TypedDict, closed=True):
    x: NotRequired[ReadOnly[Vector]]


class ToySpringNode(ResponseNode[ToySpringInputs, ToyLoad]):
    READS = "x"
    READS_UNITS = "m"

    def __init__(self, name: str, timestep: float, k: float = 1.7,
                 x0: Sequence[float] = (0.05, -0.02, 0.01)) -> None:
        super().__init__(name, timestep, k=k, x0=tuple(x0))

    @override
    def law(self, p: Mapping[str, Any], given: ToySpringInputs) -> ToyLoad:
        return ToyLoad(load=-p["k"] * (given.get("x", _zero3()) - _float(p["x0"])))


class ToyDragInputs(TypedDict, closed=True):
    v: NotRequired[ReadOnly[Vector]]


class ToyDragNode(ResponseNode[ToyDragInputs, ToyLoad]):
    READS = "v"
    READS_UNITS = "m/s"

    def __init__(self, name: str, timestep: float, c: float = 0.37, rung: str = "quadratic") -> None:
        super().__init__(name, timestep, c=c, rung=rung)

    @override
    def law(self, p: Mapping[str, Any], given: ToyDragInputs) -> ToyLoad:
        v = given.get("v", _zero3())
        return ToyLoad(load=-p["c"] * v * _norm(v) if p["rung"] == "quadratic" else -p["c"] * v)


class ToyLiftInputs(TypedDict, closed=True):
    u: NotRequired[ReadOnly[Vector]]


class ToyLiftNode(ResponseNode[ToyLiftInputs, ToyLoad]):
    READS = "u"
    READS_UNITS = "1"

    def __init__(self, name: str, timestep: float, a: float = 0.83, w: Sequence[float] = (1.0, -0.7, 0.45)) -> None:
        super().__init__(name, timestep, a=a, w=tuple(w))

    @override
    def law(self, p: Mapping[str, Any], given: ToyLiftInputs) -> ToyLoad:
        return ToyLoad(load=p["a"] * jnp.tanh(given.get("u", _zero3())) * _float(p["w"]))


class ToyHoldInputs(TypedDict, closed=True):
    push: NotRequired[ReadOnly[Vector]]


class ToyHoldState(TypedDict, closed=True):
    hold_z: ReadOnly[Scalar]


class ToyHoldNode(ResponseNode[ToyHoldInputs, ToyHoldState]):
    READS = "push"
    READS_UNITS = "1"
    OUTPUT = "hold_z"

    def __init__(self, name: str, timestep: float, gain: float = 0.5) -> None:
        super().__init__(name, timestep, gain=gain)

    @override
    def law(self, p: Mapping[str, Any], given: ToyHoldInputs) -> ToyHoldState:
        return ToyHoldState(hold_z=p["gain"] * given.get("push", _zero3())[2])


class ToyFieldInputs(TypedDict, extra_items=ReadOnly[Vector]):
    """``points_<slot>`` for each reader slot."""


class ToyFieldNode(ToyNode):
    """An evaluated field, ``push = amplitude * sin(wavenumber * points + phase)``, with one points input and one
    output per reader slot. ``phase`` is :data:`PHASE`, held as static data."""

    def __init__(self, name: str, timestep: float, slots: Sequence[str] = (), amplitude: float = 0.4,
                 wavenumber: float = 1.3) -> None:
        super().__init__(name, timestep, slots=tuple(slots), amplitude=amplitude, wavenumber=wavenumber)
        self._phase = StaticArray(np.asarray(PHASE, dtype=np.dtype(jnp.zeros(()).dtype)))

    def _slots(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], node_params(self)["slots"])

    @property
    @override
    def static_data(self) -> dict[str, Any]:
        return {"phase": self._phase}

    @override
    def initial_state(self) -> NodeFields:
        return node_fields({f"push_{slot}": _zero3() for slot in self._slots()})

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {f"points_{slot}": _input("m") for slot in self._slots()}

    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        p = node_params(self, params)
        phase = jnp.asarray(self._phase.value)
        given = typed_view(ToyFieldInputs, boundary_inputs)
        return node_fields({f"push_{slot}": p["amplitude"] * jnp.sin(p["wavenumber"] * given.get(f"points_{slot}",
                                                                                                   _zero3()) + phase)
                            for slot in self._slots()})


class ToyRecorderState(TypedDict, closed=True):
    last: ReadOnly[Vector]
    samples: ReadOnly[Scalar]


class ToyRecorderInputs(TypedDict, closed=True):
    position: NotRequired[ReadOnly[Vector]]


class ToyRecorderNode(ToyNode):
    """Keeps the last position it read and how many times it fired."""

    def __init__(self, name: str, timestep: float) -> None:
        super().__init__(name, timestep)

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(ToyRecorderState(last=_zero3(), samples=_float(0.0)))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"position": _input("m")}

    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        held, given = typed_view(ToyRecorderState, state), typed_view(ToyRecorderInputs, boundary_inputs)
        return node_fields(ToyRecorderState(last=given.get("position", held["last"]), samples=held["samples"] + 1.0))


class ToyRegionProbeState(TypedDict, closed=True):
    distance: ReadOnly[Scalar]
    normal: ReadOnly[Vector]
    nearest: ReadOnly[Vector]


class ToyRegionProbeInputs(TypedDict, closed=True):
    position: NotRequired[ReadOnly[Vector]]


class ToyRegionProbeNode(MimeNode):
    """A region's shape at a position, as a node evaluates a shape: by its registered name, with its parameters (each
    ``region_<name>``) as constants. It keeps the signed distance, the outward normal and the nearest point."""

    def __init__(self, name: str, timestep: float, *, shape: str, **region: object) -> None:
        shape_named(shape)
        super().__init__(name, timestep, shape=shape, **{f"region_{key}": value for key, value in region.items()})

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(ToyRegionProbeState(distance=_float(0.0), normal=_zero3(), nearest=_zero3()))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"position": _input("m")}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        own = self.constants()
        shape = shape_named(own["shape"])
        region = {key: p.get(f"region_{key}", own[f"region_{key}"]) for key in shape.parameter_names}
        position = typed_view(ToyRegionProbeInputs, boundary_inputs).get("position", _zero3())
        return node_fields(ToyRegionProbeState(distance=shape.signed_distance(region, position),
                                               normal=shape.outward_normal(region, position),
                                               nearest=nearest_point(shape, region, position)))
