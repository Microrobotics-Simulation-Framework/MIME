"""The toy entries' MADDENING nodes. Each keeps every constructor argument in ``params``, so its saved form is whole;
names its initial conditions ``initial_*``, which MADDENING takes as initial conditions rather than step parameters;
and declares the units of every input, spelt as the vocabulary spells them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast, override

import jax.numpy
import numpy as np
from jax import Array
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.static_data import StaticArray

from mime.build._maddening import node_constructor_arguments

type NodeState = dict[str, Array]

#: ``jax.numpy``'s hints name a device type from jaxlib's compiled core, which ships no stubs, so a strict checker
#: reads its array constructors as partly unknown; the toys call them through this name.
jnp: Any = jax.numpy

#: The toy field's phase per axis. A fixed table the field node holds as static data, so that the derived hash has a
#: static array's contents to cover: MADDENING hashes static data by shape and dtype only.
PHASE: tuple[float, float, float] = (0.0, 0.25, -0.5)


def _float(value: Any) -> Array:
    return jnp.asarray(value, dtype=jnp.zeros(()).dtype)


def _zero3() -> Array:
    return _float((0.0, 0.0, 0.0))


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


class ToyBodyNode(ToyNode):
    """A point mass: ``v += dt * load / mass``, then ``x += dt * v``. A delivered ``hold_z`` replaces the velocity
    along z."""

    def __init__(self, name: str, timestep: float, mass: float = 1.3,
                 initial_position: Sequence[float] = (0.0, 0.0, 0.0),
                 initial_velocity: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__(name, timestep, mass=mass, initial_position=tuple(initial_position),
                         initial_velocity=tuple(initial_velocity))

    @override
    def initial_state(self) -> NodeState:
        p = node_params(self)
        return {"x": _float(p["initial_position"]), "v": _float(p["initial_velocity"])}

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"load": _input("N", additive=True), "hold_z": _input("m/s", ())}

    @override
    def update(self, state: NodeState, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeState:
        p = node_params(self, params)
        load = boundary_inputs.get("load", jnp.zeros_like(state["v"]))
        v = state["v"] + dt * load / p["mass"]
        if "hold_z" in boundary_inputs:
            v = v.at[2].set(boundary_inputs["hold_z"])
        return {"x": state["x"] + dt * v, "v": v}


class ToyAnchoredBodyNode(ToyNode):
    """A body that stays where it starts, whatever it is delivered."""

    def __init__(self, name: str, timestep: float, initial_position: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__(name, timestep, initial_position=tuple(initial_position))

    @override
    def initial_state(self) -> NodeState:
        return {"x": _float(node_params(self)["initial_position"])}

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"load": _input("N", additive=True)}

    @override
    def update(self, state: NodeState, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeState:
        return {"x": state["x"]}


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

    def _forces(self, u: Array, gain: Any) -> NodeState:
        return {f"force_{body}": gain * u for body in self._bodies()}

    @override
    def initial_state(self) -> NodeState:
        p = node_params(self)
        u = _float(p["initial_value"])
        return {"u": u, **self._forces(u, p["drag_gain"])}

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {f"pos_{body}": _input("m") for body in self._bodies()}

    @override
    def update(self, state: NodeState, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeState:
        p = node_params(self, params)
        zero = jnp.zeros_like(state["u"])
        drives = [jnp.sin(p["coupling"] * boundary_inputs.get(f"pos_{body}", zero)) for body in self._bodies()]
        drive = drives[0] if len(drives) == 1 else (sum(drives[1:], drives[0]) / len(drives) if drives else zero)
        u = state["u"] + dt * p["relax_rate"] * (drive - state["u"])
        return {"u": u, **self._forces(u, p["drag_gain"])}


class ResponseNode(ToyNode):
    """A response node: ``output = law(reading)``, held as state. ``READS`` is its input, in ``READS_UNITS``."""

    READS = "q"
    READS_UNITS = "1"
    OUTPUT = "load"

    @override
    def initial_state(self) -> NodeState:
        return {self.OUTPUT: jnp.zeros_like(self.law(node_params(self), _zero3()))}

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {self.READS: _input(self.READS_UNITS)}

    @override
    def update(self, state: NodeState, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeState:
        reading = boundary_inputs.get(self.READS, _zero3())
        return {self.OUTPUT: self.law(node_params(self, params), reading)}

    def law(self, p: Mapping[str, Any], reading: Array) -> Array:
        raise NotImplementedError


class ToySpringNode(ResponseNode):
    READS = "x"
    READS_UNITS = "m"

    def __init__(self, name: str, timestep: float, k: float = 1.7,
                 x0: Sequence[float] = (0.05, -0.02, 0.01)) -> None:
        super().__init__(name, timestep, k=k, x0=tuple(x0))

    @override
    def law(self, p: Mapping[str, Any], reading: Array) -> Array:
        return -p["k"] * (reading - _float(p["x0"]))


class ToyDragNode(ResponseNode):
    READS = "v"
    READS_UNITS = "m/s"

    def __init__(self, name: str, timestep: float, c: float = 0.37, rung: str = "quadratic") -> None:
        super().__init__(name, timestep, c=c, rung=rung)

    @override
    def law(self, p: Mapping[str, Any], reading: Array) -> Array:
        return -p["c"] * reading * jnp.abs(reading) if p["rung"] == "quadratic" else -p["c"] * reading


class ToyLiftNode(ResponseNode):
    READS = "u"
    READS_UNITS = "1"

    def __init__(self, name: str, timestep: float, a: float = 0.83, w: Sequence[float] = (1.0, -0.7, 0.45)) -> None:
        super().__init__(name, timestep, a=a, w=tuple(w))

    @override
    def law(self, p: Mapping[str, Any], reading: Array) -> Array:
        return p["a"] * jnp.tanh(reading) * _float(p["w"])


class ToyHoldNode(ResponseNode):
    READS = "push"
    READS_UNITS = "1"
    OUTPUT = "hold_z"

    def __init__(self, name: str, timestep: float, gain: float = 0.5) -> None:
        super().__init__(name, timestep, gain=gain)

    @override
    def law(self, p: Mapping[str, Any], reading: Array) -> Array:
        return p["gain"] * reading[2]


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
    def initial_state(self) -> NodeState:
        return {f"push_{slot}": _zero3() for slot in self._slots()}

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {f"points_{slot}": _input("m") for slot in self._slots()}

    @override
    def update(self, state: NodeState, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeState:
        p = node_params(self, params)
        phase = jnp.asarray(self._phase.value)
        return {f"push_{slot}": p["amplitude"] * jnp.sin(p["wavenumber"] * boundary_inputs.get(f"points_{slot}",
                                                                                              _zero3()) + phase)
                for slot in self._slots()}


class ToyRecorderNode(ToyNode):
    """Keeps the last position it read and how many times it fired."""

    def __init__(self, name: str, timestep: float) -> None:
        super().__init__(name, timestep)

    @override
    def initial_state(self) -> NodeState:
        return {"last": _zero3(), "samples": _float(0.0)}

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"position": _input("m")}

    @override
    def update(self, state: NodeState, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeState:
        return {"last": boundary_inputs.get("position", state["last"]), "samples": state["samples"] + 1.0}
