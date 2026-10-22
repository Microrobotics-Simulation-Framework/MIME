"""The toy entries' MADDENING nodes. Each keeps every constructor argument in ``params``, so its saved form is whole."""

from __future__ import annotations

import jax.numpy as jnp
from maddening.core.node import BoundaryInputSpec, SimulationNode


def _float(value) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.zeros(()).dtype)


def _vector_input() -> BoundaryInputSpec:
    return BoundaryInputSpec(shape=(3,), dtype=jnp.zeros(()).dtype, default=_float((0.0, 0.0, 0.0)))


def _params(node: SimulationNode, params: dict | None) -> dict:
    return node.params if params is None else {**node.params, **params}


class ToyBodyNode(SimulationNode):
    """A point mass: ``v += dt * load / mass``, then ``x += dt * v``. A delivered ``hold_z`` replaces the velocity
    along z."""

    def __init__(self, name, timestep, mass=1.3, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)):
        super().__init__(name, timestep, mass=mass, position=position, velocity=velocity)

    def initial_state(self):
        return {"x": _float(self.params["position"]), "v": _float(self.params["velocity"])}

    def boundary_input_spec(self):
        return {"load": BoundaryInputSpec(shape=(3,), dtype=jnp.zeros(()).dtype, default=_float((0.0, 0.0, 0.0)),
                                          coupling_type="additive"),
                "hold_z": BoundaryInputSpec(shape=(), dtype=jnp.zeros(()).dtype, default=_float(0.0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = _params(self, params)
        load = boundary_inputs.get("load", jnp.zeros_like(state["v"]))
        v = state["v"] + dt * load / p["mass"]
        if "hold_z" in boundary_inputs:
            v = v.at[2].set(boundary_inputs["hold_z"])
        return {"x": state["x"] + dt * v, "v": v}


class ToyAnchoredBodyNode(SimulationNode):
    """A body that stays where it starts, whatever it is delivered."""

    def __init__(self, name, timestep, position=(0.0, 0.0, 0.0)):
        super().__init__(name, timestep, position=position)

    def initial_state(self):
        return {"x": _float(self.params["position"])}

    def boundary_input_spec(self):
        return {"load": _vector_input()}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": state["x"]}


class ToyMediumNode(SimulationNode):
    """A 3-vector ``u`` relaxing towards ``sin(coupling * pos)`` of its registered bodies (their mean when there are
    several), publishing ``force_<body> = drag_gain * u`` for each."""

    def __init__(self, name, timestep, bodies=(), relax_rate=2.1, coupling=1.9, drag_gain=0.61,
                 value=(0.1, 0.2, -0.1)):
        super().__init__(name, timestep, bodies=tuple(bodies), relax_rate=relax_rate, coupling=coupling,
                         drag_gain=drag_gain, value=value)

    def _forces(self, u, gain):
        return {f"force_{body}": gain * u for body in self.params["bodies"]}

    def initial_state(self):
        u = _float(self.params["value"])
        return {"u": u, **self._forces(u, self.params["drag_gain"])}

    def boundary_input_spec(self):
        return {f"pos_{body}": _vector_input() for body in self.params["bodies"]}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = _params(self, params)
        zero = jnp.zeros_like(state["u"])
        drives = [jnp.sin(p["coupling"] * boundary_inputs.get(f"pos_{body}", zero)) for body in p["bodies"]]
        drive = drives[0] if len(drives) == 1 else (sum(drives[1:], drives[0]) / len(drives) if drives else zero)
        u = state["u"] + dt * p["relax_rate"] * (drive - state["u"])
        return {"u": u, **self._forces(u, p["drag_gain"])}


class _ResponseNode(SimulationNode):
    """A response node: ``output = law(reading)``, held as state."""

    READS = "q"
    OUTPUT = "load"

    def initial_state(self):
        return {self.OUTPUT: jnp.zeros_like(self._law(self.params, _float((0.0, 0.0, 0.0))))}

    def boundary_input_spec(self):
        return {self.READS: _vector_input()}

    def update(self, state, boundary_inputs, dt, *, params=None):
        reading = boundary_inputs.get(self.READS, _float((0.0, 0.0, 0.0)))
        return {self.OUTPUT: self._law(_params(self, params), reading)}

    def _law(self, p, reading):
        raise NotImplementedError


class ToySpringNode(_ResponseNode):
    READS = "x"

    def __init__(self, name, timestep, k=1.7, x0=(0.05, -0.02, 0.01)):
        super().__init__(name, timestep, k=k, x0=x0)

    def _law(self, p, x):
        return -p["k"] * (x - _float(p["x0"]))


class ToyDragNode(_ResponseNode):
    READS = "v"

    def __init__(self, name, timestep, c=0.37, rung="quadratic"):
        super().__init__(name, timestep, c=c, rung=rung)

    def _law(self, p, v):
        return -p["c"] * v * jnp.abs(v) if p["rung"] == "quadratic" else -p["c"] * v


class ToyLiftNode(_ResponseNode):
    READS = "u"

    def __init__(self, name, timestep, a=0.83, w=(1.0, -0.7, 0.45)):
        super().__init__(name, timestep, a=a, w=w)

    def _law(self, p, u):
        return p["a"] * jnp.tanh(u) * _float(p["w"])


class ToyHoldNode(_ResponseNode):
    READS = "push"
    OUTPUT = "hold_z"

    def __init__(self, name, timestep, gain=0.5):
        super().__init__(name, timestep, gain=gain)

    def _law(self, p, push):
        return p["gain"] * push[2]


class ToyFieldNode(SimulationNode):
    """An evaluated field, ``push = amplitude * sin(wavenumber * points)``, with one points input and one output per
    reader slot."""

    def __init__(self, name, timestep, slots=(), amplitude=0.4, wavenumber=1.3):
        super().__init__(name, timestep, slots=tuple(slots), amplitude=amplitude, wavenumber=wavenumber)

    def initial_state(self):
        return {f"push_{slot}": _float((0.0, 0.0, 0.0)) for slot in self.params["slots"]}

    def boundary_input_spec(self):
        return {f"points_{slot}": _vector_input() for slot in self.params["slots"]}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = _params(self, params)
        zero = _float((0.0, 0.0, 0.0))
        return {f"push_{slot}": p["amplitude"] * jnp.sin(p["wavenumber"] * boundary_inputs.get(f"points_{slot}", zero))
                for slot in p["slots"]}


class ToyRecorderNode(SimulationNode):
    """Keeps the last position it read and how many times it fired."""

    def __init__(self, name, timestep):
        super().__init__(name, timestep)

    def initial_state(self):
        return {"last": _float((0.0, 0.0, 0.0)), "samples": _float(0.0)}

    def boundary_input_spec(self):
        return {"position": _vector_input()}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"last": boundary_inputs.get("position", state["last"]), "samples": state["samples"] + 1.0}
