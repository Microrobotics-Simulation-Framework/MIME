"""The coupled toy assembled by hand with plain MADDENING calls, from node classes written apart from
``mime.testing.toys``: the reference the build must reproduce bit for bit.

A relaxing medium ``u`` driven by ``sin`` of the body's position publishes ``force = drag_gain * u``, which loads the
body; three laws (a spring on the position, a quadratic drag on the velocity, a lift from the medium's value) load
the body too. There is no coupling group, so every loop is staggered and the order of ``add_node`` and ``add_edge``
calls is part of the model.
"""

import jax.numpy as jnp
import numpy as np
from maddening import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

F64 = jnp.float64
DT = 0.02


def _p(node, params):
    return node.params if params is None else {**node.params, **params}


class Medium(SimulationNode):
    def __init__(self, name, timestep, relax_rate=2.1, coupling=1.9, drag_gain=0.61):
        super().__init__(name, timestep, relax_rate=relax_rate, coupling=coupling, drag_gain=drag_gain)

    def initial_state(self):
        u = jnp.asarray([0.1, 0.2, -0.1], F64)
        return {"u": u, "force": self.params["drag_gain"] * u}

    def boundary_input_spec(self):
        return {"pos": BoundaryInputSpec(shape=(3,), dtype=F64, default=jnp.zeros(3, F64))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = _p(self, params)
        pos = boundary_inputs.get("pos", jnp.zeros(3, F64))
        u = state["u"] + dt * p["relax_rate"] * (jnp.sin(p["coupling"] * pos) - state["u"])
        return {"u": u, "force": p["drag_gain"] * u}


class Body(SimulationNode):
    def __init__(self, name, timestep, mass=1.3):
        super().__init__(name, timestep, mass=mass)

    def initial_state(self):
        return {"x": jnp.asarray([0.3, -0.2, 0.1], F64), "v": jnp.asarray([0.5, 0.1, -0.3], F64)}

    def boundary_input_spec(self):
        return {"load": BoundaryInputSpec(shape=(3,), dtype=F64, default=jnp.zeros(3, F64), coupling_type="additive")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = _p(self, params)
        load = boundary_inputs.get("load", jnp.zeros(3, F64))
        v = state["v"] + dt * load / p["mass"]
        return {"x": state["x"] + dt * v, "v": v}


class _Law(SimulationNode):
    READS = "q"

    def initial_state(self):
        return {"load": jnp.zeros(3, F64)}

    def boundary_input_spec(self):
        return {self.READS: BoundaryInputSpec(shape=(3,), dtype=F64, default=jnp.zeros(3, F64))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"load": self.law(_p(self, params), boundary_inputs.get(self.READS, jnp.zeros(3, F64)))}


class Spring(_Law):
    READS = "x"

    def __init__(self, name, timestep, k=1.7, x0=(0.05, -0.02, 0.01)):
        super().__init__(name, timestep, k=k, x0=x0)

    def law(self, p, x):
        return -p["k"] * (x - jnp.asarray(p["x0"], F64))


class Drag(_Law):
    READS = "v"

    def __init__(self, name, timestep, c=0.37):
        super().__init__(name, timestep, c=c)

    def law(self, p, v):
        return -p["c"] * v * jnp.abs(v)


class Lift(_Law):
    READS = "u"

    def __init__(self, name, timestep, a=0.83, w=(1.0, -0.7, 0.45)):
        super().__init__(name, timestep, a=a, w=w)

    def law(self, p, u):
        return p["a"] * jnp.tanh(u) * jnp.asarray(p["w"], F64)


# The build's node names, so that the histories line up field by field.
MEDIUM, BODY = "bath.flow", "probe"
LAWS = {"toy.drag:probe": (Drag, "v", BODY), "toy.lift:probe": (Lift, "u", MEDIUM),
        "toy.spring:probe": (Spring, "x", BODY)}
COUPLE = "toy.couple:probe"

ORDER_A = (*sorted(LAWS), MEDIUM, BODY)
ORDER_B = (BODY, MEDIUM, "toy.spring:probe", "toy.lift:probe", "toy.drag:probe")
LOADS_A = (COUPLE, *sorted(LAWS))
LOADS_B = tuple(reversed(LOADS_A))


def _node(name):
    if name == MEDIUM:
        return Medium(name, DT)
    if name == BODY:
        return Body(name, DT)
    return LAWS[name][0](name, DT)


def assemble(nodes=ORDER_A, loads=LOADS_A) -> GraphManager:
    """The reads, then the loads into the body in the order ``loads`` names their deliverers."""
    graph = GraphManager()
    for name in nodes:
        graph.add_node(_node(name))
    for law, (_, field, source) in sorted(LAWS.items()):
        graph.add_edge(source, law, field, field)
    graph.add_edge(BODY, MEDIUM, "x", "pos")
    for by in loads:
        if by == COUPLE:
            graph.add_edge(MEDIUM, BODY, "force", "load", additive=True)
        else:
            graph.add_edge(by, BODY, "load", "load", additive=True)
    return graph


def history(graph: GraphManager, steps: int) -> dict:
    """The run's history, with the medium's ``force`` under the build's per-body name ``force_probe``."""
    graph.compile()
    _, stacked = graph.run_scan_with_history(steps)
    out = {node: {field: np.asarray(value) for field, value in fields.items()} for node, fields in stacked.items()}
    out[MEDIUM]["force_probe"] = out[MEDIUM].pop("force")
    return out
