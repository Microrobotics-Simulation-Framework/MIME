"""A dynamics' step identity with the energy ledger attached (F0-1 M6.2), on drawn states: the family-suite case of a
dynamics, as the energy case (``CaseKind.ENERGY``, ``mime.testing.conformance``) is a law's.

:func:`step_identity`: the receiver's node, metered for one synthetic law on each input its port kind pairs (a load
read at the step's start, an operator solved for in the step), books ``H' - H = sum W - N`` within 16 eps of the step's
scale on drawn states, and **work fed into its unattributed input is booked as unattributed** (a draw is fed into the
input that sums what no law owns, the unattributed work comes out above zero, and the identity closes with it): a node
that books such work under a law, or loses it, fails the case on every draw. Drawn per port kind
(:data:`PORT_DRAWS`): a rigid body's at a millimetre swimmer's scales (:func:`rigid_body_states`); a port kind with no
draws yet fails the case, naming it. A dynamics that names no port kind passes with its reason for being unmetered, and fails without
one."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
from hypothesis import strategies as st
from maddening.core.node import SimulationNode
from maddening.testing.verification import VerificationResult

from mime.core import jnp, ledger
from mime.registry import Dynamics
from mime.vocabulary import port_kind

#: The rounding floor the case judges against, in units of eps on the step's scale: the ledger's.
ULPS = ledger.IDENTITY_ULPS


def _failed(detail: str) -> VerificationResult:
    return VerificationResult("step identity", "FAIL", detail=detail)


#: The rig's body, as a dynamics' step identity is checked on it.
_RIG_BODY = "probe"
#: A millimetre swimmer's scales (FL-9's): its mass, and the size of its principal moments.
SWIMMER_MASS, SWIMMER_MOMENT = 8.491116020725203e-05, 1e-10


def _unit(draw: Callable[..., Any], size: int) -> np.ndarray:
    """A unit vector of ``size`` components, each 0.1 to 1 in size with its own sign, normalised: none underflows and
    none sits on an axis plane."""
    magnitudes = draw(st.lists(st.floats(0.1, 1.0), min_size=size, max_size=size))
    signs = draw(st.lists(st.sampled_from((-1.0, 1.0)), min_size=size, max_size=size))
    held = np.asarray(magnitudes, dtype=np.float64) * np.asarray(signs)
    return held / np.linalg.norm(held)


def _vector(draw: Callable[..., Any], least: float, most: float) -> np.ndarray:
    return draw(st.floats(least, most)) * _unit(draw, 3)


def _rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@st.composite
def rigid_body_states(draw: Callable[..., Any]) -> dict[str, np.ndarray]:
    """A rigid body's state and what its laws deliver, at a millimetre swimmer's scales: its mass to a factor of two
    of FL-9's, principal moments 0.5 to 3 e-10 kg m^2 within the triangle inequality about drawn axes, its position to
    a centimetre, speeds to a centimetre a second, rates to 100 rad/s, a load (and a second, fed to the unattributed
    input) to 1e-5 N and 1e-8 N m, and a symmetric positive definite resistance of FL-9's in a tube (4e-5 N s/m, 8e-10 N m s, coupled up to 0.3 of their geometric
    mean) turned by a drawn rotation; each component drawn 0.1 to 1 of its scale with its own sign."""
    least = draw(st.floats(0.5, 1.5)) * SWIMMER_MOMENT
    middle = least * (1.0 + draw(st.floats(0.0, 0.6)))
    largest = min(middle * (1.0 + draw(st.floats(0.0, 0.6))), 0.975 * (least + middle))
    axes = _rotation(_unit(draw, 4))
    tensor = axes @ np.diag([least, middle, largest]) @ axes.T
    translation = np.diag(4e-5 * np.asarray(draw(st.lists(st.floats(1.0, 2.0), min_size=3, max_size=3))))
    turning = np.diag(8e-10 * np.asarray(draw(st.lists(st.floats(1.0, 3.0), min_size=3, max_size=3))))
    entries = np.asarray(draw(st.lists(st.floats(-1.0, 1.0), min_size=9, max_size=9)), dtype=np.float64)
    coupling = 0.3 * math.sqrt(4e-5 * 8e-10) * entries.reshape(3, 3) / 3.0
    turn = np.zeros((6, 6))
    turn[:3, :3] = turn[3:, 3:] = _rotation(_unit(draw, 4))
    resistance = turn @ np.block([[translation, coupling], [coupling.T, turning]]) @ turn.T
    return {"mass": np.asarray(SWIMMER_MASS * draw(st.floats(0.5, 2.0))), "tensor": tensor,
            "pose": np.concatenate([_vector(draw, 1e-5, 1e-2), _unit(draw, 4)]),
            "twist": np.concatenate([_vector(draw, 1e-4, 1e-2), _vector(draw, 0.1, 100.0)]),
            "load": np.concatenate([_vector(draw, 1e-8, 1e-5), _vector(draw, 1e-12, 1e-8)]),
            "unattributed": np.concatenate([_vector(draw, 1e-8, 1e-5), _vector(draw, 1e-12, 1e-8)]),
            "resistance": resistance}


#: The draws for each port kind a dynamics' step identity is checked on, by its name.
PORT_DRAWS: dict[str, Callable[[], Any]] = {"rigid_body": rigid_body_states}


def metered_twin(node: SimulationNode, dynamics: Dynamics
                 ) -> tuple[SimulationNode, tuple[ledger.LedgerDelivery, ...]] | str:
    """``node`` (a receiver as a build makes it, of ``dynamics``) metered for one synthetic law on each input its port
    kind pairs (a load read at the step's start, an operator solved for), or why it cannot be: its dynamics names no
    registered port kind, or its node writes no step identity. A confinement is left to the law that holds it."""
    held = cast(Any, node)
    if getattr(type(node), "LEDGER_TERMS", None) is None:
        return f"{type(node).__name__} writes no step identity (MimeNode.LEDGER_TERMS)"
    kind = port_kind(dynamics.port)
    if kind is None:
        return f"{type(dynamics).label()} names the port kind {dynamics.port!r}, which is not registered"
    deliveries: list[ledger.LedgerDelivery] = []
    for interface, port in sorted(dynamics.accepts.items(), key=lambda item: item[1].field):
        paired = kind.deliveries.get(interface)
        if paired is None or paired.value not in (ledger.LOAD, ledger.SOLVED) or port.field == "confinement":
            continue
        energy_class = ledger.UNMETERED if paired.value == ledger.LOAD else ledger.DISSIPATIVE
        deliveries.append(ledger.LedgerDelivery(port.field, f"case:{port.field}", energy_class, paired.value))
    params = {key: value for key, value in dict(held.params).items() if key != ledger.LEDGER}
    twin = type(node)(node.name, float(held.delta_t), **params, ledger=tuple(d.encoded() for d in deliveries))
    return twin, tuple(deliveries)


def step_identity(dynamics: Dynamics, node: SimulationNode | None = None, *,
                  max_examples: int = 200) -> VerificationResult:
    """``dynamics``' step identity on drawn states (the module's docstring), on ``node`` (its node as a build makes
    it; by default as :func:`dynamics_rig` builds it): one step of its metered twin from each draw of its port kind's
    states, the identity within its floor and work fed into its unattributed input booked as unattributed. A dynamics
    that names no port kind passes with its reason for being unmetered, and fails with none. A failing draw is
    reported as drawn."""
    from hypothesis import HealthCheck, Phase, given, settings

    if not dynamics.port:
        if dynamics.unmetered:
            return VerificationResult("step identity", "PASS", detail=f"unmetered: {dynamics.unmetered} (a pass by that "
                                                                     f"declaration alone: nothing was checked)")
        return _failed(f"{type(dynamics).label()} names no port kind (Dynamics.port) and states no reason for "
                       f"being unmetered (Dynamics.unmetered)")
    draws = PORT_DRAWS.get(dynamics.port)
    if draws is None:
        return _failed(f"no draws for the port kind {dynamics.port!r} yet")
    built = dynamics_rig(dynamics) if node is None else node
    if isinstance(built, str):
        return _failed(built)
    found = metered_twin(built, dynamics)
    if isinstance(found, str):
        return _failed(found)
    twin, deliveries = found
    held = cast(Any, twin)
    dtype = np.dtype(np.asarray(next(iter(held.initial_state().values()))).dtype)
    floor = ULPS * float(np.finfo(dtype).eps)
    last: dict[str, Any] = {}
    count = 0
    # The input that sums what no law owns, which the case feeds a load of its own: work that must come out booked as
    # unattributed.
    unattributed_input = next((d.field for d in deliveries if d.kind == ledger.LOAD), "")
    if not unattributed_input:
        return _failed(f"{type(dynamics).label()}'s port kind pairs no load, so there is no input to feed unattributed "
                       f"work")

    def step(drawn: Mapping[str, Any]) -> Mapping[str, Any]:
        p = {key: jnp.asarray(value, dtype=dtype) for key, value in held._step_constants(None).items()}
        inertia = np.asarray(drawn["tensor"]) if np.size(p["inertia"]) == 9 else np.diag(drawn["tensor"])
        p = {**p, "mass": jnp.asarray(drawn["mass"], dtype=dtype),
             "inertia": jnp.asarray(np.reshape(inertia, (-1,)), dtype=dtype)}
        state = {**{key: jnp.asarray(value, dtype=dtype) for key, value in held.initial_state().items()},
                 "pose": jnp.asarray(drawn["pose"], dtype=dtype), "twist": jnp.asarray(drawn["twist"], dtype=dtype)}
        state = ledger.with_records(held, {key: value for key, value in state.items()
                                           if not key.startswith(ledger.RECORDS)})
        inputs = {name: jnp.zeros(tuple(entry.shape), dtype=dtype) for name, entry in held.boundary_input_spec().items()
                  if ledger.own_input(name) is None}
        inputs[unattributed_input] = jnp.asarray(drawn["unattributed"], dtype=dtype)
        for index, delivery in enumerate(deliveries):
            value = drawn["resistance"] if delivery.kind == ledger.SOLVED else drawn["load"]
            inputs[ledger.law_input(delivery.field, index)] = jnp.asarray(value, dtype=dtype)
        return cast(Mapping[str, Any], held.advance(p, state, inputs, float(held.delta_t)))

    @settings(max_examples=max_examples, deadline=None, database=None, phases=(Phase.explicit, Phase.generate),
              suppress_health_check=[HealthCheck.too_slow])
    @given(draws())
    def prop(drawn: Mapping[str, Any]) -> None:
        nonlocal count
        count += 1
        last.clear()
        last.update(drawn)
        stepped = step(drawn)
        over = float(stepped[f"{ledger.RECORDS}step_identity"]) / floor
        assert over <= 1.0, f"the step identity's residual is {over:.3g} of its floor"
        booked = float(stepped[f"{ledger.RECORDS}unattributed_worst"])
        assert booked > 0.0, (f"work fed into the unattributed input ({unattributed_input}) was not booked as "
                              f"unattributed work: it was booked under a law or lost")

    try:
        prop()
    except AssertionError as error:
        return VerificationResult("step identity", "FAIL", detail=str(error), counterexample=dict(last),
                                  n_examples=count)
    return VerificationResult("step identity", "PASS", n_examples=count)


def dynamics_rig(dynamics: Dynamics) -> SimulationNode | str:
    """The node of ``dynamics`` as a build makes it for the smallest experiment holding it: one body of a millimetre
    swimmer's mass and moments, at rest, with nothing acting on it; or why it cannot be built so (a dynamics of
    something other than a body: its rig comes with its port kind's draws)."""
    from mime.experiment import Experiment
    from mime.scene import Scene, Z
    from mime.things import Body, State, ThingKind
    from mime.units import kg, kg_m2, m, m_per_s, ms, rad_per_s, standard_gravity

    if ThingKind.BODY not in type(dynamics).applies_to:
        return f"{type(dynamics).label()} is not a body's dynamics, and a rig for it comes with its port kind's draws"
    experiment = Experiment("dynamics_rig", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=ms(0.5))
    experiment.add_body(Body(_RIG_BODY, dynamics=dynamics, initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare(f"{_RIG_BODY}.mass", kg(SWIMMER_MASS), reason="the rig's body")
    experiment.declare(f"{_RIG_BODY}.inertia", kg_m2((SWIMMER_MOMENT,) * 3), reason="the rig's body")
    built = experiment.build()
    names = [name for plan in built.plans.entries if plan.thing == _RIG_BODY
             for name in plan.node_names()]
    if len(names) != 1:
        return f"{type(dynamics).label()} builds {len(names)} nodes for one body, and the case steps one"
    return built.graph.get_node(names[0])


__all__ = ["PORT_DRAWS", "dynamics_rig", "metered_twin", "rigid_body_states", "step_identity"]
