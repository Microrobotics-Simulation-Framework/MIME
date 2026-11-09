"""A dynamics' step identity with the energy ledger attached (F0-1 M6.2), on drawn states: the family-suite case of a
dynamics, as the energy case (``CaseKind.ENERGY``, ``mime.testing.conformance``) is a law's.

:func:`step_identity`: the receiver's node, metered for one synthetic law on each input its port kind pairs (a load
read at the step's start, an operator solved for in the step), books ``H' - H = sum W - N`` within 16 eps of the step's
scale on drawn states, and **work fed into its unattributed input is booked as unattributed** (a draw is fed into the
input that sums what no law owns, the unattributed work comes out above zero, and the identity closes with it): a node
that books such work under a law, or loses it, fails the case on every draw. Drawn per port kind
(:data:`PORT_DRAWS`): a rigid body's at a millimetre swimmer's scales (:func:`rigid_body_states`); a chain segment's and
a fluid's on their own rigs (:data:`PORT_CASES`: a fluid's is a water-filled millimetre tube's flow,
:func:`fluid_rig`); a port kind with no draws yet fails the case, naming it. A dynamics that names no port kind passes with its reason for being unmetered, and fails without
one."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax
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


def _symmetric(draw: Callable[..., Any], scale: float) -> np.ndarray:
    """A symmetric 3 x 3 matrix, each entry 0.1 to 1 of ``scale`` with its own sign."""
    entries = draw(st.lists(st.floats(0.1, 1.0), min_size=6, max_size=6))
    signs = draw(st.lists(st.sampled_from((-1.0, 1.0)), min_size=6, max_size=6))
    a, b, c, d, e, f = (scale * size * sign for size, sign in zip(entries, signs, strict=True))
    return np.array([[a, d, e], [d, b, f], [e, f, c]])


#: The joint rig's tree: three joints (revolute about z, revolute about y, prismatic along x), each body's frame named
#: after it, the base fixed in the world; the second joint bounded and the third with Coulomb friction, so that the
#: stops' and friction's impulses act on the draws that reach them.
_JOINTS = (("swing", 0, True, (0.0, 0.0, 1.0), (0.0, 0.0, 0.1)), ("tilt", 1, True, (0.0, 1.0, 0.0), (0.2, 0.0, 0.0)),
           ("slide", 2, False, (1.0, 0.0, 0.0), (0.15, 0.0, 0.0)))
#: The spinner rig's host: one frame, the motor mounted on it.
_MOUNT_COUNT = 1


def joint_rig(dt: float = 5e-4) -> SimulationNode:
    """A dynamic arm's node of :data:`_JOINTS`' tree at an arm's scales (links of 0.2 to 0.6 kg, 10 to 20 cm), as a
    build makes it: what the joint port kind's step identity is checked on."""
    from mime.nodes.actuation import DynamicArmNode

    dofs = len(_JOINTS)
    identity = (1.0, 0.0, 0.0, 0.0)

    def flat(rows: Any) -> tuple[float, ...]:
        return tuple(float(v) for v in np.ravel(np.asarray(rows, dtype=np.float64)))

    return DynamicArmNode(
        "case_arm", dt, dof_names=[name for name, *_ in _JOINTS], parents=[parent - 1 for _, parent, *_ in _JOINTS],
        revolute=[revolute for _, _, revolute, *_ in _JOINTS], frame_names=[name for name, *_ in _JOINTS],
        frame_bodies=list(range(dofs)), axes=flat([axis for *_, axis, _ in _JOINTS]),
        origin_translations=flat([origin for *_, origin in _JOINTS]), origin_rotations=flat([identity] * dofs),
        frame_translations=flat([(0.05, 0.02, 0.0)] * dofs), frame_rotations=flat([identity] * dofs),
        masses=[0.6, 0.4, 0.2], centres_of_mass=flat([(0.05, 0.0, 0.02), (0.08, 0.01, 0.0), (0.04, 0.0, -0.01)]),
        inertias=flat([np.diag([2e-3, 3e-3, 2.5e-3]), np.diag([1e-3, 1.5e-3, 1.2e-3]), np.diag([3e-4, 4e-4, 3.5e-4])]),
        armatures=[1e-4, 5e-5, 0.0], damping=[0.02, 0.01, 0.5], rubbing=[2], friction=[0.3], bounded=[1],
        lower=[-0.6], upper=[0.6], placement=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), gravity=(0.0, 0.0, -9.80665),
        initial_joint_positions=[0.0] * dofs)


@st.composite
def joint_states(draw: Callable[..., Any]) -> dict[str, np.ndarray]:
    """A dynamic arm's state and what is delivered to it, at an arm's scales: joint positions within half a radian
    (the bounded joint, on half the draws, a fraction of a step's travel short of its stop and moving towards it, so
    that the step reaches it), rates to 2 rad/s, joint
    torques to 1 N m (a force to 5 N on the prismatic joint), loads on its frames to 1 N and 0.1 N m (a second set,
    fed to the unattributed input), spinners' momenta to 1e-3 N m s, and an inertia its frames carry of 1e-4 kg m^2 at
    the step's start and that changed by up to a hundredth at its end; each component drawn 0.1 to 1 of its scale
    with its own sign."""
    frames = len(_JOINTS)

    def signed(scale: float, size: int) -> np.ndarray:
        magnitudes = draw(st.lists(st.floats(0.1, 1.0), min_size=size, max_size=size))
        signs = draw(st.lists(st.sampled_from((-1.0, 1.0)), min_size=size, max_size=size))
        return scale * np.asarray(magnitudes) * np.asarray(signs)

    q = signed(0.5, frames)
    v = signed(2.0, frames)
    if draw(st.booleans()):
        # The bounded joint a fraction of a step's travel short of its stop, moving towards it: the step reaches it.
        q[1] = np.sign(v[1]) * (0.6 - draw(st.floats(0.1, 0.9)) * 5e-4 * abs(v[1]))
    carried = np.zeros((frames, 2, 3, 3))
    for k in range(frames):
        start = _symmetric(draw, 1e-4)
        carried[k, 0] = start
        carried[k, 1] = start + _symmetric(draw, 1e-6)
    return {"q": q, "v": v, "torques": signed(1.0, 2), "forces": signed(5.0, 1),
            "wrench": np.concatenate([signed(1.0, 3 * frames).reshape(frames, 3),
                                      signed(0.1, 3 * frames).reshape(frames, 3)], axis=1),
            "unattributed": np.concatenate([signed(1.0, 3 * frames).reshape(frames, 3),
                                            signed(0.1, 3 * frames).reshape(frames, 3)], axis=1),
            "momentum": signed(1e-3, 3 * frames).reshape(frames, 3), "inertia": carried}


def spinner_rig(dt: float = 5e-4) -> SimulationNode:
    """A dynamic motor's node mounted on one host frame, its rotor of 2e-5 kg m^2 about its axis with a half
    difference across it, its bearing damped: what the spinner port kind's step identity is checked on."""
    from mime.nodes.actuation import DynamicMotorNode

    axis = np.array([0.3, -0.2, 1.0]) / np.linalg.norm([0.3, -0.2, 1.0])
    across = np.cross(axis, [1.0, 0.0, 0.0])
    across /= np.linalg.norm(across)
    asymmetry = 3e-6 * (np.outer(across, across) - np.outer(np.cross(axis, across), np.cross(axis, across)))
    return DynamicMotorNode("case_motor", dt, axis=tuple(axis), inertia=2e-5, damping=1e-6,
                            placement=(0.01, -0.02, 0.03, 1.0, 0.0, 0.0, 0.0), asymmetry=tuple(asymmetry.ravel()),
                            mount_index=0, mount_count=_MOUNT_COUNT, initial_angle=0.0)


@st.composite
def spinner_states(draw: Callable[..., Any]) -> dict[str, np.ndarray]:
    """A dynamic motor's state and what is delivered to it: its angle in a turn, its spin to 600 rad/s, its torque to
    1e-2 N m, loads on its two frames to 0.1 N and 1e-3 N m (a second set, fed to the unattributed input), and its
    host frame's pose anywhere within a metre and its angular velocity to 10 rad/s; each component drawn 0.1 to 1 of
    its scale with its own sign."""
    def signed(scale: float, size: int) -> np.ndarray:
        magnitudes = draw(st.lists(st.floats(0.1, 1.0), min_size=size, max_size=size))
        signs = draw(st.lists(st.sampled_from((-1.0, 1.0)), min_size=size, max_size=size))
        return scale * np.asarray(magnitudes) * np.asarray(signs)

    def wrench() -> np.ndarray:
        return np.concatenate([signed(0.1, 6).reshape(2, 3), signed(1e-3, 6).reshape(2, 3)], axis=1)

    return {"angle": signed(3.0, 1)[0], "spin": signed(600.0, 1)[0], "torque": signed(1e-2, 1)[0], "wrench": wrench(),
            "unattributed": wrench(), "mount": np.concatenate([signed(1.0, 3), _unit(draw, 4)]),
            "rate": signed(10.0, 3)}


def _arm_step(twin: Any, deliveries: Sequence[ledger.LedgerDelivery], drawn: Mapping[str, Any],
              dtype: Any) -> Mapping[str, Any]:
    p = {key: jnp.asarray(value, dtype=dtype) for key, value in twin._step_constants(None).items()}
    held = twin.initial_state()
    state = {key: jnp.asarray(value, dtype=dtype) for key, value in held.items() if not key.startswith(ledger.RECORDS)}
    q, v = jnp.asarray(drawn["q"], dtype=dtype), jnp.asarray(drawn["v"], dtype=dtype)
    articulation = twin.articulation(p, dtype)
    from mime.core.multibody import forward_kinematics, frame_motions

    base = jnp.asarray(p["placement"], dtype=dtype)
    _, local = frame_motions(articulation, q, v)
    state.update(joint_positions=q, joint_velocities=v, frame_poses=twin.in_world(base, forward_kinematics(
        articulation, q)), frame_angular_velocities=twin.rates_in_world(base, jnp.zeros(3, dtype=dtype), local))
    state = ledger.with_records(twin, state)
    values = {"link_wrench": drawn["wrench"], "link_momentum": drawn["momentum"], "link_inertia": drawn["inertia"],
              "joint_torques": drawn["torques"], "joint_forces": drawn["forces"]}
    inputs = {"link_wrench": jnp.asarray(drawn["unattributed"], dtype=dtype)}
    for index, delivery in enumerate(deliveries):
        if delivery.kind != ledger.HOSTED:
            inputs[ledger.law_input(delivery.field, index)] = jnp.asarray(values[delivery.field], dtype=dtype)
    return _compiled(twin)(p, state, inputs)


def _compiled(twin: Any) -> Callable[..., Mapping[str, Any]]:
    """``twin``'s step compiled once for every draw (their shapes are one): eager, a dynamic arm's projected
    Gauss-Seidel sweeps dispatch each operation alone."""
    held = getattr(twin, "_case_step", None)
    if held is None:
        dt = float(twin.delta_t)

        def step(p: Mapping[str, Any], state: Mapping[str, Any], inputs: Mapping[str, Any]) -> Mapping[str, Any]:
            return cast(Mapping[str, Any], twin.advance(p, state, inputs, dt))

        held = cast(Any, jax).jit(step)
        object.__setattr__(twin, "_case_step", held)
    return cast(Callable[..., Mapping[str, Any]], held)


def _spinner_step(twin: Any, deliveries: Sequence[ledger.LedgerDelivery], drawn: Mapping[str, Any],
                  dtype: Any) -> Mapping[str, Any]:
    p = {key: jnp.asarray(value, dtype=dtype) for key, value in twin._step_constants(None).items()}
    held = twin.initial_state()
    state = {key: jnp.asarray(value, dtype=dtype) for key, value in held.items() if not key.startswith(ledger.RECORDS)}
    state.update(angle=jnp.asarray(drawn["angle"], dtype=dtype), spin=jnp.asarray(drawn["spin"], dtype=dtype))
    state = ledger.with_records(twin, state)
    values = {"link_wrench": drawn["wrench"], "torque": drawn["torque"]}
    inputs = {"link_wrench": jnp.asarray(drawn["unattributed"], dtype=dtype),
              "mount_poses": jnp.asarray(drawn["mount"], dtype=dtype)[None],
              "mount_rates": jnp.asarray(drawn["rate"], dtype=dtype)[None]}
    for index, delivery in enumerate(deliveries):
        if delivery.kind != ledger.HOSTED:
            inputs[ledger.law_input(delivery.field, index)] = jnp.asarray(values[delivery.field], dtype=dtype)
    return _compiled(twin)(p, state, inputs)


#: The fluid rig's tube: a millimetre lumen of water, eight cells across, four cells long (the periodic box's), at a
#: step of 0.1 ms (IMEX Euler's CFL 0.4 holds to 0.5 m/s).
_TUBE_RADIUS, _WATER_VISCOSITY, _WATER_DENSITY, _FLUID_STEP = 0.5e-3, 1e-3, 1000.0, 1e-4


def fluid_rig(dt: float = _FLUID_STEP) -> SimulationNode:
    """A medium's flow as a build makes it, IMEX Euler with convection on: a periodic box holding a straight tube of
    water 1 mm across, eight cells across the lumen and four cells long, its wall under its own name. What the fluid
    port kind's step identity is checked on."""
    from mime.nodes.flow.fvm import ImplicitFlowNode, periodic_tube_layout

    across = 8
    layout = periodic_tube_layout(_TUBE_RADIUS, 4 * 2.0 * _TUBE_RADIUS / across, cells_across=across)
    return ImplicitFlowNode("case_flow", dt, cells_x=layout.cells[0], cells_y=layout.cells[1],
                            cells_z=layout.cells[2], spacing=layout.spacing, viscosity=_WATER_VISCOSITY,
                            density=_WATER_DENSITY, tube_centre=layout.centre, tube_axis=layout.axis,
                            tube_radius=layout.radius, wall_name="case_wall")


@st.composite
def fluid_states(draw: Callable[..., Any]) -> dict[str, Any]:
    """A millimetre tube's water and what drives it: a Poiseuille flow along the tube, its mean speed to 5 cm/s with
    its sign, beside a swirl of the same order (a seeded field of 1e-4 to 5e-2 m/s, made divergence-free by the step),
    and a body force per unit mass to 10 m/s^2 (a second, fed to the unattributed input); each component drawn 0.1 to 1
    of its scale with its own sign."""
    def signed(least: float, most: float) -> float:
        return draw(st.floats(least, most)) * draw(st.sampled_from((-1.0, 1.0)))

    return {"mean": signed(1e-4, 5e-2), "swirl": draw(st.floats(1e-4, 5e-2)),
            "seed": draw(st.integers(0, 2 ** 31 - 1)), "load": _vector(draw, 1e-2, 10.0),
            "unattributed": _vector(draw, 1e-2, 10.0)}


def _fluid_step(twin: Any, deliveries: Sequence[ledger.LedgerDelivery], drawn: Mapping[str, Any],
                dtype: Any) -> Mapping[str, Any]:
    p = twin._step_constants(None)
    grid = twin.grid(p)
    c = twin.constants()
    rng = np.random.default_rng(int(drawn["seed"]))
    points = np.asarray(grid.face_points(2, dtype), dtype=np.float64)
    centre = c["tube_centre"]
    across = ((points[..., 0] - centre[0]) ** 2 + (points[..., 1] - centre[1]) ** 2) / c["tube_radius"] ** 2
    axial = 2.0 * float(drawn["mean"]) * np.clip(1.0 - across, 0.0, None)
    cells = tuple(int(n) for n in grid.cells)
    swirl: list[np.ndarray] = [float(drawn["swirl"]) * rng.standard_normal(cells) for _ in range(3)]
    flow = grid.project((jnp.asarray(swirl[0], dtype=dtype), jnp.asarray(swirl[1], dtype=dtype),
                         jnp.asarray(swirl[2] + axial, dtype=dtype)))
    held = twin.initial_state()
    state = {key: jnp.asarray(value) for key, value in held.items() if not key.startswith(ledger.RECORDS)}
    state.update(velocity_x=flow[0], velocity_y=flow[1], velocity_z=flow[2])
    state = ledger.with_records(twin, state)
    values = {"body_force": drawn["load"], "waveform": np.zeros(np.shape(c.get("waveform", np.zeros(7))))}
    inputs = {"body_force": jnp.asarray(drawn["unattributed"], dtype=dtype)}
    for index, delivery in enumerate(deliveries):
        if delivery.kind != ledger.HOSTED:
            inputs[ledger.law_input(delivery.field, index)] = jnp.asarray(values[delivery.field], dtype=dtype)
    return _compiled(twin)(p, state, inputs)


@dataclass(frozen=True)
class PortCase:
    """How a port kind's step identity is checked: its rig (a receiver's node as a build makes it), its draws, the
    inputs one synthetic law each delivers into (with the commands its node pairs), and one step of the metered twin
    from a draw, with a second load fed to the input that sums what no law owns."""

    rig: Callable[[], SimulationNode]
    draws: Callable[[], Any]
    inputs: tuple[str, ...]
    step: Callable[[Any, Sequence[ledger.LedgerDelivery], Mapping[str, Any], Any], Mapping[str, Any]]


#: The port kinds checked on rigs of their own: a dynamic arm's joints, a spinner on a host frame, and a medium's flow.
PORT_CASES: dict[str, PortCase] = {
    "fluid": PortCase(fluid_rig, fluid_states, ("body_force", "waveform"), _fluid_step),
    "joint": PortCase(joint_rig, joint_states, ("joint_forces", "joint_torques", "link_inertia", "link_momentum",
                                                "link_wrench"), _arm_step),
    "spinner": PortCase(spinner_rig, spinner_states, ("link_wrench", "torque"), _spinner_step),
}

#: The draws for each port kind a dynamics' step identity is checked on, by its name.
PORT_DRAWS: dict[str, Callable[[], Any]] = {"rigid_body": rigid_body_states,
                                            **{name: case.draws for name, case in PORT_CASES.items()}}


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
    if dynamics.port in PORT_CASES:
        return _port_case_identity(PORT_CASES[dynamics.port], node, max_examples)
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


def _port_case_identity(case: PortCase, node: SimulationNode | None, max_examples: int) -> VerificationResult:
    """A chain segment's step identity on drawn states: its node (the case's rig by default) metered for one synthetic
    law on each input the case delivers into and the laws it hosts itself, one step from each draw, the identity within
    its floor and work fed into its unattributed input booked as unattributed."""
    from hypothesis import HealthCheck, Phase, given, settings

    built: Any = case.rig() if node is None else node
    spec = built.boundary_input_spec()
    deliveries = [ledger.LedgerDelivery(field, f"case:{field}", ledger.UNMETERED, ledger.LOAD)
                  for field in case.inputs if field in spec and field not in ("link_momentum", "link_inertia")]
    deliveries += [ledger.LedgerDelivery(field, f"case:{field}", ledger.WORKLESS, kind)
                   for field, kind in (("link_momentum", ledger.FLOW_IN), ("link_inertia", ledger.INERTIA))
                   if field in case.inputs and field in spec]
    deliveries += [ledger.LedgerDelivery(role, law, energy_class, ledger.HOSTED)
                   for role, law, energy_class in built.hosted_laws()]
    params = {key: value for key, value in dict(built.params).items() if key != ledger.LEDGER}
    twin = type(built)(built.name, float(built.delta_t), **params, ledger=tuple(d.encoded() for d in deliveries))
    dtype = np.dtype(np.asarray(next(iter(twin.initial_state().values()))).dtype)
    floor = ULPS * float(np.finfo(dtype).eps)
    last: dict[str, Any] = {}
    count = 0

    @settings(max_examples=max_examples, deadline=None, database=None, phases=(Phase.explicit, Phase.generate),
              suppress_health_check=[HealthCheck.too_slow])
    @given(case.draws())
    def prop(drawn: Mapping[str, Any]) -> None:
        nonlocal count
        count += 1
        last.clear()
        last.update(drawn)
        stepped = case.step(twin, deliveries, drawn, dtype)
        over = float(stepped[f"{ledger.RECORDS}step_identity"]) / floor
        assert over <= 1.0, f"the step identity's residual is {over:.3g} of its floor"
        booked = float(stepped[f"{ledger.RECORDS}unattributed_worst"])
        assert booked > 0.0, ("work fed into the unattributed input was not booked as unattributed work: it was "
                              "booked under a law or lost")

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


__all__ = ["PORT_CASES", "PORT_DRAWS", "PortCase", "dynamics_rig", "fluid_rig", "fluid_states", "joint_rig",
           "joint_states", "metered_twin",
           "rigid_body_states", "spinner_rig", "spinner_states", "step_identity"]
