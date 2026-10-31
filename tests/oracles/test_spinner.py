"""The spinner (phase 7a4, MIME-ANO-122): a rotor carried by its absolute axial spin, with velocities and inertias
crossing to its host. Each oracle with a control that fails on every draw; float64, the nodes as the build makes them,
stepped as the graph steps them (the host first, on the motor's reaction, momentum and inertia of the step before;
then the motor on the host's new frame poses and angular velocities, this step's), and runs of the graph itself held
to that stepping.

- **Spin momentum**: ``descriptions/spin_joint.urdf``'s hub, one joint along the spin axis, carrying a driven rotor,
  with no gravity and no damping: the axial momentum of the two together, ``I q_dot + J Omega``, is conserved to the
  integrator's order. The drive's reaction reaches the hub a step late, so the momentum moves by ``dt tau`` at most
  (measured: exactly that, from the first step), and by half that at half the step. The control: 7a3's form, the
  rotor's speed relative to its stator with no term from the stator's own turning, which gains ``J`` times the hub's
  change of speed from nothing, at any step.
- **The time levels**: the graph steps the hub and its rotor as they are stepped here, the motor reading the hub's
  rates of this step, on a hub whose rotor's axis is turned 0.3 rad off its joint and whose bearing damps it, so that
  the motor's spin depends on the hub's rate along its axis and its rotor's frame on the rate across it. The control:
  the motor reading the hub's rates of the step before, which the graph does not match, by a margin the test prints.
  The gimbal's graph likewise matches its stepping here, its momentum handed to the gimbal on the next step; the
  control, the momentum not handed.
- **Energy**: ``descriptions/gimbal.urdf`` with a rotor of no mass spun about the inner ring's y, across both joints:
  its mass matrix is constant and its velocity-product forces zero, so that a step's change of kinetic energy is the
  gyroscopic coupling's alone. Drive off, bearing undamped, no gravity, the joints moving from drawn rates and the
  rotor spun to a drawn ``nu dt`` from 1e-3 to 0.1 (``nu`` the coupling's frequency): over 1e4 steps the kinetic
  energy stays within 64 units in the last place a step of its start, and the spin does not move: the term does no
  work, and the Cayley step adds none. The control: the coupling as 7a4 first applied it, forward Euler from the
  start-of-step velocity, which grows the energy by ``1 + (nu dt)^2`` a step, past the bound at every draw. What it
  sees: any scheme by which the gyroscopic term does work. What it cannot see: the term's sign (no work either way),
  its size (no work at any size), the rotor's inertia counted twice (a rotor of no mass grafts nothing) and the drive's
  reaction (the drive is off): the closed form below and the tree oracle (``test_spinner_tree.py``) see those.
- **Gyroscopic closed form**: the gimbal at rest at zero but for its yaw at ``omega``, its rotor spun at ``Omega``:
  the torque on the pitch joint is ``J Omega omega``, and on the step after the rotor hands its momentum the pitch rate
  is that torque's over the pitch ring's inertia, within the Cayley step's own ``(nu dt / 2)^2``. The controls: the
  momentum not handed (the torque dropped), and handed with its sign flipped.
"""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import spinning_host, spinning_hub
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.core import quaternion
from mime.core.multibody import frame_angular_jacobians, mass_matrix
from mime.nodes.actuation import DynamicArmNode, DynamicMotorNode
from mime.nodes.actuation._spinner import gyroscopic_matrix
from mime.testing import first_steps

PROPERTY = settings(deadline=None, max_examples=100)
EPS = float(np.finfo(np.float64).eps)
ROUNDINGS = 256


def built(experiment):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return experiment.build()


_COMPILED: dict[tuple, object] = {}


class RelativeMotor(DynamicMotorNode):
    """The planted motor, 7a3's form: its rotor's speed relative to its stator, driven and damped as if the stator did
    not turn, handing no momentum. Its state's ``spin`` holds that relative speed."""

    def advance(self, p, state, boundary_inputs, dt):
        angle, relative = state["angle"], state["spin"]
        base = self.base(p, boundary_inputs)
        mount = self.base_rate(boundary_inputs, angle.dtype)
        axis = quaternion.rotate(base[3:], jnp.asarray(p["axis"], dtype=angle.dtype))
        torque = boundary_inputs.get("torque", jnp.zeros((), dtype=angle.dtype))
        wrenches = boundary_inputs.get("link_wrench", jnp.zeros((2, 6), dtype=angle.dtype))
        inertia, damping = (jnp.asarray(p[name], dtype=angle.dtype) for name in ("inertia", "damping"))
        relative = (inertia * relative + dt * (torque + jnp.dot(axis, wrenches[1, 3:]))) / (inertia + dt * damping)
        angle = angle + dt * relative
        return {**state, "frame_poses": self.poses(p, base, angle), "angle": angle, "spin": relative,
                "frame_angular_velocities": jnp.stack([mount, mount + relative * axis]),
                "reaction": self.passed(boundary_inputs, base, axis, torque - damping * relative, wrenches),
                "momentum": jnp.zeros_like(state["momentum"])}


class Unhanded(DynamicMotorNode):
    """The planted motor: it hands its host no momentum, so the gyroscopic torque is dropped."""

    def handed(self, p, base, axis, angle, spin, relative, dt):
        spun_up, carried = super().handed(p, base, axis, angle, spin, relative, dt)
        return jnp.zeros_like(spun_up), carried


class Flipped(DynamicMotorNode):
    """The planted motor: it hands its host its momentum with the sign flipped, so the gyroscopic torque turns the
    other way."""

    def handed(self, p, base, axis, angle, spin, relative, dt):
        spun_up, carried = super().handed(p, base, axis, angle, spin, relative, dt)
        return -spun_up, carried


class ForwardEulerArm(DynamicArmNode):
    """The planted host, the gyroscopic coupling as 7a4 first applied it: the torque from the start-of-step velocity,
    forward Euler on a skew coupling."""

    def spinning(self, articulation, q, qd, base, inertia, given, dt):
        if "link_momentum" not in given:
            return qd
        momenta = quaternion.rotate_back(base[3:], given["link_momentum"].T).T
        gyroscopic = gyroscopic_matrix(frame_angular_jacobians(articulation, q), momenta)
        return qd + dt * jnp.linalg.solve(inertia, gyroscopic @ qd)


def initial(node):
    """A node's initial state, computed once: its forward kinematics evaluated eagerly costs a tenth of a second. Held
    with the node itself, so that a node freed by another module, whose id this one may reuse, never hands over its
    state."""
    key = ("initial", id(node))
    if key not in _COMPILED or _COMPILED[key][0] is not node:
        _COMPILED[key] = (node, node.initial_state())
    return _COMPILED[key][1]


def planted(node, cls):
    return cls(node.name, node.delta_t, **node.constants())


def absolute(motor, state, host_rate, axis):
    """The rotor's absolute spin: the real motor's state, the planted one's relative speed with its host's."""
    if isinstance(motor, RelativeMotor):
        return state["spin"] + jnp.dot(axis, host_rate)
    return state["spin"]


def compiled(arm, motor, dt, steps, previous_rates=False):
    """:func:`stepped` compiled once per pair of nodes, step and length, its constants, states and inputs traced."""
    key = (id(arm), id(motor), dt, steps, previous_rates)
    if key not in _COMPILED:
        _COMPILED[key] = jax.jit(lambda pa, pm, a, r, given_arm, torque: stepped(
            arm, motor, pa, pm, a, r, given_arm, torque, dt, steps, previous_rates=previous_rates))
    return _COMPILED[key]


def stepped(arm, motor, arm_params, motor_params, arm_state, motor_state, given_arm, torque, dt, steps, *,
            previous_rates=False):
    """The host and the motor stepped as the graph steps them, compiled: the host under ``given_arm`` and the motor's
    reaction, momentum and (for a rotor unequal across its axis) inertia of the step before, then the motor on the
    host's new frame poses and rates (with ``previous_rates``, the planted time level: the rates of the step before).
    Every step's host state, motor state, and the motor's absolute spin."""
    unequal = bool(np.any(np.asarray(motor.constants()["asymmetry"])))
    index = motor.constants()["mount_index"]

    def step(held, _):
        host, rotor = held
        inputs = {**given_arm, "link_wrench": rotor["reaction"], "link_momentum": rotor["momentum"],
                  **({"link_inertia": rotor["inertia"]} if unequal else {})}
        before = host
        host = arm.advance(arm_params, host, inputs, dt)
        rates = (before if previous_rates else host)["frame_angular_velocities"]
        rotor = motor.advance(motor_params, rotor, {"torque": torque, "mount_poses": host["frame_poses"],
                                                    "mount_rates": rates}, dt)
        axis = quaternion.rotate(rotor["frame_poses"][0][3:], jnp.asarray(motor_params["axis"]))
        spin = absolute(motor, rotor, rates[index], axis)
        return (host, rotor), (host, rotor, spin)

    _, history = jax.lax.scan(step, (arm_state, motor_state), None, length=steps)
    return history


# -- spin momentum --------------------------------------------------------------------------------------------------

HORIZON = 0.1


@pytest.fixture(scope="module")
def hub():
    graph = built(spinning_hub()).graph
    motor = graph.get_node("hub.motor")
    return graph.get_node("hub.arm"), motor, planted(motor, RelativeMotor)


def physical(lower: float, upper: float):
    """A float in ``[lower, upper]``, a magnitude under a millionth of the larger bound set to zero rather than left
    subnormal (CLAUDE.md, "Draw what a body can have")."""
    scale = max(abs(lower), abs(upper))
    return st.floats(lower, upper, allow_nan=False).map(lambda x: 0.0 if abs(x) < 1e-6 * scale else x)


def either(lower: float, upper: float):
    """A magnitude in ``[lower, upper]``, either way."""
    return st.tuples(st.floats(lower, upper), st.sampled_from((-1.0, 1.0))).map(lambda pair: pair[0] * pair[1])


@st.composite
def drives(draw):
    """A drive torque of A0's scale (0.1 to 1 mN m either way), a rotor's inertia about its axis of A0's order, a
    start spin to 100 rad/s and a hub's start rate to 2 rad/s."""
    return (draw(either(0.1, 1.0)) * 1e-3, draw(st.floats(0.5, 5.0)) * 1e-5, draw(physical(-100.0, 100.0)),
            draw(physical(-2.0, 2.0)))


def momentum_drift(arm, motor, drawn, dt) -> tuple[float, float]:
    """The largest move of ``I q_dot + J Omega`` from its start over ``HORIZON`` at the step ``dt``, and the scale of
    the momentum (for its rounding)."""
    torque, inertia, spin, rate = drawn
    pa, pm = arm._step_constants(None), {**motor._step_constants(None), "inertia": jnp.asarray(inertia)}
    hub_state = {**initial(arm), "joint_velocities": jnp.asarray((rate,))}
    rotor = {**initial(motor), "spin": jnp.asarray(spin if not isinstance(motor, RelativeMotor) else spin - rate)}
    hosts, _, spins = compiled(arm, motor, dt, int(round(HORIZON / dt)))(
        pa, pm, hub_state, rotor, {"joint_torques": jnp.zeros(1)}, jnp.asarray(torque))
    held = float(mass_matrix(arm.articulation(pa, jnp.float64), jnp.zeros(1))[0, 0])
    momenta = held * np.asarray(hosts["joint_velocities"])[:, 0] + inertia * np.asarray(spins)
    start = held * rate + inertia * spin
    return float(np.max(np.abs(momenta - start))), abs(start) + abs(torque) * HORIZON


@PROPERTY
@given(drives())
def test_the_axial_momentum_of_hub_and_rotor_is_conserved_to_the_integrators_order(hub, drawn):
    arm, motor, _ = hub
    torque = abs(drawn[0])
    coarse, scale = momentum_drift(arm, motor, drawn, 1e-3)
    fine, _ = momentum_drift(arm, motor, drawn, 5e-4)
    rounding = ROUNDINGS * EPS * scale * HORIZON / 5e-4
    assert coarse <= 1e-3 * torque + rounding, (coarse, 1e-3 * torque)
    assert abs(fine - 0.5 * coarse) <= 0.01 * coarse + rounding, (coarse, fine)


@PROPERTY
@given(drives())
def test_control_7a3s_relative_speed_fails_the_momentum_on_every_draw(hub, drawn):
    arm, _, relative = hub
    coarse, _ = momentum_drift(arm, relative, drawn, 1e-3)
    fine, _ = momentum_drift(arm, relative, drawn, 5e-4)
    assert fine > 10.0 * 5e-4 * abs(drawn[0]), (coarse, fine)


# -- the time levels ------------------------------------------------------------------------------------------------

#: The turn of the time levels' rotor off its hub's joint, 0.3 rad about x, so that the hub's rate has a part along
#: the rotor's axis and a part across it.
TURNED = (math.cos(0.15), math.sin(0.15), 0.0, 0.0)
LEVEL_STEPS = 20


def graph_and_stepped(experiment, owner: str, *, previous_rates=False, motor_cls=None) -> float:
    """The largest difference between the graph's first steps of ``experiment`` and its arm and motor stepped here
    from their initial states, in the joint velocities, the spin and the rotor's frame rates, each against its
    scale."""
    graph = built(experiment)
    run = first_steps(graph, LEVEL_STEPS)
    arm, motor = graph.graph.get_node(f"{owner}.arm"), graph.graph.get_node(f"{owner}.motor")
    if motor_cls is not None:
        motor = planted(motor, motor_cls)
    dofs = len(arm.constants()["dof_names"])
    torque = float(np.asarray(run[f"drive:{owner}.motor.torque"]["value"])[0])
    hosts, rotors, _ = stepped(arm, motor, arm._step_constants(None), motor._step_constants(None), initial(arm),
                               initial(motor), {"joint_torques": jnp.zeros(dofs)}, jnp.asarray(torque), 1e-3,
                               LEVEL_STEPS, previous_rates=previous_rates)
    pairs = ((run[f"{owner}.arm"]["joint_velocities"], hosts["joint_velocities"]),
             (run[f"{owner}.motor"]["spin"], rotors["spin"]),
             (run[f"{owner}.motor"]["frame_angular_velocities"], rotors["frame_angular_velocities"]))
    return max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / float(np.max(np.abs(np.asarray(a))))
               for a, b in pairs)


def turned_hub(name: str):
    return spinning_hub(name, damping=1e-4, orientation=TURNED, initial_velocity=30.0, hub_velocity=1.0)


def test_the_graph_steps_hub_and_rotor_as_they_are_stepped_here_at_this_steps_rates():
    misfit = graph_and_stepped(turned_hub("levels_hub"), "hub")
    late = graph_and_stepped(turned_hub("levels_hub_late"), "hub", previous_rates=True)
    print(f"\nthe graph against this step's rates {misfit:.3g}; against the step before's {late:.3g}")
    assert misfit <= ROUNDINGS * EPS
    assert late > 1e6 * ROUNDINGS * EPS, late


def test_the_gimbals_graph_hands_the_rotors_momentum_to_the_gimbal_on_the_next_step():
    def gimbal(name):
        return spinning_host("gimbal", name, rates=(0.5, 0.3), spin=50.0, damping=1e-4, torque=1e-4)

    misfit = graph_and_stepped(gimbal("levels_gimbal"), "gimbal")
    unhanded = graph_and_stepped(gimbal("levels_gimbal_unhanded"), "gimbal", motor_cls=Unhanded)
    print(f"\nthe gimbal's graph against its stepping {misfit:.3g}; with the momentum not handed {unhanded:.3g}")
    assert misfit <= ROUNDINGS * EPS
    assert unhanded > 1e6 * ROUNDINGS * EPS, unhanded


# -- energy ---------------------------------------------------------------------------------------------------------

ENERGY_STEPS, ENERGY_DT = 10_000, 1e-3
#: The energy's rounding a step, in units in the last place of its start: measured, 2e-12 of it over the 1e4 steps.
ENERGY_ROUNDINGS = 64


@pytest.fixture(scope="module")
def gimbal():
    graph = built(spinning_host("gimbal")).graph
    arm = graph.get_node("gimbal.arm")
    return arm, graph.get_node("gimbal.motor"), planted(arm, ForwardEulerArm)


def coupling(arm) -> float:
    """``sqrt(det M)`` of the gimbal at zero, constant over its motion: the gyroscopic frequency is ``J Omega`` over
    it."""
    held = np.asarray(mass_matrix(arm.articulation(arm._step_constants(None), jnp.float64), jnp.zeros(2)))
    return math.sqrt(float(np.linalg.det(held)))


@st.composite
def motions(draw):
    """The gimbal's joint rates, each 0.02 to 0.2 rad/s either way, and its rotor's ``nu dt``, from 1e-3 to 0.1, even
    in its logarithm, either way."""
    return (np.asarray([draw(either(0.02, 0.2)), draw(either(0.02, 0.2))]), 10.0 ** draw(st.floats(-3.0, -1.0)),
            draw(st.sampled_from((-1.0, 1.0))))


def energy_drift(arm, motor, drawn) -> tuple[float, float, float, float]:
    """The largest move of the gimbal's kinetic energy from its start over the run, the largest move of the rotor's
    spin, the energy and the spin."""
    rates, nu_dt, sign = drawn
    pa = arm._step_constants(None)
    pm = {**motor._step_constants(None), "damping": jnp.zeros(())}
    spin = sign * nu_dt / ENERGY_DT * coupling(arm) / float(pm["inertia"])
    host = {**initial(arm), "joint_velocities": jnp.asarray(rates)}
    rotor = {**initial(motor), "spin": jnp.asarray(spin)}
    hosts, _, spins = compiled(arm, motor, ENERGY_DT, ENERGY_STEPS)(pa, pm, host, rotor,
                                                                   {"joint_torques": jnp.zeros(2)}, jnp.zeros(()))
    key = ("kinetic", id(arm))
    if key not in _COMPILED:
        def energies(pa, q, qd):
            articulation = arm.articulation(pa, jnp.float64)
            return jax.vmap(lambda a, b: 0.5 * b @ mass_matrix(articulation, a) @ b)(q, qd)
        _COMPILED[key] = jax.jit(energies)
    held = np.asarray(_COMPILED[key](pa, hosts["joint_positions"], hosts["joint_velocities"]))
    start = float(_COMPILED[key](pa, initial(arm)["joint_positions"][None], jnp.asarray(rates)[None])[0])
    return float(np.max(np.abs(held - start))), float(np.max(np.abs(np.asarray(spins) - spin))), start, spin


def energy_bound(start: float) -> float:
    return ENERGY_ROUNDINGS * EPS * start * ENERGY_STEPS


@PROPERTY
@given(motions())
def test_the_gyroscopic_coupling_does_no_work_over_ten_thousand_steps_at_every_step_size(gimbal, drawn):
    arm, motor, _ = gimbal
    drift, moved, start, spin = energy_drift(arm, motor, drawn)
    assert drift <= energy_bound(start), (drift / start, drawn[1])
    assert moved <= ENERGY_ROUNDINGS * EPS * abs(spin) * ENERGY_STEPS, moved


@PROPERTY
@given(motions())
def test_control_the_forward_euler_coupling_grows_the_energy_on_every_draw(gimbal, drawn):
    _, motor, forward = gimbal
    drift, _, start, _ = energy_drift(forward, motor, drawn)
    assert drift > energy_bound(start), (drift / start, drawn[1])


# -- the gyroscopic closed form -------------------------------------------------------------------------------------

@st.composite
def spins(draw):
    """The gimbal's yaw rate to 2 rad/s either way, a spin to 200 rad/s, and a rotor's inertia of A0's order."""
    return draw(either(0.1, 2.0)), draw(either(1.0, 200.0)), draw(st.floats(0.5, 5.0)) * 1e-5


def gyroscopic_misfit(arm, motor, drawn) -> tuple[float, float]:
    """How far the pitch rate after the second step (the first on which the gimbal holds the rotor's momentum) is
    from ``dt J Omega omega / M_pitch``, against that, and the Cayley step's own ``(nu dt / 2)^2``."""
    rate, spin, inertia = drawn
    pa, pm = arm._step_constants(None), {**motor._step_constants(None), "inertia": jnp.asarray(inertia),
                                         "damping": jnp.zeros(())}
    host = {**initial(arm), "joint_velocities": jnp.asarray((rate, 0.0))}
    rotor = {**initial(motor), "spin": jnp.asarray(spin)}
    hosts, _, _ = compiled(arm, motor, 1e-3, 2)(pa, pm, host, rotor, {"joint_torques": jnp.zeros(2)}, jnp.zeros(()))
    held = np.asarray(mass_matrix(arm.articulation(pa, jnp.float64), jnp.zeros(2)))
    expected = 1e-3 * inertia * spin * rate / held[1, 1]
    nu_dt = abs(inertia * spin) / coupling(arm) * 1e-3
    return abs(float(hosts["joint_velocities"][1, 1]) - expected) / abs(expected), (0.5 * nu_dt) ** 2


@pytest.fixture(scope="module")
def gyroscopic(gimbal):
    arm, motor, _ = gimbal
    return arm, motor, planted(motor, Unhanded), planted(motor, Flipped)


@PROPERTY
@given(spins())
def test_a_spinning_rotor_whose_axis_turns_at_a_rate_loads_its_host_by_j_omega_omega(gyroscopic, drawn):
    arm, motor, _, _ = gyroscopic
    misfit, cayley = gyroscopic_misfit(arm, motor, drawn)
    assert misfit <= cayley + ROUNDINGS * EPS, (misfit, cayley)


@PROPERTY
@given(spins())
def test_control_the_momentum_not_handed_or_flipped_fails_on_every_draw(gyroscopic, drawn):
    arm, _, unhanded, flipped = gyroscopic
    for motor in (unhanded, flipped):
        misfit, _ = gyroscopic_misfit(arm, motor, drawn)
        assert misfit > 0.5, (type(motor).__name__, misfit)
