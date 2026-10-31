"""The spinner (phase 7a4, MIME-ANO-122): a rotor carried by its absolute axial spin, with only velocities crossing to
its host, a step late. Each oracle with a control that fails on every draw; float64, the nodes as the build makes them,
stepped as the graph steps them (the host first, then the motor reading the host's new state, the motor's reaction
reaching the host on the next step), and a run of the graph itself held to that stepping.

- **Spin momentum**: ``descriptions/spin_joint.urdf``'s hub, one joint along the spin axis, carrying a driven rotor,
  with no gravity and no damping: the axial momentum of the two together, ``I q_dot + J Omega``, is conserved to the
  integrator's order. The drive's reaction reaches the hub a step late, so the momentum moves by ``dt tau`` at most
  (measured: exactly that, from the first step), and by half that at half the step. The control: 7a3's form, the
  rotor's speed relative to its stator with no term from the stator's own turning, which gains ``J`` times the hub's
  change of speed from nothing, at any step.
- **Energy**: A0's dynamic arm with the motor on its flange (5 cm out, turned 0.3 rad) and a rotor symmetric about its
  axis (the magnet turned along it, its mass and inertia across the axis welded on the flange), the drive off, the
  bearing undamped, no gravity, every joint moving from a drawn rate and the rotor from a drawn spin: the kinetic
  energy of the two, ``q_dot M q_dot / 2 + J Omega^2 / 2``, is conserved to the integrator's order (its drift halves
  with the step: the arm's semi-implicit Euler and the gyroscopic torque's step of lag, which does no work in the
  limit), and the rotor's own, ``J Omega^2 / 2``, does not move at all: nothing turns it about its axis. A draw whose
  combined drift is at its rounding at the coarsest step is conserved to rounding, and has no order to measure. The
  control: 7a3's form again, whose rotor's spin, its speed relative to its stator held, moves with the arm's turning
  about the axis.
- **Gyroscopic closed form**: a rotor spinning at ``Omega`` about z, its axis turned about x at a constant ``omega`` by
  a kinematic motor: the torque on the frame it is mounted on is ``J Omega omega``, along ``e x omega``, at every
  step, and its spin does not change. The control: the torque dropped.
"""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import massive_rpm, spinning_hub, turned_spinner
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.core import quaternion
from mime.core.multibody import mass_matrix
from mime.nodes.actuation import DynamicMotorNode
from mime.testing import first_steps

jax.config.update("jax_enable_x64", True)

PROPERTY = settings(deadline=None, max_examples=60)
EPS = float(np.finfo(np.float64).eps)
ROUNDINGS = 256


def built(experiment):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return experiment.build()


_COMPILED: dict[tuple, object] = {}


class RelativeMotor(DynamicMotorNode):
    """The planted motor, 7a3's form: its rotor's speed relative to its stator, driven and damped as if the stator did
    not turn, and no gyroscopic torque. Its state's ``spin`` holds that relative speed."""

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
                "reaction": self.passed(boundary_inputs, base, axis, torque - damping * relative, wrenches,
                                        jnp.zeros(3, dtype=angle.dtype))}


def initial(node):
    """A node's initial state, computed once: its forward kinematics evaluated eagerly costs a tenth of a second."""
    key = ("initial", id(node))
    if key not in _COMPILED:
        _COMPILED[key] = node.initial_state()
    return _COMPILED[key]


def planted(motor):
    return RelativeMotor(motor.name, motor.delta_t, **motor.constants())


def absolute(motor, state, host_rate, axis):
    """The rotor's absolute spin: the real motor's state, the planted one's relative speed with its host's."""
    if isinstance(motor, RelativeMotor):
        return state["spin"] + jnp.dot(axis, host_rate)
    return state["spin"]


def compiled(arm, motor, dt, steps):
    """:func:`stepped` compiled once per pair of nodes, step and length, its constants, states and inputs traced."""
    key = (id(arm), id(motor), dt, steps)
    if key not in _COMPILED:
        _COMPILED[key] = jax.jit(lambda pa, pm, a, r, given_arm, torque: stepped(arm, motor, pa, pm, a, r, given_arm,
                                                                                torque, dt, steps))
    return _COMPILED[key]


def stepped(arm, motor, arm_params, motor_params, arm_state, motor_state, given_arm, torque, dt, steps):
    """The host and the motor stepped as the graph steps them, compiled: the host under ``given_arm`` and the motor's
    reaction of the step before, then the motor on the host's new frame poses and rates. Every step's host state,
    motor state, and the motor's absolute spin."""

    def step(held, _):
        host, rotor, reaction = held
        host = arm.advance(arm_params, host, {**given_arm, "link_wrench": reaction}, dt)
        inputs = {"torque": torque, "mount_poses": host["frame_poses"], "mount_rates": host["frame_angular_velocities"]}
        rotor = motor.advance(motor_params, rotor, inputs, dt)
        index = motor.constants()["mount_index"]
        axis = quaternion.rotate(rotor["frame_poses"][0][3:], jnp.asarray(motor_params["axis"]))
        spin = absolute(motor, rotor, host["frame_angular_velocities"][index], axis)
        return (host, rotor, rotor["reaction"]), (host, rotor, spin)

    start = (arm_state, motor_state, jnp.zeros_like(motor_state["reaction"]))
    _, history = jax.lax.scan(step, start, None, length=steps)
    return history


# -- spin momentum --------------------------------------------------------------------------------------------------

HORIZON = 0.1


@pytest.fixture(scope="module")
def hub():
    graph = built(spinning_hub()).graph
    motor = graph.get_node("hub.motor")
    return graph.get_node("hub.arm"), motor, planted(motor)


@st.composite
def drives(draw):
    """A drive torque of A0's scale (0.1 to 1 mN m either way), a rotor's inertia about its axis of A0's order, a
    start spin to 100 rad/s and a hub's start rate to 2 rad/s."""
    return (draw(st.floats(0.1, 1.0)) * 1e-3 * draw(st.sampled_from((-1.0, 1.0))), draw(st.floats(0.5, 5.0)) * 1e-5,
            draw(st.floats(-100.0, 100.0)), draw(st.floats(-2.0, 2.0)))


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


def test_the_graph_steps_hub_and_rotor_as_they_are_stepped_here(hub):
    experiment = spinning_hub("graph_hub")
    graph_run = first_steps(built(experiment), 20)
    arm, motor, _ = hub
    pa, pm = arm._step_constants(None), motor._step_constants(None)
    hosts, rotors, _ = stepped(arm, motor, pa, pm, arm.initial_state(), motor.initial_state(),
                               {"joint_torques": jnp.zeros(1)}, jnp.asarray(1e-3), 1e-3, 20)
    assert np.allclose(graph_run["hub.arm"]["joint_velocities"], hosts["joint_velocities"], rtol=0.0, atol=1e-15)
    assert np.allclose(graph_run["hub.motor"]["spin"], rotors["spin"], rtol=1e-14, atol=0.0)
    assert float(np.max(np.abs(graph_run["hub.arm"]["joint_velocities"]))) > 0.0


# -- energy ---------------------------------------------------------------------------------------------------------

ENERGY_HORIZON, ENERGY_STEPS = 0.05, (1e-3, 5e-4, 2.5e-4)
#: How far from 1 the log2 of each halving's ratio may lie for first order.
ORDER_BAND = 0.2


@pytest.fixture(scope="module")
def flange():
    graph = built(massive_rpm("symmetric_rotor", magnet_axis=(0.0, 0.0, 1.0))).graph
    motor = graph.get_node("rpm.motor")
    return graph.get_node("rpm.arm"), motor, planted(motor)


@st.composite
def motions(draw):
    """Joint rates of the AR4, each 0.3 to 1 rad/s either way (the file's speed limits are 1.05 to 1.57), and a
    rotor's spin to 100 rad/s; the joints stay far from their stops over the run."""
    rates = [draw(st.floats(0.3, 1.0)) * draw(st.sampled_from((-1.0, 1.0))) for _ in range(6)]
    return np.asarray(rates), draw(st.floats(-100.0, 100.0))


def energy_drift(arm, motor, drawn, dt) -> tuple[float, float, float]:
    """The largest move of the combined kinetic energy from its start over the run at the step ``dt``, the largest
    move of the rotor's spin, and the energy's scale."""
    rates, spin = drawn
    pa = {**arm._step_constants(None), "gravity": jnp.zeros(3)}
    pm = {**motor._step_constants(None), "damping": jnp.zeros(())}
    host = {**initial(arm), "joint_velocities": jnp.asarray(rates)}
    rotor = {**initial(motor), "spin": jnp.asarray(spin)}
    hosts, _, spins = compiled(arm, motor, dt, int(round(ENERGY_HORIZON / dt)))(
        pa, pm, host, rotor, {"joint_torques": jnp.zeros(6)}, jnp.zeros(()))
    energies = kinetic_energies(arm, pa, pm, hosts["joint_positions"], hosts["joint_velocities"], spins)
    start = float(kinetic_energies(arm, pa, pm, initial(arm)["joint_positions"][None], jnp.asarray(rates)[None],
                                   jnp.asarray(spin)[None])[0])
    return float(np.max(np.abs(np.asarray(energies) - start))), float(np.max(np.abs(np.asarray(spins) - spin))), start


def kinetic_energies(arm, pa, pm, q, qd, omega):
    """``q_dot M q_dot / 2 + J Omega^2 / 2`` at each row, the arm's mass matrix its node's (the grafted tree's)."""
    key = ("kinetic", id(arm))
    if key not in _COMPILED:
        def energy(pa, pm, q, qd, omega):
            articulation = arm.articulation(pa, jnp.float64)
            return jax.vmap(lambda a, b, c: 0.5 * b @ mass_matrix(articulation, a) @ b + 0.5 * pm["inertia"] * c ** 2)(
                q, qd, omega)
        _COMPILED[key] = jax.jit(energy)
    return _COMPILED[key](pa, pm, q, qd, omega)


def energy_misfits(arm, motor, drawn) -> list[str]:
    """Every way the run breaks the oracle: an order of the combined energy's drift outside the band (where the drift
    is above its rounding), or the rotor's spin moved beyond its rounding."""
    runs = [energy_drift(arm, motor, drawn, dt) for dt in ENERGY_STEPS]
    found: list[str] = []
    drifts = [run[0] for run in runs]
    rounding = ROUNDINGS * EPS * runs[0][2] * ENERGY_HORIZON / ENERGY_STEPS[-1]
    if drifts[0] > rounding:
        orders = [math.log2(a / b) if b > 0.0 else math.inf for a, b in zip(drifts, drifts[1:], strict=False)]
        if any(abs(order - 1.0) > ORDER_BAND for order in orders):
            found.append(f"the energy's drift is of orders {orders}")
    moved = max(run[1] for run in runs)
    if moved > ROUNDINGS * EPS * abs(drawn[1]) * ENERGY_HORIZON / ENERGY_STEPS[-1]:
        found.append(f"the rotor's spin moved by {moved:.3g} rad/s")
    return found


@PROPERTY
@given(motions())
def test_the_kinetic_energy_of_arm_and_rotor_is_conserved_to_the_integrators_order(flange, drawn):
    arm, motor, _ = flange
    assert energy_misfits(arm, motor, drawn) == []


@PROPERTY
@given(motions())
def test_control_7a3s_relative_speed_fails_the_energy_on_every_draw(flange, drawn):
    arm, _, relative = flange
    assert energy_misfits(arm, relative, drawn)


# -- the gyroscopic closed form -------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def turning():
    graph = built(turned_spinner()).graph
    spin = graph.get_node("rig.spin")
    return graph.get_node("rig.turn"), spin, Ungyroscopic(spin.name, spin.delta_t, **spin.constants())


class Ungyroscopic(DynamicMotorNode):
    """The planted motor: its stator passes no gyroscopic torque."""

    def passed(self, given, base, axis, held, wrenches, turning):
        return super().passed(given, base, axis, held, wrenches, jnp.zeros_like(turning))


@st.composite
def spins(draw):
    """A turn of the axis to 5 rad/s either way, a spin to 200 rad/s, and a rotor's inertia of A0's order."""
    return (draw(st.floats(0.1, 5.0)) * draw(st.sampled_from((-1.0, 1.0))),
            draw(st.floats(1.0, 200.0)) * draw(st.sampled_from((-1.0, 1.0))), draw(st.floats(0.5, 5.0)) * 1e-5)


#: The gyroscopic closed form's run, in steps of 1 ms: its spin's rounding grows by an ulp a step at most.
GYROSCOPIC_STEPS = 50


def gyroscopic_misfit(turn, spin_node, drawn, steps: int = GYROSCOPIC_STEPS) -> tuple[float, float]:
    """The largest distance of the torque the stator passes from ``J Omega (e x omega)``, against the torque's own
    size, over ``steps``; and the largest change of the spin."""
    rate, spin, inertia = drawn
    pt, ps = turn._step_constants(None), {**spin_node._step_constants(None), "inertia": jnp.asarray(inertia)}
    rotor = {**initial(spin_node), "spin": jnp.asarray(spin)}
    key = ("turned", id(turn), id(spin_node), steps)
    if key not in _COMPILED:
        def run(pt, ps, host, rotor_state, rate):
            def step(held, _):
                host, rotor_state = held
                host = turn.advance(pt, host, {"velocity": rate}, 1e-3)
                rotor_state = spin_node.advance(ps, rotor_state, {
                    "torque": jnp.zeros(()), "mount_poses": host["frame_poses"],
                    "mount_rates": host["frame_angular_velocities"]}, 1e-3)
                return (host, rotor_state), rotor_state

            return jax.lax.scan(step, (host, rotor_state), None, length=steps)[1]

        _COMPILED[key] = jax.jit(run)
    rotors = _COMPILED[key](pt, ps, initial(turn), rotor, jnp.asarray(rate))
    index = spin_node.constants()["mount_index"]
    torques = np.asarray(rotors["reaction"])[:, index, 3:]
    axes = np.asarray(jax.vmap(lambda pose: quaternion.rotate(pose[3:], jnp.asarray((0.0, 0.0, 1.0))))(
        rotors["frame_poses"][:, 0]))
    expected = inertia * spin * np.cross(axes, np.asarray((rate, 0.0, 0.0)))
    size = inertia * abs(spin) * abs(rate)
    misfit = float(np.max(np.abs(torques - expected))) / size
    magnitude = float(np.max(np.abs(np.linalg.norm(torques, axis=1) - size))) / size
    return max(misfit, magnitude), float(np.max(np.abs(np.asarray(rotors["spin"]) - spin)))


@PROPERTY
@given(spins())
def test_a_spinning_rotor_turned_at_a_constant_rate_loads_its_host_by_j_omega_omega(turning, drawn):
    turn, spin_node, _ = turning
    misfit, moved = gyroscopic_misfit(turn, spin_node, drawn)
    assert misfit <= ROUNDINGS * EPS, misfit
    assert moved <= 64 * EPS * GYROSCOPIC_STEPS * abs(drawn[1]), moved


@PROPERTY
@given(spins())
def test_control_the_torque_dropped_fails_on_every_draw(turning, drawn):
    turn, _, dropped = turning
    misfit, _ = gyroscopic_misfit(turn, dropped, drawn)
    assert misfit > 0.5, misfit


def test_the_graph_passes_the_gyroscopic_torque_its_closed_form_gives():
    history = first_steps(built(turned_spinner("graph_turned", turn_rate=2.0, spin=50.0)), 10)
    torque = history["rig.spin"]["reaction"][:, :, 3:]
    size = 1e-5 * 50.0 * 2.0
    assert np.allclose(np.linalg.norm(torque[:, 1], axis=1), size, rtol=1e-12, atol=0.0)
    assert np.all(torque[:, 0] == 0.0)
