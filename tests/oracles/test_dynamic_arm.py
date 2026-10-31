"""The dynamic arm against quantities derived another way, each oracle with a control that fails on every draw (float64,
per module).

- **The energy balance**, over drawn states, torques and trees (the planar arm with its damping, the AR4 and the
  branched tree with its armature; Coulomb friction is held to zero here and has its own oracle): over a run, the
  change of ``E = (1/2) qd^T M qd + V`` (``V`` the potential of every body's centre of mass, from forward kinematics)
  is the work of the joint torques less what damping took, summed step by step, to the integrator's order: the summed
  size of each step's residual halves as ``dt`` halves, from 1 ms to 0.25 ms over 50 ms (log2 of each ratio within
  ``ORDER_BAND`` of 1). Half the draws start one joint near a stop, turning and pushed into it: a run that reaches
  a stop is held to an inequality instead, since an inelastic stop takes energy that no step size gives back: at
  every step size, no step that ends on a stop gains more energy than the largest residual of a step that does not.
  The control: the node built with no gravity, its potential still counted, fails on every draw along which gravity
  would do at least as much work as the integrator leaves in the balance at its finest step, the plant above the
  oracle's resolution (the residual then holds the work gravity did, which does not shrink with ``dt``: log2 of its
  ratios falls below the band). The stops' own control, the head's stop that zeroed one joint's velocity on a coupled
  tree, is ``test_arm_impulses.py``'s.
- **The kinematic arm against the dynamic one**, on A0's AR4 as the build makes each: driven by a stiff joint
  controller (proportional and derivative on each joint, its stiffness the joint's own inertia at the home pose times
  ``(2 pi 10 Hz)^2``, critically damped, with the weight balanced by the library's gravity torques), the dynamic arm
  follows a slow smooth trajectory about the home pose (0.05 rad, half a hertz, each joint its own phase, starting at
  the trajectory's own speed) within twice the error the controller predicts, ``K^-1 (M qdd_d + C qd_d)`` at the
  trajectory; every frame pose it publishes is the kinematic arm's at the commanded positions within what that joint
  error moves it, ``REACH`` times its largest component. The control: the same run with the arm's gravity dropped
  fails, the controller balancing a weight the arm no longer has.
- **Coulomb friction**, on one joint with a closed form (``descriptions/friction_joint.urdf``: a rod of 0.08 kg m^2
  about the joint, friction 0.5 N m): a torque below the friction leaves it exactly where it was; above it, it turns
  at ``(torque - friction) / inertia`` from rest; turning with no torque, it slows at ``friction / inertia`` and stops,
  and stays. The control: friction applied as ``-friction sign(velocity)`` at the start of the step fails the first.
  On coupled trees, ``test_arm_impulses.py``.
- **Its stops**: a joint driven past its limit arrives at it at the end of the step, its velocity the one that brings
  it there, every joint coupled to it moved by the stop's impulse through ``A^-1``; on the next step the stop takes
  the rest of its velocity. The control: the 0.2 node's stop, which clipped the position and kept the velocity, fails.
  On coupled trees, ``test_arm_impulses.py``.
- **Its loads and its weight through a turned base**, ``test_arm_loads.py``; **its order and MADDENING's
  battery**, ``test_arm_dynamics_battery.py``.
"""

import math
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import AR4_HOME, dynamic_rpm, kinematic_rpm
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import multibody_cases as cases
from mime.core import quaternion
from mime.core.multibody import gravity_forces, inverse_dynamics, mass_matrix, read_urdf
from mime.core.multibody._kinematics import body_poses, constants
from mime.nodes.actuation import DynamicArmNode

PROPERTY = settings(deadline=None, max_examples=100)
EPS = float(np.finfo(np.float64).eps)
#: An oracle's tolerance, in units in the last place of its scale.
ROUNDING = 64
DESCRIPTIONS = Path(__file__).parents[1] / "descriptions"
#: A gravity with a component along every axis, so that the planar arm (turning about z) feels it.
GRAVITY = 9.81 * np.array([0.3, -0.5, -0.8]) / np.linalg.norm([0.3, -0.5, -0.8])
TREES = {name: factory() for name, (factory, *_) in cases.TREES.items()}
#: The energy balance's run: its length, and the steps it is taken at, each half the one before.
HORIZON, STEPS = 0.05, (1e-3, 5e-4, 2.5e-4)
#: How far from 1 the log2 of each halving's ratio may lie for first order.
ORDER_BAND = 0.2
#: The drawn speed (rad/s, or m/s on the branched tree's slide) and acceleration from the torques (rad/s^2): a joint
#: driven at its own inertia's scale, the AR4's wrist with millinewton metres and its shoulder with newton metres.
SPEED, ACCELERATION = 0.3, 20.0


def node_of(tree, q0, qd0, gravity, friction=None) -> DynamicArmNode:
    a = tree.articulation
    return DynamicArmNode("arm", STEPS[0], **DynamicArmNode.described(
        a, [tree.joint(n).damping for n in a.dof_names],
        [0.0] * a.dofs if friction is None else [tree.joint(n).friction for n in a.dof_names]),
        placement=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), gravity=tuple(float(g) for g in gravity),
        initial_joint_positions=tuple(q0), initial_joint_velocities=tuple(qd0))


def potential(articulation, q):
    held = constants(articulation, q.dtype)
    bodies = body_poses(articulation, q, held)
    return sum(-held["masses"][i] * jnp.dot(jnp.asarray(GRAVITY), b[:3] + quaternion.rotate(
        b[3:], held["centres_of_mass"][i])) for i, b in enumerate(bodies))


def energy(articulation, q, qd):
    return 0.5 * qd @ mass_matrix(articulation, q) @ qd + potential(articulation, q)


def balance(tree, gravity):
    """A compiled run per step size: from ``(q0, qd0)`` under ``tau`` for ``HORIZON``, the summed size of the energy
    residual of each step that does not end on a stop, the largest residual (signed) of a step that does and the
    largest size of one that does not, whether a joint reached a stop, the summed size of each step's change of the
    potential (the work gravity did, or would have done, along the run), and the summed size of each step's residual
    with the potential left out (what the integrator's order alone leaves, where the node has no gravity)."""
    a = tree.articulation
    damping = jnp.asarray([tree.joint(n).damping for n in a.dof_names])
    lower, upper = np.maximum(a.lower, -1e3), np.minimum(a.upper, 1e3)

    def run(q0, qd0, tau, h):
        node = node_of(tree, np.zeros(a.dofs), np.zeros(a.dofs), gravity)
        p = node._step_constants(None)
        state = {**node.initial_state(), "joint_positions": q0, "joint_velocities": qd0}

        def step(held, _):
            new = node.advance(p, held, cases.joint_commands(node, tau), h)
            v = new["joint_velocities"]
            kinetic = 0.5 * (v @ mass_matrix(a, new["joint_positions"]) @ v - held["joint_velocities"] @ mass_matrix(
                a, held["joint_positions"]) @ held["joint_velocities"])
            kinetic = kinetic - h * jnp.dot(tau, v) + h * jnp.dot(v, damping * v)
            work = potential(a, new["joint_positions"]) - potential(a, held["joint_positions"])
            stopped = jnp.any((new["joint_positions"] <= lower) | (new["joint_positions"] >= upper))
            return new, (kinetic + work, stopped, jnp.abs(work), jnp.abs(kinetic))

        _, (residuals, stopped, work, kinetic) = jax.lax.scan(step, state, None, length=int(round(HORIZON / h)))
        smooth = jnp.where(stopped, 0.0, jnp.abs(residuals))
        return (jnp.sum(smooth), jnp.max(jnp.where(stopped, residuals, -jnp.inf)), jnp.max(smooth), jnp.any(stopped),
                jnp.sum(work), jnp.sum(kinetic))

    return {h: jax.jit(lambda q0, qd0, tau, h=h: run(q0, qd0, tau, h)) for h in STEPS}


BALANCES = {name: balance(tree, GRAVITY) for name, tree in TREES.items()}
WEIGHTLESS = {name: balance(tree, np.zeros(3)) for name, tree in TREES.items()}


@st.composite
def runs(draw, name):
    """A start within the middle half of each joint's range, a speed of at least a tenth of ``SPEED`` on each joint,
    so that the arm moves and gravity works on it, and torques of the joint's own scale; on half the draws, one joint
    starts near a stop, turning into it and pushed into it, 10 to 50 % of the run's travel at its speed away."""
    tree = TREES[name]
    a = tree.articulation
    lower, upper = np.maximum(a.lower, -math.pi), np.minimum(a.upper, math.pi)
    middle, half = (lower + upper) / 2, (upper - lower) / 4
    unit = arrays(np.float64, (a.dofs,), elements=st.floats(-1.0, 1.0, allow_nan=False))
    q0 = middle + half * draw(unit)
    inertia = np.diag(np.asarray(mass_matrix(a, jnp.asarray(q0))))
    speed = draw(unit)
    qd0 = SPEED * np.sign(speed + 0.5) * (0.1 + 0.9 * np.abs(speed))
    tau = ACCELERATION * inertia * draw(unit)
    if draw(st.booleans()):
        joint = draw(st.integers(0, a.dofs - 1))
        side = draw(st.sampled_from([-1.0, 1.0]))
        q0[joint] = (upper if side > 0 else lower)[joint] - side * abs(qd0[joint]) * HORIZON * draw(st.floats(0.1, 0.5))
        qd0[joint], tau[joint] = side * abs(qd0[joint]), side * abs(tau[joint])
    return name, q0, qd0, tau


def orders(compiled, q0, qd0, tau) -> tuple[list[float], list[float], bool, float, float, list[tuple[float, float]]]:
    """Each step size's summed residual of the steps that do not end on a stop, the order each halving shows,
    whether a stop was reached, at the finest step gravity's summed work and the summed residual with the potential
    left out, and at each step size the largest residual of a step that ends on a stop beside the largest size of one
    that does not."""
    sums, stopped, work, kinetic, stops = [], False, 0.0, 0.0, []
    for h in STEPS:
        total, at_stops, smooth, hit, work_done, left = compiled[h](jnp.asarray(q0), jnp.asarray(qd0),
                                                                    jnp.asarray(tau))
        sums.append(float(total))
        stops.append((float(at_stops), float(smooth)))
        stopped, work, kinetic = stopped or bool(hit), float(work_done), float(left)
    return sums, [math.log2(a / b) if a > 0 and b > 0 else math.nan for a, b in zip(sums, sums[1:], strict=False)], \
        stopped, work, kinetic, stops


@PROPERTY
@given(st.sampled_from(sorted(TREES)).flatmap(runs))
def test_the_energy_balance_closes_to_the_integrators_order(drawn):
    """A run that reaches a stop: no step ending on a stop gains more energy than the largest residual of a step that
    does not, at every step size (the stop takes energy; it never gives any)."""
    name, q0, qd0, tau = drawn
    sums, found, stopped, _, _, stops = orders(BALANCES[name], q0, qd0, tau)
    if stopped:
        assert all(at_stops <= smooth for at_stops, smooth in stops), f"{name}: at the stops, at each step size, " \
                                                                       f"{stops}"
        return
    assert all(abs(order - 1.0) <= ORDER_BAND for order in found), f"{name}: residuals {sums}, orders {found}"


@PROPERTY
@given(st.sampled_from(sorted(TREES)).flatmap(runs))
def test_control_the_arm_with_its_gravity_dropped_fails_the_balance_on_every_draw(drawn):
    """A run along which gravity would do at least as much work as the integrator leaves in the balance at its finest
    step: a plant below the oracle's resolution is no control (one run of twenty found a pose whose bodies move almost
    level, under torques whose first-order residual outweighs the weight's work)."""
    name, q0, qd0, tau = drawn
    sums, found, stopped, work, kinetic, _ = orders(WEIGHTLESS[name], q0, qd0, tau)
    assume(not stopped and work >= kinetic)
    assert any(not abs(order - 1.0) <= ORDER_BAND for order in found), f"{name}: residuals {sums}, orders {found}"


# -- the kinematic arm against the dynamic one ----------------------------------------------------------------------

DT, DURATION = 1e-3, 1.0
AMPLITUDE, FREQUENCY = 0.05, 0.5
#: The controller's bandwidth: its stiffness on each joint is that joint's inertia at home times its square.
BANDWIDTH = 2 * math.pi * 10.0
#: The AR4's reach (m): a joint error moves a frame by at most about this much per radian.
REACH = 0.75
#: Each joint's phase on the trajectory, so that no two move together.
PHASES = np.linspace(0.0, 2.0, 6)


def trajectory(t):
    w = 2 * math.pi * FREQUENCY
    angle = w * t + PHASES
    home = jnp.asarray(AR4_HOME)
    return (home + AMPLITUDE * jnp.sin(angle), AMPLITUDE * w * jnp.cos(angle),
            -AMPLITUDE * w * w * jnp.sin(angle))


@pytest.fixture(scope="module")
def arms():
    """The AR4's kinematic node from A0's kinematic case, and its dynamic node from the dynamic case, starting on the
    trajectory at its own speed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kinematic = kinematic_rpm("kinematic_arm").build().graph.get_node("rpm.arm")
        dynamic = dynamic_rpm("tracked_arm", motor=False, laws=()).build().graph.get_node("rpm.arm")
        tree = read_urdf(Path(__file__).parents[1] / "descriptions" / "ar4.urdf").tree
    return kinematic, dynamic, tree


def tracked(arms, gravity_dropped: bool = False):
    """The run: the dynamic arm under the controller, the kinematic arm at the commanded positions; the joint errors,
    the controller's predicted error at each step, and the largest difference of any frame pose's position."""
    kinematic, dynamic, tree = arms
    a = tree.articulation
    p = dynamic._step_constants(None)
    if gravity_dropped:
        p = {**p, "gravity": jnp.zeros(3)}
    pk = kinematic._step_constants(None)
    g = jnp.asarray((0.0, 0.0, -9.80665))
    inertia = jnp.diag(mass_matrix(a, jnp.asarray(AR4_HOME)))
    stiffness, damping = inertia * BANDWIDTH**2, 2.0 * inertia * BANDWIDTH
    q0, qd0, _ = trajectory(0.0)
    state = {**dynamic.initial_state(), "joint_positions": q0, "joint_velocities": qd0}

    def step(held, n):
        t = n * DT
        target, rate, _ = trajectory(t + DT)
        q, qd = held["joint_positions"], held["joint_velocities"]
        torque = gravity_forces(a, q, g) + stiffness * (target - q) + damping * (rate - qd)
        new = dynamic.advance(p, held, {"joint_torques": torque}, DT)
        commanded = kinematic.advance(pk, kinematic.initial_state(), {"joint_positions": target}, DT)
        _, rate_now, accel = trajectory(t + DT)
        predicted = jnp.abs(inverse_dynamics(a, target, rate_now, accel, jnp.zeros(3)) / stiffness)
        apart = jnp.max(jnp.abs(new["frame_poses"][:, :3] - commanded["frame_poses"][:, :3]))
        return new, (new["joint_positions"] - target, predicted, apart)

    run = jax.jit(lambda s: jax.lax.scan(step, s, jnp.arange(int(DURATION / DT)))[1])
    errors, predicted, apart = run(state)
    return np.asarray(errors), np.asarray(predicted), np.asarray(apart)


def tracking_misfits(errors, predicted, apart) -> list[str]:
    bound = 2.0 * float(np.max(predicted))
    found = []
    if float(np.max(np.abs(errors))) > bound:
        found.append(f"joint error {np.max(np.abs(errors)):.3g} rad above twice the predicted {bound / 2:.3g}")
    if float(np.max(apart)) > REACH * bound:
        found.append(f"frame positions {np.max(apart):.3g} m apart, above {REACH * bound:.3g}")
    return found


def test_the_dynamic_arm_follows_the_kinematic_one_within_a_stiff_controllers_tracking_error(arms):
    errors, predicted, apart = tracked(arms)
    print(f"\njoint error {np.max(np.abs(errors)):.3g} rad, predicted {np.max(predicted):.3g}; frames apart "
          f"{np.max(apart):.3g} m")
    assert tracking_misfits(errors, predicted, apart) == []


def test_control_the_dynamic_arm_with_its_gravity_dropped_leaves_the_kinematic_one(arms):
    errors, predicted, apart = tracked(arms, gravity_dropped=True)
    print(f"\ngravity dropped: joint error {np.max(np.abs(errors)):.3g} rad, predicted {np.max(predicted):.3g}")
    assert len(tracking_misfits(errors, predicted, apart)) == 2


# -- Coulomb friction, on one joint -------------------------------------------------------------------------------

ROD = read_urdf(DESCRIPTIONS / "friction_joint.urdf").tree
ROD_INERTIA, ROD_FRICTION, ROD_STEP = 0.08, 0.5, 1e-3


def rod_run(torque: float, speed: float, steps: int, rule=None):
    """The rod's angles and speeds over ``steps`` from rest at angle 0 (or turning at ``speed``), under ``torque``."""
    node = node_of(ROD, (0.0,), (speed,), (0.0, 0.0, -9.80665), friction=True)
    if rule is not None:
        node.constrained = rule.__get__(node)  # type: ignore[method-assign]
    held = node.initial_state()
    angles, speeds = [], []
    for _ in range(steps):
        held = node.update(held, {"joint_torques": jnp.asarray((torque,))}, ROD_STEP)
        angles.append(float(held["joint_positions"][0]))
        speeds.append(float(held["joint_velocities"][0]))
    return np.asarray(angles), np.asarray(speeds)


def explicit_friction(self, p, q, free, inverse, dt):
    """The planted law: ``-friction sign(velocity)`` at the velocity before the friction, as a torque over the step."""
    rubbing = np.asarray(self.constants()["rubbing"])
    torque = jnp.zeros_like(free).at[rubbing].set(jnp.asarray(p["friction"], dtype=free.dtype) * jnp.sign(free[rubbing]))
    velocity = free - dt * inverse @ torque
    return q + dt * velocity, velocity


#: A speed the rod's step rounds to: 64 roundings of the speed one step of the friction's torque gives it.
ROD_ROUNDING = 64 * float(np.finfo(np.float64).eps) * ROD_STEP * ROD_FRICTION / ROD_INERTIA


def test_coulomb_friction_sticks_below_its_level_and_slides_above_it_at_the_closed_form():
    angles, speeds = rod_run(0.4 * ROD_FRICTION, 0.0, 50)
    assert np.max(np.abs(speeds)) <= ROD_ROUNDING and np.max(np.abs(angles)) <= 50 * ROD_STEP * ROD_ROUNDING
    slide = 2.5 * ROD_FRICTION
    _, speeds = rod_run(slide, 0.0, 50)
    expected = ROD_STEP * np.arange(1, 51) * (slide - ROD_FRICTION) / ROD_INERTIA
    assert np.max(np.abs(speeds - expected)) <= 50 * ROD_ROUNDING
    _, speeds = rod_run(0.0, 0.2, 60)
    stopping = np.maximum(0.2 - ROD_STEP * np.arange(1, 61) * ROD_FRICTION / ROD_INERTIA, 0.0)
    assert np.max(np.abs(speeds - stopping)) <= 60 * ROD_ROUNDING


def test_control_friction_applied_explicitly_does_not_stick():
    angles, speeds = rod_run(0.4 * ROD_FRICTION, 0.0, 50, rule=explicit_friction)
    assert np.max(np.abs(speeds)) > 1e6 * ROD_ROUNDING


# -- its stops -----------------------------------------------------------------------------------------------------

STOP_STEP = 1e-3


def stop_run(rule=None):
    """The planar arm with its first joint 1e-3 rad inside its upper limit, turning out at 2 rad/s: two steps with no
    torque or weight, and the step its unconstrained twin takes from the same start."""
    tree = TREES["planar"]
    upper = float(tree.articulation.upper[0])
    start = (upper - 1e-3, 0.0, 0.0)
    node = node_of(tree, start, (2.0, 0.0, 0.0), np.zeros(3))
    if rule is not None:
        node.constrained = rule.__get__(node)  # type: ignore[method-assign]
    free = node_of(tree, start, (2.0, 0.0, 0.0), np.zeros(3))
    p = free._step_constants(None)
    free_velocity = np.asarray(free.advance({**p, "lower": jnp.full(3, -jnp.inf), "upper": jnp.full(3, jnp.inf)},
                                            free.initial_state(), {}, STOP_STEP)["joint_velocities"])
    first = node.update(node.initial_state(), {}, STOP_STEP)
    second = node.update(first, {}, STOP_STEP)
    a = tree.articulation
    system = np.asarray(mass_matrix(a, jnp.asarray(start))) + STOP_STEP * np.diag([tree.joint(n).damping
                                                                                  for n in a.dof_names])
    return upper, start, free_velocity, np.linalg.inv(system), first, second


def kept_velocity(self, p, q, free, inverse, dt):
    """The 0.2 node's stop: the position clipped to the limits, the velocity kept."""
    return jnp.clip(q + dt * free, jnp.asarray(p["lower"]), jnp.asarray(p["upper"])), free


def stop_misfits(rule=None) -> list[str]:
    """How the run departs from the stop's law: the joint at its limit after the first step, its velocity the one
    that brought it there, the others' moved by the impulse that changed it, ``A^-1 e_1 / (A^-1)_11`` times its change;
    and its velocity taken on the second step."""
    upper, start, free, inverse, first, second = stop_run(rule)
    q, qd = np.asarray(first["joint_positions"]), np.asarray(first["joint_velocities"])
    arrival = (upper - start[0]) / STOP_STEP
    expected = free + inverse[:, 0] / inverse[0, 0] * (arrival - free[0])
    found = []
    if q[0] != upper:
        found.append(f"the joint at {q[0]!r} after the first step, not at its limit {upper!r}")
    # The arrival speed is a difference of positions over the step: its rounding is the positions' over the step.
    if np.max(np.abs(qd - expected)) > ROUNDING * EPS * (abs(upper) / STOP_STEP + np.max(np.abs(free))):
        found.append(f"velocities {qd} after the first step, the impulse's {expected}")
    if float(second["joint_velocities"][0]) != 0.0:
        found.append(f"the joint at {float(second['joint_velocities'][0]):g} rad/s on its stop after the second step")
    return found


def test_a_joint_driven_past_its_limit_arrives_at_it_and_its_impulse_moves_the_joints_coupled_to_it():
    assert stop_misfits() == []


def test_control_the_stop_that_kept_the_velocity_fails():
    assert len(stop_misfits(kept_velocity)) == 2
