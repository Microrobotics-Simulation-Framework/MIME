"""The dynamic arm's rigid stops and Coulomb friction on coupled trees, against what any inelastic stop and any Coulomb
friction must do, each oracle with the 7a3 head's rule as its control (float64, per module). The trees: a two-link
planar arm whose joints are strongly coupled (its normalised coupling 0.86 at an elbow of 0.5 rad), built here, and
A0's AR4. The head's rule took each joint's friction on its own diagonal of ``M + dt D`` with the others locked, all
at once, and stopped a joint by zeroing its velocity alone: exact on one joint, wrong on coupled ones.

- **Kinetic energy never rises across a stop event**: a joint at or within one step of its stop, turning into it, the
  arm's velocity's direction drawn, no other force: the step's kinetic energy is at most the unconstrained step's.
  The control's draws turn the other joints so that their coupling opposes the stopping joint, where zeroing one
  velocity component adds ``2 l (1 - l) m^2 / M_jj`` (``m`` the coupling momentum, ``l`` drawn in 0.3 to 0.7).
- **Resting on a stop**: from rest under its weight, a joint on the stop its weight pushes it into is held, and the
  others accelerate as the held arm's closed form, ``-M_FF^-1 g_F`` (the review's planar case, -9.46 rad/s^2 on its
  second joint; the free arm gives +20.36). The control moves them as the free arm.
- **Below breakaway**: from rest with no weight, a torque on one joint below its friction leaves every joint at rest.
  The control turns the arm, the torqued joint backwards on the planar arm.
- **Held by friction**: under its weight, with friction above every joint's holding torque, the arm stays put over
  100 steps. The control creeps.
- **Kinetic energy never rises under friction alone**: the velocity's direction and each joint's friction drawn (below
  and above what would stop it), no other force. The control's draws lie along the direction in which the head's
  per-joint stick impulses, applied together, overshoot (an eigenvalue of ``diag(A)^1/2 A^-1 diag(A)^1/2`` above 2.5),
  with friction above what stops each joint, where it multiplies the energy by at least ``(1 - 2.5)^2``.

Every oracle holds to the rounding of the quantity it compares, ``ROUNDING`` units in the last place of its scale (the
unconstrained step's kinetic energy, or the speed the torques or the weight give in one step); each control misses by
orders of magnitude more on every draw.
"""

import math
import warnings
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import multibody_cases as cases
from mime.core.multibody import gravity_forces, mass_matrix
from mime.nodes.actuation import DynamicArmNode

PROPERTY = settings(deadline=None, max_examples=100)
DT = 1e-3
#: The stop events' step: short, so that the velocity products move the step's velocity by far less than the
#: control's gain.
STOP_DT = 1e-4
EPS = float(np.finfo(np.float64).eps)
#: An oracle's tolerance, in units in the last place of its scale.
ROUNDING = 64
IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
#: The planar arm's joint limits, for the stops.
PLANAR_LIMIT = 2.5
GRAVITY = {"planar": np.array([0.0, -9.81, 0.0]), "ar4": np.array([0.0, 0.0, -9.80665])}
#: Each tree's speed (rad/s), the scale of its drawn velocities.
SPEEDS = {"planar": 2.0, "ar4": 0.5}


TREES = {"planar": cases.coupled_planar(PLANAR_LIMIT), "ar4": cases.ar4()}


@cache
def compiled(name: str):
    """``name``'s mass matrix and gravity torques, compiled once: the strategies and closed forms call them per draw."""
    tree = TREES[name]
    return jax.jit(lambda q: mass_matrix(tree, q)), jax.jit(lambda q, g: gravity_forces(tree, q, g))


def inertia_of(name: str, q) -> np.ndarray:
    return np.asarray(compiled(name)[0](jnp.asarray(q, dtype=jnp.float64)))


def weight_of(name: str, q, g) -> np.ndarray:
    return np.asarray(compiled(name)[1](jnp.asarray(q, dtype=jnp.float64), jnp.asarray(g, dtype=jnp.float64)))


def todays_rule(self, p, q, free, inverse, dt):
    """The 7a3 head's rule, the controls' plant: each joint's friction impulse on its own diagonal of ``A`` with the
    others locked, all applied at once through ``A^-1``; then each bounded joint's position clipped and its velocity
    zeroed alone."""
    held = self.constants()
    rubbing, bounded = held["rubbing"], held["bounded"]
    velocity = free
    if rubbing:
        index = np.asarray(rubbing)
        needed = jnp.diagonal(jnp.linalg.inv(inverse))[index] * free[index] / dt
        friction = jnp.asarray(p["friction"])
        torque = jnp.zeros_like(free).at[index].set(jnp.clip(needed, -friction, friction))
        velocity = free - dt * inverse @ torque
    positions = q + dt * velocity
    if bounded:
        index = np.asarray(bounded)
        lower, upper = jnp.asarray(p["lower"]), jnp.asarray(p["upper"])
        reached = positions[index]
        rate = jnp.where((reached > lower) & (reached < upper), velocity[index], 0.0)
        positions = positions.at[index].set(jnp.clip(reached, lower, upper))
        velocity = velocity.at[index].set(rate)
    return positions, velocity


@cache
def arm(name: str, kind: str, planted: bool = False) -> DynamicArmNode:
    """``name``'s node with no damping: ``stops`` (every joint bounded, no friction), ``friction`` (every joint with
    friction, its levels set by each call, no limits) or ``bare`` (neither); with ``planted``, the head's rule."""
    tree = TREES[name]
    a = tree.articulation
    described = DynamicArmNode.described(a, [0.0] * a.dofs, [1.0 if kind == "friction" else 0.0] * a.dofs)
    if kind != "stops":
        described.update(bounded=(), lower=(), upper=())
    node = DynamicArmNode("arm", DT, **described, placement=IDENTITY, gravity=(0.0, 0.0, 0.0),
                          initial_joint_positions=(0.0,) * a.dofs)
    if planted:
        node.constrained = todays_rule.__get__(node)  # type: ignore[method-assign]
    return node


#: Each node's initial state, computed once: every step replaces its joints' positions and velocities.
INITIAL: dict[int, dict] = {}


def initial(node: DynamicArmNode) -> dict:
    if id(node) not in INITIAL:
        INITIAL[id(node)] = dict(node.initial_state())
    return INITIAL[id(node)]


def step(node: DynamicArmNode, q, qd, *, torques=None, dt=DT, steps: int = 1, **params):
    """The joints' positions and velocities after ``steps`` compiled steps from ``(q, qd)``."""
    state = {**initial(node), "joint_positions": jnp.asarray(q, dtype=jnp.float64),
             "joint_velocities": jnp.asarray(qd, dtype=jnp.float64)}
    given = {} if torques is None else {"joint_torques": jnp.asarray(torques, dtype=jnp.float64)}
    params = {key: jnp.asarray(v) for key, v in params.items()}
    if steps == 1:
        state = node.update(state, given, dt, params=params)
    else:
        state = run(node, dt, steps, tuple(sorted(given)), tuple(sorted(params)))(state, given, params)
    return np.asarray(state["joint_positions"]), np.asarray(state["joint_velocities"])


@cache
def run(node: DynamicArmNode, dt: float, steps: int, inputs: tuple[str, ...], overridden: tuple[str, ...]):
    """``steps`` of the node's own step in one compiled scan, its constants overridden by the parameters given (the
    held runs' hundred steps, which a step at a time would compile once and dispatch a hundred times)."""
    held = node._step_constants(None)

    def scanned(state, given, params):
        p = {**held, **params}
        return jax.lax.scan(lambda s, _: (node.advance(p, s, given, dt), None), state, None, length=steps)[0]

    return jax.jit(scanned)


def kinetic(name: str, q, qd) -> float:
    v = np.asarray(qd)
    return float(0.5 * v @ inertia_of(name, q) @ v)


def limits(name: str) -> tuple[np.ndarray, np.ndarray]:
    a = TREES[name].articulation
    return np.asarray(a.lower, dtype=np.float64), np.asarray(a.upper, dtype=np.float64)


def couplings(name: str, q) -> np.ndarray:
    """Each joint's largest normalised coupling to another, ``|M_jk| / sqrt(M_jj M_kk)``."""
    m = inertia_of(name, q)
    d = np.sqrt(np.diag(m))
    normalised = np.abs(m) / np.outer(d, d)
    np.fill_diagonal(normalised, 0.0)
    return normalised.max(axis=1)


@st.composite
def poses(draw, name: str, margin: float = 0.1):
    """Joint positions inside the middle of each joint's range."""
    lower, upper = limits(name)
    unit = draw(st.lists(st.floats(margin, 1.0 - margin), min_size=len(lower), max_size=len(lower)))
    return lower + (upper - lower) * np.asarray(unit)


def unit_vectors(n: int):
    return st.lists(st.floats(-1.0, 1.0), min_size=n, max_size=n).map(np.asarray).filter(
        lambda v: float(np.linalg.norm(v)) > 0.1).map(lambda v: v / np.linalg.norm(v))


# -- the stops ------------------------------------------------------------------------------------------------------

def at_the_stop(name: str, q, joint: int, velocity, fraction: float) -> np.ndarray:
    """``q`` with ``joint`` ``fraction`` of one step's travel inside the stop its velocity turns it into."""
    lower, upper = limits(name)
    held = np.array(q, dtype=np.float64)
    travel = fraction * STOP_DT * abs(velocity[joint])
    held[joint] = lower[joint] + travel if velocity[joint] < 0 else upper[joint] - travel
    return held


@st.composite
def stop_events(draw):
    """A pose, a joint at or within one step of its stop, and a velocity whose direction is drawn, turning the joint
    into the stop at a tenth of the tree's speed or more."""
    name = draw(st.sampled_from(sorted(TREES)))
    q = draw(poses(name))
    joint = draw(st.integers(0, len(q) - 1))
    velocity = SPEEDS[name] * draw(unit_vectors(len(q)))
    if abs(velocity[joint]) < 0.1 * SPEEDS[name]:
        velocity[joint] = math.copysign(0.1 * SPEEDS[name], velocity[joint] or 1.0)
    return name, at_the_stop(name, q, joint, velocity, draw(st.floats(0.0, 0.99))), velocity


def at_a_limit(name: str, q, joint: int, side: int) -> np.ndarray:
    """``q`` with ``joint`` at its lower limit (``side`` -1) or its upper one (+1)."""
    held = np.array(q, dtype=np.float64)
    held[joint] = limits(name)[0 if side < 0 else 1][joint]
    return held


@st.composite
def gaining_stop_events(draw):
    """A stop event along which zeroing the stopping joint's velocity alone adds energy: a joint coupled to another
    at its stop (normalised coupling 0.2 or more there), the others turning along its coupling row, and its own
    velocity ``l`` of the way to twice ``-m / M_jj`` (``m`` the others' coupling momentum), into its stop."""
    name = draw(st.sampled_from(sorted(TREES)))
    q = draw(poses(name))
    ends = [(joint, side) for joint in range(len(q)) for side in (-1, 1)
            if couplings(name, at_a_limit(name, q, joint, side))[joint] >= 0.2]
    assume(ends)
    joint, side = draw(st.sampled_from(ends))
    q = at_a_limit(name, q, joint, side)
    m = inertia_of(name, q)
    row = np.delete(m[joint], joint)
    others = row / np.linalg.norm(row) + 0.3 * draw(unit_vectors(len(row)))
    if float(row @ others) * side > 0:
        others = -others
    share = draw(st.floats(0.3, 0.7))
    velocity = np.insert(others, joint, -share * 2.0 * float(row @ others) / m[joint, joint])
    velocity *= SPEEDS[name] / np.linalg.norm(velocity)
    return name, at_the_stop(name, q, joint, velocity, draw(st.floats(0.0, 0.9))), velocity


def stop_energies(name: str, q, velocity, planted: bool = False) -> tuple[float, float, np.ndarray]:
    """The kinetic energy (at ``q``) of the unconstrained step and of the step with the stops, and the positions."""
    _, free = step(arm(name, "bare"), q, velocity, dt=STOP_DT)
    after, stopped = step(arm(name, "stops", planted), q, velocity, dt=STOP_DT)
    return kinetic(name, q, free), kinetic(name, q, stopped), after


@PROPERTY
@given(stop_events())
def test_kinetic_energy_never_rises_across_a_stop_event(drawn):
    name, q, velocity = drawn
    free, stopped, after = stop_energies(name, q, velocity)
    lower, upper = limits(name)
    assert stopped <= free * (1.0 + ROUNDING * EPS), f"{name}: {free:.17g} J to {stopped:.17g} J"
    assert np.all(after >= lower) and np.all(after <= upper)


@PROPERTY
@given(gaining_stop_events())
def test_control_the_heads_stop_adds_energy_on_every_draw_where_the_coupling_opposes_it(drawn):
    name, q, velocity = drawn
    free, stopped, _ = stop_energies(name, q, velocity, planted=True)
    assert stopped > free * (1.0 + 1e3 * ROUNDING * EPS), f"{name}: {free:.17g} J to {stopped:.17g} J"


def resting(name: str, q, joint: int, planted: bool = False, **params) -> tuple[np.ndarray, np.ndarray]:
    """One step from rest under the weight, ``joint`` on its stop: the joints' accelerations, and the held arm's."""
    g = GRAVITY[name]
    _, velocity = step(arm(name, "stops", planted), q, np.zeros(len(q)), gravity=g, **params)
    m = inertia_of(name, q)
    weight = weight_of(name, q, g)
    free = np.delete(np.arange(len(q)), joint)
    held = np.zeros(len(q))
    held[free] = -np.linalg.solve(m[np.ix_(free, free)], weight[free])
    return velocity / DT, held


def held_misfit(name: str, q, joint: int, planted: bool = False) -> tuple[float, float]:
    """The largest departure of a joint's acceleration from the held arm's, each relative to the scale of its own
    solve's inputs, ``(|M^-1| |g|)_j`` (an acceleration can cancel to near nothing: an elbow whose weight barely turns
    it), and the rounding of the solves either takes: ``ROUNDING`` units in the last place times the mass matrix's
    condition number."""
    found, held = resting(name, q, joint, planted)
    m = inertia_of(name, q)
    scale = np.abs(np.linalg.inv(m)) @ np.abs(weight_of(name, q, GRAVITY[name]))
    rounding = ROUNDING * EPS * float(np.linalg.cond(m))
    return float(np.max(np.abs(found - held) / scale)), rounding


@st.composite
def resting_poses(draw):
    """A pose with one joint on the stop its weight pushes it into, there (a joint whose free acceleration under the
    weight is a tenth of the largest or more, into the stop), the others inside their range."""
    name = draw(st.sampled_from(sorted(TREES)))
    q = draw(poses(name))
    ends = []
    for joint in range(len(q)):
        for side in (-1, 1):
            at = at_a_limit(name, q, joint, side)
            m = inertia_of(name, at)
            falling = -np.linalg.solve(m, weight_of(name, at, GRAVITY[name]))
            if falling[joint] * side >= 0.1 * np.max(np.abs(falling)):
                ends.append((joint, side))
    assume(ends)
    joint, side = draw(st.sampled_from(ends))
    return name, at_a_limit(name, q, joint, side), joint


REVIEWED = {"lower": np.array([0.3, -PLANAR_LIMIT]), "upper": np.array([3.0, PLANAR_LIMIT])}


@pytest.mark.parametrize("planted", [False, True], ids=["oracle", "control"])
def test_the_planar_arm_resting_on_its_stop_moves_its_other_joint_as_the_held_arms_closed_form(planted):
    """The review's case: the first joint on its lower stop at 0.3 rad, the elbow at 0.5 rad, in a vertical plane:
    the held arm's elbow accelerates at ``-g_2 / M_22``, -9.46 rad/s^2, and the head's (the control) at the free
    arm's, +20.36."""
    found, held = resting("planar", np.array([0.3, 0.5]), 0, planted, **REVIEWED)
    print(f"\nthe elbow's acceleration {found[1]:.6g} rad/s^2, the held arm's {held[1]:.6g}")
    assert (np.max(np.abs(found - held)) <= ROUNDING * EPS * abs(held[1])) is not planted


@PROPERTY
@given(resting_poses())
def test_an_arm_resting_on_a_stop_is_held_there_and_its_other_joints_move_as_the_held_arms_closed_form(drawn):
    name, q, joint = drawn
    found, _ = resting(name, q, joint)
    misfit, rounding = held_misfit(name, q, joint)
    # The misfit holds the joint on its stop too: its acceleration against the held arm's zero.
    assert misfit <= rounding, (name, joint, found, misfit, rounding)


@PROPERTY
@given(resting_poses().filter(lambda d: couplings(d[0], d[1])[d[2]] >= 0.05))
def test_control_the_heads_stop_moves_the_other_joints_as_the_free_arm_on_every_draw(drawn):
    name, q, joint = drawn
    misfit, rounding = held_misfit(name, q, joint, planted=True)
    assert misfit > 1e3 * rounding, (name, joint, misfit, rounding)


# -- Coulomb friction -----------------------------------------------------------------------------------------------

@st.composite
def frictions(draw, scale):
    """Friction levels from a tenth to ten times ``scale`` per joint, at its own joint's scale."""
    return np.asarray(scale) * np.asarray(draw(st.lists(st.floats(0.1, 10.0), min_size=len(scale),
                                                          max_size=len(scale))))


def inertia_scale(name: str, q) -> np.ndarray:
    """Each joint's torque scale: its own inertia times a hundred radians per second squared."""
    return 100.0 * np.diag(inertia_of(name, q))


@st.composite
def breakaways(draw):
    """From rest, a torque on one coupled joint (normalised coupling 0.2 or more) of 0.1 to 0.99 of its friction, and
    friction on every joint."""
    name = draw(st.sampled_from(sorted(TREES)))
    q = draw(poses(name))
    coupled = [j for j, c in enumerate(couplings(name, q)) if c >= 0.2]
    assume(coupled)
    joint = draw(st.sampled_from(coupled))
    friction = draw(frictions(inertia_scale(name, q)))
    torques = np.zeros(len(q))
    torques[joint] = draw(st.sampled_from([-1.0, 1.0])) * draw(st.floats(0.1, 0.99)) * friction[joint]
    return name, q, friction, torques


def breakaway_speed(name: str, q, friction, torques, planted: bool = False) -> tuple[float, float]:
    """The largest joint speed over ten steps, and the speed the torques would give in one step with no friction."""
    _, free = step(arm(name, "bare"), q, np.zeros(len(q)), torques=torques)
    _, found = step(arm(name, "friction", planted), q, np.zeros(len(q)), torques=torques, steps=10, friction=friction)
    return float(np.max(np.abs(found))), float(np.max(np.abs(free)))


def test_a_torque_below_breakaway_on_the_planar_arms_first_joint_leaves_it_at_rest():
    """The review's case: friction 1 N m on the first joint only, a torque of 0.95 N m on it: the head turned it
    backwards at 0.045 rad/s^2."""
    tree = cases.coupled_planar()
    a = tree.articulation
    node = DynamicArmNode("arm", DT, **DynamicArmNode.described(a, [0.0, 0.0], [1.0, 0.0]), placement=IDENTITY,
                          gravity=(0.0, 0.0, 0.0), initial_joint_positions=(0.3, 0.5))
    state = node.initial_state()
    for _ in range(3):
        state = node.update(state, {"joint_torques": jnp.asarray((0.95, 0.0))}, DT)
    assert np.all(np.abs(np.asarray(state["joint_velocities"])) <= ROUNDING * EPS * DT * 0.95 / 0.575)


@PROPERTY
@given(breakaways())
def test_a_torque_below_breakaway_leaves_every_joint_at_rest(drawn):
    name, q, friction, torques = drawn
    found, scale = breakaway_speed(name, q, friction, torques)
    assert found <= ROUNDING * EPS * scale, f"{name}: {found:.3g} rad/s against {scale:.3g}"


@PROPERTY
@given(breakaways())
def test_control_the_heads_friction_turns_the_arm_below_breakaway_on_every_draw(drawn):
    name, q, friction, torques = drawn
    found, scale = breakaway_speed(name, q, friction, torques, planted=True)
    assert found > 1e6 * ROUNDING * EPS * scale, f"{name}: {found:.3g} rad/s against {scale:.3g}"


@st.composite
def held_poses(draw):
    """A pose under the weight, from rest, with friction on every joint from 1.05 to 5 times its holding torque (at
    least a hundredth of the largest)."""
    name = draw(st.sampled_from(sorted(TREES)))
    q = draw(poses(name))
    weight = np.abs(weight_of(name, q, GRAVITY[name]))
    needed = np.maximum(weight, 0.01 * np.max(weight))
    factors = np.asarray(draw(st.lists(st.floats(1.05, 5.0), min_size=len(q), max_size=len(q))))
    return name, q, needed * factors


HELD_STEPS = 100


def drift(name: str, q, friction, planted: bool = False) -> tuple[float, float]:
    """How far any joint moved over ``HELD_STEPS`` from rest under the weight, and how far the weight alone moves the
    free arm in one step (its speed times the step)."""
    g = GRAVITY[name]
    _, free = step(arm(name, "bare"), q, np.zeros(len(q)), gravity=g)
    after, _ = step(arm(name, "friction", planted), q, np.zeros(len(q)), steps=HELD_STEPS, friction=friction, gravity=g)
    return float(np.max(np.abs(after - q))), float(DT * np.max(np.abs(free)))


@PROPERTY
@given(held_poses())
def test_an_arm_under_its_weight_with_friction_above_its_holding_torques_stays_put(drawn):
    name, q, friction = drawn
    moved, scale = drift(name, q, friction)
    assert moved <= ROUNDING * EPS * HELD_STEPS * scale, f"{name}: moved {moved:.3g} rad against {scale:.3g}"


@PROPERTY
@given(held_poses())
def test_control_the_heads_friction_lets_the_held_arm_creep_on_every_draw(drawn):
    name, q, friction = drawn
    moved, scale = drift(name, q, friction, planted=True)
    assert moved > 1e6 * ROUNDING * EPS * HELD_STEPS * scale, f"{name}: moved {moved:.3g} rad against {scale:.3g}"


@st.composite
def sliding(draw):
    """A velocity whose direction is drawn, and each joint's friction from a tenth to ten times what would stop it
    over the step on its own (``|M qd|_j / dt``): some joints stick, some slide."""
    name = draw(st.sampled_from(sorted(TREES)))
    q = draw(poses(name))
    velocity = SPEEDS[name] * draw(unit_vectors(len(q)))
    stopping = np.abs(inertia_of(name, q) @ velocity) / DT
    return name, q, velocity, np.maximum(stopping, 1e-3 * np.max(stopping)) * np.asarray(
        draw(st.lists(st.floats(0.1, 10.0), min_size=len(q), max_size=len(q))))


@st.composite
def overshooting(draw):
    """A pose where ``S = diag(M)^1/2 M^-1 diag(M)^1/2`` has an eigenvalue of 2.5 or more, the velocity along its
    eigenvector (scaled back by ``diag(M)^-1/2``, with a tenth of a drawn direction beside it), and each joint's
    friction from 1.5 to 5 times what the head's rule takes to stop it, ``M_jj |qd_j| / dt``, so that it sticks every
    joint at once and multiplies the energy along that direction by ``(1 - 2.5)^2`` or more."""
    name = draw(st.sampled_from(sorted(TREES)))
    q = draw(poses(name).filter(lambda q: overshoot(name, q)[0] >= 2.5))
    largest, direction = overshoot(name, q)
    m = inertia_of(name, q)
    velocity = (direction + 0.1 * draw(unit_vectors(len(q)))) / np.sqrt(np.diag(m))
    velocity *= SPEEDS[name] / np.linalg.norm(velocity)
    stopping = np.diag(m) * np.abs(velocity) / DT
    return name, q, velocity, np.maximum(stopping, 1e-3 * np.max(stopping)) * np.asarray(
        draw(st.lists(st.floats(1.5, 5.0), min_size=len(q), max_size=len(q))))


def overshoot(name: str, q) -> tuple[float, np.ndarray]:
    m = inertia_of(name, q)
    d = np.sqrt(np.diag(m))
    values, vectors = np.linalg.eigh(np.outer(d, d) * np.linalg.inv(m))
    return float(values[-1]), vectors[:, -1]


def friction_energies(name: str, q, velocity, friction, planted: bool = False) -> tuple[float, float]:
    _, free = step(arm(name, "bare"), q, velocity)
    _, found = step(arm(name, "friction", planted), q, velocity, friction=friction)
    return kinetic(name, q, free), kinetic(name, q, found)


@PROPERTY
@given(sliding())
def test_kinetic_energy_never_rises_under_friction_alone(drawn):
    name, q, velocity, friction = drawn
    free, found = friction_energies(name, q, velocity, friction)
    assert found <= free * (1.0 + ROUNDING * EPS), f"{name}: {free:.17g} J to {found:.17g} J"


@PROPERTY
@given(overshooting())
def test_control_the_heads_friction_adds_energy_along_its_overshooting_direction_on_every_draw(drawn):
    name, q, velocity, friction = drawn
    free, found = friction_energies(name, q, velocity, friction, planted=True)
    assert found > 1.5 * free, f"{name}: {free:.6g} J to {found:.6g} J"


@pytest.fixture(scope="module", autouse=True)
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
