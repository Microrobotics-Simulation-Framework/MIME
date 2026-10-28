"""The rigid-body library against quantities derived another way, each oracle with a control that fails (float64, per
module). Drawn by Hypothesis over joint states inside the limits of three trees: the planar arm, the AR4 and a branched
tree built in code (``multibody_cases.py``).

- **Forward kinematics** against the planar arm's closed form. Control: joint 2's axis reversed.
- **Jacobians** against ``jax.jacfwd`` of forward kinematics and against central differences. Control: a revolute
  column's lever crossed the wrong way round.
- **The mass matrix** symmetric positive definite, and the Hessian in ``qd`` of the kinetic energy summed over every
  body of the tree from its own velocity (its frame's pose differentiated along ``qd``), plus the armature. Control:
  the composite-rigid-body algorithm given one body with no rotational inertia.
- **Gravity**: inverse dynamics at rest equals ``dV/dq``, ``V`` the potential of every body's centre of mass. Control:
  gravity's sign reversed in inverse dynamics.
- **Velocity products**: inverse dynamics' velocity terms equal ``C(q, qd) qd`` from the Christoffel symbols of the
  mass matrix, ``M_dot qd - (1/2) d(qd^T M qd)/dq`` by automatic differentiation. Control: the recursion without its
  ``v x* I v`` term.
- **Consistency**: inverse dynamics is ``M qdd + h``, and forward dynamics inverts it. Control: forward dynamics with
  the velocity terms left out of ``h``.
- **Inverse kinematics**: from any seed it reaches a target placed by forward kinematics or reports that it did not,
  its report true of where it ended; from a seed in its basin (near the target's joint positions, the Jacobian well
  conditioned all the way) it reaches it. A target out of reach is reported as not reached, and one reached only
  outside the limits as outside them. Controls: a solve claiming every target reached, on solves known not to reach;
  one cut to two iterations.

Tolerances are the float64 rounding of each quantity's own scale, times ``ROUNDINGS``; the finite differences add
their truncation, bounded from the step.
"""

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import Phase, assume, find, given, settings
from hypothesis import strategies as st

import multibody_cases as cases
from mime.core import quaternion
from mime.core.multibody import MultibodyTree, bias_forces, forward_dynamics, forward_kinematics, frame_jacobian, \
    frame_poses, gravity_forces, inverse_dynamics, inverse_kinematics, mass_matrix, rotation_vector
from mime.core.multibody import _dynamics, _kinematics

EPS = cases.EPS
TREES = {name: factory() for name, (factory, *_) in cases.TREES.items()}
SPEED = {name: speed for name, (_, speed, _, _) in cases.TREES.items()}
ACCELERATION = {name: acceleration for name, (_, _, acceleration, _) in cases.TREES.items()}
REACH = {name: reach for name, (_, _, _, reach) in cases.TREES.items()}
#: A gravity with a component along every axis, so that the planar arm (turning about z) feels it.
GRAVITY = 9.81 * np.array([0.3, -0.5, -0.8]) / np.linalg.norm([0.3, -0.5, -0.8])
#: Every frame of the planar arm and the branched tree; of the AR4's, a link welded to the root, a moving link halfway
#: along the chain and the last one, a visual link welded to a moving one, and the end effector.
FRAMES = {"planar": TREES["planar"].frame_names, "branched": TREES["branched"].frame_names,
          "ar4": ("base_link", "link_3", "link_6", "link_2_cover", "ee_link")}
#: How many float64 roundings of its scale each comparison allows: the two sides compute one quantity by different
#: sequences of a few dozen operations each.
ROUNDINGS = 64
PROPERTY = settings(deadline=None)
#: A control needs one failing example, not the smallest: it searches as a property does, and stops at the first.
CONTROL = settings(deadline=None, database=None, phases=(Phase.generate,))

each_tree = pytest.mark.parametrize("name", sorted(TREES))


def rotation(pose: Any) -> Any:
    return quaternion.rotation_matrix(pose[3:])


def vee(skew: Any) -> Any:
    return jnp.stack([skew[2, 1], skew[0, 2], skew[1, 0]])


def largest(x: Any) -> float:
    return float(np.max(np.abs(np.asarray(x))))


FK = {name: jax.jit(lambda q, tree=tree: forward_kinematics(tree, q)) for name, tree in TREES.items()}
MASS = {name: jax.jit(lambda q, tree=tree: mass_matrix(tree, q)) for name, tree in TREES.items()}


# Forward kinematics -------------------------------------------------------------------------------------------------

def matrix_of(q: np.ndarray) -> np.ndarray:
    """The rotation matrix of a unit quaternion (wxyz), in NumPy."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def planar_misfit(tree: MultibodyTree, fk: Any, q: np.ndarray) -> float:
    """The largest difference from the closed form over the three link frames and the tip: positions relative to the
    reach, rotation matrices entry by entry. Joint ``i`` sits at the end of link ``i - 1``, heading along the sum of
    the angles up to it. ``fk`` is the tree's forward kinematics, compiled."""
    poses = np.asarray(fk(jnp.asarray(q)))
    reach, headings = sum(cases.PLANAR_LENGTHS), np.cumsum(q)
    ends = [np.zeros(3)]
    for length, heading in zip(cases.PLANAR_LENGTHS, headings, strict=True):
        ends.append(ends[-1] + length * np.array([np.cos(heading), np.sin(heading), 0.0]))
    found = []
    for i, heading in enumerate(headings):
        link = poses[tree.frame_names.index(f"link_{i + 1}")]
        c, s = np.cos(heading), np.sin(heading)
        turned = matrix_of(link[3:])
        found.append(largest(link[:3] - ends[i]) / reach)
        found.append(largest(turned - np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])))
        if i == 2:
            found.append(largest(link[:3] + turned @ [cases.PLANAR_LENGTHS[2], 0.0, 0.0] - ends[3]) / reach)
    return max(found)


def reversed_axis(tree: MultibodyTree, joint: str) -> MultibodyTree:
    joints = tuple(dataclasses.replace(j, axis=tuple(-a for a in j.axis)) if j.name == joint else j
                   for j in tree.joints)
    return dataclasses.replace(tree, joints=joints)


@PROPERTY
@given(st.data())
def test_forward_kinematics_is_the_planar_arms_closed_form(data):
    q = data.draw(cases.positions(TREES["planar"]))
    assert planar_misfit(TREES["planar"], FK["planar"], q) <= ROUNDINGS * EPS


def test_with_one_axis_reversed_the_closed_form_fails():
    planted = reversed_axis(TREES["planar"], "joint_2")
    fk = jax.jit(lambda q: forward_kinematics(planted, q))
    found = find(cases.positions(planted), lambda q: planar_misfit(planted, fk, q) > ROUNDINGS * EPS,
                 settings=CONTROL)
    print(f"\njoint_2 reversed: misfit {planar_misfit(planted, fk, found):.3g} at q = {found}")


# Jacobians ----------------------------------------------------------------------------------------------------------

def jacobians(model: Any, frames: tuple[str, ...]):
    return jax.jit(lambda q: jnp.stack([frame_jacobian(model, q, f) for f in frames]))


def autodiff_jacobians(name: str):
    """Each frame's ``[dp/dq; vee(dR/dq R^T)]`` by ``jax.jacfwd`` of forward kinematics."""
    tree = TREES[name]

    def placement(q):
        poses = frame_poses(tree, q, FRAMES[name])
        return poses[:, :3], jax.vmap(rotation)(poses)

    def angular(turning, turned):
        return jax.vmap(lambda column: vee(column @ turned.T), in_axes=2, out_axes=1)(turning)

    def jacobian(q):
        (_, turned), (moving, turning) = placement(q), jax.jacfwd(placement)(q)
        return jnp.concatenate([moving, jax.vmap(angular)(turning, turned)], axis=1)

    return jax.jit(jacobian)


ROTATION_BETWEEN = jax.jit(jax.vmap(lambda a, b: rotation_vector(quaternion.multiply(a, quaternion.conjugate(b)))))


def finite_difference_jacobians(name: str, q: np.ndarray, step: float) -> np.ndarray:
    """Central differences: positions differenced, and the rotation from one side's frame to the other's as a
    rotation vector, each over twice the step."""
    frames = [TREES[name].frame_names.index(f) for f in FRAMES[name]]
    columns = []
    for j in range(len(q)):
        shift = np.zeros_like(q)
        shift[j] = step
        ahead, behind = (np.asarray(FK[name](jnp.asarray(q + s)))[frames] for s in (shift, -shift))
        turned = np.asarray(ROTATION_BETWEEN(jnp.asarray(ahead[:, 3:]), jnp.asarray(behind[:, 3:])))
        columns.append(np.concatenate([ahead[:, :3] - behind[:, :3], turned], axis=1) / (2 * step))
    return np.stack(columns, axis=2)


JACOBIANS = {name: jacobians(tree, FRAMES[name]) for name, tree in TREES.items()}
AUTODIFF = {name: autodiff_jacobians(name) for name in TREES}


def jacobian_misfit(name: str, found: np.ndarray, expected: np.ndarray) -> float:
    """The largest difference: linear rows relative to the reach, angular rows (unitless) as they are."""
    return max(largest(found[:, :3] - expected[:, :3]) / REACH[name], largest(found[:, 3:] - expected[:, 3:]))


@each_tree
@PROPERTY
@given(data=st.data())
def test_jacobians_are_forward_kinematics_differentiated(name, data):
    q = jnp.asarray(data.draw(cases.positions(TREES[name])))
    assert jacobian_misfit(name, np.asarray(JACOBIANS[name](q)), np.asarray(AUTODIFF[name](q))) <= ROUNDINGS * EPS


#: The central difference's step, ``eps^(1/3)`` (radians, or metres on a prismatic joint), balances its truncation, at
#: most ``step^2 reach / 6`` since moving one joint carries each point along a circle of radius under the reach (or a
#: line), against its rounding, about ``ROUNDINGS eps reach / step``: both of order ``eps^(2/3) reach``.
STEP = EPS ** (1 / 3)
DIFFERENCE_TOLERANCE = (1 / 6 + ROUNDINGS) * EPS ** (2 / 3)


@each_tree
@PROPERTY
@given(data=st.data())
def test_jacobians_agree_with_central_differences(name, data):
    q = data.draw(cases.positions(TREES[name]))
    found = np.asarray(JACOBIANS[name](jnp.asarray(q)))
    assert jacobian_misfit(name, found, finite_difference_jacobians(name, q, STEP)) <= DIFFERENCE_TOLERANCE


class CrossTheWrongWayRound:
    """``jax.numpy`` with ``cross(a, b)`` giving ``b x a``: a revolute column's lever ``(p - o) x a``, not
    ``a x (p - o)``."""

    def __getattr__(self, name: str) -> Any:
        return getattr(jnp, name)

    @staticmethod
    def cross(a: Any, b: Any) -> Any:
        return jnp.cross(b, a)


@each_tree
def test_a_jacobian_with_its_lever_the_wrong_way_round_fails_both_comparisons(name, monkeypatch):
    monkeypatch.setattr(_kinematics, "jnp", CrossTheWrongWayRound())
    planted = jacobians(TREES[name], FRAMES[name])
    found = find(cases.positions(TREES[name]), lambda q: jacobian_misfit(
        name, np.asarray(planted(jnp.asarray(q))), np.asarray(AUTODIFF[name](jnp.asarray(q)))) > ROUNDINGS * EPS,
        settings=CONTROL)
    differenced = finite_difference_jacobians(name, found, STEP)
    assert jacobian_misfit(name, np.asarray(planted(jnp.asarray(found))), differenced) > DIFFERENCE_TOLERANCE


# The mass matrix ----------------------------------------------------------------------------------------------------

def body_placements(tree: MultibodyTree):
    """Every body with an inertial (its own, uncombined): a function from ``q`` to its centre of mass and rotation in
    the root's frame, from its frame's pose; and the bodies' masses and inertias."""
    bodies = [(b.name, b.inertial) for b in tree.bodies if b.inertial is not None]
    centres = jnp.asarray([i.centre_of_mass for _, i in bodies])

    def placement(q):
        poses = frame_poses(tree, q, [name for name, _ in bodies])
        turned = jax.vmap(rotation)(poses)
        return poses[:, :3] + jnp.einsum("bij,bj->bi", turned, centres), turned

    return placement, jnp.asarray([i.mass for _, i in bodies]), jnp.asarray([i.inertia for _, i in bodies])


def kinetic_energy(tree: MultibodyTree):
    """``T(q, qd) = 1/2 sum (m |v_c|^2 + w . R I R^T w)``, each body's velocities its placement's derivative along
    ``qd``."""
    placement, masses, inertias = body_placements(tree)

    def energy(q, qd):
        (_, turned), (velocity, turning) = jax.jvp(placement, (q,), (qd,))
        omega = jax.vmap(lambda dr, r: vee(dr @ r.T))(turning, turned)
        spin = jnp.einsum("bij,bjk,blk,bl->bi", turned, inertias, turned, omega)
        return 0.5 * (jnp.sum(masses * jnp.sum(velocity * velocity, axis=1)) + jnp.sum(omega * spin))

    return energy


#: The Hessian forward over forward, which compiles in about half the time of reverse over forward.
ENERGY_HESSIAN = {name: jax.jit(lambda q, tree=tree: jax.jacfwd(jax.jacfwd(kinetic_energy(tree), argnums=1), argnums=1)(
    q, jnp.zeros_like(q))) for name, tree in TREES.items()}


def energy_misfit(name: str, mass: np.ndarray, q: np.ndarray) -> float:
    """The mass matrix against the energy's Hessian plus the armature, relative to the Hessian's largest entry."""
    expected = np.asarray(ENERGY_HESSIAN[name](jnp.asarray(q))) + np.diag(TREES[name].articulation.armatures)
    return largest(mass - expected) / largest(expected)


@each_tree
@PROPERTY
@given(data=st.data())
def test_the_mass_matrix_is_symmetric_positive_definite_and_the_energys_hessian(name, data):
    q = data.draw(cases.positions(TREES[name]))
    mass = np.asarray(MASS[name](jnp.asarray(q)))
    assert np.array_equal(mass, mass.T)
    eigenvalues = np.linalg.eigvalsh(mass)
    assert eigenvalues[0] > ROUNDINGS * EPS * eigenvalues[-1]
    assert energy_misfit(name, mass, q) <= ROUNDINGS * EPS


@each_tree
def test_a_mass_matrix_missing_one_bodys_rotational_inertia_fails(name):
    """The composite-rigid-body algorithm given the rigid body with the largest rotational inertia with none."""
    articulation = TREES[name].articulation
    inertias = np.array(articulation.inertias)
    inertias[int(np.argmax([np.trace(i) for i in inertias]))] = 0.0
    planted = jax.jit(lambda q: mass_matrix(dataclasses.replace(articulation, inertias=inertias), q))
    find(cases.positions(TREES[name]), lambda q: energy_misfit(name, np.asarray(planted(jnp.asarray(q))), q)
         > ROUNDINGS * EPS, settings=CONTROL)


# Gravity ------------------------------------------------------------------------------------------------------------

def potential(tree: MultibodyTree):
    """``V(q) = -sum m g . c(q)`` over every body's centre of mass."""
    placement, masses, _ = body_placements(tree)
    return lambda q: -jnp.sum(masses * (placement(q)[0] @ jnp.asarray(GRAVITY)))


GRAVITY_FORCES = {name: jax.jit(lambda q, g, tree=tree: gravity_forces(tree, q, g)) for name, tree in TREES.items()}
#: Each oracle's constants are made when it is traced, under the module's float64, not at import.
POTENTIAL_GRADIENT = {name: jax.jit(lambda q, tree=tree: jax.grad(potential(tree))(q)) for name, tree in TREES.items()}
#: The largest moment gravity can have about a joint: the whole tree's weight at its reach.
WEIGHT_MOMENT = {name: sum(b.inertial.mass for b in tree.bodies if b.inertial is not None)
                 * float(np.linalg.norm(GRAVITY)) * REACH[name] for name, tree in TREES.items()}


def gravity_misfit(name: str, q: np.ndarray, gravity: np.ndarray) -> float:
    found = GRAVITY_FORCES[name](jnp.asarray(q), jnp.asarray(gravity))
    return largest(np.asarray(found) - np.asarray(POTENTIAL_GRADIENT[name](jnp.asarray(q)))) / WEIGHT_MOMENT[name]


@each_tree
@PROPERTY
@given(data=st.data())
def test_inverse_dynamics_at_rest_is_the_potentials_gradient(name, data):
    assert gravity_misfit(name, data.draw(cases.positions(TREES[name])), GRAVITY) <= ROUNDINGS * EPS


@each_tree
def test_gravity_with_its_sign_reversed_fails(name):
    find(cases.positions(TREES[name]), lambda q: gravity_misfit(name, q, -GRAVITY) > ROUNDINGS * EPS,
         settings=CONTROL)


# Velocity products --------------------------------------------------------------------------------------------------

def christoffel_terms(tree: MultibodyTree):
    """``M_dot qd`` and ``(1/2) d(qd^T M qd)/dq``, whose difference is ``C(q, qd) qd``, both from ``dM/dq`` by
    ``jax.jacfwd``."""
    def terms(q, qd):
        derivative = jax.jacfwd(lambda x: mass_matrix(tree, x))(q)  # [i, j, k] = dM_ij / dq_k
        return jnp.einsum("ijk,k,j->i", derivative, qd, qd), 0.5 * jnp.einsum("ijk,i,j->k", derivative, qd, qd)

    return jax.jit(terms)


def velocity_products(tree: MultibodyTree):
    return jax.jit(lambda q, qd: bias_forces(tree, q, qd, jnp.zeros(3)))


CHRISTOFFEL = {name: christoffel_terms(tree) for name, tree in TREES.items()}
VELOCITY_PRODUCTS = {name: velocity_products(tree) for name, tree in TREES.items()}


def velocity_misfit(name: str, recursion: Any, q: np.ndarray, qd: np.ndarray) -> float:
    """Relative to the inputs' scale, the mass matrix's largest entry times the largest speed squared: the terms
    cancel, to zero on a straight planar arm with only its last joint turning, so the result is no scale. At rest
    both sides are exactly zero."""
    changing, gradient = (np.asarray(x) for x in CHRISTOFFEL[name](jnp.asarray(q), jnp.asarray(qd)))
    difference = largest(np.asarray(recursion(jnp.asarray(q), jnp.asarray(qd))) - (changing - gradient))
    scale = largest(MASS[name](jnp.asarray(q))) * largest(qd) ** 2
    return difference / scale if scale > 0 else difference


@each_tree
@PROPERTY
@given(data=st.data())
def test_the_velocity_terms_are_the_christoffel_forms(name, data):
    q, qd, _ = data.draw(cases.states(TREES[name], SPEED[name], ACCELERATION[name]))
    assert velocity_misfit(name, VELOCITY_PRODUCTS[name], q, qd) <= ROUNDINGS * EPS


@each_tree
def test_without_the_gyroscopic_term_the_christoffel_forms_fail(name, monkeypatch):
    monkeypatch.setattr(_dynamics, "force_cross", lambda v, f: jnp.zeros_like(f))
    planted = velocity_products(TREES[name])
    find(cases.states(TREES[name], SPEED[name], ACCELERATION[name]),
         lambda state: velocity_misfit(name, planted, state[0], state[1]) > ROUNDINGS * EPS, settings=CONTROL)


# Consistency --------------------------------------------------------------------------------------------------------

def consistency(tree: MultibodyTree):
    """At a state: ``tau`` by inverse dynamics, ``M qdd + h``, and forward dynamics of ``tau``."""
    def run(q, qd, qdd, g):
        tau = inverse_dynamics(tree, q, qd, qdd, g)
        return tau, mass_matrix(tree, q) @ qdd + bias_forces(tree, q, qd, g), forward_dynamics(tree, q, qd, tau, g)

    return jax.jit(run)


CONSISTENCY = {name: consistency(tree) for name, tree in TREES.items()}


def consistency_misfits(name: str, run: Any, state: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, float]:
    """How far inverse dynamics is from ``M qdd + h``, relative to the larger of its two parts, and how far forward
    dynamics is from ``qdd``, relative to the accelerations' scale times the mass matrix's condition number, the
    rounding a solve with it amplifies."""
    q, qd, qdd = state
    tau, split, recovered = (np.asarray(x) for x in run(*(jnp.asarray(x) for x in (q, qd, qdd, GRAVITY))))
    mass = np.asarray(MASS[name](jnp.asarray(q)))
    bias = tau - mass @ qdd
    torques = max(largest(mass @ qdd), largest(bias))
    accelerations = max(largest(qdd), largest(np.linalg.solve(mass, bias)))
    return largest(tau - split) / torques, largest(recovered - qdd) / (accelerations * float(np.linalg.cond(mass)))


@each_tree
@PROPERTY
@given(data=st.data())
def test_inverse_dynamics_is_m_qdd_plus_h_and_forward_dynamics_inverts_it(name, data):
    split, inverted = consistency_misfits(name, CONSISTENCY[name],
                                          data.draw(cases.states(TREES[name], SPEED[name], ACCELERATION[name])))
    assert split <= ROUNDINGS * EPS and inverted <= ROUNDINGS * EPS


@each_tree
def test_forward_dynamics_without_the_velocity_terms_does_not_invert_it(name, monkeypatch):
    held = _dynamics.bias_forces
    monkeypatch.setattr(_dynamics, "bias_forces", lambda model, q, qd, gravity: held(model, q, None, gravity))
    planted = consistency(TREES[name])
    find(cases.states(TREES[name], SPEED[name], ACCELERATION[name]),
         lambda state: consistency_misfits(name, planted, state)[1] > ROUNDINGS * EPS, settings=CONTROL)


# Inverse kinematics -------------------------------------------------------------------------------------------------

#: The frame each tree's solve places: the planar arm's last link, the AR4's end effector, the branched tree's tip;
#: and how far from the target's joint positions a solve starts (rad, or m on a prismatic joint).
IK_FRAME = {"planar": "link_3", "ar4": "ee_link", "branched": "tip"}
SEED_OFFSET = {"planar": 0.3, "ar4": 0.2, "branched": 0.02}
#: Damped least squares converges linearly, at a rate set by the damping against the Jacobian's smallest singular
#: value, and stalls at a singular target (where its error is second order in the joints): the targets are drawn where
#: the Jacobian of the joints that move the frame has a condition number under ``WELL_CONDITIONED``, and solved with
#: damping 2 % of the reach, to within a thousand roundings of the reach.
WELL_CONDITIONED, IK_ITERATIONS, IK_TOLERANCE = 100.0, 200, 1024 * EPS


def solver(name: str):
    tree, frame, reach = TREES[name], IK_FRAME[name], REACH[name]
    return jax.jit(lambda target, q0: inverse_kinematics(
        tree, frame, target, q0, position_tolerance=IK_TOLERANCE * reach, rotation_tolerance=IK_TOLERANCE,
        iterations=IK_ITERATIONS, damping=0.02 * reach))


SOLVE = {name: solver(name) for name in TREES}
FRAME_JACOBIAN = {name: jax.jit(lambda q, tree=tree, frame=IK_FRAME[name]: frame_jacobian(tree, q, frame))
                  for name, tree in TREES.items()}


def condition(name: str, q: np.ndarray) -> float:
    jacobian = np.asarray(FRAME_JACOBIAN[name](jnp.asarray(q)))
    values = np.linalg.svd(jacobian[:, np.any(jacobian != 0, axis=0)], compute_uv=False)
    return float(values[0] / values[-1]) if values[-1] > 0 else math.inf


#: The basin a solve is claimed to converge from: a seed within ``SEED_OFFSET`` of the target's own joint positions on
#: every joint, with the Jacobian well conditioned (under ``WELL_CONDITIONED``) at each of ``BASIN_POINTS`` points
#: spaced along the straight path from the seed to those positions, ends included. Damped least squares is a local
#: method: from a seed outside it (one 0.025 rad from the AR4's wrist singularity, where the iterate's condition number
#: reached 6e5) it crawls, and is still millimetres away after 1000 iterations.
BASIN_POINTS = 5


def in_basin(name: str, q: np.ndarray, seed: np.ndarray) -> bool:
    return all(condition(name, seed + t * (q - seed)) <= WELL_CONDITIONED for t in np.linspace(0.0, 1.0, BASIN_POINTS))


def reported_honestly(name: str, solved: Any, target: Any) -> bool:
    """Whether a solve's report is true of where it ended: the frame's distance and angle from the target, computed
    again from forward kinematics at its ``q``, are the errors it reports, to rounding, and it says ``reached`` only
    where both are within the tolerances asked for."""
    tree, reach = TREES[name], REACH[name]
    ended = FK[name](solved.q)[tree.frame_names.index(IK_FRAME[name])]
    distance = float(np.linalg.norm(np.asarray(ended[:3] - target[:3])))
    angle = float(np.linalg.norm(np.asarray(rotation_vector(quaternion.multiply(
        jnp.asarray(target[3:]), quaternion.conjugate(ended[3:]))))))
    agrees = (abs(distance - float(solved.position_error)) <= ROUNDINGS * EPS * reach
              and abs(angle - float(solved.rotation_error)) <= ROUNDINGS * EPS)
    within = distance <= IK_TOLERANCE * reach and angle <= IK_TOLERANCE
    return agrees and (not bool(solved.reached) or within)


def draw_solve(name: str, data: Any, offset: float | None) -> tuple[np.ndarray, np.ndarray, Any]:
    """A target placed by forward kinematics at well-conditioned joint positions, and a seed: within ``offset`` of
    them on every joint (rad, or m on a prismatic joint), or with ``None`` anywhere within the joints' limits."""
    tree = TREES[name]
    q = data.draw(cases.positions(tree))
    assume(condition(name, q) <= WELL_CONDITIONED)
    if offset is None:
        seed = np.asarray(data.draw(cases.positions(tree)))
    else:
        seed = q + offset * np.asarray(data.draw(st.lists(st.floats(-1.0, 1.0), min_size=len(q), max_size=len(q))))
    target = FK[name](jnp.asarray(q))[tree.frame_names.index(IK_FRAME[name])]
    return q, seed, target


@each_tree
@PROPERTY
@given(data=st.data())
def test_from_any_seed_inverse_kinematics_reaches_the_target_or_says_it_did_not(name, data):
    """From a seed anywhere within the joints' limits, in its basin or not, the solve reports where it ended
    truthfully: never ``reached`` where forward kinematics puts the frame outside the tolerances."""
    _, seed, target = draw_solve(name, data, None)
    solved = SOLVE[name](target, jnp.asarray(seed))
    assert reported_honestly(name, solved, target), (bool(solved.reached), float(solved.position_error),
                                                     float(solved.rotation_error))


@each_tree
@PROPERTY
@given(data=st.data())
def test_from_a_seed_in_its_basin_inverse_kinematics_reaches_the_target(name, data):
    q, seed, target = draw_solve(name, data, SEED_OFFSET[name])
    assume(in_basin(name, q, seed))
    solved = SOLVE[name](target, jnp.asarray(seed))
    assert bool(solved.reached), (float(solved.position_error), float(solved.rotation_error))


def test_the_seed_ci_drew_outside_the_basin_is_outside_it_and_not_reported_reached():
    """The draw that failed the old single property about one CI run in nine (PR #21): its target is well conditioned
    (82), its seed is not (452), 0.025 rad from the wrist singularity. The basin excludes it, and the solve reports it
    not reached, truthfully."""
    q = np.array([0.0, 0.0, -1.359375, 0.0, -0.125, 0.0])
    seed = q + 0.2 * np.array([0.0, -1.0, 0.0, 1.0, 0.5, 0.0])
    target = FK["ar4"](jnp.asarray(q))[TREES["ar4"].frame_names.index("ee_link")]
    solved = SOLVE["ar4"](target, jnp.asarray(seed))
    assert condition("ar4", q) <= WELL_CONDITIONED and not in_basin("ar4", q, seed)
    assert not bool(solved.reached) and reported_honestly("ar4", solved, target)


def test_a_solve_that_claims_every_target_reached_fails_the_honesty_check_every_time():
    """The control: a solve whose report says ``reached`` whatever its errors is caught, on solves known not to reach
    their targets: CI's draw outside the basin (2.9 mm off after its iterations) and a target out of the planar arm's
    reach. Fixed, so that it fails every run: a search for a seed that does not converge found none in some CI runs."""
    q = np.array([0.0, 0.0, -1.359375, 0.0, -0.125, 0.0])
    target = FK["ar4"](jnp.asarray(q))[TREES["ar4"].frame_names.index("ee_link")]
    stalled = SOLVE["ar4"](target, jnp.asarray(q + 0.2 * np.array([0.0, -1.0, 0.0, 1.0, 0.5, 0.0])))
    beyond = jnp.asarray([3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    unreached = SOLVE["planar"](beyond, jnp.asarray([0.1, -0.1, 0.1]))
    for name, solved, aimed in (("ar4", stalled, target), ("planar", unreached, beyond)):
        assert reported_honestly(name, solved, aimed)
        assert not reported_honestly(name, solved._replace(reached=jnp.asarray(True)), aimed)


def test_a_solve_cut_to_two_iterations_fails_the_basin_property():
    """The control: two iterations do not converge from every seed in the basin, so the basin property catches a
    solver that stops early."""
    tree, reach = TREES["ar4"], REACH["ar4"]
    hurried = jax.jit(lambda target, q0: inverse_kinematics(
        tree, "ee_link", target, q0, position_tolerance=IK_TOLERANCE * reach, rotation_tolerance=IK_TOLERANCE,
        iterations=2, damping=0.02 * reach))

    def stalls(data: Any) -> bool:
        q, seed, target = draw_solve("ar4", data, SEED_OFFSET["ar4"])
        return in_basin("ar4", q, seed) and not bool(hurried(target, jnp.asarray(seed)).reached)

    find(st.data(), stalls, settings=CONTROL)


def test_a_target_out_of_reach_is_reported_not_reached():
    tree = TREES["planar"]
    target = jnp.asarray([3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    solved = SOLVE["planar"](target, jnp.asarray([0.1, -0.1, 0.1]))
    assert not bool(solved.reached)
    assert float(solved.position_error) >= 3.0 - sum(cases.PLANAR_LENGTHS[:2])  # link_3 sits at the end of link 2
    ended = np.asarray(FK["planar"](solved.q))[tree.frame_names.index("link_3")]
    assert float(np.linalg.norm(ended[:3] - np.asarray(target[:3]))) == pytest.approx(float(solved.position_error))


def test_a_target_reached_only_outside_the_limits_is_reported_outside_them():
    tree = TREES["ar4"]
    beyond = np.array([0.3, 1.9, -0.4, 0.5, 0.6, -0.2])  # joint_2's upper limit is pi/2
    assert condition("ar4", beyond) <= WELL_CONDITIONED
    target = FK["ar4"](jnp.asarray(beyond))[tree.frame_names.index("ee_link")]
    solved = SOLVE["ar4"](target, jnp.asarray(beyond + 0.05))
    assert bool(solved.reached) and not bool(solved.within_limits)
    assert float(solved.q[1]) > float(tree.articulation.upper[1])
    inside = SOLVE["ar4"](FK["ar4"](jnp.asarray(beyond - [0, 0.5, 0, 0, 0, 0]))[tree.frame_names.index("ee_link")],
                          jnp.asarray(beyond - [0, 0.45, 0, 0, 0, 0]))
    assert bool(inside.reached) and bool(inside.within_limits)
