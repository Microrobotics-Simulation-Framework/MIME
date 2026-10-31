"""Loads on a tree's links in the recursive Newton-Euler algorithm, against the Jacobian transpose (float64, per
module), over joint states inside the limits of the planar arm, the AR4 and the branched tree (``multibody_cases.py``)
and wrenches drawn on every named frame.

- **The oracle**: inverse dynamics with the loads ``W_k`` on the frames (``frame_forces``) is inverse dynamics without
  them less ``sum_k J_k^T W_k``, ``J_k`` the geometric Jacobian of frame ``k``'s origin, ``W_k`` its force and its
  torque about that origin: the principle of virtual work, computed by another route (the 0.2 node mapped its loads
  this way).
- **The control**: the loads gathered without moving each torque from its frame's origin to the root's (the
  ``o x f`` term dropped) fail on every draw: a frame's origin is never at the root's on these trees, and the drawn
  force is held to a size where the term it drops stands above the comparison's rounding.

The comparison allows ``ROUNDINGS`` float64 roundings of the larger side's scale: both sides sum a few dozen products of
the same terms in other orders.
"""

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import multibody_cases as cases
from mime.core.multibody import frame_forces, frame_jacobian, inverse_dynamics
from mime.core.multibody._articulation import ROOT

EPS = cases.EPS
ROUNDINGS = 64
PROPERTY = settings(deadline=None, max_examples=100)
TREES = {name: factory() for name, (factory, *_) in cases.TREES.items()}
SPEED = {name: speed for name, (_, speed, _, _) in cases.TREES.items()}
#: A gravity with a component along every axis, as the multibody oracles use.
GRAVITY = 9.81 * np.array([0.3, -0.5, -0.8]) / np.linalg.norm([0.3, -0.5, -0.8])
#: The size of a drawn force (N) and torque (N m): a millimetre robot's magnet pulls with micronewtons and a motor
#: turns with millinewton metres, and an arm carries newtons; the oracle is linear in the load, so one scale serves.
LOAD = 1.0


def loads(tree):
    frames = len(tree.frame_names)
    component = st.floats(-LOAD, LOAD, allow_nan=False).map(lambda x: 0.0 if abs(x) < 1e-3 * LOAD else x)
    return arrays(np.float64, (frames, 6), elements=component)


def compiled(tree):
    def without(q, qd):
        return inverse_dynamics(tree, q, qd, None, jnp.asarray(GRAVITY))

    def loaded(q, qd, w):
        return inverse_dynamics(tree, q, qd, None, jnp.asarray(GRAVITY), frame_forces(tree, q, w))

    def transposed(q, w):
        return sum(frame_jacobian(tree, q, f).T @ w[k] for k, f in enumerate(tree.frame_names))

    return jax.jit(without), jax.jit(loaded), jax.jit(transposed)


COMPILED = {name: compiled(tree) for name, tree in TREES.items()}


def misfit(name, q, qd, w, gathered=None) -> tuple[float, float]:
    """The largest difference between the two routes, and the comparison's bound."""
    without, loaded, transposed = COMPILED[name]
    by_rnea = np.asarray(without(q, qd)) - np.asarray(loaded(q, qd, w) if gathered is None else gathered(q, qd, w))
    by_jacobian = np.asarray(transposed(q, w))
    scale = max(float(np.max(np.abs(by_jacobian))), float(np.max(np.abs(np.asarray(without(q, qd))))), 1e-300)
    return float(np.max(np.abs(by_rnea - by_jacobian))), ROUNDINGS * EPS * scale


def joint_states(name):
    tree = TREES[name]
    return st.tuples(cases.positions(tree), arrays(np.float64, (tree.articulation.dofs,),
                                                   elements=st.floats(-SPEED[name], SPEED[name], allow_nan=False)),
                     loads(tree))


@PROPERTY
@given(st.sampled_from(sorted(TREES)).flatmap(lambda name: st.tuples(st.just(name), joint_states(name))))
def test_loads_on_the_frames_enter_inverse_dynamics_as_the_jacobian_transpose_maps_them(drawn):
    name, (q, qd, w) = drawn
    found, bound = misfit(name, q, qd, w)
    assert found <= bound, f"{name}: {found:.3g} > {bound:.3g}"


def unmoved(tree):
    """``frame_forces`` with each torque left about its frame's origin, as if every origin were the root's."""
    articulation = tree.articulation

    def gathered(q, qd, w):
        onto = np.zeros((articulation.dofs, len(tree.frame_names)))
        for k, body in enumerate(articulation.frame_bodies):
            if body != ROOT:
                onto[body, k] = 1.0
        return inverse_dynamics(tree, q, qd, None, jnp.asarray(GRAVITY), jnp.asarray(onto) @ jnp.asarray(w))

    return jax.jit(gathered)


UNMOVED = {name: unmoved(tree) for name, tree in TREES.items()}


@PROPERTY
@given(st.sampled_from(sorted(TREES)).flatmap(lambda name: st.tuples(st.just(name), joint_states(name))))
def test_control_the_loads_gathered_without_moving_their_torques_to_the_root_fail_on_every_draw(drawn):
    name, (q, qd, w) = drawn
    w = np.asarray(w).copy()
    w[:, :3] = np.where(np.abs(w[:, :3]) < 0.1 * LOAD, 0.1 * LOAD, w[:, :3])
    found, bound = misfit(name, q, qd, w, UNMOVED[name])
    assert found > bound, f"{name}: {found:.3g} <= {bound:.3g}"
