"""Confinement by projection, as a property of the body's node: after every step the body lies within the tube's
lumen less its extent and margin, and wherever the step was projected back its velocity has no outward part. Drawn by
Hypothesis over states inside and outside the tube, loads, resistances and timesteps.

The control: the same node with no confinement fails the property, and Hypothesis finds where.
"""

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import find, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mime.core import CYLINDER, quaternion
from mime.nodes.body import RigidInertialNode

RADIUS, EXTENT, MARGIN = 3.175e-3, 2.0748e-3, 1e-4
SURFACE = {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "radius": RADIUS}
LIMIT = RADIUS - EXTENT - MARGIN
STEPS = 3


def body(confined: bool) -> RigidInertialNode:
    confinement = dict(surface_shape="cylinder", extent=EXTENT, margin=MARGIN, velocity_rule="remove_outward",
                       **{f"surface_{k}": v for k, v in SURFACE.items()}) if confined else {}
    return RigidInertialNode("probe", 5e-4, mass=8.5e-5, inertia=(1.0e-10, 1.2e-10, 0.8e-10),
                             initial_pose=(0, 0, 0, 1, 0, 0, 0), initial_twist=(0.0,) * 6, **confinement)


CONFINED, FREE = body(True), body(False)
#: The cylinder's closed forms and the rotation's exponential, compiled once so that each example costs one call.
distance = jax.jit(lambda point: CYLINDER.signed_distance(SURFACE, point))
normal_at = jax.jit(lambda point: CYLINDER.outward_normal(SURFACE, point))
exponential = jax.jit(quaternion.exponential)


def vectors(scale: float, size: int = 3):
    """Components up to ``scale``, each zero or above a millionth of it: physical magnitudes, never within reach of
    float64's subnormals, where XLA flushes to zero and a residue of 1e-305 m/s would read as moving outward."""
    component = st.floats(-scale, scale, allow_nan=False).map(lambda x: 0.0 if abs(x) < 1e-6 * scale else x)
    return arrays(np.float64, (size,), elements=component)


EXAMPLES = st.fixed_dictionaries({
    "position": vectors(5e-3), "rotation": vectors(3.0), "velocity": vectors(0.5), "omega": vectors(50.0),
    "force": vectors(1e-3), "torque": vectors(1e-8), "dt": st.floats(1e-4, 1e-2)})


def violations(node: RigidInertialNode, example: dict) -> list[str]:
    """Every step at which the body ends outside the shrunk lumen, or moves outward where it was projected."""
    q = exponential(jnp.asarray(example["rotation"]))
    state = {"pose": jnp.concatenate([jnp.asarray(example["position"]), q]),
             "twist": jnp.asarray(np.concatenate([example["velocity"], example["omega"]]))}
    given = {"applied_wrench": jnp.asarray(np.concatenate([example["force"], example["torque"]]))}
    found = []
    for n in range(STEPS):
        free = FREE.update(state, given, example["dt"])
        state = node.update(state, given, example["dt"])
        position, velocity = np.asarray(state["pose"][:3]), np.asarray(state["twist"][:3])
        radial = float(np.linalg.norm(position[1:]))
        if radial > LIMIT * (1 + 1e-12):
            found.append(f"step {n}: radial offset {radial} beyond {LIMIT}")
        if float(distance(free["pose"][:3])) + EXTENT + MARGIN > 0:
            normal = np.asarray(normal_at(jnp.asarray(position)))
            if float(np.dot(velocity, normal)) > 1e-12 * float(np.linalg.norm(free["twist"][:3])):
                found.append(f"step {n}: projected, and moving outward at {float(np.dot(velocity, normal))} m/s")
    return found


@settings(deadline=None)
@given(EXAMPLES)
def test_after_every_step_the_body_lies_inside_and_moves_no_further_out(example):
    assert violations(CONFINED, example) == []


def test_without_the_projection_hypothesis_finds_a_body_outside():
    counterexample = find(EXAMPLES, lambda example: bool(violations(FREE, example)),
                          settings=settings(deadline=None, database=None))
    print(f"\n{violations(FREE, counterexample)}")
    assert violations(FREE, counterexample)


def bordered(system, normal):
    """The bordered matrix of the constrained step: ``A dV + lambda N = F - R V`` with ``N . (V + dV)`` given."""
    held = np.zeros((7, 7))
    held[:6, :6], held[:6, 6], held[6, :3] = system, np.concatenate([normal, np.zeros(3)]), normal
    return held


def test_on_the_surface_moving_outward_the_hold_is_the_bordered_solve():
    """rigid_inertial 0.2.0's hold (4c3, declared change 2), float64: for a body on its shrunk surface whose free step
    moves it outward, against a resistance whose translation couples to its rotation, ``held`` gives the velocity of
    the bordered system that holds its normal velocity at zero, ``N . V' = 0``, with a multiplier pushing inward; a
    body a tenth of a millimetre inside is not held. The control: with no coupling, the hold is remove_outward."""
    held_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        constants, dt = CONFINED.constants(), 5e-4
        rng = np.random.default_rng(7)
        a = rng.normal(size=(6, 6)) * np.array([1e-3] * 3 + [1e-7] * 3)[:, None]
        mass = np.diag([8.5e-5] * 3 + [1.0e-10, 1.2e-10, 0.8e-10])
        twist = np.array([0.002, 0.0005, -0.003, 40.0, 1.0, -2.0])
        load = np.array([0.0, 0.0, -8.3e-4, 0.0, 0.0, 0.0])
        position, normal = np.array([0.004, 0.0, -LIMIT]), np.array([0.0, 0.0, -1.0])
        direction = np.concatenate([normal, np.zeros(3)])
        for resistance, coupled in ((a @ a.T + np.diag([1e-3] * 3 + [1e-9] * 3), True),
                                    (np.diag([2e-3] * 3 + [1e-9] * 3), False)):
            system = mass / dt + resistance
            free = twist + np.linalg.solve(system, load - resistance @ twist)
            assert free[:3] @ normal > 0, "the free step moves the body outward"
            found = np.asarray(CONFINED.held(constants, jnp.asarray(position), jnp.asarray(system),
                                             jnp.asarray(free), dt))
            bordered = np.zeros((7, 7))
            bordered[:6, :6], bordered[:6, 6], bordered[6, :6] = system, direction, direction
            solved = np.linalg.solve(bordered, np.concatenate([system @ (free - twist), [-(direction @ twist)]]))
            expected = twist + solved[:6]
            assert solved[6] > 0, "the reaction -lambda N pushes the body inward"
            assert np.allclose(found, expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max())
            assert abs(float(found[:3] @ normal)) <= 1e-12 * float(np.abs(free[:3]).max())
            if not coupled:
                rule = free.copy()
                rule[:3] -= float(free[:3] @ normal) * normal
                assert np.allclose(found, rule, rtol=1e-12, atol=1e-15)
            else:
                assert not np.allclose(found[3:], free[3:], rtol=1e-6), "through the coupling the hold turns the body"
            inside = np.asarray(CONFINED.held(constants, jnp.asarray(position + [0.0, 0.0, 1e-4]),
                                              jnp.asarray(system), jnp.asarray(free), dt))
            assert np.array_equal(inside, free)
    finally:
        jax.config.update("jax_enable_x64", held_mode)
