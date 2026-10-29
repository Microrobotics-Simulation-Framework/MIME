"""Confinement by projection, as a property of the body's node: after every step the body lies within the tube's
lumen less its extent and margin, and wherever the step was projected back its velocity has no outward part. Drawn by
Hypothesis over states inside and outside the tube, loads, resistances and timesteps.

The control: the same node with no confinement fails the property, and Hypothesis finds where.
"""

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import assume, example, find, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mime.core import CYLINDER, quaternion
from mime.nodes.body import RigidInertialNode
from mime.nodes.body._rigid_inertial import CONTACT_ULPS
from mime.testing import FLOOR_ULPS

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


def steps(node: RigidInertialNode, example: dict):
    """Each step of the example for ``node``: its number, the state it starts from, the free step from there and the
    node's own."""
    q = exponential(jnp.asarray(example["rotation"]))
    state = {"pose": jnp.concatenate([jnp.asarray(example["position"]), q]),
             "twist": jnp.asarray(np.concatenate([example["velocity"], example["omega"]]))}
    given = {"applied_wrench": jnp.asarray(np.concatenate([example["force"], example["torque"]]))}
    for n in range(STEPS):
        free = FREE.update(state, given, example["dt"])
        after = node.update(state, given, example["dt"])
        yield n, state, free, after
        state = after


def judgements(node: RigidInertialNode, example: dict, planted: float = 0.0) -> list[tuple[int, float, float]]:
    """``(step, outward speed, allowed)`` for every step whose outward velocity the law fixes.

    A step that starts within the node's contact band of the shrunk surface is held inside the solve, along the normal
    *at the start of the step*, to close the gap ``g`` there within it: ``n . V' = max(g, 0) / dt``. The normal at the
    end is turned by the arc travelled over the surface's radius, so ``V' . n_end`` is ``|V'|`` times that turn, a
    second-order residue of the hold (``|V'|^2 dt / rho``: 9e-18 m/s on a tube of radius 1e-3 m, for 7 nm/s and
    0.2 ms) and not an outward motion; judged against the normal at the start it is what the law fixes, and a step that
    ends projected after the hold leaves ``V' . n_start`` inward by the same turn, which the bound allows. Any other
    step the free step decides: where it ends beyond the surface it is projected and the rule removes the outward part
    along the normal at its end. Rounding is on the scale of the cancellation, in the dtype of the state:
    ``FLOOR_ULPS`` units in the last place of the free step's speed, and for the gap those of the position's scale
    (``|x| + extent + margin``) over ``dt``.

    ``planted`` replaces each judged outward speed by ``(1 + planted)`` times what is allowed: the control's error.
    """
    inset, judged = EXTENT + MARGIN, []
    for n, before, free, after in steps(node, example):
        start, dt = before["pose"][:3], example["dt"]
        epsilon = float(jnp.finfo(start.dtype).eps)
        gap = -(float(distance(start)) + inset)
        position_rounding = epsilon * (float(np.linalg.norm(np.asarray(start))) + inset)
        rounding = FLOOR_ULPS * epsilon * float(np.linalg.norm(np.asarray(free["twist"][:3])))
        if abs(gap) <= CONTACT_ULPS * position_rounding:
            direction = np.asarray(normal_at(start))
            allowed = max(gap, 0.0) / dt + rounding + FLOOR_ULPS * position_rounding / dt
        elif float(distance(free["pose"][:3])) + inset > 0:
            direction, allowed = np.asarray(normal_at(after["pose"][:3])), rounding
        else:
            continue
        outward = float(np.dot(np.asarray(after["twist"][:3]), direction))
        judged.append((n, (1 + planted) * allowed if planted else outward, allowed))
    return judged


def violations(node: RigidInertialNode, example: dict) -> list[str]:
    """Every step at which the body ends outside the shrunk lumen, or moves outward past what the law lets through."""
    found = []
    for n, _, _, after in steps(node, example):
        radial = float(np.linalg.norm(np.asarray(after["pose"][1:3])))
        if radial > LIMIT * (1 + 1e-12):
            found.append(f"step {n}: radial offset {radial} beyond {LIMIT}")
    return found + [f"step {n}: moving outward at {outward} m/s, past the {allowed} m/s the law lets through"
                    for n, outward, allowed in judgements(node, example) if outward > allowed]


#: The draw CI's run 40812115177 failed on (Hypothesis 6.170.5): a body 3.9 mm out in a 1 mm lumen, pushed along the
#: tube by 1.4e-9 N. Step 0 projects it onto the shrunk surface; step 1 starts there, is held along the normal at its
#: start, and ends with the normal turned by 1.3e-9 rad, so 8.7e-18 m/s along the normal at its end.
HELD_ALONG_THE_TUBE = {
    "position": np.array([0.0, 0.00390625, 0.0002]), "rotation": np.zeros(3), "velocity": np.zeros(3),
    "omega": np.zeros(3), "force": np.array([0.0, 0.0, 1.40089049e-09]), "torque": np.zeros(3), "dt": 2e-4}


@settings(deadline=None)
@given(EXAMPLES)
@example(HELD_ALONG_THE_TUBE)
def test_after_every_step_the_body_lies_inside_and_moves_no_further_out(example):
    assert violations(CONFINED, example) == []


@settings(deadline=None)
@given(EXAMPLES)
def test_an_outward_speed_just_above_what_the_law_allows_is_found_at_every_judged_step(example):
    """The control: the judgement's resolution. Where an example is judged at a step with something allowed, an
    outward speed a tenth above it exceeds it, at every such step of every example; where nothing is allowed (a free
    step of no speed, on the surface, with no gap) there is no floor to plant above, and the example is not drawn."""
    judged = [(outward, allowed) for _, outward, allowed in judgements(CONFINED, example, planted=0.1) if allowed > 0]
    assume(judged)
    assert all(outward > allowed for outward, allowed in judged)
    assert violations(CONFINED, example) == [], "and the unplanted example is within the law"


def test_the_held_step_leaves_the_turn_of_the_normal_and_the_old_scale_misread_it():
    """The failing draw, step 1: along the normal at the start of the step the body moves no outward, and along the
    normal at its end it moves by the turn ``|V|^2 dt / rho`` that the arc over the shrunk radius gives, which is what
    a judgement against zero on the end normal read as outward motion."""
    arc_turn = None
    for n, before, free, after in steps(CONFINED, HELD_ALONG_THE_TUBE):
        if n == 1:
            speed, dt = float(np.linalg.norm(np.asarray(free["twist"][:3]))), HELD_ALONG_THE_TUBE["dt"]
            at_start = float(np.dot(np.asarray(after["twist"][:3]), np.asarray(normal_at(before["pose"][:3]))))
            at_end = float(np.dot(np.asarray(after["twist"][:3]), np.asarray(normal_at(after["pose"][:3]))))
            arc_turn = speed * speed * dt / float(np.linalg.norm(np.asarray(after["pose"][1:3])))
            assert abs(at_start) < 1e-3 * arc_turn
            assert at_end > 1e-12 * speed, "the old check's tolerance, 1e-12 of the free speed, would have flagged this"
            assert np.isclose(at_end, arc_turn, rtol=1e-3)
    assert arc_turn is not None


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
