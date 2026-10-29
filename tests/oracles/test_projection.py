"""Confinement by projection, as a property of the body's node: after every step the body lies within the tube's
lumen less its extent and margin, and wherever the step was projected back its velocity has no outward part. Drawn by
Hypothesis over states inside and outside the tube, loads, resistances and timesteps.

The control: the same node with no confinement fails the property, and Hypothesis finds where.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, find, given, settings
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


@functools.cache
def hold_of(node: RigidInertialNode):
    """``(pose, twist, free twist, dt) -> twist`` through the node's own ``held`` (no resistance is fed, so its system is
    ``M / dt``): the velocity after the hold in the solve, before the position is advanced and projected. Compiled once
    per node, so that each example costs one call."""
    constants = node.constants()

    def hold(pose, twist, free_twist, dt):
        rotation = quaternion.rotation_matrix(pose[3:])
        inertia = rotation @ jnp.diag(jnp.asarray(constants["inertia"], dtype=twist.dtype)) @ rotation.T
        mass_matrix = jnp.zeros((6, 6), dtype=twist.dtype)
        mass_matrix = mass_matrix.at[:3, :3].set(constants["mass"] * jnp.eye(3, dtype=twist.dtype))
        mass_matrix = mass_matrix.at[3:, 3:].set(inertia)
        return node.held(constants, pose[:3], mass_matrix / dt, free_twist, dt)

    return jax.jit(hold)


def judgements(node: RigidInertialNode, example: dict) -> list[tuple[int, str, float, float]]:
    """``(step, what, outward speed, allowed)`` for every step whose outward velocity the law fixes.

    A step that starts within the node's contact band of the shrunk surface is held inside the solve, along the normal
    *at the start of the step*, to close the gap ``g`` there within it: ``n . V' = max(g, 0) / dt``. That is judged on
    the hold's own output, which the rest of the step washes out of the final state (the projection and the rule remove
    the outward part along the normal at its end whatever the hold did), and on the final state along the same normal.
    The normal at the end is turned by the arc travelled over the surface's radius, so ``V' . n_end`` is ``|V'|``
    times that turn, a second-order residue of the hold (``|V'|^2 dt / rho``: 9e-18 m/s on a tube of radius 1e-3 m, for
    7 nm/s and 0.2 ms) and not an outward motion; a step that ends projected after the hold leaves ``V' . n_start``
    inward by the same turn, which the bound allows. Any other step the free step decides: where it ends beyond the
    surface it is projected and the rule removes the outward part along the normal at its end. Rounding is on the scale
    of the cancellation, in the dtype of the state: ``FLOOR_ULPS`` units in the last place of the free step's speed,
    and for the gap those of the position's scale (``|x| + extent + margin``) over ``dt``. ``FLOOR_ULPS`` is the
    suite's declared 64, not fitted to this law: the residue sits within 2 units, so the margin is wide and costs the
    plants below nothing (they are 1e-9 m, 1e-6 of the outward part, and the whole hold).
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
            held = np.asarray(hold_of(node)(before["pose"], before["twist"], free["twist"], dt)[:3])
            judged.append((n, "held", float(np.dot(held, direction)), allowed))
            judged.append((n, "after the hold", float(np.dot(np.asarray(after["twist"][:3]), direction)), allowed))
        elif float(distance(free["pose"][:3])) + inset > 0:
            direction = np.asarray(normal_at(after["pose"][:3]))
            judged.append((n, "projected", float(np.dot(np.asarray(after["twist"][:3]), direction)), rounding))
    return judged


def violations(node: RigidInertialNode, example: dict) -> list[str]:
    """Every step at which the body ends outside the shrunk lumen, or moves outward past what the law lets through."""
    found = []
    for n, _, _, after in steps(node, example):
        radial = float(np.linalg.norm(np.asarray(after["pose"][1:3])))
        if radial > LIMIT * (1 + 1e-12):
            found.append(f"step {n}: radial offset {radial} beyond {LIMIT}")
    return found + [f"step {n}: {what}, moving outward at {outward} m/s, past the {allowed} m/s the law lets through"
                    for n, what, outward, allowed in judgements(node, example) if outward > allowed]


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


class Planted(RigidInertialNode):
    """The shipped node with one fault planted, for the controls: each overrides a method of a subclass, so the node's
    own behaviour is untouched."""

    def __init__(self, fault: str) -> None:
        super().__init__("probe", 5e-4, mass=8.5e-5, inertia=(1.0e-10, 1.2e-10, 0.8e-10),
                         initial_pose=(0, 0, 0, 1, 0, 0, 0), initial_twist=(0.0,) * 6, surface_shape="cylinder",
                         extent=EXTENT, margin=MARGIN, velocity_rule="remove_outward",
                         **{f"surface_{k}": v for k, v in SURFACE.items()})
        self.fault = fault

    def held(self, p, position, system, moved, dt):
        if self.fault == "no hold":
            return moved
        if self.fault == "hold at the end of the step":
            # the same hold, read from where the free step ends: its band, gap and normal are the end's
            return super().held(p, position + dt * moved[:3], system, moved, dt)
        return super().held(p, position, system, moved, dt)

    def _confined(self, p, position, velocity):
        projected, ruled = super()._confined(p, position, velocity)
        was_projected = jnp.any(projected != position)
        if self.fault == "left 1e-9 m outside":
            away = projected - jnp.asarray(SURFACE["centre"])
            radial = away.at[0].set(0.0)
            outward = radial / jnp.linalg.norm(radial)
            return jnp.where(was_projected, projected + 1e-9 * outward, projected), ruled
        if self.fault == "rule disabled":
            return projected, velocity
        if self.fault == "1e-6 of the outward part kept":
            return projected, jnp.where(was_projected, ruled + 1e-6 * (velocity - ruled), ruled)
        return projected, ruled


FAULTS = ("left 1e-9 m outside", "rule disabled", "1e-6 of the outward part kept", "no hold",
          "hold at the end of the step")


@pytest.mark.parametrize("fault", FAULTS)
def test_a_fault_planted_in_the_node_is_found(fault):
    """The controls, in the node and not in the comparator (a scalar planted in the check cannot fail): a body left a
    nanometre outside, the rule off, a millionth of the outward part kept, the hold removed, and the hold taken at
    the end of the step instead of its start. The last two are invisible in the final state, which the projection and
    the rule rewrite; the judgement of the hold's own output sees them."""
    planted = Planted(fault)
    counterexample = find(EXAMPLES, lambda example: bool(violations(planted, example)),
                          settings=settings(deadline=None, database=None))
    assert violations(planted, counterexample)


def test_the_unplanted_subclass_is_the_node():
    """The control's own control: ``Planted`` with no fault is within the law on the failing draw and agrees with the
    node's step."""
    same = Planted("none")
    assert violations(same, HELD_ALONG_THE_TUBE) == []
    for (_, _, _, a), (_, _, _, b) in zip(steps(same, HELD_ALONG_THE_TUBE), steps(CONFINED, HELD_ALONG_THE_TUBE)):
        assert np.array_equal(np.asarray(a["twist"]), np.asarray(b["twist"]))


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
