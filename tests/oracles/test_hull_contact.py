"""The containment contact 0.2.0, which holds a body's hull points (``Contact.Projection``; the body's node,
``RigidInertialNode``'s hull contact), against its oracles, each with a control that fails it on every run:

- **Never past the wall.** A capped cylinder and FL-9's screw in script A's tube, drawn turned within what the bore
  admits and placed anywhere it admits, moving, turning and driven by a force and a torque at a millimetre body's
  scales, for four steps: no hull point ends past the wall, beyond the rounding of a position, nor past the wall
  shrunk by the margin by more than the sweeps leave of a wedged body (``HELD_WITHIN``). The control: today's
  contact, 0.1.0's, which holds the centre by the hull's radial reach, on the same draws, lets the hull through the
  wall.
- **Two-point statics.** A long cylinder turned across the tube until its two hull points farthest across its axis,
  one at each end, touch the shrunk wall, under a torque about the axis across both: each point's force is the couple's,
  ``tau / (2 (x cos(a) - y sin(a)))`` for a point at ``(x, y)`` in the body's frame turned by ``a``, equal and opposite,
  no other point takes one, and the twist after the step is still; and each force's derivative in the torque, by
  forward-mode differentiation through the step, is the couple's slope. The control: the node with the moment of each point's row dropped (the
  outward velocity taken away without the rotation coupled in) cannot hold the torque.
- **No energy gained.** Over drawn states at the wall, the contact's solve never leaves more kinetic energy in the
  step's norm than the free step's, and its forces do no positive work. The control: a solve that asks each point
  to move away by its gap in the step gains energy.
- **Unchanged where nothing touches.** Where no hull point can reach the shrunk wall in the step, the step is the
  free body's to the last bit (``np.array_equal``). The control: a contact that keeps a force of one part in 1e15.
- **The frame.** Turning the tube and the body's state together turns the step, off the ties of two of a group's
  points, where the step jumps by the choice of point (``contact.point``).
- **Admissibility only.** A contact that adds lubrication beside a drag that includes it is refused
  (``lubrication.twice``); either alone builds.

The draws are in float64, the node's dtype here, at physical scales: positions inside the tube, speeds up to 2 cm/s
and turning up to 50 rad/s about the body's axis and 10 rad/s across it, forces up to 1 mN and torques up to 10 nN m
on a body of 85 ug (CLAUDE.md's property-test causes: draw what a body can have)."""

import math
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hull_cases import FL9_SETTINGS, MARGIN, TUBE_RADIUS, hull_node, kind_points, rotation
from hypothesis import assume, find, given, settings
from hypothesis import strategies as st

import mime.nodes.body._rigid_inertial as rigid
from mime.build import BuildRefused
from mime.core import quaternion
from mime.interactions import Contact
from mime.nodes.body import RigidInertialNode
from mime.scene import Cylinder, Screw
from mime.units import dimensionless, mm

MASS, INERTIA = 8.5e-5, (1.3e-10, 4.6e-10, 4.6e-10)
ROD = (Cylinder(radius=mm(1.0), length=mm(6.0), axis=(1.0, 0.0, 0.0)),
       {"radius": 1e-3, "length": 6e-3, "axis": (1.0, 0.0, 0.0)})
SCREW = (Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
               starts=2), FL9_SETTINGS)
#: The hull's points at a coarse spacing, so that a draw's four steps are cheap: 280 of the rod's, 957 of FL-9's.
SPACING = 4e-4
KINDS = {"rod": ROD, "screw": SCREW}
#: The statics' rod, long and thin, so that turned across the tube until both ends touch, its two points' rows are
#: coupled at 0.1 and the sweeps converge to rounding (a rod pressed in at a steep tilt, its levers short, couples them
#: more closely: the property's rod, 1 mm by 6 mm, at 0.89, takes 0.8% less than its couple after the solve's sweeps).
LONG_ROD = (Cylinder(radius=mm(0.5), length=mm(8.0), axis=(1.0, 0.0, 0.0)),
            {"radius": 5e-4, "length": 8e-3, "axis": (1.0, 0.0, 0.0)})
ALL_KINDS = {**KINDS, "long_rod": LONG_ROD}
STEPS = 4
#: A position's rounding, in units in the last place of the tube's radius, which the gaps are computed from.
POSITION_ULPS = 64


@cache
def points(name: str) -> np.ndarray:
    kind, held = ALL_KINDS[name]
    return kind_points(kind, held, SPACING)


@cache
def held(name: str) -> RigidInertialNode:
    kind, settings_ = ALL_KINDS[name]
    return hull_node(kind, settings_, SPACING, mass=MASS, inertia=INERTIA)


@cache
def centre_held(name: str) -> RigidInertialNode:
    """Today's contact, 0.1.0's: the centre held by the hull's reach across its own axis (its extent) plus the
    margin."""
    kind, settings_ = KINDS[name]
    return RigidInertialNode("umr", 5e-4, mass=MASS, inertia=INERTIA, initial_pose=(0, 0, 0, 1, 0, 0, 0),
                             initial_twist=(0.0,) * 6, surface_shape="cylinder", margin=MARGIN,
                             extent=float(np.hypot(points(name)[:, 1], points(name)[:, 2]).max()),
                             velocity_rule="remove_outward",
                             surface_centre=(0.0, 0.0, 0.0), surface_axis=(1.0, 0.0, 0.0), surface_radius=TUBE_RADIUS)


def gaps(name: str, pose) -> np.ndarray:
    """Every hull point's gap to the wall shrunk by the margin at ``pose`` (negative: past it)."""
    placed = np.asarray(pose[:3], dtype=np.float64) + points(name) @ rotation(np.asarray(pose[3:])).T
    return TUBE_RADIUS - MARGIN - np.hypot(placed[:, 1], placed[:, 2])


#: The body itself, read from outside the contact: its kind's quadrature at a sixtieth of its enclosing radius (on
#: FL-9 35 um apart, its farthest point within about 0.2 um of a sample; the rod's 53 um, within about 0.35 um), far
#: finer than the contact's points. The body's own deepest point lies at most that much past the dense set's.
DENSE_SHARE = 1.0 / 60.0


@cache
def dense(name: str) -> np.ndarray:
    kind, held_ = ALL_KINDS[name]
    return kind_points(kind, held_, DENSE_SHARE * kind.enclosing_radius(held_))


def body_gap(name: str, pose) -> float:
    """The body's least gap to the wall shrunk by the margin at ``pose``, from its dense quadrature."""
    placed = np.asarray(pose[:3], dtype=np.float64) + dense(name) @ rotation(np.asarray(pose[3:])).T
    return float(TUBE_RADIUS - MARGIN - np.hypot(placed[:, 1], placed[:, 2]).max())


def about_axis(angle: float, axis) -> np.ndarray:
    """The unit quaternion of a turn by ``angle`` about the unit ``axis``."""
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * np.asarray(axis, dtype=np.float64)])


def product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The Hamilton product ``a b``: the turn ``b``, then ``a``."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def turned(spin: float, tilt: float, about: float) -> np.ndarray:
    """The body spun about its own axis (x), then tilted by ``tilt`` about the direction across the tube at the angle
    ``about`` round it."""
    return product(about_axis(tilt, (0.0, math.cos(about), math.sin(about))), about_axis(spin, (1.0, 0.0, 0.0)))


def bisect(fits, high: float) -> float:
    """The largest value in ``[0, high]`` that ``fits`` (``fits(0)`` holds), by bisection."""
    low = 0.0
    if fits(high):
        return high
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if fits(middle) else (low, middle)
    return low


def admitted(name: str, draw: dict) -> np.ndarray:
    """The drawn state as a pose the bore admits: the tilt taken down to the largest the bore admits with the centre on
    the axis where the drawn one does not fit, and the centre at the drawn share of the room along the drawn
    direction."""
    def fits_tilt(tilt: float) -> bool:
        return gaps(name, np.concatenate([np.zeros(3), turned(draw["spin"], tilt, draw["about"])])).min() > 0

    orientation = turned(draw["spin"], bisect(fits_tilt, draw["tilt"]) * draw.get("fit", 0.999), draw["about"])
    direction = np.array([0.0, math.cos(draw["towards"]), math.sin(draw["towards"])])
    room = bisect(lambda r: gaps(name, np.concatenate([r * direction, orientation])).min() > 0, TUBE_RADIUS)
    return np.concatenate([draw["share"] * room * direction, orientation])


def component(scale: float):
    return st.floats(-scale, scale, allow_nan=False).map(lambda x: 0.0 if abs(x) < 1e-6 * scale else x)


def draws(kinds=tuple(sorted(KINDS))):
    return st.fixed_dictionaries({
        "kind": st.sampled_from(kinds), "spin": st.floats(0.0, 2 * math.pi), "tilt": st.floats(0.0, 1.5),
        "about": st.floats(0.0, 2 * math.pi), "towards": st.floats(0.0, 2 * math.pi), "share": st.floats(0.0, 1.0),
        "velocity": st.tuples(*[component(0.02)] * 3), "spinning": component(50.0),
        "turning": st.tuples(component(10.0), component(10.0)), "force": st.tuples(*[component(1e-3)] * 3),
        "torque": st.tuples(*[component(1e-8)] * 3), "resisted": st.booleans(), "dt": st.floats(1e-4, 1e-3)})


DRAWS = draws()
#: A control's draws, the rod's alone: one node compiled, and every control fails on it.
ROD_DRAWS = draws(("rod",))


def start(draw: dict, node: RigidInertialNode) -> tuple[dict, dict]:
    """The drawn state and inputs for ``node``: the turning about the body's own axis turned with it."""
    pose = admitted(draw["kind"], draw)
    spin = rotation(pose[3:]) @ np.array([draw["spinning"], *draw["turning"]])
    state = {**node.initial_state(), "pose": jnp.asarray(pose),
             "twist": jnp.asarray(np.concatenate([np.asarray(draw["velocity"]), spin]))}
    resistance = np.diag([1e-2] * 3 + [3e-11] * 3) if draw["resisted"] else np.zeros((6, 6))
    inputs = {"applied_wrench": jnp.asarray(np.concatenate([draw["force"], draw["torque"]])),
              "resistance": jnp.asarray(resistance), "background_load": jnp.zeros(6)}
    return state, inputs


def deepest(draw: dict, node: RigidInertialNode) -> tuple[float, float]:
    """The least gap of the body (its dense quadrature, ``body_gap``) to the shrunk wall over the draw's steps, and
    the rounding it is judged on."""
    state, inputs = start(draw, node)
    least = math.inf
    for _ in range(STEPS):
        state = node.update(state, inputs, draw["dt"])
        least = min(least, body_gap(draw["kind"], np.asarray(state["pose"])))
    return least, POSITION_ULPS * float(np.finfo(np.float64).eps) * TUBE_RADIUS


#: How far past the wall shrunk by the margin the body may end: what the fixed count of sweeps leaves of a wedged body
#: (7e-10 m on a rod turned into both walls, its two rows coupled at 0.82; phase 7o's pull request), with the dense
#: quadrature's own shortfall from the body (``DENSE_SHARE``, at most 0.35 um), a declaration a hundredth of the
#: margin. The contact holds the body's own surface, so its body ends no further past; one held by its samples at the
#: draws' spacing (0.4 mm) ends up to their shortfall past it, 71 um on FL-9 (the control). The wall itself is never
#: reached beyond a position's rounding.
HELD_WITHIN = 1e-6


@settings(deadline=None)
@given(DRAWS)
def test_no_hull_point_ends_past_the_wall_and_the_margin_is_kept_to_the_sweeps_convergence(draw):
    least, rounding = deepest(draw, held(draw["kind"]))
    assert least + MARGIN >= -rounding, f"a hull point ended {-(least + MARGIN):.3g} m past the wall"
    assert least >= -HELD_WITHIN, f"a hull point ended {-least:.3g} m past the wall shrunk by the margin"


class Sampled(RigidInertialNode):
    """The control: the hull held by its samples, as for a kind with no support query (the contact before review
    7o's F2)."""

    @property
    def supported(self) -> bool:
        return False


@cache
def sampled(name: str) -> RigidInertialNode:
    reference = held(name)
    return Sampled("umr", 5e-4, **{key: value for key, value in reference.constants().items()
                                   if key not in ("name", "timestep")})


def test_control_a_hull_held_by_its_samples_lets_its_body_past_the_margin():
    """Held by its samples at the draws' spacing, the screw's body ends past the shrunk wall by up to the samples'
    shortfall from it, far beyond ``HELD_WITHIN``."""
    def passes(draw: dict) -> bool:
        least, _ = deepest(near_the_wall(draw), sampled(draw["kind"]))
        return least < -HELD_WITHIN

    assert passes(find(DRAWS.filter(lambda d: d["kind"] == "screw"), passes,
                       settings=settings(deadline=None, database=None)))


def test_control_todays_contact_holding_the_centre_lets_the_hull_through_the_wall():
    """0.1.0's contact holds the centre by the hull's reach across its axis: a turned body's ends pass the wall
    shrunk by the margin as soon as they move out from near it."""
    def passes(draw: dict) -> bool:
        least, rounding = deepest(near_the_wall(draw), centre_held(draw["kind"]))
        return least < -HELD_WITHIN or least + MARGIN < -rounding

    assert passes(find(ROD_DRAWS, passes, settings=settings(deadline=None, database=None)))


# -- two-point statics ---------------------------------------------------------------------------------------------

#: The torque about z that presses the rod's two ends into the wall.
TORQUE = 2e-9


def couple_place() -> tuple[float, float]:
    """The rod's rim point the couple holds at its +x end, in its frame: along its axis and across it (along its y);
    the other is its mirror at the -x end. The contact holds the rod's own rims (the cylinder's support), not a
    sample near them."""
    _, settings_ = LONG_ROD
    return settings_["length"] / 2.0, settings_["radius"]


def couple_turn() -> np.ndarray:
    """The rod tilted about z until its two rim points, ``(x, y)`` and ``(-x, -y)``, touch the shrunk wall."""
    return about_axis(couple_tilt(), (0.0, 0.0, 1.0))


def couple_tilt() -> float:
    """The tilt about z at which the two points touch the shrunk wall: ``x sin(a) + y cos(a) = R - margin``."""
    x, y = couple_place()
    return math.asin((TUBE_RADIUS - MARGIN) / math.hypot(x, y)) - math.atan2(y, x)


def couple_force() -> float:
    """Each point's force in the couple that holds the torque: ``tau / (2 (x cos(a) - y sin(a)))``, its lever the
    point's distance along the tube from the rod's centre."""
    x, y = couple_place()
    tilt = couple_tilt()
    return TORQUE / (2 * (x * math.cos(tilt) - y * math.sin(tilt)))


class Uncoupled(RigidInertialNode):
    """The control: each point's row with its moment dropped, the outward velocity taken away along the normal alone."""

    def contacts(self, p, body, pose):
        found = super().contacts(p, body, pose)
        return found._replace(rows=found.rows.at[:, 3:].set(0.0))


def couple_step(node: RigidInertialNode) -> dict:
    pose = np.concatenate([np.zeros(3), couple_turn()])
    state = {**node.initial_state(), "pose": jnp.asarray(pose), "twist": jnp.zeros(6)}
    wrench = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, TORQUE])
    return node.update(state, {"applied_wrench": wrench, "resistance": jnp.zeros((6, 6)),
                               "background_load": jnp.zeros(6)}, 5e-4)


def couple_problems(node: RigidInertialNode) -> list[str]:
    """Where the step of the rod pressed against both walls differs from the couple's closed form: each of the two
    rim points' force (``couple_force``), held at the rim itself, no other point's, and a still twist."""
    expected = couple_force()
    out = couple_step(node)
    forces, held_points = np.asarray(out["contact_forces"]), np.asarray(out["contact_points"])
    turn = rotation(couple_turn())
    x, y = couple_place()
    found = []
    for rim in (np.array([x, y, 0.0]), np.array([-x, -y, 0.0])):
        at = np.flatnonzero(np.all(np.isclose(held_points, turn @ rim, rtol=0, atol=1e-12), axis=1) & (forces > 0))
        taken = float(forces[at].sum()) if len(at) else 0.0
        if not len(at) or abs(taken - expected) > 1e-9 * expected:
            found.append(f"the rim at {turn @ rim} took {taken if len(at) else 'no force'} N, against "
                         f"{expected:.12g} N")
    rims = [turn @ np.array([x, y, 0.0]), turn @ np.array([-x, -y, 0.0])]
    others = [k for k in np.flatnonzero(forces > 0)
              if not any(np.allclose(held_points[k], rim, rtol=0, atol=1e-12) for rim in rims)]
    if others:
        found.append(f"{len(others)} points off the rims took a force")
    twist = np.asarray(out["twist"])
    if np.max(np.abs(twist[3:])) > 1e-9 * TORQUE * 5e-4 / INERTIA[2]:
        found.append(f"the rod still turns at {twist[3:]} rad/s")
    return found


def test_a_rod_pressed_against_both_walls_takes_the_couple_at_its_rims():
    assert couple_problems(held("long_rod")) == []


@cache
def uncoupled() -> RigidInertialNode:
    reference = held("long_rod")
    return Uncoupled("umr", 5e-4, **{key: value for key, value in reference.constants().items()
                                     if key not in ("name", "timestep")})


def test_control_without_its_rotation_coupled_in_the_contact_cannot_hold_the_couple():
    assert couple_problems(uncoupled()) != []


def couple_slope_problems(node: RigidInertialNode) -> list[str]:
    """Where the derivative of each rim's force in the torque (its rows' together: two groups may hold one rim), by
    forward-mode differentiation through the step, differs from the couple's ``1 / (2 (x cos(a) - y sin(a)))``: the
    contact is differentiable where its active set holds."""
    pose = jnp.asarray(np.concatenate([np.zeros(3), couple_turn()]))
    state = {**node.initial_state(), "pose": pose, "twist": jnp.zeros(6)}

    def forces(torque):
        wrench = jnp.zeros(6).at[5].set(torque)
        return node.update(state, {"applied_wrench": wrench, "resistance": jnp.zeros((6, 6)),
                                   "background_load": jnp.zeros(6)}, 5e-4)["contact_forces"]

    slope = np.asarray(jax.jacfwd(forces)(jnp.asarray(TORQUE)))
    held_points = np.asarray(node.update(state, {"applied_wrench": jnp.zeros(6).at[5].set(TORQUE),
                                                 "resistance": jnp.zeros((6, 6)),
                                                 "background_load": jnp.zeros(6)}, 5e-4)["contact_points"])
    expected = couple_force() / TORQUE
    turn = rotation(couple_turn())
    x, y = couple_place()
    found = []
    taken = np.zeros(len(slope), dtype=bool)
    for rim in (np.array([x, y, 0.0]), np.array([-x, -y, 0.0])):
        at = np.all(np.isclose(held_points, turn @ rim, rtol=0, atol=1e-12), axis=1)
        taken |= at
        value = float(slope[at].sum())
        if abs(value - expected) > 1e-9 * expected:
            found.append(f"the rim at {turn @ rim}'s force moves by {value:.12g} N per N m, against {expected:.12g}")
    if np.any(np.abs(slope[~taken]) > 1e-9 * expected):
        found.append("a point off the rims' force moves with the torque")
    return found


def test_the_couples_forces_move_with_the_torque_as_its_closed_form():
    assert couple_slope_problems(held("long_rod")) == []


def test_control_without_its_rotation_coupled_in_the_couples_forces_do_not():
    assert couple_slope_problems(uncoupled()) != []


# -- no energy gained, nothing changed where nothing touches, the frame ------------------------------------------

def solve(node: RigidInertialNode, state: dict, inputs: dict, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                                    np.ndarray]:
    """The step's system ``A``, the free twist, the held twist and the forces' power over the step: the node's own
    solve, before the pose is advanced, compiled once per node."""
    found = solver(node)(state["pose"], state["twist"], inputs, dt)
    return tuple(np.asarray(value) for value in found)  # type: ignore[return-value]


@cache
def solver(node: RigidInertialNode):
    return jax.jit(lambda pose, twist, inputs, dt: held_solve(node, pose, twist, inputs, dt))


def held_solve(node: RigidInertialNode, pose, twist, inputs: dict, dt):
    """The node's own step system and free twist (``RigidInertialNode.stepped``), its held twist, and the forces' power
    at the held twist."""
    p = node._step_constants(None)
    _, system, free = node.stepped(p, pose, twist, inputs, dt)
    body = node.hull_points(p, twist.dtype)
    moved, _, forces, _ = node.hull_held(p, body, pose, system, free, dt)
    rows = node.contacts(p, body, pose).rows
    return system, free, moved, forces @ (rows @ moved)


def near_the_wall(draw: dict) -> dict:
    return {**draw, "share": 0.98 + 0.02 * draw["share"]}


class Pushing(RigidInertialNode):
    """The control: each point is asked to move away by its gap in the step, not to close it by at most its gap."""

    def hull_held(self, p, body, pose, system, moved, dt):
        found = self.contacts(p, body, pose)
        response = jnp.linalg.solve(system, found.rows.T)
        forces = self.sweeps(found.rows @ response, found.rows @ moved - jnp.abs(found.gaps) / dt, rigid.HULL_SWEEPS,
                             owner="contact.impulse")
        return moved + response @ forces, found.points, forces, jnp.zeros(())


def energy_problems(node: RigidInertialNode, draw: dict) -> list[str]:
    draw = near_the_wall(draw)
    state, inputs = start(draw, node)
    system, free, moved, power = solve(node, state, inputs, draw["dt"])
    before, after = 0.5 * free @ system @ free, 0.5 * moved @ system @ moved
    rounding = POSITION_ULPS * float(np.finfo(np.float64).eps)
    found = []
    if after > before * (1 + rounding):
        found.append(f"the held step's energy {after:.6g} J/s exceeds the free step's {before:.6g}")
    if power > rounding * before:
        found.append(f"the contact's forces do work at {power:.3g} W")
    return found


@cache
def pushing(name: str) -> RigidInertialNode:
    reference = held(name)
    return Pushing("umr", 5e-4, **{key: value for key, value in reference.constants().items()
                                   if key not in ("name", "timestep")})


@settings(deadline=None)
@given(DRAWS)
def test_the_contact_never_leaves_the_step_more_energy_than_the_free_step(draw):
    assert energy_problems(held(draw["kind"]), draw) == []


def test_control_a_contact_that_pushes_points_away_gains_energy():
    found = find(ROD_DRAWS, lambda draw: bool(energy_problems(pushing(draw["kind"]), draw)),
                 settings=settings(deadline=None, database=None))
    assert energy_problems(pushing(found["kind"]), found)


@cache
def free_twin(name: str) -> RigidInertialNode:
    reference = held(name)
    own = {key: value for key, value in reference.constants().items() if key not in ("name", "timestep")}
    return RigidInertialNode("umr", 5e-4, **{key: value for key, value in own.items() if not key.startswith(
        ("surface_", "hull_"))})


class Lingering(RigidInertialNode):
    """The control: a contact that keeps a force of one part in 1e15 of the body's weight's order at every point."""

    def sweeps(self, coupling, start_, count, *, owner):
        return super().sweeps(coupling, start_, count, owner=owner) + 1e-15 * MASS


@cache
def lingering(name: str) -> RigidInertialNode:
    reference = held(name)
    return Lingering("umr", 5e-4, **{key: value for key, value in reference.constants().items()
                                     if key not in ("name", "timestep")})


def far_from_the_wall(draw: dict) -> dict:
    """The draw with no hull point able to reach the shrunk wall in the step: its centre near the axis, its speeds a
    hundredth of the draw's, and its tilt at most 0.3 of what the bore admits."""
    return {**draw, "share": 0.1 * draw["share"], "fit": 0.3, "velocity": tuple(
        0.01 * v for v in draw["velocity"]), "spinning": 0.01 * draw["spinning"],
        "turning": tuple(0.01 * v for v in draw["turning"])}


def changed(node: RigidInertialNode, draw: dict) -> list[str]:
    draw = far_from_the_wall(draw)
    state, inputs = start(draw, node)
    free_state = {"pose": state["pose"], "twist": state["twist"]}
    out, free = node.update(state, inputs, draw["dt"]), free_twin(draw["kind"]).update(free_state, inputs, draw["dt"])
    assert gaps(draw["kind"], np.asarray(out["pose"])).min() > 1e-5, "the draw reached the wall"
    return [field for field in ("pose", "twist") if not np.array_equal(np.asarray(out[field]),
                                                                       np.asarray(free[field]))]


@settings(deadline=None)
@given(DRAWS)
def test_where_no_point_reaches_the_wall_the_step_is_the_free_bodys_to_the_last_bit(draw):
    assert changed(held(draw["kind"]), draw) == []


def test_control_a_contact_that_keeps_a_vanishing_force_is_seen():
    found = find(ROD_DRAWS, lambda draw: bool(changed(lingering(draw["kind"]), draw)),
                 settings=settings(deadline=None, database=None))
    assert changed(lingering(found["kind"]), found)


#: Two of a group's points whose gaps lie within this of each other are a tie of ``contact.point``, where the step
#: jumps (its declared order is -1) and a turned world's rounding may choose the other: far above that rounding (about
#: 1e-17 m here), far below any gap a step reads.
TIE = 1e-12


def tied(name: str, node: RigidInertialNode, pose) -> bool:
    """Whether any group's two points nearest the shrunk wall at ``pose`` tie (``TIE``): a body lying along the wall,
    touching it along a line, or one placed symmetrically to its samples, as a draw of exact zeros places it."""
    found = gaps(name, pose)
    for row in node.groups:
        members = np.sort(found[row[row >= 0]])
        if len(members) > 1 and members[1] - members[0] < TIE:
            return True
    return False


@settings(deadline=None)
@given(DRAWS, st.tuples(st.floats(-math.pi, math.pi), st.floats(-math.pi, math.pi), st.floats(-math.pi, math.pi)))
def test_turning_the_tube_and_the_body_together_turns_the_step(draw, world):
    """Judged off the contact's choice of each group's point: where two of a group's points tie the step jumps
    (``contact.point``, a declared jump), and the turned world's rounding may choose the other, which is the switch
    and not the frame."""
    draw = near_the_wall(draw)
    node = held(draw["kind"])
    state, inputs = start(draw, node)
    assume(not tied(draw["kind"], node, np.asarray(state["pose"])))
    assume(not tied(draw["kind"], node, np.asarray(node.update(state, inputs, draw["dt"])["pose"])))
    turn = quaternion.exponential(jnp.asarray(world))
    matrix = np.asarray(quaternion.rotation_matrix(turn))
    pose, twist = np.asarray(state["pose"]), np.asarray(state["twist"])
    wrench = np.asarray(inputs["applied_wrench"])
    other = node
    moved = {**other.initial_state(),
             "pose": jnp.asarray(np.concatenate([matrix @ pose[:3], np.asarray(quaternion.multiply(turn, pose[3:]))])),
             "twist": jnp.asarray(np.concatenate([matrix @ twist[:3], matrix @ twist[3:]]))}
    block = np.kron(np.eye(2), matrix)
    other_inputs = {**inputs, "applied_wrench": jnp.asarray(block @ wrench),
                    "resistance": jnp.asarray(block @ np.asarray(inputs["resistance"]) @ block.T)}
    turned_axis = {"surface_axis": jnp.asarray(matrix[:, 0])}
    out, there = node.update(state, inputs, draw["dt"]), node.update(moved, other_inputs, draw["dt"],
                                                                      params=turned_axis)
    speed = float(np.max(np.abs(np.asarray(out["twist"])))) + 1e-6
    assert np.allclose(matrix @ np.asarray(out["pose"])[:3], np.asarray(there["pose"])[:3], rtol=0, atol=1e-12)
    assert np.allclose(block @ np.asarray(out["twist"]), np.asarray(there["twist"]), rtol=0, atol=1e-9 * speed)
    forces = np.asarray(out["contact_forces"])
    assert np.allclose(forces, np.asarray(there["contact_forces"]), rtol=1e-6,
                       atol=1e-9 * float(np.max(np.abs(forces), initial=0.0)))


# -- admissibility only --------------------------------------------------------------------------------------------

def screw_in_the_tube(lubricated: bool):
    from hull_cases import hull_contact

    from mime.data import asset
    from mime.interactions import Drag, Lubrication

    experiment = hull_contact()
    experiment.attach(Drag.Surrogate(weights=asset("dejongh_mlp_v2"),
                                     lubrication=Lubrication(epsilon=mm(0.15)) if lubricated else None),
                      body="umr", medium="water")
    experiment.declare("water.viscosity", __import__("mime.units", fromlist=["mPa_s"]).mPa_s(1.0), reason="water")
    return experiment


def test_a_contact_never_adds_lubrication_beside_a_drag_that_includes_it():
    assert screw_in_the_tube(lubricated=True).plan() is not None


def test_a_contact_that_adds_lubrication_beside_a_drag_that_includes_it_is_refused(monkeypatch):
    monkeypatch.setattr(Contact, "includes_lubrication", lambda self: True)
    with pytest.raises(BuildRefused) as refused:
        screw_in_the_tube(lubricated=True).plan()
    assert "lubrication.twice" in [r.code for r in refused.value.refusals]


def test_control_a_contact_that_adds_lubrication_alone_builds(monkeypatch):
    monkeypatch.setattr(Contact, "includes_lubrication", lambda self: True)
    assert screw_in_the_tube(lubricated=False).plan() is not None



# -- the plan's refusals -------------------------------------------------------------------------------------------

def refused_codes(experiment) -> list[str]:
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    return [r.code for r in refused.value.refusals]


def test_a_spacing_for_a_body_with_no_hull_is_refused():
    """The contact rig's body has no parts, so no hull: it is held as a sphere of its extent, and a spacing written for
    its hull's points is refused (``confinement.no-hull``); the control, the same contact with no spacing, plans."""
    from mime.testing._mime_families import contact_rig

    assert "confinement.no-hull" in refused_codes(contact_rig(Contact.Projection(margin=mm(0.1), spacing=mm(0.2))))
    assert contact_rig(Contact.Projection(margin=mm(0.1))).plan() is not None


def test_a_surface_whose_shape_takes_one_point_at_a_time_is_refused(monkeypatch):
    """A hull's points are held against the surface many at once: a shape that takes one point at a time is refused
    (``confinement.surface-shape``), here the cylinder with its declaration withdrawn; the control, the cylinder as
    it is, plans."""
    from hull_cases import hull_contact

    from mime.core import CYLINDER

    assert hull_contact().plan() is not None
    monkeypatch.setattr(type(CYLINDER), "takes_many", False)
    assert "confinement.surface-shape" in refused_codes(hull_contact())


def test_a_dynamics_that_holds_no_hull_is_refused(monkeypatch):
    """A body's dynamics that projects a confinement but cannot hold a hull's points is refused
    (``confinement.hull-not-held``), here rigid_inertial with its flag withdrawn; the control, as it is, plans."""
    from hull_cases import hull_contact

    from mime.dynamics import RigidInertial
    from mime.vocabulary import Capability

    assert hull_contact().plan() is not None
    monkeypatch.setattr(RigidInertial, "flags", RigidInertial.flags - {Capability.HOLDS_HULL})
    assert "confinement.hull-not-held" in refused_codes(hull_contact())
