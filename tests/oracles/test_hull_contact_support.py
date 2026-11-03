"""The containment contact holds a body's own surface (review 7o's F2, option B; F5 to F7), against its oracles, each
with a control that fails on every run (nothing here is drawn but the cut's property):

- **No friction from the hull** (F2). A smooth cylinder of FL-9's crests' radius and length, isotropic so that its
  gyroscopic term is exactly zero, spun at 50 Hz about its axis on the tube's floor under its weight, keeps its spin to
  rounding over a second: a frictionless contact at its touching point takes none. The control: the same body held by
  its samples (a kind with no support) loses most of it.
- **At rest it stays at rest** (F6). A sphere, a cylinder and FL-9 set on the floor under their weight, their own
  lowest point a nanometre above the shrunk wall: over a second the energy (kinetic and the weight's potential) never
  rises above an earlier value by more than rounding, and the body does not wander sideways. The control: FL-9 held by
  its samples rises and wanders.
- **No energy from the projection's turn** (review 7o's G1). A rod with its axial moment the smallest, yawed across
  the floor and spun at 50 Hz about its axis, never stands above its starting energy by more than the weight's potential
  over a micrometre; and over drawn states near the wall the projection after the step leaves the kinetic energy as
  the solve left it, to rounding: its turn carries the angular velocity with it. The control: the angular velocity left
  in the world (the contact 0.2.0's first rule) pumps energy into the rod and moves the kinetic energy.
- **No positive power** (F7). A long rod with a light transverse inertia, its two rows coupled positively (+0.85 and
  +0.98), pressed into both walls: the forces' power at the held twist is never positive, however far the sweeps got.
  The control: the sweeps' forces unscaled give positive power. **The residual**: a fat cylinder (2.6 by 3.5 mm)
  wedged at its tilt limit, its rims' rows strongly coupled, is left above ``CONVERGED`` by five sweeps and raises the
  flag (its twenty converge it, since the contact 0.3.0's searches do); the long rod, converged, does not; and script
  A's arm, every step held, stays under it by half. The control: a flag set below that run rises on it.
- **The convex cut** (F1). Each group's dropped points lie inside the group's kept hull, and a step chosen from the
  cut candidates is the step from all of them, bit for bit (Hypothesis, near the wall). The control: a cut that drops
  a hull vertex changes a step.
- **The spacing's guard** (F5). A spacing whose samples fall farther from the body than the margin is refused,
  naming one that keeps within it. The control: the case's spacing plans. A margin too small to say how finely to seed
  the body takes the floor, and a spacing that takes more samples to judge than the plan holds is refused before any
  is made.
- **The flags**: a hull raises ``touching`` and ``unconverged``; one whose kind has a support also raises
  ``support_short``, and one whose kind has none (a box) ``held_by_samples``.
- **A held point never shallower than its seed** (review 7o's G4). On the round-trip case, FL-9 held from its first
  step, no point the support's steps reach stands shallower than the sample they started from. The control: a search
  that ends shallower raises ``support_short``; and so does one that does not converge, holding its group's nearest
  sample, by the samples' shortfall (MIME-ANO-135)."""

import math
import re
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hull_cases import MARGIN, TUBE_RADIUS, hull_node, kind_points, rotation
from hypothesis import find, given, settings
from test_hull_contact import DRAWS, KINDS, ROD_DRAWS, SPACING, SPACINGS, about_axis, held, near_the_wall, start

import mime.build._plan as plan_module
import mime.interactions.contact as contact_law
import mime.nodes.body._contact as contact_module
import mime.nodes.body._rigid_inertial as rigid_module
from mime import Experiment
from mime.build import BuildRefused
from mime.core import quaternion
from mime.interactions import Contact
from mime.nodes.body import RigidInertialMidpointNode, RigidInertialNode, convex_candidates, point_groups
from mime.nodes.body._contact import CONVERGED
from mime.scene import Box, Cylinder, Sphere
from mime.units import mm, s

MASS, WEIGHT = 8.5e-5, 8.5e-5 * 9.81
DT = 5e-4
#: FL-9's crests' radius and length as a smooth cylinder, the review's control body of revolution.
SMOOTH = (Cylinder(radius=mm(2.0748), length=mm(7.47), axis=(1.0, 0.0, 0.0)),
          {"radius": 2.0748e-3, "length": 7.47e-3, "axis": (1.0, 0.0, 0.0)})
BALL = (Sphere(radius=mm(1.0)), {"radius": 1e-3})
#: The contact's default spacing, the margin.
DEFAULT = 1e-4


class Sampled(RigidInertialNode):
    """The control: the hull held by its samples, as for a kind with no support query."""

    @property
    def supported(self) -> bool:
        return False


class SampledMidpoint(RigidInertialMidpointNode):
    @property
    def supported(self) -> bool:
        return False


def node_of(body, *, inertia, cls=RigidInertialMidpointNode):
    kind, settings_ = body
    reference = hull_node(kind, settings_, DEFAULT, mass=MASS, inertia=inertia)
    return cls("umr", DT, **{key: value for key, value in reference.constants().items()
                             if key not in ("name", "timestep")})


def lowest(body) -> float:
    """How far the body reaches straight down from its centre, aligned: its own support's."""
    kind, settings_ = body
    points = kind_points(kind, settings_, DEFAULT)
    down = np.array([0.0, 0.0, -1.0])
    parameters = {k: (jnp.asarray(v) if isinstance(v, float) else v) for k, v in settings_.items()}
    found = kind.support(parameters, jnp.asarray(down), jnp.asarray(points[np.argmax(points @ down)]))
    return float(np.asarray(found.point) @ down)


def on_the_floor(node, body, spin: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """The body set on the floor under its weight, its own lowest point a nanometre above the shrunk wall, spinning
    at ``spin`` about its axis: its poses and twists over ``steps`` steps."""
    depth = TUBE_RADIUS - MARGIN - lowest(body) - 1e-9
    state = {**node.initial_state(), "pose": jnp.asarray([0.0, 0.0, -depth, 1.0, 0.0, 0.0, 0.0]),
             "twist": jnp.asarray([0.0, 0.0, 0.0, spin, 0.0, 0.0])}
    inputs = {"applied_wrench": jnp.asarray([0.0, 0.0, -WEIGHT, 0.0, 0.0, 0.0]), "resistance": jnp.zeros((6, 6)),
              "background_load": jnp.zeros(6)}

    def scan(held_):
        return jax.lax.scan(lambda s, _: (node.update(s, inputs, DT), (s["pose"], s["twist"])), held_, None,
                            length=steps)[1]

    poses, twists = jax.jit(scan)(state)
    return np.asarray(poses), np.asarray(twists)


# -- no friction from the hull ---------------------------------------------------------------------------------------

SPIN, SPIN_STEPS = 2 * math.pi * 50, 2000
ISOTROPIC = (1.3e-10,) * 3


@cache
def spun(sampled: bool) -> np.ndarray:
    node = node_of(SMOOTH, inertia=ISOTROPIC, cls=SampledMidpoint if sampled else RigidInertialMidpointNode)
    return on_the_floor(node, SMOOTH, SPIN, SPIN_STEPS)[1][:, 3]


def test_a_smooth_cylinder_spun_on_the_floor_keeps_its_spin():
    spin = spun(sampled=False)
    print(f"\nspin from {spin[0]:.12g} to {spin[-1]:.12g} rad/s over {SPIN_STEPS * DT:g} s")
    assert np.max(np.abs(spin - SPIN)) <= 1e-9 * SPIN


def test_control_held_by_its_samples_the_spun_cylinder_loses_its_spin():
    spin = spun(sampled=True)
    print(f"\nheld by its samples, spin from {spin[0]:.6g} to {spin[-1]:.6g} rad/s")
    assert spin[-1] < 0.95 * SPIN


# -- at rest it stays at rest ------------------------------------------------------------------------------------

REST_STEPS = 2000
#: The bodies at rest: their inertias (the cylinder's and FL-9's, axial the smallest; the ball's isotropic).
RESTING = {"ball": (BALL, (3.4e-11,) * 3), "cylinder": (SMOOTH, (1.3e-10, 4.6e-10, 4.6e-10)),
           "fl9": (KINDS["screw"], (1.3e-10, 4.6e-10, 4.6e-10))}


def energy(poses: np.ndarray, twists: np.ndarray, inertia) -> np.ndarray:
    found = []
    for pose, twist in zip(poses, twists, strict=True):
        turn = rotation(pose[3:])
        moment = turn @ np.diag(inertia) @ turn.T
        found.append(0.5 * MASS * twist[:3] @ twist[:3] + 0.5 * twist[3:] @ moment @ twist[3:] + WEIGHT * pose[2])
    return np.asarray(found)


def rest_problems(name: str, sampled: bool) -> list[str]:
    body, inertia = RESTING[name]
    node = node_of(body, inertia=inertia, cls=SampledMidpoint if sampled else RigidInertialMidpointNode)
    poses, twists = on_the_floor(node, body, 0.0, REST_STEPS)
    held_ = energy(poses, twists, inertia)
    rise = float(np.max(held_ - np.minimum.accumulate(held_)))
    rounding = 1e-6 * WEIGHT * 1e-6  # the weight's potential over a micrometre's millionth: far above float64's
    wander = float(np.ptp(poses[:, 1]))
    found = []
    if rise > rounding:
        found.append(f"{name}: its energy rose {rise:.3g} J above an earlier value")
    if wander > 1e-9:
        found.append(f"{name}: it wandered {wander:.3g} m sideways")
    return found


@pytest.mark.parametrize("name", sorted(RESTING))
def test_a_body_at_rest_on_the_floor_neither_gains_energy_nor_wanders(name):
    assert rest_problems(name, sampled=False) == []


def test_control_fl9_held_by_its_samples_rises_or_wanders():
    assert rest_problems("fl9", sampled=True) != []


# -- no energy from the projection's turn ----------------------------------------------------------------------------

class LeftInTheWorld(RigidInertialMidpointNode):
    """The control: the projection's turn leaves the angular velocity in the world, as the contact 0.2.0 first did
    (its D9), so a turned pose with unequal moments changes the rotational kinetic energy."""

    def carried(self, turn, twist):
        return twist


#: A rod, 1 mm by 6 mm, its axial moment the smallest, yawed across the floor and spun about its axis: the body review
#: 7o's G1 found gaining energy while the projection after the step turned its pose and left its angular velocity in the
#: world (1.2e-6 J above its start in 2 s at 50 Hz, 57 % of its spin energy, its axis tumbling to 60 degrees).
ROD = (Cylinder(radius=mm(1.0), length=mm(6.0), axis=(1.0, 0.0, 0.0)),
       {"radius": 1e-3, "length": 6e-3, "axis": (1.0, 0.0, 0.0)})
ROD_INERTIA = (0.5 * MASS * 1e-6, MASS * (3e-6 + 36e-6) / 12.0, MASS * (3e-6 + 36e-6) / 12.0)
ROD_YAW, ROD_SPIN, ROD_STEPS, ROD_SPACING = math.radians(2.0), 2 * math.pi * 50, 1000, 4e-4
#: How far the rod's energy (kinetic and the weight's potential) may stand above its start: the weight's potential over
#: a micrometre, 8.3e-10 J. Measured (phase 7o's second fix round, 0.5 s, float64): 1.3e-11 J spun at 50 Hz, 6.0e-11 J
#: unspun, both in the first 40 steps as the yawed rod settles on the curved floor, so not the spin's; with the
#: projection's turn leaving the angular velocity in the world it stands 5.0e-7 J above its start by 0.5 s.
ROD_RISE = WEIGHT * 1e-6


def rod_on_the_floor(cls) -> tuple[np.ndarray, np.ndarray]:
    """The rod yawed ``ROD_YAW`` about the vertical, its own deepest point (the contact's) a nanometre above the shrunk
    floor, spun at ``ROD_SPIN`` about its axis under its weight: its poses and twists over ``ROD_STEPS`` steps, the
    start included."""
    kind, settings_ = ROD
    reference = hull_node(kind, settings_, ROD_SPACING, mass=MASS, inertia=ROD_INERTIA)
    node = cls("umr", DT, **{key: value for key, value in reference.constants().items()
                             if key not in ("name", "timestep")})
    turn = about_axis(ROD_YAW, (0.0, 0.0, 1.0))
    p = node._step_constants(None)
    body = node.hull_points(p, jnp.float64)
    least = jax.jit(lambda depth: jnp.min(node.contacts(p, body, jnp.concatenate([
        jnp.stack([0.0, 0.0, -depth]), jnp.asarray(turn)])).gaps))
    low, high = 0.0, TUBE_RADIUS
    for _ in range(80):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if float(least(middle)) > 1e-9 else (low, middle)
    state = {**node.initial_state(), "pose": jnp.asarray([0.0, 0.0, -low, *turn]),
             "twist": jnp.asarray([0.0, 0.0, 0.0, *(rotation(turn) @ np.array([ROD_SPIN, 0.0, 0.0]))])}
    inputs = {"applied_wrench": jnp.asarray([0.0, 0.0, -WEIGHT, 0.0, 0.0, 0.0]), "resistance": jnp.zeros((6, 6)),
              "background_load": jnp.zeros(6)}

    def scan(held_):
        return jax.lax.scan(lambda s, _: (node.update(s, inputs, DT), (s["pose"], s["twist"])), held_, None,
                            length=ROD_STEPS)

    end, (poses, twists) = jax.jit(scan)(state)
    return (np.concatenate([np.asarray(poses), np.asarray(end["pose"])[None]]),
            np.concatenate([np.asarray(twists), np.asarray(end["twist"])[None]]))


@cache
def rod_rise(cls) -> float:
    poses, twists = rod_on_the_floor(cls)
    held_ = energy(poses, twists, ROD_INERTIA)
    return float(np.max(held_ - held_[0]))


def test_a_spinning_rod_with_unequal_moments_gains_no_energy_on_the_floor():
    rise = rod_rise(RigidInertialMidpointNode)
    print(f"\nthe spun rod's energy stood at most {rise:.3g} J above its start")
    assert rise <= ROD_RISE, f"its energy rose {rise:.3g} J above its start"


def test_control_the_angular_velocity_left_in_the_world_pumps_energy_into_it():
    assert rod_rise(LeftInTheWorld) > 100.0 * ROD_RISE


def projection_kinetic(node, pose, twist, inputs, dt):
    """The kinetic energy before the projection after the step (the pose the solve's twist reached) and after it
    (the projected pose with the twist it carried): the node's own step, taken apart as ``advance`` takes it."""
    p = node._step_constants(None)
    mass_matrix, system, free = node.stepped(p, pose, twist, inputs, dt)
    body = node.hull_points(p, twist.dtype)
    moved = node.hull_held(p, body, pose, system, free, dt)[0]
    orientation = quaternion.normalized(quaternion.multiply(quaternion.exponential(dt * moved[3:]), pose[3:]))
    reached = jnp.concatenate([pose[:3] + dt * moved[:3], orientation])
    ended, carried = node.hull_projected(p, body, reached, moved, mass_matrix)
    moments = jnp.asarray(p["inertia"], dtype=twist.dtype)

    def kinetic(at, velocity):
        spin = quaternion.rotate_back(at[3:], velocity[3:])
        return 0.5 * p["mass"] * velocity[:3] @ velocity[:3] + 0.5 * spin @ (moments * spin)

    return kinetic(reached, moved), kinetic(ended, carried)


@cache
def kinetic_step(node):
    return jax.jit(lambda pose, twist, inputs, dt: projection_kinetic(node, pose, twist, inputs, dt))


@cache
def left_in_the_world(name: str) -> RigidInertialNode:
    reference = held(name)
    return type("LeftInTheWorld", (type(reference),), {"carried": lambda self, turn, twist: twist})(
        "umr", 5e-4, **{key: value for key, value in reference.constants().items() if key not in ("name", "timestep")})


def kinetic_problems(node, draw: dict) -> list[str]:
    draw = near_the_wall(draw)
    state, inputs = start(draw, node)
    before, after = (float(e) for e in kinetic_step(node)(state["pose"], state["twist"], inputs, draw["dt"]))
    rounding = 64 * float(np.finfo(np.float64).eps)
    return [f"the projection moved the kinetic energy from {before:.17g} to {after:.17g} J"] \
        if abs(after - before) > rounding * before else []


@settings(deadline=None)
@given(DRAWS)
def test_the_projection_after_the_step_leaves_the_kinetic_energy_as_the_solve_left_it(draw):
    assert kinetic_problems(held(draw["kind"]), draw) == []


def test_control_left_in_the_world_the_angular_velocity_moves_it():
    found = find(ROD_DRAWS, lambda draw: bool(kinetic_problems(left_in_the_world(draw["kind"]), draw)),
                 settings=settings(deadline=None, database=None))
    assert kinetic_problems(left_in_the_world(found["kind"]), found)


# -- no positive power, and the residual -----------------------------------------------------------------------------

TORQUE = 2e-9


def wedged(radius_mm: float, length_mm: float, inertia, *, unscaled: bool = False, monkeypatch=None):
    """A cylinder centred on the tube's axis, turned about z until its hull touches the shrunk wall at both ends, under
    a torque about z that presses both in: the node's step's forces, its rows' power at the held twist and the
    residual."""
    kind = Cylinder(radius=mm(radius_mm), length=mm(length_mm), axis=(1.0, 0.0, 0.0))
    settings_ = {"radius": radius_mm * 1e-3, "length": length_mm * 1e-3, "axis": (1.0, 0.0, 0.0)}
    x, y = settings_["length"] / 2.0, settings_["radius"]
    tilt = math.asin((TUBE_RADIUS - MARGIN) / math.hypot(x, y)) - math.atan2(y, x)
    node = hull_node(kind, settings_, SPACING, mass=MASS, inertia=inertia)
    if unscaled:
        monkeypatch.setattr(rigid_module, "no_work", lambda coupling, start_, forces: forces)
    p = node._step_constants(None)
    pose = jnp.asarray(np.concatenate([np.zeros(3), about_axis(tilt, (0.0, 0.0, 1.0))]))
    inputs = {"applied_wrench": jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, TORQUE]), "resistance": jnp.zeros((6, 6)),
              "background_load": jnp.zeros(6)}
    _, system, free = node.stepped(p, pose, jnp.zeros(6), inputs, DT)
    body = node.hull_points(p, jnp.float64)
    moved, _, forces, left, _ = node.hull_held(p, body, pose, system, free, DT)
    rows = node.contacts(p, body, pose).rows
    rate = float(0.5 * free @ system @ free) / DT
    return np.asarray(forces), float(forces @ (rows @ moved)), float(left), rate


@pytest.mark.parametrize("share", [10.0, 100.0])
def test_positively_coupled_rows_do_no_positive_work(share):
    forces, power, _, rate = wedged(0.5, 8.0, (1.3e-10, 4.6e-10 / share, 4.6e-10 / share))
    assert np.sum(forces > 0) >= 2, "fewer than two points held: the check could not discriminate"
    assert power <= 1e-12 * rate, f"the forces' power is {power:.3g} W, {power / rate:.3g} of the step's energy rate"


def test_control_the_sweeps_forces_unscaled_do_positive_work(monkeypatch):
    _, power, _, rate = wedged(0.5, 8.0, (1.3e-10, 4.6e-12, 4.6e-12), unscaled=True, monkeypatch=monkeypatch)
    assert power > 1e-6 * rate


def test_a_wedged_body_is_left_unconverged_and_reports_it(monkeypatch):
    """The flag's upper side: a fat cylinder (2.6 by 3.5 mm) wedged at its tilt limit, eight rows sharing its rims,
    given five sweeps, which leave its solve unconverged (1.6e-2). With its twenty it read 6.7e-2 under the contact
    0.2.0, a figure of the searches that contact left unconverged (MIME-ANO-135); under 0.3.0 they converge it to
    7.9e-8, and of the wedges measured only the long rod with a hundredth of its transverse moments passes the
    threshold, by a third (6.5e-3)."""
    monkeypatch.setattr(rigid_module, "HULL_SWEEPS", 5)
    forces, _, left, _ = wedged(2.6, 3.5, (1.3e-10, 4.6e-10, 4.6e-10))
    assert np.sum(forces > 0) >= 2
    assert left > CONVERGED, f"the residual {left:.3g} is within {CONVERGED}: the flag would not rise"


def test_control_the_long_rod_converges_and_does_not_report_it():
    _, _, left, _ = wedged(0.5, 8.0, (1.3e-10, 4.6e-10, 4.6e-10))
    assert left <= CONVERGED


def script_as_arm_held_by_its_hull(monkeypatch):
    """Script A's reproduction arm with the contact 0.3.0 at its default spacing, 2 s of it."""
    import script_a

    monkeypatch.setattr(script_a, "CentreContact", Contact)
    return script_a.reproduction_arm().build().run(s(2.0), history=True)


def unconverged_raised(result) -> list[str]:
    return [read.path for read in result.flags if read.path.endswith("unconverged") and read.raised]


def test_a_healthy_run_stays_under_the_flag(monkeypatch):
    """The threshold's lower side: script A's arm, every step held, stays under it, by the margin it was set with, and
    its residual is no empty reading (its median is above the control's threshold below)."""
    result = script_as_arm_held_by_its_hull(monkeypatch)
    left = np.asarray(result.history["umr"]["contact_residual"], dtype=np.float64)
    assert np.median(left) > 1e-6 and left.max() < CONVERGED / 2.0
    assert unconverged_raised(result) == []
    assert [read.path for read in result.flags if read.path.endswith("support_short") and read.raised] == []


def test_control_a_flag_set_below_that_run_rises_on_it(monkeypatch):
    monkeypatch.setattr(contact_module, "CONVERGED", 1e-6)
    assert unconverged_raised(script_as_arm_held_by_its_hull(monkeypatch)) != []


# -- the convex cut --------------------------------------------------------------------------------------------------

def inside_problems(name: str, candidates) -> list[str]:
    """Each of the hull's points a group drops that lies outside the group's kept points' convex hull."""
    from scipy.spatial import ConvexHull

    kind, settings_ = KINDS[name]
    points = kind_points(kind, settings_, SPACINGS.get(name, SPACING))
    groups = point_groups(points, 8)
    found = []
    for row, kept in zip(groups, candidates, strict=True):
        members, held_ = row[row >= 0], kept[kept >= 0]
        dropped = np.setdiff1d(members, held_)
        if len(held_) < 4 or not len(dropped):
            continue
        hull = ConvexHull(points[held_])
        beyond = (points[dropped] @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1)
        if beyond.max() > 1e-12 * np.ptp(points):
            found.append(f"{name}: a dropped point lies {beyond.max():.3g} m outside its group's kept hull")
    return found


def halved(name: str) -> np.ndarray:
    """The control's cut: each group's kept hull vertices with every other one dropped."""
    kind, settings_ = KINDS[name]
    points = kind_points(kind, settings_, SPACINGS.get(name, SPACING))
    rows = convex_candidates(points, point_groups(points, 8))
    cut = np.full_like(rows, -1)
    for g, row in enumerate(rows):
        kept = row[row >= 0][::2]
        cut[g, :len(kept)] = kept
    return cut


@pytest.mark.parametrize("name", sorted(KINDS))
def test_every_point_a_group_drops_lies_inside_its_kept_hull(name):
    assert inside_problems(name, held(name).groups) == []


def test_control_a_cut_that_drops_hull_vertices_leaves_points_outside():
    assert inside_problems("screw", halved("screw")) != []


@cache
def uncut(name: str, rows: tuple | None = None) -> RigidInertialNode:
    kind, settings_ = KINDS[name]
    points = kind_points(kind, settings_, SPACINGS.get(name, SPACING))
    groups = point_groups(points, 8) if rows is None else np.asarray(rows)
    return hull_node(kind, settings_, SPACINGS.get(name, SPACING), mass=8.5e-5, inertia=(1.3e-10, 4.6e-10, 4.6e-10),
                     groups=groups)


def cut_changes(draw: dict, node: RigidInertialNode) -> list[str]:
    draw = near_the_wall(draw)
    state, inputs = start(draw, held(draw["kind"]))
    out, other = held(draw["kind"]).update(state, inputs, draw["dt"]), node.update(state, inputs, draw["dt"])
    return [field for field in ("pose", "twist") if not np.array_equal(np.asarray(out[field]),
                                                                       np.asarray(other[field]))]


@settings(deadline=None)
@given(DRAWS)
def test_the_cut_candidates_choose_the_step_all_the_points_choose_to_the_last_bit(draw):
    assert cut_changes(draw, uncut(draw["kind"])) == []


def test_control_a_cut_that_drops_hull_vertices_changes_a_step():
    planted = uncut("screw", tuple(map(tuple, halved("screw"))))
    found = find(DRAWS.filter(lambda d: d["kind"] == "screw"), lambda draw: bool(cut_changes(draw, planted)),
                 settings=settings(deadline=None, database=None))
    assert cut_changes(found, planted)


# -- the spacing's guard, and the flags ------------------------------------------------------------------------------

def test_a_spacing_whose_samples_fall_past_the_margin_is_refused_naming_one_within_it():
    from hull_cases import hull_contact

    with pytest.raises(BuildRefused) as refused:
        hull_contact(spacing=1.6e-3).plan()
    (found,) = [r for r in refused.value.refusals if r.code == "confinement.spacing-short"]
    print(f"\n{found.message}\n{found.action}")
    assert "spacing=mm(" in found.action


def test_control_the_cases_spacing_plans():
    from hull_cases import hull_contact

    assert hull_contact().plan() is not None


def default_contact(margin) -> Experiment:
    """The hull_contact case with its contact at ``margin`` and the spacing left to its default."""
    from hull_cases import hull_contact

    experiment = hull_contact()
    experiment._declarations = [d for d in experiment._declarations if not isinstance(getattr(d, "law", None), Contact)]
    experiment.attach(Contact.Projection(margin=margin), body="umr", surface="tube.inner_surface")
    return experiment


def samples_held(margin) -> int:
    (line,) = [line for lines in default_contact(margin).plan().checked.values() for line in lines
               if "at a spacing of" in line]
    return int(re.search(r"its ([\d,]+) points at a spacing", line)[1].replace(",", ""))


@pytest.mark.parametrize("margin_mm", [0.0, 1e-2])
def test_a_margin_too_small_to_say_how_finely_to_seed_the_body_takes_the_floor_and_plans(margin_mm):
    """A 64th of FL-9's size, 34,494 samples, whatever the margin below it: a thousandth of its size would take about
    ten million, and the plan's guard goes through them all (CI's runner lost its 16 GB to a negative margin's)."""
    assert samples_held(mm(margin_mm)) == 34_494


def test_a_negative_margin_is_refused_for_itself_with_no_samples_to_pay():
    with pytest.raises(BuildRefused) as refused:
        default_contact(mm(-0.1)).plan()
    assert "setting.value" in refused.value.codes()


def test_the_margins_own_spacing_stands_above_the_floor():
    assert samples_held(mm(0.1)) == 15_327


def test_a_spacing_that_takes_more_samples_to_judge_than_the_plan_holds_is_refused_before_any_is_made(monkeypatch):
    """The floor at the rule it replaced, a thousandth of the size: the refusal names a spacing, in a plan that makes
    none of the millions of samples (the shortfall would raise, were it reached)."""
    def reached(*_: object) -> float:
        raise AssertionError("the shortfall was judged from every sample")

    monkeypatch.setattr(contact_law, "SPACING_FLOOR", 1e-3)
    monkeypatch.setattr(plan_module, "samples_shortfall", reached)
    with pytest.raises(BuildRefused) as refused:
        default_contact(mm(0.0)).plan()
    (found,) = [r for r in refused.value.refusals if r.code == "confinement.spacing-short"]
    assert "write spacing=mm(" in found.action and "of them to judge" in found.message


def test_a_margin_no_spacing_can_reach_is_refused_within_what_the_plan_holds():
    """A nanometre's margin, the search for a finer spacing stopping where the plan's memory does, says so."""
    from hull_cases import hull_contact

    with pytest.raises(BuildRefused) as refused:
        hull_contact(margin=1e-9).plan()
    (found,) = [r for r in refused.value.refusals if r.code == "confinement.spacing-short"]
    assert "a larger margin=mm(...)" in found.action and "the finest the plan judges" in found.action


def flag_names(part=None, spacing: float = SPACING) -> set[str]:
    from hull_cases import hull_contact

    from mime.experiment._experiment import run_layout

    return {flag.name for flag in run_layout(hull_contact(part=part, spacing=spacing).plan()).flags if flag.name}


def test_a_hull_raises_touching_and_unconverged_and_a_box_held_by_its_samples_also_held_by_samples():
    held_ = flag_names()
    assert {"touching", "unconverged", "support_short"} <= held_ and "held_by_samples" not in held_
    boxed = flag_names(Box(size=mm((2.0, 1.0, 1.0))), 5e-5)
    assert {"touching", "unconverged", "held_by_samples"} <= boxed and "support_short" not in boxed


# -- a held point never shallower than its seed ----------------------------------------------------------------------

def held_short(monkeypatch=None, pulled: float = 0.0):
    """The round-trip case (FL-9 turned 12 degrees, held from its first step) over 20 steps: the largest
    ``contact_short`` and whether ``support_short`` rose. ``pulled``: the control's search, which ends at the seed
    pulled that share of its distance towards the body's centre, shallower than the seed."""
    from hull_cases import hull_contact

    if pulled:
        monkeypatch.setattr(contact_module, "deepest", lambda pair, seeds: seeds * (1.0 - pulled))
    result = hull_contact().build().run(s(0.01), history=True)
    held_ = np.asarray(result.history["umr"]["contact_short"])
    raised = [read.path for read in result.flags if read.path.endswith("support_short") and read.raised]
    return float(held_.max()), float(np.finfo(held_.dtype).eps) * TUBE_RADIUS, raised


def test_the_support_never_leaves_a_held_point_shallower_than_its_seed():
    short, rounding, raised = held_short()
    assert short <= 64 * rounding and raised == [], f"a held point stood {short:.3g} m shallower than its seed"


def test_control_a_search_that_ends_shallower_than_its_seed_raises_support_short(monkeypatch):
    short, _, raised = held_short(monkeypatch, pulled=5e-3)
    assert short > 1e-6 and raised == ["umr.support_short"]


def test_a_search_that_does_not_converge_holds_its_nearest_sample_and_raises_support_short(monkeypatch):
    """The round-trip case with the search given no steps, so that none converges: each group holds its nearest
    sample (``contact.search``), and where that sample is near being held the body may reach past it by up to the
    samples' shortfall, which ``contact_short`` reads and ``support_short`` raises. Beside the test above, whose
    search converges and raises nothing, the same case, this is the flag's two sides."""
    from hull_cases import FL9_SETTINGS
    from hull_cases import SPACING as CASE_SPACING
    from script_a import fl9

    from mime.scene._sampling import hull_shortfall, quadrature_counts

    monkeypatch.setattr(contact_module, "CONTACT_NEWTON", 0)
    short, _, raised = held_short()
    kind = fl9()
    patches = tuple(kind.patches)
    expected = hull_shortfall(kind, FL9_SETTINGS, patches, quadrature_counts(kind, FL9_SETTINGS, CASE_SPACING, patches))
    assert short >= expected and raised == ["umr.support_short"]
