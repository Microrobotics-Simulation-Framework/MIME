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
- **No positive power** (F7). A long rod with a light transverse inertia, its two rows coupled positively (+0.85 and
  +0.98), pressed into both walls: the forces' power at the held twist is never positive, however far the sweeps got.
  The control: the sweeps' forces unscaled give positive power. **The residual**: a fat cylinder (2.6 by 3.5 mm)
  wedged at its tilt limit, its rims' rows strongly coupled, is left above ``CONVERGED`` by the sweeps (6.7 %) and
  raises the flag; the long rod, converged, does not.
- **The convex cut** (F1). Each group's dropped points lie inside the group's kept hull, and a step chosen from the
  cut candidates is the step from all of them, bit for bit (Hypothesis, near the wall). The control: a cut that drops
  a hull vertex changes a step.
- **The spacing's guard** (F5). A spacing whose samples fall farther from the body than the margin is refused,
  naming one that keeps within it. The control: the case's spacing plans.
- **The flags**: a hull raises ``touching`` and ``unconverged``; one whose kind has no support (a box) also raises
  ``held_by_samples``."""

import math
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hull_cases import MARGIN, TUBE_RADIUS, hull_node, kind_points, rotation
from hypothesis import find, given, settings
from test_hull_contact import DRAWS, KINDS, SPACING, about_axis, held, near_the_wall, start

import mime.nodes.body._contact as contact_module
import mime.nodes.body._rigid_inertial as rigid_module
from mime.build import BuildRefused
from mime.nodes.body import RigidInertialMidpointNode, RigidInertialNode, convex_candidates, point_groups
from mime.nodes.body._contact import CONVERGED
from mime.scene import Box, Cylinder, Sphere
from mime.units import mm

jax.config.update("jax_enable_x64", True)

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
    moved, _, forces, left = node.hull_held(p, body, pose, system, free, DT)
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


def test_a_wedged_body_is_left_unconverged_and_reports_it():
    forces, _, left, _ = wedged(2.6, 3.5, (1.3e-10, 4.6e-10, 4.6e-10))
    assert np.sum(forces > 0) >= 2
    assert left > CONVERGED, f"the residual {left:.3g} is within {CONVERGED}: the flag would not rise"


def test_control_the_long_rod_converges_and_does_not_report_it():
    _, _, left, _ = wedged(0.5, 8.0, (1.3e-10, 4.6e-10, 4.6e-10))
    assert left <= CONVERGED


# -- the convex cut --------------------------------------------------------------------------------------------------

def inside_problems(name: str, candidates) -> list[str]:
    """Each of the hull's points a group drops that lies outside the group's kept points' convex hull."""
    from scipy.spatial import ConvexHull

    kind, settings_ = KINDS[name]
    points = kind_points(kind, settings_, SPACING)
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
    points = kind_points(kind, settings_, SPACING)
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
    points = kind_points(kind, settings_, SPACING)
    groups = point_groups(points, 8) if rows is None else np.asarray(rows)
    return hull_node(kind, settings_, SPACING, mass=8.5e-5, inertia=(1.3e-10, 4.6e-10, 4.6e-10), groups=groups)


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


def flag_names(part=None, spacing: float = SPACING) -> set[str]:
    from hull_cases import hull_contact

    from mime.experiment._experiment import run_layout

    return {flag.name for flag in run_layout(hull_contact(part=part, spacing=spacing).plan()).flags if flag.name}


def test_a_hull_raises_touching_and_unconverged_and_a_box_held_by_its_samples_also_held_by_samples():
    held_ = flag_names()
    assert {"touching", "unconverged"} <= held_ and "held_by_samples" not in held_
    assert {"touching", "unconverged", "held_by_samples"} <= flag_names(Box(size=mm((2.0, 1.0, 1.0))), 5e-5)
