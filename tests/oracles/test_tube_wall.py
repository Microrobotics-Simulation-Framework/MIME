"""The tube's image system and the wall-kind interface: the Fourier-Bessel series of the image (the oracle of the
tube's tables), the read of a table through the evaluator, and the interface's refusals and controls, with a wall kind
the test registers on a small table of the series (``tube_cases``): the tube's own wall kind is reserved, since the
2026-05-21 tables fail their check (MIME-ANO-117, ``differential/check_wall_tables.py``).

**The series is checked against what defines it**, not against itself: with the free Stokeslet it vanishes on the
wall (the no-slip condition); the image is divergence-free inside the tube and reciprocal, ``G(x, y) = G(y, x)^T``;
and on the axis it gives the first wall coefficient of a sphere moving along the axis of a tube, ``2.10444`` (Bohlin's
and Faxen's; Happel & Brenner 1965). Tolerances, declared before the first run: the residual on the wall within
``1e-3`` of the free Stokeslet's largest component there, for a source at most 0.6 of the radius from the axis at the
tables' truncation; the coefficient within ``1e-5``. The divergence and reciprocity tolerances were declared after
the v0.2 builder's defect was found, from the defect-free series' own: the divergence, by a central difference of
step ``1e-3`` of the radius, within ``1e-4`` of the image's largest component over the radius; reciprocity within
``1e-5`` of the pair's largest component. Controls: the series for a tube 1% wider fails the no-slip condition, so
the check can tell the radius a table was built at; the series as the v0.2 builder computed it (``as_built``) meets
the no-slip condition and fails both divergence and reciprocity, so neither of those is the no-slip check again.

**The read** of a table through the tube's correction (cubic along every axis, the angle and the axial offset
mirrored past their symmetric ends) reproduces the series at its knots: a small table built from the series itself,
read at samples whose pairs lie on its knots in every quadrant of the angle and of the axial offset, in a tube turned
and moved off the origin, within ``1e-5`` of each block (the series' own binning of a source's radius rounds it at
``1e-6`` of the tube's radius, which the table and the direct series do at different radii). Control: the odd
components' signs dropped fails.

**The interface's controls**, each failing on every run: the tube's correction replaced at the node by free space's
fails the confined sphere (Haberman & Sayre), whose smallest tested confinement deviates from free space by many
times its tolerance (printed with the test); a wall kind named for a confining geometry it does not declare (a
test-registered geometry whose region is a ``slab``, a channel between planes) is refused when planned, with
``drag.wall-kind``, the valid wall kinds listed; a body moved off the table publishes ``wall_accepted`` false, and a
node that does not publish it fails that test; a body longer along the axis than the table reaches is refused when
planned (``drag.wall-range``), and a plan that omits the check fails that test."""

import dataclasses
import math
from typing import Annotated, override

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tube_fourier_bessel as fb
from bem_cases import drag_node, free_space, resistance_at
from tube_cases import TUBE_RADIUS, haberman_sayre, polar_spacing, sphere_in_a_tube

import mime.interactions.hydrodynamic as hydrodynamic
import mime.nodes.hydro.bem._tube as tube_module
import mime.nodes.hydro.bem._walls as walls
from mime.build._maddening import node_state, parameter_leaves
from mime.build import BuildRefused
from mime.gradients import GradientStatus
from mime.nodes.hydro.bem import UNBOUNDED, BEMDragNode, frame_across, tube_correction, unit_tube_table, \
    wall_correction_named
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.interactions import Drag
from mime.registry import Geometry, register_geometry
from mime.scene import Capsule, Scene, Sphere, Z
from mime.things import Body, Medium, Part, State, Structure
from mime.units import Length, Quantity, kg, kg_m2, m, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

jax.config.update("jax_enable_x64", True)

AS_BUILT = {"modes": 15, "wavenumbers": 80, "angles": 64}


# -- the oracle --------------------------------------------------------------------------------------------------

def no_slip_residual(source_radius: float, built_at: float = 1.0, as_built: bool = False) -> float:
    """The largest velocity component of the series' image plus the free Stokeslet at points of the wall of the tube
    of radius 1, per unit force at a source ``source_radius`` from the axis, over the free Stokeslet's largest there;
    the series taken for a tube of radius ``built_at``."""
    rng = np.random.default_rng(7)
    angle, height = rng.uniform(0.0, 2 * math.pi, 24), rng.uniform(-2.0, 2.0, 24)
    wall = np.stack([np.cos(angle), np.sin(angle), height], axis=1)
    source = np.array([[max(source_radius, 1e-9), 0.0, 0.0]])
    image = fb.image(wall, source, built_at, 1.0, **AS_BUILT, as_built=as_built)[:, 0]
    free = fb.free_stokeslet(wall, source, 1.0)[:, 0]
    return float(np.max(np.abs(image + free)) / np.max(np.abs(free)))


@pytest.mark.parametrize("source_radius", [0.0, 0.3, 0.6])
def test_the_series_with_the_free_stokeslet_vanishes_on_the_wall(source_radius):
    found = no_slip_residual(source_radius)
    print(f"\nsource at {source_radius} of the radius: the no-slip residual {found:.2e} of the free Stokeslet")
    assert found <= 1e-3


def test_control_the_series_for_a_tube_one_percent_wider_fails_the_no_slip_condition():
    found = no_slip_residual(0.3, built_at=1.01)
    print(f"\nthe series for a tube 1% wider: the residual {found:.2e} of the free Stokeslet")
    assert found > 10 * 1e-3


def test_on_the_axis_the_image_gives_the_first_wall_coefficient_of_a_sphere_on_the_axis():
    """The image velocity at a source on the axis of the tube of radius 1, along the axis, is ``-k / (6 pi)`` per unit
    force (``mu = 1``): a sphere of radius ``a`` there feels ``F = 6 pi mu a U (1 + k a / R + ...)``."""
    source = np.array([[1e-9, 0.0, 0.0]])
    found = -float(fb.image(source, source, 1.0, 1.0, **AS_BUILT)[0, 0, 2, 2]) * 6 * math.pi
    print(f"\nthe first wall coefficient along the axis: {found:.7f}")
    assert abs(found - 2.10444) <= 1e-5


def divergence(source: np.ndarray, point: np.ndarray, as_built: bool = False, step: float = 1e-3) -> np.ndarray:
    """The image's divergence at ``point`` for a unit force along each axis at ``source``, by a central difference of
    ``step``, over the image's largest component there divided by the radius (1)."""
    probes = np.array([point + sign * step * np.eye(3)[d] for d in range(3) for sign in (1.0, -1.0)])
    image = fb.image(probes, source[None, :], 1.0, 1.0, **AS_BUILT, as_built=as_built)[:, 0]
    found = sum((image[2 * d][d] - image[2 * d + 1][d]) / (2 * step) for d in range(3))
    return np.abs(found) / np.max(np.abs(image))


#: A source off the axis, and points inside the tube at other angles and heights.
SOURCE = np.array([0.5, 0.0, 0.0])
INSIDE = np.array([[0.2, 0.3, 0.1], [-0.4, 0.1, -0.3], [0.1, -0.6, 0.4]])


def reciprocity(as_built: bool = False) -> float:
    """The largest ``|G(x, y) - G(y, x)^T|`` over pairs of the points and the source, over the pair's largest."""
    points = np.vstack([SOURCE, INSIDE])
    image = fb.image(points, points, 1.0, 1.0, **AS_BUILT, as_built=as_built)
    return max(float(np.max(np.abs(image[t, s] - image[s, t].T)) / np.max(np.abs(image[t, s])))
               for t in range(len(points)) for s in range(t))


def test_the_series_image_is_divergence_free_and_reciprocal():
    worst = max(float(np.max(divergence(SOURCE, point))) for point in INSIDE)
    print(f"\nthe divergence {worst:.2e}, reciprocity {reciprocity():.2e}")
    assert worst <= 1e-4 and reciprocity() <= 1e-5


def test_control_the_series_as_the_v02_builder_computed_it_meets_the_wall_and_fails_inside():
    """The defect the 2026-05-21 tables hold (``tube_fourier_bessel.image``'s ``as_built``): the no-slip condition
    holds, and the divergence and reciprocity do not."""
    worst = max(float(np.max(divergence(SOURCE, point, as_built=True))) for point in INSIDE)
    on_wall = no_slip_residual(0.3, as_built=True)
    print(f"\nas built: on the wall {on_wall:.2e}; the divergence {worst:.2e}, reciprocity "
          f"{reciprocity(as_built=True):.2e}")
    assert on_wall <= 1e-3 and worst > 100 * 1e-4 and reciprocity(as_built=True) > 100 * 1e-5


# -- the read ----------------------------------------------------------------------------------------------------

#: A small table built from the series: a tube of radius 2, radii, angles and offsets on uniform grids.
BUILT_AT, RADII, ANGLES, OFFSETS = 2.0, np.linspace(0.1, 1.6, 5), np.linspace(0.0, math.pi, 9), np.linspace(0.0, 1.2, 5)


@pytest.fixture(scope="module")
def series_table():
    data = np.zeros((len(RADII), len(RADII), len(ANGLES), len(OFFSETS), 3, 3))
    angle, offset = (grid.ravel() for grid in np.meshgrid(ANGLES, OFFSETS, indexing="ij"))
    targets = np.stack([np.zeros_like(angle), np.zeros_like(angle), offset], axis=1)
    turn = fb.to_cylindrical(angle)
    for i, target_radius in enumerate(RADII):
        targets[:, 0], targets[:, 1] = target_radius * np.cos(angle), target_radius * np.sin(angle)
        for j, source_radius in enumerate(RADII):
            image = fb.image(targets, np.array([[source_radius, 0.0, 0.0]]), BUILT_AT, 1.0, **AS_BUILT)[:, 0]
            data[i, j] = np.einsum("pab,pbc->pac", turn, image).reshape(len(ANGLES), len(OFFSETS), 3, 3)
    return unit_tube_table(BUILT_AT, RADII, ANGLES, OFFSETS, data).arrays(jnp.float64)


#: The tube the read is placed in: turned, moved off the origin, in SI, and a viscosity that is not 1.
TUBE, MU = 3e-3, 2e-3
AXIS = np.array([0.3, -0.4, 0.86]) / np.linalg.norm([0.3, -0.4, 0.86])
CENTRE = np.array([1e-3, -2e-3, 0.5e-3])


def on_knots() -> np.ndarray:
    """Samples in the unit tube's cylindrical coordinates whose pairs' angles and axial offsets are the table's knots
    or their negatives, a radius of the table each: one at angle 0.7 and height 0.2, and others a knot either way
    (a radius and an offset inside the first and last, which the samples' placement in the turned tube can round
    past)."""
    rng = np.random.default_rng(3)
    angles = 0.7 + np.concatenate([[0.0], rng.choice([-1, 1], 9) * ANGLES[rng.integers(1, len(ANGLES), 9)]])
    heights = 0.2 + np.concatenate([[0.0], rng.choice([-1, 1], 9) * OFFSETS[rng.integers(1, len(OFFSETS) - 1, 9)]]) \
        / BUILT_AT
    radii = RADII[rng.integers(1, len(RADII) - 1, 10)] / BUILT_AT
    return np.stack([radii * np.cos(angles), radii * np.sin(angles), heights], axis=1)


def read_misses(table) -> np.ndarray:
    """Each pair from the first sample, and each pair to it, read through the tube's correction against the series
    at the same points, relative to the pair's largest component."""
    local = on_knots()
    frame = np.asarray(frame_across(AXIS)).reshape(3, 3)
    world = CENTRE + (local * TUBE) @ frame
    read = tube_correction(jnp.asarray(world), {"centre": tuple(CENTRE), "frame": tuple(frame.ravel()),
                                                "radius": TUBE, "table": table}, MU)
    count = len(local)
    found = np.asarray(read.matrix).reshape(count, 3, count, 3).transpose(0, 2, 1, 3)
    series = np.einsum("ia,tsab,jb->tsij", frame.T, fb.image(local, local, 1.0, 1.0, **AS_BUILT), frame.T) \
        / (TUBE * MU)
    pairs = [(0, s) for s in range(count)] + [(t, 0) for t in range(1, count)]
    return np.array([np.max(np.abs(found[t, s] - series[t, s])) / np.max(np.abs(series[t, s])) for t, s in pairs])


def test_the_read_reproduces_the_series_at_its_knots_in_every_quadrant(series_table):
    misses = read_misses(series_table)
    print(f"\nthe read at the knots against the series: worst {misses.max():.2e} over {misses.size} pairs")
    assert misses.max() <= 1e-5


def test_control_the_odd_components_signs_dropped_fail_the_read(series_table, monkeypatch):
    monkeypatch.setattr(tube_module, "ODD_ALONG", np.zeros((3, 3), dtype=bool))
    monkeypatch.setattr(tube_module, "ODD_IN_ANGLE", np.zeros((3, 3), dtype=bool))
    misses = read_misses(series_table)
    print(f"\nthe odd components' signs dropped: worst {misses.max():.2e}")
    assert misses.max() > 1e-2


# -- the interface ------------------------------------------------------------------------------------------------

@register_geometry
@dataclasses.dataclass(frozen=True, kw_only=True)
class Channel(Geometry):
    """A channel between two parallel planes, its lumen a ``slab``: a confining geometry the tube's wall kind does not
    declare, and that no wall kind matches in 0.3."""

    name = "oracle.channel"
    version = "0.0.1"
    differentiability = GradientStatus.PIECEWISE_JUMP
    builds = {"": ()}
    regions = ("lumen",)
    region_shapes = {"lumen": "slab"}
    half_gap: Annotated[Quantity, Length]

    @override
    def region_parameters(self, region, settings):
        return {"centre": (0.0, 0.0, 0.0), "axis": (0.0, 0.0, 1.0), "half_thickness": settings["half_gap"]}


def channel_experiment(wall_kind: str):
    """A sphere in water filling a channel between planes, with the tube's wall kind named for it."""
    experiment = Experiment("channel", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_structure(Structure("channel", geometry=Channel(half_gap=mm(3.0))))
    experiment.add_medium(Medium("water", fills="channel.lumen"))
    experiment.declare("water.viscosity", mPa_s(1.0), reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=mm(0.5)))},
                             initial=State(position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                           velocity=m_per_s((0.0, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("ball.mass", kg(1e-5), reason="a millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
    experiment.attach(Drag.BEM(spacing=mm(0.3), blob=0.5, wall_kind=wall_kind), body="ball", medium="water")
    return experiment


def test_a_wall_kind_named_for_a_geometry_it_does_not_declare_is_refused_listing_the_wall_kinds():
    with pytest.raises(BuildRefused) as refused:
        channel_experiment("oracle.tube_series").plan()
    message = str(refused.value)
    print(f"\n{message}")
    assert refused.value.codes() == ["drag.wall-kind"]
    assert "a slab region" in message and "free_space 0.1.0" in message and "oracle.tube_series 0.0.1" in message
    assert "a plane wall has none in 0.3" in message and "tube (reserved" in message


def test_control_the_tube_in_a_tube_is_not_refused_for_its_geometry():
    """The control of the refusal above: the same wall kind in the geometry it declares plans with no
    ``drag.wall-kind``, so the refusal is the geometry's, not the wall kind's."""
    sphere_in_a_tube("matched", radius=0.1, spacing=mm(0.2)).plan()


# -- the confined sphere ------------------------------------------------------------------------------------------

#: The sphere's radius over the tube's, the smallest confinement tested, and its two resolutions (polar Gauss nodes;
#: twice as many round). Gauss's first node keeps every sample more than a fifth of the sphere's radius from the axis,
#: inside the table's first radius, up to eleven nodes.
RATIO, POLAR = 0.1, (10, 11)
#: The oracle's tolerance on the ratio of the drag along the axis to free space's at the same samples, extrapolated to
#: zero spacing, declared before the first run; the confinement's own effect, Haberman & Sayre's ``K - 1``, is printed
#: with it. The protocol first declared held each of 8 and 10 polar nodes within it, unextrapolated, and failed at 8
#: (+0.58%): the ratio's own discretisation error does not cancel, and falls as one over the nodes (+0.80%, +0.58%,
#: +0.44%, +0.40% at 6, 8, 10 and 11), so the oracle reads the first-order extrapolation in ``1 / n``, as the
#: free-space sphere's does, with the tolerance unchanged.
SPHERE_TOLERANCE = 5e-3


def drag_ratio(polar: int) -> float:
    """The drag along the tube's axis on the sphere centred in the tube, through ``Drag.BEM`` with the tube's
    correction on the series' table, over its drag in free space at the same samples."""
    spacing = polar_spacing(RATIO, polar)
    confined = resistance_at(sphere_in_a_tube("confined", radius=RATIO, spacing=spacing).build(), "ball")
    free = resistance_at(free_space("free", {"ball": Sphere(radius=m(RATIO * TUBE_RADIUS.value))},
                                    Drag.BEM(spacing=spacing, blob=0.5, wall_kind="free_space")).build(), "ball")
    return float(confined[2, 2] / free[2, 2])


def sphere_misses() -> tuple[list[float], float, float]:
    """Each resolution's miss from Haberman & Sayre, the first-order extrapolation's in ``1 / n``, and ``K``."""
    expected = haberman_sayre(RATIO)
    coarse, fine = POLAR
    found = [drag_ratio(polar) / expected - 1.0 for polar in POLAR]
    return found, (fine * found[1] - coarse * found[0]) / (fine - coarse), expected


def test_the_confined_sphere_drags_as_haberman_and_sayre_say():
    each, extrapolated, expected = sphere_misses()
    print(f"\nK = {expected:.5f} at a / R = {RATIO}; its confinement's effect K - 1 = {expected - 1:.4f}, "
          f"{(expected - 1) / SPHERE_TOLERANCE:.0f} times the tolerance; the misses at {POLAR} polar nodes: "
          f"{np.round(each, 5)}, extrapolated {extrapolated:.5f}")
    assert expected - 1 >= 3 * SPHERE_TOLERANCE
    assert abs(each[1]) < abs(each[0]) and abs(extrapolated) <= SPHERE_TOLERANCE


def test_control_the_tubes_correction_replaced_by_free_spaces_fails_the_confined_sphere(monkeypatch):
    """At the node, not through the support table: the tube's correction function (the read of its table) swapped
    for free space's, which adds nothing."""
    held = wall_correction_named("oracle.tube_series", "0.0.1")
    monkeypatch.setitem(walls._corrections, ("oracle.tube_series", "0.0.1"),
                        dataclasses.replace(held, correction=lambda points, parameters, viscosity:
                                            UNBOUNDED.correction(points, parameters, viscosity)))
    each, extrapolated, _ = sphere_misses()
    print(f"\nthe free space's correction in the tube's place: misses {np.round(each, 5)}, extrapolated "
          f"{extrapolated:.5f}")
    assert abs(extrapolated) > SPHERE_TOLERANCE


# -- what the node publishes, and what the plan checks -------------------------------------------------------------

def published_at(built, offset: float) -> tuple[bool, np.ndarray]:
    """``wall_accepted`` and the resistance of the drag node with the ball moved ``offset`` of the tube's radius
    along x."""
    node = drag_node(built, "ball")
    held = node_state(built.graph, "drag:ball")
    pose = jnp.asarray((offset * TUBE_RADIUS.value, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=jnp.float64)
    out = node.update(held, {"pose": pose}, 1e-3, params=parameter_leaves(built.graph)["drag:ball"])
    return bool(out["wall_accepted"]), np.asarray(out["resistance"])


def off_the_table_is_published(built) -> bool:
    inside, value = published_at(built, 0.0)
    outside, beyond = published_at(built, 0.5)
    return inside and np.all(np.isfinite(value)) and not outside and not np.all(np.isfinite(beyond))


@pytest.fixture(scope="module")
def centred():
    return sphere_in_a_tube("published", radius=RATIO, spacing=polar_spacing(RATIO, 6)).build()


def test_a_body_moved_off_the_table_is_published_as_not_accepted(centred):
    assert off_the_table_is_published(centred)


class _AlwaysAccepted(BEMDragNode):
    """A node whose step does not publish the reads' flag: it says every read was accepted."""

    def advance(self, p, state, boundary_inputs, dt):
        return {**super().advance(p, state, boundary_inputs, dt), "wall_accepted": jnp.asarray(True)}


def test_control_a_node_that_does_not_publish_the_flag_fails(monkeypatch):
    monkeypatch.setattr(hydrodynamic, "BEMDragNode", _AlwaysAccepted)
    built = sphere_in_a_tube("unpublished", radius=RATIO, spacing=polar_spacing(RATIO, 6)).build()
    assert not off_the_table_is_published(built)


def too_long() -> object:
    """A capsule along the tube's axis longer than the series' table reaches along it."""
    return sphere_in_a_tube("too_long", radius=RATIO, spacing=mm(0.3),
                            hull=Capsule(radius=mm(0.3), height=mm(3.0), axis=(0.0, 0.0, 1.0)))


def test_a_body_longer_than_the_table_reaches_is_refused_when_planned():
    with pytest.raises(BuildRefused) as refused:
        too_long().plan()
    print(f"\n{refused.value}")
    assert "drag.wall-range" in refused.value.codes()


def test_control_a_plan_that_omits_the_reach_check_does_not_refuse_it(monkeypatch):
    held = wall_correction_named("oracle.tube_series", "0.0.1")
    monkeypatch.setitem(walls._corrections, ("oracle.tube_series", "0.0.1"),
                        dataclasses.replace(held, beyond=lambda parameters, reach: []))
    try:
        too_long().plan()
    except BuildRefused as refused:
        assert "drag.wall-range" not in refused.value.codes()
