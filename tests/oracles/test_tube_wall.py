"""The tube's image system and the wall-kind interface: the Fourier-Bessel series of the image (the oracle of the
tube's tables), the read of a table through the evaluator, and the interface's refusals and controls, with a wall kind
the test registers on a small table of the series (``tube_cases``): the tube's own wall kind is reserved, since the
2026-05-21 tables fail their check (MIME-ANO-117, ``differential/check_wall_tables.py``).

**The series is checked against what defines it**, not against itself: with the free Stokeslet it vanishes on the
wall (the no-slip condition); the image is divergence-free and satisfies the momentum equation inside the tube
(``lap(curl u) = 0``) and is reciprocal, ``G(x, y) = G(y, x)^T``, as a Stokes flow's Green's function is; on the
axis its self-image across the axis is the same in every direction, by the tube's symmetry, and a point exactly on the
axis takes the limit the series approaches; and on the axis it gives the first wall coefficient of a sphere moving
along the axis of a tube, ``2.10444`` (Bohlin's and Faxen's; Happel & Brenner 1965). ``test_tube_mfs.py`` adds a
solution that shares no code with any series. Tolerances: the residual on the wall within ``1e-3`` of the free
Stokeslet's largest component there, for a source at most 0.6 of the radius from the axis at the tables' truncation,
declared before the first run; the coefficient within ``1e-5``. The divergence, momentum and reciprocity tolerances
were declared after the v0.2 builder's defect was found, and are derived from the series' own floor with a stated
margin (the comment above ``DIVERGENCE_FLOOR``: 20, 4 and 40 times). Controls: the series for a tube 1% wider fails the
no-slip condition, so the check can tell the radius a table was built at; the series as the v0.2 builder computed it
(``as_built``) meets the no-slip condition and fails divergence, the momentum equation, reciprocity and the symmetry
about the axis (its self-image across the axis has the opposite sign along y to along x), so none of those is the
no-slip check again; the series without its limits on the axis (``axis_limits=False``) gives order ``2e7`` there. The
reference the wall-table check uses (40, 160, 128) is itself converged at 0.8 of the radius and not at 0.95
(``test_the_finer_reference_...``), which bounds what the check's off-axis figures mean (MIME-ANO-117). The wall table
is not a trainable leaf and is a constant of the compiled step, not an argument (the tests of its spec and of the
lowered step, in the interface's section).

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
import re
from typing import Annotated, override

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tube_fourier_bessel as fb
from bem_cases import drag_node, free_space, resistance_at
from maddening import sysid
from maddening.core.params import ParamSpec, trainable_mask
from tube_cases import TUBE_RADIUS, haberman_sayre, polar_spacing, sphere_in_a_tube

import mime.interactions.hydrodynamic as hydrodynamic
import mime.nodes.hydro.bem._tube as tube_module
import mime.nodes.hydro.bem._walls as walls
from mime.build._maddening import node_state, parameter_leaves, parameter_specs
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


def reference_residual(source_radius: float, truncation: tuple[int, int, int]) -> float:
    """The no-slip residual (as ``no_slip_residual``) for a source ``source_radius`` of the radius from the axis, at
    ``truncation`` (modes, wavenumbers, angles), on points of the wall including those nearest the source."""
    rng = np.random.default_rng(7)
    angle, height = rng.uniform(0.0, 2 * math.pi, 24), rng.uniform(-0.5, 0.5, 24)
    wall = np.stack([np.cos(angle), np.sin(angle), height], axis=1)
    wall = np.vstack([wall, [[1.0, 0.0, 0.0], [math.cos(0.05), math.sin(0.05), 0.02]]])
    source = np.array([[source_radius, 0.0, 0.0]])
    image = fb.image(wall, source, 1.0, 1.0, modes=truncation[0], wavenumbers=truncation[1], angles=truncation[2])[:, 0]
    free = fb.free_stokeslet(wall, source, 1.0)[:, 0]
    return float(np.max(np.abs(image + free)) / np.max(np.abs(free)))


FINER = (40, 160, 128)


def test_the_finer_reference_is_converged_at_eight_tenths_of_the_radius_and_not_at_95_hundredths():
    """The series at 40, 160 and 128, which the wall-table check takes as its reference, meets the no-slip condition
    to a few parts in 10^4 for a source at 0.8 of the radius, to 1.6% at 0.9 and to 11% at 0.95 (printed). The check's
    figures where FL-9 reaches 0.95 of the tube's radius (an off-axis body in the 2.035 tube) are therefore not
    estimates of the tables' error: the reference is itself off by more than the 1% the tables were held to. The
    series at 80, 320 and 256 leaves 1.8% at 0.95 (measured in the review of the check, too slow to run here), so
    converging to 1% there needs more than 80 modes: a constraint on the tables' rebuild."""
    found = {rho: reference_residual(rho, FINER) for rho in (0.8, 0.9, 0.95)}
    print(f"\nthe no-slip residual of the series at {FINER}: " + ", ".join(f"{r:.1e} at {rho}" for rho, r in found.items()))
    assert found[0.8] <= 1e-3 and found[0.95] > 5e-2


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
    """The image velocity at a source exactly on the axis of the tube of radius 1, along the axis, is ``-k / (6 pi)``
    per unit force (``mu = 1``): a sphere of radius ``a`` there feels ``F = 6 pi mu a U (1 + k a / R + ...)``."""
    source = np.zeros((1, 3))
    found = -float(fb.image(source, source, 1.0, 1.0, **AS_BUILT)[0, 0, 2, 2]) * 6 * math.pi
    print(f"\nthe first wall coefficient along the axis: {found:.7f}")
    assert abs(found - 2.10444) <= 1e-5


def across_the_axis(as_built: bool = False, axis_limits: bool = True, rho: float = 0.0) -> tuple[float, float]:
    """The self-image's coefficients across the tube's axis, along x and y, for a source ``rho`` from the axis (exactly
    on it by default): equal by the tube's symmetry about its axis."""
    source = np.array([[rho, 0.0, 0.0]])
    image = fb.image(source, source, 1.0, 1.0, **AS_BUILT, as_built=as_built, axis_limits=axis_limits)[0, 0]
    return -float(image[0, 0]) * 6 * math.pi, -float(image[1, 1]) * 6 * math.pi


def test_on_the_axis_the_image_across_it_is_the_same_in_every_direction():
    along_x, along_y = across_the_axis()
    print(f"\nacross the axis: {along_x:.7f} along x, {along_y:.7f} along y")
    assert abs(along_x - along_y) <= 1e-6 * abs(along_x)


def test_a_point_exactly_on_the_axis_takes_the_limit_the_series_approaches():
    """The modes' limits on the axis (``n I_n / rho`` and ``q_n I_n`` are not zero there for ``n = 1``): the self-image
    exactly on the axis equals the one a millionth of the radius off it, within 1e-5 of itself."""
    exact, near = across_the_axis(), across_the_axis(rho=1e-6)
    print(f"\nexactly on the axis {exact[0]:.7f}, {exact[1]:.7f}; at 1e-6 {near[0]:.7f}, {near[1]:.7f}")
    assert max(abs(e - n) for e, n in zip(exact, near, strict=True)) <= 1e-5 * abs(near[0])


def test_control_without_the_axis_limits_a_point_on_the_axis_is_garbage():
    """The v0.2 builder (and this series before ``axis_limits``) set the modes' limits on the axis to zero, which gave
    ``k`` of order ``2e7`` along x and y for a point exactly on it."""
    along_x, along_y = across_the_axis(axis_limits=False)
    print(f"\nwithout the limits, exactly on the axis: {along_x:.3e} along x, {along_y:.3e} along y")
    assert abs(along_x) > 1e4 and abs(along_y) > 1e4


#: Two sources off the axis (the second turned and off the plane z = 0) and points inside the tube at other angles
#: and heights; the sources are among the points reciprocity pairs.
SOURCES = np.array([[0.5, 0.0, 0.0], [0.2, -0.35, 0.3]])
INSIDE = np.array([[0.25, 0.1, 0.2], [-0.3, -0.35, -0.15], [0.05, 0.55, 0.4], [0.45, -0.2, 0.05]])
BASIS = np.eye(3)


def gradient(point: np.ndarray, as_built: bool = False, step: float = 2e-2) -> np.ndarray:
    """``(3, 3, Ns, 3)``: ``d u_i / d x_d`` of the image at ``point`` for every source and force direction
    (``[i, d, source, column]``), by a 4th-order central difference of ``step`` (of the tube's radius)."""
    probes = np.array([point + c * step * BASIS[d] for d in range(3) for c in (2, 1, -1, -2)])
    g = fb.image(probes, SOURCES, 1.0, 1.0, **AS_BUILT, as_built=as_built).reshape(3, 4, len(SOURCES), 3, 3)
    return ((-g[:, 0] + 8 * g[:, 1] - 8 * g[:, 2] + g[:, 3]) / (12 * step)).transpose(2, 0, 1, 3)


def divergence(as_built: bool = False, step: float = 2e-2) -> float:
    """The image's divergence over the largest entry of its gradient, at each of the points, for every source and force
    direction: judged on the whole gradient's scale, since a column's diagonal terms can all be small."""
    worst = 0.0
    for point in INSIDE:
        grad = gradient(point, as_built, step)
        divergence_at = sum(grad[d, d] for d in range(3))
        worst = max(worst, float(np.max(np.abs(divergence_at) / np.max(np.abs(grad), axis=(0, 1)))))
    return worst


def curl_laplacian(as_built: bool = False, step: float = 4e-2) -> float:
    """The momentum equation's identity: a Stokes flow has ``lap(curl u) = 0`` (the pressure's gradient drops out of
    the curl). The Laplacian of the curl by central differences of ``step``, over the largest of its three terms (the three directions of the
    second derivative, all three components), at each of the points, for every source and force direction."""
    worst = 0.0
    for point in INSIDE:
        centres = [point] + [point + c * step * BASIS[d] for d in range(3) for c in (1, -1)]
        probes = np.array([q + c * step * BASIS[d] for q in centres for d in range(3) for c in (1, -1)])
        u = fb.image(probes, SOURCES, 1.0, 1.0, **AS_BUILT, as_built=as_built).reshape(7, 3, 2, len(SOURCES), 3, 3)
        du = (u[:, :, 0] - u[:, :, 1]) / (2 * step)
        curl = np.stack([du[:, 1, :, 2] - du[:, 2, :, 1], du[:, 2, :, 0] - du[:, 0, :, 2],
                         du[:, 0, :, 1] - du[:, 1, :, 0]], axis=2)
        terms = np.stack([(curl[1 + 2 * d] - 2 * curl[0] + curl[2 + 2 * d]) / step ** 2 for d in range(3)])
        worst = max(worst, float(np.max(np.abs(terms.sum(0)).max(axis=1) / np.max(np.abs(terms), axis=(0, 2)))))
    return worst


def reciprocity(as_built: bool = False) -> float:
    """The largest ``|G(x, y) - G(y, x)^T|`` over pairs of the points and the sources, over the pair's largest."""
    points = np.vstack([SOURCES, INSIDE])
    image = fb.image(points, points, 1.0, 1.0, **AS_BUILT, as_built=as_built)
    return max(float(np.max(np.abs(image[t, s] - image[s, t].T)) / np.max(np.abs(image[t, s])))
               for t in range(len(points)) for s in range(t))


#: The tolerances, derived from the series' own floor, not set after seeing a pass. The divergence is judged by a
#: 4th-order stencil of step 0.02 on the scale of the terms that cancel: its floor, the series' rounding (the
#: coefficients of the small wavenumbers are large and cancel), rises as the step shrinks (2.4e-6 at 0.02, 3.3e-5 at
#: 0.0025 in the review's measurement, the same at 5, 10 and 15 modes), so the step is fixed at 0.02 and the tolerance
#: is 20 times the floor measured there (printed with the test), where the builder's image gives order one.
#: The momentum identity's floor is the stencil's truncation, 0.024 at step 0.04 and 0.014 at 0.02 (it rises to 0.25
#: at 0.01, where the rounding takes over: the step is fixed, not tuned), so both steps are read, with a tolerance of 4 times the larger.
#: Reciprocity needs no stencil: its floor is the series' own rounding, about 2.5e-7, and the tolerance is 40 times it.
DIVERGENCE_FLOOR, MOMENTUM_FLOOR, RECIPROCITY_FLOOR = 2.4e-6, 2.4e-2, 2.5e-7
DIVERGENCE_TOLERANCE, MOMENTUM_TOLERANCE, RECIPROCITY_TOLERANCE = (20 * DIVERGENCE_FLOOR, 4 * MOMENTUM_FLOOR,
                                                                   40 * RECIPROCITY_FLOOR)


def test_the_series_image_is_divergence_free_and_reciprocal():
    found, recip = divergence(), reciprocity()
    print(f"\nthe divergence {found:.2e} (floor {DIVERGENCE_FLOOR:.1e}, tolerance {DIVERGENCE_TOLERANCE:.1e}), "
          f"reciprocity {recip:.2e} (floor {RECIPROCITY_FLOOR:.1e}, tolerance {RECIPROCITY_TOLERANCE:.0e})")
    assert found <= DIVERGENCE_TOLERANCE and recip <= RECIPROCITY_TOLERANCE


def test_the_series_image_satisfies_the_momentum_equation():
    """``lap(curl u) = 0`` for the image, a Stokes flow, at two steps."""
    found = [curl_laplacian(step=step) for step in (4e-2, 2e-2)]
    print(f"\nlap(curl u) over its terms: {found[0]:.2e} at step 0.04, {found[1]:.2e} at 0.02 "
          f"(floor {MOMENTUM_FLOOR:.1e}, tolerance {MOMENTUM_TOLERANCE:.1e})")
    assert max(found) <= MOMENTUM_TOLERANCE


def test_control_the_series_as_the_v02_builder_computed_it_meets_the_wall_and_fails_inside():
    """The defect the 2026-05-21 tables hold (``tube_fourier_bessel.image``'s ``as_built``): the no-slip condition
    holds, and the divergence, the momentum equation, reciprocity and the symmetry about the axis do not."""
    div, momentum, recip = divergence(as_built=True), curl_laplacian(as_built=True), reciprocity(as_built=True)
    on_wall = no_slip_residual(0.3, as_built=True)
    along_x, along_y = across_the_axis(as_built=True)
    print(f"\nas built: on the wall {on_wall:.2e}; the divergence {div:.2e}, lap(curl u) {momentum:.2e}, "
          f"reciprocity {recip:.2e}; across the axis {along_x:.5f} along x, {along_y:.5f} along y")
    assert on_wall <= 1e-3
    assert div > 100 * DIVERGENCE_TOLERANCE and momentum > 10 * MOMENTUM_TOLERANCE
    assert recip > 100 * RECIPROCITY_TOLERANCE
    assert abs(along_x - along_y) > 0.1 * abs(along_x)


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


def table_bytes(built) -> int:
    return sum(leaf.nbytes for leaf in jax.tree.leaves(drag_node(built, "ball").constants()["wall_table"]))


def test_the_table_is_never_trainable_in_a_mask_a_gradient_or_the_fim(centred):
    """The table is declared ``trainable=False`` and is not a leaf of the graph's parameters at all (MADDENING's
    ``params_pytree`` leaves out a value it cannot make a floating array of), so the graph's trainable mask, a gradient
    taken over the parameters and ``sysid.fim`` over that mask cannot reach it: the mask has an entry for each leaf and
    none for the table, and the Fisher matrix has a column for each trainable scalar and none for a table entry."""
    graph, node = centred.graph, drag_node(centred, "ball")
    specs, leaves = parameter_specs(graph)["drag:ball"], parameter_leaves(graph)["drag:ball"]
    mask = graph.trainable_mask()["nodes"]["drag:ball"]
    held = node_state(graph, "drag:ball")
    pose = jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=jnp.float64)
    report = sysid.fim(lambda params: node.update(held, {"pose": pose}, 1e-3, params=params)["resistance"],
                       leaves, mask=mask)
    print(f"\nthe table's spec trainable {specs['wall_table'].trainable}; the parameters {sorted(leaves)}; the Fisher "
          f"matrix's columns {report.param_names}")
    assert specs["wall_table"].trainable is False and specs["wall_frame"].trainable is False
    assert "wall_table" not in leaves and set(mask) == set(leaves)
    assert len(report.param_names) == sum(int(jnp.size(leaves[name])) for name in leaves if mask[name])
    assert not any("table" in name for name in report.param_names) and len(report.param_names) > 0


def test_control_a_table_left_trainable_would_be_a_column_of_the_mask():
    """The mask is read from the specs: a spec that says nothing about a leaf (the default) leaves it trainable, so the
    absence above is the declaration's and the leaf's absence, not the mask's blindness to a table."""
    table = {"values": jnp.zeros(4), "breaks": jnp.zeros(2)}
    assert trainable_mask({"wall_table": table}, {}) == {"wall_table": {"values": True, "breaks": True}}
    assert trainable_mask({"wall_table": table}, {"wall_table": {"values": ParamSpec(trainable=False),
                                                                   "breaks": ParamSpec(trainable=False)}}) \
        == {"wall_table": {"values": False, "breaks": False}}


def test_the_table_is_a_constant_of_the_compiled_step_not_an_argument(centred):
    """What D6 (the table as a node parameter, so a compiled step takes it as an argument) turned out to be as built:
    the table is a node constant, since ``params_pytree`` leaves it out, so the lowered step carries it as a constant
    of its text (about two characters a byte, as ``static_data`` would). Pinned as it is; when the table becomes an
    argument, the assertion flips to the table's absence from the text."""
    node, graph = drag_node(centred, "ball"), centred.graph
    held, leaves = node_state(graph, "drag:ball"), parameter_leaves(graph)["drag:ball"]
    pose = jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=jnp.float64)
    lowered = jax.jit(lambda params: node.update(held, {"pose": pose}, 1e-3, params=params)["resistance"]) \
        .lower(leaves).as_text()
    largest = max((len(found.group(0)) for found in re.finditer(r"dense<[^>]*>", lowered)), default=0)
    print(f"\nthe table {table_bytes(centred)} bytes; the largest constant in the lowered step {largest} characters")
    assert largest > table_bytes(centred)


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
