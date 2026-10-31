"""The screw's caps in rings (MIME-ANO-119): no two samples of a cap lie closer than ``CAP_SEPARATION`` spacings,
over every setting the kind accepts (drawn) and over the kind-conformance harness's grid of starts, modulations,
wavenumbers and spacings; a cap carries the polar grid's count, ``outward x around``, wherever that holds, and fewer
only where the next count up would crowd; each cap's area and first and second moments are as accurate as the 0.1.0
polar grid's or better, within the harness's rounding floor, against a reference in extended precision; and the
samples are a traced function of the shape, whose derivative is the closed form's.

**The controls**, each failing on every run (nothing in them is drawn): a pair ``CROWDED`` spacings apart (the
anomaly's closest pair) re-inserted into a cap fails the separation check; 0.1.0's polar grid fails it at every setting
of the grid; the outer ring's weights moved by one part in 10^9 fail the moments' comparison."""

import math

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial import cKDTree
from screw_grid import FL9, HELIX_SETTINGS, REACH_FRACTIONS

from mime.scene import Screw
from mime.scene._caps import cap_rings
from mime.scene._screw import CAP_SEPARATION, LARGEST_MODULATION, LARGEST_WAVENUMBER, STARTS
from mime.testing import KIND_ROUNDING_ULPS
from mime.units import dimensionless, m

#: The anomaly's closest pair on 0.1.0's caps, in spacings: the control's planted pair.
CROWDED = 0.005
#: A wall-table BEM's working spacings for FL-9 (7j-a's two and the convergence ladder's next), m.
BEM_SPACINGS = (0.3e-3, 0.178e-3, 0.140e-3, 0.100e-3)
EPS = float(np.finfo(np.float64).eps)


def kind_at(starts, modulation, wavenumber, radius=FL9["radius"], length=FL9["length"]):
    held = {"radius": radius, "length": length, "wavenumber": wavenumber, "modulation": modulation}
    return Screw(radius=m(radius), length=m(length), wavenumber=dimensionless(wavenumber),
                 modulation=dimensionless(modulation), starts=starts), held


def caps_of(kind, held, counts) -> tuple[np.ndarray, np.ndarray]:
    """Each cap's points ``(2, n, 3)`` and weights ``(2, n)``, the cap at ``-L/2`` first."""
    found = kind.samples(held, "caps", counts)
    points = np.asarray(found.points, dtype=np.float64)
    return points.reshape(2, -1, 3), np.asarray(found.weights, dtype=np.float64).reshape(2, -1)


def closest_within_a_cap(points: np.ndarray) -> float:
    return min(float(cKDTree(cap).query(cap, 2)[0][:, 1].min()) for cap in points)


def polar_grid(kind, held, counts) -> tuple[np.ndarray, np.ndarray]:
    """0.1.0's caps, its rule restated: Gauss-Legendre nodes in the fraction out and the trapezoid rule round the
    turn, weighted by ``s rho^2 ds dtheta``."""
    outward, around = counts
    radius, length, nu, epsilon = (held[k] for k in ("radius", "length", "wavenumber", "modulation"))
    node, weight = np.polynomial.legendre.leggauss(outward)
    fraction = np.repeat((node + 1.0) / 2.0, around)
    share = np.repeat(weight / 2.0, around)
    angle = np.tile(2.0 * math.pi * (np.arange(around) + 0.5) / around, outward)
    rho = radius * (1.0 + epsilon * np.sin(kind.starts * angle))
    points, weights = [], []
    for end in (-1.0, 1.0):
        turned = angle + nu / radius * (end * length / 2.0 + length / 2.0)
        points.append(np.stack([np.full_like(rho, end * length / 2.0), fraction * rho * np.cos(turned),
                                fraction * rho * np.sin(turned)], axis=-1))
        weights.append(share * fraction * rho ** 2 * (2.0 * math.pi / around))
    return np.stack(points), np.stack(weights)


def exact_moments(kind, held) -> np.ndarray:
    """Each cap's area, first moments (y, z) and second moments (yy, yz, zz), ``(2, 6)``: integrals round the turn
    of ``rho^(k+2) / (k+2)`` times the turned direction's powers, trigonometric polynomials of degree at most
    ``4 starts + 2``, which the trapezoid rule at 64 points takes exactly, in extended precision."""
    radius, length, nu, epsilon = (np.longdouble(held[k]) for k in ("radius", "length", "wavenumber", "modulation"))
    count = 64
    theta = np.longdouble(2) * np.longdouble(math.pi) * np.arange(count, dtype=np.longdouble) / count
    rho = radius * (1 + epsilon * np.sin(kind.starts * theta))
    found = []
    for end in (-1, 1):
        turned = theta + nu / radius * (end * length / 2 + length / 2)
        c, s = np.cos(turned), np.sin(turned)
        step = np.longdouble(2) * np.longdouble(math.pi) / count
        found.append([np.sum(rho ** 2 / 2) * step, np.sum(rho ** 3 / 3 * c) * step, np.sum(rho ** 3 / 3 * s) * step,
                      np.sum(rho ** 4 / 4 * c * c) * step, np.sum(rho ** 4 / 4 * c * s) * step,
                      np.sum(rho ** 4 / 4 * s * s) * step])
    return np.asarray(found, dtype=np.longdouble)


def moments(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    y, z = points[..., 1].astype(np.longdouble), points[..., 2].astype(np.longdouble)
    w = weights.astype(np.longdouble)
    return np.stack([np.sum(w, -1), np.sum(w * y, -1), np.sum(w * z, -1), np.sum(w * y * y, -1),
                     np.sum(w * y * z, -1), np.sum(w * z * z, -1)], axis=-1)


def moment_errors(kind, held, points, weights) -> np.ndarray:
    """Each cap's moments' errors, ``(2, 6)``, each over its own scale: the area, the area times the reach, and the
    area times the reach squared (a first moment is zero, so its scale is its inputs')."""
    exact = exact_moments(kind, held)
    reach = held["radius"] * (1.0 + held["modulation"])
    area = exact[:, :1]
    scale = np.concatenate([area, area * reach, area * reach, area * reach ** 2, area * reach ** 2,
                            area * reach ** 2], axis=1)
    return np.asarray(np.abs(moments(points, weights) - exact) / scale, dtype=np.float64)


def floor_of(weights: np.ndarray) -> float:
    """The harness's rounding allowance for a sum of this many terms."""
    return KIND_ROUNDING_ULPS * EPS * math.sqrt(weights.shape[-1])


# -- the separation ------------------------------------------------------------------------------------------------------

@settings(deadline=None)
@given(starts=st.sampled_from(STARTS), modulation=st.floats(0.02, LARGEST_MODULATION),
       wavenumber=st.floats(-LARGEST_WAVENUMBER, LARGEST_WAVENUMBER), radius=st.floats(5e-4, 3e-3),
       slenderness=st.floats(1.0, 10.0), fineness=st.floats(10.0, 40.0))
def test_no_two_samples_of_a_cap_lie_closer_than_the_stated_separation(starts, modulation, wavenumber, radius,
                                                                       slenderness, fineness):
    """Over the settings the kind accepts, at the harness's range of spacings (a tenth to a fortieth of the reach);
    and the cap carries the polar grid's count unless the next count up crowds."""
    kind, held = kind_at(starts, modulation, wavenumber, radius, slenderness * radius)
    spacing = kind.radial_reach(held) / fineness
    outward, around = kind.sample_counts(held, spacing, "caps")
    points, _ = caps_of(kind, held, (outward, around))
    assert closest_within_a_cap(points) >= CAP_SEPARATION * spacing
    grid = math.ceil(kind.radial_reach(held) / spacing)
    assert around == kind.sample_counts(held, spacing, "lateral")[0] and 1 <= outward <= grid
    if outward < grid:
        above, _ = caps_of(kind, held, (outward + 1, around))
        assert closest_within_a_cap(above) < CAP_SEPARATION * spacing


@pytest.mark.parametrize("fineness", REACH_FRACTIONS)
@pytest.mark.parametrize(("starts", "modulation", "nu"), HELIX_SETTINGS)
def test_the_separation_holds_over_the_harness_grid(starts, modulation, nu, fineness):
    kind, held = kind_at(starts, modulation, nu)
    spacing = kind.radial_reach(held) / fineness
    points, _ = caps_of(kind, held, kind.sample_counts(held, spacing, "caps"))
    assert closest_within_a_cap(points) >= CAP_SEPARATION * spacing


@pytest.mark.parametrize("spacing", BEM_SPACINGS)
def test_fl9_keeps_the_polar_grids_count_at_a_bems_spacings(spacing):
    """FL-9's caps carry ``outward x around`` at every working spacing, the polar grid's own count: a BEM's blob, set
    by the samples' mean spacing, is the one 0.1.0 gave."""
    kind, held = kind_at(2, 0.33, 2.33)
    outward, around = kind.sample_counts(held, spacing, "caps")
    assert outward == math.ceil(kind.radial_reach(held) / spacing)
    assert around == kind.sample_counts(held, spacing, "lateral")[0]
    points, weights = caps_of(kind, held, (outward, around))
    assert points.shape[1] == outward * around
    assert closest_within_a_cap(points) >= CAP_SEPARATION * spacing


@pytest.mark.parametrize("fineness", REACH_FRACTIONS)
@pytest.mark.parametrize(("starts", "modulation", "nu"), HELIX_SETTINGS[::4])
def test_control_a_crowded_pair_re_inserted_fails_the_separation_check(starts, modulation, nu, fineness):
    """A sample ``CROWDED`` spacings from one of each cap's rings' samples, added to the cap, fails the check."""
    kind, held = kind_at(starts, modulation, nu)
    spacing = kind.radial_reach(held) / fineness
    points, _ = caps_of(kind, held, kind.sample_counts(held, spacing, "caps"))
    planted = np.concatenate([points, points[:, 1:2] + np.array([0.0, CROWDED * spacing, 0.0])], axis=1)
    assert closest_within_a_cap(planted) < CAP_SEPARATION * spacing


@pytest.mark.parametrize(("starts", "modulation", "nu"), HELIX_SETTINGS)
def test_control_the_polar_grid_fails_the_separation_check_at_every_setting(starts, modulation, nu):
    kind, held = kind_at(starts, modulation, nu)
    spacing = kind.radial_reach(held) / REACH_FRACTIONS[0]
    outward = math.ceil(kind.radial_reach(held) / spacing)
    around = kind.sample_counts(held, spacing, "lateral")[0]
    points, _ = polar_grid(kind, held, (outward, around))
    assert closest_within_a_cap(points) < CAP_SEPARATION * spacing


# -- the quadrature ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fineness", REACH_FRACTIONS)
@pytest.mark.parametrize(("starts", "modulation", "nu"), HELIX_SETTINGS)
def test_each_caps_area_and_moments_are_as_accurate_as_the_polar_grids(starts, modulation, nu, fineness):
    """Each cap's area, first and second moments, against the extended-precision reference: within the harness's
    rounding floor, or no further off than 0.1.0's polar grid at the same spacing."""
    kind, held = kind_at(starts, modulation, nu)
    spacing = kind.radial_reach(held) / fineness
    points, weights = caps_of(kind, held, kind.sample_counts(held, spacing, "caps"))
    rings = moment_errors(kind, held, points, weights)
    outward = math.ceil(kind.radial_reach(held) / spacing)
    grid_points, grid_weights = polar_grid(kind, held, (outward, kind.sample_counts(held, spacing, "lateral")[0]))
    grid = moment_errors(kind, held, grid_points, grid_weights)
    print(f"\nrings {rings.max():.2e} (floor {floor_of(weights):.2e}), polar grid {grid.max():.2e}")
    assert np.all(rings <= np.maximum(grid, floor_of(weights)))


@pytest.mark.parametrize("spacing", BEM_SPACINGS)
def test_control_the_outer_rings_weights_moved_by_a_part_in_a_billion_fail_the_moments(spacing):
    kind, held = kind_at(2, 0.33, 2.33)
    counts = kind.sample_counts(held, spacing, "caps")
    points, weights = caps_of(kind, held, counts)
    outer = cap_rings(math.prod(counts), kind.starts).counts[-1]
    planted = weights.copy()
    planted[:, -outer:] *= 1.0 + 1e-9
    assert np.any(moment_errors(kind, held, points, planted) > floor_of(weights))


def test_the_rings_weights_are_above_zero_at_every_count():
    """Every count of samples a cap may carry, from one ring of the fewest round a centre to a fine cap's: the radial
    weights are above zero, so is every sample's."""
    for starts in STARTS:
        for total in (*range(4 * starts + 4, 400), 1000, 5000, 20000):
            rings = cap_rings(total, starts)
            assert len(rings.fraction) == total
            assert np.all(rings.share > 0.0) and min(rings.counts) >= 4 * starts + 3


# -- the matrix a boundary-element solve factorises ----------------------------------------------------------------------

#: The least eigenvalue of one cap's free-space regularised-Stokeslet matrix over its largest, below which it reads as
#: singular. A declaration: two orders under the rings' at FL-9 (4.7e-4 at 0.178 mm, 7.8e-4 at 0.3 mm, measured on the
#: CPU when the rings landed) and eleven over the float64 rounding at which 0.1.0's polar grid sits (its ratio is
#: about -1e-16 at both, a negative eigenvalue at rounding: MIME-ANO-119's free-space matrix).
CONDITIONED = 1e-5


def conditioning(points: np.ndarray, kind, held, spacing) -> float:
    """``lambda_min / lambda_max`` of the regularised-Stokeslet matrix of ``points`` with the blob a closed surface at
    ``spacing`` gives (0.5 of the samples' mean spacing), as a BEM would solve with."""
    from mime.nodes.hydro.bem import stokeslet_matrix

    lateral = kind.sample_counts(held, spacing, "lateral")
    count = math.prod(lateral) + 2 * math.prod(kind.sample_counts(held, spacing, "caps"))
    eps_squared = 0.25 * (kind.area(held, "lateral") + kind.area(held, "caps")) / count
    matrix = np.asarray(stokeslet_matrix(jax.numpy.asarray(points), eps_squared, 1e-3), dtype=np.float64)
    values = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    return float(values[0] / values[-1])


@pytest.mark.parametrize("spacing", [0.3e-3])
def test_a_caps_free_space_matrix_is_well_conditioned_and_the_polar_grids_is_singular(spacing):
    """One cap of FL-9 on its own, in free space: with the rings its matrix is well conditioned; the control, the same
    cap on 0.1.0's polar grid at the same count, reads as singular, on every run (nothing is drawn)."""
    kind, held = kind_at(2, 0.33, 2.33)
    counts = kind.sample_counts(held, spacing, "caps")
    rings, _ = caps_of(kind, held, counts)
    grid, _ = polar_grid(kind, held, (counts[0], kind.sample_counts(held, spacing, "lateral")[0]))
    assert conditioning(rings[1], kind, held, spacing) >= CONDITIONED
    assert conditioning(grid[1], kind, held, spacing) < CONDITIONED


# -- the shape's derivative through the samples ------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["radius", "modulation"])
def test_the_caps_polar_moment_has_the_closed_forms_derivative(name):
    """``sum w (y^2 + z^2)`` over the caps, traced through the samples in the shape's settings, is the closed form
    ``2 int rho^4 / 4 dtheta = pi R^4 (2 + 6 eps^2 + 3 eps^4 / 4) / 2`` (the class's ``P0``, over two), and so is its
    derivative, compiled: the quadrature is exact for it at every value, and a sample's position follows the shape."""
    kind, held = kind_at(2, 0.33, 2.33)
    counts = kind.sample_counts(held, 0.178e-3, "caps")

    def polar(value):
        found = kind.samples({**held, name: value}, "caps", counts)
        return jax.numpy.sum(found.weights * (found.points[:, 1] ** 2 + found.points[:, 2] ** 2))

    def closed(radius, epsilon):
        return math.pi * radius ** 4 * (2.0 + 6.0 * epsilon ** 2 + 0.75 * epsilon ** 4) / 2.0

    radius, epsilon = held["radius"], held["modulation"]
    derivative = (4.0 * closed(radius, epsilon) / radius if name == "radius"
                  else math.pi * radius ** 4 * (12.0 * epsilon + 3.0 * epsilon ** 3) / 2.0)
    value = jax.numpy.asarray(held[name], dtype=jax.numpy.float64)
    assert float(polar(value)) == pytest.approx(closed(radius, epsilon), rel=1e-13)
    assert float(jax.jit(jax.grad(polar))(value)) == pytest.approx(derivative, rel=1e-12)


def test_the_caps_samples_are_the_same_made_twice():
    kind, held = kind_at(2, 0.33, 2.33)
    counts = kind.sample_counts(held, 0.178e-3, "caps")
    first, second = kind.samples(held, "caps", counts), kind.samples(held, "caps", counts)
    assert np.array_equal(np.asarray(first.points), np.asarray(second.points))
    assert np.array_equal(np.asarray(first.weights), np.asarray(second.weights))
