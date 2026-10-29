"""The tube's image against a solution that shares no code with any Fourier-Bessel series: the method of fundamental
solutions (MFS).

Singular Stokeslets are placed on a cylinder of radius ``OUTSIDE`` (1.3 of the tube's) round the tube, over
``|z| <= HALF_LENGTH``, with strengths found by least squares so that, with the source's free Stokeslet, the velocity
vanishes at collocation points on the wall. The field inside the tube is a sum of Stokeslets whose singularities lie
outside it, so it is a Stokes flow exactly; what remains of it is the wall's residual (measured at fresh points) and
the open ends. Only forces across the axis are used: a net axial force in a finite open tube drives a flux the
infinite tube does not have.

This is the independent half of the check that the image is a Stokes flow. The divergence, momentum and reciprocity
identities (``test_tube_wall.py``) are necessary; this is a solution of the boundary-value problem itself, and the
builder's image of a force along the source's azimuthal direction (MIME-ANO-117) misses it by order one, where the
series' own meets it to the MFS's wall residual.

The resolution here is small, for the lane's budget: in the review that wrote this check, three resolutions (up to
7,392 unknowns) had the series' distance to the MFS fall with the MFS's wall residual (off the axis 1.2e-4, 1.1e-4,
1.2e-6) while the builder's stayed at 0.8 to 2.4; the test prints both at this resolution. Tolerance, declared from
that: the series within ``TOLERANCE`` of the MFS, and the builder's image farther than ``100 * TOLERANCE``."""

import math

import numpy as np
import pytest
import tube_fourier_bessel as fb
from scipy import linalg

AS_BUILT = {"modes": 15, "wavenumbers": 80, "angles": 64}
#: The Stokeslets' cylinder over the tube's radius, the number round it and along it, and the half-length they cover.
OUTSIDE, AROUND, ALONG, HALF_LENGTH = 1.3, 24, 30, 3.0
TOLERANCE = 5e-3
#: The sources, and the force directions used at each (x and y: across the axis).
CASES = {"on the axis": (np.zeros(3), (0, 1)), "off the axis": (np.array([0.5, 0.0, 0.0]), (0, 1)),
         "off the axis, turned": (np.array([0.2, -0.35, 0.3]), (0, 1))}
#: Points inside the tube the images are compared at, and the source itself.
INSIDE = np.array([[0.25, 0.1, 0.2], [-0.3, -0.35, -0.15], [0.05, 0.55, 0.4], [0.45, -0.2, 0.05]])


def stokeslet_matrix(targets: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """``(3 Nt, 3 Ns)``: the free Stokeslet's velocity at each target per unit force at each source (``mu = 1``)."""
    return fb.free_stokeslet(targets, sources, 1.0).transpose(0, 2, 1, 3).reshape(3 * len(targets), 3 * len(sources))


def ring(radius: float, around: int, along: int, shift: float = 0.0) -> np.ndarray:
    angle = (np.arange(around) + shift) * 2 * math.pi / around
    height = np.linspace(-HALF_LENGTH, HALF_LENGTH, along)
    a, z = np.meshgrid(angle, height, indexing="ij")
    return np.stack([radius * np.cos(a).ravel(), radius * np.sin(a).ravel(), z.ravel()], axis=1)


@pytest.fixture(scope="module")
def solved() -> dict[str, dict[str, object]]:
    """Per case and force direction: the MFS's image at the points and at the source, the MFS's wall residual at
    fresh points over the free Stokeslet there, and the two series' images, all over the series' largest component."""
    stokeslets = ring(OUTSIDE, AROUND, ALONG)
    wall = ring(1.0, int(1.5 * AROUND), int(1.5 * ALONG), shift=0.25)
    columns = [(name, force) for name, (_, forces) in CASES.items() for force in forces]
    rhs = np.column_stack([-fb.free_stokeslet(wall, CASES[name][0][None], 1.0)[:, 0, :, force].ravel()
                           for name, force in columns])
    strengths = linalg.lstsq(stokeslet_matrix(wall, stokeslets), rhs, lapack_driver="gelsy", check_finite=False)[0]
    rng = np.random.default_rng(5)
    angle, height = rng.uniform(0, 2 * math.pi, 200), rng.uniform(-1.5, 1.5, 200)
    fresh = np.stack([np.cos(angle), np.sin(angle), height], axis=1)
    on_wall = stokeslet_matrix(fresh, stokeslets) @ strengths
    out: dict[str, dict[str, object]] = {}
    for index, (name, force) in enumerate(columns):
        source = CASES[name][0][None]
        targets = np.vstack([source, INSIDE])
        mfs = (stokeslet_matrix(targets, stokeslets) @ strengths[:, index]).reshape(len(targets), 3)
        corrected = fb.image(targets, source, 1.0, 1.0, **AS_BUILT)[:, 0, :, force]
        built = fb.image(targets, source, 1.0, 1.0, **AS_BUILT, as_built=True)[:, 0, :, force]
        free = fb.free_stokeslet(fresh, source, 1.0)[:, 0, :, force]
        scale = float(np.max(np.abs(corrected)))
        out[f"{name}, force along {'xyz'[force]}"] = {
            "wall": float(np.max(np.abs(on_wall[:, index].reshape(-1, 3) + free)) / np.max(np.abs(free))),
            "series": float(np.max(np.abs(corrected - mfs)) / scale), "builder": float(np.max(np.abs(built - mfs)) / scale),
            "coefficient": -6 * math.pi * float(mfs[0, force])}
    return out


def test_the_series_image_is_the_mfs_solution_to_the_walls_residual(solved):
    for name, found in solved.items():
        print(f"\n{name}: the MFS's wall residual {found['wall']:.1e}; the series' distance to it "
              f"{found['series']:.1e}; the builder's {found['builder']:.1e}")
    assert all(found["series"] <= TOLERANCE for found in solved.values())


def test_the_mfs_meets_the_wall_closely_enough_to_judge_by(solved):
    """The MFS is a valid judge only where its own wall residual is small: a few per cent of the free Stokeslet's
    largest component at the wall, beside the builder's order-one miss."""
    assert all(found["wall"] <= 5e-2 for found in solved.values())


def test_on_the_axis_the_mfs_has_the_transverse_coefficient_of_the_series(solved):
    """The first wall coefficient of a sphere pulled across the tube's axis, ``1.80436``, from the MFS (no series)."""
    found = [float(solved[f"on the axis, force along {axis}"]["coefficient"]) for axis in "xy"]
    print(f"\nthe MFS's transverse coefficient on the axis: {found[0]:.5f} along x, {found[1]:.5f} along y")
    assert all(abs(each - 1.80436) <= 1e-3 for each in found)


def test_control_the_builders_image_misses_the_mfs_on_the_force_along_the_azimuth(solved):
    """The v0.2 builder's image (``as_built``) meets the wall but is not a Stokes flow, so it misses the MFS by order
    one for a force along the source's azimuthal direction, and not for the radial one."""
    azimuthal = [found["builder"] for name, found in solved.items() if name.endswith("along y")]
    radial = solved["off the axis, force along x"]["builder"]
    print(f"\nthe builder's distance to the MFS, azimuthal forces: {np.round(azimuthal, 3)}; radial {radial:.1e}")
    assert min(azimuthal) > 100 * TOLERANCE
    assert radial <= TOLERANCE
