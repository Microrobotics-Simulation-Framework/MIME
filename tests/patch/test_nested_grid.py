"""The nested grid kind (every face given) and the patch's own field, checked on their own: the projector exact,
idempotent and self-adjoint, the operator symmetric, the lift divergence-free with its given values kept; an exterior
Stokeslet's field reproduced inside the box at second order (no wall, its exact values on every face), with its control
(the given values 1 % off, which the interior cannot lose); the Brinkman difference's series and closed form against an
exact rational sum, judged on the Stokes part's scale (the difference is a cancellation); the patch's own response
against the singular Brinkman Stokeslet written again, far from the blob, and the steady limit's regularised Stokeslet."""

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.flow.fvm._nested import NestedMACGrid
from mime.nodes.flow.fvm._patch import RefinementPatch, brinkman_difference, own_response, patch_reflection, \
    patch_stopping, placed
from mime.nodes.flow.fvm._solve import REPORT_SIZE


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    yield jnp.dtype(request.param)
    jax.config.update("jax_enable_x64", held)


@pytest.fixture
def float64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jnp.dtype("float64")
    jax.config.update("jax_enable_x64", held)


def eps_of(dtype):
    return float(jnp.finfo(dtype).eps)


@pytest.mark.parametrize("cells", [(5, 6, 7), (8, 8, 8)])
def test_the_projector_is_exact_idempotent_and_self_adjoint_and_the_operator_symmetric(dtype, cells):
    grid = NestedMACGrid(cells, 2.5e-5)
    rng = np.random.default_rng(1)
    a = tuple(jnp.asarray(rng.normal(size=grid.interior_shape(c)), dtype=dtype) for c in range(3))
    b = tuple(jnp.asarray(rng.normal(size=grid.interior_shape(c)), dtype=dtype) for c in range(3))
    pa = grid.project(a)
    size = max(float(jnp.max(jnp.abs(x))) for x in a)
    line = 200 * eps_of(dtype)
    assert max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(grid.project(pa), pa)) <= line * size
    assert float(jnp.max(jnp.abs(grid.divergence(pa)))) * grid.spacing <= line * size
    scale = float(grid.inner_magnitude(a, b))
    assert abs(float(grid.inner(grid.project(a), b) - grid.inner(a, grid.project(b)))) <= line * scale
    lap_scale = float(grid.inner(grid.laplacian_magnitude(tuple(jnp.abs(x) for x in a)), tuple(jnp.abs(x) for x in b)))
    assert abs(float(grid.inner(grid.laplacian(a), b) - grid.inner(a, grid.laplacian(b)))) <= line * lap_scale


def test_the_lift_keeps_the_given_values_made_compatible_and_is_divergence_free(dtype):
    grid = NestedMACGrid((6, 7, 8), 2.5e-5)
    rng = np.random.default_rng(2)
    values = tuple(jnp.asarray(rng.normal(size=grid.given_count(c)), dtype=dtype) for c in range(3))
    fixed = grid.compatible(values)
    lift = grid.lifted(fixed)
    line = 400 * eps_of(dtype)
    assert max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(grid.given_values(lift), fixed)) == 0.0
    assert float(jnp.max(jnp.abs(grid.divergence_padded(lift)))) * grid.spacing <= line
    assert abs(float(grid.flux_share(fixed))) <= line
    assert abs(float(grid.flux_share(values))) > 1e-3


def stokeslet(points, source, force, viscosity):
    """The singular Stokeslet's velocity at ``points`` of ``force`` at ``source``, written again."""
    r = points - source
    d = np.linalg.norm(r, axis=-1, keepdims=True)
    return (force / d + r * (r @ force)[..., None] / d ** 3) / (8.0 * math.pi * viscosity)


def exterior_error(cells, scale_given=1.0):
    """The patch with no wall and no force, its faces given an exterior Stokeslet's exact values: the largest error
    of its interior field against the same Stokeslet, over the field's largest value."""
    side = 1e-3
    h = side / cells
    patch = RefinementPatch((cells, cells, cells), h, read="point")
    grid = patch.grid()
    source = np.array([-0.6, 0.4, 0.5]) * side
    force = np.array([0.3, -0.2, 1.0]) * 1e-6
    far = jnp.asarray([[0.5 * side, 0.5 * side, 0.5 * side]])
    placement = placed(patch, [], jnp.zeros(3, jnp.int32), 0, far, 1e-12, 1e-3)
    given = tuple(jnp.asarray(scale_given * stokeslet(np.asarray(grid.given_points(c, jnp.float64)), source, force,
                                                      1e-3)[:, c]) for c in range(3))
    solved = patch_reflection(patch, placement, given, jnp.zeros((1, 3)), None,
                              patch_stopping(patch, jnp.float64, tolerance=1e-12), jnp.zeros(REPORT_SIZE))
    errors, sizes = [], []
    for c in range(3):
        exact = stokeslet(np.asarray(grid.face_points(c, jnp.float64)), source, force, 1e-3)[..., c]
        errors.append(np.max(np.abs(np.asarray(solved.field[c]) - exact)))
        sizes.append(np.max(np.abs(exact)))
    return max(errors) / max(sizes)


def test_an_exterior_stokeslet_is_reproduced_inside_at_second_order(float64):
    coarse, fine = exterior_error(8), exterior_error(16)
    assert fine < 0.01
    assert coarse / fine > 3.0, (coarse, fine)


def test_control_given_values_one_percent_off_are_not_lost_inside(float64):
    assert exterior_error(16, scale_given=1.01) > 5e-3


def exact_difference(big_r, terms=200):
    big_r = Fraction(big_r)
    fa, fb, power = Fraction(0), Fraction(0), Fraction(1)
    for m in range(1, terms):
        sign = (-1) ** m
        f = [Fraction(1, math.factorial(m + k)) for k in range(3)]
        fa += 2 * sign * (f[0] - f[1] + f[2]) * power
        fb += -2 * sign * (f[0] - 3 * f[1] + 3 * f[2]) * power
        power *= big_r
    return float(fa), float(fb)


def test_the_brinkman_difference_against_an_exact_sum_on_the_stokes_parts_scale(dtype):
    """Below and above the series' edge and across it; the error over the Stokes part's size (1/R in these units)."""
    radii = np.array([1e-4, 1e-2, 0.3, 0.999, 1.0, 1.001, 2.0, 5.0, 20.0])
    found = brinkman_difference(jnp.asarray(radii, dtype=dtype))
    exact = np.array([exact_difference(r) for r in radii])
    scale = np.maximum(1.0, 1.0 / radii)
    for k in range(2):
        assert np.max(np.abs(np.asarray(found[k], dtype=np.float64) - exact[:, k]) / scale) <= 64 * eps_of(dtype)


def singular_brinkman(points, source, force, viscosity, alpha):
    """The singular Brinkman Stokeslet, ``[A F / r + B r (r . F) / r^3] / (8 pi mu)``, written again."""
    r = points - source
    d = np.linalg.norm(r, axis=-1, keepdims=True)
    big = alpha * d
    a = 2.0 * np.exp(-big) * (1.0 + 1.0 / big + 1.0 / big ** 2) - 2.0 / big ** 2
    b = -2.0 * np.exp(-big) * (1.0 + 3.0 / big + 3.0 / big ** 2) + 6.0 / big ** 2
    return (a * force / d + b * r * (r @ force)[..., None] / d ** 3) / (8.0 * math.pi * viscosity)


@pytest.mark.parametrize("shift", [0.0, 2.52e6])
def test_the_own_response_is_the_brinkman_stokeslet_far_from_the_blob(float64, shift):
    """A plane wall through a 12-cell patch; one force point 6 cells above it; the shell's faces against the singular
    Brinkman Stokeslet (the steady Stokeslet at shift zero), where the blob is a thousandth of the distance."""
    h = 5e-5
    patch = RefinementPatch((12, 12, 12), h, read="point")

    class Plane:
        region = "wall"

        def signed_distance(self, points):
            return 3.0 * h - points[..., 2]

    source = np.array([[6.2 * h, 5.9 * h, 9.0 * h]])
    viscosity = 0.05
    placement = placed(patch, [Plane()], jnp.zeros(3, jnp.int32), 0, jnp.asarray(source), (1e-3 * 3 * h) ** 2,
                       viscosity, shift=shift)
    force = np.array([[0.2, -0.4, 1.0]]) * 1e-6
    found = np.asarray(own_response(placement, jnp.asarray(force)))
    used = np.asarray(placement.shell_weight) > 0
    at = np.asarray(placement.shell_points)[used]
    comp = np.asarray(placement.shell_components)[used]
    if shift > 0:
        exact = singular_brinkman(at, source[0], force[0], viscosity, math.sqrt(shift / viscosity))
    else:
        exact = stokeslet(at, source[0], force[0], viscosity)
    exact = exact[np.arange(at.shape[0]), comp]
    scale = np.max(np.abs(exact))
    assert np.max(np.abs(found[used] - exact)) <= 1e-5 * scale


def test_the_own_field_on_every_solid_face_and_not_a_shell_four_cells_deep(float64):
    """Every face where the band or the solid is takes the own field: a shell 4 cells deep into a plane's solid, on a patch
    9 cells deep in it, moves the reads by tens of per cent (the pressure carries the solid's constraint to the fluid; the
    penalty screens only the velocity). A capacity that holds every such face reads what every interior face does, and a
    capacity below them is counted, not silent."""
    h = 5e-5

    class Plane:
        region = "wall"

        def signed_distance(self, points):
            return 9.0 * h - points[..., 2]

    rng = np.random.default_rng(5)
    sources = jnp.asarray(np.array([7.0, 7.2, 11.5]) * h + rng.uniform(-1.5, 1.5, size=(12, 3)) * h)
    forces = jnp.asarray(rng.normal(size=(12, 3)) * 1e-9)

    def reads(capacity, deep=None):
        patch = RefinementPatch((14, 14, 18), h, read="point", shell_capacity=capacity)
        placement = placed(patch, [Plane()], jnp.zeros(3, jnp.int32), 0, sources, (0.3 * h) ** 2, 1e-3)
        if deep is not None:
            kept = placement.shell_points[:, 2] >= (9.0 - deep) * h
            placement = placement._replace(shell_weight=jnp.where(kept, placement.shell_weight, 0.0))
        grid = patch.grid()
        zero = tuple(jnp.zeros(grid.given_count(c)) for c in range(3))
        found = patch_reflection(patch, placement, zero, forces, None,
                                 patch_stopping(patch, jnp.float64, tolerance=1e-12), jnp.zeros(REPORT_SIZE))
        return np.asarray(found.values), int(placement.shell_needed)

    every, needed = reads(0)
    planned, _ = reads(needed)
    shallow, _ = reads(0, deep=4.0)
    scale = np.max(np.abs(every))
    assert np.max(np.abs(planned - every)) <= 1e-10 * scale
    assert np.max(np.abs(shallow - every)) > 0.05 * scale
    short = RefinementPatch((14, 14, 18), h, read="point", shell_capacity=needed // 2)
    assert int(placed(short, [Plane()], jnp.zeros(3, jnp.int32), 0, sources, (0.3 * h) ** 2, 1e-3).shell_needed) > \
        short.capacity()
