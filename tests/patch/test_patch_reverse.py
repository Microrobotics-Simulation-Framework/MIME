"""The patch in reverse mode (the review's B1): the own response's transpose is the hand-written one, so neither
:func:`patch_reflection_transpose` nor JAX's reverse mode through :func:`patch_reflection` transposes its chunked map,
which held every face-point pair's kernel (83 GiB at stage 1's 107^3 box).

- the dot-product identity of a column and :func:`patch_reflection_transpose`, and of ``jax.jvp`` against
  ``jax.vjp`` through the column in its given values, its forces and its force points, both reads, both dtypes, under
  the step operator with a wall in the box; control ``PATCH_PLANT=transpose_scaled``, the hand-written transpose 1 %
  off;
- the compiled temporaries of the transpose and of the vjp at 24^3 with 338 force points and every interior face's own
  field, against the forward's; control ``PATCH_PLANT=map_transposed``, the own response as the plain chunked map that
  JAX transposes (the PR's head before this fix: about 1 GiB here against 0.06)."""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.flow.fvm import _patch
from mime.nodes.flow.fvm._patch import RefinementPatch, patch_reflection, patch_reflection_transpose, \
    patch_stopping, placed
from mime.nodes.flow.fvm._patch_transfers import Quadrature
from mime.nodes.flow.fvm._solve import REPORT_SIZE

#: The compiled temporaries of the transpose and the vjp may exceed the forward's by this factor at most: the PR's
#: head before the fix read 17 times at 24^3 (and 35 times at 32^3), the fix about 0.85.
MEMORY_FACTOR = 2.0


class Plane:
    region = "wall"

    def __init__(self, at):
        self.at = at

    def signed_distance(self, points):
        return self.at - points[..., 2]


def plant():
    return os.environ.get("PATCH_PLANT") or None


def plain_map(place, forces):
    """The own response as the PR's head computed it, a chunked map JAX transposes by stacking each chunk's kernel."""
    held_at, held_c, _ = _patch._chunked(place)

    def chunk(pair):
        x, component = pair
        r, diagonal, outer = _patch._kernel(place, x)
        r_c = jnp.take_along_axis(r, component[:, None, None], axis=2)[..., 0]
        along = jnp.sum(r * forces[None, :, :], axis=-1)
        return jnp.sum(diagonal * forces.T[component] + outer * r_c * along, axis=1)

    return jax.lax.map(chunk, (held_at, held_c)).reshape(-1)[:place.at.shape[0]]


@pytest.fixture
def planted(monkeypatch):
    if plant() == "map_transposed":
        monkeypatch.setattr(_patch, "_own", plain_map)
    if plant() == "transpose_scaled":
        held = _patch._own_transposed
        monkeypatch.setattr(_patch, "_own_transposed", lambda place, cotangent: 1.01 * held(place, cotangent))
    yield


@pytest.fixture(params=["float32", "float64"])
def dtype(request, planted):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    yield jnp.dtype(request.param)
    jax.config.update("jax_enable_x64", held)


def case(dtype, read, n=12, count=9, seed=7):
    """A patch of 50 um cells, a plane wall a third of the way up, ``count`` force points above it, a step of 1 ms in a
    0.1 Pa s fluid (the shift 1.26e6); for the cell read, four quadrature points about each force point."""
    rng = np.random.default_rng(seed)
    h = 5e-5
    patch = RefinementPatch((n, n, n), h, read=read)
    points = rng.uniform(0.35 * n, 0.65 * n, size=(count, 3)) * h + np.array([0.0, 0.0, 0.1 * n * h])
    quadrature = None
    if read == "cell":
        at = np.repeat(points, 4, axis=0) + rng.uniform(-0.4, 0.4, size=(4 * count, 3)) * h
        weights = rng.uniform(0.5, 1.5, size=4 * count) * h ** 2
        owners = np.repeat(np.arange(count), 4)
        quadrature = Quadrature(jnp.asarray(at, dtype=dtype), jnp.asarray(weights, dtype=dtype),
                                jnp.asarray(owners, dtype=jnp.int32),
                                jnp.asarray(np.bincount(owners, weights), dtype=dtype))
    placement = placed(patch, [Plane(0.3 * n * h)], jnp.zeros(3, jnp.int32), 0, jnp.asarray(points, dtype=dtype),
                       (0.3 * h) ** 2, 0.1, shift=1260.0 / 1e-3, quadrature=quadrature)
    return patch, placement, rng


def tolerance_of(dtype):
    return 1e-12 if dtype == jnp.float64 else None


def identity_line(dtype):
    """The dot-product identity's line on the products' scale, about ten times the clean readings' largest (8.5e-13 in
    float64 at the CG's 1e-12, 1.7e-5 in float32 at its floor); the 1 % plant reads 2e-6 and 1e-3."""
    return 1e-11 if dtype == jnp.float64 else 2e-4


def drawn(rng, patch, placement, dtype):
    grid = patch.grid()
    boundary = tuple(jnp.asarray(rng.normal(size=grid.given_count(c)) * 1e-4, dtype=dtype) for c in range(3))
    forces = jnp.asarray(rng.normal(size=placement.points.shape) * 1e-6, dtype=dtype)
    cotangent = jnp.asarray(rng.normal(size=placement.points.shape), dtype=dtype)
    return boundary, forces, cotangent


def pairing(a, b):
    return sum(float(np.vdot(np.asarray(x, np.float64), np.asarray(y, np.float64)))
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))


def size_of(*trees):
    return math.sqrt(sum(float(np.vdot(np.asarray(x, np.float64), np.asarray(x, np.float64))) for tree in trees
                         for x in jax.tree.leaves(tree)))


@pytest.mark.parametrize("read", ["cell", "point"])
def test_the_transpose_and_the_vjp_are_the_columns_transposes(dtype, read):
    """Each of the column's inputs on its own scale (the given values', the forces' and the force points'), so that the
    own response's transpose is not lost under the given values' larger share."""
    patch, placement, rng = case(dtype, read)
    boundary, forces, cotangent = drawn(rng, patch, placement, dtype)
    zero_b = tuple(jnp.zeros_like(b) for b in boundary)
    stopping = patch_stopping(patch, dtype, tolerance=tolerance_of(dtype))
    tap = jnp.zeros(REPORT_SIZE, dtype=dtype)

    def column(b, f, p):
        return patch_reflection(patch, placement._replace(points=p), b, f, None, stopping, tap).values

    given, force, _ = jax.jit(lambda c: patch_reflection_transpose(patch, placement, c, None, stopping, tap))(cotangent)
    _, pull = jax.vjp(column, boundary, forces, placement.points)
    back = jax.jit(pull)(cotangent)
    steps = {"given": (boundary, jnp.zeros_like(forces), jnp.zeros_like(placement.points)),
             "forces": (zero_b, forces, jnp.zeros_like(placement.points)),
             "points": (zero_b, jnp.zeros_like(forces),
                        jnp.asarray(rng.normal(size=placement.points.shape) * 1e-7, dtype=dtype))}
    transposed = {"given": given, "forces": force}
    for k, (name, direction) in enumerate(steps.items()):
        if name != "points":
            values = jax.jit(column)(direction[0], direction[1], placement.points)
            left, right = pairing(values, cotangent), pairing(direction[k], transposed[name])
            scale = size_of(values) * size_of(cotangent) + size_of(direction[k]) * size_of(transposed[name])
            assert abs(left - right) <= identity_line(dtype) * scale, ("transpose", name, abs(left - right) / scale)
        _, tangent = jax.jit(lambda d: jax.jvp(column, (boundary, forces, placement.points), d))(direction)
        left, right = pairing(tangent, cotangent), pairing(direction[k], back[k])
        scale = size_of(tangent) * size_of(cotangent) + size_of(direction[k]) * size_of(back[k])
        assert abs(left - right) <= identity_line(dtype) * scale, ("vjp", name, abs(left - right) / scale)


def test_the_transpose_and_the_vjp_hold_no_more_than_the_forwards_temporaries(planted):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        patch, placement, _ = case(jnp.float64, "point", n=24, count=338, seed=3)
        grid = patch.grid()
        stopping = patch_stopping(patch, jnp.float64)
        tap = jnp.zeros(REPORT_SIZE)
        given = tuple(jax.ShapeDtypeStruct((grid.given_count(c),), jnp.float64) for c in range(3))
        forces = jax.ShapeDtypeStruct((338, 3), jnp.float64)

        def forward(p, b, f):
            return patch_reflection(patch, p, b, f, None, stopping, tap).values

        def transpose(p, c):
            return patch_reflection_transpose(patch, p, c, None, stopping, tap)[:2]

        def reverse(p, b, f, c):
            _, pull = jax.vjp(lambda bb, ff: forward(p, bb, ff), b, f)
            return pull(c)

        def temporaries(fn, *args):
            return jax.jit(fn).lower(*args).compile().memory_analysis().temp_size_in_bytes

        base = temporaries(forward, placement, given, forces)
        for name, fn, args in (("transpose", transpose, (placement, forces)),
                               ("vjp", reverse, (placement, given, forces, forces))):
            found = temporaries(fn, *args)
            assert found <= MEMORY_FACTOR * base, (name, found / 2 ** 30, base / 2 ** 30)
    finally:
        jax.config.update("jax_enable_x64", held)
