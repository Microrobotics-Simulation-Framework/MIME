"""The plane wall kind's image (``mime.nodes.hydro.mrs._plane``) holds the identities that make it the image of MIME's
regularised Stokeslet above a no-slip plane, at drawn planes, sources and targets at a body's physical scales (a body of
size ``a`` from a micrometre to a centimetre; a plane along any normal within ten sizes of the origin; sources 0.02 to 2
``a`` above it; blobs 0.01 to 0.5 ``a``), in float64. Each is judged on the scale of its inputs' rounding, never of its
result (``wall_rounding``, ``fluid_rounding``, ``image_rounding``), and each has a planted error that fails it on every
draw:

- **no slip:** the free Stokeslet of the source's blob plus its image vanishes on the plane, against the free
  Stokeslet's own velocity at the target (a wall target within two heights of the source's foot, where every term is
  of the free Stokeslet's order). Plants: each of the four terms with its sign flipped; the image given no blob
  (Blake's singular image against the regularised free Stokeslet: a blob dropped on its way).
- **no divergence:** the total's divergence for each of the three unit forces at a fluid point, against its gradient's
  largest entry (a divergence is linear in the force, so one force could miss what another shows). Each term is
  divergence-free on its own, so a sign flip cannot fail it; the plant is a term that is not, the Stokeslet's ``R
  R^T`` part alone, at a millionth of the image.
- **reciprocity:** ``G(x, y) = G(y, x)^T`` for two points in the medium with the same blob. Plant: a term that is not
  reciprocal even with the target the source, a millionth of the image.
- **Blake's limit:** with no blob the image is Blake's singular image (``plane_oracles.blake``, written out again from
  its formula, sharing no code with MIME) to rounding, and the difference falls as the blob squared. Plants: the
  image's doublet with its sign flipped (the comparison with no blob must fail it), and an error of the blob's own
  order (the rate must fail it).
- **its frame:** a plane along any normal at any offset, the normal given at any length, is the canonical image
  turned and moved.

Blake's image as the oracle writes it is held by the identity it serves: with the singular Stokeslet it vanishes on the
wall. The phase's measurement (1,000 draws each, the residual in units of the inputs' rounding): no slip at most 4.4,
the divergence 3.3, the reciprocity 92, Blake's image with no blob 1.1, the turned frame 2.5, the rate 99.97 to
100.00; each line holds a factor of ten or more over the largest. The plants' least, in the same units: the flipped
terms 2.1e10 (the rotlet) and above, the blob dropped 2.6e9, the divergence 1.5e6, the reciprocity 3.5e8, the
doublet against Blake's 1.2e11; the planted rate at most 10.1."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from plane_oracles import blake, frame, free_stokeslet

from mime.nodes.hydro.mrs._plane import image_terms, plane_image

EPS = float(np.finfo(np.float64).eps)
#: Each identity's line, in units of its inputs' rounding.
NO_SLIP = 64.0
DIVERGENCE = 64.0
RECIPROCITY = 1024.0
LIMIT = 32.0
TERMS = ("stokeslet", "doublet", "dipole", "rotlet")
CANONICAL = {"normal": (0.0, 0.0, 1.0), "offset": 0.0}


def size_of(plane, *points) -> float:
    """The largest coordinate of the points and of the plane's foot (m)."""
    return max(max(float(np.abs(p).max()) for p in points), abs(plane["offset"]))


def height(plane, point) -> float:
    """A point's (``(1, 3)``) height above the plane (m)."""
    return float(np.dot(point[0], frame(np.asarray(plane["normal"]))[0]) - plane["offset"])


def mirror_of(plane, point) -> np.ndarray:
    n = frame(np.asarray(plane["normal"]))[0]
    return point[0] - 2.0 * height(plane, point) * n


def wall_rounding(plane, source, target) -> float:
    """``eps (1 + L / h)``, ``L`` the largest coordinate and ``h`` the source's height: the rounding of a kernel from
    the source to a wall target, whose inputs are rounded at their own size while it varies over the height."""
    return EPS * (1.0 + size_of(plane, source, target) / height(plane, source))


def fluid_rounding(plane, x, y, eps_squared) -> float:
    """``eps (1 + L / d)``, ``d`` the points' least height and their distance with the blob, ``sqrt(|x - y|^2 +
    eps^2)``: the rounding of the whole kernel between two points in the medium."""
    apart = math.sqrt(float(np.sum((x[0] - y[0]) ** 2)) + eps_squared)
    return EPS * (1.0 + size_of(plane, x, y) / min(height(plane, x), height(plane, y), apart))


def image_rounding(plane, x, y) -> float:
    """``eps (1 + L / d)``, ``d`` the points' least height: the rounding of the image alone, whose size is set by the
    distance to the source's mirror, at least twice its height."""
    return EPS * (1.0 + size_of(plane, x, y) / min(height(plane, x), height(plane, y)))


@st.composite
def planes(draw):
    """A body's size ``a`` (m), a plane (its normal at any length, its offset within ten sizes of the origin), and the
    plane's unit normal and two directions along it."""
    a = draw(st.floats(1e-6, 1e-2))
    normal = np.array([draw(st.floats(-1.0, 1.0)) for _ in range(3)])
    if np.linalg.norm(normal) < 0.2:
        normal = np.array([0.0, 0.0, 1.0])
    length = draw(st.floats(0.5, 2.0))
    offset = draw(st.floats(-10.0, 10.0)) * a
    n, u, v = frame(normal)
    return a, {"normal": tuple(float(c) for c in length * n), "offset": offset}, n, u, v


@st.composite
def wall_pairs(draw):
    """A plane, a source ``h`` above it, a target on it within two heights of the source's foot, and a blob's
    ``eps^2``."""
    a, plane, n, u, v = draw(planes())
    h = draw(st.floats(0.02, 2.0)) * a
    eps = draw(st.floats(0.01, 0.5)) * a
    along = np.array([draw(st.floats(-1.0, 1.0)), draw(st.floats(-1.0, 1.0))]) * a
    foot = plane["offset"] * n + along[0] * u + along[1] * v
    rho, phi = draw(st.floats(0.0, 2.0)) * h, draw(st.floats(0.0, 2 * math.pi))
    target = foot + rho * (math.cos(phi) * u + math.sin(phi) * v)
    return plane, (foot + h * n)[None, :], target[None, :], eps * eps


@st.composite
def fluid_pairs(draw):
    """A plane, two points in the medium 0.02 to 3 sizes above it and within three sizes of each other along it, and a
    blob's ``eps^2``."""
    a, plane, n, u, v = draw(planes())
    points = []
    for _ in range(2):
        along = np.array([draw(st.floats(-1.5, 1.5)), draw(st.floats(-1.5, 1.5))]) * a
        points.append(plane["offset"] * n + along[0] * u + along[1] * v + draw(st.floats(0.02, 3.0)) * a * n)
    eps = draw(st.floats(0.01, 0.5)) * a
    return plane, points[0][None, :], points[1][None, :], eps * eps


def image(targets, sources, plane, eps_squared, *, flip=None):
    """The image at unit viscosity, ``(T, S, 3, 3)``, the term ``flip`` (if any) with its sign flipped."""
    terms = image_terms(jnp.asarray(targets), jnp.asarray(sources), plane, eps_squared)._asdict()
    return np.asarray(sum((-held if name == flip else held) for name, held in terms.items())) / (8.0 * math.pi)


def local(plane, *points):
    """The points in the plane's own frame: its normal +z, its foot at the origin."""
    n, u, v = frame(np.asarray(plane["normal"]))
    turn = np.stack([u, v, n])
    return turn, [(p - plane["offset"] * n) @ turn.T for p in points]


def slip(plane, source, target, eps_squared, *, flip=None, image_block=None):
    """The wall's velocity per unit force (the block's largest entry) over the free Stokeslet's own there, in units of
    the inputs' rounding."""
    free = free_stokeslet(target, source, eps_squared)[0, 0]
    held = image_block if image_block is not None else image(target, source, plane, eps_squared, flip=flip)[0, 0]
    return float(np.abs(free + held).max() / np.abs(free).max()) / wall_rounding(plane, source, target)


@settings(deadline=None)
@given(wall_pairs())
def test_the_free_stokeslet_and_its_image_vanish_on_the_plane(drawn):
    found = slip(*drawn)
    assert found <= NO_SLIP, found


@settings(deadline=None)
@given(wall_pairs(), st.sampled_from(TERMS))
def test_control_each_term_with_its_sign_flipped_slips_on_every_draw(drawn, term):
    found = slip(*drawn, flip=term)
    assert found > NO_SLIP, (term, found)


@settings(deadline=None)
@given(wall_pairs())
def test_control_the_image_given_no_blob_slips_on_every_draw(drawn):
    """A blob dropped on its way: the image with no blob against the regularised free Stokeslet."""
    plane, source, target, eps_squared = drawn
    singular = np.asarray(plane_image(jnp.asarray(target), jnp.asarray(source), plane, 1.0, 0.0).matrix)[0, 0]
    found = slip(plane, source, target, eps_squared, image_block=singular)
    assert found > NO_SLIP, found


def field(plane, source, eps_squared, planted=None):
    """The whole kernel of a force at ``source`` as a function of the target, ``3 x 3`` (the velocity by the force),
    with ``planted(x)`` added, for ``jax.jacfwd``."""

    def g(x):
        r = x - jnp.asarray(source[0])
        r2 = jnp.sum(r * r)
        free = ((r2 + 2 * eps_squared) * jnp.eye(3) + r[:, None] * r[None, :]) / (8 * math.pi * (r2 + eps_squared) ** 1.5)
        held = sum(image_terms(x[None, :], jnp.asarray(source), plane, eps_squared))[0, 0] / (8 * math.pi)
        return free + held + (planted(x) if planted is not None else 0.0)

    return g


def divergence(plane, source, target, eps_squared, planted=None):
    """The divergence of each unit force's velocity, the largest of the three, over the gradient's largest entry, in
    units of the inputs' rounding."""
    gradient = np.asarray(jax.jacfwd(field(plane, source, eps_squared, planted))(jnp.asarray(target[0])))
    found = float(np.abs(np.einsum("iji->j", gradient)).max() / np.abs(gradient).max())
    return found / fluid_rounding(plane, source, target, eps_squared)


@settings(deadline=None)
@given(fluid_pairs())
def test_the_free_stokeslet_and_its_image_have_no_divergence(drawn):
    found = divergence(*drawn)
    assert found <= DIVERGENCE, found


@settings(deadline=None)
@given(fluid_pairs())
def test_control_a_term_with_a_divergence_a_millionth_of_the_image_fails_on_every_draw(drawn):
    """The image's Stokeslet's ``R R^T`` part alone, whose divergence is ``R / |R|^3``, never zero (the mirror lies at
    least twice the source's height away), at a millionth of its size."""
    plane, source, target, eps_squared = drawn
    mirror = jnp.asarray(mirror_of(plane, source))

    def planted(x):
        r = x - mirror
        return 1e-6 * r[:, None] * r[None, :] / (8 * math.pi * jnp.sum(r * r) ** 1.5)

    found = divergence(plane, source, target, eps_squared, planted)
    assert found > DIVERGENCE, found


def kernel(plane, x, y, eps_squared, planted=0.0):
    """The whole kernel ``G(x, y)`` at unit viscosity: the free Stokeslet of the blob plus the image, and a planted term
    ``planted`` times ``n u^T / |R|^2`` (``n`` the normal, ``u`` a direction along the plane, ``R`` the target less the
    source's mirror), not reciprocal even with the target the source."""
    held = free_stokeslet(x, y, eps_squared)[0, 0] + image(x, y, plane, eps_squared)[0, 0]
    if planted:
        n, u, _ = frame(np.asarray(plane["normal"]))
        r = x[0] - mirror_of(plane, y)
        held = held + planted * np.outer(n, u) / (8 * math.pi * float(np.dot(r, r)))
    return held


def reciprocity(plane, x, y, eps_squared, planted=0.0):
    forward, backward = kernel(plane, x, y, eps_squared, planted), kernel(plane, y, x, eps_squared, planted)
    found = float(np.abs(forward - backward.T).max() / np.abs(forward).max())
    return found / fluid_rounding(plane, x, y, eps_squared)


@settings(deadline=None)
@given(fluid_pairs())
def test_the_kernel_is_reciprocal(drawn):
    found = reciprocity(*drawn)
    assert found <= RECIPROCITY, found


@settings(deadline=None)
@given(fluid_pairs())
def test_control_a_non_reciprocal_term_a_millionth_of_the_kernel_fails_on_every_draw(drawn):
    found = reciprocity(*drawn, planted=1e-6)
    assert found > RECIPROCITY, found


def against_blake(plane, x, y, *, flip=None):
    """The image with no blob against Blake's singular image in the plane's own frame, in units of the inputs'
    rounding."""
    _, (at, of) = local(plane, x, y)
    found = image(at, of, CANONICAL, 0.0, flip=flip)[0, 0]
    oracle = blake(at, of)[0, 0]
    return float(np.abs(found - oracle).max() / np.abs(oracle).max()) / image_rounding(plane, x, y)


@settings(deadline=None)
@given(fluid_pairs())
def test_with_no_blob_the_image_is_blakes_singular_image(drawn):
    plane, x, y, _ = drawn
    found = against_blake(plane, x, y)
    assert found <= LIMIT, found


@settings(deadline=None)
@given(fluid_pairs())
def test_control_the_image_with_its_doublet_flipped_is_not_blakes_on_every_draw(drawn):
    plane, x, y, _ = drawn
    found = against_blake(plane, x, y, flip="doublet")
    assert found > LIMIT, found


@settings(deadline=None)
@given(st.floats(0.0, 2 * math.pi), st.floats(0.0, 2.0))
def test_blakes_image_and_the_singular_stokeslet_vanish_on_the_wall(phi, rho):
    """The oracle's own check, with a source 0.3 above the wall and a target ``rho`` heights from its foot."""
    source = np.array([[0.1, -0.2, 0.3]])
    target = np.array([[0.1 + 0.3 * rho * math.cos(phi), -0.2 + 0.3 * rho * math.sin(phi), 0.0]])
    free = free_stokeslet(target, source, 0.0)[0, 0]
    found = float(np.abs(free + blake(target, source)[0, 0]).max() / np.abs(free).max())
    assert found <= NO_SLIP * wall_rounding(CANONICAL, source, target), found


def limit_rate(plane, x, y, eps_squared, planted: float = 0.0) -> float:
    """How much nearer Blake's image the image comes for a blob ten times smaller: 100 for a difference of the blob's
    square; ``planted`` adds ``planted * eps / |R|`` times the singular Stokeslet at the mirror, an error of the blob's
    own order."""
    zero = image(x, y, plane, 0.0)[0, 0]
    mirror = mirror_of(plane, y)[None, :]
    scale = float(np.linalg.norm(x[0] - mirror[0]))

    def gap(e2: float) -> float:
        held = image(x, y, plane, e2)[0, 0] + planted * math.sqrt(e2) / scale * free_stokeslet(x, mirror, 0.0)[0, 0]
        return float(np.abs(held - zero).max())

    return gap(eps_squared) / gap(eps_squared / 100.0)


def small_blob(plane, x, y, eps_squared) -> float:
    """The drawn blob's ``eps^2``, at most a hundredth of the mirror's distance squared."""
    return min(eps_squared, (1e-2 * float(np.linalg.norm(x[0] - mirror_of(plane, y)))) ** 2)


@settings(deadline=None)
@given(fluid_pairs())
def test_the_image_comes_to_blakes_as_the_blob_squared(drawn):
    """Between a blob of at most a hundredth of the mirror's distance and one ten times smaller, the difference falls a
    hundredfold, to a fifth (the next order's share; measured 99.98 to 100.00)."""
    plane, x, y, eps_squared = drawn
    rate = limit_rate(plane, x, y, small_blob(plane, x, y, eps_squared))
    assert 80.0 <= rate <= 120.0, rate


@settings(deadline=None)
@given(fluid_pairs())
def test_control_an_error_of_the_blobs_order_fails_the_rate_on_every_draw(drawn):
    plane, x, y, eps_squared = drawn
    rate = limit_rate(plane, x, y, small_blob(plane, x, y, eps_squared), planted=1.0)
    assert not 80.0 <= rate <= 120.0, rate


@settings(deadline=None)
@given(fluid_pairs())
def test_a_plane_along_any_normal_at_any_offset_is_the_canonical_image_turned_and_moved(drawn):
    plane, x, y, eps_squared = drawn
    turn, (at, of) = local(plane, x, y)
    canonical = image(at, of, CANONICAL, eps_squared)[0, 0]
    found = turn @ image(x, y, plane, eps_squared)[0, 0] @ turn.T
    error = float(np.abs(found - canonical).max() / np.abs(canonical).max()) / image_rounding(plane, x, y)
    assert error <= LIMIT, error
