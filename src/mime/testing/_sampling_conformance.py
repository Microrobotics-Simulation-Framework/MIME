"""The sampling conformance suite: does a geometry kind's sampling (``Geometry.sampling``) spread its samples evenly,
does its hull query (``Geometry.reach``) hold its body, and does a wall kind's distance (``WallKind.distance``) say
where its wall is? Run on every registered kind, MIME's and every installed namespace's
(``tests/scene/test_sampling.py``), each check with a control that fails it on every run.

**Spacing** (:func:`sampling_quality`): each sample's nearest neighbour over the samples' mean spacing, ``sqrt(A /
N)``, the length the MRS rung's blob is a fraction of. Its 5th percentile must be at least ``NEAREST_FIFTH`` and its
least at least ``NEAREST_LEAST``. A kind's quadrature (which a sampling chooses from) puts a quarter of a sphere's
samples within half the mean spacing of another, and fails.

**Coverage**: each region's share of the samples against its share of the area, for each patch and for the strip
within ``SEAM_WIDTH`` mean spacings of where two patches meet (a cap's rim, a box's edges), and for any region a caller
labels (the screw's crests and troughs); each within ``COVERAGE_BAND`` of even, where the region expects at least
``COVERAGE_LEAST`` samples (fewer is noise, not coverage).

**Conditioning** (:func:`condition_number`): the condition number of the MRS rung's matrix on the samples, at the
blob fractions of ``CONDITION_BOUNDS``: today's blob, 0.5, and the larger fractions a smoother blob needs (1 to 2,
``plans`` of the MRS rung's later kernels); each below its bound.

**Reach** (:func:`reach_problems`): the hull query against the largest ``x . d`` over a dense quadrature of the body,
over ``DIRECTIONS`` directions: never below it (the body is held), and above it by no more than the quadrature's own
gap (its samples' distance from the farthest point), unless the kind declares its reach an envelope, which may
overstate by its declared share (``Geometry.reach_envelope``); and its gradient is the farthest point it reports,
``x . d`` of it the reach (it is homogeneous of degree one).

**Distance** (:func:`distance_problems`): a wall kind's signed distance and normal at points of the medium: the
normal a unit vector and the distance's gradient, the distance against a closed form the caller writes from the wall's
definition; free space has none."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np

from mime.core import jnp
from mime.nodes.hydro.mrs import stokeslet_matrix
from mime.registry import Geometry, Sampling, Solid, WallKind, implements
from mime.scene._sampling import nearest, quadrature_arrays, quadrature_counts
from mime.units import Magnitude

#: The nearest-neighbour thresholds, over the mean spacing: a declaration, design #195's ("the 5th percentile at
#: least about 0.7 and no pair under 0.5"). Measured on every declared kind at ``SUITE_SAMPLES`` and ``FINER_SAMPLES``:
#: the quasi-uniform samplings' 5th percentile 0.759 to 0.807 and least 0.609 to 0.730; the kinds' quadratures' 0.086 to
#: 0.358 and 0.062 to 0.358.
NEAREST_FIFTH = 0.7
NEAREST_LEAST = 0.5
#: A region's share of the samples over its share of the area within this of one: a declaration, under the 2.4 to 2.9
#: times the screw's quadrature gives its caps and above the 0.956 to 1.064 the quasi-uniform samplings' regions measured
#: (patches, seams, the screw's crests and troughs).
COVERAGE_BAND = 0.15
#: The seams' strip, within this many mean spacings of another patch: two, so that it holds whole the rows the samples
#: form along an edge, about a mean spacing apart (a strip one spacing wide cuts the second row and reads 0.81 to 0.89
#: on a cylinder whose samples are even across it).
SEAM_WIDTH = 2.0
#: The fewest samples a region must expect (its area share of the count) for its coverage to be judged: below it the
#: count's own scatter, about its square root, is a seventh of it.
COVERAGE_LEAST = 50
#: The bound on the MRS matrix's condition number at each blob fraction, at the suite's spacing (``suite_spacing``):
#: a declaration, ten times the largest the quasi-uniform samplings of the declared kinds measured there, rounded up
#: (measured, 500 to 616 samples: 108 to 132 at 0.5, 1.5e3 to 1.65e3 at 1.0, 1.35e4 to 1.93e4 at 1.5, 8.4e4 to 1.93e5
#: at 2.0; the kinds' quadratures: the sphere's 2.3e4, 5.2e7, 6.1e10 and 3.4e13, the screw's 701, 4.7e4, 1.9e6 and
#: 5.0e7, the cylinder's, whose caps are rings, 207 to 1.9e5).
CONDITION_BOUNDS: Mapping[float, float] = {0.5: 1.5e3, 1.0: 2.0e4, 1.5: 2.0e5, 2.0: 2.0e6}
#: The directions the reach is checked along: a Fibonacci set of 61 over the sphere, then the six axes and the eight
#: diagonals, where a box's or a cylinder's farthest set is a face, an edge or a corner.
DIRECTIONS = 61
#: The brute-force quadrature's spacing, over the kind's enclosing size, and the gap allowed between the reach and its
#: largest ``x . d``, over that spacing squared over the size: a sampled surface misses its farthest point by about
#: ``h^2 / (2 r)`` for a curvature radius ``r``. Measured: at most 1.2e-4 of the size (a capsule's), against the 5.6e-4
#: allowed.
REACH_SPACING = 1.0 / 120.0
REACH_GAP = 8.0


#: The samples the suite asks of a kind (its quadrature's count at the spacing it samples at, ``suite_spacing``): a few
#: hundred, where a 5th percentile is twenty samples deep and the matrix's condition takes about a second; and the
#: larger count its spread is checked at as well, where the interior outweighs the edges.
SUITE_SAMPLES = 500
FINER_SAMPLES = 2000


def quadrature_size(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float) -> int:
    """How many samples ``kind``'s quadrature has at ``spacing`` over its closed surface: what its sampling there
    takes."""
    patches = tuple(kind.patches)
    return len(quadrature_arrays(kind, settings, patches, quadrature_counts(kind, settings, spacing, patches))[0])


def suite_spacing(kind: Geometry, settings: Mapping[str, Magnitude], samples: int = SUITE_SAMPLES) -> float:
    """The largest spacing, to a percent, at which ``kind``'s quadrature has at least ``samples`` samples."""
    coarse = _size(kind, settings) or 1.0
    fine = coarse
    while quadrature_size(kind, settings, fine) < samples:
        fine /= 2.0
    while coarse / fine > 1.01:
        middle = math.sqrt(coarse * fine)
        coarse, fine = (coarse, middle) if quadrature_size(kind, settings, middle) >= samples else (middle, fine)
    return fine


def _parameters(settings: Mapping[str, Magnitude]) -> dict[str, Any]:
    return {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
            for name, value in settings.items()}


def sampled_points(kind: Geometry, settings: Mapping[str, Magnitude], sampling: Sampling) -> tuple[np.ndarray,
                                                                                                    np.ndarray]:
    """``sampling``'s samples of ``kind`` in float64: their points and weights."""
    with cast(Any, jax).enable_x64(True):
        held = sampling.samples(kind, _parameters(settings))
        return (np.asarray(held.points, dtype=np.float64).reshape(-1, 3),
                np.asarray(held.weights, dtype=np.float64).reshape(-1))


type Labeller = Callable[[np.ndarray, np.ndarray], Mapping[str, np.ndarray]]


@dataclass(frozen=True)
class SamplingQuality:
    """A sampling's spread: its count and mean spacing; its samples' nearest neighbours over the mean spacing (least,
    5th percentile, median, the share below half); and each region's share of the samples over its share of the area,
    with the samples it expects."""

    samples: int
    mean_spacing: float
    nearest_least: float
    nearest_fifth: float
    nearest_median: float
    below_half: float
    coverage: Mapping[str, float]
    expected: Mapping[str, float]


def _regions(points: np.ndarray, labels: np.ndarray, patches: Sequence[str], dense: np.ndarray,
             dense_labels: np.ndarray, strip: float, labeller: Labeller | None) -> dict[str, np.ndarray]:
    found = {f"patch {patch}": labels == i for i, patch in enumerate(patches)}
    if len(patches) > 1:
        near = np.zeros(len(points), dtype=bool)
        for i in range(len(patches)):
            others = dense[dense_labels != i]
            if len(others):
                near |= (labels == i) & (nearest(others, points)[0] < strip)
        found["seams"] = near
    if labeller is not None:
        found |= {name: np.asarray(mask, dtype=bool) for name, mask in labeller(points, labels).items()}
    return found


def sampling_quality(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float, *,
                     sampling: Sampling | None = None, labeller: Labeller | None = None) -> SamplingQuality:
    """The spread of ``kind``'s sampling at ``spacing`` over its closed surface (or of ``sampling``, a sampling of it
    written elsewhere, a control's), its coverage judged against its quadrature at the sampling's dense counts."""
    patches = tuple(kind.patches)
    held = sampling if sampling is not None else kind.sampling(settings, spacing, patches)
    points, weights = sampled_points(kind, settings, held)
    dense_counts = quadrature_counts(kind, settings, spacing / 8.0, patches)
    dense, dense_weights, dense_labels = quadrature_arrays(kind, settings, patches, dense_counts)
    _, _, own_labels = quadrature_arrays(kind, settings, held.patches, held.counts)
    labels = own_labels[held.chosen] if held.chosen is not None else own_labels
    count = len(points)
    mean = math.sqrt(float(np.sum(weights)) / count)
    closest = nearest(points, points, k=2)[0][:, 1] / mean
    samples = _regions(points, labels, patches, dense, dense_labels, SEAM_WIDTH * mean, labeller)
    areas = _regions(dense, dense_labels, patches, dense, dense_labels, SEAM_WIDTH * mean, labeller)
    total = float(np.sum(dense_weights))
    coverage: dict[str, float] = {}
    expected: dict[str, float] = {}
    for name, mask in areas.items():
        share = float(np.sum(dense_weights[mask])) / total
        expected[name] = share * count
        coverage[name] = float(np.mean(samples[name])) / share if share > 0.0 else math.nan
    return SamplingQuality(count, mean, float(closest.min()), float(np.percentile(closest, 5.0)),
                           float(np.median(closest)), float(np.mean(closest < 0.5)), coverage, expected)


def sampling_problems(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float, *,
                      sampling: Sampling | None = None, labeller: Labeller | None = None) -> list[str]:
    """Every way the sampling's spread fails the suite's thresholds; and its weights' sum against the patches'
    area."""
    quality = sampling_quality(kind, settings, spacing, sampling=sampling, labeller=labeller)
    label = kind.label()
    found: list[str] = []
    if not quality.nearest_fifth >= NEAREST_FIFTH:
        found.append(f"{label}: the 5th percentile of its samples' nearest neighbours is {quality.nearest_fifth:.3f} "
                     f"of the mean spacing, below {NEAREST_FIFTH}")
    if not quality.nearest_least >= NEAREST_LEAST:
        found.append(f"{label}: its closest pair of samples is {quality.nearest_least:.3f} of the mean spacing apart, "
                     f"below {NEAREST_LEAST} ({quality.below_half:.1%} of its samples are closer than half to another)")
    for name, ratio in quality.coverage.items():
        if quality.expected[name] >= COVERAGE_LEAST and not abs(ratio - 1.0) <= COVERAGE_BAND:
            found.append(f"{label}: its {name} hold {ratio:.3f} times their share of the area in samples, beyond "
                         f"{COVERAGE_BAND:.0%} of even")
    held = sampling if sampling is not None else kind.sampling(settings, spacing, tuple(kind.patches))
    _, weights = sampled_points(kind, settings, held)
    area = math.fsum(kind.area(settings, patch) for patch in held.patches)
    if not abs(float(np.sum(weights)) - area) <= 1e-12 * area:
        found.append(f"{label}: its samples' weights sum to {float(np.sum(weights))!r} m^2, and its patches' area is "
                     f"{area!r} m^2")
    return found


def condition_number(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float, blob: float, *,
                     sampling: Sampling | None = None) -> float:
    """The 2-norm condition number of the MRS rung's matrix (``stokeslet_matrix``, unit viscosity) on the sampling's
    samples at ``spacing``, its blob ``blob`` times their mean spacing: the ratio of its extreme eigenvalues, in
    float64 (``inf`` where its least is not above zero)."""
    held = sampling if sampling is not None else kind.sampling(settings, spacing, tuple(kind.patches))
    points, weights = sampled_points(kind, settings, held)
    with cast(Any, jax).enable_x64(True):
        eps_squared = blob * blob * float(np.sum(weights)) / len(weights)
        matrix = np.asarray(stokeslet_matrix(jnp.asarray(points), eps_squared, 1.0), dtype=np.float64)
    values = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    return float(values[-1] / values[0]) if values[0] > 0.0 else math.inf


def conditioning_problems(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float, *,
                          sampling: Sampling | None = None) -> list[str]:
    found: list[str] = []
    for blob, bound in CONDITION_BOUNDS.items():
        held = condition_number(kind, settings, spacing, blob, sampling=sampling)
        if not held <= bound:
            found.append(f"{kind.label()}: the MRS matrix on its samples at blob {blob} has a condition number of "
                         f"{held:.3g}, above {bound:.3g}")
    return found


def directions() -> np.ndarray:
    """The reach's directions: ``DIRECTIONS`` on a Fibonacci sphere, then the six axes and the eight diagonals."""
    i = np.arange(DIRECTIONS) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / DIRECTIONS)
    turn = math.pi * (1.0 + math.sqrt(5.0)) * i
    spread = np.stack([np.sin(polar) * np.cos(turn), np.sin(polar) * np.sin(turn), np.cos(polar)], axis=1)
    axes = np.concatenate([np.eye(3), -np.eye(3)])
    corners = np.asarray([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                         dtype=np.float64) / math.sqrt(3.0)
    return np.concatenate([spread, axes, corners])


def _size(kind: Geometry, settings: Mapping[str, Magnitude]) -> float | None:
    """The kind's enclosing size, which the reach's brute force is scaled by: a solid's enclosing radius, or its
    radial reach."""
    if isinstance(kind, Solid):
        return float(kind.enclosing_radius(settings))
    if implements(kind, "radial_reach"):
        return float(kind.radial_reach(settings))
    return None


def reach_problems(kind: Geometry, settings: Mapping[str, Magnitude], *,
                   reach: Callable[[Mapping[str, Any], Any], Any] | None = None) -> list[str]:
    """Every way ``kind``'s reach (or ``reach``, a control's) fails against a dense quadrature of the body over
    :func:`directions`; for a kind that samples no surface, nothing (its reach is held to its closed form by its own
    test)."""
    if not kind.patches:
        return []
    coarse = _size(kind, settings)
    if coarse is None:
        return [f"{kind.label()}: it gives no size (an enclosing radius or a radial reach) to scale the brute force by"]
    patches = tuple(kind.patches)
    spacing = coarse * REACH_SPACING
    points, _, _ = quadrature_arrays(kind, settings, patches, quadrature_counts(kind, settings, spacing, patches))
    gap = REACH_GAP * spacing * spacing / coarse
    envelope = float(kind.reach_envelope)
    query = reach if reach is not None else kind.reach
    label = kind.label()
    found: list[str] = []
    with cast(Any, jax).enable_x64(True):
        parameters = _parameters(settings)

        def reached(d: Any) -> Any:
            return query(parameters, d)

        def value_and_point(d: Any) -> tuple[Any, Any]:
            return cast(Any, jax).value_and_grad(reached)(d)

        for d in cast(list[np.ndarray], list(directions())):
            value, point = (np.asarray(x, dtype=np.float64) for x in value_and_point(jnp.asarray(d)))
            projected = points @ d
            brute = float(np.max(projected))
            rounding = 64.0 * float(np.finfo(np.float64).eps) * coarse
            if not float(value) >= brute - rounding:
                found.append(f"{label}: its reach along {np.round(d, 3).tolist()} is {float(value):.9g} m, below "
                             f"{brute:.9g} m, a point of its body: the hull query does not hold the body")
            allowed = gap + envelope * coarse
            if not float(value) <= brute + allowed + rounding:
                found.append(f"{label}: its reach along {np.round(d, 3).tolist()} is {float(value):.9g} m, above its "
                             f"body's farthest {brute:.9g} m by {float(value) - brute:.3g} m, beyond the "
                             f"{allowed:.3g} m allowed")
            if not abs(float(point @ d) - float(value)) <= rounding + 1e-12 * abs(float(value)):
                found.append(f"{label}: along {np.round(d, 3).tolist()} its reach's gradient, {point.tolist()}, does "
                             f"not reach as far as its reach says ({float(point @ d):.9g} m, against "
                             f"{float(value):.9g} m)")
    return found


def distance_problems(wall: WallKind, parameters: Mapping[str, Any], points: np.ndarray,
                      closed_form: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None) -> list[str]:
    """Every way ``wall``'s distance at ``points`` (each of the medium, world frame) fails: its normal not a unit
    vector or not its distance's gradient, or the distance and normal not ``closed_form``'s (the caller's, written
    from the wall's definition); and, for a wall kind that matches no region, a distance at all."""
    label = wall.label()
    with cast(Any, jax).enable_x64(True):
        held = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                for name, value in parameters.items()}
        first = wall.distance(held, jnp.asarray(points[0], dtype=jnp.float64))
        if wall.matches is None:
            return [] if first is None else [f"{label}: it matches no region, so it has no wall, and gives a "
                                             f"distance"]
        if first is None:
            return [f"{label}: it matches the region {wall.matches!r} and gives no distance"]

        def distance(point: Any) -> Any:
            found = wall.distance(held, point)
            assert found is not None
            return found.distance

        def normal(point: Any) -> Any:
            found = wall.distance(held, point)
            assert found is not None
            return found.normal

        at = jnp.asarray(points, dtype=jnp.float64)
        values = np.asarray(cast(Any, jax).vmap(distance)(at), dtype=np.float64)
        normals = np.asarray(cast(Any, jax).vmap(normal)(at), dtype=np.float64)
        gradients = np.asarray(cast(Any, jax).vmap(cast(Any, jax).grad(distance))(at), dtype=np.float64)
    found: list[str] = []
    rounding = 64.0 * float(np.finfo(np.float64).eps)
    scale = float(np.max(np.abs(points))) + float(np.max(np.abs(values)))
    if not np.all(np.abs(np.linalg.norm(normals, axis=1) - 1.0) <= rounding * 16):
        found.append(f"{label}: its normal is not a unit vector at every point")
    if not np.all(np.abs(gradients - normals) <= rounding * 16):
        found.append(f"{label}: its normal is not its distance's gradient (apart by up to "
                     f"{float(np.max(np.abs(gradients - normals))):.3g})")
    if closed_form is not None:
        distances, directions_ = closed_form(points)
        if not np.all(np.abs(values - distances) <= rounding * scale):
            found.append(f"{label}: its distance differs from the closed form by up to "
                         f"{float(np.max(np.abs(values - distances))):.3g} m")
        if not np.all(np.abs(normals - directions_) <= rounding * 16):
            found.append(f"{label}: its normal differs from the closed form's by up to "
                         f"{float(np.max(np.abs(normals - directions_))):.3g}")
    return found


__all__ = ["CONDITION_BOUNDS", "COVERAGE_BAND", "COVERAGE_LEAST", "DIRECTIONS", "FINER_SAMPLES", "NEAREST_FIFTH",
           "NEAREST_LEAST", "REACH_GAP", "REACH_SPACING", "SEAM_WIDTH", "SUITE_SAMPLES", "quadrature_size",
           "SamplingQuality", "condition_number", "conditioning_problems", "directions", "distance_problems",
           "reach_problems", "sampled_points", "sampling_problems", "sampling_quality", "suite_spacing"]
