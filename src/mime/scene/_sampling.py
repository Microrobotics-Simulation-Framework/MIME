"""Quasi-uniform surface sampling, built from a geometry kind's own quadrature, so that any kind that gives its samples
at counts (``Geometry.samples`` and ``sample_counts``), MIME's or a lab's, gets it by one call
(:func:`quasi_uniform`); and the hull query of any kind from its samples (:func:`sampled_reach`), its own body's from
its quadrature (:func:`body_reach`).

**Why.** A kind's quadrature is a tensor grid, exact for its integrals and crowded where its coordinates meet: a
sphere's poles, a cap's centre, a cylinder's rims, the ends of Gauss-Legendre nodes. The method of regularised
Stokeslets puts an unknown at each sample, and samples far closer than the spacing make its matrix poorly conditioned
and cost accuracy (on de Jongh's FL-9, 26 to 27 % of the grid's samples are closer than half the mean spacing, and a
quasi-uniform sampling of the same count moves its free-space speed from 8.5 % to 3.5 % below a converged reference).

**How.** The kind's quadrature at ``REFINEMENT`` times finer spacing is the dense set: points exactly on the surface,
with their areas. The sampling takes as many samples as the kind's quadrature has at the spacing (so that a spacing
gives the unknowns it gave before, placed better), times the density's mean over the surface:

1. **Farthest points**: from the dense set's first point, each next one the dense point farthest from those chosen, its
   squared distance weighted by the density.
2. **Lloyd's relaxation**, ``LLOYD_STEPS`` steps: each dense point to its nearest generator, and each generator to its
   cell's centroid, weighted by area times density, in space (a centroid lies just inside a curved surface).
3. **Each generator to the dense set**: to the dense point of its own cell nearest it, so that every sample is a
   quadrature sample, on the surface exactly, and its normal the surface's, and two generators either side of an edge
   are not both drawn onto it.
4. **Its cell**: each dense point to its nearest sample; a sample's weight is its cell's area, so the weights sum to
   the surface's area exactly.

At every step squared distances within ``TIE`` of the body's squared size are one tie, broken by the lower index, so
that a choice made on one machine is made on another whose arithmetic rounds the dense points' last bits otherwise,
and on the same body scaled (a symmetric body's grid has exact ties in exact arithmetic, which rounding would
otherwise decide).

The choice is made once, when the experiment is built, from the settings in SI, in float64 whatever the run's
precision: the run takes the dense quadrature at those counts and gathers the chosen samples, so the samples move with
the kind's parameters and a study's derivative flows through them, and the build is the same in float32 and float64.
What the choice reproduces across machines is bounded by the tie rule: a dense point whose distance lies within a
rounding of a tie's edge can still fall the other way on another machine, which the saved graph's hash shows (it holds
the chosen indices), never silently.

**What it costs.** The dense set has about ``REFINEMENT^2`` points per sample; the farthest-point pass is ``O(N M)``
(compiled), the relaxation ``O(M log N)`` per step: about 0.2 s for a few hundred samples, and some 10 s for FL-9's
5,984 (``tests/scene/test_sampling.py`` records the measured times).

**It is not continuous in the kind's settings.** The choice is discrete, so a small change of a setting can choose
other samples and a result steps: on FL-9 at 416 samples the wavenumber moved by 1e-5 of itself moves the MRS
resistance by 0.37% of its diagonal, a thousand times its smooth change, and by up to 0.42% within 1e-3 (measured in
phase 7m-a's review; MIME-ANO-127). Within one build the choice is held, so a derivative is exact; a study that builds
again at moved settings sees the steps (the package's README says more)."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import jax
import numpy as np

from mime.core import jnp
from mime.registry import Density, Geometry, Samples, Sampling, uniform
from mime.units import Magnitude

#: The dense set's spacing is the sampling's over this. A declaration, from a measurement on the sphere, the capsule and
#: FL-9 at a few hundred to a thousand samples (phase 7m-a's pull request): at 6 the screw's closest pair fell to 0.48
#: of the mean spacing and its 5th percentile to 0.75, at 8 they were 0.69 and 0.78 and the sphere's 0.78 and 0.80;
#: finer costs the square in points.
REFINEMENT = 8
#: Lloyd's steps. A declaration: on FL-9 at 957 samples (phase 7m-a's pull request), 20 steps gave the spread of the
#: relaxation's fixed point, which 60 reach (the same closest pair, 0.69 of the mean spacing, and 5th percentile, 0.78;
#: the caps' coverage 1.005 against 1.013).
LLOYD_STEPS = 20
#: Squared distances within this share of the body's squared size are one tie, broken by the lower index: 1e-12, far
#: above float64's rounding of a squared distance (2e-16 of it) and far below any spacing a sampling uses (a body
#: sampled at a thousandth of its size has squared spacings 1e-6 of its squared size).
TIE = 1e-12
#: A sampling that is the kind's own quadrature, and one chosen by :func:`quasi_uniform`, by what ``Sampling.method``
#: records.
QUADRATURE, QUASI_UNIFORM = "quadrature", "quasi_uniform"

_jit: Any = cast(Any, jax).jit
_fori_loop: Any = cast(Any, jax).lax.fori_loop
#: SciPy's k-d tree, whose module ships no type hints.
_spatial: Any = cast(Any, importlib.import_module("scipy")).spatial


def nearest(points: np.ndarray, queries: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Each query's ``k`` nearest of ``points`` (a k-d tree's, exact): their distances and indices, ``(n,)`` for one
    and ``(n, k)`` for more."""
    distance, index = _spatial.cKDTree(points).query(queries, k=k)
    return np.asarray(distance, dtype=np.float64), np.asarray(index, dtype=np.int64)


def _parameters(settings: Mapping[str, Magnitude]) -> dict[str, Any]:
    return {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
            for name, value in settings.items()}


def quadrature_arrays(kind: Geometry, settings: Mapping[str, Magnitude], patches: Sequence[str],
                      counts: Sequence[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The kind's quadrature on ``patches`` at ``counts``, concatenated in order, in float64 whatever the run's
    precision: its points ``(M, 3)``, weights ``(M,)`` and each point's patch (its index in ``patches``)."""
    with cast(Any, jax).enable_x64(True):
        held = _parameters(settings)
        found = [kind.samples(held, patch, tuple(count)) for patch, count in zip(patches, counts, strict=True)]
        points = [np.asarray(s.points, dtype=np.float64).reshape(-1, 3) for s in found]
        weights = [np.asarray(s.weights, dtype=np.float64).reshape(-1) for s in found]
    labels = [np.full(len(p), i, dtype=np.int64) for i, p in enumerate(points)]
    return np.concatenate(points), np.concatenate(weights), np.concatenate(labels)


def quadrature_counts(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float,
                      patches: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(c) for c in kind.sample_counts(settings, spacing, patch)) for patch in patches)


def quadrature(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float, patches: Sequence[str]) -> Sampling:
    """The kind's own quadrature at ``spacing`` as its sampling: what MIME's kinds gave the MRS rung before 7m-a, and
    what each version kept registered from then gives, so that a folder saved with it reproduces."""
    return Sampling(tuple(patches), quadrature_counts(kind, settings, spacing, patches), QUADRATURE)


def _select(points: Any, weight: Any, quantum: Any, count: int) -> Any:
    def step(i: Any, carry: tuple[Any, Any]) -> tuple[Any, Any]:
        chosen, nearest = carry
        j = jnp.argmax(jnp.floor(nearest * weight / quantum))
        return chosen.at[i].set(j), jnp.minimum(nearest, jnp.sum((points - points[j]) ** 2, axis=-1))

    start = (jnp.zeros(count, dtype=jnp.int64), jnp.sum((points - points[0]) ** 2, axis=-1))
    return _fori_loop(1, count, step, start)[0]


_SELECT = _jit(_select, static_argnames=("count",))


def _farthest(points: np.ndarray, density: np.ndarray, count: int, quantum: float) -> np.ndarray:
    """``count`` indices of ``points`` by farthest-point selection from the first, each squared distance weighted by
    ``density`` and read in whole ``quantum``s, so that ties go to the lower index (``argmax`` takes the first)."""
    with cast(Any, jax).enable_x64(True):
        return np.asarray(_SELECT(jnp.asarray(points, dtype=jnp.float64), jnp.asarray(density, dtype=jnp.float64),
                                  jnp.asarray(quantum, dtype=jnp.float64), count=count), dtype=np.int64)


def _owners(generators: np.ndarray, points: np.ndarray, quantum: float) -> np.ndarray:
    """Each point's nearest generator, the lower index where its two nearest lie within ``quantum`` of each other in
    squared distance: a tie, which rounding would otherwise decide."""
    if len(generators) < 2:
        return np.zeros(len(points), dtype=np.int64)
    distance, index = nearest(generators, points, k=2)
    tied = distance[:, 1] ** 2 - distance[:, 0] ** 2 <= quantum
    return np.where(tied, np.minimum(index[:, 0], index[:, 1]), index[:, 0])


def _relaxed(points: np.ndarray, mass: np.ndarray, chosen: np.ndarray, quantum: float) -> np.ndarray:
    """``chosen`` after Lloyd's steps in space, each generator then taken to the dense point of its own cell nearest
    it (so that two generators either side of an edge, whose centroids lie inside the body, are not both drawn onto
    the edge), or, for a generator whose cell is empty, to its nearest dense point not yet taken; ties within
    ``quantum`` go to the lower index."""
    count = len(chosen)
    generators = np.asarray(points[chosen], dtype=np.float64)
    for _ in range(LLOYD_STEPS):
        owner = _owners(generators, points, quantum)
        held = np.bincount(owner, weights=mass, minlength=count)
        centroid = np.stack([np.bincount(owner, weights=mass * points[:, i], minlength=count) for i in range(3)],
                            axis=1) / np.where(held > 0.0, held, 1.0)[:, None]
        generators = np.asarray(np.where((held > 0.0)[:, None], centroid, generators), dtype=np.float64)
    owner = _owners(generators, points, quantum)
    away = np.floor(np.sum((points - generators[owner]) ** 2, axis=1) / quantum)
    order = np.lexsort((np.arange(len(points)), away, owner))
    first = np.searchsorted(owner[order], np.arange(count), side="left")
    found = np.full(count, -1, dtype=np.int64)
    filled = first < len(points)
    filled[filled] = owner[order[first[filled]]] == np.arange(count)[filled]
    found[filled] = order[first[filled]]
    taken = np.zeros(len(points), dtype=bool)
    taken[found[filled]] = True
    for i in np.flatnonzero(~filled).tolist():
        offsets: np.ndarray = points - cast(np.ndarray, generators[i])
        away_from = np.floor(np.sum(offsets * offsets, axis=1) / quantum)
        order_of: list[int] = np.lexsort((np.arange(len(points)), away_from)).tolist()
        found[i] = next(j for j in order_of if not taken[j])
        taken[found[i]] = True
    return found


def _members(points: np.ndarray, chosen: np.ndarray, quantum: float) -> np.ndarray:
    """Each chosen sample's cell, the dense points nearest it (ties to the lower sample), as rows padded with
    ``len(points)``."""
    owner = _owners(points[chosen], points, quantum)
    order = np.argsort(owner, kind="stable")
    sizes = np.bincount(owner, minlength=len(chosen))
    rows = np.full((len(chosen), int(sizes.max())), len(points), dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    for i, (start, size) in enumerate(zip(starts.tolist(), sizes.tolist(), strict=True)):
        rows[i, :size] = order[start:start + size]
    return rows


#: The samplings made in this process, by kind, settings, spacing, patches and density: a build asks once for its
#: plan's counts and once for its node's choice, and a study builds the same body many times.
_MADE: dict[tuple[object, ...], Sampling] = {}
_MADE_MOST = 32


def quasi_uniform(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float, patches: Sequence[str],
                  density: Density = uniform) -> Sampling:
    """``kind``'s surface over ``patches`` sampled quasi-uniformly at ``spacing`` (m) where ``density`` is 1: as many
    samples as its quadrature has at the spacing, times the density's mean, chosen from its quadrature at
    ``REFINEMENT`` times finer (the module's docstring). Any kind whose ``samples`` and ``sample_counts`` it can call
    takes it as its ``sampling``."""
    patches = tuple(patches)
    try:
        key: tuple[object, ...] | None = (type(kind), kind, tuple(sorted(settings.items())), float(spacing),
                                          patches, density)
        hash(key)
    except TypeError:
        key = None
    if key is not None and key in _MADE:
        return _MADE[key]
    made = _quasi_uniform(kind, settings, spacing, patches, density)
    if key is not None:
        if len(_MADE) >= _MADE_MOST:
            _MADE.pop(next(iter(_MADE)))
        _MADE[key] = made
    return made


def _quasi_uniform(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float, patches: tuple[str, ...],
                   density: Density) -> Sampling:
    own, _, _ = quadrature_arrays(kind, settings, patches, quadrature_counts(kind, settings, spacing, patches))
    counts = quadrature_counts(kind, settings, spacing / REFINEMENT, patches)
    points, weights, _ = quadrature_arrays(kind, settings, patches, counts)
    held = np.asarray(density(points), dtype=np.float64)
    if held.shape != (len(points),) or not np.all(np.isfinite(held) & (held > 0.0)):
        raise ValueError(f"{kind.label()}: a sampling's density is a number above zero at every point")
    count = max(1, round(len(own) * float(np.sum(weights * held) / np.sum(weights))))
    if count > len(points):
        raise ValueError(f"{kind.label()}: {count} samples asked of a dense set of {len(points)}")
    quantum = TIE * float(np.max(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    chosen = _relaxed(points, weights * held, _farthest(points, held, count, quantum), quantum)
    members = _members(points, chosen, quantum)
    chosen = chosen.astype(np.int32)
    for array in (chosen, members):
        array.flags.writeable = False
    return Sampling(patches, counts, QUASI_UNIFORM, chosen, members)


def length_of(v: Any) -> Any:
    """``|v|`` for a reach's part across an axis, zero with a zero derivative where ``v`` is zero: there the farthest
    set is a whole face (a cap), and the derivative is its middle. Not yet a switch of ``mime.core.switch``: no node's
    step evaluates a reach in 7m-a, and the contact that will (7o) declares the reach's kinks with their owners."""
    squared = jnp.sum(jnp.asarray(v) * jnp.asarray(v))
    held = squared > 0.0
    return jnp.where(held, jnp.sqrt(jnp.where(held, squared, 1.0)), 0.0)


def magnitude_of(x: Any) -> Any:
    """``|x|`` elementwise for a reach's part along an axis, with a zero derivative where ``x`` is zero (where JAX's
    ``abs`` takes one): there the farthest set is a face, a segment or a rim, and the derivative is its middle. Not yet
    a switch, as :func:`length_of` is not."""
    held = jnp.asarray(x)
    return jnp.where(held == 0.0, jnp.zeros_like(held), jnp.abs(held))


def sampled_reach(samples: Callable[[Mapping[str, Any]], Samples]) -> Callable[[Mapping[str, Any], Any], Any]:
    """A hull query from samples, for a kind with no closed form of its own: ``samples`` gives the kind's samples from
    its parameters (its quadrature at counts it fixes), and the query is the largest ``x . direction`` over them, which
    is the body's reach to within the samples' distance from its farthest point, from inside (it never overstates)."""
    def reach(parameters: Mapping[str, Any], direction: Any) -> Any:
        return jnp.max(jnp.asarray(samples(parameters).points) @ jnp.asarray(direction))

    return reach


def body_reach(kind: Geometry, settings: Mapping[str, Magnitude],
               spacing: float) -> Callable[[Mapping[str, Any], Any], Any]:
    """The hull query of ``kind``'s own body: :func:`sampled_reach` over its quadrature on every patch at ``spacing``,
    the counts fixed at ``settings``. Each sample is a point of the body, so the reach never overstates the body's, and
    its gradient, the farthest sample (the mean of those tied for farthest, the middle of a face or a segment, as the
    closed forms take it), is a point of the body a contact can act at; it falls short of the body's reach by no more
    than the body's farthest point lies from the nearest sample. Where a kind's own ``reach`` is an envelope
    (the screw's: on FL-9 its gradient lies 1.1 to 1.7 mm from the body's farthest point at tilts from 5 to 85 degrees),
    this is the one a contact's point is taken from (the containment contact, phase 7o)."""
    patches = tuple(kind.patches)
    counts = quadrature_counts(kind, settings, spacing, patches)

    def samples(parameters: Mapping[str, Any]) -> Samples:
        found = [kind.samples(parameters, patch, count) for patch, count in zip(patches, counts, strict=True)]
        points = jnp.concatenate([jnp.reshape(jnp.asarray(s.points), (-1, 3)) for s in found])
        return Samples(points, None, None)

    return sampled_reach(samples)


#: The directions a hull's samples' shortfall is taken over (:func:`samples_shortfall`): on a Fibonacci sphere, with
#: the six axes and the eight diagonals; and how much finer the quadrature is that a kind with no support is judged
#: against.
SHORTFALL_DIRECTIONS = 2048
SHORTFALL_REFINEMENT = 4
#: The most samples the plan judges a hull's shortfall from (``judged_samples``). A declaration, so that a plan never
#: takes more of a machine than a test does: measured at 1.3 million on a sphere, every sample a hull vertex, 2.7 GB and
#: 47 s, and at 1.5 million on FL-9, 0.7 GB and 10 s; FL-9 at the default spacing has 15,327.
JUDGED_SAMPLES = 2_000_000
_DIRECTIONS_AT_ONCE = 64
_shortfalls: dict[tuple[object, ...], float] = {}


def shortfall_directions() -> np.ndarray:
    i = np.arange(SHORTFALL_DIRECTIONS) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / SHORTFALL_DIRECTIONS)
    turn = np.pi * (1.0 + np.sqrt(5.0)) * i
    spread = np.stack([np.sin(polar) * np.cos(turn), np.sin(polar) * np.sin(turn), np.cos(polar)], axis=1)
    corners = np.asarray([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    return np.concatenate([spread, np.eye(3), -np.eye(3), corners / np.sqrt(3.0)])


def _hull_points(points: np.ndarray) -> np.ndarray:
    """The vertices of ``points``' convex hull (every point where there is none): the only points a support needs."""
    from scipy.spatial import ConvexHull, QhullError

    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        return points


def _farthest_along(points: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each direction's farthest of ``points`` and its reach along it, a few directions at a time: all of them at once
    would hold a number per point and direction, 16 kB a point, and every sample of a round body is a hull vertex."""
    index, reach = np.empty(len(directions), dtype=np.int64), np.empty(len(directions))
    for start in range(0, len(directions), _DIRECTIONS_AT_ONCE):
        along = points @ directions[start:start + _DIRECTIONS_AT_ONCE].T
        index[start:start + _DIRECTIONS_AT_ONCE] = np.argmax(along, axis=0)
        reach[start:start + _DIRECTIONS_AT_ONCE] = np.max(along, axis=0)
    return index, reach


def judged_samples(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float) -> int:
    """About how many samples :func:`samples_shortfall` reads to judge ``spacing``, known before any is made (each
    patch's counts' product, which a patch of two faces has twice of): its quadrature's at ``spacing``, and
    ``SHORTFALL_REFINEMENT`` times finer for a kind with no support."""
    from mime.registry import implements

    finest = spacing if implements(kind, "support") else spacing / SHORTFALL_REFINEMENT
    return sum(math.prod(counts) for counts in quadrature_counts(kind, settings, finest, tuple(kind.patches)))


def samples_shortfall(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float) -> float:
    """How far the body reaches past its quadrature's samples at ``spacing`` on every patch (m): the largest, over
    ``shortfall_directions``, of its farthest point's reach less its farthest sample's. The body's reach is its
    support's where its kind has one (from the farthest sample, exact), and its quadrature's at
    ``SHORTFALL_REFINEMENT`` times finer spacing otherwise (its own shortfall about that factor squared smaller). An
    estimate over finitely many directions, which may miss the largest by a little; a contact holding a body by its
    samples keeps the body within the margin less this, so the plan refuses a spacing whose shortfall passes the
    margin (``confinement.spacing-short``)."""
    key = (type(kind).__name__, getattr(kind, "version", ""), tuple(sorted((k, repr(v)) for k, v in settings.items())),
           spacing)
    held = _shortfalls.get(key)
    if held is not None:
        return held
    from mime.registry import implements

    patches = tuple(kind.patches)
    points = quadrature_arrays(kind, settings, patches, quadrature_counts(kind, settings, spacing, patches))[0]
    coarse = _hull_points(points)
    directions = shortfall_directions()
    index, sampled = _farthest_along(coarse, directions)
    if implements(kind, "support"):
        with cast(Any, jax).enable_x64(True):
            found = kind.support(_parameters(settings), jnp.asarray(directions), jnp.asarray(coarse[index]))
            body = np.einsum("ij,ij->i", np.asarray(found.point, dtype=np.float64), directions)
    else:
        fine = quadrature_arrays(kind, settings, patches,
                                 quadrature_counts(kind, settings, spacing / SHORTFALL_REFINEMENT, patches))[0]
        body = _farthest_along(_hull_points(fine), directions)[1]
    held = max(float(np.max(body - sampled)), 0.0)
    _shortfalls[key] = held
    return held


__all__ = ["JUDGED_SAMPLES", "LLOYD_STEPS", "QUADRATURE", "QUASI_UNIFORM", "REFINEMENT", "TIE", "body_reach",
           "judged_samples", "length_of", "magnitude_of", "nearest", "quadrature", "quadrature_arrays",
           "quadrature_counts", "quasi_uniform", "sampled_reach", "samples_shortfall", "shortfall_directions"]
