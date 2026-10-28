"""The kind-conformance harness: does a geometry kind's sampled surface agree with its run-time shapes and its closed
forms?

For each of the kind's patches, sampled at a spacing (or at counts the caller chooses): the samples lie on the boundary
of the region the patch names (its shape's signed distance zero, to rounding), and on or inside every other region of
a solid, whose body is its regions' intersection (a cap within the screw's cross-section, the screw's surface within
the span of its caps); the sample normals equal the shape's outward normal there, to the rounding of both: the
sample's position and the shape's evaluation there each round at the sample's rounding length ``l`` (its
``rounding_length``, or its distance from the kind's origin), which the normal carries by its sensitivity ``|dn/dp|``
(autodiff at the sample, the largest singular value of the normal's Jacobian: the section's curvature and twist), and
the normal itself rounds, so ``NORMAL_ROUNDING_ULPS`` units in the last place of ``1 + l |dn/dp|``; and the weights
sum to the patch's area. Over every patch together, a closed kind's volume by the divergence theorem, ``1/3``
of the integral of ``x . n``, equals its closed form; and, for a solid, its second moments by the same theorem, ``1/5``
of the integral of ``x_i x_j (x . n)``, give its inertia per unit mass. Each is judged on rounding of the inputs' scale:
``KIND_ROUNDING_ULPS`` units in the last place of the sum's terms, growing as the square root of their number. The
drawing check is the stage's (7e)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import jax
import numpy as np

from mime.core import jnp, shape_named
from mime.registry import Geometry, Solid, implements
from mime.units import Magnitude

#: The tolerance's units in the last place, for a value computed in a few operations from inputs of its scale.
KIND_ROUNDING_ULPS = 64.0
#: A sample's normal against its shape's, in units in the last place of ``1 + l |dn/dp|`` (the module's docstring):
#: the sample's rounding and the shape's at its point, each a position rounded at ``l`` carried through the normal's
#: sensitivity, and the normal's own. A declaration, the next power of two above the sum of the two parts' largest
#: values measured against exact arithmetic (mpmath, 40 digits) on the helix at the 36 settings
#: ``tests/saved/test_registry_walk.py`` runs, at 1, 3 and 10 times FL-9's length: the sample's, at every lateral
#: sample (every fourth beyond FL-9's length), 2.41; the shape's at the very same point, at the 48 samples per setting
#: where the comparison or the sensitivity is largest and 16 spread evenly, 1.69; the comparison itself, at every
#: sample, at most 2.95. The sphere's and the capsule's, ``l`` the distance from their centre: 0.49 and 0.60, the
#: comparison 0.79.
NORMAL_ROUNDING_ULPS = 8.0
_EPSILON = float(np.finfo(np.float64).eps)
_vmap: Any = cast(Any, jax).vmap
_jit: Any = cast(Any, jax).jit
_jacfwd: Any = cast(Any, jax).jacfwd


def _at_samples(shape: Any, parameters: Mapping[str, Any], points: Any) -> tuple[np.ndarray, np.ndarray]:
    """The shape's signed distance and outward normal at every sample, in float64."""
    held = dict(parameters)

    def one(point: Any) -> tuple[Any, Any]:
        return shape.signed_distance(held, point), shape.outward_normal(held, point)

    distance, normal = _jit(_vmap(one))(jnp.asarray(points, dtype=jnp.float64))
    return np.asarray(distance, dtype=np.float64), np.asarray(normal, dtype=np.float64)


def _sensitivity(shape: Any, parameters: Mapping[str, Any], points: np.ndarray) -> np.ndarray:
    """The largest singular value of the shape's outward normal's Jacobian at each point: how far the normal turns
    per metre the point moves."""
    held = dict(parameters)

    def normal(point: Any) -> Any:
        return shape.outward_normal(held, point)

    def jacobian(point: Any) -> Any:
        return _jacfwd(normal)(point)

    found = _jit(_vmap(jacobian))(jnp.asarray(points, dtype=jnp.float64))
    return np.linalg.norm(np.asarray(found, dtype=np.float64), ord=2, axis=(1, 2))


def kind_problems(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float,
                  counts: Mapping[str, tuple[int, ...]] | None = None) -> list[str]:
    """Every way ``kind``, with its ``settings`` in SI, sampled about ``spacing`` apart (or at ``counts`` for each
    patch it names), fails the harness; none for a kind that samples no patch (it has nothing to check), which the
    registry walk reports."""
    if not kind.patches:
        return []
    label = kind.label()
    found: list[str] = []
    parameters = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                  for name, value in settings.items()}
    flux, moments, terms, scale = 0.0, np.zeros((3, 3)), 0, 0.0
    for patch, region in kind.patches.items():
        chosen = (counts or {}).get(patch) or kind.sample_counts(settings, spacing, patch)
        samples = kind.samples(parameters, patch, chosen)
        points, normals, weights = (np.asarray(held, dtype=np.float64) for held in
                                    (samples.points, samples.normals, samples.weights))
        size = float(np.max(np.linalg.norm(points, axis=1)))
        scale = max(scale, size)
        shape = shape_named(kind.region_shapes[region])
        distance, normal = _at_samples(shape, kind.region_parameters(region, settings), points)
        terms += len(weights)
        allowed = KIND_ROUNDING_ULPS * _EPSILON * math.sqrt(len(weights))
        off = float(np.max(np.abs(distance)))
        if not off <= KIND_ROUNDING_ULPS * _EPSILON * size:
            found.append(f"{label}: patch {patch!r}'s samples lie up to {off:.3g} m off the boundary of its region "
                         f"{region!r} ({kind.region_shapes[region]})")
        for other in kind.regions if isinstance(kind, Solid) else ():
            if other == region or other not in kind.region_shapes:
                continue
            beyond, _ = _at_samples(shape_named(kind.region_shapes[other]), kind.region_parameters(other, settings),
                                    points)
            outside = float(np.max(beyond))
            if not outside <= KIND_ROUNDING_ULPS * _EPSILON * size:
                found.append(f"{label}: patch {patch!r}'s samples lie up to {outside:.3g} m outside its region "
                             f"{other!r} ({kind.region_shapes[other]}), which the body lies within")
        apart = np.max(np.abs(normal - normals), axis=1)
        allowed_turn = np.full(len(apart), NORMAL_ROUNDING_ULPS * _EPSILON)
        beyond = apart > allowed_turn  # the sensitivity is read only where it is needed
        if np.any(beyond):
            length = np.linalg.norm(points[beyond], axis=1)  # a kind's rounding length can only be below it
            if samples.rounding_length is not None:
                length = np.minimum(length, np.asarray(samples.rounding_length, dtype=np.float64)[beyond])
            allowed_turn[beyond] *= 1.0 + length * _sensitivity(shape, kind.region_parameters(region, settings),
                                                                points[beyond])
        if not np.all(apart <= allowed_turn):
            worst = int(np.argmax(apart / allowed_turn))
            found.append(f"{label}: patch {patch!r}'s sample normals differ from {kind.region_shapes[region]}'s "
                         f"outward normal by up to {float(apart[worst]):.3g}, beyond {float(allowed_turn[worst]):.3g} "
                         f"there")
        area = kind.area(settings, patch)
        if not abs(float(np.sum(weights)) - area) <= allowed * area:
            found.append(f"{label}: patch {patch!r}'s weights sum to {float(np.sum(weights))!r} m^2, and its area is "
                         f"{area!r} m^2")
        reach = np.sum(points * normals, axis=1) * weights
        flux += float(np.sum(reach)) / 3.0
        moments += np.einsum("ni,nj,n->ij", points, points, reach) / 5.0
    if isinstance(kind, Solid):
        volume = kind.volume(settings)
        allowed = KIND_ROUNDING_ULPS * _EPSILON * math.sqrt(terms)
        if not abs(flux - volume) <= allowed * volume:
            found.append(f"{label}: its volume by the divergence theorem is {flux!r} m^3, and its closed form "
                         f"{volume!r} m^3")
        second = moments / volume
        inertia = np.trace(second) * np.eye(3) - second
        stated = np.asarray(kind.inertia_per_mass(settings), dtype=np.float64)
        apart = float(np.max(np.abs(inertia - stated)))
        if not apart <= allowed * scale * scale:
            found.append(f"{label}: its inertia per unit mass by the divergence theorem is {inertia.tolist()} m^2, "
                         f"and its closed form {stated.tolist()} m^2 (apart by {apart:.3g})")
    return found


def sampling_kinds(kinds: list[type[Geometry]]) -> list[type[Geometry]]:
    """The kinds among ``kinds`` that sample their surface: those the harness checks."""
    return [kind for kind in kinds if implements(kind, "samples")]


__all__ = ["KIND_ROUNDING_ULPS", "NORMAL_ROUNDING_ULPS", "kind_problems", "sampling_kinds"]
