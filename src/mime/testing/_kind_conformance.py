"""The kind-conformance harness: does a geometry kind's sampled surface agree with its run-time shapes and its closed
forms?

For each of the kind's patches, sampled at a spacing: the samples lie on the boundary of the region the patch names
(its shape's signed distance zero, to rounding), the sample normals equal the shape's outward normal there, and the
weights sum to the patch's area. Over every patch together, a closed kind's volume by the divergence theorem, ``1/3``
of the integral of ``x . n``, equals its closed form; and, for a solid, its second moments by the same theorem, ``1/5``
of the integral of ``x_i x_j (x . n)``, give its inertia per unit mass. Each is judged on rounding of the inputs' scale:
``KIND_ROUNDING_ULPS`` units in the last place of the sum's terms, growing as the square root of their number. The drawing
check is the stage's (7e)."""

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
_EPSILON = float(np.finfo(np.float64).eps)
_vmap: Any = cast(Any, jax).vmap
_jit: Any = cast(Any, jax).jit


def _at_samples(shape: Any, parameters: Mapping[str, Any], points: Any) -> tuple[np.ndarray, np.ndarray]:
    """The shape's signed distance and outward normal at every sample, in float64."""
    held = dict(parameters)

    def one(point: Any) -> tuple[Any, Any]:
        return shape.signed_distance(held, point), shape.outward_normal(held, point)

    distance, normal = _jit(_vmap(one))(jnp.asarray(points, dtype=jnp.float64))
    return np.asarray(distance, dtype=np.float64), np.asarray(normal, dtype=np.float64)


def kind_problems(kind: Geometry, settings: Mapping[str, Magnitude], spacing: float) -> list[str]:
    """Every way ``kind``, with its ``settings`` in SI, sampled about ``spacing`` apart, fails the harness; none for a
    kind that samples no patch (it has nothing to check), which the registry walk reports."""
    if not kind.patches:
        return []
    label = kind.label()
    found: list[str] = []
    parameters = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                  for name, value in settings.items()}
    flux, moments, terms, scale = 0.0, np.zeros((3, 3)), 0, 0.0
    for patch, region in kind.patches.items():
        counts = kind.sample_counts(settings, spacing, patch)
        samples = kind.samples(parameters, patch, counts)
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
        turned = float(np.max(np.abs(normal - normals)))
        if not turned <= 4.0 * KIND_ROUNDING_ULPS * _EPSILON:
            found.append(f"{label}: patch {patch!r}'s sample normals differ from {kind.region_shapes[region]}'s outward "
                         f"normal by up to {turned:.3g}")
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


__all__ = ["KIND_ROUNDING_ULPS", "kind_problems", "sampling_kinds"]
