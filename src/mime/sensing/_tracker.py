"""A JAX-native tracker (camera spike R2): a soft centroid of a backlit frame's counts before noise (``expected``),
back-projected through a believed pinhole onto a declared plane. Every step is a smooth function of the frame, so a
derivative passes from the frame's counts to the 3-D point.

- **The soft centroid**: each pixel weighs its darkness, ``(full - counts) / full``, the share of the pixel the
  shadow covers for an opaque body; the centroid is the weighted mean of the pixel centres (``u`` along a row, ``v``
  down the columns, a pixel's centre at whole numbers, as :class:`mime.sensing.Pinhole` places ``cx``). It is the
  silhouette's centroid, which for a ball off the axis sits slightly outward of its centre's projection (a perspective
  bias, the camera arm's own).
- **Depth from a declared plane**: the ray through the centroid (no distortion believed) meets the plane through
  ``plane_point`` with normal ``plane_normal``; out-of-plane motion is not observed. ``spread``, the shadow's area
  over the area a ball of ``radius`` on the plane would cast, is one where the body is on the plane, and its departure
  is the out-of-plane flag's indicator."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax

jnp: Any = jax.numpy


class Tracked(NamedTuple):
    point: Any
    pixel: Any
    mass: Any


def soft_centroid(expected: Any, full: float) -> tuple[Any, Any, Any]:
    """The darkness-weighted centroid ``(u, v)`` of ``expected`` ``(rows, columns)`` and the darkness's total."""
    dark = (full - expected) / full
    rows, columns = expected.shape
    v, u = jnp.meshgrid(jnp.arange(rows, dtype=expected.dtype), jnp.arange(columns, dtype=expected.dtype),
                        indexing="ij")
    total = jnp.sum(dark)
    return jnp.sum(dark * u) / total, jnp.sum(dark * v) / total, total


def rotation(q: Any) -> Any:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.stack([jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
                      jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
                      jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])])


def tracked(expected: Any, *, full: float, fx: Any, fy: Any, cx: Any, cy: Any, position: Any, orientation: Any,
            plane_point: Any, plane_normal: Any) -> Tracked:
    """The 3-D point the frame's soft centroid back-projects to on the declared plane, through the believed camera
    (intrinsics ``fx, fy, cx, cy``; its optical frame at ``position`` turned by ``orientation`` into the world)."""
    u, v, mass = soft_centroid(expected, full)
    ray = rotation(jnp.asarray(orientation, dtype=expected.dtype)) @ jnp.stack([(u - cx) / fx, (v - cy) / fy,
                                                                                 jnp.ones((), expected.dtype)])
    origin = jnp.asarray(position, dtype=expected.dtype)
    normal = jnp.asarray(plane_normal, dtype=expected.dtype)
    along = jnp.dot(jnp.asarray(plane_point, dtype=expected.dtype) - origin, normal) / jnp.dot(ray, normal)
    return Tracked(origin + along * ray, jnp.stack([u, v]), mass)


__all__ = ["Tracked", "soft_centroid", "tracked"]
