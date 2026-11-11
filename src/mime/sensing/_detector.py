"""The detector: a pinhole with Brown-Conrady distortion (OpenCV's parameters and conventions), its ray table, and the
projection of points into it.

**Conventions** (OpenCV's, which a calibration file states): the camera's optical frame has +x to the right of the image,
+y down it and +z forward, out of the lens; a pixel's centre is at integer coordinates, the first pixel's at ``(0, 0)``;
``u`` runs along a row (the image's columns), ``v`` down a column. The pose is the optical frame's place in the world:
its origin (the pinhole) and the unit quaternion ``(w, x, y, z)`` turning optical-frame vectors into the world's.

**The ray table** is traced from the intrinsics' leaves, so a derivative in the focal length, the principal point or a
distortion coefficient flows through every ray. Each pixel's ray is the undistorted direction through its centre, by
Newton's method on the distortion: ``UNDISTORT_STEPS`` steps with their derivative held, then one step with it flowing,
whose derivative is the implicit function's (exact where the steps converged). With each ray come its derivatives along
``u`` and ``v`` (one ``jvp`` each), from which the pixel integration reads an edge's line within the pixel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import jax
import numpy as np

from mime.core import quaternion
from mime.core._node import jnp
from mime.scene import CameraPrim

#: Bracketed Newton steps of the undistortion's radial solve, their derivative held (a step leaving the bracket bisects
#: it, so a hard lens converges at worst as bisection); the distortion's inverse gate measures what they reach.
UNDISTORT_STEPS = 16
#: Newton steps of the whole map after the radial solve, for the tangential terms (|p| about 1e-3): eight, since four
#: left a pixel near the validity radius unconverged on about one drawn lens in eight (the gate's clean arm, 4 of 30).
TANGENTIAL_STEPS = 8

lax: Any = jax.lax
#: JAX's transforms, whose hints name a device type from jaxlib's compiled core, which ships no stubs.
_jax: Any = cast(Any, jax)


@dataclass(frozen=True)
class BrownConrady:
    """Brown-Conrady lens distortion in OpenCV's five parameters and order: radial ``k1``, ``k2``, ``k3`` and tangential
    ``p1``, ``p2``, acting on undistorted normalised coordinates ``(x, y)``::

        r2 = x^2 + y^2,  radial = 1 + k1 r2 + k2 r2^2 + k3 r2^3
        x_d = x radial + 2 p1 x y + p2 (r2 + 2 x^2)
        y_d = y radial + p1 (r2 + 2 y^2) + 2 p2 x y

    Each coefficient is a JAX leaf: a fit moves it, and every ray follows. ``source`` says where the numbers come from:
    ``"declared"`` (written in the script) or a fit's file; a declaration is not a calibration."""

    k1: Any = 0.0
    k2: Any = 0.0
    p1: Any = 0.0
    p2: Any = 0.0
    k3: Any = 0.0
    source: str = field(default="declared", compare=False)

    def distort(self, x: Any, y: Any) -> tuple[Any, Any]:
        r2 = x * x + y * y
        radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        return (x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x),
                y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y)

    def _jacobian(self, x: Any, y: Any) -> tuple[Any, Any, Any, Any]:
        r2 = x * x + y * y
        radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        slope = self.k1 + r2 * (2.0 * self.k2 + 3.0 * self.k3 * r2)
        cross = 2.0 * x * y * slope + 2.0 * self.p1 * x + 2.0 * self.p2 * y
        return (radial + 2.0 * x * x * slope + 2.0 * self.p1 * y + 6.0 * self.p2 * x, cross,
                cross, radial + 2.0 * y * y * slope + 6.0 * self.p1 * y + 2.0 * self.p2 * x)

    def _newton(self, x: Any, y: Any, xd: Any, yd: Any) -> tuple[Any, Any]:
        fx, fy = self.distort(x, y)
        a, b, c, d = self._jacobian(x, y)
        rx, ry = fx - xd, fy - yd
        determinant = a * d - b * c
        return x - (d * rx - b * ry) / determinant, y - (a * ry - c * rx) / determinant

    def _slope(self, s: Any) -> Any:
        """The radial map's slope ``d(r radial)/dr`` at ``r^2 = s``."""
        return 1.0 + s * (3.0 * self.k1 + s * (5.0 * self.k2 + 7.0 * self.k3 * s))

    def first_turn(self, cap: Any) -> Any:
        """The square of the validity radius: the first ``r^2`` in ``(0, cap]`` where the radial map stops
        increasing, or ``cap`` where it does not. The slope is a cubic in ``r^2``; between the roots of its derivative
        it is monotone, so the first of those pieces whose end is not positive holds the first turn, found there by
        bisection."""
        a, b, c = 21.0 * self.k3, 10.0 * self.k2, 3.0 * self.k1
        discriminant = b * b - 4.0 * a * c
        root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        flat = jnp.abs(a) < 1e-30
        linear = jnp.where(jnp.abs(b) > 1e-30, -c / jnp.where(jnp.abs(b) > 1e-30, b, 1.0), cap)
        held_a = jnp.where(flat, 1.0, a)
        first = jnp.where(flat, linear, (-b - root) / (2.0 * held_a))
        second = jnp.where(flat, cap, (-b + root) / (2.0 * held_a))
        real = flat | (discriminant >= 0.0)
        ends = [jnp.where(real & (e > 0.0) & (e < cap), e, cap) for e in (first, second)]
        ends = [jnp.minimum(ends[0], ends[1]), jnp.maximum(ends[0], ends[1]), cap]
        low, high, found = jnp.zeros_like(cap), cap, jnp.zeros((), dtype=bool)
        previous = jnp.zeros_like(cap)
        for end in ends:
            here = ~found & (self._slope(end) <= 0.0)
            low, high = jnp.where(here, previous, low), jnp.where(here, end, high)
            found = found | here
            previous = end
        def bisect(_: Any, bracket: tuple[Any, Any]) -> tuple[Any, Any]:
            low, high = bracket
            middle = 0.5 * (low + high)
            positive = self._slope(middle) > 0.0
            return jnp.where(positive, middle, low), jnp.where(positive, high, middle)

        low, _ = lax.fori_loop(0, 60, bisect, (low, high))
        return jnp.where(found, low, cap)

    def undistort(self, xd: Any, yd: Any, steps: int | None = None) -> tuple[Any, Any, Any]:
        """The undistorted normalised point on the first monotone branch whose distortion is ``(xd, yd)``, and whether
        it was found there. The radius by a bracketed Newton solve of the radial map on ``[0, validity radius]``
        (a step leaving the bracket bisects it), ``steps`` (``UNDISTORT_STEPS``) of them, then ``TANGENTIAL_STEPS`` Newton steps of
        the whole map for the tangential terms, all with their derivative held; then one more Newton step with it flowing,
        whose derivative is the implicit function's."""
        held = lax.stop_gradient(self)
        hx, hy = lax.stop_gradient(xd), lax.stop_gradient(yd)
        distorted = jnp.sqrt(hx * hx + hy * hy)
        cap = (4.0 * jnp.max(distorted) + 1.0) ** 2
        turn = jnp.sqrt(held.first_turn(cap))

        def radial(r: Any) -> Any:
            r2 = r * r
            return r * (1.0 + r2 * (held.k1 + r2 * (held.k2 + r2 * held.k3)))

        def radial_step(_: Any, carry: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            low, high, r = carry
            value = radial(r) - distorted
            low, high = jnp.where(value < 0.0, r, low), jnp.where(value < 0.0, high, r)
            stepped = r - value / jnp.maximum(held._slope(r * r), 1e-30)
            return low, high, jnp.where((stepped >= low) & (stepped <= high), stepped, 0.5 * (low + high))

        def tangential_step(_: Any, point: tuple[Any, Any]) -> tuple[Any, Any]:
            return held._newton(point[0], point[1], hx, hy)

        high = jnp.broadcast_to(turn, distorted.shape)
        _, _, r = lax.fori_loop(0, UNDISTORT_STEPS if steps is None else steps, radial_step,
                                (jnp.zeros_like(distorted), high, jnp.minimum(distorted, high)))
        scale = jnp.where(distorted > 0.0, r / jnp.where(distorted > 0.0, distorted, 1.0), 1.0)
        x, y = lax.fori_loop(0, TANGENTIAL_STEPS, tangential_step, (hx * scale, hy * scale))
        x, y = lax.stop_gradient(x), lax.stop_gradient(y)
        fx, fy = held.distort(x, y)
        residual = jnp.sqrt((fx - hx) ** 2 + (fy - hy) ** 2)
        eps = float(jnp.finfo(residual.dtype).eps)
        found = (residual <= 1024.0 * eps * (1.0 + distorted)) & (x * x + y * y < turn * turn)
        x, y = self._newton(x, y, xd, yd)
        return x, y, found

    def monotone(self, x: Any, y: Any) -> Any:
        """Whether the radial distortion is still increasing at ``(x, y)``: inside its validity radius, where the
        map from undistorted to distorted radius is one to one."""
        r2 = x * x + y * y
        return 1.0 + r2 * (3.0 * self.k1 + r2 * (5.0 * self.k2 + 7.0 * self.k3 * r2)) > 0.0


jax.tree_util.register_dataclass(BrownConrady, data_fields=["k1", "k2", "p1", "p2", "k3"], meta_fields=["source"])


@dataclass(frozen=True)
class Pinhole:
    """A pinhole detector of ``width`` by ``height`` pixels: focal lengths ``fx``, ``fy`` and principal point ``cx``,
    ``cy`` in pixels, and its lens distortion. Its numbers are JAX leaves (the sizes are not), so a derivative in any of
    them flows through the ray table."""

    width: int
    height: int
    fx: Any
    fy: Any
    cx: Any
    cy: Any
    distortion: BrownConrady = field(default_factory=BrownConrady)

    def __post_init__(self) -> None:
        if not (self.width > 0 and self.height > 0):
            raise ValueError(f"a detector is a whole number of pixels each way, not {self.width!r} by {self.height!r}")

    def pixels(self, u: Any, v: Any) -> tuple[Any, Any]:
        """Normalised distorted coordinates of the pixel point ``(u, v)``."""
        return (u - self.cx) / self.fx, (v - self.cy) / self.fy

    def to_pixels(self, xd: Any, yd: Any) -> tuple[Any, Any]:
        return self.fx * xd + self.cx, self.fy * yd + self.cy


jax.tree_util.register_dataclass(Pinhole, data_fields=["fx", "fy", "cx", "cy", "distortion"],
                                 meta_fields=["width", "height"])


@dataclass(frozen=True)
class CameraPose:
    """Where the camera is: its optical frame's origin (the pinhole) in the world, in metres, and the unit quaternion
    ``(w, x, y, z)`` turning optical-frame vectors (x right, y down, z forward) into the world's."""

    position: Any
    orientation: Any


jax.tree_util.register_dataclass(CameraPose, data_fields=["position", "orientation"], meta_fields=[])


class Rays(NamedTuple):
    """The ray table: the pinhole in the world, and per pixel (``(height, width)`` arrays, a component each) the unit
    direction through its centre, its derivatives along ``u`` and ``v``, and whether the pixel lies beyond the
    distortion's validity (not invertible there, or not converged)."""

    origin: tuple[Any, Any, Any]
    direction: tuple[Any, Any, Any]
    along_u: tuple[Any, Any, Any]
    along_v: tuple[Any, Any, Any]
    beyond: Any


def _directions(camera: Pinhole, rotation: Any, u: Any, v: Any) -> tuple[tuple[Any, Any, Any], Any]:
    xd, yd = camera.pixels(u, v)
    x, y, found = camera.distortion.undistort(xd, yd)
    beyond = ~found
    scale = 1.0 / jnp.sqrt(x * x + y * y + 1.0)
    cx, cy, cz = x * scale, y * scale, scale
    world = tuple(rotation[i, 0] * cx + rotation[i, 1] * cy + rotation[i, 2] * cz for i in range(3))
    return (world[0], world[1], world[2]), beyond


def ray_table(camera: Pinhole, pose: CameraPose, *, u: Any = None, v: Any = None) -> Rays:
    """The ray through each pixel's centre (or through the points ``u``, ``v`` when given), with its derivatives along
    the image's axes."""
    dtype = jnp.result_type(camera.fx, float)
    if u is None or v is None:
        v, u = jnp.meshgrid(jnp.arange(camera.height, dtype=dtype), jnp.arange(camera.width, dtype=dtype),
                            indexing="ij")
    rotation = quaternion.rotation_matrix(jnp.asarray(pose.orientation, dtype=dtype))
    def through(uu: Any, vv: Any) -> tuple[tuple[Any, Any, Any], Any]:
        return _directions(camera, rotation, uu, vv)

    direction, linear, beyond = _jax.linearize(through, u, v, has_aux=True)
    along_u, along_v = linear(jnp.ones_like(u), jnp.zeros_like(v)), linear(jnp.zeros_like(u), jnp.ones_like(v))
    position = jnp.asarray(pose.position, dtype=dtype)
    return Rays((position[0], position[1], position[2]), direction, along_u, along_v, beyond)


def project(camera: Pinhole, pose: CameraPose, points: Any) -> tuple[Any, Any]:
    """Pixel coordinates ``(..., 2)`` of world points ``(..., 3)`` and their depths along the optical axis (positive in
    front of the camera): OpenCV's ``projectPoints`` with this model."""
    dtype = jnp.result_type(camera.fx, float)
    rotation = quaternion.rotation_matrix(jnp.asarray(pose.orientation, dtype=dtype))
    local = (jnp.asarray(points, dtype=dtype) - jnp.asarray(pose.position, dtype=dtype)) @ rotation
    depth = local[..., 2]
    xd, yd = camera.distortion.distort(local[..., 0] / depth, local[..., 1] / depth)
    u, v = camera.to_pixels(xd, yd)
    return jnp.stack([u, v], axis=-1), depth


#: The pixel pitch a camera's stage prim gives its filmback when none is declared, in metres: USD's camera has no
#: pixels, so its apertures are the detector's pixels times a pitch; a declaration only, read by nothing but the prim.
PIXEL_PITCH = 5e-6


def to_camera_prim(camera: Pinhole, pose: CameraPose, *, pitch: float = PIXEL_PITCH) -> CameraPrim:
    """The camera's numbers as its stage prim holds them (``mime.scene.write_camera`` writes them)."""
    d = camera.distortion
    position = tuple(float(c) for c in np.asarray(pose.position, dtype=np.float64))
    orientation = tuple(float(c) for c in np.asarray(pose.orientation, dtype=np.float64))
    return CameraPrim(camera.width, camera.height, float(camera.fx), float(camera.fy), float(camera.cx),
                      float(camera.cy), (float(d.k1), float(d.k2), float(d.p1), float(d.p2), float(d.k3)), d.source,
                      pitch, (position[0], position[1], position[2]),
                      (orientation[0], orientation[1], orientation[2], orientation[3]))


def from_camera_prim(held: CameraPrim) -> tuple[Pinhole, CameraPose]:
    """The camera and its pose a stage prim's numbers describe (``mime.scene.read_camera`` reads them)."""
    k1, k2, p1, p2, k3 = held.distortion
    return (Pinhole(held.width, held.height, held.fx, held.fy, held.cx, held.cy,
                    BrownConrady(k1, k2, p1, p2, k3, source=held.distortion_source)),
            CameraPose(jnp.asarray(held.position), jnp.asarray(held.orientation)))
