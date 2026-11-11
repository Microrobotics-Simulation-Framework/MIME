"""The camera's chain, its stages kept apart: the detector's rays, each subject's path length and coverage (route 1),
the backlit reduction over media on two levels, the exposure's sub-frames, and the response (PSF, gain, seeded noise,
quantisation). A pure observer of the scene's state: nothing here acts on the scene.

**The reduction** (backlit transmission, one bin). Each subject ``b`` multiplies the backlight by
``1 - c_b + c_b tau_b``, ``c_b`` its coverage of the pixel and ``tau_b`` what it transmits where it covers it: zero if
opaque (coverage, never an infinite attenuation: no ``inf x 0``), ``1 - blocks`` if a flat absorber, and
``exp(-(mu_b - mu_parent) L_b)`` along its chord ``L_b`` if it attenuates, ``mu_parent`` its parent medium's (zero at
the top). The parent's own term runs along its whole chord, so the child's region is taken from the parent's material
and given its own: the displaced-medium rule, exact along each ray while a child lies inside its parent. A child that
leaves its parent (a robot past the wall it rides on) by more than its tolerance is flagged.

**Static structure** (subjects not moving) is rendered once per camera into a :class:`StaticCache`, keyed by the hash
of every number it was made from; a cache whose key differs from the camera's and the static subjects' numbers now is
refused as stale, and one whose numbers are traced (a derivative in them) is refused too: the cache carries no
derivative in the intrinsics or the structure, so a calibration renders without it."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax
import numpy as np

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.core import quaternion
from mime.core._node import jnp
from mime.sensing import _coverage as coverage
from mime.sensing import _detector as detector
from mime.sensing import _forms as forms
from mime.sensing._detector import CameraPose, Pinhole, Rays, project, ray_table
from mime.sensing._subjects import Appearance, Look, Subject, AppearanceSource, order, resolve

lax: Any = jax.lax
#: JAX's transforms, whose hints name a device type from jaxlib's compiled core, which ships no stubs.
_jax: Any = cast(Any, jax)

#: Below this projected radius (pixels) a silhouette's box edge loses accuracy (5.5 % in its derivative at 6 px, 8 % at
#: 3); a subject imaged with the box edge there is flagged.
SMALL_SILHOUETTE_PX = 6.0


@dataclass(frozen=True)
class Exposure:
    """A global shutter integrating ``sub_frames`` poses evenly over its exposure (the in-loop clock that gives them
    comes later)."""

    sub_frames: int = 1


@dataclass(frozen=True)
class Response:
    """The sensor's response to the transmitted share ``T``: a Gaussian PSF of ``psf`` pixels (its standard deviation;
    zero for none), ``full_scale`` electrons at full backlight over the exposure, shot noise (Gaussian, of variance the
    signal's) and ``read_noise`` electrons, ``gain`` counts per electron with ``offset``, then quantisation to ``bits``.
    Every number is declared, never a measurement of a camera."""

    full_scale: Any = 1000.0
    gain: Any = 1.0
    offset: Any = 0.0
    read_noise: Any = 0.0
    shot_noise: bool = True
    bits: int = 12
    psf: float = 0.0


class Capture(NamedTuple):
    """A rendered frame: the quantised counts (``image``), the counts expected before noise and quantisation
    (``expected``, the differentiable output), the transmitted share before the PSF (``transmission``), and the flags."""

    image: Any
    expected: Any
    transmission: Any
    flags: dict[str, Any]


@dataclass(frozen=True)
class StaticCache:
    """The static subjects' transmission and the ray table for one camera, and the hash of every number they were
    made from."""

    key: str
    transmission: Any
    rays: Rays
    names: tuple[str, ...]
    flags: Mapping[str, Any]


def _rotation(q: Any, dtype: Any) -> Any:
    return quaternion.rotation_matrix(jnp.asarray(q, dtype=dtype))


def _local(subject: Subject, position: Any, orientation: Any, dtype: Any) -> tuple[Any, Any]:
    """The subject's form's frame in the world: its rotation and its origin."""
    placed = subject.placed
    world = _rotation(orientation, dtype)
    turn = jnp.asarray(placed.turn, dtype=dtype)
    return world @ turn, jnp.asarray(position, dtype=dtype) + world @ jnp.asarray(placed.centre, dtype=dtype)


def _into(rotation: Any, vector: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
    """A world vector's components in a frame of ``rotation`` (``R^T v``)."""
    return tuple(rotation[0, i] * vector[0] + rotation[1, i] * vector[1] + rotation[2, i] * vector[2] for i in range(3))


class Seen(NamedTuple):
    """One subject on a set of rays: its coverage of each pixel and the interval each ray spends inside it."""

    coverage: Any
    interval: tuple[Any, Any]


def _parameters(subject: Subject, dtype: Any) -> dict[str, Any]:
    return {name: jnp.asarray(value, dtype=dtype) for name, value in subject.placed.parameters.items()}


def _box_coverage(subject: Subject, p: Mapping[str, Any], o: Any, d: Any, du: Any, dv: Any,
                  corner_samples: int) -> Any:
    form = subject.placed.form

    def edges(dx: Any, dy: Any, dz: Any) -> tuple[Any, tuple[Any, ...]]:
        direction = (dx, dy, dz)
        return form.edge(p, o, direction), form.pieces(p, o, direction)

    (g, pieces), linear = _jax.linearize(edges, *d)
    gu, pieces_u = linear(*du)
    gv, pieces_v = linear(*dv)
    single = coverage.box(g, gu, gv)
    if not form.corners or corner_samples <= 1:
        return single
    crossing = sum((coverage.near(pk, pu, pv).astype(g.dtype) for pk, pu, pv in zip(pieces, pieces_u, pieces_v,
                                                                                     strict=True)),
                   start=jnp.zeros_like(g))
    corner = crossing >= 2.0

    def edge(dx: Any, dy: Any, dz: Any) -> Any:
        return form.edge(p, o, (dx, dy, dz))

    def one(total: Any, offset: Any) -> tuple[Any, None]:
        su, sv = offset[0], offset[1]
        moved: tuple[Any, Any, Any] = (d[0] + su * du[0] + sv * dv[0], d[1] + su * du[1] + sv * dv[1],
                                       d[2] + su * du[2] + sv * dv[2])
        scale = 1.0 / jnp.sqrt(forms.dot(moved, moved))
        h, sub_linear = _jax.linearize(edge, moved[0] * scale, moved[1] * scale, moved[2] * scale)
        return total + coverage.box(h, sub_linear(*du), sub_linear(*dv), 1.0 / corner_samples), None

    def sub_sampled() -> Any:
        offsets = jnp.asarray(coverage.sub_offsets(corner_samples), dtype=g.dtype)
        total, _ = lax.scan(one, jnp.zeros_like(g), offsets)
        return jnp.where(corner, total / corner_samples ** 2, single)

    # The sub-samples only where some pixel holds a corner: a tube whose ends are out of view pays nothing for them.
    return lax.cond(jnp.any(corner), sub_sampled, lambda: single)


def _corners(camera: Pinhole, start: tuple[Any, Any] | None, size: tuple[int, int]) -> tuple[Any, Any]:
    """The undistorted normalised coordinates of the pixels' corners: of the whole image, or of a window starting at
    ``start`` (its first pixel's ``(v, u)``) of ``size`` pixels."""
    dtype = jnp.result_type(camera.fx, float)
    rows, columns = size
    v, u = jnp.meshgrid(jnp.arange(rows + 1, dtype=dtype) - 0.5, jnp.arange(columns + 1, dtype=dtype) - 0.5,
                        indexing="ij")
    if start is not None:
        v, u = v + start[0].astype(dtype), u + start[1].astype(dtype)
    xd, yd = camera.pixels(u, v)
    x, y, _ = camera.distortion.undistort(xd, yd)
    return x, y


def _seen(subject: Subject, rays: Rays, rotation: Any, origin: Any, dtype: Any, camera: Pinhole, pose: CameraPose,
          start: tuple[Any, Any] | None, corner_samples: int) -> Seen:
    p = _parameters(subject, dtype)
    o = _into(rotation, tuple(rays.origin[i] - origin[i] for i in range(3)))
    d, du, dv = _into(rotation, rays.direction), _into(rotation, rays.along_u), _into(rotation, rays.along_v)
    form = subject.placed.form
    interval = form.interval(p, o, d)
    if subject.coverage == "disc":
        camera_rotation = _rotation(pose.orientation, dtype)
        centre = camera_rotation.T @ (origin - jnp.asarray(pose.position, dtype=dtype))
        corners_x, corners_y = _corners(camera, start, rays.beyond.shape)
        covered = coverage.disc(corners_x, corners_y, centre, p["radius"])
    else:
        covered = _box_coverage(subject, p, o, d, du, dv, corner_samples)
    return Seen(covered, interval)


def _transmits(appearance: Appearance, parent: Appearance | None, chord: Any, dtype: Any) -> Any:
    """What a subject transmits where it covers a pixel (``tau``), its parent medium's attenuation displaced."""
    if appearance.look is Look.OPAQUE:
        return jnp.zeros_like(chord)
    if appearance.look is Look.FLAT:
        return jnp.full_like(chord, 1.0) - jnp.asarray(appearance.blocks, dtype=dtype)
    around = parent.attenuation if parent is not None and parent.look is Look.ATTENUATING else 0.0
    return jnp.exp(-(jnp.asarray(appearance.attenuation, dtype=dtype) - jnp.asarray(around, dtype=dtype)) * chord)


def _window_rays(rays: Rays, start: tuple[Any, Any], size: int) -> Rays:
    def cut(a: Any) -> Any:
        return lax.dynamic_slice(a, start, (size, size))

    return Rays(rays.origin, tuple(cut(a) for a in rays.direction), tuple(cut(a) for a in rays.along_u),
                tuple(cut(a) for a in rays.along_v), cut(rays.beyond))


def _implicit_lumens(subjects: Sequence[Subject]) -> list[Subject]:
    """A clear lumen for each tube's wall whose lumen no subject images: the lumen displaces the wall's material."""
    filled = {s.parent for s in subjects if s.kind == "tube" and s.region == "lumen"}
    return [Subject(f"{s.name}.lumen", "tube", s.settings, "lumen", Appearance.clear(), s.name, pose=s.pose)
            for s in subjects if s.kind == "tube" and s.region == "wall" and s.name not in filled]


def _numbers(value: Any) -> Any:
    """``value``'s numbers as plain Python numbers (raising on a traced value)."""
    if isinstance(value, Mapping):
        held = cast(Mapping[str, Any], value)
        return {str(k): _numbers(held[k]) for k in sorted(held, key=str)}
    if isinstance(value, (tuple, list)):
        return [_numbers(v) for v in cast(Sequence[Any], value)]
    if isinstance(value, (str, bool)) or value is None:
        return value
    return np.asarray(value, dtype=np.float64).tolist()


def cache_key(subjects: Sequence[Subject], camera: Pinhole, pose: CameraPose, dtype: Any) -> str:
    """The hash of every number the static subjects' transmission is made from: the camera's, its pose, each static
    subject's, the dtype and the renderer's own declared constants."""
    try:
        record = {
            "camera": [camera.width, camera.height, _numbers([camera.fx, camera.fy, camera.cx, camera.cy]),
                       _numbers([camera.distortion.k1, camera.distortion.k2, camera.distortion.p1,
                                 camera.distortion.p2, camera.distortion.k3])],
            "pose": _numbers([pose.position, pose.orientation]),
            "subjects": [[s.name, s.kind, s.region, _numbers(s.settings), _numbers(list(s.pose)), s.parent, s.coverage,
                          None if s.appearance is None else [s.appearance.look.value,
                                                             _numbers([s.appearance.attenuation,
                                                                       s.appearance.blocks])]]
                         for s in subjects if not s.moving],
            "dtype": str(np.dtype(dtype)),
            "constants": [forms.GRAZING, coverage.CORNER_SAMPLES, detector.UNDISTORT_STEPS],
        }
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError) as error:
        raise ArgumentRefused([Refusal(
            "camera.cache-traced", RefusalClass.USER,
            "a static cache was given while the camera's or a static subject's numbers are traced (a derivative in them)",
            "render without the cache (cache=None) to differentiate in the intrinsics or the structure",
            why="the cache holds the structure's image at fixed numbers, with no derivative in them")]) from error
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _window_of(subject: Subject, camera: Pinhole, pose: CameraPose, origin: Any) -> int | None:
    """A subject's window's half-size in pixels: its declared one, or for exact disc coverage (always windowed) its
    silhouette's reach at its pose with a margin, which needs the pose and the camera as numbers; ``None`` for a
    box-edged subject with none declared (it is rendered over the whole frame)."""
    if subject.window is not None or subject.coverage != "disc":
        return subject.window
    try:
        reach = float(np.asarray(_projected_reach(camera, pose, origin, subject.enclosing())))
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError) as error:
        raise ArgumentRefused([Refusal(
            "camera.subject", RefusalClass.USER,
            f"subject {subject.name!r}'s exact disc coverage is windowed, and its window cannot be sized from a traced "
            "pose or camera", "declare its half-size in pixels: Subject(..., window=24), or take the box edge: "
            "Subject(..., coverage=\"box_edge\")",
            why="a window's size is fixed when the program is made; a derivative traces the numbers it would be made "
                "from")]) from error
    return math.ceil(1.1 * reach) + 2


def _factor(subject: Subject, parent: Subject | None, parent_pose: tuple[Any, Any] | None, rays: Rays, position: Any,
            orientation: Any, camera: Pinhole, pose: CameraPose, dtype: Any, corner_samples: int, shape: Any,
            flags: dict[str, Any]) -> tuple[Any, Any]:
    """One subject's factor on the whole image (``1 - c + c tau``; ones outside its window), and its projected centre;
    its flags written into ``flags``."""
    rotation, origin = _local(subject, position, orientation, dtype)
    pixel, depth = project(camera, pose, origin)
    window = _window_of(subject, camera, pose, origin)
    if window is not None:
        size = min(2 * window + 1, camera.height, camera.width)
        start = (jnp.round(pixel[1]).astype(jnp.int32) - window, jnp.round(pixel[0]).astype(jnp.int32) - window)
        start = (jnp.clip(start[0], 0, camera.height - size), jnp.clip(start[1], 0, camera.width - size))
        seen_rays = _window_rays(rays, start, size)
    else:
        start, seen_rays = None, rays
    seen = _seen(subject, seen_rays, rotation, origin, dtype, camera, pose, start, corner_samples)
    assert subject.appearance is not None
    chord = forms.chord(seen.interval)
    factor = 1.0 - seen.coverage + seen.coverage * _transmits(
        subject.appearance, parent.appearance if parent is not None else None, chord, dtype)
    name = subject.name
    if parent is not None and parent_pose is not None:
        parent_rotation, parent_origin = _local(parent, parent_pose[0], parent_pose[1], dtype)
        o = _into(parent_rotation, tuple(seen_rays.origin[i] - parent_origin[i] for i in range(3)))
        inside = parent.placed.form.interval(_parameters(parent, dtype), o, _into(parent_rotation, seen_rays.direction))
        limit = 2.0 * subject.enclosing() * math.sqrt(2.0 * subject.tolerance)
        outside = jnp.sum((chord - forms.overlap(seen.interval, inside)) > limit)
        flags[f"outside_parent:{name}"] = flags.get(f"outside_parent:{name}", 0) + outside
    if start is not None and window is not None:
        factor = lax.dynamic_update_slice(jnp.ones(shape, dtype=factor.dtype), factor, start)
        reach = _projected_reach(camera, pose, origin, subject.enclosing())
        flags[f"window_overflow:{name}"] = flags.get(f"window_overflow:{name}", False) | (reach + 1.5 > window)
    if subject.coverage == "box_edge":
        small = camera.fx * subject.feature() / depth < SMALL_SILHOUETTE_PX
        flags[f"small_silhouette:{name}"] = flags.get(f"small_silhouette:{name}", False) | small
    flags[f"behind:{name}"] = flags.get(f"behind:{name}", False) | (depth < subject.enclosing())
    if subject.placed.form.corners:
        flags[f"thin_cap:{name}"] = flags.get(f"thin_cap:{name}", False) | _thin_cap(subject, rotation, origin, camera,
                                                                                      pose, dtype)
    return factor, pixel


#: A cap whose ellipse is thinner than this many pixels across (seen within a pixel of edge-on) and in view raises
#: ``thin_cap``: its outline is resolved to about a seventh of a pixel's area by the corner sub-samples there (the cap's
#: edge function changes across its ellipse faster than a sub-pixel's line follows), measured on a side-on cylinder.
THIN_CAP_PX = 1.0


def _thin_cap(subject: Subject, rotation: Any, origin: Any, camera: Pinhole, pose: CameraPose, dtype: Any) -> Any:
    """Whether a capped cylinder shows a cap in view seen nearly edge-on: the cap's ellipse's minor axis (twice
    ``f R |cos|`` over its depth, the cosine between the line of sight and the axis) under ``THIN_CAP_PX``."""
    radius, half = subject.placed.parameters["radius"], subject.placed.parameters["length"] / 2.0
    axis = rotation[:, 2]
    position = jnp.asarray(pose.position, dtype=dtype)
    found = jnp.zeros((), dtype=bool)
    for end in (-1.0, 1.0):
        centre = origin + end * half * axis
        sight = centre - position
        distance = jnp.sqrt(jnp.sum(sight * sight))
        pixel, depth = project(camera, pose, centre)
        across = 2.0 * camera.fx * radius * jnp.abs(jnp.sum(sight * axis)) / (distance * depth)
        reach = camera.fx * radius / depth
        seen = (pixel[0] > -reach) & (pixel[0] < camera.width + reach) & (pixel[1] > -reach) \
            & (pixel[1] < camera.height + reach) & (depth > 0)
        found = found | (seen & (across < THIN_CAP_PX))
    return found


def _static(subjects: Sequence[Subject], rays: Rays, camera: Pinhole, pose: CameraPose, dtype: Any,
            corner_samples: int, flags: dict[str, Any] | None = None) -> Any:
    by_name = {s.name: s for s in subjects}
    held_flags: dict[str, Any] = {} if flags is None else flags
    shape = rays.direction[0].shape
    total = jnp.ones(shape, dtype=dtype)
    for subject in subjects:
        if subject.moving:
            continue
        parent = by_name[subject.parent] if subject.parent is not None else None
        factor, _ = _factor(subject, parent, parent.pose if parent is not None and not parent.moving else None, rays,
                            subject.pose[0], subject.pose[1], camera, pose, dtype, corner_samples, shape, held_flags)
        total = total * factor
    return total


def _prepared(subjects: Sequence[Subject], stage: Mapping[str, Any] | None) -> tuple[list[Subject], dict[str, AppearanceSource],
                                                                                     tuple[str, ...]]:
    resolved = resolve(list(subjects) + _implicit_lumens(subjects), stage)
    return order(resolved.subjects), dict(resolved.sources), resolved.notes


def static_cache(subjects: Sequence[Subject], camera: Pinhole, pose: CameraPose, *, stage: Mapping[str, Any] | None = None,
            corner_samples: int = coverage.CORNER_SAMPLES) -> StaticCache:
    """The static subjects' transmission for this camera, once, keyed by the hash of what it was made from."""
    held, _, _ = _prepared(subjects, stage)
    dtype = jnp.result_type(camera.fx, float)
    rays = lax.stop_gradient(ray_table(camera, pose))
    flags: dict[str, Any] = {}
    transmission = lax.stop_gradient(_static(held, rays, camera, pose, dtype, corner_samples, flags))
    return StaticCache(cache_key(held, camera, pose, dtype), transmission, rays,
                       tuple(s.name for s in held if not s.moving), flags)


def _psf(image: Any, sigma: float) -> Any:
    if sigma <= 0.0:
        return image
    radius = max(1, math.ceil(3.0 * sigma))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-offsets ** 2 / (2.0 * sigma ** 2))
    weights = weights / weights.sum()
    padded: Any = jnp.pad(image, radius, mode="edge")
    height, width = image.shape
    rows: Any = jnp.zeros((height, width + 2 * radius), dtype=padded.dtype)
    for k, w in zip(offsets, weights, strict=True):
        rows = rows + float(w) * padded[radius + int(k):radius + int(k) + height, :]
    out: Any = jnp.zeros((height, width), dtype=padded.dtype)
    for k, w in zip(offsets, weights, strict=True):
        out = out + float(w) * rows[:, radius + int(k):radius + int(k) + width]
    return out


def respond(transmission: Any, response: Response, key: Any) -> tuple[Any, Any, Any]:
    """The expected counts, the quantised counts and the share saturated, from the transmitted share."""
    electrons = response.full_scale * _psf(transmission, response.psf)
    expected = response.gain * electrons + response.offset
    if key is None:
        noisy = expected
    else:
        shot, read = jax.random.normal(key, (2, *electrons.shape), dtype=electrons.dtype)
        spread = (jnp.sqrt(jnp.maximum(electrons, 0.0)) * shot if response.shot_noise else 0.0) \
            + response.read_noise * read
        noisy = expected + response.gain * spread
    top = float(2 ** response.bits - 1)
    image = jnp.clip(jnp.round(noisy), 0.0, top)
    return expected, image, jnp.mean((noisy >= top).astype(expected.dtype))


def for_display(image: Any, response: Response) -> Any:
    """The quantised counts as a display's grey levels in ``[0, 1]`` (full scale white), for a viewer; a controller
    reads the counts."""
    return image / float(2 ** response.bits - 1)


def _pose_at(poses: Mapping[str, Any], subject: Subject, k: int, frames: int) -> tuple[Any, Any]:
    if subject.name not in poses:
        raise ArgumentRefused([Refusal("camera.subject", RefusalClass.USER,
                                       f"subject {subject.name!r} is moving, and no pose of it was given",
                                       f"give poses={{{subject.name!r}: (positions, orientations)}}")])
    position, orientation = poses[subject.name]
    position, orientation = jnp.asarray(position), jnp.asarray(orientation)
    if position.ndim == 1:
        return position, orientation
    if position.shape[0] != frames:
        raise ArgumentRefused([Refusal("camera.subject", RefusalClass.USER,
                                       f"subject {subject.name!r} has {position.shape[0]} poses for "
                                       f"{frames} sub-frames", "give one pose per sub-frame: poses={name: (positions of shape (K, 3), orientations of shape (K, 4))}")])
    return position[k], orientation[k]


def render(subjects: Sequence[Subject], camera: Pinhole, pose: CameraPose, poses: Mapping[str, Any] | None = None, *,
           exposure: Exposure = Exposure(), response: Response = Response(), key: Any = None,
           cache: StaticCache | None = None, stage: Mapping[str, Any] | None = None,
           corner_samples: int = coverage.CORNER_SAMPLES) -> Capture:
    """One frame of ``subjects`` seen by ``camera`` at ``pose``: the moving subjects at ``poses`` (a position and an
    orientation each, or ``sub_frames`` of each), the static ones at their own, through the exposure and the response;
    noise drawn from ``key`` (none without one). Pure JAX: differentiable in the poses, the intrinsics and the subjects'
    numbers (without a cache)."""
    held, sources, notes = _prepared(subjects, stage)
    del sources, notes
    dtype = jnp.result_type(camera.fx, float)
    if cache is not None:
        if cache_key(held, camera, pose, dtype) != cache.key:
            raise ArgumentRefused([Refusal(
                "camera.cache-stale", RefusalClass.USER,
                "the static cache was made from other numbers than this camera's and its static subjects'",
                "make it again with static_cache(subjects, camera, pose), or render without it",
                why="a changed parameter (an override, a moved camera) changes the structure's image")])
        static, rays = cache.transmission, cache.rays
        flags: dict[str, Any] = dict(cache.flags)
    else:
        rays = ray_table(camera, pose)
        flags = {}
        static = _static(held, rays, camera, pose, dtype, corner_samples, flags)
    by_name = {s.name: s for s in held}
    frames = exposure.sub_frames
    flags["beyond_distortion"] = jnp.sum(rays.beyond)
    total = jnp.zeros_like(static)
    centres: dict[str, list[Any]] = {s.name: [] for s in held if s.moving}
    for k in range(frames):
        product = jnp.ones_like(static)
        for subject in held:
            if not subject.moving:
                continue
            position, orientation = _pose_at(poses or {}, subject, k, frames)
            parent = by_name[subject.parent] if subject.parent is not None else None
            parent_pose = None if parent is None else parent.pose if not parent.moving else \
                _pose_at(poses or {}, parent, k, frames)
            factor, pixel = _factor(subject, parent, parent_pose, rays, position, orientation, camera, pose, dtype,
                                    corner_samples, static.shape, flags)
            centres[subject.name].append(pixel)
            product = product * factor
        total = total + product
    for name, held_centres in centres.items():
        if len(held_centres) > 1:
            steps = jnp.stack([jnp.linalg.norm(b - a) for a, b in zip(held_centres, held_centres[1:], strict=False)])
            flags[f"ghosting:{name}"] = jnp.max(steps) > 1.0
    transmission = static * total / frames
    expected, image, saturated = respond(transmission, response, key)
    flags["saturated"] = saturated
    return Capture(image, expected, transmission, flags)


def _projected_reach(camera: Pinhole, pose: CameraPose, origin: Any, radius: float) -> Any:
    """A ball's image's half-size in pixels, to a window's flag: the largest distance from its projected centre to the
    projection of a point on it along the camera's axes, widened by the perspective of its silhouette (whose half-angle
    is ``asin(R / D)``, not ``atan``), and by a twentieth for the distortion's change across it."""
    dtype = jnp.result_type(camera.fx, float)
    axes = _rotation(pose.orientation, dtype)
    points = jnp.stack([origin + sign * radius * axes[:, i] for i in range(2) for sign in (-1.0, 1.0)])
    pixels, _ = project(camera, pose, points)
    centre, _ = project(camera, pose, origin)
    distance = jnp.linalg.norm(origin - jnp.asarray(pose.position, dtype=dtype))
    widened = 1.0 / jnp.sqrt(jnp.maximum(1.0 - (radius / distance) ** 2, 1e-6))
    return 1.05 * widened * jnp.max(jnp.linalg.norm(pixels - centre, axis=-1))


def appearance_sources(subjects: Sequence[Subject], stage: Mapping[str, Any] | None = None) -> tuple[dict[str, AppearanceSource],
                                                                                                     tuple[str, ...]]:
    """Where each subject's appearance comes from (the implicit clear lumens among them), and the build note."""
    _, sources, notes = _prepared(subjects, stage)
    return sources, notes
