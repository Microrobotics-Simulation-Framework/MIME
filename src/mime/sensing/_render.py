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
of every input it was made from (every field of the camera, its pose and each static subject, read from the
dataclasses, and the corner sub-samples); a cache whose key differs from the render's inputs now is refused as stale,
and one whose numbers are traced (a derivative in them) is refused too: the cache carries no derivative in the
intrinsics or the structure, so a calibration renders without it.

**Behind the camera:** every form is taken on the ray (``t >= 0``), so a body behind the camera covers nothing and one
straddling its plane only what lies in front; ``behind`` reports a body whose enclosing ball reaches the plane."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
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


#: Sub-frames this many pixels or more apart (a subject's projected centre between neighbours) raise ``ghosting``.
GHOSTING_PX = 1.0


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


def _lines(subject: Subject, p: Mapping[str, Any], o: Any, d: Any, du: Any, dv: Any) -> list[coverage.Line]:
    """Each of the subject's pieces as its edge's line in the pixel, its gradient read along the ray table's
    derivatives by one ``jvp`` each way."""
    form = subject.placed.form

    def pieces(dx: Any, dy: Any, dz: Any) -> tuple[Any, ...]:
        return form.pieces(p, o, (dx, dy, dz))

    values, linear = _jax.linearize(pieces, *d)
    along_u, along_v = linear(*du), linear(*dv)
    lines = [coverage.line(g, gu, gv) for g, gu, gv in zip(values, along_u, along_v, strict=True)]
    for inner, bound in form.bounds:
        lines[inner] = coverage.greater(lines[inner], lines[bound])
    return lines


def _composed(subject: Subject, lines: Sequence[coverage.Line]) -> coverage.Line:
    return subject.placed.form.compose(lines, coverage.lesser, coverage.greater)


def _box_coverage(subject: Subject, p: Mapping[str, Any], o: Any, d: Any, du: Any, dv: Any,
                  corner_samples: int) -> Any:
    lines = _lines(subject, p, o, d, du, dv)
    single = coverage.covered(_composed(subject, lines))
    if not subject.placed.form.corners or corner_samples <= 1:
        return single
    form = subject.placed.form
    crossing = sum((coverage.near(line).astype(single.dtype) for line in lines), start=jnp.zeros_like(single))
    turning = sum((coverage.near(lines[i]).astype(single.dtype) for i in form.turning), start=jnp.zeros_like(single))
    corner = (crossing >= 2.0) | (turning >= 1.0)

    def one(total: Any, offset: Any) -> tuple[Any, None]:
        su, sv = offset[0], offset[1]
        moved: tuple[Any, Any, Any] = (d[0] + su * du[0] + sv * dv[0], d[1] + su * du[1] + sv * dv[1],
                                       d[2] + su * du[2] + sv * dv[2])
        scale = 1.0 / jnp.sqrt(forms.dot(moved, moved))
        sub = _composed(subject, _lines(subject, p, o, tuple(c * scale for c in moved), du, dv))
        return total + coverage.covered(sub, 1.0 / corner_samples), None

    def sub_sampled() -> Any:
        offsets = jnp.asarray(coverage.sub_offsets(corner_samples), dtype=single.dtype)
        total, _ = lax.scan(_jax.checkpoint(one, prevent_cse=False), jnp.zeros_like(single), offsets)
        return total / corner_samples ** 2

    # Sub-sampled where some pixel holds a corner (two pieces cross it) or a piece whose own outline turns within a
    # pixel (a cap seen near edge-on, its ellipse's ends a fraction of a pixel round): every pixel then takes its
    # sub-samples, which cost the same once computed, and which no piece's line can steer from (inside a cap's thin
    # ellipse its first-order distance is inflated, past the corner reach); a tube whose ends are out of view pays
    # nothing for them.
    # Each sub-sample is checkpointed: reverse mode keeps a cond's residuals as large as its largest branch at every
    # step whichever branch runs, and a scene whose corners never come into view would otherwise keep 16 sub-samples'
    # (camera spike R1: 96 % of the in-loop camera's tape). A program that takes no reverse derivative is unchanged: a
    # checkpoint lowers to its body, and prevent_cse=False (inside a scan, which keeps the recomputation apart) adds no
    # barrier.
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
    interval = forms.ahead(form.interval(p, o, d))
    if subject.coverage == "disc":
        camera_rotation = _rotation(pose.orientation, dtype)
        centre = camera_rotation.T @ (origin - jnp.asarray(pose.position, dtype=dtype))
        corners_x, corners_y = _corners(camera, start, rays.beyond.shape)

        def exact() -> Any:
            return coverage.disc(corners_x, corners_y, centre, p["radius"])

        def on_the_ray() -> Any:
            return _box_coverage(subject, p, o, d, du, dv, corner_samples)

        # The conic is an ellipse only for a ball wholly in front of the camera; one reaching behind it (its image
        # unbounded, or nothing) is covered by the box edge on the ray.
        covered = lax.cond(centre[2] > p["radius"], exact, on_the_ray)
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


def _window_rays(rays: Rays, start: tuple[Any, Any], size: tuple[int, int]) -> Rays:
    def cut(a: Any) -> Any:
        return lax.dynamic_slice(a, start, size)

    return Rays(rays.origin, tuple(cut(a) for a in rays.direction), tuple(cut(a) for a in rays.along_u),
                tuple(cut(a) for a in rays.along_v), cut(rays.beyond))


def _implicit_lumens(subjects: Sequence[Subject]) -> list[Subject]:
    """A clear lumen for each tube's wall whose lumen no subject images: the lumen displaces the wall's material."""
    filled = {s.parent for s in subjects if s.kind == "tube" and s.region == "lumen"}
    return [Subject(f"{s.name}.lumen", "tube", s.settings, "lumen", Appearance.clear(), s.name, pose=s.pose)
            for s in subjects if s.kind == "tube" and s.region == "wall" and s.name not in filled]


#: A subject's fields made from its others (its placed form, from its kind, settings and region): left out of the key.
DERIVED_FIELDS = frozenset({"placed"})


def _numbers(value: Any) -> Any:
    """``value``'s numbers as plain Python numbers, a dataclass's by its every field (but those derived from others),
    an enumeration by its value (raising on a traced value)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _numbers(getattr(value, f.name)) for f in dataclasses.fields(value)
                if f.name not in DERIVED_FIELDS}
    if isinstance(value, Enum):
        return _numbers(value.value)
    if isinstance(value, Mapping):
        held = cast(Mapping[str, Any], value)
        return {str(k): _numbers(held[k]) for k in sorted(held, key=str)}
    if isinstance(value, (tuple, list)):
        return [_numbers(v) for v in cast(Sequence[Any], value)]
    if isinstance(value, (str, bool)) or value is None:
        return value
    return np.asarray(value, dtype=np.float64).tolist()


def cache_key(subjects: Sequence[Subject], camera: Pinhole, pose: CameraPose, dtype: Any,
              corner_samples: int = coverage.CORNER_SAMPLES) -> str:
    """The hash of every input the static subjects' transmission and flags are made from: the camera's every field, its
    pose, each static subject's every field (with its parent's appearance, which it displaces), the corner sub-samples,
    the dtype and the renderer's own declared constants. Read from the dataclasses' fields, so a field added later is
    keyed without a change here."""
    by_name = {s.name: s for s in subjects}
    try:
        record = {
            "camera": _numbers(camera),
            "pose": _numbers(pose),
            "subjects": [[_numbers(s), None if s.parent is None else _numbers(by_name[s.parent].appearance)]
                         for s in subjects if not s.moving],
            "corner_samples": corner_samples,
            "dtype": str(np.dtype(dtype)),
            "constants": [forms.GRAZING, forms.PARALLEL, forms.FAR, coverage.CORNER_REACH, coverage.FLAT,
                          detector.UNDISTORT_STEPS, detector.TANGENTIAL_STEPS, SMALL_SILHOUETTE_PX, THIN_CAP_SUBPIXELS,
                          SILHOUETTE_POINTS, list(SILHOUETTE_MARGIN)],
        }
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError) as error:
        raise ArgumentRefused([Refusal(
            "camera.cache-traced", RefusalClass.USER,
            "a static cache was given while the camera's or a static subject's numbers are traced (a derivative in them)",
            "render without the cache (cache=None) to differentiate in the intrinsics or the structure",
            why="the cache holds the structure's image at fixed numbers, with no derivative in them")]) from error
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _window_of(subject: Subject, camera: Pinhole, pose: CameraPose, origin: Any) -> tuple[int, int] | None:
    """A subject's window's half-sizes in pixels, ``(rows, columns)``: its declared one each way, or for exact disc
    coverage (always windowed) its silhouette's bounding box about its projected centre at its pose with a pixel's
    margin, which needs the pose and the camera as numbers (the whole image where the ball reaches behind the camera);
    ``None`` for a box-edged subject with none declared (it is rendered over the whole frame)."""
    if subject.window is not None:
        return subject.window, subject.window
    if subject.coverage != "disc":
        return None
    try:
        low, high, in_front = (np.asarray(x) for x in _silhouette_box(camera, pose, origin, subject.enclosing()))
        centre = np.round(np.asarray(project(camera, pose, origin)[0]))
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError) as error:
        raise ArgumentRefused([Refusal(
            "camera.subject", RefusalClass.USER,
            f"subject {subject.name!r}'s exact disc coverage is windowed, and its window cannot be sized from a traced "
            "pose or camera", "declare its half-size in pixels: Subject(..., window=24), or take the box edge: "
            "Subject(..., coverage=\"box_edge\")",
            why="a window's size is fixed when the program is made; a derivative traces the numbers it would be made "
                "from")]) from error
    if not bool(in_front):
        return camera.height, camera.width
    reach = np.maximum(centre - low, high - centre)
    return math.ceil(float(reach[1])) + 1, math.ceil(float(reach[0])) + 1


def _factor(subject: Subject, parent: Subject | None, parent_pose: tuple[Any, Any] | None, rays: Rays, position: Any,
            orientation: Any, camera: Pinhole, pose: CameraPose, dtype: Any, corner_samples: int, shape: Any,
            flags: dict[str, Any]) -> tuple[Any, Any]:
    """One subject's factor on the whole image (``1 - c + c tau``; ones outside its window), and its projected centre;
    its flags written into ``flags``."""
    rotation, origin = _local(subject, position, orientation, dtype)
    pixel, depth = project(camera, pose, origin)
    window = _window_of(subject, camera, pose, origin)
    if window is not None:
        size = _window_size(window, camera)
        start = (jnp.clip(jnp.round(pixel[1]).astype(jnp.int32) - window[0], 0, camera.height - size[0]),
                 jnp.clip(jnp.round(pixel[0]).astype(jnp.int32) - window[1], 0, camera.width - size[1]))
        seen_rays = _window_rays(rays, start, size)
    else:
        size, start, seen_rays = None, None, rays
    seen = _seen(subject, seen_rays, rotation, origin, dtype, camera, pose, start, corner_samples)
    assert subject.appearance is not None
    chord = forms.chord(seen.interval)
    factor = 1.0 - seen.coverage + seen.coverage * _transmits(
        subject.appearance, parent.appearance if parent is not None else None, chord, dtype)
    name = subject.name
    if parent is not None and parent_pose is not None:
        parent_rotation, parent_origin = _local(parent, parent_pose[0], parent_pose[1], dtype)
        o = _into(parent_rotation, tuple(seen_rays.origin[i] - parent_origin[i] for i in range(3)))
        inside = forms.ahead(parent.placed.form.interval(_parameters(parent, dtype), o,
                                                         _into(parent_rotation, seen_rays.direction)))
        limit = 2.0 * subject.enclosing() * math.sqrt(2.0 * subject.tolerance)
        outside = jnp.sum((chord - forms.overlap(seen.interval, inside)) > limit)
        flags[f"outside_parent:{name}"] = flags.get(f"outside_parent:{name}", 0) + outside
    if start is not None and size is not None:
        factor = lax.dynamic_update_slice(jnp.ones(shape, dtype=factor.dtype), factor, start)
        flags[f"window_overflow:{name}"] = flags.get(f"window_overflow:{name}", False) | _overflows(
            camera, pose, origin, subject.enclosing(), start, size)
    if subject.coverage == "box_edge":
        small = camera.fx * subject.feature() / depth < SMALL_SILHOUETTE_PX
        flags[f"small_silhouette:{name}"] = flags.get(f"small_silhouette:{name}", False) | small
    flags[f"behind:{name}"] = flags.get(f"behind:{name}", False) | _behind(depth, subject)
    if subject.placed.form.corners:
        flags[f"thin_cap:{name}"] = flags.get(f"thin_cap:{name}", False) | _thin_cap(
            subject, rotation, origin, camera, pose, dtype, corner_samples)
    return factor, pixel


def _window_size(window: tuple[int, int], camera: Pinhole) -> tuple[int, int]:
    """A window's rows and columns: each axis clipped to the image on its own (a window wider than the image's shorter
    side keeps its width)."""
    return min(2 * window[0] + 1, camera.height), min(2 * window[1] + 1, camera.width)


def _behind(depth: Any, subject: Subject) -> Any:
    """Whether a subject is not wholly in front of the camera: its enclosing ball reaches the camera's plane (the part
    behind it is not imaged)."""
    return depth < subject.enclosing()


def _overflows(camera: Pinhole, pose: CameraPose, origin: Any, radius: float, start: tuple[Any, Any],
               size: tuple[int, int]) -> Any:
    """Whether any part of a ball's silhouette that lies in the image lies outside the window (``start``, ``size``:
    rows, then columns); for a ball reaching behind the camera, whether the window is less than the whole image."""
    low, high, in_front = _silhouette_box(camera, pose, origin, radius)
    lost = jnp.zeros((), dtype=bool)
    for axis, extent in ((0, camera.width), (1, camera.height)):
        first, count = start[1 - axis], size[1 - axis]
        seen_low = jnp.clip(low[axis], -0.5, extent - 0.5)
        seen_high = jnp.clip(high[axis], -0.5, extent - 0.5)
        showing = seen_high > seen_low
        lost = lost | (showing & ((seen_low < first - 0.5) | (seen_high > first + count - 0.5)))
    whole = size == (camera.height, camera.width)
    return jnp.where(in_front, lost, not whole)


#: A cap whose ellipse is narrower than this many sub-pixels across (seen near edge-on) and in view raises
#: ``thin_cap``. The sub-samples' straight lines resolve a cap's outline where its ellipse spans two sub-pixels or more;
#: in a narrower one a sub-pixel holds both of its sides and the cap's first-order line there is inflated (its edge
#: function changes fastest across the ellipse), and a pixel misses by up to about one sub-pixel's area. Measured
#: against the exact hull (``tests/sensing/test_corners.py``): with 4 x 4 sub-samples up to 7.7 % at ellipses 0.16 to
#: 0.25 px across, within 3.9 % at every view measured with no cap narrower than half a pixel; at the worst thin view 2.7 % with
#: 8 x 8 and 0.7 % with 16 x 16 (resolution). The threshold is declared and measured at the default sub-samples only.
THIN_CAP_SUBPIXELS = 2.0


def _thin_cap(subject: Subject, rotation: Any, origin: Any, camera: Pinhole, pose: CameraPose, dtype: Any,
              corner_samples: int) -> Any:
    """Whether a capped cylinder shows a cap in view whose ellipse is narrower than ``THIN_CAP_SUBPIXELS`` sub-pixels
    (``1 / corner_samples`` px each): its minor axis, twice ``f R |cos|`` over its depth, the cosine between the line
    of sight and the axis."""
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
        found = found | (seen & (across < THIN_CAP_SUBPIXELS / max(corner_samples, 1)))
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
    return StaticCache(cache_key(held, camera, pose, dtype, corner_samples), transmission, rays,
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


def frame_key(seed: int, step: Any) -> Any:
    """The noise key of the frame made at ``step`` (the in-loop camera's count at that step) by the stream whose seed is
    ``seed``: rendering the recorded poses again with it reproduces the frame."""
    return jax.random.fold_in(jax.random.PRNGKey(seed), step)


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
        if cache_key(held, camera, pose, dtype, corner_samples) != cache.key:
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
    return exposed(held, camera, pose, rays, static, flags, poses or {}, exposure, response, key, corner_samples)


def exposed(held: Sequence[Subject], camera: Pinhole, pose: CameraPose, rays: Rays, static: Any, flags: dict[str, Any],
            poses: Mapping[str, Any], exposure: Exposure, response: Response, key: Any,
            corner_samples: int = coverage.CORNER_SAMPLES) -> Capture:
    """The moving subjects at ``poses`` (one per sub-frame) over the static structure's transmission ``static`` and the
    ``rays`` it was made with, through the exposure and the response: ``render`` after its static part. The in-loop
    camera calls it with the static part it holds."""
    dtype = jnp.result_type(camera.fx, float)
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
            position, orientation = _pose_at(poses, subject, k, frames)
            parent = by_name[subject.parent] if subject.parent is not None else None
            parent_pose = None if parent is None else parent.pose if not parent.moving else \
                _pose_at(poses, parent, k, frames)
            factor, pixel = _factor(subject, parent, parent_pose, rays, position, orientation, camera, pose, dtype,
                                    corner_samples, static.shape, flags)
            centres[subject.name].append(pixel)
            product = product * factor
        total = total + product
    for name, held_centres in centres.items():
        if len(held_centres) > 1:
            steps = jnp.stack([jnp.linalg.norm(b - a) for a, b in zip(held_centres, held_centres[1:], strict=False)])
            flags[f"ghosting:{name}"] = jnp.max(steps) >= GHOSTING_PX
    transmission = static * total / frames
    expected, image, saturated = respond(transmission, response, key)
    flags["saturated"] = saturated
    return Capture(image, expected, transmission, flags)


#: Points of a ball's tangent circle projected to bound its silhouette, and the margin (pixels, plus a share of the
#: silhouette's size) added for the arcs between them: a chord of a circle cut in N arcs lies within
#: ``1 - cos(pi / N)`` of its radius (3e-4 at 128) of the arc, and the distortion's change across an arc is smaller.
SILHOUETTE_POINTS = 128
SILHOUETTE_MARGIN = (0.5, 1e-3)


def _silhouette_box(camera: Pinhole, pose: CameraPose, origin: Any, radius: float) -> tuple[Any, Any, Any]:
    """The bounding box in pixels of the silhouette of a ball of ``radius`` at ``origin`` (its least and greatest
    ``(u, v)``), from points of its tangent circle through the camera's projection (distortion included), and whether
    the ball is wholly in front of the camera (otherwise its silhouette is unbounded or empty, and the box is not
    read)."""
    dtype = jnp.result_type(camera.fx, float)
    axes = _rotation(pose.orientation, dtype)
    sight = origin - jnp.asarray(pose.position, dtype=dtype)
    distance = jnp.sqrt(jnp.sum(sight * sight))
    axis = sight / distance
    # A direction across the line of sight: the camera's x axis made orthogonal to it, or its y axis where the line of
    # sight is near the x axis.
    across = [axes[:, i] - jnp.sum(axes[:, i] * axis) * axis for i in range(2)]
    lengths = [jnp.sum(a * a) for a in across]
    first = jnp.where(lengths[0] > 0.25, across[0] / jnp.sqrt(lengths[0]), across[1] / jnp.sqrt(lengths[1]))
    second = jnp.cross(axis, first)
    sine = jnp.minimum(radius / distance, 1.0)
    cosine = jnp.sqrt(1.0 - sine * sine)
    phi = jnp.arange(SILHOUETTE_POINTS, dtype=dtype) * (2.0 * math.pi / SILHOUETTE_POINTS)
    ring = (cosine * axis[None, :] + sine * (jnp.cos(phi)[:, None] * first[None, :]
                                             + jnp.sin(phi)[:, None] * second[None, :]))
    points = jnp.asarray(pose.position, dtype=dtype)[None, :] + distance * cosine * ring
    pixels, _ = project(camera, pose, points)
    low, high = jnp.min(pixels, axis=0), jnp.max(pixels, axis=0)
    margin = SILHOUETTE_MARGIN[0] + SILHOUETTE_MARGIN[1] * (high - low)
    _, depth = project(camera, pose, origin)
    return low - margin, high + margin, depth > radius


def appearance_sources(subjects: Sequence[Subject], stage: Mapping[str, Any] | None = None) -> tuple[dict[str, AppearanceSource],
                                                                                                     tuple[str, ...]]:
    """Where each subject's appearance comes from (the implicit clear lumens among them), and the build note."""
    _, sources, notes = _prepared(subjects, stage)
    return sources, notes


#: The renderer's parts the in-loop camera's node calls, under public names: the node holds the static structure's image
#: and the rays itself, guarded by its leaves, and sizes each disc subject's window from its pose at the start.
prepared = _prepared
static_part = _static


def projected_reach(camera: Pinhole, pose: CameraPose, origin: Any, radius: float) -> Any:
    """The greatest distance in pixels from a ball's projected centre to its silhouette's bounding box (``_silhouette_box``,
    the distortion included), the largest over both axes; not finite where the ball reaches behind the camera."""
    low, high, in_front = _silhouette_box(camera, pose, origin, radius)
    centre = project(camera, pose, origin)[0]
    reach = jnp.max(jnp.maximum(centre - low, high - centre))
    return jnp.where(in_front, reach, jnp.inf)
