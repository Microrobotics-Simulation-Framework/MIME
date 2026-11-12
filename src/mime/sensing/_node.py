"""The in-loop camera's node: the sensor core of ``mime.sensing`` on the base step's clock.

**The clock.** The node steps at the base step (``rate=`` is refused: a slower node is computed every base step and its
result discarded, so it would save nothing). Its own count ``steps`` runs from zero; a *frame step* is a step whose count is
a multiple of ``period`` (the camera's ``frame_period`` in base steps), and there the node renders; at every other step it
holds the image. ``frame_step`` is the count of the step the held image was made at (``-1`` before the first), ``age`` the
steps since (zero on a frame step), and ``new_frame`` one on a frame step.

**The exposure.** A ring of the last ``sub_frames`` poses of every body is written at every base step, the newest being the
pose read at this step; a frame integrates those ``sub_frames`` poses, so the exposure is ``sub_frames`` base steps ending
at the frame step. Before the run's first poses the ring holds the bodies' initial poses.

**The noise.** A frame's key is the stream's seed folded with the frame's step (:func:`frame_key`), so that rendering the
recorded poses again with that key reproduces the frame exactly.

**The leaves.** The detector's nine numbers (``intrinsics``: ``fx, fy, cx, cy, k1, k2, p1, p2, k3``) are the node's leaves,
so a study can fit them; the ray table and the structure's image follow them. The build's static cache (a camera fixed in
the world) is used only while the leaves equal the numbers it was made from, and a derivative in them takes the uncached
path, so a stale cache is never read. ``expected`` (the counts before noise and quantisation) is the differentiable
output; ``image`` is the quantised frame.

**Forward mode only.** A derivative passes in forward mode (the ring's poses and the leaves); reverse mode and a batched
study are refused by the study (``Observation.forward_only``), naming the offline renderer, because the cond is a
``select`` under ``vmap`` and reverse mode would keep every frame step's render.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast, override

import jax
import numpy as np
from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, node_input, \
    typed_view
from mime.core.multibody import compose
from mime.nodes.actuation._chain import POSE_UNITS, MountedNode
from mime.sensing import _scene as scene
from mime.sensing._detector import BrownConrady, CameraPose, Pinhole, Rays
from mime.sensing._coverage import CORNER_SAMPLES
from mime.sensing._render import (Exposure, Response, StaticCache, exposed, frame_key, prepared, projected_reach,
                                  static_cache, static_part)
from mime.sensing._detector import ray_table
from mime.sensing._subjects import Subject

lax: Any = jax.lax
_jax: Any = cast(Any, jax)

#: The flags a frame raises, each a state field of the node holding, on a frame step, the number of subjects (or pixels,
#: or the share of saturated ones) it raised for in that frame, and zero at every other step.
#: The detector's nine numbers, the node's leaves, in the order a Pinhole and OpenCV give them.
INTRINSICS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
#: How far a focal length may move from the build's, as a share: the box a fit stays in, and what the disc windows are
#: sized for. Past it the silhouette could leave its window, which a flag would report, so the leaf's bounds keep it in.
FOCAL_SHARE = 0.25
#: How far a distortion coefficient may move from the build's: the radial ones by this, the tangential ones by a fifth.
DISTORTION_BOX = 0.3
FLAG_FIELDS = ("beyond_distortion", "saturated", "outside_parent", "small_silhouette", "window_overflow", "behind",
               "ghosting", "thin_cap")

EQUATIONS = (r"n_{f} = P \, \mathrm{floor}(n / P), \quad a = n - n_{f}, \quad "
             r"I = \mathrm{render}\left(\mathbf{c}, s, n_{f}\right)")

_NODE = "mime.sensing._node.CameraNode"


class CameraState(TypedDict, closed=True):
    steps: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    frame_step: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    age: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    new_frame: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    image: ReadOnly[Annotated[Float[Array, "rows columns"], Units("1")]]
    expected: ReadOnly[Annotated[Float[Array, "rows columns"], Units("1")]]
    ring: ReadOnly[Annotated[Float[Array, "sub_frames bodies 7"], Units(POSE_UNITS)]]
    cache_origin: ReadOnly[Annotated[Float[Array, "3"], Units("m")]]
    cache_rays: ReadOnly[Annotated[Float[Array, "10 rows columns"], Units("1")]]
    cache_transmission: ReadOnly[Annotated[Float[Array, "rows columns"], Units("1")]]
    cache_flags: ReadOnly[Annotated[Float[Array, "static_flags"], Units("1")]]
    beyond_distortion: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    saturated: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    outside_parent: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    small_silhouette: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    window_overflow: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    behind: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    ghosting: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    thin_cap: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class CameraInputs(TypedDict, extra_items=ReadOnly[Float[Array, "7"]]):
    """``pose_<body>`` for each body (its position and orientation), and ``mount_poses`` for the frames of the chain
    segment the camera is mounted on."""

    mount_poses: NotRequired[ReadOnly[Float[Array, "mounted 7"]]]
    mount_rates: NotRequired[ReadOnly[Float[Array, "mounted 3"]]]


def detector_of(spec: Mapping[str, Any], intrinsics: Any) -> Pinhole:
    """The detector of ``spec`` (its size and where its numbers come from) with the nine ``intrinsics``
    ``(fx, fy, cx, cy, k1, k2, p1, p2, k3)``, which may be traced: they are the node's leaves."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = (intrinsics[i] for i in range(9))
    return Pinhole(spec["width"], spec["height"], fx, fy, cx, cy, BrownConrady(k1, k2, p1, p2, k3, source=spec["source"]))


def response_of(spec: Mapping[str, Any]) -> Response:
    return Response(**spec["response"])


def _composed(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """``inner``, a pose in ``outer``'s frame, in the world, in float64 (a pose is ``x, y, z, w, x, y, z``)."""
    q, p = outer[3:], inner[3:]
    w, x, y, z = q
    turned = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                       [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                       [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    pw, px, py, pz = p
    product = np.array([w * pw - x * px - y * py - z * pz, w * px + x * pw + y * pz - z * py,
                        w * py - x * pz + y * pw + z * px, w * pz + x * py - y * px + z * pw])
    return np.concatenate([outer[:3] + turned @ inner[:3], product])


SCENE_HELD = ("held as built: the scene's numbers (the subjects' settings and appearances, the response) are consumed when "
              "the node is built, into its subjects and the static structure's image, so that a new value takes a "
              "rebuild (a study refuses to vary it)")


class CameraNode(MountedNode):
    """The camera's frame on the base step's clock: the whole scene (its ``spec``, a JSON string of the detector, the
    subjects, the response and the key's seed), rendered every ``period`` steps from the ring of the last ``sub_frames``
    poses of every body in ``bodies`` (the inputs ``pose_<body>``), seen from the camera's pose: its ``placement`` in the
    world, or on the frame ``mount_index`` of the ``mount_poses`` it is mounted on. The static structure is rendered once,
    at the node's construction, where the camera is fixed in the world; where it is mounted on a frame that may move it is
    rendered in every frame, and no cache is ever held (a stale one would be a wrong image)."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-133",
        description="A declared camera model in the loop: the sensor core's frame of the whole scene, rendered on "
                    "frame steps of the base step's clock from a ring of the bodies' poses, held between them.",
        governing_equations=EQUATIONS,
        assumptions=("The frame period is a whole number of base steps and a frame is rendered at every step whose count "
                     "is a multiple of it, exposed over the sub-frames' base steps ending at that step",
                     "The scene is the experiment's bodies (their outermost parts) and structures, imaged by the "
                     "closed form of their kinds; a body is taken to move, a structure to stand"),
        limitations=("Forward only: a derivative through it is refused",
                     "A camera on a frame that moves is exposed at the frame step's pose throughout the exposure",
                     "A declared camera model: backlit, one bin, no optics beyond the pinhole and its distortion"),
    )
    symbols = (
        Symbol("n", SymbolKind.STATE, "steps", "1", f"{_NODE}.advance"),
        Symbol("P", SymbolKind.PARAMETER, "period", "1", f"{_NODE}.advance"),
        Symbol("s", SymbolKind.PARAMETER, "seed", "1", f"{_NODE}.advance"),
        Symbol("n_{f}", SymbolKind.STATE, "frame_step", "1", f"{_NODE}.advance"),
        Symbol("a", SymbolKind.STATE, "age", "1", f"{_NODE}.advance"),
        Symbol("I", SymbolKind.STATE, "image", "1", f"{_NODE}.advance"),
        Symbol(r"\mathbf{c}", SymbolKind.PARAMETER, "placement", POSE_UNITS, f"{_NODE}.camera_pose"),
    )

    def __init__(self, name: str, timestep: float, *, spec: str, fx: float, fy: float, cx: float, cy: float, k1: float,
                 k2: float, p1: float, p2: float, k3: float, bodies: Sequence[str], initial_poses: Sequence[float],
                 period: int, sub_frames: int, seed: int, placement: Sequence[float], mount_index: int = -1,
                 mount_count: int = 0, initial_mount_pose: Sequence[float] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)) -> None:
        super().__init__(name, timestep, spec=spec, scene_numbers=scene.numbers(json.loads(spec)), fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), k1=float(k1),
                         k2=float(k2), p1=float(p1), p2=float(p2), k3=float(k3), bodies=tuple(bodies),
                         initial_poses=tuple(float(c) for c in initial_poses), period=int(period),
                         sub_frames=int(sub_frames), seed=int(seed), placement=tuple(float(c) for c in placement),
                         mount_index=int(mount_index), mount_count=int(mount_count),
                         initial_mount_pose=tuple(float(c) for c in initial_mount_pose))
        self._held = self._made()

    # -- what the node holds, made once from its constants ----------------------------------------------------------

    def _made(self) -> tuple[Pinhole, tuple[Subject, ...], StaticCache | None]:
        """The detector, the subjects (each disc-covered body's window sized from its initial pose) and, for a camera
        fixed in the world, the static structure's cache."""
        held = self.constants()
        spec = json.loads(cast(str, held["spec"]))
        intrinsics = self.build_intrinsics()
        camera = detector_of(spec, intrinsics)
        reach_camera = detector_of(spec, (intrinsics[0] * (1.0 + FOCAL_SHARE), intrinsics[1] * (1.0 + FOCAL_SHARE),
                                          *intrinsics[2:]))
        poses = np.asarray(held["initial_poses"], dtype=np.float64).reshape(-1, 7)
        bodies = cast(tuple[str, ...], held["bodies"])
        initial = np.asarray(held["placement"], dtype=np.float64)
        if cast(int, held["mount_index"]) >= 0:
            initial = _composed(np.asarray(held["initial_mount_pose"], dtype=np.float64), initial)
        pose0 = CameraPose(initial[:3], initial[3:])
        owner = {subject: body for body, names in spec["body_subjects"].items() for subject in names}
        made: list[Subject] = []
        for each in spec["subjects"]:
            subject = scene.subject_of(each)
            if subject.moving and subject.coverage == "disc" and subject.window is None:
                at = poses[bodies.index(owner[subject.name])]
                origin = _composed(at, np.concatenate([np.asarray(subject.placed.centre), [1.0, 0.0, 0.0, 0.0]]))[:3]
                reach = float(np.asarray(projected_reach(reach_camera, pose0, origin, subject.enclosing())))
                subject = scene.subject_of({**each, "window": scene.frame_window(reach, camera.width, camera.height)})
            made.append(subject)
        cache = static_cache(made, camera, pose0) if self.holds_cache() else None
        return camera, tuple(prepared(made, None)[0]), cache

    def build_intrinsics(self) -> tuple[float, ...]:
        """The detector's nine numbers as the node was built with them."""
        held = self.constants()
        return tuple(float(held[name]) for name in INTRINSICS)

    def holds_cache(self) -> bool:
        """Whether the build holds the static structure's image: only where the camera is fixed in the world, where the
        camera and the static structure share a frame. A camera on a frame that may move never does."""
        return cast(int, self.constants()["mount_index"]) < 0

    def _spec(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], json.loads(cast(str, self.constants()["spec"])))

    def _bodies(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.constants()["bodies"])

    def image_shape(self) -> tuple[int, int]:
        camera = self._held[0]
        return camera.height, camera.width

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(),
                **self.intrinsic_specs(),
                "scene_numbers": ParamSpec(trainable=False, units="1", description=SCENE_HELD),
                "period": ParamSpec(trainable=False, units="1", description="base steps between frames"),
                "seed": ParamSpec(trainable=False, units="1", description="the noise stream's seed, folded with the "
                                                                          "frame's step"),
                "placement": ParamSpec(trainable=False, units=POSE_UNITS,
                                       description="the camera's pose in the world, or on its mount frame: held with "
                                                   "the static structure's cache, so a move needs a new build"),
                "initial_poses": ParamSpec(trainable=False, units=POSE_UNITS,
                                           description="the bodies' poses at the start, which fill the ring")}

    def intrinsic_specs(self) -> dict[str, ParamSpec]:
        """The detector's nine leaves, each bounded about its build value: a focal length by ``FOCAL_SHARE``, a centre by
        a fifth of the image, a distortion coefficient by ``DISTORTION_BOX`` (a tangential one a fifth of it). The box
        keeps a fit where the disc windows, sized for it, hold the silhouettes."""
        spec = self._spec()
        held = dict(zip(INTRINSICS, self.build_intrinsics(), strict=True))
        size = {"cx": spec["width"], "cy": spec["height"]}
        box: dict[str, float] = {"fx": FOCAL_SHARE * abs(held["fx"]), "fy": FOCAL_SHARE * abs(held["fy"]),
                                 "cx": 0.2 * size["cx"], "cy": 0.2 * size["cy"], "k1": DISTORTION_BOX,
                                 "k2": DISTORTION_BOX, "k3": DISTORTION_BOX, "p1": DISTORTION_BOX / 5,
                                 "p2": DISTORTION_BOX / 5}
        units = {name: "1" for name in INTRINSICS}
        return {name: ParamSpec(units=units[name], bounds=(held[name] - box[name], held[name] + box[name]),
                                description=f"the detector's {name}: a leaf a calibration fits, within its box")
                for name in INTRINSICS}

    # -- the step ---------------------------------------------------------------------------------------------------

    @override
    def initial_state(self) -> NodeFields:
        held = self.constants()
        sub_frames = cast(int, held["sub_frames"])
        poses = as_run_array(np.asarray(held["initial_poses"]).reshape(-1, 7))
        scalar = as_run_array(0.0)
        zero_int = jnp.asarray(0, dtype=jnp.int32)
        rows, columns = self.image_shape()
        flags = {name: scalar for name in FLAG_FIELDS}
        return node_fields(CameraState(
            steps=zero_int, frame_step=jnp.asarray(-1, dtype=jnp.int32), age=zero_int, new_frame=zero_int,
            image=as_run_array(np.zeros((rows, columns))), expected=as_run_array(np.zeros((rows, columns))),
            ring=jnp.broadcast_to(poses[None], (sub_frames, *poses.shape)),
            cache_origin=as_run_array(np.zeros(3)), cache_rays=as_run_array(np.zeros((10, rows, columns))),
            cache_transmission=as_run_array(np.zeros((rows, columns))),
            cache_flags=as_run_array(np.zeros(len(self._static_flag_names()))), **cast(Any, flags)))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {**{f"pose_{body}": node_input(POSE_UNITS, (7,)) for body in self._bodies()}, **self.mounted_inputs()}

    def camera_pose(self, given: Mapping[str, Array], dtype: Any) -> CameraPose:
        """The camera's optical frame in the world: its placement, composed onto its mount frame's pose where it has
        one."""
        held = self.constants()
        placement = jnp.asarray(np.asarray(held["placement"]), dtype=dtype)
        index = cast(int, held["mount_index"])
        if index < 0:
            return CameraPose(np.asarray(held["placement"], dtype=np.float64)[:3],
                              np.asarray(held["placement"], dtype=np.float64)[3:])
        base = compose(jnp.asarray(given["mount_poses"][index], dtype=dtype), placement)
        return CameraPose(base[:3], base[3:])

    def _static_flag_names(self) -> tuple[str, ...]:
        """The flags the static structure raises, by name: those of the cache the build made (none for a camera that holds
        none)."""
        cache = self._held[2]
        return tuple(sorted(cache.flags)) if cache is not None else ()

    def _static_part(self, intrinsics: Array, pose: CameraPose, dtype: Any, count: Array,
                     held: tuple[Array, Array, Array, Array]) -> tuple[Rays, Array, dict[str, Any], tuple[Array, ...]]:
        """The rays and the static structure's transmission at ``intrinsics``, and the cache fields to carry on. Where the
        camera is fixed in the world the node holds them in its state: made from the step's own ``intrinsics`` at the first
        step, read from the state after, so that no number made from a leaf is a constant of the program and a leaf a study
        moved is the cache's too. A camera on a frame that may move has no cache: it renders them in every frame."""
        _, subjects, cache = self._held
        spec = self._spec()
        moved = detector_of(spec, intrinsics)

        def fresh() -> tuple[Rays, Array, dict[str, Any]]:
            rays = ray_table(moved, pose)
            flags: dict[str, Any] = {}
            return rays, static_part(subjects, rays, moved, pose, dtype, CORNER_SAMPLES, flags), flags

        if cache is None:
            rays, static, flags = fresh()
            return rays, static, flags, held
        names = self._static_flag_names()

        def made() -> tuple[Array, ...]:
            rays, static, flags = fresh()
            return (jnp.stack([jnp.asarray(c, dtype=dtype) for c in rays.origin]),
                    jnp.stack([jnp.asarray(c, dtype=dtype) for part in (rays.direction, rays.along_u, rays.along_v)
                               for c in part] + [jnp.asarray(rays.beyond, dtype=dtype)]),
                    jnp.asarray(static, dtype=dtype),
                    jnp.stack([jnp.asarray(flags[name], dtype=dtype) for name in names]) if names
                    else jnp.zeros((0,), dtype=dtype))

        carried = lax.cond(count == 0, made, lambda: held)
        origin, packed, static, flag_values = carried
        rays = Rays(origin=(origin[0], origin[1], origin[2]), direction=(packed[0], packed[1], packed[2]),
                    along_u=(packed[3], packed[4], packed[5]), along_v=(packed[6], packed[7], packed[8]),
                    beyond=packed[9] > 0.5)
        return rays, static, {name: flag_values[i] for i, name in enumerate(names)}, carried

    def _rendered(self, ring: Array, intrinsics: Array, pose: CameraPose, key: Any, dtype: Any, count: Array,
                  cached: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, dict[str, Array], tuple[Array, ...]]:
        _, subjects, _ = self._held
        held = self.constants()
        spec = self._spec()
        sub_frames = cast(int, held["sub_frames"])
        moves: dict[str, Any] = {}
        for index, body in enumerate(self._bodies()):
            for name in spec["body_subjects"][body]:
                moves[name] = (ring[:, index, :3], ring[:, index, 3:]) if sub_frames > 1 \
                    else (ring[0, index, :3], ring[0, index, 3:])
        rays, static, flags, carried = self._static_part(intrinsics, pose, dtype, count, cached)
        camera = detector_of(spec, intrinsics)
        captured = exposed(subjects, camera, pose, rays, static, dict(flags), moves, Exposure(sub_frames),
                           response_of(spec), key)
        totals = {name: jnp.zeros((), dtype=dtype) for name in FLAG_FIELDS}
        for flag, value in captured.flags.items():
            kind = flag.split(":")[0]
            if kind in totals:
                totals[kind] = totals[kind] + jnp.asarray(value, dtype=dtype)
        return captured.image.astype(dtype), captured.expected.astype(dtype), totals, carried

    def frame_due(self, count: Array, period: int) -> Array:
        """Whether the step with count ``count`` is a frame step."""
        return (count % period) == 0

    def pushed(self, ring: Array, poses: Array) -> Array:
        """The ring with this step's poses its newest."""
        return jnp.concatenate([ring[1:], poses[None]])

    def exposed(self, before: Array, ring: Array) -> Array:
        """The poses a frame is exposed over: the ring as this step leaves it."""
        del before
        return ring

    def key_of(self, seed: int, count: Array) -> Any:
        """The noise key of the frame made at ``count``."""
        return frame_key(seed, count)

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held, given = typed_view(CameraState, state), typed_view(CameraInputs, boundary_inputs)
        constants = self.constants()
        period, seed = cast(int, constants["period"]), cast(int, constants["seed"])
        count = held["steps"]
        dtype = held["image"].dtype
        poses = jnp.stack([jnp.asarray(given[f"pose_{body}"], dtype=dtype) for body in self._bodies()]) \
            if self._bodies() else jnp.zeros((0, 7), dtype=dtype)
        ring = self.pushed(held["ring"], poses)
        exposure = self.exposed(held["ring"], ring)
        pose = self.camera_pose(cast(Mapping[str, Array], given), dtype)
        intrinsics = jnp.stack([jnp.asarray(p[name], dtype=dtype) for name in INTRINSICS])
        key = self.key_of(seed, count)

        cached = (held["cache_origin"], held["cache_rays"], held["cache_transmission"], held["cache_flags"])

        def make() -> tuple[Array, Array, dict[str, Array], tuple[Array, ...]]:
            return self._rendered(exposure, intrinsics, pose, key, dtype, count, cached)

        def keep() -> tuple[Array, Array, dict[str, Array], tuple[Array, ...]]:
            return (held["image"], held["expected"], {name: jnp.zeros((), dtype=dtype) for name in FLAG_FIELDS},
                    cached)

        frame = self.frame_due(count, period)
        image, expected, flags, carried = lax.cond(frame, make, keep)
        frame_step = jnp.where(frame, count, held["frame_step"])
        return node_fields(CameraState(
            steps=count + 1, frame_step=frame_step, age=count - frame_step, new_frame=frame.astype(jnp.int32),
            image=image, expected=expected, ring=ring, cache_origin=carried[0], cache_rays=carried[1],
            cache_transmission=carried[2], cache_flags=carried[3], **flags))


def period_steps(period: float, dt: float) -> int | None:
    """``period`` as a whole number of base steps of ``dt``, or ``None`` where it is not one."""
    ratio = period / dt
    whole = round(ratio)
    return whole if whole >= 1 and math.isclose(ratio, whole, rel_tol=1e-9) else None
