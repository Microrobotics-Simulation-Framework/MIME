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

**Forward only.** The frame is rendered by the offline renderer under a ``lax.cond``; a derivative that reaches it is
refused (``camera.reverse``), because the cond's branch is not the differentiable path: the offline ``render`` is, and a
batched run (where a ``cond`` is a ``select``) is refused by the study."""

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

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.core import NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, node_input, quaternion, \
    typed_view
from mime.core.multibody import compose
from mime.nodes.actuation._chain import POSE_UNITS, MountedNode
from mime.sensing import _scene as scene
from mime.sensing._detector import BrownConrady, CameraPose, Pinhole
from mime.sensing._render import Exposure, Response, StaticCache, _projected_reach, render, static_cache
from mime.sensing._subjects import Subject

lax: Any = jax.lax
_jax: Any = cast(Any, jax)

#: The flags a frame raises, each a state field of the node holding, on a frame step, the number of subjects (or pixels,
#: or the share of saturated ones) it raised for in that frame, and zero at every other step.
FLAG_FIELDS = ("beyond_distortion", "saturated", "outside_parent", "small_silhouette", "window_overflow", "behind",
               "ghosting", "thin_cap")

EQUATIONS = (r"\mathrm{frame}_{k} = \mathbb{1}\left[n \bmod P = 0\right], \quad "
             r"I = \mathrm{render}\left(\mathbf{x}_{n-K+1 \ldots n}, \mathbf{q}_{n-K+1 \ldots n}, \mathbf{c}, "
             r"\mathrm{key}(s, n)\right), \quad \mathrm{age} = n - n_{f}")

_NODE = "mime.sensing._node.CameraNode"


class CameraState(TypedDict, closed=True):
    steps: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    frame_step: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    age: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    new_frame: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    image: ReadOnly[Annotated[Float[Array, "rows columns"], Units("1")]]
    ring: ReadOnly[Annotated[Float[Array, "sub_frames bodies 7"], Units(POSE_UNITS)]]
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


@_jax.custom_jvp
def forward_only(held: Any) -> Any:
    """``held`` itself, which no derivative may pass through: the camera's frames are made in the forward run."""
    return held


@forward_only.defjvp
def _refuse_derivative(primals: Any, tangents: Any) -> Any:
    raise ArgumentRefused([Refusal(
        "camera.reverse", RefusalClass.USER,
        "a derivative reaches the in-loop camera: the frame is rendered under a cond on frame steps, and a "
        "derivative through it is not the offline renderer's",
        "differentiate the offline renderer, mime.sensing.render, in the poses and the intrinsics, with the poses "
        "recorded from the run; the in-loop camera is a forward observation",
        why="the in-loop camera is forward only: its cond, its ring and its held image carry no derivative")])


def frame_key(seed: int, step: Any) -> Any:
    """The noise key of the frame made at ``step`` (the node's count at that step) by the stream whose seed is ``seed``."""
    return jax.random.fold_in(jax.random.PRNGKey(seed), step)


def detector_of(spec: Mapping[str, Any]) -> Pinhole:
    k1, k2, p1, p2, k3 = spec["distortion"]
    return Pinhole(spec["width"], spec["height"], spec["fx"], spec["fy"], spec["cx"], spec["cy"],
                   BrownConrady(k1, k2, p1, p2, k3, source=spec["source"]))


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


class CameraNode(MountedNode):
    """The camera's frame on the base step's clock: the whole scene (its ``spec``, a JSON string of the detector, the
    subjects, the response and the key's seed), rendered every ``period`` steps from the ring of the last ``sub_frames``
    poses of every body in ``bodies`` (the inputs ``pose_<body>``), seen from the camera's pose: its ``placement`` in the
    world, or on the frame ``mount_index`` of the ``mount_poses`` it is mounted on. The static structure is rendered once,
    at the node's construction, where the camera is fixed in the world; where it is mounted on a frame that may move it is
    rendered in every frame, and no cache is ever held (a stale one would be a wrong image)."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-130",
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
        Symbol("K", SymbolKind.PARAMETER, "sub_frames", "1", f"{_NODE}.advance"),
        Symbol("s", SymbolKind.PARAMETER, "seed", "1", f"{_NODE}.advance"),
        Symbol("n_{f}", SymbolKind.STATE, "frame_step", "1", f"{_NODE}.advance"),
        Symbol(r"\mathrm{age}", SymbolKind.STATE, "age", "1", f"{_NODE}.advance"),
        Symbol(r"\mathrm{frame}_{k}", SymbolKind.STATE, "new_frame", "1", f"{_NODE}.advance"),
        Symbol("I", SymbolKind.STATE, "image", "1", f"{_NODE}.advance"),
        Symbol(r"\mathbf{c}", SymbolKind.PARAMETER, "placement", POSE_UNITS, f"{_NODE}.camera_pose"),
    )

    def __init__(self, name: str, timestep: float, *, spec: str, bodies: Sequence[str], initial_poses: Sequence[float],
                 period: int, sub_frames: int, seed: int, placement: Sequence[float], mount_index: int = -1,
                 mount_count: int = 0, initial_mount_pose: Sequence[float] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)) -> None:
        super().__init__(name, timestep, spec=spec, bodies=tuple(bodies),
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
        camera = detector_of(spec)
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
                reach = float(np.asarray(_projected_reach(camera, pose0, origin, subject.enclosing())))
                subject = scene.subject_of({**each, "window": scene.frame_window(reach, camera.width, camera.height)})
            made.append(subject)
        cache = static_cache(made, camera, pose0) if cast(int, held["mount_index"]) < 0 else None
        return camera, tuple(made), cache

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
                "placement": ParamSpec(trainable=False, units=POSE_UNITS,
                                       description="the camera's pose in the world, or on its mount frame: held with "
                                                   "the static structure's cache, so a move needs a new build"),
                "initial_poses": ParamSpec(trainable=False, units=POSE_UNITS,
                                           description="the bodies' poses at the start, which fill the ring")}

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
            image=as_run_array(np.zeros((rows, columns))),
            ring=jnp.broadcast_to(poses[None], (sub_frames, *poses.shape)), **cast(Any, flags)))

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

    def _rendered(self, ring: Array, pose: CameraPose, key: Any, dtype: Any) -> tuple[Array, dict[str, Array]]:
        camera, subjects, cache = self._held
        held = self.constants()
        spec = self._spec()
        sub_frames = cast(int, held["sub_frames"])
        moves: dict[str, Any] = {}
        for index, body in enumerate(self._bodies()):
            for name in spec["body_subjects"][body]:
                moves[name] = (ring[:, index, :3], ring[:, index, 3:]) if sub_frames > 1 \
                    else (ring[0, index, :3], ring[0, index, 3:])
        captured = render(subjects, camera, pose, moves, exposure=Exposure(sub_frames), response=response_of(spec),
                          key=key, cache=cache)
        totals = {name: jnp.zeros((), dtype=dtype) for name in FLAG_FIELDS}
        for flag, value in captured.flags.items():
            kind = flag.split(":")[0]
            if kind in totals:
                totals[kind] = totals[kind] + jnp.asarray(value, dtype=dtype)
        return captured.image.astype(dtype), totals

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
        ring = jnp.concatenate([held["ring"][1:], poses[None]])
        pose = self.camera_pose(cast(Mapping[str, Array], given), dtype)
        if cast(int, constants["mount_index"]) >= 0:
            ring, position, orientation = forward_only((ring, pose.position, pose.orientation))
            pose = CameraPose(position, orientation)
        else:
            ring = forward_only(ring)
        key = frame_key(seed, count)

        def make() -> tuple[Array, dict[str, Array]]:
            return self._rendered(ring, pose, key, dtype)

        def keep() -> tuple[Array, dict[str, Array]]:
            return held["image"], {name: jnp.zeros((), dtype=dtype) for name in FLAG_FIELDS}

        frame = (count % period) == 0
        image, flags = lax.cond(frame, make, keep)
        frame_step = jnp.where(frame, count, held["frame_step"])
        return node_fields(CameraState(
            steps=count + 1, frame_step=frame_step, age=count - frame_step, new_frame=frame.astype(jnp.int32),
            image=image, ring=ring, **cast(Any, flags)))


def period_steps(period: float, dt: float) -> int | None:
    """``period`` as a whole number of base steps of ``dt``, or ``None`` where it is not one."""
    ratio = period / dt
    whole = round(ratio)
    return whole if whole >= 1 and math.isclose(ratio, whole, rel_tol=1e-9) else None
