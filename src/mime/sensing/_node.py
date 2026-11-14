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
so a study can fit them; the ray table and the structure's image follow them. **The static structure** (a camera fixed in
the world: the ray table, the standing subjects' transmission and their flags, packed flat) is a leaf of its own,
``static_structure``, beside the numbers it was made at, ``static_structure_from`` (the nine intrinsics, the scene's numbers
as built, the camera's pose): both made when the node's parameters are made, by one compiled maker, so the build's tree, a
study's point's and an override's hold the structure at their own leaves, and the save's rerun is given it at its nudged
ones (``made_again``). A frame step reads it as its tree (or its run-constants) holds it, made at the run's leaves: read,
never carried, so the state holds no cache, a held step costs what holding the image costs, and a frame step what drawing the
frame over it costs, at the build's leaves or any other. Where a derivative may be taken (MIME's differentiated run, a
chunk a caller differentiates) the run makes it from its own leaves once, before its steps (``MimeNode.run_constants``, by
JAX's own rules, no custom one), so the derivative in the leaves passes through it from a continued state as from the
first step, and a tree whose leaves were written without it is drawn at its leaves. A forward run whose tree holds a
structure made at other numbers than its leaves (written leaf by leaf) makes it again once, on the host, before its steps
(``refreshed_held``, a compare of the numbers). A step given its run-constants (a differentiated run, a chunk that
tallies the run's flags) holds no branch that makes it, and one that still meets a structure made at other numbers draws
a frame that is not a number, never one at other leaves (no flag names the cause: MIME's runs make the structure again
first, so only a tree given past them could reach it). A step not given them (MADDENING's own scan, which MIME's
recorded runs use, ``graph.step``, MADDENING's verification, which injects a moved leaf and asks for the frame a node
built with it draws) takes whatever tree it is given, so it chooses per frame: the held structure where its numbers are
the step's, one made from them where any differs (never taken in a run of MIME's, which made it again first). A
derivative taken through such a step carries no part of the structure where its numbers hold: MIME's derivatives are
taken through its differentiated run, which makes it from the leaves. Nothing of the structure is a constant of the
program.
``expected`` (the counts before noise and quantisation) is the differentiable output; ``image`` is the quantised frame.

**Both modes; a batched study refused.** A derivative passes in forward and in reverse mode (the ring's poses and the
leaves). Reverse mode keeps a cond's residuals as large as its largest branch at every step, whichever branch runs, so
two checkpoints hold its tape down: each corner sub-sample's (the static structure's make and every frame keep one
sub-sample's residuals at a time, not sixteen) and each frame's render (a step keeps the render's inputs, and the
backward pass draws the frame again), which leave the tape a few hundred frames, flat in the run's steps. A program that
takes no reverse derivative is unchanged (a checkpoint lowers to its body). A study's reverse derivative whose estimate
would not fit the memory a run can have is refused before it compiles (:meth:`CameraNode.derivative_problem`,
``camera.reverse-memory``), naming the offline renderer. A batched study is refused by the study
(``Observation.unbatched``): under ``vmap`` the frame's cond is a ``select``, so every step would render.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast, override

import jax
import numpy as np
from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import NodeFields, Symbol, SymbolKind, Units, as_run_array, jit, jnp, node_fields, node_input, \
    run_constants_given, run_dtype, typed_view
from mime.core._node import for_derivative
from mime.core.multibody import compose
from mime.nodes.actuation._chain import POSE_UNITS, MountedNode
from mime.registry import Problem
from mime.sensing import _scene as scene
from mime.sensing._coverage import CORNER_SAMPLES
from mime.sensing._detector import BrownConrady, CameraPose, Pinhole, Rays, ray_table
from mime.sensing._render import Exposure, Response, exposed, frame_key, prepared, projected_reach, static_part
from mime.sensing._subjects import Subject

lax: Any = jax.lax
_checkpoint: Any = cast(Any, jax).checkpoint

#: The detector's nine numbers, the node's leaves, in the order a Pinhole and OpenCV give them.
INTRINSICS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
#: How far a focal length may move from the build's, as a share: the box a fit stays in, and what the disc windows are
#: sized for. Past it the silhouette could leave its window, which a flag would report, so the leaf's bounds keep it in.
FOCAL_SHARE = 0.25
#: How far a distortion coefficient may move from the build's: the radial ones by this, the tangential ones by a fifth.
DISTORTION_BOX = 0.3
#: The flags a frame raises, each a state field of the node holding, on a frame step, the number of subjects (or pixels,
#: or the share of saturated ones) it raised for in that frame, and zero at every other step.
FLAG_FIELDS = ("beyond_distortion", "saturated", "outside_parent", "small_silhouette", "window_overflow", "behind",
               "ghosting", "thin_cap")
#: The static structure of a camera fixed in the world, a leaf (:meth:`CameraNode.run_constants`): the ray table's origin,
#: its ten planes (the direction, its derivatives along u and v, whether a pixel lies beyond the distortion), the standing
#: subjects' transmission and the static flags in their names' order, packed flat (:func:`packed`).
STRUCTURE = "static_structure"
#: The numbers :data:`STRUCTURE` was made at, a leaf beside it: the nine intrinsics, the scene's numbers as built and the
#: camera's pose, the ones the step compares its own with.
STRUCTURE_FROM = "static_structure_from"
#: The image-sized planes of the packed structure: the ray table's ten and the transmission.
PLANES = 11
#: Reverse mode's peak through the camera past what the process holds once the experiment is built, for MIME's
#: differentiated run of ``n`` steps in checkpointed chunks of ``L = ceil(sqrt(n))`` steps: the larger of its two phases,
#: and :data:`REVERSE_POSE` more for each pose a frame renders past one (``m``, the moving subjects times the exposure's
#: sub-frames: each is traced into the program, which grows its compile and its code, not its tape). The derivative's
#: compile, :data:`REVERSE_COMPILE` bytes. Its run, :data:`REVERSE_HELD` bytes (the compiled program and what the runtime
#: keeps) and the tape, in frames of the run's dtype (rows x columns values), ``fixed + start (n // L)``: ``fixed`` the
#: static structure's make (its residuals, kept for the derivative in the leaves) and one step's backward pass (each
#: corner sub-sample and each frame's render are checkpointed, so a step keeps their inputs, not their residuals),
#: ``start`` a chunk start's held image and counts. :data:`REVERSE_MEASURED` says how they were measured.
REVERSE_COMPILE = 1.45 * 2 ** 30
REVERSE_POSE = 0.16 * 2 ** 30
REVERSE_HELD = 1.06 * 2 ** 30
REVERSE_FRAMES = {"fixed": 464.0, "start": 2.0}
#: The refusal's margin on the estimate, the largest measured peak over its estimate rounded up, and never under 1.1 (the
#: memory route's floor); and its headroom, the share of the memory a run can have that it keeps free (the near field's
#: rule, ``MEMORY_HEADROOM``).
REVERSE_MARGIN = 1.1
REVERSE_HEADROOM = 0.1
REVERSE_MEASURED = ("fitted to the reverse phases' peak resident memory, less the resident memory after the build, that "
                    "node_measure.py measured through the built graph on GitHub's runners (4 vCPU, AMD EPYC and Intel "
                    "Xeon), at 160x120 to 640x480, 10 to 1000 steps, float64 and float32, one to six poses a frame, 22 "
                    "runs: 0.85 to 1.00 of the estimate. A first fit (the poses' term at 0.08 GiB, from 320x240 alone) "
                    "held the eight runs it left out within its margin, the largest at 1.098 of the estimate, and the "
                    "term was refitted on all of them. fixed is XLA's compiled memory analysis of the reverse program, "
                    "453 to 467 frames at every size, dtype, step count and pose count measured; start is derived")

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


def packed(rays: Rays, static: Any, flags: Mapping[str, Any], names: Sequence[str], dtype: Any) -> Array:
    """The static structure as one flat array (:data:`STRUCTURE`): the origin, the :data:`PLANES` planes, the flags of
    ``names`` in that order."""
    origin = jnp.stack([jnp.asarray(c, dtype=dtype) for c in rays.origin])
    planes = jnp.stack([jnp.asarray(c, dtype=dtype) for part in (rays.direction, rays.along_u, rays.along_v) for c in part]
                       + [jnp.asarray(rays.beyond, dtype=dtype), jnp.asarray(static, dtype=dtype)])
    values = jnp.stack([jnp.asarray(flags[name], dtype=dtype) for name in names]) if names \
        else jnp.zeros((0,), dtype=dtype)
    return jnp.concatenate([origin, jnp.reshape(planes, (-1,)), values])


def unpacked(flat: Array, rows: int, columns: int, names: Sequence[str]) -> tuple[Rays, Array, dict[str, Array]]:
    """The rays, the transmission and the flags :func:`packed` holds."""
    size = PLANES * rows * columns
    planes = jnp.reshape(flat[3:3 + size], (PLANES, rows, columns))
    values = flat[3 + size:]
    rays = Rays(origin=(flat[0], flat[1], flat[2]), direction=(planes[0], planes[1], planes[2]),
                along_u=(planes[3], planes[4], planes[5]), along_v=(planes[6], planes[7], planes[8]),
                beyond=planes[9] > 0.5)
    return rays, planes[10], {name: values[i] for i, name in enumerate(names)}


def static_subjects(spec: Mapping[str, Any]) -> tuple[Subject, ...]:
    """The subjects of ``spec`` as the static part reads them: prepared (each tube's implicit lumen, the drawing order),
    a moving disc's window left unsized, which the static part never reads (it images the standing subjects alone)."""
    return tuple(prepared([scene.subject_of(each) for each in spec["subjects"]], None)[0])


@functools.lru_cache(maxsize=16)
def structure_maker(spec: str, placement: tuple[float, ...]) -> tuple[Callable[[Array], Array], list[str]]:
    """The static structure of a camera fixed in the world at ``placement``, a function of its nine intrinsics compiled
    once per scene and pose (a study's points, an override's build and the save's rerun share it), and the static flags'
    names in their packed order, known once it has been traced."""
    held = json.loads(spec)
    subjects = static_subjects(held)
    pose = CameraPose(np.asarray(placement[:3], dtype=np.float64), np.asarray(placement[3:], dtype=np.float64))
    names: list[str] = []

    def made(intrinsics: Array) -> Array:
        dtype = intrinsics.dtype
        moved = detector_of(held, intrinsics)
        rays = ray_table(moved, pose)
        flags: dict[str, Any] = {}
        static = static_part(subjects, rays, moved, pose, dtype, CORNER_SAMPLES, flags)
        names[:] = sorted(flags)
        return packed(rays, static, flags, names, dtype)

    return cast(Callable[[Array], Array], jit(made)), names


SCENE_HELD = ("held as built: the scene's numbers (the subjects' settings and appearances, the response) are consumed when "
              "the node is built, into its subjects and the static structure's image, so that a new value takes a "
              "rebuild (a study refuses to vary it)")
STRUCTURE_HELD = ("the static structure, made when the node's parameters are made, from the leaves beside it: no quantity "
                  "of its own, made again once a run from the run's leaves where a derivative may be taken "
                  "(run_constants)")


class CameraNode(MountedNode):
    """The camera's frame on the base step's clock: the whole scene (its ``spec``, a JSON string of the detector, the
    subjects, the response and the key's seed), rendered every ``period`` steps from the ring of the last ``sub_frames``
    poses of every body in ``bodies`` (the inputs ``pose_<body>``), seen from the camera's pose: its ``placement`` in the
    world, or on the frame ``mount_index`` of the ``mount_poses`` it is mounted on. Where the camera is fixed in the world
    the static structure is a leaf (:data:`STRUCTURE`), made with the node's parameters, and once a run from the run's
    leaves where a derivative may be taken, and read; where it is mounted on a frame that may move it is rendered in every
    frame, and none is held (a stale one would be a wrong image)."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-133",
        description="A declared camera model in the loop: the sensor core's frame of the whole scene, rendered on "
                    "frame steps of the base step's clock from a ring of the bodies' poses, held between them.",
        governing_equations=EQUATIONS,
        assumptions=("The frame period is a whole number of base steps and a frame is rendered at every step whose count "
                     "is a multiple of it, exposed over the sub-frames' base steps ending at that step",
                     "The scene is the experiment's bodies (their outermost parts) and structures, imaged by the "
                     "closed form of their kinds; a body is taken to move, a structure to stand"),
        limitations=("A batched study through it is refused (under vmap every step would render), and reverse mode where "
                     "its tape would not fit the memory a run can have",
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
        self._structure = self._built_structure() if self.holds_cache() else None

    # -- what the node holds, made once from its constants ----------------------------------------------------------

    def _made(self) -> tuple[Pinhole, tuple[Subject, ...]]:
        """The detector and the subjects, each disc-covered body's window sized from its initial pose."""
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
        return camera, tuple(prepared(made, None)[0])

    def _fixed_pose(self) -> tuple[float, ...]:
        """The camera's pose in the world as built: its placement, on its mount frame's initial pose where it has one."""
        held = self.constants()
        placement = np.asarray(held["placement"], dtype=np.float64)
        if cast(int, held["mount_index"]) >= 0:
            placement = _composed(np.asarray(held["initial_mount_pose"], dtype=np.float64), placement)
        return tuple(float(c) for c in placement)

    def _maker(self) -> tuple[Callable[[Array], Array], list[str]]:
        return structure_maker(cast(str, self.constants()["spec"]), self._fixed_pose())

    def _built_structure(self) -> tuple[Array, Array]:
        """The static structure at the build's intrinsics in the run's dtype, and the numbers it was made at."""
        intrinsics = as_run_array(np.asarray(self.build_intrinsics()))
        return self._maker()[0](intrinsics), self._structure_numbers(intrinsics)

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
        held = {key: ParamSpec(trainable=False, units="1", description=STRUCTURE_HELD)
                for key in (STRUCTURE, STRUCTURE_FROM)} if self.holds_cache() else {}
        return {**super().param_specs(),
                **self.intrinsic_specs(),
                "scene_numbers": ParamSpec(trainable=False, units="1", description=SCENE_HELD),
                "period": ParamSpec(trainable=False, units="1", description="base steps between frames"),
                "seed": ParamSpec(trainable=False, units="1", description="the noise stream's seed, folded with the "
                                                                          "frame's step"),
                "placement": ParamSpec(trainable=False, units=POSE_UNITS,
                                       description="the camera's pose in the world, or on its mount frame: held with "
                                                   "the static structure, so a move needs a new build"),
                "initial_poses": ParamSpec(trainable=False, units=POSE_UNITS,
                                           description="the bodies' poses at the start, which fill the ring"),
                **held}

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

    @override
    def params_pytree(self) -> dict[str, Any]:
        inherited: Any = super()
        leaves = cast(dict[str, Any], inherited.params_pytree())
        if self._structure is None:
            return leaves
        structure, numbers = self._structure
        if structure.dtype != run_dtype():  # the run's precision set after the node was made
            structure, numbers = self._built_structure()
        return {**leaves, STRUCTURE: structure, STRUCTURE_FROM: numbers}

    # -- the static structure: a leaf, made once a run ---------------------------------------------------------------

    def _structure_numbers(self, intrinsics: Array) -> Array:
        """Every number the static structure is made from, as the step has them: the nine ``intrinsics``, the scene's
        numbers as built (constants of the program, so that a structure made by a build with other numbers differs) and
        the camera's pose."""
        dtype = intrinsics.dtype
        placement = np.asarray(self._fixed_pose(), dtype=np.float64)
        scene_numbers = np.asarray(self.constants()["scene_numbers"], dtype=np.float64)
        return jnp.concatenate([intrinsics, jnp.asarray(scene_numbers, dtype=dtype).reshape(-1),
                                jnp.stack([jnp.asarray(c, dtype=dtype) for c in placement])])

    def _make(self, numbers: Array) -> Array:
        """The static structure made at ``numbers`` (their first nine, the intrinsics)."""
        return self._maker()[0](numbers[:len(INTRINSICS)])

    def _intrinsics(self, p: Mapping[str, Any], dtype: Any) -> Array:
        return jnp.stack([jnp.asarray(p[name], dtype=dtype) for name in INTRINSICS])

    def _made_at(self, params: Mapping[str, Any]) -> dict[str, Array]:
        """The static structure made at ``params``' intrinsics, in its leaf's dtype, and the numbers it was made at."""
        dtype = jnp.asarray(params[STRUCTURE]).dtype
        intrinsics = self._intrinsics(params, dtype)
        return {STRUCTURE: self._maker()[0](intrinsics).astype(dtype), STRUCTURE_FROM: self._structure_numbers(intrinsics)}

    @override
    def run_constants(self, params: Mapping[str, Any]) -> dict[str, Array]:
        """For a run a derivative may be taken through (``for_derivative``): the static structure made from the run's
        leaves, once for the run (a differentiated run makes its run-constants before its chunks), with the numbers it
        was made at, so that the derivative in the leaves passes through it by JAX's own rules wherever the run starts
        and whatever the tree held. Otherwise none: a forward chunk's steps read the structure its tree holds, made at
        its leaves, with no make and no copy."""
        if STRUCTURE not in params or not for_derivative():
            return {}
        return self._made_at(params)

    @override
    def refreshed_held(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The static structure made again at ``params``' intrinsics where the numbers it was made at are not theirs
        (a tree whose leaves were written without it), compared on the host in the leaf's dtype; none where they are."""
        if STRUCTURE not in params:
            return {}
        made_from = np.asarray(params[STRUCTURE_FROM])
        intrinsics = np.asarray([np.asarray(params[name]) for name in INTRINSICS], dtype=made_from.dtype)
        numbers = np.concatenate([intrinsics, np.asarray(self.constants()["scene_numbers"], dtype=np.float64)
                                  .reshape(-1).astype(made_from.dtype),
                                  np.asarray(self._fixed_pose(), dtype=np.float64).astype(made_from.dtype)])
        return {} if np.array_equal(numbers, made_from) else dict(self._made_at(params))

    def reverse_estimate(self, steps: int, members: int = 1) -> float:
        """The bytes reverse mode through the camera holds at its peak over a run of ``steps`` steps of ``members``
        members, past what the process holds once the experiment is built, in the run's dtype: the larger of the
        derivative's compile and its run (:data:`REVERSE_COMPILE`, :data:`REVERSE_HELD`, :data:`REVERSE_FRAMES`), and
        :data:`REVERSE_POSE` for each pose a frame renders past one."""
        rows, columns = self.image_shape()
        length = max(1, math.ceil(math.sqrt(steps)))
        poses = max(1, sum(1 for each in self._held[1] if each.moving) * cast(int, self.constants()["sub_frames"]))
        frame = rows * columns * int(jnp.zeros((), dtype=run_dtype()).dtype.itemsize)
        tape = frame * (REVERSE_FRAMES["fixed"] + REVERSE_FRAMES["start"] * (steps // length))
        return REVERSE_POSE * (poses - 1) + max(REVERSE_COMPILE, REVERSE_HELD + members * tape)

    def derivative_problem(self, mode: str, directions: int, members: int, steps: int) -> Problem | None:
        """A study's derivative in reverse mode through the camera over ``steps`` steps whose tape would not fit the
        memory a run can have, refused before the study compiles it (``camera.reverse-memory``): the estimate
        (:meth:`reverse_estimate`) times :data:`REVERSE_MARGIN` against that memory less :data:`REVERSE_HEADROOM` of it.
        Forward mode keeps no tape and is not refused."""
        del directions
        if mode != "reverse":
            return None
        needed = REVERSE_MARGIN * self.reverse_estimate(steps, members)
        where, available = device_memory()
        if available is None or needed <= (1.0 - REVERSE_HEADROOM) * available:
            return None
        rows, columns = self.image_shape()
        return Problem(
            "camera.reverse-memory",
            f"reverse mode through the in-loop camera's {columns} x {rows} frames over {steps} steps needs about "
            f"{needed / 2 ** 30:.3g} GiB (its compile or its run with its tape, as measured runs held, times "
            f"{REVERSE_MARGIN:g}), more than {where}, "
            f"{available / 2 ** 30:.3g} GiB, less the {REVERSE_HEADROOM:.0%} of it the check keeps free",
            "differentiate the offline renderer, mime.sensing.render (differentiable, and it vmaps), in the poses "
            "recorded from the run; or take forward mode along a few axes, study.gradient(..., mode='forward'), which "
            "keeps no tape; or fewer steps, a smaller detector, or a machine with more memory")

    @override
    def made_again(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The static structure made from ``params``' intrinsics, with the numbers it was made at: what the save's
        resolution rerun gives the step (its nudged leaves), which then reads it."""
        if STRUCTURE not in params:
            return {}
        return dict(self._made_at(params))

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

    def _rendered(self, p: Mapping[str, Any], ring: Array, intrinsics: Array, pose: CameraPose, key: Any,
                  dtype: Any) -> tuple[Array, Array, dict[str, Array]]:
        """The frame over the static structure, for a camera fixed in the world. A step given its run-constants (a
        differentiated run, a chunk a caller differentiates, a chunk that tallies the run's flags) reads the one the
        run holds (:meth:`structure_of`, made at the run's leaves), never makes it, and draws a frame that is not a
        number where its numbers are not the step's. A step not given them (MADDENING's own scan, which MIME's recorded
        runs use, ``graph.step``, MADDENING's verification) takes whatever tree it is given, so it chooses per frame:
        the one the tree holds where its numbers are the step's, one made from the step's numbers where any differs
        (the frame rendered in each branch, so that no branch gives the structure out; a run of MIME's never takes the
        second, having made the tree's again first). Made where the step was given none (the node's constants alone).
        A camera on a frame that may move renders it."""
        _, subjects = self._held
        held = self.constants()
        spec = self._spec()
        sub_frames = cast(int, held["sub_frames"])
        moves: dict[str, Any] = {}
        for index, body in enumerate(self._bodies()):
            for name in spec["body_subjects"][body]:
                moves[name] = (ring[:, index, :3], ring[:, index, 3:]) if sub_frames > 1 \
                    else (ring[0, index, :3], ring[0, index, 3:])
        camera = detector_of(spec, intrinsics)

        def framed(rays: Rays, static: Array, flags: dict[str, Any]) -> tuple[Array, Array, dict[str, Array]]:
            captured = exposed(subjects, camera, pose, rays, static, dict(flags), moves, Exposure(sub_frames),
                               response_of(spec), key)
            totals = {name: jnp.zeros((), dtype=dtype) for name in FLAG_FIELDS}
            for flag, value in captured.flags.items():
                kind = flag.split(":")[0]
                if kind in totals:
                    totals[kind] = totals[kind] + jnp.asarray(value, dtype=dtype)
            return captured.image.astype(dtype), captured.expected.astype(dtype), totals

        if self._structure is None:
            rays = ray_table(camera, pose)
            flags: dict[str, Any] = {}
            return framed(rays, static_part(subjects, rays, camera, pose, dtype, CORNER_SAMPLES, flags), flags)
        rows, columns = self.image_shape()
        names = self._maker()[1]
        numbers = self._structure_numbers(intrinsics)
        if STRUCTURE not in p:
            return framed(*unpacked(self._make(numbers), rows, columns, names))
        held = self.holds_at(p, numbers)
        if run_constants_given(p):
            image, expected, totals = framed(*unpacked(self.structure_of(p, numbers), rows, columns, names))
            missing = jnp.asarray(jnp.nan, dtype=dtype)
            return jnp.where(held, image, missing), jnp.where(held, expected, missing), totals

        def read() -> tuple[Array, Array, dict[str, Array]]:
            return framed(*unpacked(self.structure_of(p, numbers), rows, columns, names))

        def made() -> tuple[Array, Array, dict[str, Array]]:
            return framed(*unpacked(self._make(numbers), rows, columns, names))

        return lax.cond(held, read, made)

    def structure_of(self, p: Mapping[str, Any], numbers: Array) -> Array:
        """The static structure a frame step draws over: the one its tree holds, or its run's run-constant (made from
        the run's leaves for a derivative). ``numbers`` are the step's, which a frame drawn over it is judged against
        (:meth:`holds_at`), never made from here."""
        del numbers
        return p[STRUCTURE]

    def holds_at(self, p: Mapping[str, Any], numbers: Array) -> Array:
        """Whether the structure ``p`` holds was made at ``numbers``: every one of them equal to its own."""
        return jnp.all(numbers == p[STRUCTURE_FROM])

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
        intrinsics = self._intrinsics(p, dtype)
        key = self.key_of(seed, count)

        def make() -> tuple[Array, Array, dict[str, Array]]:
            return self._rendered(p, exposure, intrinsics, pose, key, dtype)

        def keep() -> tuple[Array, Array, dict[str, Array]]:
            return held["image"], held["expected"], {name: jnp.zeros((), dtype=dtype) for name in FLAG_FIELDS}

        # The frame's render is checkpointed: reverse mode keeps a cond's residuals as large as its largest branch at
        # every step whichever branch runs, so a step's tape would hold a render's (tens of frames) at every step of the
        # chunk it differentiates; checkpointed, it holds the render's inputs and draws the frame again in the backward
        # pass. A program that takes no reverse derivative is unchanged: a checkpoint lowers to its body, and
        # prevent_cse=False (inside the run's scan, which keeps the recomputation apart) adds no barrier.
        frame = self.frame_due(count, period)
        image, expected, flags = lax.cond(frame, _checkpoint(make, prevent_cse=False), keep)
        frame_step = jnp.where(frame, count, held["frame_step"])
        return node_fields(CameraState(
            steps=count + 1, frame_step=frame_step, age=count - frame_step, new_frame=frame.astype(jnp.int32),
            image=image, expected=expected, ring=ring, **flags))


def device_memory() -> tuple[str, int | None]:
    """The memory a run's arrays can have and its size (``mime.build._maddening.device_memory``)."""
    from mime.build._maddening import device_memory as held

    return held()


def period_steps(period: float, dt: float) -> int | None:
    """``period`` as a whole number of base steps of ``dt``, or ``None`` where it is not one."""
    ratio = period / dt
    whole = round(ratio)
    return whole if whole >= 1 and math.isclose(ratio, whole, rel_tol=1e-9) else None
