from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, cast, override

from mime.gradients import GradientStatus
from mime.registry import (
    DESIGN,
    PHYSICAL,
    BuildContext,
    Flag,
    GradientRecord,
    Observation,
    Output,
    Plan,
    PlanContext,
    Problem,
    Read,
    SettingGroup,
    convention,
    register_observation,
    register_setting_group,
)
from mime.sensing import BrownConrady, Pinhole
from mime.sensing import _scene as scene
from mime.sensing._node import FLAG_FIELDS, CameraNode, period_steps
from mime.sensing._switches import SWITCHES
from mime.units import DIMENSIONLESS, Length, Quantity, Time, m
from mime.vocabulary import POSE

if TYPE_CHECKING:
    from mime.build import ObservationBuilder

#: What each flag the node raises means, by its state field.
FLAG_MEANINGS = {
    "beyond_distortion": "pixels of the frame beyond the lens distortion's validity (no ray through them)",
    "saturated": "the share of the frame's pixels at full scale",
    "outside_parent": "pixels where a subject's chord lies outside its parent medium beyond the contact's tolerance",
    "small_silhouette": "subjects imaged by the box edge whose silhouette is under a few pixels in radius",
    "window_overflow": "subjects whose silhouette reached past the window their exact disc coverage is rendered in",
    "behind": "subjects whose enclosing ball reaches behind the camera's pinhole",
    "ghosting": "subjects that moved more than a pixel between the exposure's sub-frames",
    "thin_cap": "capped cylinders showing a cap nearly edge-on, where its outline is least resolved",
}


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class Detector(SettingGroup):
    """A pinhole detector of ``width`` by ``height`` pixels with focal lengths ``fx``, ``fy`` and principal point ``cx``,
    ``cy`` in pixels, and OpenCV's five distortion coefficients ``k1, k2, p1, p2, k3``. ``source`` says where the
    numbers come from: ``"declared"``, or a calibration file's name and sha256 (``fitted: ...``), which
    ``Pinhole.from_calibration`` writes. A declaration is not a calibration."""

    group = "detector"
    setting_roles = {
        "width": PHYSICAL, "height": PHYSICAL, "fx": PHYSICAL, "fy": PHYSICAL, "cx": PHYSICAL, "cy": PHYSICAL,
        **dict.fromkeys(("k1", "k2", "p1", "p2", "k3"), convention(PHYSICAL, "no lens distortion unless declared or fitted")),
        "source": convention(PHYSICAL, "the numbers are a declaration unless a calibration file is the source")}
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    source: str = "declared"

    @staticmethod
    def of(pinhole: Pinhole) -> Detector:
        d = pinhole.distortion
        return Detector(width=pinhole.width, height=pinhole.height, fx=float(pinhole.fx), fy=float(pinhole.fy),
                        cx=float(pinhole.cx), cy=float(pinhole.cy),
                        k1=float(d.k1), k2=float(d.k2), p1=float(d.p1), p2=float(d.p2), k3=float(d.k3), source=d.source)

    def fitted(self, values: Mapping[str, float], how: str) -> Detector:
        """This detector with the numbers a fit found (``values``, by setting name: ``fx``, ``k1``, ...), labelled fitted
        with how it was fitted: ``source`` is then ``"fitted: <how>"``, and a declared camera is never mistaken for it."""
        return dataclasses.replace(self, source=f"fitted: {how}", **{k: float(v) for k, v in values.items()})

    def pinhole(self) -> Pinhole:
        return Pinhole(self.width, self.height, self.fx, self.fy, self.cx, self.cy,
                       BrownConrady(self.k1, self.k2, self.p1, self.p2, self.k3, source=self.source))


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class Sensor(SettingGroup):
    """The sensor's response to the transmitted share: ``full_scale`` electrons at full backlight over the exposure, shot
    noise (Gaussian, of the signal's variance) and ``read_noise`` electrons, ``gain`` counts per electron with ``offset``,
    a Gaussian PSF of ``psf`` pixels and quantisation to ``bits``. Every number is declared, never a measurement of a
    camera."""

    group = "sensor"
    setting_roles = {name: convention(PHYSICAL, "a declared default of the sensor core, never a measured camera")
                     for name in ("full_scale", "gain", "offset", "read_noise", "shot_noise", "bits", "psf")}
    full_scale: float = 1000.0
    gain: float = 1.0
    offset: float = 0.0
    read_noise: float = 0.0
    shot_noise: bool = True
    bits: int = 12
    psf: float = 0.0


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class World(SettingGroup):
    """The camera fixed in the world: its optical frame (x right, y down, z forward, out of the lens) at ``position``,
    turned by the unit quaternion ``orientation`` ``(w, x, y, z)`` into the world's axes."""

    group = "camera_world"
    setting_roles = {"position": PHYSICAL, "orientation": PHYSICAL}
    position: Annotated[Quantity, Length]
    orientation: tuple[float, float, float, float]


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class Mounted(SettingGroup):
    """The camera fixed to a frame of an apparatus's chain, ``frame`` (``<apparatus>.<segment>.<frame>``, the AR4's
    wrist as ``rpm.arm.link_6``): its optical frame at ``position`` in that frame, turned by ``orientation``."""

    group = "camera_mounted"
    setting_roles = {"frame": PHYSICAL, "position": convention(PHYSICAL, "the camera sits at the frame's origin"),
                     "orientation": convention(PHYSICAL, "the camera's optical frame is the frame's axes")}
    frame: str
    position: Annotated[Quantity, Length] = m((0.0, 0.0, 0.0))
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@register_observation
@dataclass(frozen=True, kw_only=True)
class Camera(Observation):
    """A declared camera model in the loop: ``exp.observe(Camera(detector=..., mount=World(...), frame_period=ms(10)))``.
    It sees the whole scene: every body (its outermost parts, at the pose its dynamics publishes) and every structure
    the experiment declares, listed by nobody; ``hide`` leaves some out, and the build report names them. Backlit, one bin:
    an opaque subject is its shadow, a subject named in ``absorbers`` attenuates by Beer-Lambert (per metre), and every
    other subject takes the stated default (opaque, the build note ``appearance defaulted for:``).

    It renders a frame every ``frame_period`` (a whole number of base steps), exposed over the ``sub_frames`` base
    steps ending at the frame step, and holds it between: its outputs are ``image`` (counts), ``frame_step`` (the step
    the held image was made at), ``age`` (steps since) and ``new_frame``. A frame's noise is drawn from the run's seed and the
    frame's step, so rendering its recorded poses again reproduces it exactly. Frames are not recorded by default, and
    recording them in the loop is refused. Its derivative passes in forward and reverse mode; reverse mode whose memory
    estimate would not fit the memory a run can have (``camera.reverse-memory``), a batched study (``camera.batched``) and
    ``rate=`` are refused."""

    name = "camera"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(
        switches=SWITCHES,
        where="expected, the frame's counts before noise and quantisation, in forward and reverse mode: the poses and the "
              "detector's nine numbers; image, frame_step, age and new_frame carry none (a rounding and counts). Reverse "
              "mode whose memory estimate would not fit the memory a run can have is refused by name "
              "(camera.reverse-memory)",
        reason="the status is the one its declared switches give (mime.core.switch, mime.sensing._switches): the "
               "renderer's switches are declared as camera.* and each is measured at the function that evaluates it "
               "by the switch harness (tests/saved/test_switch_harness.py): a jump where two pieces of a body's outline "
               "meet inside a pixel, where the pinhole crosses a cap's plane and at the exact disc's series threshold, a "
               "kink or smoother at the others, the corner choice continuous at its threshold. The harness bounds each "
               "order at one placed point, from above; it does not show the derivative across a kink is the branch "
               "taken's. The closed form of a capsule is the exception: its sites (the interval's least entry and last "
               "exit and its selects of the pieces the ray meets, the edge function's clamped foot) are listed raw and "
               "neither routed nor declared nor measured, so the status does not cover the capsule's own kinks and "
               "jumps; routing them is a follow-up. The undistortion's iterations run on held values, and the ray's "
               "derivative is the implicit "
               "function's, which tests/sensing/test_undistortion.py checks in the five coefficients and the pixel, "
               "forward and reverse, in both dtypes, near the validity radius and away from it, against the implicit "
               "function's and (float64) central differences, with planted wrong derivatives failing every case; a "
               "pixel beyond the validity radius is flagged and its derivative is not checked. Forward mode is checked "
               "through the built graph against central differences of the same run "
               "(tests/sensing/test_in_loop_forward.py): a ball's settling and the principal point across a silhouette "
               "(the box edge, whose central differences are piecewise, so judged with an allowance per unit of step) and "
               "the focal length in the tube wall's interior (quadratic), from the run's start and from a continued state, "
               "in both dtypes, with a stop_gradient on the edge measure planted as the control. Reverse mode "
               "(tests/sensing/test_in_loop_reverse.py) is checked against forward mode, against central differences "
               "(float32 against float64 where they cannot discriminate) and a held step's against its frame's, from the "
               "run's start, from a continued state (the tree's structure, or a leaf written over it) and over a CALLBACK "
               "controller's chunks, in both dtypes, with three planted controls (the edge measure, the structure the "
               "run makes from its leaves, the held frame)",
        route="a smooth union and intersection of a body's pieces within a pixel (a softened least and greatest, in place "
              "of camera.pieces.*), a smooth facing term for a cap seen edge-on (camera.cap_side) and the exact disc's "
              "series one term longer (camera.disc.series) would leave kinks or smoother: the status piecewise_kink")
    unbatched: ClassVar[str] = ("the frame is rendered under a cond on frame steps, a select under vmap, so a batched "
                                "study would render every member's frame at every step, not once a frame period")
    references = ()
    builds = {"": (CameraNode,)}
    roles = {}
    sees_scene: ClassVar[bool] = True
    setting_roles = {
        **Observation.setting_roles,
        "detector": PHYSICAL, "mount": PHYSICAL, "frame_period": DESIGN,
        "sub_frames": convention(DESIGN, "one pose per frame: no motion blur unless the exposure is asked for"),
        "sensor": convention(PHYSICAL, "the sensor core's declared defaults"),
        "hide": convention(DESIGN, "nothing is hidden: the camera sees the whole scene"),
        "absorbers": convention(PHYSICAL, "no subject attenuates: every subject is opaque, defaulted and noted"),
    }
    raises = tuple(Flag(name, FLAG_MEANINGS[name]) for name in FLAG_FIELDS)
    outputs = (
        Output("image", DIMENSIONLESS, "the held frame, in counts, rows by columns", shape_from_settings=True,
               recordable=False),
        Output("expected", DIMENSIONLESS, "the held frame's counts before noise and quantisation, rows by columns: the "
                                          "differentiable output", shape_from_settings=True, recordable=False),
        Output("frame_step", DIMENSIONLESS, "the count of the step the held frame was made at (none before the first step)"),
        Output("age", DIMENSIONLESS, "the base steps since the held frame was made (zero on a frame step)"),
        Output("new_frame", DIMENSIONLESS, "one on a step a frame was made at, zero otherwise"),
    )
    detector: Detector | Pinhole
    mount: World | Mounted
    frame_period: Annotated[Quantity, Time]
    sub_frames: int = 1
    sensor: Sensor | None = None
    hide: tuple[str, ...] = ()
    absorbers: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.detector, Pinhole):
            object.__setattr__(self, "detector", Detector.of(self.detector))

    def _detector(self) -> Detector:
        return cast(Detector, self.detector)

    @override
    def output_shape(self, output: Output) -> tuple[int, ...]:
        detector = self._detector()
        return (detector.height, detector.width) if output.shape_from_settings else output.shape

    @override
    def setting_problems(self, settings: Mapping[str, Any]) -> list[str]:
        d = self._detector()
        found: list[str] = []
        if not (d.width >= 1 and d.height >= 1 and d.fx > 0 and d.fy > 0):
            found.append("a detector has at least one pixel each way and positive focal lengths")
        if self.sub_frames < 1:
            found.append(f"sub_frames is {self.sub_frames!r}, not a whole number from 1")
        if not (math.isfinite(float(settings["frame_period"])) and float(settings["frame_period"]) > 0):
            found.append("frame_period is a time above zero")
        norm = math.sqrt(sum(c * c for c in self.mount.orientation))
        if abs(norm - 1.0) > 1e-6:
            found.append(f"the mount's orientation {self.mount.orientation!r} is not a unit quaternion (w, x, y, z)")
        return found

    def _found(self, ctx: PlanContext | BuildContext) -> scene.Found:
        return scene.subjects_of(ctx.roster, self.hide, dict(self.absorbers))

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        problems: list[Problem] = []
        period = period_steps(ctx.scalar("frame_period"), ctx.dt)
        if self.rate is not None:
            problems.append(Problem(
                "camera.rate", "rate= is refused: a slower observation is computed at every base step and its result "
                               "discarded, so it would save nothing, and the camera has a clock of its own",
                "write frame_period=, a whole number of base steps"))
        if period is None:
            below = max(1, math.floor(ctx.scalar("frame_period") / ctx.dt))
            near = " and ".join(f"{k} step{'s' * (k > 1)} ({k * ctx.dt:.6g} s)" for k in (below, below + 1))
            problems.append(Problem(
                "camera.frame-period", f"frame_period is {ctx.scalar('frame_period'):.6g} s, not a whole multiple of the "
                                       f"base step dt = {ctx.dt:.6g} s",
                f"write frame_period= a whole number of base steps; the nearest are {near}"))
            period = 1
        if self.sub_frames > period:
            problems.append(Problem(
                "camera.frame-period", f"sub_frames is {self.sub_frames}, more than the {period} base steps of a frame "
                                       f"period: the exposure would reach back into the previous frame's",
                f"write sub_frames= at most {period}"))
        found = self._found(ctx)
        for name in found.unknown_hidden:
            problems.append(Problem("camera.hide", f"hide names {name!r}, which is no body or structure of the experiment",
                                    f"write `hide=(...)` with names the experiment declares, or a scenery prim's prim "
                                    f"path: {', '.join(sorted({*ctx.roster.bodies, *ctx.roster.structures, *(m.path for m in ctx.roster.scenery)}))}"))
        unknown = sorted({n for n, _ in self.absorbers} - {s.name for s in found.subjects})
        for name in unknown:
            problems.append(Problem("camera.hide", f"absorbers names {name!r}, which is no subject the camera images",
                                    f"write `absorbers=((name, per_metre),)` with one of "
                                    f"{', '.join(sorted(s.name for s in found.subjects))}"))
        for name, code, message in found.refused:
            if code == scene.MESH_UNTRACED:
                problems.append(Problem("camera.mesh-untraced", f"the scenery prim {name!r} cannot be imaged yet: {message}",
                                        f"hide it with hide=({name!r},), or wait for the mesh tracer"))
                continue
            problems.append(Problem(code, f"{name!r} cannot be imaged: {message.split(': ', 1)[-1]}",
                                    f"hide it with hide=({name!r},) until the sampled route lands, or give its kind a "
                                    f"closed form"))
        reads = {f"pose_{body}": Read("", POSE.name, of=body) for body, names in found.bodies.items()
                 if names and ctx.roster.bodies[body].pose is not None}
        mounted = self.mount.frame if isinstance(self.mount, Mounted) else None
        return Plan(reads=reads, streams=("noise",), problems=tuple(problems), mounted_on=mounted,
                    notes=self._notes(ctx, found, period, mounted))

    def _notes(self, ctx: PlanContext, found: scene.Found, period: int, mounted: str | None) -> tuple[str, ...]:
        d = self._detector()
        route = ", ".join(sorted(f"{s.name} ({s.kind})" for s in found.subjects)) or "none"
        sub = self.sub_frames
        dt = ctx.dt
        lines = [
            f"camera: a declared camera model, {d.width} x {d.height} pixels, {d.source}",
            f"route 1 (closed form) for every subject: {route}",
            f"frame: {d.width * d.height} rays, {sub} sub-frame{'s' * (sub > 1)}, rendered every {period} base steps "
            f"({period * dt:.6g} s); exposure {sub} step{'s' * (sub > 1)} ending at the frame step",
            f"latency: a read sees a frame between 0 and {period - 1} steps ({(period - 1) * dt:.6g} s) old; a controller "
            f"called between chunks reads the state after the previous chunk, up to a chunk later",
        ]
        if found.hidden:
            lines.append(f"hidden: {', '.join(found.hidden)}")
        staged = [s.name for s in found.subjects if s.name in found.display]
        defaulted = [s.name for s in found.subjects if s.name not in dict(self.absorbers) and s.name not in found.display]
        if staged:
            lines.append(f"appearance from the stage's display primvars for: {', '.join(staged)}")
        lines += [f"appearance: {line} for: {', '.join(names)}" for line, names in found.reduced.items()]
        if defaulted:
            lines.append(f"appearance defaulted for: {', '.join(defaulted)}")
        lines.append("bodies are taken to move and structures to stand; apparatus parts are not imaged")
        if mounted is None:
            lines.append("static cache: the camera and the static structure share the world's frame, so the structure is "
                         "rendered once, at the build, and held as a leaf beside the numbers it was made at; a run whose "
                         "detector numbers differ renders it once, before its steps")
        else:
            statics = [s.name for s in found.subjects if not s.moving]
            lines.append(f"static cache: the camera is mounted on {mounted}, taken to move, so the static structure "
                         f"({', '.join(statics) or 'none'}) is rendered again in every frame: {d.width * d.height} rays "
                         f"each, no cache is held (a stale one would be a wrong image)")
        return tuple(lines)

    @override
    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        d = self._detector()
        found = self._found(ctx)
        period = period_steps(ctx.scalar("frame_period"), ctx.dt)
        assert period is not None
        sensor = self.sensor or Sensor()
        bodies = [name for name, names in found.bodies.items() if names and ctx.roster.bodies[name].pose is not None]
        spec = {
            "width": d.width, "height": d.height, "source": d.source,
            "subjects": [scene.subject_spec(s, None, found.display.get(s.name)) for s in found.subjects],
            "body_subjects": {name: list(found.bodies[name]) for name in bodies},
            "response": {"full_scale": sensor.full_scale, "gain": sensor.gain, "offset": sensor.offset,
                         "read_noise": sensor.read_noise, "shot_noise": sensor.shot_noise, "bits": sensor.bits,
                         "psf": sensor.psf}}
        position = ctx.vector("mount.position")
        placement = (*position, *self.mount.orientation)
        mount = ctx.mount
        initial = [c for name in bodies for c in cast(tuple[float, ...], ctx.roster.bodies[name].pose)]
        node = b.add_node(CameraNode(
            ctx.node_name(), ctx.timestep, spec=scene.dumps(spec), fx=d.fx, fy=d.fy, cx=d.cx, cy=d.cy, k1=d.k1, k2=d.k2, p1=d.p1, p2=d.p2, k3=d.k3,
            bodies=bodies, initial_poses=initial, period=period,
            sub_frames=self.sub_frames, seed=ctx.seeds["noise"], placement=placement,
            mount_index=mount.index if mount is not None else -1, mount_count=mount.count if mount is not None else 0,
            initial_mount_pose=mount.initial_pose if mount is not None else (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)))
        for name in bodies:
            b.connect(b.read(f"pose_{name}"), into=node[f"pose_{name}"])
        if mount is not None:
            b.mount(node["mount_poses"])
