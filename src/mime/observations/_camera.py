from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, cast, override

import numpy as np

from mime.core import jnp
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
from mime.scene._kinds import rotate
from mime.sensing import Appearance, BrownConrady, Pinhole
from mime.sensing import _sampled as sampled_route
from mime.sensing import _scene as scene
from mime.sensing._mesh import MESH_SAMPLES
from mime.sensing._detector import CameraPose
from mime.sensing._detector import project as projected
from mime.sensing._node import FLAG_FIELDS, FOCAL_SHARE, SAMPLED_FLAG_FIELDS, CameraNode, period_steps
from mime.sensing._render import projected_reach
from mime.sensing._switches import SWITCHES
from mime.units import DIMENSIONLESS, LENGTH, Length, Quantity, Time, m, si
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
    "corner_unverified": "pixels where two branches of a sampled body's outline cross (the sampled route's corners) "
                         "whose reading could not be verified (a least reaching a sub-pixel left unread), each within "
                         "its line and corner_bound; their share of the frame is this over its pixels",
    "corner_bound": "the most an unverified corner pixel's share may lie beyond its line (a share of a pixel, the "
                    "largest in the frame, summed over subjects); zero where every corner is verified",
}
#: The flags warned of after a run that raised them: a pixel whose stated accuracy is not verified.
FLAG_WARNS = frozenset({"corner_unverified"})


def _nearest_depth(triangles: Any, position: Any, orientation: tuple[float, float, float, float]) -> float | None:
    """The least depth along the camera's optical axis of a mesh's corners in front of it, or ``None`` for none."""
    w, x, y, z = orientation
    axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    depth = (np.reshape(np.asarray(triangles, dtype=np.float64), (-1, 3)) - position) @ axis
    ahead = depth[depth > 0.0]
    return float(ahead.min()) if ahead.size else None


def _mesh_problems(found: scene.Found, mounted: str | None) -> list[Problem]:
    """The traced meshes' refusals: one the stage animates, one that attenuates and is not a closed solid, and any on a
    camera that may move; each naming its route."""
    problems: list[Problem] = []
    for each in found.meshes:
        if each.animated:
            problems.append(Problem(
                "camera.mesh-moving", f"the scenery mesh {each.name!r} moves (the stage samples its points or its "
                                      f"placement in time), and the in-loop camera traces static meshes once a run",
                f"hide it with hide=({each.name!r},); a moving mesh is not imaged in the loop"))
        if each.open is not None:
            problems.append(Problem(
                "camera.mesh-open", f"{each.name!r} is named in absorbers, so it attenuates along its chord, which "
                                    f"needs a closed solid, and {each.open}",
                f"leave it out of absorbers, absorbers=((name, per_metre),) without {each.name!r} (it is then imaged by "
                f"its display opacity, or opaque), or close it: every edge shared by two triangles wound opposite ways, "
                f"counter-clockwise seen from outside"))
    if mounted is not None and found.meshes:
        names = [each.name for each in found.meshes]
        problems.append(Problem(
            "camera.mesh-mounted", f"the camera is mounted on {mounted}, which may move, and it would trace static "
                                   f"meshes ({', '.join(names)}): a moving camera keeps no static structure, and a mesh "
                                   f"traced every frame would cost seconds a frame",
            f"fix the camera in the world, mount=World(...), or hide the meshes, hide=({', '.join(repr(n) for n in names)},)"))
    return problems


#: The camera's own model of its noise and quantisation, which its image carries and its expected image is before.
SENSOR_RESPONSE = "the camera's sensor response (Sensor: shot and read noise, gain and quantisation to its bits)"


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
    raises = tuple(Flag(name, FLAG_MEANINGS[name], warns=name in FLAG_WARNS)
                   for name in (*FLAG_FIELDS, *SAMPLED_FLAG_FIELDS))
    outputs = (
        Output("image", DIMENSIONLESS, "the held frame, in counts, rows by columns", shape_from_settings=True,
               recordable=False, noise_model=SENSOR_RESPONSE),
        Output("expected", DIMENSIONLESS, "the held frame's counts before noise and quantisation, rows by columns: the "
                                          "differentiable output", shape_from_settings=True, recordable=False,
               before=SENSOR_RESPONSE),
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

    def project(self, points: Any, *, pose: tuple[Any, Any] | None = None) -> tuple[Any, Any]:
        """Where world points ``(..., 3)`` (metres) fall on this camera's detector, ``(..., 2)`` pixels ``(u, v)``,
        and their depths along its optical axis (positive in front): a pure function, in ``jax.numpy``, of its
        detector's nine numbers and its pose, OpenCV's ``projectPoints`` with its model. ``pose`` is its optical
        frame's position in the world and its unit quaternion ``(w, x, y, z)``; a camera fixed in the world
        (``World``) is at its own where none is given, and one ``Mounted`` on a frame, which moves, is given it.
        What a controller's model of its camera computes; the camera's own step renders, and never calls it."""
        if pose is None:
            if not isinstance(self.mount, World):
                raise TypeError("a camera mounted on a frame moves with it: give project(..., pose=(position, "
                                "orientation)), its optical frame in the world")
            pose = (si(self.mount.position, LENGTH, what="the camera's position"), self.mount.orientation)
        return projected(self._detector().pinhole(), CameraPose(pose[0], pose[1]), points)

    @override
    def output_shape(self, output: Output) -> tuple[int, ...]:
        detector = self._detector()
        return (detector.height, detector.width) if output.shape_from_settings else output.shape

    @override
    def clock(self, settings: Mapping[str, Any], dt: float) -> int | None:
        return period_steps(float(settings["frame_period"]), dt)

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
        imaged = {s.name for s in found.subjects} | {each.name for each in found.meshes}
        unknown = sorted({n for n, _ in self.absorbers} - imaged)
        for name in unknown:
            problems.append(Problem("camera.hide", f"absorbers names {name!r}, which is no subject the camera images",
                                    f"write `absorbers=((name, per_metre),)` with one of {', '.join(sorted(imaged))}"))
        for name, code, message in found.refused:
            if code == scene.MESH_UNTRACED:
                problems.append(Problem("camera.mesh-untraced", f"the scenery prim {name!r} cannot be imaged: {message}",
                                        f"hide it with hide=({name!r},), or give the stage a Mesh prim in its place"))
                continue
            problems.append(Problem(code, f"{name!r} cannot be imaged: {message.split(': ', 1)[-1]}",
                                    f"hide it with hide=({name!r},), or give its kind a closed form, or a cheap sign "
                                    f"function (and, if creased, its crease measure)"))
        mounted = self.mount.frame if isinstance(self.mount, Mounted) else None
        problems += _mesh_problems(found, mounted)
        reads = {f"pose_{body}": Read("", POSE.name, of=body) for body, names in found.bodies.items()
                 if names and ctx.roster.bodies[body].pose is not None}
        # The sampled route's flags are the node's state only where it images a subject by that route.
        sampled = any(each.placed.sampled is not None for each in found.subjects)
        return Plan(reads=reads, streams=("noise",), problems=tuple(problems), mounted_on=mounted,
                    notes=self._notes(ctx, found, period, mounted),
                    raises=FLAG_FIELDS + (SAMPLED_FLAG_FIELDS if sampled else ()))

    def _sampled_notes(self, ctx: PlanContext, sampled: list[Any], period: int) -> list[str]:
        """Route 2's line for each sampled body, and its structural cost estimate (its window's rays at the start
        pose times the sub-frames times its evaluations a ray and their count), warned above ``COST_LINE``."""
        lines: list[str] = []
        for subject in sampled:
            marched = subject.placed.sampled
            rays = self._window_rays(ctx, subject)
            cost = sampled_route.frame_cost(marched, subject.placed.parameters, rays, self.sub_frames,
                                             jnp.result_type(float))
            lines.append(f"route 2 (sampled) for {subject.name} ({subject.kind}): {marched.samples} samples a ray, the "
                         f"two least minima as pieces, corner sub-samples where both cross a pixel; structural cost "
                         f"estimate {cost / 1e6:.4g} M flops-equivalents a frame ({rays} rays of its window, "
                         f"{self.sub_frames} sub-frame{'s' * (self.sub_frames > 1)}), every {period} base steps: a count "
                         f"of the program, not a time")
            if cost > sampled_route.COST_LINE:
                lines.append(f"warning: {subject.name}'s frames are estimated at {cost / 1e9:.3g} G flops-equivalents, "
                             f"over the line of {sampled_route.COST_LINE / 1e9:.3g} G (about 35 to 45 ms in float32 and "
                             f"70 to 90 ms in float64 on a four-core CPU): render its frames offline from the recorded poses (mime.sensing.render), "
                             f"or lengthen frame_period")
        return lines

    def _window_rays(self, ctx: PlanContext, subject: Any) -> int:
        """The rays of a sampled body's window at its start pose, as the node sizes it; the frame's where the camera
        is mounted (its pose is a frame's) or the body has no start pose."""
        d = self._detector()
        frame = d.width * d.height
        body = subject.name.split(".", 1)[0]
        start = ctx.roster.bodies[body].pose if body in ctx.roster.bodies else None
        if not isinstance(self.mount, World) or start is None:
            return frame
        position = np.asarray(ctx.vector("mount.position"), dtype=np.float64)
        pose = CameraPose(position, np.asarray(self.mount.orientation, dtype=np.float64))
        at = np.asarray(start, dtype=np.float64)
        turned = rotate((float(at[3]), float(at[4]), float(at[5]), float(at[6])),
                        tuple(float(c) for c in subject.placed.centre))  # type: ignore[arg-type]
        origin = at[:3] + np.asarray(turned, dtype=np.float64)
        reach = (1.0 + FOCAL_SHARE) * float(np.asarray(projected_reach(d.pinhole(), pose, origin, subject.enclosing())))
        window = scene.frame_window(reach, d.width, d.height)
        return min(frame, (2 * window + 1) ** 2)

    def _notes(self, ctx: PlanContext, found: scene.Found, period: int, mounted: str | None) -> tuple[str, ...]:
        d = self._detector()
        route = ", ".join(sorted(f"{s.name} ({s.kind})" for s in found.subjects)) or "none"
        sub = self.sub_frames
        dt = ctx.dt
        sampled = [s for s in found.subjects if s.placed.sampled is not None]
        closed = ", ".join(sorted(f"{s.name} ({s.kind})" for s in found.subjects if s.placed.sampled is None))
        lines = [
            f"camera: a declared camera model, {d.width} x {d.height} pixels, {d.source}",
            *([f"route 1 (closed form) for every subject: {route}"] if not sampled else
              [f"route 1 (closed form) for: {closed or 'none'}",
               *self._sampled_notes(ctx, sampled, period)]),
            *self._mesh_notes(ctx, found),
            f"frame: {d.width * d.height} rays, {sub} sub-frame{'s' * (sub > 1)}, rendered every {period} base steps "
            f"({period * dt:.6g} s); exposure {sub} step{'s' * (sub > 1)} ending at the frame step",
            f"latency: a read sees a frame between 0 and {period - 1} steps ({(period - 1) * dt:.6g} s) old; a controller "
            f"called between chunks reads the state after the previous chunk, up to a chunk later",
        ]
        if found.hidden:
            lines.append(f"hidden: {', '.join(found.hidden)}")
        staged = [s.name for s in found.subjects if s.name in found.display] + \
            [each.name for each in found.meshes if each.appearance is None and each.look() is not None]
        defaulted = [s.name for s in found.subjects if s.name not in dict(self.absorbers) and s.name not in found.display] + \
            [each.name for each in found.meshes if each.look() is None]
        if staged:
            lines.append(f"appearance from the stage's display primvars for: {', '.join(staged)}")
        lines += [f"appearance: {line} for: {', '.join(names)}" for line, names in found.reduced.items()]
        if defaulted:
            lines.append(f"appearance defaulted for: {', '.join(defaulted)}")
        lines.append("bodies are taken to move and structures to stand; apparatus parts are not imaged")
        if mounted is None:
            lines.append("static cache: the camera and the static structure share the world's frame, so the structure is "
                         "rendered once, at the build, and held as a leaf beside the numbers it was made at; a run whose "
                         "detector numbers differ renders it once, before its steps"
                         + ("; with the traced meshes it is made once a run, and again at each study point that varies "
                            "the detector's numbers, and each frame reads the tree's checksum (its cost, measured on runners, is on "
                            "the camera's entry page)" if found.meshes else ""))
        else:
            statics = [s.name for s in found.subjects if not s.moving]
            lines.append(f"static cache: the camera is mounted on {mounted}, taken to move, so the static structure "
                         f"({', '.join(statics) or 'none'}) is rendered again in every frame: {d.width * d.height} rays "
                         f"each, no cache is held (a stale one would be a wrong image)")
        return tuple(lines)

    def _mesh_notes(self, ctx: PlanContext, found: scene.Found) -> list[str]:
        """The build notes of the traced meshes: each with its triangles and look, the sub-rays a pixel, a drawing's
        facets and sagitta and a scenery mesh's decimation in pixels at this camera, and each body's mesh parent."""
        if not found.meshes:
            return []
        d = self._detector()
        position = np.asarray(ctx.vector("mount.position"), dtype=np.float64) if isinstance(self.mount, World) else None
        lines = [f"traced static meshes (a bounding volume hierarchy, once a run, {MESH_SAMPLES} x {MESH_SAMPLES} "
                 f"sub-rays a pixel): " + ", ".join(
                     f"{each.name} ({len(each.triangles)} triangles, {(each.look() or Appearance.opaque()).look.value})"
                     for each in found.meshes)]

        def pixels(length: float, each: scene.TracedMesh) -> str:
            if position is None:
                return "unknown (the camera moves)"
            depth = _nearest_depth(each.triangles, position, self.mount.orientation)
            return "behind the camera" if depth is None else f"{length * max(d.fx, d.fy) / depth:.3g} px"

        drawn = [each for each in found.meshes if each.source != "scenery"]
        if drawn:
            lines.append("structures imaged by their drawing: " + ", ".join(
                f"{each.name} ({each.source}, exact)" if each.sagitta == 0.0 else
                f"{each.name} ({each.source}, {each.segments} facets a turn, sagitta about {each.sagitta:.3g} m, "
                f"{pixels(each.sagitta, each)} at this camera)" for each in drawn))
        recorded = [each for each in found.meshes if each.source == "scenery" and each.decimation is not None]
        if recorded:
            lines.append("decimation distance at this camera: " + ", ".join(
                f"{each.name} {pixels(cast(float, each.decimation), each)} ({cast(float, each.decimation):.3g} m)"
                for each in recorded))
        unrecorded = [each.name for each in found.meshes if each.source == "scenery" and each.decimation is None]
        if unrecorded:
            lines.append(f"decimation distance not recorded for: {', '.join(unrecorded)} (its layer holds none, so what "
                         f"a decimation moved is unknown)")
        if found.inside:
            lines.append("a mesh as parent medium, outside_parent not checked against it (no signed distance in the "
                         "step): " + ", ".join(f"{subject} in {mesh}" for subject, mesh in sorted(found.inside.items())))
        return lines

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
        meshes: dict[str, Any] = {}
        if found.meshes:
            spec["meshes"] = [scene.mesh_spec(each) for each in found.meshes]
            spec["mesh_samples"] = MESH_SAMPLES
            meshes = {"mesh_triangles": np.concatenate([each.triangles for each in found.meshes]),
                      "mesh_owners": np.concatenate([np.full(len(each.triangles), i, dtype=np.int64)
                                                     for i, each in enumerate(found.meshes)])}
        position = ctx.vector("mount.position")
        placement = (*position, *self.mount.orientation)
        mount = ctx.mount
        initial = [c for name in bodies for c in cast(tuple[float, ...], ctx.roster.bodies[name].pose)]
        node = b.add_node(CameraNode(
            ctx.node_name(), ctx.timestep, spec=scene.dumps(spec), fx=d.fx, fy=d.fy, cx=d.cx, cy=d.cy, k1=d.k1, k2=d.k2, p1=d.p1, p2=d.p2, k3=d.k3,
            bodies=bodies, initial_poses=initial, period=period,
            sub_frames=self.sub_frames, seed=ctx.seeds["noise"], placement=placement,
            mount_index=mount.index if mount is not None else -1, mount_count=mount.count if mount is not None else 0,
            initial_mount_pose=mount.initial_pose if mount is not None else (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            **meshes))
        for name in bodies:
            b.connect(b.read(f"pose_{name}"), into=node[f"pose_{name}"])
        if mount is not None:
            b.mount(node["mount_poses"])
