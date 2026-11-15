"""A magnetic sphere steered by what a camera sees: the frame's soft centroid, a PID and the exact allocation of the
coils' currents, all inside the compiled step, so that a derivative passes from the run's objective back through the
coils, the body, the camera and the tracker to the controller's gains and its model of the camera.

**The loop**: camera port -> tracker -> PID -> allocator -> coil command port, one controller in GRAPH mode
(``mode="graph"``). It reads the camera's expected frame through a measurement port (``imaging.frames``, of
``camera:.expected``: the counts before the sensor's noise, the differentiable output), and writes the coils' currents
through a command port that holds them within Navion_2's declared 25 A. Its law composes two plain functions of this
file, :func:`tracked` and :func:`pid`, with the allocation its context gives (``ctx.allocator``) and its model of the
camera (``ctx.believed("camera")``, the declaration's numbers, or a ``believed=`` camera that differs).

**The case** is camera spike R2's: a 1 mm sphere of 4e-6 kg with a fixed magnetic moment in Navion_2's field, the force
alone (no torque, so the moment keeps its direction), no gravity and no medium, dt = 1 ms. It starts at the multipole
oracle's point 52 (inside Navion_2's declared region) with that point's moment. A camera 0.1 m above it looks down, 96 x
72 pixels at 400 px focal length (0.25 mm a pixel there, the sphere 8 px across), a frame every 5 ms; the controller
ticks every 5 ms with 1 ms of latency. The reference is a step to a point a few millimetres off the start, then a 1.5
mm circle at 2 Hz about it, in the plane the sphere starts in.

**What the tracker sees, and what it cannot.** The camera sees two axes. Depth comes from a declared plane (the
sphere's starting height): the tracker back-projects the shadow's centroid onto it, so a sphere that leaves the plane is
mis-placed in x and y by its height change times its offset from the optical axis over the camera's distance. Nothing
holds this sphere on the plane, and on a tick where a coil sits at its bound the allocation trades the z row: the
sphere then drifts in z, which the camera does not measure. The tracker flags it (``off_plane``) where the shadow's area
says the depth has moved by more than ``plane_tolerance``. A target that leaves the frame, or whose shadow reaches the
frame's border, is lost: the tracker holds its last point, the law writes no current while it is lost, and ``lost``
counts the ticks. Both flags are the controller's state, read from the run (``result.final["controller"]``). The state
also keeps what each tick asked of the allocation, the measured point, the force wanted and the rows there (``point``,
``wanted``, ``rows``: the force per unit current, N/A, as the law's model evaluated it in the run's arithmetic), so that
the problem the allocation solved can be solved again by another solver.

**The derivative's status** is ``piecewise_jump``: the allocation's active set changes where a coil reaches its bound
(a jump in the derivative, never in the currents), the lost guard switches the law off, and the camera's own coverage
makes the objective only C^(1,1/2) in a parameter (a disc's segment past a pixel's side grows as its depth to the power
3/2), so a difference check through it measures its order and never assumes two. The coils' limit is a port's bound, a
constant of the build: no derivative is taken in it.

**What this example shows, and what it does not.** It shows a closed loop through a camera that is differentiable end to
end, and the checks that hold its derivative, its allocation and its guard. It does not track well: the default run ends
3.47 mm from the reference against a step of 3.35 mm, the sphere having drifted 3.4 mm in depth (flagged off the plane on
32 ticks), while the true-state arm, the same loop reading the sphere's pose, ends 0.61 mm away.

``python examples/camera_loop.py`` builds the loop, runs it for 0.2 s, prints the build report's controller block, the
tracking miss and the flags, then the derivative of the final miss in the gains (reverse mode, through ``Study``).
``tests/examples/test_camera_loop.py`` and ``test_camera_loop_tracker.py`` hold it."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax

from mime import Apparatus, Body, CommandPort, Experiment, Measurement, State
from mime.control import allocation_switch
from mime.core import quaternion, switch
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.fields import CalibratedMultipole
from mime.gradients import GradientStatus
from mime.interactions import Magnetic
from mime.observations import Camera, Detector, World
from mime.registry import ControlLaw, Controller, GradientRecord, Signal, register_controller
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import A, A_m2, N, kg, kg_m2, m, m_per_s, m_per_s2, ms, rad_per_s
from mime.vocabulary import register_namespace

jnp: Any = jax.numpy

# region: case
register_namespace("example", "0.1.0")

#: The multipole oracle's point 52 in Navion_2's declared region, and the moment drawn there (A m^2).
START = (0.00134162727289842, 0.02642104454083892, 0.02148800181565995)
MOMENT = (-2.5128544555197885e-04, 7.7870258610772701e-05, 8.4890458193167159e-05)
MASS, RADIUS = 4e-6, 1e-3
#: Navion_2's declared limit on each coil's current (A).
LIMIT = 25.0
WIDTH, HEIGHT, FOCAL = 96, 72, 400.0
#: The camera's optical frame (x right, y down, z out of the lens) looking down -z: a half turn about x.
LOOK_DOWN = (0.0, 1.0, 0.0, 0.0)
CAMERA_AT = (START[0], START[1], START[2] + 0.1)
DETECTOR = Detector(width=WIDTH, height=HEIGHT, fx=FOCAL, fy=FOCAL, cx=(WIDTH - 1) / 2, cy=(HEIGHT - 1) / 2)
#: The expected frame's counts where nothing shades it: the camera's default sensor, 1000 electrons at full backlight
#: at one count each.
BACKLIGHT = 1000.0
FRAME_MS, RATE_MS, LATENCY_MS = 5, 5, 1
#: The reference's default step off the start (m) and its circle: radius (m) and frequency (Hz).
STEP = (3e-3, -1.5e-3)
CIRCLE, FREQUENCY = 1.5e-3, 2.0
# endregion case

#: The switches' owners, which the controller declares.
ACTIVE_SET = "example.camera_pid.active_set"
LOST = "example.camera_pid.lost"
#: A target is lost where its shadow holds less than this share of the ball's area on the plane, or where it reaches
#: the frame's border by more than BORDER of a pixel's darkness: half a ball is in view when its centre is on the
#: frame's edge, and a shadow clipped by the border pulls its centroid inwards.
LOST_SHARE, BORDER = 0.5, 1e-3
#: Newton steps of the back-projection onto the plane, from the pinhole's guess: none move it without a believed lens
#: distortion, and three take a mild one to rounding.
NEWTON_STEPS = 3

LOST_SWITCH = switch.Declaration(
    owner=LOST,
    indicator="the shadow's darkness less half the ball's area on the plane, in pixels (and its darkness on the frame's "
              "border, which must be nil)",
    orders={"currents": -1},
    route="none: a target out of view has no measurement to differentiate; the law writes no current while it is lost")


class Tracked(NamedTuple):
    """What the tracker gives each tick: the point on the declared plane (m, the world), the shadow's centroid (px),
    its darkness in pixels (``mass``), whether the target is ``lost`` (its point then the one held), and the depth the
    shadow's area implies less the plane's (``depth_miss``, m, positive away from the camera), and how far the
    darkness is above what a lost target leaves (``margin``, px: the lost guard's indicator)."""

    point: Any
    pixel: Any
    mass: Any
    lost: Any
    depth_miss: Any
    margin: Any


def soft_centroid(frame: Any, backlight: float) -> tuple[Any, Any, Any, Any]:
    """The darkness ``(backlight - frame) / backlight`` of each pixel (the share of it the shadow covers, for an opaque
    body), weighted by each pixel's centre: ``(sum of darkness * u, sum of darkness * v, total darkness, darkness on the
    border)``, with ``u`` along a row and ``v`` down the columns, a pixel's centre at whole numbers."""
    dark = (backlight - frame) / backlight
    rows, columns = frame.shape
    v, u = jnp.meshgrid(jnp.arange(rows, dtype=frame.dtype), jnp.arange(columns, dtype=frame.dtype), indexing="ij")
    border = jnp.sum(dark[0]) + jnp.sum(dark[-1]) + jnp.sum(dark[1:-1, 0]) + jnp.sum(dark[1:-1, -1])
    return jnp.sum(dark * u), jnp.sum(dark * v), jnp.sum(dark), border


def _optical(model: Any, point: Any) -> Any:
    """``point`` (the world) in the believed camera's optical frame."""
    rotation = quaternion.rotation_matrix(jnp.asarray(model.theta["orientation"], dtype=point.dtype))
    return (point - jnp.asarray(model.theta["position"], dtype=point.dtype)) @ rotation


def ball_area(model: Any, centre: Any, radius: float) -> Any:
    """The area in pixels of a ball's silhouette through the believed pinhole (its lens distortion left out), centred
    at ``centre`` (the world): the ellipse the cone tangent to the ball cuts from the image plane,
    ``pi fx fy r^2 sqrt(D^2 - r^2) / (Z^2 - r^2)^(3/2)`` for a centre at distance ``D`` and depth ``Z``."""
    local = _optical(model, centre)
    held = jnp.asarray(radius, dtype=centre.dtype)
    distance2 = jnp.sum(local * local)
    fx, fy = (jnp.asarray(model.theta[k], dtype=centre.dtype) for k in ("fx", "fy"))
    return math.pi * fx * fy * held * held * jnp.sqrt(distance2 - held * held) / (local[2] ** 2 - held * held) ** 1.5


def on_plane(model: Any, pixel: Any, height: Any) -> Any:
    """The point of the plane ``z = height`` (the world) whose image through the believed camera is ``pixel``: the
    pinhole's ray through it as the first guess, then Newton steps on the belief's own projection, so that a believed
    lens distortion is undone by the projection that applies it."""
    dtype = pixel.dtype
    theta = model.theta
    rotation = quaternion.rotation_matrix(jnp.asarray(theta["orientation"], dtype=dtype))
    position = jnp.asarray(theta["position"], dtype=dtype)
    fx, fy, cx, cy = (jnp.asarray(theta[k], dtype=dtype) for k in ("fx", "fy", "cx", "cy"))
    ray = rotation @ jnp.stack([(pixel[0] - cx) / fx, (pixel[1] - cy) / fy, jnp.ones((), dtype)])
    guess = position[:2] + (height - position[2]) / ray[2] * ray[:2]

    def miss(xy: Any) -> Any:
        return model.project(jnp.stack([xy[0], xy[1], height]))[0] - pixel

    for _ in range(NEWTON_STEPS):
        jac = jax.jacfwd(miss)(guess)
        r = miss(guess)
        det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
        guess = guess - jnp.stack([jac[1, 1] * r[0] - jac[0, 1] * r[1], jac[0, 0] * r[1] - jac[1, 0] * r[0]]) / det
    return jnp.stack([guess[0], guess[1], jnp.asarray(height, dtype)])


def tracked(frame: Any, model: Any, *, height: Any, radius: float, backlight: float, held: Any) -> Tracked:
    """The 3-D point the frame's soft centroid back-projects to on the plane ``z = height`` through the believed
    camera ``model``, or ``held`` (the last point) where the target is lost; never a division by nothing."""
    su, sv, mass, border = soft_centroid(frame, backlight)
    expected = ball_area(model, held, radius)
    clear = switch.predicate(border <= BORDER, owner=LOST, resolution=lambda: switch.rounding(1.0),
                             indicator=lambda: BORDER - border)
    seen = clear & (mass >= LOST_SHARE * expected)
    safe = switch.domain_guard(seen, mass, 1.0)
    pixel = jnp.stack([su / safe, sv / safe])
    margin = mass - LOST_SHARE * expected
    point = switch.where(seen, on_plane(model, pixel, height), held, owner=LOST,
                         resolution=lambda: switch.rounding(expected), indicator=lambda: margin)
    depth = _optical(model, point)[2]
    implied = depth * jnp.sqrt(ball_area(model, point, radius) / safe)
    return Tracked(point, pixel, mass, ~seen, jnp.where(seen, implied - depth, 0.0), margin)


class Gains(NamedTuple):
    kp: Any
    ki: Any
    kd: Any


def pid(state: Mapping[str, Any], measured: Any, reference: Any, gains: Gains, period: float, *,
        frozen: Any) -> tuple[dict[str, Any], Any]:
    """The force (N) a PID on the 3-D error asks of the sphere, ``m (kp e + ki int e - kd dp/dt)``, its derivative taken
    on the measurement; the integral held where ``frozen`` (a lost tick)."""
    error = reference - measured
    integral = jnp.where(frozen, state["integral"], state["integral"] + error * period)
    rate = (measured - state["previous"]) / period
    wanted = MASS * (gains.kp * error + gains.ki * integral - gains.kd * rate)
    return {"integral": integral, "previous": measured}, wanted


def reference(t: Any, step: tuple[float, float], settle: float = 0.0) -> Any:
    """The reference at time ``t`` (s): the start until ``settle`` (s), then the start moved by ``step`` (m), then a
    circle about that point."""
    omega = 2.0 * math.pi * FREQUENCY
    since = t - settle
    moved = jnp.stack([START[0] + step[0] + CIRCLE * (jnp.cos(omega * since) - 1.0),
                       START[1] + step[1] + CIRCLE * jnp.sin(omega * since), jnp.full(jnp.shape(t), START[2])])
    return jnp.where(since >= 0.0, moved, jnp.asarray(START, dtype=moved.dtype).reshape((3,) + (1,) * jnp.ndim(t)))


def _law(ctx: Any, measure: Callable[[Mapping[str, Any], Any, Any], Tracked]) -> ControlLaw:
    gains = Gains(ctx.scalar("kp"), ctx.scalar("ki"), ctx.scalar("kd"))
    step = (ctx.scalar("step_x"), ctx.scalar("step_y"))
    settle = ctx.scalar("settle")
    period = ctx.timestep
    solve = ctx.allocator("coils", commands=("currents",), objective={"force": N(1e-3)}, owner=ACTIVE_SET)
    dtype = jnp.zeros(()).dtype
    moment = jnp.asarray(MOMENT, dtype=dtype)

    def count(flag: Any) -> Any:
        return flag.astype(dtype)

    zero = jnp.zeros((), dtype=dtype)

    def step_(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        t = state["tick"] * period
        found = measure(reads, state["point"], dtype)
        held, wanted = pid(state, found.point, reference(t, step, settle), gains, period, frozen=found.lost)
        at = {"point": found.point, "moment": moment}
        currents, _ = solve({"force": wanted}, at)
        rows, _ = solve.model.affine(at, commands=("currents",), dtype=dtype)
        written = switch.where(found.lost, jnp.zeros_like(currents["currents"]), currents["currents"], owner=LOST,
                               resolution=lambda: switch.rounding(found.mass), indicator=lambda: -found.margin)
        off = jnp.abs(found.depth_miss) > ctx.scalar("plane_tolerance")
        first = jnp.where((state["first_lost"] < 0) & found.lost, state["tick"], state["first_lost"])
        after = {**held, "tick": state["tick"] + 1, "point": found.point, "wanted": wanted,
                 "rows": rows["force"]["currents"],
                 "age": jnp.asarray(reads.get("age", zero), dtype=dtype),
                 "lost": state["lost"] + count(found.lost), "first_lost": first,
                 "off_plane": state["off_plane"] + count(off),
                 "depth_miss": jnp.maximum(state["depth_miss"], jnp.abs(found.depth_miss))}
        return after, {"currents": written}

    origin = jnp.asarray(START, dtype=dtype)
    start = {"tick": zero, "age": zero, "integral": jnp.zeros((3,), dtype=dtype), "previous": origin, "point": origin,
             "wanted": jnp.zeros((3,), dtype=dtype), "rows": jnp.zeros((3, 3), dtype=dtype), "lost": zero,
             "first_lost": zero - 1.0, "off_plane": zero,
             "depth_miss": zero}
    return ControlLaw(state=start, held={"currents": jnp.zeros((3,), dtype=dtype)}, step=step_)


_RECORD = GradientRecord(switches=(allocation_switch(ACTIVE_SET, {"currents": -1}), LOST_SWITCH))


# region: controller
@register_controller
@dataclass(frozen=True, kw_only=True)
class CameraPID(Controller):
    """The PID through the exact allocation, measuring by the tracker on the camera's expected frame, through its
    believed camera; ``plane_tolerance`` (m) the depth change the shadow's area flags."""

    name = "example.camera_pid"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = _RECORD
    builds = {"": ()}
    reads = {"frames": Signal("1", (HEIGHT, WIDTH)), "age": Signal("1", ())}
    writes = {"currents": Signal("A", (3,))}
    kp: float = 2500.0
    ki: float = 3000.0
    kd: float = 80.0
    step_x: float = STEP[0]
    step_y: float = STEP[1]
    settle: float = 0.0
    plane_tolerance: float = 5e-4

    def law(self, ctx: Any) -> ControlLaw:
        model = ctx.believed("camera")

        def measure(reads: Mapping[str, Any], held: Any, dtype: Any) -> Tracked:
            return tracked(reads["frames"], model, height=jnp.asarray(START[2], dtype=dtype), radius=RADIUS,
                           backlight=BACKLIGHT, held=held)

        return _law(ctx, measure)
# endregion controller


@register_controller
@dataclass(frozen=True, kw_only=True)
class StatePID(CameraPID):
    """The same law measuring the sphere's true position (ground truth, not deployable): the true-state run."""

    name = "example.state_pid"
    reads = {"pose": Signal("m; 1", (7,))}

    def law(self, ctx: Any) -> ControlLaw:
        def measure(reads: Mapping[str, Any], held: Any, dtype: Any) -> Tracked:
            zero = jnp.zeros((), dtype=dtype)
            return Tracked(reads["pose"][:3], jnp.zeros((2,), dtype=dtype), zero, jnp.zeros((), bool), zero, zero + 1.0)

        return _law(ctx, measure)


def camera(detector: Detector = DETECTOR, frame_ms: float = FRAME_MS) -> Camera:
    return Camera(detector=detector, mount=World(position=m(CAMERA_AT), orientation=LOOK_DOWN), frame_period=ms(frame_ms))


# region: experiment
def experiment(name: str = "camera_loop", *, arm: str = "camera", believed: Camera | None = None,
               frame_ms: float = FRAME_MS, rate_ms: float = RATE_MS, latency_ms: float = LATENCY_MS,
               port_latency_ms: float = 0.0, command_latency_ms: float = 0.0, mode: str = "graph",
               **settings: float) -> Experiment:
    """The loop: ``arm`` ``"camera"`` (the tracker on the camera's port) or ``"state"`` (the sphere's true pose); the
    controller's ``believed`` camera (the declaration's where none is given); the frame period, the controller's rate
    and latency, and the measurement and command ports' latencies, in milliseconds; the controller's ``settings``."""
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    exp.add_apparatus(Apparatus(
        "coils", fields={"magnetic": CalibratedMultipole(calibration=asset("navion_2"))},
        ports={"currents": CommandPort(bounds=A(LIMIT), latency=ms(command_latency_ms) if command_latency_ms else None,
                                       reason="Navion_2's declared limit on each coil")}))
    exp.add_body(Body("sphere", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=m(RADIUS)))},
                      initial=State(position=m(START), orientation=(1.0, 0.0, 0.0, 0.0),
                                    velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("sphere.mass", kg(MASS)), ("sphere.inertia", kg_m2((1.6e-12,) * 3)),
                        ("sphere.magnetic_moment", A_m2(MOMENT)), ("sphere.magnet_centre", m((0.0, 0.0, 0.0)))):
        exp.declare(path, value, reason="a millimetre sphere with a magnet at its centre, the multipole oracle's moment")
    exp.attach(Magnetic.Force(), body="sphere", field="coils.magnetic")
    exp.observe(camera(frame_ms=frame_ms))
    exp.add_apparatus(Apparatus("imaging", ports={"frames": Measurement(
        of="camera:.expected", latency=ms(port_latency_ms) if port_latency_ms else None,
        reason="the frame read as it is made" if not port_latency_ms else "a declared transfer delay")}))
    controller = (CameraPID if arm == "camera" else StatePID)(**settings)
    reads = {"frames": "imaging.frames", "age": "imaging.frames.age"} if arm == "camera" else {"pose": "sphere.pose"}
    exp.set_controller(controller, reads=reads, writes={"currents": "coils.currents"}, mode=mode, rate=ms(rate_ms),
                       latency=ms(latency_ms) if latency_ms else None,
                       believed={"camera": believed} if believed is not None else None)
    return exp
# endregion experiment


def miss(final: Mapping[str, Any], step: tuple[float, float] = STEP, steps: int = 200) -> Any:
    """The final miss squared, in mm^2: the sphere's true position after ``steps`` against the reference then."""
    gap = final["sphere"]["pose"][:3] - reference(jnp.asarray(steps * 1e-3), step)
    return jnp.sum(gap * gap) / 1e-6


def main() -> None:
    from mime import Study
    from mime.study import grid

    # region: run
    exp = experiment()
    built = exp.build()
    print("\n".join(built.report.controller))
    result = built.run(ms(200))
    held = result.final["controller"]
    print(f"final miss {math.sqrt(float(miss(result.final))):.3f} mm; lost on {int(held['state_lost'])} ticks; "
          f"off the plane on {int(held['state_off_plane'])} (largest depth change "
          f"{1e3 * float(held['state_depth_miss']):.2f} mm)")
    # endregion run
    # region: gradient
    study = Study(exp).sweep(grid({"controller.kp": [2500.0], "controller.kd": [80.0]}), duration=ms(200))
    found = study.gradient(miss, mode="reverse")
    print(found)
    # endregion gradient


if __name__ == "__main__":
    main()
