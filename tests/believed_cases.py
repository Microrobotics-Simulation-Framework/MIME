"""Test-only controllers that read a believed model and allocate through it (``ctx.believed``, ``ctx.allocator``),
registered under the ``oracle`` namespace in this process only, and the experiments they run in.

The rig is camera spike R2's STATE arm, as ``allocation_cases`` builds it: a 1 mm sphere of 4e-6 kg with a fixed moment
of 2.8e-4 A m^2 at the centre of Navion_2's declared region, the force alone, no gravity, no medium, dt = 1 ms; its
controller reads the sphere's true pose every 5 ms, one step late, and pulls it towards a point 3 mm off its start by a
PD force. :class:`BelievedPD` allocates that force through ``ctx.allocator`` on its believed model of the coils, within
the bounds their port declares (25 A), where ``allocation_cases.AllocatingPD`` writes the same allocation by hand."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax

from allocation_cases import MASS, MOMENT, OFFSET, _start

from mime import Apparatus, Body, Experiment, State
from mime.control import allocation_switch
from mime.assets import asset
from mime.dynamics import RigidInertial
from mime.fields import CalibratedMultipole
from mime.gradients import GradientStatus
from mime.interactions import Magnetic
from mime.registry import ControlLaw, Controller, GradientRecord, Signal, register_controller
from mime.scene import Scene, Sphere, Z
from mime.things import CommandPort, Part
from mime.units import A, A_m2, N, kg, kg_m2, m, m_per_s, m_per_s2, ms, rad_per_s

jnp: Any = jax.numpy  # the oracle namespace is registered by allocation_cases

#: The owner the believed controllers declare and pass to the allocation.
OWNER = "believed_cases.active_set"
FORCE_SCALE = 1e-3
LIMIT = 25.0
#: The port's reason, which a port with numbers needs.
REASON = "the test rig's declared limit"


@register_controller
@dataclass(frozen=True, kw_only=True)
class BelievedPD(Controller):
    """A PD force towards a point, allocated by ``ctx.allocator`` on the controller's model of the coils (its belief,
    the truth where none is declared), within the coils' port's bounds. ``smooth``, a current, regularises towards the
    last command at that scale (``toward="last"``), which a port's slew needs as ``last``."""

    name = "oracle.believed_pd"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(allocation_switch(OWNER, {"currents": 0}),))
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"currents": Signal("A", (3,))}
    kp: float = 2500.0
    kd: float = 80.0
    smooth: float = 0.0

    def law(self, ctx: Any) -> ControlLaw:
        smooth = ctx.scalar("smooth")
        solve = ctx.allocator("coils", commands=("currents",), objective={"force": N(FORCE_SCALE)}, owner=OWNER,
                              effort=A(smooth) if smooth > 0.0 else None, toward="last")
        dtype = jnp.zeros(()).dtype
        kp, kd, period = ctx.scalar("kp"), ctx.scalar("kd"), ctx.timestep
        start = jnp.asarray(_start(), dtype=dtype)
        goal = start + jnp.asarray(OFFSET, dtype=dtype)
        moment = jnp.asarray(MOMENT, dtype=dtype)

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            point = reads["pose"][:3]
            wanted = MASS * (kp * (goal - point) - kd * (point - state["previous"]) / period)
            commands, _ = solve({"force": wanted}, {"point": point, "moment": moment}, {"currents": state["last"]})
            return {"previous": point, "last": commands["currents"]}, {"currents": commands["currents"]}

        zero = jnp.zeros((3,), dtype=dtype)
        return ControlLaw(state={"previous": start, "last": zero}, held={"currents": zero}, step=step)


@register_controller
@dataclass(frozen=True, kw_only=True)
class GlitchedPD(BelievedPD):
    """:class:`BelievedPD` whose write at its tick ``glitch_at`` is ``glitch`` amperes beyond what it allocated, in every
    coil: a fault on the line the allocation cannot know of (what trips a port that trips at its bounds). Its ``last``
    stays the allocation's, so the box goes on being taken around commands the stage no longer stands at."""

    name = "oracle.glitched_pd"
    glitch_at: float = 3.0
    glitch: float = 30.0

    def law(self, ctx: Any) -> ControlLaw:
        inner = super().law(ctx)
        at, extra = ctx.scalar("glitch_at"), ctx.scalar("glitch")

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            after, written = inner.step({k: v for k, v in state.items() if k != "tick"}, reads)
            beyond = jnp.where(state["tick"] == at, extra, 0.0).astype(written["currents"].dtype)
            return {**after, "tick": state["tick"] + 1}, {"currents": written["currents"] + beyond}

        return ControlLaw(state={**inner.state, "tick": jnp.asarray(0.0, dtype=jnp.zeros(()).dtype)}, held=inner.held,
                          step=step)


def model_of(experiment: Experiment, owner: str) -> Any:
    """The forward model of the entry ``owner`` of ``experiment``, as its plan makes it (``forward_model``)."""
    from mime.build._construct import entry_context

    plans = experiment.plan()
    plan = next(p for p in (*plans.entries, *plans.declared) if p.owner == owner)
    made = plan.entry.forward_model(entry_context(plan, plans))
    assert made is not None
    return made


def coils(*, at: tuple[float, float, float] = (0.0, 0.0, 0.0), system: str = "navion_2") -> CalibratedMultipole:
    return CalibratedMultipole(calibration=asset(system), at=m(at))


def believed_sphere(name: str = "believed_sphere", *, truth: CalibratedMultipole | None = None,
                    believed: Mapping[str, Any] | None = None, port: CommandPort | None = None, mode: str = "graph",
                    rate: Any = None, latency: Any = None, controller: Controller | bool | None = None) -> Experiment:
    """The sphere in the coils' field, its currents written by ``controller`` (a :class:`BelievedPD` by default;
    ``False``, none: the currents a constant 1 A in the first coil, a document a load can read) through the coils' port
    (25 A bounds unless ``port`` is given), the coils ``truth`` (Navion_2 at the world's origin by default) and the
    controller's belief ``believed``."""
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    held = port if port is not None else CommandPort(bounds=A(LIMIT), reason=REASON)
    exp.add_apparatus(Apparatus("coils", fields={"magnetic": truth if truth is not None else coils()},
                                ports={"currents": held},
                                commands={"currents": A((1.0, 0.0, 0.0))} if controller is False else {}))
    exp.add_body(Body("sphere", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=m(1e-3)))},
                      initial=State(position=m(_start()), orientation=(1.0, 0.0, 0.0, 0.0),
                                    velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("sphere.mass", kg(MASS)), ("sphere.inertia", kg_m2((1.6e-12,) * 3)),
                        ("sphere.magnetic_moment", A_m2(MOMENT)), ("sphere.magnet_centre", m((0.0, 0.0, 0.0)))):
        exp.declare(path, value, reason="a millimetre sphere with a magnet at its centre")
    exp.attach(Magnetic.Force(), body="sphere", field="coils.magnetic")
    if controller is False:
        return exp
    exp.set_controller(controller if isinstance(controller, Controller) else BelievedPD(), reads={"pose": "sphere.pose"},
                       writes={"currents": "coils.currents"}, mode=mode, rate=rate if rate is not None else ms(5),
                       latency=latency if latency is not None else ms(1), believed=believed)
    return exp


@register_controller
@dataclass(frozen=True, kw_only=True)
class AllocatingProbe(Controller):
    """A controller whose law asks ``ctx.allocator`` for ``commands`` of ``apparatus`` with ``objective`` (pairs of a
    quantity and its scale) and writes zeros: what the allocator refuses at the build, and its control, which builds.
    ``raising`` makes its law raise a plain ``ValueError`` instead (an entry's defect, never a refusal's code)."""

    name = "oracle.allocating_probe"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(allocation_switch(OWNER, {"currents": 0}),))
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"currents": Signal("A", (3,))}
    apparatus: str = "coils"
    commands: tuple[str, ...] = ("currents",)
    objective: tuple[tuple[str, Any], ...] = (("force", N(FORCE_SCALE)),)
    raising: bool = False

    def law(self, ctx: Any) -> ControlLaw:
        if self.raising:
            raise ValueError("a law's own defect")
        ctx.allocator(self.apparatus, commands=self.commands, objective=dict(self.objective), owner=OWNER)
        zero = jnp.zeros((3,), dtype=jnp.zeros(()).dtype)
        return ControlLaw(state={}, held={"currents": zero}, step=lambda state, reads: ({}, {"currents": zero}))



@register_controller
@dataclass(frozen=True, kw_only=True)
class AllocatingArmProbe(AllocatingProbe):
    """:class:`AllocatingProbe` writing an arm's joint positions: what the allocator refuses for a model affine in
    none of its commands."""

    name = "oracle.allocating_arm_probe"
    gradient = GradientRecord(switches=(allocation_switch(OWNER, {"joint_positions": 0}),))
    writes = {"joint_positions": Signal("1", (-1,))}

    def law(self, ctx: Any) -> ControlLaw:
        ctx.allocator(self.apparatus, commands=self.commands, objective=dict(self.objective), owner=OWNER)
        raise AssertionError("the allocator refuses an arm's joint positions")


#: The camera the camera-belief rig sees the sphere with: 1 cm from the origin on -y, looking across (+y).
CAMERA_DETECTOR = dict(width=160, height=120, fx=750.0, fy=750.0, cx=79.5, cy=59.5)
CAMERA_LOOK_ACROSS = (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)
#: The world point the camera probe projects, and its writes' scale (A per pixel, A per metre of depth).
PROJECTED = (1e-3, -5e-4, 7e-4)
PIXEL_SCALE, DEPTH_SCALE = 0.1, 100.0


@register_controller
@dataclass(frozen=True, kw_only=True)
class CameraProbe(Controller):
    """A controller whose law projects a fixed world point through ``ctx.believed("camera")`` and writes the pixel
    ``(u, v)`` and the depth as currents (scaled): the believed camera's projection in the compiled step (graph) or on
    the host (callback), which a build must run for the leaf names of its numbers (a belief's key is an observation id,
    ``camera:``, with its colon)."""

    name = "oracle.camera_probe"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    gradient = GradientRecord()
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"currents": Signal("A", (3,))}

    def law(self, ctx: Any) -> ControlLaw:
        camera = ctx.believed("camera")
        dtype = jnp.zeros(()).dtype
        point = jnp.asarray(PROJECTED, dtype=dtype)

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            pixel, depth = camera.project(point)
            return {}, {"currents": jnp.stack([PIXEL_SCALE * pixel[0], PIXEL_SCALE * pixel[1],
                                               DEPTH_SCALE * depth]).astype(dtype)}

        zero = jnp.zeros((3,), dtype=dtype)
        return ControlLaw(state={}, held={"currents": zero}, step=step)


def camera_sphere(*, believed: Mapping[str, Any] | None = None, mode: str = "graph") -> Experiment:
    """:func:`believed_sphere` with a camera in the world observing it, its currents written by :class:`CameraProbe`."""
    from mime.observations import Camera, Detector, World

    exp = believed_sphere("camera_sphere", controller=CameraProbe(), mode=mode, believed=believed)
    exp.observe(Camera(detector=Detector(**CAMERA_DETECTOR), mount=World(position=m((0.0, -1e-2, 0.0)),
                                                                          orientation=CAMERA_LOOK_ACROSS),
                       frame_period=ms(5)))
    return exp


__all__ = ["FORCE_SCALE", "LIMIT", "OWNER", "REASON", "AllocatingArmProbe", "AllocatingProbe", "BelievedPD", "CameraProbe", "camera_sphere", "GlitchedPD", "believed_sphere", "coils",
           "model_of"]
