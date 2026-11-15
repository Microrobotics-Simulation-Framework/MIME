"""Experiments with a coil array on a frame of its own apparatus's chain (``CoilArray(mount=...)``) and with the exposure
envelope (``mime.observations.Exposure``), and the test-only controller that allocates through a mounted array.

The mounted rig is A0's kinematic AR4 at its home pose carrying a motor on its flange, the motor's rotor turning at
``SPEED`` about the flange's z, and Navion_2 mounted on the rotor: placed in the rotor's frame by ``ROTOR_AT``, the point
a sphere at the world's origin starts at in that frame (to a millimetre), and turned by ``TURN``, so that the sphere
starts near the array's centre and the array moves round it as the rotor turns (a millimetre a step at most). The sphere
has no moment, no gravity acts and it starts at rest, so it stays where it starts: what the array's field is read at is
known exactly at every step. The magnetic force reads the field at the sphere's centre (a provider nothing reads is not
planned); the currents are constant (``CURRENTS``), or a controller's (:class:`MountedFieldControl`).

The exposure's sphere is Navion_2 placed in the world, the same still sphere at its centre, and its currents through a
port with a latency of one step, so that the field is off at step 0 and on from step 1."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax

import allocation_cases  # noqa: F401 - registers the oracle namespace
from chain_cases import AR4, AR4_BASE, AR4_HOME

from mime import Apparatus, Body, Experiment, State
from mime.control import allocation_switch
from mime.data import asset as packaged
from mime.drives import Constant
from mime.dynamics import Arm, Motor, RigidInertial
from mime.fields import CalibratedMultipole
from mime.gradients import GradientStatus
from mime.interactions import Magnetic
from mime.registry import ControlLaw, Controller, GradientRecord, Signal, register_controller
from mime.scene import Scene, Sphere, Z, asset
from mime.things import CommandPort, Part
from mime.units import A, A_m2, N_m, kg, kg_m2, m, m_per_s, m_per_s2, mT, ms, rad, rad_per_s

jnp: Any = jax.numpy

#: The sphere's point in the rotor's frame at the start, to a millimetre: where the array's centre is placed.
ROTOR_AT = (-0.129, -0.0765, 0.0)
#: The array's turn in the rotor's frame: 40 degrees about (1, 2, 2) / 3, a unit quaternion (w, x, y, z).
TURN = (math.cos(math.radians(20.0)), *(c / 3.0 * math.sin(math.radians(20.0)) for c in (1.0, 2.0, 2.0)))
#: The rotor's speed (rad/s) and the coils' currents (A).
SPEED, CURRENTS = 2.0, (5.0, -3.0, 2.0)
#: The rotor's row in the motor's frame poses (its frames are its base and its rotor).
ROTOR = 1
#: The field the controller allocates at the sphere (T), and the scale of its objective.
WANTED = (2.0e-3, -1.0e-3, 1.5e-3)
#: The owner the mounted controller declares and passes to the allocation.
OWNER = "mount_cases.active_set"
LIMIT = 25.0
REASON = "the test rig's declared limit"


def coils(*, mount: str | None = "motor.rotor", at=ROTOR_AT, orientation=TURN, system: str = "navion_2"
          ) -> CalibratedMultipole:
    return CalibratedMultipole(calibration=packaged(system), mount=mount, at=m(at), orientation=orientation)


def still_sphere(exp: Experiment, *, position=(0.0, 0.0, 0.0), moment=(0.0, 0.0, 0.0)) -> None:
    """A millimetre sphere at rest at ``position``, its magnet (of ``moment``) at its centre: with no moment, no force
    or torque moves it."""
    exp.add_body(Body("sphere", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=m(1e-3)))},
                      initial=State(position=m(position), orientation=(1.0, 0.0, 0.0, 0.0),
                                    velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("sphere.mass", kg(4e-6)), ("sphere.inertia", kg_m2((1.6e-12,) * 3)),
                        ("sphere.magnetic_moment", A_m2(moment)), ("sphere.magnet_centre", m((0.0, 0.0, 0.0)))):
        exp.declare(path, value, reason="a millimetre test sphere")


def scene_experiment(name: str) -> Experiment:
    return Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))


def mounted_rig(name: str = "mounted_coils", *, array: CalibratedMultipole | None = None, speed: float = SPEED,
                controller: Controller | None = None, believed: Mapping[str, Any] | None = None,
                observe: Any = None, mode: str = "callback") -> Experiment:
    """The mounted rig: the AR4 and its motor, kinematic, the array on the rotor (:func:`coils` unless ``array`` is
    given), its currents constant or written by ``controller`` (every 2 ms, no latency, ``mode`` a callback unless
    given: inside the compiled step a controller cannot read the chain's frames, ``controller.unreached-read``)
    through a 25 A port; the still sphere under the array's force; ``observe``, an observation to add."""
    exp = scene_experiment(name)
    exp.add_apparatus(Apparatus("rig", chain={
        "arm": Arm.Kinematic(description=asset(AR4), initial_positions=rad(AR4_HOME), at=m(AR4_BASE)),
        "motor": Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0), mount="link_6")},
        fields={"coils": array if array is not None else coils()},
        commands={"arm.joint_positions": Constant(rad(AR4_HOME)), "motor.velocity": Constant(rad_per_s(speed)),
                  **({} if controller is not None else {"currents": Constant(A(CURRENTS))})},
        ports={"currents": CommandPort(bounds=A(LIMIT), reason=REASON)}))
    still_sphere(exp)
    exp.attach(Magnetic.Force(), body="sphere", field="rig.coils")
    if observe is not None:
        exp.observe(observe)
    if controller is not None:
        exp.set_controller(controller, reads={"pose": "sphere.pose", "frames": "rig.motor.frame_poses"},
                           writes={"currents": "rig.currents"}, mode=mode, rate=ms(2), latency=ms(0),
                           believed=believed)
    return exp


def arm_mounted(name: str = "arm_mounted", *, rung: str = "kinematic", mount: str = "arm.link_6") -> Experiment:
    """Navion_2 on the AR4's flange, the arm on its ``rung`` (``"kinematic"`` or ``"dynamic"``, held by zero torques
    with no gravity), the still sphere under its force."""
    exp = scene_experiment(name)
    if rung == "kinematic":
        arm = Arm.Kinematic(description=asset(AR4), initial_positions=rad(AR4_HOME), at=m(AR4_BASE))
        drive = {"arm.joint_positions": Constant(rad(AR4_HOME))}
    else:
        arm = Arm.Dynamic(description=asset(AR4), initial_positions=rad(AR4_HOME), at=m(AR4_BASE))
        drive = {"arm.joint_torques": Constant(N_m((0.0,) * 6))}
    exp.add_apparatus(Apparatus("rig", chain={"arm": arm},
                                fields={"coils": coils(mount=mount, at=ROTOR_AT)},
                                commands={**drive, "currents": Constant(A(CURRENTS))}))
    still_sphere(exp)
    exp.attach(Magnetic.Force(), body="sphere", field="rig.coils")
    return exp


@register_controller
@dataclass(frozen=True, kw_only=True)
class MountedFieldControl(Controller):
    """Each tick, the currents that make ``WANTED`` at the sphere, allocated by ``ctx.allocator`` on its model of the
    mounted array at the rotor's pose it reads that tick (``x.mount_pose``): the exact path, the model affine in the
    currents at every pose. Its state keeps what it read and what the model says the currents make (``made``)."""

    name = "oracle.mounted_field"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(allocation_switch(OWNER, {"currents": 0}),))
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,)), "frames": Signal("m; 1", (2, 7))}
    writes = {"currents": Signal("A", (3,))}

    def law(self, ctx: Any) -> ControlLaw:
        solve = ctx.allocator("rig", commands=("currents",), objective={"flux_density": mT(1.0)}, owner=OWNER)
        dtype = jnp.zeros(()).dtype
        wanted = jnp.asarray(WANTED, dtype=dtype)

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            del state
            point, mount = reads["pose"][:3], reads["frames"][ROTOR]
            commands, report = solve({"flux_density": wanted}, {"point": point, "mount_pose": mount})
            return ({"point": point, "mount": mount, "made": report.made["flux_density"]},
                    {"currents": commands["currents"]})

        zero = jnp.zeros((3,), dtype=dtype)
        start = {"point": zero, "mount": jnp.zeros((7,), dtype=dtype), "made": zero}
        return ControlLaw(state=start, held={"currents": zero}, step=step)


def exposure_sphere(name: str = "exposure_sphere", *, observe: Any = None, currents=CURRENTS, latency: bool = True,
                    laws: tuple[Any, ...] = (Magnetic.Force,)) -> Experiment:
    """Navion_2 at the world's origin and the still sphere at its centre, the field's ``laws`` on it, the currents
    constant through a port delaying them a step where ``latency`` (the field off at step 0, on from step 1), and
    ``observe``, an observation to add."""
    exp = scene_experiment(name)
    port = CommandPort(latency=ms(1), reason="the field switched on at step 1") if latency else None
    exp.add_apparatus(Apparatus("coils", fields={"magnetic": coils(mount=None, at=(0.0, 0.0, 0.0),
                                                                   orientation=(1.0, 0.0, 0.0, 0.0))},
                                commands={"currents": Constant(A(currents))},
                                ports={"currents": port} if port is not None else {}))
    still_sphere(exp)
    for law in laws:
        exp.attach(law(), body="sphere", field="coils.magnetic")
    if observe is not None:
        exp.observe(observe)
    return exp


__all__ = ["CURRENTS", "LIMIT", "OWNER", "ROTOR", "ROTOR_AT", "SPEED", "TURN", "WANTED", "MountedFieldControl",
           "arm_mounted", "coils", "exposure_sphere", "mounted_rig", "still_sphere"]
