"""Script A0 on its dynamic rungs, through the public API: :mod:`script_a0`'s experiment with the arm and the motor
dynamic, closed by one controller (:class:`a0_loops.A0Loops`) that writes the arm's joint torques by the tag's
computed-torque loop on ``arm_tracking``'s targets and the rotor's torque by the tag's PI velocity loop, every step.

Two arms of it. **As declared** (``density=None``): the rotor's inertia about its axis is the motor's declared 1e-5
kg m^2 (the 0.2 line's), its magnet has no mass, nothing is grafted on the arm, and the rotor's spin, its drive's and
bearing's reaction and its gyroscopic torque load the arm's flange, which the 0.2 line's did not. **With the magnet's
mass** (``density`` given, sintered NdFeB's typical 7.5 g/cm^3, an inference): its moment about the spin axis adds to
the rotor's (2.59e-5 kg m^2 in all), its mass and mean inertia across the axis are welded on ``link_6``, and its half
difference across the axis turns with it; ``magnet_axis`` is its geometric axis in the rotor's frame (A0's along x,
the moment's, spun about a diameter; ``(0, 0, 1)``, spun about its own axis, the symmetric case). The second arm is
reported as data and never compared with the 0.2 reference, which carried neither."""

from __future__ import annotations

import math
from typing import Any

from a0_loops import A0Loops
from chain_cases import AR4, AR4_BASE, AR4_HOME, MAGNET_LENGTH, MAGNET_MOMENT, MAGNET_RADIUS, ROTOR_DAMPING, \
    ROTOR_INERTIA
from script_a import EPSILON, INERTIA, MASS, MOMENT, REASON, REFERENCE_START, VISCOSITY, VOLUME, WATER, fl9

from mime import Body, Experiment, Medium, State, Structure
from mime.data import asset
from mime.dynamics import Arm, Motor, RigidInertial
from mime.fields import PointDipole
from mime.interactions import Contact, Drag, Gravity, Lubrication, Magnetic
from mime.observations import SwimSpeed
from mime.scene import Cylinder, Scene, Tube, X, Z
from mime.scene import asset as file_asset
from mime.things import Apparatus, Part, PermanentMagnet
from mime.units import A_m2, Hz, N_m_s, kg_m2, kg_per_m3, m, m_per_s, m_per_s2, mm, ms, rad, rad_per_s

__all__ = ["DYNAMIC_READS", "DYNAMIC_WRITES", "a0_dynamic", "loops"]

#: What A0's controller reads on the dynamic rungs: ``arm_tracking``'s reads, the arm's joint velocities, and the
#: motor's frames' angular velocities and poses; and what it writes.
DYNAMIC_READS = {"body_pose": "umr.pose", "body_twist": "umr.twist", "joint_positions": "rpm.arm.joint_positions",
                 "joint_velocities": "rpm.arm.joint_velocities",
                 "rotor_rates": "rpm.motor.frame_angular_velocities", "rotor_poses": "rpm.motor.frame_poses"}
DYNAMIC_WRITES = {"joint_torques": "rpm.arm.joint_torques", "torque": "rpm.motor.torque"}


def loops(frequency: float, **settings: Any) -> A0Loops:
    """A0's controller on its dynamic rungs with the tag's settings: ``arm_tracking``'s A0 settings, ``Kp`` 1e4 and
    ``Kd`` 200, the rotor held at ``2 pi frequency`` by ``Kp`` 1e-3 N m s and ``Ki`` 1e-2 N m. ``settings`` override
    them (the controls' planted gains)."""
    chosen: dict[str, Any] = dict(description=file_asset(AR4), end_effector="link_6", home=rad(AR4_HOME), axis=X, up=Z,
                                  at=m(AR4_BASE), arm="rpm.arm", speed=2.0 * math.pi * frequency)
    chosen.update(settings)
    return A0Loops(**chosen)


def a0_dynamic(name: str = "script_a0_dynamic", *, frequency=Hz(3.0), dt=ms(0.5), mode: Any = "graph", rate=None,
               latency=None, controller=None, density: float | None = None, magnet_axis=(1.0, 0.0, 0.0),
               top_speed: float | None = None, start=REFERENCE_START, rotor_inertia: float = ROTOR_INERTIA,
               initial_angle: float = 0.0, magnet_direction=(1.0, 0.0, 0.0), **settings: Any) -> Experiment:
    """A0 on its dynamic rungs: the arm and the motor dynamic, the rotor's magnet on ``motor.rotor``, closed by
    :func:`loops` every ``rate`` (every step when not given), ``latency`` late. ``density`` (kg/m^3) gives the magnet
    a mass, ``magnet_axis`` its geometric axis in the rotor's frame, ``top_speed`` (rad/s) the motor's, which an
    unequal rotor on a dynamic arm needs. ``controller`` replaces the controller; ``settings`` are its own.
    ``rotor_inertia`` (kg m^2), ``initial_angle`` (rad) and ``magnet_direction`` (in the rotor's frame) are A0's as
    declared unless a control plants a fault in them."""
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=dt)
    tube = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                             length=mm(100.0))))
    water = exp.add_medium(Medium("water", fills=tube.lumen))
    motor: dict[str, Any] = {"top_speed": rad_per_s(top_speed)} if top_speed is not None else {}
    exp.add_apparatus(Apparatus(
        "rpm",
        chain={"arm": Arm.Dynamic(description=file_asset(AR4), initial_positions=rad(AR4_HOME), at=m(AR4_BASE)),
               "motor": Motor.Dynamic(axis=Z, inertia=kg_m2(rotor_inertia), damping=N_m_s(ROTOR_DAMPING),
                                      initial_angle=rad(initial_angle), mount="link_6", **motor)},
        parts={"rotor_magnet": Part(Cylinder(radius=m(MAGNET_RADIUS), length=m(MAGNET_LENGTH), axis=magnet_axis),
                                    mount="motor.rotor")},
        magnets={"rotor_magnet": PermanentMagnet(direction=tuple(magnet_direction))},
        fields={"magnetic": PointDipole(part="rotor_magnet")}))
    exp.declare("rpm.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's 35 x 20 mm N45 rotor magnet, 18.89 A m^2")
    if density is not None:
        exp.declare("rpm.parts.rotor_magnet.density", kg_per_m3(density), reason="sintered NdFeB, its typical value "
                                                                                 "(an inference)")
    umr = exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=start, orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(fl9())}))
    for path, value in (("umr.inertia", INERTIA), ("umr.mass", MASS), ("umr.buoyant_volume", VOLUME),
                        ("umr.magnetic_moment", MOMENT), ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0))),
                        ("umr.magnet_centre", m((0.0, 0.0, 0.0))), ("water.density", WATER),
                        ("water.viscosity", VISCOSITY)):
        exp.declare(path, value, reason=REASON)
    exp.attach(Drag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=EPSILON)),
               body=umr, medium=water)
    exp.attach(Magnetic.Torque(), body=umr, field="rpm.magnetic")
    exp.attach(Magnetic.Force(), body=umr, field="rpm.magnetic")
    exp.attach(Gravity(), body=umr, medium=water)
    exp.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    exp.observe(SwimSpeed(umr, along=tube.axis))
    exp.set_controller(controller if controller is not None else loops(frequency.value, **settings),
                       reads=DYNAMIC_READS, writes=DYNAMIC_WRITES, mode=mode, rate=rate if rate is not None else dt,
                       latency=latency)
    return exp
