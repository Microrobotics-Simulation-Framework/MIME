"""Tutorial A0: a permanent magnet on an AR4 arm, driving the same FL-9 screw in the same tube, in a closed loop. The
arm's controller reads the body's pose and moves the arm to keep the magnet above it. The user guide
(``docs/guide/tutorial_a0.md``) includes this file by its regions and prints its output, which
``tests/docs/test_tutorials.py`` and ``mime docs check`` regenerate.

The arm's description is the AR4's URDF in ``examples/descriptions/ar4/`` (MIT, beside its meshes' manifest); a lab
uses its own.
"""

import math
from pathlib import Path

import numpy as np

from script_a import FL9, REFERENCE, CentreContact

from mime import Apparatus, Body, Experiment, Medium, PermanentMagnet, State, Structure
from mime.control import ArmTracking
from mime.data import asset
from mime.drives import Constant
from mime.dynamics import Arm, Motor, RigidInertial
from mime.fields import PointDipole
from mime.interactions import Drag, Gravity, Lubrication, Magnetic
from mime.observations import SwimSpeed
from mime.scene import Cylinder, Scene, Tube, X, Z
from mime.scene import asset as description
from mime.things import Part
from mime.units import (
    A_m2,
    Hz,
    kg,
    kg_m2,
    kg_per_m3,
    m,
    m3,
    m_per_s,
    m_per_s2,
    mm,
    mPa_s,
    ms,
    rad,
    rad_per_s,
    s,
)

AR4 = Path(__file__).resolve().parent / "descriptions" / "ar4" / "ar4.urdf"
#: Where the arm's base stands, and the joint positions that put its flange about 15 cm above the world's origin.
BASE = (-0.05, 0.328, -0.43)
HOME = (-0.02761, 0.23896, -0.77475, 1.55611, 1.54628, 0.0)
FREQUENCY = Hz(3.0)
SHORT = s(1.0)


# region: apparatus
def rotor_magnet_on_an_arm() -> Apparatus:
    """The AR4 carrying a motor on its last link; the rotor carries de Jongh's 35 mm by 20 mm magnet."""
    return Apparatus(
        "rpm",
        chain={"arm": Arm.Kinematic(description=description(AR4), initial_positions=rad(HOME), at=m(BASE)),
               "motor": Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0), mount="link_6")},
        parts={"rotor_magnet": Part(Cylinder(radius=mm(17.5), length=mm(20.0), axis=(1.0, 0.0, 0.0)),
                                    mount="motor.rotor")},
        magnets={"rotor_magnet": PermanentMagnet(direction=(1.0, 0.0, 0.0))},
        fields={"magnetic": PointDipole(part="rotor_magnet")},
        # The arm's joint positions are given no drive: the controller writes them.
        commands={"motor.velocity": Constant(rad_per_s(2 * math.pi * FREQUENCY.value))})
# endregion apparatus


# region: experiment
def a0() -> Experiment:
    experiment = Experiment("tutorial_a0", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5),
                            duration=SHORT)
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                    length=mm(100.0))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen))
    experiment.add_apparatus(rotor_magnet_on_an_arm())
    experiment.declare("rpm.magnetic_moment", A_m2(18.89), reason="de Jongh's 35 x 20 mm N45 rotor magnet")
    umr = experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(FL9)}))
    for path, value in (("umr.mass", kg(8.491116020725203e-05)),
                        ("umr.inertia", kg_m2((1.0331989974018431e-10,) * 3)),
                        ("umr.buoyant_volume", m3(6.022068099805108e-08)),
                        ("umr.magnetic_moment", A_m2((0.0, 2 * 8.4e-4, 0.0))),
                        ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0))), ("umr.magnet_centre", m((0.0, 0.0, 0.0))),
                        ("water.density", kg_per_m3(1000.0)), ("water.viscosity", mPa_s(1.0))):
        experiment.declare(path, value, reason=REFERENCE)
    experiment.attach(Drag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=mm(0.15))),
                      body=umr, medium=water)
    experiment.attach(Magnetic.Torque(), body=umr, field="rpm.magnetic")
    experiment.attach(Magnetic.Force(), body=umr, field="rpm.magnetic")
    experiment.attach(Gravity(), body=umr, medium=water)
    experiment.attach(CentreContact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    experiment.observe(SwimSpeed(umr, along=tube.axis))
    return experiment
# endregion experiment


# region: controller
def close_the_loop(experiment: Experiment) -> Experiment:
    """The controller reads the body's pose and twist and the arm's joints, and writes the arm's joint positions
    every step."""
    tracking = ArmTracking(description=description(AR4), end_effector="link_6", home=rad(HOME), axis=X, up=Z, at=m(BASE))
    experiment.set_controller(
        tracking, reads={"body_pose": "umr.pose", "body_twist": "umr.twist", "joint_positions": "rpm.arm.joint_positions"},
        writes={"joint_positions": "rpm.arm.joint_positions"}, mode="graph", rate=ms(0.5))
    return experiment
# endregion controller


# region: run
built = close_the_loop(a0()).build()
print(built.report.controller[0].split(" (")[0])
history = built.run(SHORT, history=True).history
# The rotor's position is the motor's second frame; the body's is its pose's first three numbers.
magnet = np.asarray(history["rpm.motor"]["frame_poses"])[:, 1, :3]
body = np.asarray(history["umr"]["pose"])[:, :3]
distance = np.linalg.norm(magnet - body, axis=1)
print(f"after a short run of {SHORT.value:g} s, the magnet is {distance[-1] * 1e3:.1f} mm from the body")
# endregion run
