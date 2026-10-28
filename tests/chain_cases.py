"""Experiments with an apparatus chain, for the round trips and the chain's oracles: the AR4 arm carrying a motor whose
rotor carries a magnet, with its home pose and the magnet's moment from the 0.2 line's A0 rig (de Jongh's 35 x 20 mm
N45 magnet), above a body held in a tube as script A's is; and the lab fixture's gantry, a chain segment with no URDF,
carrying a magnet."""

import importlib
from pathlib import Path

from mime.drives import Constant
from mime.dynamics import Motor, RigidInertial, RobotArm
from mime.experiment import Experiment
from mime.fields import PointDipoleField
from mime.interactions import Contact, Magnetic
from mime.scene import X, Z, Scene, SolidCylinder, Tube, asset
from mime.things import Apparatus, Body, Part, PermanentMagnet, State, Structure
from mime.units import A_m2, kg, kg_m2, m, m_per_s, mm, ms, rad, rad_per_s, standard_gravity

DESCRIPTIONS = Path(__file__).parent / "descriptions"
AR4 = DESCRIPTIONS / "ar4.urdf"
#: The AR4's base, and the joint positions that put its flange about 15 cm above the world's origin (A0's home pose).
AR4_BASE = (-0.05, 0.328, -0.43)
AR4_HOME = (-0.02761, 0.23896, -0.77475, 1.55611, 1.54628, 0.0)
#: De Jongh's rotor magnet: 18.89 A m^2, a cylinder 35 mm across and 20 mm long.
MAGNET_MOMENT, MAGNET_RADIUS, MAGNET_LENGTH = 18.89, 17.5e-3, 20e-3
#: A millimetre-scale body's mass, inertia and moment, round numbers of its order.
BODY_MASS, BODY_INERTIA, BODY_MOMENT = 8.5e-5, (1.0e-10, 1.2e-10, 0.8e-10), (0.0, 1.7e-3, 0.4e-3)


def magnet_part(mount: str | None, at=None) -> Part:
    return Part(SolidCylinder(radius=m(MAGNET_RADIUS), length=m(MAGNET_LENGTH), axis=(1.0, 0.0, 0.0)), mount=mount,
                at=at)


def experiment(name: str) -> Experiment:
    return Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=ms(1))


def add_body(exp: Experiment, *, position=mm((0.0, 0.0, -1.0)), magnet_centre=m((0.0, 0.0, 0.0)),
             velocity=m_per_s((0.0, 0.0, -1e-3))) -> None:
    """A millimetre body in script A's tube (along x, a quarter inch across), held in it by projection: it starts
    0.2 um above the floor, moving down into it at 1 mm/s, so that the projection holds it from the first step, as
    script A's body rides its floor."""
    exp.add_structure(Structure("tube", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(3.175), wall=mm(1.6),
                                                      length=mm(100.0))))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=position, orientation=(1.0, 0.0, 0.0, 0.0), velocity=velocity,
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.declare("umr.mass", kg(BODY_MASS), reason="a millimetre body, round numbers of its order")
    exp.declare("umr.inertia", kg_m2(BODY_INERTIA), reason="a millimetre body, round numbers of its order")
    exp.declare("umr.magnetic_moment", A_m2(BODY_MOMENT), reason="a millimetre body's magnet, its order")
    exp.declare("umr.magnet_centre", magnet_centre, reason="its magnet at its centre of mass")
    exp.declare("umr.extent", mm(2.0748), reason="script A's body")
    exp.attach(Contact.Projection(margin=mm(0.1)), body="umr", surface="tube.inner_surface")


def rpm(*, joints=AR4_HOME, speed=6.0, motor_at=None, motor_orientation=None, magnet_at=None, description=AR4,
        acknowledged=(), servo=None, prescribed="velocity") -> Apparatus:
    """The AR4 at its home pose carrying a motor on its flange (``link_6``), turning about the flange's z at ``speed``
    rad/s (``prescribed="angle"``: to ``speed`` rad), the magnet on the rotor, magnetised along the rotor's x. The
    motor is placed on the flange by ``motor_at`` and ``motor_orientation``, the magnet on the rotor by
    ``magnet_at``; ``servo`` is the arm's time constant."""
    return Apparatus(
        "rpm",
        chain={"arm": RobotArm.Kinematic(description=asset(description), initial_positions=rad(AR4_HOME),
                                         at=m(AR4_BASE), acknowledged=acknowledged, servo=servo),
               "motor": Motor.Kinematic(axis=Z, prescribed=prescribed, initial_angle=rad(0.0), mount="link_6",
                                        at=motor_at, orientation=motor_orientation)},
        parts={"rotor_magnet": magnet_part("motor.rotor", at=magnet_at)},
        magnets={"rotor_magnet": PermanentMagnet(direction=(1.0, 0.0, 0.0))},
        fields={"magnetic": PointDipoleField(part="rotor_magnet")},
        commands={"arm.joint_positions": Constant(rad(joints)),
                  **({"motor.velocity": Constant(rad_per_s(speed))} if prescribed == "velocity"
                     else {"motor.angle": Constant(rad(speed))})})


def kinematic_rpm(name: str = "kinematic_rpm", **options) -> Experiment:
    """A0's apparatus, kinematic: the AR4 carrying the motor and its magnet, the field's torque and force on a body
    1 mm below the origin."""
    exp = experiment(name)
    exp.add_apparatus(rpm(**options))
    exp.declare("rpm.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's 35 x 20 mm N45 rotor magnet")
    add_body(exp)
    exp.attach(Magnetic.Torque(), body="umr", field="rpm.magnetic")
    exp.attach(Magnetic.Force(), body="umr", field="rpm.magnetic")
    return exp


def gantry_magnet(name: str = "gantry_magnet", *, magnet_at=None) -> Experiment:
    """The lab fixture's gantry, a segment with no URDF, 10 cm above the origin, its carriage at 2 cm along x with a
    magnet on it: the field's force on a body below, its magnet half a millimetre off its centre of mass, so the force
    turns it too. The mount and everything downstream as on MIME's arm."""
    gantry = importlib.import_module("mime_lab_fixture").Gantry
    exp = experiment(name)
    exp.add_apparatus(Apparatus(
        "stage", chain={"gantry": gantry(travel=X, at=m((0.0, 0.0, 0.1)))},
        parts={"magnet": magnet_part("gantry.carriage", at=magnet_at)},
        magnets={"magnet": PermanentMagnet(direction=(0.0, 0.0, 1.0))},
        fields={"magnetic": PointDipoleField(part="magnet")},
        commands={"gantry.position": Constant(m(0.02))}))
    exp.declare("stage.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's rotor magnet, on a stage")
    add_body(exp, magnet_centre=mm((0.5, 0.0, 0.0)))
    exp.attach(Magnetic.Force(), body="umr", field="stage.magnetic")
    return exp


def fixed_magnet(name: str = "fixed_magnet", *, at=m((0.0, 0.0, 0.1)), law=None, body_position=None) -> Experiment:
    """A magnet fixed in the world, with no chain, above a body: the smallest apparatus a dipole needs."""
    exp = experiment(name)
    exp.add_apparatus(Apparatus("stand", parts={"magnet": magnet_part(None, at=at)},
                                magnets={"magnet": PermanentMagnet(direction=(0.0, 0.0, 1.0))},
                                fields={"magnetic": PointDipoleField(part="magnet")}))
    exp.declare("stand.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's rotor magnet, on a stand")
    add_body(exp, **({} if body_position is None else {"position": body_position}))
    exp.attach(law if law is not None else Magnetic.Force(), body="umr", field="stand.magnetic")
    return exp
