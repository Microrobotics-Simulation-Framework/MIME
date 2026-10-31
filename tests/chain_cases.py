"""Experiments with an apparatus chain, for the round trips and the chain's oracles: the AR4 arm carrying a motor whose
rotor carries a magnet, with its home pose and the magnet's moment from the 0.2 line's A0 rig (de Jongh's 35 x 20 mm
N45 magnet), above a body held in a tube as script A's is; and the lab fixture's gantry, a chain segment with no URDF,
carrying a magnet; A0's apparatus closed by its controller; and A0's apparatus dynamic, its arm held at its home pose
by the torques that balance its weight there, its motor driven by a torque, the magnet's reaction on the rotor."""

import importlib
import math
import warnings
from functools import cache
from pathlib import Path

import jax
import numpy as np

from mime.control import ArmTracking
from mime.drives import Constant
from mime.dynamics import Arm, Motor, RigidInertial
from mime.core.multibody import gravity_forces, read_urdf
from mime.experiment import Experiment
from mime.fields import PointDipole
from mime.interactions import Contact, Magnetic
from mime.scene import X, Z, Scene, Cylinder, Tube, asset
from mime.things import Apparatus, Body, Part, PermanentMagnet, State, Structure
from mime.units import A_m2, N_m, N_m_s, kg, kg_m2, kg_per_m3, m, m_per_s, m_per_s2, mm, ms, rad, rad_per_s, \
    standard_gravity

DESCRIPTIONS = Path(__file__).parent / "descriptions"
AR4 = DESCRIPTIONS / "ar4.urdf"
C_ARM, SPIN_JOINT = DESCRIPTIONS / "c_arm.urdf", DESCRIPTIONS / "spin_joint.urdf"
#: The AR4's base, and the joint positions that put its flange about 15 cm above the world's origin (A0's home pose).
AR4_BASE = (-0.05, 0.328, -0.43)
AR4_HOME = (-0.02761, 0.23896, -0.77475, 1.55611, 1.54628, 0.0)
#: De Jongh's rotor magnet: 18.89 A m^2, a cylinder 35 mm across and 20 mm long.
MAGNET_MOMENT, MAGNET_RADIUS, MAGNET_LENGTH = 18.89, 17.5e-3, 20e-3
#: A millimetre-scale body's mass, inertia and moment, round numbers of its order.
BODY_MASS, BODY_INERTIA, BODY_MOMENT = 8.5e-5, (1.0e-10, 1.2e-10, 0.8e-10), (0.0, 1.7e-3, 0.4e-3)


def magnet_part(mount: str | None, at=None) -> Part:
    return Part(Cylinder(radius=m(MAGNET_RADIUS), length=m(MAGNET_LENGTH), axis=(1.0, 0.0, 0.0)), mount=mount,
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
        acknowledged=(), servo=None, prescribed="velocity", controlled=(), arm_at=AR4_BASE) -> Apparatus:
    """The AR4 at its home pose carrying a motor on its flange (``link_6``), turning about the flange's z at ``speed``
    rad/s (``prescribed="angle"``: to ``speed`` rad), the magnet on the rotor, magnetised along the rotor's x. The
    motor is placed on the flange by ``motor_at`` and ``motor_orientation``, the magnet on the rotor by
    ``magnet_at``; ``servo`` is the arm's time constant; ``arm_at`` is where the arm's base is placed. The commands
    named in ``controlled`` take no drive, for a controller to write."""
    return Apparatus(
        "rpm",
        chain={"arm": Arm.Kinematic(description=asset(description), initial_positions=rad(AR4_HOME),
                                         at=m(arm_at), acknowledged=acknowledged, servo=servo),
               "motor": Motor.Kinematic(axis=Z, prescribed=prescribed, initial_angle=rad(0.0), mount="link_6",
                                        at=motor_at, orientation=motor_orientation)},
        parts={"rotor_magnet": magnet_part("motor.rotor", at=magnet_at)},
        magnets={"rotor_magnet": PermanentMagnet(direction=(1.0, 0.0, 0.0))},
        fields={"magnetic": PointDipole(part="rotor_magnet")},
        commands={name: drive for name, drive in {
            "arm.joint_positions": Constant(rad(joints)),
            **({"motor.velocity": Constant(rad_per_s(speed))} if prescribed == "velocity"
               else {"motor.angle": Constant(rad(speed))})}.items() if name not in controlled})


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
        fields={"magnetic": PointDipole(part="magnet")},
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
                                fields={"magnetic": PointDipole(part="magnet")}))
    exp.declare("stand.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's rotor magnet, on a stand")
    add_body(exp, **({} if body_position is None else {"position": body_position}))
    exp.attach(law if law is not None else Magnetic.Force(), body="umr", field="stand.magnetic")
    return exp


#: What A0's controller reads: the body's pose and twist, and the arm's joint positions; and what it writes.
TRACKING_READS = {"body_pose": "umr.pose", "body_twist": "umr.twist", "joint_positions": "rpm.arm.joint_positions"}
TRACKING_WRITES = {"joint_positions": "rpm.arm.joint_positions"}


def tracking(**settings) -> ArmTracking:
    """A0's controller for the AR4 of :func:`rpm`, with the 0.2 line's A0 settings: its flange 15 cm above the body
    along +z, following it along x within 5 cm either side."""
    return ArmTracking(description=asset(AR4), end_effector="link_6", home=rad(AR4_HOME), axis=X, up=Z,
                       at=m(AR4_BASE), **settings)


def tracked_rpm(name: str = "tracked_rpm", *, rate=ms(2), latency=ms(1), controller=None, reads=None, writes=None,
                mode="callback", controlled=("arm.joint_positions",), **options) -> Experiment:
    """A0 closed by its controller, a callback: :func:`kinematic_rpm` with the arm's joint positions written by
    :func:`tracking` every ``rate``, ``latency`` late, and given no drive (``controlled``)."""
    exp = experiment(name)
    exp.add_apparatus(rpm(controlled=controlled, **options))
    exp.declare("rpm.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's 35 x 20 mm N45 rotor magnet")
    add_body(exp)
    exp.attach(Magnetic.Torque(), body="umr", field="rpm.magnetic")
    exp.attach(Magnetic.Force(), body="umr", field="rpm.magnetic")
    exp.set_controller(controller if controller is not None else tracking(),
                       reads=TRACKING_READS if reads is None else reads,
                       writes=TRACKING_WRITES if writes is None else writes, mode=mode, rate=rate, latency=latency)
    return exp


#: A0's rotor, as the 0.2 line's rig declared it: its inertia about its axis and its bearing's damping, and a drive
#: torque of a millinewton metre.
ROTOR_INERTIA, ROTOR_DAMPING, ROTOR_TORQUE = 1e-5, 1e-4, 1e-3
#: Where the dynamic motor sits on the flange: 5 cm out along it, as the 0.2 line's A0 mounted it, and turned 0.3 rad
#: about the flange's x, so that its stator's lever on the flange and the turn of its axis are both exercised.
MOTOR_OFFSET, MOTOR_TURN = (0.0, 0.0, 0.05), (math.cos(0.15), math.sin(0.15), 0.0, 0.0)
GRAVITY = (0.0, 0.0, -9.80665)


@cache
def holding_torques(joints=AR4_HOME) -> tuple[float, ...]:
    """The AR4's joint torques that balance its weight at ``joints``, its base unturned so that its root's frame is
    the world's: the joints' share of gravity, from the rigid-body library, in float64."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = read_urdf(AR4).tree
    with jax.enable_x64(True):
        return tuple(float(t) for t in np.asarray(gravity_forces(tree, np.asarray(joints), np.asarray(GRAVITY))))


def friction_arm(name: str = "friction_arm") -> Experiment:
    """A dynamic arm with Coulomb friction: ``descriptions/friction_joint.urdf``'s rod on one joint, driven by a torque
    below its friction, beside the body. A0's AR4 declares no friction, so its dynamic arm never reaches the friction's
    switches or the sweeps' start; this one does."""
    exp = experiment(name)
    exp.add_apparatus(Apparatus("rod", chain={"arm": Arm.Dynamic(description=asset(DESCRIPTIONS / "friction_joint.urdf"),
                                                                 initial_positions=rad((0.0,)))},
                                commands={"arm.joint_torques": Constant(N_m((0.2,)))}))
    add_body(exp)
    return exp


def dynamic_rpm(name: str = "dynamic_rpm", *, mount: str | None = None, motor: bool = True,
                torque: float = ROTOR_TORQUE, laws=("torque", "force")) -> Experiment:
    """A0's apparatus on its dynamic rungs: the AR4 starting at rest at its home pose, held there at the start by the
    torques that balance its weight, carrying a motor on its flange (``motor=False``: none), placed at ``MOTOR_OFFSET``
    and turned by ``MOTOR_TURN``, whose rotor is driven by
    ``torque`` against its inertia and damping, and the magnet on ``mount``; the body below, under the field's torque
    and force (``laws``), each delivering its reaction to the frame the magnet is mounted on (by default the rotor, or
    the flange with no motor)."""
    exp = experiment(name)
    chain = {"arm": Arm.Dynamic(description=asset(AR4), initial_positions=rad(AR4_HOME), at=m(AR4_BASE))}
    commands = {"arm.joint_torques": Constant(N_m(holding_torques()))}
    if motor:
        chain["motor"] = Motor.Dynamic(axis=Z, inertia=kg_m2(ROTOR_INERTIA), damping=N_m_s(ROTOR_DAMPING),
                                       initial_angle=rad(0.0), mount="link_6", at=m(MOTOR_OFFSET),
                                       orientation=MOTOR_TURN)
        commands["motor.torque"] = Constant(N_m(torque))
    mount = mount or ("motor.rotor" if motor else "arm.link_6")
    exp.add_apparatus(Apparatus("rpm", chain=chain, parts={"rotor_magnet": magnet_part(mount)},
                                magnets={"rotor_magnet": PermanentMagnet(direction=(1.0, 0.0, 0.0))},
                                fields={"magnetic": PointDipole(part="rotor_magnet")}, commands=commands))
    exp.declare("rpm.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's 35 x 20 mm N45 rotor magnet")
    add_body(exp)
    for law in laws:
        exp.attach(Magnetic.Torque() if law == "torque" else Magnetic.Force(), body="umr", field="rpm.magnetic")
    return exp


#: A0's magnet's density: sintered NdFeB's typical 7.5 g/cm^3 (inference: the record MIME packages holds its remanence
#: alone). Through its solid, 0.144 kg, 1.59e-5 kg m^2 about the rotor's z, its spin axis (a diameter), and 2.21e-5
#: and 1.59e-5 across it: the accepted case of a rotor unequal across its axis.
MAGNET_DENSITY = 7500.0
#: The most A0's rotor spins at: the 0.2 line's drive at 10 Hz, the highest it ran (inference from its field
#: frequencies, 3 and 10 Hz).
TOP_SPEED = 2.0 * math.pi * 10.0


def massive_rpm(name: str = "massive_rpm", *, arm: str = "dynamic", magnet_axis=(1.0, 0.0, 0.0), magnet_at=None,
                density: float | None = MAGNET_DENSITY, top_speed: float | None = TOP_SPEED,
                extra: bool = False, solid=None, motor: str = "dynamic") -> Experiment:
    """A0's apparatus with its masses mounted: :func:`dynamic_rpm`'s arm (or its kinematic rung, ``arm="kinematic"``)
    and motor, the magnet on the rotor of ``density`` (``None``: no mass), its axis ``magnet_axis`` in the rotor's
    frame (A0's along x, across the spin axis z), at ``magnet_at``, or ``solid`` in its place; the motor's
    ``top_speed``; ``motor="kinematic"``, the motor on its kinematic rung at A0's speed. With ``extra``, a camera of
    120 g on link 3 and a stator part of 80 g on the motor's base besides."""
    exp = experiment(name)
    segment = Arm.Dynamic if arm == "dynamic" else Arm.Kinematic
    chain = {"arm": segment(description=asset(AR4), initial_positions=rad(AR4_HOME), at=m(AR4_BASE)),
             "motor": Motor.Dynamic(axis=Z, inertia=kg_m2(ROTOR_INERTIA), damping=N_m_s(ROTOR_DAMPING),
                                    initial_angle=rad(0.0), mount="link_6", at=m(MOTOR_OFFSET), orientation=MOTOR_TURN,
                                    **({"top_speed": rad_per_s(top_speed)} if top_speed is not None else {}))
             if motor == "dynamic" else Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0),
                                                        mount="link_6", at=m(MOTOR_OFFSET), orientation=MOTOR_TURN)}
    commands = {"arm.joint_torques": Constant(N_m(holding_torques()))} if arm == "dynamic" else \
        {"arm.joint_positions": Constant(rad(AR4_HOME))}
    commands.update({"motor.torque": Constant(N_m(ROTOR_TORQUE))} if motor == "dynamic" else
                    {"motor.velocity": Constant(rad_per_s(6.0))})
    held = solid if solid is not None else Cylinder(radius=m(MAGNET_RADIUS), length=m(MAGNET_LENGTH), axis=magnet_axis)
    parts = {"rotor_magnet": Part(held, mount="motor.rotor", at=magnet_at)}
    if extra:
        parts["camera"] = Part(Cylinder(radius=mm(15.0), length=mm(40.0), axis=(0.0, 1.0, 0.0)), mount="arm.link_3",
                               at=mm((20.0, 10.0, -30.0)))
        parts["stator"] = Part(Cylinder(radius=mm(20.0), length=mm(30.0), axis=(0.0, 0.0, 1.0)), mount="motor.base",
                               at=mm((0.0, 0.0, -20.0)))
    exp.add_apparatus(Apparatus("rpm", chain=chain, parts=parts,
                                magnets={"rotor_magnet": PermanentMagnet(direction=(1.0, 0.0, 0.0))},
                                fields={"magnetic": PointDipole(part="rotor_magnet")}, commands=commands))
    exp.declare("rpm.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's 35 x 20 mm N45 rotor magnet")
    if density is not None:
        exp.declare("rpm.parts.rotor_magnet.density", kg_per_m3(density), reason="sintered NdFeB, its typical value")
    if extra:
        exp.declare("rpm.parts.camera.mass", kg(0.12), reason="a small camera, round numbers of its order")
        exp.declare("rpm.parts.stator.mass", kg(0.08), reason="a motor's stator, round numbers of its order")
    add_body(exp)
    exp.attach(Magnetic.Torque(), body="umr", field="rpm.magnetic")
    exp.attach(Magnetic.Force(), body="umr", field="rpm.magnetic")
    return exp


def spinning_hub(name: str = "spinning_hub", *, inertia: float = ROTOR_INERTIA, torque: float = ROTOR_TORQUE,
                 damping: float = 0.0, initial_velocity: float = 0.0) -> Experiment:
    """``descriptions/spin_joint.urdf``'s hub on its dynamic rung, with no gravity, carrying a dynamic motor whose axis
    is the joint's, driven by ``torque``: the spinner's momentum oracle."""
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    exp.add_apparatus(Apparatus("hub", chain={
        "arm": Arm.Dynamic(description=asset(SPIN_JOINT), initial_positions=rad((0.0,))),
        "motor": Motor.Dynamic(axis=Z, inertia=kg_m2(inertia), damping=N_m_s(damping), initial_angle=rad(0.0),
                               initial_velocity=rad_per_s(initial_velocity), mount="hub")},
        commands={"arm.joint_torques": Constant(N_m((0.0,))), "motor.torque": Constant(N_m(torque))}))
    add_body(exp)
    return exp


def turned_spinner(name: str = "turned_spinner", *, turn_rate: float = 2.0, spin: float = 50.0,
                   inertia: float = ROTOR_INERTIA) -> Experiment:
    """A kinematic motor turning about x at ``turn_rate``, and on its rotor a dynamic motor spinning about z at
    ``spin``, undriven and undamped: the gyroscopic closed form, ``J Omega omega`` on the frame it is mounted on."""
    exp = experiment(name)
    exp.add_apparatus(Apparatus("rig", chain={
        "turn": Motor.Kinematic(axis=X, prescribed="velocity", initial_angle=rad(0.0), at=m((0.0, 0.0, 0.2))),
        "spin": Motor.Dynamic(axis=Z, inertia=kg_m2(inertia), damping=N_m_s(0.0), initial_angle=rad(0.0),
                              initial_velocity=rad_per_s(spin), mount="rotor")},
        commands={"turn.velocity": Constant(rad_per_s(turn_rate)), "spin.torque": Constant(N_m(0.0))}))
    add_body(exp)
    return exp
