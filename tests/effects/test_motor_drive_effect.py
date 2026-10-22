"""M2 — rotating magnetic drive as an effect + SWIM_MODE (free | held).

Design finding (verified): the rotating motor-magnet drive needs **no new effect
family**. ``MagneticModel.PointDipole(ext_magnet, response, pose_source=motor)``
already composes MotorNode → PermanentMagnetNode → PermanentMagnetResponseNode into
**one** attachable rotating-dipole drive — the motor spins the external magnet
(``rotor_pose_world`` → ``magnet_pose_world``), the field is evaluated at the body,
and the response delivers ``magnetic_torque``/``magnetic_force`` to the body. So per
the plan's own escape hatch we reuse ``PointDipole`` rather than write a redundant
``MotorDrive`` (effects-first = reuse the registered effect).

``SWIM_MODE`` is the **body-node** choice, set in ``Experiment.set_body``:
  * ``free``  → ``RigidBodyNode(use_inertial=True)`` integrates net torque/force and
    rotates under the field (the natural swim mode);
  * ``held``  → ``RigidBodyNode(kinematic_mode=True)`` with prescribed V/Ω (zero ⇒
    fixed pose), the mode used to read a reaction drag in a flow (the hydro readout
    is M5; here we only pin that held stays put while the drive is attached).

This is also the first **effects-first** ``Experiment.attach`` composition in the
gate work (M0/M1 used the raw harness): it proves a magnetic-only body builds and
runs through ``Experiment`` → ``build`` → ``step``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mime.effects.experiment import Experiment, Body, Medium
from mime.effects.magnetic import MagneticModel
from mime.nodes.actuation.motor import MotorNode
from mime.nodes.actuation.permanent_magnet import PermanentMagnetNode
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode
from mime.nodes.robot.rigid_body import RigidBodyNode

pytestmark = pytest.mark.x64

_DT = 1e-3
_OMEGA = 2 * np.pi * 3.0                                # 3 Hz (de Jongh RPM)
_STANDOFF = 0.1                                         # body 10 cm below magnet (z/R≈5.7)
# de Jongh magnet + onboard-moment params (ar4 params.py / dejongh.py).
_DIPOLE, _R_MAG, _L_MAG = 18.89, 17.5e-3, 20e-3
_N_MAG, _M_SINGLE = 2, 8.4e-4


def _drive(pose_source=True):
    motor = MotorNode("motor", _DT, inertia_kg_m2=1e-5, kt_n_m_per_a=0.05, r_ohm=1.0,
                      l_henry=1e-3, damping_n_m_s=1e-4, axis_in_parent_frame=(0, 0, 1),
                      tool_offset_in_rotor_frame=(0, 0, 0, 1, 0, 0, 0))
    magnet = PermanentMagnetNode("ext_magnet", _DT, dipole_moment_a_m2=_DIPOLE,
                                 magnetization_axis_in_body=(1, 0, 0),
                                 magnet_radius_m=_R_MAG, magnet_length_m=_L_MAG,
                                 field_model="point_dipole", earth_field_world_t=(0, 0, 0))
    response = PermanentMagnetResponseNode("magnet", _DT, n_magnets=_N_MAG,
                                           m_single=_M_SINGLE, moment_axis=(1, 0, 0))
    return MagneticModel.PointDipole(magnet, response,
                                     pose_source=motor if pose_source else None)


def _body(swim_mode):
    common = dict(semi_major_axis_m=2e-3, semi_minor_axis_m=1e-3,
                  density_kg_m3=1100.0, fluid_viscosity_pa_s=1e-3,
                  fluid_density_kg_m3=1000.0)
    if swim_mode == "free":
        node = RigidBodyNode("body", _DT, use_inertial=True, I_eff=1e-9, m_eff=1e-5,
                             **common)
    else:                                               # held: prescribed kinematics
        node = RigidBodyNode("body", _DT, kinematic_mode=True, **common)
    return Body("body", node=node, properties={"magnetic": {}})


def _experiment(swim_mode):
    exp = Experiment(name=f"motor_drive_{swim_mode}", mime_version_min="0.1.0")
    exp.set_body(_body(swim_mode))
    exp.set_medium(Medium({"density": 1000.0, "viscosity": 1e-3}))
    exp.attach(_drive())
    exp.add_external_input("motor", "commanded_velocity", ())
    if swim_mode == "held":
        exp.add_external_input("body", "external_angular_velocity", (3,))
        exp.add_external_input("body", "external_velocity", (3,))
    gm, _ = exp.build()
    # place the body at a standoff below the magnet so the dipole field is finite.
    st = gm.get_node_state("body")
    st = {**st, "position": jnp.array([0.0, 0.0, -_STANDOFF])}
    gm.set_node_state("body", st)
    return gm


def _quat_dev(q):
    """Angle (rad) of a unit quaternion away from identity (1,0,0,0)."""
    return 2.0 * float(np.arccos(np.clip(abs(np.asarray(q)[0]), 0.0, 1.0)))


def test_pointdipole_motor_drive_attaches_and_spins_free_body():
    gm = _experiment("free")
    ext = {"motor": {"commanded_velocity": jnp.asarray(_OMEGA)}}
    q0 = np.asarray(gm.get_node_state("body")["orientation"]).copy()
    for _ in range(250):
        gm.step(ext)
    bod = gm.get_node_state("body")
    dev = _quat_dev(bod["orientation"])
    wmag = float(np.linalg.norm(np.asarray(bod["angular_velocity"])))
    print(f"\n[M2 free] |q dev from identity|={dev:.3e} rad, "
          f"|angular_velocity|={wmag:.3e} rad/s")
    # the rotating field drove a magnetic torque → the free body rotated.
    assert np.all(np.isfinite(bod["orientation"]))
    assert dev > 1e-3                                    # orientation moved off identity
    assert wmag > 1e-3                                   # body picked up angular velocity
    assert not np.allclose(np.asarray(bod["orientation"]), q0)


def test_held_body_stays_put_with_drive_attached():
    gm = _experiment("held")
    ext = {"motor": {"commanded_velocity": jnp.asarray(_OMEGA)},
           "body": {"external_angular_velocity": jnp.zeros(3),
                    "external_velocity": jnp.zeros(3)}}
    for _ in range(100):
        gm.step(ext)
    bod = gm.get_node_state("body")
    dev = _quat_dev(bod["orientation"])
    pos = np.asarray(bod["position"])
    print(f"\n[M2 held] |q dev|={dev:.3e} rad, position={pos}")
    # held (prescribed zero V/Ω): pose is fixed despite the drive being attached.
    assert dev < 1e-6
    assert np.allclose(pos, [0.0, 0.0, -_STANDOFF], atol=1e-9)
