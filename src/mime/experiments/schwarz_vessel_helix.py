"""schwarz_vessel_helix — the gate experiment, composed effects-first, in **SI**.

The de Jongh screw inside a flowing vessel, driven by a rotating magnetic dipole,
with the **coupled two-scale** solver (inertial Cartesian+IBM FVM far-field ⊗
confined Liron–Shahar BEM near-field). Proves a usable finite-Re, flow-rate-aware,
confined simulator — the assumptions ar4_helical_drive's strictly-Stokes MLP
surrogate omits.

**Units are physical SI throughout** (metres, Pa·s, Newtons, A·m²), mirroring
``ar4_helical_drive``/``dejongh_new_chain`` — so the real AR4 arm and the metric
controller fit naturally. The only dimensionless object, the confined wall table,
is encapsulated by the **SI-native** BEM (``StokesletFluidNode(length_scale=...)``,
validated to the de Jongh ~3 mm/s swim speed). See ``memory: si-native-confined-bem``.

Composed via the effects-first ``Experiment`` API (``set_body`` + ``set_medium`` +
``.attach`` → ``.build()``): ``HydrodynamicModel.TwoScale`` (FVM flow+wall ⊗ confined
BEM), ``MagneticModel.PointDipole(pose_source=motor)`` (rotating drive — de Jongh RPM
values), ``GravityEffect``, optional ``RobotArmEffect`` (the AR4 arm carrying the
magnet, ar4 values).

``SWIM_MODE``: free (RigidBody integrates → swims) | held (kinematic, reaction drag).
``FLOW_PROFILE``: poiseuille (steady, rate U_MEAN) | womersley (pulsatile).

Caveat: the coupled drag is **differential** (self-wake subtraction is a known
blocker, OFF); absolute drag is not physical. Confined-BEM near-field is the
straight-segment model (cylindrical wall table); a bifurcation apex needs the FVM
far-field geometry. Primary validity Re ≲ 25 (vessel), λ ≤ 0.8.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mime.effects.experiment import Experiment
from mime.effects.body_medium import Body, Medium
from mime.effects.hydrodynamic import HydrodynamicModel
from mime.effects.magnetic import MagneticModel
from mime.effects.mechanical import GravityEffect, RobotArmEffect

from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig
from mime.nodes.environment.fvm.fluid_node import FVMFluidNode
from mime.nodes.environment.fvm.ibm import IBMBody
from mime.nodes.environment.fvm.lifting import (
    make_poiseuille_lift, make_womersley_lift_analytical)
from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
from mime.nodes.environment.stokeslet.interface import create_interface_mesh
from mime.nodes.environment.stokeslet.dejongh_geometry import dejongh_umr_surface
from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table
from mime.nodes.actuation.motor import MotorNode
from mime.nodes.actuation.permanent_magnet import PermanentMagnetNode
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode
from mime.nodes.robot.rigid_body import RigidBodyNode


# ── repo paths (absolute — the runner may launch from any CWD) ───────────────
_REPO = Path(__file__).resolve().parents[3]
_EXP = _REPO / "experiments" / "schwarz_vessel_helix"


# ── defaults (physical SI — de Jongh FL-9 / ar4 values) ──────────────────────
_DEFAULTS: dict[str, Any] = {
    "SWIM_MODE": "free",            # free | held
    "FLOW_PROFILE": "poiseuille",   # poiseuille | womersley

    # fluid
    "MU_PA_S": 1e-3, "RHO_FLUID": 1000.0, "DELTA_RHO": 410.0,   # de Jongh

    # de Jongh FL-9 screw (SI, metres). R_cyl 1.56 mm, L 7.47 mm.
    "NU_FL": 2.33, "R_CYL_UMR_M": 1.56e-3, "L_UMR_M": 7.47e-3,
    "EPS": 0.33, "N_STARTS": 2, "N_THETA": 24, "N_ZETA": 36,

    # confined vessel: 1/4" tube R_ves 3.175 mm ⇒ ratio 2.035 (λ≈0.49).
    "R_VES_M": 3.175e-3,
    "WALL_TABLE": str(_REPO / "data" / "dejongh_benchmark" / "wall_tables"
                      / "wall_R2.035.npz"),

    # FVM far-field
    "CPR": 4,                       # cells per body radius
    "DT": 5e-4,
    "U_MEAN": 3e-3,                 # vessel mean velocity [m/s] (Re≈19, ≲25 valid)
    "U_MEAN_AMP": 1e-3,             # Womersley AC amplitude [m/s]
    "OMEGA_FLOW": 2 * np.pi * 1.2,  # ~heart-rate pulsation

    # magnetic drive (de Jongh RPM — ar4 values)
    "DRIVE_HZ": 3.0,                # motor spin [Hz]
    "MAG_DIPOLE": 18.89, "MAG_R": 17.5e-3, "MAG_L": 20e-3,
    # 0.15 m standoff (the arm's EE target; reachable by the AR4). ~0.56 mT at the
    # screw. Closer → stronger field → over-spin/tumble (the project's step-out
    # physics). With the arm in, the magnet pose comes from the EE, not this value.
    "MAG_STANDOFF_M": 0.15,         # magnet standoff above the screw [m] (no-arm)
    "N_MAGNETS": 2, "M_SINGLE": 8.4e-4, "MOMENT_AXIS": (1.0, 0.0, 0.0),
    "MOTOR_INERTIA": 1e-5, "MOTOR_KT": 0.05, "MOTOR_R": 1.0,
    "MOTOR_L": 1e-3, "MOTOR_DAMPING": 1e-4,

    # body initial orientation: body-z (screw long axis) → world-x (the pipe axis),
    # so the screw lies along the flow and corkscrews *through* the pipe. +90° about y.
    "BODY_INIT_ORIENT": (0.7071068, 0.0, 0.7071068, 0.0),

    # arm (AR4) — ar4's verbatim base + home (the EE holds the magnet above the x-tube,
    # motor axis = EE-z along world-x = the pipe axis). Reusable now that the BEM is
    # frame-aware and the vessel is along x like ar4.
    "INCLUDE_ARM": False,
    "ARM_URDF": str(_EXP / "assets" / "ar4.urdf"),
    "ARM_EE_LINK": "link_6",
    "ARM_BASE_POSE": (-0.05, 0.328, -0.43, 1.0, 0.0, 0.0, 0.0),
    "ARM_HOME_RAD": (-0.02761, 0.23896, -0.77475, 1.55611, 1.54628, 0.0),
}


def _p(params, key):
    return params.get(key, _DEFAULTS[key]) if params else _DEFAULTS[key]


def _screw_mesh(params):
    """de Jongh FL-9 surface mesh in metres, z-centred + axis-aligned."""
    m = dejongh_umr_surface(
        nu=_p(params, "NU_FL"), L_UMR=_p(params, "L_UMR_M") * 1e3,   # generator is mm
        R_cyl=_p(params, "R_CYL_UMR_M") * 1e3, epsilon=_p(params, "EPS"),
        N=_p(params, "N_STARTS"), n_theta=_p(params, "N_THETA"),
        n_zeta=_p(params, "N_ZETA"))
    P = (np.asarray(m.points, np.float64) * 1e-3)        # mm → m
    P -= P.mean(axis=0)
    return dataclasses.replace(m, points=P, weights=np.asarray(m.weights) * 1e-6)


def screw_points(params: dict | None = None) -> jnp.ndarray:
    """Screw reference (body-frame) points [m] — feed to the FVM coupling geometry."""
    return jnp.asarray(_screw_mesh(params or {}).points)


def _far_node(params, body, R_ves, mu, rho):
    """Cartesian FVM far-field [SI] carrying the vessel flow + IBM cylinder wall.

    ar4 axes: vessel + flow along world-**x** (horizontal), gravity ⊥ along −z. The
    screw's long axis is body-z, rotated body-z→world-x by the body's initial
    orientation, so it lies along the pipe and corkscrews *through* it (not down it).
    """
    nb = body.n_points
    a = _p(params, "R_CYL_UMR_M")
    dx = a / _p(params, "CPR")
    # screw long axis is body-z; its world-x extent (after the body-z→world-x rotation)
    # equals the body-z half-length. The pipe (x) box is elongated; the cross-section
    # (y,z) spans the vessel.
    x_half = float(np.abs(body.points[:, 2]).max())
    Lx = 2 * (x_half + 3 * a)
    Lyz = 2 * 1.25 * R_ves
    nx, nyz = int(np.ceil(Lx / dx)), int(np.ceil(Lyz / dx))
    mesh = make_cartesian_mesh_3d(nx, nyz, nyz, Lx, Lyz, Lyz,
                                  origin=(-Lx / 2, -Lyz / 2, -Lyz / 2),
                                  dtype=jnp.float64, periodic_x=True)
    profile = _p(params, "FLOW_PROFILE")
    nu = mu / rho
    if profile == "poiseuille":
        lift = make_poiseuille_lift(mesh, R_pipe=R_ves, U_mean=_p(params, "U_MEAN"), axis=0)
    elif profile == "womersley":
        lift = make_womersley_lift_analytical(
            mesh, R_pipe=R_ves, U_mean_dc=_p(params, "U_MEAN"),
            U_mean_amp=_p(params, "U_MEAN_AMP"), omega=_p(params, "OMEGA_FLOW"),
            nu=nu, axis=0)
    else:
        raise ValueError(f"FLOW_PROFILE must be poiseuille|womersley, got {profile!r}")
    # IBM cylinder along x: φ > 0 in the lumen (y²+z² < R_ves²).
    wall = IBMBody(name="wall", sdf=lambda x: R_ves - jnp.sqrt(
        x[..., 1] ** 2 + x[..., 2] ** 2 + 1e-30))
    bcs = {pn: VelocityBC(u_wall=jnp.zeros(3))
           for pn in ("y_min", "y_max", "z_min", "z_max")}
    cfg = PisoConfig(nu=nu, rho=rho, gamma_conv=0.5, n_corrector=2,
                     pressure_bc=("periodic", "neumann", "neumann"),
                     velocity_bc=("periodic", "dirichlet", "dirichlet"),
                     ibm_alpha=1e5, ibm_eps=1.0 * dx, transform_backend="dense")
    return FVMFluidNode("fvm", _p(params, "DT"), mesh=mesh, bcs=bcs, cfg=cfg,
                        static_bodies=[wall], lifting=lift, n_sample_points=nb,
                        n_forcing_points=nb, forcing_sigma=1.5 * dx)


def _near_node(params, body_mesh, table, mu):
    """SI-native confined Stokes BEM (Liron–Shahar wall table via length_scale)."""
    return StokesletFluidNode(
        "bem", _p(params, "DT"), mu=mu, body_mesh=body_mesh,
        interface_mesh=create_interface_mesh(radius=1.8, n_refine=1),
        wall_table=table, R_cyl=float(table.R_cyl),
        length_scale=_p(params, "R_CYL_UMR_M"))


def _body_node(params, mu, rho):
    dt = _p(params, "DT")
    a = _p(params, "R_CYL_UMR_M")
    L = _p(params, "L_UMR_M")
    vol = np.pi * a ** 2 * L                              # screw volume ≈ cylinder
    m_eff = (rho + _p(params, "DELTA_RHO")) * vol
    I_eff = 0.5 * m_eff * a ** 2
    common = dict(semi_major_axis_m=a, semi_minor_axis_m=0.5 * a,
                  density_kg_m3=rho + _p(params, "DELTA_RHO"),
                  fluid_viscosity_pa_s=mu, fluid_density_kg_m3=rho)
    if _p(params, "SWIM_MODE") == "held":
        return RigidBodyNode("body", dt, kinematic_mode=True, **common)
    return RigidBodyNode("body", dt, use_inertial=True, I_eff=I_eff, m_eff=m_eff,
                         **common)


def _magnetic_effect(params):
    dt = _p(params, "DT")
    motor = MotorNode("motor", dt, inertia_kg_m2=_p(params, "MOTOR_INERTIA"),
                      kt_n_m_per_a=_p(params, "MOTOR_KT"), r_ohm=_p(params, "MOTOR_R"),
                      l_henry=_p(params, "MOTOR_L"), damping_n_m_s=_p(params, "MOTOR_DAMPING"),
                      axis_in_parent_frame=(0, 0, 1),
                      tool_offset_in_rotor_frame=(0, 0, 0, 1, 0, 0, 0))
    magnet = PermanentMagnetNode("ext_magnet", dt, dipole_moment_a_m2=_p(params, "MAG_DIPOLE"),
                                 magnetization_axis_in_body=(1, 0, 0),
                                 magnet_radius_m=_p(params, "MAG_R"),
                                 magnet_length_m=_p(params, "MAG_L"),
                                 field_model="point_dipole", earth_field_world_t=(0, 0, 0))
    response = PermanentMagnetResponseNode("magnet", dt, n_magnets=_p(params, "N_MAGNETS"),
                                           m_single=_p(params, "M_SINGLE"),
                                           moment_axis=_p(params, "MOMENT_AXIS"))
    return MagneticModel.PointDipole(magnet, response, pose_source=motor)


def build_experiment(params: dict | None = None) -> Experiment:
    """Compose the SI gate experiment via effects-first ``Experiment.attach``."""
    params = params or {}
    mu, rho = _p(params, "MU_PA_S"), _p(params, "RHO_FLUID")
    R_ves = _p(params, "R_VES_M")
    table = load_wall_table(str(_p(params, "WALL_TABLE")))

    body_mesh = _screw_mesh(params)
    rho_max = float(np.sqrt(body_mesh.points[:, 0] ** 2 + body_mesh.points[:, 1] ** 2).max())
    if rho_max >= R_ves:
        raise ValueError(f"screw leaves lumen: max ρ={rho_max:.4g} m ≥ R_ves={R_ves} m")

    far = _far_node(params, body_mesh, R_ves, mu, rho)
    near = _near_node(params, body_mesh, table, mu)

    exp = Experiment(name="schwarz_vessel_helix", mime_version_min="0.1.0")
    exp.set_medium(Medium({"density": rho, "viscosity": mu}))
    exp.set_body(Body("body", node=_body_node(params, mu, rho),
                      properties={"hydrodynamic": {}, "magnetic": {}}))

    exp.attach(HydrodynamicModel.TwoScale(
        far, near, body_points=jnp.asarray(body_mesh.points),
        body_weights=jnp.asarray(body_mesh.weights)))
    exp.attach(_magnetic_effect(params))
    a = _p(params, "R_CYL_UMR_M")
    vol = np.pi * a ** 2 * _p(params, "L_UMR_M")
    exp.attach(GravityEffect(timestep=_p(params, "DT"),
                             delta_rho_kg_m3=_p(params, "DELTA_RHO"),
                             volume_m3=vol, direction=(0.0, 0.0, -1.0)))
    if _p(params, "INCLUDE_ARM"):
        exp.attach(RobotArmEffect(
            urdf_path=str(_p(params, "ARM_URDF")), carrier_node="motor",
            end_effector_link_name=_p(params, "ARM_EE_LINK"),
            base_pose_world=_p(params, "ARM_BASE_POSE"), timestep=_p(params, "DT")))

    exp.add_external_input("motor", "commanded_velocity", ())
    if not _p(params, "INCLUDE_ARM"):
        exp.add_external_input("motor", "parent_pose_world", (7,))
    if _p(params, "SWIM_MODE") == "held":
        exp.add_external_input("body", "external_velocity", (3,))
        exp.add_external_input("body", "external_angular_velocity", (3,))
    return exp


def _seed_arm_home(gm, params):
    """Seed the AR4 at its IK home pose so the EE holds the magnet at the standoff
    from frame 0 (else the arm starts collapsed at q=0 and the magnet is mislocated).
    Mirrors ar4_helical_drive/physics/setup.py."""
    from mime.control.kinematics.urdf import parse_urdf
    from mime.control.kinematics.fk import joint_to_world_transforms
    from mime.control.kinematics.transform import pose_to_matrix, _rotation_matrix_to_quat
    from mime.nodes.actuation.robot_arm import _resolve_link_index

    tree = parse_urdf(str(_p(params, "ARM_URDF")))
    ee_idx = _resolve_link_index(tree, _p(params, "ARM_EE_LINK"))
    home_q = jnp.asarray(_p(params, "ARM_HOME_RAD"), dtype=jnp.float32)
    base = jnp.asarray(_p(params, "ARM_BASE_POSE"), dtype=jnp.float32)
    ee_off = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    link_xf = pose_to_matrix(base)[None] @ joint_to_world_transforms(tree, home_q)
    quats = jax.vmap(_rotation_matrix_to_quat)(link_xf[:, :3, :3])
    link_poses = jnp.concatenate([link_xf[:, :3, 3], quats], axis=1)
    T_ee = link_xf[ee_idx] @ pose_to_matrix(ee_off)
    ee_pose = jnp.concatenate([T_ee[:3, 3], _rotation_matrix_to_quat(T_ee[:3, :3])])

    st = dict(gm.get_node_state("arm"))
    st.update(joint_angles=home_q, joint_velocities=jnp.zeros_like(home_q),
              link_poses_world=link_poses, end_effector_pose_world=ee_pose)
    gm.set_node_state("arm", st)


def _seed_body_orientation(gm, params):
    """Seed the screw's initial orientation (body-z → world-x) so it lies along the
    pipe. The BEM is frame-aware (rotates by body_orientation), so this is correct."""
    st = dict(gm.get_node_state("body"))
    st["orientation"] = jnp.asarray(_p(params, "BODY_INIT_ORIENT"), dtype=st["orientation"].dtype)
    gm.set_node_state("body", st)


def build_graph(params: dict | None = None):
    """Runner entry point: build the experiment and return its GraphManager."""
    params = params or {}
    gm, _ = build_experiment(params).build()
    _seed_body_orientation(gm, params)
    if _p(params, "INCLUDE_ARM"):
        _seed_arm_home(gm, params)
    if hasattr(gm, "compile"):
        gm.compile()
    return gm


def body_world_points(body_points_ref, state=None) -> jnp.ndarray:
    """World-frame BEM sampling/forcing points = body pose · reference points [m].

    The two-scale coupling samples the FVM velocity (and spreads the BEM reaction)
    at the body's world surface locations; ``make_two_scale_coupling`` exposes these
    as the far node's ``sample_points``/``forcing_points`` external inputs (the
    ``TwoScale`` effect does not bake them — see its docstring). Constant for a held
    body, pose-tracked for a free one.
    """
    P = jnp.asarray(body_points_ref)
    if state is None or "body" not in state:
        return P
    from mime.core.quaternion import rotate_vector
    b = state["body"]
    q = jnp.asarray(b["orientation"])
    x = jnp.asarray(b["position"])
    return jax.vmap(lambda p: rotate_vector(q, p))(P) + x


def default_external_inputs(params: dict | None = None, *, body_points_ref=None,
                            state=None) -> dict:
    """External-input dict for a basic driven run (no arm).

    Pass ``body_points_ref`` so the FVM geometry is supplied each step (**required**
    or the coupling samples at the origin and diverges). Pass ``state`` to pose-track
    a moving (free) body.
    """
    params = params or {}
    drive = jnp.asarray(2 * np.pi * _p(params, "DRIVE_HZ"))
    ext: dict[str, Any] = {"motor": {"commanded_velocity": drive}}
    if _p(params, "INCLUDE_ARM"):
        # Joint-space PD hold to the seeded home pose (on top of the node's gravity
        # compensation). A placeholder for the closed-loop standoff *tracker* the
        # control work-packages will build — here it just keeps the magnet steady.
        home_q = jnp.asarray(_p(params, "ARM_HOME_RAD"))
        kp, kd = _p(params, "ARM_KP"), _p(params, "ARM_KD")
        if state is not None and "arm" in state:
            q = jnp.asarray(state["arm"]["joint_angles"])
            qd = jnp.asarray(state["arm"].get("joint_velocities", jnp.zeros_like(q)))
            tau = kp * (home_q - q) - kd * qd
        else:
            tau = jnp.zeros_like(home_q)
        ext["arm"] = {"commanded_joint_torques": tau}
    else:
        # no-arm: magnet above the x-pipe at (0,0,standoff), motor axis along world-x
        # (the pipe) via the +90°-about-y orientation (so it drives the x-screw).
        z = _p(params, "MAG_STANDOFF_M")
        ext["motor"]["parent_pose_world"] = jnp.array(
            [0.0, 0.0, z, 0.7071068, 0.0, 0.7071068, 0.0])
    if _p(params, "SWIM_MODE") == "held":
        ext["body"] = {"external_velocity": jnp.zeros(3),
                       "external_angular_velocity": jnp.zeros(3)}
    if body_points_ref is not None:
        pts = body_world_points(body_points_ref, state)
        ext["fvm"] = {"sample_points": pts, "forcing_points": pts}
    return ext
