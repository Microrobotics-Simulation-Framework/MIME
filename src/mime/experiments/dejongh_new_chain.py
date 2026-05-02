"""De Jongh confined-swimming graph driven by the new actuation chain.

This module is the experiment-side wiring for MIME-VER-130 / 131 / 132.
It mirrors :mod:`mime.experiments.dejongh` but replaces the lone
``ExternalMagneticFieldNode`` with the Motor + PermanentMagnet pair
introduced in the actuation-decomposition plan:

    MotorNode (rotor angle θ(t))
       │ rotor_pose_world (parent ⊕ R(z, θ))
       ▼
    PermanentMagnetNode  ←─── target_position_world (UMR position)
       │ field_vector, field_gradient at UMR location
       ▼
    PermanentMagnetResponseNode (existing — unchanged)
       │ magnetic_torque, magnetic_force
       ▼
    RigidBodyNode  …  rest of dejongh stack identical

The arm is *static* in this scenario (Wave C-1 scope) — its presence
is captured by ``magnet_base_pose_world``, which we simply hardcode.
A future RobotArm-driven variant will compose this base pose with
``RobotArmNode.end_effector_pose_world``.

The body→magnet feedback (UMR position drives the field-evaluation
target) is wired through a coupling group, per dejongh deliverable
A.2 (V2 confirmed Gauss-Seidel converges within a single timestep).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager

from mime.nodes.actuation.motor import MotorNode
from mime.nodes.actuation.permanent_magnet import PermanentMagnetNode
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode
from mime.nodes.environment.gravity_node import GravityNode
from mime.nodes.environment.stokeslet.mlp_resistance_node import MLPResistanceNode
from mime.nodes.environment.stokeslet.lubrication_node import (
    LubricationCorrectionNode,
)
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.nodes.robot.constraints import CylindricalVesselConstraint

from mime.experiments.dejongh import (
    FL_PARAMS,
    VESSELS,
    R_CYL_UMR_MM,
    R_MAX_BODY_FACTOR,
    N_MAGNETS,
    M_SINGLE,
    umr_volume_m3_fl,
    default_mlp_weights_path,
)


def build_graph(
    design_name: str = "FL-9",
    vessel_name: str = '1/4"',
    *,
    mu_Pa_s: float = 1e-3,
    delta_rho: float = 410.0,
    volume_m3: Optional[float] = None,
    dt: float = 5e-4,
    use_lubrication: bool = True,
    lubrication_epsilon_mm: float = 0.15,
    mlp_weights_path: Optional[Path] = None,
    # New-chain magnet placement -----------------------------------------
    magnet_base_xyz_m: Tuple[float, float, float] = (0.0, 0.0, 0.05),
    magnet_lateral_offset_m: float = 0.0,
    magnet_dipole_a_m2: float = N_MAGNETS * M_SINGLE,
    magnet_radius_m: float = 1e-3,
    magnet_length_m: float = 2e-3,
    field_model: str = "point_dipole",
    motor_axis_in_parent: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    motor_inertia_kg_m2: float = 1e-5,
    motor_kt_n_m_per_a: float = 0.05,
    motor_r_ohm: float = 1.0,
    motor_l_henry: float = 1e-3,
    motor_damping_n_m_s: float = 1e-4,
    use_coupling_group: bool = True,
    # Vessel axis (0=x, 1=y, 2=z). Default 2 (z) matches dejongh.
    # AR4 experiment overrides to 0 to lay the tube horizontal.
    vessel_axis: int = 2,
    # Body gravity direction (unit vector). Default (0,-1,0) matches
    # dejongh. AR4 experiment overrides to (0,0,-1) so gravity points
    # at the floor.
    body_gravity_direction: Tuple[float, float, float] = (0.0, -1.0, 0.0),
    # Permanent magnet's moment direction in the rotor body frame.
    # Must be perpendicular to ``motor_axis_in_parent`` so the moment
    # vector rotates with the rotor (otherwise it stays parallel to
    # the spin axis and the field doesn't rotate).
    magnet_axis_in_body: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> GraphManager:
    """Build the de Jongh graph with Motor + PermanentMagnet replacing
    the legacy ``ExternalMagneticFieldNode``.

    Geometry conventions (matching :mod:`mime.experiments.dejongh`):
    vessel axis is +z, gravity is along −y, the UMR's body magnetic
    moment is along +x in body frame and rotates with the body
    quaternion. The new permanent magnet's *body* moment is along +x
    and is rotated by the motor's rotor frame about ``motor_axis_in_parent``
    — which is +z by default — so the magnet's moment sweeps the xy
    plane just like the legacy node's uniform B(t) did.

    The magnet's *position* in the world is
    ``magnet_base_xyz_m + (magnet_lateral_offset_m, 0, 0)``. With
    ``magnet_lateral_offset_m == 0`` the magnet sits directly above
    the vessel centerline, reproducing the symmetric configuration of
    the legacy uniform-field node. Non-zero offset triggers the
    misalignment physics that MIME-VER-132 exercises.
    """
    fl = FL_PARAMS[design_name]
    R_ves_mm = VESSELS[vessel_name]
    if volume_m3 is None:
        volume_m3 = umr_volume_m3_fl(fl["L_UMR_mm"])
    if mlp_weights_path is None:
        mlp_weights_path = default_mlp_weights_path()

    # --- Magnet base pose (4-DOF Xform — translation + identity quat) -----
    base_t = (
        float(magnet_base_xyz_m[0] + magnet_lateral_offset_m),
        float(magnet_base_xyz_m[1]),
        float(magnet_base_xyz_m[2]),
    )
    motor_parent_pose = (base_t[0], base_t[1], base_t[2], 1.0, 0.0, 0.0, 0.0)

    # --- Nodes ------------------------------------------------------------
    motor_node = MotorNode(
        "motor",
        dt,
        inertia_kg_m2=motor_inertia_kg_m2,
        kt_n_m_per_a=motor_kt_n_m_per_a,
        r_ohm=motor_r_ohm,
        l_henry=motor_l_henry,
        damping_n_m_s=motor_damping_n_m_s,
        axis_in_parent_frame=tuple(motor_axis_in_parent),
        # Identity tool-offset: magnet body sits at rotor origin.
        tool_offset_in_rotor_frame=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )

    magnet_node = PermanentMagnetNode(
        "ext_magnet",
        dt,
        dipole_moment_a_m2=magnet_dipole_a_m2,
        magnetization_axis_in_body=tuple(magnet_axis_in_body),
        magnet_radius_m=magnet_radius_m,
        magnet_length_m=magnet_length_m,
        field_model=field_model,
        # Earth field is captured but, for V&V apples-to-apples with the
        # legacy dejongh runs, set to zero by default.
        earth_field_world_t=(0.0, 0.0, 0.0),
    )

    # UMR's onboard magnetic moment is in the *body* frame and must
    # be perpendicular to the helix's own body axis (body-z by the
    # dejongh mesh convention). World-frame alignment with the
    # vessel comes from the body's *initial orientation* — the
    # caller rotates the body so body-z aligns with the vessel
    # axis. So the body-frame moment stays at (1,0,0) regardless of
    # vessel_axis: that's perpendicular to body-z, and the body
    # rotation then maps it into the field's rotation plane.
    response_node = PermanentMagnetResponseNode(
        "magnet", dt,
        n_magnets=N_MAGNETS, m_single=M_SINGLE,
        moment_axis=(1.0, 0.0, 0.0),
    )
    gravity_node = GravityNode(
        "gravity", dt,
        delta_rho_kg_m3=delta_rho,
        volume_m3=volume_m3,
        direction=tuple(body_gravity_direction),
    )
    mlp_node = MLPResistanceNode(
        "mlp_drag", dt,
        mlp_weights_path=str(mlp_weights_path),
        nu=fl["nu"], L_UMR_mm=fl["L_UMR_mm"],
        R_cyl_UMR_mm=R_CYL_UMR_MM,
        R_ves_mm=R_ves_mm,
        mu_Pa_s=mu_Pa_s,
    )

    R_max_body_mm = R_CYL_UMR_MM * R_MAX_BODY_FACTOR
    effective_R_mm = max(R_ves_mm - R_max_body_mm - 0.1, 0.1)
    vessel_constraint = CylindricalVesselConstraint(
        radius=effective_R_mm * 1e-3,
        # 1 m vessel — long enough that the helix can corkscrew at
        # paper-rate (~3 mm/s) for hundreds of seconds without
        # reaching the end-cap and triggering the constraint
        # instability we hit with the legacy 100 mm tube.
        half_length=0.5,
        axis=int(vessel_axis),
    )

    m_eff = delta_rho * volume_m3 + 1000.0 * volume_m3
    I_eff = 0.5 * m_eff * (R_CYL_UMR_MM * 1e-3) ** 2

    body_node = RigidBodyNode(
        "body", dt,
        semi_major_axis_m=R_CYL_UMR_MM * 1e-3 * R_MAX_BODY_FACTOR,
        semi_minor_axis_m=R_CYL_UMR_MM * 1e-3,
        density_kg_m3=delta_rho + 1000.0,
        fluid_viscosity_pa_s=mu_Pa_s,
        fluid_density_kg_m3=1000.0,
        use_analytical_drag=False,
        use_inertial=True,
        I_eff=I_eff,
        m_eff=m_eff,
        constraint=vessel_constraint,
    )

    # --- Graph ------------------------------------------------------------
    gm = GraphManager()
    nodes = [motor_node, magnet_node, response_node, gravity_node, mlp_node, body_node]
    if use_lubrication:
        lub_node = LubricationCorrectionNode(
            "lub", dt,
            R_ves_mm=R_ves_mm,
            R_max_body_mm=R_CYL_UMR_MM * R_MAX_BODY_FACTOR,
            mu_Pa_s=mu_Pa_s,
            epsilon_mm=lubrication_epsilon_mm,
            a_eff_mm=R_CYL_UMR_MM * R_MAX_BODY_FACTOR,
        )
        nodes.append(lub_node)

    for n in nodes:
        gm.add_node(n)

    # Motor parent pose comes from a constant external input (the arm is
    # static in this Wave C-1 scenario). Adding it as an external input
    # keeps the wiring uniform with future arm-driven variants where it
    # would come from RobotArmNode.end_effector_pose_world.
    gm.add_external_input("motor", "parent_pose_world", shape=(7,),
                          dtype=jnp.float32)

    # Motor → magnet pose
    gm.add_edge("motor", "ext_magnet", "rotor_pose_world", "magnet_pose_world")

    # Body → magnet target (closes the b→f→b loop — handled by coupling group)
    gm.add_edge("body", "ext_magnet", "position", "target_position_world")

    # Magnet → response
    gm.add_edge("ext_magnet", "magnet", "field_vector", "field_vector")
    gm.add_edge("ext_magnet", "magnet", "field_gradient", "field_gradient")
    gm.add_edge("body", "magnet", "orientation", "orientation")

    # Body → MLP drag
    gm.add_edge("body", "mlp_drag", "position", "robot_position")
    gm.add_edge("body", "mlp_drag", "orientation", "robot_orientation")
    gm.add_edge("body", "mlp_drag", "velocity", "body_velocity")
    gm.add_edge("body", "mlp_drag", "angular_velocity", "body_angular_velocity")

    # Forces + torques into body
    gm.add_edge("gravity", "body", "gravity_force", "external_force",
                additive=True)
    gm.add_edge("magnet", "body", "magnetic_force", "magnetic_force")
    gm.add_edge("magnet", "body", "magnetic_torque", "magnetic_torque")

    if use_lubrication:
        gm.add_edge("mlp_drag", "lub", "resistance_matrix", "resistance_matrix")
        gm.add_edge("body", "lub", "position", "robot_position")
        gm.add_edge("body", "lub", "velocity", "body_velocity")
        gm.add_edge("body", "lub", "angular_velocity", "body_angular_velocity")
        gm.add_edge("lub", "body", "drag_force", "drag_force")
        gm.add_edge("lub", "body", "drag_torque", "drag_torque")
    else:
        gm.add_edge("mlp_drag", "body", "drag_force", "drag_force")
        gm.add_edge("mlp_drag", "body", "drag_torque", "drag_torque")

    # Coupling group: body↔ext_magnet↔magnet form a cycle through
    # target_position → field → torque/force → body. Resolve with
    # Gauss-Seidel iteration (per dejongh deliverable Appendix A.2)
    # for high-fidelity work.
    #
    # For visualisation / iteration the coupling group is the dominant
    # per-step cost — its while_loop can run up to 20 inner iterations,
    # each doing CRBA + RNEA + cuSolver. Setting
    # ``use_coupling_group=False`` falls back to the same staggered
    # back-edges that the legacy ``mime.experiments.dejongh`` graph
    # uses (a one-step phase lag in the body→field link, ~10° at
    # 60 Hz — invisible on cm-scale microrobot trajectories). The
    # speedup vs. the coupling group is ~10×.
    if use_coupling_group:
        gm.add_coupling_group(
            ["body", "ext_magnet", "magnet"],
            max_iterations=20,
            tolerance=1e-6,
        )

    # Live-controllable inputs (motor command — no longer the legacy
    # frequency_hz/field_strength_mt of the old node).
    gm.add_external_input("motor", "commanded_velocity", shape=(),
                          dtype=jnp.float32)

    return gm


def constant_motor_parent_pose(magnet_base_xyz_m, lateral_offset_m=0.0):
    """Return the constant 7-vector parent pose used by ``MotorNode``."""
    return jnp.array([
        float(magnet_base_xyz_m[0]) + float(lateral_offset_m),
        float(magnet_base_xyz_m[1]),
        float(magnet_base_xyz_m[2]),
        1.0, 0.0, 0.0, 0.0,
    ], dtype=jnp.float32)
