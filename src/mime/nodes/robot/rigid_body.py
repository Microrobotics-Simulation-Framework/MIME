"""RigidBodyNode — 6-DOF rigid body dynamics for microrobots.

Three modes:
1. **Overdamped + analytical drag** (default): V = R_T^{-1} @ F_ext.
   Stokes regime, inertia negligible, drag computed from Oberbeck-Stechert.
2. **Overdamped + external drag** (use_analytical_drag=False): same algebra
   but drag_force/drag_torque come from IB-LBM via boundary inputs.
3. **Inertial + external drag** (use_inertial=True): Newton's 2nd law
   with explicit Euler integration. Required for FSI coupling where the
   overdamped model causes step-0 blowup (zero drag → infinite omega).

State: position (3D) + orientation (quaternion) + velocity + angular velocity.

Reference: Ch 2 and Ch 4 of "Mathematical Modelling of Swimming Soft Microrobots"
"""

from __future__ import annotations

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole, BiocompatibilityMeta, BiocompatibilityClass,
    AnatomicalRegimeMeta, AnatomicalCompartment, FlowRegime,
)
from mime.core.quaternion import (
    quat_normalize, quat_from_angular_velocity, quat_multiply, identity_quat,
)


def oberbeck_stechert_coefficients(e: float) -> tuple:
    """Compute Oberbeck-Stechert drag coefficients for a prolate ellipsoid.

    Parameters
    ----------
    e : float or jnp scalar
        Eccentricity = sqrt(1 - b^2/a^2). Range [0, 1).

    Returns
    -------
    C_1, C_2, C_3 : drag coefficients (dimensionless)

    For a sphere (e -> 0): C_1 = C_2 = 1, C_3 = 1.
    Uses jnp.where to handle the e -> 0 singularity safely.
    """
    e2 = e * e
    log_term = jnp.log((1.0 + e) / jnp.maximum(1.0 - e, 1e-30))

    # Denominators (may be zero at e=0)
    denom_1 = -2.0 * e + (1.0 + e2) * log_term
    denom_2 = 2.0 * e + (3.0 * e2 - 1.0) * log_term

    C_1_raw = (8.0 / 3.0) * e * e2 / jnp.maximum(jnp.abs(denom_1), 1e-30)
    C_2_raw = (16.0 / 3.0) * e * e2 / jnp.maximum(jnp.abs(denom_2), 1e-30)
    C_3_raw = (4.0 / 3.0) * e * e2 * (2.0 - e2) / (
        (1.0 + e2) * jnp.maximum(jnp.abs(denom_1), 1e-30)
    )

    # For e < epsilon, use sphere limit: C_1 = C_2 = C_3 = 1
    is_sphere = e < 1e-6
    C_1 = jnp.where(is_sphere, 1.0, C_1_raw)
    C_2 = jnp.where(is_sphere, 1.0, C_2_raw)
    C_3 = jnp.where(is_sphere, 1.0, C_3_raw)

    return C_1, C_2, C_3


@stability(StabilityLevel.EXPERIMENTAL)
class RigidBodyNode(MimeNode):
    """6-DOF rigid body in overdamped Stokes flow.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    semi_major_axis_m : float
        Semi-major axis a [m] (along body e1).
    semi_minor_axis_m : float
        Semi-minor axis b [m]. For sphere: b = a.
    density_kg_m3 : float
        Body density [kg/m^3].
    fluid_viscosity_pa_s : float
        Surrounding fluid dynamic viscosity [Pa.s].
    fluid_density_kg_m3 : float
        Surrounding fluid density [kg/m^3].
    use_analytical_drag : bool
        If True (default), compute drag from Oberbeck-Stechert.
        If False, expect drag_force and drag_torque as boundary inputs
        from CSFFlowNode/IB-LBM.

    Boundary Inputs (additive forces/torques)
    -----------------------------------------
    magnetic_force : (3,)
        From MagneticResponseNode. Additive.
    magnetic_torque : (3,)
        From MagneticResponseNode. Additive.
    drag_force : (3,)
        From CSFFlowNode (IB-LBM mode). Additive.
    drag_torque : (3,)
        From CSFFlowNode (IB-LBM mode). Additive.
    external_force : (3,)
        Any additional external force (gravity, contact). Additive.
    external_torque : (3,)
        Any additional external torque. Additive.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-003",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="6-DOF rigid body dynamics in overdamped Stokes regime",
        governing_equations=(
            r"F_total ≈ 0 => V = R_T^{-1} F_ext; "
            r"T_total ≈ 0 => omega = R_R^{-1} T_ext; "
            r"x += V*dt; q = dq(omega, dt) * q"
        ),
        discretization="Explicit Euler for position; exact quaternion rotation for orientation",
        assumptions=(
            "Stokes regime: Re << 1, inertia negligible",
            "Rigid body — no deformation",
            "Prolate ellipsoid shape for analytical drag coefficients",
            "Fluid at rest (quiescent) when using analytical drag — background "
            "flow drag comes from CSFFlowNode as boundary input",
        ),
        limitations=(
            "Analytical drag only valid for Re < 0.1",
            "No near-wall corrections in this node (SurfaceContactNode needed)",
            "Quaternion integration uses first-order approximation",
        ),
        validated_regimes=(
            ValidatedRegime("Re", 0.0, 0.1, "",
                            "Stokes regime — inertia negligible"),
            ValidatedRegime("semi_major_axis_m", 1e-6, 1e-3, "m",
                            "Microrobot size range"),
        ),
        references=(
            Reference("Lighthill1976", "Flagellar Hydrodynamics"),
            Reference("Rodenborn2013", "Propulsion of microorganisms by helical flagellum"),
        ),
        hazard_hints=(
            "Re > 1 invalidates Stokes drag — nonlinear convective term dominates",
            "Oberbeck-Stechert coefficients have singularity at e=0; "
            "jnp.where guard switches to sphere limit",
        ),
        implementation_map={
            "V = F_total / (6*pi*eta*a*C)": (
                "mime.nodes.robot.rigid_body.RigidBodyNode.update"
            ),
            "Oberbeck-Stechert C_1,C_2,C_3": (
                "mime.nodes.robot.rigid_body.oberbeck_stechert_coefficients"
            ),
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ROBOT_BODY,
        anatomical_regimes=(
            AnatomicalRegimeMeta(
                compartment=AnatomicalCompartment.CSF,
                anatomy="general CSF space",
                flow_regime=FlowRegime.STAGNANT,
                re_min=0.0, re_max=0.1,
                viscosity_min_pa_s=7e-4, viscosity_max_pa_s=1e-3,
            ),
        ),
        biocompatibility=BiocompatibilityMeta(
            materials=("SU-8", "NdFeB"),
            iso_10993_class=BiocompatibilityClass.NOT_ASSESSED,
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        semi_major_axis_m: float = 100e-6,
        semi_minor_axis_m: float = 100e-6,
        density_kg_m3: float = 1100.0,
        fluid_viscosity_pa_s: float = 8.5e-4,
        fluid_density_kg_m3: float = 1002.0,
        use_analytical_drag: bool = True,
        use_inertial: bool = False,
        I_eff: float | None = None,
        m_eff: float | None = None,
        omega_max: float | None = None,
        **kwargs,
    ):
        if use_inertial and I_eff is None:
            raise ValueError(
                "use_inertial=True requires I_eff (effective rotational inertia "
                "in kg*m^2). For d2.8 UMR: I_eff ≈ 1e-10."
            )
        if use_inertial and m_eff is None:
            m_eff = density_kg_m3 * (4.0 / 3.0) * 3.14159 * semi_major_axis_m * semi_minor_axis_m**2

        super().__init__(
            name, timestep,
            semi_major_axis_m=semi_major_axis_m,
            semi_minor_axis_m=semi_minor_axis_m,
            density_kg_m3=density_kg_m3,
            fluid_viscosity_pa_s=fluid_viscosity_pa_s,
            fluid_density_kg_m3=fluid_density_kg_m3,
            use_analytical_drag=use_analytical_drag,
            use_inertial=use_inertial,
            I_eff=I_eff,
            m_eff=m_eff,
            omega_max=omega_max,
            **kwargs,
        )

    def initial_state(self) -> dict:
        return {
            "position": jnp.zeros(3),
            "orientation": identity_quat(),
            "velocity": jnp.zeros(3),
            "angular_velocity": jnp.zeros(3),
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "magnetic_force": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Magnetic force [N]",
            ),
            "magnetic_torque": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Magnetic torque [N.m]",
            ),
            "drag_force": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Fluid drag force [N] (from IB-LBM)",
            ),
            "drag_torque": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Fluid drag torque [N.m] (from IB-LBM)",
            ),
            "external_force": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Additional external force [N]",
            ),
            "external_torque": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Additional external torque [N.m]",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        a = self.params["semi_major_axis_m"]
        b = self.params["semi_minor_axis_m"]
        eta = self.params["fluid_viscosity_pa_s"]
        use_analytical = self.params["use_analytical_drag"]
        use_inertial = self.params["use_inertial"]

        pos = state["position"]
        q = state["orientation"]

        # Sum all external forces and torques
        F_ext = (
            boundary_inputs.get("magnetic_force", jnp.zeros(3))
            + boundary_inputs.get("external_force", jnp.zeros(3))
        )
        T_ext = (
            boundary_inputs.get("magnetic_torque", jnp.zeros(3))
            + boundary_inputs.get("external_torque", jnp.zeros(3))
        )

        if use_inertial:
            # Inertial mode: I_eff * dΩ/dt = T_ext + T_drag
            I_eff = self.params["I_eff"]
            m_eff = self.params["m_eff"]
            F_drag = boundary_inputs.get("drag_force", jnp.zeros(3))
            T_drag = boundary_inputs.get("drag_torque", jnp.zeros(3))

            omega_old = state["angular_velocity"]
            V_old = state["velocity"]

            # Euler integration of rotational dynamics
            T_total = T_ext + T_drag
            omega = omega_old + T_total / I_eff * dt

            # Ma safety clamp: limit angular velocity magnitude
            omega_max = self.params.get("omega_max", None)
            if omega_max is not None:
                omega_mag = jnp.linalg.norm(omega)
                scale = jnp.where(
                    omega_mag > omega_max,
                    omega_max / jnp.maximum(omega_mag, 1e-30),
                    1.0,
                )
                omega = omega * scale

            # Euler integration of translational dynamics
            F_total = F_ext + F_drag
            V = V_old + F_total / m_eff * dt

        elif use_analytical:
            e = jnp.sqrt(jnp.maximum(1.0 - (b/a)**2, 0.0))
            C_1, C_2, C_3 = oberbeck_stechert_coefficients(e)
            R_T_diag = 6.0 * a * jnp.pi * eta * jnp.array([C_1, C_2, C_2])
            R_R_diag = 8.0 * a * b**2 * jnp.pi * eta * jnp.array([C_3, C_3, C_3])
            V = F_ext / jnp.maximum(R_T_diag, 1e-30)
            omega = T_ext / jnp.maximum(R_R_diag, 1e-30)
        else:
            F_drag = boundary_inputs.get("drag_force", jnp.zeros(3))
            T_drag = boundary_inputs.get("drag_torque", jnp.zeros(3))
            e = jnp.sqrt(jnp.maximum(1.0 - (b/a)**2, 0.0))
            C_1, C_2, C_3 = oberbeck_stechert_coefficients(e)
            R_T_diag = 6.0 * a * jnp.pi * eta * jnp.array([C_1, C_2, C_2])
            R_R_diag = 8.0 * a * b**2 * jnp.pi * eta * jnp.array([C_3, C_3, C_3])
            V = (F_ext + F_drag) / jnp.maximum(R_T_diag, 1e-30)
            omega = (T_ext + T_drag) / jnp.maximum(R_R_diag, 1e-30)

        # Integrate position (Euler)
        new_pos = pos + V * dt

        # Integrate orientation (quaternion)
        dq = quat_from_angular_velocity(omega, dt)
        new_q = quat_normalize(quat_multiply(dq, q))

        return {
            "position": new_pos,
            "orientation": new_q,
            "velocity": V,
            "angular_velocity": omega,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "position": state["position"],
            "orientation": state["orientation"],
            "velocity": state["velocity"],
            "angular_velocity": state["angular_velocity"],
        }
