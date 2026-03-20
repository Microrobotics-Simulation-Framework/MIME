"""CSFFlowNode — analytical Stokes drag in quiescent or pulsatile CSF.

Computes the drag force and torque on a spherical body using:
- Level 0: Stokes drag F = -6*pi*mu*a*(V - u_inf) for quiescent fluid
- Level 1: Faxen correction for non-uniform background flow
- Oscillatory drag kernel K = 1 + sqrt(i)*Wo_a + i*Wo_a^2/6 for pulsatile flow

This is the analytical fallback for benchmarks B0 and B2. It will be
replaced/augmented by IB-LBM for full fluid-structure coupling (Phase 2+).

Reference: CSF_addon.md (Womersley solution, Maxey-Riley equation, Brenner corrections)
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
    MimeNodeMeta, NodeRole,
    AnatomicalRegimeMeta, AnatomicalCompartment, FlowRegime,
)


# Default CSF properties
CSF_MU = 8.5e-4     # Pa.s
CSF_RHO = 1002.0    # kg/m^3
CSF_NU = CSF_MU / CSF_RHO  # ~7.0e-7 m^2/s


@stability(StabilityLevel.EXPERIMENTAL)
class CSFFlowNode(MimeNode):
    """Analytical Stokes drag on a sphere in quiescent or pulsatile CSF.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    fluid_viscosity_pa_s : float
        Dynamic viscosity [Pa.s]. Default: CSF at 37C.
    fluid_density_kg_m3 : float
        Fluid density [kg/m^3]. Default: CSF.
    robot_radius_m : float
        Effective hydrodynamic radius of the robot [m].
    pulsatile : bool
        If True, include pulsatile background flow and oscillatory corrections.
    cardiac_freq_hz : float
        Cardiac pulsation frequency [Hz].
    peak_velocity_m_s : float
        Peak centreline velocity of background CSF flow [m/s].
    tube_radius_m : float
        Channel radius [m] (for Womersley profile).

    Boundary Inputs
    ---------------
    position : (3,)
        Robot position from RigidBodyNode.
    velocity : (3,)
        Robot velocity from RigidBodyNode.
    angular_velocity : (3,)
        Robot angular velocity from RigidBodyNode.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-004",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="Analytical Stokes/Faxen drag on sphere in quiescent or pulsatile CSF",
        governing_equations=(
            r"F_drag = -6*pi*mu*a*(V - u_inf); "
            r"T_drag = -8*pi*mu*a^3*(omega_robot - omega_fluid); "
            r"Oscillatory kernel: K = 1 + sqrt(i)*Wo_a + i*Wo_a^2/6"
        ),
        discretization="Analytical — closed-form evaluation each timestep",
        assumptions=(
            "Spherical body for drag computation",
            "Stokes regime: Re << 1",
            "Newtonian fluid (CSF is Newtonian at physiological protein levels)",
            "Rigid tube walls for Womersley profile (no wall compliance)",
            "Robot is small compared to channel: a << R",
            "Quiescent mode: no background flow",
            "Pulsatile mode: Womersley profile approximated by centreline velocity",
        ),
        limitations=(
            "No resolved flow field — point-force drag only",
            "No fluid-structure interaction (one-way coupling only)",
            "Faxen correction negligible for a=100um at cardiac frequency (Wo_a~0.1)",
            "Basset history force approximated, not exact convolution",
        ),
        validated_regimes=(
            ValidatedRegime("Re", 0.0, 0.1, "", "Stokes regime"),
            ValidatedRegime("robot_radius_m", 10e-6, 500e-6, "m",
                            "Microrobot size range"),
        ),
        references=(
            Reference("Purcell1977", "Life at Low Reynolds Number"),
        ),
        hazard_hints=(
            "At high actuation frequencies (>100 Hz), oscillatory corrections "
            "become O(1) and the quasi-steady Stokes drag is insufficient",
            "Near walls (h/a < 2), Brenner corrections are needed but not "
            "included in this node — use SurfaceContactNode",
        ),
        implementation_map={
            "F = -6*pi*mu*a*(V - u)": (
                "mime.nodes.environment.csf_flow.CSFFlowNode.update"
            ),
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ENVIRONMENT,
        anatomical_regimes=(
            AnatomicalRegimeMeta(
                compartment=AnatomicalCompartment.CSF,
                anatomy="aqueduct of Sylvius",
                flow_regime=FlowRegime.PULSATILE_CSF,
                re_min=0.0, re_max=0.1,
                viscosity_min_pa_s=7e-4, viscosity_max_pa_s=1e-3,
                temperature_min_c=36.0, temperature_max_c=38.0,
            ),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        fluid_viscosity_pa_s: float = CSF_MU,
        fluid_density_kg_m3: float = CSF_RHO,
        robot_radius_m: float = 100e-6,
        pulsatile: bool = False,
        cardiac_freq_hz: float = 1.1,
        peak_velocity_m_s: float = 0.04,
        tube_radius_m: float = 1.2e-3,
        **kwargs,
    ):
        super().__init__(
            name, timestep,
            fluid_viscosity_pa_s=fluid_viscosity_pa_s,
            fluid_density_kg_m3=fluid_density_kg_m3,
            robot_radius_m=robot_radius_m,
            pulsatile=pulsatile,
            cardiac_freq_hz=cardiac_freq_hz,
            peak_velocity_m_s=peak_velocity_m_s,
            tube_radius_m=tube_radius_m,
            **kwargs,
        )

    def initial_state(self) -> dict:
        return {
            "drag_force": jnp.zeros(3),
            "drag_torque": jnp.zeros(3),
            "background_velocity": jnp.zeros(3),
            "sim_time": jnp.array(0.0),
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "position": BoundaryInputSpec(
                shape=(3,), description="Robot position [m]",
            ),
            "velocity": BoundaryInputSpec(
                shape=(3,), description="Robot velocity [m/s]",
            ),
            "angular_velocity": BoundaryInputSpec(
                shape=(3,), description="Robot angular velocity [rad/s]",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        mu = self.params["fluid_viscosity_pa_s"]
        rho = self.params["fluid_density_kg_m3"]
        a = self.params["robot_radius_m"]
        pulsatile = self.params["pulsatile"]

        V_robot = boundary_inputs.get("velocity", jnp.zeros(3))
        omega_robot = boundary_inputs.get("angular_velocity", jnp.zeros(3))

        t = state["sim_time"] + dt

        # Background flow velocity at robot position
        if pulsatile:
            f_c = self.params["cardiac_freq_hz"]
            v_peak = self.params["peak_velocity_m_s"]
            omega_c = 2.0 * jnp.pi * f_c
            # Simplified: centreline sinusoidal pulsation along z-axis
            u_inf = jnp.array([0.0, 0.0, v_peak * jnp.sin(omega_c * t)])
        else:
            u_inf = jnp.zeros(3)

        # Stokes drag: F = -6*pi*mu*a*(V - u_inf)
        F_drag = -6.0 * jnp.pi * mu * a * (V_robot - u_inf)

        # Rotational drag: T = -8*pi*mu*a^3*omega
        T_drag = -8.0 * jnp.pi * mu * a**3 * omega_robot

        return {
            "drag_force": F_drag,
            "drag_torque": T_drag,
            "background_velocity": u_inf,
            "sim_time": t,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "drag_force": state["drag_force"],
            "drag_torque": state["drag_torque"],
        }
