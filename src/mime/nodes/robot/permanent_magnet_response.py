"""PermanentMagnetResponseNode — torque and force from field on a permanent magnet.

Computes:
- Fixed moment in body frame: m_body = n_magnets * m_single * normalize(moment_axis)
- Rotate to lab frame: m_lab = R(q) @ m_body
- Magnetic torque: T = m_lab x B (m is already in A*m^2, no volume factor)
- Magnetic force: F = (grad B) @ m_lab (zero for uniform field)

No susceptibility tensor, no saturation, no volume multiplier.

Reference: de Boer et al. (2025), Wireless mechanical thrombus fragmentation.
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
)
from mime.core.quaternion import rotate_vector


@stability(StabilityLevel.EXPERIMENTAL)
class PermanentMagnetResponseNode(MimeNode):
    """Permanent-magnet response: torque and force from external field.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    n_magnets : int
        Number of permanent magnets in the robot body.
    m_single : float
        Magnetic moment per magnet [A*m^2].
    moment_axis : tuple[float, float, float]
        Direction of moment in body frame (will be normalised).

    Boundary Inputs
    ---------------
    field_vector : (3,)
        External B field in Tesla, from ExternalMagneticFieldNode.
    field_gradient : (3,3)
        Spatial gradient dB/dx in T/m.
    orientation : (4,)
        Robot quaternion [w,x,y,z] from RigidBodyNode.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-008",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="Permanent-magnet torque and force from external field interaction",
        governing_equations=(
            r"T = m_lab x B; "
            r"F = (grad B) @ m_lab; "
            r"m_body = n * m_single * axis_hat"
        ),
        discretization="Analytical — no discretisation",
        assumptions=(
            "Rigid permanent moment (no demagnetization)",
            "Moment axis fixed in body frame",
            "No temperature dependence of moment",
        ),
        limitations=(
            "No demagnetization effects",
            "No hysteresis",
            "No temperature-dependent moment",
        ),
        validated_regimes=(
            ValidatedRegime("field_strength_T", 0.0, 0.1, "T",
                            "NdFeB permanent magnet in mT-range field"),
        ),
        references=(
            Reference("deBoer2025",
                       "Wireless mechanical thrombus fragmentation"),
        ),
        hazard_hints=(
            "If external field exceeds coercivity (~1 T for NdFeB N45), "
            "demagnetization invalidates the constant-moment assumption",
            "Moment saturation is not modelled; the moment is assumed "
            "constant regardless of applied field strength",
        ),
        implementation_map={
            "m_body = n * m_single * axis_hat": (
                "mime.nodes.robot.permanent_magnet_response."
                "PermanentMagnetResponseNode.update"
            ),
            "T = m_lab x B": (
                "mime.nodes.robot.permanent_magnet_response."
                "PermanentMagnetResponseNode.update"
            ),
            "F = (grad B) @ m_lab": (
                "mime.nodes.robot.permanent_magnet_response."
                "PermanentMagnetResponseNode.update"
            ),
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ROBOT_BODY,
        biocompatibility=BiocompatibilityMeta(
            materials=("NdFeB N45",),
            iso_10993_class=BiocompatibilityClass.NOT_ASSESSED,
            biocompatibility_hazard_hints=(
                "NdFeB magnets are not inherently biocompatible; "
                "manufacturer must perform ISO 10993 evaluation. "
                "Typically encapsulated in biocompatible resin or parylene coating.",
            ),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        n_magnets: int = 1,
        m_single: float = 1.07e-3,
        moment_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
        **kwargs,
    ):
        super().__init__(
            name, timestep,
            n_magnets=n_magnets,
            m_single=m_single,
            moment_axis=moment_axis,
            **kwargs,
        )

    def initial_state(self) -> dict:
        return {
            "magnetization": jnp.zeros(3),     # A*m^2, lab frame
            "magnetic_torque": jnp.zeros(3),    # N.m, lab frame
            "magnetic_force": jnp.zeros(3),     # N, lab frame
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "field_vector": BoundaryInputSpec(
                shape=(3,), description="External B field [T]",
            ),
            "field_gradient": BoundaryInputSpec(
                shape=(3, 3), description="Spatial gradient dB/dx [T/m]",
            ),
            "orientation": BoundaryInputSpec(
                shape=(4,), default=jnp.array([1.0, 0.0, 0.0, 0.0]),
                description="Robot quaternion [w,x,y,z]",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        B_lab = boundary_inputs.get("field_vector", jnp.zeros(3))
        grad_B = boundary_inputs.get("field_gradient", jnp.zeros((3, 3)))
        q = boundary_inputs.get("orientation", jnp.array([1.0, 0.0, 0.0, 0.0]))

        n_magnets = self.params["n_magnets"]
        m_single = self.params["m_single"]
        moment_axis = jnp.array(self.params["moment_axis"], dtype=jnp.float32)

        # Normalise the moment axis (safe for zero vector)
        axis_norm = jnp.linalg.norm(moment_axis)
        axis_hat = moment_axis / jnp.maximum(axis_norm, 1e-30)

        # Fixed moment in body frame [A*m^2]
        m_body = n_magnets * m_single * axis_hat

        # Rotate moment to lab frame
        m_lab = rotate_vector(q, m_body)

        # Magnetic torque: T = m_lab x B (no volume factor — m is in A*m^2)
        torque = jnp.cross(m_lab, B_lab)

        # Magnetic force: F = (grad B) @ m_lab (zero for uniform field)
        force = grad_B @ m_lab

        return {
            "magnetization": m_lab,
            "magnetic_torque": torque,
            "magnetic_force": force,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "magnetic_torque": state["magnetic_torque"],
            "magnetic_force": state["magnetic_force"],
        }
