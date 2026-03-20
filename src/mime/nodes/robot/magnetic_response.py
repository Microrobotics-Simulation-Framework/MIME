"""MagneticResponseNode — torque and force from magnetic field on a soft-magnet body.

Computes:
- Induced magnetization: m = (1/mu_0) * chi_a @ B (in body frame)
- Magnetic torque: T = v * (m x B) (in lab frame)
- Magnetic force: F = v * (m . grad)B (in lab frame)

With saturation clipping: |m| <= m_sat.

The susceptibility tensor chi_a is diagonal in the body frame with
entries [1/n_axi, 1/n_rad, 1/n_rad] where n_axi + 2*n_rad = 1.
The field B must be rotated into the body frame for the magnetization
calculation, then the torque/force are computed in the lab frame.

Reference: Ch 3 and Ch 10 of "Mathematical Modelling of Swimming Soft Microrobots"
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
from mime.core.quaternion import rotate_vector_inverse, rotate_vector


MU_0 = 4.0 * jnp.pi * 1e-7  # Permeability of free space [T.m/A]


@stability(StabilityLevel.EXPERIMENTAL)
class MagneticResponseNode(MimeNode):
    """Soft-magnetic response: torque and force from external field.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    volume_m3 : float
        Volume of magnetic material [m^3].
    n_axi : float
        Axial demagnetization factor (along body e1 axis).
        Must satisfy n_axi + 2*n_rad = 1.
    n_rad : float
        Radial demagnetization factor. Default computed from n_axi.
    m_sat : float
        Saturation magnetization [A/m]. 0 = no saturation limit.

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
        algorithm_id="MIME-NODE-002",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="Soft-magnetic torque and force from external field interaction",
        governing_equations=(
            r"m = (1/mu_0) * chi_a @ B_body; "
            r"T = v * (m_lab x B_lab); "
            r"F = v * (m_lab . grad)B"
        ),
        discretization="Analytical — no discretisation",
        assumptions=(
            "Linear magnetization below saturation (m < m_sat)",
            "No hysteresis or remnant magnetization (ideal soft-magnet)",
            "Susceptibility tensor is diagonal in body frame",
            "Demagnetization factors satisfy n_axi + 2*n_rad = 1 (ellipsoidal body)",
        ),
        limitations=(
            "Linear approximation fails above saturation magnetization",
            "No hysteresis modelling — cannot capture field-history effects",
            "Assumes ellipsoidal body shape for demagnetization factors",
        ),
        validated_regimes=(
            ValidatedRegime("field_strength_T", 0.0, 0.1, "T",
                            "Below saturation for Co80Ni20"),
        ),
        references=(
            Reference("Abbott2009", "How Should Microrobots Swim?"),
        ),
        hazard_hints=(
            "If B exceeds saturation, linear approximation m ~ B fails; "
            "jnp.clip caps magnetization but torque calculation becomes approximate",
            "Demagnetization factors must satisfy n_axi + 2*n_rad = 1; "
            "violation produces unphysical magnetization",
        ),
        implementation_map={
            "m = chi_a @ B / mu_0": (
                "mime.nodes.robot.magnetic_response."
                "MagneticResponseNode.update"
            ),
            "T = v * (m x B)": (
                "mime.nodes.robot.magnetic_response."
                "MagneticResponseNode.update"
            ),
            "F = v * (m . grad)B": (
                "mime.nodes.robot.magnetic_response."
                "MagneticResponseNode.update"
            ),
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ROBOT_BODY,
        biocompatibility=BiocompatibilityMeta(
            materials=("Co80Ni20",),
            iso_10993_class=BiocompatibilityClass.NOT_ASSESSED,
            biocompatibility_hazard_hints=(
                "Default material (Co80Ni20) biocompatibility not assessed; "
                "manufacturer must perform ISO 10993 evaluation",
            ),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        volume_m3: float = 1e-15,
        n_axi: float = 0.2,
        n_rad: float | None = None,
        m_sat: float = 0.0,
        **kwargs,
    ):
        if n_rad is None:
            n_rad = (1.0 - n_axi) / 2.0
        super().__init__(
            name, timestep,
            volume_m3=volume_m3,
            n_axi=n_axi,
            n_rad=n_rad,
            m_sat=m_sat,
            **kwargs,
        )

    def initial_state(self) -> dict:
        return {
            "magnetization": jnp.zeros(3),     # A/m, lab frame
            "magnetic_torque": jnp.zeros(3),   # N.m, lab frame
            "magnetic_force": jnp.zeros(3),    # N, lab frame
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

        v = self.params["volume_m3"]
        n_axi = self.params["n_axi"]
        n_rad = self.params["n_rad"]
        m_sat = self.params["m_sat"]

        # Susceptibility tensor in body frame: chi_a = diag(1/n_axi, 1/n_rad, 1/n_rad)
        chi_diag = jnp.array([1.0 / n_axi, 1.0 / n_rad, 1.0 / n_rad])

        # Rotate B into body frame
        B_body = rotate_vector_inverse(q, B_lab)

        # Magnetization in body frame: m = (1/mu_0) * chi_a @ B
        m_body = (1.0 / MU_0) * chi_diag * B_body

        # Saturation clipping
        if m_sat > 0:
            m_mag = jnp.linalg.norm(m_body)
            scale = jnp.where(m_mag > m_sat, m_sat / jnp.maximum(m_mag, 1e-30), 1.0)
            m_body = m_body * scale

        # Rotate magnetization back to lab frame
        m_lab = rotate_vector(q, m_body)

        # Magnetic torque: T = v * (m x B) in lab frame
        torque = v * jnp.cross(m_lab, B_lab)

        # Magnetic force: F = v * (m . grad)B in lab frame
        # F_i = v * sum_j m_j * (dB_i/dx_j) = v * grad_B @ m
        force = v * grad_B @ m_lab

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
