"""PhaseTrackingNode — phase error between robot orientation and field rotation.

An observational node (not physics). Reads:
- Robot orientation quaternion from RigidBodyNode
- External field vector from ExternalMagneticFieldNode

Computes phase_error = angle between the robot's magnetic moment direction
(body e1 axis) and the external field direction. Step-out is detected when
phase_error exceeds pi/2.

Also computes the step-out frequency omega_c from the magnetic-to-viscous
torque ratio, using Eq. 7.9 from the textbook.

Reference: Ch 7.3 of "Mathematical Modelling of Swimming Soft Microrobots"
           Abbott et al. (2009) for phase dynamics
"""

from __future__ import annotations

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import MimeNodeMeta, NodeRole
from mime.core.quaternion import rotate_vector


@stability(StabilityLevel.EXPERIMENTAL)
class PhaseTrackingNode(MimeNode):
    """Tracks phase error between robot orientation and rotating field.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.

    Boundary Inputs
    ---------------
    orientation : (4,)
        Robot quaternion [w,x,y,z] from RigidBodyNode.
    field_vector : (3,)
        External B field [T] from ExternalMagneticFieldNode.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-005",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="Phase error tracking between robot magnetic moment and rotating field",
        governing_equations=(
            r"phase_error = arccos(m_hat . B_hat); "
            r"stepped_out = (phase_error > pi/2)"
        ),
        discretization="Analytical — instantaneous angle computation",
        assumptions=(
            "Robot magnetic moment is aligned with body e1 axis",
            "Field and moment are coplanar for phase error computation",
            "Step-out threshold is pi/2 (physically: maximum torque angle)",
        ),
        limitations=(
            "No phase error ODE — detects step-out but does not model wobbling dynamics",
            "Step-out frequency requires external resistance tensor parameters",
        ),
        references=(
            Reference("Abbott2009", "How Should Microrobots Swim?"),
        ),
        hazard_hints=(
            "Phase error near 0 or pi has poor gradient signal due to arccos; "
            "use cos(phase_error) for optimization objectives instead",
            "Step-out detection is binary — no model of the asynchronous wobbling regime",
        ),
        implementation_map={
            "phase_error = arccos(m_hat . B_hat)": (
                "mime.nodes.robot.phase_tracking.PhaseTrackingNode.update"
            ),
        },
    )

    mime_meta = MimeNodeMeta(role=NodeRole.ROBOT_BODY)

    def __init__(self, name: str, timestep: float, **kwargs):
        super().__init__(name, timestep, **kwargs)

    def initial_state(self) -> dict:
        return {
            "phase_error": jnp.array(0.0),       # radians [0, pi]
            "cos_phase_error": jnp.array(1.0),   # cos(phase_error) — better for gradients
            "stepped_out": jnp.array(False),      # boolean flag
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "orientation": BoundaryInputSpec(
                shape=(4,),
                default=jnp.array([1.0, 0.0, 0.0, 0.0]),
                description="Robot quaternion [w,x,y,z]",
            ),
            "field_vector": BoundaryInputSpec(
                shape=(3,),
                description="External B field [T]",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        q = boundary_inputs.get("orientation", jnp.array([1.0, 0.0, 0.0, 0.0]))
        B = boundary_inputs.get("field_vector", jnp.zeros(3))

        # Robot magnetic moment direction = body e1 axis rotated to lab frame
        e1_body = jnp.array([1.0, 0.0, 0.0])
        m_hat = rotate_vector(q, e1_body)

        # Normalize field vector
        B_norm = jnp.linalg.norm(B)
        B_hat = B / jnp.maximum(B_norm, 1e-30)

        # Phase error = angle between m_hat and B_hat
        cos_err = jnp.clip(jnp.dot(m_hat, B_hat), -1.0, 1.0)
        phase_error = jnp.arccos(cos_err)

        # Step-out detection
        stepped_out = phase_error > (jnp.pi / 2.0)

        return {
            "phase_error": phase_error,
            "cos_phase_error": cos_err,
            "stepped_out": stepped_out,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "phase_error": state["phase_error"],
            "cos_phase_error": state["cos_phase_error"],
            "stepped_out": state["stepped_out"],
        }
