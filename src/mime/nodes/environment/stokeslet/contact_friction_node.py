"""Wall-contact rolling/sliding friction — calibration stub.

When a rigid microrobot is pressed against the vessel wall by gravity
or magnetic force, a **tangential contact friction** opposes swimming
in addition to the purely hydrodynamic lubrication drag
(:class:`LubricationCorrectionNode`). The friction force depends on
robot surface material, vessel lining, coating, and fluid conditions
and must be calibrated from phantom-vessel experiments — no
first-principles expression is available for resin microrobots in
silicone tubes.

This node is an **architectural stub**: it carries the correct
input/output ports for the downstream-coupled friction equation, but
outputs zero force until ``mu_roll`` is set to a measured value.

Future physics (when calibrated)
--------------------------------
    F_friction = -μ_roll · F_wall_normal · v̂_tangential

where F_wall_normal comes from LubricationCorrectionNode's
``wall_normal_force_coef`` × V_normal (at quasi-equilibrium:
F_wall_normal ≈ external force pressing the robot onto the wall,
e.g. the radial component of gravity).

References
----------
The friction model is taken to be Coulomb-type; see Popov (2010)
*Contact Mechanics and Friction* Ch. 10 for the surface-independent
form used here. Calibration procedure: run a known-force pull-test on
a fabricated robot segment against a silicone vessel phantom and
regress F_tangential vs F_normal.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec, BoundaryFluxSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference, UQReadiness,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole,
    AnatomicalRegimeMeta, AnatomicalCompartment, FlowRegime,
)

logger = logging.getLogger(__name__)


@stability(StabilityLevel.EXPERIMENTAL)
class ContactFrictionNode(MimeNode):
    """Stub: wall-contact rolling/sliding friction.

    Outputs zero friction until ``mu_roll`` is set from calibration.
    See module docstring for the planned model.

    Parameters
    ----------
    mu_roll : float
        Rolling friction coefficient. Default 0.0 = no friction
        (stub behaviour). Populate from phantom pull-tests.
    external_normal_force_N : float
        Steady radial force pressing the robot onto the wall [N]. If
        the robot settles under gravity in a horizontal vessel, this
        equals the buoyancy-corrected weight.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-014",
        algorithm_version="0.1.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Stub for wall-contact rolling/sliding friction. Outputs "
            "zero friction until μ_roll is calibrated from phantom "
            "experiments. Interface prepared so downstream nodes can "
            "be wired now without refactoring later."
        ),
        governing_equations=(
            "F_friction = -μ_roll · F_wall_normal · v̂_tangential "
            "(Coulomb; disabled when μ_roll = 0)."
        ),
        discretization="Pointwise force at body nearest-wall point.",
        assumptions=(
            "Robot in persistent wall contact (δ → 0).",
            "Coulomb friction — independent of contact-area and "
            "velocity magnitude.",
            "Tangential velocity is the projection of body velocity "
            "onto the plane orthogonal to the wall normal.",
        ),
        limitations=(
            "μ_roll is not calibrated — this node is a stub. "
            "Any non-zero friction result requires experimental "
            "calibration that does not yet exist.",
            "No stick-slip transition modelled.",
            "Ignores velocity-weakening / velocity-strengthening "
            "effects.",
        ),
        validated_regimes=(
            ValidatedRegime("mu_roll", 0.0, 0.0, "",
                             "Calibration missing; kept at 0."),
        ),
        references=(
            Reference("Popov2010",
                       "Popov (2010) Contact Mechanics and Friction "
                       "Ch. 10 — Coulomb model"),
        ),
        uq_readiness=UQReadiness.NOT_READY,
        deprecation_notice=(
            "Stub node. Will be promoted to PROVISIONAL once "
            "μ_roll is measured from phantom pull-tests."
        ),
        hazard_hints=(
            "Using this node with mu_roll ≠ 0 before calibration "
            "gives unvalidated drag; flag calibration_status in the "
            "NodeMeta before any downstream use.",
        ),
        implementation_map={
            "Coulomb friction (disabled for μ=0)":
                "mime.nodes.environment.stokeslet.contact_friction_node."
                "ContactFrictionNode.update",
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ENVIRONMENT,
        anatomical_regimes=(
            AnatomicalRegimeMeta(
                compartment=AnatomicalCompartment.BLOOD,
                anatomy="wall-contacting regions",
                flow_regime=FlowRegime.STAGNANT,
                notes=("Stub — requires experimental μ_roll for any "
                        "vessel/robot material combination."),
            ),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        mu_roll: float = 0.0,
        external_normal_force_N: float = 0.0,
        **kwargs,
    ):
        super().__init__(name, timestep, **kwargs)
        self._mu_roll = float(mu_roll)
        self._F_n_ext = float(external_normal_force_N)
        if self._mu_roll != 0.0:
            logger.warning(
                "%s: μ_roll=%.3f set without calibration. Ensure this "
                "value comes from a validated phantom pull-test.",
                name, self._mu_roll,
            )

    @property
    def requires_halo(self) -> bool:
        return False

    def initial_state(self) -> dict:
        return {
            "friction_force": jnp.zeros(3),
            "friction_active": jnp.asarray(0.0),
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "wall_normal_force_coef": BoundaryInputSpec(
                shape=(), default=jnp.asarray(0.0),
                description=(
                    "Lubrication normal-stiffness w(δ)·R_nn [N·s/m]; "
                    "combined with normal velocity gives F_n."
                ),
            ),
            "body_velocity": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Body centre velocity [m/s]",
            ),
            "robot_position": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Robot centre [m] — used to compute n̂",
            ),
        }

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        return {
            "friction_force": BoundaryFluxSpec(
                shape=(3,), description="Wall-contact friction force [N]",
                output_units="N",
            ),
            "friction_active": BoundaryFluxSpec(
                shape=(), description="1.0 if |friction| > 0, else 0.0",
                output_units="bool",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        if self._mu_roll == 0.0:
            # Stub behaviour — zero friction, zero flags
            return {
                "friction_force": jnp.zeros(3),
                "friction_active": jnp.asarray(0.0),
            }

        U = boundary_inputs.get("body_velocity", jnp.zeros(3))
        pos = boundary_inputs.get("robot_position", jnp.zeros(3))

        radial = jnp.array([pos[0], pos[1], 0.0])
        radial_norm = jnp.linalg.norm(radial)
        n_hat = jnp.where(
            radial_norm > 1e-10,
            radial / jnp.maximum(radial_norm, 1e-12),
            jnp.array([1.0, 0.0, 0.0]),
        )

        U_t = U - jnp.dot(U, n_hat) * n_hat
        U_t_norm = jnp.linalg.norm(U_t)
        t_hat = jnp.where(
            U_t_norm > 1e-10,
            U_t / jnp.maximum(U_t_norm, 1e-12),
            jnp.zeros(3),
        )

        # Coulomb: |F_fric| = μ · F_n, opposite to v̂_tangential
        F_fric = -self._mu_roll * self._F_n_ext * t_hat
        return {
            "friction_force": F_fric,
            "friction_active": (U_t_norm > 1e-8).astype(jnp.float32),
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "friction_force": state["friction_force"],
            "friction_active": state["friction_active"],
        }
