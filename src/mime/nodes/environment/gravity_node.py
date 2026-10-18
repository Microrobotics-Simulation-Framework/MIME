"""GravityNode — constant buoyancy-corrected gravitational force.

A trivial node that outputs F_gravity = (ρ_robot - ρ_fluid) × V_robot × g
in the world frame, with configurable direction (default y-down).

Calibrated defaults match the de Jongh UMR:
- Body: Phrozen Aqua-Gray resin, ρ ≈ 1180 kg/m³
- Embedded magnets: 2 × (1 mm)³ N45 NdFeB, ρ ≈ 7500 kg/m³
- Effective density ≈ 1410 kg/m³ (with magnet volume weighting)
- Δρ vs water: 410 kg/m³
- UMR volume ≈ π × R_cyl² × L (≈ 5.7e-8 m³ for L=7.47 mm, R_cyl=1.56 mm)
- F_gravity ≈ 2.3e-4 N

Output: `gravity_force` (3,) in Newtons. Wire additively to the
MLPResistanceNode's `external_force` input.
"""

from __future__ import annotations

import jax.numpy as jnp

from maddening.core.node import BoundaryFluxSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole,
    AnatomicalRegimeMeta, AnatomicalCompartment, FlowRegime,
)


@stability(StabilityLevel.EXPERIMENTAL)
class GravityNode(MimeNode):
    """Constant buoyancy-corrected gravitational force.

    Parameters
    ----------
    name : str
    timestep : float
    delta_rho_kg_m3 : float
        (ρ_robot - ρ_fluid) in kg/m³. Default 410 (de Jongh UMR in water).
    volume_m3 : float
        Robot volume in m³. Default 5.7e-8 (de Jongh FL-9).
    g : float
        Gravitational acceleration magnitude. Default 9.81.
    direction : tuple
        Unit vector of gravity in world frame. Default (0, -1, 0) = y-down.
        Use (0, 0, -1) if z is vertical.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-012",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Constant buoyancy-corrected gravitational force output for "
            "suspended microrobot bodies."
        ),
        governing_equations=(
            "F = (ρ_robot - ρ_fluid) · V · g · ĝ  (Archimedes)"
        ),
        discretization="Time-invariant scalar force; no integration required.",
        assumptions=(
            "Uniform robot density (effective bulk value).",
            "Incompressible, uniform-density fluid.",
            "Vertical gravity direction is fixed in the world frame.",
        ),
        limitations=(
            "Does not model density stratification or thermal buoyancy.",
            "No coupling to deformation of flexible bodies (use FlexibleBody "
            "for distributed weight).",
        ),
        validated_regimes=(
            ValidatedRegime("delta_rho_kg_m3", -2000.0, 10000.0, "kg/m³",
                             "Density contrast range for UMR-in-water "
                             "through dense metal-in-blood scenarios."),
        ),
        references=(
            Reference("deJongh2025",
                       "de Jongh et al. (2025) UMR density calibration"),
        ),
        hazard_hints=(
            "Direction convention mismatch (y-down vs z-down) silently "
            "applies gravity on the wrong axis — verify against coordinate "
            "frame of downstream drag/pose nodes.",
        ),
        implementation_map={
            "Archimedes buoyancy force":
                "mime.nodes.environment.gravity_node.GravityNode._force",
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ENVIRONMENT,
        anatomical_regimes=(
            AnatomicalRegimeMeta(
                compartment=AnatomicalCompartment.BLOOD,
                anatomy="any vascular segment",
                flow_regime=FlowRegime.STAGNANT,
                notes="Body-force term; independent of local flow.",
            ),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        delta_rho_kg_m3: float,  # material-valued: no default
        volume_m3: float = 5.7e-8,
        g: float = 9.81,
        direction: tuple = (0.0, -1.0, 0.0),
        **kwargs,
    ):
        super().__init__(
            name, timestep,
            delta_rho_kg_m3=float(delta_rho_kg_m3),
            volume_m3=float(volume_m3),
            g=float(g),
            direction=tuple(float(c) for c in direction),
            **kwargs,
        )

    @staticmethod
    def _force(p) -> jnp.ndarray:
        direction = jnp.asarray(p["direction"])
        direction_norm = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-10)
        return direction_norm * p["delta_rho_kg_m3"] * p["volume_m3"] * p["g"]

    def initial_state(self) -> dict:
        return {"gravity_force": self._force(self.params)}

    def boundary_input_spec(self) -> dict:
        return {}

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        return {
            "gravity_force": BoundaryFluxSpec(
                shape=(3,),
                description="Buoyancy-corrected gravitational force [N]",
                output_units="N",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float,
               *, params=None) -> dict:
        # Constant force, recomputed from the (possibly written) parameters.
        p = self.params if params is None else {**self.params, **params}
        return {"gravity_force": self._force(p)}

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        return {"gravity_force": state["gravity_force"]}
