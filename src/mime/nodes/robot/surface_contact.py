"""SurfaceContactNode — near-wall drag corrections and contact forces.

Implements Brenner wall corrections for a sphere near a plane wall:
- Perpendicular: F_perp = 6*pi*mu*a*V_perp * (1 + 9a/(8h))  [Brenner 1961]
- Parallel: F_par = 6*pi*mu*a*V_par / (1 - 9a/(16h))  [Goldman-Cox-Brenner 1967]

Also provides a soft penalty-based contact force to prevent wall penetration.

Reference: CSF_addon.md Eq. D.11, D.12; Brenner (1961); Goldman, Cox & Brenner (1967)
"""

from __future__ import annotations

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import MimeNodeMeta, NodeRole


def brenner_correction_perpendicular(a: float, h: jnp.ndarray) -> jnp.ndarray:
    """Brenner wall correction factor for perpendicular translation.

    Uses the leading-order expansion: correction ≈ 1 + 9a/(8h).
    This is numerically stable for all h > 0 (unlike the reciprocal form
    1/(1-9a/8h) which diverges at h = 9a/8).

    Parameters
    ----------
    a : float
        Sphere radius [m].
    h : jnp.ndarray
        Centre-to-wall distance [m]. Must be > a.
    """
    h_safe = jnp.maximum(h, a * 1.5)  # valid regime: h/a > 1.5
    return 1.0 + 9.0 * a / (8.0 * h_safe)


def brenner_correction_parallel(a: float, h: jnp.ndarray) -> jnp.ndarray:
    """Goldman-Cox-Brenner correction for parallel translation.

    Uses leading-order expansion: correction ≈ 1 + 9a/(16h).

    Parameters
    ----------
    a : float
        Sphere radius [m].
    h : jnp.ndarray
        Centre-to-wall distance [m]. Must be > a.
    """
    h_safe = jnp.maximum(h, a * 1.5)
    return 1.0 + 9.0 * a / (16.0 * h_safe)


def penalty_contact_force(
    position: jnp.ndarray,
    wall_position: float,
    wall_normal: jnp.ndarray,
    robot_radius: float,
    stiffness: float = 1e-6,
) -> jnp.ndarray:
    """Soft penalty force preventing wall penetration.

    Parameters
    ----------
    position : (3,) robot position
    wall_position : float, wall location along normal direction
    wall_normal : (3,) outward normal of the wall
    robot_radius : float [m]
    stiffness : float [N/m], penalty spring stiffness

    Returns
    -------
    (3,) contact force [N]. Zero if not in contact.
    """
    # Signed distance from robot centre to wall (positive = inside fluid)
    d = jnp.dot(position, wall_normal) - wall_position
    # Gap = distance from robot surface to wall
    gap = d - robot_radius
    # Penalty force: linear spring, only when gap < 0 (penetrating)
    penetration = jnp.maximum(-gap, 0.0)
    return stiffness * penetration * wall_normal


@stability(StabilityLevel.EXPERIMENTAL)
class SurfaceContactNode(MimeNode):
    """Near-wall drag corrections and contact forces for a sphere.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    robot_radius_m : float
        Effective hydrodynamic radius [m].
    wall_position : float
        Wall location along the wall normal axis [m].
    wall_normal_axis : int
        Which axis the wall normal points along (0=x, 1=y, 2=z). Default: 2 (z).
    wall_side : int
        +1 if wall is at +z side, -1 if at -z side. Default: -1 (floor).
    contact_stiffness : float
        Penalty spring stiffness [N/m] for contact force.
    fluid_viscosity_pa_s : float
        Dynamic viscosity [Pa.s].

    Boundary Inputs
    ---------------
    position : (3,) robot position from RigidBodyNode
    velocity : (3,) robot velocity from RigidBodyNode
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-007",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="Near-wall Brenner/Goldman-Cox-Brenner drag corrections + penalty contact",
        governing_equations=(
            r"F_perp_corrected = F_stokes / (1 - 9a/(8h)); "
            r"F_par_corrected = F_stokes / (1 - 9a/(16h)); "
            r"F_contact = k * max(-gap, 0) * n_wall"
        ),
        discretization="Analytical — closed-form evaluation",
        assumptions=(
            "Sphere near an infinite plane wall",
            "Stokes regime (Re << 1)",
            "h/a > 1.01 (correction series diverges at contact)",
            "Wall is flat — no curvature corrections",
        ),
        limitations=(
            "Corrections are first-order truncation of infinite series",
            "Penalty contact force is not physical — it is a numerical regularisation",
            "No adhesion modelling (van der Waals, electrostatic)",
            "Single wall only — no channel confinement from two walls",
        ),
        validated_regimes=(
            ValidatedRegime("h_over_a", 1.01, 100.0, "",
                            "Wall correction valid for h/a > 1"),
        ),
        references=(
            Reference("Purcell1977", "Life at Low Reynolds Number"),
        ),
        hazard_hints=(
            "At h/a < 1.01, corrections diverge; jnp.maximum clamps h",
            "Penalty contact stiffness introduces artificial dynamics — "
            "timestep must be small enough to resolve the contact spring",
            "DIFFERENTIABILITY-LIMITED: penalty force has a kink at gap=0; "
            "jax.grad through contact events produces unreliable gradients",
        ),
        implementation_map={
            "Brenner perpendicular": (
                "mime.nodes.robot.surface_contact.brenner_correction_perpendicular"
            ),
            "Goldman-Cox-Brenner parallel": (
                "mime.nodes.robot.surface_contact.brenner_correction_parallel"
            ),
            "Penalty contact": (
                "mime.nodes.robot.surface_contact.penalty_contact_force"
            ),
        },
    )

    mime_meta = MimeNodeMeta(role=NodeRole.ROBOT_BODY)

    def __init__(
        self,
        name: str,
        timestep: float,
        robot_radius_m: float = 100e-6,
        wall_position: float = 0.0,
        wall_normal_axis: int = 2,
        wall_side: int = -1,
        contact_stiffness: float = 1e-6,
        fluid_viscosity_pa_s: float = 8.5e-4,
        **kwargs,
    ):
        super().__init__(
            name, timestep,
            robot_radius_m=robot_radius_m,
            wall_position=wall_position,
            wall_normal_axis=wall_normal_axis,
            wall_side=wall_side,
            contact_stiffness=contact_stiffness,
            fluid_viscosity_pa_s=fluid_viscosity_pa_s,
            **kwargs,
        )

    def initial_state(self) -> dict:
        return {
            "wall_correction_perp": jnp.array(1.0),
            "wall_correction_par": jnp.array(1.0),
            "contact_force": jnp.zeros(3),
            "gap_distance": jnp.array(1.0),  # h/a ratio
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "position": BoundaryInputSpec(
                shape=(3,), description="Robot position [m]",
            ),
            "velocity": BoundaryInputSpec(
                shape=(3,), description="Robot velocity [m/s]",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        a = self.params["robot_radius_m"]
        wall_pos = self.params["wall_position"]
        axis = self.params["wall_normal_axis"]
        side = self.params["wall_side"]
        k = self.params["contact_stiffness"]

        pos = boundary_inputs.get("position", jnp.zeros(3))

        # Wall normal vector
        n_wall = jnp.zeros(3).at[axis].set(float(side))

        # Distance from robot centre to wall
        h = jnp.abs(pos[axis] - wall_pos)

        # Brenner corrections
        corr_perp = brenner_correction_perpendicular(a, h)
        corr_par = brenner_correction_parallel(a, h)

        # Contact force (penalty)
        f_contact = penalty_contact_force(pos, wall_pos, -n_wall, a, k)

        return {
            "wall_correction_perp": corr_perp,
            "wall_correction_par": corr_par,
            "contact_force": f_contact,
            "gap_distance": h / a,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "contact_force": state["contact_force"],
            "wall_correction_perp": state["wall_correction_perp"],
            "wall_correction_par": state["wall_correction_par"],
        }
