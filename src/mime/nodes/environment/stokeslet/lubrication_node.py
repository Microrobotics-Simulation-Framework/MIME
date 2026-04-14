"""Analytical near-wall lubrication correction for confined Stokes drag.

Regularised Stokeslet BEM (``mime.nodes.environment.stokeslet``) smooths
the Green's function at scales ≲ ε, so it cannot reproduce the singular
near-wall hydrodynamic forces that dominate when a rigid body is within
a gap δ ≪ a_eff of the vessel wall. The standard textbook treatment is
to add analytical asymptotic lubrication corrections on top of the BEM
solution — the framework formalised by Brady & Bossis (1988) as
Stokesian Dynamics.

This node takes the 6×6 resistance matrix from
``MLPResistanceNode`` (or any other R source) and adds the sphere-plane
lubrication terms of Goldman, Cox & Brenner (1967) + Cox & Brenner
(1967). All coefficients are textbook asymptotics — zero free
parameters.

Physics
-------
Near-wall corrections added to R (body represented by its nearest point
as a sphere of effective radius a_eff):

* **Normal translation**:   R_nn_lub = 6πμ a_eff² / δ
      (Cox & Brenner 1967, leading singularity)
* **Tangential translation**: R_tt_lub = 6πμ a_eff (8/15) ln(a_eff/δ)
      (Goldman, Cox & Brenner 1967a)
* **Rotation (axis ∥ wall)**: R_rr_lub = 8πμ a_eff³ (2/5) ln(a_eff/δ)
      (Goldman, Cox & Brenner 1967a)

A Brady-Bossis style **smooth blending weight** w(δ) = exp(−δ/ε) avoids
double-counting the far-field (BEM already contains the regularised
analogue at larger gaps). The correction is negligible at δ > 3ε and
full strength at δ < ε.

Blocks affected
---------------
In the wall-aligned frame (n̂ = radial unit from vessel axis to robot,
ẑ parallel to vessel axis, t̂₂ = n̂ × ẑ):

* Translation–translation block (R[:3, :3]): rank-3 update adding
  R_nn along n̂n̂ᵀ and R_tt along ẑẑᵀ + t̂₂t̂₂ᵀ.
* Rotation–rotation block (R[3:, 3:]): rank-2 update adding R_rr along
  ẑẑᵀ + t̂₂t̂₂ᵀ.

Translation–rotation coupling from wall proximity is set to zero in
this version — the effect enters at the same asymptotic order but
requires a separate coupled-motion calibration. See
:class:`ContactFrictionNode` for the tangential-contact companion.

References
----------
* Goldman, Cox & Brenner (1967a) — Slow viscous motion of a sphere
  parallel to a plane wall (translation), *Chem. Eng. Sci.* 22, 637.
* Goldman, Cox & Brenner (1967b) — Rotation of a sphere near a plane
  wall, *Chem. Eng. Sci.* 22, 653.
* Cox & Brenner (1967) — Slow viscous motion of a sphere perpendicular
  to a plane wall — II. Small gap widths, *Chem. Eng. Sci.* 22, 1753.
* Brady & Bossis (1988) — Stokesian Dynamics,
  *Annu. Rev. Fluid Mech.* 20, 111.
* Kim & Karrila (2005) — Microhydrodynamics, Ch. 7 (wall-resistance
  functions).
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
class LubricationCorrectionNode(MimeNode):
    """Analytical sphere-wall lubrication correction on top of BEM/MLP R.

    Parameters
    ----------
    name, timestep : standard MimeNode.
    R_ves_mm : float
        Vessel radius [mm]. Wall is assumed locally flat at δ ≪ R_ves.
    R_max_body_mm : float
        Non-dimensional body radius times R_cyl_UMR — i.e. the farthest
        point of the UMR from its axis. Default 2.08 = (1 + ε_mod) ×
        R_cyl_UMR for the de Jongh geometry.
    mu_Pa_s : float
        Fluid viscosity [Pa·s]. Default 1e-3 (water).
    epsilon_mm : float
        Blending scale for w(δ) = exp(−δ/ε). Default 0.2 mm, matching
        the BEM mesh-spacing regularisation.
    a_eff_mm : float
        Effective radius at the nearest point [mm]. Default 2.08, same
        as R_max_body_mm — a spherical cap with that curvature
        approximates the convex envelope locally.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-013",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Analytical near-wall lubrication correction for confined "
            "Stokes resistance matrices. Adds the singular sphere-wall "
            "forces that regularised Stokeslet BEM cannot resolve."
        ),
        governing_equations=(
            "R_corrected = R_BEM + w(δ) · R_lub, w(δ) = exp(-δ/ε). "
            "R_lub_nn = 6πμ a² / δ  (Cox & Brenner 1967). "
            "R_lub_tt = 6πμ a (8/15) ln(a/δ) "
            "(Goldman, Cox & Brenner 1967a). "
            "R_lub_rr = 8πμ a³ (2/5) ln(a/δ) "
            "(Goldman, Cox & Brenner 1967a)."
        ),
        discretization=(
            "Pointwise wall-aligned frame; rank-3 translation-translation "
            "update + rank-2 rotation-rotation update added to R."
        ),
        assumptions=(
            "Wall locally flat at δ ≪ R_ves "
            "(curvature correction is O(δ/R_ves)).",
            "Body locally spherical at its nearest point; effective "
            "radius a_eff approximates the convex envelope curvature.",
            "Leading-order asymptotics only — higher-order terms "
            "(const + O(δ/a)) are absorbed into the BEM far-field.",
            "No translation-rotation coupling added (kept for a "
            "future calibrated version).",
        ),
        limitations=(
            "Zero free parameters — cannot absorb rolling/sliding "
            "contact mechanics. Pair with ContactFrictionNode "
            "(MIME-NODE-014) once experimental μ_roll is available.",
            "Blending weight ε is a model hyperparameter; default "
            "matches BEM regularisation but should be tuned for other "
            "mesh densities.",
            "Assumes rigid body. Deformation under near-wall stress "
            "would modify a_eff dynamically.",
        ),
        validated_regimes=(
            ValidatedRegime("gap_over_radius", 0.0, 1.0, "",
                             "δ/a_eff range in which the leading "
                             "asymptotic is considered accurate."),
        ),
        references=(
            Reference("GoldmanCoxBrenner1967a",
                       "Goldman, Cox & Brenner (1967) Chem. Eng. Sci. "
                       "22:637 — tangential translation"),
            Reference("GoldmanCoxBrenner1967b",
                       "Goldman, Cox & Brenner (1967) Chem. Eng. Sci. "
                       "22:653 — rotation"),
            Reference("CoxBrenner1967",
                       "Cox & Brenner (1967) Chem. Eng. Sci. 22:1753 — "
                       "normal approach"),
            Reference("BradyBossis1988",
                       "Brady & Bossis (1988) Annu. Rev. Fluid Mech. "
                       "20:111 — Stokesian Dynamics framework"),
            Reference("KimKarrila2005",
                       "Kim & Karrila (2005) Microhydrodynamics Ch. 7"),
        ),
        uq_readiness=UQReadiness.PARAMETER_SWEEP,
        hazard_hints=(
            "δ → 0 produces R_nn → ∞; the jnp.maximum(δ, δ_floor) guard "
            "prevents numerical blow-up. Verify δ_floor is small enough "
            "to not bias contact dynamics.",
            "Blending weight ε larger than mean gap over the trajectory "
            "blends the correction into regimes BEM already covers, "
            "double-counting drag.",
        ),
        implementation_map={
            "Normal lubrication term":
                "mime.nodes.environment.stokeslet.lubrication_node."
                "LubricationCorrectionNode._lub_R_in_wall_frame",
            "Wall-aligned frame + rank updates":
                "mime.nodes.environment.stokeslet.lubrication_node."
                "LubricationCorrectionNode.update",
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ENVIRONMENT,
        anatomical_regimes=(
            AnatomicalRegimeMeta(
                compartment=AnatomicalCompartment.BLOOD,
                anatomy="any vessel segment where robot contacts the wall",
                flow_regime=FlowRegime.STAGNANT,
                viscosity_min_pa_s=1e-3, viscosity_max_pa_s=4e-3,
                notes="Correction is relevant once δ/a_eff ≲ 1.",
            ),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        R_ves_mm: float,
        R_max_body_mm: float = 2.08,
        mu_Pa_s: float = 1e-3,
        epsilon_mm: float = 0.2,
        a_eff_mm: float = 2.08,
        delta_floor_mm: float = 1e-6,
        **kwargs,
    ):
        super().__init__(name, timestep, mu=mu_Pa_s, **kwargs)
        self._R_ves_m = float(R_ves_mm) * 1e-3
        self._R_max_body_m = float(R_max_body_mm) * 1e-3
        self._mu = float(mu_Pa_s)
        self._epsilon_m = float(epsilon_mm) * 1e-3
        self._a_eff_m = float(a_eff_mm) * 1e-3
        self._delta_floor_m = float(delta_floor_mm) * 1e-3

    @property
    def requires_halo(self) -> bool:
        return False

    def initial_state(self) -> dict:
        return {
            "corrected_resistance_matrix": jnp.eye(6),
            "drag_force": jnp.zeros(3),
            "drag_torque": jnp.zeros(3),
            "min_gap": jnp.asarray(self._R_ves_m - self._R_max_body_m),
            "wall_normal_force_coef": jnp.asarray(0.0),
            "lubrication_active": jnp.asarray(0.0),
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "resistance_matrix": BoundaryInputSpec(
                shape=(6, 6), default=jnp.eye(6),
                description="Upstream R (from MLPResistanceNode or BEM) [SI]",
            ),
            "robot_position": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Robot centre [m]",
            ),
            "body_velocity": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Translational velocity [m/s]",
            ),
            "body_angular_velocity": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Angular velocity [rad/s]",
            ),
            "background_velocity": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Ambient fluid velocity at robot centre [m/s]",
            ),
        }

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        return {
            "corrected_resistance_matrix": BoundaryFluxSpec(
                shape=(6, 6), description="R + w(δ)·R_lub [SI]",
                output_units="mixed",
            ),
            "drag_force": BoundaryFluxSpec(
                shape=(3,), description="Drag force from corrected R [N]",
                output_units="N",
            ),
            "drag_torque": BoundaryFluxSpec(
                shape=(3,), description="Drag torque from corrected R [N·m]",
                output_units="N*m",
            ),
            "min_gap": BoundaryFluxSpec(
                shape=(), description="Minimum body-wall gap δ [m]",
                output_units="m",
            ),
            "wall_normal_force_coef": BoundaryFluxSpec(
                shape=(), description=(
                    "Normal lubrication stiffness w(δ)·R_nn [N·s/m]. "
                    "Multiplied by the normal velocity gives the "
                    "repulsive wall force — consumed by "
                    "ContactFrictionNode."
                ),
                output_units="N*s/m",
            ),
            "lubrication_active": BoundaryFluxSpec(
                shape=(), description="1.0 if w(δ) > 0.01, else 0.0",
                output_units="bool",
            ),
        }

    # ── Pure JAX core ────────────────────────────────────────────────
    def _lub_R_in_wall_frame(self, delta_m):
        """Return (R_nn, R_tt, R_rr) scalars at gap δ [m]."""
        a = self._a_eff_m
        mu = self._mu
        delta = jnp.maximum(delta_m, self._delta_floor_m)

        R_nn = 6.0 * jnp.pi * mu * a ** 2 / delta
        # ln(a/δ) is only positive for δ < a; clamp to 0 past that point
        log_ratio = jnp.maximum(jnp.log(a / delta), 0.0)
        R_tt = 6.0 * jnp.pi * mu * a * (8.0 / 15.0) * log_ratio
        R_rr = 8.0 * jnp.pi * mu * a ** 3 * (2.0 / 5.0) * log_ratio
        return R_nn, R_tt, R_rr

    def _build_lub_matrix(self, R_nn, R_tt, R_rr, n):
        """Assemble the 6×6 lubrication block in the wall-aligned frame.

        n is the unit outward normal (body → wall); orthogonal tangents
        use vessel axis ẑ and n̂ × ẑ for the two tangential directions.
        """
        z = jnp.array([0.0, 0.0, 1.0])
        t2 = jnp.cross(n, z)
        # t2 is zero when n is itself along z (robot on the axis) — guard
        t2_norm = jnp.linalg.norm(t2)
        t2 = jnp.where(t2_norm > 1e-10, t2 / jnp.maximum(t2_norm, 1e-12),
                        jnp.array([1.0, 0.0, 0.0]))
        t1 = z  # vessel axis

        def _pad_F(v):
            return jnp.concatenate([v, jnp.zeros(3)])

        def _pad_T(v):
            return jnp.concatenate([jnp.zeros(3), v])

        R_lub = jnp.zeros((6, 6))
        R_lub = R_lub + R_nn * jnp.outer(_pad_F(n), _pad_F(n))
        R_lub = R_lub + R_tt * jnp.outer(_pad_F(t1), _pad_F(t1))
        R_lub = R_lub + R_tt * jnp.outer(_pad_F(t2), _pad_F(t2))
        R_lub = R_lub + R_rr * jnp.outer(_pad_T(t1), _pad_T(t1))
        R_lub = R_lub + R_rr * jnp.outer(_pad_T(t2), _pad_T(t2))
        return R_lub

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        R_in = boundary_inputs.get("resistance_matrix", jnp.eye(6))
        pos = boundary_inputs.get("robot_position", jnp.zeros(3))
        U = boundary_inputs.get("body_velocity", jnp.zeros(3))
        Omega = boundary_inputs.get("body_angular_velocity", jnp.zeros(3))
        u_bg = boundary_inputs.get("background_velocity", jnp.zeros(3))

        # Gap in SI: vessel radius − (offset + body radius)
        offset_m = jnp.sqrt(pos[0] ** 2 + pos[1] ** 2)
        delta = self._R_ves_m - (offset_m + self._R_max_body_m)
        delta = jnp.maximum(delta, self._delta_floor_m)

        # Outward wall normal — radial direction from vessel axis
        radial = jnp.array([pos[0], pos[1], 0.0])
        radial_norm = jnp.linalg.norm(radial)
        n_hat = jnp.where(
            radial_norm > 1e-10,
            radial / jnp.maximum(radial_norm, 1e-12),
            jnp.array([1.0, 0.0, 0.0]),  # default when robot on axis
        )

        # Blending weight
        w = jnp.exp(-delta / self._epsilon_m)

        R_nn, R_tt, R_rr = self._lub_R_in_wall_frame(delta)
        R_lub = self._build_lub_matrix(R_nn, R_tt, R_rr, n_hat)
        R_corrected = R_in + w * R_lub

        U_rel = U - u_bg
        motion = jnp.concatenate([U_rel, Omega])
        drag = -R_corrected @ motion
        drag_force = drag[:3]
        drag_torque = drag[3:]

        wall_normal_coef = w * R_nn
        active = (w > 0.01).astype(jnp.float32)

        return {
            "corrected_resistance_matrix": R_corrected,
            "drag_force": drag_force,
            "drag_torque": drag_torque,
            "min_gap": delta,
            "wall_normal_force_coef": wall_normal_coef,
            "lubrication_active": active,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "corrected_resistance_matrix":
                state["corrected_resistance_matrix"],
            "drag_force": state["drag_force"],
            "drag_torque": state["drag_torque"],
            "min_gap": state["min_gap"],
            "wall_normal_force_coef": state["wall_normal_force_coef"],
            "lubrication_active": state["lubrication_active"],
        }


def apply_lubrication_to_R_SI(
    R_SI: jnp.ndarray,
    offset_m: float,
    R_ves_m: float,
    R_max_body_m: float = 2.08e-3,
    a_eff_m: float = 2.08e-3,
    mu_Pa_s: float = 1e-3,
    epsilon_m: float = 0.2e-3,
    delta_floor_m: float = 1e-9,
):
    """Stateless SI-only version of the lubrication correction.

    Intended for offline / static analyses (paper-comparison re-runs,
    sensitivity sweeps) where we don't want to spin up a full
    MADDENING graph. Uses the same asymptotic coefficients as the node.

    Returns (R_corrected, delta, w).
    """
    delta = max(R_ves_m - (offset_m + R_max_body_m), delta_floor_m)
    w = float(jnp.exp(-delta / epsilon_m))

    a = a_eff_m
    R_nn = 6.0 * jnp.pi * mu_Pa_s * a ** 2 / delta
    log_ratio = max(float(jnp.log(a / delta)), 0.0)
    R_tt = 6.0 * jnp.pi * mu_Pa_s * a * (8.0 / 15.0) * log_ratio
    R_rr = 8.0 * jnp.pi * mu_Pa_s * a ** 3 * (2.0 / 5.0) * log_ratio

    n = jnp.array([1.0, 0.0, 0.0])  # training data has offset along +x
    z = jnp.array([0.0, 0.0, 1.0])
    t2 = jnp.cross(n, z)
    t2 = t2 / jnp.linalg.norm(t2)
    t1 = z

    def _pad_F(v):
        return jnp.concatenate([v, jnp.zeros(3)])

    def _pad_T(v):
        return jnp.concatenate([jnp.zeros(3), v])

    R_lub = jnp.zeros((6, 6))
    R_lub = R_lub + R_nn * jnp.outer(_pad_F(n), _pad_F(n))
    R_lub = R_lub + R_tt * jnp.outer(_pad_F(t1), _pad_F(t1))
    R_lub = R_lub + R_tt * jnp.outer(_pad_F(t2), _pad_F(t2))
    R_lub = R_lub + R_rr * jnp.outer(_pad_T(t1), _pad_T(t1))
    R_lub = R_lub + R_rr * jnp.outer(_pad_T(t2), _pad_T(t2))
    return R_SI + w * R_lub, delta, w
