"""MLPResistanceNode — real-time surrogate for confined Stokes drag.

Replaces StokesletFluidNode for real-time 6DOF dynamic simulation. A
trained Cholesky MLP predicts the 6×6 SPD resistance matrix R as a
function of (ν, L_UMR, κ, offset_x, offset_y, log_min_gap). At each
timestep the node:

1. Extracts the robot position from the pose, computes (offset_x, offset_y)
2. Evaluates the MLP → 21 Cholesky entries → R_nd (non-dim, SPD by construction)
3. Converts R_nd to SI units (four block-wise scalings by μ·a, μ·a², μ·a³)
4. Solves overdamped Stokes: [U; Ω] = -R_SI⁻¹ @ [F_ext; T_ext]
5. Outputs velocity for kinematic integration downstream

Why this output design: the overdamped 6DOF solve requires a full-6×6 R
matrix inversion. RigidBodyNode's existing modes use diagonal
Oberbeck-Stechert drag (element-wise division). Rather than changing that
validated node, MLPResistanceNode does the full coupled solve internally
and outputs velocity. RigidBodyNode then runs in kinematic_mode=True,
integrating position + orientation from the velocity input.

Non-dimensionalisation convention (from the de Jongh benchmark):
    Length scale: a = R_cyl_UMR = 1.56 mm
    R_SI[:3,:3] = μ * a   * R_nd[:3,:3]   (force per velocity)
    R_SI[:3,3:] = μ * a²  * R_nd[:3,3:]   (force per angular velocity)
    R_SI[3:,:3] = μ * a²  * R_nd[3:,:3]   (torque per velocity)
    R_SI[3:,3:] = μ * a³  * R_nd[3:,3:]   (torque per angular velocity)

SPD is guaranteed by the Cholesky parameterisation: L has softplus
diagonal, R = LLᵀ is always SPD. Reciprocity is exact by construction.

Wall contact clamp: the MLP was trained on offset fractions ≤ 0.3 R_ves.
The node clamps the offset to (R_ves - R_max_UMR) × 0.95 to keep the MLP
inside its training range and prevent the robot from intersecting the wall.
"""

from __future__ import annotations

import logging
from typing import Callable, Union

import numpy as np
import jax
import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec, BoundaryFluxSpec
from maddening.core.edge import EdgeSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference, UQReadiness,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole,
    AnatomicalRegimeMeta, AnatomicalCompartment, FlowRegime,
)
from mime.surrogates.cholesky_mlp import (
    load_weights, mlp_forward, L_flat_to_R_jax, CholeskyMLPWeights,
)

logger = logging.getLogger(__name__)

# UMR geometry constants (from de Jongh 2025)
EPSILON_MOD = 0.33  # cross-section modulation
N_STARTS = 2        # number of helical starts
R_MAX_UMR_FACTOR = 1.0 + EPSILON_MOD  # R_max = R_cyl_UMR × 1.33


def _R_nd_to_SI(R_nd, mu_Pa_s, a_m):
    """Convert non-dimensional R to SI (4 block-wise scalings).

    Non-dim convention (from de Jongh benchmark):
        R_SI[:3,:3] = μ * a   * R_nd[:3,:3]   (force per velocity)
        R_SI[:3,3:] = μ * a²  * R_nd[:3,3:]   (force per angular vel)
        R_SI[3:,:3] = μ * a²  * R_nd[3:,:3]   (torque per velocity)
        R_SI[3:,3:] = μ * a³  * R_nd[3:,3:]   (torque per angular vel)
    """
    R_SI = jnp.zeros((6, 6))
    R_SI = R_SI.at[:3, :3].set(mu_Pa_s * a_m * R_nd[:3, :3])
    R_SI = R_SI.at[:3, 3:].set(mu_Pa_s * a_m ** 2 * R_nd[:3, 3:])
    R_SI = R_SI.at[3:, :3].set(mu_Pa_s * a_m ** 2 * R_nd[3:, :3])
    R_SI = R_SI.at[3:, 3:].set(mu_Pa_s * a_m ** 3 * R_nd[3:, 3:])
    return R_SI


@stability(StabilityLevel.EXPERIMENTAL)
class MLPResistanceNode(MimeNode):
    """Real-time confined Stokes drag via trained Cholesky MLP surrogate.

    Parameters
    ----------
    name : str
    timestep : float
    mlp_weights_path : str
        Path to .npz with MLP weights + normalization stats.
    nu : float
        Normalized wavenumber (geometry, fixed per simulation).
    L_UMR_mm : float
        UMR body length [mm] (fixed per simulation).
    R_cyl_UMR_mm : float
        UMR reference cylinder radius [mm]. Default 1.56 (de Jongh).
    R_ves_mm : float or Callable[float, float]
        Vessel radius [mm], constant or callable R_ves(z_m) for transitions.
    mu_Pa_s : float
        Fluid viscosity [Pa·s]. Default 1e-3 (water).
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-011",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Real-time 6×6 resistance-matrix surrogate for confined Stokes "
            "drag on a helical UMR, trained via Cholesky-parameterised MLP "
            "against regularised-Stokeslet BEM ground truth (de Jongh 2025 "
            "benchmark)."
        ),
        governing_equations=(
            "Overdamped Stokes balance [U; Ω] = -R⁻¹ @ [F_ext; T_ext]; "
            "R = L Lᵀ with L the MLP-predicted Cholesky factor "
            "(softplus diagonal guarantees SPD)."
        ),
        discretization=(
            "Stateless feed-forward MLP (SiLU activations, 21-dim "
            "lower-triangular output). Cylindrical symmetry exploited by "
            "rotating offset to +x canonical frame before evaluation."
        ),
        assumptions=(
            "Quasi-steady Stokes flow (Re → 0, instantaneous force balance).",
            "Rigid helical UMR with fixed geometry (ν, L_UMR) per simulation.",
            "Cylindrical vessel with axis along +z; offset rotations preserve R.",
            "Training ground truth (BEM + Liron-Shahar wall table) is trusted.",
        ),
        limitations=(
            "Extrapolation outside training envelope "
            "(ν ∈ [2.33, 3.7], κ ∈ [0.25, 0.66], offset_frac ≤ 0.3) is "
            "uncharacterised — wall-contact clamp projects back into range.",
            "Single fixed geometry per node instance; geometry change "
            "requires reloading weights.",
            "No Faxén correction for ambient flow curvature; background "
            "velocity enters only through relative velocity U - U_bg.",
        ),
        validated_regimes=(
            ValidatedRegime("nu", 2.33, 3.7, "",
                             "Helical wavenumber range in training set"),
            ValidatedRegime("kappa", 0.25, 0.66, "",
                             "Confinement ratio R_cyl_UMR / R_ves"),
            ValidatedRegime("offset_frac", 0.0, 0.3, "",
                             "Lateral offset / R_ves in training set"),
            ValidatedRegime("mlp_test_rel_mae", 0.0, 0.01, "",
                             "0.7% held-out swim-speed MAE on LHS test set"),
        ),
        references=(
            Reference("deJongh2025",
                       "de Jongh et al. (2025) confined helical swimmer benchmark"),
            Reference("CortezFauci2005",
                       "Cortez (2005) regularised Stokeslets SIAM J. Sci. Comput."),
            Reference("LironShahar1978",
                       "Liron & Shahar (1978) J. Fluid Mech. 86, 727-744"),
        ),
        uq_readiness=UQReadiness.PARAMETER_SWEEP,
        hazard_hints=(
            "Operating outside training envelope yields silently wrong R; "
            "the wall-contact clamp masks this by projecting offsets back "
            "inside — monitor clamp_fired output for persistent activation.",
            "Unit conversion R_nd → R_SI uses four distinct scalings "
            "(μa, μa², μa², μa³); a swapped exponent produces velocities "
            "off by 10³× without failing SPD checks.",
        ),
        implementation_map={
            "MLP forward pass":
                "mime.surrogates.cholesky_mlp.mlp_forward",
            "Cholesky → SPD reconstruction":
                "mime.surrogates.cholesky_mlp.L_flat_to_R_jax",
            "Non-dim → SI block scaling":
                "mime.nodes.environment.stokeslet.mlp_resistance_node._R_nd_to_SI",
            "Overdamped Stokes solve":
                "mime.nodes.environment.stokeslet.mlp_resistance_node."
                "MLPResistanceNode._compute_R_and_drag",
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ENVIRONMENT,
        anatomical_regimes=(
            AnatomicalRegimeMeta(
                compartment=AnatomicalCompartment.BLOOD,
                anatomy="iliac artery segment (3.175–6.35 mm diameter)",
                flow_regime=FlowRegime.STAGNANT,
                re_min=0.0, re_max=0.1,
                viscosity_min_pa_s=1e-3, viscosity_max_pa_s=4e-3,
                temperature_min_c=20.0, temperature_max_c=38.0,
                notes=(
                    "Trained on water at 20°C (μ=1e-3 Pa·s); μ enters "
                    "linearly in SI rescaling so blood-viscosity "
                    "extrapolation is exact."
                ),
            ),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        mlp_weights_path: str,
        nu: float,
        L_UMR_mm: float,
        R_cyl_UMR_mm: float = 1.56,
        R_ves_mm: Union[float, Callable[[float], float]] = 3.175,
        mu_Pa_s: float = 1e-3,
        **kwargs,
    ):
        super().__init__(name, timestep, mu=mu_Pa_s, **kwargs)
        self._nu = float(nu)
        self._L_UMR_mm = float(L_UMR_mm)
        self._R_cyl_UMR_mm = float(R_cyl_UMR_mm)
        self._R_cyl_UMR_m = self._R_cyl_UMR_mm * 1e-3  # for SI conversion
        self._mu = float(mu_Pa_s)
        self._R_ves_mm = R_ves_mm  # may be scalar or callable
        self._mlp_weights_path = str(mlp_weights_path)

        # Load MLP weights + normalization via the shared surrogate module
        weights = load_weights(mlp_weights_path)
        self._weights: CholeskyMLPWeights = weights
        self._params = weights.layers
        self._X_mean = weights.X_mean
        self._X_std = weights.X_std
        self._L_mean = weights.L_mean
        self._L_std = weights.L_std
        self._use_squared_features = bool(weights.use_squared_features)
        logger.info(
            "MLPResistanceNode %s: loaded MLP (%d layers, ν=%.3f, L=%.2f mm, "
            "μ=%.3e, squared_features=%s)",
            name, len(weights.layers), self._nu, self._L_UMR_mm, self._mu,
            self._use_squared_features,
        )

        # Precompute feature scalars (nu, L_nd are constant)
        self._L_nd = self._L_UMR_mm / self._R_cyl_UMR_mm

    @property
    def requires_halo(self) -> bool:
        return False

    def initial_state(self) -> dict:
        return {
            "drag_force": jnp.zeros(3),
            "drag_torque": jnp.zeros(3),
            "resistance_matrix": jnp.eye(6),
            "clamp_fired": jnp.array(0.0),
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "robot_position": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Robot center position [m]",
            ),
            "robot_orientation": BoundaryInputSpec(
                shape=(4,), default=jnp.array([1.0, 0.0, 0.0, 0.0]),
                description="Robot orientation quaternion",
            ),
            "body_velocity": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Current translational velocity [m/s]",
            ),
            "body_angular_velocity": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Current angular velocity [rad/s]",
            ),
            "background_velocity": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Optional Womersley/pulsatile flow at robot center [m/s]",
            ),
        }

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        return {
            "drag_force": BoundaryFluxSpec(
                shape=(3,), description="Fluid drag force on body = -R @ [U_rel; Ω] [N]",
                output_units="N",
            ),
            "drag_torque": BoundaryFluxSpec(
                shape=(3,), description="Fluid drag torque on body [N·m]",
                output_units="N*m",
            ),
            "resistance_matrix": BoundaryFluxSpec(
                shape=(6, 6), description="Resistance matrix R (SI)",
                output_units="mixed",
            ),
            "clamp_fired": BoundaryFluxSpec(
                shape=(), description="1.0 if wall-contact clamp fired this step",
                output_units="bool",
            ),
        }

    def _compute_R_and_drag(self, pose_pos_m, U, Omega, u_bg, R_ves_nd):
        """Pure JAX pipeline: position → R → drag = -R @ motion.

        Exploits cylindrical symmetry: training data has offsets along +x only.
        For arbitrary (offset_x, offset_y), rotate to align offset with +x,
        evaluate MLP in that canonical frame, then rotate R back.
        """
        # Position in mm for feature computation (MLP trained in mm-based non-dim)
        x_m, y_m, z_m = pose_pos_m[0], pose_pos_m[1], pose_pos_m[2]

        # Non-dimensional offset
        offset_x_nd = x_m * 1e3 / self._R_cyl_UMR_mm
        offset_y_nd = y_m * 1e3 / self._R_cyl_UMR_mm

        kappa = 1.0 / R_ves_nd

        # Wall contact clamp: keep offset within MLP training range / wall gap
        max_offset_nd = (R_ves_nd - R_MAX_UMR_FACTOR) * 0.95
        max_offset_nd = jnp.maximum(max_offset_nd, 0.0)
        offset_mag = jnp.sqrt(offset_x_nd**2 + offset_y_nd**2)
        scale = jnp.where(
            offset_mag > max_offset_nd,
            max_offset_nd / jnp.maximum(offset_mag, 1e-10),
            1.0,
        )
        offset_mag_clamped = offset_mag * scale
        clamp_fired = (offset_mag > max_offset_nd).astype(jnp.float32)

        # Rotation angle: align offset with +x in canonical frame
        theta = jnp.arctan2(offset_y_nd, offset_x_nd)
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)

        # In canonical frame, offset is along +x with magnitude offset_mag_clamped
        canonical_offset_x = offset_mag_clamped
        canonical_offset_y = 0.0

        # min_gap: distance from farthest body point to wall
        effective_body_edge = offset_mag_clamped + R_MAX_UMR_FACTOR
        min_gap_nd = jnp.maximum(R_ves_nd - effective_body_edge, 1e-3)
        log_min_gap = jnp.log(min_gap_nd)

        # Feature vector (canonical frame). v2 weights append ν², κ².
        if self._use_squared_features:
            X = jnp.array([self._nu, self._L_nd, kappa,
                            canonical_offset_x, canonical_offset_y, log_min_gap,
                            self._nu ** 2, kappa ** 2])
        else:
            X = jnp.array([self._nu, self._L_nd, kappa,
                            canonical_offset_x, canonical_offset_y, log_min_gap])
        X_n = (X - self._X_mean) / self._X_std

        # MLP forward → Cholesky entries → R_nd in canonical frame
        L_flat_n = mlp_forward(self._params, X_n)
        L_flat = L_flat_n * self._L_std + self._L_mean
        R_nd_canonical = L_flat_to_R_jax(L_flat)

        # Rotate R from canonical frame (offset along +x) to actual frame
        # (offset at angle θ). The rotation acts on both translation and
        # rotation blocks: R_actual = M @ R_canonical @ M^T where M is the
        # 6×6 block-diagonal rotation.
        Rz = jnp.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0],
        ])
        M = jnp.zeros((6, 6))
        M = M.at[:3, :3].set(Rz)
        M = M.at[3:, 3:].set(Rz)
        R_nd = M @ R_nd_canonical @ M.T

        # Convert to SI
        R_SI = _R_nd_to_SI(R_nd, self._mu, self._R_cyl_UMR_m)

        # Drag force/torque = -R @ [U_rel; Ω] where U_rel = U - U_bg is relative velocity
        # (drag opposes motion relative to the surrounding flow).
        U_rel = U - u_bg
        motion_rel = jnp.concatenate([U_rel, Omega])
        drag = -R_SI @ motion_rel
        drag_force = drag[:3]
        drag_torque = drag[3:]

        return drag_force, drag_torque, R_SI, clamp_fired

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        pos = boundary_inputs.get("robot_position", jnp.zeros(3))
        if callable(self._R_ves_mm):
            raise NotImplementedError("Callable R_ves_mm requires a JAX-traceable function.")
        R_ves_nd = float(self._R_ves_mm) / self._R_cyl_UMR_mm

        U = boundary_inputs.get("body_velocity", jnp.zeros(3))
        Omega = boundary_inputs.get("body_angular_velocity", jnp.zeros(3))
        u_bg = boundary_inputs.get("background_velocity", jnp.zeros(3))

        drag_F, drag_T, R_SI, clamp = self._compute_R_and_drag(
            pos, U, Omega, u_bg, R_ves_nd)

        return {
            "drag_force": drag_F,
            "drag_torque": drag_T,
            "resistance_matrix": R_SI,
            "clamp_fired": clamp,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "drag_force": state["drag_force"],
            "drag_torque": state["drag_torque"],
            "resistance_matrix": state["resistance_matrix"],
            "clamp_fired": state["clamp_fired"],
        }


def make_mlp_rigid_body_edges(
    mlp_node_name: str,
    body_node_name: str,
    gravity_node_name: str | None = None,
    magnet_node_name: str | None = None,
) -> list[EdgeSpec]:
    """Canonical edge wiring between an MLPResistanceNode and a RigidBodyNode.

    Connects pose/velocity from body → MLP (for R evaluation at current
    offset), drag force/torque from MLP → body (consumed by body's
    inertial or external-drag integration), and optionally routes
    gravity/magnetic force & torque into the body.

    Parameters
    ----------
    mlp_node_name, body_node_name : str
        Names of the registered nodes.
    gravity_node_name, magnet_node_name : str, optional
        If provided, adds the corresponding additive force/torque edges.

    Returns
    -------
    list[EdgeSpec]
    """
    edges: list[EdgeSpec] = [
        EdgeSpec(body_node_name, mlp_node_name, "position", "robot_position"),
        EdgeSpec(body_node_name, mlp_node_name, "orientation", "robot_orientation"),
        EdgeSpec(body_node_name, mlp_node_name, "velocity", "body_velocity"),
        EdgeSpec(body_node_name, mlp_node_name, "angular_velocity",
                  "body_angular_velocity"),
        EdgeSpec(mlp_node_name, body_node_name, "drag_force", "drag_force"),
        EdgeSpec(mlp_node_name, body_node_name, "drag_torque", "drag_torque"),
    ]
    if gravity_node_name is not None:
        edges.append(EdgeSpec(
            gravity_node_name, body_node_name,
            "gravity_force", "external_force", additive=True,
        ))
    if magnet_node_name is not None:
        edges.extend([
            EdgeSpec(magnet_node_name, body_node_name,
                      "magnetic_force", "magnetic_force"),
            EdgeSpec(magnet_node_name, body_node_name,
                      "magnetic_torque", "magnetic_torque"),
        ])
    return edges
