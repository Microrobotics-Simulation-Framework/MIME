"""De Jongh confined-swimming MADDENING graph builder.

Single source of truth for the helical-microrobot graph used by both
``scripts/dejongh_dynamic_simulation.py`` (offline analysis + USDC
recordings) and the ``MIME/experiments/dejongh_confined`` experiment
directory consumed by the MIME runner / MICROROBOTICA viewport.

The graph wires:

    ExternalMagneticFieldNode  →  PermanentMagnetResponseNode
                                   ↓ magnetic torque/force
    GravityNode  →  RigidBodyNode (inertial, kinematic_mode=False)
                       ↑                ↓ pose, velocity
                       drag             ↓
                       └── LubricationCorrectionNode (optional)
                                ↑
                          MLPResistanceNode  ←  body pose

Two external inputs are exposed so a controller can drive them live:

    field.frequency_hz       (scalar, Hz)
    field.field_strength_mt  (scalar, mT)

These match the parameter names that ``MIME/experiments/dejongh_confined``'s
``control/controller.py`` reads from ``params`` each tick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager

from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode
from mime.nodes.environment.gravity_node import GravityNode
from mime.nodes.environment.stokeslet.mlp_resistance_node import MLPResistanceNode
from mime.nodes.environment.stokeslet.lubrication_node import LubricationCorrectionNode
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.nodes.robot.constraints import CylindricalVesselConstraint


# ── Geometry constants (de Jongh 2025) ────────────────────────────────
R_CYL_UMR_MM = 1.56          # reference body radius
EPSILON_MOD = 0.33           # cross-section modulation
R_MAX_BODY_FACTOR = 1.0 + EPSILON_MOD  # 1.33

# 2 × 1mm³ N45 magnets (Supermagnete data: Mx = 0.84 Am² per magnet)
N_MAGNETS = 2
M_SINGLE = 8.4e-4

# Per-design ν values (helical wavenumber)
FL_PARAMS = {
    "FL-3": {"nu": 1.0, "L_UMR_mm": 7.47},
    "FL-5": {"nu": 1.4, "L_UMR_mm": 7.47},
    "FL-7": {"nu": 1.8, "L_UMR_mm": 7.47},
    "FL-9": {"nu": 2.33, "L_UMR_mm": 7.47},
}

# Vessel radii (mm) by colloquial name
VESSELS = {
    '1/2"':  6.35,
    '3/8"':  4.765,
    '1/4"':  3.175,
    '3/16"': 2.38,
}


def umr_volume_m3_fl(L_UMR_mm: float = 7.47) -> float:
    """Approximate FL-family UMR volume.

    π R_cyl² L · (1 + ε²/2) — the modulation correction averages the
    cross-section over θ ∈ [0, 2π).
    """
    return float(
        np.pi * (R_CYL_UMR_MM * 1e-3) ** 2 * (L_UMR_mm * 1e-3)
        * (1 + EPSILON_MOD ** 2 / 2)
    )


def default_mlp_weights_path() -> Path:
    """Repo-relative path to the trained Cholesky MLP weights (v2)."""
    repo = Path(__file__).resolve().parents[3]
    return repo / "data" / "dejongh_benchmark" / "mlp_cholesky_weights_v2.npz"


def build_graph(
    design_name: str = "FL-9",
    vessel_name: str = '1/4"',
    *,
    mu_Pa_s: float = 1e-3,
    delta_rho: float = 410.0,
    volume_m3: Optional[float] = None,
    dt: float = 5e-4,
    use_lubrication: bool = True,
    lubrication_epsilon_mm: float = 0.15,
    mlp_weights_path: Optional[Path] = None,
) -> GraphManager:
    """Build the de Jongh confined-swimming graph.

    Parameters
    ----------
    design_name :
        Key into :data:`FL_PARAMS` (FL-3, FL-5, FL-7, FL-9 today).
    vessel_name :
        Key into :data:`VESSELS` (silicone tube inner-diameter labels).
    mu_Pa_s :
        Fluid viscosity. Default water at 20 °C.
    delta_rho :
        Body − fluid density contrast [kg/m³]. Default UMR-in-water 410.
    volume_m3 :
        UMR volume. ``None`` → :func:`umr_volume_m3_fl`.
    dt :
        Physics timestep. 0.5 ms is well inside quaternion stability for
        10 Hz rotation.
    use_lubrication :
        Insert :class:`LubricationCorrectionNode` between MLP and body.
    lubrication_epsilon_mm :
        Blending scale ε for w(δ) = exp(−δ/ε).
    mlp_weights_path :
        Override the default ``mlp_cholesky_weights_v2.npz`` location.

    Returns
    -------
    GraphManager
        A *not-yet-compiled* graph. Caller invokes ``gm.compile()``.
    """
    fl = FL_PARAMS[design_name]
    R_ves_mm = VESSELS[vessel_name]
    if volume_m3 is None:
        volume_m3 = umr_volume_m3_fl(fl["L_UMR_mm"])
    if mlp_weights_path is None:
        mlp_weights_path = default_mlp_weights_path()

    field_node = ExternalMagneticFieldNode("field", dt)
    magnet_node = PermanentMagnetResponseNode(
        "magnet", dt,
        n_magnets=N_MAGNETS, m_single=M_SINGLE,
        moment_axis=(1.0, 0.0, 0.0),  # perpendicular to vessel axis (de Jongh)
    )
    gravity_node = GravityNode(
        "gravity", dt,
        delta_rho_kg_m3=delta_rho,
        volume_m3=volume_m3,
        direction=(0.0, -1.0, 0.0),  # y-down
    )
    mlp_node = MLPResistanceNode(
        "mlp_drag", dt,
        mlp_weights_path=str(mlp_weights_path),
        nu=fl["nu"], L_UMR_mm=fl["L_UMR_mm"],
        R_cyl_UMR_mm=R_CYL_UMR_MM,
        R_ves_mm=R_ves_mm,
        mu_Pa_s=mu_Pa_s,
    )

    # Vessel constraint with 0.1 mm safety margin
    R_max_body_mm = R_CYL_UMR_MM * R_MAX_BODY_FACTOR
    effective_R_mm = max(R_ves_mm - R_max_body_mm - 0.1, 0.1)
    vessel_constraint = CylindricalVesselConstraint(
        radius=effective_R_mm * 1e-3,
        half_length=0.05,  # 100 mm vessel length
        axis=2,
    )

    # Inertial integration: m, I derived from density × volume
    m_eff = delta_rho * volume_m3 + 1000.0 * volume_m3
    I_eff = 0.5 * m_eff * (R_CYL_UMR_MM * 1e-3) ** 2

    body_node = RigidBodyNode(
        "body", dt,
        semi_major_axis_m=R_CYL_UMR_MM * 1e-3 * R_MAX_BODY_FACTOR,
        semi_minor_axis_m=R_CYL_UMR_MM * 1e-3,
        density_kg_m3=delta_rho + 1000.0,
        fluid_viscosity_pa_s=mu_Pa_s,
        fluid_density_kg_m3=1000.0,
        use_analytical_drag=False,
        use_inertial=True,
        I_eff=I_eff,
        m_eff=m_eff,
        constraint=vessel_constraint,
    )

    gm = GraphManager()
    node_list = [field_node, magnet_node, gravity_node, mlp_node, body_node]

    if use_lubrication:
        lub_node = LubricationCorrectionNode(
            "lub", dt,
            R_ves_mm=R_ves_mm,
            R_max_body_mm=R_CYL_UMR_MM * R_MAX_BODY_FACTOR,
            mu_Pa_s=mu_Pa_s,
            epsilon_mm=lubrication_epsilon_mm,
            a_eff_mm=R_CYL_UMR_MM * R_MAX_BODY_FACTOR,
        )
        node_list.append(lub_node)

    for n in node_list:
        gm.add_node(n)

    # B field → magnet
    gm.add_edge("field", "magnet", "field_vector", "field_vector")
    gm.add_edge("field", "magnet", "field_gradient", "field_gradient")
    # Body orientation → magnet
    gm.add_edge("body", "magnet", "orientation", "orientation")

    # Body pose + velocity → MLP
    gm.add_edge("body", "mlp_drag", "position", "robot_position")
    gm.add_edge("body", "mlp_drag", "orientation", "robot_orientation")
    gm.add_edge("body", "mlp_drag", "velocity", "body_velocity")
    gm.add_edge("body", "mlp_drag", "angular_velocity", "body_angular_velocity")

    # Forces + torques into body
    gm.add_edge("gravity", "body", "gravity_force", "external_force",
                additive=True)
    gm.add_edge("magnet", "body", "magnetic_force", "magnetic_force")
    gm.add_edge("magnet", "body", "magnetic_torque", "magnetic_torque")

    if use_lubrication:
        gm.add_edge("mlp_drag", "lub", "resistance_matrix", "resistance_matrix")
        gm.add_edge("body", "lub", "position", "robot_position")
        gm.add_edge("body", "lub", "velocity", "body_velocity")
        gm.add_edge("body", "lub", "angular_velocity", "body_angular_velocity")
        gm.add_edge("lub", "body", "drag_force", "drag_force")
        gm.add_edge("lub", "body", "drag_torque", "drag_torque")
    else:
        gm.add_edge("mlp_drag", "body", "drag_force", "drag_force")
        gm.add_edge("mlp_drag", "body", "drag_torque", "drag_torque")

    # Live-controllable inputs
    gm.add_external_input("field", "frequency_hz", shape=(),
                          dtype=jnp.float32)
    gm.add_external_input("field", "field_strength_mt", shape=(),
                          dtype=jnp.float32)

    return gm


def build_graph_with_meta(
    design_name: str = "FL-9",
    vessel_name: str = '1/4"',
    *,
    B_strength_mT: float = 1.2,
    field_freq_hz: float = 10.0,
    pulsatile_U_mean_mm_s: float = 0.0,
    pulsatile_freq_hz: float = 1.2,
    pulsatile_amplitude: float = 0.6,
    **kwargs,
) -> Tuple[GraphManager, dict]:
    """Backwards-compatible wrapper used by ``scripts/dejongh_dynamic_simulation.py``.

    Returns ``(gm, meta_dict)`` so callers can record the scenario
    parameters alongside the graph. New code should prefer
    :func:`build_graph` and assemble its own metadata.
    """
    gm = build_graph(design_name=design_name, vessel_name=vessel_name, **kwargs)
    fl = FL_PARAMS[design_name]
    meta = {
        "design": design_name, "vessel": vessel_name,
        "R_ves_mm": VESSELS[vessel_name],
        "dt": kwargs.get("dt", 5e-4),
        "B_strength_mT": B_strength_mT, "field_freq_hz": field_freq_hz,
        "pulsatile_U_mean_mm_s": pulsatile_U_mean_mm_s,
        "pulsatile_freq_hz": pulsatile_freq_hz,
        "pulsatile_amplitude": pulsatile_amplitude,
        **fl,
    }
    return gm, meta
