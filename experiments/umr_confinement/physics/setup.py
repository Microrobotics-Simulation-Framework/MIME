"""Build the 4-node FSI GraphManager for UMR confinement.

Called by mime.runner: build_graph(params) -> GraphManager.
Same graph topology as run_single_node_fsi() in run_confinement_sweep.py.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from mime.nodes.environment.lbm.fluid_node import (
    IBLBMFluidNode, make_iblbm_rigid_body_edges,
)
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode


def build_graph(params: dict) -> GraphManager:
    """Construct the FSI-coupled UMR graph from experiment parameters.

    Parameters
    ----------
    params : dict
        Namespace from physics/params.py (executed by mime.runner).

    Returns
    -------
    GraphManager
        Compiled graph ready for stepping.
    """
    N = params["RESOLUTION"]
    tau = params["TAU"]
    ratio = params["CONFINEMENT_RATIO"]
    use_bouzidi = params.get("USE_BOUZIDI", False)
    subcycle_factor = params.get("SUBCYCLE_FACTOR", 10)

    cs = 1.0 / math.sqrt(3)
    dx_mm = params["VESSEL_DIAMETER_MM"] / N
    dx_physical = params["VESSEL_DIAMETER_MM"] * 1e-3 / N
    nu_lattice = (tau - 0.5) / 3.0
    nu_physical = params["FLUID_VISCOSITY"] / params["FLUID_DENSITY"]
    dt_physical = nu_lattice * dx_physical ** 2 / nu_physical

    geom_lu = {k: v / dx_mm for k, v in params["UMR_GEOM_MM"].items()}
    R_fin_lu = geom_lu["fin_outer_radius"]
    R_vessel_lu = geom_lu["body_radius"] / ratio

    omega_max_lattice = 0.05 * cs / R_fin_lu
    omega_max_physical = omega_max_lattice / dt_physical

    # Target field frequency matching the sweep omega
    omega_lu = 0.05 * cs / R_fin_lu
    omega_physical = omega_lu / dt_physical
    f_field_hz = omega_physical / (2.0 * math.pi)

    # --- Construct nodes ---
    body_geometry_params = dict(nx=N, ny=N, nz=N, **geom_lu)
    lbm = IBLBMFluidNode(
        name="lbm_fluid", timestep=dt_physical,
        nx=N, ny=N, nz=N, tau=tau,
        vessel_radius_lu=R_vessel_lu,
        body_geometry_params=body_geometry_params,
        use_bouzidi=use_bouzidi, dx_physical=dx_physical,
    )

    rigid = RigidBodyNode(
        name="rigid_body", timestep=dt_physical / subcycle_factor,
        semi_major_axis_m=params["SEMI_MAJOR"],
        semi_minor_axis_m=params["SEMI_MINOR"],
        density_kg_m3=params["BODY_DENSITY"],
        fluid_viscosity_pa_s=params["FLUID_VISCOSITY"],
        fluid_density_kg_m3=params["FLUID_DENSITY"],
        use_analytical_drag=False,
        use_inertial=True,
        I_eff=params["I_EFF"],
        omega_max=omega_max_physical,
    )

    field = ExternalMagneticFieldNode(
        name="ext_field", timestep=dt_physical,
    )

    magnet = PermanentMagnetResponseNode(
        name="magnet_response", timestep=dt_physical,
        n_magnets=params["N_MAG"],
        m_single=params["M_SINGLE"],
        moment_axis=(0.0, 1.0, 0.0),
    )

    # --- Construct graph ---
    gm = GraphManager()
    gm.add_node(field)
    gm.add_node(magnet)
    gm.add_node(rigid)
    gm.add_node(lbm)

    gm.add_edge("ext_field", "magnet_response", "field_vector", "field_vector")
    gm.add_edge("ext_field", "magnet_response", "field_gradient", "field_gradient")
    gm.add_edge("magnet_response", "rigid_body", "magnetic_torque",
                "magnetic_torque", additive=True)
    gm.add_edge("magnet_response", "rigid_body", "magnetic_force",
                "magnetic_force", additive=True)
    gm.add_edge("rigid_body", "magnet_response", "orientation", "orientation")

    for edge in make_iblbm_rigid_body_edges(
        "lbm_fluid", "rigid_body", dx_physical, dt_physical,
        params["FLUID_DENSITY"],
    ):
        gm.add_edge(edge.source_node, edge.target_node,
                     edge.source_field, edge.target_field,
                     transform=edge.transform, additive=edge.additive)

    gm.add_external_input("ext_field", "frequency_hz", shape=())
    gm.add_external_input("ext_field", "field_strength_mt", shape=())

    gm.add_coupling_group(
        ["rigid_body", "lbm_fluid"],
        max_iterations=1,
        subcycling=True,
        boundary_interpolation="linear",
    )

    return gm
