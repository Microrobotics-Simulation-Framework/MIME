"""Build the FSI GraphManager for UMR confinement.

Called by mime.runner: build_graph(params) -> GraphManager.

Two modes:
- USE_SCHWARZ=False: 4-node graph (LBM resolves the body directly)
- USE_SCHWARZ=True:  5-node graph (BEM near-field + LBM far-field via
                     Schwarz domain decomposition). No Mach constraint.
"""

from __future__ import annotations

import math
import logging

import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from mime.nodes.environment.lbm.fluid_node import (
    IBLBMFluidNode, make_iblbm_rigid_body_edges,
)
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode

logger = logging.getLogger(__name__)


def build_graph(params: dict) -> GraphManager:
    """Construct the FSI-coupled UMR graph from experiment parameters.

    Dispatches to _build_schwarz_graph when USE_SCHWARZ=True.

    Parameters
    ----------
    params : dict
        Namespace from physics/params.py (executed by mime.runner).

    Returns
    -------
    GraphManager
        Compiled graph ready for stepping.
    """
    if params.get("USE_SCHWARZ", False):
        logger.info("Building Schwarz BEM-LBM graph (USE_SCHWARZ=True)")
        return _build_schwarz_graph(params)

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

    from mime.nodes.robot.constraints import CylindricalVesselConstraint

    vessel_radius_m = params["VESSEL_DIAMETER_MM"] * 1e-3 / 2.0
    constraint = CylindricalVesselConstraint(
        radius=vessel_radius_m,
        half_length=vessel_radius_m * 2.0,
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
        constraint=constraint,
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


def _build_schwarz_graph(params: dict) -> GraphManager:
    """Build 5-node Schwarz BEM-LBM graph for clinical-frequency UMR.

    Near-field (BEM): body surface only, no Mach constraint.
    Far-field (LBM): vessel + interface sphere, no resolved body.
    """
    from mime.nodes.environment.stokeslet.fluid_node import (
        StokesletFluidNode,
        make_stokeslet_rigid_body_edges,
        make_schwarz_coupling_edges,
    )
    from mime.nodes.environment.stokeslet.surface_mesh import (
        sphere_surface_mesh, sdf_surface_mesh,
    )
    from mime.nodes.environment.stokeslet.interface import create_interface_mesh
    from mime.nodes.environment.lbm.far_field_node import LBMFarFieldNode

    N_lbm = params.get("SCHWARZ_LBM_RESOLUTION", 64)
    tau = params["TAU"]
    ratio = params["CONFINEMENT_RATIO"]
    subcycle_factor = params.get("SUBCYCLE_FACTOR", 10)
    iface_factor = params.get("SCHWARZ_IFACE_RADIUS_FACTOR", 2.0)

    # Physical dimensions
    vessel_d_m = params["VESSEL_DIAMETER_MM"] * 1e-3
    vessel_R_m = vessel_d_m / 2.0
    body_R_m = params["UMR_GEOM_MM"]["body_radius"] * 1e-3
    fin_R_m = params["UMR_GEOM_MM"]["fin_outer_radius"] * 1e-3
    body_bounding_R = fin_R_m  # bounding radius includes fins
    iface_R_m = iface_factor * body_bounding_R

    # LBM lattice parameters
    dx = vessel_d_m / (N_lbm * 0.8)  # vessel fills ~80% of domain
    nu_lattice = (tau - 0.5) / 3.0
    nu_physical = params["FLUID_VISCOSITY"] / params["FLUID_DENSITY"]
    dt_physical = nu_lattice * dx ** 2 / nu_physical
    vessel_R_lu = vessel_R_m / dx
    iface_R_lu = iface_R_m / dx
    center_lu = (N_lbm / 2, N_lbm / 2, N_lbm / 2)

    logger.info(
        "Schwarz graph: N_lbm=%d, dx=%.4e m, dt=%.4e s, "
        "vessel_R=%.1f lu, iface_R=%.1f lu",
        N_lbm, dx, dt_physical, vessel_R_lu, iface_R_lu,
    )

    # BEM body surface mesh (from SDF or simple sphere approximation)
    # For now, use a sphere approximation of the body
    body_mesh = sphere_surface_mesh(
        center=(0, 0, 0), radius=body_R_m, n_refine=3,
    )

    # Interface sphere mesh (BEM evaluation grid)
    iface_mesh_bem = create_interface_mesh(
        center=(0, 0, 0), radius=iface_R_m, n_refine=2,
    )

    # Interface sphere mesh in LBM physical coordinates
    lbm_center_phys = (N_lbm / 2 * dx, N_lbm / 2 * dx, N_lbm / 2 * dx)
    iface_mesh_lbm = sphere_surface_mesh(
        center=lbm_center_phys, radius=iface_R_m, n_refine=2,
    )

    # BEM epsilon
    gap = vessel_R_m - body_R_m
    epsilon_bem = min(0.05 * body_R_m, 0.02 * gap)

    # --- Construct nodes ---
    bem = StokesletFluidNode(
        name="bem_near",
        timestep=dt_physical,
        mu=params["FLUID_VISCOSITY"],
        body_mesh=body_mesh,
        interface_mesh=iface_mesh_bem,
        epsilon=epsilon_bem,
    )

    lbm = LBMFarFieldNode(
        name="lbm_far",
        timestep=dt_physical,
        nx=N_lbm, ny=N_lbm, nz=N_lbm, tau=tau,
        vessel_radius_lu=vessel_R_lu,
        interface_center_lu=center_lu,
        interface_radius_lu=iface_R_lu,
        interface_points_physical=np.array(iface_mesh_lbm.points),
        dx_physical=dx,
    )

    from mime.nodes.robot.constraints import CylindricalVesselConstraint
    constraint = CylindricalVesselConstraint(
        radius=vessel_R_m, half_length=vessel_R_m * 2.0,
    )

    # No Mach constraint — use clinical frequency directly
    f_field_hz = params.get("F_STEP_UNCONFINED", 128.0)
    omega_max = 2 * math.pi * f_field_hz * 2.0  # generous limit

    rigid = RigidBodyNode(
        name="rigid_body",
        timestep=dt_physical / subcycle_factor,
        semi_major_axis_m=params["SEMI_MAJOR"],
        semi_minor_axis_m=params["SEMI_MINOR"],
        density_kg_m3=params["BODY_DENSITY"],
        fluid_viscosity_pa_s=params["FLUID_VISCOSITY"],
        fluid_density_kg_m3=params["FLUID_DENSITY"],
        use_analytical_drag=False,
        use_inertial=True,
        I_eff=params["I_EFF"],
        omega_max=omega_max,
        constraint=constraint,
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
    gm.add_node(bem)
    gm.add_node(lbm)

    # Magnetic edges (unchanged)
    gm.add_edge("ext_field", "magnet_response", "field_vector", "field_vector")
    gm.add_edge("ext_field", "magnet_response", "field_gradient", "field_gradient")
    gm.add_edge("magnet_response", "rigid_body", "magnetic_torque",
                "magnetic_torque", additive=True)
    gm.add_edge("magnet_response", "rigid_body", "magnetic_force",
                "magnetic_force", additive=True)
    gm.add_edge("rigid_body", "magnet_response", "orientation", "orientation")

    # BEM ↔ RigidBody edges (SI units, no transforms)
    for edge in make_stokeslet_rigid_body_edges("bem_near", "rigid_body"):
        gm.add_edge(edge.source_node, edge.target_node,
                     edge.source_field, edge.target_field,
                     transform=edge.transform, additive=edge.additive)

    # BEM ↔ LBM Schwarz coupling edges
    # Unit conversion: BEM outputs m/s, LBM expects lattice units
    def phys_to_lattice_vel(u_phys):
        return u_phys * dt_physical / dx

    def lattice_to_phys_vel(u_lu):
        return u_lu * dx / dt_physical

    gm.add_edge("bem_near", "lbm_far",
                "interface_velocity", "interface_velocity",
                transform=phys_to_lattice_vel)
    gm.add_edge("lbm_far", "bem_near",
                "interface_background_velocity", "background_flow",
                transform=lattice_to_phys_vel)

    # External inputs
    gm.add_external_input("ext_field", "frequency_hz", shape=())
    gm.add_external_input("ext_field", "field_strength_mt", shape=())

    # Coupling groups
    schwarz_max_iter = params.get("SCHWARZ_MAX_ITER", 3)
    schwarz_tol = params.get("SCHWARZ_TOLERANCE", 1e-4)

    # Single coupling group for all FSI nodes:
    # rigid_body ↔ bem_near ↔ lbm_far
    gm.add_coupling_group(
        ["rigid_body", "bem_near", "lbm_far"],
        max_iterations=schwarz_max_iter,
        tolerance=schwarz_tol,
        convergence_norm="interface",
        acceleration="iqn-ils",
        subcycling=True,
        boundary_interpolation="linear",
    )

    return gm
