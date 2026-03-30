"""Hook: mesh_generator — generates UMR helical microrobot mesh.

Returns (vertices, triangles) for the UMR body + cone + helical fins,
or None if geometry parameters aren't available.
"""

from __future__ import annotations

from .helix_mesh import generate_umr_mesh


def generate_mesh(params: dict):
    """Generate UMR surface mesh from experiment parameters.

    Parameters
    ----------
    params : dict
        Experiment parameters. Requires 'UMR_GEOM_MM' key with
        body_radius, body_length, cone_length, etc.

    Returns
    -------
    (vertices, triangles) or None
        (N,3) float32 vertices and (M,3) int32 triangle indices.
    """
    geom_mm = params.get("UMR_GEOM_MM")
    if geom_mm is None:
        return None

    scale = 1e-3  # mm → m
    return generate_umr_mesh(
        body_radius=geom_mm["body_radius"] * scale,
        body_length=geom_mm["body_length"] * scale,
        cone_length=geom_mm["cone_length"] * scale,
        cone_end_radius=geom_mm["cone_end_radius"] * scale,
        fin_outer_radius=geom_mm["fin_outer_radius"] * scale,
        fin_length=geom_mm["fin_length"] * scale,
        fin_thickness=geom_mm["fin_thickness"] * scale,
        helix_pitch=geom_mm["helix_pitch"] * scale,
    )
