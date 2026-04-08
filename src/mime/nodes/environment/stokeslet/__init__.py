"""Regularised Stokeslet BEM fluid solver for confined microrobot flows.

Provides StokesletFluidNode — a quasi-static Stokes flow solver using
the method of regularised Stokeslets (Cortez, Fauci & Medovikov 2005)
with Nyström BEM discretization. Computes drag force and torque on a
rigid body via a 6×6 resistance matrix, optionally including vessel
wall confinement via explicit wall surface discretization or the
Liron-Shahar cylindrical Green's function (precomputed wall table).

No Mach number constraint — operates at any rotation frequency.
"""

from .fluid_node import StokesletFluidNode, make_stokeslet_rigid_body_edges
from .surface_mesh import SurfaceMesh, sphere_surface_mesh, cylinder_surface_mesh
from .cylinder_wall_table import WallTable, load_wall_table, save_wall_table

__all__ = [
    "StokesletFluidNode",
    "make_stokeslet_rigid_body_edges",
    "SurfaceMesh",
    "sphere_surface_mesh",
    "cylinder_surface_mesh",
    "WallTable",
    "load_wall_table",
    "save_wall_table",
]
