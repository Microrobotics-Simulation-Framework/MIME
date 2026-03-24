"""Lattice Boltzmann Method (LBM) fluid solver with Immersed Boundary coupling.

Submodules:
- d2q9: 2D D2Q9 lattice constants, equilibrium, collision, streaming
- d3q19: 3D D3Q19 lattice constants, equilibrium, collision, streaming
- bounce_back: Simple and Bouzidi interpolated bounce-back boundary conditions
- rotating_body: Per-step rotating solid mask update
- ib: Immersed Boundary delta function interpolation and force spreading
- solver: IB-LBM coupled solver combining LBM and IB
- fluid_node: IBLBMFluidNode — MimeNode wrapper for 3D LBM solver

Nodes:
- IBLBMFluidNode: 3D IB-LBM fluid solver as a MADDENING SimulationNode
"""

from mime.nodes.environment.lbm.fluid_node import (
    IBLBMFluidNode,
    make_iblbm_rigid_body_edges,
)

__all__ = ["IBLBMFluidNode", "make_iblbm_rigid_body_edges"]
