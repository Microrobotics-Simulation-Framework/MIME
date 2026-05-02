"""Graph-native finite volume method for MIME.

A collocated, JAX-native FVM solver for incompressible Navier-Stokes,
designed around the gather-compute-scatter pattern (DiFVM, Du et al.
arXiv:2603.15920) so every operator is a single message-pass over a
face graph. This makes the entire PISO/SIMPLE loop fully JIT-fusible
and autodiff-transparent.

Core abstractions
-----------------
- :class:`FVMMesh` — face-graph topology + precomputed geometry
- :class:`BoundaryPatch` — a labelled set of boundary faces
- gather/compute/scatter operators in :mod:`.operators`
- FFT-diagonalised pressure Poisson in :mod:`.pressure`
- Diffuse penalty IBM in :mod:`.ibm`
- SIMPLE / PISO solver loops in :mod:`.simple` / :mod:`.piso`
- :class:`FVMFluidNode` — MADDENING-compatible fluid node in :mod:`.fluid_node`
"""

from mime.nodes.environment.fvm.mesh import (
    FVMMesh,
    BoundaryPatch,
    make_cartesian_mesh_2d,
    make_cartesian_mesh_3d,
)
from mime.nodes.environment.fvm.fluid_node import (
    FVMFluidNode,
    make_sphere_body_factory,
)

__all__ = [
    "FVMMesh",
    "BoundaryPatch",
    "make_cartesian_mesh_2d",
    "make_cartesian_mesh_3d",
    "FVMFluidNode",
    "make_sphere_body_factory",
]
