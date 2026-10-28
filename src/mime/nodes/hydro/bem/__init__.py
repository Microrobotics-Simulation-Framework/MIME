"""The regularised-Stokeslet boundary-element method: the matrix of a body's surface samples, its solve and the 6 x 6
resistance (``resistance``), the wall corrections a confining wall adds to it, by name (free space adds none), and the
node that evaluates them at a body's pose (``BEMDragNode``)."""

from mime.nodes.hydro.bem._node import SHAPE, WALL, BEMDragInputs, BEMDragNode, BEMDragState, kind_from_settings
from mime.nodes.hydro.bem._solve import (
    PRECISION,
    blob_squared,
    point_forces,
    resistance,
    rigid_motions,
    stokeslet_matrix,
)
from mime.nodes.hydro.bem._walls import (
    FREE_SPACE,
    WallCorrection,
    register_wall_correction,
    wall_correction_named,
    wall_corrections,
)

__all__ = ["FREE_SPACE", "PRECISION", "SHAPE", "WALL", "BEMDragInputs", "BEMDragNode", "BEMDragState",
           "WallCorrection", "blob_squared", "kind_from_settings", "point_forces", "register_wall_correction",
           "resistance", "rigid_motions", "stokeslet_matrix", "wall_correction_named", "wall_corrections"]
