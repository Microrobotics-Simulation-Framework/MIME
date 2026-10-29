"""The regularised-Stokeslet boundary-element method: the matrix of a body's surface samples, its solve and the 6 x 6
resistance (``resistance``), the wall corrections a confining wall adds to it, by its wall kind's name and version
(free space adds none; the straight circular tube's, read from a table of its image system, is in place for its wall
kind), and the node that evaluates them at a body's pose (``BEMDragNode``)."""

from mime.nodes.hydro.bem._node import SHAPE, WALL, BEMDragInputs, BEMDragNode, BEMDragState, kind_from_settings
from mime.nodes.hydro.bem._solve import (
    PRECISION,
    blob_squared,
    point_forces,
    resistance,
    rigid_motions,
    stokeslet_matrix,
)
from mime.nodes.hydro.bem._tube import (
    frame_across,
    tube_correction,
    tube_reads,
    tube_table_from,
    tube_wall,
    unit_tube_table,
)
from mime.nodes.hydro.bem._walls import (
    FREE_SPACE,
    UNBOUNDED,
    Corrected,
    WallCorrection,
    register_wall_correction,
    wall_correction_named,
    wall_corrections,
)

__all__ = ["FREE_SPACE", "PRECISION", "SHAPE", "UNBOUNDED", "WALL", "BEMDragInputs", "BEMDragNode",
           "BEMDragState", "Corrected", "WallCorrection", "blob_squared", "frame_across", "kind_from_settings",
           "point_forces", "register_wall_correction", "resistance", "rigid_motions", "stokeslet_matrix",
           "tube_correction", "tube_reads", "tube_table_from", "tube_wall", "unit_tube_table", "wall_correction_named",
           "wall_corrections"]
