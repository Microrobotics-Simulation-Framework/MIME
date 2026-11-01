"""The method of regularised Stokeslets: the matrix of a body's surface samples, its solve and the 6 x 6
resistance (``resistance``), the wall corrections a confining wall adds to it, by its wall kind's name and version
(free space adds none; the straight circular tube's, read from a table of its image system, is in place for its wall
kind), and the node that evaluates them at a body's pose (``MRSDragNode``)."""

from mime.nodes.hydro.mrs._node import SHAPE, WALL, MRSDragInputs, MRSDragNode, MRSDragState, kind_from_settings
from mime.nodes.hydro.mrs._solve import (
    DEPTH,
    PRECISION,
    blob_squared,
    force_sums,
    point_forces,
    resistance,
    rigid_motions,
    stokeslet_matrix,
)
from mime.nodes.hydro.mrs._tube import (
    frame_across,
    tube_correction,
    tube_reads,
    tube_table_from,
    tube_wall,
    unit_tube_table,
)
from mime.nodes.hydro.mrs._walls import (
    FREE_SPACE,
    UNBOUNDED,
    Corrected,
    WallCorrection,
    register_wall_correction,
    wall_correction_named,
    wall_corrections,
)

__all__ = ["DEPTH", "FREE_SPACE", "PRECISION", "SHAPE", "UNBOUNDED", "WALL", "MRSDragInputs", "MRSDragNode",
           "MRSDragState", "Corrected", "WallCorrection", "blob_squared", "force_sums", "frame_across",
           "kind_from_settings", "point_forces", "register_wall_correction", "resistance", "rigid_motions",
           "stokeslet_matrix", "tube_correction", "tube_reads", "tube_table_from", "tube_wall", "unit_tube_table",
           "wall_correction_named", "wall_corrections"]
