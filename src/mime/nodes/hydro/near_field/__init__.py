"""The near field of a medium's interacting bodies: the near-field body-model contract (``BodyOperator``), the MRS as
its first body model (``MRSOperator``, composed from the per-body ``MRSDragNode``), the cross-kernel between bodies with
its reach, the joint system and its grand resistance (a pure function of the bodies' poses, their leaves and the
wall's parameters; by a dense LU, or past the takeover line by GMRES with each body's own block its preconditioner),
and the joint node that solves it each step for a medium's bodies (``JointDragNode``)."""

from mime.nodes.hydro.near_field._contract import (
    BODY_FRAME_CONSTANT,
    CROSS_KERNEL,
    OVERLAP_RESOLUTION,
    PRECONDITIONER,
    BodyOperator,
    Joint,
    Placed,
    blocks_of,
    booked_work,
    cross_image,
    cross_loads,
    grand_resistance,
    grand_resistance_by_gmres,
    joint_matrix,
    lagged_amplification,
    lagged_gain,
    pair_gaps,
    quadratic_form,
)
from mime.nodes.hydro.near_field._mrs import CROSS_SHARE, REACH, MRSOperator, cross_weight, point_block
from mime.nodes.hydro.near_field._node import (
    CONTACT_GAP,
    DEFAULT_TOLERANCE,
    DENSE_UNKNOWNS,
    MEMORY_COUNTS,
    MEMORY_MEASURED,
    JointDragInputs,
    JointDragNode,
    JointDragState,
    member_prefix,
)

__all__ = ["BODY_FRAME_CONSTANT", "CONTACT_GAP", "CROSS_KERNEL", "CROSS_SHARE", "DEFAULT_TOLERANCE", "DENSE_UNKNOWNS",
           "MEMORY_COUNTS", "MEMORY_MEASURED", "OVERLAP_RESOLUTION", "PRECONDITIONER", "REACH", "BodyOperator",
           "Joint", "JointDragInputs", "JointDragNode", "JointDragState", "MRSOperator", "Placed", "blocks_of",
           "booked_work", "cross_image", "cross_loads", "cross_weight", "grand_resistance", "grand_resistance_by_gmres",
           "joint_matrix", "lagged_amplification", "lagged_gain", "member_prefix", "pair_gaps", "point_block",
           "quadratic_form"]
