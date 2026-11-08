"""The near field of a medium's interacting bodies: the near-field body-model contract (``BodyOperator``), the MRS as
its first body model (``MRSOperator``, composed from the per-body ``MRSDragNode``), the cross-kernel between bodies with
its reach, the joint system and its grand resistance (a pure function of the bodies' poses, their leaves and the
wall's parameters), and the joint node that solves it each step for a medium's bodies (``JointDragNode``)."""

from mime.nodes.hydro.near_field._contract import (
    BODY_FRAME_CONSTANT,
    CROSS_KERNEL,
    PRECONDITIONER,
    BodyOperator,
    Joint,
    Placed,
    blocks_of,
    cross_image,
    cross_loads,
    grand_resistance,
    joint_matrix,
    lagged_gain,
    quadratic_form,
)
from mime.nodes.hydro.near_field._mrs import CROSS_SHARE, REACH, MRSOperator, cross_weight, point_block
from mime.nodes.hydro.near_field._node import DENSE_UNKNOWNS, JointDragNode, member_prefix

__all__ = ["BODY_FRAME_CONSTANT", "CROSS_KERNEL", "CROSS_SHARE", "DENSE_UNKNOWNS", "PRECONDITIONER", "REACH",
           "BodyOperator", "Joint", "JointDragNode", "MRSOperator", "Placed", "blocks_of", "cross_image", "cross_loads",
           "cross_weight", "grand_resistance", "joint_matrix", "lagged_gain", "member_prefix", "point_block",
           "quadratic_form"]
