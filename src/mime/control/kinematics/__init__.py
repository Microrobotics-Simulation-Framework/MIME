"""URDF + rigid-body multibody helpers for MIME's robot-control layer.

Pure JAX implementation: every public function is jit/grad/vmap traceable.
This package mirrors patterns from `frax.core.robot` and `frax.utils.*` but
is reimplemented in MIME (no `frax` import) so it can ship as part of the
MIME stack without an extra dependency.

Public surface
--------------

URDF parsing
~~~~~~~~~~~~
* :func:`parse_urdf` — stdlib ``xml.etree`` parser → :class:`KinematicTree`.
* :class:`KinematicTree` — frozen dataclass (registered as a JAX static
  pytree) describing the robot's structure.
* :func:`ancestor_mask` — boolean ancestor matrix used by CRBA / RNEA.

Transforms
~~~~~~~~~~
* :func:`revolute_transform`, :func:`prismatic_transform` — joint transforms.
* :func:`compose_transform` — homogeneous matrix product.
* :func:`pose_to_matrix`, :func:`matrix_to_pose` — pose ⇄ 4×4 conversions.
  (Pose convention: ``[x, y, z, qw, qx, qy, qz]``, WXYZ scalar-first
  quaternion to match ``mime.core.quaternion``.)

Spatial algebra (6D motion / force)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* :func:`spatial_motion_cross`, :func:`spatial_force_cross`
* :func:`spatial_inertia_from_link`, :func:`spatial_axis_for_joint`
* Convention: ``[linear; angular]`` ordering (matches frax / Pinocchio /
  MuJoCo).

Forward kinematics
~~~~~~~~~~~~~~~~~~
* :func:`joint_to_world_transforms`, :func:`link_to_world_transforms`
* :func:`link_world_poses` — convenience pose form
* :func:`frame_jacobian` — geometric Jacobian for a given link

Dynamics
~~~~~~~~
* :func:`mass_matrix` — joint-space inertia matrix M(q) (CRBA).
* :func:`rnea`, :func:`gravity_vector`, :func:`nonlinear_bias` — RNEA-based
  inverse dynamics utilities.

Wave B (``RobotArmNode``) will consume this surface — see the plan at
``.claude/plans/hi-familiarize-yourself-with-giggly-toucan.md``.
"""

from .urdf import (
    KinematicTree,
    JOINT_TYPE_REVOLUTE,
    JOINT_TYPE_PRISMATIC,
    parse_urdf,
    ancestor_mask,
)
from .transform import (
    revolute_transform,
    prismatic_transform,
    joint_transform,
    compose_transform,
    pose_to_matrix,
    matrix_to_pose,
    transform_points,
)
from .spatial import (
    spatial_motion_cross,
    spatial_force_cross,
    spatial_inertia_from_link,
    spatial_inertias_from_links,
    spatial_axis_for_joint,
    spatial_axes_for_joints,
)
from .fk import (
    joint_to_world_transforms,
    link_to_world_transforms,
    link_world_poses,
    frame_jacobian,
)
from .crba import mass_matrix
from .rnea import rnea, gravity_vector, nonlinear_bias

__all__ = [
    # URDF
    "KinematicTree",
    "JOINT_TYPE_REVOLUTE",
    "JOINT_TYPE_PRISMATIC",
    "parse_urdf",
    "ancestor_mask",
    # Transforms
    "revolute_transform",
    "prismatic_transform",
    "joint_transform",
    "compose_transform",
    "pose_to_matrix",
    "matrix_to_pose",
    "transform_points",
    # Spatial
    "spatial_motion_cross",
    "spatial_force_cross",
    "spatial_inertia_from_link",
    "spatial_inertias_from_links",
    "spatial_axis_for_joint",
    "spatial_axes_for_joints",
    # FK
    "joint_to_world_transforms",
    "link_to_world_transforms",
    "link_world_poses",
    "frame_jacobian",
    # Dynamics
    "mass_matrix",
    "rnea",
    "gravity_vector",
    "nonlinear_bias",
]
