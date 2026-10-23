"""The rigid-body library: description readers into one canonical tree, and the maths of an open tree on it.

Nodes, dynamics entries and controllers import it; it imports nothing from MIME above ``mime.core``.

- **The tree** (:class:`MultibodyTree`): bodies with inertials, joints (revolute, prismatic, fixed) with their axes,
  limits, damping, friction and armature, named frames to mount parts on (every body's own, and extra ones), and loop
  closures held as data. A reader records the file it read and its SHA-256.
- **The URDF reader** (:func:`read_urdf`): every element of the file read and used, recognised and not used, or
  refused by name, each kind not used warned about once with what MIME does instead, and every element listed in
  the classification it returns.
- **The maths**, pure JAX (``jit``, ``grad`` and ``vmap`` traceable, finite at zero rotation, in the dtype of the
  joint positions given), everything in the root's frame: poses ``(x, y, z, qw, qx, qy, qz)`` on ``mime.core``'s
  quaternions; forward kinematics to every frame; frame Jacobians; inverse kinematics by damped least squares; the
  mass matrix by the composite-rigid-body algorithm; inverse dynamics by the recursive Newton-Euler algorithm (both
  after Featherstone, *Rigid Body Dynamics Algorithms*, 2008); forward dynamics ``q_ddot = M^-1 (tau - h)``.

**Closed chains are out of scope.** A closed-chain (parallel) mechanism, a delta robot or a four-bar linkage, is a tree
plus loop closures; the tree holds the closures, but every algorithm here is for open trees and refuses a tree that
holds one, naming it. They stay out of scope until a downstream user needs one: if you have that use case, ask at
https://github.com/Microrobotics-Simulation-Framework/MIME/issues. Until then you can register your own: a lab's
dynamics entry for a chain segment (``mime.registry.register_dynamics``) that takes its own description of the
mechanism and publishes the same frame poses as MIME's chains, so that mounts and everything downstream work unchanged.
"""

from mime.core.multibody._articulation import ROOT, Articulation, articulate
from mime.core.multibody._dynamics import bias_forces, forward_dynamics, gravity_forces, inverse_dynamics, mass_matrix
from mime.core.multibody._ik import InverseKinematics, damped_least_squares, inverse_kinematics
from mime.core.multibody._kinematics import Model, forward_kinematics, frame_jacobian, frame_pose
from mime.core.multibody._se3 import Pose, compose, inverse, pose_error, rotation_vector, transform_point
from mime.core.multibody._tree import IDENTITY, Body, ClosedChainRefused, ClosureKind, DescriptionSource, Frame, \
    Inertial, InvalidDescription, Joint, JointKind, JointLimits, LoopClosure, MultibodyTree, RigidTransform
from mime.core.multibody._urdf import CATEGORY_MESSAGES, DescriptionClassification, DescriptionElement, \
    DescriptionReading, DescriptionRefused, DescriptionWarning, ElementGroup, InertiaGap, read_urdf

__all__ = [
    "CATEGORY_MESSAGES",
    "IDENTITY",
    "ROOT",
    "Articulation",
    "Body",
    "ClosedChainRefused",
    "ClosureKind",
    "DescriptionClassification",
    "DescriptionElement",
    "DescriptionReading",
    "DescriptionRefused",
    "DescriptionSource",
    "DescriptionWarning",
    "ElementGroup",
    "Frame",
    "Inertial",
    "InertiaGap",
    "InvalidDescription",
    "InverseKinematics",
    "Joint",
    "JointKind",
    "JointLimits",
    "LoopClosure",
    "Model",
    "MultibodyTree",
    "Pose",
    "RigidTransform",
    "articulate",
    "bias_forces",
    "compose",
    "damped_least_squares",
    "forward_dynamics",
    "forward_kinematics",
    "frame_jacobian",
    "frame_pose",
    "gravity_forces",
    "inverse",
    "inverse_dynamics",
    "inverse_kinematics",
    "mass_matrix",
    "pose_error",
    "read_urdf",
    "rotation_vector",
    "transform_point",
]
