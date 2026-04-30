"""Stdlib URDF parser → JAX-friendly ``KinematicTree``.

This module parses a URDF file using only ``xml.etree.ElementTree`` (no
``urdfpy``, no ``xacro``). It supports v1 of the kinematics layer:

  * Joint types: ``revolute`` (mapped to type 0), ``continuous`` (also
    mapped to revolute, type 0, with infinite limits), ``prismatic`` (type 1),
    and ``fixed`` (merged at parse time — see :func:`_merge_fixed_joints`).
  * One root link (no kinematic loops).
  * Inertia, origin, axis, dynamics damping, joint limits.
  * Mesh URIs from ``<visual>`` and ``<collision>`` elements (kept as raw
    strings; resolution is the caller's responsibility).

The resulting :class:`KinematicTree` is a frozen dataclass registered as a
JAX *static* pytree (via ``jax.tree_util.register_static``), so the entire
tree can be passed as a closed-over argument to jit/vmap/grad without
triggering retracing on each call.

Numeric arrays on the tree are stored as ``numpy.ndarray`` (not jnp) so
that the dataclass remains hashable for jit caching; downstream functions
(``fk.py``, ``crba.py``, ``rnea.py``) cast them to ``jnp.ndarray`` once at
the top of each computation.

Fixed-joint merging
-------------------
``fixed`` joints are eliminated by absorbing the fixed-joint transform into
each non-fixed descendant's parent-frame transform, and combining the
inertial properties of the child link of a fixed joint into its parent
link (parallel-axis theorem applied at the merged frame). The merge is
applied iteratively until no fixed joints remain. This matches the
behavior of frax / Genesis URDF parsers and is the standard approach in
``urdfpy.merge_fixed_links``. We *exclude* the world→base fixed joint if
the URDF declares one (the root is always treated as world-fixed).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import jax
import numpy as np


# Joint-type integer encoding (matches frax)
JOINT_TYPE_REVOLUTE = 0
JOINT_TYPE_PRISMATIC = 1


# ---------------------------------------------------------------------------
# KinematicTree dataclass
# ---------------------------------------------------------------------------


@jax.tree_util.register_static
@dataclass(frozen=True, eq=True)
class KinematicTree:
    """Static description of a URDF-derived kinematic tree.

    All array fields are ``numpy.ndarray`` (so the dataclass is hashable for
    JAX jit caching). They are converted to ``jnp.ndarray`` inside the
    consumers (``fk.py``, ``crba.py``, ``rnea.py``).

    Attributes
    ----------
    num_joints : int
        Number of non-fixed joints (= number of articulated links). Equal
        to the configuration-space dimension.
    joint_types : (N,) ndarray of int
        0 = revolute, 1 = prismatic.
    joint_axes_local : (N, 3) ndarray of float
        Joint axis in each joint's local frame (URDF ``<axis>``).
    joint_parent_frame_xforms : (N, 4, 4) ndarray of float
        Transform from each joint's frame at q=0 to its parent joint's frame
        (URDF ``<origin>`` of the joint, possibly composed with absorbed
        fixed-joint transforms).
    link_masses : (N,) ndarray of float
    link_com_local : (N, 3) ndarray of float
        COM position in the joint's frame (i.e. the inertial origin's xyz).
    link_inertia_local : (N, 3, 3) ndarray of float
        Inertia tensor about the COM, expressed in the inertial frame.
        (For now the inertial-frame rotation, if any, is folded in by
        rotating the tensor at parse time — see :func:`_parse_inertial`.)
    parent_idxs : (N,) ndarray of int
        ``parent_idxs[i] = j`` means joint ``i``'s parent joint is ``j``.
        ``-1`` for joints whose parent is the (fixed) root link.
    link_names, joint_names : tuple[str, ...]
    joint_limits : (N, 2) ndarray of float
        ``[lower, upper]`` per joint.
    joint_friction : (N,) ndarray of float
        URDF ``<dynamics damping=...>`` (0 if absent).
    urdf_mesh_paths : tuple[tuple[str, str], ...]
        Tuple of ``(key, mesh_uri)`` pairs. Keys are
        ``f"{link_name}.visual"`` and ``f"{link_name}.collision"``. Stored
        as a tuple-of-tuples (rather than a dict) so the dataclass remains
        hashable for ``register_static``.
    root_link_name : str
        Name of the URDF root link (the kinematic-tree root).
    """

    num_joints: int
    joint_types: np.ndarray
    joint_axes_local: np.ndarray
    joint_parent_frame_xforms: np.ndarray
    link_masses: np.ndarray
    link_com_local: np.ndarray
    link_inertia_local: np.ndarray
    parent_idxs: np.ndarray
    link_names: tuple
    joint_names: tuple
    joint_limits: np.ndarray
    joint_friction: np.ndarray
    urdf_mesh_paths: tuple = field(default_factory=tuple)
    root_link_name: str = ""

    # Make hashable: hash by identity (cheap and JAX only needs a stable id).
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other) -> bool:
        return self is other

    # Convenience views ------------------------------------------------------

    @property
    def revolute_mask(self) -> np.ndarray:
        return (self.joint_types == JOINT_TYPE_REVOLUTE).astype(np.float32)

    @property
    def prismatic_mask(self) -> np.ndarray:
        return (self.joint_types == JOINT_TYPE_PRISMATIC).astype(np.float32)

    @property
    def mesh_paths_dict(self) -> dict:
        return dict(self.urdf_mesh_paths)


# ---------------------------------------------------------------------------
# Ancestor mask
# ---------------------------------------------------------------------------


def ancestor_mask(parent_idxs: np.ndarray) -> np.ndarray:
    """Build the boolean ancestor-or-self mask used by CRBA / RNEA.

    Returns
    -------
    (N, N) bool ndarray where ``mask[i, j] = True`` iff ``j`` is an ancestor
    of ``i`` or ``j == i``.

    Assumes a topological sort: parents always have a smaller index than
    their children (this is enforced by :func:`parse_urdf`).
    """
    parent_idxs = np.asarray(parent_idxs)
    n = len(parent_idxs)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        mask[i, i] = True
        p = int(parent_idxs[i])
        while p != -1:
            mask[i, p] = True
            p = int(parent_idxs[p])
    return mask


# ---------------------------------------------------------------------------
# URDF helpers (stdlib only)
# ---------------------------------------------------------------------------


def _parse_xyz(text: Optional[str], default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in text.split()], dtype=float)


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """RPY (extrinsic XYZ Euler) → 3x3 rotation matrix.

    URDF spec: rotation about fixed axes, applied as Rz·Ry·Rx (i.e. rotate
    by R first, then P about the new frame, etc., but expressed extrinsically).
    Equivalent: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _parse_origin(elem: Optional[ET.Element]) -> np.ndarray:
    """Parse a URDF ``<origin xyz="..." rpy="..."/>`` into a 4×4 transform."""
    if elem is None:
        return np.eye(4)
    xyz = _parse_xyz(elem.get("xyz"), default=(0.0, 0.0, 0.0))
    rpy = _parse_xyz(elem.get("rpy"), default=(0.0, 0.0, 0.0))
    return _make_homogeneous(_rotation_from_rpy(rpy), xyz)


def _parse_inertial(elem: Optional[ET.Element]) -> tuple:
    """Parse ``<inertial>`` block.

    Returns
    -------
    (mass, com, inertia_tensor_about_com_in_link_frame)

    The URDF inertial-frame rotation (rpy on ``<origin>``) is folded in by
    rotating the inertia tensor: ``I_link = R · I_inertial · Rᵀ``.
    """
    if elem is None:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    origin = _parse_origin(elem.find("origin"))
    com = origin[:3, 3]
    R_in = origin[:3, :3]

    mass_elem = elem.find("mass")
    mass = float(mass_elem.get("value")) if mass_elem is not None else 0.0

    i_elem = elem.find("inertia")
    if i_elem is None:
        I_in = np.zeros((3, 3))
    else:
        ixx = float(i_elem.get("ixx", 0.0))
        ixy = float(i_elem.get("ixy", 0.0))
        ixz = float(i_elem.get("ixz", 0.0))
        iyy = float(i_elem.get("iyy", 0.0))
        iyz = float(i_elem.get("iyz", 0.0))
        izz = float(i_elem.get("izz", 0.0))
        I_in = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

    I_link = R_in @ I_in @ R_in.T
    return mass, com, I_link


def _translate_inertia(I: np.ndarray, mass: float, d: np.ndarray) -> np.ndarray:
    """Parallel-axis: shift inertia tensor by displacement ``d``.

    ``I_new = I + m · (||d||² · I_3 - d·dᵀ)``
    """
    return I + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


# ---------------------------------------------------------------------------
# Internal IR (parsed but unmerged)
# ---------------------------------------------------------------------------


@dataclass
class _LinkIR:
    name: str
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    visual_mesh: Optional[str] = None
    collision_mesh: Optional[str] = None


@dataclass
class _JointIR:
    name: str
    joint_type: str  # "revolute" / "prismatic" / "fixed" / "continuous"
    parent_link: str
    child_link: str
    origin: np.ndarray = field(default_factory=lambda: np.eye(4))  # parent-link → child-joint
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    lower: float = -np.inf
    upper: float = np.inf
    damping: float = 0.0


def _parse_xml(path: str) -> tuple:
    """Parse the URDF XML into intermediate link/joint records."""
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"URDF root element must be <robot>; got <{root.tag}>")

    links: dict[str, _LinkIR] = {}
    for lk in root.findall("link"):
        name = lk.get("name")
        if name is None:
            raise ValueError("URDF <link> missing 'name' attribute")
        mass, com, I_link = _parse_inertial(lk.find("inertial"))
        link = _LinkIR(name=name, mass=mass, com=com, inertia=I_link)

        vis = lk.find("visual")
        if vis is not None:
            geom = vis.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None:
                    link.visual_mesh = mesh.get("filename")

        col = lk.find("collision")
        if col is not None:
            geom = col.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None:
                    link.collision_mesh = mesh.get("filename")

        links[name] = link

    joints: list[_JointIR] = []
    for jt in root.findall("joint"):
        name = jt.get("name")
        joint_type = jt.get("type")
        if name is None or joint_type is None:
            raise ValueError("URDF <joint> requires both 'name' and 'type'")

        parent = jt.find("parent")
        child = jt.find("child")
        if parent is None or child is None:
            raise ValueError(f"URDF joint '{name}' missing <parent> or <child>")
        parent_link = parent.get("link")
        child_link = child.get("link")

        origin = _parse_origin(jt.find("origin"))

        axis_elem = jt.find("axis")
        if axis_elem is not None:
            axis = _parse_xyz(axis_elem.get("xyz"), default=(1.0, 0.0, 0.0))
        else:
            axis = np.array([1.0, 0.0, 0.0])

        lower, upper = -np.inf, np.inf
        limit_elem = jt.find("limit")
        if limit_elem is not None:
            lower = float(limit_elem.get("lower", -np.inf))
            upper = float(limit_elem.get("upper", np.inf))

        damping = 0.0
        dyn_elem = jt.find("dynamics")
        if dyn_elem is not None:
            damping = float(dyn_elem.get("damping", 0.0))

        joints.append(
            _JointIR(
                name=name,
                joint_type=joint_type,
                parent_link=parent_link,
                child_link=child_link,
                origin=origin,
                axis=axis,
                lower=lower,
                upper=upper,
                damping=damping,
            )
        )

    return links, joints


def _find_root_link(links: dict, joints: list) -> str:
    """The root link is the unique link that is never a child of any joint."""
    children = {j.child_link for j in joints}
    roots = [name for name in links if name not in children]
    if len(roots) == 0:
        raise ValueError("URDF has no root link (kinematic loop?)")
    if len(roots) > 1:
        raise ValueError(f"URDF has multiple root links: {roots}")
    return roots[0]


def _merge_fixed_joints(
    links: dict, joints: list, root: str
) -> tuple:
    """Eliminate fixed (non-root) joints by absorbing them into descendants.

    Strategy
    --------
    Iterate until no fixed joints remain. For each fixed joint J with parent
    link P and child link C:

      1. Combine C's inertial into P's inertial. C's COM and inertia are
         expressed in the C-frame; we transform them into the P-frame via
         J's origin (``T_PC = J.origin``) and apply the parallel-axis
         theorem at P's frame using the shifted COM.
      2. For every joint K whose parent_link is C, redirect K's parent to
         P and prepend J's origin to K's origin: ``K.origin ← T_PC · K.origin``.
      3. Likewise for visual/collision meshes: if C has meshes, attach them
         to P keyed under C's name (so we don't clobber P's own meshes).
         For v1 we keep only the *first* mesh per link / per role, matching
         the URDF spec assumption that each ``<link>`` has at most one
         ``<visual>`` and one ``<collision>``.
      4. Remove C from ``links`` and J from ``joints``.

    Special case: if C is a leaf (no joints have C as parent), merging is
    purely an inertial accumulation onto P.

    The world→base fixed joint (one whose parent_link == root and whose
    child has no inertial mass) is treated identically — its child becomes
    the new "kinematic base", which is what we want for a fixed-base
    manipulator.

    Returns
    -------
    (links, joints) — both mutated in place and returned.
    """
    changed = True
    while changed:
        changed = False
        for joint in list(joints):
            if joint.joint_type != "fixed":
                continue
            P_name = joint.parent_link
            C_name = joint.child_link
            if P_name not in links or C_name not in links:
                # Already merged transitively; skip.
                joints.remove(joint)
                changed = True
                break

            P = links[P_name]
            C = links[C_name]
            T_PC = joint.origin

            # 1. Merge inertials. Express C's COM and inertia in P's frame.
            R_PC = T_PC[:3, :3]
            t_PC = T_PC[:3, 3]
            com_C_in_P = R_PC @ C.com + t_PC
            inertia_C_in_P = R_PC @ C.inertia @ R_PC.T

            m_total = P.mass + C.mass
            if m_total > 0:
                com_total = (P.mass * P.com + C.mass * com_C_in_P) / m_total
            else:
                com_total = P.com
            # Parallel-axis shift each contribution to com_total
            I_P_at_total = _translate_inertia(P.inertia, P.mass, com_total - P.com)
            I_C_at_total = _translate_inertia(
                inertia_C_in_P, C.mass, com_total - com_C_in_P
            )
            P.mass = m_total
            P.com = com_total
            P.inertia = I_P_at_total + I_C_at_total

            # Inherit C's meshes only if P has none in that role.
            if C.visual_mesh is not None and P.visual_mesh is None:
                P.visual_mesh = C.visual_mesh
            if C.collision_mesh is not None and P.collision_mesh is None:
                P.collision_mesh = C.collision_mesh

            # 2. Redirect child-joints of C → parent P, with origin pre-multiplied.
            for k in joints:
                if k is joint:
                    continue
                if k.parent_link == C_name:
                    k.parent_link = P_name
                    k.origin = T_PC @ k.origin

            # 4. Remove C and the fixed joint.
            del links[C_name]
            joints.remove(joint)
            changed = True
            break  # restart iteration (joints list mutated)

    return links, joints


def _topological_sort(joints: list, root: str) -> list:
    """Return joints sorted so each parent appears before its children.

    Joints whose parent is the root link come first, then their children,
    etc. (BFS).
    """
    children_of = {}
    for j in joints:
        children_of.setdefault(j.parent_link, []).append(j)

    ordered = []
    queue = list(children_of.get(root, []))
    while queue:
        j = queue.pop(0)
        ordered.append(j)
        queue.extend(children_of.get(j.child_link, []))

    if len(ordered) != len(joints):
        # Some joint references an unknown parent → kinematic island
        missing = [j.name for j in joints if j not in ordered]
        raise ValueError(
            f"URDF has joints disconnected from the kinematic tree rooted at "
            f"'{root}': {missing}"
        )
    return ordered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_urdf(path: str) -> KinematicTree:
    """Parse a URDF file and return a :class:`KinematicTree`.

    Supports ``revolute``, ``continuous``, ``prismatic``, and ``fixed``
    joints. ``fixed`` joints are merged into adjacent links (see
    :func:`_merge_fixed_joints`). Floating / planar / multi-DOF joints
    are not supported in v1.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"URDF file not found: {path}")

    links, joints = _parse_xml(path)
    if not joints:
        raise ValueError("URDF has no joints — cannot build a kinematic tree")

    root = _find_root_link(links, joints)

    # Merge fixed joints (drops links + joints in place).
    links, joints = _merge_fixed_joints(links, joints, root)

    if not joints:
        # A robot consisting solely of a single inertial link after merging
        # has no DOFs; treat as an error rather than silently returning empty.
        raise ValueError(
            "URDF has zero non-fixed joints after merging — kinematic tree is empty"
        )

    # The new root may have shifted if the original root was inside a chain
    # of fixed joints. Re-detect.
    root = _find_root_link(links, joints)

    # Topologically sort the surviving joints.
    ordered = _topological_sort(joints, root)
    name_to_idx = {j.name: i for i, j in enumerate(ordered)}
    link_to_joint_idx = {j.child_link: i for i, j in enumerate(ordered)}

    n = len(ordered)
    joint_types = np.zeros(n, dtype=np.int32)
    joint_axes_local = np.zeros((n, 3), dtype=np.float64)
    joint_parent_frame_xforms = np.zeros((n, 4, 4), dtype=np.float64)
    link_masses = np.zeros(n, dtype=np.float64)
    link_com_local = np.zeros((n, 3), dtype=np.float64)
    link_inertia_local = np.zeros((n, 3, 3), dtype=np.float64)
    parent_idxs = np.full(n, -1, dtype=np.int32)
    joint_limits = np.zeros((n, 2), dtype=np.float64)
    joint_friction = np.zeros(n, dtype=np.float64)
    link_names = []
    joint_names = []
    mesh_paths: list = []

    for i, j in enumerate(ordered):
        if j.joint_type in ("revolute", "continuous"):
            joint_types[i] = JOINT_TYPE_REVOLUTE
        elif j.joint_type == "prismatic":
            joint_types[i] = JOINT_TYPE_PRISMATIC
        else:
            raise ValueError(
                f"Unsupported joint type after merging: '{j.joint_type}' "
                f"(joint name: {j.name}). v1 supports revolute / continuous / "
                f"prismatic only."
            )
        joint_axes_local[i] = j.axis
        joint_parent_frame_xforms[i] = j.origin
        joint_limits[i] = [j.lower, j.upper]
        joint_friction[i] = j.damping

        # Parent index: parent is the joint whose child_link is our parent_link,
        # or -1 if our parent_link is the root.
        if j.parent_link == root:
            parent_idxs[i] = -1
        else:
            parent_idxs[i] = link_to_joint_idx[j.parent_link]

        link = links[j.child_link]
        link_masses[i] = link.mass
        link_com_local[i] = link.com
        link_inertia_local[i] = link.inertia

        link_names.append(link.name)
        joint_names.append(j.name)

        if link.visual_mesh is not None:
            mesh_paths.append((f"{link.name}.visual", link.visual_mesh))
        if link.collision_mesh is not None:
            mesh_paths.append((f"{link.name}.collision", link.collision_mesh))

    return KinematicTree(
        num_joints=n,
        joint_types=joint_types,
        joint_axes_local=joint_axes_local,
        joint_parent_frame_xforms=joint_parent_frame_xforms,
        link_masses=link_masses,
        link_com_local=link_com_local,
        link_inertia_local=link_inertia_local,
        parent_idxs=parent_idxs,
        link_names=tuple(link_names),
        joint_names=tuple(joint_names),
        joint_limits=joint_limits,
        joint_friction=joint_friction,
        urdf_mesh_paths=tuple(mesh_paths),
        root_link_name=root,
    )
