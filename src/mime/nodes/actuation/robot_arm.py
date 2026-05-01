"""RobotArmNode — articulated rigid-body manipulator driven by a URDF.

This node simulates a fixed-base, multi-DOF articulated arm whose
kinematic and dynamic structure is parsed once from a URDF file at
construction time. The joint-space dynamics are integrated with
semi-implicit Euler:

    M(q)·q̈ + c(q, q̇) + g(q) = τ_eff + Jᵀ·F_ext
    q̇_{n+1} = q̇_n + q̈ · Δt
    q_{n+1}  = q_n  + q̇_{n+1} · Δt

where

* ``M(q)``         is the joint-space mass matrix (CRBA),
* ``c(q,q̇) + g(q)`` is the nonlinear bias (Coriolis + gravity, RNEA),
* ``τ_eff = τ_cmd − f·q̇`` is the commanded torque after viscous joint
  friction,
* ``F_ext`` are external 6-DOF wrenches applied per link, mapped into
  joint torque space by ``Jᵀ_i`` (the geometric Jacobian of link ``i``).

Joint limits are enforced by hard clipping on ``q`` after the position
update — this is a non-compliant (rigid) stop and is documented as a
hazard hint. v1 has no contact / collision detection and no joint
flexibility.

The kinematic tree is captured at init as a ``KinematicTree`` (a frozen,
JAX-static dataclass) so the entire ``update`` body is JAX-traceable
under jit / grad / vmap with no runtime URDF parsing.

Algorithm ID: MIME-NODE-102
Stability: EXPERIMENTAL
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.node import BoundaryInputSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta,
    NodeRole,
    ActuationMeta,
    ActuationPrinciple,
    ArticulatedArmMeta,
)
from mime.control.kinematics import (
    parse_urdf,
    KinematicTree,
    link_world_poses,
    mass_matrix,
    nonlinear_bias,
    frame_jacobian,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_pose7() -> jnp.ndarray:
    """Return ``[0,0,0, 1,0,0,0]`` — origin at identity orientation
    (scalar-first quaternion ``[w, x, y, z]`` per project convention)."""
    return jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def _resolve_link_index(tree: KinematicTree, link_name: str) -> int:
    """Resolve a URDF link name to its index in ``tree.link_names``.

    Raised at construction time only — the result is captured as static
    Python state on the node, so this lookup never re-runs during a JAX
    trace.
    """
    for i, name in enumerate(tree.link_names):
        if name == link_name:
            return i
    raise ValueError(
        f"Link '{link_name}' not found in URDF tree. Available links: "
        f"{tree.link_names}"
    )


def _compose_pose(parent_pose: jnp.ndarray, child_pose: jnp.ndarray) -> jnp.ndarray:
    """Compose two 7-vector poses ``[x,y,z, qw,qx,qy,qz]`` → world pose.

    ``world = parent ⊕ child``: the child pose is expressed in the
    parent's frame, the result is the child expressed in the parent's
    parent's frame (typically world).
    """
    from mime.core.quaternion import quat_multiply, quat_normalize, quat_to_rotation_matrix

    p_t = parent_pose[0:3]
    p_q = parent_pose[3:7]
    c_t = child_pose[0:3]
    c_q = child_pose[3:7]

    R_p = quat_to_rotation_matrix(p_q)
    t_world = p_t + R_p @ c_t
    q_world = quat_normalize(quat_multiply(p_q, c_q))
    return jnp.concatenate([t_world, q_world])


# ---------------------------------------------------------------------------
# Step-time dynamics — pure JAX
# ---------------------------------------------------------------------------


def _wrench_to_joint_torques(
    tree: KinematicTree,
    q: jnp.ndarray,
    wrenches_per_link: jnp.ndarray,
) -> jnp.ndarray:
    """Map per-link 6-vector wrenches to joint torques via Jᵀ.

    ``wrenches_per_link[i] = [F_lin (3); F_ang (3)]`` follows the
    ``mime.control.kinematics`` ``[linear; angular]`` ordering. For each
    link ``i`` we evaluate ``J_i = frame_jacobian(tree, q, i)`` and
    accumulate ``τ_ext += J_iᵀ · wrench_i``.

    The unrolled Python loop is OK at trace time: ``tree.num_joints`` is
    a static integer (the tree is a JAX-static pytree), so the loop is
    unrolled into XLA ops once per trace. With typical ``N ≤ 10`` this
    is negligible.
    """
    n = tree.num_joints
    tau = jnp.zeros(n, dtype=q.dtype)
    for i in range(n):
        J_i = frame_jacobian(tree, q, link_idx=i)  # (6, N)
        tau = tau + J_i.T @ wrenches_per_link[i]
    return tau


# ---------------------------------------------------------------------------
# RobotArmNode
# ---------------------------------------------------------------------------


@stability(StabilityLevel.EXPERIMENTAL)
class RobotArmNode(MimeNode):
    """Fixed-base articulated rigid-body arm driven by a URDF.

    Parameters
    ----------
    name : str
    timestep : float
        Simulation timestep [s].
    urdf_path : str
        Path to the URDF file. Resolved relative to CWD by default; the
        caller should normally supply an absolute path.
    base_pose_world : tuple[7], default identity
        World pose of the kinematic-tree root link
        ``[x, y, z, qw, qx, qy, qz]`` (scalar-first quaternion per
        project convention).
    end_effector_link_name : str
        URDF link name to expose as the end-effector flux. Resolved at
        init time against the parsed URDF; raises if not found.
    end_effector_offset_in_link : tuple[7], default identity
        Fixed transform from the EE link's frame to the actual tool-tip
        frame, applied after the link pose lookup.
    joint_friction_n_m_s : tuple[float] or None, default None
        Per-joint viscous friction coefficient [N·m·s/rad]. If ``None``
        the URDF's ``<dynamics damping="...">`` values are used; specify
        explicitly to override.
    gravity_world : tuple[3], default ``(0, 0, -9.80665)``
        World-frame gravity vector [m/s²].
    joint_limit_override : tuple[(N,2)] or None, default None
        Optional per-joint ``(lower, upper)`` overrides in radians (or
        meters for prismatic). If ``None`` the URDF's ``<limit>`` values
        are used.

    Boundary Inputs
    ---------------
    commanded_joint_torques : (N,)
        Direct joint torques [N·m]. Additive; default zeros.
    external_wrenches_per_link : (N, 6)
        Per-link world-frame 6-vector wrenches ``[F_lin; F_ang]``.
        Additive; default zeros.

    Boundary Fluxes
    ---------------
    end_effector_pose_world : (7,)
        World pose of the end-effector tool-tip frame.
    link_poses_world : (N, 7)
        World pose of every URDF link (in tree order).
    joint_angles : (N,)
        Current joint angles.
    joint_velocities : (N,)
        Current joint velocities.
    joint_torques_actual : (N,)
        Joint torques actually applied this step
        (``τ_cmd − f·q̇ + Jᵀ·wrench``).
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-102",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Fixed-base articulated rigid-body arm parameterised by a "
            "URDF. Joint-space dynamics integrated with semi-implicit "
            "Euler; mass matrix from CRBA, nonlinear bias from RNEA, "
            "external wrenches mapped via Jacobian transpose."
        ),
        governing_equations=(
            r"M(q) qdd + c(q, qd) + g(q) = tau_eff + sum_i J_i^T F_ext_i; "
            r"tau_eff = tau_cmd - f*qd; "
            r"qd_{n+1} = qd_n + qdd*dt; "
            r"q_{n+1}  = q_n  + qd_{n+1}*dt; "
            r"q_{n+1}  = clip(q_{n+1}, q_lo, q_hi)"
        ),
        discretization=(
            "Semi-implicit (symplectic) Euler with explicit forces; "
            "mass matrix M(q) via CRBA (vectorised, [linear, angular] "
            "spatial ordering); nonlinear bias c+g via RNEA in a single "
            "pass; geometric Jacobian J_i for the inertial frame of "
            "each link (ancestor-chain construction). Joint limits are "
            "enforced by hard clipping after the position update."
        ),
        assumptions=(
            "Rigid joints — no flexibility, backlash, or compliant stops",
            "Linear viscous joint friction tau_f = f·q̇; Coulomb friction not modelled",
            "Fixed base — base_pose_world is held constant by the caller",
            "Gravity is treated as a uniform inertial field in world coords",
            "External wrenches are expressed in the world frame and applied at each link's inertial origin",
            "URDF fixed joints are merged at parse time (handled by mime.control.kinematics.parse_urdf)",
        ),
        limitations=(
            "No contact / collision detection — wrist or links can pass through obstacles",
            "Joint limit clipping is hard (no compliant stop) — q̇ is not zeroed at the wall",
            "Rigid joints only — no link or joint flexibility",
            "Single solve per step — no fixed-point iteration for inter-node coupling beyond what GraphManager supplies",
        ),
        validated_regimes=(
            ValidatedRegime("num_joints", 1.0, 10.0, "-",
                            "Tree size validated on 3-link planar fixture"),
            ValidatedRegime("timestep", 1e-4, 1e-2, "s",
                            "Stable for typical revolute manipulators"),
        ),
        references=(
            Reference(
                "Featherstone2008",
                "Rigid Body Dynamics Algorithms — CRBA, RNEA, spatial-vector form",
            ),
            Reference(
                "Sciavicco2000",
                "Modelling and Control of Robot Manipulators — Jacobian transpose mapping",
            ),
        ),
        hazard_hints=(
            "fully rigid joints / no flexibility",
            "no contact / collision detection",
            "joint limit clipping is hard (no compliant stop)",
            "gravity assumed inertial",
        ),
        implementation_map={
            "M(q) — joint-space mass matrix (CRBA)": (
                "mime.control.kinematics.crba.mass_matrix"
            ),
            "c(q, qd) + g(q) — nonlinear bias (RNEA)": (
                "mime.control.kinematics.rnea.nonlinear_bias"
            ),
            "J_i — geometric Jacobian for link i": (
                "mime.control.kinematics.fk.frame_jacobian"
            ),
            "tau_ext = sum_i J_i^T F_ext_i": (
                "mime.nodes.actuation.robot_arm._wrench_to_joint_torques"
            ),
            "qdd = M^{-1} (tau_eff + tau_ext - bias)": (
                "mime.nodes.actuation.robot_arm.RobotArmNode.update"
            ),
            "Semi-implicit Euler q, qd update + clip to joint limits": (
                "mime.nodes.actuation.robot_arm.RobotArmNode.update"
            ),
            "Forward kinematics for end-effector & link poses": (
                "mime.control.kinematics.fk.link_world_poses"
            ),
            "Pose composition base ⊕ link ⊕ tool_offset": (
                "mime.nodes.actuation.robot_arm.RobotArmNode."
                "compute_boundary_fluxes"
            ),
        },
    )

    # mime_meta is built at __init__ from the parsed URDF, since num_dof
    # depends on the supplied URDF. We seed a coarse class-level meta so
    # MimeNode's class-level invariant (mime_meta is not None on the
    # class) is satisfied; instances overwrite ``self.mime_meta`` with
    # the full per-arm version.
    mime_meta = MimeNodeMeta(
        role=NodeRole.EXTERNAL_APPARATUS,
        actuation=ActuationMeta(
            principle=ActuationPrinciple.ARTICULATED_ARM,
            is_onboard=False,
            commandable_fields=("commanded_joint_torques",),
        ),
        articulated_arm=ArticulatedArmMeta(
            num_dof=0,
            convention="urdf",
            urdf_path=None,
            joint_friction_modelled=False,
            commandable_fields=("commanded_joint_torques",),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        urdf_path: str,
        end_effector_link_name: str,
        base_pose_world: tuple = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        end_effector_offset_in_link: tuple = (
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ),
        joint_friction_n_m_s: Optional[tuple] = None,
        gravity_world: tuple = (0.0, 0.0, -9.80665),
        joint_limit_override: Optional[tuple] = None,
        **kwargs,
    ):
        if len(base_pose_world) != 7:
            raise ValueError(
                f"base_pose_world must be length 7 (x,y,z,qw,qx,qy,qz), got "
                f"{len(base_pose_world)}"
            )
        if len(end_effector_offset_in_link) != 7:
            raise ValueError(
                f"end_effector_offset_in_link must be length 7, got "
                f"{len(end_effector_offset_in_link)}"
            )
        if len(gravity_world) != 3:
            raise ValueError(
                f"gravity_world must be length 3, got {len(gravity_world)}"
            )

        # Parse URDF once at construction time. The resulting
        # KinematicTree is a JAX-static pytree (numpy-backed, hashable
        # by identity) — passing it through ``self`` does not break
        # tracing.
        tree = parse_urdf(urdf_path)
        n = tree.num_joints

        # Resolve EE link name now so the lookup never reaches the
        # JAX trace.
        ee_idx = _resolve_link_index(tree, end_effector_link_name)

        # Determine joint-friction values:
        #   - explicit override → use as-is (length must match N)
        #   - None              → fall back to URDF damping
        if joint_friction_n_m_s is None:
            friction = np.asarray(tree.joint_friction, dtype=np.float64).copy()
        else:
            if len(joint_friction_n_m_s) != n:
                raise ValueError(
                    f"joint_friction_n_m_s length {len(joint_friction_n_m_s)} "
                    f"!= num_joints {n}"
                )
            friction = np.asarray(joint_friction_n_m_s, dtype=np.float64).copy()

        # Joint limits: override or URDF
        if joint_limit_override is None:
            limits = np.asarray(tree.joint_limits, dtype=np.float64).copy()
        else:
            limits = np.asarray(joint_limit_override, dtype=np.float64)
            if limits.shape != (n, 2):
                raise ValueError(
                    f"joint_limit_override shape {limits.shape} != ({n}, 2)"
                )

        # joint_friction_modelled flag = any nonzero entry
        friction_modelled = bool(np.any(friction != 0.0))

        super().__init__(
            name, timestep,
            # Use object dtype storage in self.params so the parameters
            # remain Python-side static; we re-cast to jnp at update
            # time.
            urdf_path=str(urdf_path),
            base_pose_world=tuple(float(c) for c in base_pose_world),
            end_effector_link_name=str(end_effector_link_name),
            end_effector_offset_in_link=tuple(
                float(c) for c in end_effector_offset_in_link
            ),
            joint_friction_n_m_s=tuple(float(x) for x in friction.tolist()),
            gravity_world=tuple(float(c) for c in gravity_world),
            joint_limits=tuple(
                tuple(float(x) for x in row) for row in limits.tolist()
            ),
            **kwargs,
        )

        # Static (Python-side) state captured at init time. These are
        # NOT in ``self.params`` because they are non-numeric / non-
        # serialisable.
        self._tree: KinematicTree = tree
        self._ee_idx: int = ee_idx
        self._num_joints: int = n

        # Refresh mime_meta with per-instance num_dof and friction flag.
        self.mime_meta = MimeNodeMeta(
            role=NodeRole.EXTERNAL_APPARATUS,
            actuation=ActuationMeta(
                principle=ActuationPrinciple.ARTICULATED_ARM,
                is_onboard=False,
                commandable_fields=("commanded_joint_torques",),
            ),
            articulated_arm=ArticulatedArmMeta(
                num_dof=n,
                convention="urdf",
                urdf_path=str(urdf_path),
                joint_friction_modelled=friction_modelled,
                commandable_fields=("commanded_joint_torques",),
            ),
        )

    # ------------------------------------------------------------------
    # State / spec
    # ------------------------------------------------------------------

    def initial_state(self) -> dict:
        n = self._num_joints
        # Seed link_poses_world / end_effector_pose_world to identity-at-base
        # so the dict shape is fixed; the first ``update`` call overwrites
        # them with the FK of the actual joint angles.
        identity_pose = _identity_pose7()
        return {
            "joint_angles":     jnp.zeros(n),
            "joint_velocities": jnp.zeros(n),
            "link_poses_world": jnp.tile(identity_pose, (n, 1)),
            "end_effector_pose_world": identity_pose,
        }

    def state_fields(self) -> list[str]:
        return [
            "joint_angles", "joint_velocities",
            "link_poses_world", "end_effector_pose_world",
        ]

    def observable_fields(self) -> list[str]:
        return [
            "joint_angles", "joint_velocities",
            "link_poses_world", "end_effector_pose_world",
        ]

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        n = self._num_joints
        return {
            "commanded_joint_torques": BoundaryInputSpec(
                shape=(n,),
                default=jnp.zeros(n),
                coupling_type="additive",
                description="Per-joint commanded torques [N.m]",
                expected_units="N.m",
            ),
            "external_wrenches_per_link": BoundaryInputSpec(
                shape=(n, 6),
                default=jnp.zeros((n, 6)),
                coupling_type="additive",
                description=(
                    "Per-link world-frame 6-vector wrench [F_lin (3); "
                    "F_ang (3)]; mapped to joint torques via J_i^T."
                ),
                expected_units="N, N.m",
            ),
        }

    # ------------------------------------------------------------------
    # Update — semi-implicit Euler, pure JAX
    # ------------------------------------------------------------------

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        n = self._num_joints
        tree = self._tree

        q  = state["joint_angles"]
        qd = state["joint_velocities"]

        tau_cmd = boundary_inputs.get(
            "commanded_joint_torques", jnp.zeros(n, dtype=q.dtype),
        )
        wrenches = boundary_inputs.get(
            "external_wrenches_per_link", jnp.zeros((n, 6), dtype=q.dtype),
        )

        friction = jnp.asarray(self.params["joint_friction_n_m_s"], dtype=q.dtype)
        limits = jnp.asarray(self.params["joint_limits"], dtype=q.dtype)
        g_world = jnp.asarray(self.params["gravity_world"], dtype=q.dtype)

        # Effective joint torques after viscous friction
        tau_eff = tau_cmd - friction * qd

        # External wrench → joint torque via J^T
        tau_ext = _wrench_to_joint_torques(tree, q, wrenches)

        # Mass matrix and bias (Coriolis + gravity)
        M = mass_matrix(tree, q)
        bias = nonlinear_bias(tree, q, qd, g_world)

        # Joint accelerations
        rhs = tau_eff + tau_ext - bias
        qdd = jnp.linalg.solve(M, rhs)

        # Semi-implicit Euler
        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        # Hard clip to joint limits (no compliant stop — see hazard hint)
        q_new = jnp.clip(q_new, limits[:, 0], limits[:, 1])

        # Compute per-link world poses + EE pose and store them in
        # state, so downstream consumers (graph edges, runner result
        # frame) can read them by name from ``state["arm"]`` without
        # invoking compute_boundary_fluxes. Uses the URDF *link frame*
        # (= joint origin) rather than the inertial COM frame so visual
        # meshes attach correctly.
        from mime.control.kinematics.fk import joint_to_world_transforms
        from mime.control.kinematics.transform import (
            pose_to_matrix, _rotation_matrix_to_quat,
        )
        base_pose = jnp.asarray(self.params["base_pose_world"], dtype=q.dtype)
        ee_offset = jnp.asarray(
            self.params["end_effector_offset_in_link"], dtype=q.dtype,
        )
        link_xforms = joint_to_world_transforms(tree, q_new)
        T_base = pose_to_matrix(base_pose)
        link_xforms = T_base[None] @ link_xforms
        link_t = link_xforms[:, :3, 3]
        link_R = link_xforms[:, :3, :3]
        link_quats = jax.vmap(_rotation_matrix_to_quat)(link_R)
        link_poses = jnp.concatenate([link_t, link_quats], axis=1)

        # End-effector world pose: EE-link world transform composed with
        # the static tool offset.
        T_ee_link = link_xforms[self._ee_idx]
        T_ee_offset = pose_to_matrix(ee_offset)
        T_ee = T_ee_link @ T_ee_offset
        ee_t = T_ee[:3, 3]
        ee_q = _rotation_matrix_to_quat(T_ee[:3, :3])
        ee_pose = jnp.concatenate([ee_t, ee_q])

        return {
            "joint_angles":     q_new,
            "joint_velocities": qd_new,
            "link_poses_world": link_poses,
            "end_effector_pose_world": ee_pose,
        }

    # ------------------------------------------------------------------
    # Boundary fluxes — exposed to downstream nodes
    # ------------------------------------------------------------------

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        # Note: ``link_poses_world`` and ``end_effector_pose_world`` are
        # ALSO stored in state by ``update`` so the runner can read them
        # by name. We *recompute* them here because callers (tests, ad-
        # hoc graph queries) may pass a hand-built state without those
        # keys; XLA hoists the duplicate FK call into a single kernel
        # when this runs inside the same compiled gm.step.
        n = self._num_joints
        tree = self._tree

        q  = state["joint_angles"]
        qd = state["joint_velocities"]

        base_pose = jnp.asarray(self.params["base_pose_world"], dtype=q.dtype)
        ee_offset = jnp.asarray(
            self.params["end_effector_offset_in_link"], dtype=q.dtype,
        )

        # Use joint_to_world_transforms (URDF link frames), not
        # link_world_poses (COM frames) — visual meshes are anchored
        # to the link frame, and downstream consumers (the motor's
        # parent_pose_world) need the actual physical EE position. See
        # update() above for the matching convention.
        from mime.control.kinematics.fk import joint_to_world_transforms
        from mime.control.kinematics.transform import (
            pose_to_matrix, _rotation_matrix_to_quat,
        )
        link_xforms = joint_to_world_transforms(tree, q)
        T_base = pose_to_matrix(base_pose)
        link_xforms = T_base[None] @ link_xforms
        link_t = link_xforms[:, :3, 3]
        link_R = link_xforms[:, :3, :3]
        link_quats = jax.vmap(_rotation_matrix_to_quat)(link_R)
        link_poses = jnp.concatenate([link_t, link_quats], axis=1)
        T_ee_link = link_xforms[self._ee_idx]
        T_ee_offset = pose_to_matrix(ee_offset)
        T_ee = T_ee_link @ T_ee_offset
        ee_t = T_ee[:3, 3]
        ee_q = _rotation_matrix_to_quat(T_ee[:3, :3])
        ee_pose = jnp.concatenate([ee_t, ee_q])

        tau_cmd = boundary_inputs.get(
            "commanded_joint_torques", jnp.zeros(n, dtype=q.dtype),
        )
        wrenches = boundary_inputs.get(
            "external_wrenches_per_link", jnp.zeros((n, 6), dtype=q.dtype),
        )
        friction = jnp.asarray(self.params["joint_friction_n_m_s"], dtype=q.dtype)
        tau_ext = _wrench_to_joint_torques(tree, q, wrenches)
        tau_actual = tau_cmd - friction * qd + tau_ext

        return {
            "end_effector_pose_world": ee_pose,
            "link_poses_world":        link_poses,
            "joint_angles":            q,
            "joint_velocities":        qd,
            "joint_torques_actual":    tau_actual,
        }
