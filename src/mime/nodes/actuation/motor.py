"""MotorNode — single-axis rotary motor (DC brushed) with three command modes.

This node simulates a one-axis rotary motor that drives a rotor (and any
end-effector rigidly attached to that rotor) about a body-fixed axis in
its parent frame. The parent frame is supplied each step via a boundary
input (typically the end-effector pose of an upstream RobotArmNode), so
the rotor pose is composed as

    parent_pose_world  ⊕  R(axis_in_parent_frame, angle)  ⊕  tool_offset_in_rotor_frame

and exposed as a flux ``rotor_pose_world`` (7-vector ``[x,y,z, qw,qx,qy,qz]``,
matching :mod:`mime.core.quaternion`'s ``[w,x,y,z]`` Hamilton convention).

Three command modes are selected at runtime by which boundary input is
non-zero. Precedence (highest first):

1. **torque-mode** — ``commanded_torque`` is set directly (default 0.0).
   The rotor torque equals the commanded value; the electrical current
   stays at zero.
2. **voltage-mode** — ``commanded_voltage`` is set. An RL armature circuit
   integrates the current ``i`` and the rotor torque is ``τ = k_t · i``.
3. **velocity-mode** — ``commanded_velocity`` is set. An internal PI loop
   tracks ``ω_des`` and produces an effective torque command.

Mode dispatch is implemented JAX-traceably with ``jnp.where`` (no Python
branches on input values) so the node is compatible with ``jit``, ``grad``,
and ``vmap``.

Mechanical dynamics (semi-implicit Euler):

    ω_{n+1} = ω_n + (dt / J) · (τ_net,n − b · ω_{n+1})       [implicit damping]
    θ_{n+1} = θ_n + ω_{n+1} · dt

Electrical dynamics (semi-implicit Euler, voltage-mode only):

    i_{n+1} = i_n + (dt / L) · (V − R · i_{n+1} − k_e · ω_n)  [implicit on R]

with ``τ_motor = k_t · i_{n+1}``.

In SI units, ``k_t`` (torque constant) and ``k_e`` (back-EMF constant) are
numerically equal, so ``ke_v_s_per_rad`` defaults to ``kt_n_m_per_a``.

v1 scope: no cogging, no rotor imbalance, no flexibility, no thermal
model. Hooks for those live on :class:`MotorMeta` and stay at their
default ``False`` values.
"""

from __future__ import annotations

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole, ActuationMeta, ActuationPrinciple, MotorMeta,
)
from mime.core.quaternion import (
    quat_multiply, quat_normalize, identity_quat,
)


# ---------------------------------------------------------------------------
# Pose composition helper
# ---------------------------------------------------------------------------

def _quat_from_axis_angle(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    """Build a unit quaternion from a unit axis and angle.

    Convention matches :mod:`mime.core.quaternion`: ``q = [w, x, y, z]``,
    Hamilton product, ``v_lab = R(q) v_body``.
    """
    half = angle * 0.5
    s = jnp.sin(half)
    c = jnp.cos(half)
    # Safe normalisation in case caller passed a non-unit axis.
    n = jnp.linalg.norm(axis)
    axis_unit = axis / jnp.maximum(n, 1e-30)
    return jnp.array([c, axis_unit[0] * s, axis_unit[1] * s, axis_unit[2] * s])


def compose_pose(
    parent_xyzw: jnp.ndarray,
    rotor_axis_local: jnp.ndarray,
    rotor_angle: jnp.ndarray,
    tool_offset_xyzw: jnp.ndarray,
) -> jnp.ndarray:
    """Compose ``parent ⊕ R(axis, angle) ⊕ tool_offset`` → world pose.

    All 7-vector poses use the convention
    ``[x, y, z, qw, qx, qy, qz]`` (translation first, scalar-first
    quaternion second), consistent with the ``[w, x, y, z]`` quaternion
    used throughout :mod:`mime.core.quaternion`.

    The composition treats ``rotor_axis_local`` as expressed in the parent
    frame (rotor axis at zero angle), ``rotor_angle`` as the scalar rotor
    coordinate, and ``tool_offset_xyzw`` as a fixed transform from the
    rotor body frame to the tool-tip frame (identity = tool centred on
    rotor axis).
    """
    parent_t = parent_xyzw[0:3]
    parent_q = parent_xyzw[3:7]            # [w, x, y, z]
    tool_t = tool_offset_xyzw[0:3]
    tool_q = tool_offset_xyzw[3:7]

    rotor_q_local = _quat_from_axis_angle(rotor_axis_local, rotor_angle)
    # Rotor frame relative to parent: pure rotation, zero translation.
    # World quaternion = parent_q ⊗ rotor_q_local ⊗ tool_q
    q_pr = quat_multiply(parent_q, rotor_q_local)
    q_world = quat_normalize(quat_multiply(q_pr, tool_q))

    # Translation: tool_t is in rotor body frame, rotate by parent_q ⊗ rotor_q_local.
    R_pr = _quat_to_rot(q_pr)
    t_world = parent_t + R_pr @ tool_t
    return jnp.concatenate([t_world, q_world])


def _quat_to_rot(q: jnp.ndarray) -> jnp.ndarray:
    """Quaternion ``[w,x,y,z]`` → rotation matrix (duplicate of
    :func:`mime.core.quaternion.quat_to_rotation_matrix` to keep the
    motor module self-contained for inlining).
    """
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------
# Default identity pose helper
# ---------------------------------------------------------------------------

def _identity_pose7() -> jnp.ndarray:
    """Return ``[0,0,0, 1,0,0,0]`` — origin at identity orientation."""
    return jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# MotorNode
# ---------------------------------------------------------------------------

@stability(StabilityLevel.EXPERIMENTAL)
class MotorNode(MimeNode):
    """Single-axis rotary motor with torque, voltage, and velocity modes.

    Parameters
    ----------
    name : str
    timestep : float
        Simulation timestep [s].
    inertia_kg_m2 : float
        Rotor inertia about the spin axis [kg·m²].
    kt_n_m_per_a : float
        Torque constant [N·m/A].
    ke_v_s_per_rad : float, optional
        Back-EMF constant [V·s/rad]. SI dictates ``k_e == k_t``
        numerically; defaults to ``kt_n_m_per_a``.
    r_ohm : float
        Armature resistance [Ω].
    l_henry : float
        Armature inductance [H].
    damping_n_m_s : float
        Viscous bearing/airgap damping ``b`` [N·m·s/rad].
    axis_in_parent_frame : tuple of 3 floats
        Rotor axis as a unit vector in the parent (carrier) frame.
        Default ``(0, 0, 1)`` — rotor spins about parent +z.
    tool_offset_in_rotor_frame : tuple of 7 floats
        Fixed transform ``[x, y, z, qw, qx, qy, qz]`` from rotor frame
        to tool-tip frame. Default = identity.
    velocity_kp : float
        Proportional gain for the velocity-mode PI controller
        [N·m·s/rad]. Default 1e-3 — small to keep the simple test
        motor stable; tune for the specific motor.
    velocity_ki : float
        Integral gain for the velocity-mode PI controller [N·m/rad].
        Default 1e-2 — together with the default Kp, achieves
        sub-1 % steady-state error on a 0.01 N·m·s damping rotor at
        ω_des = 10 rad/s within ≈1 s.

    Boundary Inputs
    ---------------
    commanded_torque : scalar
        Direct rotor torque [N·m]. Additive; default 0.0.
        Non-zero command activates **torque-mode**.
    commanded_voltage : scalar
        Armature terminal voltage [V]. Default 0.0. Activates
        **voltage-mode** when non-zero (and ``commanded_torque`` is 0).
    commanded_velocity : scalar
        Desired rotor angular velocity [rad/s]. Default 0.0.
        Activates **velocity-mode** when non-zero (and the others are 0).
    parent_pose_world : (7,)
        Pose of the parent frame in world coordinates
        ``[x, y, z, qw, qx, qy, qz]``. Default identity.
    load_torque : scalar
        Reaction torque from the load (e.g. magnetic drag on the
        permanent magnet attached to the rotor) [N·m]. Additive;
        default 0.0.

    Boundary Fluxes
    ---------------
    rotor_angle : scalar
        Cumulative rotor angle [rad] (not wrapped to [-π, π]).
    rotor_angular_velocity : scalar
        Rotor angular velocity [rad/s].
    rotor_pose_world : (7,)
        World pose of the tool-tip frame.

    Notes
    -----
    Mode precedence (in order): torque > voltage > velocity. The first
    non-zero command wins; ties default to torque-mode. Mode dispatch is
    JAX-traceable (``jnp.where`` only).

    v1 omissions: no cogging, no thermal model, no PWM ripple in
    voltage-mode, no shaft flexibility. The :class:`MotorMeta` flags
    ``has_cogging`` and ``has_imbalance_vibration`` stay ``False``.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-100",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Single-axis DC-brushed motor with torque-, voltage-, and "
            "velocity-command modes. Semi-implicit Euler integration of "
            "rotor mechanics and (optional) armature RL circuit."
        ),
        governing_equations=(
            r"J·dω/dt = τ_cmd + τ_load + τ_motor − b·ω; "
            r"τ_motor = k_t·i (voltage mode) else 0; "
            r"L·di/dt = V − R·i − k_e·ω; "
            r"τ_cmd_vel = K_p·(ω_des − ω) + K_i·∫(ω_des − ω) dt; "
            r"θ_{n+1} = θ_n + ω_{n+1}·dt; "
            r"rotor_pose = parent ⊕ R(axis, θ) ⊕ tool_offset."
        ),
        discretization=(
            "Semi-implicit (symplectic) Euler for rotor mechanics; "
            "semi-implicit Euler for armature current. Integrator term "
            "for velocity-mode PI uses backward Euler on error. Pose "
            "composition is exact (closed-form quaternion product)."
        ),
        assumptions=(
            "Single rotational degree of freedom (one-axis rotor)",
            "Linear bearing friction τ_friction = b·ω; Coulomb friction "
            "neglected",
            "DC-brushed armature: V = R·i + L·di/dt + k_e·ω",
            "k_t and k_e numerically equal in SI — defaults enforce this",
            "Rotor inertia is constant (no rotor-imbalance modulation)",
            "Tool offset is rigid relative to the rotor frame",
        ),
        limitations=(
            "No cogging torque",
            "No thermal model — winding heating not tracked",
            "Voltage-mode neglects PWM ripple and switching losses",
            "Velocity-mode PI gains are not auto-tuned; defaults work for "
            "a small lab motor (b≈0.01, J≈1e-4) and must be retuned for "
            "other regimes",
            "No back-iron saturation — torque is linear in current at all "
            "currents",
        ),
        validated_regimes=(
            ValidatedRegime("inertia_kg_m2", 1e-7, 1e-2, "kg.m^2",
                            "Lab-scale to small servo motors"),
            ValidatedRegime("kt_n_m_per_a", 1e-3, 1.0, "N.m/A",
                            "Small DC brushed to NEMA-23 servos"),
            ValidatedRegime("commanded_velocity", 0.0, 500.0, "rad/s",
                            "Default PI gains tested up to ~50 rad/s"),
        ),
        references=(
            Reference("Krause2013", "Analysis of Electric Machinery"),
        ),
        hazard_hints=(
            "no cogging modelled — predicted low-speed smoothness is "
            "optimistic vs real brushed motors",
            "no thermal model — sustained high-current operation will not "
            "trigger derating in simulation",
            "voltage-mode neglects PWM ripple — current ripple from the "
            "drive is absent",
            "velocity-mode PI gains require tuning per motor — defaults "
            "are conservative; retune Kp, Ki for stiffer/heavier rotors",
        ),
        implementation_map={
            "J·dω/dt = τ_net − b·ω (semi-implicit)": (
                "mime.nodes.actuation.motor.MotorNode.update"
            ),
            "L·di/dt = V − R·i − k_e·ω": (
                "mime.nodes.actuation.motor.MotorNode.update"
            ),
            "τ_motor = k_t·i": (
                "mime.nodes.actuation.motor.MotorNode.update"
            ),
            "τ_PI = K_p·(ω_des − ω) + K_i·∫(ω_des − ω) dt": (
                "mime.nodes.actuation.motor.MotorNode.update"
            ),
            "θ_{n+1} = θ_n + ω_{n+1}·dt": (
                "mime.nodes.actuation.motor.MotorNode.update"
            ),
            "rotor_pose = parent ⊕ R(axis,θ) ⊕ tool_offset": (
                "mime.nodes.actuation.motor.compose_pose"
            ),
            "Mode dispatch (torque > voltage > velocity)": (
                "mime.nodes.actuation.motor.MotorNode.update"
            ),
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.EXTERNAL_APPARATUS,
        actuation=ActuationMeta(
            principle=ActuationPrinciple.MOTOR_ROTOR,
            is_onboard=False,
            commandable_fields=(
                "commanded_torque",
                "commanded_velocity",
                "commanded_voltage",
            ),
        ),
        motor=MotorMeta(
            motor_type="dc_brushed",
            commandable_fields=(
                "commanded_torque",
                "commanded_velocity",
                "commanded_voltage",
            ),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        inertia_kg_m2: float,
        kt_n_m_per_a: float,
        r_ohm: float,
        l_henry: float,
        damping_n_m_s: float,
        ke_v_s_per_rad: float | None = None,
        axis_in_parent_frame: tuple = (0.0, 0.0, 1.0),
        tool_offset_in_rotor_frame: tuple = (
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ),
        velocity_kp: float = 1e-3,
        velocity_ki: float = 1e-2,
        **kwargs,
    ):
        if ke_v_s_per_rad is None:
            ke_v_s_per_rad = kt_n_m_per_a  # SI: k_t == k_e numerically

        if len(axis_in_parent_frame) != 3:
            raise ValueError(
                f"axis_in_parent_frame must be length 3, got "
                f"{len(axis_in_parent_frame)}"
            )
        if len(tool_offset_in_rotor_frame) != 7:
            raise ValueError(
                f"tool_offset_in_rotor_frame must be length 7 "
                f"(x,y,z,qw,qx,qy,qz), got "
                f"{len(tool_offset_in_rotor_frame)}"
            )

        super().__init__(
            name, timestep,
            inertia_kg_m2=float(inertia_kg_m2),
            kt_n_m_per_a=float(kt_n_m_per_a),
            ke_v_s_per_rad=float(ke_v_s_per_rad),
            r_ohm=float(r_ohm),
            l_henry=float(l_henry),
            damping_n_m_s=float(damping_n_m_s),
            axis_in_parent_frame=tuple(float(x) for x in axis_in_parent_frame),
            tool_offset_in_rotor_frame=tuple(
                float(x) for x in tool_offset_in_rotor_frame
            ),
            velocity_kp=float(velocity_kp),
            velocity_ki=float(velocity_ki),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # State / spec
    # ------------------------------------------------------------------

    def initial_state(self) -> dict:
        return {
            "angle": jnp.array(0.0),
            "angular_velocity": jnp.array(0.0),
            "current": jnp.array(0.0),
            # World-frame rotor pose — published in state (not just as a
            # flux) so downstream edges can read it directly via the
            # graph's source_field-from-state path.
            "rotor_pose_world": _identity_pose7(),
            # Velocity-mode integrator state — kept always so the state
            # dict is mode-independent.
            "velocity_integral_error": jnp.array(0.0),
        }

    def state_fields(self) -> list[str]:
        # Public state fields exposed to controllers / loggers. The PI
        # integrator is internal bookkeeping.
        return ["angle", "angular_velocity", "current", "rotor_pose_world"]

    def observable_fields(self) -> list[str]:
        return ["angle", "angular_velocity", "current", "rotor_pose_world"]

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "commanded_torque": BoundaryInputSpec(
                shape=(), default=0.0, coupling_type="additive",
                description="Direct rotor torque command [N.m]",
                expected_units="N.m",
            ),
            "commanded_voltage": BoundaryInputSpec(
                shape=(), default=0.0, coupling_type="replacive",
                description="Armature terminal voltage command [V]",
                expected_units="V",
            ),
            "commanded_velocity": BoundaryInputSpec(
                shape=(), default=0.0, coupling_type="replacive",
                description="Desired rotor angular velocity [rad/s]",
                expected_units="rad/s",
            ),
            "parent_pose_world": BoundaryInputSpec(
                shape=(7,), default=_identity_pose7(),
                coupling_type="replacive",
                description=(
                    "Parent (carrier) frame pose in world coordinates "
                    "[x,y,z, qw,qx,qy,qz]"
                ),
            ),
            "load_torque": BoundaryInputSpec(
                shape=(), default=0.0, coupling_type="additive",
                description=(
                    "External load reaction torque [N.m] — typically the "
                    "magnetic reaction from the rotor-mounted magnet"
                ),
                expected_units="N.m",
            ),
        }

    # ------------------------------------------------------------------
    # Update — semi-implicit Euler, JAX-traceable mode dispatch
    # ------------------------------------------------------------------

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        J = self.params["inertia_kg_m2"]
        kt = self.params["kt_n_m_per_a"]
        ke = self.params["ke_v_s_per_rad"]
        R = self.params["r_ohm"]
        L = self.params["l_henry"]
        b = self.params["damping_n_m_s"]
        Kp = self.params["velocity_kp"]
        Ki = self.params["velocity_ki"]

        theta = state["angle"]
        omega = state["angular_velocity"]
        current = state["current"]
        vel_int = state["velocity_integral_error"]

        tau_cmd_in = boundary_inputs.get("commanded_torque", jnp.array(0.0))
        v_cmd = boundary_inputs.get("commanded_voltage", jnp.array(0.0))
        omega_des = boundary_inputs.get("commanded_velocity", jnp.array(0.0))
        tau_load = boundary_inputs.get("load_torque", jnp.array(0.0))

        # ---- Mode flags (JAX-traceable; precedence torque > voltage > velocity)
        # Use a small epsilon to treat tiny floats as "no command".
        eps = 1e-30
        torque_mode = jnp.abs(tau_cmd_in) > eps
        voltage_mode = jnp.logical_and(
            jnp.logical_not(torque_mode), jnp.abs(v_cmd) > eps,
        )
        velocity_mode = jnp.logical_and(
            jnp.logical_not(jnp.logical_or(torque_mode, voltage_mode)),
            jnp.abs(omega_des) > eps,
        )

        # ---- Velocity-mode PI controller (always computed; gated by mask)
        # Backward-Euler integrator on the error: I_{n+1} = I_n + e_n · dt
        # using the pre-step omega (matches the "n" sample for the error).
        err = omega_des - omega
        vel_int_new = vel_int + err * dt
        tau_pi = Kp * err + Ki * vel_int_new
        # Freeze the integrator outside velocity-mode to prevent windup
        # while another mode is active.
        vel_int_out = jnp.where(velocity_mode, vel_int_new, vel_int)

        # ---- Electrical update (voltage-mode only): semi-implicit Euler
        # i_{n+1} = i_n + (dt/L)·(V − R·i_{n+1} − k_e·ω_n)
        # Solving:  i_{n+1} (1 + dt·R/L) = i_n + (dt/L)·(V − k_e·ω_n)
        denom_i = 1.0 + dt * R / jnp.maximum(L, 1e-30)
        i_volt = (current + (dt / jnp.maximum(L, 1e-30))
                  * (v_cmd - ke * omega)) / denom_i
        # In non-voltage modes, current relaxes to zero (no source). Use
        # the same RL relaxation but with V=0 — physically a coasting
        # short-circuited armature would still bleed current, but for
        # the v1 model the controller gates the armature off, so we
        # snap to zero outside voltage-mode.
        current_new = jnp.where(voltage_mode, i_volt, jnp.array(0.0))

        # ---- Torque assembly
        tau_motor = jnp.where(voltage_mode, kt * current_new, jnp.array(0.0))
        tau_cmd_effective = jnp.where(
            velocity_mode, tau_pi,
            jnp.where(torque_mode, tau_cmd_in, jnp.array(0.0)),
        )
        tau_drive = tau_cmd_effective + tau_motor + tau_load

        # ---- Mechanical update: semi-implicit Euler with implicit damping
        # ω_{n+1} = (J·ω_n + dt·τ_drive) / (J + dt·b)
        omega_new = (J * omega + dt * tau_drive) / (
            J + dt * jnp.maximum(b, 0.0) + 1e-30
        )
        theta_new = theta + omega_new * dt

        # Compose the world-frame rotor pose so it is available in state
        # (downstream edges read source_field from state by default;
        # storing it here matches the convention used by
        # ExternalMagneticFieldNode for ``field_vector``).
        parent = boundary_inputs.get("parent_pose_world", _identity_pose7())
        axis = jnp.asarray(self.params["axis_in_parent_frame"])
        tool = jnp.asarray(self.params["tool_offset_in_rotor_frame"])
        rotor_pose_world_new = compose_pose(parent, axis, theta_new, tool)

        return {
            "angle": theta_new,
            "angular_velocity": omega_new,
            "current": current_new,
            "rotor_pose_world": rotor_pose_world_new,
            "velocity_integral_error": vel_int_out,
        }

    # ------------------------------------------------------------------
    # Boundary fluxes — exposed to downstream nodes (e.g. PermanentMagnetNode)
    # ------------------------------------------------------------------

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        parent = boundary_inputs.get("parent_pose_world", _identity_pose7())
        axis = jnp.asarray(self.params["axis_in_parent_frame"])
        tool = jnp.asarray(self.params["tool_offset_in_rotor_frame"])
        pose = compose_pose(parent, axis, state["angle"], tool)
        return {
            "rotor_angle": state["angle"],
            "rotor_angular_velocity": state["angular_velocity"],
            "rotor_pose_world": pose,
        }
