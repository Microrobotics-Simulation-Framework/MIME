"""RigidBodyNode — 6-DOF rigid body dynamics for microrobots.

Three modes:
1. **Overdamped + analytical drag** (default): V = R_T^{-1} @ F_ext.
   Stokes regime, inertia negligible, drag computed from Oberbeck-Stechert.
2. **Overdamped + external drag** (use_analytical_drag=False): same algebra
   but drag_force/drag_torque come from IB-LBM via boundary inputs.
3. **Inertial + external drag** (use_inertial=True): Newton's 2nd law
   with explicit Euler integration. Required for FSI coupling where the
   overdamped model causes step-0 blowup (zero drag → infinite omega).

State: position (3D) + orientation (quaternion) + velocity + angular velocity.

Reference: Ch 2 and Ch 4 of "Mathematical Modelling of Swimming Soft Microrobots"
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole, BiocompatibilityMeta, BiocompatibilityClass,
    AnatomicalRegimeMeta, AnatomicalCompartment, FlowRegime,
)
from mime.core.quaternion import (
    quat_normalize, quat_from_angular_velocity, quat_multiply, identity_quat,
)


def oberbeck_stechert_coefficients(e: float) -> tuple:
    """Compute Oberbeck-Stechert drag coefficients for a prolate ellipsoid.

    Parameters
    ----------
    e : float or jnp scalar
        Eccentricity = sqrt(1 - b^2/a^2). Range [0, 1).

    Returns
    -------
    C_1, C_2, C_3 : drag coefficients (dimensionless)

    For a sphere (e -> 0): C_1 = C_2 = 1, C_3 = 1.
    Uses jnp.where to handle the e -> 0 singularity safely.
    """
    e2 = e * e

    # ``log((1+e)/(1-e)) == 2 * atanh(e) == 2*(e + e^3/3 + e^5/5 + ...)``.
    # Both denominators below contain a leading ``±2e`` that cancels
    # analytically against the ``2e`` term of ``2*atanh(e)``. Computing
    # them in the naive ``-2e + (1+e^2)*log(...)`` form loses that
    # cancellation and is catastrophically inaccurate in float32 near
    # e=0 (e.g. C_1 = 1.037 instead of 1.000 at e=0.01 — only masked when
    # a test happened to run under a leaked jax_enable_x64). Reformulate
    # in terms of the cancellation-free ``g = atanh(e) - e = e^3/3 +
    # e^5/5 + ...`` so the near-sphere limit is accurate in single
    # precision too. Small e: Maclaurin series (fast convergence for
    # e < ~0.3). Larger e: direct atanh, where no cancellation occurs.
    g_series = e2 * e * (
        1.0 / 3.0 + e2 * (1.0 / 5.0 + e2 * (1.0 / 7.0 + e2 / 9.0))
    )
    g_direct = jnp.arctanh(jnp.minimum(e, 1.0 - 1e-7)) - e
    g = jnp.where(e < 0.25, g_series, g_direct)

    # log_term retained for any downstream use / clarity; equals the
    # original jnp.log((1+e)/(1-e)) but reconstructed cancellation-free.
    log_term = 2.0 * (e + g)

    # Denominators (zero at e=0), in cancellation-free form:
    #   denom_1 = -2e + (1+e^2)*2(e+g) = 2g(1+e^2) + 2e^3
    #   denom_2 =  2e + (3e^2-1)*2(e+g) = 6e^3 + 2g(3e^2-1)
    denom_1 = 2.0 * g * (1.0 + e2) + 2.0 * e * e2
    denom_2 = 6.0 * e * e2 + 2.0 * g * (3.0 * e2 - 1.0)

    C_1_raw = (8.0 / 3.0) * e * e2 / jnp.maximum(jnp.abs(denom_1), 1e-30)
    C_2_raw = (16.0 / 3.0) * e * e2 / jnp.maximum(jnp.abs(denom_2), 1e-30)
    C_3_raw = (4.0 / 3.0) * e * e2 * (2.0 - e2) / (
        (1.0 + e2) * jnp.maximum(jnp.abs(denom_1), 1e-30)
    )

    # For e < epsilon, use sphere limit: C_1 = C_2 = C_3 = 1
    is_sphere = e < 1e-6
    C_1 = jnp.where(is_sphere, 1.0, C_1_raw)
    C_2 = jnp.where(is_sphere, 1.0, C_2_raw)
    C_3 = jnp.where(is_sphere, 1.0, C_3_raw)

    return C_1, C_2, C_3


def offcenter_resistance_si(position, q, d_knots, R_grid, vessel_axis=0,
                            d_clamp=None):
    """Interpolate the SI 6×6 resistance for a body at a lateral offset.

    The confined wall table is axisymmetric about the vessel axis, so the
    resistance at any lateral offset equals the canonical R(|offset|) — tabulated
    along body-x in ``R_grid`` over the radial knots ``d_knots`` (in body-radius /
    length_scale units, from :meth:`StokesletFluidNode.resistance_grid_si`) —
    rotated by the offset azimuth about the vessel axis.

    The azimuth is measured in the **body frame** (the body applies R there): the
    world radial offset (``position`` ⊥ ``vessel_axis``, vessel through the origin)
    is rotated world→body by ``q⁻¹`` and lands in body x-y (body-z = the vessel
    axis). ``M = blockdiag(Rz(φ), Rz(φ))`` then rotates both the force and torque
    blocks: ``R(offset) = M · R_canonical(d) · Mᵀ``. (Note the axial sub-block —
    indices 2, 5 — is invariant under this rotation, so the force-free swim depends
    only on R_canonical(d).)
    """
    from mime.core.quaternion import rotate_vector_inverse

    d_knots = jnp.asarray(d_knots)
    R_grid = jnp.asarray(R_grid)
    if d_clamp is None:
        d_clamp = d_knots[-1]

    # World radial offset (perpendicular to the vessel axis through the origin).
    e_axis = jnp.asarray(jnp.eye(3)[vessel_axis], dtype=position.dtype)
    off_world = position - jnp.dot(position, e_axis) * e_axis
    # → body frame; the in-plane (x, y) components (body-z is the vessel axis).
    off_body = rotate_vector_inverse(q, off_world)
    ox, oy = off_body[0], off_body[1]
    d = jnp.minimum(jnp.sqrt(ox * ox + oy * oy + 1e-30), d_clamp)
    phi = jnp.arctan2(oy, ox)

    # Component-wise linear interpolation R_canonical(d) over the radial grid.
    R_flat = R_grid.reshape(R_grid.shape[0], -1)            # (n_knots, 36)
    R_can = jax.vmap(lambda col: jnp.interp(d, d_knots, col),
                     in_axes=1, out_axes=0)(R_flat).reshape(6, 6)

    # Rotate about the vessel axis (body-z): M = blockdiag(Rz(φ), Rz(φ)).
    c, s = jnp.cos(phi), jnp.sin(phi)
    Rz = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=R_can.dtype)
    M = jnp.zeros((6, 6), dtype=R_can.dtype).at[:3, :3].set(Rz).at[3:, 3:].set(Rz)
    return M @ R_can @ M.T


@stability(StabilityLevel.EXPERIMENTAL)
class RigidBodyNode(MimeNode):
    """6-DOF rigid body in overdamped Stokes flow.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    semi_major_axis_m : float
        Semi-major axis a [m] (along body e1).
    semi_minor_axis_m : float
        Semi-minor axis b [m]. For sphere: b = a.
    density_kg_m3 : float
        Body density [kg/m^3].
    fluid_viscosity_pa_s : float
        Surrounding fluid dynamic viscosity [Pa.s].
    fluid_density_kg_m3 : float
        Surrounding fluid density [kg/m^3].
    use_analytical_drag : bool
        If True (default), compute drag from Oberbeck-Stechert.
        If False, expect drag_force and drag_torque as boundary inputs
        from CSFFlowNode/IB-LBM.

    Boundary Inputs (additive forces/torques)
    -----------------------------------------
    magnetic_force : (3,)
        From MagneticResponseNode. Additive.
    magnetic_torque : (3,)
        From MagneticResponseNode. Additive.
    drag_force : (3,)
        From CSFFlowNode (IB-LBM mode). Additive.
    drag_torque : (3,)
        From CSFFlowNode (IB-LBM mode). Additive.
    external_force : (3,)
        Any additional external force (gravity, contact). Additive.
    external_torque : (3,)
        Any additional external torque. Additive.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-003",
        algorithm_version="1.1.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "6-DOF rigid body dynamics with four integration modes: "
            "overdamped analytical drag, overdamped external drag, "
            "inertial Euler, and kinematic (pre-computed V/Ω)."
        ),
        governing_equations=(
            r"Overdamped: V = R_T^{-1} F_ext; omega = R_R^{-1} T_ext. "
            r"Inertial: m dV/dt = F_total; I dΩ/dt = T_total. "
            r"Kinematic: V, Ω prescribed by upstream node. "
            r"All modes: x += V*dt; q = dq(omega, dt) * q."
        ),
        discretization="Explicit Euler for position; exact quaternion rotation for orientation",
        assumptions=(
            "Stokes regime: Re << 1, inertia negligible (overdamped modes)",
            "Rigid body — no deformation",
            "Prolate ellipsoid shape for analytical drag coefficients",
            "Fluid at rest (quiescent) when using analytical drag — background "
            "flow drag comes from CSFFlowNode as boundary input",
            "Kinematic mode assumes upstream node (e.g. MLPResistanceNode) "
            "has already performed the overdamped 6×6 R⁻¹ solve.",
        ),
        limitations=(
            "Analytical drag only valid for Re < 0.1",
            "No near-wall corrections in this node (SurfaceContactNode needed)",
            "Quaternion integration uses first-order approximation",
            "Kinematic mode does not see F_ext/T_ext — any force-balance "
            "must be handled upstream before velocity is injected.",
        ),
        validated_regimes=(
            ValidatedRegime("Re", 0.0, 0.1, "",
                            "Stokes regime — inertia negligible"),
            ValidatedRegime("semi_major_axis_m", 1e-6, 1e-3, "m",
                            "Microrobot size range"),
        ),
        references=(
            Reference("Lighthill1976", "Flagellar Hydrodynamics"),
            Reference("Rodenborn2013", "Propulsion of microorganisms by helical flagellum"),
        ),
        hazard_hints=(
            "Re > 1 invalidates Stokes drag — nonlinear convective term dominates",
            "Oberbeck-Stechert coefficients have singularity at e=0; "
            "jnp.where guard switches to sphere limit",
        ),
        implementation_map={
            "V = F_total / (6*pi*eta*a*C)": (
                "mime.nodes.robot.rigid_body.RigidBodyNode.update"
            ),
            "Oberbeck-Stechert C_1,C_2,C_3": (
                "mime.nodes.robot.rigid_body.oberbeck_stechert_coefficients"
            ),
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ROBOT_BODY,
        anatomical_regimes=(
            AnatomicalRegimeMeta(
                compartment=AnatomicalCompartment.CSF,
                anatomy="general CSF space",
                flow_regime=FlowRegime.STAGNANT,
                re_min=0.0, re_max=0.1,
                viscosity_min_pa_s=7e-4, viscosity_max_pa_s=1e-3,
            ),
        ),
        biocompatibility=BiocompatibilityMeta(
            materials=("SU-8", "NdFeB"),
            iso_10993_class=BiocompatibilityClass.NOT_ASSESSED,
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        semi_major_axis_m: float = 100e-6,
        semi_minor_axis_m: float = 100e-6,
        density_kg_m3: float = 1100.0,
        fluid_viscosity_pa_s: float = 8.5e-4,
        fluid_density_kg_m3: float = 1002.0,
        use_analytical_drag: bool = True,
        use_inertial: bool = False,
        kinematic_mode: bool = False,
        I_eff: float | None = None,
        I_transverse: float | None = None,
        m_eff: float | None = None,
        omega_max: float | None = None,
        constraint=None,
        mobility_matrix=None,
        resistance_matrix=None,
        locked_spin_axis_body=None,
        resistance_grid=None,
        vessel_axis=0,
        d_clamp=None,
        **kwargs,
    ):
        # Migration guard: old vessel_radius_m parameter removed
        for removed_param in ("vessel_radius_m", "vessel_half_length_m", "wall_stiffness"):
            if removed_param in kwargs:
                raise TypeError(
                    f"'{removed_param}' is removed from RigidBodyNode. "
                    f"Use constraint=CylindricalVesselConstraint(radius=..., "
                    f"half_length=...) instead. See mime.nodes.robot.constraints."
                )

        self._constraint = constraint

        # Overdamped 6×6 mobility mode (the confined-BEM microswimmer model):
        # given the full BEM resistance R (off-diagonal R_FΩ = the helical
        # chirality coupling that turns rotation into thrust), solve the
        # instantaneous overdamped force balance [V;ω] = R⁻¹·L_total each step.
        # No inertia integration → first-order → the screw locks to the magnetic
        # drive (the inertial mode librates: tiny rotational inertia makes the
        # magnetic-rotational mode stiff/under-damped). M = R⁻¹ is body-frame and
        # constant for a centred body. See StokesletFluidNode.resistance_matrix_si.
        self._mobility = None
        if mobility_matrix is not None:
            self._mobility = jnp.asarray(mobility_matrix)

        # Locked-rotation overdamped mode (the de Jongh quasi-static lock): below
        # step-out the magnetic alignment relaxes ~1000× faster than the swim, so
        # the screw is rigidly locked to the rotating field. Rather than integrate
        # that stiff alignment explicitly (it needs sub-ms dt), PRESCRIBE the spin
        # about the screw axis (drive rate, ``spin_rate`` input) and solve the
        # force-free translation from the 6×6 resistance:
        #     R_FU·V = (F_ext + F_bg) − R_FΩ·ω   ⇒  V = R_FU⁻¹·(… − R_FΩ·ω).
        # This is exactly the validated de Jongh swimming_velocity relation (+the
        # external/background load), robust at any dt. See schwarz_vessel_helix.
        self._resistance = None
        self._lock_axis = None
        if resistance_matrix is not None:
            self._resistance = jnp.asarray(resistance_matrix)
            axis = (0.0, 0.0, 1.0) if locked_spin_axis_body is None \
                else locked_spin_axis_body
            self._lock_axis = jnp.asarray(axis, dtype=jnp.float32)

        # OFF-CENTER resistance grid (near-wall, ar4-MLP-style): R(d) over radial
        # offsets (SI metres knots) from StokesletFluidNode.resistance_grid_si. When
        # present, the locked/overdamped branches use R at the body's CURRENT radial
        # offset instead of the centred constant — the dense screw rides the tube
        # floor where the centred R underestimates drag. The offset is read from the
        # body STATE pose, so it is FROZEN across a Schwarz step (state is re-seeded
        # from step-start each inner iteration), not the live iterate — see
        # offcenter_resistance_si and the freeze-check test.
        self._res_grid = None
        self._vessel_axis = int(vessel_axis)
        self._d_clamp = d_clamp
        if resistance_grid is not None:
            dk, rg = resistance_grid
            self._res_grid = (jnp.asarray(dk), jnp.asarray(rg))
            if self._d_clamp is None:
                self._d_clamp = float(jnp.asarray(dk)[-1])

        if use_inertial and I_eff is None:
            raise ValueError(
                "use_inertial=True requires I_eff (effective rotational inertia "
                "in kg*m^2). For d2.8 UMR: I_eff ≈ 1e-10."
            )
        if use_inertial and m_eff is None:
            m_eff = density_kg_m3 * (4.0 / 3.0) * 3.14159 * semi_major_axis_m * semi_minor_axis_m**2

        super().__init__(
            name, timestep,
            semi_major_axis_m=semi_major_axis_m,
            semi_minor_axis_m=semi_minor_axis_m,
            density_kg_m3=density_kg_m3,
            fluid_viscosity_pa_s=fluid_viscosity_pa_s,
            fluid_density_kg_m3=fluid_density_kg_m3,
            use_analytical_drag=use_analytical_drag,
            use_inertial=use_inertial,
            kinematic_mode=kinematic_mode,
            I_eff=I_eff,
            I_transverse=I_transverse,
            m_eff=m_eff,
            omega_max=omega_max,
            **kwargs,
        )

    def initial_state(self) -> dict:
        return {
            "position": jnp.zeros(3),
            "orientation": identity_quat(),
            "velocity": jnp.zeros(3),
            "angular_velocity": jnp.zeros(3),
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        spec = {
            "magnetic_force": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Magnetic force [N]",
            ),
            "magnetic_torque": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Magnetic torque [N.m]",
            ),
            "drag_force": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Fluid drag force [N] (from IB-LBM)",
            ),
            "drag_torque": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Fluid drag torque [N.m] (from IB-LBM)",
            ),
            "external_force": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Additional external force [N]",
            ),
            "external_torque": BoundaryInputSpec(
                shape=(3,), coupling_type="additive",
                description="Additional external torque [N.m]",
            ),
        }
        if self._mobility is not None or self._resistance is not None:
            # Overdamped modes: the force/torque ON the body from the ambient flow
            # alone (from the confined BEM), kept separate from the motion-coupled
            # drag so the force balance is a one-shot solve.
            spec["background_force"] = BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3), coupling_type="additive",
                description="Force from background flow [N]",
            )
            spec["background_torque"] = BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3), coupling_type="additive",
                description="Torque from background flow [N.m]",
            )
        if self._resistance is not None and not self.params["use_inertial"]:
            # Only the locked (overdamped) path prescribes the spin; the inertial
            # body lets rotation emerge from the magnetic torque (no spin_rate).
            spec["spin_rate"] = BoundaryInputSpec(
                shape=(), default=jnp.float32(0.0),
                description="Prescribed lock spin about the screw axis [rad/s]",
            )
        if self.params.get("kinematic_mode", False):
            spec["external_velocity"] = BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Pre-computed velocity (from MLPResistanceNode) [m/s]",
            )
            spec["external_angular_velocity"] = BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Pre-computed angular velocity [rad/s]",
            )
        return spec

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        a = self.params["semi_major_axis_m"]
        b = self.params["semi_minor_axis_m"]
        eta = self.params["fluid_viscosity_pa_s"]
        use_analytical = self.params["use_analytical_drag"]
        use_inertial = self.params["use_inertial"]
        kinematic = self.params.get("kinematic_mode", False)

        pos = state["position"]
        q = state["orientation"]

        # Sum all external forces and torques
        F_ext = (
            boundary_inputs.get("magnetic_force", jnp.zeros(3))
            + boundary_inputs.get("external_force", jnp.zeros(3))
        )
        T_ext = (
            boundary_inputs.get("magnetic_torque", jnp.zeros(3))
            + boundary_inputs.get("external_torque", jnp.zeros(3))
        )

        if kinematic:
            # Velocity comes from upstream (e.g. MLPResistanceNode which does the
            # full 6×6 R inversion internally). We just integrate position + orient.
            V = boundary_inputs.get("external_velocity", jnp.zeros(3))
            omega = boundary_inputs.get("external_angular_velocity", jnp.zeros(3))
        elif self._mobility is not None:
            # Overdamped 6×6 mobility (the confined-BEM microswimmer). Solve the
            # instantaneous force balance [V;ω] = R⁻¹·L_total in the BODY frame,
            # where the BEM resistance R (and so M = R⁻¹) is constant for a
            # centred body. L_total = external (magnetic + gravity) + the
            # background-flow load (force ON the body from the ambient flow).
            #
            # The motion-resistance −R·[V;ω] is NOT added: it is exactly the LHS
            # of the balance, supplied by M. Using the BEM's motion-coupled drag
            # here instead would make [V;ω] depend on its own previous value and
            # the coupling iteration would oscillate (eigenvalue −1) and diverge.
            from mime.core.quaternion import rotate_vector, rotate_vector_inverse

            F_bg = boundary_inputs.get("background_force", jnp.zeros(3))
            T_bg = boundary_inputs.get("background_torque", jnp.zeros(3))
            # world → body (M lives in the body frame; no-op at identity q).
            load_b = jnp.concatenate([
                rotate_vector_inverse(q, F_ext + F_bg),
                rotate_vector_inverse(q, T_ext + T_bg),
            ])
            # Off-center: M = R(offset)⁻¹ at the (frozen-per-step) radial offset.
            if self._res_grid is not None:
                R_oc = offcenter_resistance_si(
                    pos, q, self._res_grid[0], self._res_grid[1],
                    self._vessel_axis, self._d_clamp)
                vw_b = jnp.linalg.solve(R_oc, load_b)
            else:
                vw_b = self._mobility @ load_b
            V = rotate_vector(q, vw_b[:3])
            omega = rotate_vector(q, vw_b[3:])
        elif self._resistance is not None and not use_inertial:
            # Locked-rotation overdamped (de Jongh quasi-static lock): prescribe
            # ω about the screw axis at the drive rate; solve force-free V.
            from mime.core.quaternion import rotate_vector, rotate_vector_inverse

            F_bg = boundary_inputs.get("background_force", jnp.zeros(3))
            T_bg = boundary_inputs.get("background_torque", jnp.zeros(3))
            spin = boundary_inputs.get("spin_rate", jnp.float32(0.0))
            omega_b = spin * self._lock_axis.astype(F_ext.dtype)

            # Off-center: R at the (frozen-per-step) radial offset, else centred.
            if self._res_grid is not None:
                R = offcenter_resistance_si(
                    pos, q, self._res_grid[0], self._res_grid[1],
                    self._vessel_axis, self._d_clamp)
            else:
                R = self._resistance
            R_FU = R[:3, :3]
            R_FW = R[:3, 3:]
            F_b = rotate_vector_inverse(q, F_ext + F_bg)
            # R_FU·V = F_applied − R_FΩ·ω  (axial-rotation thrust via R_FΩ)
            rhs = F_b - R_FW @ omega_b
            V_b = jnp.linalg.solve(R_FU, rhs)
            V = rotate_vector(q, V_b)
            omega = rotate_vector(q, omega_b)
        elif use_inertial:
            # Inertial mode: I_eff * dΩ/dt = T_ext + T_drag (and m_eff for V).
            # The body's rotational dynamics are strongly UNDER-damped (ζ≈0.014 for
            # the confined FL-9 screw), so the lock is NOT quasi-static — the screw
            # can wobble and lose synchrony (step-out). This is the emergent body
            # for D1 / EP-1.
            I_eff = self.params["I_eff"]
            m_eff = self.params["m_eff"]
            F_drag = boundary_inputs.get("drag_force", jnp.zeros(3))
            T_drag = boundary_inputs.get("drag_torque", jnp.zeros(3))

            omega_old = state["angular_velocity"]
            V_old = state["velocity"]

            # Confined-BEM motion drag: when the 6×6 resistance (centred constant or
            # off-center grid) is supplied, the genuine physical damping wrench is
            # −R·[V;ω] evaluated in the BODY frame at the FROZEN step-start velocity
            # (state is re-seeded each Schwarz inner iteration → naturally frozen-
            # per-step). Unlike the overdamped R⁻¹ solve, here −R·[V;ω] is an
            # explicit additive load, not the LHS of a balance, so it does NOT
            # double-count / oscillate (eigenvalue −1) — it is stable explicit Euler
            # at the emergent dt (≈1e-4, validated in mod1_stepout_spike / M1). The
            # background-flow load (force ON a held body from the ambient flow) is
            # added separately, same split as the locked/overdamped paths.
            if self._resistance is not None or self._res_grid is not None:
                from mime.core.quaternion import rotate_vector, rotate_vector_inverse

                if self._res_grid is not None:
                    R_mot = offcenter_resistance_si(
                        pos, q, self._res_grid[0], self._res_grid[1],
                        self._vessel_axis, self._d_clamp)
                else:
                    R_mot = self._resistance
                vw_b = jnp.concatenate([
                    rotate_vector_inverse(q, V_old),
                    rotate_vector_inverse(q, omega_old),
                ])
                drag_b = -R_mot @ vw_b
                F_bg = boundary_inputs.get("background_force", jnp.zeros(3))
                T_bg = boundary_inputs.get("background_torque", jnp.zeros(3))
                F_drag = F_drag + rotate_vector(q, drag_b[:3]) + F_bg
                T_drag = T_drag + rotate_vector(q, drag_b[3:]) + T_bg

            # Rotational dynamics — Euler's equations for a symmetric top with an
            # ANISOTROPIC inertia tensor: body-z is the long/spin axis (I_axial =
            # I_eff), body-x,y are transverse/tumble (I_transverse). For an elongated
            # screw I_transverse ≈ m(3a²+L²)/12 ≫ I_axial, and the gyroscopic term
            # (I_axial−I_transverse)·ω×ω is exactly what makes the spinning body
            # PRECESS/wobble — essential for the step-out/wobble physics (D1). Using
            # a scalar I_eff (the old code) for the transverse mode over-damps the
            # wobble. ``I_transverse`` defaults to I_eff (isotropic) → byte-identical
            # to the old scalar integration when not supplied (ar4 unchanged).
            from mime.core.quaternion import rotate_vector, rotate_vector_inverse
            I_a = I_eff
            I_t = self.params.get("I_transverse", None)
            I_t = I_a if I_t is None else I_t
            T_total = T_ext + T_drag
            Tb = rotate_vector_inverse(q, T_total)
            wb = rotate_vector_inverse(q, omega_old)
            dwx = (Tb[0] - (I_a - I_t) * wb[1] * wb[2]) / I_t
            dwy = (Tb[1] - (I_t - I_a) * wb[2] * wb[0]) / I_t
            dwz = Tb[2] / I_a
            wb_new = wb + dt * jnp.stack([dwx, dwy, dwz])
            omega = rotate_vector(q, wb_new)

            # Ma safety clamp: limit angular velocity magnitude
            omega_max = self.params.get("omega_max", None)
            if omega_max is not None:
                omega_mag = jnp.linalg.norm(omega)
                scale = jnp.where(
                    omega_mag > omega_max,
                    omega_max / jnp.maximum(omega_mag, 1e-30),
                    1.0,
                )
                omega = omega * scale

            # Euler integration of translational dynamics
            F_total = F_ext + F_drag
            V = V_old + F_total / m_eff * dt

        elif use_analytical:
            e = jnp.sqrt(jnp.maximum(1.0 - (b/a)**2, 0.0))
            C_1, C_2, C_3 = oberbeck_stechert_coefficients(e)
            R_T_diag = 6.0 * a * jnp.pi * eta * jnp.array([C_1, C_2, C_2])
            R_R_diag = 8.0 * a * b**2 * jnp.pi * eta * jnp.array([C_3, C_3, C_3])
            V = F_ext / jnp.maximum(R_T_diag, 1e-30)
            omega = T_ext / jnp.maximum(R_R_diag, 1e-30)
        else:
            F_drag = boundary_inputs.get("drag_force", jnp.zeros(3))
            T_drag = boundary_inputs.get("drag_torque", jnp.zeros(3))
            e = jnp.sqrt(jnp.maximum(1.0 - (b/a)**2, 0.0))
            C_1, C_2, C_3 = oberbeck_stechert_coefficients(e)
            R_T_diag = 6.0 * a * jnp.pi * eta * jnp.array([C_1, C_2, C_2])
            R_R_diag = 8.0 * a * b**2 * jnp.pi * eta * jnp.array([C_3, C_3, C_3])
            V = (F_ext + F_drag) / jnp.maximum(R_T_diag, 1e-30)
            omega = (T_ext + T_drag) / jnp.maximum(R_R_diag, 1e-30)

        # Integrate position (Euler)
        new_pos = pos + V * dt

        # Apply contact constraint (if provided)
        if self._constraint is not None:
            new_pos, V = self._constraint.apply(new_pos, V)

        # Integrate orientation (quaternion)
        dq = quat_from_angular_velocity(omega, dt)
        new_q = quat_normalize(quat_multiply(dq, q))

        return {
            "position": new_pos,
            "orientation": new_q,
            "velocity": V,
            "angular_velocity": omega,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "position": state["position"],
            "orientation": state["orientation"],
            "velocity": state["velocity"],
            "angular_velocity": state["angular_velocity"],
        }
