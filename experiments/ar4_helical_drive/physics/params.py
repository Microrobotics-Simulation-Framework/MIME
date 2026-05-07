# Parameters for the AR4 + helical-UMR drive experiment.
# Pure assignments + arithmetic — no imports, no classes. Executed via
# exec() into a namespace by mime.runner.

# ── UMR + vessel + fluid (matches dejongh_confined defaults) ────────
DESIGN_NAME       = "FL-9"
VESSEL_NAME       = "1/4\""
MU_PA_S           = 1e-3
DELTA_RHO_KG_M3   = 410.0
DT_PHYS           = 5e-4   # matches legacy dejongh

# ── Lubrication correction (Goldman-Cox-Brenner) ─────────────────────
USE_LUBRICATION   = True
LUB_EPSILON_MM    = 0.15

# ── Initial UMR pose (pre-sunk to gravity floor) ─────────────────────
# Vessel runs along world-x; gravity is along -z.  The gravity floor
# of the vessel cross-section is at z = -R_eff (≈ -1 mm). Putting the
# body anywhere else means it starts off-equilibrium and orbits the
# cross-section indefinitely under the rotating gradient force —
# the bug behind the apparent instability when comparing to legacy
# dejongh, where the same INIT_Y_M=-1mm was correctly gravity-pinned
# because gravity there was along -y rather than -z.
INIT_X_M          = 0.0
INIT_Y_M          = 0.0
INIT_Z_M          = -1.0e-3

# ── Arm: AR4 (Annin Robotics open-source 6-DOF) ──────────────────────
URDF_PATH                   = "assets/ar4.urdf"
END_EFFECTOR_LINK_NAME      = "link_6"
# AR4 base is positioned so that — with the proper home-pose seeded
# BASE_POSE_WORLD: AR4 base mesh fits ArmDesk's footprint at
# y=0.328 ±0.343 m. The EE is brought to its target above the
# horizontal tube at world origin by joint angles (ARM_HOME_RAD)
# rather than by moving the base. Standoff and magnet strength
# calibrated by the sweep in scripts/ar4_calibration_sweep.py
# against the de Jongh (2025) RPM setup.
BASE_POSE_WORLD             = (-0.05, 0.328, -0.43,  1.0, 0.0, 0.0, 0.0)
# Rotor sits at the EE link centre. With the post-IK home pose, the
# motor spin axis (EE-z) is aligned with world-x, so the rotor
# cylinder lies along world-x. There's no asymmetry to compensate
# for; any non-zero offset would just visually displace the rotor
# from the wrist. Keep at identity.
END_EFFECTOR_OFFSET_IN_LINK = (0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0)
# Per-joint viscous friction. None → use the URDF's <dynamics damping>.
JOINT_FRICTION_N_M_S        = None
GRAVITY_WORLD               = (0.0, 0.0, -9.80665)

# Initial home pose (zero-vector home is unstable for AR4 — link 2
# would point straight up; we drop the elbow into a relaxed pose so
# the magnet sits above the vessel at startup).
# Home pose found by position-IK so the EE (with EE_OFFSET) lands
# at world (0, 0, 0.07) — 7 cm above the horizontal tube at world
# origin, well clear of any wrist-mesh collision. Computed offline
# with optax.adam against the position objective + small joint-
# regularisation for a "neutral-shaped" solution.
# 15 cm standoff — matches the de Jongh paper's RPM placement. At
# this distance the elliptical-field gradient at the UMR is small
# enough that the helix's tilt instability grows slowly enough for
# the AR4 to track via the closed-loop controller. Earlier attempt
# at 20 cm (max reach) had the gradient too strong for the
# controller's mechanical bandwidth to suppress.
ARM_HOME_RAD                = (-0.02761, 0.23896, -0.77475, 1.55611, 1.54628, 0.00000)

# Add the RNEA gravity vector to commanded_joint_torques each step
# so a zero-torque controller (or any controller that doesn't include
# gravity-compensation in its command) holds the home pose statically.
# Without this the AR4 sags under gravity, oscillates, and clips
# against its joint limits — visible in the viewport as "arm collapses
# toward base, waves around, snaps back".
AUTO_GRAVITY_COMPENSATION   = True

# ── Motor (rotor that spins the magnet) ──────────────────────────────
MOTOR_INERTIA_KG_M2 = 1e-5
MOTOR_KT_N_M_PER_A  = 0.05
MOTOR_R_OHM         = 1.0
MOTOR_L_HENRY       = 1e-3
MOTOR_DAMPING_N_M_S = 1e-4
# Spin axis is the AR4 J6 axis (+x in the flange frame, which is the
# EE offset's local frame).
# Motor spin axis = EE-y (orthogonal to EE-x = the wrist's pointing
# direction). Visually this makes the rotor cylinder lie ACROSS the
# wrist rather than extend out from it.
MOTOR_AXIS_IN_PARENT = (0.0, 0.0, 1.0)

# ── Permanent magnet ─────────────────────────────────────────────────
# External rotor magnet — de Jongh paper RPM (NdBFe N45 ⌀35×20 mm,
# m = 18.89 A·m²).  At our 20 cm standoff (max AR4 reach), the
# field at the UMR is ~360 µT — just above the step-out threshold
# for FL-9 at 10 Hz, so the helix synchronises and corkscrews
# along its long axis.  Smaller magnets put the field below
# step-out and the helix oscillates / precesses instead of
# rotating cleanly (manifests visually as "rotating around the
# wrong axis").  The vessel was extended to ±0.5 m to let the
# corkscrew swim run for hundreds of seconds without hitting the
# end-cap.
MAGNET_DIPOLE_A_M2  = 18.89              # de Jongh paper magnet

# Permanent-magnet moment direction in the rotor body frame.
# MUST be perpendicular to MOTOR_AXIS_IN_PARENT. With motor axis
# (0,1,0) we put the moment along (1,0,0) (rotor body x) so the
# rotating dipole sweeps the x-z plane.
MAGNET_AXIS_IN_BODY = (1.0, 0.0, 0.0)
MAGNET_RADIUS_M     = 17.5e-3  # 35 mm diameter (de Jongh RPM)
MAGNET_LENGTH_M     = 20e-3    # 20 mm height
FIELD_MODEL         = "point_dipole"     # {point_dipole | current_loop | coulombian_poles}
EARTH_FIELD_WORLD_T = (0.0, 0.0, 0.0)

# ── Live-editable actuation knob (ParameterPanel) ────────────────────
# Drive frequency — matches the de Jongh paper's 10 Hz at the
# 15 cm standoff. With B ≈ 757 µT at the helix (Mahoney/Abbott
# Eq 1, RMS over the elliptical cycle) the FL-9 helix's step-out
# is around 21 Hz, so 10 Hz is well below step-out and the helix
# locks to the field. Going lower hurts swim speed; going higher
# crosses step-out at this geometry and causes asynchronous
# tumble.  Compare to the 20 cm setup where step-out was 9.5 Hz
# and we had to drop to 3 Hz.
FIELD_FREQUENCY_HZ  = 3.0

# ── Visualisation-fast vs. publication-fidelity ──────────────────────
# True  : Gauss-Seidel coupling group on body↔magnet (high fidelity,
#         ~10× slower per step).
# False : staggered back-edges (one-step phase lag of ~10° at 60 Hz —
#         invisible at cm-scale UMR motion). Default for this
#         experiment is False since the headline use case is
#         interactive viz iteration.
# Coupling group across body ↔ ext_magnet ↔ magnet ↔ mlp_drag ↔ lub:
# resolves the body↔drag implicit system within each timestep using
# Gauss-Seidel iteration. Required because the lubrication drag near
# the vessel wall is stiffer than dt·R/m allows for explicit Euler —
# without the coupling group the body's velocity flips sign with
# growing magnitude each step and teleports between vessel end-caps.
# 10× slower than staggered back-edges but the simulation actually
# stays inside the tube.
USE_COUPLING_GROUP  = True

# ── Closed-loop controller (control/controller.py) ───────────────────
# Joint-space PD gains used by the AR4 tracking controller. These run
# *on top of* RobotArmNode's auto_gravity_compensation so the PD law
# only handles the tracking residual, not gravity load.
CONTROL_K_P         = 10000.0   # 1/s² (10 ms time const, ζ=1)
CONTROL_K_D         = 200.0     # 1/s
CONTROL_IK_LAMBDA   = 0.05    # damped-LS Newton regulariser
# Differential-IK step gain (dimensionless). 1.0 = one full Newton
# step per call: q_target = q + J⁺·e, which puts the joint target
# at ≈ the IK solution. The IDPD then tracks it at K_p rate. Larger
# values overshoot (q_target = q + (K_ik-1)·J⁺·e past the target);
# smaller values settle slower but more conservatively.
CONTROL_K_IK        = 1.0
# Inner Newton iters for the differential IK each physics step. 5
# is enough to converge from q (current state) to q_target through
# rotation gaps up to ~π/2; single-step (1 iter) lags during fast
# helix tumble.
CONTROL_IK_INNER_ITERS = 5

# Closed-loop position tracking (M3): EE follows the body's x
# position at this height above the tube.  15 cm matches de Jongh.
CONTROL_STANDOFF_M  = 0.15

# Low-pass coefficient on the target pose (per physics step).
# 0.005 = 100 ms time const at dt = 5e-4. Heavily filters
# orientation feedback: only AVERAGE helix tilt drift (over many
# rotation cycles) drives the rotor. High-frequency body wobble
# from the elliptical-field tilt mode is ignored — chasing it
# would whip the AR4 around (the wrist's IK couples rotation to
# substantial position changes), which itself drives field-source
# motion → helix tumble → positive feedback.  Slow tracking
# breaks the loop.
CONTROL_TARGET_ALPHA = 0.005

# Orientation feedback is OFF by default after diagnostics (see
# scene/_diagnose_compensation.py).  Turning it ON causes the AR4
# to thrash: the IK that aligns EE-z with body-z also translates
# the wrist by 10s of cm (a 6-DOF AR4's position and orientation
# subspaces are coupled), which moves the field source erratically
# and itself drives helix tumble.  With FB OFF the position-
# tracking controller keeps the rotor above the helix while the
# rotor's spin axis stays along world-x; the field is steady, and
# the helix swims cleanly at paper rate (~4 mm/s).
#
# Set ON to study the closed-loop instability mechanism described
# above, or once the AR4-position coupling is fixed (e.g. via
# task-priority IK that doesn't translate the wrist when only
# orientation is required).
ENABLE_ORIENTATION_FEEDBACK = False

# Velocity feedforward lookahead (s). The controller projects body
# orientation forward by this much using its angular velocity, so
# the IDPD's steady-state τ·ω lag is cancelled.  Should match the
# IDPD time constant 1/sqrt(K_p) — at K_p=10000, that's 10 ms.
CONTROL_LOOKAHEAD_S = 0.010

# Reach envelope (M5): the AR4 at the configured base position +
# 20 cm standoff above the tube can only reach roughly x ∈ [-0.05,
# +0.05] m without driving its joints into limits or singularities
# (verified by the IK reach probe in scripts/check_ar4_reach.py).
# Beyond this, the controller clamps the target to the envelope —
# the rotor stays at the edge of its reach while the helix swims
# past, at which point the field at the helix loses the
# perpendicular geometry and the misalignment effect returns
# (which is fine for visualisation; real labs widen the workspace
# by mounting the manipulator on a linear stage).
CONTROL_X_MIN_M = -0.05
CONTROL_X_MAX_M = +0.05

# Jacobian-conditioning safety: if σ_min(J) drops below this the
# IK is approaching singularity and we freeze the target instead
# of trying to push through. Logged when triggered.
CONTROL_SIGMA_MIN = 1e-3
