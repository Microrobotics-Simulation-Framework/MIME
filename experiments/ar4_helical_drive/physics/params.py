# Parameters for the AR4 + helical-UMR drive experiment.
# Pure assignments + arithmetic — no imports, no classes. Executed via
# exec() into a namespace by mime.runner.

# ── UMR + vessel + fluid (matches dejongh_confined defaults) ────────
DESIGN_NAME       = "FL-9"
VESSEL_NAME       = "1/4\""
MU_PA_S           = 1e-3
DELTA_RHO_KG_M3   = 410.0
DT_PHYS           = 5e-4

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
# 20 cm standoff (max AR4 reach above the tube). Going further is
# infeasible — IK fails beyond 20 cm given the AR4 base position
# fixed by the desk geometry.  At 20 cm the orbital amplitude is
# halved vs 15 cm (~0.37 mm vs 0.80 mm) while |B| at the helix is
# still ~360 µT, well above the step-out edge for FL-9.
ARM_HOME_RAD                = (-0.02763, 0.49873, -1.51875, 1.54673, 1.55147, 0.00000)

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
FIELD_FREQUENCY_HZ  = 10.0

# ── Visualisation-fast vs. publication-fidelity ──────────────────────
# True  : Gauss-Seidel coupling group on body↔magnet (high fidelity,
#         ~10× slower per step).
# False : staggered back-edges (one-step phase lag of ~10° at 60 Hz —
#         invisible at cm-scale UMR motion). Default for this
#         experiment is False since the headline use case is
#         interactive viz iteration.
USE_COUPLING_GROUP  = False
