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

# ── Initial UMR pose (pre-sunk) ──────────────────────────────────────
INIT_X_M          = 0.0
INIT_Y_M          = -1.0e-3
INIT_Z_M          = 0.0

# ── Arm: AR4 (Annin Robotics open-source 6-DOF) ──────────────────────
URDF_PATH                   = "assets/ar4.urdf"
END_EFFECTOR_LINK_NAME      = "link_6"
# AR4 base sits at world origin; the URDF's `world` link is the static
# anchor and the `world_to_base` fixed joint introduces no offset.
BASE_POSE_WORLD             = (0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0)
# Mount the magnet 5 cm beyond the AR4 tool flange along +x of the
# flange frame.
END_EFFECTOR_OFFSET_IN_LINK = (0.05, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0)
# Per-joint viscous friction. None → use the URDF's <dynamics damping>.
JOINT_FRICTION_N_M_S        = None
GRAVITY_WORLD               = (0.0, 0.0, -9.80665)

# Initial home pose (zero-vector home is unstable for AR4 — link 2
# would point straight up; we drop the elbow into a relaxed pose so
# the magnet sits above the vessel at startup).
ARM_HOME_RAD                = (0.0, -0.6, 0.6, 0.0, -0.5, 0.0)

# ── Motor (rotor that spins the magnet) ──────────────────────────────
MOTOR_INERTIA_KG_M2 = 1e-5
MOTOR_KT_N_M_PER_A  = 0.05
MOTOR_R_OHM         = 1.0
MOTOR_L_HENRY       = 1e-3
MOTOR_DAMPING_N_M_S = 1e-4
# Spin axis is the AR4 J6 axis (+x in the flange frame, which is the
# EE offset's local frame).
MOTOR_AXIS_IN_PARENT = (1.0, 0.0, 0.0)

# ── Permanent magnet ─────────────────────────────────────────────────
MAGNET_DIPOLE_A_M2  = 1.68e-3            # 2 × 1mm³ N45 (matches dejongh)
MAGNET_AXIS_IN_BODY = (1.0, 0.0, 0.0)
MAGNET_RADIUS_M     = 1e-3
MAGNET_LENGTH_M     = 2e-3
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
