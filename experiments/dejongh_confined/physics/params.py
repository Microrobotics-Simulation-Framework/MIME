# Parameters for the de Jongh confined-swimming experiment.
# Pure assignments + arithmetic — no imports, no classes. Executed via
# exec() into a namespace by mime.runner.

# ── Geometry & fluid ─────────────────────────────────────────────────
DESIGN_NAME       = "FL-9"     # one of FL-3, FL-5, FL-7, FL-9
VESSEL_NAME       = "1/4\""    # one of 1/2", 3/8", 1/4", 3/16"
MU_PA_S           = 1e-3       # water at 20 °C
DELTA_RHO_KG_M3   = 410.0      # buoyancy-corrected body density contrast
DT_PHYS           = 5e-4       # 0.5 ms physics step

# ── Actuation (live-editable from MICROROBOTICA ParameterPanel) ──────
FIELD_FREQUENCY_HZ = 10.0
FIELD_STRENGTH_MT  = 1.2

# ── Lubrication correction (Goldman-Cox-Brenner asymptotics) ────────
USE_LUBRICATION   = True
LUB_EPSILON_MM    = 0.15

# ── Initial pose (pre-sunk to skip the ~3 ms gravity transient) ─────
# y is the gravity-down axis in this graph.
INIT_X_M          = 0.0
INIT_Y_M          = -1.0e-3
INIT_Z_M          = 0.0

# ── MLP surrogate weights ────────────────────────────────────────────
# Resolved relative to the MIME repo root if not absolute.
MLP_WEIGHTS_PATH  = "data/dejongh_benchmark/mlp_cholesky_weights_v2.npz"
