"""schwarz_vessel_helix — physics parameters (live-editable namespace, SI).

The runner exec()s this in a bare namespace and passes the UPPERCASE names as the
``params`` dict. Anything omitted falls back to the composer defaults in
``mime.experiments.schwarz_vessel_helix._DEFAULTS`` (and the controller merges those
defaults too, so the arm geometry / control gains resolve). Units are physical SI.

NOTE: no ``__file__`` / path resolution here (the runner's exec gives no ``__file__``
and keeps only scalars) — asset paths live in the composer defaults (absolute).
"""

import numpy as np

# ── run mode ─────────────────────────────────────────────────────────────────
SWIM_MODE = "free"            # free (swims) | held (pinned; reaction-drag readout)
FLOW_PROFILE = "poiseuille"   # poiseuille (steady, rate U_MEAN) | womersley (pulsatile)
INCLUDE_ARM = True            # AR4 arm holds the magnet (gates the control work-packages)

# ── vessel flow (SI) ─────────────────────────────────────────────────────────
U_MEAN = 3e-3                 # vessel mean velocity [m/s] (Re≈19; ≲25 local-Stokes-valid)

# ── magnetic drive ───────────────────────────────────────────────────────────
# NOTE: 3 Hz is the de Jongh / MLP-training regime. Real ops against physiological
# flow are much higher (≳128 Hz) — but that pushes the robot-scale Reynolds number
# out of the confined Stokes-BEM's validity (see the plan's forward-looking section).
# Treat frequency as a swept axis, not a silent default.
DRIVE_HZ = 3.0

# ── screw resolution / step ──────────────────────────────────────────────────
N_THETA = 24
N_ZETA = 36
DT = 5e-4

# Control gains use the composer/controller defaults (ar4's stable inverse-dynamics
# values: CONTROL_K_P=100, CONTROL_K_D=20, ...). Override here to retune live.
