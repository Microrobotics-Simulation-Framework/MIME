"""Physical parameters for d2.8 UMR confinement experiment.

All values from deboer2025_params.md and umr_ode.py.
This file is executed by mime.runner and the resulting namespace
is serialized to JSON for MICROROBOTICA's parameter panel.
"""

import math

# --- UMR geometry (mm, converted to lattice units by setup.py) ---
VESSEL_DIAMETER_MM = 9.4
UMR_GEOM_MM = dict(
    body_radius=0.87, body_length=4.1, cone_length=1.9,
    cone_end_radius=0.255, fin_outer_radius=1.42,
    fin_length=2.03, fin_width=0.55, fin_thickness=0.15,
    helix_pitch=8.0,
)

# --- Physical properties ---
N_MAG = 1
M_SINGLE = 1.07e-3           # A*m^2 per magnet
B_FIELD = 3e-3                # T (3 mT)
FLUID_VISCOSITY = 0.69e-3     # Pa.s (water at 37C)
FLUID_DENSITY = 997.0         # kg/m^3
BODY_DENSITY = 1100.0         # kg/m^3 (SU-8 + NdFeB)
I_EFF = 1e-10                 # kg*m^2 (effective rotational inertia)

# --- Geometry ---
SEMI_MAJOR = 2.05e-3          # m (half body length)
SEMI_MINOR = 0.87e-3          # m (body radius)
CONFINEMENT_RATIO = 0.30

# --- LBM parameters ---
RESOLUTION = 192              # lattice nodes per vessel diameter
TAU = 0.8                     # BGK relaxation time
USE_BOUZIDI = False           # simple BB for demo (Bouzidi for production)

# --- Derived ---
F_STEP_UNCONFINED = 128.0     # Hz (de Boer et al.)
C_ROT = N_MAG * M_SINGLE * B_FIELD / (2 * math.pi * F_STEP_UNCONFINED)

# --- Subcycling ---
SUBCYCLE_FACTOR = 10

# --- Controller mode ---
MODE = "stepout"          # "steady" (constant freq) or "stepout" (frequency ramp)
F_STEADY_FRAC = 0.8       # fraction of Mach-safe max for steady mode
RAMP_STEPS = 20000         # steps over which to ramp (stepout mode)
F_RAMP_START = 0.3         # start ramp at this fraction of Mach-safe max
F_RAMP_END = 1.0           # end ramp at this fraction of Mach-safe max
