# `scripts/sim2real/`: the model at script A's operating point (dev1 phase 7j-a)

What MIME predicts for script A's operating point, rung by rung, with a band on each number, before any comparison with
a measurement: FL-9 (de Jongh et al. 2025) on the floor of the 1/4" water-filled tube, turned by a uniform field at
10 Hz and 1.2 mT. No measured speed enters any script here. Run by hand, never by CI (`testpaths` is `tests/`).

| File | What it does |
|---|---|
| `operating_point.py` | script A's reproduction arm with every declared value and setting as an argument (the drag's rung, the surrogate's weights, the lubrication, the contact margin, the time step, the moment, the field, the densities, the viscosity, the inertia); the floored pose from the graph's own leaves; the swim speeds read from a resistance (`axial`, `lateral_free`, `wall_held`); a run's steady speed by phase 0's rule |
| `surrogate.py` | the surrogate's rungs on the CPU: its resistance alone and with its lubrication at the floored pose, the full dynamics with the lubrication on and off, the time step's convergence, the margin's sweep, the dynamics' finite differences across the budget's ranges (the rung is `piecewise_jump`), the coupling groups, and that the experiment is script A's |
| `bem.py` | the BEM rung (`Drag.MRS` in the `tube` wall kind) at the floored pose over eight phases of the half turn: closed and lateral, two spacings, MIME-ANO-114's offsets with the surrogate beside, the blob, the margin, the table's read error, and the budget by `maddening.sysid.fim(..., scale="nominal")` with its finite-difference check |
| `modal_app.py` | `bem.py`'s parts and the BEM rung's full dynamics on an A100, float64 at the highest matmul precision (`modal` is no dependency of MIME) |
| `assemble.py` | the breakdown from the parts' JSON: the ladder, the model-internal gaps, the budget's bands, every control's verdict, the bit-for-bit check |
| `prereg.py` | the preregistration: the commit, the scripts, the table, the weights and the meshes, hashed before any run |
| `budget_inputs.json` | every input's plausible range with its source: the `ParamSpec` bounds of the budget |
| `predictions_7ja.json`, `preregistration_7ja.json` | committed before any of the phase's computations, never edited after a result |
| `capfix.py` | MIME-ANO-119's fix checked at A: the screw kind's ring caps against its earlier polar grid (rebuilt here, its hash checked against the preregistration's): the speed, the matrix's spectrum and Cholesky factor at each phase, the free-space resistance, the derivative's finite-difference check with its planted control, and the correction's fine scale scan |
| `capfix_modal.py`, `capfix_launch.py` | `capfix.py`'s stages in one detached call on an A100-40GB, each stage's result written to a Volume as it finishes; the launcher starts it at most three times and never beside a running or started run |
| `predictions_capfix.json` | committed before any of the fix's computations at A, never edited after a result |

The library's rung `bem` is the rung `mrs` since the rename of 0.3.0's drag (the method of regularised Stokeslets). These
scripts keep their own names for their arms and outputs (`bem.py`, `bem_*.json`, the arm `"bem"`), which the frozen
predictions and the preregistration key on; only what calls the library (`Drag.MRS`, `mime.nodes.hydro.mrs`,
`MRSDragNode`) was renamed, so these scripts' bytes are no longer the ones `preregistration_7ja.json` hashed.

A file `preregistration_7ja.json` hashed changes only by a listed change: `preregistration_drift.json` lists each one with its new hash, its commits and its reason (the register's amendment A2), and `tests/differential/test_preregistration_drift.py` fails on a change to any of them that is not listed, or listed and changed again.

Every experiment here builds its geometry kinds at the versions 7j's register froze (`operating_point.FROZEN_VERSIONS`:
the screw 0.2.0, whose samples are its quadrature, not the 0.3.0 that quasi-uniform sampling brought), through
`operating_point.pinned`, until a dated amendment of the register moves them: built on the screw 0.3.0, the MRS arm's
speed at point A moved by +15.6%. `tests/differential/test_frozen_register_pins.py` fails on a script here that builds
a kind unpinned. `capfix.py` and `prereg.py` rebuild the samples from the node's counts, which are its samples only on
a kind that samples by its quadrature, and refuse any other (`operating_point.require_quadrature`).

The surrogate's rows use `dejongh_mlp_v2`, trained through the 2026-05-21 tables (MIME-ANO-117) until phase 7i ships
retrained weights; `--weights <asset>` re-runs every surrogate row with another (`surrogate.py`, `bem.py --part
offsets`, `modal_app.py`). Every script refuses to run without `JAX_ENABLE_X64=1`.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/sim2real/surrogate.py --part dynamics --out DIR
    MIME_7JA_WHEEL=... MIME_7JA_REQUIREMENTS=... MIME_7JA_TABLE=... modal run scripts/sim2real/modal_app.py --stage static --out DIR
    python scripts/sim2real/assemble.py --runs DIR --repeat DIR2 --mesh MESH.json --out SUMMARY.json
