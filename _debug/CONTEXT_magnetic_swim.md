# Context: fix the magnetic-driven corkscrew swim (schwarz_vessel_helix)

---
## ✅ RESOLVED (2026-06-13) — read this first

The screw now **corkscrews through the x-pipe** at ~mm/s with its spin **locked to
the drive** (`omega_x = 18.85 rad/s` at 3 Hz). Default `BODY_MODEL="locked"`.

What was done (commits on `debug/magnetic-swim-coupling`):
1. **Exposed the confined BEM's SI 6×6 resistance** `R` in Schwarz mode
   (`StokesletFluidNode.resistance_matrix_si()`), extracted from the factored
   interface system — same algebra as the de-risk, recovers ~3.2 mm/s force-free.
2. **Overdamped mobility body (Option A, `BODY_MODEL="overdamped"`)**: solves
   `[V;ω] = R⁻¹·(L_ext + L_bg)` one-shot. The **off-diagonal R_FΩ chirality
   coupling** is what turns rotation into thrust. The background-flow load `L_bg`
   is exposed SEPARATELY from the motion-coupled drag (new BEM
   `background_force/torque` fluxes, wired by `TwoScale` when the body declares
   them) — folding it into the drag makes the coupling iterate oscillate
   (eigenvalue −1) and diverge.
3. **Finding on Option A**: it is correct and **dt-converged** (5e-5 ≡ 2e-5), but
   the *emergent magnetic lock* is **stiff** (free-spin ≈ 2200 rad/s ⇒ DT=5e-4
   aliases the field). At resolving dt it does NOT lock to 18.85 — it settles to a
   **slip** (`omega_x ≈ −3.5 rad/s`). That is the genuine, dt-independent
   **magnetic sync / step-out** behaviour = the project's **"Problem 1"**, not a
   bug. Use `overdamped` for that study.
4. **`BODY_MODEL="locked"` (default, the gate deliverable)**: the de Jongh
   quasi-static lock — prescribe the spin to the drive rate about the screw axis,
   solve force-free translation `V = R_FU⁻¹·(F_applied − R_FΩ·ω)`. Robust at
   DT=5e-4 (no stiff integration), validated to the de Jongh ~3 mm/s. The screw
   locks + translates monotonically along +x. e2e test asserts this.

The earlier "under-damped whirl" diagnosis was right that the inertial mode is
wrong; the deeper finding is that even the *correct* overdamped emergent lock is
numerically stiff, and the robust+validated answer is the quasi-static lock.

Everything below is the original (pre-fix) handoff, kept for provenance.
---


## TL;DR of the task
The de Jongh screw, magnetically driven by the AR4-held rotating magnet, should
**corkscrew through the x-pipe**. Right now it **spins but doesn't translate**: the
screw whirls (spins faster than the 3 Hz field, librating) instead of synchronizing,
so no clean thrust. **The swim mechanism is proven correct; only the magnetic sync /
body model is broken.** Implement the **overdamped (mobility-based) swimmer** (Option A
below) so it locks and swims.

## Where everything is
- Branch: **`debug/magnetic-swim-coupling`** (off `release/v0.3.0`). 6 commits, all
  checkpoints. `release/v0.3.0` is untouched.
- Experiment: `experiments/schwarz_vessel_helix/` (experiment.yaml, physics/{setup,params}.py,
  control/controller.py, scene/world.usda, assets/{ar4_meshes.usdc, ar4.urdf, umr.usda}).
- Composer (the real builder): `src/mime/experiments/schwarz_vessel_helix.py`
  (`build_experiment` / `build_graph` / `default_external_inputs` / `screw_points` /
  `body_world_points` / `_seed_body_orientation` / `_seed_arm_home`).
- Debug scripts + findings: `_debug/` (d1..d12c, `SUMMARY.md`). Run with
  `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python _debug/<f>.py` from the MIME dir
  (`source ../.venv/bin/activate` first).
- Memory: `si-native-confined-bem.md`, `effects-first-composition.md` (auto-loaded).
- Plan doc (append progress here): `../plans/MIME_v0.3.0_PLAN.md`.

## What WORKS (validated — do not re-debug)
1. **SI-native confined BEM**: `StokesletFluidNode(..., length_scale=R_body)` normalises
   the body for the dimensionless wall table, keeps SI in/out (raw solve = L·t_phys → ÷L).
   Validated: force-free FL-9 swim = **3.27 mm/s** vs de Jongh ~3 mm/s
   (`tests/verification/test_si_confined_bem.py`).
2. **Frame-aware BEM** (`_update_schwarz` in `stokeslet/fluid_node.py`): rotates inputs
   (U, ω, background_flow) world→body by `body_orientation`, outputs body→world. Lets the
   vessel lie along **world-x** (screw body-z→world-x). Identity = no-op (legacy unchanged).
   Force AND torque frame-invariant to machine precision.
3. **ar4 axes**: vessel+flow along x (FVM `periodic_x`, lift `axis=0`, IBM cylinder along x),
   gravity ⊥ along −z. AR4 arm holds the magnet via ar4's inverse-dynamics controller
   (`control/controller.py`, reused verbatim, target = `[body.x, 0, standoff]`). Arm holds
   with 0 drift.
4. **Swim mechanism in the coupled experiment** (D11): rotating the screw produces axial
   thrust that grows with rotation rate (chirality coupling R_FΩ). So given rotation, it
   swims. The hydro is correct.
5. Tests green with all the above: experiments suite + C1/C2 (16), effects + M0 (79),
   e2e differentiability machine-precision.

## The BUG (what to fix)
The screw **whirls** instead of synchronizing to the 3 Hz drive. Root cause (D12c):
**genuine under-damped rotational dynamics** — the screw's tiny rotational inertia
(I_eff ≈ 7e-11) + small axial rotational drag make the magnetic-rotational mode
second-order and under-damped, so it whirls. **Confirmed dt-independent** (identical
libration at dt 5e-4 vs 1.5e-4) → NOT a numerical artifact.

A real microswimmer is **overdamped** (Re≪1, *zero* rotational inertia → first-order →
locks, no whirl). The current body uses the **inertial** RigidBodyNode mode → whirls.

### Things already tried (don't repeat)
- Drag-torque sign is fine (reaction + −1 edge flip damps; D3).
- Orientation-lag negative damping: real, partly fixed by a coupling group (D4) but not
  the whirl.
- Big implicit coupling group `[fvm,bem,body,ext_magnet,magnet]` (added via
  `make_two_scale_coupling(extra_coupling_members=)` + `TwoScale(extra_coupling_members=)`,
  composer attaches TwoScale **last**): doesn't fix the whirl (it's the rotational ODE).
- Smaller dt: no change (D12c).
- Standoff sweep: red herring (AR4 reach-limited to ~0.15 m → field constant). Vary
  `MAG_DIPOLE` to change field strength instead.
- `BODY_MODEL=overdamped` (RigidBodyNode `use_inertial=False, use_analytical_drag=False`):
  NaNs at step 0 — that mode needs a **mobility** drag node (velocity-out), but the BEM is
  a **resistance** node (force-out). Incompatible as-is. (Composer has `BODY_MODEL` knob;
  default is `inertial`.)

## THE FIX — Option A (recommended): mobility-based overdamped swimmer
The overdamped force balance is `[U;Ω] = M·F_ext` where `M = R⁻¹` (mobility) and `F_ext` =
magnetic torque/force + gravity (+ the background-flow contribution). No inertia integration
→ first-order → locks, no whirl.

Concretely, one clean route:
1. Have the confined BEM expose its **6×6 resistance matrix R** in Schwarz/coupled mode
   (it already computes it in *standalone* mode → `_init_confined_standalone` sets `self._R`;
   the schwarz path LU-factorises `A_conf` but doesn't expose a 6×6 R). Either expose R, or
   add a node that assembles R from the BEM operator each step.
2. An **overdamped body** that, given the external load `F_ext` (magnetic + gravity) and the
   background-flow-induced force, solves `[U;Ω] = R⁻¹·(F_ext + F_background)` directly. This
   is the de Jongh `swimming_velocity` relation generalised (force-free: with F=0 along the
   free DOFs, U = −R_FU⁻¹ R_FΩ Ω). See `scripts/dejongh_benchmark.py::swimming_velocity` and
   `compute_R_matrix` for the exact algebra (this is what the de-risk used to get 3.27 mm/s).
3. Wire it into the coupling group so the background flow + magnetic torque are
   self-consistent each step.

Watch: the wall table assumes a **centred, axis-aligned** body (`_check_centering` warns
off-axis). Keep the screw near the vessel axis (it already does with DELTA_RHO small).

## THE FIX — Option B (faster demo, less correct)
Prescribe the **locked rotation** Ω = drive-rate about world-x (valid below step-out — the
operating regime) and let translation emerge force-free. Shows the swim through the pipe at
~de Jongh speed; treats magnetic SYNC as the separate Problem-1 study (the lit review's
Problem 1 = step-out/Mason, so "does it lock" is partly a physics question, not just a bug).
Implementation sketch: a body mode with Ω prescribed (kinematic about x at the drive rate)
and V from the force balance / inertial translation (translation is NOT stiff, so it's fine).

## How to reproduce / test the current behaviour
```python
# from MIME/, source ../.venv/bin/activate, JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
from mime.experiments import schwarz_vessel_helix as S
import importlib.util
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ctrl=load('experiments/schwarz_vessel_helix/control/controller.py','c')
p={'N_THETA':16,'N_ZETA':24,'SWIM_MODE':'free','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':True,'DELTA_RHO':0.0}
gm=S.build_graph(p); st=None
for i in range(200): st=gm.step(ctrl.get_external_inputs(p,i,st))
b=gm.get_node_state('body')   # b['angular_velocity'][0] = spin about x (whirls); b['position'][0] = x-swim (~0)
```
`_debug/d11_thrust.py` (prescribed-rotation thrust, proves the mechanism) and
`_debug/d10_bigroup.py` (inertial + big group, shows the whirl) are the best starting refs.

## Success criteria
With `BODY_MODEL` overdamped (or the locked-rotation demo): the screw rotation **locks**
(≈ the drive rate, not whirling) and the body **translates along x** (through the pipe) at
~mm/s order (compare to the 3.27 mm/s force-free de-risk number). Then update the e2e test
+ the plan, and clean up `_debug/`.

## Process notes (standing constraints)
- Commit checkpoints as you go (this is the debug branch); on a snag, break down / change
  angle. End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Never touch** the pre-existing uncommitted ar4_helical_drive working-tree changes,
  `probe_cache.py`, or the `output/` dirs. Commit only my files. Push only if asked.
- Effects-first composition is the house style (see `[[effects-first-composition]]`).
