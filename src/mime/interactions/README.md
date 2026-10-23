# `mime.interactions`

**The physical laws between things**, one module per family.

## Families

- **`magnetic`**: torque, and the full wrench.
- **`hydrodynamic`**: drag, through its rungs. The rungs are a learned surrogate, a penalty term inside the flow
  solver, and later Stokes boundary elements.
- **`gravity`**: weight and buoyancy.
- **`contact`**: by projection (inside the body's own update) or by penalty (as loads into each side). Velocity rules
  include a friction cone.
- **`acoustic`**: radiation force, from a2.

## The contract

- An interaction is **one physical law among named things**. It reads declared interfaces and delivers into a declared
  target of a thing's dynamics.
- It never touches another interaction's nodes.
- A law that needs another law's output reads it from the thing that received it.
- **The model choice is written in the script**: `Drag(rung="surrogate")`, or the alias `Drag.Surrogate(...)`, which
  saves the same way. Nothing is picked automatically. A choice outside its validated range is reported.
- Transfers between a body's points and a grid go through `mappings/`.

## Status

Since phase 4b:
- **`magnetic.py`**: `Magnetic.Torque()` (`magnetic.torque` 0.1.0), `tau = m x B` on the body's permanent moment
  turned by its orientation, from a uniform field; it reads the pose at the start of the step, a one-step lag of
  `omega dt` in phase that lag accounting shows;
- **`gravity.py`**: `Gravity()` (`gravity` 0.1.0), weight minus buoyancy `(m - rho_f V) g` with gravity from the
  declared up axis, and the couple when the body's centre of buoyancy is off its centre of mass (reading the pose
  only then);
- **`contact.py`**: `Contact(rung="projection" | "penalty")` (`contact` 0.1.0), `Contact.Projection(...)` its alias;
  projection binds a fixed structure's surface (`surface=tube.inner_surface`) and goes to the body's dynamics at
  construction, with its margin, the body's extent and the velocity rule `remove_outward`; penalty is reserved, and
  projection against a moving thing is refused as reserved.

Each passes its family's suite (`mime.testing.MAGNETIC`, `GRAVITY`, `CONTACT`).

Since phase 4c1:
- **`hydrodynamic.py`**: `Drag(rung="surrogate")` (`drag` 0.1.0), `Drag.Surrogate(...)` its alias, which saves
  identically. The Cholesky MLP's resistance for a de Jongh screw body in a straight tube, from `weights=asset(
  "dejongh_mlp_v2")`, with `lubrication=Lubrication(epsilon=...)` blended in as one model when given (the
  sphere-wall asymptotes of Cox & Brenner and Goldman, Cox & Brenner, weighted by `exp(-gap / epsilon)`; the gap
  floored at 1 nm). It binds `body=` and `medium=`, reads the body's pose at the start of the step, and the tube's
  axis, a point on it and its radius from the region the medium fills, through the shape protocol: a medium filling
  no region, or a region with no axis, is refused. It reads the medium's `viscosity` and the body's
  `helix_wavenumber`, `length` and `radius`, declared in the experiment for now (`mime.materials` is the one place
  that reads them). It delivers `resistance` (an operator the body solves with implicitly) and `background_load`
  (zero in a medium at rest). A body or tube outside the surrogate's training data in `nu` or `kappa` is refused at
  plan time, naming the entry, the value, the range and where the range comes from; recording an acceptance instead
  is for a1's profiles. The offset from the axis keeps the 0.2 line's clamp, reported as `clamp_fired` in the node's
  state. The resistance is about the end of the body's axis, as the surrogate was trained, and is delivered as the
  body's (MIME-ANO-112, for the maintainer). `Lubrication` is a group of settings (`mime.registry.SettingGroup`),
  which the saved form writes by its group name.

It passes the hydrodynamic family's suite (`mime.testing.HYDRODYNAMIC`), whose analytic oracle and second member, a
Stokes sphere, is registered by MIME's tests only; no public `Drag.Stokes` is built yet.

Phases and releases are explained in the [package manifest](../README.md#status).
