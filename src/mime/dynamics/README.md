# `mime.dynamics`

**How each thing's state evolves**: the registered dynamics entries.

## What goes here

- **`body/`**:
  - `locked`, `rigid_overdamped`, `rigid_inertial`. The inertial body integrates resistance implicitly,
    `(M/dt + R)·V' = (M/dt)·V + F`.
  - Confinement and its velocity rules: remove the outward velocity, inelastic, or a friction cone.
- **`flow/`**: `quiescent`, `prescribed`, and the finite-volume flow solver. The solver is rewritten for 0.3. Coupling
  a coarse flow to a fine one near the body (multi-resolution) is a stretch goal.
- **`apparatus/`**: `fixed`, `arm`, `motor`, and a coil driver.

## How an entry works

In phase 1, an entry plans. It declares its targets (what interactions may deliver into it), its reads, its capability
record, its validity and its differentiability.

In phase 2, it builds its node or nodes once, with every registration addressed to it. For example, a flow solver is
constructed with the bodies it must include.

## Boundaries

- Built by `build`, using nodes from `nodes/`.
- MIME owns the solvers its value depends on, such as the fluid solver, and keeps each one narrow.

## Status

Empty. The rigid body and confinement land in dev1's phase 4, and the arm and motor in phase 7. The rewritten flow
solver comes in a1.

Phases and releases are explained in the [package manifest](../README.md#status).
