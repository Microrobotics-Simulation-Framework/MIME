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

Each passes its family's suite (`mime.testing.MAGNETIC`, `GRAVITY`, `CONTACT`). Drag (the surrogate with
lubrication) and the hydrodynamic family come in 4c.

Phases and releases are explained in the [package manifest](../README.md#status).
