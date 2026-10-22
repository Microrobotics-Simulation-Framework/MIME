# `mime.nodes`

**The MADDENING nodes underneath the entries.**

## What goes here

- **`body/`**, **`actuation/`**, **`magnetic/`**, **`hydro/`**, **`acoustic/`**.
- **`flow/fvm/`**, the finite-volume flow solver, rewritten for 0.3.

## How nodes arrive

Nodes are ported one at a time from the 0.2 line, each behind the new contracts. Each one must pass:
- MADDENING's `verify_node` battery;
- a small physics oracle;
- a golden output;
- a differential test against frozen outputs of the 0.2 code.

## Boundaries

- Imports nothing above it, so every node also runs on MADDENING alone.
- Importable, but **outside 0.3.x's stability promise**.

## Status

Empty. The first experiment's nodes arrive in dev1's phase 4, the arm and motor in phase 7, and the rewritten flow
solver in a1.

Phases and releases are explained in the [package manifest](../README.md#status).
