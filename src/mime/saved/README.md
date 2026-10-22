# `mime.saved`

**The saved form: what makes an experiment reproducible by someone else.**

## What goes here

- **The document.** It records what the experiment declares: registered names and versions, settings, bindings,
  scene paths, and material references (record id, pin, conditions). It never holds code, and never holds copies of
  record values.
- **The lock.** It pins the versions of MIME, MADDENING and OpenMaterica; every material record's id, version and
  content hash; the scene's identity; and the hash of every asset.
- **The resolved record** and **the derived hash** of the built graph.
- **Loading.** A load reads the document and never executes a script. It runs three load checks.
- **`verify`**, which reruns and compares.
- Later, the **experiment folder** (a standard, self-contained layout) and **pack**.

## How it stays complete

Saving and loading land together, early, and grow with every feature. A CI test walks the registry and requires a
round-trip case (build, save, load, rebuild, then the same hash and the same first steps) for every registered entry
and every kind of thing. An entry added without one fails CI.

## Status

Empty. It lands in dev1's phase 3. The folder and `pack` come at 0.3.0.

Phases and releases are explained in the [package manifest](../README.md#status).
