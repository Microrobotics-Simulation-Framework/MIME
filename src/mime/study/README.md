# `mime.study`

**Many runs of one experiment**: sweeps, variants, fits and sensitivities.

## What goes here

- **Sweeps**, batched with `jax.vmap` over graph parameters.
- **Variant loops**, for changes that cannot be batched. A sweep over a material condition is one example, since it
  re-reads materials at build.
- **Fits**, with datasets and fit reports. Each result is a value carrying its provenance.
- **Sensitivities**.

## Boundaries

- It builds and runs experiments through `experiment`. Each sweep member starts from a fresh graph.
- It stays in MIME until a second library needs it.

## Status

Empty. Sweeps land in dev2, and fits in a1.

Phases and releases are explained in the [package manifest](../README.md#status).
