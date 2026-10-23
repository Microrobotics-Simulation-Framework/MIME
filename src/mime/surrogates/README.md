# `mime.surrogates`

**Learned models that nodes evaluate.**

## What goes here

- **The Cholesky MLP drag surrogate** for helical robots in a tube.
- Its weights are package data in `data/`, pinned by sha256.

## A lesson carried over

In the 0.2 line, this surrogate assumed the tube ran along z. In an experiment whose tube ran along x, it read the
robot's position along the tube as a sideways offset, and clamped on every step. That robot advanced about nine times
too far per turn. In 0.3, the surrogate takes the axis from the tube's geometry, and a surrogate that cannot take one
is refused.

## Status

Since phase 4c1:
- **the Cholesky MLP** (`read_cholesky_mlp`, `predict_resistance`): the network's 21 outputs are the lower triangle
  of a Cholesky factor, its diagonal through a softplus plus 1e-5, so the 6 x 6 resistance `L L^T` is symmetric
  positive definite by construction. Plain JAX, ported from the 0.2 line; the training script stays there;
- **what each packaged surrogate was trained on** (`SURROGATES`, `DEJONGH_MLP_V2`): de Jongh's screw bodies (the
  cross-section modulation 0.33 and two starts fixed in training), with the ranges a build checks (`nu` from 0.25 to
  3.5, the training data's, where the 0.2 line declared 2.33 to 3.7; `kappa` from 0.25 to 0.66, as declared) and the
  offset fraction (up to 0.3, as declared), which moves during a run and is clamped instead. Read from the training
  data (which reproduce the weights' stored feature statistics): `kappa` was trained at 0, 0.12 and 0.246 to 0.655,
  so the declared upper bound, 0.66, lies just above it; offsets reach 0.35; lengths 4.13 to 9.73 mm, which the 0.2
  line declares no range for. `Drag.Surrogate` evaluates it (`mime.interactions`).

Phases and releases are explained in the [package manifest](../README.md#status).
