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
  3.5, the training data's, where the 0.2 line declared 2.33 to 3.7; `kappa` from 0.25 to 0.655, the largest trained value; `L / R` from 2.65 to 6.24, since phase 4c3) and the
  offset fraction (up to 0.3, as declared), which moves during a run and is clamped instead. Read from the training
  data (which reproduce the weights' stored feature statistics): `kappa` was trained at 0, 0.12 and 0.246 to 0.655,
  the 0.2 line's declared upper bound, 0.66, lay just above it; offsets reach 0.35, and for the operating geometries
  are sampled only at 0, 0.024, 0.1, 0.2 and 0.3 of the radius; lengths 4.13 to 9.73 mm, which the 0.2 line declared
  no range for. `Drag.Surrogate` evaluates it (`mime.interactions`).
- **Its accuracy** against the BEM it was checked against misses the 0.2 line's stated 1%: up to 16.2% in swim speed
  between 0.2 and 0.32 of the tube's radius, in the gaps between those samples (MIME-ANO-114, open: the BEM's
  near-wall check, dense data and retraining come first).

Phases and releases are explained in the [package manifest](../README.md#status).
