# `mime.surrogates`

**Learned models that nodes evaluate.**

## What goes here

- **A surrogate as a type** (`Surrogate`): an asset record naming its evaluator, the outputs it delivers (a resistance,
  and a flow-load operator where it was trained for one), the features it reads in order, its accuracy per use and its
  provenance, with the shipped derivation record that holds its builder, inputs by hash, parameters and checks.
- **The Cholesky MLP drag surrogate** for helical robots in a tube, the first evaluator (`cholesky_mlp`).
- Its weights are package data in `data/`, pinned by sha256.
- **The generator** of a screw surrogate's training data and its held-out protocol (`mime.surrogates.generator`,
  imported on its own, since it builds experiments), called from `scripts/surrogate/generate.py`.

## A lesson carried over

In the 0.2 line, this surrogate assumed the tube ran along z. In an experiment whose tube ran along x, it read the
robot's position along the tube as a sideways offset, and clamped on every step. That robot advanced about nine times
too far per turn. In 0.3, the surrogate takes the axis from the tube's geometry, and a surrogate that cannot take one
is refused.

## Status

Since phase 4c1:
- **the Cholesky MLP** (`read_cholesky_mlp`, `predict_resistance`): the network's 21 outputs are the lower triangle
  of a Cholesky factor, its diagonal through a softplus plus 1e-5, so the 6 x 6 resistance `L L^T` is symmetric
  positive definite by construction. Plain JAX, ported from the 0.2 line; the training script stays there. `L` is
  built whole-array (a gather through a constant index map and two `jnp.where`s), not by 21 single-entry writes;
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

Since phase 7i:
- **the type** (`Surrogate`, `ScrewSurrogate` its screw form, `Accuracy`, `Provenance`, `Evaluator`): the drag's node
  reads a surrogate's weights through the evaluator its record names (`register_evaluator`, `evaluator_named`), and
  refuses a record whose features are not the ones it builds. An accuracy nothing measured is `None`: unknown, not
  bounded. `dejongh_mlp_v2`'s record states its swim-speed accuracy as MIME-ANO-114's, what that bounds, and that its
  training sampled the lateral surface alone, about the end of the body's axis;
- **staleness, mechanically** (`staleness`, `current_inputs`): a surrogate's derivation record names the tables its
  training read as its `wall:tube` inputs, by hash, and is stale when the tube's wall kind reads another table now.
  `dejongh_mlp_v2` is stale, as it declares (MIME-ANO-117); its retraining is the next phase's;
- **the generator and the held-out protocol** (`mime.surrogates.generator`): de Jongh's screw bodies resting on the
  floor of a straight tube, through `Drag.BEM` in the `tube` wall kind (the corrected table, never the old ones), in
  float64, the resistance in the surrogate's canonical frame about the body's centroid. The domain (`DOMAIN`) is de
  Jongh's seventeen designs in his four tubes, at the contact margins a resting body keeps, and no nearer the wall
  than a use's near-wall edge, a multiple of the spacing measured by the near-wall check (MIME-ANO-114), below which the
  `Lubrication` blend takes the body over. The swim speed's edge lies inside the contact margin; the whole
  resistance's (its translation normal to the wall, its pitching rotation) lies past it at every spacing the check
  reached, which the domain names among what lies outside (`Domain.outside`). Two held-out sets are
  disjoint from the training set by construction: the gaps between the training margins, and the body turned about
  its own axis (an input the surrogate does not read). `accuracy` judges a surrogate on a held-out set per use,
  stating that it estimates the error against this method at this discretisation and bounds nothing about the flow.
  Each case's checks are the build's: a case with a failed check produces no data (`mime.assets.derive`).

Phases and releases are explained in the [package manifest](../README.md#status).
