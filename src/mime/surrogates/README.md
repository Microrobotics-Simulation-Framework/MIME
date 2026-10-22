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

Empty. It is ported in dev1's phase 4.

Phases and releases are explained in the [package manifest](../README.md#status).
