# `mime.gradients`

**Whether a gradient through the experiment can be trusted.**

## What goes here

- **Gradient statuses.** Every entry declares one, and the statuses compose along each path.
- **The rule for a differentiated study.** A path that is cut, or whose gradient is unverified, is refused unless
  acknowledged.
- **The gradient map**, and its checks:
  - the forward response to each input;
  - a finite difference that rebuilds the graph;
  - a stated probe protocol.

## Why the checks are strict

A finite difference taken only through graph parameters can pass on a wrong gradient. Two known cases are a value
baked in as a Python float, and a capped iteration. So the checks always rebuild, and keep planted failures as
controls.

## Status

`GradientStatus`, the five statuses, landed in dev1's phase 2: every registered entry declares one. Composition, the
map and the checks arrive with the first fit, in a1.

Phases and releases are explained in the [package manifest](../README.md#status).
