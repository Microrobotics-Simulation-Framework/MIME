# `mime.gradients`

**Whether a gradient through the experiment can be trusted.**

## What goes here

- **Gradient statuses.** Every entry declares one, and the statuses compose along each path. A piecewise status is
  derived from the switches the entry declares (`mime.core.switch`): `piecewise_jump` where one is a jump,
  `piecewise_kink` where all are kinks, `exact` where there are none.
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

`GradientStatus` landed in dev1's phase 2: every registered entry declares one. Since phase 7s, `piecewise` is split
into `piecewise_kink` and `piecewise_jump`, and `status_of_switches`, `derived_status` and `status_problem` give the
status an entry's switches and its listed raw sites allow, which `mime docs check` holds every entry to. Composition,
the map and the checks arrive with the first fit, in a1.

Phases and releases are explained in the [package manifest](../README.md#status).
