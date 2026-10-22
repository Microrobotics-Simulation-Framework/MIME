# `mime.build`

**Turns a checked declaration into a MADDENING graph.** This is the only code that calls MADDENING's graph API.

## What goes here

- **Phase 1, plan.** No graph exists yet.
  - Resolve handles, typed paths and strings.
  - Take the world frame from the scene once, and refuse a missing axis.
  - Ask every entry for its plan, and read materials through `materials/`.
  - Check reads and targets against `vocabulary/`.
  - Validate each written model choice against its support table, and check each `rate=` divides evenly.
  - Merge the requests to solve things together into coupling groups.
  - Derive ids from names (`drag:umr`) and seeds, and compute the operating point.
  - Refuse with every reason at once, each with a class and an action.
- **Phase 2, construct.**
  - Construct each node from resolved arguments.
  - Add nodes in the canonical order: providers, interactions by id, then media, bodies and apparatus.
  - Add edges through restricted builders, coupling groups, and the controller's external inputs.
- **Post-build checks.**
  - Edges into summed targets are additive, and an exclusive target has at most one edge.
  - No target is also an external input.
  - No node silently drops a constructor argument from its saved form.
  - The derived hash.
- **The build report**: one orientation line per thing, the validity outcomes, and where every material value came
  from.

## Why it is strict

On MADDENING 0.4, node and edge order is part of the model. A non-additive edge silently replaces whatever was summed
into that input before it. The canonical order and the builders' additive flags remove both hazards. Three kinds of
test check they stay removed:
- graph snapshots;
- shuffled declarations that must give bitwise-identical runs;
- a mutation test per post-build check.

Entries never see the graph itself, only a restricted builder.

## Status

Empty. It lands in dev1's phase 2, tested on toy entries before any physics, since most of the risk sits here.

Phases and releases are explained in the [package manifest](../README.md#status).
