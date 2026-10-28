# `mime.core`

**What every node imports.**

## What goes here

- **`MimeNode`**, the base class of MIME's nodes, on MADDENING's `SimulationNode`.
- **Node metadata**: MADDENING's `NodeMeta` on a node class, and the symbol table beside it (`Symbol`,
  `SymbolKind`; `MimeNode.symbols`), which joins each symbol of the node's equations to its code. The checks are
  `mime.docs`'s.
- **Quaternion helpers**, with a norm that is safe at zero rotation. The 0.2 version produced a NaN there.
- **The switch functions** (`mime.core.switch`): every non-smooth operation a node's step evaluates, so that each one
  has a name. See [Switches](#switches).
- **Typed node state**: `NodeFields`, `typed_view` and `node_fields`, the static views by which a node types its
  state and inputs inside MADDENING's signatures. The convention is set out in
  [`nodes/README.md`](../nodes/README.md#typed-node-state).
- **[`multibody/`](multibody/)**, the rigid-body library: description readers into one canonical tree (URDF now),
  forward and inverse kinematics, frame Jacobians, the mass matrix, inverse and forward dynamics. Nodes, dynamics
  entries and controllers import it.

## Switches

A **switch** is a relation in a node's step together with the branches it selects: a maximum, a `where`, a norm
at zero. Its derivative is not the computed value's there, so MIME names every one. A step never evaluates a raw
non-smooth operation; it calls `mime.core.switch` instead.

**The functions.** `maximum`, `minimum`, `absolute`, `clip`, `norm` (MIME's safe norm, with the derivative zero at
zero), `where` and `predicate` each evaluate a switch. Each returns its operation's value and JAX's derivative of
it, ties included, so writing a site through them changes no program. Each takes three keyword arguments:

- `owner`: the switch's dotted name, `<entry or shape>.<switch>` (`contact.band`), shared by all its sites. A core
  helper that evaluates a switch (`remove_outward` and every velocity rule) takes its owner from its caller, as a
  keyword-only `owner` parameter: owners are keyword-only everywhere, and the switch scan reads one only as `owner=`,
  failing on an owner passed by position to a helper it can see. A lab's owners start with its package's name.
- `indicator`: the signed value whose sign selects the branch, in its own units, positive on the side the call names
  first (`where`'s first branch, `maximum`'s first argument); for a band, the value whose magnitude within the
  resolution counts as on it.
- `resolution`: the width within which the indicator counts as zero, in units in the last place of a named scale
  (`switch.rounding(scale)` where nothing sets one).

The indicator and the resolution are each a value or a zero-argument callable. A callable is evaluated only inside a
collector, so with no context open the step's program is exactly the operation's. Pass a value that is already
computed at the site as it is; pass anything else as a lambda.

```python
from mime.core import switch

beyond = shape.signed_distance(parameters, position) + inset
projected = position - switch.maximum(beyond, 0.0, owner="contact.project", indicator=beyond,
                                      resolution=lambda: 64 * eps * scale) * normal
length = switch.where(off, jnp.sqrt(switch.domain_guard(off, squared, 1.0)), 0.0, owner="cylinder.axis",
                      resolution=resolution, indicator=lambda: jnp.sqrt(squared))
```

Three selections are no switch and take no owner: `domain_guard(condition, a, b)`, whose fallback `b` only keeps an
undefined value (a square root's derivative at zero, a division by zero) out of a branch that is discarded wherever
`condition` is false; `nonzero_norm(x)`, the safe norm of a vector that is zero at no valid state (a direction the
plan refuses at zero, a quaternion an integrator keeps near unit length), whose zero branch is a domain guard; and
`static_select(mask, a, b)`, whose mask is fixed at build (a traced mask is refused). A norm whose zero a valid state
reaches is `norm`, with its owner (`contact.scale`, the body's distance from the origin). A square root takes a domain
guard's output. Where an operation's fixed formula differs from the site's, use the more general function with the
site's own predicate (the cylinder's band test goes through `where`, not `norm`). The domain-guard test
(`tests/saved/test_switch_harness.py`) must reach every guard a step calls with its condition false, and every
`nonzero_norm` away from zero: add a point there for a new one.

**Declare the owner** in the gradient record of the entry whose law it is, as a `switch.Declaration`: the owner,
what the indicator measures with its units, its smoothness order per output group of the node that evaluates it (a
jump, a kink, or continuously differentiable to some order: the integers are in `Declaration`'s docstring), a time
switch's settings, and the route to an exact derivative. A shape declares its switches beside its class
(`register_shape(..., switches=...)`), and a geometry kind naming the shape inherits them. The entry's status is
derived from its switches, and `mime docs check` fails on one declared otherwise. Give each owner a row in
`tests/saved/test_switch_harness.py`, which forces its branch each way and measures its orders; a switch no output
can show (never live, or live with no effect an output shows) says why in its declaration's `unmeasured`, and its
row then measures nothing.

**No switch function inside a transform** (`lax.cond`, `lax.switch`, `lax.scan`, `fori_loop`, `while_loop`, `vmap`,
`checkpoint`, `remat`, a nested `jit`): write it at the step's top level. A switch that must sit in a loop stays raw
and is listed in `docs/ratchets/raw_switches.txt` with its reason (the rotation's exponential, which the inverse
kinematics' loop reaches, is the first). The switch scan (`tests/saved/test_switch_scan.py`) traces every registered
node's step and fails on a raw non-smooth operation, a switch function inside a transform, an owner no entry declares
and a declared owner nothing passes, unless the site is listed with its reason.

**A new registered node class needs a configuration that traces it**, or the scan fails, naming it: the
configurations are script A's arms and the round-trip cases (`tests/saved/switch_configurations.py`), so a new
entry's round-trip case is enough; a node no case builds needs a configuration of its own there.

**The context.** One context, empty by default, set only by a context manager: `collecting()`, which gathers a row
(owner, indicator, resolution, branch) per evaluation, or the test-only setting, `forcing({owner: branch})` and
`replacing_fallbacks(scale)`. It is read while a step is traced, so whatever reads it traces the step afresh inside
it (`MimeNode.fresh_update`, or a new `jax.jit`); `update`'s compiled step was traced once and sees nothing. The
switch log that the collector feeds is a later phase's.

## Status

Since phase 4a: the typed node state helpers.

Since phase 4b:
- **`MimeNode`**: every constructor argument kept in `params`; `constants(params)` gives what a step reads, the
  graph's live leaves over the constructor's; a node writes its step as `advance(p, state, inputs, dt)`, and
  `update` runs it compiled once per node, so the verification battery's eager calls are one call each and the
  graph's trace inlines it; `node_input` declares an input with its units; `run_dtype` and `as_run_array`;
- **`quaternion`**: products, rotation matrices and `exponential`, whose value and derivatives are finite at a zero
  rotation (0.2 divided by `|omega|` and differentiated to NaN at rest);
- **shapes** (`Shape`, the protocol a geometry kind's surfaces are evaluated with: signed distance, outward normal,
  the axis where there is one), `Cylinder` (a point within `AXIS_ULPS` units in the last place of its radius of its
  axis is on it, whatever the axis's direction: the resolution drag's on-axis frame shares), `register_shape` and
  `shape_named`; and the confinements' velocity rules
  (`VELOCITY_RULES`: `remove_outward`). Node metadata (MADDENING's `NodeMeta`) is not carried over from 0.2.

Since the first documentation phase: **the symbol table's form**, `Symbol` and `SymbolKind`, and `MimeNode.symbols`.
`NodeMeta` returns as the node pages' source, its checks in `mime.docs`; no node carries either yet.

Since phase 7s: **`mime.core.switch`**, the switch functions, `nonzero_norm`, their declarations and their one
context; every node site written through it (the rotation's exponential, the field's unit vector and drag's clamp
flag stay raw, listed); `MimeNode.fresh_update`; a shape's switch table (`shape_switches`).

Since phase 7d: **the solids' shapes**, `sphere`, `capsule`, `slab` (between two parallel planes) and `dejongh_screw`
(de Jongh's screw surface, unbounded along its axis: its distance by a root find whose gradients are stopped, then a
Newton step with them flowing), each declaring its switches beside it; and `nearest_point`, the point less its signed
distance along its normal. Within a band about a medial set (a centre, a segment, a mid-plane) the distance is its value
there and the normal zero, each with the derivative zero in the point; the cylinder keeps its own.

Since phase 7a1: **`multibody`**, the canonical tree (bodies, joints, named frames, loop closures as data), the URDF
reader sorting every element into read, recognised and not used, or refused, and the rigid-body maths in pure JAX,
ported from the 0.2 line's `control/kinematics/` with its oracles (see its [README](multibody/README.md)).

Phases and releases are explained in the [package manifest](../README.md#status).
