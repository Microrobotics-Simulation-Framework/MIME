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
- **[`tables/`](tables/)**, the table evaluator: one differentiable reader of tables on OpenMaterica's table
  description, for MIME's solver-output tables and OpenMaterica's material tables, refused at an edge and never
  clipped outside its readable range.
- **The energy ledger's machinery** (`mime.core.ledger`), which a receiver's step runs only where the ledger is
  attached: the one pairing rule, each law's paired work recorded beside the step, the step identity's terms and
  running sums. See [The energy ledger](#the-energy-ledger).

## Switches

A **switch** is a relation in a node's step together with the branches it selects: a maximum, a `where`, a norm
at zero. Its derivative is not the computed value's there, so MIME names every one. A step never evaluates a raw
non-smooth operation; it calls `mime.core.switch` instead.

**The functions.** `maximum`, `minimum`, `absolute`, `clip`, `norm` (MIME's safe norm, with the derivative zero at
zero), `where`, `predicate` and `searchsorted` (a query's interval among sorted breaks: a table's cell lookup) each
evaluate a switch. Each returns its operation's value and JAX's derivative of it, ties included, so writing a site
through them changes no program. Each takes three keyword arguments:

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
switch log that the collector feeds comes later.

## The energy ledger

A **receiver** (a dynamics' node: the rigid body's) is metered when it is built with `ledger`, the deliveries it
takes, each a law's id, its class, its delivery kind and its run-time check (`LedgerDelivery`). Only the build gives
a node its ledger, and only where the experiment attaches it; without it the node is the node it always was, its
parameters, inputs, state and step included, every bit of the program unchanged.

- **Per-law inputs.** Each law delivers into an input of its own (`law_input`) beside the one every law sums into
  without the ledger, which stays for anything unattributed. Attribution is then the check that the unattributed
  inputs paired no work: exactly zero.
- **One pairing rule** (`paired_work`), keyed by `mime.vocabulary.DeliveryKind`'s values as strings: a load read at
  the step's start with the configuration's actual change; a force the step solves for with the step's solution; a
  flow handed in with the effort returned; a delivered inertia by its change. A law hosted in its receiver's step (a
  contact) is paired with the solution under its own name.
- **The step identity.** A metered node names its own scheme's numerical terms (`MimeNode.LEDGER_TERMS`) and gives
  its storage (`MimeNode.storage`) and its step's account (`MimeNode.identity_terms`, a `StepAccount`): `dH - W + N`
  is judged against `IDENTITY_ULPS` units of rounding on the step's scale, the magnitude of the arithmetic that forms
  its terms (each product's factors in magnitude, the solve's included), so that terms that cancel, and a stiff
  solve, are judged on the size of what was rounded. `recorded` keeps running sums over the run's steps and
  sub-steps in the receiver's state (the fields prefixed `ledger_`); `balance` and `balance_floor` read the run's
  books from them.
- **Definiteness at run time**, for a law whose operator's definiteness depends on the state:
  `definite_at_floor`, one Cholesky factorisation of its scaled symmetric part shifted by `DEFINITE_ULPS` units of
  rounding, so that a semidefinite operator passes where a plain factorisation fails on it (a zero diagonal entry
  passes only with its row zero, judged exactly).
- **A dissipative law's sign**, read from its own row: `gift_steps` counts the steps in which its paired work was
  positive beyond the identity's floor. The identity does not see it: it books the gift under the law's name.
- **A potential law** states its family's energy (`MimeNode.energy`) on its node, from the parameters a study moves;
  its class is settled on the law alone, and no per-step residual of it is kept.

## Status

The typed node state helpers.

Also:
- **`MimeNode`**: every constructor argument kept in `params`; `constants(params)` gives what a step reads, the
  graph's live leaves over the constructor's; a node writes its step as `advance(p, state, inputs, dt)`, and
  `update` runs it compiled once per node, so the verification battery's eager calls are one call each and the
  graph's trace inlines it; `node_input` declares an input with its units; `run_dtype` and `as_run_array`;
- **run-constants** (`MimeNode.run_constants`): quantities a node makes from its leaves that are constant over a run
  (the MRS drag's held body-frame matrix; a one-way field solved once per run; a precomputed operator), each in place
  of a leaf of its own of the same shape and dtype. MIME's chunked runs make them once per chunk inside the compiled
  program, before the step loop (`mime.build._maddening.with_run_constants`, `chunk_function`), so that reverse mode
  derives each once per chunk rather than at every step; inside the loop the step's constants hold
  `RUN_CONSTANTS_GIVEN` (`run_constants_given(p)`; set by `run_constants_made` around the loop's trace, part of the
  node's compiled step's cache key) and the step reads them as given. Called otherwise (MADDENING's own scan,
  `graph.step`) the step makes them itself. A held value's pattern: the value made at build as a `trainable=False`
  leaf, the leaves it was made at beside it, and `held_or_made` choosing between them in the program, never a value
  compared with a constant;
- **`quaternion`**: products, rotation matrices and `exponential`, whose value and derivatives are finite at a zero
  rotation (0.2 divided by `|omega|` and differentiated to NaN at rest);
- **shapes** (`Shape`, the protocol a geometry kind's surfaces are evaluated with: signed distance, outward normal,
  the axis where there is one), `Cylinder` (a point within `AXIS_ULPS` units in the last place of its radius of its
  axis is on it, whatever the axis's direction: the resolution drag's on-axis frame shares), `register_shape` and
  `shape_named`; and the confinements' velocity rules
  (`VELOCITY_RULES`: `remove_outward`). Node metadata (MADDENING's `NodeMeta`) is not carried over from 0.2.

**The symbol table's form**, `Symbol` and `SymbolKind`, and `MimeNode.symbols`.
`NodeMeta` returns as the node pages' source, its checks in `mime.docs`; no node carries either yet.

**`mime.core.switch`**, the switch functions, `nonzero_norm`, their declarations and their one
context; every node site written through it (the rotation's exponential, the field's unit vector and drag's clamp
flag stay raw, listed); `MimeNode.fresh_update`; a shape's switch table (`shape_switches`).

**`tables`**, the table evaluator (see its [README](tables/README.md)), and the switch function
`searchsorted`, its cell lookup, whose raw forms (`jnp.searchsorted`, `digitize`, `interp`, `histogram`, `bincount`,
`unique`, `argwhere`, `lax.sort`, a floor division, a cast to an integer dtype) the switch scan refuses in a step;
and `without_slope`, the one selection that makes a derivative NaN on purpose (a table's one-point readable range).

**the solids' shapes**, `sphere`, `capsule`, `slab` (between two parallel planes) and `screw`
(de Jongh's screw surface, unbounded along its axis: its distance by a root find whose gradients are stopped, polished,
then a Newton step with them flowing, not taken on its focal set, where the step is singular), each declaring its
switches beside it; and `nearest_point`, the point less its signed
distance along its normal. Within a band about a medial set (a centre, a segment, a mid-plane) the distance is its value
there and the normal zero, each with the derivative zero in the point; the cylinder keeps its own.

**`multibody`**, the canonical tree (bodies, joints, named frames, loop closures as data), the URDF
reader sorting every element into read, recognised and not used, or refused, and the rigid-body maths in pure JAX,
ported from the 0.2 line's `control/kinematics/` with its oracles (see its [README](multibody/README.md)).

**`ledger`**, the energy ledger's machinery on a receiver (above), with `MimeNode`'s hooks (`ledger`, `metered`,
`LEDGER_TERMS`, `storage`, `identity_terms`, `configuration`, `energy`); the rigid body's node is its first receiver.
The build wires it only where `EnergyLedger` is attached (`mime.build`); a run's account reads the records
(`mime.results`), its arithmetic here and traceable (`losses`, `stored`, `balance`), so a derivative passes through it.
A dissipative law's sign counts a step in which its paired work was positive beyond the floor, or, held in the step, its
rule's removal at arrival negative beyond it.

Releases are explained in the [package manifest](../README.md#status).
