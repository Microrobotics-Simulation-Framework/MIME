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
  - Add edges through restricted builders, coupling groups, and the controller's external inputs: one per write,
    derived from its declaration and from nothing else, for a callback. A controller inside the compiled step adds
    its node (and its latency's delay line) first instead, the rest in the order MADDENING runs the callback's graph,
    its reads and writes as edges; a step that cannot run there is refused (`controller.untraceable`).
- **Post-build checks.**
  - The experiment builds some node: one that builds none is refused (`run.nothing-to-step`) before MADDENING,
    which has no step to take, is asked for one.
  - Edges into summed targets are additive, and an exclusive target has at most one edge.
  - No target is also an external input. Every command has one writer, its drive's edge, the external input the
    controller's write derives or the controller's edge, and no external input is one no write derives.
  - A controller inside the compiled step reads every node at the start of the step and writes within it (each read
    edge reads the previous step, each write edge this step), and every other edge reads the step it reads with the
    controller called between chunks (`post-build.controller-order`); a read of a node its writes do not reach runs
    before it whatever the order, and is refused (`controller.unreached-read`).
  - No node silently drops a constructor argument from its saved form.
  - Every node input declares its units, and every edge carries the vocabulary's units at both ends.
  - MADDENING's own checks: an issue that would stop `compile()` is a refusal, a units warning among them; every
    other warning goes into the report with its source.
  - The compiled step, traced once: no host callback inside it.
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

Landed in dev1's phase 2, tested on toy entries with no physics:
- phase 1 (`plan_experiment`): handles, typed paths and strings resolved, with suggestions; every read and target
  checked against the vocabulary and the bound dynamics; rungs against support tables; rates against `dt`;
  registrations, reader slots, coupling groups (merged requests), ids and seeds; every refusal at once;
- the world frame as a stand-in (`up` declared once, gravity along `-up`) until the scene lands;
- phase 2 (`construct`): the canonical order, the restricted builders, groups with `solver="ift"`, `atol=0` and an
  explicit tolerance and cap, then the post-build checks and `compile()`;
- the post-build checks: summed targets take only additive edges, an exclusive target one edge, no target is an
  external input, no transform on an edge into a target, no node drops a constructor argument from its saved form,
  and MADDENING's base step is `dt`;
- the build report and the graph's canonical text.

Since phase 3:
- the restricted builders label every edge with the vocabulary's units (`source_units` and `target_units`, spelt as
  the vocabulary spells them; MIME uses no edge transforms, so both ends carry the same label), and a post-build
  check refuses an edge without labels, a label that differs from its target input's `expected_units`, and a node
  input that declares none;
- MADDENING's `validate()` and `compile()` warnings go into the build report, each with its source; a units warning
  is a refusal;
- the run's structure in the report: a run is MADDENING's one jit-compiled step, scanned, and the report says "one
  compiled scan" or names what runs on the host between compiled chunks: a controller in its callback mode, called
  once every control period. The report's controller lines give its mode, period and latency, each read with its
  class and each write with the command it replaces the drive of;
- a probe of each node's step code: every node is traced once at build through MADDENING's public node interface
  (its state, the inputs the graph resolves for it, its `update` and `compute_boundary_fluxes`), each trace walked
  with every scan, loop and branch body, and a host callback (`pure_callback`, `io_callback`, `jax.debug.callback`,
  `jax.debug.print`) is refused, naming the node. Nothing private is read;
- the derived hash: canonical JSON of MADDENING's description of the built graph plus the contents of every static
  array (which MADDENING describes by shape and dtype only), observations hashed apart from the physics.

MADDENING's and JAX's calls whose hints leave a type unknown go through `_maddening.py`, once, so the rest of the
package is checked strictly. The operating point, material reads and the other probes come with physics and
materials.

Since phase 4b:
- **declared values** (`exp.declare("umr.mass", kg(...), reason=...)`) checked in phase 1 against the things declared
  (a value the thing has, in its units and shape, with a reason, declared once) and read through `mime.materials`,
  each entry receiving the values its plan reads in SI (`ctx.values`); an undeclared read is refused with the line
  to declare it, and the report lists every value with "declared in the experiment", its reason and its readers;
- **geometry**: a structure's geometry kind is planned first, so a direction can refer to it (`tube.axis`) in any
  order of the script; a medium's `fills=` names a region; a setting marked `Direction` becomes a unit vector;
- **confinement**: a `Confinement` in a plan (an inequality-constraint target) is resolved against a fixed
  structure's surface into a `ResolvedConfinement` (the surface's shape and parameters, the margin, the body's
  extent, the velocity rule) handed to the body's dynamics at construction, at most one per body; a surface of a
  moving thing is refused as reserved; a reserved rung (`reserved_rungs`) is refused with its reason;
- **drives**: each command of an apparatus's built providers gets its drive's node (`drive:<apparatus>.<command>`,
  after the observations in the canonical order) and one exclusive edge into the command; a drive's quantity is
  checked against the command's dimension; a command with no drive, or a drive with no command, is refused;
- **a uniform field** (`Capability.UNIFORM`) gives its reader a slot with no points;
- **the canonical order gathers each coupling group's members at the place of its last member**: MADDENING 0.4.x
  runs a group as one block at its first member's place, so a law added between a group's law and its body would
  otherwise run after the group, and the body read its load a step late (it changed the full toy's node order);
- **lag accounting** (`_lags.py`): every edge classified, from MIME's own plan (the node order, the edges its
  builders declared, the groups), as reading this step, the previous step (an edge inside a feedback loop whose
  source is scheduled at or after its target) or the current iterate (inside a coupling group); shown in the report
  and in the graph's canonical text. A previous-step read is **planned** when a law, provider, dynamics or
  observation reads the published state of a thing bound to it; any other (a delivered load, a field, a command,
  read a step late) is refused as `lag.unplanned` unless a coupling group closes its loop. After `compile()` the
  classification is checked against MADDENING's compiled schedule (its back edges, less a group's internal edges)
  and `validate()`'s loop lines, a disagreement refused as `lag.cross-check`. `latency_in_steps` counts a path's
  previous-step reads; a controller inside the compiled step (phase 7b2) takes one, its start-of-step read, and its
  latency's steps are its delay line's.

Since phase 6:
- **the world frame from the scene**: phase 1 takes up from `Scene.new(up=..., units=...)`, once, and refuses an
  experiment's own `up=`; it composes the scene's references (`mime.scene.compose`), places every thing at its prim
  (refusing a missing, taken or shared prim, and a name held twice), and reads each magnet's design data, a
  magnet's direction then directing its body's moment (`mime.materials` reads the moment as a magnitude);
- **geometry reads are static unless declared `moving`**: a surface or direction of a thing with a dynamics, read by
  an entry that does not declare `moving`, is refused as `geometry.moving-read`, naming the entry, the thing and why;
- **the report** has the scene's line, one orientation line per thing (`OrientationLine`) and the stage's hash;
  `stage_spec(plans)` is what the stage writer is given.

Since the first documentation phase, **the refusal table**: every code MIME constructs, with the classes it is refused
in, is a row of `refusal_table.txt` beside this module, generated from a scan of the source (`mime.docs`) and checked
against it by `mime docs check`. A `Refusal` constructed with a MIME code the table does not hold, or in a class it
does not give the code, raises `UncataloguedRefusal`; a code whose first word is a registered namespace (`labx.`) is
that namespace's own, accepted and `uncatalogued`. A problem an entry's plan returns with a code it cannot be refused
with is refused as that entry's defect (`entry.defect`), so a lab's code never crashes a build.

Since phase 7a2, **chains**: each segment planned in chain order (its rung, its settings, what they make it), each
after the first mounted on a frame of the one before it by an edge the build derives (`b.mount`), the world poses
of every frame at the start composed along the chain; an apparatus's parts resolved onto their mounts, a provider
reading its part's pose by the same edge (`b.part`); each segment's commands driven as `<segment>.<command>`; a law
delivering into a chain that takes no load refused (`target.kinematic-chain`); the canonical order keeping a chain's
segments in chain order. The report lists each segment's description as read, every element not used or read past,
and the stage holds the chain's prims and its parts.

Phases and releases are explained in the [package manifest](../README.md#status).
