# `mime`: the package manifest

MIME (MIcrorobotics Multiphysics Engine) simulates field-actuated microrobots, magnetic and acoustic, moving through
confined, low-Reynolds-number flows such as vessels and channels. Gradients run through the whole simulation, so the
same experiment that predicts a result can also be fitted to measurements.

This package is **MIME 0.3, a full rewrite**. Every folder exists and says what it is for, and the code arrives phase
by phase (see [Status](#status)): so far the core that turns a declared experiment into a graph, tested on toy entries
with no physics, and the first experiment's body side (a rigid inertial body in a tube, a uniform rotating field, the
magnetic torque, gravity and contact), and the scene written as a USD stage. The 0.2 line is preserved at the git
tag `pre-0.3`.

MIME stands on two libraries:
- [MADDENING](https://microrobotica.org/maddening/), the coupling engine that runs the graph. MIME 0.3 targets
  MADDENING 0.4.x.
- [OpenMaterica](https://github.com/OpenMaterica/OpenMaterica), the materials library, which
  gives every material value a record, a source and a pin.

The scene is a USD stage ([OpenUSD](https://openusd.org)).

## The idea

A script **declares an experiment**: the things in it, how each one moves, the fields that act on them, the physical
laws between them, what to observe, and an optional controller. Nothing is built while the script runs.

`build()` checks the whole declaration first. If anything is wrong, it refuses with every reason at once, each with
what to do about it. Only then does it construct a MADDENING graph.

A built experiment saves as a document plus a lock file. Anyone can reload and rerun that pair without executing the
original script; the load resolves every entry among installed packages and checks the rebuild against the lock.
**The experiment, not the script, is the deliverable.**

```python
exp = Experiment("fl9_quarter_inch_water", scene=scene, profile="research", dt=..., ...)
tube  = exp.add_structure(Structure("tube", geometry=Tube(...), material=record(...)))
water = exp.add_medium(Medium("water", fills=tube.lumen, material=record("water")))
umr   = exp.add_body(Body("umr", prim=..., dynamics=RigidInertial(), initial=State(...)))
field = exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(...)}, ...))

exp.attach(Drag.Surrogate(weights=...), body=umr, medium=water)
exp.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
exp.attach(Gravity(), body=umr, medium=water)
exp.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
exp.observe(SwimSpeed(umr, along=tube.axis))

built = exp.build()        # checks everything, then constructs the graph; writes the stage, document and lock
built.run(duration=s(5))
```

This is a sketch of the intended API. The declaration and build machinery works today, and so do the scene, the
magnetic torque and force, gravity, contact, drag, materials read from OpenMaterica's records, and an apparatus's
kinematic chain carrying a magnet.

## Three parts, and what sits beside them

- **Things with dynamics.** These are `Structure` (a passive solid such as a tube wall), `Medium` (water, blood),
  `Body` (the robot) and `Apparatus` (a coil array, a robot arm carrying a magnet). Each thing's dynamics, such as a
  rigid body, a flow solver or an arm, becomes the stateful node or nodes of the graph.
- **Fields with providers.** A field is anything read at points. Its provider either evaluates it (a dipole), solves
  it over a medium (the flow), or reads it from a table.
- **Interactions.** Each is one physical law between named things, such as `Magnetic.Torque`, `Drag`, `Gravity` or
  `Contact`. It reads declared interfaces and delivers into a declared target of a thing's dynamics. A target is one
  of: a load, a resistance, a constraint, or a registration inside a solver.
- **Beside them** sit **observations**, which read and never write back, and the **controller**. A controller is
  declared once and can run four ways: inside the graph, as a Python callback, as an external process, or in an
  exported FMU.

Choices are explicit. The script names every binding and every model choice (`Drag(rung="surrogate")`). Nothing is
picked silently, and a choice outside its validated range is reported, never swapped for another.

## From declaration to graph

1. **The script** calls `add_*`, `attach`, `observe` and `set_controller`. These only record declarations.
2. **Phase 1, plan** (`build/`):
   - resolve names;
   - take the world frame from the scene;
   - ask every entry for its plan;
   - read materials through `materials/`;
   - check every read and target against `vocabulary/`;
   - merge the requests to solve things together into coupling groups;
   - refuse, with every reason.
3. **Phase 2, construct** (`build/`):
   - build the nodes in a fixed canonical order;
   - add edges through restricted builders, which set each edge's `additive` flag from its target;
   - add the coupling groups and the controller's inputs;
   - run the post-build checks.
4. **Write and run**: write the stage, the document and the lock, then run.

The order matters. On MADDENING 0.4, the order nodes and edges are added in is part of the model. So the order a
script attaches things must never change a result, and the tests check that it doesn't.

## Dependency direction

```
experiment  ->  build  ->  dynamics, fields, interactions, observations, control  ->  nodes  ->  MADDENING
```

- `nodes` imports nothing above it, so every node also runs on MADDENING alone.
- Each outside API has exactly one importer:
  - `mappings` for MADDENING's mapping registry and geometry edges;
  - `materials` for OpenMaterica;
  - `scene` and `viz` for `pxr` (OpenUSD).
- Platform-specific simulators and renderers never enter this package. Bridges to them live in separate packages.

## Manifest

| Package | What it is for | Origin | Lands in |
|---|---|---|---|
| [`vocabulary/`](vocabulary/) | the names of every interface (kind, units, frame, shape, sign); capability flags; reserved names | new | dev1, phase 2 |
| [`things/`](things/) | `Structure`, `Medium`, `Body`, `Apparatus`; parts, roles and state; handles | new | dev1, phase 2 |
| [`registry/`](registry/) | registration of every kind of entry, keyed by name and version | new | dev1, phase 2 |
| [`build/`](build/) | the two phases, the canonical order, restricted builders, post-build checks, the build report | new | dev1, phase 2 |
| [`experiment/`](experiment/) | `Experiment` and `BuiltExperiment`; run settings | new | dev1, phase 2 |
| [`saved/`](saved/) | the document, the lock, loading, `verify`; later the experiment folder | new | dev1, phase 3 (landed) |
| [`testing/`](testing/) | harnesses labs call from their own tests | new | dev1, phases 1 to 4 |
| [`cli/`](cli/) | the `mime` command, thin over the Python API | new | dev1, phase 3 (`mime verify`) |
| [`docs/`](docs/) | the documentation's machine-readable sources and their checks: the registry export, the symbol-table checks, the refusal scan, the ratchet lists, `mime docs check` | new | dev1, the first documentation phase |
| [`dynamics/`](dynamics/) | how bodies, flows and apparatus evolve | new | dev1, phases 4 and 7 |
| [`fields/`](fields/) | field providers | new | dev1, phases 4 and 7 |
| [`interactions/`](interactions/) | physical laws, one module per family | new | dev1, phase 4 |
| [`observations/`](observations/) | what an experiment measures | new | dev1, phase 4 |
| [`errors/`](errors/) | the boundary of the public calls: every error a MIME error with a code in one shape (what, why, what to do, where, its page), errors from below translated, the script's own frames shown and the full chain kept, written to a file and shown with `MIME_TRACEBACK=full` | new | dev1, phase 7l-b |
| [`results/`](results/) | a run's result, from a run and from a run's folder: each observation's declared outputs by name with their units, series with their time axes, windows by time, the flags raised over them, and the exports (a table, CSV, NPZ, a run folder's manifest) | new | dev1, phase 7k |
| [`nodes/`](nodes/) | the MADDENING nodes underneath the entries | ported; the fluid solver rewritten | dev1, phases 4 and 7 |
| [`core/`](core/) | the base node class, quaternion helpers, shapes; typed node state (phase 4a); `multibody/`, the rigid-body library (phase 7a1); `tables/`, the table evaluator (phase 7f) | ported | dev1, phases 4 and 7 |
| [`surrogates/`](surrogates/) | learned models that nodes evaluate: a surrogate as a type naming its evaluator (phase 7i), the generator of training data and its held-out protocol (phase 7i) | ported | dev1, phases 4 and 7 |
| [`data/`](data/) | small package data: surrogate weights, calibrations | ported | dev1, phase 4 |
| [`env/`](env/) | the environment a run is made in: each variant's hashed pins and the base images, `mime env check`, what the lock records of what no pin holds, XLA's cache directory, and the Modal script's generator | new | dev1, with the Modal template |
| [`assets/`](assets/) | large files referenced by sha256 and fetched on request: the cache, the one resolver, the wall tables' references; derived assets' derivation records (phase 7i) | new | dev1, before phase 7h |
| [`credentials/`](credentials/) | the token `mime push` will use, kept where a commit cannot publish it and shown nowhere | new | dev1, after the assets |
| [`materials/`](materials/) | the seam to OpenMaterica | new | dev1, phase 5 |
| [`scene/`](scene/) | the USD stage: frames, prim paths, geometry, the stage writer | new | dev1, phase 6 (landed) |
| [`control/`](control/) | the controller declaration and its modes: the callback and the chunked run (phase 7b1), the mode inside the compiled step (phase 7b2) | rewritten | dev1, phase 7 |
| [`mappings/`](mappings/) | transfers between discretisations | new | with the first such transfer |
| [`study/`](study/) | sweeps, fits and sensitivities over experiments | new | dev2 (sweeps), a1 (fits) |
| [`validity/`](validity/) | validity lines, watches, evidence, profiles | new | a1 |
| [`gradients/`](gradients/) | gradient statuses and their checks | new | declared from dev1; checked in a1 |
| [`viz/`](viz/) | recording over the scene; viewports | ported | when recording is needed |

Three modules sit beside the packages: `units.py` (SI quantities such as `mm(...)`, `Hz(...)`, `mT(...)`, checked by
dimension in phase 1 and turned into plain SI floats, landed in phase 2; the dimension markers an entry's settings
declare in their annotations, `Annotated[Quantity, Length]`, since phase 4a) and `drives.py` (command generators,
each a node whose numbers are `params` leaves with an exclusive edge into an apparatus's command: the constant drive
since phase 4b, the ramps in dev2) and, since phase 7h, `walls.py` (wall kinds: how a confining wall reaches a solver
of the flow; free space and the tube, the plane reserved).

The toy entries the build is tested on are in `mime.testing.toys`, so a lab's tests can use them too, with the toy
family, the family suites' control.

## Status

The rewrite moves through releases: `0.3.0.dev1`, `dev2`, `a1`, `a2`, then `0.3.0`. Each release is split into
phases when it is next. Every phase ends at a gate: a test that can fail, plus a control showing that it can. dev1's
phases are:

1. Skeleton and CI. **Done.**
2. The core (declarations, registry, both build phases, the canonical order), tested on toy entries with no physics.
   **Done.**
3. The saved form on those toys: save, load, rebuild, the same result; type checking in CI; units labels on every
   edge. **Done.**
4. The first experiment's physics, node by node: a magnetic helical robot swimming in a tube, compared with frozen
   outputs of the 0.2 code. In three parts: 4a, the infrastructure every physics entry uses (MADDENING 0.4.2, the
   Apache-2.0 licence, test durations on every CI lane, units declared in settings' annotations, typed node state,
   and the family suites' harness with the node walk), **done**; 4b, the body-side entries (the rigid inertial body
   with implicit resistance and confinement by projection, the tube, values declared in the experiment, the uniform
   rotating field and the constant drive, the magnetic torque, gravity, contact, the swim speed, the magnetic,
   gravity and contact families, and lag accounting), **done**; 4c1, drag (the surrogate's port with its fixes, the
   `Drag` entry and the hydrodynamic family), **done**; 4c2, the phase's gate, **done**: script A, built through
   the public API, against its frozen reference, with every expected difference declared before the run, the two
   graphs' lag structures compared, and stateful tests of the `Experiment` API; 4c3, three structural corrections,
   each a declared change measured against its prediction (the step solved for the change in velocity, the contact
   held in the implicit solve, the surrogate's resistance about the centre), **done**: every quantity of A, in
   float64 and in float32, the float32 phase lag included, holds its declared prediction within its tolerance.
   **Phase 4 is closed.**
5. Materials from OpenMaterica records. **Done**: records read at the experiment's conditions through
   `mime.materials`, OpenMaterica's verdict passed through (`met` builds, `not_met` is refused, `unknown` needs an
   acceptance with its reason), records pinned as files, a body's parts and the part resolver, and script A's record
   arm, each record switch's change predicted before the run and measured within its stated tolerance; the
   reproduction arm, through the seam, is phase 4's bit for bit (since 4c3, the arm 4c3 left, with each switch's
   change predicted again on 4c3's physics).
6. The scene: the USD stage writer and frames. **Done**, before phase 5 (the two are independent): `Scene.new`,
   the stage written beside the document and the lock, robot assets referenced by hash, bodies bound to their
   prims, the `mime:` schema for a magnet's direction, the frame proposals and the two frontier rules.
7. The same robot driven by a magnet on a robot arm, closed loop. In parts: phase 7a1, the rigid-body library
   (`mime.core.multibody`), **done**; phase 7a2, the kinematic apparatus chain (the arm and the motor as chain
   segments with their kinematic rungs, mounts, the point-dipole field and the force of its gradient,
   `Magnetic.Force`), **done**; phase 7b1, the controller's declaration and its callback mode, **done**; phase 7b2,
   the controller inside the compiled step, **done**; phase 7a3, the dynamic arm and motor, their loads and the
   magnetic laws' reactions on the source (MIME-ANO-103), **done**; phase 7a4, mounted mass and spinners on dynamic
   hosts (MIME-ANO-122), **done**; then the experiment's gate.
8. The `0.3.0.dev1` release.

Nothing here is stable until 0.3.0. Even then, `mime.nodes` stays outside the 0.3.x stability promise.
