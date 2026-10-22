# `mime`: the package manifest

MIME (MIcrorobotics Multiphysics Engine) simulates field-actuated microrobots, magnetic and acoustic, moving through
confined, low-Reynolds-number flows such as vessels and channels. Gradients run through the whole simulation, so the
same experiment that predicts a result can also be fitted to measurements.

This package is **MIME 0.3, a full rewrite**. Every folder exists and says what it is for, and the code arrives phase
by phase (see [Status](#status)): so far the core that turns a declared experiment into a graph, tested on toy entries
with no physics. The 0.2 line is preserved at the git tag `pre-0.3`.

MIME stands on two libraries:
- [MADDENING](https://microrobotica.org/maddening/), the coupling engine that runs the graph. MIME 0.3 targets
  MADDENING 0.4.x.
- [OpenMaterica](https://github.com/Microrobotics-Simulation-Framework/OpenMaterica), the materials library, which
  gives every material value a record, a source and a pin.

The scene is a USD stage ([OpenUSD](https://openusd.org)).

## The idea

A script **declares an experiment**: the things in it, how each one moves, the fields that act on them, the physical
laws between them, what to observe, and an optional controller. Nothing is built while the script runs.

`build()` checks the whole declaration first. If anything is wrong, it refuses with every reason at once, each with
what to do about it. Only then does it construct a MADDENING graph.

A built experiment saves as a document plus a lock file. Anyone can reload and rerun that pair without executing the
original script. **The experiment, not the script, is the deliverable.**

```python
exp = Experiment("fl9_quarter_inch_water", scene=scene, profile="research", dt=..., ...)
tube  = exp.add_structure(Structure("tube", geometry=Tube(...), material=record(...)))
water = exp.add_medium(Medium("water", fills=tube.lumen, material=record("water")))
umr   = exp.add_body(Body("umr", prim=..., dynamics=RigidInertial(), initial=State(...)))
field = exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotatingField(...)}, ...))

exp.attach(Drag.Surrogate(weights=...), body=umr, medium=water)
exp.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
exp.attach(Gravity(), body=umr, medium=water)
exp.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
exp.observe(SwimSpeed(umr, along=tube.axis))

built = exp.build()        # checks everything, then constructs the graph; writes the stage, document and lock
built.run(duration=s(5))
```

This is a sketch of the intended API. The declaration and build machinery works today; the physics it names
(`Drag`, `Magnetic.Torque`, the scene and materials) arrives in later phases.

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
| [`saved/`](saved/) | the document, the lock, loading, `verify`; later the experiment folder | new | dev1, phase 3 |
| [`testing/`](testing/) | harnesses labs call from their own tests | new | dev1, phases 1 to 4 |
| [`cli/`](cli/) | the `mime` command, thin over the Python API | new | dev1, phase 3 |
| [`dynamics/`](dynamics/) | how bodies, flows and apparatus evolve | new | dev1, phases 4 and 7 |
| [`fields/`](fields/) | field providers | new | dev1, phases 4 and 7 |
| [`interactions/`](interactions/) | physical laws, one module per family | new | dev1, phase 4 |
| [`observations/`](observations/) | what an experiment measures | new | dev1, phase 4 |
| [`nodes/`](nodes/) | the MADDENING nodes underneath the entries | ported; the fluid solver rewritten | dev1, phases 4 and 7 |
| [`core/`](core/) | the base node class, node metadata, quaternion helpers | ported | dev1, phase 4 |
| [`surrogates/`](surrogates/) | learned models that nodes evaluate | ported | dev1, phase 4 |
| [`data/`](data/) | small package data: surrogate weights, calibrations | ported | dev1, phase 4 |
| [`materials/`](materials/) | the seam to OpenMaterica | new | dev1, phase 5 |
| [`scene/`](scene/) | the USD stage: frames, prim paths, geometry, the stage writer | new | dev1, phase 6 |
| [`control/`](control/) | the controller declaration and its modes; arm kinematics | rewritten; kinematics ported | dev1, phase 7 |
| [`mappings/`](mappings/) | transfers between discretisations | new | with the first such transfer |
| [`study/`](study/) | sweeps, fits and sensitivities over experiments | new | dev2 (sweeps), a1 (fits) |
| [`validity/`](validity/) | validity lines, watches, evidence, profiles | new | a1 |
| [`gradients/`](gradients/) | gradient statuses and their checks | new | declared from dev1; checked in a1 |
| [`viz/`](viz/) | recording over the scene; viewports | ported | when recording is needed |

Two modules sit beside the packages: `units.py` (SI quantities such as `mm(...)`, `Hz(...)`, `mT(...)`, checked by
dimension in phase 1 and turned into plain SI floats; landed in phase 2) and `drives.py` (command generators such as
a constant or a ramp; planned for phase 4).

The toy entries the build is tested on are in `mime.testing.toys`, so a lab's tests can use them too.

## Status

The rewrite moves through releases: `0.3.0.dev1`, `dev2`, `a1`, `a2`, then `0.3.0`. Each release is split into
phases when it is next. Every phase ends at a gate: a test that can fail, plus a control showing that it can. dev1's
phases are:

1. Skeleton and CI. **Done.**
2. The core (declarations, registry, both build phases, the canonical order), tested on toy entries with no physics.
   **Done.**
3. The saved form on those toys: save, load, rebuild, the same result.
4. The first experiment's physics, node by node: a magnetic helical robot swimming in a tube, compared with frozen
   outputs of the 0.2 code.
5. Materials from OpenMaterica records.
6. The scene: the USD stage writer and frames.
7. The same robot driven by a magnet on a robot arm, closed loop.
8. The `0.3.0.dev1` release.

Nothing here is stable until 0.3.0. Even then, `mime.nodes` stays outside the 0.3.x stability promise.
