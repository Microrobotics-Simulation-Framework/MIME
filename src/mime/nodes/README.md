# `mime.nodes`

**The MADDENING nodes underneath the entries.**

## What goes here

- **`body/`**, **`actuation/`**, **`magnetic/`**, **`hydro/`**, **`acoustic/`**.
- **`control/`**: a controller inside the compiled step, its law's step at its own rate, and the delay line of its
  latency. Its step is the controller's law, made at the build from the experiment's declaration, so this node is
  the one not rebuilt from its `params` alone.
- **`flow/fvm/`**, the finite-volume flow solver, rewritten for 0.3.

## How nodes arrive

Nodes are ported one at a time from the 0.2 line, each behind the new contracts. Each one must pass:
- MADDENING's `verify_node` battery, and `verify_node_order` where it declares an order;
- a small physics oracle;
- a golden output;
- a differential test against frozen outputs of the 0.2 code.

## Typed node state

Every node types its state and its inputs, statically, in the same way.

- **Write the step as `MimeNode.advance(p, state, boundary_inputs, dt)`**, `p` the numeric constants with the graph's
  live `params` over them; `update` runs it compiled. A toy or lab node may override `update` itself.
- **Keep MADDENING's signatures.** `SimulationNode` annotates `initial_state`, `update`, `compute_boundary_fluxes` and
  `boundary_input_spec` with bare `dict`. An override may not narrow a parameter: `state: BodyState` in `update` is
  refused by pyright's override rule (a plain `dict` is not a TypedDict). So a node writes `state: NodeFields`,
  `boundary_inputs: Mapping[str, Array]` and `-> NodeFields` (`NodeFields` is `dict[str, Array]`, from `mime.core`).
- **One closed TypedDict for the state and one for the inputs**, named after the node (`ToyBodyState`,
  `ToyBodyInputs`), from `typing_extensions` with `closed=True`. Every value is a jaxtyping array giving its shape,
  `Float[Array, "3"]`, `Float[Array, ""]` for a scalar; values are `ReadOnly`, and an input a node may not be given is
  `NotRequired` (read it with `.get`). Several nodes with one output share its TypedDict (`ToyLoad`).
- **Fields named at construction**, one per registered body or reader slot (`force_<body>`, `points_<slot>`), are
  the TypedDict's `extra_items` (`extra_items=ReadOnly[Float[Array, "3"]]`), beside the fields it names.
- **Take a view inside, hand back a dict.** `typed_view(BodyState, state)` reads the state, and
  `node_fields(BodyState(x=..., v=...))` returns it. Constructing the TypedDict is the check: a missing field, an
  extra one, or a value that is not an array is a type error, and `typed_view` refuses a TypedDict that is not closed
  or holds anything but arrays. Both are casts: nothing is checked at run time, and no run-time type-checking
  dependency is added.
- **`params`** stays `Mapping[str, Any] | None`, read through the node's own helper.

```python
class BodyState(TypedDict, closed=True):
    x: ReadOnly[Float[Array, "3"]]
    v: ReadOnly[Float[Array, "3"]]


class BodyInputs(TypedDict, closed=True):
    load: NotRequired[ReadOnly[Float[Array, "3"]]]


class BodyNode(MimeNode):
    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held, given = typed_view(BodyState, state), typed_view(BodyInputs, boundary_inputs)
        v = held["v"] + dt * given.get("load", jnp.zeros(3)) / p["mass"]
        return node_fields(BodyState(x=held["x"] + dt * v, v=v))
```

The toy nodes (`mime.testing.toys`) and the lab fixture's node follow it. `tests/unit/test_node_fields.py` holds each
toy node's TypedDicts to the fields it holds and the inputs it declares at run time, and `tests/typing/` holds the
static controls: a missing or extra field, a view of a TypedDict that is not closed, a write into a view, and the
override that narrows a parameter are each refused.

## Boundaries

- Imports nothing above it, so every node also runs on MADDENING alone.
- Importable, but **outside 0.3.x's stability promise**.

## Status

Since phase 4b, the body side of the first experiment, each ported from the 0.2 line behind the new contracts and
writing its step as `MimeNode.advance`:
- **`body/`**: `RigidInertialNode` (implicit resistance, the exponential turn, the confinement's projection),
  `GravityNode` (weight, buoyancy and its couple), `SwimSpeedNode`;
- **`magnetic/`**: `MagneticTorqueNode`;
- **`actuation/`**: `UniformRotatingNode` and `ConstantNode`, a drive's.

Each passes `verify_node` in the node walk, has a golden output, and (but for the swim speed and the drive) a
differential against the 0.2 port's node on fixed inputs (`tests/differential/`).

Since phase 4c1, **`hydro/`**: `SurrogateDragNode`, the drag surrogate's resistance with its lubrication correction
in one node (the 0.2 line's `MLPResistanceNode` and `LubricationCorrectionNode`), the tube's axis, a point on it and
its radius as parameters, the weights as static arrays. Its differential against the port agrees within 1e-12 of
each block's largest magnitude off the axis (2.8e-15 at worst); on the axis the frame differs, as declared, and the
port's own frame there turned out to follow the sign of zero of the position. The arm and motor come in phase 7,
and the rewritten flow solver in a1. Phase 4a set the typed node state convention above.

Since phase 7a2, `actuation/`'s chain segment nodes, `KinematicArmNode` and `KinematicMotorNode` (each publishing
`frame_poses`, placed on the row of its mount's frame poses it reads, `MountedNode`), `PointDipoleNode`, and
`magnetic/`'s `MagneticForceNode`, each with `NodeMeta` and its symbol table.

Since phase 7a3, `actuation/`'s dynamic segment nodes, `DynamicArmNode` (joint torques and the loads on its frames
in, the mass matrix and the bias from `mime.core.multibody`, implicit damping, rigid stops and Coulomb friction as
impulses through the step's system by projected Gauss-Seidel sweeps) and `DynamicMotorNode` (a torque in, its rotor's inertia and damping, and the wrench its stator passes to
its mount, `reaction`), each first order in time; and `SourceReactionNode`, a law's wrench on a body as its reaction
on the field's source, moved to the frame the source is mounted on. Each with `NodeMeta` and its symbol table.

Since phase 7a4, every segment node publishes `frame_angular_velocities` beside `frame_poses`, and reads its mount's
as `mount_rates`, this step's (the kinematic arm's from its joints' rates over the step, the dynamic arm's from its
forward pass; a kinematic node computes them only where the build finds a reader, its `rates`); the arm nodes take
their commands by joint kind (`joint_torques` and `joint_forces`, `joint_positions` and `joint_displacements`); and
`actuation/_spinner.py`, the spinner, which `DynamicMotorNode` is: the rotor's absolute spin about its axis, the
host's angular velocity in; out, a step late, the drive's and bearing's reactions, the rotor's momentum `J Omega e`
and its half difference across its axis, now and a step on. `DynamicArmNode` takes them on its frames
(`link_momentum`, `link_inertia`): the gyroscopic torque by a Cayley step at its own velocity, which does no work, and
the turning inertia in its mass matrix.

Since phase 7g, **`hydro/mrs/`**: the method of regularised Stokeslets, ported from the earlier line's
`stokeslet/` behind its contracts. A library (the matrix of a body's samples in point forces, symmetric whatever the
quadrature weights; the solve by the linear-solve rule; the 6 x 6 resistance about a centre), the wall corrections a
confining wall adds to the matrix, registered by name (`free_space` adds none), and `MRSDragNode`, which samples a
body's hull kind at each step, places the samples by the pose, adds the wall's correction and solves, checkpointed so
that the reverse mode keeps no matrix between its passes. It holds no geometry's formula: the samples are the kind's.

Since phase 7h, the wall corrections are the MRS's own use of a wall kind (`mime.walls`), registered by the kind's
name and version, each with the constants it prepares at build and the reads it checks when planned; the node
publishes `wall_accepted`, whether every read its correction made was inside its data. The straight circular tube's
correction, its image system read from a table of the Fourier-Bessel series in the tube's own units (the angle and
the axial offset mirrored past their symmetric ends so that the read is continuously differentiable there), is in
place, and the tube's wall kind, `tube`, registers it on the corrected table of the unit tube's image system, the
shipped asset `unit_tube_R0.997` (MIME-ANO-117), which `mime assets fetch unit_tube_R0.997` stores in the cache: the
table is read when an experiment with the wall kind is planned, never downloaded there, and a missing one is refused by
name (`asset.missing`). The tests of the wall kind need the asset in the cache; CI stores it once per job.

Phases and releases are explained in the [package manifest](../README.md#status).
