# `mime.nodes`

**The MADDENING nodes underneath the entries.**

## What goes here

- **`body/`**, **`actuation/`**, **`magnetic/`**, **`hydro/`**, **`acoustic/`**.
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
- **`actuation/`**: `UniformRotatingFieldNode` and `ConstantNode`, a drive's.

Each passes `verify_node` in the node walk, has a golden output, and (but for the swim speed and the drive) a
differential against the 0.2 port's node on fixed inputs (`tests/differential/`).

Since phase 4c1, **`hydro/`**: `SurrogateDragNode`, the drag surrogate's resistance with its lubrication correction
in one node (the 0.2 line's `MLPResistanceNode` and `LubricationCorrectionNode`), the tube's axis, a point on it and
its radius as parameters, the weights as static arrays. Its differential against the port agrees within 1e-12 of
each block's largest magnitude off the axis (2.8e-15 at worst); on the axis the frame differs, as declared, and the
port's own frame there turned out to follow the sign of zero of the position. The arm and motor come in phase 7,
and the rewritten flow solver in a1. Phase 4a set the typed node state convention above.

Phases and releases are explained in the [package manifest](../README.md#status).
