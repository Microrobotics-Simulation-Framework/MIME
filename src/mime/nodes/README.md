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


class BodyNode(SimulationNode):
    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        held, given = typed_view(BodyState, state), typed_view(BodyInputs, boundary_inputs)
        v = held["v"] + dt * given.get("load", jnp.zeros(3)) / mass
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

Empty. The first experiment's nodes arrive in dev1's phase 4 (4b and 4c), the arm and motor in phase 7, and the
rewritten flow solver in a1. Phase 4a set the typed node state convention above, on the toy nodes.

Phases and releases are explained in the [package manifest](../README.md#status).
