# Compile-time edge validation

MADDENING v0.2 added a graph-edge consistency check. Every edge wires one
node's output field to another node's `BoundaryInputSpec`; the check flags
edges whose two ends disagree.

## What it checks

For each edge, `GraphManager.validate()` compares the source field against
the target node's `boundary_input_spec()` entry on three axes:

* **shape** — source array shape vs. `BoundaryInputSpec.shape`.
* **dtype** — source array dtype vs. `BoundaryInputSpec.dtype`.
* **units** — the edge's declared `source_units` / `target_units` vs. the
  spec's `expected_units`.

Shape and dtype checks are skipped when the edge carries a `transform`
(which may legitimately reshape or recast) or when the spec leaves a
dimension symbolic. Unit checks are advisory only.

## How it runs

```python
issues = gm.validate()        # returns a list of strings
```

Edge issues come back as strings prefixed `WARNING[shape]`,
`WARNING[dtype]`, or `WARNING[units]`. `validate()` is also run
automatically inside `gm.compile()`, which routes each prefix to a typed
warning:

| Prefix | Warning class |
|---|---|
| `WARNING[shape]` | `ShapeMismatchWarning` |
| `WARNING[dtype]` | `DtypeMismatchWarning` |
| `WARNING[units]` | `UnitMismatchWarning` |

All three subclass `EdgeValidationWarning` (in `maddening/warnings.py`),
so a caller can `warnings.catch_warnings()` on the base class to handle
them uniformly. In v0.2 these are warnings; the v0.2.1 plan flips shape
and dtype to hard `EdgeValidationError`s — unit mismatches stay advisory
permanently, since units are documentation, not contract.

## How MIME stays clean

MIME's representative experiment graphs are pinned edge-validation-clean
(v0.2 fit-up §4) by `tests/verification/test_edge_validation.py`. The test
builds each experiment graph, calls `validate()` directly, and asserts no
issue carries an edge prefix:

```python
_EDGE_ISSUE_PREFIXES = ("WARNING[shape]", "WARNING[dtype]", "WARNING[units]")
edge_issues = [i for i in gm.validate() if i.startswith(_EDGE_ISSUE_PREFIXES)]
assert not edge_issues
```

It calls `validate()` rather than `compile()` because `validate()` *is* the
edge check — skipping the JIT keeps the test cheap. LBM/FVM graphs are
built at reduced resolution; edge-spec consistency does not depend on
lattice size. A new node that introduces a mismatched edge fails here
loudly, instead of letting MADDENING v0.2.1 escalate it to a hard error.

## Adding a custom node

To keep `validate()` clean for a new node:

* Declare a complete `boundary_input_spec()` — give every boundary input
  a concrete `shape`, `dtype`, and an `expected_units` tag where a
  physical unit applies.
* Make sure each edge feeding the node produces a field of the matching
  shape and dtype, or attach a `transform` that reconciles them.
* Add the node's experiment to `test_edge_validation.py` (register a
  builder) and confirm `gm.validate()` reports no edge issue.
