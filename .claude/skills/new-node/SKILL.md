# New Node Skill — MIME

This skill guides the agent through adding a new `MimeNode` subclass. Follow every step — MIME nodes are compliance artifacts.

## Trigger

When the user asks to add a new physics node, or invokes `/new-node`.

---

## Step 0 — Classify the node

Decide the node's role:

| Role | Directory | Requires |
|------|-----------|----------|
| `external_apparatus` | `src/mime/nodes/actuation/` | `ActuationMeta` |
| `robot_body` | `src/mime/nodes/robot/` | `BiocompatibilityMeta` |
| `environment` | `src/mime/nodes/environment/` | — |
| `sensing` | `src/mime/nodes/sensing/` | `SensingMeta` |
| `therapeutic` | `src/mime/nodes/therapeutic/` | `TherapeuticMeta` |

Check `ARCHITECTURE_PLAN.md` §5 for the current node hierarchy and assign the next available `MIME-NODE-XXX` ID.

---

## Step 1 — Implement the node

```python
"""MyNode -- one-line description."""

import jax.numpy as jnp

from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import MimeNodeMeta, NodeRole, AnatomicalRegimeMeta, ...


@stability(StabilityLevel.EXPERIMENTAL)
class MyNode(MimeNode):
    """NumPy-style docstring."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-XXX",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="One-line description",
        governing_equations=r"...",
        discretization="...",
        assumptions=("...",),
        limitations=("...",),
        validated_regimes=(
            ValidatedRegime("param", min_val, max_val, "units"),
        ),
        references=(
            Reference("AuthorYear", "Description"),
        ),
        hazard_hints=("...",),
        implementation_map={
            "term description": "mime.nodes.subpackage.MyNode.update",
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ROBOT_BODY,
        anatomical_regimes=(
            AnatomicalRegimeMeta(compartment=..., ...),
        ),
        biocompatibility=...,  # Required if robot_body
    )

    def __init__(self, name, timestep, **kwargs):
        super().__init__(name, timestep, **kwargs)

    def initial_state(self):
        return {"field": jnp.zeros(...)}

    def update(self, state, boundary_inputs, dt):
        # Pure JAX operations only
        ...
        return new_state
```

---

## Step 2 — Write the algorithm guide

Copy `docs/algorithm_guide/nodes/_template.md` to `docs/algorithm_guide/nodes/my_node.md`. Fill in every section — no empty tables or placeholder text.

Declare **Verification Mode**: Mode 1 (Wrapping) or Mode 2 (Independent).

---

## Step 3 — Write tests

**Unit test** (`tests/nodes/test_my_node.py`):
- Test each public method
- Verify JAX-traceability: `jax.jit(node.update)(state, {}, dt)` must work

**Verification benchmark** (`tests/verification/test_my_node.py`):
```python
from maddening.core.compliance.validation import verification_benchmark

@verification_benchmark(
    benchmark_id="MIME-VER-XXX",
    description="Description",
    node_class="MyNode",
    reference="AuthorYear",
)
def test_my_analytical_comparison():
    ...
```

**Integration test**: test within a `GraphManager` with edges to other nodes.

---

## Step 4 — Add bibliography entries

Add BibTeX entries for any cited references to `docs/bibliography.bib`.

---

## Step 5 — Run consistency check

```python
node = MyNode("test", 0.001)
errors = node.validate_mime_consistency()
assert errors == [], errors
```

---

## Step 6 — Document known limitations

If the node has known failure modes, add entries to `docs/validation/known_anomalies.yaml` using the **new-anomaly** skill.

---

## Quick checklist

```markdown
- [ ] `MimeNode` subclass with `initial_state()` and `update()`
- [ ] `update()` is JAX-traceable
- [ ] `meta` ClassVar set (MIME-NODE-* ID)
- [ ] `mime_meta` ClassVar set with correct role
- [ ] `@stability(StabilityLevel.EXPERIMENTAL)` applied
- [ ] Role-specific metadata present (BiocompatibilityMeta, ActuationMeta, etc.)
- [ ] `validate_mime_consistency()` returns no errors
- [ ] Algorithm guide in docs/algorithm_guide/nodes/
- [ ] Implementation Mapping complete
- [ ] At least one @verification_benchmark (MIME-VER-*)
- [ ] Unit tests + integration test
- [ ] Known limitations in known_anomalies.yaml
- [ ] Bibliography entries for cited references
- [ ] All CI checks pass
```
