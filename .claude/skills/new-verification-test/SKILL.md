# New Verification Test Skill — MIME

This skill guides the agent through adding a new verification test. Verification tests are formal compliance artifacts registered via MADDENING's `@verification_benchmark` decorator.

## Trigger

When the user asks to add a verification test, or invokes `/new-verification-test`.

---

## Unit test vs. verification test

| | Unit test | Verification test |
|---|---|---|
| Location | `tests/nodes/test_*.py` | `tests/verification/test_*.py` |
| Registration | None | `@verification_benchmark` decorator |
| Purpose | Test implementation correctness | Prove a physics-relevant property holds |
| Linked to | Nothing formal | A `NodeMeta.algorithm_id` and optionally an anomaly fix |

Write a verification test when testing:
- Analytical solution comparison (drag force vs. Stokes law)
- Experimental data comparison (trajectory vs. published data)
- Conservation properties (mass, energy, momentum)
- Convergence order verification
- Regression against a known fixed anomaly

---

## Step 1 — Assign the benchmark ID

Check existing `MIME-VER-*` IDs in `tests/verification/` and assign the next sequential one.

---

## Step 2 — Write the test

```python
# tests/verification/test_my_node.py

from maddening.core.compliance.validation import verification_benchmark

@verification_benchmark(
    benchmark_id="MIME-VER-XXX",
    node_class="MyNode",
    benchmark_type=BenchmarkType.ANALYTICAL,  # or CONVERGENCE, REFERENCE_DATA, REGRESSION
    description="One sentence describing what this proves",
    reference="AuthorYear",
    tolerance=1e-3,
)
def test_my_analytical_comparison():
    """Compare MyNode output against analytical solution.

    Acceptance criterion: relative error < 1e-3.
    Reference: [@AuthorYear] in docs/bibliography.bib.
    """
    # Arrange: set up node, graph, initial conditions
    # Act: run simulation
    # Assert: compare to analytical/reference solution
    assert relative_error < 1e-3
```

---

## Step 3 — Update documentation

Add the benchmark to the relevant algorithm guide's Verification Evidence section:

```markdown
## Verification Evidence

- `MIME-VER-XXX`: [description] — `tests/verification/test_my_node.py`
```

---

## Step 4 — Link to anomaly resolution (if applicable)

If this test proves a previously open anomaly is fixed:
1. Update the anomaly's `resolution_status` to `"fixed"` in `known_anomalies.yaml`
2. Add CHANGELOG entry under `### Fixed`

---

## Step 5 — Run the test

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/verification/test_my_node.py -v
```

---

## Quick checklist

```markdown
- [ ] Next MIME-VER-XXX ID assigned
- [ ] @verification_benchmark decorator with all fields
- [ ] BenchmarkType chosen (ANALYTICAL, CONVERGENCE, REFERENCE_DATA, REGRESSION)
- [ ] Clear acceptance criterion documented in docstring
- [ ] Reference cited ([@Key] in bibliography.bib)
- [ ] Algorithm guide Verification Evidence section updated
- [ ] If fixing anomaly: resolution_status updated
- [ ] Test runs and passes (or fails with clear message if intentionally failing)
```
