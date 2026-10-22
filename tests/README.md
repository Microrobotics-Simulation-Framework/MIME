# `tests/`

**MIME's own tests.** Users write theirs with the harnesses in [`mime.testing`](../src/mime/testing/).

## What is here now

- `test_package.py`: the package layout matches the design. Every subpackage imports and states its role, the version
  is the distribution's, and `mime.testing` names its extra when Hypothesis is missing.
- `test_ci_lane.py`: each CI lane installed what it claims. It reads the distributions' own metadata. For example,
  the OpenMaterica-`main` lane must have OpenMaterica from git, not PyPI. A lane that doesn't name itself fails, so no
  lane can skip the check. The test also runs its own control: claims meant for other lanes must fail in this one.
- `test_canary.py`: CI's planted failures. A manual run with `plant_failure` set must turn every lane red, which
  proves each lane really runs its tests; one with `plant_slow_tests` set makes two tests take 21 s and 6 s, which the
  time budget must fail and warn on.
- `conftest.py`: each test's own CPU time and its children's, written into the JUnit XML for the durations report.
- `test_report_test_durations.py` and `duration_allowlist.txt`: the time budget's gate (see *The time budget*).
- `test_licence.py`: the licence is the Apache-2.0 text, NOTICE names the copyright and the 0.2 line's licence, and
  no other tracked file but the changelog names that licence, with a planted mention as the control.
- `test_collection.py`: every test file under `tests/` is collected. pytest skips a folder named `build` by default,
  so `pyproject.toml` lists the folders to skip without it, and this test fails if one is ever left out again.
- `test_lab_fixture.py`: an entry registered from outside MIME, by the [lab fixture package](../lab_fixture/), builds
  beside MIME's own. Every CI lane installs that package, and the test fails under CI without it.
- `test_packaging.py`: the built wheel ships `py.typed`, the `mime` command and the toy namespace's entry point, and
  declares its licence by SPDX expression with both licence files (PEP 639).
- `test_typing_check.py`: the typing gate (`scripts/typing_check.py`) refuses each kind of run it cannot trust, fails
  on an error, and passes a clean run; CI's `typecheck` job runs the gate itself, with a planted type error as its
  control. `typing/` holds the consumer snippet that job type-checks against the built wheel.
- `unit/`: units (with the dimension markers and the settings an entry reads in SI), the vocabulary, things and
  handles, the registry (with the dimensions settings declare), and the toy nodes' typed state
  (`test_node_fields.py`: each toy node's TypedDicts name the fields it holds and the inputs it declares).
- `families/`: the family suites' control. `test_toy_family.py`: the toy drag's two rungs conform to the toy family,
  the planted sign error fails the sign case and nothing else, and a law planted with each other kind of defect fails
  its case (frame, gradients, material reach, units, nodes, swap, oracle, and equal and opposite on a family of
  two-body laws). `test_node_battery.py`: a declared order of accuracy is measured, and a wrong claim or one with no
  ladder fails.
- `build/`, the graph-invariant layer, on the toy entries of `mime.testing.toys` (run in float64, scoped per module):
  - `test_canonical_order.py`: all 48 orders of the coupled toy's calls give one graph and a history bitwise equal to
    the same toy assembled by hand (`hand_built.py`); its controls show the hand-built graph is order-sensitive and
    that a build skipping the canonical order is not order-invariant;
  - `test_order_invariance.py`: the Hypothesis property (any order of the full toy's nine calls gives the same graph
    text and a bitwise-identical run), and its control, in which Hypothesis must find an order that breaks it;
  - `test_post_build_checks.py`: a mutation per post-build check, each refused by its own check;
  - `test_snapshots.py`: the toys' graphs against the golden text in `build/golden/`;
  - `test_plan.py` and `test_builders.py`: phase 1's refusals and the restricted builders;
  - `test_maddening_guards.py`: the MADDENING 0.4.x behaviours the build works around, pinned, so the
    MADDENING-`main` lane shows when one changes (the private step function the probe traces among them);
  - `test_compiled_step.py`: a run is one compiled scan; a host callback in the step is refused naming its node; and
    a leaf baked into the trace as a constant is caught. Each with its control;
  - `test_report_warnings.py`: MADDENING's warnings in the report, and a units warning as a refusal.
- `saved/`, the saved form (float64, as `build/`):
  - `test_registry_walk.py`: every registered entry and kind of thing has a round-trip case in `cases.py`, and every
    case round-trips through a fresh process with the same derived hash and bitwise-identical first steps; its
    controls remove each load-bearing case and add an entry with none. Each case's parameter leaves reach its step;
  - `test_node_walk.py`: every registered entry has a node case in `node_cases.py`, and every node it builds passes
    MADDENING's `verify_node` (and `verify_node_order` on each declared order); its controls remove each case and add
    an entry with none;
  - `test_document.py`: what the document holds, that order and formatting leave it and its hash alone, that a
    Python tag is refused and constructs nothing (with an unsafe loader as the control that the payload is live);
  - `test_load.py`: an unknown entry and another version refused with the line to install, an edited document
    refused naming what changed, a static array's contents changing the derived hash where MADDENING's description
    cannot see it, the factory guard, reports across environments, `mime verify`, and what `save` refuses;
  - `test_schema.py`: the TypedDicts and the loader's run-time check in step, on instances Hypothesis generates from
    the annotations.

## The planned layout

| Folder | What it will hold |
|---|---|
| `unit/` | grows with the part resolver and frames |
| `build/` | grows with each new kind of construction (mapped edges, multi-rate physics) |
| `families/` | grows a conformance suite per family as its entries arrive: hydrodynamic, magnetic, acoustic radiation, gravity, contact |
| `differential/` | comparisons with frozen outputs of the 0.2 code, with tolerances declared before the first run |
| `oracles/` | analytic solutions; small physics oracles per interaction; software oracles, one lane each |
| `acceptance/` | comparisons with measurements, each stating what it bounds and what it estimates |
| `gradients/` | gradient checks that rebuild the graph, with planted failures as controls |
| `saved/` | grows with every entry and every feature of the document (materials, scene, controller) |
| `scene/` | the stage writer, frames, orientation lines |

## Markers and lanes

- `slow` tests are too expensive for CI's default lane. Run them with `-m "slow or not slow"`.
- `gpu` tests need a CUDA device and **skip without one**. A skip looks like a pass in a summary line, so count skips.
- CI's lanes are: the default (Python 3.12 to 3.14, released dependencies), OpenMaterica `main`, MADDENING `main`, and
  a scheduled run against the newest usd-core. The last three report without blocking. Beside them, `typecheck`
  blocks: pyright, pinned, strict over `src/mime`, at zero errors.

Run the tests locally with `JAX_PLATFORMS=cpu python -m pytest`, after `uv pip install --no-deps ./lab_fixture`
(without it, `test_lab_fixture.py` skips outside CI). CI runs them on pytest-xdist workers
(`-n auto --dist worksteal`); add the same options locally to do the same. The slowest test is the order-invariance
property, about 45 s on a laptop CPU: Hypothesis's default 100 examples, each of which builds, compiles and runs a
fresh graph.

## The time budget

MADDENING's, read from each test's wall time (setup, call and teardown) in the JUnit XML every CI lane writes:

| Time on CI | What it means |
|---|---|
| up to 1 s | Fine. |
| 1 to 5 s | The watch list: optimise it when you are in the file. |
| over 5 s | Mark it `slow`, make it much faster, or keep it on every push by listing it in `duration_allowlist.txt` as `<node id> # kept: <why>`. |

An unlisted test over 5 s gets a warning annotation on the run. An unlisted test over 20 s is re-run once, alone, and
fails the job if it is over 20 s again. An allowlisted test over 20 s passes with a warning, since the allowlist has
no ceiling. Each lane's `Test durations` step (`scripts/report_test_durations.py`, a copy of MADDENING's) writes the
bands, the slowest tests and files, and the totals of wall time and CPU time to the run page. `conftest.py` records
each test's own CPU time and that of the processes it starts, since the xdist workers share the runner's cores and a
test's wall time partly measures the others.

**A property test is never shrunk and never marked `slow` to meet the budget**: `slow` runs in no CI job, so a
slow-marked property is checked nowhere. It goes on the allowlist with its reason instead.

The controls are in `test_report_test_durations.py` (the gate on hand-written reports, and the check that every lane
writes its report and lets the gate fail the job, with a mutation per way it could stop) and in a planted CI run: the
`plant_slow_tests` input makes the two planted tests in `test_canary.py` take 21 s and 6 s, and every lane must fail
on the first and warn on the second.
