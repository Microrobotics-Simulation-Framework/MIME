# `tests/`

**MIME's own tests.** Users write theirs with the harnesses in [`mime.testing`](../src/mime/testing/).

## What is here now

- `test_package.py`: the package layout matches the design. Every subpackage imports and states its role, the version
  is the distribution's, and `mime.testing` names its extra when Hypothesis is missing.
- `test_ci_lane.py`: each CI lane installed what it claims. It reads the distributions' own metadata. For example,
  the OpenMaterica-`main` lane must have OpenMaterica from git, not PyPI. A lane that doesn't name itself fails, so no
  lane can skip the check. The test also runs its own control: claims meant for other lanes must fail in this one.
- `test_ci_wiring.py`: CI runs the checks it must, read from `.github/workflows/`. Each job name v0.3's branch
  protection requires is made by one job, of a workflow run on a pull request into v0.3 and a push to it, and skipped
  by nothing but the schedule; the default lane's `mime docs check --base origin/v0.3` and the `confidential` job's
  scan, with the key's secret, `--messages`, `--diffs` and `--ref-name`, keep their exit status, read by running each
  step's script in bash with its commands stubbed. Each control plants one break in a copy of the workflows.
- `test_ci_docs_artifact.py` (N6): the workflow builds the documentation pages, scans the built directory and uploads
  it, in that order, in one job, over one directory; the scan's failure fails the job; the upload runs only after
  every earlier step succeeded; and the scan step's own script, run in bash on a directory, exits non-zero on a planted
  private path. Each fault planted in a copy of the workflow (the scan moved after the upload, removed, in a second job,
  under `continue-on-error`, `|| true`, `if: false`, given no directory; `always()` on the upload) is reported by name.
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
  (`test_node_fields.py`: each toy node's TypedDicts name the fields it holds and the inputs it declares), and MIME's
  nodes' (`test_mime_node_fields.py`, which also requires every input to declare its units); the quaternion helpers
  (`test_quaternion.py`: the rotation's exponential at rest is finite with finite derivatives, and so is a body
  stepped at rest, with 0.2's increment as the control that gives NaN); shapes and the tube (`test_geometry.py`: the
  cylinder's distance and normal by their closed forms, finite with finite derivatives on its axis; its derivatives
  against central differences off the axis, at a resolution chosen per draw, with a derivative planted off by two
  and a half times that as the control; on any axis, within its resolution, the values and derivatives its record states, with the
  resolution removed as the control; the lumen as a region with its axis and a point on it); the drag on its tube's
  axis in float32 (`test_drag_axis.py`: finite value and derivative on the axis and 1e-30 m from it, the 0.2 line's
  formula giving NaN as the control).
- `test_packaged_data.py`: the surrogate's weights are committed, ignored by no `.gitignore` rule, load through the
  loader and match the pinned sha256, and so are the committed reference arrays; `test_packaging.py` checks the wheel
  and the sdist ship them. `test_hypothesis_profile.py`: the profile `conftest.py` loads prints the blob that replays a failure, and changes nothing else from what Hypothesis would load (its `ci` profile
  on CI).
- `unit/test_multibody_tree.py` and `unit/test_urdf_reader.py`, the rigid-body library's contracts: the tree's
  refusals, its articulation (depth-first numbering, fixed joints welded, inertials combined), every algorithm
  keeping the joint positions' dtype, finite with finite derivatives at zero rotation, compiled and mapped over a
  batch, and refusing a tree with a loop closure, naming it (the open tree as the control); the URDF reader's tree,
  its hash pin, URDF's roll-pitch-yaw, the three groups with one warning per kind and a refusal naming each refused
  element (the file without it as the control), acknowledgements, and the mass a dynamic chain would need.
- `multibody_cases.py`: the trees the rigid-body library is tested on (the planar arm and the AR4 from
  `descriptions/`, and a branched tree built in code with a prismatic joint, a welded body with mass, armature and an
  extra frame), and joint states inside their limits.
- `script_a.py`: script A as a script declares it, with the declared values of phase 0's frozen reference, and its
  body side (`drag=False`); the round trip, the lag list, the snapshot and the golden outputs build it, and
  `reproduction_arm()`, the body starting where the reference's did, is what `differential/` holds to the reference.
  `rigid_oracles.py`: the rigid body's closed forms and two test-only laws (a constant wrench, a constant resistance)
  under the `oracle` namespace, registered in the test process only. `hydro_oracles.py`: the hydrodynamic family's
  test-only member, a Stokes sphere, with its closed form; the surrogate's formula again in float64 NumPy, the drag
  family's oracle case's reference; and the 0.2 line's +z frame, planted.
- `materials/`, the seam to OpenMaterica (phase 5): `test_seam.py`, what the build does with each verdict (`met`,
  `not_met` with OpenMaterica's reason, `unknown` with and without an acceptance, `no_pick`, an interval's bounds,
  a table and its extrapolations), each refusal with its control; `test_pins.py`, records as pinned files (a byte
  changed is refused at build and at load), the saved document holding no record value (the lock holding them is the
  search's control), the lock's pins and reads, and only `mime.materials` importing `materica`; `test_parts.py`, the
  solids against a Monte Carlo integral (a planted inertia as the control) and the part resolver; `records.py`, the
  records and small experiments they share.
- `differential/test_script_a_records.py`, phase 5's gate: the reproduction arm through the seam is phase 4's
  (`references/script_a_phase4_reproduction.json`, frozen before phase 5 changed anything; bit for bit in the
  environment that froze it), with one inline value 1% off as the control; each record switch of the record arm
  changes A's steady quantities as `script_a_record_switches.json` predicted before the record arm existed
  (`predict_script_a_records.py`), and misses the other switches' predictions; the record arm differs from the
  reproduction only in what its report lists.
- `oracles/`, the physics oracles, each with a control that fails (float64, per module):
  - `test_integrator.py`: from rest under a constant force and torque against constant resistances, the velocity
    and angular velocity approach their terminal values by the closed form, with a first-order error (halving `dt`
    halves it); the quaternion stays unit-norm; through a built experiment the twist is the node's. The control: the
    resistance integrated explicitly (planted as a subclass) diverges at `dt = 2.5 m/c`, where the implicit scheme
    relaxes;
  - `test_projection.py`: a Hypothesis property of the confined body's node, after every step inside the lumen less
    its extent and margin and moving no further out where projected; the control, the unconfined node, fails it;
  - `test_laws.py`: gravity's closed form and couple, the field (constant magnitude, perpendicular to its axis, at
    the commanded frequency, its time by count), the torque `m x B` and zero when aligned (to the products'
    rounding: XLA fuses a multiply into the subtraction); the control: a planted sign flip fails the magnetic
    family's sign case and only it;
  - `test_observation_and_drive.py`: the swim speed (the velocity along its direction, its mean over the steps after
    `after`) and the constant drive (its value every step, moved through `params`), with a control;
  - `test_multibody.py`, the rigid-body library on three trees, each oracle with its control: forward kinematics
    against the planar arm's closed form (one axis reversed); Jacobians against `jax.jacfwd` of forward kinematics
    and central differences (a lever crossed the wrong way round); the mass matrix symmetric positive definite and the
    Hessian of the kinetic energy summed over every body (one body's rotational inertia dropped); gravity against the
    potential's gradient (gravity reversed); the velocity terms against the Christoffel symbols (the gyroscopic term
    dropped); inverse dynamics as `M qdd + h` and forward dynamics inverting it (`h` without its velocity terms);
    inverse kinematics reaching targets placed by forward kinematics, a target out of reach reported as such, and one
    reached only outside the limits reported as outside them;
  - `test_synchronous_rotation.py`, the integration oracle: a body in the rotating field against a constant
    rotational resistance locks to it with the lag `arcsin(R omega / (m B))` less one step's turn `omega dt` (the
    tolerance stated before the run), halving `dt` halves the discrepancy, and above step-out the body slips;
  - `test_drag.py`: the drag's surrogate against the BEM values it was checked against (bounding it against that BEM,
    not the flow), its three fixes (nu and kappa refused outside the training data, with the 0.2 line's range as the
    control; the axis from the geometry, a tube along x the turned tube along z, with the 0.2 line's +z frame as the
    control; finite values and derivatives on the axis, matching finite differences where they can tell), the
    lubrication's formula and its gap floor, the clamp, the reference point MIME-ANO-112 records, the Stokes sphere's
    closed form and terminal twist, and the entry's refusals, saving and swap (references in `references/`).
- `differential/`: the node-level differentials against the 0.2 port (`references/`, generated once in the port's
  own environment by `generate_port_references.py` and `generate_port_drag_references.py`), with their declared
  differences, two anomalies of the old code found on the first run (MIME-ANO-110 and -111), and planted mutations as
  the controls; the drag's (`test_port_drag.py`) on poses inside the training range, past the clamp, near the axis
  and on it; and the rigid-body library's against the port's on the AR4 (`test_port_multibody.py`, from
  `generate_port_multibody_references.py`): frames, Jacobians, centres of mass, mass matrix and inverse dynamics
  within 64 float64 roundings, with no difference declared, and a planted mass and a reversed axis as the controls.
  `test_provenance.py`: the generators record how they were run through `_provenance.py`, with no private path,
  judged by the scan's own path patterns.
  And phase 4's gate, script A against its frozen 0.2 reference (`references/script_a/`: the reference's
  steady quantities and provenance, float64, float32 and float64 at 11 Hz, and every tenth sample of its float64
  trajectory):
  - `script_a_expected_differences.json`: every expected difference with its predicted size and its reasoning, the
    tolerances, what is gated and what only reported, the controls, the lag-structure mapping, and a `post_hoc`
    section for what the first run found, each part committed before the run it governs. Written by
    `predict_script_a.py`, which integrates A in NumPy with 0.2's update rules (reproducing the reference to
    rounding) and with 0.3's; `script_a_steady.py` is the reference's window rule; `script_a_runs.py` runs
    `script_a.py`'s `reproduction_arm` and compares;
  - `test_script_a.py`: float64, every gated quantity on its prediction (the integrator's +2.35% in swim speed
    among them), the quantities that cannot discriminate reported with why; the controls: at 11 Hz it matches the
    11 Hz reference and at 10 Hz misses it everywhere; at half the step it matches its prediction and the
    integrator's gap halves; the emulation still reproduces the reference; `examples/script_a.py` declares the same
    experiment;
  - `test_script_a_float32.py`: float32 against the float32 reference, every quantity gated, the phase lag
    included (MIME-ANO-113's fix); change 1's float32-float64 lag bound on its own scheme; the control, 0.1.0's
    form planted, misses; and the post hoc item of 4c3 (a float32 body held inside the contact band): the lag within
    3.5e-8 rad of float64 at three amplitude draws, the body on its surface, the control missing;
  - 4c3's declared changes (`declared_changes` in the file, `predict_script_a.py --declare`), each committed before
    its measurement, and `script_a_runs.planted_out`, which undoes a change in one test to measure a declaration on
    its own scheme or to run a control;
  - `test_script_a_lag_structure.py`: both graphs' edges classified by MIME's lag rule and mapped as declared; the
    torque reads the orientation at the start of the step in both; the reference's late drag is what 0.3's rule
    refuses; the control without the canonical order is noticed.
- `descriptions/`: the description files the rigid-body library reads, the planar arm and the AR4 as the 0.2 line
  ships them.
- `docs/`, the documentation's sources and their checks (`mime.docs`), every control running the real check on a
  planted class or on a doctored copy of the checkout (`checkout.py`):
  - `test_registry_export.py`: `mime registry --json` in a fresh process lists every installed entry with its fields;
    the completeness test, with its controls (B3): a class attribute not declared a `ClassVar`, and a `ClassVar` the
    export cannot encode, each fail by name; a declared one passes; a namespace failing to import fails the export;
  - `test_symbol_table.py`: the symbol-table checks on the test-only nodes of `planted_nodes.py` (no MIME node
    carries `NodeMeta` yet): S1 to S6 and N3, each arm failing by name and each passing arm passing;
  - `test_refusal_table.py`: the table is what the source constructs, and the scan's and the registry's golden
    numbers (`golden_numbers.json`) reproduce; R1 (a lab's `Problem` code refused, never raised), R2 (a planted
    tuple code, a code passed by a variable, a stray code-shaped string and a stale row, each found) and R3 (an
    uncatalogued code, or one in the wrong class, raises where it is constructed);
  - `test_lists.py`: the list reader, and W4 on the refusal list: a listed code a test now names, a code no test
    names and not listed, an empty reason, a bare count and a duplicate each fail;
  - `test_docs_check.py`: `mime docs check` passes on the checkout, printing each list's length and its wall time,
    and fails on a doctored copy and on a folder that is no checkout; the entry fields' checks on planted entries;
  - `test_package_boundary.py`: `mime.testing` never imports `mime.docs`, what the export imports never imports
    `mime.testing`, and `mime.cli` imports `mime.docs` only inside a function (C1), each scan with a planted import
    as its control; without the `verify` extra, `mime.cli` imports, `mime registry --json` and `mime verify` run, and
    `mime docs check` and `mime docs build` name the extra with no traceback.
  - `test_pages.py`: the page generators through the real `mime docs build` in a fresh process, the registry export
    as the oracle: every entry of the export has a page naming its version, the lab fixture's among them (the control
    for the generated half, by the namespace rules), every class variable, node and family has its page and every
    relative link resolves; the toy family page compares its two rungs with no blank cell and the hydrodynamic family
    prints its contract and rung list; no page holds an absolute path; the output option, the ignored default
    directory, and a build that clears only a directory a build made; a hand-written source is printed in its page,
    and one that no longer resolves fails the build naming the file and writes no page (W8's second half); W7 (an empty
    equations, assumptions or references field prints `none declared` and is counted, a filled one does not); a
    display equation on two pages fails; a gradient record's every field is printed;
  - `test_reproducible.py`: W8, two builds in fresh processes under two hash seeds: a planted generator that writes a
    set's order fails, through `mime docs check` too, naming the page, and the real generator passes;
  - `test_citations.py`: the citation parser and N2 (`[@a; @TYPO]`), with the old script's pattern as the control that
    sees one key of the two; the forms of Pandoc's grammar, what is not a citation, the bibliography reader, and the
    zero-citation failure;
  - `test_rendered_numbers.py`: the number rule over the text the generators print from code, a number planted in
    each kind of string in each of three forms failing by page and string, every recorded string on its page, the
    keying of a rendered string, and the namespace rules through the real command.
- `families/`: `test_family_walk.py`, the registry walk's other half: every registered interaction has a case in
  `family_cases.py`, declares its family or says why it has none, and every one in a family passes its whole suite,
  with controls (a case removed, an interaction
  with none, gravity with its weight reversed, a projection that keeps the outward velocity or projects too far).
  The family suites' control. `test_toy_family.py`: the toy drag's two rungs conform to the toy family,
  the planted sign error fails the sign case and nothing else, and a law planted with each other kind of defect fails
  its case (frame, gradients, material reach, units, nodes, swap, oracle, and equal and opposite on a family of
  two-body laws). `test_node_battery.py`: a declared order of accuracy is measured, and a wrong claim or one with no
  ladder fails. `test_hydrodynamic_family.py`: the Stokes sphere conforms, swapped with the surrogate, and planted
  drags fail their cases (a flipped resistance the sign case, the 0.2 line's +z frame the frame case, a rung ignoring
  the viscosity or squaring it material reach).
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
  - `test_report_warnings.py`: MADDENING's warnings in the report, and a units warning as a refusal;
  - `test_lags.py`: lag accounting. Script A's previous-step reads are those of its planned graph; the report and the
    snapshot show every edge's lag; inside a coupling group an edge reads the current iterate. The controls: in
    declaration order the list changes (and, asked to, the build refuses the unplanned read); without gathering a
    group's members the full toy reads a load a step late and is refused; a body registered with a body added before
    it is refused; a classification planted wrong is refused by the cross-check with MADDENING's schedule;
  - `test_body_side.py`: phase 1 on the body side's declarations: declared values, the tube and its typed paths,
    fills, directions by reference, contact's binding and its reserved rung, commands and drives, the body's initial
    state, each refusal with what to write.
- `saved/`, the saved form (float64, as `build/`):
  - `test_registry_walk.py`: every registered entry and kind of thing has a round-trip case in `cases.py`, and every
    case round-trips through a fresh process with the same derived hash and bitwise-identical first steps; its
    controls remove each load-bearing case and add an entry with none. Each case's parameter leaves reach its step;
  - `test_node_walk.py`: every registered entry has a node case in `node_cases.py`, and every node it builds passes
    MADDENING's `verify_node` (and `verify_node_order` on each declared order); its controls remove each case and add
    an entry with none. The same case holds the entry to its `builds` for the rung it builds, a node's symbol rows to
    their names and units, and a rung no case builds is listed in `docs/ratchets/builds_unchecked.txt`; its controls
    (B1, B2, B4): a node declared and not built, one built and not declared, and a rung no case builds;
  - `test_document.py`: what the document holds, that order and formatting leave it and its hash alone, that a
    Python tag is refused and constructs nothing (with an unsafe loader as the control that the payload is live);
  - `test_load.py`: an unknown entry and another version refused with the line to install, an edited document
    refused naming what changed, a static array's contents changing the derived hash where MADDENING's description
    cannot see it, the factory guard, reports across environments, `mime verify`, and what `save` refuses;
  - `test_schema.py`: the TypedDicts and the loader's run-time check in step, on instances Hypothesis generates from
    the annotations;
  - `test_golden_outputs.py`: a golden output per registered name and version (`golden/outputs/`), within 1e-10 of
    each field's scale, an entry without one failing; its control moves one value;
  - `test_experiment_states.py`: the `Experiment` API as a Hypothesis state machine: random scripts of `add_*`,
    `declare`, `attach`, `observe`, `build`, `save` and `load` over the toys and script A's entries, drawn as a
    script writes them; a build succeeds or refuses with reasons, `load(save(x))` keeps the derived hash, and a
    refused build leaves the experiment unchanged. Its controls, a save dropping a setting and a build that changes
    the experiment before refusing, are caught on every run (20 of 20 locally);
  - `test_body_side_document.py`: what script A's document and lock hold (the geometry, the fill, the drives, the
    declared values and their labels, the drag's weights by name and hash and its lubrication as a group, the
    asset's source and hash in the lock) and what a load refuses of an edited one, or of an installation shipping
    other weights under the same name.
- `scene/`, the scene (float64, as `build/`), on script A's body side and on `bound_robot` (`robot_asset.py`: a small
  robot asset the tests write, its magnet part's direction in the `mime:` schema, and script A's body side with the
  robot bound to its prim, also a round-trip case of the registry walk):
  - `test_stage.py`: the stage read back with pxr has the declared prims, up axis, `metersPerUnit` and
    `defaultPrim`; the control, a stage written without `defaultPrim` by a planted switch, fails that check and
    usd-core's validators, which find no error otherwise; the tube's settings on its prim in SI; the body at its
    pose; a micro-scale mesh's points local; up shown up for any declared up; the scene's units; two builds writing
    the same bytes, with their hash in the lock, and a changed declaration changing them;
  - `test_assets.py`: two names per thing (the leaf as the default, a duplicate name or prim refused), a missing or
    taken prim, the magnet's direction read from the asset and turned into the body's frame (the moment follows
    what the asset writes: the control), a direction declared twice refused, design data not in the schema's form
    refused, and an asset the scene cannot read as written refused;
  - `test_saved_scene.py`: the round trip keeps the scene, the prim paths and the asset's pin; a changed asset or an
    edited stage is refused at load, the unchanged folder loading as the control;
  - `test_moving_geometry.py`: a read of a moving thing's geometry refused naming the reader, the thing and why, and
    building once the reader declares `moving`; script A's reads of the fixed tube unaffected;
  - `test_frames.py`: up declared once, on the scene, and gravity derived; a missing up refused naming up, and no
    default axes; directions by reference following the tube; one orientation line per thing;
  - `test_rotated_world.py`: nodes never hard-code an axis. The body side in a world turned by a rotation Hypothesis
    draws runs as the first world turned, within 16 eps per step of each value's scale (measured at up to 1.1); the
    control, a law with +z in its code, fails it on every rotation tried, far above the tolerance.
- `test_import_boundaries.py`: only `mime.scene` and `mime.viz` import `pxr`, and `mime.core` (with
  `mime.core.multibody`) imports nothing from MIME above it, each with a planted import as the control.

## The planned layout

| Folder | What it will hold |
|---|---|
| `unit/` | grows with the part resolver and frames |
| `build/` | grows with each new kind of construction (mapped edges, multi-rate physics) |
| `families/` | grows a conformance suite per family as its entries arrive: acoustic radiation next |
| `differential/` | grows with each ported node; A0 against its frozen reference in phase 7 |
| `oracles/` | grows with each law and dynamics; software oracles, one lane each |
| `acceptance/` | comparisons with measurements, each stating what it bounds and what it estimates |
| `gradients/` | gradient checks that rebuild the graph, with planted failures as controls |
| `saved/` | grows with every entry and every feature of the document (materials, controller) |
| `scene/` | grows with stage-first scenes, regions and bindings, and OpenMaterica's scene fixtures |

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
