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
- `platforms.txt`, `test_platforms_selection.py` and `platforms/`: the smoke job for macOS and native Windows
  (`.github/workflows/platforms.yml`) runs the test files the list names, each with its reason, through
  `scripts/platforms_smoke.py`. The test holds that every named file exists and is a test file, and reads the workflow:
  both platforms, two Pythons, the default lane's install, a timeout on every job, and no `PYTHONUTF8` or
  `PYTHONIOENCODING`. `platforms/test_smoke.py` is the smoke's own: the console script through pipes, and a save and
  a run read as bytes. What each platform supports is in [`docs/platforms.md`](../docs/platforms.md).
- `test_canary.py`: CI's planted failures. A manual run with `plant_failure` set must turn every lane red, which
  proves each lane really runs its tests; one with `plant_slow_tests` set makes two tests take 21 s and 6 s, which the
  time budget must fail and warn on.
- `conftest.py`: each test's own CPU time and its children's, written into the JUnit XML for the durations report;
  and one OpenBLAS thread per process, set before SciPy loads it, since its spinning threads slowed many small solves
  sevenfold with every core busy.
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
  nodes' (`test_mime_node_fields.py`, which also requires every input to declare its units); the switch functions
  (`test_switch.py`: each one's value and derivative are its operation's, ties included, and its traced program is
  the operation's with no context open, `nonzero_norm`'s the safe norm's, row by row along an axis, recording whether
  any row was zero; the module imports only JAX, jaxtyping and
  the standard library; a collector records each switch with its owner, and the test-only setting acts only on a
  fresh trace, replacing a zero fallback by the scale and recording each guard's call site, each with its control);
  the quaternion helpers (`test_quaternion.py`: the rotation's exponential at rest is finite with finite derivatives,
  and so is a body stepped at rest, with 0.2's increment as the control that gives NaN); shapes and the tube (`test_geometry.py`: the
  cylinder's distance and normal by their closed forms, finite with finite derivatives on its axis; its derivatives
  against central differences off the axis, at a resolution chosen per draw, with a derivative planted off by two
  and a half times that as the control; on any axis, within its resolution, the values and derivatives its record states, with the
  resolution removed as the control; the lumen as a region with its axis and a point on it); the drag on its tube's
  axis in float32 (`test_drag_axis.py`: finite value and derivative on the axis and 1e-30 m from it, the 0.2 line's
  formula giving NaN as the control); the table evaluator (`test_tables.py`: the form refused when made, each scheme
  exact on the polynomials it reproduces at random tables, derivatives against central differences, a cubic's
  one-sided derivatives equal at its knots and a linear table's not, the Cholesky variant positive definite over
  resistance-like draws at physical scales (a regression check) and rough draws (which tell the construction from a
  plain interpolation), its switch declared by test entries and checked by the switch harness for a jump,
  a step reading every scheme passing the switch scan over `mime`'s package, the flags and both refusals; the
  controls: entry-by-entry cubic interpolation losing definiteness, secant slopes failing the quadratic oracle on
  every draw, a raw cell lookup flagged; and `test_table_cases.py`: OpenMaterica's table-evaluation cases through
  the materials seam, all 35 files in float64 and float32, each side, with and without an extrapolation, and
  planted faults (the side, the clip, the cell, a kept slope, OpenMaterica's edge faults, a shifted value) each
  failing cases in every configuration).
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
- `control/`, the controller (phase 7b1), on `control_cases.py`'s test-only controllers (under the `oracle`
  namespace) and a motor alone above a body, whose angle is the command it was given, and on A0's rig
  (`chain_cases.tracked_rpm`):
  - `test_declaration.py`: every refusal of the declaration raised, each naming what to do; the external inputs
    derived from the writes and nothing else, the 0.2 double declaration refused, a scan that leaves one unfed
    refused; each read classed in the report and the lock alike; the profile seam refusing a state read when given a
    judgement that does; the document's controller and its load; controls (a), a write dropped from the derived
    inputs, refused by the post-build check, and (d), a state read classed as an observation, caught;
  - `test_callback.py`: the latency rule pinned step by step at latencies 0 to 7 over a period of 3; the hook seeing
    each chunk's end, time and state; script A chunked for a hook bit for bit its one scan; the controller called
    once a period with the state as the period starts; runs continuing with its waiting writes; A0's controller
    holding the flange at its standoff above the body within its reach, and moving its target by its smoothing per
    tick; controls (b), a latency off by one step, (c), a hook skipped on the last chunk, and (e), a callback
    called twice in one period, each caught.
- `oracles/`, the physics oracles, each with a control that fails (float64, per module):
  - `test_integrator.py`: from rest under a constant force and torque against constant resistances, the velocity
    and angular velocity approach their terminal values by the closed form, with a first-order error (halving `dt`
    halves it); the quaternion stays unit-norm; through a built experiment the twist is the node's. The control: the
    resistance integrated explicitly (planted as a subclass) diverges at `dt = 2.5 m/c`, where the implicit scheme
    relaxes;
  - `test_projection.py`: a Hypothesis property of the confined body's node, after every step inside the lumen less
    its extent and margin and moving no further out where projected; the control, the unconfined node, fails it;
  - `test_hull_contact.py`: the contact 0.2.0, which holds a body's hull points (`hull_cases.py`'s nodes and its
    FL-9 held from the first step), each oracle with a control that fails it on every run: a capped cylinder and
    FL-9 drawn turned and placed within what the tube admits, moving and driven, never past the wall nor past the
    margin by more than the sweeps leave (today's contact, the centre held by its extent, lets the hull through); a
    long rod pressed into both walls by a torque takes the couple's closed form at its two points (the rows' moments
    dropped cannot); the solve never leaves more energy than the free step and its forces do no positive work (a
    solve that pushes points away gains it); where nothing touches, the step is the free body's to the last bit (a
    vanishing kept force is seen); turning the tube and the body together turns the step; a contact that adds
    lubrication beside a drag that includes it is refused; and the plan's refusals of a hull it cannot hold;
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
  - `test_drag.py`: the drag's surrogate against the MRS values it was checked against (bounding it against that MRS,
    not the flow), its three fixes (nu and kappa refused outside the training data, with the 0.2 line's range as the
    control; the axis from the geometry, a tube along x the turned tube along z, with the 0.2 line's +z frame as the
    control; finite values and derivatives on the axis, matching finite differences where they can tell), the
    lubrication's formula and its gap floor, the clamp, the reference point MIME-ANO-112 records, the Stokes sphere's
    closed form and terminal twist, and the entry's refusals, saving and swap (references in `references/`);
  - `test_mrs.py`: the drag's MRS rung in free space against the sphere's Stokes drag and torque (two
    resolutions, extrapolated) and the prolate spheroid's closed form, on each kind it serves, with its refusals;
    the free-space wall replaced by a non-zero one and the sphere's radius read too large as the controls; the
    prolate spheroid through the rung, on a kind the test registers, extrapolated from three resolutions, with its
    semi-axes read 1% too large as the control its translation entries catch; and every product of the node's step
    at the highest precision, a node placing its samples at the default one as the control;
    `test_mrs_gradients.py`: its derivatives, DEFINITION 5.3's three checks (the pose against a part placed there at
    construction, differences by rebuilding the experiment for the sphere's radius, the viscosity and the helix's
    wavenumber, and for every trainable leaf of the node on each kind, each with its leaf stopped as the control; the
    protocol stated before the run), and on the shipped solve the forward mode's one factorisation, the forward and
    reverse modes against the difference, and the reverse mode's checkpoint, with a baked orientation, a baked blob,
    an adjoint taking a traction matrix for symmetric and the linear-solve rule without its `dA x` term as the
    controls; `test_mrs_family.py`: the hydrodynamic family's whole suite on the rung, on a rig of its own in an
    unbounded medium; the experiments in `mrs_cases.py`.
- `differential/`: the node-level differentials against the 0.2 port (`references/`, generated once in the port's
  own environment by `generate_port_references.py` and `generate_port_drag_references.py`), with their declared
  differences, two anomalies of the old code found on the first run (MIME-ANO-110 and -111), and planted mutations as
  the controls; the drag's (`test_port_drag.py`) on poses inside the training range, past the clamp, near the axis
  and on it; and the rigid-body library's against the port's on the AR4 (`test_port_multibody.py`, from
  `generate_port_multibody_references.py`): frames, Jacobians, centres of mass, mass matrix and inverse dynamics
  within 64 float64 roundings, with no difference declared, and a planted mass and a reversed axis as the controls;
  and the table evaluator against the 0.2 line's wall-table lookup on two blocks of its own grid, the near-wall
  corner and the axis end with every angle and offset (`test_port_table.py`, from `generate_port_table_references.py`),
  within 64 float64 and 128 float32 roundings declared before the run, with a cell one below and a knot moved as the
  controls; `table_budget.py`, not collected, times a read against that lookup's on its whole grid. And the
  MRS library against the 0.2 line's free-space MRS on its own meshes (`test_port_mrs.py`, from
  `generate_port_mrs_references.py`, frozen before the port): the helix, a sphere and a prolate spheroid, the full
  sizes marked slow; the 0.2 helix mesh tied to the de Jongh kind; the kind's own quadrature and its caps reported as
  model differences; a dropped weight and the traction matrix transposed as the controls. How the helix's resistance
  converges (`test_mrs_convergence.py`, recorded by `measure_mrs_convergence.py`, not collected, in
  `references/mrs_convergence.json`): the 0.2 line's triangle rule, written again here and tied to the frozen 0.2
  output, and the kind's samples through the rung's node, at five levels; the record's bracket of the diagonal, and
  the kind's path at the CI levels bounded by it, with its samples read 2% too large as the control.
  A0's controller against the 0.2 controller (`test_port_controller.py`, from
  `generate_port_controller_references.py`, run once with x64 off and once on): `mime.control.ArmTracking` stepped on
  the recorded input sequence the 0.2 controller was fed, with the orientation feedback off and on, its target joint
  positions, target position and orientation (compared by the angle of the rotation) within bounds derived in the
  test's docstring from float32's epsilon, the quantity's size and the operations it passes through, and committed
  before the references were regenerated; one difference found when measured (the target's quaternion is the other
  sign, one rotation) is stated there as such. The recorded sequence passes the reach envelope (a third of its ticks
  beyond it, some on it) with the arm's joints scattered off home, so that the iterations have not converged. The
  controls: ten planted changes (gain, damping, iterations, smoothing, standoff, lookahead, the reach clip removed or
  widened), a conjugate and a flipped axis, and the casts' dtype; a change below the bounds' resolution is stated there.
  `controller_budget.py`, not collected, times A0's rig with and without its controller as a callback.
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
  And phase 7c1's gate, A0 against its axis-fixed reference (`references/a0/`: the pre-0.3 tag's A0 with its drag
  surrogate given the tube's axis, at 3 Hz and 10 Hz, in float32 and mixed float64; the steady quantities, the
  provenance with its paths left out, and the float32 runs' arrays at every tenth step; `freeze_a0_reference.py`
  copies them in):
  - `script_a0.py`: A0 through the public API (the AR4 carrying a motor and the rotor magnet, a point dipole, closed
    by `arm_tracking` inside the compiled step with the 0.2 line's settings, the body script A's, the drag surrogate
    pinned to `dejongh_mlp_v2`, which this does not validate: MIME-ANO-117); `a0_runs.py` reads the steady quantities
    from the reference or a run;
  - `predict_a0.py` and `a0_expected_differences.json`, committed before the comparison was first run: the emulator
    of script A extended to the arm (the tag's computed-torque joints against the kinematic arm, their inverse
    kinematics and the rigid-body library's forward kinematics), the dipole, the rotor's PI loop and the controller;
    it reproduces the reference with the tag's scheme (swim speed to 1.5e-5, rotation rates to 1e-9, the rotor's speed
    exactly, the single-precision accumulators of the rotor's angle and the controller's filter bit for bit), and
    predicts 0.3's A0 with four declared differences with predicted sizes (a fifth, mu_0, declared negligible: 1.35e-10), six declared none, and tolerances from the reference's own
    float32 gap and the emulation's miss of it; `a0_emulation.json` is its record;
  - `test_a0.py`: float32, every gated quantity on its prediction at both frequencies, what cannot discriminate
    reported with why; the control for the whole comparison, held to the other frequency's prediction it misses all
    but the distance (which the controller fixes at either); each declared difference larger than the tolerance of
    the quantity that gates it; the build's lags and weights; the reference's values read back; the emulation rerun;
  - `test_a0_controls.py` and `a0_plants.py`: a fault planted in the arm's pose, the dipole's position and moment, the
    controller's latency and its law, the motor's speed and the gradient force alone, at three resolutions in both
    directions, fails the comparison; `measure_a0_controls.py` measures the sensitivities the resolutions rest on and
    the catch rate over sizes from below the resolution to ten times it (`references/a0_controls_measured.json`),
    which `test_a0_controls.py` checks against the constants the plants use. `rotor_angle_rate_mean_hz`, the motor's
    output, is gated beside the commanded speed; it was added in the review of phase 7c1 (not declared beforehand).
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
    through the real command, an `exact` entry holding a jump switch and a raw-switch list line of the wrong form
    each fail;
  - `test_switch_declarations.py`: every installed entry's status is the one its switches and listed raw sites give,
    no `piecewise` remains, its four users carry their statuses and `rigid_inertial` is `exact` with no record, each
    owner is declared once; the controls:
    `exact` with a jump switch, `exact` with a site listed `cut`, a status named `piecewise` or none, and an owner
    declared twice, each fail;
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
  - `test_reproducible.py`: W8, builds in fresh processes under eight hash seeds: a planted generator that writes a
    set's order fails, through `mime docs check` too, naming the page, and the real generator passes; the seeds'
    detection of two-, three- and four-item sets, measured over drawn sets; and a test that no module of `mime.docs`
    imports a clock, a random source or the machine's identity;
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
    an entry with none. Every case that uses the entry holds it to its `builds` for each rung, in the first case that
    builds that rung, a node's symbol rows to their names and units, and a rung no case builds is listed in
    `docs/ratchets/builds_unchecked.txt`; its controls (B1, B2, B4): a node declared and not built, one built and not
    declared, a rung no case builds, and a rung only a later case builds, which is checked there;
  - `test_rung_gradients.py`: each rung's own gradient status: a saved lock records the attached rung's (the drag's
    mrs rung `exact`, its surrogate `piecewise_jump`, an entry without rungs of its own its own), a declaration that
    is not the entry's is refused at registration, and a rung's status is held to its own switches; the controls, a
    lock that writes the entry's status for every rung and a rung declared `exact` beside a switch;
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
  - `test_switch_scan.py`: the switch scan over every registered node's step, traced in `switch_configurations.py`
    (script A's four arms and the round-trip cases): the tree passes with its list and its owners; the traced scope
    holds every census site a node's step reaches; every raw operation planted in a package of its own fails in its
    plain, aliased and `jax.numpy.` forms, as does every other family and a switch function inside each transform
    (`vmap` batched and unbatched), directly or through a helper, and so do a custom rule through
    `functools.partial` or among a module's statements, an operation passed uncalled or through a local alias, a
    power of a half, a comparison cast by a call or in arithmetic, a raw operation in the second of two lambdas on a
    line or in a generator expression, and an owner passed by position to a helper; host code, guarded square roots,
    clean generator expressions and lazy indicators pass; its controls (a converted site still listed, a listed
    `stop_gradient` naming no status, an undeclared owner, a declared owner nothing passes, an untraced node class,
    and steps whose equations carry no frame, which the scan reports first and alone) fail; a lab's owners carry its
    package's name;
  - `test_switch_harness.py`: each of MIME's switch owners, on script A's nodes or the cylinder, forced each way a
    quarter of a step from its switch, measures within its declared order, with its indicator at each placed point,
    or measures nothing where its declaration says so; the kink limit fitted on drag's clamp and end and checked on an
    analytic kink and jump, with its resolution on two mixed rows; replacing every domain guard's fallback changes
    nothing where its condition is false, every guard the switch scan lists is reached so and every nonzero norm
    away from zero; the controls (a jump declared a kink, drag's clamp divisor written as a jump, a row measuring
    nothing or something against its declaration, a jump in a small output beside a large one, a flipped or rescaled
    placement, a step traced before the setting, a fallback reaching a value or a derivative, a zero fallback
    reaching a value, a guard no point reaches false, a nonzero norm at zero) fail.
    `step_jaxprs.py` is the merge-time comparison: every node's step in every switch configuration as jaxpr text, run
    on a pull request's merge base and its head, which must be equal;
  - `test_shape_suite.py`: the shape conformance suite over every registered shape and capability (Hypothesis:
    probes off, near and on each declared set, turned and moved at random, float64 and float32, autodiff against the
    float64 difference on the point's side); every shape has a case placing each switch; at an explicit example near
    each set, for each moved parameter and each dtype, autodiff equals the difference, and its control, a tangent
    scaled wrong, fails; the cylinder's normal within its axis band has the projection across the axis as its
    Jacobian, a recorded artefact (`artefacts/`), with the constant-zero branch as its control; and on the screw's
    focal set the distance is exact with the normal its derivative (the review's settings, FL-9's and the most
    twisted), measured once differentiable there, with the Newton step forced on the set as its control.
    `test_registry_walk.py` also runs the kind-conformance harness on every kind that samples its surface and on the
    helix at every setting it accepts, with its controls (a closed form or the samples planted wrong, a cap sampled
    without its twist at every count of starts);
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
  - `test_screw.py`: the solid kinds against what they are: the helix's volume equal to A's declared volume to two
    units in the last place, the screw's distance equal to a brute-force minimum (Hypothesis, over screws and
    points), the earlier line's surface points on the re-centred screw, its thread across its normal; and the plan's
    refusals of a value the hull's kind gives, of an extent no hull gives, of a drag on a body of another kind or on a
    helix turned or moved in the body, and of a tube whose shape is not a circular tube's;
  - `test_rotated_world.py`: nodes never hard-code an axis. The body side in a world turned by a rotation Hypothesis
    draws runs as the first world turned, within 16 eps per step of each value's scale (measured at up to 1.1); the
    control, a law with +z in its code, fails it on every rotation tried, far above the tolerance.
- `examples/` (phase 7l-c, the first run), each running what a new user runs, as written: `test_readme.py` extracts
  the README's quickstart and executes it in a temporary directory, runs the `mime pending` command the README gives on
  the folder it saved, and holds the install section's distribution, extras and branch to the project's own;
  `test_sphere_settling.py` and `test_screw_in_tube.py` run the two examples at their starting sizes and hold the
  screw's speed to its row of `docs/data/spacing_tables.json` and its inertia to what the build's refusal prints. They
  bound that each example builds, runs and gives the number its table states; the screw's number is no accuracy claim.
- `results/test_watch.py`: the post-run Reynolds-number watch. Its two controls are properties: a run planted a
  factor of at least 1.05 above the bound is flagged on every draw and one below it on none, which the stress workflow
  runs twenty times with fresh seeds; its no-graph-change test compares the derived and input hashes of a build with
  and without the watch, and a build that does differ must hash differently. `build/test_inertia_refusal.py`: the
  refusal of a tensor with products prints the tensor, its principal moments and the declaration to paste, and the
  declaration builds, and its action is ordered by the products' size; `docs/test_spacing_tables.py` holds the tables
  and the numbers in the prose of `docs/choosing_spacing.md` to the measurements.
- `oracles/test_tilt_flag.py` and `results/test_tilt_arms.py`: the surrogate drag's two flags for a tilt it cannot see
  (friction 4c): `tilt_fired` at the declared validity range and `bore_fired` where a body no longer fits the tube's
  bore. The first holds each to its bound (the bore's in closed form) at tilts planted a factor of 1.05 either side of
  it, over bores, bodies and axes Hypothesis draws, that no derivative passes through them, and the declared limit to
  the measurement it rests on; its controls are plausible mistakes (the bore read as a radius, caught on every draw;
  the axis read as a row of the rotation, the limit as a half angle and the cosine unsigned, each found by search). The
  second runs friction 4c's arms for a second and holds both flags, step by step, to the recorded poses; its plants
  are deterministic. The stress workflow runs both twenty times.
- `test_import_boundaries.py`: only `mime.scene` and `mime.viz` import `pxr`, only `mime.materials` imports
  OpenMaterica (`materica`), and `mime.core` (with `mime.core.multibody`) imports nothing from MIME above it, each
  with a planted import as the control. Nothing in MIME imports `modal`, by the scan and in a fresh process that
  imports every submodule; the controls are a planted import and a planted module importing a stand-in `modal`.
- `test_naming_rules.py`: two of CONTRIBUTING.md's naming rules: no public class of `mime.fields`, `mime.interactions`,
  `mime.dynamics`, `mime.observations`, `mime.drives` or `mime.scene` ends with its package's category word, and
  no subpackage exports another's class under a name it defines itself. Each check has a planted violation it
  reports on every run.
- `env/`, the environment a run is made in (`mime.env`), with no Modal account, call or network:
  - `test_requirements.py`: each variant's requirements file is `uv export`'s output from `uv.lock`, byte for byte,
    run with the uv the dev group pins (CI's, and the Modal script's); the control, the lock with one hash changed,
    exports other bytes. Every pin has a wheel for every supported Python on its base image's glibc (the control,
    the CUDA plugin's wheels for one Python taken out, is found), and `bases.json` pins each series by digest and
    `uv.json` the uv that installs the pins;
  - `test_check.py`: `mime env check` refuses a planted pin, a missing distribution, an unpinned one, another
    Python, another OS, a requested GPU JAX does not see (the CPU fallback), a GPU of another type (an A100 request
    is met by any A100) and a GPU where none was requested, a MIME other than the pinned wheel's, a lab file that is not
    pins, a GPU run whose lock records a matmul precision other than the highest, and a package shadowed through
    `PYTHONPATH` (in a real process), each from its plant into an environment it accepts, and every difference at once;
    it evaluates markers for the system checked, and applies the lock's precision;
  - `test_lock_environment.py`: the lock's new environment fields round-trip, and a change in any one of them, in
    the lock or in the loading process (`XLA_FLAGS`, the matmul precision, the runner), makes `verify` compare within
    tolerances and claim no bits; two records that differ only in the runner's information (an image ID, a client's
    version) are the same environment; a device field neither run could read is never equal;
  - `test_modal_script.py`: the script `mime modal` writes, for a toy experiment and for script A, imported against
    a stand-in for `modal` (`modal_stub.py`): it parses and imports `modal` and nothing of MIME; its layers are the
    base, the variant's pins, MIME's wheel (from PyPI by hash, or the checkout's by hash), then the lab's, and the
    experiment is mounted, not copied; it names exact GPU types only, and a loose one is refused with nothing written;
    it copies MIME's requirements file byte for byte, pins the image builder and every uv by the digest of its image,
    refuses a GPU run of a lock saved below the highest matmul precision and a lab file that is not pins, and
    records the image ID and the client's version apart from what decides the results;
  - `test_run.py`: `mime run` writes its outputs and the run's own saved form in the lock's precision, which
    `mime verify` reproduces bit for bit, in a float32 process too (it applies the lock's precision, as `mime run`
    does); `run_to` writes the same as the command; XLA's cache directory is one per jaxlib, platform and, on a GPU,
    its compute capability.

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
- `memory_heavy` tests peak above 2 GiB of their own process (`memory_peaks.json` lists what was measured, with the line
  and its basis; `scripts/measure_test_memory.py` measures). Under pytest-xdist they run one at a time across the
  workers, whatever `--dist` says, and give their free memory back when they end (`conftest.py`): CI's runner has 16 GB
  for four workers, and the shard that ran the 8 GB arm battery beside the 6 GB helix reference peaked at 14.4 GB.
  `test_memory_heavy.py` fails a listed test over the line that is unmarked; it cannot know a test nobody measured.
- CI's lanes are: the default (Python 3.12 to 3.14, released dependencies), OpenMaterica `main`, MADDENING `main`, and
  a scheduled run against the newest usd-core. The last three report without blocking. Beside them, `typecheck`
  blocks: pyright, pinned, strict over `src/mime`, at zero errors.

Run the tests locally with `JAX_PLATFORMS=cpu python -m pytest`, after `uv pip install --no-deps ./lab_fixture`
(without it, `test_lab_fixture.py` skips outside CI), and after `mime assets fetch unit_tube_R0.997` (209 MB, into the
assets cache, or into a directory of your own named by `MIME_ASSETS_DIR`): the tube wall kind reads that table, so its
tests, the registry walk and the switch scan need it, and a test never downloads it: a missing one fails by name
(`asset.missing`), with the command. CI stores it once per job (`.github/actions/unit-tube`). CI runs them on pytest-xdist workers
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
