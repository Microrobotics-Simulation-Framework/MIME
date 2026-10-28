# Changelog

All notable changes to MIME are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- `scripts/check_citations.py`, replaced by the citation parser of `mime docs check` (below, under Added): the script read one
  key of `[@a; @b]`, none of `@key`, `[-@key]` and `[see @key]`, and the first word of a hyphenated key, and passed
  when no page existed.
- `scripts/check_impl_mapping.py`, which tested only that a name in one table of a 0.2 algorithm guide imported, and
  passed when no guide existed; the symbol table's checks replace it. CI's separate compliance step goes with it:
  `mime docs check` runs `scripts/check_anomalies.py`, unchanged.
- The 0.2 line, for the 0.3 rewrite on the `v0.3` branch: the package, its tests, experiments, examples, job and
  cloud-task files, docker files, debug folders, verification and analysis scripts, the root plan documents, and the
  docs. All of it remains at the tag `pre-0.3`. Kept: the licence files, `CITATION.cff`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CLAUDE.md`, `.claude/`, the anomaly registry (`docs/validation/known_anomalies.yaml`; anomaly IDs
  are permanent records), `docs/bibliography.bib`, and the compliance scripts.
- The 0.2 line's `.gitignore` rules for a root `data/` folder (`/data/`, `data/gnn_training/`,
  `data/gnn_sweep_dryrun/`, `data/sweep_manifest_test/`): the first kept the drag surrogate's weights out of git and
  so out of every distribution, and no 0.3 file is under a root `data/` (phase 4c1).

- The anomaly registry (`docs/validation/known_anomalies.yaml`), now one continuous record across the 0.2 line and
  0.3, in schema 2.0: MADDENING's field names where it has them (`status` is `resolution_status`, `closed` is
  `resolved`, `affected_nodes` is `affected_components`, the node ids merged into the existing list where an entry
  had both), `affected_versions` as a PEP 440 range on every entry, and `resolution_version` and `verification` on
  resolved ones. Resolved for 0.3: MIME-ANO-001 and 002 (removed with their code), 003, 107, 110 and 111; 108 stays
  resolved. Open: 100 to 106 and 109, each with its plan. Corrected: 002's component, 100's workaround, 101's margin.
  The header no longer cites IEC 62304 or any standard.
- `scripts/check_anomalies.py` validates that schema: the status values, no unknown field, a `resolution_version` and a
  `verification` list on a resolved entry, and that each `verification` id names an existing test. It has a `main()`,
  and `tests/test_anomaly_registry.py` runs the real registry and controls against it.
- A scene's up is +Y or +Z only, the two axes USD's `upAxis` names. Phase 6 wrote any other up as `upAxis = Z` with
  `/World` turned and `mime:up` recording it; a USD consumer reads `upAxis` and does not compose a root rotation to
  find up, and MIME takes its world frame from the stage's metadata. `Scene.new` raises `ValueError` for any other
  direction, before any build, naming the two ups: up is a choice of coordinates, so the scene is re-expressed with
  up along +Y or +Z (for example by swapping two axes), and nothing physical is lost. A saved document whose scene
  declares another up is refused at load with the same message (`frame.up-not-y-or-z`), and so is a scene changed
  after its declaration, at the build. The stage's `upAxis` is the declared up, `/World` carries no transform, and
  `mime:up` is gone. `mime.scene` loses `onto_z` and gains `UP_AXES` and `up_axis_problem`. The rotated-world test
  draws the rotations that keep up on +Y or +Z: a turn about +Z, followed on half the draws by the quarter turn that
  takes +Z onto +Y, which is the part that shows its planted +z law.

### Changed
- **The drag surrogate reads its shape from the body's kind** (phase 7d): ν, κ and L from the hull's family
  parameters (the de Jongh helix only: `drag.body-kind`; another modulation or count of starts:
  `drag.outside-training`),
  the outer radius from its radial reach, and the tube through the region's registered shape. Script A's body carries
  FL-9's helix as its shell; its golden outputs and frozen reference hold. The record arm's shell is the helix, whose
  closed-form volume equals A's declared volume to two units in the last place, in place of the cylinder of equal area.
- **The tube reads `piecewise_jump`** (phase 7d): the cylinder's in-band Jacobian, the projection across the axis, is
  held by a standing test as a recorded artefact, which clears its `stop_gradient` line. A geometry kind's switches are
  its shapes', inherited, and `mime docs check` reads them so.
- **`piecewise` is split into `piecewise_kink` and `piecewise_jump`** (dev1, phase 7s): a kink's value is continuous
  where its derivative is not; across a jump the value itself is not. An entry's piecewise status is derived from the
  switches it declares, and `mime docs check` fails on one declared otherwise. Its users: `contact`, `drag` and
  `swim_speed` are `piecewise_jump`; `tube` is `unverified`, capped by the cylinder's in-band `stop_gradient` until a
  standing test of the in-band Jacobian clears it; `rigid_inertial` stays `exact` with no gradient record: its
  quaternion's norm, zero at no valid state, is a domain guard (`nonzero_norm`), not a switch. `GradientRecord`'s
  `where`, `reason` and `route` are keywords, optional for a piecewise entry, whose `switches` are its record.
  Contact's route for its landing is now a landing located within the step (MADDENING's events, planned), not another
  rung. The refusal scan takes a declared switch owner, and a module's attribute under `mime`, `maddening`, `jax` or
  `builtins`, as a name.
- `mime.core.remove_outward` and every `VELOCITY_RULES` rule take the `owner` of the switch they evaluate, as a
  keyword; `mime.core`'s `safe_norm` delegates to the switch module's formula, which `switch.norm` evaluates with an
  owner and `switch.nonzero_norm`, for `quaternion.normalized` and the drag's frame, as a domain guard.
- The first documentation phase's follow-ups (dev1):
  - **The 0.2 algorithm ids: a node that changed in any way takes a new id.** Each of the 20 ids in
    `docs/data/node_ids_0.2.yaml` now records its verdict, `retired` with what changed (or that the node has no 0.3
    counterpart), or `kept_by` a 0.3 class's dotted path with its evidence; `mime docs check` accepts a 0.2 id only on
    the class that keeps it, and the rule giving a class the id of a 0.2 class of the same name is gone. No 0.2 node
    is the same in 0.3 (the six ids with counterparts all changed their parameters, ports or method), so all 20 are
    retired, and a new node still takes `MIME-NODE-104` onwards.
  - **Every MIME entry's `references` are established**, each key checked against its work, and `mime docs check`
    fails on a MIME entry's key that `docs/bibliography.bib` lacks, naming the entry and the key (a lab's keys are
    counted, not checked). Twelve verified entries are added to the bibliography: de Jongh et al. (2025) under
    `deJongh2025NonlinDyn`, Cox & Brenner (1967), Goldman, Cox & Brenner (1967, part I), Brady & Bossis (1988),
    Peraire & Widnall's MIT 16.07 lecture L28 (2009), Solà's quaternion kinematics (arXiv, 2017), Ling, Moebs &
    Sanny's University Physics Volume 2 (OpenStax, 2016), Tatum's Classical Mechanics (LibreTexts), and four MathWorld
    entries for the solids' volumes and inertias. `rigid_inertial` cites Peraire & Widnall for its equations of motion
    and Solà for its orientation update; its implicit resistance solve and its confinement are its own method, cited
    to no work. Drag's lubrication docstrings now say it keeps the asymptotes' leading terms only, naming the terms it
    drops.
    The dataset entry `deJongh2025` keeps its key, with its first author corrected to Luuc de Jongh and its path to
    the tag `pre-0.3`, where the summary it names still is. Five entries rest on no work and declare `()`: the
    constant drive, the uniform rotating field, the swim speed, the tube's geometry and contact's projection.
  - **The gradient records**: Tube stays piecewise, its switch the axis, where neither its distance (a cone point)
    nor its normal (no limit) has a derivative, so its route is "none". `Cylinder` now takes a point within
    `AXIS_ULPS` units in the last place of its radius of its axis as on it, whatever the axis's direction (the
    resolution drag's on-axis frame used, moved to `mime.core`): a point put on a tilted axis computes an offset of
    rounding, not zero, so the record's on-axis values held only for a tube along a coordinate axis, and now hold for
    any. Off the axis, a float64 property compares the derivatives (in the point and every parameter) with central
    differences at a step chosen per draw from their rounding and truncation: it bounds the derivatives' error from
    above at its resolution, from 7e-10 to 2.2e-5 of the derivative over the drawn range (8e-8 at the median draw), and
    a derivative off by two and a half times that fails on every draw. Their exactness is the closed form's; the test
    bounds it.
    Contact's, drag's and swim speed's routes to `exact` are filled; contact's `where` and `reason` are corrected (the
    zero along the normal in contact is the exact derivative of the projected value; the switches are the contact's
    branch changes) and drag's completed (the clamp itself, the axis, the end flip and the lubrication's floors).
    `docs/ratchets/not_established.txt` is empty. Drag's node docstrings now say the body's y axis is taken on the
    tube's axis where its x axis lies within 30 degrees of the tube's, as the code does, not 60.
  - **A lab's values not established are counted and named** in `mime docs check`'s summary, not failed.
  - **`register_namespace` refuses the first word of a MIME entry's name** (`gravity`), as it does a refusal code's,
    since `namespace_of` would then answer that namespace for MIME's entry.
- Phase 4c3 (dev1): three structural corrections to script A's physics, each a declared change, predicted by the
  emulator and committed before it was measured (`tests/differential/script_a_expected_differences.json`,
  `declared_changes`):
  - `rigid_inertial` 0.2.0 solves its step for the change in velocity, `(M/dt + R) dV = F - R V` (MIME-ANO-113,
    resolved): float64 moves by rounding (1.8e-15); every float32 quantity of A, the lag included, is gated again,
    and phase 4 is closed;
  - `rigid_inertial` 0.2.0 holds its confinement inside the implicit solve: on the surface and moving outward, a
    multiplier along the normal removes the outward velocity through `M/dt + R`; the projection after the step stays.
    Script A's speed moves from +2.35% to -0.110% against its reference, and its change between dt and dt/2 from
    1.18% to 0.026%. The normal velocity is held at the gap closed in one step, so a body inside the contact band
    (about 0.1 um wide in float32) reaches its surface (post hoc: held at zero, the float32 lag missed by 1.6e-7);
  - `drag` 0.2.0 moves the surrogate's resistance from the end of the body's axis to its centre of mass, `R_C = T^T R_P
    T`, the helix's axis the body's x axis (MIME-ANO-112, resolved): script A's speed -7.80% against its reference,
    3.2086 mm/s;
  - a document naming `rigid_inertial` 0.1.0 or `drag` 0.1.0 is refused at load (`load.version`).
- The surrogate's declared domain: `kappa` up to 0.655, the largest trained value; `L / R` from 2.65 to 6.24, checked at
  plan time; the offset fraction's 0.3 with its reason.
- Script A's record arm (phase 5) on 4c3's physics: each record switch's change declared again with 4c3's scheme
  before the arm ran on it (`script_a_record_switches.json`, `corrected_physics_4c3`) and measured on it to 8e-16.
  The density switch, whose speed change phase 5 credited to the implicit step's O(dt) load, falls from +1.05e-4 to
  -2.6e-8, below every tolerance, and is reported as such. The seam's reproduction is held to the arm 4c3 left
  (`references/script_a_4c3_reproduction.json`), and its control moves water's viscosity.
- `cholesky_resistance` builds the 6 x 6 factor whole-array (one gather through a constant index map, `jnp.where` for
  the diagonal's softplus and the zeroed upper triangle) instead of 21 single-entry writes, one operation each in the
  compiled step: equal to the loop bit for bit in value and gradient, in float32 and float64 (the first item of the
  performance budget).

### Added
- **Geometry kinds, the physics side** (dev1, phase 7d). `registry.Geometry` gains optional build-side capabilities:
  `samples` by named patch (a JAX function of the kind's parameters), `sample_counts`, `area`, `family_parameters`,
  `radial_reach` and `tangent`, with `implements` saying which a kind has. New solid kinds: `Sphere`, `Capsule` and
  `DeJonghHelix` (de Jongh et al. 2025, Eq. 1; centred on its centroid, the 0.2 body moved half its length along its
  axis with its turn kept), evaluated at run time by the new shapes `sphere`, `capsule`, `slab` and `dejongh_screw`,
  each declaring its switches (`sphere.centre`, `capsule.segment`, `capsule.end.low` and `.high`, `slab.mid`,
  `dejongh_screw.seam`); `nearest_point`. A body's hull, its one bulk part of a kind with a radial reach, gives it its
  `extent` (`parts.extent` where it cannot), and the values `extent`, `radius`, `length` and `helix_wavenumber` are
  refused beside it (`value.from-kind`); laws read it through `ctx.geometry(of=...)`. `mime.testing` gains the
  kind-conformance harness (`kind_problems`) and the shape conformance suite (`check_shape`, `shape_probes`,
  `SHAPE_CASES`), and `toy.region_probe`.
- The documentation's generators (dev1, phase D1b), in `mime.docs` and CI. Generated pages are never committed:
  - **`mime docs build [--root <checkout>] [--out <directory>]`** writes `index.md` and `pages/` to `docs/_build/`
    (untracked, in `.gitignore`) or the directory named, and needs the `verify` extra. It generates an **entry page**
    for every registered entry (a lab's too, from the registry export, with no work from the lab), a **node page** for
    every node an entry builds (`NodeMeta` printed once, the symbol table, the entries that build it), a **family
    page** for every registered family (its contract, its members, and, once two built rungs can replace each other, a
    comparison in which no cell is blank: tested on the toy drag's two rungs; the hydrodynamic family prints its
    contract and rung list), a **mechanism page** for every target kind, the **vocabulary reference** (every
    interface with its kind, units, frame, shape, sign, magnitude and description, a reserved name marked), a
    generated **index** with a front door by goal, and the hand-written **concepts note** (`docs/concepts.md`). Each
    entry page opens with an "at a glance" block that links the node page's equation and the family page. A
    hand-written source is printed in its page. The pages hold no absolute path; a source whose entry, version or id
    no longer resolves fails the build, naming the file, and writes no page (W8's second half); a display equation on
    two pages fails; an empty equations, assumptions or references field prints `none declared` and is counted with
    the other hatches in the build's summary (W7). A gradient record's every field is printed, so a field the switch
    foundation adds to it is rendered with no edit.
  - **`mime docs check` builds the pages** in process for the citation check and the number rule, and **in fresh
    processes under eight hash seeds**, failing on a page that differs by a byte (W8: that is all it bounds; a test of
    the generators' imports keeps a clock out, and address-dependent order, directory order and locale are not
    bounded). `--skip-hash-seed-builds` leaves out the second for the tests of the other checks; CI never passes it.
  - **The citation parser** (`mime.docs._citations`, control N2) reads Pandoc's grammar (`@key`, `[@a; @b]`,
    `[see @key, p. n]`, `[-@key]`, `@{any key}`), counts the citations it checked and the ones it did not find, and
    fails on none at all when there are pages. It replaces `scripts/check_citations.py`, which read one key of
    `[@a; @b]`, none of the other forms and the first word of a hyphenated key, and passed when no page existed.
  - **The number rule reads the text the pages print from code** (a validity line, a gradient record's text, a
    node's assumptions, an interface's description), keyed by the page's name under `generated/`, the tokens and an
    occurrence index. MIME's namespace is judged and the others counted. Ten strings are listed in
    `docs/ratchets/number_allowlist.txt`, each with its reason: the gradient records' and validity lines' numbers
    (D2 moves the drag clause, a1 structures the lines) and a sign condition's zero.
  - **CI builds the pages, scans the built directory and uploads it** as a run artifact, in that order, in the
    default lane. `scripts/check_confidential.py` gains `--directory DIR` (same key, same fork exception) for text
    found in no tracked file; a pytest control shows its exit status on a planted word and a planted private path
    (N5's output case), and `tests/test_ci_docs_artifact.py` reads the workflow for the step order, the scan's
    failing the job and the upload running only after every earlier step succeeded (N6).
  - `mime.testing.registered_families()`.
- **Switches** (dev1, phase 7s): every non-smooth operation a node's step evaluates is named.
  - **`mime.core.switch`**: `maximum`, `minimum`, `absolute`, `clip`, `norm` (MIME's safe norm), `where` and
    `predicate`, each with its value and JAX's derivative, ties included, and a keyword-only `owner`, `resolution` and
    `indicator`, the last two evaluated only inside a collector, so a step's program is unchanged; `domain_guard`,
    `nonzero_norm` (the safe norm of a vector zero at no valid state) and `static_select`, no switches. One context,
    set by `collecting()` or the test-only `forcing()` and `replacing_fallbacks()`, read while a step is traced.
    `Declaration` declares a switch (its owner, its indicator with units, its orders per output group, a time
    switch's settings, its route, and why no output can show it where none can); `rounding(scale)` gives a
    resolution. The module imports only JAX, jaxtyping and the standard library.
  - **Every node site written through it**, each step's jaxpr equal to the merge base's in every switch configuration
    (`tests/saved/step_jaxprs.py`), with 21 owners: contact's 8 (evaluated in the body's step), drag's 9, the swim
    speed's 2 and the cylinder's 2 (declared beside the shape, `register_shape(..., switches=)`, inherited by `Tube`).
    `MimeNode.fresh_update` traces a step afresh, for what reads a switch context.
  - **The switch scan** (`mime.docs._switches`; `tests/saved/test_switch_scan.py`): its scope is the functions JAX
    records for the equations of every registered node's step, traced in script A's arms and the round-trip cases; in
    it, a raw non-smooth operation (called or passed), an unguarded square root or power of a half, a comparison cast
    to a number or in arithmetic, `stop_gradient`, `custom_jvp`, `custom_vjp` (decorators, calls, through
    `functools.partial`, or among a module's statements) and a switch function inside a transform fail unless listed
    in `docs/ratchets/raw_switches.txt` (seven sites: the rotation's exponential, the cylinder's in-band normal, the
    rotating field's unit vector and drag's clamp flag); every owner passed is declared, by keyword, and every
    declared owner passed, a lab's under its package's name. It lists every domain guard and nonzero norm in scope,
    and says first when JAX's source information did not hold.
  - **The switch harness** (`mime.testing.classify`, `fallback_problems`; `tests/saved/test_switch_harness.py`): each
    of MIME's owners, forced each way a quarter of a step from its switch, measures within its declared order (a
    kink's forced-branch difference shrinks with the step, a jump's does not), its indicator at each placed point
    being that point's offset, or measures nothing where its declaration says no output can show it; the kink limit
    is measured, with its resolution; the rounding floor is each output's own; replacing every domain guard's
    fallback changes nothing where its condition is false, every guard the scan lists is reached so, and every
    nonzero norm away from zero.
- The documentation's second half of the first phase (dev1), in `mime.docs` and CI:
  - **the docs walk** (`mime.docs._walk`), over a docs root it takes as a parameter: a hand-written source must exist
    for each entry of MIME's (`docs/entries/<kind>/<name>.md`, naming in its front matter the version it describes,
    which must be the registered one), each target kind (`docs/mechanisms/<kind>.md`), each node of MIME's an entry
    builds (`NodeMeta` and a symbol table) and each packaged asset (`docs/assets/<name>.md`); every page source's id,
    and every link to one, must name something that exists. `toy` is exempt by namespace; a lab's entries and nodes
    without a source are counted, not failed, unless the lab opts in; a walk that finds no entry of MIME's fails;
  - **the ledger**, `docs/unpaged.txt`, generated from the registry: the 28 sources that merged before the walk
    without them (12 entries, 6 target kinds, 7 nodes, 3 assets), each with when it is paid. It only shrinks: a
    missing source not listed fails, and so does a listed item that has its source now;
  - **the number rule** (`mime.docs._numbers`): no measured number in a hand-written page source or a package README,
    a line carrying a number token failing unless listed, keyed by its file, its number tokens in order and its
    occurrence index, so a wording edit keeps an entry and a changed, added, removed or swapped number, or a copy of a
    listed line, is a new key. `docs/ratchets/number_allowlist.txt` (a line with its own reason, empty today) and
    `docs/ratchets/readme_numbers.txt` (the READMEs' frozen start, one shared reason, only shrinking) are the lists;
    `docs/number_idioms.txt` exempts `2D`, `3D` and `6 x 6`, each standing alone; code fences are exempt, and an
    unclosed one fails; so are MIME's identifiers by prefix (`MIME-NODE-`, `MIME-ANO-`, `MIME-VER-`, `MADD-ANO-` and
    three digits) and a numbered reference (`#146`), a release the code reads after its product's name (the version,
    the pins and the installed MADDENING and OpenMaterica, printed), a phase name of a digit with at most a letter and a
    digit, and an ordered list's number where it counts its list; a heading's number is a token. What one token is
    (superscript exponents, vulgar fractions and a middle-dot decimal among them), and the same-token groups, are
    printed with every run. `mime docs check --base <revision>`, which CI runs against `origin/v0.3`, fails on a start
    list line the base's list does not hold. A warning, never a failure, names each number spelled in words before a
    unit (`cm`, `m`, `s` and `N` among them), "per cent", "times" or "fold";
  - **the confidentiality scan**, `scripts/check_confidential.py` and `scripts/confidential_digests.txt`, OpenMaterica's
    copied: CI's new `confidential` job checks every tracked file's path and content, what each commit of the push or
    pull request adds (its paths and added lines, `--diffs`) and its message, and the branch's name, for the
    maintainer's private names (by keyed digests, the names never committed) and private paths. Without its key it
    fails, so it cannot pass having checked nothing, except on a pull request from a fork, told apart by the event's
    payload, which checks the paths and warns;
  - `tests/test_ci_wiring.py`: the job names v0.3's branch protection requires each made by one job, the default
    lane's `mime docs check --base origin/v0.3` and the `confidential` job's scan, with the key and its ranges, each
    run with its exit status intact (the step's script run in bash with its commands stubbed), and a planted break in
    a copy of the workflows for each;
  - the `new-node` and `commit-and-push` skills rewritten for 0.3: `NodeMeta`, the symbol table and its lists, the
    next free id from `MIME-NODE-104`, the entry's fields and hand-written file, `mime docs check`, the scan.
- The documentation's machine-readable sources and their checks (dev1, the first documentation phase), in
  `mime.docs`:
  - `mime registry --json`, the registry export: every registered entry, MIME's and each installed namespace's, with
    every `ClassVar` and dataclass field it declares, each with its type, default and value, taken in a fresh process
    and runnable on a base install (a namespace that needs the `verify` extra is listed as not loaded); a completeness
    test fails on a value an entry class holds that the export does not carry;
  - four entry fields the pages are generated from, declared for every registered entry: `builds` (each rung's node
    classes, checked by the node walk against what each entry's case builds, a rung no case builds listed as
    declared, unchecked), `gradient` (a `GradientRecord`: where the derivative is not exact, why, and the route to
    `exact`; a `cut` or `unverified` entry must state its reason and route), `references` (bibliography keys, `()`
    for none), and `family` on interactions, which the family walk now reads
    (`FamilyCase.family` is gone). A value the code does not establish is `NOT_ESTABLISHED`, listed with its reason;
  - the symbol-table form, `mime.core.Symbol` and `MimeNode.symbols`, beside MADDENING's `NodeMeta`, and its checks:
    a row's symbols are tokens of the equations, its name and units are what the code declares, `implemented_by`
    names a function where it is defined (the node's class or a base, or a module), every equation symbol outside a
    braced subscript has a row, pinned rows stay (a floor), algorithm ids are unique and a 0.2 line's id is taken
    only by the class the frozen list keeps it for (`docs/data/node_ids_0.2.yaml`), `hazard_hints` stays empty. No node carries them yet; the checks run on
    test-only nodes;
  - the refusal table, `mime/build/refusal_table.txt`: every code MIME constructs with the classes it is refused
    in, generated from a scan of the source and checked against it. A `Refusal` with a MIME code not in it, or in a
    class it does not give, raises `UncataloguedRefusal`; a code in a registered namespace is accepted, uncatalogued,
    and a problem code an entry's plan returns that it cannot be refused with (a lab's outside its namespace, or a
    MIME entry's not in the table) is refused as the entry's defect. A code on two rows of the table, and a code
    built where it is used (an f-string), fail; `register_namespace` refuses a name that is the first word of a MIME
    code (`load`), which would take those codes out of the check;
  - the ratchets as lists, `docs/ratchets/<name>.txt`, one `item  # reason` per line, each length derived and
    printed: the refusal codes no test names, the code-shaped names the scan's cross-check allows, the pinned symbol
    rows, the symbol indices, the symbol rows whose units are declared and not checked, the rungs no case builds, and
    the values not established;
  - `mime docs check`: every check above over a checkout, with `scripts/check_citations.py` and
    `scripts/check_anomalies.py` unchanged; it builds no graph, and CI's default lane runs it.
- Materials from OpenMaterica's records (dev1, phase 5), through `mime.materials`, the only importer of `materica`:
  - `Experiment(conditions=...)`, the operating conditions every record is read at, each with its tolerance
    (`degC(20, plus_minus=0.5)`, `mime.units.Condition`);
  - `mime.materials.Record`, a material read from an OpenMaterica record file, pinned by the SHA-256 of its bytes
    (`record_file(path, sha256=...)`, or `packaged_record(name)` for a copy MIME ships) and by the record's id,
    version and content hash; with the tokens it requires, its acceptances of reads OpenMaterica answers `unknown`
    (each naming the reasons it covers, with its reason), the value it uses from an interval (`Nominal`), and how it
    extends a table (`Extrapolation`, `constant`, or `linear` computed by MIME);
  - what the build does with each verdict, until profiles land in a1: `met` builds; `not_met` is refused with
    OpenMaterica's reasons; `unknown` is refused unless an acceptance covers every reason (`no_pick` always); an
    interval's bounds are read and the declared value must lie inside them; a table read outside its range takes the
    experiment's extrapolation;
  - `Medium(material=...)`, and a body's parts: `Body(parts={...})`, each a `mime.things.Part` with a solid
    (`SolidCylinder`, `SolidEllipsoid`, `SolidBox`, registered geometry kinds with closed-form volume and inertia),
    a material, a placement (`at=`, or the asset's transforms for a bound body) and the part it lies `inside`; the
    part resolver derives the body's mass, inertia, buoyant volume, centre of buoyancy and magnetic moment
    (`B_r / mu_0` times the volume, along the magnet's direction), each with its derivation in the report, and a value
    declared in the experiment takes the place of a derived or recorded one, saying so; a part's own values are
    declared at `<body>.parts.<part>.<value>`;
  - the saved form: the document's `conditions` and `materials` (each a reference: the record file, its SHA-256, the
    record's id, version and content hash, and the experiment's own terms; never a value from the record), a medium's
    material and a body's parts; the lock's resolved materials (every read as sent and answered, each record's pins,
    each part) and its record files, which `save` writes under `records/` and a load holds to their hashes;
  - copies of OpenMaterica's example records `example/water` and `example/ndfeb_n45` (CC0-1.0 values; at its tag
    `openmaterica-v0.1.0.dev3`) in `mime.data`, as `openmaterica_example_water` and `openmaterica_example_ndfeb_n45`;
  - script A's record arm (`examples/script_a.py --records`, `tests/script_a.py`'s `record_arm()`): water's density
    and viscosity at 20 degC and N45's remanence through two magnet parts, switched on one at a time, each switch's
    change in A's steady quantities predicted before the record arm existed and measured within the stated tolerance
    (`tests/differential/test_script_a_records.py`); the reproduction arm, read through the seam, is phase 4's arm bit
    for bit.
- Script A against its frozen 0.2 reference (dev1, phase 4c2), phase 4's gate:
  - `examples/script_a.py`, script A's reproduction arm as a user writes it, every number the 0.2 line's run used
    declared in the experiment with where it comes from; `tests/script_a.py`'s `reproduction_arm()` is the same
    experiment, and a test holds the two to one document;
  - the expected differences, declared and committed before the comparison first ran
    (`tests/differential/script_a_expected_differences.json`), predicted by integrating A in NumPy with 0.2's update
    rules, which reproduce the reference to rounding, and with 0.3's. The largest: 0.3's implicit resistance, with
    the body confined by projection, swims 2.35% faster at dt = 0.5 ms, a first-order error in dt (the solve
    resists the outward velocity of a fall the projection then removes) that halves with dt and that holding the
    contact inside the solve removes; the 0.2 reference's commands were float32 inputs (-4.3e-8 on its rates);
    MIME-ANO-110 and the field's timekeeping are absent from this reference;
  - the comparison in float64 (every gated quantity on its prediction to 1e-14) and float32, with its controls (the
    +10% frequency reference, half the step), the two graphs' lag structures compared (the torque reads the
    orientation at the start of the step in both), and one post hoc item: the float32 phase lag (MIME-ANO-113);
  - Hypothesis stateful tests of the `Experiment` API (`tests/saved/test_experiment_states.py`): a build succeeds or
    refuses with reasons, `load(save(x))` keeps the derived hash, a refused build leaves the experiment unchanged;
    planted controls for the last two.
- The rigid-body library, `mime.core.multibody` (dev1, phase 7a1), ported from the 0.2 line's `control/kinematics/`
  and moved below the nodes, which will import it:
  - one canonical tree, `MultibodyTree`: bodies with inertials, revolute, prismatic and fixed joints with axes,
    limits, damping, friction and armature, named frames (every body's own and extra ones, to mount parts on), loop
    closures held as data (`LoopClosure`) and refused by every algorithm, naming them (`ClosedChainRefused`, which
    says closed chains are out of scope until someone needs one and how to register one's own chain segment), and the
    description file's SHA-256; compiled once into an `Articulation` that welds fixed joints, keeps every frame's
    offset, and whose arrays are JAX pytree leaves;
  - the URDF reader, `read_urdf`: every element read and used, recognised and not used (visuals, collisions,
    materials, transmissions, `gazebo`, `ros2_control`, calibration, safety controllers), or refused by name (mimic
    joints and elements URDF does not define, which can be read past with an acknowledged reason; floating and planar
    joints and unexpanded xacro, which cannot), one `DescriptionWarning` per kind with what MIME does instead and how
    to ask, the classification returned, a link with geometry and no `<inertial>` recorded as an `InertiaGap`, and a
    reading pinned to an expected hash refusing any other file;
  - the maths in pure JAX (`jit`, `grad`, `vmap`, finite at zero rotation, float32 in and float32 out): forward
    kinematics to every frame, frame Jacobians, damped-least-squares inverse kinematics that reports an unreached
    target or a solution outside the limits rather than clamping, the mass matrix by the composite-rigid-body
    algorithm, inverse dynamics by the recursive Newton-Euler algorithm, and forward dynamics;
  - tests: oracles with a control each (the planar arm's closed form, Jacobians against `jax.jacfwd` and central
    differences, the mass matrix against the Hessian of the kinetic energy, gravity against the potential's gradient,
    the velocity terms against the Christoffel symbols, inverse against forward dynamics, inverse kinematics), the
    reader's three groups and messages, loop closures refused by every algorithm, and a differential against the 0.2
    port's library on the AR4, within 11 float64 roundings.
- Drag (dev1, phase 4c1):
  - `interactions.Drag` (`drag` 0.1.0, rung `surrogate`, alias `Drag.Surrogate`, which saves identically) with
    `interactions.Lubrication`: the 0.2 line's Cholesky MLP drag surrogate and its near-wall lubrication correction,
    ported as one model behind the new contracts, with three fixes. The tube's axis, a point on it and its radius come
    from the region the medium fills, through the shape protocol, not from a +z tube (the 0.2 line clamped every step
    of a tube along x); a medium filling no region, or one with no axis, is refused; the 0.2 line's callable vessel
    radius is gone. The value and its derivative are finite on the axis, the frame built from vectors, the body's
    own axis across the tube taking the place of `arctan2(0, 0)` there. The surrogate's training domain is declared
    data (`mime.surrogates.SURROGATES`): `nu` from 0.25 to 3.5, the training data's (the 0.2 line declared 2.33 to
    3.7), and `kappa` from 0.25 to 0.66, each checked at plan time (`drag.outside-training`); the offset keeps the 0.2
    line's clamp and reports `clamp_fired`. It reads the body's pose at the start of the step and delivers the
    resistance and the background load;
  - `mime.surrogates`: the Cholesky MLP in plain JAX and the packaged surrogate's training domain; `mime.nodes.hydro`:
    `SurrogateDragNode`;
  - `mime.data`: the surrogate's weights as package data (`mlp_cholesky_weights_v2.npz`, sha256 pinned), committed
    and shipped in the wheel and the sdist, and `mime.data.asset(name)`: a packaged asset by name and sha256, whose
    contents are read only if the hash matches (`mime.asset`, phase 6's, names a file by its path); the saved form writes an asset by name and hash, the
    lock records its source and hash (`resolved.assets`), and a load refuses other contents under the name;
  - groups of settings (`mime.registry.SettingGroup`, `register_setting_group`): written in a script as one value,
    handed to the entry in SI as `<setting>.<field>`, saved by their group name;
  - `units.DYNAMIC_VISCOSITY`, `Pa_s`, `mPa_s` and `dimensionless`; a medium's `viscosity` and a body's `radius`,
    `length` and `helix_wavenumber` among the values a thing has; an entry's plan may refuse what it is bound to
    (`Plan.problems`) and read the region a bound medium fills (`PlanContext.region`); a geometry kind names the
    shapes of its regions (`region_shapes`, the tube's `lumen`), and the shape protocol gives a point on an axis;
  - the hydrodynamic family (`mime.testing.HYDRODYNAMIC`): sign (the resistance symmetric positive definite), frame
    (turning the tube and the body together, and moving along a straight tube's axis), units, material reach (in
    proportion to the viscosity), oracle, gradients, swap and nodes; equal and opposite does not apply;
  - the family harness: rounding floors per output field and per part of a composite (a pose's position, an
    operator's block), inputs nudged and differentiated on the scale of their own part; a family's constants held by
    some members only, displacements its setup is uniform along, and laws in proportion to a material; the contact
    family's planted-error control tightened from 0.1 mm to 70 nm, which it catches in every run;
  - a Hypothesis profile in `tests/conftest.py` printing the blob that
    reproduces a failure in every run, otherwise the profile Hypothesis would load (its `ci` profile on CI);
  - tests: the drag's oracles and controls (`tests/oracles/test_drag.py`), a test-only Stokes sphere as the family's
    analytic oracle and second member, the node-level differential against the 0.2 port's drag
    (`tests/differential/test_port_drag.py`), and the weights' presence in git and in both distributions.
  - `docs/validation/known_anomalies.yaml`: MIME-ANO-112, the surrogate's resistance is about the end of the body's
    axis and is applied about its centre of mass; MIME-ANO-114 (phase 4c3), the surrogate's accuracy against the BEM
    it was checked against, up to 16% in swim speed against the 0.2 line's stated 1%.
- The scene (dev1, phase 6): MIME writes its scene as a USD stage, Python-first. No experiment computes a different
  number:
  - `Scene.new(up=..., units=...)` and `Experiment(scene=...)`: up is declared once, on the scene, and gravity is
    `-g * up`. `Experiment(up=...)` is refused (`frame.up-on-experiment`, or `frame.up-twice` beside a scene); a
    missing scene or up is refused naming up (`frame.up-missing`);
  - the stage writer in `mime.scene`: `save` writes `scene/scene.usda` beside the document and the lock, with the
    declared up axis (+Y or +Z, the two USD names; see Changed),
    `metersPerUnit` from the scene's units, a `/World` root set as `defaultPrim`, and one prim per thing (`/World/<name>`
    for a thing written Python-first) carrying `mime:thing` and `mime:name`. A geometry kind's `drawing` shows it (the
    tube as its wall, an annular mesh) with its settings in SI as `mime:geometry:<setting>`; a body stands at its
    initial pose. Mesh points stay local and near the origin, placed by double-precision transform ops. The same
    declarations write the same bytes, and the lock records the stage's SHA-256 (`stage`);
  - `scene.reference(asset(path), at="/World/...")` references a robot asset layer pinned by SHA-256 (refused when
    the bytes differ from a given pin, when it sets no `defaultPrim`, authors no or another `metersPerUnit`, or
    composes other files); `save` writes its bytes to `scene/layers/`, and a load refuses a changed asset
    (`load.asset-changed`) or a hand-edited stage (`load.stage-changed`);
  - `Body(..., prim=scene["/World/UMR"])` binds a body to a prim an asset brings in: two names per thing, the
    document's defaulting to the prim's last element; a duplicate name or prim is refused;
  - the `mime:` applied schema for design data on part prims: `MimeMagnetAPI` in the prim's `apiSchemas` and
    `mime:magnet:direction`, a magnet's magnetisation direction in its part's frame. `Body(magnets={"Magnets":
    PermanentMagnet()})` gives the part its role; the build reads the direction, turns it into the body's frame, and
    a body's moment is then declared as a magnitude along it. Declared in both places, a direction is refused
    (`magnet.direction-twice`);
  - frontier rule 1: every thing's prim path is saved beside its name, in the document and the resolved record;
  - frontier rule 2: a reference to the geometry of a thing with a dynamics (a surface, a direction) is refused
    unless the referring entry declares `moving` (`geometry.moving-read`), naming the entry, the thing and why;
  - the build report's scene line and one orientation line per thing, a direction given by reference written as
    the path it refers to beside the vector it resolved to;
  - `mime.scene.check_stage`, usd-core's validators (its `UsdValidation` framework) over a written stage;
  - the lock's environment records usd-core's version;
  - tests: the stage read back with pxr and its planted control (no `defaultPrim`), the validators, determinism,
    the asset and its schema, the saved scene and its refusals, the geometry rule, the frame proposals, a round-trip
    case for a body bound to an asset (`bound_robot`), the rotated world (`tests/scene/test_rotated_world.py`: the
    body side in a world turned by a drawn rotation runs as the first world turned, within 16 eps per step; a law
    with +z in its code fails it), and the rule that only `mime.scene` and `mime.viz` import `pxr`.
- The body side of the first experiment (dev1, phase 4b):
  - `mime.core`: `MimeNode` (every constructor argument in `params`, the step written as `advance` and run compiled
    once per node), the quaternion helpers with an exponential whose value and derivatives are finite at a zero
    rotation, the `Shape` protocol and `Cylinder`, and the confinements' velocity rule `remove_outward`;
  - `mime.scene.Tube`, a registered geometry kind (axis, inner radius, wall, length, centre) with typed paths
    `tube.axis`, `tube.inner_surface` and `tube.lumen`; `Structure(geometry=...)`, `Medium(fills=...)`; settings
    marked `Direction` refer to a thing's direction (`rotation_axis=tube.axis`);
  - values declared in the experiment, `exp.declare("umr.mass", kg(...), reason=...)`, read through `mime.materials`,
    the one place that reads a thing's values, and labelled "declared in the experiment" in the report and the saved
    form;
  - `dynamics.RigidInertial` (`rigid_inertial` 0.1.0): implicit resistance, `(M/dt + R) V' = (M/dt) V + F`, with no
    explicit setting, and confinement by projection in its own update;
  - `interactions.Contact` (`contact` 0.1.0, rung `projection`, alias `Contact.Projection`; `penalty` reserved),
    `interactions.Gravity` (`gravity` 0.1.0, weight minus buoyancy and its couple), `interactions.Magnetic.Torque`
    (`magnetic.torque` 0.1.0, `tau = m x B`);
  - `fields.UniformRotatingField` (`uniform_rotating_field` 0.1.0), its time by step count, and `mime.drives` with the
    constant drive (`drive.constant` 0.1.0); an apparatus's `commands`, a quantity being short for the constant drive;
  - `observations.SwimSpeed` (`swim_speed` 0.1.0);
  - MIME's vocabulary for a body and the magnetic field, and `Capability.UNIFORM`; the registry's drive and geometry
    kinds, `Value` reads, `Confinement`, `Command`, reserved rungs, and entries judging their own settings;
  - lag accounting: every edge classified as reading this step, the previous step or the current iterate, from
    MIME's own plan and canonical order, shown in the build report and the graph snapshot, cross-checked against
    MADDENING's compiled schedule and `validate()`'s loop lines; an unplanned feedback loop refused
    (`lag.unplanned`); `latency_in_steps` for a controller's realised latency;
  - the magnetic, gravity and contact families as conformance suites (`mime.testing.MAGNETIC`, `GRAVITY`, `CONTACT`),
    the harness reading declared values, turning composites in the frame case and probing a confining law through
    its body's node; the family walk (every registered interaction passes its family's suite, or CI fails);
  - tests: the physics oracles with their controls (`tests/oracles/`), golden outputs per registered name and
    version, node cases and round-trip cases for every new entry, and node-level differentials against the 0.2 port
    (`tests/differential/`).
- The infrastructure for the physics (dev1, phase 4a):
  - test durations on every CI lane: each lane's tests run on pytest-xdist workers and write JUnit XML, and a
    `Test durations` step writes the bands, the slowest tests and files and the wall and CPU totals to the run page
    and holds MADDENING's budget (a warning over 5 s, a failure over 20 s once re-run alone, an allowlist with
    reasons in `tests/duration_allowlist.txt`). `scripts/report_test_durations.py` is a copy of MADDENING's at
    v0.4.2 with OpenMaterica's CPU columns, every change listed in its header; `tests/conftest.py` records each
    test's own and its children's CPU time. The `plant_slow_tests` input of a manual run is the budget's control;
  - units in annotations: an entry's dimensioned setting declares its dimension once, `Annotated[Quantity,
    Length]`, with markers in `mime.units`; registration reads them (`mime.registry.dimensioned_settings`), phase 1
    refuses a setting of another dimension (`setting.units`) naming the entry, the setting and both dimensions, and
    an entry reads its settings in SI from its context (`ctx.scalar`, `ctx.vector`);
  - typed node state: `mime.core.NodeFields`, `typed_view` and `node_fields`, by which a node types its state and
    inputs as closed TypedDicts of jaxtyping arrays inside MADDENING's own signatures; the convention is in
    `nodes/README.md`, and the toy nodes and the lab fixture's node follow it;
  - the family suites' harness in `mime.testing`: `Family`, `register_family`, `conformance` and `assert_conforms`
    with the case kinds sign, frame, units, material reach, oracle, gradients, swap, equal and opposite, and nodes,
    the sampled ones as invariants of MADDENING's `verify_node`; the node battery (`node_failures`,
    `assert_nodes_verified`, `NodeCase`, `OrderLadder`: `verify_node` on every node an entry builds, and
    `verify_node_order` on each order a node declares); the toy family (`mime.testing.toys.TOY_FAMILY`) as the
    control; and the node walk, which requires a node case for every registered entry;
  - `BuiltExperiment.plans`, phase 1's result, for inspection.
- The saved form on the toys (dev1, phase 3):
  - `mime.saved`: the document (`experiment.mime.yaml`: registered names and versions, settings written differently
    from their defaults, bindings by name, initial states, the run settings and world frame, and empty slots for
    materials, the scene and the controller), read with PyYAML's safe loader and hashed over canonical JSON; the
    lock (`mime.lock`: the environment, every entry with its distribution and version, derived seeds, hashes, the
    resolved record and the derived graph); the reference outputs (`reference.npz`); both files as closed TypedDicts,
    with a run-time check driven by the same annotations;
  - `Experiment.save`, `Experiment.load`, `Experiment.from_document`, `Experiment.document`,
    `BuiltExperiment.save` and `BuiltExperiment.verify`: a load finds entries through installed `mime.entries` entry
    points only, refuses a missing entry or another version with the line to install, rebuilds, and checks the input
    hash exactly, the environment by report, and the derived graph hash for hash within one environment and within
    tolerance across environments;
  - `mime verify <folder>` (`mime.cli`), and the `mime` command;
  - the derived hash: MADDENING's description of the built graph plus every static array's contents, observations
    apart from the physics;
  - units labels on every edge from the vocabulary, `expected_units` on every toy input, and a post-build check
    refusing a missing or mismatched label or an input with no units;
  - MADDENING's `validate()` and `compile()` warnings in the build report with their source, a units warning as a
    refusal;
  - the run's structure in the build report ("one compiled scan" for every experiment so far); a probe that traces
    the compiled step once at build and refuses a host callback inside it, naming the node;
  - `mime.testing`: `round_trip_failures`, `refused_round_trip_failures`, `installed_entries`, `covered`,
    `load_in_fresh_process` and `param_reach_failures` (every trainable leaf reaches the compiled step without a
    recompilation);
  - tests: the registry walk in the default lane (an entry or kind of thing without a round-trip case fails CI), with
    its controls; the document, load, schema and compiled-step tests; a packaging test that `py.typed` ships in the
    wheel; the typing gate's own tests;
  - typing: `pyrightconfig.json` (Python 3.12, strict over `src/mime`, `reportDeprecated`,
    `reportImplicitOverride` and unnecessary ignores as errors), `scripts/typing_check.py`, and a blocking
    `typecheck` CI job that refuses an untrustworthy run, plants a type error as its control, and type-checks a
    consumer snippet against the built wheel, with and without `py.typed`; `@override` on every overriding method.
- The core of 0.3 (dev1, phase 2), tested on toy entries with no physics:
  - `mime.units`: SI quantities and their constructors (`mm`, `ms`, `Hz`, `mT`, ...), checked by dimension in phase 1
    and turned into plain SI floats;
  - `mime.vocabulary`: interface entries with kind, units, frame, shape, sign, dtype and typical magnitude, versioned
    per namespace; the six target kinds; the closed list of capability flags; reserved names, refused as "reserved,
    not built";
  - `mime.things`: `Structure`, `Medium`, `Body` and `Apparatus` as declarations; `State`; handles, and typed paths
    to an apparatus's fields;
  - `mime.registry`: `register_interaction`, `_observation`, `_dynamics`, `_field_provider` and `_controller`,
    keyed by (name, version); entries as frozen dataclasses of their settings; plans as data; support tables as
    predicates over capability records, with no default rung and no default binding;
  - `mime.build`: phase 1, which resolves handles, typed paths and strings (with "did you mean"), checks reads,
    targets, rungs, rates and registrations, merges solve-together requests into coupling groups, derives ids and
    seeds, and refuses with every reason at once, each with a class and an action line; phase 2, which adds nodes in
    the canonical order (providers, interactions by id, media, bodies, apparatus, then observations) and edges by
    delivering id through restricted builders that set `additive` from the target's kind, and every coupling group
    with `solver="ift"`, `atol=0` and an explicit tolerance and cap; the post-build checks; the build report; the
    world frame as a stand-in until the scene lands;
  - `mime.experiment`: `Experiment` (`add_*`, `attach`, `observe`, `plan`, `build`; `dt` as the base step with no
    default, `rate=` a whole multiple of it) and `BuiltExperiment` (report, graph text, `run`);
  - `mime.gradients.GradientStatus`, which every entry declares;
  - `mime.testing`: the graph-invariant helpers (graph text, order invariance) and `mime.testing.toys`;
  - tests: the canonical order against a hand-assembled reference over every order of the calls, with its controls;
    the order-invariance property in Hypothesis, with its control; a mutation test per post-build check; golden
    graph text; pins of the MADDENING 0.4.x behaviours the build works around;
  - `lab_fixture/`, a package outside `src/mime` that registers an interaction from outside MIME; every CI lane
    builds and installs it.
- The 0.3 package skeleton: the subpackages of the 0.3 layout, each empty with a line stating its role; `py.typed`;
  `mime.testing` on `maddening.testing`, needing the new `verify` extra.
- Packaging for 0.3: version `0.3.0.dev0` (single-sourced from `mime.__version__`), Python 3.12 to 3.14,
  `maddening>=0.4.1,<0.5`, `jax>=0.10,<0.13` (MADDENING 0.4.1's own range), and `openmaterica>=0.1.0.dev3` and
  `usd-core>=26.5` as required dependencies; the `verify` and `dev` extras; a committed `uv.lock`.
- CI on uv with four lanes, each checking from the installed metadata that it installed what it claims: the locked
  PyPI releases on Python 3.12 to 3.14 (blocking); OpenMaterica's `main`; MADDENING's `main`; the newest usd-core,
  weekly. A manual run's `plant_failure` input fails a canary test in every lane.

### Changed
- Development environments use Python 3.14: `.python-version` steers `uv sync` and `uv venv`, which otherwise take the
  newest Python installed, and so failed on a machine with Python 3.15, for which usd-core ships no wheels. The
  package still supports Python 3.12 to 3.14, and CI still runs each of them.
- The canonical order gathers each coupling group's members at the place of its last member, so a law added between
  a group's members runs before the group: MADDENING 0.4.x runs a group as one block at its first member's place,
  and the full toy's body read the spring's load a step late. The full toy's node order and history change.
- The build report lists the values read and every edge's lag, and the graph snapshot each edge's lag; the toys'
  golden snapshots are regenerated.
- A delivery into an inequality-constraint target is an entry defect: a confinement is planned as a `Confinement`.
- MIME is licensed under the Apache License, Version 2.0, from the 0.3 line on, as MADDENING is from 0.4.2: `LICENSE`
  holds the Apache-2.0 text, `NOTICE` the copyright and the note that the 0.2 line (the tag `pre-0.3` and earlier)
  was distributed under the GNU Lesser General Public License, version 3 or later; `pyproject.toml` declares
  `license = "Apache-2.0"` and `license-files` (PEP 639) with no licence classifier; `CITATION.cff`, the README and the
  lab fixture say Apache-2.0; `COPYING.GPL-3` is removed. `CONTRIBUTING.md` states the licensing of contributions
  (inbound equals outbound) and a DCO sign-off on each commit, documented and not yet enforced.
- MIME requires MADDENING 0.4.2 (`maddening>=0.4.2,<0.5`).
- The toy drag is version 0.2.0: its quadratic rung takes the velocity's norm (with a derivative of zero at rest), so
  turning the world turns its load; 0.1.0 took each component's magnitude. The toys' settings are quantities with
  declared dimensions, at the SI values they held as plain numbers.
- `tests/build/test_canonical_order.py` runs 100 steps, which the hand-built control needs to tell the summation
  orders apart with the toy drag at 0.2.0.
- The build's host-callback probe traces each node alone through MADDENING's public node interface (its state, the
  inputs `resolve_boundary_inputs` gives it, `update` and `compute_boundary_fluxes`) instead of the whole step
  through MADDENING's private step function. Nothing private is read, the probe no longer needs the compiled step,
  and a callback in a node's flux hook is now refused too.
- Development tooling moved from the `dev` extra to a PEP 735 `dev` dependency group (pytest, pytest-xdist,
  Hypothesis, `pyright==1.1.414`, hatchling); the `verify` extra stays for labs. New direct dependencies:
  `typing_extensions>=4.16.0`, `jaxtyping>=0.3.11` and `pyyaml>=6.0`.
- The toy nodes name their initial conditions `initial_*`, which MADDENING takes as initial conditions rather than
  step parameters; the toy field holds a static phase table.
- A built graph's canonical text shows each edge's units labels.
- `RunSettings` holds the script's values as written (`Any`), since phase 1 is what checks them.
- `scripts/check_anomalies.py` checks the registry with its own check (top-level fields, unique prefixed IDs,
  required fields), the one the 0.2 line's CI ran: MADDENING 0.4's validator does not accept the registry's 0.2
  schema.

### Known Anomalies
- MIME-ANO-110: the 0.2 permanent-magnet response computes its moment in float32 even under x64 (found by the
  node-level differential; 0.3's torque does not).
- MIME-ANO-111: the 0.2 vessel constraint removes only `(R/r)^2` of the outward velocity of a body it projects back
  (found by the node-level differential; 0.3's projection removes all of it).
- MIME-ANO-113 (open, minor): `RigidInertial`'s implicit step rounds a load far below the inertial term in float32,
  which a locked body keeps as a fixed phase-lag offset (2.3e-7 rad in script A, bound 4.5e-7); solving the same step
  for the change in velocity removes it. Planned as a new version of the rung (found by phase 4c2's comparison).
- MIME-ANO-110's reach: in a graph run on MADDENING 0.4.x the 0.2 port's moment is float64, so script A's frozen
  reference does not carry the float32 rounding; the tag's run on 0.3.1 does.

### Fixed
- The differential tests' reference provenance files (`tests/differential/references/`) carried absolute home,
  scratchpad and planning-directory paths, which the confidentiality scan refuses; each is now written relative to the
  workspace (`MIME-port041/...`) or names the private directory without its path, and no value a test reads changed.
  The port's reference generators record the port by its name, and their command line through
  `tests/differential/_provenance.py`, relative to the workspace, with the planning directory by its role.
- `.gitignore` anchors `build/` and `dist/` at the root; unanchored, they ignored `src/mime/build/`.
- pytest collects `tests/build/`: its default `norecursedirs` skips any folder named `build`, so `pyproject.toml`
  lists the default folders without it, and `tests/test_collection.py` fails if a test file is ever left out.
- Building an experiment that builds no node raised MADDENING's `RuntimeError` ("No nodes registered."); it is now
  refused as `run.nothing-to-step`, with what to declare. Found by the stateful tests (phase 4c2).
- CI no longer derandomises MIME's property tests, and the contact family's sign case no longer fails a correct
  projection. `tests/conftest.py`'s profile turns off the `derandomize` that Hypothesis's `ci` profile turns on, and
  keeps the blob and the absent database. The sign rule took the reached speed's norm in float32, where the square of
  a speed below about 3e-23 m/s underflows to zero, and so demanded an outward velocity of exactly zero; it now judges
  in float64, with a floor of eight of the dtype's smallest normal number, below which XLA flushes the projection's
  arithmetic to zero. Drawing anew on CI then found the rotation's exponential (`mime.core.quaternion.exponential`)
  with a derivative off by angle^3/96 below 1e-4 rad, 1e-14 at the threshold, where its series stopped at the square
  of the angle; it now carries the fourth power, which leaves every value bitwise unchanged.

## [0.2.0] - 2026-06-11

Brings MIME into full alignment with **MADDENING v0.2.1** and lands the two
pieces of new architecture the v1.0 stack roadmap puts on the critical path:
**multi-GPU sharding of the IB-LBM fluid node** and the **EffectModel contract
pilot**. Adopts the v0.2 node-API surfaces (`halo_width()`, the `static_data`
channel, compile-time edge validation) and folds in the TF32 / float32
numerical-precision fixes. Narrative summary in
[`docs/release_notes/v0.2.md`](docs/release_notes/v0.2.md).

### Added
- **IBLBM multi-GPU sharding** — `IBLBMFluidNode.update_padded` now has a
  working multi-device path (previously `NotImplementedError`): halo-aware
  streaming (`d3q19.stream_padded`), per-slab pipe + UMR missing-link masks
  (`bounce_back.compute_missing_mask_sharded`), the UMR body rebuilt on each
  slab's global coordinates (an `origin` offset threaded through the geometry
  helpers), and psum'd drag force/torque. Bit-identical to the single-device
  step under jit on a 4-device mesh (`tests/verification/test_lbm_sharded_contract.py`).
  Requires `maddening>=0.2.1`.
- **EffectModel contract pilot** (`mime.effects`, per ADR-2026-EFFECT-MODEL) —
  the Protocol surface (`EffectModel` / `SourcedEffectModel[S]` /
  `EffectHandle` / `CouplingSpec` / polymorphic `Regime`), a `@register_effect`
  registry, an `Experiment` with the six-pass `build()` validation contract
  (each pass a typed error) and load-time MIME-version compatibility checks,
  and the `HydrodynamicModel` family (LBM / FVM / Stokeslet / DefectCorrection
  backends adapting the existing fluid nodes over the shared drag contract).
  The MagneticModel family, `SourceInputProvider`, and cross-effect coupling
  implementation are deferred to v0.3.

### Changed
- Dependency floor raised to `maddening>=0.2.1,<0.3` (was `>=0.2.0`); CI now
  pins the MADDENING checkout to the `v0.2.1` tag. Unlocks the IBLBM sharded
  `static_data` / `domain_integral_fields` path.

### Fixed
- **`jax_enable_x64` test-isolation leak.** Seven verification modules enabled
  x64 at *import* time, which pytest runs during collection — so x64 leaked on
  for the whole session, made results order-dependent (a hazard under
  pytest-xdist), and produced spurious `float64 → float32` cast FutureWarnings
  in float32 nodes. Replaced with a `@pytest.mark.x64` marker + an autouse
  conftest fixture that scopes x64 per-test with deterministic teardown;
  `tests/test_x64_isolation.py` guards against regressions.
- **`oberbeck_stechert_coefficients` float32 cancellation.** The prolate-
  spheroid drag-coefficient denominators subtract a leading ±2e that cancels
  against `2·atanh(e)`, losing all float32 precision near e=0 (C₁=1.037 instead
  of 1.000 at e=0.01 — only masked when a test ran under the leaked x64).
  Reformulated cancellation-free via `g = atanh(e) − e` (series for small e);
  algebraically identical, validated to ≤2.6e-7 vs a float64 reference across
  e ∈ [0.001, 0.99].
- **AR4 safety-clamp test** updated for the controller perf refactor that fused
  `_update_target_from_body` into the jitted step law; it now drives the target
  filter to convergence through the public `compute()` path.

### Test infrastructure
- `@pytest.mark.x64` marker + an autouse conftest fixture scope
  `jax_enable_x64` to opting-in tests with deterministic teardown (replaces
  module-level `jax.config.update` calls that leaked into the whole session).

### Performance
- **`ar4_helical_drive` declarative actor poses.** The recorder now sources
  each arm-link and rotor/magnet pose directly from its producing node's output
  field (`pose_from` → `arm.link_poses_world[i]`, `motor.rotor_pose_world`)
  instead of re-deriving poses from per-actor `position`/`orientation` state.
  Significantly faster experiment iteration at identical physical output.

## [0.1.0] - 2026-05-11

First tagged release. Establishes the MIME package, the `MimeNode` ABC over
MADDENING's `SimulationNode`, the actuation-apparatus node decomposition
(Motor + PermanentMagnet + RobotArm), and the initial verification suite.

### Added
- Project scaffold: pyproject.toml, package structure, core modules
- `MimeNode` ABC extending MADDENING's `SimulationNode`
- `MimeNodeMeta` and all domain metadata dataclasses
- `GeometrySource` protocol with `CylinderGeometry` and `MeshGeometry`
- `USDViewport` protocol
- Known anomalies registry (empty — no anomalies yet)
- Bibliography with initial microrobotics references
- DOCUMENTATION_ARCHITECTURE.md (self-contained)
- ARCHITECTURE_PLAN.md v0.6
- MIME_NODE_TAXONOMY.md
- GitHub issue template for anomaly reporting
- Claude skills: commit-and-push, new-node, new-anomaly, new-verification-test
- **Actuation decomposition** — three new external-apparatus nodes
  introduced as a Motor + PermanentMagnet + RobotArm chain that
  replaces `ExternalMagneticFieldNode` whenever spatial field
  variation, finite-magnet near-field, or apparatus-mediated effects
  matter. The legacy node stays as a first-class peer for
  uniform-Helmholtz cases.
  - `MotorNode` (`MIME-NODE-100`) — single-axis rotary motor (DC
    brushed) with torque / voltage / velocity command modes.
  - `PermanentMagnetNode` (`MIME-NODE-101`) — bar-magnet field
    producer with three field models (`point_dipole`, `current_loop`,
    `coulombian_poles`). Drop-in for the existing
    `PermanentMagnetResponseNode` consumer.
  - `RobotArmNode` (`MIME-NODE-102`) — URDF-driven articulated
    rigid-body manipulator. FK / CRBA / RNEA / forward dynamics, all
    pure JAX, jit/grad/vmap traceable.
- New domain metadata: `MotorMeta`, `ArticulatedArmMeta`, plus
  `ActuationPrinciple.MOTOR_ROTOR` and `ActuationPrinciple.ARTICULATED_ARM`.
- `mime.control.kinematics` package (URDF parser, SE(3) helpers,
  6D spatial algebra, FK / CRBA / RNEA) — patterns mirrored from
  `tmp/frax`, no frax import.
- `mime.experiments.dejongh_new_chain.build_graph` — dejongh graph
  driven by Motor + PermanentMagnetNode (with body↔magnet coupling
  group per dejongh deliverable A.2).
- `MICROROBOTICA/experiments/_template_helical_drive/` — generic
  template for any URDF arm.
- `MICROROBOTICA/experiments/ar4_helical_drive/` — AR4 (Annin
  Robotics) demo bundled as the default-arm working example.
- `scripts/urdf_to_usd_scene.py` — author-time helper that derives
  a `world.usda` link hierarchy from a URDF.
- `scripts/run_ar4_helical_drive.py` — standalone runner for the AR4
  experiment that loads `physics/`, builds the graph, and writes
  per-frame JSON-lines matching the MICROROBOTICA `ResultFrame`
  schema. Useful for iterating on the simulation without launching
  the Qt viewer; runs on GPU when available.
- `tests/conftest.py` — sets the GPU env vars and JAX persistent
  compile cache before any test module imports JAX. Three knobs
  matter on this rig:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
    `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` so cuSolver's handle
    creation (called by `RobotArmNode.update` via `jnp.linalg.solve`)
    succeeds without JAX preallocating the whole device.
  - `XLA_FLAGS=--xla_gpu_autotune_level=0` skips XLA's per-shape
    autotune; for our tiny matrices the autotune cost dominates the
    runtime cost.
  - `jax_compilation_cache_dir` (default
    `~/.cache/jax_compilation_cache`) — persistent on-disk cache
    for compiled XLA executables. First run still pays the full
    compile, subsequent runs load the artefact from disk.

### Changed
- `mime.core.metadata.MimeNodeMeta` now carries optional `motor` and
  `articulated_arm` fields (additive — no breaking change).
- `MimeNode.validate_mime_consistency` now cross-checks
  `MotorMeta.commandable_fields` and `ArticulatedArmMeta.commandable_fields`
  against `ActuationMeta.commandable_fields` when both metas are set.
- `MotorNode.update` now also stores `rotor_pose_world` in state so
  downstream edges read it as a state field (matches the convention
  used by `ExternalMagneticFieldNode` for `field_vector`).

### Deprecated
- *None.* `ExternalMagneticFieldNode` (`MIME-NODE-001`) **stays as a
  first-class peer** — it remains the right choice for uniform-
  Helmholtz workspace simulations.

### Performance
- **Standalone runner XLA-compile speedup**:
  - Before: 678 ms/step steady-state, 55 s pre-warm (cold cache).
  - After: 76 ms/step steady-state with `--no-coupling-group`,
    95 ms/step with the coupling group; 14 s pre-warm (warm cache);
    40 s pre-warm (cold cache, autotune-off).
  - ~9× per-step speedup, ~4× pre-warm speedup. Wins came from
    (a) persistent JAX compile cache, (b) skipping XLA gemm/solver
    autotune, (c) pre-jitting the per-frame `link_world_poses`
    closure in `run_ar4_helical_drive.py` so the sample callback
    no longer retraces, (d) `dejongh_new_chain.build_graph` now
    accepts `use_coupling_group=False` for visualisation work
    (staggered back-edges; one-step phase lag, ~10° at 60 Hz).
  - Remaining per-step cost is structural — Gauss-Seidel inner
    loop + per-node cuSolver dispatch — and addressed in a future
    MADDENING-level optimisation.
- New bibliography entries: Featherstone2008, Sciavicco2000,
  Furlani2001, Annin2024AR4, Jackson1998, Krause2013, deJongh2025.

### Test infrastructure
- `pyproject.toml` `[tool.pytest.ini_options]` now defaults to
  `addopts = "-m 'not slow'"` so CI runs the fast suite by default.
  The four expensive tests
  (`test_torque_mode_step_response_analytical`,
  `test_voltage_mode_steady_state`, `test_velocity_mode_pi_tracking`,
  `test_ver131_dejongh_reproduction_short_window`) are tagged with
  `@pytest.mark.slow`. Run the full suite locally with
  `pytest -m 'slow or not slow'`.

### Verification
- `MIME-VER-100` — MotorNode torque-mode step response vs analytical
  first-order solution.
- `MIME-VER-110` / `MIME-VER-111` — PermanentMagnetNode dipole field
  and gradient vs analytical at far-field standoffs.
- `MIME-VER-112` — Earth-field superposition.
- `MIME-VER-120` — RobotArmNode forward kinematics on a 3-link planar
  fixture vs closed-form analytical reference.
- `MIME-VER-121` — Mass-matrix symmetry & PD across random configs.
- `MIME-VER-122` — RNEA / CRBA round-trip consistency at zero velocity.
- `MIME-VER-123` — Gravity-compensated static hold (1 s, < 1e-3 rad
  drift).
- `MIME-VER-124` — Inverse-dynamics + PD trajectory tracking on a
  0.5 Hz sinusoid (RMS error < 0.5°).
- `MIME-VER-130` — Motor + PermanentMagnetNode reproduce the legacy
  `ExternalMagneticFieldNode` rotating field at the UMR location in
  the far-field aligned limit.
- `MIME-VER-131` — Dejongh helical-swim short-window reproduction.
  Both chains produce a finite, same-sign axial velocity.
  Quantitative magnitude disagreement is by design — the new chain
  restores the previously-zero `(∇B)·m` gradient-force term flagged
  in dejongh deliverable Appendix A.1.
- `MIME-VER-132` — Misalignment-induced field tilt at the UMR grows
  monotonically with the magnet's lateral offset over a 0–6 mm sweep.

### Known Anomalies
- `MIME-ANO-100` — `point_dipole` model not faithful at r < 5·R_magnet.
- `MIME-ANO-101` — RobotArmNode v1 has fully rigid joints / links.
- `MIME-ANO-102` — MotorNode v1 is an ideal torque source.
- `MIME-ANO-103` — Motor reaction torque on arm is zero (one-way
  coupling); reframed as a future SENSING plan concern.
- `MIME-ANO-104` — v1 experiment controllers read microrobot position
  from physics truth (sensor pipeline deferred to SENSING plan).

[Unreleased]: https://github.com/Microrobotics-Simulation-Framework/MIME/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Microrobotics-Simulation-Framework/MIME/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Microrobotics-Simulation-Framework/MIME/releases/tag/v0.1.0
