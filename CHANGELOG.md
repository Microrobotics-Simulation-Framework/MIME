# Changelog

All notable changes to MIME are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Quasi-uniform surface sampling, a hull query and a wall's distance, as seams of the geometry protocol** (dev1 phase
  7m-a, designs #195 and #196):
  - **`Geometry.sampling(settings, spacing, patches, density=uniform)`**: a kind's surface sampled for a solver whose
    unknowns sit at the samples, a `mime.registry.Sampling`, chosen among the kind's quadrature samples when the
    experiment is built, so that the samples stay a JAX function of its parameters and a study's derivative flows
    through them. `mime.scene.quasi_uniform` gives it to any kind with a quadrature, a lab's as MIME's: as many samples
    as the quadrature has at the spacing, chosen from it at eight times finer by farthest points and twenty of Lloyd's
    steps, each a quadrature sample (on the surface exactly) weighted by its cell's area, ties broken by index so the
    choice is the same on another machine and on the body scaled, made in float64 whatever the run's precision. A
    `density` grades it toward a feature with no change of the protocol. The MRS rung takes it, and refuses a kind that
    does not implement it by name (`drag.no-samples`); its node holds the choice as it holds its counts
    (`sampling_chosen`, `sampling_members`), and `MRSDragNode.sample_count` says how many samples it solves at.
  - **Every solid kind samples**: the cylinder, the box and the ellipsoid gain a quadrature (the cylinder's caps in
    rings, as the screw's; the box's faces and the ellipsoid by Gauss-Legendre) checked by the kind-conformance harness
    (the ellipsoid's against a level set written from its definition), the sampling and the regions their patches bound
    (the cylinder's `side` and `span`, the box's three slabs, the ellipsoid's `solid`, which no shape evaluates); a body
    whose hull is one of them serves the MRS rung (a hull may now be a kind that samples and has no radial reach; its
    extent is then declared where a law reads it). The lab fixture registers a kind of its own, `labx.bead`, its own
    quadrature and `mime.scene.quasi_uniform`, which the rung and the suites serve unchanged.
  - **`Geometry.reach(parameters, direction)`, the hull query**: the largest `x . d` over the body, homogeneous of
    degree one in `d`, its gradient the farthest point (a face's, a segment's or a rim's middle where that is not one
    point); closed forms for the sphere, the capsule, the cylinder, the box, the ellipsoid and the tube, an envelope for
    the screw (the cylinder of its crests, `reach_envelope` 0.15: on FL-9 it overstates by up to 0.138 of its enclosing
    radius between its axis and across it, and by nothing along either), and `mime.scene.sampled_reach` from any kind's
    samples. No node's step evaluates it yet; its kinks become declared switches with the containment contact (7o).
  - **`WallKind.distance(parameters, point)`**: a `WallDistance`, the wall's signed distance from a point of the medium
    (above zero in it) and its normal into it, by default minus the matched region's shape's, so the tube's is its
    radius less the point's distance from its axis, with the cylinder's switches; free space gives `None`.
  - **The sampling conformance suite** (`mime.testing`: `sampling_problems`, `conditioning_problems`, `reach_problems`,
    `distance_problems`; `tests/saved/test_sampling_walk.py` runs it on every registered kind, MIME's and every
    installed namespace's): the 5th percentile of the samples' nearest neighbours at least 0.7 of the mean spacing and
    none under 0.5 (measured 0.759 to 0.807 and 0.609 to 0.730 at 500 and 2,000 samples, against the quadratures' 0.086
    to 0.358 and 0.062 to 0.358), each patch's, the seams' and the screw's crests' and troughs' share of the samples
    within 15% of their share of the area (measured within 7%; the screw's quadrature gives its caps 2.4 to 2.9 times),
    the MRS matrix's condition at blob 0.5, 1, 1.5 and 2 under bounds ten times the measured (108 to 132 at 0.5,
    against the sphere quadrature's 2.3e4), the reach against a dense quadrature's farthest point, and the tube's
    distance against its closed form. Each check has a control that fails on every run: every kind's own quadrature, and
    each kept version, fails the spacing check; a sampling that leaves out the screw's caps fails the coverage check
    and passes the spacing; the sphere's quadrature fails the conditioning bound; a reach planted short, long or without
    a capsule's caps, and a wall's distance planted off or its normal reversed, fail.
  - **The helix reference as shipped data** (`tests/oracles/references/helix_fl9_free_space.json`, pinned by its
    SHA-256): FL-9's free-space swim speed at 10 Hz, 3.0573 +- 0.057 mm/s post hoc and 3.060 +- 0.147 mm/s by the
    preregistered rule, with its two routes, runs and controls; and `tests/oracles/test_helix_reference.py`, the rung
    at the frozen 0.16 mm (5,984 samples) and blob 0.5: on the screw 0.3.0's sampling 2.9523 mm/s (-3.43% from the
    post hoc point), inside the preregistered band, and on the screw 0.2.0's quadrature 2.7979 mm/s (-8.48%), outside
    it, the control. An estimate against an extrapolated reference, not a bound on the model, and no measured speed.
- **Joint armature and transmissions on dynamic chains, and a check that predicts a spinner's kick** (dev1 phase 7a5,
  design #193): a geared joint's drive reflects its motor's rotor onto the joint (the rotor's inertia times the square
  of the reduction), which a URDF's links do not hold, so the AR4's wrist was lighter than the rotor it carries and A0's
  spin-up kicked it to 21 times its speed limit (7c2's review, item 3).
  - **Armature on the dynamic arm**, per joint, on the diagonal of the mass matrix the step forms once, so that every
    use of it carries the armature: the solve's system and its momentum `M q_dot`, the Cayley step's `M`, and the
    stops' and friction's impulses through the same factor; also in inverse dynamics, and on the tree `ctx.models`
    hands a controller. kg m^2 on a revolute joint, kg on a sliding one; differentiable (a parameter of the node).
    `Arm.Dynamic(armature={"joint_6": kg_m2(..., reason=...)})` declares it; `actuators={"<actuator>": kg_m2(...) |
    "<registered actuator>"}` gives the rotor behind a simple transmission of the description, whose reduction
    reflects it; neither, a convention of none (the URDF's own meaning), which the build report says per joint and
    research lists. A joint given both ways or twice, an actuator given twice, an unknown joint or actuator, or a unit
    of the wrong kind is refused (`chain.armature`); on the kinematic rung, refused. A zero armature builds the graph of
    none, bit for bit.
  - **The URDF reader holds simple transmissions** (`<transmission>` of type `SimpleTransmission`, one joint and one
    actuator, its `<mechanicalReduction>`; the older form with the type an attribute too) in
    `MultibodyTree.transmissions` (`mime.core.multibody.Transmission`), as read: every transmission stays recognised
    and not used, as before, and a description that loaded still loads. A transmission is judged only where it is
    used, where `actuators=` names its actuator: there one on a joint the tree lacks or a fixed joint, two on one
    joint, a reduction of zero or not a number, or an element URDF does not define in it is refused
    (`chain.armature`), and the used one is read and used in the build report and the saved form, its hardware
    interfaces, roles and offsets not used. Not used, the report names it with what is wrong with it, on both rungs.
    `mime.core.multibody.with_armatures` sets joints' armatures on a tree.
  - **Actuators** (`mime.dynamics.Actuator`, `register_actuator`, `actuator`): a motor's datasheet record, its rotor's
    inertia with its source; one registered, StepperOnline's 17HS19-2004S1 (82 g cm^2, from its drawing).
  - **The kick check** (`mime.build._kicks`): at build, for each spinner on a segment that carries mass, each revolute
    joint's ratio `J |(M^-1 S^T e)_j|`, the joint's speed per unit of the rotor's spin with the joints held by nothing
    (on one joint along the axis, `J / (I + a)`), from the carried tree with its armature at the start's joint
    positions; the largest listed with the mounted masses, and a validity line (`validity of ... (computed at build)`),
    never a refusal and the same in every profile, for each joint whose predicted speed at the motor's top speed passes
    its speed limit (where the limit or the top speed is not given, whose ratio passes 0.1, a labelled decision the
    line names), with `armature=` as the fix: 7c2's stand-in of 1e-4 kg m^2 on the wrist, a ratio of 0.093 and 3.73
    times the limit, gives one. Predicted against measured on A0's free arm over six armatures: within 5.3e-4 of the
    largest ratio, from the bare wrist's 3.93 to 0.0095.
  - **A post-run flag** `<segment>.beyond_speed_limit` where a joint's speed passed its description's limit: a 7k flag
    read from the joint velocities against per-joint limits (`FlagLayout.name` and `.limits`, in the manifest, a joint
    with no limit `null` there), adding nothing to the graph. A URDF `velocity="0"` is no limit
    (`JointLimits.speed`), as MoveIt reads a URDF.
  - **A0**: as declared (7c2's preregistered comparison) its arm states no armature with its reason
    (`script_a0_dynamic.NO_ARMATURE`), and the build names the bare wrist's kick (ratio 4.11). Two examples, not
    preregistered (`tests/differential/test_a0_armature.py`), each armature an inference and not the AR4's drive: a
    geared NEMA 17 wrist (the registered stepper's rotor behind an assumed reduction of 10, 8.2e-4 kg m^2), which A0's
    own controller holds at 0.18 rad/s at 10 Hz against its limit of 1.57 (bare: 33 rad/s), the flag down; and 7c2's
    stand-in of 1e-4 kg m^2, whose free kick (6.1 rad/s) passes the limit, held at 1.44 rad/s by the controller on the
    built tree and passing it (4.6 rad/s, the flag raised) on a stale tree with the armature zeroed.
- **The first run** (dev1 phase 7l-c, design #191 item 7):
  - **The README's install and quickstart**: the install section names the distribution, its extras and the branch it is
    installed from until 0.3 is released; the quickstart (a steel ball settling in water, in exploratory, with no
    paperwork) is run by a test as written, with the `mime pending` command it ends on.
  - **Two examples, each run by a test**, `examples/sphere_settling.py` (a sphere in free space, compared with Stokes's
    law) and `examples/screw_in_tube.py` (an FL-9 screw in a tube of glycerol, every value declared), and
    `docs/choosing_spacing.md`, which answers "how fine should `spacing` be, and what does it cost" with tables measured
    on the examples (`scripts/example_tables.py`, held to `docs/data/spacing_tables.json` by a test), stating which
    numbers are measured, on what, and what the screw's number is worth.
  - **A post-run Reynolds-number watch** (`mime.results.WatchRead`, `RunResult.watches`): after every run, the Reynolds
    number of the body's translation and of its turning against the bound the entry declares
    (`mime.registry.ReynoldsRange`, declared on both rungs of `Drag` with its basis), with the size of the neglected
    inertia where it is known (Oseen's first correction for a sphere's translation) and the statement that it is not
    where it is not. A flag on the result, in `text()`, `outputs.json` (`watches`), the `.mat` file and the build report's
    post-run section; never a refusal, and no part of the graph: the derived hash is the same with and without it.
  - **`parts.inertia-axes` prints the derived tensor**: in the body's frame with its units, its principal moments and
    axes, and the `exp.declare("<body>.inertia", ...)` line, reason included, that declares its diagonal.
  - `scripts/inertia_arms.py`, which measured friction 4c of the first-hour walk: a screw whose axial moment is its
    smallest (as a real screw's is) turns broadside about half a second into a rotating field and, in a bore narrower
    than it is long, goes through the tube's wall, which the surrogate cannot see (its resistance reads the body's
    centre and which way its axis points along the tube, and nothing else of the orientation).
  - **Two flags of the surrogate drag for a tilt it cannot see** (`Drag(rung="surrogate")`, in the node's state beside
    `clamp_fired`): `tilt_fired`, raised at the step the body's axis passes the tilt the surrogate is declared valid to
    (`mime.nodes.hydro.TILT_VALIDITY`, 5 degrees: a declaration from one boundary-element measurement, the largest
    measured tilt at which FL-9's swim speed moves by less than the 16% the entry states for the surrogate,
    `scripts/tilt_validity.py` and `docs/data/tilt_validity.json`), and `bore_fired`, at the step the body, as the cylinder
    of its length and outer radius, no longer fits the tube's bore (about 19 degrees for FL-9 in the quarter-inch bore,
    whichever way it points): the geometric limit beside the declared one, so that a user sees how far the body is from
    passing through the wall. Carried as every flag is: in the result's `text()`, `outputs.json`, the `.mat` file. The
    surrogate's resistance, and so every number a run gave, is unchanged; the node's state gained two fields, so the
    derived hash of a surrogate drag graph changed. The containment itself, a contact that reads the orientation, is a
    later phase.
  - **The inertia refusal orders its action by the products' size**: declaring the diagonal first where the largest
    product is at most 1% of the largest moment, placing the parts symmetrically first (with how far the diagonal is
    from the tensor) where it is not.
- **A root `.gitattributes`** (`* text=auto eol=lf`, binaries marked) so a file MIME pins by sha256 keeps its bytes on a
  Windows checkout, where git's default `core.autocrlf=true` wrote `records/water.json` with CRLF and the build refused
  it; `tests/test_gitattributes.py` holds that no tracked file is left to the user's configuration, and the two error
  corpus cases skipped on Windows for this run there.
- **A macOS and Windows smoke job** (`.github/workflows/platforms.yml`, `docs/platforms.md`): the test files
  `tests/platforms.txt` names, then `mime docs check`, on `macos-latest` (Apple Silicon) and `windows-latest`, each on
  Python 3.12 and 3.14, installed from the lock as the default lane is and with no encoding variable set. No required
  check. `scripts/platforms_smoke.py` reads the list, `tests/test_platforms_selection.py` holds that every file it names
  exists, and `tests/platforms/test_smoke.py` is the smoke's own. `docs/platforms.md` says what each platform supports,
  what the job tests and what nothing does; a load off Linux still reports its environment as different (the CPU's flags
  are unreadable there), and five tests that need it are skipped, with the reason, on macOS and Windows.
- **A label-triggered stress workflow** (`.github/workflows/stress.yml`, CONTRIBUTING.md "CI: stress runs"): the label
  `stress` on a pull request runs the test files it changes, and those a `stress:` line of its description lists, 1 to 40
  times (20 unless `runs=` says) in parallel, each run with its own Hypothesis seed, on the default lane's setup and the
  compilation caches of the shards holding the files (restored, never saved). A summary job gives per test how many runs
  passed, each failure's seed and `@reproduce_failure` blob; not a required check. `scripts/stress_select.py` chooses
  the run (and is tested), `scripts/stress_summary.py` reads the JUnit reports.
- **Mounted mass and spinners on dynamic hosts** (dev1 phase 7a4, design #185, MIME-ANO-122 resolved):
  - **Rigid payloads grafted as welded bodies**: `mime.core.multibody.graft(tree, payloads)` welds each `Payload` (a
    name, the frame it is fixed to, its placement there and its inertial) to its frame's body by a fixed joint, the
    link's own inertial untouched, and `articulate(tree, frames=...)` keeps the description's frames; the dynamic
    arm's node holds the grafted tree. An apparatus's part has a mass: its density from its material's record or
    declared (`<apparatus>.parts.<part>.density`), or its mass declared (`<apparatus>.parts.<part>.mass`), a solid of
    uniform density; a part with neither changes nothing. The build walks every mass mounted on a chain to what carries
    it (`mime.build._masses`): welded on a dynamic arm's link, the rotor of a spinner, riding on a motor's stator, or
    nothing on a prescribed motion, and lists each in the build report (`mounted masses`). `ChainSegment` gains
    `carries_mass`, `riding`, `spinner`, `carried_model` and `reads_mount_rates`, and `Segment` `initial_rates` and
    `massive`, so that a lab's segment can take part.
  - **The spinner**: `motor` 0.2.0's dynamic rung carries the rotor's absolute spin about its axis
    (`mime.nodes.actuation._spinner`, one piece for any segment with a spinning body), `J Omega_dot` the axial torques
    with no term from the stator's turning; its angle is relative to its stator, turned by `Omega - e . omega` at the
    mount's rates of this step, and its bearing damps that relative speed. The parts on its rotor add their moment about
    the axis to its declared `inertia`; their mass and mean inertia across the axis are welded on the arm's link, and
    their half difference across it turns with the spin: the spin feels its torque `(omega x e) . D omega`, and the arm
    carries it in its mass matrix (`apparatus.link_inertia`, at the step's start and its end). Its stator passes the
    drive's and bearing's reactions, a step late, and hands the arm the rotor's momentum `J Omega e`
    (`apparatus.link_momentum`), whose gyroscopic torque the arm applies at its own velocity by a Cayley step
    (Crank-Nicolson on the skew joint-space matrix `G = sum_k S_k^T [h_k]x S_k`): it does no work at any step size.
    The spinner converges at first order to the rotor as a jointed child on A0's magnet spun about a diameter and on a
    rod spun end over end. `Motor.Dynamic(top_speed=...)` judges a rotor's imbalance and tilt below.
  - **Frames' angular velocities**: every MIME segment publishes `apparatus.frame_angular_velocities` on every rung
    (the dynamic arm from its forward pass, `mime.core.multibody.frame_motions`; the kinematic arm from its joints'
    rates over the step; the motors from their mount's and their rotor's turn), the kinematic rungs only where the
    build finds a reader (a spinner or a rate-publishing segment mounted on them, a controller's or an observation's
    read), and each mounted segment reads its mount's (`mount_rates`, `DynamicsBuilder.mount_rates`,
    `ResolvedMount.rates` and `initial_rate`; zero where no edge feeds them).
  - **Refused on a dynamic host, each saying the kinematic rung takes it**: an articulated payload, a segment with
    joints of its own carrying mass (`chain.articulated-payload`); a rotor whose inertia turns out of its axis, or whose
    centre of mass lies off it, by more than a residue, its rotating load at the top speed above a thousandth of its
    weight (`chain.spinner-tilted`, `chain.spinner-unbalanced`, with the load's size; a CAD placement's residue is
    accepted and reported); a residue with no top speed to judge it (`chain.spinner-speed`); a spinner on a segment
    that publishes no angular velocities, and a controller's read of an unknown one (`chain.mount-rates`).
  - **Joint commands typed by joint kind**: a dynamic arm's revolute joints take `<segment>.joint_torques` (N m) and
    its sliding ones `<segment>.joint_forces` (N); a kinematic arm's `<segment>.joint_positions` (rad) and
    `<segment>.joint_displacements` (m), each over its kind's joints in order (PR #64's review, Q-12).
  - **Mounts on an earlier segment**: `mount="<segment>.<frame>"` names a frame of any earlier segment of the chain, so
    that a mid-arm aimer and a motor on the flange both ride on the arm.
  - **A controller's models**: `BuildContext.models` holds the built tree of each segment that carries mass, what is
    mounted on it grafted on, for a model-based controller; one reading a bare description of its own declares it, and
    the build report names the trees its law read (`mime.control.law_and_models`).
  - Oracles, each with its control failing every draw: the graft equals the link's inertial edited by hand in the
    file, on the AR4 and a C-arm-like tree with a sliding joint (`tests/descriptions/c_arm.urdf`); the axial momentum
    of a hub and its rotor conserved to the integrator's order (`tests/descriptions/spin_joint.urdf`); a gimbal's
    kinetic energy kept to rounding over 1e4 steps at `nu dt` from 1e-3 to 0.1, its forward-Euler control growing it
    (`tests/descriptions/gimbal.urdf`); the gyroscopic closed form `J Omega omega`; the spinner against its rotor as a
    jointed child integrated finely, at first order, on a joint along the axis, one across it and a gimbal, symmetric
    and unequal rotors (`tests/oracles/test_spinner_tree.py`); the frames' rates against differences of their poses
    with the base turned (`tests/oracles/test_frame_rates.py`). The kinematic rungs' outputs on A0 are unchanged to the
    last bit.
- **Profiles, setting roles and defaults** (dev1 phase 7l-a, designs #190 and #191 items 1 to 5): exploratory's easy
  path, on a1's seams. `Experiment(..., profile=)`, `exploratory` the default, saved in the document and hashed by the
  lock; `research`; `strict` refused by name (`profile.reserved`) until a1. The profiles act by two tables of data in
  `mime.validity` (`finding_classes.txt`, each finding code's class with what covers it; `profile_rules.txt`,
  (profile, class) to generate, basis, list or refuse), and read a finding's class, never a code; profiles act on
  waivers alone, which a test holds for every row with planted controls. In exploratory an unstated condition or a
  value stated without its origin is waived with the reason `pending()`, a declared value with no reason is pending, a
  numerical default is taken: each recorded in the saved form and stamped on every output (the build report,
  `RunResult`'s printout, `outputs.json` and the `.mat`). Research refuses a pending reason by name
  (`value.reason-pending`, `material.accept-pending`, `source.pending`) and a numerical default
  (`setting.numerical-default`), and lists a physical value with no source and a numerical value with no evidence.
  `built.report.pending()` and `mime pending <folder>` print each item with the lines to paste, in Python and in the
  document's YAML. Every setting of every registered entry declares its role (`mime.registry.Role`: design, physical,
  numerical, standard) and its default's kind (`DefaultKind`: determined, convention, numerical, none), checked at
  registration; `Drag`'s `wall_kind` is determined from the medium, its `blob` and `spacing` numerical defaults, and
  gravity is standard by convention (`Experiment`'s signature no longer invites `None`). A quantity takes a reason
  (`mm(3.175, reason=...)`, `chosen(...)`, `pending()`), and `exp.source(path, reason=...)` gives a setting's source
  apart from its value. The saved form is version 2 (`mime.document/2`, `mime.lock/2`); a folder of version 1 loads,
  read as exploratory and held to its own lock. `docs/profiles.md`, the profiles' path for a new user. After its
  refuting review: the profile's record is kept in a canonical order (two pending declarations in any order load), one
  item per read, one dispatcher for every action at every site (`mime.validity.decide`, with `profile.tables` refusing
  a row no site can apply), a site for every reason (`source.no-site` for one nothing judges), a written default its
  setting declares no role for reported as `undeclared` (stamped in exploratory, refused in research), defaults
  recorded by kind, value and entry, never their words; the bem rung's default spacing by sample count per body
  family, its measured misses stated on the stamp (`scripts/default_spacing_table.py`); `Screw.starts` required.
- **Post-run results** (dev1 phase 7k, design #184 items 5 to 10): `mime.results.RunResult`, one object from
  `BuiltExperiment.run(...)` and from `RunResult.read(folder)` (a `run_to` or `mime run` folder, local or on a Modal
  Volume's mount, or an NPZ `to_npz` wrote), indexed by the handle `observe` returns or by the observation's id, each
  declared output read by name with its units: an output at the end as a `Reading` over its window, a per-step output
  as a `Series` with its time axis when the run records it (`run(..., record=[swim.speed], every=ms(5))`; `history=True`
  records every one), and a series not recorded refused naming the `record=` that keeps it (`NotRecorded`).
  Observations declare their outputs (`mime.registry.Output`, `OutputKind`, `Reduction`), checked at registration and
  at the build, and pages and the run's manifest are generated from them; entries declare the flags their nodes raise
  (`Flag`, `raises`; the surrogate drag's `clamp_fired`), shown on every window, export and printout, never blocking a
  read. Windows by time under one inclusion rule, the observations' own (the steps whose end lies after the start and
  not after the end); `mean()` and `integral()`, the rules a node declares. `Quantity` holds arrays (`Quantity.of`) and
  reads any quantity in a unit with `.to(unit)` (units are `mime.units.Unit`; `mm_per_s` and `um_per_s` added). Exports:
  `table()` (columns named with their units, a flags column), CSV, NPZ, `.mat` (`to_mat`, through scipy, now a declared
  dependency), and `outputs.json` beside `outputs.npz` in a run
  folder. `mime run --record ... --every ...`, and the Modal script's `::main --record --every`. The lab fixture
  registers `labx.height`, an observation declaring its outputs the same way. A run that records nothing scans as
  before. A window's start is decided as a node decides its `after`, in the run's float type, so a series over a node's
  window has the steps the node counted on the grid in float32 too; a flag of a run that records nothing is unknown
  before its last step (`was_raised` is `null` in the manifest), never "not raised"; an array quantity cannot be made
  writeable again; `.to(Hz)` of a `rad_per_s` quantity (and the reverse) and `.to(degC)` are refused with the reason.
- **The dynamic arm and motor, their loads, and the magnetic laws' reaction on the source** (dev1 phase 7a3):
  - **`Arm.Dynamic`** (`arm` 0.1.0, rung `dynamic`, added before any release with the kinematic rung unchanged): joint
    torques in (`<segment>.joint_torques`, N m), against the mass matrix and the bias of `mime.core.multibody` (the
    experiment's gravity, the velocity products and the loads on its frames), each joint's damping, taken at the end
    of the step, and Coulomb friction from the description; its limits rigid, inelastic stops. The stops and friction
    are impulses through the step's system `A = M + dt D`: the step's velocity is the one nearest the unconstrained
    step's in `A`'s norm, less what friction dissipates, that keeps every bounded joint within its limits over the step
    (a joint driven past a limit arrives at it at the end of the step, and the stop takes the rest of its velocity on
    the next), solved by 20 projected Gauss-Seidel sweeps over the joints with a stop or friction, each joint's stop and
    friction one exact update with the others free (`1 / (A^-1)_jj`), from a start that holds every joint with friction
    still as far as its friction can. It never adds kinetic energy at any sweep; one joint, and an arm whose friction
    holds it, are exact; on coupled joints the change a converged solve would still make to the velocity, relative to
    the unconstrained velocity, is at worst 3e-6 on the AR4 and 1.6e-2 on a strongly coupled planar arm both of whose
    joints reach a stop in one step (measured through the node over drawn states, the median exact; it falls by about
    0.8 a sweep there: 40 sweeps would leave 1.6e-4, at four times the compile time of the step's derivative). Semi-implicit Euler, first order in time
    (`DynamicArmNode`, MIME-NODE-109). It publishes its joint velocities (`apparatus.joint_velocities`). Its base is
    fixed in the world (`chain.fixed-base`), a description whose moving body has geometry and no inertial is refused
    (`chain.inertia-gap`), and so is a joint that moves no inertia about its axis at the initial positions
    (`chain.massless-joint`). Each rung declares its own gradient status: kinematic `piecewise_kink`, dynamic
    `piecewise_kink` (`arm.stop.low`, `arm.stop.high`, `arm.friction.low`, `arm.friction.high`, `arm.stick`, each with
    its row in the switch harness; the stops' rows also measure the impulse's change of every joint's velocity).
  - **`Motor.Dynamic`** (`motor` 0.1.0, rung `dynamic`): a torque in (`<segment>.torque`) against its rotor's inertia,
    its bearing's viscous damping and the loads on its rotor about the axis; an ideal torque source, MIME-ANO-102's
    torque-source record (`DynamicMotorNode`, MIME-NODE-110). Its stator passes the drive's and the damping's reaction
    and the rotor's loads across its axis, with its own, to the frame it is mounted on (MIME-ANO-103, now partly
    resolved).
  - **Loads on a chain**: `apparatus.link_wrench` is built, a row per frame of a segment (a force and a torque about the
    frame's origin), which a dynamic segment accepts; a mount names the loads port of a segment that takes them
    (`ResolvedMount.loads`). A segment passes its reaction to the one it is mounted on by an edge the build derives
    (`DynamicsBuilder.reaction`), read a step late: the lag accounting plans it (`REACTION`) and the report notes the
    staggered coupling. A law delivers its reaction on a field's source with `Delivery("field",
    "apparatus.link_wrench")`, planned where `PlanContext.reaction` names the segment the source is mounted on; its
    builder wires that segment's frame poses (`InteractionBuilder.mount`) and one of its nodes into another
    (`InteractionBuilder.connect`). A law naming an apparatus and no link is refused (`target.chain-link`).
  - **`Magnetic.Torque` and `Magnetic.Force` deliver their reaction** on a source mounted on a dynamic segment:
    equal and opposite, moved to the frame's origin (`SourceReactionNode`, MIME-NODE-111); together, what the body's
    dipole exerts on a point dipole source. Elsewhere they build what they built.
  - **`mime.core.multibody`**: loads on the links in the recursive Newton-Euler algorithm (`frame_forces`, and
    `link_forces` in `inverse_dynamics` and `forward_dynamics`), and `cholesky_solve`.
  - **Units**: `TORQUE` (`N_m`) and `ROTATIONAL_DAMPING` (`N_m_s`).
  - **Oracles, each with a control that fails on every draw**: loads in the recursion against the Jacobian transpose;
    the energy balance to the integrator's order, a run that reaches a stop held to the inequality an inelastic stop
    obeys (the control, gravity dropped); the dynamic arm under a stiff joint
    controller against the kinematic arm on a slow smooth trajectory (the control, gravity dropped); Coulomb friction's
    closed form on one joint (the control, friction applied explicitly); the stops (the control, the earlier line's stop
    that kept the velocity); on a strongly coupled planar arm and the AR4, the stops and friction against what any
    inelastic stop and Coulomb friction must do (kinetic energy never rising across a stop event or under friction
    alone, an arm resting on a stop moving as the held arm's closed form, a torque below breakaway moving nothing, an
    arm held by its friction staying put), each with the first rule of this phase as its control (a stop that zeroed one
    joint's velocity, friction on each joint's own diagonal with the others locked); the weight and a load moving the
    arm through a turned base as their closed forms (the controls: the weight turned the wrong way, the loads not
    turned, the loads reversed); the motor's bookkeeping, off its mount's origin (the control, a motor that passes
    nothing on) and its reaction
    reaching the arm a step late through the build (the control, the edge left out); the laws and their reactions on
    the source equal and opposite (the control, the reactions dropped), and the reactions against the force and torque
    of two point dipoles (the control, the force's reaction alone). The new nodes pass MADDENING's battery, and the
    dynamic ones their first order. Measured over 1000 fresh draws each, the largest oracle misfit against its bound and
    the smallest control misfit: the loads 0.17 and 6e10; the energy balance's order off first by 0.010 and 0.54 (its
    band 0.2; the control drawn where gravity's work outweighs the integrator's residual: twenty runs found one draw
    below it otherwise); the motor's balance 0.049 and 1.3e11; equal and opposite 0.0023 and 7e13; the pair 9.5e-5 and
    1.0e4 of its tolerance. The tracking oracle, one run: a joint error of 0.33 mrad against a bound of 0.77 mrad (twice
    the controller's predicted 0.38), 10.5 mrad with gravity dropped.
  - **Its cost, measured** (a budget, not a gate; one CPU, the best of five 2000-step runs after a first): A0's
    apparatus at 69 us a step on its dynamic rungs against 2.8 us on its kinematic ones (float32; 71 and 2.9 in
    float64), of which the stops' and friction's 20 sweeps take 19 us (50 us with none); the dynamic arm's node alone
    23 us a step (26 in float64); compiling and running the 2000 steps the first time, 3.1 s against 1.2 s.
  - **The node walk** looks past a rung's first case for a node class built only on a condition (a law's reaction
    node), failing only when no case that declares the rung builds it; and reads a segment's publishes from the
    record of the rung a case built.
  - **A0's apparatus on its dynamic rungs** is a round-trip case (`dynamic_rpm`).
  - **Golden outputs per rung**: an entry with rungs has a golden output per rung its first round-trip case does not
    build, from the first case that builds it (the arm's and the motor's dynamic rungs, the drag's boundary-element
    rung); a flag is compared exactly. A dynamic arm with Coulomb friction (`chain_cases.friction_arm`) is a switch
    configuration, so that the scan traces the friction's switches and the sweeps' start.
- **A surrogate as a type** (dev1 phase 7i-a): `mime.surrogates.Surrogate`, an asset record naming its evaluator
  (registered by name and version, `cholesky_mlp` 0.1.0 the first), its outputs (a resistance, and a flow-load operator
  where trained for one), the features it reads in order, its accuracy per use (`Accuracy`; an unmeasured use is
  `None`, unknown and not bounded) and its provenance (`Provenance`: what it was trained on, the patches its training
  sampled, the reference point, its derivation record). `ScrewSurrogate` is its screw form and `dejongh_mlp_v2` its first
  record: swim speed known to MIME-ANO-114's 16.2% against the BEM it was checked against, the resistance's entries
  unmeasured, trained on the lateral surface alone about the end of the body's axis. The drag's node reads the weights
  through the evaluator the record names, refuses a record whose features it does not build, and delivers a surrogate
  trained about the centre unshifted. Gate: the node through the type reproduces the 4c1 MLP's golden outputs to
  2.8e-15 (a record naming a planted evaluator, the diagonal floored at 1e-4, misses by 4.1e-6); inference cost per
  step 16.4 µs against 16.8 µs before the type (ratio 0.99, quartiles 0.97 to 1.00; CPU, float64; bitwise the same
  resistance), `tests/differential/surrogate_budget.py`.
- **Derivation records of derived assets** (design #181, its minimum): `mime.assets.DerivationRecord` (builder and
  commit, inputs by role and hash, parameters, checks with results, validity domain), `derive` running a `Builder` and
  refusing a build with a failed check (`asset.derivation-check`), `stale_inputs`, and the shipped records in
  `mime.data`'s `derivations/`: the corrected tube table's (its parameters from its file's provenance arrays, its builder
  the committed `scripts/tube_table/full.py` and build spec with the factor 1.5, at their commit on `v0.3`, each source
  file's hash recorded), and `dejongh_mlp_v2`'s, whose `wall:tube`
  inputs are the six 2026-05-21 tables its training read. `mime.surrogates.staleness` finds `dejongh_mlp_v2` stale
  against the table the tube wall kind reads now, as it declares (MIME-ANO-117).
- **Saved experiments record the assets they read**: an entry's plan names the shipped references it reads (a wall
  kind's correction its table), and the build writes them under the lock's `assets` key, so a tube experiment's lock
  names `unit_tube_R0.997` and `mime assets fetch <folder>` and `mime modal` fetch it. A load refuses a lock that does
  not name an asset its experiment reads, or names it under another hash (`load.assets`).
- **The surrogate generator and its held-out protocol** (`mime.surrogates.generator`, `scripts/surrogate/`): de
  Jongh's seventeen designs resting on the floor of his four tubes through `Drag.BEM` in the `tube` wall kind, in
  float64, the resistance in the canonical frame about the centroid; held-out sets at the gaps between the training
  margins and with the body turned about its axis (floored again at each roll); accuracy per use stated as what it
  estimates. **The near-wall check** (MIME-ANO-114; FL-9 in the 1/4 in tube, spacings 0.30 to 0.10 mm, gaps 0.02 to
  0.5 mm, on Modal, $1.14): the whole resistance is resolved from 2.08 spacings at the declared far gap, past the
  largest contact margin at every spacing reached (its normal translation and pitching rotation). **The domain's
  near-wall edge is open, the maintainer's decision** (review of the PR, B1): the measured edges move with the far gap
  (whole resistance 2.083, 0.833, 0.625 spacings at far gaps 0.5, 0.3, 0.2 mm) and differ by design (FL-1's swim speed
  2.50 spacings against FL-9's 0.625), and the 3/16 in tube cannot run the analysis as declared; they are recorded as
  data in the domain, with the options, and the generator refuses to produce training data while the edge is open.
  7i-b is blocked on that decision.
- **Review fixes of phase 7i-a**: the tube table's record points at the committed builder and spec (#62); the build refuses by name an open edge, a case outside the domain and an empty set
  (`generator.near-wall-edge-open`, `generator.outside-domain`, `generator.no-cases`); `derive` refuses a build that ran
  no check; `staleness` refuses a role nothing plays (`asset.unknown-role`); the derivation record gains the sha256 of
  each source file the build ran and a dirty-tree flag (`working_state`); a body turned about its axis is floored at its
  nearest point (the roll set's FL-1 cases at a quarter turn were 0.288 mm from the wall against 0.137 mm); MIME-ANO-118
  records a 14% jump at FL-9 in the 3/16 in tube that passes every check; MIME-ANO-116 and 114 carry the review's
  figures (the "plateau of 0.6% to 2.3%" holds at 0.12 and 0.15 mm only).
- **A0 against its axis-fixed reference** (dev1 phase 7c1): the AR4 carrying a motor and the rotor magnet (a point
  dipole), closed by `arm_tracking` inside the compiled step, driving script A's body, written through the public API
  (`tests/script_a0.py`) and compared in float32 with the pre-0.3 tag's A0 (its drag surrogate given the tube's axis) at 3 Hz and
  10 Hz (`tests/differential/references/a0/`). The differences were declared before the comparison was run, four
  with predicted sizes, mu_0 (negligible, 1.35e-10) and six declared none (`a0_expected_differences.json`), by the script A emulator extended
  to the arm, the dipole, the rotor's PI loop and the controller (`predict_a0.py`), which reproduces the reference with
  the tag's scheme (the swim speed to 1.5e-5, the rotation rates to 1.5e-9, the single-precision accumulators of the rotor's
  angle and the controller's filter bit for bit). Every gated quantity holds to its prediction, and a fault planted in
  the arm's pose, the dipole's position or moment, the controller's latency or law, the motor's speed or the gradient
  force fails the comparison, caught 20 of 20 over sizes from 1.5 to 10 resolutions
  (`references/a0_controls_measured.json`). Its limits, measured: a gradient-force error below about 2e-3 of the force
  is not seen; the distance from the body cannot tell 3 Hz from 10 Hz, so eight of the nine declared quantities are
  effectively gated; the motor's speed is gated through its angle (`rotor_angle_rate_mean_hz`, added after the
  declaration), a single-precision sum that shows a speed fault only above about 1e-3 of the speed. The drag surrogate is
  `dejongh_mlp_v2` on both sides, which this does not validate (MIME-ANO-117). The largest declared difference is
  MIME-ANO-112's, the surrogate's resistance referred to the body's centre: the swim speed up 10.2% (3 Hz) and 9.2%
  (10 Hz). A finding off the list: the controller's single-precision target filter stops 3.3e-7 m short of the standoff
  here and 1.3e-6 m short in the tag's run (a fused multiply-add there), 9.2e-7 m on the distance from the body.
- **A0 on its dynamic rungs against its reference, the swap test, and the kinematic arm's evidence** (dev1 phase 7c2):
  A0 on `Arm.Dynamic` and `Motor.Dynamic` (`tests/script_a0_dynamic.py`), closed by one controller that carries the
  tag's computed-torque joint loop on `arm_tracking`'s targets (Kp 1e4, Kd 200, the weight in the command, from the
  built tree and the experiment's gravity) and its rotor's PI velocity loop (Kp 1e-3, Ki 1e-2, the speed relative to
  the stator read through the motor's frames), each a `law(ctx)`, run in one step (`tests/a0_loops.py`, test-only
  entries under `oracle`), compared in float32 with the same reference as 7c1's at 3 Hz and 10 Hz. Every difference
  was declared before it was measured (`tests/differential/a0_dynamic_expected_differences.json`, from
  `predict_a0_dynamic.py`, 7c1's emulator extended to the dynamic rungs; with every change off it is 7c1's to the bit):
  7c1's arm and rotor differences close; three of 7c1's carry over; five are new, each with its size made alone: the
  rotor's axial inertia on the arm (its spin-up turns joint 6 by 0.153 rad at 3 Hz and 0.511 rad at 10 Hz, at up to 9.9
  and 33 rad/s against the description's 1.57; measured to 4e-5), the magnet's reaction on the rotor and the flange,
  the rotor loop's read, the arm's step, and one found in the dynamic build's report and not on 7a3's list (the field
  reads the rotor's pose a step late there, since the reactions reach the rotor within the step); seven more declared
  none. Every gate holds at 3 Hz on the build's machine. **A finding off the list**: the rotor's single-precision angle on the dynamic motor
  is reproduced by the emulation to about one unit in its last place, not bit for bit, and at 10 Hz reaches the window
  one unit (3.05e-5 rad) apart, which moves the rotation rates by phase by 1.3e-7 (4.5 and 9.4 tolerances), the
  magnet's phase by 6.3e-5 rad and the motor's angle rate to 0.97 of its tolerance; the emulation with its angle set to
  0.3's at 4.95 s gives 0.3's rates to 3e-10, and XLA's single-thread CPU flags move 0.3's own angle by a unit
  (`references/a0_dynamic_findings.json`). The first declaration stays as committed, its 10 Hz miss recorded as
  measured; on some of CI's runners its angle-reading gates miss at 3 Hz too (5 of 20 stress runners), which the
  first round's CI passed only on the allowance first written for it (a widening of those tolerances after the miss
  was seen), no longer a gate.
  **A second declaration** (`a0_dynamic_expected_differences_b.json`, from `predict_a0_dynamic_b.py`, written before
  its run; the maintainer's option A) reads every phase against the angle of the same run (`a0_phases.py`): the
  field's turning and phase at the body against the motor's angle, the body's turning against the field's, the rotor
  frame's phase against its angle, and the magnet's phase in its mount, inverted from the field the body felt, which
  nothing gated; it gates the absolute phase at the start and in the rotor loop's integral, and the angle's own
  rounding to one unit of its last place below each power of two crossed, derived. Its tolerances add to 7c1's rule
  ten times the gap to the emulation's single-precision twin (the arm's step in float32), which reproduces the finding
  from the emulation alone; the kick's is 2.3e-4 (the first's 2.8e-2 passed a 4 % error in the rotor's J). Every gate
  holds at 3 Hz and 10 Hz, on every runner. Its controls (`test_a0_dynamic_b_controls.py`), at both frequencies: the motor's angle
  offset at its start, the magnet turned 1e-4 rad in its mount and a downstream drift (the three faults the first rule
  let through), the rotor's J 4 % large and the rotor loop's Ki at three resolutions each fail their gate, every one
  above its gate's resolution as measured (`references/a0_dynamic_b_controls_measured.json`).
  The swap test (`test_a0_swap.py`): the same law inside the compiled step and as a callback, 1000 steps through the
  spin-up, every field of every node within 8 single-precision ulps per step run (at most 0.098 of it), with no lag
  in mode 1 (7b2's measurement; the code architecture's "one base step of lag" is superseded); a step of lag removed or
  added fails it, 1049 times the bound. The first declaration's controls (`test_a0_dynamic_controls.py`, at 3 Hz): a
  fault in the joint loop's Kd, the rotor loop's Ki or the joint loop's gravity term fails it, caught 20 of 20 over
  sizes from 1.5 to 10 size units (`references/a0_dynamic_controls_measured.json`); the units were not every gate's
  resolution (Ki's measured threshold is 1e-4 to 2e-4, gravity's 2.2e-3 by the body's rate), and every plant is above
  its measured one. The
  kinematic rung against the dynamic, as data for the validity line (`references/a0_rungs_measured.json`, 1 to 10 Hz):
  the commanded joint speeds stay below 1% of the description's limits, and the rungs differ by a fixed share that does
  not vanish with them (the tracking error +18.5% to +18.8%: a 19.5 ms servo lag over the controller's 100 ms,
  speed-invariant by construction, so the declared relative crossing rule could not cross; #99's line needs a lag
  condition and an absolute tolerance; the swim speed +0.6% to +0.8%); the kinematic rung carries no rotor kick, which
  takes the dynamic rung's joint 6 to 2.1 to 21 times its velocity limit, a bare-wrist artefact (the description has
  no armature and massless motor links) that #99 does not cite. A0 with its magnet's mass is reported beside it and never compared: its kick is 0.27 and 0.91 rad.
- **The tube wall kind**, `tube` 0.1.0 (`mime.walls.TubeWall`), reading the corrected table of the unit tube's image
  system (MIME-ANO-117): `Drag.BEM(wall_kind="tube")` in a medium that fills a tube's lumen, the boundary-element
  method's correction registered on the shipped asset `unit_tube_R0.997`, readable from the axis to 0.997 of the radius
  and 5 radii along it. The asset is stored by `mime assets fetch unit_tube_R0.997`, read when an experiment is
  planned and never downloaded there; one that is missing is refused as `asset.missing`, naming the command (an
  entry's plan may now raise `BuildRefused`, whose refusals stand). CI stores it once per job in a cache keyed by its
  sha256 (`.github/actions/unit-tube`). The read declares its four switches (`tube.table`, `tube.angle`,
  `tube.axial`, `tube.axis`), the reads sitting under the solve's checkpoint, which the switch scan's raw-site list names
  (`nested_remat2`: its item form takes a family name with digits). From about 3 radii along the axis the read's error exceeds the tube's whole Green's
  function: fit for a body's drag, not for long-range interactions.
- **The corrected table of the unit tube's image system** (MIME-ANO-117): one table for every tube radius, built from
  the defect-free Fourier-Bessel series by its builder (`scripts/tube_table/`) under one truncation for every source
  radius (2500 modes, 16000 Gauss nodes, k_max 3000: the rule's factors times 1.5), readable from the axis to 0.997 of
  the radius, gated against the builder, reciprocity, divergence, the axis's coefficients and a method of fundamental
  solutions (consistent with the table, not a bound: its own wall residual is 3.8%) with the defective series as its
  control; shipped as the reference `unit_tube_R0.997` (`mime.data`'s `assets/unit_tube_tables.json`, CC0, pinned to its
  Hugging Face revision). **Reciprocity is gated at every knot of the file**: the maximum is 8.4e-10, none of the
  2,903,181 knots is above the declared 1e-8 (the defective control's is 2.28). The table at the first truncation (1667
  modes) missed it, 2.0e-7 at 129 knots above 1e-8 at the wall's edge, so "reciprocal to rounding", which was said of it
  from interior points, was wrong; it is kept as the reference `unit_tube_R0.997_1667_modes`, superseded and read by
  nothing, beside the eleven 2026-05-21 references, labelled wrong and read by nothing. The gate's read check now
  compares in the shipped evaluator's frame (it read about 2 for any table before). The tube's evaluator now takes
  a wall's table as arguments of the compiled step (`params` leaves `wall_table.<path>`, `trainable=False`), not as
  constants of its program, so its size no longer grows the program; reads through the tube's axis (each pair's rows
  in the source's frame, blended with its reciprocal read by `rho_s^2 / (rho_t^2 + rho_s^2)`, owner `tube.axis`, at
  every pair, so the evaluator's own reads are reciprocal whatever the table holds and reciprocity is checked on the
  file's knots and on `tube_reads`, the reads before the blend); and takes its readable range from the table file's
  `readable_rho`.
- **Wall kinds** (dev1 phase 7h, stopped at its check of the wall tables): `mime.registry.WallKind` and
  `register_wall_kind`, an entry kind that declares, apart from any solver, the region shape a confining wall matches
  exactly, the parameters it reads from it, its form (analytic or meshed), its validity and its gradient status;
  `mime.walls` holds MIME's, `free_space` now, with the tube and the plane reserved, each with its reason. The
  boundary-element method registers its own correction under a kind's name and version (`WallCorrection`: the
  constants it prepares at build, the reads it checks when planned), and `BEMDragNode` publishes `wall_accepted`.
  `Drag.BEM(wall_kind=...)` names a registered kind, its one version or `"name version"`, and is refused when the kind
  does not match the medium's region (`drag.wall-kind`, listing the kinds) or the body outreaches its data
  (`drag.wall-range`). The tube's correction, its image system read from a table in the tube's own units, cubic on
  every axis with the angle and the axial offset mirrored past their symmetric ends, is in place
  (`mime.nodes.hydro.bem.tube_correction`). The build's derived hash takes a registered pytree of arrays (a compiled
  table) by its structure and its leaves.
- **The Fourier-Bessel check of the 2026-05-21 wall tables** (`tests/differential/check_wall_tables.py`): the tables
  are what the v0.2 builder computed, at the radius each names, and that builder's image of a force along the
  source's azimuthal direction is not a Stokes flow (MIME-ANO-117): the tables fail, and the tube's wall kind is
  reserved until they are built again. On the axis the defect is the whole of the miss (FL-9 in the 1/4-inch tube:
  25% in the resistance, 1.077 times in the swim, the truncation alone 0.000); off the axis the tables' truncation
  matters more than the defect there, and the finer series they are held against is itself unconverged beyond 0.9 of
  the tube's radius, so those figures are not estimates. The record gives the cause (one basis serving two columns),
  an independent confirmation (divergence, momentum, reciprocity, the axis and a method of fundamental solutions),
  and the constraint on the rebuild.
- **Running a saved experiment on Modal, fast and reproducible**, `mime.env`: the environment is built once per MIME
  release, Python series and variant, never per experiment, and every input to it is pinned by hash or digest.
  - A `cuda12` extra (`jax[cuda12]`), locked in `uv.lock`, so JAX's CUDA plugin, its PJRT and every `nvidia-*` library
    are pinned with hashes. Each variant's `uv export` (`cpu`, `cuda12`) ships in the wheel under `mime/data/env/`, with
    a test that regenerates it from the lock byte for byte; every pin has a wheel for Python 3.12, 3.13 and 3.14.
    `bases.json` pins the base image of each Python series by digest, with its Python, Debian, glibc and own
    distributions, and `uv.json` the uv that installs the pins (`scripts/bump_bases.py`).
  - `mime env check`: the installed distributions against the pins (markers evaluated), Python and the OS against the
    base image, JAX's devices against the GPU type requested (a CPU fallback refused; an A100 request met by any A100),
    MIME against the pinned wheel's version, and a module Python imports from outside the environment's site-packages
    refused, every difference named; then the lock's precision applied (a GPU run whose lock records a matmul
    precision other than the highest is refused), and XLA's cache in its directory (by jaxlib, platform and, on a GPU,
    its compute capability, a CPU's by the hash of its flags).
  - The lock's environment records what no pin holds: the GPU's device kind, driver, CUDA and cuDNN, the CPU's model
    and the hash of its flags, the matmul precision, `XLA_FLAGS` and XLA's deterministic GPU mode, and a free-form
    `runner` record (`MIME_RUNNER`). "The same environment", for a bit claim, means equal on all of them; the runner's
    information (`MIME_RUNNER_INFO`: a workspace's image ID, a client's version) is stored and never compared, and a
    device field that could not be read is never equal to anything.
  - `BuiltExperiment.run_to(folder, duration)` writes what `mime run` does; `mime verify` applies the lock's precision.
  - `mime run <folder> --duration <s> --out <dir>`: run a saved experiment in its lock's precision and write the
    outputs, a record and the run's own saved form.
  - `mime modal <folder> [--gpu <exact type>] [--deterministic]`: writes the Modal script beside the experiment, and
    never imports `modal`. Its layers: the digest-pinned base, the variant's hashed pins, MIME's wheel by its PyPI hash
    (before a release, a wheel of the checkout by its hash), and the lab's distributions; the experiment is mounted,
    assets and XLA's cache live on Volumes, the image builder is pinned, and XLA's deterministic GPU mode is off unless
    asked for and recorded either way. See `docs/running_on_modal.md`.
  New core dependency: packaging (Apache-2.0 or BSD-2-Clause), already installed through pooch. The dev group pins
  uv, whose export the pins are.
- **Remote assets** (before dev1's phase 7h), `mime.assets`: a large file is referenced by the sha256 of its bytes with
  its size, licence and an ordered tuple of locations (the first the primary, the rest backups), never by where it
  lives. `resolve` is the one lookup (the hash-keyed cache, a file already in `MIME_ASSETS_DIR`, then the locations
  through pooch with `known_hash`); `locate` is what a build calls and never downloads, refusing `asset.missing` with
  the command to run; `MIME_OFFLINE` (`1`, `true`, `yes` or `on`; any other unrecognised value is refused, never read as
  online) downloads nothing. One cache directory, which `MIME_ASSETS_DIR` replaces; a file a person placed there is
  accepted by its name (the sha256) and its hash and is never deleted or overwritten (`asset.placed-file`); only a file
  MIME downloaded is replaced. Several processes may fetch at once: a per-asset lock, one download per asset, and unique
  temporary names. `fetch` hashes a cached file in full and repairs one whose bytes are wrong; `check` marks a corrupt
  file so that `locate` refuses it until a fetch repairs it. The lock's `assets` key is part of the saved form's `Lock`.
  `mime assets list`, `fetch <name-or-folder>... | --all` and `check`; a folder stands for the assets its lock
  references, and the `asset.missing` refusal gives the exact command, scoped to the folder where it is known. The eleven wall
  tables' references ship as data, pinned to one revision of the public dataset `NickERoy/mime-wall-tables`.
  New core dependency: pooch (BSD-3-Clause), with platformdirs, requests, urllib3, idna, charset-normalizer and
  certifi.
- **Bundles and credentials** (dev1, after the assets): `mime pack <folder>` writes a saved folder as one file (a zip
  with nothing compressed, so the same folder makes the same bytes): a manifest of every file's sha256 and size, the
  lock's asset references and the bundle's own hash, then the folder's files. It loads the folder first, carries only
  the files a saved folder names (the document, the lock, the reference outputs, `scene/`, `records/`,
  `descriptions/`; never a path part starting with a dot, a name a filesystem treats specially, or two names that
  collide in case), and refuses to carry a path of the machine that packed it, any token MIME holds (as written,
  percent-encoded, JSON-escaped, in base64, hex or UTF-16, or inside gzip or a compressed zip member), a link or
  special file, or a file over fifty megabytes that the lock does not reference as an asset (`bundle.*`). `mime pull
  <doi | address | file>` gets a bundle through the assets resolver (a new `mime.assets.retrieve` stores a file whose
  hash is not yet known, held to the repository's checksum and to a size: the one the record states, or 500 MB, raised
  by `--max-bundle-bytes`; a Zenodo DOI needs no account), checks the destination before it downloads, then every name
  against the saved form's layout, every member kind and every hash, writes only into a new folder through its own
  opens, accepts only the files the lock and the document name, holds the lock's distribution names and versions to
  PEP 503 and PEP 440, lists the packages the lock names that are not installed with the command to install them,
  quoted (it installs and runs nothing), then fetches the bundle's assets unless `--no-assets`, each held to the size
  its lock declares while it downloads, asking first above a gigabyte (`--yes` goes ahead; without a terminal it stops
  with the command), and ends with a summary in names and sizes, in which a bundle's control and escape characters are
  written as text. `mime credentials set | status | remove <service>` for `zenodo` and `zenodo-sandbox`, ready for
  `push`: the token is read with `getpass` and never accepted as an argument (a refusal does not print the word),
  kept in a file in the user's config directory (owner-only, never inside a git working tree, a dotfiles repository
  with a separate work tree included) that `ZENODO_TOKEN` and `ZENODO_SANDBOX_TOKEN` override, and held in a type that
  prints masked and shows nothing of itself to a traceback printed with its locals. `status` shows the last four
  characters of a token of 32 or more and nothing of a shorter one. A file a person placed in the assets cache under
  the asset's name, or the file name its location serves (`wall_R2.500.npz`), is accepted when its full hash matches,
  as one named by its sha256 is. New direct dependencies: platformdirs and packaging (already pooch's).

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
- **The sphere 0.2.0, the capsule 0.2.0 and the screw 0.3.0 sample their surfaces quasi-uniformly** (phase 7m-a), so
  every result of the MRS rung on them moves (the same count of samples, placed so that none crowd: on FL-9 in free
  space at 0.16 mm and blob 0.5 the swim speed rises from 2.7979 to 2.9523 mm/s against a reference of 3.057). Their
  quadratures, shapes and closed forms are unchanged. The versions superseded stay registered (`mime.scene.KEPT`),
  each sampling by its quadrature, so that a folder saved with one loads and reproduces (the saved-form fixtures of
  versions 1 and 2 name the sphere 0.1.0; 7j's saved experiments of its frozen model name the screw 0.2.0); a script
  that writes `Sphere(...)` gets the current version. The drag stays 0.3.0 and the wall kinds 0.1.0: what changed is
  the kinds' samples. The golden outputs that moved were regenerated, each listed in the pull request with its reason.
  MIME-ANO-123 (the polar grids crowding the sphere's and the capsule's poles) is resolved for the current versions.
- **The drag's `bem` rung is the `mrs` rung, and the saved form is version 3** (design #192, item 1): the node is the
  method of regularised Stokeslets (Cortez, Fauci and Medovikov, 2005), not a boundary-element method, and the name is
  kept free for a true boundary-element rung. `Drag.BEM()` is `Drag.MRS()`, `BEMDragNode` is `MRSDragNode`,
  `mime.nodes.hydro.bem` is `mime.nodes.hydro.mrs`, and every refusal, page, test, script and example follows; there is
  no alias (the entries of this release below say `bem` as they were written: "formerly the bem rung"). The wall tables
  and the asset `unit_tube_R0.997` are method-agnostic, and unchanged. The rename was made by a script from a census of
  every occurrence, with the exceptions listed by hand, and a test (`tests/test_naming_rules.py`) fails on any use of
  the old name outside them.
  - **The saved form is version 3** (`mime.document/3`, `mime.lock/3`), because the rung's name is in a saved document
    and its hash, and the drag node's type is in the derived graph's. The same bump fixes the possessive in the sources
    of values declared in the experiment (they read `body 'umr''s mass`, a quoted name followed by an apostrophe; they
    read `body umr's mass`). A folder of version 1 or 2 still loads: its document is read as the current schema (a
    document that names `bem` names `mrs`, and the load report says so), and its lock is compared in its own form, the
    rebuilt experiment written back as that version wrote it, so `mime verify` reproduces it. A new save is version 3.
    Tests with controls: a version 2 folder edited after its lock is refused, versions mixed between document and lock
    are refused, and a load that does not write the form back is refused. Version 2 folders of the 7j study that are on
    branches and not in the repository load through the same reader.
  - **Every derived hash of a build that does not use the rung is unchanged** (script A's surrogate arm and A0's rungs,
    measured before and after); a build that uses it has another derived hash, the node's type having changed.
- **The `mime` command survives a narrow redirected stream, and the text MIME writes is the same bytes on every
  platform** (found by the smoke job on Windows): a report that holds a character the stream's code page lacks (a minus
  sign, a Greek letter in a path) is written with its escape, where it raised after the work was done; every text file
  a save or a run writes is written with `\n`, where Windows wrote `\r\n`; and `mime modal` shows a script on another
  drive than the working directory as it is, where `os.path.relpath` raised.
- **Class renames** (before dev1; no alias, no shim, no migration of saved documents): `mime.scene.DeJonghHelix` is
  `Screw` (registry name `screw`; de Jongh et al. 2025 is cited in its docstring and its `references`), `SolidBox`,
  `SolidCylinder` and `SolidEllipsoid` are `Box`, `Cylinder` and `Ellipsoid` (registry names `box`, `cylinder`,
  `ellipsoid`), and `mime.scene` no longer re-exports `mime.core.Cylinder`, which is imported from `mime.core`;
  `mime.core.DeJonghScrew` is `Screw` (shape `screw`, singleton `SCREW`, switch owners `screw.seam` and `screw.focus`;
  its thread profile is sinusoidal, so a later profile is a setting, not a new name); `mime.fields.UniformRotatingField`
  and `PointDipoleField` are `UniformRotating` and `PointDipole` (their nodes, states and inputs, `UniformRotatingNode`,
  `PointDipoleNode` and so on, follow, as do the registry names `uniform_rotating` and `point_dipole`);
  `mime.dynamics.apparatus.RobotArm` is `Arm` (registry name `arm`, switch owners `arm.limit.low` and `arm.limit.high`).
  Asset names (`dejongh_mlp_v2`, the wall tables) and citations keep their names. The naming rules these follow are in
  CONTRIBUTING.md; rules 2 and 6 are enforced by `tests/test_naming_rules.py`.
- **`arm_tracking`'s step**, unchanged as a callback bit for bit with its orientation feedback off (A0's setting): its
  inverse kinematics' iterations unrolled, so that its switches sit in its step, not in a loop's body; with the
  feedback on, its target's quaternion is taken from `q q^T`, linear in the turn's matrix, in the held orientation's
  hemisphere, which needs no case of the largest diagonal term and has a derivative everywhere off the hemisphere's
  edge (the case form's reverse-mode derivative was NaN where a discarded case took the square root of zero, at a
  turn with a zero component, the identity among them); within the 0.2 oracle's bounds.
- **The de Jongh screw carries its turn exact** (a 7d follow-up): `screw`'s turn along its axis
  (`Screw.turn`), which the helix's samples share, is carried to twice the working precision, so that a point
  rounds at its distance from the axis however far along the body it lies, not at the turn's size (`eps nu L / R` at
  the far end). The kind-conformance harness judges a sample's normal on `NORMAL_ROUNDING_ULPS` units in the last
  place of `1 + l |dn/dp|`, `l` the sample's rounding length (`Samples.rounding_length`, new and optional; the distance
  from the origin where a kind gives none), a declaration recorded with its measurement against exact arithmetic, in
  place of 7d's tolerance fitted from its failures, 64 units of `1 + |p| |dn/dp|`. The screw's values change in their
  last digits only.
- **The drag surrogate reads its shape from the body's kind** (phase 7d), as `drag` 0.3.0: ν, κ and L from the hull's
  family parameters (the de Jongh helix only, unturned and at the body's origin: `drag.body-kind`; another modulation
  or count of starts: `drag.outside-training`), the outer radius from its radial reach, and the tube through the
  region's registered shape. It no longer reads the values `helix_wavenumber`, `length` and `radius`, and its node
  takes the outer radius and the tube's shape, so a document naming `drag` 0.2.0 is refused at load (`load.version`);
  what it computes is unchanged. Script A's body carries
  FL-9's helix as its shell; its golden outputs and frozen reference hold. The record arm's shell is the helix, whose
  closed-form volume equals A's declared volume to two units in the last place, in place of the cylinder of equal area.
- **Breaking, for a geometry kind registered outside MIME** (phase 7e). `Geometry.drawing(settings)` is now
  `drawing(settings, segments)` and raises `NotImplementedError` by default, so a kind that does not draw itself leaves
  it unimplemented (`implements(kind, "drawing")` says which); its mesh is in the kind's own frame, `segments` facets a
  turn, and `Drawing` loses `translation` and `orientation` (the prim's transform places it). `GeometryPlan.drawing` is
  gone: the stage spec draws, at `stage_spec(plans, segments=...)` (`DRAWING_SEGMENTS`, 64). A kind still written as
  `drawing(self, settings)` is refused at the build as `entry.defect`, naming `drawing() raised TypeError`: rewrite it
  with the second argument, in its own frame.
- **A stage saved before phase 7e does not load**: the stage writer writes other bytes from the same document (native
  primitives, `MimeGeometryAPI`, the proxy and region prims, a placement that is the prim's transform), so a folder
  from before is refused with `load.stage-hash` ("the stage writer wrote other bytes from the same declarations"),
  checked on script A saved by phase 7d and loaded by 7e. Save it again. No kind's version moved: a version names what
  a kind computes and accepts, which 7e does not change (script A's results are bit for bit 7d's), and the stage's form
  belongs to the writer, whose change the lock's stage hash exists to report; that holds only while no folder outside
  development exists, and a kind's change to what it accepts or computes is a new version, as `drag` 0.3.0.
- **The tube reads `piecewise_jump`** (phase 7d): the cylinder's in-band Jacobian, the projection across the axis, is
  held by a standing test as a recorded artefact, which clears its `stop_gradient` line. A geometry kind's switches are
  its shapes', inherited, and `mime docs check` reads them so.
- **`Magnetic.Torque` reads a field that varies in space** at the body's centre of mass, by a second support row
  (dev1, phase 7a2), whose note states its validity: the field varying little across the body, and outside a point
  dipole's near field (MIME-ANO-100). A uniform field is read as before, with no points: its outputs are unchanged.
- The inverse kinematics' property is split in two (phase 7a2): from any seed within the joints' limits a solve reports
  where it ended truthfully, never `reached` where forward kinematics says otherwise; from a seed in its stated basin
  (near the target's joint positions, the Jacobian well conditioned all the way) it reaches the target. The single
  property claimed convergence from seeds outside a local solver's basin, and failed about one CI run in nine. The
  reach tolerance is unchanged.
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
- **The table evaluator's read, profiled by its operations** (ahead of phase 7h): by operation counts on the lowered
  and compiled read, never by wall time (`tests/differential/table_profile.py`), and held there by guards on the
  compiled read with the 7f read as their failing control (`tests/unit/test_table_operations.py`).
  - An axis is located without a search wherever a bucket index of its breaks compares a query with no more breaks
    than the search takes steps (`mime.core.switch.Buckets`, `switch.searchsorted(..., buckets=)`): the search's index
    exactly, on every number a dtype holds; the read is the 7f read bit for bit.
  - Each axis's stencil weights are evaluated first, and a window's values summed with them as differences from its
    first knot's, the windows as differences from the first corner's value, so that a derivative along any axis is a
    weighted sum of differences. The last axis's window is gathered for each combination of the leading axes' window
    knots: those of the leading axes nearest the last together, as many as keep one gather within
    `Table.gathered_at_once` elements (by default `GATHERED_AT_ONCE`), the rest in a loop that sums each gather as it
    is gathered. A step's few queries are read in one gather and a few kernels; millions in a loop that holds one
    iteration's windows. Each axis is located at its own argument's shape, so a read over pairs of points, a radius
    given as a column and a row, locates each point's radius once.
  - An axis wider than its dtype's range, or whose buckets' scale would be a denormal, keeps the search: a bucket
    there would misplace a query (a float32 axis from -3e38 to 3e38 read 0.29 for 0.41, every flag the same).
  - `Table.scheme` takes a scheme per axis, `linear` or `cubic`, for MIME's own tables. Four settings of a table trade
    memory for time: `stored_polynomials` and `float32_values` (off), `recomputed_in_reverse` (on: on the CPU it keeps
    fewer bytes, gathers fewer elements and launches fewer kernels than keeping the windows, for a few more
    operations; the GPU is unmeasured) and `gathered_at_once`. `float32_values` refuses a table whose stored values
    float32 holds only past its range or below its normal numbers. The settings are recorded when tables enter an
    experiment's saved form, with their first consumers.
  - `MIME-ANO-115`'s boundary reworded: a query on a one-point axis that carries a tangent (even a zero one given to
    `jvp`) keeps the forward NaN, one that carries none in the mode used does not; pinned by a test. The Cholesky
    property's check gains a control at its resolution: a loss of definiteness past its rounding allowance with every
    diagonal positive is refused on every draw.
- `cholesky_resistance` builds the 6 x 6 factor whole-array (one gather through a constant index map, `jnp.where` for
  the diagonal's softplus and the zeroed upper triangle) instead of 21 single-entry writes, one operation each in the
  compiled step: equal to the loop bit for bit in value and gradient, in float32 and float64 (the first item of the
  performance budget).

### Added
- **The controller inside the compiled step, its first mode** (dev1, phase 7b2): `set_controller(..., mode="graph")`
  (or `1`).
  - **Its node** (`mime.nodes.control.ControllerNode`) runs the very step its law gives the callback, at the
    controller's rate, which MADDENING's multi-rate schedule keeps: between ticks it holds the controller's writes.
    Its reads are edges from the state into it, its writes edges into the commands; the graph declares no external
    input. A latency is a delay line (`DelayNode`) of that many base steps. The run is one compiled scan, continued
    from run to run with the controller's state in the graph's.
  - **The latency rule holds as for the callback**: the same law in both modes writes the same at the same steps on
    7b1's grid of periods 1, 2, 3 and 5 by latencies 0, 1, 2, 4, 5, 9 and 11, and off it (periods 7 and 4 by latencies
    3 and 12, and 3 by 3): a law that writes whole numbers (`Counter`) bit for bit in every field of every node, a law
    that reads the pose and filters it (`Follower`) within 8 single-precision units in the last place per step of each
    field's largest magnitude (measured at 2% of that bound). The controller's node runs first in each step, so its
    reads see the state at the start of the step and its writes act within it; the rest runs in the order MADDENING
    runs the callback's graph, so no other edge reads another step than it does there (with the canonical order
    alone, a commanded segment ran after the field it carries, which read its pose a step late). The build refuses
    an order that breaks either (`post-build.controller-order`), and a read of a node the controller's writes do not
    reach, which runs before it whatever the order (`controller.unreached-read`).
  - **The trace check**: a step that cannot run inside the compiled step (Python on a traced value, a call to the
    host, a state or writes that change fields, shapes or dtypes) is refused for this mode, naming why, with the
    callback as its action (`controller.untraceable`). The step is also traced twice, and one whose two programs differ
    (it reads a counter, a global or a random generator its own calls change, which the compiled step would freeze at
    its trace) is refused (`controller.impure`). A step with a side effect on the world (`print`, a file) is not seen
    by any trace: it must be a pure function of its state and reads (documented on `Controller` and `ControlLaw`),
    or mode 1 runs the effect once, at the build, where mode 2 runs it at every tick.
  - **A run need not be whole periods** in mode 1; the callback refuses one that is not (`ValueError`). Splits off the
    period rejoin the whole run bit for bit, so the swap test must run whole periods.
  - **Gradients**: a derivative of the run passes through the controller's writes inside the compiled step, and
    stops at them as a callback. `arm_tracking` declares its switches, written through `mime.core.switch` (its reach,
    the inverse kinematics' half turn and series, the orientation feedback's hemisphere and the fallback where the
    body's z lies along the home pose's x), and is `piecewise_jump`, derived from them, no longer `cut`; the build
    report says what each mode gives. The status is recorded per placement, as #176 records it per rung: a controller
    placed as a callback is `cut` whatever its law (`Controller.callback_gradient`, the binding's), in the lock's
    resolved entry, in the registry export and on the entry's page, and `mime docs check` keeps its line under
    "gradients cut or unverified", with its route (mode 1).
  - `mime.core.multibody`: `shorter_way` and `shorter_rotation_vector`, the rotation vector's two switches, each with
    the caller's owner, and `relative_turn`; `rotation_vector`, `pose_error` and `inverse_kinematics` take an `owner`.
- **`Drag.BEM`, the boundary-element rung in free space** (dev1, phase 7g), with its library under
  `mime.nodes.hydro.bem`.
  - **The 0.2 BEM's outputs frozen first**, in the port's own environment, before any of it was ported
    (`tests/differential/generate_port_bem_references.py`): de Jongh's FL-9 at the G0 anchor's and the surrogate
    training data's meshes, a sphere and a prolate spheroid; the library reproduces each on its 0.2 samples within
    rounding, and the 0.2 helix mesh is shown to be a tessellation of the de Jongh kind's lateral surface.
  - **The library**: the regularised Stokeslet's matrix in point forces (symmetric whatever the quadrature weights,
    where the 0.2 traction matrix was not), the solve by `jax.numpy.linalg.solve`'s linear-solve rule, the resistance
    about a centre; the blob a setting times the samples' mean spacing; wall corrections registered by name, the
    free space adding none, so that the tube's plugs in without touching the node.
  - **The rung** samples any hull kind that gives samples (sphere, capsule, de Jongh helix) on the patches written,
    at a spacing written, placed by the body's pose each step; refuses a kind without samples
    (`drag.no-samples`), a patch its kind lacks (`drag.patches`) and a medium filling a region with the free-space
    wall (`drag.wall-kind`). Its derivative is exact: DEFINITION 5.3's three checks pass, with shape derivatives
    through the kind's samples, check 2 on every trainable leaf of the node; the reverse mode is checkpointed and
    keeps no matrix between its passes. The node places its samples at the highest precision, as the solve
    multiplies.
  - **A rung's own gradient status** (`Entry.rung_gradients`, `mime.registry.RungGradient`): an entry whose rungs
    differ declares each rung's status and record, and what a user differentiates through is the rung attached. The
    lock's resolved record holds the attached rung's status, so an experiment whose drag is the bem rung records
    `exact` and one whose drag is the surrogate records `piecewise_jump`, as before; the entry's own is the least of
    its rungs'. The registry export carries it, the entry page prints each rung's status and switches in the rung's
    section, and the family page each rung's status; the docs check holds each rung's status to its own switches.
    The lock's form is unchanged (its `differentiability` is a status, as before), and no saved document on a
    merged revision changes: only the bem rung, which none holds, records another status than its entry's.
  - The node walk holds an entry to its `builds` for every rung any round-trip case builds, not only the first case's.
  - How the de Jongh helix's resistance converges, the 0.2 line's triangle rule against the kind's samples at five
    levels, recorded (`tests/differential/references/bem_convergence.json`), and the kind's path through the rung
    bounded by it.
  - The capsule's sample frame is taken in NumPy from its axis, held fixed when the samples are made, so a traced
    radius or height leaves no switch in where its samples sit. An unconfined rigid body's `extent` and `margin`,
    read by no step, are declared not trainable.
- **The controller, declared once, and its callback mode** (dev1, phase 7b1). `Experiment.set_controller(controller,
  reads=..., writes=..., mode="callback", rate=..., latency=...)` declares the experiment's one controller, a registered
  entry (`mime.registry.Controller`, `register_controller`) whose fields are its settings and which declares the labels
  it reads and writes (`reads`, `writes`, each a `Signal`: units and shape). `law(ctx)` makes it ready from its
  settings in SI: its state at the start, the writes it holds before its first acts, and its step, `(state, reads) ->
  (state, writes)`, a function of arrays the mode inside the compiled step can trace.
  - **Bindings.** A read label is bound to a thing's published state (`"umr.pose"`, a chain segment's
    `"rpm.arm.joint_positions"`) or an observation's output (`"swim_speed:umr.speed"`); a write label to an apparatus's
    command (`"rpm.arm.joint_positions"`), which then takes no drive: the controller replaces it.
  - **Refusals**, each with its action: a second controller (`controller.more-than-one`); a mode that names none
    (`controller.mode-unknown`) or is not built yet (`controller.mode-unbuilt`: the mode inside the compiled step lands
    in phase 7b2, the client and FMU modes in 0.3.0a2); a rate not a whole multiple of `dt` (`run.rate-not-multiple`)
    and a latency not a whole number of steps (`controller.latency`), naming the nearest (a rate, in steps, seconds and
    hertz); a run not a whole number of control periods, naming the nearest runs; a label unbound, unknown or
    bound to no path (`controller.labels`); a write no command has (`controller.write-unknown`); a command written
    twice, by a drive and the controller or by two labels (`controller.written-twice`); a read of nothing published
    (`controller.read-unknown`), or of another signal than the label's (`controller.signal`).
  - **External inputs are derived from the writes**, and from nothing else: the build adds one per write, at its node
    input's shape and dtype, and a post-build check refuses a command without exactly one writer
    (`post-build.command-writers`) and an external input no write derives (`post-build.underived-input`). MIME's own
    scans feed every external input or raise, so no input runs on MADDENING's zeros. The 0.2 line's A0 graph, which
    declared the motor's mount pose an external input and wired it too, cannot be written: a mount is no command.
  - **Mode 2, a callback.** A run is compiled chunks of one control period: the controller is called on the host
    between them with the state as each period starts, never inside a compiled step, and its writes are fed as
    external inputs held through the steps they act on. A run with a controller is a whole number of periods; runs
    continue from one to the next with the controller's state and its waiting writes. A step whose writes are not
    its labels stops the run (`ControllerError`). No derivative passes through a callback, and the build report says
    so.
  - **Latency** in whole base steps: a tick at time t reads the state at t, and its writes act on every step from the
    one that starts at t + latency until the next tick's do; the writes held before the first act on the steps before
    them. Latency 0 is the 0.2 line's: the state at the start of a step, the command over it. The build report states
    the period, the latency in steps and seconds, and the rule.
  - **The per-chunk hook**: `run(duration, hook=..., chunk=...)` calls `hook(Chunk)` after every chunk with its index,
    step, time and every node's state. With a controller the chunks are its period; without one `chunk=` sizes them,
    and a chunked run computes bit for bit what one scan does.
  - **Classed reads** (MIME-ANO-104): each read is `state` ("ground truth, not deployable") or `observation`; the build
    report and the lock's `resolved.controller` list each with its class. `mime.build.refused_reads` is the seam where
    a profile will judge them (`controller.read-refused`); none refuses one yet.
  - **The saved form**: the document's `controller` holds the entry, its mode by word, its bindings, rate and latency;
    a load rebuilds it, and the lock records its mode, period, latency, each read with its class and each write with
    the input it feeds.
  - **`ArmTracking` (`arm_tracking` 0.1.0)**, A0's controller ported from the 0.2 rig: a magnet on an arm's end effector
    held at a standoff above a body along an axis, within a reach, by a per-tick filter and damped-least-squares
    inverse kinematics, writing a kinematic arm's joint positions. Its outputs match the 0.2 controller's on a recorded
    sequence within differences declared before the comparison. It computes in float32, as the 0.2 controller did.
  - **`apparatus.joint_positions`**, a state interface the robot arm publishes, so a controller can read its joints.
- The build report's `controller` lines, the run line naming the controller between chunks, and `RunStructure` filled
  from the declaration; `BuiltExperiment.held_inputs()`; `param_reach_failures` feeds a controller's held writes.
- **The table evaluator** (dev1, phase 7f), `mime.core.tables`: one differentiable reader of tables in pure JAX, in
  the query's dtype, for MIME's solver-output tables and OpenMaterica's material tables.
  - **The description** is OpenMaterica's table (axes with units, knots, declared edges, source range, the scheme)
    on as many axes as a table has, with the consumer's `ConstantExtrapolation` and its basis; every rule of its form
    refuses a table that breaks it when it is made (`TableInvalid`). A readable range of one point is read as
    OpenMaterica's tables T12 and T13 specify: the value there, a derivative that is NaN
    (`mime.core.switch.without_slope`). The evaluator takes one query argument per axis, so a query there that
    carries no tangent in the mode used leaves the other axes' derivatives finite; one that carries a tangent, even
    a zero one, keeps the NaN (`MIME-ANO-115`, narrowed): along that axis in reverse mode, in every direction in
    forward mode.
  - **Schemes**: `linear` and `loglog` (OpenMaterica's, a kink at each knot) and `cubic` (a piecewise cubic Hermite
    with each knot's slope from the parabola through it and its neighbours: continuously differentiable, exact on
    quadratics in each axis); `structure="cholesky"` interpolates a positive definite value's Cholesky factor, its
    diagonal as logarithms, so every value is positive definite. The divergence- and curl-free variant waits for a2.
  - **Never clipped silently**: a read returns `in_range`, `refused` (at or across a declared edge) and `accepted`
    beside the value, which is NaN where the run must stop; `query_range_problems` finds a query range that would
    stop it at build, and `refuse` raises `TableRefused` on the host.
  - **One switch**: locating a query among an axis's breaks, `mime.core.switch.searchsorted` (new: `jnp.searchsorted`
    with an owner, a resolution and an indicator), under the owner of the entry that owns the table;
    `Table.value_order` gives its order. The switch scan's raw families gain `searchsorted`, `digitize`, `interp`,
    `histogram`, `bincount`, `unique`, `argwhere`, `lax.sort`, a floor division and a cast to an integer dtype.
  - A break beside zero is the smallest normal number of the dtype, not a denormal (a compiled read flushes it to
    zero), so an edge or a range end at zero reads correctly; a denormal knot is refused when compiled.
  - `mime.materials.evaluator_table` converts a record's table; only `mime.materials` imports OpenMaterica, now
    tested (`tests/test_import_boundaries.py`).
  - Its oracles: OpenMaterica's table-evaluation cases, analytic functions, finite differences, positive
    definiteness at physical scales and on rough draws, and the 0.2 line's wall-table lookup on two blocks of its own
    grid, the near-wall corner and the axis end with every angle and offset
    (`tests/differential/test_port_table.py`, generated in the port's environment), with its per-lookup cost measured
    against that lookup's (`tests/differential/table_budget.py`).
- **Geometry kinds, the USD side** (dev1, phase 7e). A kind with a native form (`Geometry.native`, a `Native`) is written
  as USD's own primitive (`Sphere`, `Capsule`, `Cylinder`), its attributes the kind's parameters through
  `metersPerUnit`; any other is an `Xform` with its shape settings and a drawn `proxy` child. Every kind's prim carries
  `MimeGeometryAPI`, `mime:geometry` and `mime:geometry:version`, and its placement (`Geometry.placement`) is the prim's
  transform. Derived prims (a proxy mesh, a region's primitive) record `mime:derived_from` and `mime:derived_hash`; a load
  reports one edited after it was written (`derived_prims_edited`) and `mime.scene.derived_findings` finds it, and a
  stale one, on any composed stage. `mime.scene.read_kind` reads a kind back from its prim, its counts (the helix's
  starts) beside its settings; taking a kind from a stage as the authority is stage-first and refused
  (`scene.stage-first`; the reader's `geometry.schema-not-applied`, `geometry.schema-incomplete`,
  `geometry.unknown-kind` and `geometry.native`). `Geometry.region_primitive` gives a region as one of OpenMaterica's
  primitives (`sphere`, `cylinder`, `slab`), exactly or refused with its reason (the tube's lumen clipped at the tube's
  length), and `mime.build.region_handoff` hands them to OpenMaterica's region-writing helper, with the prim each is
  written as. A structure whose kind gives region primitives is named in lowercase letters, digits and underscores
  (`thing.name`), as its regions are `<structure>_<region>` in the manifest. The kind-conformance harness's drawing
  check (`mime.testing.drawing_problems`) holds every drawn kind's vertices on its surface and its volume and area to
  their closed forms; the entry pages gain a Capabilities section, generated from which optional methods a kind
  implements.
- **Geometry kinds, the physics side** (dev1, phase 7d). `registry.Geometry` gains optional build-side capabilities:
  `samples` by named patch (a JAX function of the kind's parameters), `sample_counts`, `area`, `family_parameters`,
  `radial_reach` and `tangent`, with `implements` saying which a kind has. New solid kinds: `Sphere`, `Capsule` and
  `Screw` (de Jongh et al. 2025, Eq. 1; centred on its centroid, the 0.2 body moved half its length along its
  axis with its turn kept), evaluated at run time by the new shapes `sphere`, `capsule`, `slab` and `screw`,
  each declaring its switches (`sphere.centre`, `capsule.segment`, `capsule.end.low` and `.high`, `slab.mid`,
  `screw.seam` and `screw.focus`, where the seam ends at a crest's or a trough's centre of curvature);
  `nearest_point`. A body's hull, its one bulk part of a kind with a radial reach, gives it its
  `extent` (`parts.extent` where it cannot), and the values `extent`, `radius`, `length` and `helix_wavenumber` are
  refused beside it (`value.from-kind`); laws read it through `ctx.geometry(of=...)`. `mime.testing` gains the
  kind-conformance harness (`kind_problems`) and the shape conformance suite (`check_shape`, `shape_probes`,
  `SHAPE_CASES`), and `toy.region_probe`.
- **The kinematic apparatus chain** (dev1, phase 7a2): an apparatus's `chain`, segments by name in chain order
  (`Apparatus(..., chain={"arm": Arm.Kinematic(...), "motor": Motor.Kinematic(..., mount="link_6")})`), each a
  `mime.registry.ChainSegment` publishing the world pose of every frame it names (`apparatus.frame_poses`) the same way
  under every rung. The first segment is placed in the world; each later one mounts on a frame of the one before it,
  with an offset, by an edge the build derives. Its commands are `<segment>.<command>`, each with its drive.
  - `Arm` (`arm` 0.1.0), general over its URDF (a file asset pinned by its SHA-256, which the saved form
    carries under `descriptions/`), with its `kinematic` rung: joint positions commanded, held within the joints'
    limits (the switches `arm.limit.low` and `.high`, kinks), an optional first-order servo, every frame by
    forward kinematics; no mass matrix, and no load: a law loading the chain is refused (`target.kinematic-chain`),
    naming the reserved `dynamic` rung. Every element of the description is classified, each kind not used warned
    about once and listed in the build report and the saved form's resolved record (`descriptions`); a refused element
    is read past only when the experiment acknowledges it with its reason. A tree with a loop closure is refused, with
    the invitation to ask or to register one's own segment. Its capability record states its limits: rigid joints
    and links (MIME-ANO-101), and no load applied, a mounted magnet's reaction among them (MIME-ANO-103); the build
    report lists each entry's limits (`limits of <owner>`). Joint positions at the start beyond a joint's limits are
    refused (`chain.initial`), since the arm would jump to the limit at its first step. The saved form keeps a
    description under `descriptions/<sha256 prefix>-<file name>`, so that two files of one name in different folders
    keep each its own copy.
  - `Motor` (`motor` 0.1.0), with its `kinematic` rung: a prescribed angle or speed about an axis of the frame it
    mounts on; its frames `base` and `rotor`; an ideal source of that motion (MIME-ANO-102).
  - **Mounts**: an apparatus's `parts`, each fixed to a frame of a segment (`Part(..., mount="motor.rotor")`) or to
    the world, with its `at`; `magnets` gives one a magnet's role, its direction declared. In the stage, each segment
    is a prim below the apparatus's, each frame a part is mounted on below its segment's, and the part below its frame
    with its authored transform, a magnet with `MimeMagnetAPI`.
  - The lab fixture registers a chain segment with no URDF, `labx.gantry`, on which a magnet mounts unchanged.
- **`PointDipole`** (`point_dipole` 0.1.0), the field of a magnet part of an apparatus as a point dipole,
  and its gradient by its closed form, at each reader's point; the magnet follows its mount. Its moment is the
  apparatus's `magnetic_moment`, declared as a magnitude along the magnet's declared direction. Its node holds the
  magnet's radius for the near-field limit (MIME-ANO-100). It is evaluated outside the magnet: the field is singular
  at its centre, which no valid state reaches (a `nonzero_norm`, so the entry is `exact`), and the build refuses a body
  that starts inside the smallest sphere that holds the magnet (`provider.reader-inside`). A field provider
  may now read its apparatus's parts (`parts`) and values (`values`).
- **`Magnetic.Force`** (`magnetic.force` 0.1.0), the force of a field's gradient on a body's permanent moment, `F =
  G^T m`, read at the body's centre of mass and acting at its magnet's centroid (`magnet_centre`, a new value of a
  body, derived from its one magnet part or declared), so with its torque `l x F`. In the magnetic family, whose sign
  rule now holds both laws and whose rig takes a fixed magnet for a law that reads a gradient. Its reaction on the
  source is not delivered.
- A dynamics may have rungs (`rungs`, `reserved_rungs`, a `rung` setting), checked as an interaction's are, and states
  its limits in its capability record (`limits`). A solid gives its enclosing radius (`enclosing_radius`).
- The vocabulary's `apparatus.frame_poses`, and `apparatus.link_wrench`, reserved for the dynamic arm.
- The node metadata and symbol tables of `KinematicArmNode` (MIME-NODE-104), `KinematicMotorNode` (105),
  `PointDipoleNode` (106) and `MagneticForceNode` (107), and the hand-written entry files of `arm`, `motor`,
  `point_dipole` and `magnetic.force`. A symbol row may name a reader slot's field as `<name>_{slot}`.
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
  - **The generality pilot** (`tests/docs/test_generality_pilot.py`): the pages of a body that does not swim, the
    `Magnetic.Force` set (`Magnetic.Force`, `PointDipole`, `Magnetic.Torque`, `Gravity`, their nodes, the magnetic
    and gravity families, the `load` mechanism), build with the existing page types, and `Drag` is read the same way.
    Found and fixed: the glance counted "built rungs" for entries with none and said a family of complementary laws
    was "compared on its page" (the family page says it is not); the validity sentence said a condition is checked
    against "the operating point" (now the point or the region each line describes); a hand-written source's title
    left the "What it models" section empty under a second page title (it is inlined one level down); the family page's
    placeholder spoke only of rungs, and a family with one member said its members are not alternatives. Found by the
    pilot's review and fixed: `Gravity` declares its `values`, so its page carries what its family names as a
    material's property; an entry that writes its own `plan` says on its page that what `plan` adds is not in its
    tables; `Magnetic.Torque` declares the validity line its support note states (a gap in #36); interface names and
    delivery targets on an entry page link to the vocabulary and the mechanism page, and the vocabulary page names the
    entries that publish each interface; a count of one reads "1 declared line". Twenty-two planted defects are each
    named by the check, and a hand-written source that uses a swimmer's word is named by nothing; the index column is
    "Built members".
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
    (`Cylinder`, `Ellipsoid`, `Box`, registered geometry kinds with closed-form volume and inertia),
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
  - `fields.UniformRotating` (`uniform_rotating` 0.1.0), its time by step count, and `mime.drives` with the
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
- MIME-ANO-124 (open, major): a motor's unwrapped single-precision angle turns at a biased rate (5.4e-4 at 1 Hz over
  10 s, up to half a percent after a minute), doubling with each doubling of the angle; the rotor's frame, the magnet
  and the field share it. Planned: wrap the angle, or carry it compensated.
- MIME-ANO-118 (open, major): FL-9 in the 3/16 in tube at a spacing of 0.10 mm and a 0.30 mm gap jumps 14% in the swim
  speed (17% axial translation, 34% axial coupling) against its neighbours, reproduced bit for bit, and passes every
  check of the generator; mechanism not isolated, the pose outside the protocol.
- MIME-ANO-123 (open, minor): the capsule's and the sphere's polar-grid caps crowd their poles (0.115 spacings for the
  capsule at 0.178 mm; 0.074 for the sphere at 0.1), the defect MIME-ANO-119 fixed on the screw; unmeasured on them.
- MIME-ANO-122 (open, major): the dynamic motor's rotor inertia about its axis rides on its stator without
  loading the arm: on A0 it is 4.1 times the inertia of the link it is mounted on about that axis, so the arm's last
  joint turns with about a fifth of its inertia; the coupled form is planned for dev1 phase 7a4 (design #185).
- MIME-ANO-121 (open, major): a wall read refused during a run produces NaN while the library documents that it stops
  the run; nothing in `src/mime` reads `wall_accepted`.
- MIME-ANO-120 (open, major): the lateral screw surface's confined resistance has one negative mode on its open ends,
  0.10 to 0.14 mm from the wall. Gains a candidate cause: the lateral patch's ends crowd under refinement (closest
  pair 0.265 spacings at 0.178 mm, 0.15 at 0.10, 0.075 at 0.05).
- MIME-ANO-119 (resolved in the screw kind 0.2.0, major): the closed screw surface's confined resistance matrix was
  singular in float64 (crowded cap centres with the tube's correction), and its derivative not reproducible. With the
  caps in rings the matrix at A has no negative eigenvalue at any phase and the derivative passes its check against
  finite differences; the speed moves by 0.13%.
- MIME-ANO-114 (open, major): its record of the near-wall check gains the margin to indefiniteness at A (phase 7j's
  study M3): the resistance turns indefinite at a coherent x1.030 of the wall correction, unchanged by MIME-ANO-119's
  fix; the table's error at A's own sample pairs is unmeasured. The unchanged onset excludes the caps and does not
  separate the table's resolution from the lateral patch's open ends (MIME-ANO-120). Its near-wall record is on
  0.1.0's caps and not re-measured on 0.2.0.
- MIME-ANO-116 (open, minor): the de Jongh helix's boundary-element resistance is not converged at the 0.2 line's
  counts, and the kind's samples differ from the 0.2 mesh there by up to 5.3% on the diagonal and 5.2% in the axial
  coupling; the converged diagonal lies between the two, the kind's the nearer in translation.
- MIME-ANO-115 (partially resolved, minor): the table evaluator's derivative through a one-point readable range is
  NaN along that axis, in every direction in forward mode when its query carries a tangent (a query there that carries
  none in the mode used leaves the other axes' derivatives finite).
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
- **The screw kind's caps no longer crowd their centres** (MIME-ANO-119; `screw` 0.2.0). Each cap is sampled in rings:
  a sample at its centre and rings out to the rim, each a copy of the section at a fraction of the way out, its count a
  multiple of the starts growing with its radius (above `4 starts + 2`, which the second moments need), equally spaced
  in the section's angle and staggered from the next ring, the outer ring set in from the rim, where the twisted lateral
  surface leans over the cap. No two samples of a cap lie closer than `CAP_SEPARATION` spacings (a declaration, 0.3);
  the samples are themselves after a turn by one start, as the section is; the radial weights integrate the area and
  the first and second moments exactly, solved in exact rational arithmetic, every one above zero. A cap carries as
  many samples as 0.1.0's polar grid, `outward x around`, less the few its symmetry needs (one more than a multiple of
  the starts), so that a boundary-element solve's blob (the samples' mean spacing) barely moves; it carries fewer only
  for a section whose troughs and flanks are too narrow for that many at the separation. The caps' counts are one
  number, `(samples,)`, the samples on each cap. The lateral patch, the shapes and every closed form are unchanged. The polar grid put as many samples on its innermost ring, near the centre, as on its rim; with
  the tube wall kind's correction that made the closed surface's matrix singular in float64 and its derivative noise.
  The screw's golden output moves to its new version; the BEM's free-space golden output, whose case samples a screw's
  closed surface, is regenerated.
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
