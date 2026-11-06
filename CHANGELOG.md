# Changelog

All notable changes to MIME are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The experiment's tree and its compiled graph, shown without running it** (`docs/guide/graph_and_tree.md`).
  `experiment.tree()` is what the experiment declares (structures, media and what each fills, bodies and their parts,
  apparatus with their chain, parts, fields and drives, each law with the things it binds, observations with what they
  read, the controller with its reads and writes), each with its registered id and version: plain indented text
  (`str(tree)`) or data (`tree.record()`, `mime.tree/1`); it is the topology, not the declared values, settings or run
  settings. `built.to_mermaid()` is MADDENING's `to_mermaid` of the
  compiled graph with the registered entry added to each node, and a notebook shows it as a diagram where its front end
  draws a Mermaid fence (JupyterLab 4.1 and after and Notebook 7 do, measured; 4.0 shows the text with a pointer to a
  renderer; VS Code was not measured). Colab does not draw a fence (measured), so there it is `text/html` that draws it with
  Mermaid 11.17.2 from jsDelivr, held to its `integrity` hash; `to_mermaid().display(offline=True)` loads nothing. `mime show <folder>` prints the tree (`--tree`, the default, drawn as a
  tree on a terminal, plain text piped, `--json` as data) or the graph (`--graph`), or both. Either is the same for the
  same experiment whatever the order its script declared things in.

- **Colab notebooks** (`docs/notebooks/`), linked first in the README: a steel ball settling in a tube of water, and
  de Jongh's screw-shaped magnetic robot driven along a tube by a rotating field (tutorial A's experiment). Written
  for a newcomer, step by step in plain words, each installs the released MIME from PyPI, runs its experiment with the
  live display, plots it with matplotlib, shows the browser view, and ends with changes to try.
  `tests/examples/test_notebooks.py` builds each notebook's experiment against the checkout and runs A's for a few
  steps; `scripts/colab_links.py` points every Colab link at one branch.

- **Download gates: a licence's conditions are accepted before a download, and the disk is checked** (design #237), in
  the library, so a script and a notebook meet them as `mime assets fetch`, `mime visuals fetch` and `mime pull` do. An
  asset under a licence with conditions beyond credit (the AR4's meshes; a non-commercial Creative Commons licence; any
  custom licence) is downloaded only once its conditions are accepted: on a terminal a panel with an amber border titled
  "Licence conditions: your acceptance is needed" shows the conditions as short lines, the licence's title and link and
  the notice, then asks, in bold, `Do you accept these conditions? [y/N]` (no by default; a no downloads nothing, exit
  2); a yes is remembered per licence and version (its id and title) in `licences.json` in the config directory (`mime
  assets licences` lists the acceptances, `--revoke <id>` forgets one); the condition lines are looked up by the
  licence's id, so a licence read from a saved lock shows the lines its manifest states; `NOASSERTION` (terms not
  stated) is gated with a panel that says so; a file already in the cache whose licence has conditions and no recorded
  acceptance asks once at its first use (a bake, `visuals view`, a fetch, a pull); a record's text is written as plain
  characters at every prompt. With no terminal it refuses naming the licence and
  `--accept-licence <id>` (Python: `accept_licences={...}`), which accepts that licence for the command and no other;
  `--yes` and `--force` accept none. Every download also reads the free space of the drive that holds the cache: over
  1 GB or a tenth of it, it is asked about (`-y/--yes`); if it would leave under 2 GB or 5 % of the drive, it is stopped
  (`-f/--force`); `visuals fetch` counts 30 % of the meshes' size more for what baking writes. New refusals:
  `asset.licence-not-accepted`, `asset.licence-declined`, `asset.disk-low`, `asset.download-large`,
  `asset.download-declined`, `asset.licences-file`. `mime assets list`, `assets sources`, `assets licences` and `visuals
  list` are tables on a terminal; `mime pending` sets the lines to paste apart as blocks that are never wrapped; `mime
  cache clear` says how much it freed. Output that is not going to a terminal is byte for byte what it was.
  **Moves:** the asset manifest format is `mime.asset_manifest/2` (it adds `conditions` to a custom licence; `/1` is still
  read, and a `/1` manifest holding `conditions` is refused); the AR4's `visuals.json` is `/2` with its licence's four
  condition lines; a saved folder's lock (form 6) holds the manifest as `/1` without the lines. `mime pull`'s question
  about a large download follows the new rule (1 GB or a tenth of the free space, not 1 GB alone), its refusal without a
  terminal is `asset.download-large` (it was `pull.assets-large`), and `mime.saved.ASSET_CONFIRM_BYTES` is gone
  (`mime.assets.CONFIRM_ABOVE_BYTES`). `-y`, `-f` and `--accept-licence` are options of `assets fetch`, `visuals fetch`
  and `pull`.

### Changed

- **A folder saved by another development release is refused by its own refusal, `load.other-release`.** Development
  releases (0.3.0.devN) make no compatibility promise, and MIME's version is part of a saved folder's input hash, so
  `0.3.0.dev1`'s folders are not loaded by `0.3.0.dev2`. Until now the refusal was `load.input-hash`, whose message said
  the document was changed. The new one names both versions ("saved by MIME 0.3.0.dev1, and this is MIME 0.3.0.dev2"),
  says that development releases do not load each other's folders, and says what to do: run the script that built the
  experiment under this MIME and save it again, `built.save(folder)`, or load the folder as it is under the MIME that
  saved it. A folder unpacked from a published record (`mime pull`) says so and names the record's page.
  It is the first thing `Experiment.load` does, before any entry resolves, so a folder whose document no longer resolves
  under this release is still refused for the release that saved it. `load.input-hash` is raised only where the
  versions agree, for a document changed after its lock was written; one loaded and then edited (a notebook cell run
  after the load) is refused in words about the edit ("edited after it was loaded"), with the actions that work: load
  the folder again, or make the same edit in the script that built the experiment and save from there (a loaded,
  edited experiment can be neither built nor saved; `Experiment.based_on` will be named when it lands). The
  compatibility policy for later releases is designed before the first users; this is not it, and two final releases
  (0.3.0 and 0.3.1) still meet `load.input-hash` and its words about a changed document.
- **`mime verify`, `mime env check` and `mime registry` draw tables on a terminal.** `mime verify` shows its checks as a
  table (the saved form, the input hash, the environment, the derived graph, the rerun and what it was held to, and a
  row for each field compared: "equal, bit for bit" where the comparison was bit for bit, otherwise the tolerance it was
  held to ("within rtol ..., atol ..."), and a field that moved with the numbers that moved it and the tolerance in
  words), with the verdict as the one line under it. `mime env check`, when it passes, shows the pins against what is installed, a row for
  each pinned distribution (wanted, found, verdict, who pins it), then its other lines. `mime registry` shows one table
  grouped by kind, a row for each entry with its name, its version and the settings it declares; a long cell wraps and
  is never cut. Output that is not going to a terminal, and a terminal under `--plain`, `NO_COLOR` or `TERM=dumb`, is
  byte for byte what it was; `mime registry --json` and `mime env check --record` are unchanged. A failing
  `mime env check` on a terminal opens its refusal's block with the table of the pinned distributions that differ
  (wanted, found, verdict, who pins it); the block keeps its codes, what to do and every reason the table does not
  hold, and drops the rows the table holds. On a terminal too narrow for five columns (under about 90) who pins is a
  line under the table instead of a column. Piped, it is the refusal's text as before.
- **A declaration made twice is kept once when it is the same, and a notebook's cell can run again.** The same
  `exp.add_*`, `attach`, `observe`, `declare`, `source`, `set_controller`, `licence` or `permission` made a second
  time, the same in its type, its settings with their units and reasons, and what it binds, is kept once with a
  warning, in a script and in a notebook; it was refused (`id.duplicate`, `thing.duplicate`, ...). A changed one is
  still refused in a script. In a notebook (Jupyter, Colab or IPython) it replaces the earlier one, with a warning, so
  a cell edited and run again takes effect. That includes a structure, medium, body or apparatus (the same name and kind)
  that laws, observations, values or the controller name: they hold it by its name, so they find the new one, and the one
  warning names each (change `field_strength` and run the field cell again). A name declared again as another kind, or a
  declaration that holds the earlier thing's object and not its name, is kept beside the new one and refused at the build,
  saying a line or cell has probably run twice. A law or an observation that a source or the controller names is
  replaced too, and the source or controller follows it by its id (the note lists it). A value declared of a thing replaced still applies and overrides what the new declaration brings, and the
  warning says so. The warning is also recorded on the experiment (`plan().replaced` and the build report), because a
  warnings filter removes a warning; a warning raised as an error stops the call before anything changes. The words
  say "a notebook or IPython": the rule also applies to a script run by `ipython script.py` or `%run`. The same scene
  reference made again is kept once. The notebooks pin 0.3.0.dev1 and keep its advice (run again from the cell that makes the experiment) until they move to the release that ships this.

- **`exp.remove(name or handle)`** takes a declaration out, in a script and in a notebook: a thing, a law (by the id
  `attach` returned), an observation, a declared value, a source, or the controller (`exp.remove(kind="controller")`).
  It is refused while another declaration still names it (`remove.named`), listing each with the cell or line that made
  it and the call that removes it; nothing is removed with it. A removal is recorded
  (`plan().removed`, the build report's "removed in this session") and raises no warning, being your own call
  (a replacement warns, because it happens without your asking). Removing and declaring
  again under the same name is how a thing's kind changes.
- **A build, a result and a view that predate a replacement or a removal say so.** `built.stale`, `result.stale` and the
  view's `stale` name what changed since (`this build predates the replacement of the apparatus 'field'`). Running
  (`run`, `run_to`), saving or opening a live session on a stale build is refused in a notebook or IPython
  (`run.stale`, action: build the experiment again) and warned about in a script; the build report's first lines, a
  result shown (`repr`, `text()`) and a view (`viz.display` of a run folder `run_to` wrote, `RunWritten.view`) carry the
  same line. A fresh build clears it; a change put back, or a declaration only added, is not stale. The run folder and
  the report's summary written into it are not changed. A file inside the run folder and `viz.web` carry the mark too; a copy of the folder, which no session knows, is shown
  with no mark. A result pickled (sent back from a worker process) keeps no link to the experiment: it still pickles and
  deep-copies, and says "staleness unknown" (`stale_unknown`), neither stale nor fresh.

### Fixed

- **In float32 a screw's second build in a process held the same matrix as its first, to the bit** (MIME-ANO-138). The
  first build of the screw in a process made the MRS node's held matrix from cap samples in float64; every later build
  made it from the same samples rounded to float32 (1e-7 of themselves), because the cap layout the process caches was
  handed to each build as the same arrays, and JAX gave back an earlier conversion of them. The drag's resistance then
  differed between the first build and the rest by 1e-5 of its entries from step 0. `cap_rings` now hands out a copy
  of the layout each call. Float64 is unchanged; `mime verify` was never affected (it compares two later builds).
- **`Experiment.save` and `Experiment.load` warn at the script's line.** A `DescriptionWarning` or a range warning raised
  while they build showed MIME's own line (`return self.build().save(...)`); the build now finds the script's frame
  itself, whichever public call it was reached through (`mime.errors._frames.warn_at_script`), and so does the view's
  output-size warning when it comes through `RunWritten.view`. Under `mime run`, where the only caller is the console
  script's stub, a range warning is printed as `mime: warning: <text>`, once, with no source line, as a description's is.
- **A `WebPage` in a notebook is no longer twice its size.** Its plain-text form held the whole escaped page (427 of the
  553 kB in tutorial A's view output); it is now one line saying what it is, its size and how to open it. The HTML form
  is unchanged.
- **A field arrow in the browser view is capped at twice its body's largest extent.** The field at a body, drawn at
  8.3 m per tesla, was 83 mm long at 10 mT beside tutorial A0's body, 7.5 mm long. The view now draws an arrow at a body
  no longer than twice the largest extent of that body, and no arrow longer than a quarter of the diagonal of the scene
  without its arrows (an arrow at no body has the scene's alone); under the cap an arrow's length stays proportional to
  its vector. A0's own field, about 1 mT in its first samples, draws an 8 mm arrow, within the cap and unchanged. The cap is in the view alone (the
  glTF and the page), the page's notes name each capped arrow by its prim path, and the stage and the recording are
  unchanged.

## [0.3.0.dev1] - 2026-11-05

The first release of MIME 0.3, a rewrite of MIME on [MADDENING](https://microrobotica.org/maddening/) 0.4. It is a
development pre-release: install it with `pip install --pre mime-engine`, since a plain `pip install mime-engine` still
installs 0.2.0. Its API, its saved form and its results may change before 0.3.0. It replaces the 0.2 line rather than
extending it, so a 0.2 script does not run on it unchanged (see *Upgrading from 0.2* below).

In MIME 0.3 an experiment is declared, not wired: its bodies, media, structures and apparatus, the laws attached between
them, and what is observed. `exp.build()` checks the declaration (units, frames, inputs, each law's declared range) and
turns it into a MADDENING graph, and `built.run()` runs it. Every value is declared with its source or taken from a
stated default, and the build report says which.

### New against 0.2

- **Experiments as declarations**: `mime.Experiment` with `Body`, `Medium`, `Structure`, `Apparatus` and `State`;
  laws are registered entries attached with `exp.attach(...)`, and observations are added with `exp.observe(...)`.
- **Quantities checked by dimension** (`mime.units`: `mm(...)`, `Hz(...)`, `mT(...)`), turned into SI at the build.
  A frequency in `Hz` counts turns and an angular velocity in `rad_per_s` radians: each site says which it takes
  and refuses the other with both conversions to write, and arithmetic keeps what a quantity counts
  (`Hz(5.0) * (2 * pi)` is still in Hz; write `rad_per_s(2 * pi * 5.0)`).
- **Profiles**: `exploratory`, the default, runs with nothing stated and lists what it left pending (`mime pending`);
  `research` asks for every reason and refuses a pending one by name. `strict` is reserved, and refused until it is built.
- **Saved experiments that reproduce**: `built.save(folder)` writes the experiment's document, a lock that pins every
  entry, asset and environment by hash, and its USD stage; `Experiment.load(folder)` rebuilds it, and `mime verify`
  reruns it against its reference steps, bit for bit in the environment that saved it.
- **The scene as a USD stage** (`Scene.new`): frames, regions, and bodies bound to prims.
- **Materials from OpenMaterica records**, read at the experiment's conditions, with the record's verdict passed through.
- **Physics entries**: the rigid inertial body; gravity and buoyancy; a containment contact that keeps a body's hull in
  its lumen; the magnetic torque and the force of a field's gradient; a uniform rotating field and a point dipole; the
  drag, with a surrogate rung and the method of regularised Stokeslets (`Drag.MRS()`) in free space and in a tube; and
  geometry kinds from the sphere to the screw and the tube.
- **Robot arms and motors as apparatus chains**, kinematic or dynamic, read from a URDF, with mounted magnets and
  spinners, and a controller in the loop (`arm_tracking`), as a callback or inside the compiled step.
- **Results to read**: `RunResult`, each observation's outputs by name with their units, flags, post-run watches
  against each law's declared range (and a warning before a run whose declared operating point is past it), exports to
  CSV, NPZ and MATLAB files.
- **Live sessions and recordings**: `built.live()` and `mime serve` speak the `mime.live/1` protocol over ZMQ to a viewer
  such as MICROROBOTICA, and `mime run --stage-every` records a USD layer of the run.
- **A recording in a browser**: `mime visuals web <folder>` writes one self-contained page, the viewer inside it, so it
  needs no network (`--open` opens it), and `mime.viz.display(folder)` shows the same view in a notebook such as
  Colab. A mesh whose licence forbids redistribution (the AR4's) is never put into a page: simple shapes are drawn
  in its place, and the page says so. `examples/sphere_settling_tube.py`, a steel ball settling across a tube of
  water onto its floor, is made to be viewed.
- **A run's live display in a notebook**: in Jupyter or Colab, `built.run`, `built.run_to` and `built.save` show
  `mime run`'s live display through the notebook (the phase, then the bar, then, in `run_to`, the saving of the run
  folder); `progress=False` turns it off, and outside a notebook nothing is shown unless `progress=True` asks for it.
- **Sharing**: `mime pack` writes a saved experiment as one bundle, `mime pull` checks a bundle and unpacks it, and
  `mime push --to zenodo` publishes one as a record with its card.
- **Assets by hash**: a large file is referenced by its sha256 with its licence and fetched on request
  (`mime assets fetch`); a file whose terms let no copy be hosted is never shipped (the AR4 arm's meshes are fetched
  from their owner), and `mime assets audit` checks a package for one.
- **The `mime` command**: `run`, `serve`, `verify`, `pending`, `pack`, `pull`, `push`, `card`, `env check`, `modal`,
  `assets`, `visuals`, `registry`, `docs`, `completion`, and `mime --version`.
- **Errors with codes**: every refusal has a code, what to change, and its own documentation page. Three inputs
  that would give a number are refused in every profile: a dynamic viscosity of zero or less
  (`value.not-positive`), a body whose centre starts outside the region its medium fills (`state.outside-region`),
  and a reading of an observation whose window starts at or after the run's end (`run.window-past-end`). A
  renamed entry says its new name (`call.renamed`, as for the drag's `Drag.MRS`), and an entry made without a
  required setting names it and why it has no default (`Screw`'s `starts`, `call.missing-setting`).
- **Extending MIME**: a lab's package registers its own entries through the `mime.entries` entry point, and tests them
  with `mime.testing`'s harnesses.
- **Generated documentation**: entry, node, error and reference pages generated from the code, and a user guide whose
  tutorials are run by tests.

### Changed

- **Licence: Apache-2.0** from 0.3 on. The 0.2 line (the tag `pre-0.3` and earlier) keeps the licence it was
  distributed under, LGPL-3.0-or-later, as `NOTICE` says. The wheel also carries the glTF converter derived from usd2gltf (MIT) and the
  viewer's bundle (model-viewer, Apache-2.0, holding three.js, fflate and gainmap-js under MIT and lit under
  BSD-3-Clause), each with its licence, so its licence expression is `Apache-2.0 AND MIT AND BSD-3-Clause`.
- **Requirements**: Python 3.12 to 3.14 (0.2: 3.10 or later), MADDENING 0.4.2 or later within 0.4 (0.2.0: MADDENING 0.2.1 or
  later within 0.2; the tag `pre-0.3`: 0.3.x), and OpenMaterica.
- **0.2's nodes and effect models are entries.** The
  [migration table](https://microrobotica.org/mime/0.3.0/pages/reference/migration) gives each public name of 0.2 the
  entry an experiment attaches in its place: `RigidBodyNode` is `dynamics
  rigid_inertial`; `MLPResistanceNode` and `StokesletFluidNode` are the `surrogate` and `mrs` rungs of `interaction
  drag`; `ExternalMagneticFieldNode` is `field_provider uniform_rotating`; `PermanentMagnetNode` is `field_provider
  point_dipole`; `PermanentMagnetResponseNode` is `interaction magnetic.torque`; `GravityNode` is `interaction
  gravity`; `SurfaceContactNode` is `interaction contact`; `RobotArmNode` is `dynamics arm`; `MotorNode` is `dynamics
  motor`; `ConstantInputNode` is `drive drive.constant`; the effect models `GravityEffect`, `HydrodynamicModel`,
  `MagneticModel` and `RobotArmEffect` are `gravity`, `drag`, `magnetic.torque` and `arm`.

### Removed

- What has no 0.3 counterpart yet: the fluid solvers (the IB-LBM and FVM fluid nodes, the GNN-corrected FVM, the LBM
  far field, the defect correction, the CSF flow), the flexible body, the friction contact, the induced and
  superparamagnetic responses, the coil array's field (it returns with its calibration asset), the field superposition,
  the phase tracking, and the rigid body's overdamped mode. The effect-model pilot's package, `mime.effects`, is gone;
  its models are the entries above.
- The 0.2 line's experiments, examples, scripts and documentation. All of it remains at the tag `v0.2.0` (the code PyPI holds as 0.2.0) and at the tag `pre-0.3` (the line's later,
  unreleased last commit).

### Upgrading from 0.2

- A 0.2 script built a graph from nodes by hand; a 0.3 script declares the experiment and attaches registered entries,
  which build their own nodes. Rewrite it: find each node it built in the
  [migration table](https://microrobotica.org/mime/0.3.0/pages/reference/migration), attach that entry, and declare the
  values the node took as arguments. [Tutorial A](https://microrobotica.org/mime/0.3.0/pages/guide/tutorial_a) is 0.2's
  script A rewritten this way, and held to the 0.2 code's outputs.
- To stay on 0.2, pin `mime-engine<0.3`. That installs 0.2.0, whose code and documentation are at the tag
  [`v0.2.0`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/v0.2.0) and which needs MADDENING 0.2.1 or later within 0.2. The tag
  [`pre-0.3`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/pre-0.3) is a later commit of the 0.2 line that was never uploaded to PyPI: it runs on
  MADDENING 0.3.x, and a pin on `<0.3` does not install it.

### Known limits

[Scope and limits](https://microrobotica.org/mime/0.3.0/pages/guide/scope) says what MIME declares and what it does not
claim: a declared limit is a statement, not a measurement, and agreement with the 0.2 code is not accuracy. In short:

- The drag surrogate's weights are 0.2's, and inherit the defect of 0.2's wall tables (MIME-ANO-117). Experiments that
  use them are held to a reference on the same weights, not to the physics.
- The MRS rung in a tube reads a table of 209 MB, which a build never downloads: fetch it once with
  `mime assets fetch unit_tube_R0.997`.
- Development releases (0.3.0.devN) make no compatibility promise. A folder saved by one development release is refused
  by the next, so keep the script that built it and save it again with the new release. That holds of a saved folder in every
  use (load, verify, run, pack), of the folder a bundle unpacks to, and of the `experiment/` folder inside a run
  folder; a run folder's outputs still read.
- Saving a run (`run_to`, or `built.save`) builds the experiment again, then compiles and runs the reference steps
  that `mime verify` checks against: a cost of about ten seconds on a laptop's CPU, and up to about two minutes on a
  Colab GPU, mostly compiling, whatever the run's length. A run that is not saved (`built.run`) does not pay it. A
  cheaper save is planned for a later development release.
- Off Linux, a run is verified within its lock's tolerances, not bit for bit, and there is no GPU build for macOS or
  native Windows ([platforms](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/docs/platforms.md)).
- The open anomalies are in the
  [registry](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/docs/validation/known_anomalies.yaml).

Every change made on the way to this release, as it was logged, is in the
[development record](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/docs/development/changes_0.3.0.dev1.md).

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

[Unreleased]: https://github.com/Microrobotics-Simulation-Framework/MIME/compare/v0.3.0.dev1...HEAD
[0.3.0.dev1]: https://github.com/Microrobotics-Simulation-Framework/MIME/compare/v0.2.0...v0.3.0.dev1
[0.2.0]: https://github.com/Microrobotics-Simulation-Framework/MIME/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Microrobotics-Simulation-Framework/MIME/releases/tag/v0.1.0
