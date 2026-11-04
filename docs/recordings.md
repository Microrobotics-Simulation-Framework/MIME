# Recordings and the stage's visuals

A recording is a run's poses and field vectors as time samples over the stage of the experiment it ran, so a run can
be replayed in any USD viewer, scrubbed, and read back. It lives in the run's folder, beside its outputs, and is its
experiment's by construction.

## Writing one

<!-- mime-fragment: the recording call in a reader's script, with a built experiment the reader has -->
```python
written = built.run_to("a0_run", s(2), stage_every=ms(16.5))   # a sample every 33 base steps
written.recording                                              # a0_run/recording.usdc
```

<!-- mime-fragment: the run command for a saved experiment the reader has -->
```sh
mime run a0_saved --duration 2s --out a0_run --stage-every 16.5ms
```

A live session records between `session.record(True)` and `session.record(False)` (the `record` command), a sample
at each chunk's end ([live sessions](live.md)). `stage_every` is a whole number of the run's chunks: the run is
chunked for it, which changes nothing it computes, and keeps no step's state (no `history=True`).

## What it holds

- **`recording.usdc`**, in the run folder. Its one sublayer is the folder's own `experiment/scene/scene.usda`, by a
  path inside the folder, so the folder still resolves when it is moved or copied to another machine.
- **Overs on the bound prims** (`built.binding()`): a body's and each chain frame's `translate` and `orient`, in double
  precision, so a single-precision state's values are held exactly; a field vector's `mime:vector:value`, with its
  arrow placed at the body, turned along the vector and scaled to its length. A `translate` is in the stage's units:
  the state's metres divided by the stage's `metersPerUnit`, so that composed with it (as a viewer composes it) a
  prim is where the run put it, on a stage in millimetres as on one in metres.
- **Its time code is the sample's index.** `timeCodesPerSecond` and `framesPerSecond` are the samples per simulated
  second, so a time code over `timeCodesPerSecond` is the simulated seconds since the layer's first sample, which is
  the state where recording began (the run's start for `run_to`; for a live session, the step `record(True)` took
  effect at). A live session that turns recording off and on joins its stretches with consecutive codes, so across the
  gap the codes are not seconds. `customLayerData` holds `mime:recording`, with the experiment's name and input hash,
  the base step, the spacing, and every sample's step.
- **Crate 0.8.0**, the oldest version usd-core writes for what it holds, which older OpenUSD releases (MICROROBOTICA's)
  read; exported fresh, with no dead bytes of an earlier save.

Its poses are the run's states at its samples' steps exactly, it opens with no finding of usd-core's validators
(`mime.scene.check_stage`), and a moved folder composes: each is a test.

## The stage it is over

The stage (`scene/scene.usda` in a saved folder, `experiment/scene/scene.usda` in a run folder) is written from the
experiment's declarations alone, the same bytes for the same declarations. For a live client and a recording it holds:

- **A prim for every frame of every chain segment**, below the segment's prim, with its frame's name as `mime:frame`
  (a name USD does not take as a prim's is written with each such character `_`). Each is placed in the world, its
  transform stack reset, at its pose at the start: an arm's links, a motor's base and rotor.
- **A prim for every field a law reads at a body** (a flux density), below its apparatus: an arrow, hidden until a
  recording or a client gives it a value.
- **What each frame draws** below it: its description's visuals (`visuals/`), a box, cylinder or sphere as its USD
  primitive, a mesh as a prim naming its manifest and the sha256 of its source; and the simple shapes that stand in
  for meshes the stage does not hold (`fallback/`): a bone from the frame's origin to each child's, and a link's
  collision primitives.
- **Drawn proxies within a triangle budget**: a geometry kind with no USD primitive (the screw) is drawn at fewer facets
  a turn where its drawing would hold more triangles than the budget (`mime.build._stage.PROXY_TRIANGLES`), which kept
  the screw's proxy from being nearly all of every run folder; a test holds script A's screw within its budget and
  near its full drawing, measured on every vertex.

The stage writer has a version, which the lock records (`stage.writer`, `mime.scene.STAGE_WRITER`). A folder saved by
the first writer (before every chain frame, visual and vector was drawn, and every proxy was drawn whole) is rebuilt
by the first writer when it loads, held to its lock byte for byte, and verifies as it did; saving it again writes the
current writer's stage.

## Visuals MIME does not ship

Some descriptions' meshes cannot be redistributed. The AR4's are Annin Robotics' design files, under its
non-commercial licence, which forbids redistributing or hosting them in any form, converted or decimated copies
included, and allows educational, research and personal non-commercial use. The description's kinematics and inertial
values are MIT. Neither distribution of MIME holds the AR4: its description folder, `examples/descriptions/ar4/` in the
sdist and the repository, holds the URDF with its MIT licence and a **manifest** of its meshes (`visuals.json`), never
their bytes. An experiment that reads the URDF finds the manifest beside it, so choosing the robot brings its meshes'
records:

<!-- mime-fragment: visuals commands that fetch from the network, shown and not run here -->
```sh
mime visuals fetch examples/descriptions/ar4/   # prints the credit and the terms, then downloads and bakes
mime visuals fetch a0_saved/                    # the same, from the records a saved folder's lock holds
mime visuals list                               # each set an installed package ships, and each baked one
mime visuals view a0_run                        # the layer that opens the run's recording with the meshes drawn
```

- **The manifest** is a named group of ordinary asset records (`mime.assets`): each file by the path a description
  writes it with, its sha256 and its git blob, at the upstream repository at a pinned commit, with its licence (an
  SPDX `LicenseRef-` whose custom licence says redistribution is not permitted) and its role (presentation: no result
  depends on it). Its tier is the licence's verdict, `referenced`, never a value written by hand. A saved folder's lock
  holds the manifests its stage draws (`visuals`), so the records travel with the folder and its bytes never do.
- **Fetching** downloads each file from that commit through the asset cache's one resolver, held to its sha256,
  decimates it to its share of the set's triangle budget (vertex clustering; every vertex moves at most a cell's
  diagonal), measures its distance to the original from every vertex of each to the other's surface, both ways (so
  that no denser sample of vertices finds more), and writes it as a USD layer in the asset cache, under `visuals/`,
  with a record of its source, parameters, checks and output, and its tier: a baked mesh is a derived file, and takes
  the strictest terms of its source, so the AR4's are referenced too. The record keeps both directions: a baked vertex lies
  within the clustering cell's diagonal of the original, but an original vertex need not lie within it of the bake,
  where a feature thinner than a cell collapsed and its triangles were dropped, so that direction is measured, not
  bounded.
- **The stage refers to a mesh by its manifest's name and its source's sha256**, never by a path into the cache, and
  holds none of its bytes. A saved stage's names are its pins: loaded, the experiment's stage names each mesh as it was
  saved, whatever manifest is beside its description now, and the build report says where one names another file. When it is opened through `mime visuals view` or `mime.viz.open_stage`, or by a client from the overlay
  `hello` names, an overlay in the cache references each baked mesh into its prim and hides the bones. Without it, the
  links are drawn as bones, and `mime visuals view` and `hello` say which sets are missing and how to fetch them (in
  `hello`, `visuals.fetch`: the command for each missing set, its folder quoted for a shell), with each set's notice
  (its credit and terms) whether it is drawn or not.
- **Nothing of it is ever written into a saved folder or a run folder**, so `mime pack` cannot carry it: a test packs
  A0's saved folder and a live session's run folder with a stand-in baked in the cache, and finds none of its bytes.
  `mime assets audit` finds a baked mesh copied anywhere (a tree, a wheel), by its hash, and `mime pack` and `mime push`
  refuse one copied into an experiment as its author's own file (`bundle.referenced-copy`, `push.referenced-copy`).

The triangle budget is a presentation setting with its basis (`mime.viz.TriangleBudget`): the earlier line's AR4
meshes at CAD density shimmered in the viewport, their triangles smaller than its pixels, and drew slowly on a
software renderer.

## A view in the browser

A run, a saved experiment or a stage can be shown as **one web page**: the scene and its recording as a glTF
binary inside the page, drawn by Google's `<model-viewer>` 4.3.1 with play and pause, a speed, a slider over the
recording, its simulated time, the scene's size, and the terms of every set of meshes it draws or leaves out. A page
needs no MIME, no Python and no USD viewer to open, so it is the form to send to someone or to put beside a report.

<!-- mime-fragment: commands for a run folder the reader has; the command lines are parsed against the CLI -->
```sh
mime visuals web a0_run                          # a0_run.html here: one file, the viewer inside it, no network needed
mime visuals web a0_run --out a0.html --open     # named, and opened in the default browser
mime visuals web a0_run --cdn --glb a0.glb       # the viewer from jsDelivr (a smaller page), and the GLB alone
```

<!-- mime-fragment: the notebook call, for a run folder the reader has -->
```python
import mime.viz

mime.viz.display("a0_run")                # the last line of a cell: the page, in the cell's output
mime.viz.display("a0_run", offline=True)  # the viewer inlined too, for a notebook read without a network
result = mime.viz.gltf("a0_run")          # the GLB itself (result.glb), with what it holds and leaves out
written.view()                            # after built.run_to(...): the same as display() of its folder
```

- **Only meshes that may travel.** A page is a copy that leaves the machine, so it holds a baked mesh only where its
  tier is `open` or `conditional` (`mime.viz.SHAREABLE_TIERS`; a conditional set's conditions are printed on the
  page). The AR4's meshes are `referenced`, and a mesh no record states a tier for is treated as one: the page draws
  their frames' simple shapes in their place and says how many it left out, as `mime visuals view` would draw them on
  this machine. Two guards hold it, each failing closed: the overlay the view composes leaves such meshes out, and a
  composed stage that still reaches one (a view layer written for a USD viewer, named as the target) is refused
  (`visuals.not-shareable`). `tests/viz/test_gltf_tiers.py` holds both, against a plant stated `open` that must be drawn.
- **Never into an experiment's folder.** A page is presentation, so it is written beside a saved or run folder,
  never inside one (`visuals.web-out`).
- **What the page shows.** glTF is +Y up and in metres, so the GLB's root node turns the stage's up axis to +Y and
  scales its units to a power of ten metres that makes the scene between one and ten units across; the page reads the scene's size
  back in its own unit. A recording shorter than four simulated seconds is slowed to about eight seconds of display
  (`--slow-motion` sets it), and the slider and the clock read simulated time. The camera frames the box of the whole
  motion, not the first pose. Spheres, cylinders, cubes and capsules are tessellated (`mime.viz._primitives.SEGMENTS` facets a turn); meshes
  are drawn as the stage holds them; translucent parts are blended as their `displayOpacity` says.
- **The viewer.** `mime visuals web` inlines the copy of `<model-viewer>` 4.3.1 shipped in the wheel
  (`mime/viz/_viewer/`, pinned by its sha256, with the licence of every part its bundle holds: the Apache License, and
  three.js, fflate and gainmap-js under MIT, lit under BSD), so the page draws from a file with no network.
  `mime.viz.display` leaves the viewer to jsDelivr by default, held to the same file's hash, because an inlined
  viewer adds about a megabyte and a half to every saved notebook; `offline=True` inlines it. `display` warns above a million bytes, where
  Jupyter's default output rate may drop the cell, and says how to make it smaller.
- **Where it does not draw.** GitHub's notebook preview runs no script, so there the cell is blank. That the inlined
  page draws with the network blocked is shown by hand in headless Chrome by `scripts/offline_view_proof.py` (WebGL
  through SwiftShader, every request sent to a closed port, the network log read; its control, the same page with the
  viewer from jsDelivr, must fail to load). It is not a CI test: CI has no WebGL. Colab, JupyterLab and VS Code were
  not tried.

### In Colab

The settling ball of `examples/sphere_settling_tube.py`, in a notebook. It fetches the tube's table (about two hundred
megabytes) once, then runs the experiment on Colab's CPU:

<!-- mime-fragment: a notebook cell that installs MIME and downloads files from the network, shown and not run by a check -->
```python
!pip install --pre mime-engine
!mime assets fetch unit_tube_R0.997
!curl -sSLO https://raw.githubusercontent.com/Microrobotics-Simulation-Framework/MIME/v0.3.0.dev1/examples/sphere_settling_tube.py

import mime.viz
import sphere_settling_tube as example

written = example.run("ball_run")
print(f"early speed {example.early_speed(written) / example.stokes_speed():.2f} of Stokes's free-space speed")
mime.viz.display("ball_run", caption=example.LIMIT)
```

**Its limit, as the example states it.** The drag includes the tube's wall, from its table, down to a gap of about
one sample spacing (2.5 um here). Closer than that, near-contact lubrication is not modelled in 0.3.0.dev1, so the
last approach is too fast, and the contact, not the fluid, holds the ball on the floor. Near-contact lubrication, and a
flat floor's wall (so a dish), are planned for 0.3.0.dev2. The tube's end caps are drawn, not computed: the physics is
the infinite tube's, and the example's docstring says why the caps' absence does not matter at their distance.
