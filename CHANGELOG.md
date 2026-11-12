# Changelog

All notable changes to MIME are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The moving refinement patch's kind** (the patch phase; not reachable from an experiment yet: the Schwarz coupling
  that hosts it wires it). A fixed fine box near a body by a wall (`RefinementPatch`, a grid kind with every face given,
  `NestedMACGrid`, whose fast solve is the cosine transform on three axes), its corner in whole fine cells snapped to
  the body (so a move never recompiles), its walls resampled from the same signed distance and dilated by whole fine
  cells with the two neighbouring dilations blended; it solves the cluster's near-wall reflection under the flow's step
  operator (`shift - mu L + mu/K`, the steady limit at zero shift) through the FVM's projected CG, its given values the
  coarse grid's reflection, its source the cluster's own field (the current step's in closed form, the regularised
  Stokeslet and the Brinkman difference; the history's from the flow's own box), read at the near field's points or
  over their cells; one column of forces a call, vmappable, linear without history, with its transpose
  (`patch_reflection`, `patch_reflection_transpose`); the backward-Euler form carries the last step's field as state.
  Its two transfers are registered mapping kinds, `mime.patch_boundary` and `mime.patch_read`, with MADDENING 0.4's
  workarounds in `mime.mappings`.

- **A static porous plug in the FVM: `Darcy()`, a porous structure's drag on the fluid in it** (lane F, dev2). A
  structure of the new geometry kind `Plug` (a solid circular cylinder placed in the world: `centre`, `axis`, `radius`,
  `length`; its region `solid`), its `permeability` declared (a structure's value, m^2, above zero; the units `m2`,
  `um2` and the dimension `AREA`; no OpenMaterica record holds one yet), and `exp.attach(Darcy(), medium=...,
  structure=...)`: the medium's flow holds it in its step as a resistivity `mu / K` added to the wall's (B5's seam,
  `wall_fields`), on IMEX Euler or its steady limit, on a periodic box (a straight tube along z or a tube tree) or an
  open vessel. Its resistivity in the wall's band is its share of the fluid over the face's fluid fraction
  (`FLUID_FLOOR`, 0.01, at least), carried into the wall's solid behind it, so a pressure-driven Darcy flux is carried by
  the band's fluid and the band's solid does not leak round a plug that closes the vessel (the plug's forms measured
  against each other: the PR's table). Its loss is booked under its id; its sampled permeability not a finite number
  above zero is flagged at sampling and refuses the run after it (`flow.permeability-not-positive`); each step reports
  its mean pressure on the planes a band's half width outside its two ends (`porous_face_pressure`). The build reports
  its `sqrt(K)` in cells where its Brinkman layer is under one, and that its flux is unmeasured where `sqrt(K)/h` is
  under the range measured: a plug closing a straight tube is resolved down to `sqrt(K)/h` 0.003 with the default 0.1h
  interior (within 0.4 % at D/h 32, a `K` of about 1e-14 m^2 in a 1 mm vessel) and 0.01 with 0.01h, unmeasured below
  (the first step's cold solve stops there, flagged); the floor, the wall's leak past a plug closing the tube (the
  intercept of its flux against `K`), is under 4e-11 of the open tube's flux at D/h 32. These are on `lumen_flux`, which
  with a porous region reports the box's whole axial flux, the plug's throughflow: past a closing plug in a box driven
  by a body force the open part carries a loop through the wall's band (2.5e-3 of the open flux at D/h 16 with 0.1h,
  K-independent), which the lumen's own mean would carry. A region outside what was measured (under that range, or
  leaving a gap to the wall under `sqrt(K)/h` 1, where a coaxial core's gap flux read 12 to 16 % low) raises
  `porous_flagged` with a warning. A periodic box reads a plug anywhere along the tube at its image in the period.
  Refused: a structure of another kind (`flow.porous-kind`), RK3 (`flow.porous-scheme`), a tilted box
  (`flow.porous-axis`), a plug longer than a periodic box's period (`flow.porous-period`) or outside an open vessel's
  span (`flow.porous-outside`). A tube tree holding a plug runs but is unmeasured. The stall detector's denominator
  takes the porous losses beside the viscous one.

- **A body resolved on the FVM's grid: `Drag.Penalty()`** (drag 0.5.0's penalty rung, F4, its first part). The body's
  solid is lifted onto the grid's faces by the wall's own band from its solid region's exact signed distance (a
  sphere's, a capsule's; another kind is refused, `drag.penalty-kind`) at its pose each step, on a window of faces
  about it, the solid the union of the wall's and the bodies'; its penalty drives its interior to its rigid twist in
  the field's own solve, and its load is the penalty's force and torque plus the interior fluid's momentum change
  taken from the field (the field formulation, B8 point 5). The coupling is lagged, a declared form: its loop is stable
  up to an edge in the step against the body's relaxation times (dt/tau_t and dt/tau_r), measured at rest against N,
  the scheme and the density ratio; the build refuses a step past it (`drag.penalty-lagged`, naming the largest step
  admitted) and a body lighter than the least ratio measured, and a run past it is stopped by its body's motion, more
  than a cell a step (`drag.penalty-motion`); the one-pass affine coupling is its next part. A body within three cells
  of the wall, and one whose density ratio lies outside the measured range, are flagged each step
  (`body_<i>_near_wall`, `body_<i>_ratio_unmeasured`), and the rung declares a Reynolds line for the post-run watch
  (150 at the default N, falling as N squared below it). The FVM's spacing with no spacing written is the finer of
  the lumen's and N cells across the body's smallest feature's radius (`feature_cells`, 6 by default); a grid under the
  thinnest-feature line is
  refused (`drag.penalty-thin`), as are a `feature_cells` a written spacing does not give (`drag.penalty-spacing`), RK3
  (`drag.penalty-scheme`), an open vessel (`flow.open-resolved`), a tube tree's flow (`flow.tree-resolved`) and an
  added mass delivered to a resolved body (`drag.penalty-added-mass`). The body takes the displaced share of the
  medium's drive (the drive acts on the fluid alone) under its drag's part `<id>.displaced`. The flow node holds the
  penalty in its step under the drag's id and judges its sign at one velocity (`body_<i>_gift_steps`, read onto the
  drag's rows); a law held in its receiver's step and judged in its sum is judged by its host. `ResolvedFlowNode`
  (MIME-NODE-131); the registration `flow.resolved` (kind `flow.resolved_partner`).
- **A study over a kind's vector setting, one build per value.** A kind's setting that is not one number (an
  ellipsoid's semi-axes, a box's size, a tube tree's radii, a direction, a count), refused as an axis until now
  (`study.held-geometry`), is an axis built per value (`Axis.built`): whole, each point the whole vector
  (`"egg.parts.shell.semi_axes"`), or one component by its index (`"vessel.radii[3]"`, `Axis.component`). Each point
  rebuilds the kind with the value written, and each value is a group of its own, split by it (the point's key is its
  structure key with the setting's value), the split reason naming the axis; its inertness is judged on the points'
  traced steps as well. No derivative is taken along one, a declared cut stated in the plan and refused where a study
  or a fit asks for one (`gradient.built-value`); an override of one is refused (`override.structural`), and so is a
  range (`range.units`). New refusals: `study.component-path` (a component of what is not a vector of numbers, or a
  second index, `vessel.radii[3][0]`), `study.component-spelling` (an index in another spelling than one whole number
  from 0, `vessel.radii[03]`, so that one component has one path), `study.component-index`, `study.vector-value`;
  `study.held-geometry` now refuses only a kind's structure, names or reference (a tree's segments); `study.axis-twice`
  judges what each axis writes, not its spelling. The study folder and the spec file hold a tuple of bare numbers (a
  direction, a body's initial orientation, refused at save until now) as the list of its numbers, read back as the
  tuple. Gated (`tests/study/test_vector_axes.py`): a study over the MRS egg's semi-axes (whole, and one component by
  the water's viscosity) and over a plane's normal (whole, and one component: two groups of one structure key, split
  by the value alone) equals the same points written by hand, bit for bit per point; the FVM tree's fourth radius
  equals them at its points (structure, leaves, initial state and traced step; no run).

- **A point body coupled two-way to the FVM: `Drag.Stokes()`** (drag 0.5.0's stokes rung, F3). A sphere's Stokes drag
  and couple on the flow read through the exchange law (a Gaussian kernel on the MAC grid's face sets, gather and
  scatter a transpose pair on MADDENING's geometry edges, the mapping kind `mime.gaussian_mac`, its modes the kernel's
  average, its Laplacian and its antisymmetric dipole), Faxén's corrections paired so the exchange stays workless, the
  kernel's own steady field removed from the drag's and the couple's readings, and Maxey and Riley's
  fluid-acceleration force and added mass read on the flow without the body: by default the body's own free-box flow
  advanced exactly by the grid's operator each step (`self_history="exact"`, with Hasimoto's image term), or fitted
  decay modes (`self_history="modes"`, the declared opt-in for many bodies, its gate a swap test against the exact
  route), and under the field's steady limit the steady response to last step's force. Lagged, two-way by default for
  the force and the couple, `one_way=True` the declared approximation. The build widens the kernel (`kernel_width=`, on
  a ladder at or above max(h, a/1.7)) until the coupled one-step margins of the force and the couple clear their floor,
  and refuses a body too large for the widest kernel (`drag.stokes-size`) or under the floor at every allowed width
  (`drag.stokes-floor`), each naming the coarser grid and the larger step that plan it; a sphere only
  (`drag.stokes-sphere`), in a flow on a grid stepped by IMEX Euler or its steady limit (`drag.stokes-scheme`), on a
  periodic box (an open vessel refuses its exchange partners, `flow.open-exchange`). The drag's sign is judged by its
  node each step (the law's power at one velocity, `cluster_gift_steps`, read onto both of its ledger rows), and its
  own dissipation booked net of the fluid's dissipation of the kernel's field, so the lagged exchange's unpaired work is
  first order. The FVM takes exchange partners (`flow.exchange_x`..`_z`, `flow.partners`), publishes its face sets
  (`flow.faces`) and books the exchange's work per law; the energy ledger books the Maxey-Riley part under its own row
  (`<id>.maxey_riley`). Point bodies in one medium interact through its flow, lagged a step; the build reports it on
  each one's drag, with their nearest starting separation in radii and their volume fraction in the grid's box.
  Flags `below_floor`, `history_flagged`, `rate_feedback`, `near_wall`, `reynolds_flagged` (stokes).
- **Two openings at one plane, laws over several openings (`surface=` a tuple or a region's openings), `Valve` and
  `SharedDistal`, the kinetic-energy flux split by plane.** The open flow's node takes any number of openings, each a
  patch of one of the box's two planes (a plane holding several is split between their centres, each of its faces, the
  solid's included, in the nearest opening's patch), and laws that bind one opening or several: a law over several
  states a block over them (`PortLaw.openings`, its rows' `B`), which the port system solves whole each step. Through
  the experiment a law's `surface=` names its openings as a tuple, `surface=(tree.end_0, tree.end_1)`, or as a region's
  declared openings, `surface=tube.lumen`, and a law given another count than it binds is refused
  (`binding.opening-count`); `FVM.openings_of` reads the geometry kind's declared openings, and the build lays each on
  the plane its surface faces; a law refused at its openings' binding is refused alone (the flow no longer adds that
  no law is attached at the openings it named), and the routes name every law an opening takes.
  `exp.attach(Valve(), medium=blood, surface=...)`: a closed valve, a flux row of zero, workless. It holds its
  opening's net flux at zero, not each face's: the lumen behind it carries out what the solid's faces on the opening
  bring back in, the solid's Darcy flow, falling as the square of the stiff interior's depth (on two parallel tubes,
  1.5 to 5.0 % of the inflow at the default stiff interior, 1.8e-4 to 7.0e-4 at `Band(delta_in=0.01)`).
  It is gated at the default interior against the walled-off vessel (that outlet capped by solid, whose cap and the
  solid around it leak as much), the inlet pressures within the two runs' solid flows carried through the open tube's
  resistance, which sees a valve error from about 8 % of the inflow there (from 1e-3 in float64 and 2e-3 in
  float32 at `Band(delta_in=0.01)`). The stiff resistance that would stand in for it differs
  from it only by its leak, exactly one over its resistance: the port system's condition number with its rows scaled
  is the valve's at every stiffness (the raw number mixes a flux row's units with a pressure row's).
  `exp.attach(SharedDistal(first=..., second=..., distal=...), medium=blood, surface=(a, b))`: two openings each
  through its own resistance into one shared distal resistance, its off-diagonals solved, its own books kept (a lagged
  off-diagonal leaves a first-order residual in them). `port_kinetic_flux`, an active flag, an operator check of
  convection at the planes: the kinetic energy the books hold as crossing the openings against each opening's own, the
  scheme's convective power split by plane (exact to the field's divergence: the divergence form is conservative), the
  planes' net judged on each opening's flux in magnitude, so its scale does not cancel; it sees no port work (the
  record's port work against the kinetic-energy flux stays not built), and its lines are about three times the clean
  worst, measured. `OpenFlowNode.opening_solid_fluxes`: each opening's flux through the solid, which for an `Inflow`
  (it pins the opening's whole net flux, never the lumen's alone) runs either way: with the inflow in steady flow, and
  in pulsed flow swinging with the drive, the lumen then carrying up to 1.32 times the inflow (the FVM's validity).
  Only a Tube's two ends build through the experiment until the tube tree's wall is in the open box; two outlets are
  gated at the node level on a rig of two parallel tubes, every run of it metered and its books judged (the ledger on
  in every gate).
- **The near-contact law, one seam for every rung, and a sphere's exact form (`mime.nodes.hydro.near_contact`).** A
  rung declares the gap it resolves down to and evaluates itself with the contact opened to it (`opening`); the law
  adds the increment to the true gap from the contact's geometry (`NearContact`: its point, frame, gap, and the gap's
  and the body's own principal curvatures) by the form the body's kind declares (`Geometry.near_contact`), which the
  plan picks and the build report names. **A sphere's form** (`Sphere.near_contact = "sphere"`) is exact: the plane's
  exact remainder from the new packaged table `sphere_plane_resistance` (MIME's, from the bispherical solutions:
  Brenner 1961, Jeffery 1915, Dean and O'Neill 1963, O'Neill 1964; built by `scripts/sphere_plane_table.py`, its
  sha256 pinned, its derivation record shipped; read as its singular terms plus a C2 spline in `ln(h/a)`, continued C1
  below its range), plus the contact's own elliptic singular terms with the body's curvature (Cox 1974, Claeys and
  Brady 1989): exact against a plane, the hybrid in a tube. Drag 0.5.0's mrs rung takes it for a sphere near a wall
  that opens at any spacing (the film's window is the film's); every other kind keeps the thin film's form, its
  numbers unchanged, MIME-ANO-154 open for it (now `partially_resolved`: closed for spheres near a plane and in a
  tube). In a tube the sphere's form is gated at sphere-to-tube ratios of 0.22 to 0.34, every entry of its 6 x 6
  against an independent tube MFS where the MFS resolves it; past 0.34 a step within the opening's band raises the new
  flag `handover_tube_ratio` and the build report says so (the rolling along the axis measured -18 % at 0.40 and -25 %
  at 0.46, MIME-ANO-154 open there). A sphere whose force points are graded towards its contact is refused the form
  (`drag.near-contact-graded`, its measured error named), and the build line states the form's measured error at
  uniform spacings. The table's file holds no times or versions (they go to its build's log), and its script pins the
  kernels its numbers are made in, so that a rebuild reproduces its sha256; a test rebuilds three knots and compares.
  The seam's roughness cutoff (`roughness_cutoff`) and the ambient's zero on the opened wall (`ambient_point`,
  `AMBIENT_ZERO`; no drag rung reads an ambient flow yet) are the same for every rung. The gates' oracles, the graded
  and the tube methods of fundamental solutions of the MIME-ANO-154 probes, are committed with their scripts
  (`tests/differential/graded_mfs.py`, `tube_mfs.py`).
- **Drag 0.5.0's near-contact setting and fields renamed, and its surrogate blend retired.** The mrs rung's hand-over
  is `Drag.MRS(near_contact=True | False)` (was `lubrication=`); its state fields are `contact_increment`,
  `contact_coupling` and `contact_cutoff` (were `film_*`; `film_residual`, `film_unsolved` and `film_annulus` stay the
  film's). `lubrication=` written on drag 0.5.0 is the surrogate's blend, which 0.5.0 retires: refused
  (`lubrication.retired`), its way out by the rung it was written on: on the mrs rung `near_contact=` at the value
  written (so `lubrication=False` keeps the hand-over off), on the surrogate drag 0.4.0, which keeps the blend. Drag
  0.5.0's surrogate rung has its own round-trip case again (script A unblended). Script A, A0, the inertia arms and the tests built on
  them pin drag 0.4.0 (`KEPT["drag", "0.4.0"]`), their programs bit for bit. Drag 0.5.0 is unreleased; no released
  program changes.

- **A flow open at the vessel's ends, and the laws at its openings: `VesselRegion`, `Inflow`, `Windkessel`, the
  `opening` port kind.** `FVM(domain=VesselRegion())` on a straight tube along z cuts the region the medium fills at the
  openings its geometry kind declares for it (`Geometry.openings`, read through `Region.openings`; named by geometry,
  never by role): the tube's own two ends, its new surfaces `tube.start` and `tube.end` along its axis, each an opening
  whose law is attached there:
  `exp.attach(Inflow(steady=mL_per_s(...)), medium=blood, surface=tube.start)`, a volume flux pinned at the opening's
  evolved face (a flux row), and `exp.attach(Windkessel(proximal=..., compliance=..., distal=...), medium=blood,
  surface=tube.end)`, the RCR stepped by backward Euler at the flow's instant (a pressure row). The grid is open
  along z with its planes' faces at half weight, so the projector stays orthogonal; the step holds each opening's law,
  coupled implicitly through the field's affine response (one unit field a step for the opening that is not the
  gauge's reference, made in the program and solved again from the last), its openings' rows solved together (A2's
  hybrid rows) for each relative pressure's move before the pressure level, so no move is the difference of two
  absolute pressures and the inflow's flux holds to its summation floor at any downstream resistance (the state keeps
  `opening_pressure_relative` beside `opening_pressure`), the build refusing rows with no solution
  (`flow.openings-ill-posed`), a law at no opening, an opening with none or two, a law on a periodic box, RK3 with
  openings and a tilted open vessel. An open vessel has no step rule yet: `dt='auto'` is refused saying so
  (`StepRule.unbounded`), and its CFL stop names a timestep from the inflow's peak. With the energy ledger the
  open flow is metered through `fluid` and the new `opening` port kind (a flow's openings' outward fluxes, a row per
  opening, against their pressures): each law's work under its name, the inflow a source supplying it, the kinetic
  energy convection carries through the openings as one line (`<node>.openings`, the scheme's own convective power
  `dt rho <C(u), u>`), `N_proj` booked (rounding at the half-weight faces) and flagged (`open_face_projection`); a
  law that stores energy of its own keeps its own books beside the flow's (`mime.core.ledger.law_records`,
  `LawRecords`: its storage at the step's start and end, its loss, its numerical term, its own identity judged on its
  own scale), and its loss is read net of what it gives back. `mime.testing.port_law_suite`, the port-law family
  suite: the block of the row a law gives the solver passive over draws, at the step and its limit `dt -> 0` (CI check
  (a)), its own books closing. The open grid's operator is gated at its planes against closed forms (its viscous and
  convective half cells). Not built, and no flag declared for them: the port work against the kinetic-energy flux
  through the openings (it needs the convective power split by plane), and the periodic box's flags on the open box and
  on RK3 (RK3 declares none). Units `Pa`, `m3_per_s`, `mL_per_s`, `Pa_s_per_m3`, `m3_per_Pa`. No released program changes.

- **The FVM on a tube tree's wall** (F2a's second part). A medium filling a `TubeTree`'s lumen
  (`Medium(fills=vessel.lumen)`) takes the FVM on a periodic box: the tree periodic along z with the box's period and
  reaching past its two ends (`flow.tree-period` refuses one that does not repeat where the band reads it), held in its
  box with its spacing from the narrowest lumen, stepped by IMEX Euler or its steady limit (RK3 on a tree is refused,
  `flow.tree-scheme`). The node takes the tree as its leaves (`tree_points`, `tree_radii`, `tree_fillet`, with
  `tree_segments` its structure) and makes its wall from them in the program: the union's exact distance (`ExactUnion`)
  wherever a field the step reads is not flat, its candidates run at the faces where its check fails within the band's
  fluid half, chosen in the program at the run's values (`ExactUnion.evaluate_chosen`), only how many may run held at
  build (a rung above the build's own count, as the cell counts are held), the plain minimum elsewhere; the wall is
  constant over a run, so a chunk makes it once (the node's run-constants `tree_wall`, `tree_wall_counts`), and a
  moved radius or period keeps one program. Its state reports where the distance is not exact (`wall_unrefined`,
  `wall_unconverged`, `wall_unplanned`, and the warning flag `wall_inexact`), and its build reports the wall's
  resolution entries below their lines (`wall_resolution`: the least radius of curvature, the narrowest lumen); a
  straight tube's state and programs are unchanged. Gated through the node, steady
  Stokes, every vessel turned about its period's axis at random and moved by a sub-cell offset: a helical tube's lumen
  not displaced against Dean's first-order profile, an elliptic section's semi-axes, a taper against lubrication, a 75 %
  stenosis and the split-and-merge loop's split at 30 degrees against Richardson references, the symmetric split a half,
  and a daughter's flux's derivative in its radius against a central difference within one discretisation.
- **`TubeTree` is the geometry kind `tube_tree`**: declared in quantities, `TubeTree(points=mm(...), segments=...,
  radii=mm(...), fillet=0.1, ends=(...))`, three lengths a point and two radii a segment, flat (`TubeTree.si` takes them
  in SI as nested tuples); its regions the `lumen` (the shape `tube_tree`) and the `wall`; its free ends named by
  geometry, never by a law's role: `root` at the first segment's start, the others `end_0`, `end_1`, ... in the order of
  their points or as `ends` names them (`TubeTree.end_names`), each a surface reached through the structure's handle
  (`vessel.root`) and an opening of the lumen (`Geometry.opening_names`, per instance beside the class's `openings`).
  Its junctions are named by order, `junction_0`, `junction_1`, ... in the order they are found walking outward from the
  first segment's start (`TubeTree.junction_names`), or as `junctions=(...)` names the first of them; `mime.scene.polyline`
  names its junctions `junction_0`, ... along the line.
  A tree whose fillet would fuse tubes that share no point (solid between them thinner than its diameter) is refused
  when planned (`setting.value`), naming a smaller fillet or a thicker wall. The round-trip case `tree_flow`.

- **Composed walls in `mime.scene`: the seam-aware union, the fillet as its closing, the tube tree** (F2a's first
  part). `ExactUnion` is the exact signed distance of a union of regions that each have one (`RoundCone`, `Ball`,
  or a lab's `Region`, by `mime.core.Shape`'s method names): the minimum, a check at each point of where it is exact (the deepest region's
  nearest wall point lies outside the regions that meet it), and Newton at a wall, a seam or a corner where it is not,
  at a plan's points held for a build. The plain minimum reads the fluid too shallow near every seam, a false wall of
  0.4 to 1.75 cells at a Y's carina; the rule is exact to a brute force's floor at random rotations and sub-cell
  offsets, and where two walls' normals are opposed (the plane of two parallel tubes' axes on an aligned grid), in both
  dtypes, and in float32 at a scanner's coordinates (its regions are evaluated about the union's own origin). Every
  value carries counts of where it is not exact (`UnionDistance`, `UnionNormal`: points outside the plan, points whose
  own seam was never reached, and pairs of regions the run's values bring within reach that the union does not hold as
  meeting). A fillet of radius `r` is the union's closing by a ball: the same rule on the regions widened by `r`, moved
  back, exact in the fluid and against closed forms (a wedge, two balls) on both sides, and in the solid except within
  `r` of a neck of the widened union thinner than `2 r` (tubes sharing no point whose wall is a little under `2 r`).
  Derivatives by implicit differentiation. `TubeTree`: points, segments with a radius at each end (round cones), its
  centreline with a junction where two segment ends or more meet, its lumen, one fillet (`FILLET_SHARE`, 0.1 of the
  smallest radius at a junction, declared: no measurement of vessels gives it yet), its values parameters a derivative
  flows through, and `turned` for tilted vessels; a malformed tree is refused (`call.tube_tree`). The walls' half of
  the resolution report (`wall_resolution`, `stokes_layer`, `steady_layer`): the narrowest solid between tubes sharing
  no point and the solid the fillet fills between them, and the wall layer against lines set from a Womersley sweep's
  worst arm (wall shear within 5 % from 2.7 cells of layer, within 2 % from 4.5). No node runs it yet: the FVM's wall
  is the first.
- **The first fit: `Study.fit(residual, *, start, noise, duration, iterations=20, bounds=None, tolerance=None,
  acknowledged=None)`** (lane S's third PR, its second half), returning `mime.results.FitResult` (each `FitIterate`
  with its `LineMargin`s, the Fisher information at the start). Damped Gauss-Newton steps on the gradient path's
  derivative under the fit-loop rule: each iterate planned again through `build_point` and run on the start's
  compiled program, every validity line judged there and given to the steps with its margin, the iterates clipped to
  each axis's declared range and its leaves' bounds, MADDENING's `fim` with the declared noise (required) at the start
  (every insensitive axis named before the first step). On the gently pulled sphere the radius fits to a target inside
  its range where the spike's rebuilt descent ended 0.79 % short; across a Reynolds line the fit ends on the line,
  naming it (a fresh build at the radius it ended at reads the number within 1e-4 of the limit), where `sysid.fit` on
  the built graph runs through it silently. The default `tolerance` is 1e-9 of the values, or ten times the run's machine epsilon where that is higher (float32:
  1.2e-6); a fit whose loss does not fall in three iterates in a row, short of any line, ends `stagnated`. A line that
  cannot be judged (no density declared) is never taken as
  inside: the start is refused (`fit.line-unjudged`) and a later iterate ends the fit `unjudged`; a range that pins
  the step ends it `at a line` (`box:<axis>`); an iterate that needs another compiled program refuses the fit
  (`fit.value-keyed`) with the iterates taken in the error's `partial`. **Controller settings as leaves**: a setting a
  study's axis names (`controller.<setting>`) is a leaf of the controller's node inside the compiled step where its
  step reads it (`controller.setting-unread` for one read through `self`; `controller.setting-not-leaf` for a count, a
  switch, a name, a direction or a file, which the compiled step reads as built); `ArmTracking` reads its smoothing,
  gain and damping through its context, and an integer written for a float setting (`gain=2`) is that float.
  MIME-ANO-146 is partly resolved: for the settings the step reads; open for a start-only one (`home`) in the graph
  mode, where the two-builds check says `quiet` of runs that differ (the callback mode stays outside `params` by
  design). Nine refusal codes (`fit.*`, `gradient.objective`, `controller.setting-*`).

- **Lubrication's hand-over on the MRS rung, body to wall: `Drag.MRS(lubrication=...)`** (drag 0.5.0, a numerical
  setting, on by default near a wall that opens, a plane or a tube, about a body whose hull kind has a support and
  whose least radius of curvature holds the hand-over gap within the film's window). The near-wall resistance is the
  coarse MRS evaluated once with the body's near contact opened to the hand-over gap (`HANDOVER_SPACINGS`, 0.75, of the
  force spacing; the finest graded spacing where graded), never closer, plus the thin film's increment between the
  contact's gap and the opened one: Reynolds' equation on the gap's second-order form (the body's support curvature
  less the wall's), solved by exact block-tridiagonal elimination (`mime.nodes.hydro._film`). It is the MRS's own
  completion in the gap, not physics beyond Stokes flow; `lubrication=False` keeps the coarse MRS alone. **The film's
  window is the build's rule:** where the hand-over gap lies past `WINDOW` (0.15) of the body's least radius of
  curvature over its whole surface (a crest or an edge sharper than the spacing resolves, the de Jongh screw at its
  family's spacing, a sphere at a spacing of 0.3 radii), the default is False, the coarse MRS carries every gap and the
  build report gives the gap against the radius; `lubrication=True` written there is refused
  (`drag.lubrication-window`). The contact is opened by moving the wall (`WallCorrection.opened`: the plane
  translated, the tube dilated), smoothly (a C2 lift, the identity beyond the hand-over band, so a body farther off is
  the coarse MRS's, step for step, at the cost of the contact's search). **Each surface's roughness** is a declared
  length (`<body>.roughness`, `<structure>.roughness`, 1 um where none is declared, never fitted): below the pair's
  composite roughness (`mime.interactions.roughness_cutoff`, the root of the sum of squares) the film's gap is floored,
  so its resistance stops growing and a contact carries the gap; the build report says where that floor reaches the
  lift's band. The near-wall resistance owns every hydrodynamic coupling of rotation and translation down to that
  cutoff, and the node reports the film's own share (`film_coupling`) and increment (`film_increment`). With it, a
  compact body's default blob near a wall is 0.05 and the default force spacing no longer follows a contact's margin,
  and a body held on the wall (a margin of 0) is no longer refused (`drag.mrs-on-wall`). Flags: `handover_window`,
  `film_annulus` (a body nearly filling a lumen, whose ring contact the point film does not cover),
  `handover_far_side` (a tube opened by dilation about a body filling more than half its diameter) and
  `film_unsolved` (warned of). Known: the coupling of rotation and translation is over-counted against its own size
  below the hand-over gap (MIME-ANO-154). Not yet: several contacts of one body, a ring's film, bodies sharing a medium
  (refused with the hand-over on, `drag.lubrication-near-field`: write `lubrication=False` there) and a body between
  two walls; the surrogate's blend (`Lubrication(epsilon=...)`) is unchanged, and on drag 0.5.0 an override of
  `lubrication` is a variant (a numerical setting), where 0.4.0's is structural. Released programs (drag 0.3.0 and
  0.4.0) are unchanged.
- **A study's derivative: `Study.gradient(of, wrt=None, mode="auto", acknowledged=None)`** (lane S's third PR, the
  gradient path), returning `mime.results.GradientResult` (each member's `MemberGradient`: the objective, its
  derivative in each axis, the flags its run raised, its nodes' stops, its plan's extrapolated table reads, each
  iterative solve's `SolveReport` forward and in the derivative solves, the warnings, and a `Recheck` where a solve
  stopped short: the member re-solved with the cause removed, the cap lifted and the stall restarts raised). One mechanism: the sweep's points, planned and grouped as a run's, each
  group differentiated together by `mime.build._maddening.differentiated_run` beside the batched scan (forward runs'
  programs untouched), forward mode for a few numbers and reverse mode with a `jax.checkpoint` a chunk. The
  derivative is read through the leaves and initial fields each axis writes, found by probe points, each by its
  factor; what it cannot be read through is refused with its route (twelve `gradient.*` codes). The path's status
  composed from its entries (`mime.gradients.composed`), a `cut` or `unverified` entry refused unless acknowledged
  (`unacknowledged`): a controller called on the host is `cut`, its writes held as the forward run made them. The
  guards' flag form in a differentiated run (`mime.core.differentiated`, marked as `run_constants_given` is): the MRS
  drag's indefinite resistance gives the body the resistance of the step before, with the same flag and stop, so the
  derivative is finite and the result names the stop. `ReadOutcome.extrapolated` marks a read extrapolated by the
  experiment's declaration.
- **The energy ledger as a gate's tool, in `mime.testing`.** `ledger_closes(books)`, one check that a metered run's
  books close at every step (a receiver's records or a run's result: the step identity within its floor, attribution,
  no gift or indefinite step, route 1's every step dissipated, no unowned jump between steps, the balance), which
  `step_identity`, SF-12's ledger case and the flow's metered gates now judge with; **ratio flags** a node declares
  (`MimeNode.LEDGER_FLAGS`, `mime.core.ledger.LedgerFlag` with `FlagEvidence`), each a ratio of a metered step's terms
  with its line measured on drawn correct runs and each planted bug's measured detection rate, read by `run_flags` and
  `judge_flags`, a flag that cannot be gated yet declared pending and never counted (`flag_problems`); and the order
  check `term_orders`, each numerical term against the order its scheme declares (`mime.core.ledger.TermOrder`, a
  scheme's `ledger_orders`): dt halved and the tolerance a hundredfold looser, apart, the order its fall shows rounding
  to the declared one, read from each term summed over a drawn ensemble (on one drawn case a term whose leading
  coefficient cancels can show an order its scheme does not have). The flow declares its first: the stall
  (`|N_Krylov| / D_visc`), convection doing work beyond its time level's (`N_conv` against `-dt rho <C(u), u' - u>`:
  a stale level, the wrong field, the advective form; not an error in the operator that keeps it skew, scaled or
  negated, which the manufactured-solution gate sees), the drive reaching into a solid (the wall's loss inside the solid over its whole), and the openings' two pending; and
  its terms' orders (`N_BE` and `N_conv` second in dt, `N_Krylov` at least first in the tolerance). Test-side: no
  program changes.

- **A medium's flow metered by the energy ledger: the `fluid` port kind, and the rule for a source.** With
  `exp.observe(EnergyLedger())` the build meters an FVM flow as a receiver of the new `fluid` port kind: its fluid's
  momentum (`flow.momentum`, kg m/s, over the fluid alone, as the drive acts on it alone) against the body force each
  law driving it delivers (`flow.body_force`, a load; the drive's waveform a mapped load, which RK3 evaluates at its
  stage instants). The flow's step pairs each law's drive with the fluid's momentum at the instant it applied it; the
  viscosity and the wall's band are held in the step and booked under `<medium>.flow.viscosity` and the wall's own
  region (`tube.wall`); IMEX Euler and its steady limit close the step identity to rounding, and RK3 judges each step
  to have dissipated and checks its drive's attribution (the work its stages applied against the law's own delivery
  at the stage instants) to the same floor. The ledger's boundary gains a rule: **a law that declares no class, reads
  nothing of the components (every read it declares, the points a field is read at among them) and delivers only
  loads its receivers pair is outside**, and what it delivers is supplied work under its own name (a `BodyForce`
  driving a metered flow); a law that declares a class keeps it whatever its node reads (gravity reads nothing without
  a couple, and stays a potential). A source that states no reason for declaring no class is still logged under
  exploratory and refused under research (`ledger.unmetered`). The flow's reason for being unmetered is gone, and
  `mime.testing`'s port cases gain the fluid's (`fluid_rig`, `fluid_states`).

- **The coil array returns: `mime.fields.CalibratedMultipole`** (`calibrated_multipole` 0.1.0, its node
  `CalibratedMultipoleNode`), an electromagnet array whose coils are a calibrated multipole model of order one, from one of three packaged calibrations (ETH MSRL's published calibrations of Navion_2, OctoMag and CardioMag, evaluated by MIME; not affiliated with or endorsed by the systems' makers), `asset("navion_2")`, `asset("octomag")` and `asset("cardiomag")`: Apache-2.0,
  Multi-Scale Robotics Lab, ETH Zurich, byte for byte their upstreams' at pinned commits, each with its card
  (`docs/assets/`). On `mime.fields.CoilArray`, the contract every kind of coil array keeps: the command `currents`, one
  per coil, reaches the node as `delivered_currents`; the array is placed by `at` and `orientation`, one exact
  transform; `background` is a uniform field in the world frame, zero unless declared (the calibrations are fitted to
  the coil field alone; Earth's 50 uT is 0.1 to 1 % of a 5 to 50 mT field and has no gradient). The sources' positions,
  directions and coefficients, the pose and the background are the node's leaves. It publishes the flux density and
  its gradient at the reader's point or its magnet's centroid, so dev1's magnetic pair reads it; it raises
  `outside_region` (outside the calibration's declared region, which is not a fit domain) and `near_source` (a coil's
  gradient per unit current above 30 times its median over the region, which is a fitted source's own term;
  `source_clearance` replaces the rule with a distance), and refuses float32 where the sources' cancellation puts its rounding above
  1e-4 of the field (OctoMag; `setting.value`). `provider.actuation(point)` gives the field and gradient per unit
  current at a point (`Actuation`, its `matrix()` the 8 x K actuation matrix), and `allocate(actuation, field=,
  gradient=, limits=)` bounded least-squares currents within limits the user declares (`allocation.wanted` refuses a
  wanted gradient that is not symmetric and trace-free, and limits that are not above zero). The units `T_per_m` and
  `mT_per_m`, of the dimension `FIELD_GRADIENT`. Held to ETH's `mag_manip` 3.0.0 per system at 10 eps over the field's
  cancellation (its archive's generator in `tests/oracles/`); its sign convention is mag_manip's, which no hardware
  grounds here (MIME-ANO-153). Not returned from the 0.2 line's `CoilArrayFieldNode`: the sign setting (a validity line
  and MIME-ANO-153 instead), S-MPEM saturation (no shipped calibration fits it; it makes the field nonlinear in
  current), orders above one. CardioMag's file carries an `Offset` block of eight current-independent sources (about 1.5 mT at the
  median over its region) that neither mag_manip nor this evaluates, stated on its card and in the validity lines; a
  `Coil_List` that names the block is refused. No system ships a current limit: the figures published for each are cited
  in its card and the validity lines (Navion_2 25 A an operating restriction and 45 A the hardware's, CardioMag 35 A a coil with its 15 kW cap not modelled, OctoMag 16 A), MIME's declarations with their sources and no maker's specification, and the previous line's unsourced 35 A and 20 A are dropped.

- **Graded sampling on the MRS rung** (`drag` 0.5.0, design #296; `mime.scene._grading`).
  `Drag.MRS(sampling=ForceSampling(grading="features"))` grades the force points towards the hull kind's declared
  features, and `ForceSampling(contact=(0, 0, -1), contact_gap=...)` towards a contact held facing the wall (opt-in, for
  a body whose orientation to the wall is held; `contact_gap` defaults to the experiment's closest approach where a
  contact law declares one). One policy turns them into a local
  spacing: an edge a third of the force spacing, a crest 0.3 of its radius across, a contact its gap over 1.5, each
  growing by the distance from it, never finer than a quarter of the force spacing; the count is the uniform one times
  the density's mean. **Each graded force point sits at its cell's area centre** (the dense point of its cell nearest
  it), the cells the relaxed generators', which the separated quadrature, the reference and the wall quadratures take
  (`Sampling.generators`): with the points where the relaxation leaves them, grading raised the error near a wall (the
  measurements behind design #296). Lloyd's mass on a graded density is the area times the density's square, so the
  samples' density is the density. **Geometry kinds declare features** (`Geometry.features`, a new optional capability;
  `mime.registry.SurfaceFeature`): the cylinder its rims and the box its twelve edges, each with its distance in closed form
  (so a sampling graded on them keeps the body's symmetries and is chosen alike under a setting's move far below the
  spacing, which a distance read from traced points ripples enough to break), the screw its rims and crests, read from
  their traces (each crest's radius across it from the thread's height and turn, within 0.1 % of the surface's own
  principal curvature on FL-9). **A graded sampling is chosen at its settings rounded to four figures**
  (`mime.scene._grading.FIGURES`: the body's numbers, the spacing, the contact's gap and direction) and moved with the
  body's own shape: a graded choice is discrete and a graded density no scale of the body's, so a relative move of a
  millionth tipped near-ties without it (measured: FL-9's features at 0.4 mm, twenty of 2227 force points chosen
  afresh under its length; a helix's contact, eight of 770 under its filament's radius). **Over a declared range** a
  graded body's count is the graded sampling's own (`graded_force_points`), chosen by `GRADED_FORCE_POINTS`, sampled
  in every ranged number: a larger body's patch is a smaller share of its surface, so its graded count can fall as it
  grows (the 1 mm sphere's contact at 0.3 mm: 106 force points at a radius of 0.5 mm, 104 at 0.525 mm). The sampling is
  fixed on the body: **`off_patch`** flags a step whose body lies within one force spacing of the wall with any point
  of its quadrature within `OFF_PATCH_BAND` (0.05) force spacings of its least distance outside the graded patch,
  judged at each quadrature point (a body turned past it, a face turned towards the wall included), raised only where
  grading or contact is written (`Flag.written`, new). In graded runs the wall's self-image measures each cell's
  distance from the wall in the force spacing, as a uniform run's (design #329, superseding #308's cell's own size;
  `GRADED_WALL_MEASURE` in `mime.nodes.hydro.mrs._node`, one switch: at the gate's sphere +3.33 % against +5.24 % at blob
  0.1, -0.04 % against +1.67 % at 0.025) and a written `wall_refinement` is taken from the finest spacing; the wall's cell
  budget follows the graded density (decision #353: every cell of the graded patch and `WALL_CELL_SHARE` of the rest),
  so a default graded run raises no `wall_cells_short`, which is kept for a real shortfall. The build reports
  what was graded, the counts and the finest spacing. **What grading buys, measured**, against a uniform sampling of
  the graded count as well where that arm was run (the count-matched uniform arm, design #329's standard control, which
  the sphere's oracle runs; the cylinder's rim's figures have none, so its gain may be the count's): the sphere's contact
  grading
  helps beyond its count (at half a force spacing from a plane, +5.31 % against +10.16 % for a uniform sampling of as
  many force points, blob 0.1; +1.74 % against +6.00 % at 0.025); FL-9's crest grading at 0.4 mm does not (+2.23 %
  graded against +2.37 % uniform at the graded count, half a spacing, blob 0.25): its gain over the uniform sampling's
  957 points is the count. Refused: features of a kind with none (`drag.grading-features`), a kept version
  (`drag.grading-kind`), a contact with no wall (`drag.grading-contact`), no gap to grade for
  (`drag.grading-approach`) or a gap of 1.5 force spacings or more, which grades nothing (`drag.grading-gap`), and
  graded bodies in a medium's joint near field (`drag.grading-near-field`, before the medium's other near-field
  refusals: its cross-kernel is measured on uniform samplings only, and with a wall it raises no `off_patch`). The
  uniform sampling is untouched: every
  released program's steps trace the same programs (`tests/saved/step_jaxprs.py`, base against head, float32 and
  float64). The wall's self-image and the near field's cross-kernel share one private switch
  (`mime.nodes.hydro._near_singular`: the C1 smoothstep, the nearest-first budget and its shortfall, the cond that skips
  the near work, the kernel integrated over cells), each caller keeping its kernel, distance, reach, budget and
  granularity.

- **A medium's near field: its mrs bodies solved together** (`drag` 0.5.0; `JointDragNode`, MIME-NODE-127;
  `mime.nodes.hydro.near_field`). Two bodies or more in one medium on `Drag.MRS(...)` are built as one joint node,
  `near_field:<medium>`, in place of their own: each step it solves their joint system by one dense LU for the
  cluster's grand resistance (each body's own block dev1's; each pair coupled by the source's regularised Stokeslet,
  integrated over every cell where the pair is within a reach of each other, in the source's cells' own size, and taken
  at its force points beyond; the wall's image between them) and delivers each body its own coupled resistance and, as
  its background load, the others' load on it a step behind. The lagged step's gain, the spectral radius of
  `(M/dt + R_d)^-1 R_off` (the per-step fixed point's contraction), is checked every step from each body's mass matrix,
  which the rigid body now publishes (`body.mass_matrix`) only where the near field reads it: at one or above the run
  stops (`drag.near-field-gain`), and the build refuses a configuration that starts there. The check is sufficient for
  the lagged step's stability, not necessary (it refuses three spheres a tenth of a radius apart at tau/dt about 0.05 to
  0.11, which are stable); the step's amplification `rho((M/dt + R_d)^-1 (M/dt - R_off))` and its relaxation in steps
  are reported. The build refuses a near field whose bodies step at another rate than the base step
  (`drag.near-field-rates`: bodies at 1 s against a node at 1 ms passed a gain of 0.877 while their own step amplified
  by 1.0146). Each body's coupled resistance and the cluster's are judged definite as dev1's is near a wall
  (`drag.near-field-indefinite`), their margins reported. **Gaps between bodies** are estimated each step from the
  force points and their normals: two bodies that overlap by more than a quarter of a cell stop the run, and the build
  refuses bodies that start so, coincident ones with the overlap's message (`drag.near-field-overlap`); a pair closer
  than a twentieth of the smaller body's equivalent radius raises `near_contact`, the validity line's edge. **The
  cross-kernel's budget** is every quadrature point of a source by default, by measurement: a quarter of them held the
  two-sphere and gain gates at MIME's default sphere but left five hull kinds of six past their conformance line near
  contact (the box 12 % at a third of a cell); `Drag.MRS(near_field=NearField(share=...))` lowers it as a cost setting, a cell
  within the reach that does not fit raising `cross_cells_short`. A body alone in its medium keeps its own node, so no
  program with one mrs body a medium moves; a folder saved with `drag` 0.4.0, kept as `UncoupledDrag`, rebuilds its
  bodies' own nodes. The build refuses a medium mixing the two versions (`drag.near-field-mixed`). With the energy
  ledger attached, each body's own
  resistance is its drag's law and the cross loads are the near field's own, `near_field:<medium>`, booked on each body
  by the one pairing rule and never judged on one body's row, where they have no sign (one body drags another): the
  node judges the cluster's books a step late, from what it delivered and the twists the bodies reached, as the
  ledger's route 1 judges a step (`mime.core.ledger.CHECK_CLUSTER`, the gift steps read into the law's lines). That
  check sees an error that makes the fluid a source (cross loads delivered three times over, caught to a radius's gap
  along the line of centres, a third of one across it) and none that leaves it dissipating (the same at wider gaps,
  half their value, their sign flipped, an error in the grand resistance that leaves it definite), which the force
  oracles hold. The near field's contract (`BodyOperator`: the points where a body reads and
  delivers, its own block, its cross-kernel, its normals) takes the MRS as its first body model; its joint system is a
  pure function of the bodies' poses, their leaves and the wall's parameters, and a body model of another kind in the
  same medium (a slender body's, a resistive-force body's) needs the joint node to hold a member kind per index.

- **The near field's GMRES route, declared and never silent** (`drag` 0.5.0, unreleased; `JointDragNode`). A cluster
  whose drags declare it, `Drag.MRS(near_field=NearField(solve="gmres"))`, is solved by GMRES on its grand resistance's
  right-hand sides at once, each body's held free part's inverse (a new leaf, made once per build in its frame, made
  again with the free part where a leaf moves) its preconditioner block, through `custom_linear_solve`: each solve, the
  primal's and each derivative's, from a cold start, stopping on its true residual relative to its own right-hand side,
  the transpose solve preconditioned by the transposed blocks. `NearField(tolerance=...)` sets the tolerance (1e-10 in
  float64, 1e-5 in float32, above the floor its arithmetic leaves a pair's residual at down to the near-contact line), a
  numerical setting; the build report says which route a cluster took and how its starting solve ended. **The dense
  route is taken at every size unless GMRES is declared** (design #330, superseding the takeover line of 16,000 unknowns
  first built): neither memory nor a measured time crossover places a line, and the memory refusal is the only bound.
  Through the node GMRES's run peaks above the dense route's
  (1.11 to 1.35 times at 3,042 to 21,168 unknowns, either dtype: the bare solve's halving does not carry). In time the
  bare solve on a quiet CPU in float64 had the dense LU faster or tied at 3,042, 4,056 and 8,112 unknowns; through the
  node on a shared laptop (one run a case) GMRES's step was the shorter at every size a radius apart (1.3 to 4.1 times)
  and the dense route's at four sizes of five a twentieth of a radius apart in float64. The solve's report is the FVM's:
  each step's a state field (`solve_report`, its running sum beside it) with a flag the run warns of with its count
  (`solve_flagged`: at the cap, stagnated, or a residual not finite; zero on the dense route), the tangent solves' as
  those fields' tangents, and the transpose solves' as the gradient of the node's zero `solve_report` parameter (a row a
  step with the node's `solve_report_rows`). **The dense route's size refusal, `drag.near-field-size`, becomes a memory
  refusal on both routes** (`drag.near-field-memory`): the run's peak, estimated from counts of the joint matrix and of
  the members' held blocks measured through the node, against the memory a run can have (the FVM's: a GPU's limit, or on a
  CPU the smaller of the available memory and the control group's limit), so that no cluster that cannot fit runs, on
  either route; **and a gradient study's derivative** (`JointDragNode.derivative_problem`, before the study compiles it,
  `mime.study._gradient.derivative_refusals`): reverse mode, forward mode by its axes and a batch by its points, counted
  through the node (`DERIVATIVE_COUNTS`), each refused where `MEMORY_MARGIN` times the estimate passes the memory less
  `MEMORY_HEADROOM` of it. **A gradient study rechecks it** (design #337): the joint node declares `rechecks`, and a member whose
  solve stopped short is re-solved with GMRES's cap lifted and its stall restarts raised (`Stopping.stall_restarts`, new
  for GMRES and none in a run: a stagnation begins the window again), by the FVM's CG's rule, and a stagnation past them
  stopping a solve only at its rounding floor (`Stopping.floor_stop`: the least true residual within `FLOOR_MARGIN` of
  `eps || |A| |x| + |b| ||`; above it the solve runs on to the lifted cap). The recheck names each node's cause from its
  solves and the member's (`Recheck.causes`): `floor-limited` only where the recheck stopped at its floor and the node's
  solve tests it (`MimeNode.floor_stops`), `stagnated` for a stall above it (or a stagnation the FVM's CG does not test),
  `cap-limited` and `not-finite`, each with its node's route out by the run's dtype (`Recheck.routes`,
  `MimeNode.recheck_route`); `NO_RECHECK_REASON` no longer says a solve reads no cap.

- **The FVM's declaration at the build, and the quantity rule.** The build report shows the flow's box (its cells,
  spacing and resolution) and the run's memory estimate, the cells times the bytes a cell a CPU run was measured to
  hold at its peak, against the memory a run can have (a GPU's limit; on a CPU the smaller of the host's available
  memory and its control group's limit), and a box whose run will not fit is refused when planned (`flow.memory`, with
  a coarser grid or a shorter period as its route). A dynamics a medium carries is a field kind: its solved-field
  declarations (`Dynamics.field_declarations()`) state the quantities it provides, the FVM and the prescribed flow
  `field.flow.velocity`, and one that declares none, or provides nothing, is refused at registration.
  `PlanContext.fields_providing` and `field_for` find a field by quantity, refusing none and two unless the law names
  one (`binding.field-quantity` and `binding.field-ambiguous`): the surface a later rung's law plans through, since no
  law reads a field by quantity yet. Every drag rung (surrogate, mrs, rft, on drag 0.5.0, 0.4.0 and 0.3.0, and each
  member of a medium's near field) assumes its medium at rest, so it is refused wherever the region its medium fills
  carries a field, its own medium's or another's, whatever the field declares it provides (`drag.medium-flows`, naming
  the field), until a rung reads a provided flow.
  With the energy ledger attached, a run's account names what the build leaves unmetered, a medium's flow among them,
  with its reason (`result.energy.unmetered`, in `result.energy.text()` and `result.text()`), where it was `None`.
  Every dot in the FVM's programs is checked to state the highest precision (a CPU test of the traced programs).
- **Declared ranges** (`exp.declare_range(path, low, high, reason=...)`, saved form 12): the range of a value the
  experiment writes, on any axis a study names (a thing's value, a kind's number, a command's constant value, a law's
  setting, a body's initial value); refused on `dt`, the precision, a rung or a numerical setting, on a value that is not
  one quantity, where the experiment's own value lies outside it, and past six ranges. Its reason, pending or not
  given, is judged by the profile as a declared value's is (refused in research, `range.reason-pending`; stamped in
  exploratory). Every choice an entry makes from
  the value is made over the range (`mime.registry.choose_over`, with the rule the entry declares: `ChoiceRule.Monotone`,
  judged at the range's corners, or `ChoiceRule.Sampled`, inside as well; neither is the entry's defect), so a sweep across it is one group
  and one compile, and a run's override inside it runs as data. The build prices what the range moved in its report
  and plans the experiment at every corner, so each validity line is judged over the range (`range.corner`). A choice
  is made in every one of those plans at each of its rule's points, and past 1024 evaluations in all (points times
  plans) it is refused before its first (`range.cost`, naming the numbers its rule samples); the MRS rung's cost per
  evaluation is in the guide. Only a body's hull's numbers reach an entry as a range: the price report says "no
  consumer reads this range" for any other. A study's point outside a range is built for its own value, a group of its
  own; a run's override outside one is refused (`override.outside-range`); members inside a range that still split,
  against the first group inside it, are an entry's defect where an entry is offered the range and the build's
  otherwise, reported (refused in strict, `study.range-split`). Before its first sweep a ranged study checks, in
  float64 and once for the experiment's content, that the range reaches the build only through the choices entries
  report (the independence check, at the experiment's own value and the ranges' two extreme corners, all low and all
  high: not at the mixed corners) and that a move of 1e-6 inside each range moves no constant of the step or of a
  sub-program nested in it (the two-builds trace form, `mime.build._in_graph.nested_program`, beside a leaf the step
  reads or not; `held` where the move writes only a leaf the step does not read, `not-discriminating` where it changes
  nothing the build holds, each named in the plan). New refusals: `range.corner`, `range.cost`, `range.duplicate`,
  `range.kind`, `range.order`, `range.outside`, `range.path`, `range.reason-pending`, `range.too-many`, `range.units`,
  `override.outside-range`, `study.held-geometry`, `study.range-split`, and `study.range-bake` and
  `study.range-dependent` (MIME's defect).
- **Geometry axes**: a number of a body's part's kind (`"ball.parts.shell.radius"`) or a structure's
  (`"tube.inner_radius"`, a plane's `"floor.offset"`) is an axis of a study, an override and a range; a kind's setting
  that is not one number is refused as one (`study.held-geometry`). A structure's number reaches the drag node's leaf
  as a build at that value does (the plane's `wall_offset`: the resistance with the override equals a fresh build's,
  entry for entry), each point judged by the build's validity lines; until now it was refused (`value.unknown`).
- **`Drag.MRS` over a range** (its first consumer, no version change): over a declared range of its body's numbers the
  rung samples at the range's point with the most force points, its default spacing the family's at the range's
  finest, and keeps the body's own numbers as its leaves, so the chosen samples move with each member's shape. The
  force points are monotone in the sphere's radius and the capsule's radius and height (measured from half to twice
  the ball's and the pill's own, at 0.3 and 0.6 mm) and the screw's length, wavenumber and modulation (0.7 to 1.6
  times FL-9's, at 1.0 and 1.5 mm), each scanned in CI, and sampled at nine points elsewhere: the screw's radius is
  not monotone (106 at 2.153 mm, 91 at 2.184 mm, at 1.5 mm). The default spacing is sampled at nine points in every
  number, since it falls as the body grows: over the screw's modulation (0.98 % finer inside a window than at its
  ends), and wherever its rounding to three figures steps under a power of ten (the ball's 0.999 mm at a radius of
  3.976 mm, 0.99 mm at 3.9785 mm: the step is a unit of the rounded value's third figure, ten of the value below it,
  which the default of an experiment with no range has as well; unchanged here). On the pulled sphere over [1.0, 1.15] mm at 0.6 mm: 91 force points at every member, against 66 at 1.0 mm
  alone. An experiment with no range builds as before, bit for bit.

- **The two-builds check** (`mime.testing.two_builds`, `two_builds_of`, `step_program`, `graph_program`, `run_form`;
  float64 only): whether a compiled program holds a value made from a declared one as a constant of its own, which a
  study's batched members would all run at the group's first value and a derivative would not see. Its trace form
  compares two builds' steps, traced as MIME's chunked run executes them with the state, the inputs and the whole
  parameter tree as arguments; its run form writes one build's leaves into the other's compiled chunk and judges each
  field against `RUN_FACTOR` times the program's own resolution there (a start one ulp off, as a saved folder's
  resolution is measured), and sees a value that reaches the run outside the step. `RUN_FACTOR` is declared, 16,
  provisional: measured in a window whose margins are 1.1 and 1.2, so the run form's gating plants stand four times
  its line up. The trace form compares the constants of every nested sub-program as well as the top level's (a node's
  step runs inside a `jit`, where an array it holds as built is a constant that neither the printed program nor the
  top-level constants carry): a NumPy or JAX array of half the magnetic torque's moment taken from the build is a `bake`
  at every move from 1e-1 to 1e-15, in the step's `jit`, a `scan` and a `cond`, its twin quiet. A pair that moves nothing
  is `not-discriminating`, never `quiet`; a run form that fires with the program the same and no state or input moved
  is the check's own gap and is refused (`two-builds.undetected`). On the registry walk's cases three pairs fire, each
  `held` and each pinned: the ellipsoid's semi-axes of `bound_robot_with_parts` and the brick's size and the egg's
  semi-axes of `mrs_solids`, which their nodes consume when they are built and which an override and a sweep refuse; the
  walk bounds what it walked (the trace form over the nonzero floats of those cases' documents), not nodes no case
  builds. Every planted bake of the bake-check spike's set is caught by the trace form at every move from 1e-1 to one
  ulp (`tests/two_builds/`). New refusals: `two-builds.float32`, `two-builds.leaf-shapes`, `two-builds.not-movable`,
  `two-builds.undetected` and `run.run-constant` (MIME's defects).
- **Run-constants** (`MimeNode.run_constants`): a quantity a node makes from its leaves and holds over a run, made once
  per chunk inside the compiled program, before the step loop, by MIME's chunked runs (the flagged run, a study's
  batched run, and `mime.build._maddening.chunk_function`, a pure function of the state and the parameters to
  differentiate through), and read as given by the loop's steps (`run_constants_given`). Reverse mode then derives it
  once per chunk: on FL-9 in free space at 222 force points the gradient through a chunk equals the per-step form's to
  1e-12, and the 666-wide LU leaves the backward loop (MIME-ANO-147). Forward runs keep their arrays bit for bit.
- **A medium's flow, solved: the FVM on a periodic box** (`fvm` 0.1.0, with `body_force` 0.1.0 and
  `mime.scene.PeriodicBox`). `Medium(..., fields={"flow": FVM(domain=PeriodicBox(length=...))})` solves the flow of the
  fluid filling a straight tube's lumen on a Cartesian MAC grid of cubic cells, the wall a smooth Brinkman band (one
  registered band, its stiff interior beyond), stepped by IMEX Euler on the projected monolithic conjugate gradient
  (`Scheme.IMEXEuler()`, convection explicit) or by its steady limit (`Scheme.Steady()`, with `stokes=True`), driven
  by `exp.attach(BodyForce(along=..., steady=...), medium=water)` on the fluid. Each step reports its solve (its
  iterations, its true residual against the tolerance and how it stopped); a step whose solve stopped other than
  converging raises `solve_flagged`, read in `result.flags` with its count and warned of after the run (a flag may now
  declare that it warns, `Flag.warns`); a solve whose residual is not finite returns NaN, so the run is refused as not
  finite, naming the flow. It reports the lumen's and the solid's fluxes, flagging the solid's bypass and,
  with `stokes=True`, the lumen's Reynolds number past their declared lines, lists every validity line it carries, and
  with the energy ledger on the node its step identity; the derivative solves report through a zero parameter's
  gradient, a row a step with `solve_report_rows` (a run past the last row refused before it runs). With convection on,
  at the first step past the scheme's CFL line the flow's state is held, the run ends at the end of that chunk and is
  refused, naming the step and a timestep to try. The wall's settings are its group, `wall=Band(delta_in=...)`. Its
  derivative is checked exact in the viscosity, the density and the drive, and declared unverified in the tube's
  geometry (the entry's status). `Medium(flow=...)` is kept as `fields={"flow": ...}`'s alias; the saved form is
  unchanged (a medium's flow is written where dev1 wrote it). The drive is delivered once a step and as its
  waveform's coefficients (`flow.waveform`), for a scheme whose stages read it at their own instants.

- **Resistive force theory, `Drag.RFT`, on kinds with a centreline** (phase R1). A slender body's resistance from local
  drag coefficients along its centreline, its segments feeling no interaction through the free fluid, with a wall's
  image between its points (the tube's table, through the wall kind, never by name). The coefficients are a setting,
  `coefficients=`: Lighthill's on a segment with a wavelength (a helix), a slender prolate spheroid's on a straight one
  (the default per segment), each computed in the step from the viscosity and the kind's numbers, so its derivatives
  in the geometry and the viscosity are exact (checked against a central difference that rebuilds, both modes, both
  loss scales, with a `stop_gradient` and a baked coefficient as controls). Its resistance is judged each step: a straight body's (a capsule's, a prolate spheroid's) as
  **semidefinite at a floor** (`floored_least_eigenvalue`, 64 units in the last place of the run's dtype), since its spin
  about itself meets no drag; a helix's, which is definite, as the MRS rung's is, against zero. With a wall, a body whose local drag outweighs its true resistance
  (a helix the size of its vessel) is refused at build, `drag.rft-indefinite`, and a step that reaches one
  stops the run, naming the MRS as the route. A kind without a centreline is refused, `drag.no-centreline`, the
  message naming the new guide page on adding one (a custom geometry, an STL or CAD mesh). The rung was added to drag
  0.4.0, changing nothing the other rungs compute: its setting `coefficients` is left out of 0.4.0's resolved record of
  a program on another rung (`Entry.unrecorded_settings`), so every such program keeps its input hash and a folder saved
  before the rung loads. Drag 0.5.0 (the near field) carries the rung too, and records `coefficients` on every rung, null
  off the rft rung, since 0.5.0 has no earlier release whose hash to keep (pinned in `tests/saved/test_drag_record.py`). Its tube family was measured at one gap to the wall, and a step at another gap, wider or
  narrower, raises `unmeasured`. Its validity lines, measured against the MRS rung on the same
  bodies, are recorded in `docs/data/slender_validity/`.
- **The centreline capability** (`mime.registry.Centreline`): a tree of segments joined at named junctions, each a
  JAX function of its kind's parameters giving the point, the unit tangent and the rotation-minimising frame's normal
  at arc length, with an optional local radius, a filament radius and a wavelength. Declared by the new **`Helix`**
  (registry `helix` 0.1.0: a helical filament with hemispherical caps, its origin its centroid, its samples
  quasi-uniform), by `Capsule` (tip to tip), by a prolate `Ellipsoid` (its long axis) and by `Tube` (its axis, the
  radius from the lumen's signed distance). `mime.scene.polyline` declares one through points; `mime.testing.
  centreline_problems` holds a kind's to the contract.
- **The plane wall kind** (`plane` 0.1.0, `mime.walls.PlaneWall`): an unbounded no-slip plane for the drag's MRS rung,
  its image system in closed form, regularised by the drag's own blob, so that with the free Stokeslet of that blob it
  vanishes on the wall and has no divergence to rounding, and as the blob vanishes it is Blake's singular image. The
  plane is a structure, `mime.scene.Plane(normal=+Z, offset=...)`, whose `half_space` a medium fills
  (`Medium("water", fills=floor.half_space)`); its shape is `mime.core.HalfSpace` (`half_space`), convex, taking many
  points, with no switch. The offset is the drag node's leaf `wall_offset`, which a gradient and a caller of the node
  reach; **a structure's settings are not yet study or override axes** (`override={"floor.offset": ...}` is refused,
  `value.unknown`, and so are `drag:<body>.wall_offset` and `drag:<body>.offset`, `study.unknown-path`), and they come
  with the study layer's declared ranges. The normal is held as built and normalised in the
  program, so declaring it a leaf later changes no program. The gradient is exact. The plane has no surface yet, so no
  contact can be held against it; the build refuses a body that starts through it, and the drag's near-wall flags
  speak in a run. **The wall protocol passes the drag's regularisation to every wall kind**: a correction and its
  image between point sets take `eps_squared`, the blob's square, and `WallCorrection.regularised` declares whether a
  kind uses it (the tube's table image is singular and declares it does not); the build report says per wall which
  image it adds. A correction a lab registers must take the new argument (a default of `None`
  serves), and registration refuses one that does not (`entry.defect`, naming the function and the argument) instead
  of failing with a bare `TypeError` at the first trace. Every released program on the tube and in free space runs to the same bits.

- **RK3, the step rule and `dt="auto"`; the prescribed flow; the solved-field contract's suite** (F1b). `FVM(...,
  scheme=Scheme.RK3())` steps the flow by S1's explicit RK3 (three low-storage stages, the wall's penalty implicit in
  each, the incremental fast projection, the pressure carried, a consistent pressure made in the program at the run's
  start and after a state handed over), reading the drive's waveform at its stage times. With no `scheme=` the build
  picks RK3 where its explicit viscosity holds at `dt` and IMEX Euler otherwise; it predicts the flow's CFL number from
  the drive (Poiseuille's centreline and twice Womersley's peak, the bound on a run from rest, in the empty tube) and
  refuses a `dt` past the chosen
  scheme's line before any step (`flow.cfl-predicted`, `flow.viscous-limit`), with the step that keeps it;
  `Experiment(..., dt="auto")` takes the largest stable step (stability only, its steps a period reported beside it),
  saved as the step it chose. The rule is judged at the flow's own step (`FVM(rate=...)` included), its prediction in
  float64 whatever the run's dtype, with a bound past the series' Womersley number of 50. The lines are one value each, along a grid axis, scaled along a tilted vessel's axis by
  `max|n_d| / sum|n_d|`: RK3's is 1.7, not the 2.0 first decided, at which a plug flow grows (the maintainer's
  decision). RK3 is a scheme of the registry's interface: its state the velocity, the pressure and whether the pressure
  is made, its stages reading the drive at their own instants, its terms its own; its node holds its state past the CFL
  line as the implicit node does. RK3's books are judged on the ledger's first route, per step (`ledger_route_one_*`).
  `Womersley()` (`womersley` 0.1.0) is a medium's flow prescribed in a straight tube, Poiseuille and Womersley by an in-program
  series valid to a Womersley number of 50 in float64 and 25 in float32, outside and with no grid; `BodyForce` drives
  either rung (it delivers its waveform where the field reads one). `mime.fields.SolvedField` holds a field kind's
  declarations and `mime.testing.field_conformance` its suite, each case applicable by declaration and every skipped
  case listed with its reason.

- **The energy ledger on an apparatus, the magnetic pair's class, and the build's structural flags**. With
  `EnergyLedger` attached, **a dynamic arm and a dynamic motor are metered** (the `joint` port kind, now taking the
  spinners' momenta and the inertia its frames carry, and the new `spinner` kind): each records its step identity
  inside its own step, closing within the ledger's floor on A0's dynamic loop in both dtypes, nothing arriving
  unattributed. The arm books its commands (under their writer, the controller or a drive, outside), each law's loads
  on its frames by the joint forces they make, its spinners' gyroscopic exchange (workless, by the Cayley step), the
  inertia its frames carry by its change as the step used it, and the laws it holds itself under their names: its
  weight (a potential, its energy taken from the run's ends), its damping and its stops' and Coulomb friction's
  impulses (dissipative). The motor books its drive by its angle's change, the axial loads on its rotor, its bearing's
  loss on the rotor's turn on its stator, its rotor's inequality (the rotor's half of the exchange whose other half is
  the inertia its host carries, `<motor>.inertia` on both, workless) and its stator's torques on its host's turn. What
  one segment hands another is booked under the output that carries it (`rpm.motor.reaction`). **A receiver's storage
  between steps** is recorded apart (`ledger_between`): where a delivered inertia moves it, it is that exchange's own,
  reported with the exchange (`EnergyAccount.exchanges`); in a coupling group, the group's residual; **whatever no
  declared producer explains is judged as unattributed work is**: the storage of the receiver's own state at a step's
  start less the one the step before ended at, kept as its worst over the step's scale (`ledger_between_worst`), over
  16 eps a failure of the step identity (`EnergyAccount.identity_worst`, `between_worst`), and left in the balance, so
  that a state edited between steps (a 50 rad/s gift to A0's motor, 1.25e-2 J) fails every run; its resolution is the
  identity's, one floor (a gift of half a floor passes, one of two fails). **A run says when a check failed**
  (`EnergyAccount.findings`, `RunResult.energy_findings`, the first lines of `result.energy.text()`, `result.text()`
  and `mime run`'s attention lines, with the amount): in particular work that arrived where no law or writer owns it,
  which is what a controller called on the host (mode `callback`) writes into a metered receiver, its books closing
  and its work named by no writer; the build report names such a controller outside. `<motor>.reaction`, the
  stator's reaction on its host, declares no class: its paired work is on both receivers' books, and the pair's sum,
  the coupling's first-order error (+5.8e-5 J, 1.8 % of A0's supply over 50 ms), is reported and not bounded by a test. **Outside, by the mount tree**: on an apparatus with a
  chain each segment and each field it carries is judged apart, a field outside unless its laws deliver their
  reactions into a segment of it, so a magnet on a dynamic arm is inside and one on a kinematic arm outside, and in an
  apparatus with a dynamic and a kinematic part only the kinematic part is outside; the controller and every drive are
  outside, and a command into a metered receiver is supplied work named by its writer. **The magnetic laws 0.2.0 are
  potential laws**: the family states their energy once for the pair, `U = -m . B` at the magnet's centroid; the
  energy case judges the pair's wrench against its negative gradient (the field from the provider's own function), and
  the closed-loop case now drives the source round closed loops too, the laws' reactions on it included, with the body
  held and with the two moving together (`pair_loop_works`); 0.1.0 stays unmetered by MIME-ANO-137, and fails the
  extended case where it did, and round its source's turns (its reaction carries the lever's couple). Where the field's
  source is outside, the pair's work is supplied by it; where it is inside, the account takes the pair's energy's
  change from the run's ends (`PointDipoleNode.energy_at`). **Structural flags**, in every profile and with or without
  the ledger: a law delivering to receivers that step at different rates is refused (`law.rate-divided`), a damper
  between two receivers that neither one step holds nor one coupling group solves is flagged by a line the build
  computes on it, and every run reports each coupling group's verdict from MADDENING's own (`RunResult.groups`), a
  group stopped at its cap said so. `mime.testing`: the joint and spinner port kinds' step identities on drawn states
  (`PORT_CASES`), with a planted gift that fails them. **Without the ledger nothing moves**: A0 on both rungs, A0 with
  its magnet's mass and script A keep every history array to the bit in both dtypes. Counted: one refusal code, one port
  kind.
- **Plots, figures kept as recipes, comparisons, the ghosted overlay and the aliasing warning** (`docs/results.md`). A new
  `plot` extra (matplotlib and plotly; the `verify` extra holds it). `result.plot()` (matplotlib) and
  `result.plot_interactive()` (plotly) on a run and a study draw one backend-neutral description: flags as intervals where the
  result's track raised them, numeric validity limits and uncertainties as bands, and a note on what the figure estimates and
  what it bounds. They read a small result protocol, so a result built by hand (`mime.results.Measured`) plots and compares
  as a simulated one. `result.figure(...).write_recipe()` saves `figures/<name>.json` (the data files by sha256) and plotly's
  JSON; `FigureRecipe.read(folder, name)` redraws it, refused by name where a data file changed. `StudyResult.compare` puts
  members against a named reference on the reference's time base, with a tolerance that is the study's rule or a stated
  quantity and never a fixed absolute, refused where the reference cannot resolve a declared drive frequency.
  `sweep(..., stage_every=...)` keeps each member's bound poses, and `result.write_compare()` writes `compare.usda` (one prim
  `/Compare/<member>` each, with visibility and metadata). A recording as sparse as half the period of a declared drive warns.
  New refusal codes `plot.*`, `compare.*` and `study.stage-every`.
- **The energy ledger in one line, `exp.observe(EnergyLedger())`, its wiring, its account and its family cases** (the
  second half of the ledger; `energy_ledger` 0.1.0). Attached, the build works out what is outside per thing, from each
  entry's plan (a thing that receives no delivery from the components: script A's field), moves each law's delivery
  into its own input on each metered body, in the order its summed input took them, each paired as the body's port kind
  says, and constructs the body again with its ledger; the build report's `energy` lines name each law's class or its
  reason for being unmetered, what supplies it and what is outside. `result.energy` is the run's account in plain words:
  what each body stores, what each dissipative law lost, what was supplied, named by what supplies it and taken from
  the dissipation side (what was lost plus the change in what is stored), and each potential's change from the run's
  two ends with the law's own parameters; with `details=True`, each law's paired work, the scheme's own terms apart,
  the paired supply and the checks; `efficiency(useful)`. **The account's sums are differentiable**: formed by
  `mime.results.energy_account` inside a function being differentiated (the final state and the parameters traced),
  `supplied_total`, `efficiency` and the potentials' change are traced arithmetic with no conversion to NumPy, and the
  derivative through them agrees with central differences as the step squared (a `result.energy` read from a finished
  run holds numbers). A post-build check compares the graph as built, edge for edge, with the deliveries the wiring
  made (`post-build.ledger-inputs`): a delivery in another law's own input (the same field's or another's, whatever
  the shapes), in a summed input, dropped, or made twice is refused on every build, by name. **That check is what
  catches a swap of two laws' loads between their inputs**, which the books cannot see; the swap shows in a residual
  only where gravity is involved: gravity's, with a moving load (gravity against the torque: thousands of floors in
  float32), and not reliably otherwise (gravity against drag's load: 0.008 of its floor in float32, as the body rests
  on the wall), so the residual is a witness for those swaps alone. An entry that declares nothing and states no reason is logged under exploratory and refused under research
  (`ledger.unmetered`). `mime.testing`: the **energy case** (`CaseKind.ENERGY`) in the magnetic, gravity, contact and
  hydrodynamic families (a potential's wrench against its energy's negative gradient; a dissipative law's operators
  definite and its loads doing no positive work; a contact giving no energy through the solve nor by its rule at
  arrival; a law with no class states its reason), each with a planted law that fails it; **a dynamics' step
  identity** on drawn states (`step_identity`); the registry walk requires every dynamics and interaction of MIME's to
  declare how it is metered or why not (the arm, the motor and the toys state why); **the magnetic family's energy case
  passes the two magnetic laws by their declaration alone** (their reason for being unmetered), which checks nothing about
  them, until their family's energy and loop test are built. The step identity feeds work into the unattributed
  input and requires it booked as unattributed (a plant that books it under a law fails it on every draw). A dissipative law held in its
  body's step now also counts as giving energy where its rule at arrival adds energy. **Without the ledger nothing
  moves**: every history array of the five programs is identical to the bit to the release before it, in both dtypes,
  and a save with the ledger off is byte-identical; with it, the bodies' programs are new, judged by their identity.
  What energy cannot see stays the families' oracles': a drag scaled by 0.9, or its coupling's sign flipped, passes
  every ledger check, and the hydrodynamic family's oracle case fails each.
- **A command ramped, a body's phase lag, and a study's step-out by the quantity it is** (`docs/guide/studies.md`).
  `drives.Ramp(start, to=..., rate=..., hold=...)` holds a command, moves it at a rate (the command's units per
  second) and holds it where it ends, writing each step its value at the step's middle (`drive.ramp` 0.1.0).
  `observations.PhaseLag(body, field=..., about=..., after=...)` (`phase_lag` 0.1.0) is the angle from the field to the
  body's moment in the plane of rotation, net of the lag the build declares for the laws bound to the body and field (the
  build report's lags, read from its lag accounting; refused, `observation.lag-ambiguous`, where they disagree), and the body's spin about the axis in turns per second, with their means over a window; it reads a uniform
  field where a law reads it, adding no slot to the provider. A study's `protocol=` is `FromRest()` (the default) or
  `Ramp(field.commands.frequency, ...)`, which ramps each point's command from its value; `step_out(result, lag, axis)`
  reads the bracket from rest and `ramp_crossings(result, lag)` where a ramp's member steps out or is caught. Measured
  on script A (float64, 0.5 ms, acceptance tests, not accuracy claims): from rest the body is last locked at 43.25 Hz
  and first unlocked at 43.5 Hz (41.0 and 41.25 Hz at 0.9 times the amplitude); ramped down from 100 Hz at 0.25 Hz/s it
  is caught at 31.73 Hz. Five refusal codes: `command.held-constant`, `observation.lag-ambiguous`,
  `study.protocol-command`, `study.protocol-quantity`, `study.series-every-step`. `ramp_crossings` reads the state a ramp
  starts from over the last window of its hold: a ramp up from an unlocked body and a ramp down from a locked one report
  that start, not a step-out or a catch at the start frequency.
  A study run on a ramp runs in memory: a study folder records no protocol yet, so `run_to` refuses it
  (`study.protocol-folder`) rather than write a folder that reads back as run from rest; a ramp's member made an
  experiment (`as_experiment`) declares its ramp from the member's value, as the study did, and a ramp's protocol is
  part of the study's id.
- **`--deterministic` on `mime run` and `mime verify`, and `mime.env.deterministic_mode()`**: XLA's deterministic GPU
  mode turned on before JAX's backend starts, since XLA reads its flags only then (measured: a flag set after the
  backend started took no effect, one set after `import jax` and `import mime` did), and recorded in the lock as before.
  The Python call is refused once the backend has started (`env.deterministic-late`), naming the routes that work,
  unless the flags already turn the mode on. On the CPU either says in one line that it changes nothing there.
- **Held points' type in a node's typed state** (`mime.core.HeldPoints`, `HeldForces`, read by `field_kinds`):
  `rigid_inertial`'s contact points are judged by `verify` only on the rows whose force acts; a new node whose state
  writes a field as `HeldPoints` beside one `HeldForces` field is judged so, with nothing declared beside it.
- **The energy ledger's machinery on the body, off unless attached** (`mime.core.ledger`; the first half of the
  ledger, whose build wiring, observation and account come next). The vocabulary's port kinds (`PortKind`,
  `register_port_kind`; MIME's `rigid_body`, `joint`, `delivered_inertia`, `flow_in`), each delivery's units paired
  with its flow's to watts when the kind is made, and the energy classes (`EnergyClass`: dissipative, potential,
  workless; no source class) with a dissipative law's `Definiteness`. The recorder: one pairing rule (a load read at
  the step's start with the configuration's actual change, a force solved for in the step with the step's solution),
  per-law inputs on a receiver so that attribution is the check that the unattributed inputs paired no work, exactly
  zero, and running sums of each law's paired work and of the receiver's step identity. Any node takes `ledger`;
  `rigid_inertial`'s node (every version) books `V^T M V / 2`, backward Euler's and the gyroscopic term's numerical
  terms and the hosted contact's loss and projection term under the contact's name, judged within 16 units of
  rounding on the step's scale (the magnitude of the arithmetic its terms are formed by, a stiff solve's included).
  The laws' classes: gravity a potential, its family's energy on its node (`GravityNode.energy`); the drag
  dissipative, definite by construction on the surrogate rung and checked at the state on the MRS rung, by a Cholesky
  factorisation at its floor (a semidefinite resistance passes); the contact dissipative, hosted in its body's step;
  the magnetic laws unmetered, each with its reason. **Without the ledger nothing moves**: every history array of
  script A (the example and the reproduction arm), A0, script A with the hull contact and the screw-in-tube example is
  identical to the bit to the release before it, in float32 and float64 (at 6 usable CPUs and one OpenBLAS thread), and
  script A saved with the ledger off is byte-identical. `tests/energy/` holds the cases, their predictions written
  before they ran: the body's identity (free, its centre held, its hull held, the hull contact's unrolled and looped
  programs) at 0.13 to 0.28 of its floor over 300 fresh draws a case and dtype, an unbooked loss of 0.1 % of the
  free body's twist caught on every draw (a sample minimum of 290 times the floor in float32 over 200 draws, not a
  bound); on every arm, a gift of 64 floors, a contact's loss and its projection's shift and each of the scheme's
  three terms misbooked by 64 floors (either sign), each read as 64 floors on every draw in both dtypes, and a
  contact that gives energy caught by the dissipative law's sign (`ledger_gift_steps`) on every draw; the unattributed
  work exactly zero, and a hidden 1e-3 and 1e-6 of a load caught on every draw;
  gravity's wrench the negative gradient of its energy on the rotation group (at most 0.16 of its floor), a weight 1 %
  heavier caught at least 477 times over; the surrogate's resistance definite at every drawn pose (its scaled least
  eigenvalue 0.68 to 0.93, measured over script A's tube and 5 degrees of tilt) and an indefinite part planted in it
  (1.1 times its least scaled eigenvalue taken off) caught on every draw (-0.17 to -0.10); the definiteness check
  fails a zero diagonal entry that has a coupled row, whatever the coupling's size. **Not in this part, by design:**
  attribution sees a load delivered where no law owns it, not a swap of two laws' loads between their own inputs
  (refused by the second part's post-build check); and no declaration of a port, a class or a definiteness was read by
  anything in this part (the second part's wiring and registry walk read them).
- **A study written to a folder, resumed, read back and made an experiment of** (`mime.results`, `mime.study`).
  `StudyResult` is complete: a long `table()`, `to_csv`, `to_npz`, `to_mat` (the member axis first, units in the file),
  `grid()`, `manifest()`, `order(output, along=...)` (the observed order of convergence over three levels and the
  extrapolated value, refused where the levels are not in the asymptotic range) and `read(folder)` of a partial study.
  `study.run_to(folder)` writes a study folder (`study.json`, the spec file `study.mime.yaml`, the base saved once, each
  group's arrays, each group file's sha256 in the manifest and each member's content key in its entries; closed
  manifest, checked layout, no saved-form step) batch by batch, into a new or empty folder or one that holds the same
  study (anything else is refused and left untouched), and a run into the same folder reuses only the members that are
  done under the same content key (the point, the group's structure key, the base's input hash, the run, and the code:
  every file of the installed `mime` package and of a package its classes come from, with the versions of MADDENING,
  OpenMaterica, JAX, jaxlib, NumPy and SciPy, so any change to MIME's files runs every member again). `read` refuses a
  group file that is not the manifest's (`study.group-file`). `mime study plan` and `mime study run` work on the spec
  file (`--shard i/k` by structure group); `mime modal --study` adds a `study` entrypoint (one container by default,
  `--shards` explicit). The research profile
  needs one reason per swept axis (`study.axis-reason`). `result.member(i).as_experiment(name)` makes a member an
  experiment of its own through `Experiment.based_on`. A law's axis given as a string now resolves when the law's id
  holds a dot of its own (`toy.spring:probe.k`), which S2b-1's resolution did not.
- **A containment watch: a body the centre-held contact holds whose hull leaves its region is flagged after the run**
  (in every profile that can be built, exploratory and research; never a refusal, no node or edge, so the graph and its
  derived hash are what they were). The contact 0.1.0 holds a body's centre by its extent and never reads its
  orientation, so script A's body given FL-9's own inertia tensor, instead of the isotropic effective inertia, turns its
  axis across the tube (45 degrees at 0.34 s) and its hull is through the wall by millimetres, which nothing flagged but
  the surrogate drag's own tilt and bore flags, later and in words about the drag. `result.containment` (and
  `result.text()`, the manifest, `outputs.json`, the `.mat` file and the end-of-run summary) now says the step, the depth
  past the wall and the fix (`Contact.Projection` reads the orientation). A run that kept the pose is read at every step
  it kept (`record=` keeps it for this); one that kept nothing is read at its last step alone, and says that it knows
  nothing of the steps before (the summary says so in one line). A centre-held body the watch cannot check (a surface
  whose shape takes one point at a time, a hull kind that gives no samples, a body whose dynamics publish no pose) is
  listed in the build report and read as not checked, with the reason. It is read when first asked for (the result's
  `containment`, its text, manifest and exports, the summary), so a run that never looks pays nothing. The depth is of
  the hull's samples, which fall up to 23 um short of the body at the watch's spacing (`shortfall`, quoted in the text):
  a bound on how far a flagged depth can understate the true one. `tests/results/test_containment.py` holds the
  reproduced case, the false-alarm cases (script A as it is, the record arm), the hull contact's run, independent
  depths, and the mutants the reviews planted.
- **The experiment's tree and its compiled graph, shown without running it** (`docs/guide/graph_and_tree.md`).
  `experiment.tree()` is what the experiment declares (structures, media and what each fills, bodies and their parts,
  apparatus with their chain, parts, fields and drives, each law with the things it binds, observations with what they
  read, the controller with its reads and writes), each with its registered id and version: plain indented text
  (`str(tree)`) or data (`tree.record()`, `mime.tree/1`); it is the topology, not the declared values, settings or run
  settings. `built.to_mermaid()` is MADDENING's `to_mermaid` of the
  compiled graph with the registered entry added to each node, and a notebook shows it as a page in an `<iframe srcdoc>`
  (the same in Colab, JupyterLab, Notebook 7 and VS Code) in which Mermaid 11.17.2 draws it and a reader pans and zooms it
  (drag, wheel or pinch, buttons for zoom in, zoom out, fit and 1:1, full screen), in a frame 720 pixels high as the 3-D
  view's is. It takes the 3-D view's names and behaviour: the notebook's page loads Mermaid from jsDelivr, held to its
  `integrity` hash; `to_mermaid().display(offline=True)` carries a copy shipped in the wheel (3.6 MB, with the licence of every
  package it can hold) so the page needs no network, and returns a `GraphPage` as the 3-D view's `display` returns a `WebPage`;
  `to_mermaid().save("graph.html")` writes the page with it; a graph built before a replacement or a removal says it is stale.
  `mime show <folder>` prints the tree (`--tree`, the default, drawn as a
  tree on a terminal, plain text piped, `--json` as data) or the graph (`--graph`), or both, or `--out [FILE]` writes the
  graph as that page (`<experiment>.graph.html` by default). Either is the same for the
  same experiment whatever the order its script declared things in.

- **Colab notebooks** (`docs/notebooks/`), linked first in the README: a steel ball settling in a tube of water, and
  de Jongh's screw-shaped magnetic robot driven along a tube by a rotating field (tutorial A's experiment). Written
  for a newcomer, step by step in plain words, each installs the released MIME from PyPI, runs its experiment with the
  live display, plots it with matplotlib, shows the browser view, and ends with changes to try.
  `tests/examples/test_notebooks.py` builds each notebook's experiment against the checkout and runs A's for a few
  steps; `scripts/colab_links.py` points every Colab link at one branch.
- **`Experiment.based_on(source, name=...)`: a saved experiment changed as a new one, which records it.** `source` is
  a saved folder, a bundle file, or what `mime pull` takes (a DOI, an address), pulled first into `./<its name>/` as
  `mime pull` pulls it; where that folder already holds the same experiment (the same input hash: a notebook's cell
  run again), it is used as it is, its assets passing the same gates, and where it holds anything else it is refused.
  The saved experiment is loaded and checked against its lock as `Experiment.load` does, and its folder is never
  written. The new experiment holds its declarations and run settings under the new name with no
  lock; a declaration taken from it, made again and changed, replaces the one taken, in a script as in a notebook,
  with a warning recorded in `plan().replaced` (`exp.declare("rpm.magnetic_moment", A_m2(25.0), reason=...)`). Saving
  it writes a new folder with its own lock, whose document records the saved experiment as `based_on`: its name, the
  input hash its lock records, the MIME version that saved it, and its DOI: the record it was pulled from, where
  `source` is a DOI (a bundle uploaded by hand is named by it too, and a lock that names another record is warned of,
  naming both), and otherwise the record its lock names, if any. A DOI that is not a DOI's shape (`10.<registrant>/...`,
  no whitespace or control character, at most 256 characters) is not recorded, and the warning says why. Only the
  immediate one: each parent's document records its own. `mime card` names it (`Based on <name> (<its
  DOI, or its input hash>)`). The parent's third-party files keep their terms and tiers, so `mime pack` carries and
  references the same files. **`mime push` sends what it sent before**: no relation to the parent and no line naming
  it in the description, until that relation (DataCite's `IsDerivedFrom`) is rehearsed on Zenodo's sandbox.
- **The saved form is version 7** (`mime.document/7`, `mime.lock/7`): the document's `based_on`, written only where
  an experiment was made from another, inside the document and input hashes. A folder of version 6 loads as before,
  and one of version 6 holding `based_on` is refused. 0.3.0.dev1, which reads up to version 6, refuses a folder of
  version 7 as `load.newer-form`, the document and the lock each once.

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

- **A study's reverse-mode derivative in float32 through a body resolved on the FVM's grid (`Drag.Penalty`) is
  refused outside the envelope where it was measured right** (`drag.penalty-float32-reverse`, before the derivative
  compiles), routed to float64 or to forward mode. The envelope, allowed: a sphere in the default 0.1h interior at up
  to 6 cells a radius, where float32 read within 3.6e-4 of float64 (solved to 1e-12) through the steady step in the
  body's position, the viscosity and the drive, and within 2.1e-3 on the study's own path (R3's sphere free at Re 76,
  IMEX Euler, 40 steps), as the rung's validity line now states. Refused: the 0.01h interior (measured 3.2 % off in
  position at 6 cells a radius, 0.16 % in the viscosity and the drive, every transpose solve reporting convergence),
  more than 6 cells a radius and a capsule (unmeasured; the gap grew 4.5 times from 3 cells to 6). The cause is
  float32's stop on the transpose's right-hand side, not a derivative defect: float64 at float32's 1e-5 reproduces each
  gap, the solves run in float64 inside the float32 program close it (a guard not built), and float64 at 1e-12 holds
  reverse against forward mode to 1e-11. Forward mode, whose tangent solve was flagged wherever its derivative was off
  by more than 0.1 %, and float64 are not refused. Predictions and outcomes: `tests/flow/predictions_f4_float32.json`.

- **`Drag.MRS`'s 0.5.0 settings grouped by concern** (design #326 as the names sitting, #336, decided it, the names approved by the maintainer, #353; drag 0.5.0 is
  unreleased, so no saved-form step). The core stays flat (`spacing`, `blob`, `patches`, `wall_kind`, and 0.4.0's
  others); the force points' grading is `sampling=ForceSampling(grading=..., contact=..., contact_gap=...)` and the
  near field's joint solve `near_field=NearField(solve=..., tolerance=..., share=...)`, both exported from
  `mime.interactions` (`near_field_share`, `near_field_tolerance`, `near_field_dense_unknowns`, `grading`, `contact`
  and `contact_gap` are gone); near contact stays a flat switch, `near_contact=` (the near-contact law's entry). A group's setting
  may declare a default computed by the entry's own method (`contact_gap`'s, the closest approach), which the build takes
  and records at `<setting>.<field>`, and registration checks the method is the entry's. `mime.registry.Feature` is
  `SurfaceFeature`. Registered: MIME-ANO-155, a reverse-mode derivative in a shape leaf through MADDENING's own
  `run_scan` remaking the MRS rung's held matrix at every backward step (32 to 660 s where MIME's chunked run takes
  about a minute; MADDENING Ask 51).

- **`Dynamics.port` is now `port_kinds`, a tuple** (no alias): a dynamics may name several port kinds, each delivery
  paired by the one kind whose deliveries name its interface, and two kinds pairing one interface are refused at
  registration. The rigid body, the arm, the motor and the FVM (now `("fluid", "opening")`) declare theirs; a lab's
  dynamics writes `port_kinds = ("rigid_body",)` where it wrote `port = "rigid_body"`.

- `TubeTree.features()` is `TubeTree.wall_features()`: `features` is every geometry kind's surface features, which a
  sampling grades towards.

- **The FVM's conjugate gradient restarts from its true residual where it would stop stagnated.** With the stiffer
  interior (`Band(delta_in=0.01)`) in a flow with a pressure gradient, float32's recurrence stalled far above its
  rounding floor (a 75 % stenosis at 16 cells across, 1e-4 against 4.8e-6), flagged. Now, where the stop would find a
  stagnation, the CG restarts from the true residual, its window begun again, up to three times before it stops
  flagged (`Stopping.stall_restarts`): every stall measured at 8, 16 and 32 cells across converges, unflagged, at its
  tolerance or floor. A solve that never stagnates never restarts, so its arithmetic is unchanged, in either dtype; no
  saved or released program moves. A restart every 100 iterations and the residual replaced with the direction kept
  were measured first and not adopted (the entry's validity says why).

- **A round length keeps its count of samples** (`mime.scene`'s solids and screw). A kind's count along a length was
  the ceiling of the length over the spacing, so a round input (a 2 mm capsule at a 1 mm spacing, a whole number of
  spacings) took the next count, and a recompiled program, where its last digit or a study's move of a millionth lifted
  the ratio past the whole number: the two-builds walk saw the near field's case's program change. The ceiling now
  takes a slack of a ten-thousandth of the ratio (`COUNT_SLACK`), so a length up to that share past a whole number of
  spacings keeps the lower count and the count's step lies off the round values. **The counts at such round inputs
  were a float artefact of 0.3.0.dev1, fixed under the kinds' own versions**: a configuration whose length lies within
  that share above a whole number of spacings takes one row fewer than in 0.3.0.dev1 (5 of 400 round inputs on a 0.1 mm
  grid, pinned by `tests/scene/test_sampling.py`: 1.3 mm at 0.1 mm takes 13 rows where dev1 took 14, 2.1 mm at 0.1 mm
  21 for 22, 2.1 mm at 0.7 mm 3 for 4, 2.6 mm at 0.1 mm 26 for 27, 2.6 mm at 0.2 mm 13 for 14), and 0.3.0.dev2 refuses
  a folder saved by another release, so none is rebuilt with the moved count. No registered case's golden output and
  no released program's arrays moved (each compared bit for bit).
- **Saved form 11: a medium's fields are written by role.** A medium's fields are saved under its own key `fields`
  (`fields: {flow: {name: fvm, ...}}`), as an apparatus's are, where form 10 wrote a medium's flow as a dynamics aspect
  (`dynamics: {flow: ...}`). A folder of form 10 or earlier loads as before: its medium's flow is read as `fields.flow`
  and written back so for its input hash; one of them whose medium holds `fields` is refused. A reader of form 10
  refuses a folder of form 11 as `load.newer-form`.

- **Saved form 12: the experiment's declared ranges.** The document holds `ranges`, each range `exp.declare_range`
  declares, by its path (its ends in SI, their unit and its reason); nothing else moves. A folder of form 11 loads as
  before, and one of form 10 through form 11's step; one of them holding `ranges` is refused. A reader of form 11
  refuses a folder of form 12 as `load.newer-form`.

- **Saved form 13: the FVM's `report_rows` is `solve_report_rows`.** The setting, its refusal
  (`flow.solve-report-rows`) and the flow node's parameter take the new name. A folder of form 12 or earlier loads as
  before: its old name is read as the new and written back with the old name for its input hash. A reader of form 12
  refuses a folder of form 13 as `load.newer-form`.

- **The MRS drag's held matrix is compared with leaves, not constants, and its second key is gone.** Beside
  `held_in_body_frame` each separated-quadrature drag holds `held_from.<leaf>`, the leaves the matrix was made at;
  `held_or_made` compares the step's leaves with them, so the step's program is the same at every value (the trace form
  is quiet on the rung), a study's member keeps its own float64-made matrix, and the save's resolution rerun gives the
  step the matrix with the leaves it was made at, which keeps it (`made_again`). The value-keyed second key (#201) is
  removed, and the stale read it allowed (a step given every body leaf as the rerun moves it kept the build's matrix)
  cannot be reached. No array moves. Folders of four experiments (the MRS sphere in free space, `mrs_tube`, `hull_contact`, `script_a`) saved by the tree before this change load with it, in the same environment and in float64, as "the same graph, hash for hash" [measured]; no golden hash is in the diff, and whether the derived hash moves for any other experiment was not measured.
- **A hull contact's search values are leaves** (`search_apart_squared`, `search_tolerance`, `search_shortfall`,
  `trainable=False`): made when the body is built from its hull's shape, they were Python numbers in the step's program
  (MIME-ANO-145), so a study whose members' hulls differed under one program would have run the first member's. The
  step reads the same numbers in the run's dtype, so every array is bit for bit what it was (the `hull_contact` folder saved before loads as "the same graph, hash for hash"
  [measured, as above]). They stay outside every derivative: where a search starts and when it stops, which the point it
  converges to does not depend on.
- **The rft rung judges in its step whether a body lies within its measured lines**, from the step's leaves: the family
  of its shape and its slenderness, its coefficient set, its spacing and, with a wall, the gap its family was measured
  at. R1 decided the family and the gap at build, from leaves the step reads, so the step held them as constants: the
  two-builds walk fired on `rft_tube`'s helix turns, whose move crossed the family's one measured slenderness, and a
  leaf written into a compiled chunk would have kept the build's `unmeasured`. The node holds the `spacing` its counts
  were made with (a leaf, `trainable=False`) in place of `measured` and `gap_least`; its build report evaluates the
  same judgement at the leaves as built. A control plants R1's build-time values back and the walk fires on the same
  pair. In float32 R1 flagged every body of `rft_free_space` and `rft_tube` unmeasured (its build compared the spacing
  with a filament radius rounded to float32); the step's judgement agrees with float64's, tested in both.
- **The FVM's float32 solve stops at its rounding floor where 1e-5 is out of reach.** With no tolerance given, a
  float32 run's solve now stops at a true residual of 1e-5 or, where that is higher, at the solve's own rounding
  floor, `eps || |A| |x| + |b| || / ||b||`, made at each check from the iterate (a tolerance given is held as given;
  float64 is unchanged). On the empty tube at 64 cells across with `wall=Band(delta_in=0.01)` the solve that
  stagnated, flagged, after 1000 iterations now stops unflagged after 69, its flux within 1.3e-7 of float64's. A solve
  that stalls above the floor still stops stagnated and is flagged: with the 0.01h interior in a flow with a pressure
  gradient, float32's conjugate gradient can stall well above it. The floor counts only while it is under a hundred
  times the tolerance (1e-3 in float32), and past that the stop holds the tolerance: made from the iterate, the floor
  rises with an iterate that grows, and a singular system's solve was passed as converged at a true residual up to
  thousands of times the right-hand side's; such a solve now ends flagged. The line does not close that case entirely:
  a growing iterate can still stop under it (measured in the entry's validity).

- **CI runs each lane as twenty shards, and a test file the shard table does not list goes to the lightest shard.**
  Two pushes on 2026-11-11 had a shard of two lanes each cancelled at its 30-minute limit: the table (2026-11-10) did not
  list the test files added since, and a file outside it went by a hash of its path, so shards 14 and 10 held 2000 to
  2400 s of them beside their own share. `tests/shard_weights.json` is rebuilt from the push run of `dface1d`, listing
  every file its reports hold, `tests/_sharding.py` places a file the table lacks after the listed ones, on the shard with
  the least time so far, at the table's 90th-percentile file weight (an assumption, replaced at the next rebuild), and
  `scripts/shard_matrix.py`'s default is 20 (the repository variable and a manual run's input still win). Ten tests the
  two runs showed over 30 s cold are listed in `tests/duration_allowlist.txt` with their reasons. Every lane runs cold
  once: the compilation cache's keys name the shard count.

- **CI runs each lane as sixteen shards, not ten, from a rebuilt table.** `tests/shard_weights.json` is rebuilt from the
  JUnit reports of the push run of `fe7c73f` (the old table dated from 2026-11-05, when the suite weighed 22920 s a lane;
  it weighs 28876 s now), and `scripts/shard_matrix.py`'s default is 16 (the repository variable and a manual run's input
  still win). A file heavier than a shard's mean can now be dealt by test (the table's `split`, `scripts/shard_weights.py
  --split FILE`); at sixteen none is, so none is split. Every lane runs cold once: the compilation cache's keys name the
  shard count.

- **`mime verify` judges each field block by block, and the saved form is version 10** (`mime.document/10`,
  `mime.lock/10`). Where `verify` claims no bits, in another environment or of a reference no run repeats bit for bit,
  every field is split into blocks by the units MIME declares for it (a pose into its position and its
  quaternion, a twist into its velocity and its angular velocity, a wrench into its force and its torque, a
  resistance into its translational, coupling and rotational blocks; a field of one unit, or whose units are not
  declared, is one block), and each block is within tolerance when its largest difference at every reference step is
  at most `max(rtol` times its largest magnitude, `k` times its **resolution**`)`, with no absolute part. The save
  measures the resolution: the reference steps run once more, by the same compiled program, from a start one ulp off
  (every nonzero floating number of the initial state and of the small parameters the step reads, one ulp up), and the
  lock records each block's divergence (`reference.resolution`) and `k` (`reference.resolution_factor`), the two keys
  the form adds. Judged whole, a field that mixed units hid a change in its smaller part: in float32 an 8e-6 m shift of
  the tube ball's position under its quaternion's 1.0, and its whole angular velocity, zeroed, flipped or doubled, under
  the surface points' speed (the review of #195, design #266); each is now reported, and a control holds each on every
  run. Held points are judged only on the rows whose force acts (MIME-ANO-144). A save writes its precision's
  tolerance, 1e-5 in float32 and 2e-14 in float64, and `k` = 30: 5.5 times the largest need measured (the MRS drag's
  asymmetry, the spread of an NVIDIA T4 GPU against its CPU's one-ulp resolution) and a quarter of the tube ball's
  angular velocity over its resolution. Each is pinned in a table of the measurements it rests on, and the
  measurements the repository holds are computed again from it. The measurement covers one T4, one laptop's CPU at
  three counts, one demo, eight steps, float32; float64 on the CPU only. A folder keeps the tolerance it was written
  with, and one of the ninth form or earlier is judged as its form says, element by element with its own numbers (1e-6
  and 1e-9), in `verify`, `verify --run` and the load's comparison of the derived graph, which from the tenth form
  holds each list of numbers and each static array to its own largest magnitude. The cost is one more run of the
  reference steps at save: where the start moves a parameter an MRS drag holds a value made from, that run makes the
  value once from the moved parameters (`MimeNode.made_again`) and the step keeps it, rather than making it at each
  step, the resolution the same bit for bit (on the MRS sphere in float64 on a laptop, 8 steps took 1.1 to 2.3 s
  against 11 to 18 s made at each step and 0.01 s for the state moved alone, which measured no divergence there).
- **A node's typed state declares the units of the fields nothing else declares** (`mime.core.Units`, beside the
  field's type: `Annotated[HeldPoints, Units("m")]`; read by `mime.core.typed_units`, after the symbol table and before
  the edges, the binding, the observations and the flags). Every field a save writes now has declared units, and a
  test fails a new one without them: the hull contact's held points (m), forces (N, read as a flag's `1` before) and
  shortfall (m); the MRS drag's stop record (its pose split as a pose is), its diagnostics and its wall gap; the kept
  `rigid_inertial` 0.2.0's pose and twist, whose twist had been judged whole where no edge read it; the swim speed's
  count; the toy body's position and velocity.
- **A GPU reference made without XLA's deterministic mode is judged within tolerance, never bit for bit**, on every
  machine, the GPU it was made on included: two such runs of one program differ in their last bits, so `verify` no
  longer reports a tenth-form folder of one moved where it was made (a folder of an earlier form keeps its
  element-by-element rule, under which such a folder can still read moved; saving it again writes the tenth). Its save
  warns once, and `verify`'s report ends with the same
  line: deterministic mode is off, so the reference verifies within tolerance, not bit for bit; for bit-for-bit
  reproducibility, rerun with `--deterministic`. A reference made in deterministic mode verifies bit for bit in the same
  environment, and the CPU is as it was. The card's tolerance line states each level and the rule of the folder's form.
  `docs/running_on_modal.md`'s "the flag stays optional" is replaced by the measurement.
- **`rigid_inertial` 0.4.0 carries a body's full inertia tensor** (F0-3, phase I1). A body's `inertia` is three principal
  moments about its axes, as before, or the symmetric 3 x 3 tensor about its centre of mass in its frame, nine numbers
  row-major (`kg_m2((Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz))`), checked symmetric and positive definite at build
  (`value.not-positive`). A body's parts now give the full tensor, its products of inertia carried as they are, in every
  profile: the refusal `parts.inertia-axes` (research) and exploratory's waiver that dropped the products with a pending
  stamp (#225) are gone, with the finding class `simplification`, its two profile rules and the general
  simplification channel that had no other producer. Research may still declare the diagonal with a reason, a modelling
  choice; the build report names the parts' tensor it takes the place of. The node (`RigidInertialTensorNode`) is
  0.3.0's midpoint step with the gyroscopic Jacobian split: 0.3.0's difference formula for the diagonal and the general
  form for the products alone, so a body with no products moves as it did on 0.3.0, every history array identical in
  float32 and float64 on the released programs (script A, its reproduction and record arms, A with the hull contact,
  A0 and its dynamic rungs, script A on FL-9's own diagonal and the frozen register's transverse arm, each of which a
  Jacobian from the world inertia, or one unit in the last place on a moment, moves; and the screw example, which
  neither moves: in its viscous regime its inertia does not reach its bits), and its derivative is finite through
  equal moments, where carrying the tensor as principal moments and axes is not. The derived graph's `inertia`
  is now nine numbers for every rigid body, so every such experiment has a new derived hash; nothing it computes moved.
  0.3.0 stays registered for saved folders (`RigidInertialMoments`): it takes the diagonal and refuses a tensor whose
  products it did not take as none; 0.2.0 is its explicit variant, as before. The frozen register keeps 0.2.0. A small
  shared 6 x 6 mass-matrix type, `mime.core.multibody.MassMatrix`, checked symmetric and positive definite where it is
  built, is what the multibody library and a later added mass share; the rigid node keeps its own arithmetic.
- **The saved form is version 9** (`mime.document/9`, `mime.lock/9`). No key is added: the resolved record now holds
  the inertia a body's parts give as the full tensor, nine numbers, where version 8 and before held its diagonal, and
  a declaration in place of it names the parts' nine numbers in its source. A declared inertia keeps the form it was
  declared in. A folder of an earlier form loads as before: its document names `rigid_inertial` 0.3.0 (or 0.2.0),
  which builds its node from the diagonal, and its record is compared in its own form. A folder that rested on
  exploratory's waiver of a body's products is refused at load (`value.products-of-inertia`), saying that 0.3.0 takes
  three moments and that the folder is re-created from its script on the current version, where the products are
  carried. A reader of version 8 refuses a
  folder of version 9 as `load.newer-form`, the document and the lock each once.
- **Known anomaly MIME-ANO-143 (filed open, resolved in this release: see Fixed): under jit, the tube wall read's derivative can have the wrong sign for a pair of force points at exactly
  equal height**, in forward and in reverse mode, with the pair set depending on XLA's fusion (measured on the tube demo's
  338 points in float64 and on two-point cases; eager evaluation and points nudged apart match finite differences). No
  derivative in `src/` goes through the wall read; nothing about the primal values changes. Filed with its reach and the
  route (one value of the offset for the sign and the absolute value); no code changes here.
- **Known anomaly MIME-ANO-144 (open, minor): the hull contact's Newton search can end at points millimetres apart under a
  change of one unit in the last place of the pose, for a group seeded beyond the margin of the shrunk wall.** In float32,
  FL-9 in the tube: five draws have such a row (3 of 80 random draws), whose held point moves by 2.0 to 8.2 mm, to another
  converged point or to the group's nearest sample. Measured on those draws, it reaches no force, net wrench, flag or
  20 ms run beyond what a row that does not move shows: the rows it ends on are out of the step's reach, or tied with rows that
  carry the force; the state's `contact_points` for that row does move. Filed with its reach, what was not measured and the
  routes; no code changes here.
- **Eight known anomalies from F0's design work, all open, none changing code** (the bake-check spike, the open-ports spike
  and the frozen register's decisions on M2, M4 and N4). `MIME-ANO-145` (major): the rigid body keeps three values derived
  from its trainable hull settings as Python floats, so an override, a sweep or a gradient over those settings reads the
  build's values. `MIME-ANO-146` (minor): `arm_tracking`'s home pose reaches the run through the controller's held write
  and not through `params`. `MIME-ANO-147` (minor): in reverse mode the MRS rung's held-or-made resistance is made again
  at every step of the backward loop (about 0.5 s a step on 222 force points, forward runs unaffected). `MIME-ANO-148`
  (minor): the screw's held resistance differed between a float32 process's first build and later ones by 2.4e-10 on a
  worktree before `MIME-ANO-138`'s fix; whether the branch still does is not measured. `MIME-ANO-149` (minor, resolved in this release: see Fixed): the
  energy ledger's float32 running sums of a mixed-sign series reach about 15 eps by 1e5 steps. `MIME-ANO-150`,
  `MIME-ANO-151`, `MIME-ANO-152`: the register's M2 (the near-contact margin), M4 (the lubrication counted twice) and N4's
  near-contact part (soft-wall lubrication), the first two placed in phase W2 of the dev2 plan and the third in 0.4.0;
  `MIME-ANO-114` and `MIME-ANO-130` now name that phase.
- **The contact 0.5.0: the hold's sweeps and its search's Newton steps are XLA loops** (`Contact.Projection` builds it).
  The same updates in the same order, with bounds fixed when the step is traced, so both modes of derivative pass
  through; the step's program is smaller and compiles faster. Its rounding is not 0.4.0's: a body held from its first
  step (FL-9 in script A's tube) moves, in float32, its velocity by about 1e-6 of its size, the forces among the rows of
  a body by up to 2e-3 of their scale (their split is not unique) and the net force by under 1e-3 of the momentum rate
  it came from, on the runners measured; in float64 each moves by 1e-11 of its scale or less; a sphere in the tube
  demo is bit for bit (`tests/oracles/test_contact_loops.py` holds the two versions to each other, with tolerances
  stated in units in the last place). The contact 0.4.0 stays registered (`mime.interactions.KEPT`): a folder saved with it loads and
  reproduces to the last bit, its program the one it was derived from. While a switch collector is open (the switch
  harness, the switch scan's check of owners) the loops are unrolled, so the collector reads each update's values. The
  search's ill-conditioning for a group seeded far off the wall, which both versions share, is MIME-ANO-144.
- **A save runs its reference steps on the run's own graph, and after a run chunked by one step builds and compiles
  nothing for them (saved form 8).** `built.save` and the saved form `run_to` writes built a second graph and compiled
  the history scan for the reference outputs; now the reference steps run on the build's own graph, reset to its
  initial state and put back where its runs left it, so a run goes on from there bit for bit. Where a run chunked by
  one step that kept its steps' state (`stage_every`, a hook's `chunk`, a controller called every step, with `record`
  or `history`) compiled the history scan of one step, they are run by it, and the lock names it
  (`reference.program: one-step`); otherwise by the history scan of whole control periods, compiled on the run's graph
  once and kept. A graph edited since the build (a parameter leaf or static array replaced, a NumPy array edited in
  place, a recompile pending) is saved from a graph built afresh, as before. `verify` reruns the lock's program on a
  graph built afresh. Measured on a laptop's CPU, float32, three fresh processes each: the tube demo's save in
  `run_to` (`examples/sphere_settling_tube.py`, `stage_every` 2 ms) from 11.9 s to 1.6 s on 24 CPUs and from 15.6 s to
  1.3 s on 2 CPUs of one core, nothing built or compiled; `sphere_settling.py`'s (no `stage_every`) from 1.9 s to 0.3 s
  and from 2.7 s to 1.0 s, the history scan compiled once. A scan of one step and a longer scan can differ in their
  last bits (in 15 of 28 round-trip cases in float32 and 12 of 28 in float64, at most 4.2e-7 and 4.4e-16), so a folder
  saved after a run chunked by one step can hold other reference outputs than the save before would have written; it
  verifies bit for bit against itself, which is what the lock names the program for. The live display's saving phase
  says "running its reference steps".
- **The saved form is version 8** (`mime.document/8`, `mime.lock/8`): the lock's `reference.program`,
  `environment.usable_cpus` and `environment.blas_threads`, all outside the input hash; the document holds no new key, its schema moving with the
  lock's. A folder of version 7 loads as before, and its reference outputs are rerun by the history scan; one of
  version 7 holding any of the keys is refused. A reader of version 7 (this release before the step) or of version 6
  (0.3.0.dev1) refuses a folder of version 8 as `load.newer-form`, the document and the lock each once.
- **A folder saved under one count of usable CPUs, or of BLAS threads, and verified under another is another
  environment** (MIME-ANO-140, resolved for folders of saved form 8). XLA's CPU backend sizes its thread pools by the
  process's affinity and partitions a program's work by it, and the OpenBLAS under JAX's CPU LAPACK kernels splits a
  routine's sums over its own pool (`OPENBLAS_NUM_THREADS`, else the affinity), so one program computes other bits on
  2 CPUs than on 24 and under one BLAS thread than under 24 (measured on the tube demo and on the MRS point-force
  case); the lock recorded the CPU's model but neither count, so a folder saved unpinned and verified under
  `taskset -c 0,1`, or under another `OPENBLAS_NUM_THREADS`, was read as the same environment and reported moved, bit
  for bit. The lock's environment now records `usable_cpus` and `blas_threads`, and a load or `verify` under another
  count says so and compares within the lock's tolerance. That comparison still reports the tube demo moved
  (MIME-ANO-141, open: its tolerance is absolute below each element's scale, and the differences it reports are about
  10^-3 of the field's largest element, not last bits). A folder of an earlier form records neither count and is
  compared as before: verify it at the counts it was saved at.
- **The documents no longer say that a run chunked for `stage_every` changes nothing it computes**: chunks of two
  steps or more change nothing, and a chunk of one step can change its last bits (`BuiltExperiment.run_to`,
  `docs/recordings.md`, the CLI's README).
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
  the folder again, or keep the edit in a new experiment made from the folder, `Experiment.based_on(<folder>,
  name=...)` (a loaded, edited experiment can be neither built nor saved). The
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

- **The open vessel's reverse-mode derivative under IMEX Euler was wrong** (since the open vessel's node, #246). Its
  projected CG's operator is self-adjoint under the open grid's half-weighted inner product, not the plain one reverse
  mode transposes in, so the transpose solve was the wrong one: through two steps the viscosity's derivative of the
  openings' pressures read 5.27 in reverse mode against forward mode's 7.24, and the inflow's amplitude's 6.69
  against 6.13 (the laws' resistances and compliance right); forward mode was right. The transpose solve is now the
  weighted one, `V (P A P)^-1 V^-1` (`mime.nodes.flow.fvm._solve.projected_solve`); the periodic box's grid has no
  weights and is unchanged. Found by F2b's gradient gate with ports.
- **A body force on the open vessel's fluid booked its work over the planes' half cells counted whole**: its pairing,
  the fluid's momentum, now weighs each face as the open box's inner product does, so a driven open vessel's books close
  (they failed at 5e12 of their floor; the open box was admitted with a body force and not gated with one).
- **The union's derivative in reverse mode** (`ExactUnion`): on a batch of thousands of points it read 4 to 50 % off
  forward mode and the central difference, where two tubes' distances tie exactly (a shared end ball): `jnp.min` and
  `jnp.minimum` find the least's derivative by comparing it again with their operands, which XLA computed twice in
  fusions that rounded them differently. The least is now selected by one comparison, its derivative the selected
  operand's: a guard holds every exact tie's derivative to one operand's (the reduction put back reads their average,
  caught on every run), and a gate holds reverse to forward mode on the node's wall (a reverse-only error its control).
  And reverse mode made a NaN on a batch where a discarded seam's or corner's implicit step was singular: a discarded
  lane now passes no derivative into it.

- **The FVM's stall flag is set from clean runs, in the solve's tolerance, and stops firing on correct float32 runs.** The
  stall detector `|N_Krylov| / D_visc` (`stall_flagged`, with the energy ledger attached) was flagged above a fixed
  1e-3 that was inferred, not measured: correct float32 runs read up to 2.0e-3 (tau/dt near 3e3), and in float64 it
  caught 114 of 200 planted tolerance stalls. The reading scales with the solve's tolerance (a drawn run's reading moves
  by a median 0.11 and 9.6 times for a tolerance a tenfold tighter and looser, in both dtypes), so the line is
  `STALL_PER_TOLERANCE` solve tolerances, 1000 in float64 and 500 in float32 (about twice the largest clean reading over
  the tolerance, 510 and 245, on 200 drawn runs at each of three tolerances), read at the tolerance the run was built
  with: a run that asks for a looser solve is judged against that solve, and the flag cannot see a tolerance that is wrong
  for the question. The ledger harness's stall flag is the same number at the default tolerance (1e-5, 5e-3), one source
  for both. The planted stalls are caught on 200 of 200 runs in both dtypes (a solve whose tolerance is 1e4 times the
  default; a CG stopped at 1e4 times the tolerance the node declares) with no clean run flagged; a solve stopped after two
  iterations is caught on 50 of 50 (float64) and 43 of 50 (float32): a stall that small is inside float32's clean
  readings. The line is validated for tau/dt up to 4.1e3, D/h 8 to 16, Re 1 to 200, and beyond that it is unvalidated.
  `mime.nodes.flow.fvm.STALL_LINE` is replaced by `STALL_PER_TOLERANCE` and `STALL_LINES`. No program's arrays move except
  `stall_flagged` where a reading lay between the old and the new line (measured, float32 and float64, bit for bit).
  Test-side: `FlagReport.unlined` is `without_line`.

- **`drag` 0.5.0's default MRS spacing steps one unit of the third figure below its bound, not ten units under a power
  of ten.** The rule rounds the largest spacing that gives the family's sample count to three figures and, where that
  lies past the bound, steps down one unit of the third figure, sized from the rounded value: where the bound is in the
  last half unit under a power of ten (a ball of radius 3.9785 mm: bound 0.99953 mm) the rounded value is the power, and the
  step went to 0.99 mm, not 0.999 mm, about 0.9 % finer spacing than needed, never coarser. 0.5.0 sizes the step
  from the bound, so the spacing is the bound rounded down to three figures, in the family's path and in the confined
  path alike (a closest approach over `CONFINED_GAP`). Drag 0.3.0 and 0.4.0 keep the released step (`step_from_boundary`),
  so released programs keep their bits (measured, float32 and float64: the resolved spacing and the resistance's bits of
  the ball, pill and FL-9 on both, and FL-9 confined on 0.4.0, equal to the tip's). Of 0.5.0's programs, only those whose
  bound lies in that band move (the 1 mm ball, the pill and FL-9 at their defaults do not).

- **The rft rung's resistance is the same bit for bit whichever kind of core a process started on** (MIME-ANO-125
  again; the rung is unreleased). R1's node summed `M^T f` over its 3n unknowns in one product (282 on
  `rft_free_space`'s helix), and with a wall took its block-diagonal local drag into `B M` and `B G` as dense products
  as long; XLA:CPU's Eigen blocks such a sum by the L1 data cache of the core it first ran on, so on a CPU with two
  sizes of core the registry walk's round trip of `rft_free_space` failed by where each process started. It sums
  `M^T f` by `force_sums` (pieces of at most `DEPTH` terms, added in order) and takes `B M` and `B G` point by point,
  three terms a sum; every product in its compiled step sums at most `DEPTH` terms
  (`tests/oracles/test_mrs_force_sums.py`, with a control that puts the whole product back). Its numbers move at
  rounding only: R6's oracle and the rung's golden outputs hold within their tolerances.
- **The energy ledger's running sums are pairs, and no longer drift with the steps** (MIME-ANO-149; the ledger is
  unreleased, so no version step). A plain float32 sum of a series of 1e5 terms drifted by tens of units of eps on the
  terms' summed magnitudes (45 on a mixed-sign series, 107 on a positive one, measured through the ledger's own update;
  the same in float64: 52 and 83). Each running sum of a signed series (`ledger_work`, `ledger_hosted`, `ledger_terms`,
  `ledger_unattributed`, `ledger_between`, `ledger_between_own`) is now a total and the part below its last place beside it
  (`<name>_low`), renormalised by an error-free sum every step: the drift at 1e5 steps is under 3e-7 units in float32
  and exact in float64, against a line of 0.012 units set by the pair's own rounding. `balance` and `losses` read the low
  parts; a reader of a field alone gets the sum rounded to the run's dtype. The error-free sum is shared with the
  rotating field's phase (`mime.core._error_free`). The step scale, a sum of magnitudes, stays plain. A program without
  the ledger is unchanged to the bit; a metered one gains the `_low` fields and its sums move by their rounding (the
  `energy_ledger` golden is regenerated; its pose and twist are unchanged).
- **A bundle changed in place in the cache, its size and modification time kept, is downloaded again and the good file
  served** (MIME-ANO-142, partially resolved). The cache's cheap check could not see a byte flipped in the middle of a
  file with both restored, so `based_on` and `pull` of a DOI served the changed bundle and refused it with advice that
  did not apply. Now, where the caller holds the repository's checksum of the file (a bundle's `md5:...`), the check
  reads the cached file once and compares its digest (`locate(ref, checksum=...)`, and `assets.retrieve`'s lookup by
  its `known_hash`), so the changed file takes the download; and a download that passed the repository's checksum is
  installed, in one rename, over a cached file MIME wrote whose full sha256 fails (a file a person placed is refused, as
  before), where before a cached file that passed the cheap check was never replaced. Measured on the pull tests' fake
  repository: the review's reproduction made one more bundle request (1 to 2) and served the good file. Cost, warm page
  cache: `locate` with no checksum 15 to 19 microseconds at any size; with the md5, 1.0 ms at 1 MB, 34 ms at 32 MB and
  1.08 s at 900 MB (a whole-file read). The checksum-free path is unchanged: an asset's file is judged by size and
  modification time, and a byte flipped in place there is found only by `mime assets check` or `fetch`; the asset
  cache's documentation says so.
  A bundle named by a bare `https://` address, which is downloaded each time, now also repairs a cached file of those
  bytes that was changed in place (the download is installed over it when MIME wrote it), where before the cached file
  was kept; and a cached bundle that cannot be read is refused as `asset.missing`, naming the file and the route
  (its permissions, or delete it and fetch again), not as a bare `PermissionError`.
- **The uniform rotating field turns by the integral of its frequency: `uniform_rotating` 0.2.0** (MIME-ANO-139). 0.1.0
  made the phase as the present frequency times the time, so a ramp turned the field at twice the commanded rate and a
  step jumped its angle. 0.2.0 holds the phase in turns as a pair of floating numbers, to which each step adds the frequency times the
  timestep as an exact product by an error-free sum, for any writer of the frequency; in float32 its angle stays within
  float32's spacing of an angle in a turn of the exact integral over a 1000 s run, where a product of frequency and time
  is off by up to 0.13 rad. For a constant frequency it is no longer 0.1.0's field to the last bit (0.1.0 carries the
  product's rounding), and the frozen register's scripts build the field at 0.1.0. `UniformRotating(...)` builds 0.2.0; 0.1.0 stays registered (`mime.fields.KEPT`) for saved folders and
  refuses any writer of its frequency but a constant drive (`command.held-constant`), naming 0.2.0.
- **The tube wall read's derivative has the right sign through samples tied in height or in azimuth under jit**
  (MIME-ANO-143). Jitted, the read's derivative along a rigid motion came out of the opposite sign for some pairs of
  force points at exactly equal height or azimuth (2.0 of the column's largest entry), in forward and in reverse mode,
  which pairs depending on how XLA fused the program; and in reverse mode with a cotangent for each entry
  (`jax.jacrev`) also where no two samples tied, through a sample's pair with itself, which is a tie in both (one of
  the test's eight pairs apart, and 10 of 12 random pairs apart in a reviewer's own harness, in both precisions). XLA computes
  the axial offset and the angle's sine inside each fusion that reads them, and near zero a difference of products is
  not the same number in every fusion, so the sign of an odd component and the derivative of the absolute value it is
  read at could see numbers of opposite sign. `pair_block` now holds the sine and the offset once (`held_once`, a
  custom derivative rule: the value itself in a program that takes no derivative, an optimization barrier's output in
  one that does) wherever a derivative is taken through them. A run takes no derivative, so its program is unchanged:
  measured, every field of the tube demo's 40 steps (the history scan and one-step chunks) and of the screw in the
  tube's 30 steps, in float32 and float64, is the program before the fix to the last bit, so the tube wall kind keeps
  its version 0.1.0; the primal of a program that does take a derivative is not bit for bit the plain program's (measured
  on the tube demo's step under `jax.jvp`: 4 to 7 of 27 state leaves differ in the last bits, see the entry's residual risk).
  `tests/oracles/test_tube_tie_derivative.py` holds the derivative through ties and self-pairs at
  seventeen pairs (jitted in both modes, eager, in both precisions, and under a barrier on the sine) and on the tube demo's 338 force
  points, against central differences, with the read before the fix as the control that fails.

- **A DOI run again does not download its bundle again** (`Experiment.based_on("10.5281/zenodo.N", name=...)`, a second
  `mime pull` of the same record). Measured on the fake repository: a re-run read the record and downloaded the bundle's
  bytes again before comparing its lock with the folder's. The asset cache now keeps a note, beside the stored bundle, of
  which sha256 the repository's checksum named (`md5-<digits>.checksum`), and `assets.retrieve` finds the bundle by it
  where the file is there and passes the cache's cheap check: the re-run reads the record's metadata (which names the
  checksum) and nothing else. A cleared cache, a missing or changed file, or a new version of the record (another
  checksum) downloads as before. A folder is still judged by its input hash against the bundle's, never by the
  checksum, so a folder that holds another experiment is refused as it was. The licence and download gates apply to the
  assets on a re-run as on the first run.
  A bare `https://` bundle address has no repository checksum and downloads again. A cached bundle changed in place with its
  size and modification time kept is served as the verified file, here as before this change (MIME-ANO-142; fixed for a
  bundle in the entry under Fixed).
- **In float32 every build of a screw or a cylinder in a process holds the first build's matrix, to the bit** (MIME-ANO-138).
  The first build in a process made the MRS node's held matrix from cap samples in float64; every later build made it
  from the same samples rounded to float32 (1e-7 of themselves), because the cap layout the process caches was handed
  to each build as the same arrays. Measured at MIME's level: `jnp.asarray` with x64 on returned float32 for the same
  array object the traced step had used under x64 off, and float64 for a copy of it (as did `jax.device_put` and
  `jax.clear_caches()`); the cached arrays themselves were unchanged. It was not reduced to a case without MIME, so it
  is not attributed to JAX in general. The drag's resistance differed between the first build and the rest by 1.1e-5
  of its first entry (about 90 float32 epsilons, from the samples' 2 to 3) from step 0, so in a script the process's
  first build, run and then saved, reported moved against the reference its own save wrote (`built.verify()` was
  clean). `cap_rings` now hands out a copy of the layout each call. Float64 is unchanged; `mime verify` was never
  affected (its reference and its rerun are both later builds, and `Experiment.load` builds once before the user's
  `.build()`).
- **A document of version 6 that declares a run's length reads.** The load refused `run.duration` in any document of
  a version below the current one, where version 5 added it; with the form at 7, that would have refused every
  version 6 folder that declares its run's length. It is refused only below version 5, as the key's message says.
- **The magnetic laws read the field where the magnet is: `magnetic.torque` 0.2.0 and `magnetic.force` 0.2.0.** In a
  field that varies in space, both read the field and its gradient at the body's `magnet_centre` turned by its
  orientation, the point a read now names fixed in the body (`Read(..., offset=...)`) and a provider that claims
  `Capability.BODY_POINTS` evaluates (`PointDipole` does). 0.1.0 read them at the centre of mass while the force acted
  at the centroid, so with the magnet off the centre of mass the pair was not conservative: a closed loop of the pose in
  a static field did net work, `m l pi G_xx` for a turn about x on a dipole's axis, 2.35 % of the loop's summed work at
  a 1 mm lever 50 mm from a 10 A m^2 dipole (MIME-ANO-137). The magnetic family's suite gains a closed-loop case: the
  laws together do no net work round closed loops of the pose (turns about drawn axes, a translation) in a drawn point
  dipole's field, within the loop's quadrature error plus its rounding floor; 0.1.0, built through its registered
  entries, fails it on every draw with a lever in float64 (but the geometry that leaves it exactly no work; in float32,
  on all but the few that are below its resolution). Through the build, a
  body spun in a dipole's field for whole turns at three steps keeps no part of its energy drift independent of the step
  on 0.2.0, and on 0.1.0 keeps that work a turn whatever the step (`tests/oracles/test_closed_loop_dynamic.py`). 0.1.0
  stays registered (`mime.interactions.KEPT`), so a folder saved with it loads and reproduces, and the frozen sim2real
  register keeps it; a script that writes `Magnetic.Torque()` or `Magnetic.Force()` gets 0.2.0. With the magnet at the
  centre of mass, or in a uniform field, the two compute the same: A0, A0 on its dynamic rungs and script A were
  measured the same to the last bit on 0.2.0 as on 0.1.0, in float32 and float64. A body with parts and several magnets
  among them, or none (its moment declared), is refused for the torque too in a field that varies in space, as it was
  for the force, unless its `magnet_centre` is declared (the refusal names both cures). The near-field check
  (`provider.reader-inside`) judges the point the law reads, the magnet's centroid for 0.2.0, as 0.1.0 keeps judging
  the centre of mass.
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
