# `mime.interactions`

**The physical laws between things**, one module per family.

## Families

- **`magnetic`**: torque, and the full wrench.
- **`hydrodynamic`**: drag, through its rungs. The rungs are a learned surrogate, a penalty term inside the flow
  solver, and later Stokes boundary elements.
- **`gravity`**: weight and buoyancy.
- **`contact`**: by projection (inside the body's own update) or by penalty (as loads into each side). Velocity rules
  include a friction cone.
- **`acoustic`**: radiation force, from a2.

## The contract

- An interaction is **one physical law among named things**. It reads declared interfaces and delivers into a declared
  target of a thing's dynamics.
- It never touches another interaction's nodes.
- A law that needs another law's output reads it from the thing that received it.
- **The model choice is written in the script**: `Drag(rung="surrogate")`, or the alias `Drag.Surrogate(...)`, which
  saves the same way. Nothing is picked automatically. A choice outside its validated range is reported.
- Transfers between a body's points and a grid go through `mappings/`.

## Status

Also:
- **`magnetic.py`**: `Magnetic.Torque()` (`magnetic.torque` 0.2.0), `tau = m x B` on the body's permanent moment
  turned by its orientation, from a field read at the magnet's centroid (a uniform field as before); it reads the pose at the start of the step, a one-step lag of
  `omega dt` in phase that lag accounting shows;
- **`gravity.py`**: `Gravity()` (`gravity` 0.1.0), weight minus buoyancy `(m - rho_f V) g` with gravity from the
  declared up axis, and the couple when the body's centre of buoyancy is off its centre of mass (reading the pose
  only then);
- **`contact.py`**: `Contact(rung="projection" | "penalty")` (`contact` 0.1.0), `Contact.Projection(...)` its alias;
  projection binds a fixed structure's surface (`surface=tube.inner_surface`) and goes to the body's dynamics at
  construction, with its margin, the body's extent and the velocity rule `remove_outward`; penalty is reserved, and
  projection against a moving thing is refused as reserved.

Each passes its family's suite (`mime.testing.MAGNETIC`, `GRAVITY`, `CONTACT`).

Also:
- **`hydrodynamic.py`**: `Drag(rung="surrogate")` (`drag` 0.1.0), `Drag.Surrogate(...)` its alias, which saves
  identically. The Cholesky MLP's resistance for a de Jongh screw body in a straight tube, from
  `weights=mime.data.asset("dejongh_mlp_v2")`, with `lubrication=Lubrication(epsilon=...)` blended in as one model
  when given (the leading terms of the sphere-wall asymptotes of Cox & Brenner and Goldman, Cox & Brenner, weighted
  by `exp(-gap / epsilon)`, the 0.2 line's blend; the gap floored at 1 nm). It binds `body=` and `medium=`, reads the body's pose at the start of the step, and the tube's
  axis, a point on it and its radius from the region the medium fills, through the shape protocol: a medium filling
  no region, or a region with no axis, is refused. It reads the medium's `viscosity` and the body's
  `helix_wavenumber`, `length` and `radius`, declared in the experiment for now (`mime.materials` is the one place
  that reads them). It delivers `resistance` (an operator the body solves with implicitly) and `background_load`
  (zero in a medium at rest). A body or tube outside the surrogate's training data in `nu`, `kappa` (up to 0.655, the
  largest trained value) or `L / R` (2.65 to 6.24) is refused at plan time, naming the entry, the value, the range and where the range comes from; recording an acceptance instead
  is for a1's profiles. The offset from the axis keeps the 0.2 line's clamp, reported as `clamp_fired` in the node's
  state. `drag` 0.2.0 moves the surrogate's resistance from the end of the body's axis, about which it
  was trained, to the centre of mass, `L / 2` along the body's x axis taken as the helix's (MIME-ANO-112, resolved);
  0.1.0 delivered it about the end. Its accuracy against the MRS it was checked against is MIME-ANO-114's: up to 16%
  in swim speed between 0.2 and 0.32 of the tube's radius. `Lubrication` is a group of settings (`mime.registry.SettingGroup`),
  which the saved form writes by its group name.

It passes the hydrodynamic family's suite (`mime.testing.HYDRODYNAMIC`), whose analytic oracle and second member, a
Stokes sphere, is registered by MIME's tests only; no public `Drag.Stokes` is built yet.

**`Magnetic.Force`** (`magnetic.force` 0.2.0): the force of a field's gradient on a body's
permanent moment, `F = G^T m`, read at its magnet's centroid (`magnet_centre`) and acting there, so with its torque. It is in the magnetic family, whose sign rule holds both its laws, and its pair law, the equal and
opposite force between two point dipoles, is tested on the law alone. Its reaction on the source is delivered where the source is mounted on a segment that takes loads (below).

**the magnetic laws' reactions**: where the field's source is a part mounted on a segment that takes
loads (a dynamic one), `Magnetic.Torque` and `Magnetic.Force` each deliver the reaction of their wrench on the body to
the frame the source is mounted on, equal and opposite and moved to that frame's origin (`SourceReactionNode`); the
two together are what the body's dipole exerts on a point dipole source. Elsewhere they build what they built.
**`Magnetic.Torque`** reads a field that varies in space at the body's magnet's centroid, by a second support row; a
uniform field is read as before.

**the magnetic laws read the field where the magnet is** (`magnetic.torque` 0.2.0, `magnetic.force` 0.2.0): a field that
varies in space, and its gradient, are read at the body's `magnet_centre` turned by its orientation from its position, a
point fixed in the body that the read names (`Read.offset`) and a provider that claims `Capability.BODY_POINTS`
evaluates (`PointDipole` does), so that the two together are the derivative of a point dipole's energy there and do no
net work round a closed loop of the pose. The magnetic family's closed-loop case holds them to it. Their 0.1.0, which
read it at the centre of mass (MIME-ANO-137), stay registered (`KEPT`), and the frozen sim2real register keeps the
torque's. A body with parts and several magnets among them, or none (its moment declared), is refused for either in a
field that varies in space unless its `magnet_centre` is declared (the refusal says so); several magnets each need their own point, later. The near-field
check (`provider.reader-inside`) judges the point a law reads: the magnet's centroid, turned by the orientation the body
starts in, for 0.2.0, and the centre of mass for the kept 0.1.0.

Releases are explained in the [package manifest](../README.md#status).

The drag reads the body's shape from its hull's kind (`ctx.geometry(of="body")`), not from declared
values: the kind must be the one its surrogate was trained on (`screw`, at the surrogate's modulation and
starts, along the body's x and centred on its origin), refused otherwise (`drag.body-kind`, `drag.outside-training`).
Its node evaluates the tube by the region's registered shape (the body's offset, the direction across the tube and the
lubrication's gap come from the cylinder's distance and normal) and takes the body's outer radius from the kind's
radial reach, so it holds no formula of either's. What it computes is unchanged: script A's golden outputs and its
frozen reference hold.

The surrogate rung reads its weights through the evaluator its surrogate's record names
(`mime.surrogates.Surrogate`), which also states the surrogate's accuracy per use and its provenance; what it computes
is unchanged. The mrs rung's plan names the shipped table its wall kind reads, and a saved experiment's lock records it.

**`Contact.Projection`** (`contact` 0.5.0, which runs the hold's sweeps and its search's Newton steps as XLA loops where
0.4.0, kept registered for its folders, unrolled them; 0.2.0 and 0.3.0, in no release, are not kept) holds the body's
hull, its orientation read, each point kept `margin` from the surface, by impulses through the body's implicit solve
with the rotation coupled in (projected Gauss-Seidel over one point per group of the hull's samples at `spacing`) and a
projection of the pose after the step
in two passes, the second choosing each group's point again where the first left the body and holding it beside the
first's (MIME-ANO-136),
inelastic and frictionless, in the body's own update (`RigidInertialNode`). The point each group holds is the body's
own surface's point that the search from the deepest of its starts converges to: its nearest sample and each next
nearest apart from those taken, as many as the hull's shape can hold local deepest points that its samples' gaps do
not order (a screw's crest passes and its rims; counted when the experiment is built); a search that does not
converge within its bound holds the group's nearest sample (MIME-ANO-134, MIME-ANO-135). The point is
found by the hull kind's **support query**
(`Geometry.support`: a direction in, the farthest point and its derivative out, reading no wall; a closed form for the
sphere, the ellipsoid, the cylinder and the capsule, Newton's steps on its own surface for the screw), so the touching
point moves continuously as the body turns and the contact takes no spin from a body of revolution (a point the
support's steps leave shallower than the sample they started from, or the samples' shortfall where the search did
not converge and held its nearest sample, is the state's `contact_short`, flagged `<body>.support_short`); a kind with no support (a lab's mesh) is held by its samples, which act as a friction set by
the spacing, and raises `<body>.held_by_samples`. The solve is laid out over contact pairs (a body's hull against a wall; two bodies' hulls,
by their supports, as GJK finds them, are the layout's next, `mime.nodes.body._contact`). Against a surface whose
shape declares its region convex (the tube's lumen) each group's candidates are its own convex hull's vertices, fixed
at build from the hull's settings as built; the projection after the step turns the angular velocity with the pose, so
it changes no kinetic energy; the forces are scaled to do no work at the held twist; and the solve's residual is `<body>`'s state's `contact_residual`,
flagged `unconverged` above `CONVERGED`. A spacing whose samples fall farther from the body than the margin, or that
takes more than `JUDGED_SAMPLES` to judge, is refused (`confinement.spacing-short`); left unwritten, the spacing is the
margin, no finer than `SPACING_FLOOR` of the hull's size. A turned body that touches at two points takes the
couple that holds it, and a screw that turns broadside in a narrow tube is held at the bore's limit, where 0.1.0
held its centre by its extent and let it through the wall (a body held by 0.1.0 whose hull leaves the region is
flagged after the run by the containment watch, in exploratory and research: [the results note](../../../docs/results.md)). A body with no hull is held as a sphere of its extent, as
before; a body whose hull no point reaches the margin steps as it did, to the last bit. 0.1.0 is kept registered
(`mime.interactions.KEPT`) for folders saved with it and for the experiments whose comparisons were declared against
it, script A's and A0's reproductions; 0.4.0 is kept for the folders saved with it, its program the one they were
derived from (0.5.0's loops round otherwise: a held body's float32 velocity moves by about a millionth of its size, its rows' forces
by up to two parts in a thousand of their scale and its net force by under a thousandth of the momentum rate, float64
by a hundred-billionth or less: `tests/oracles/test_contact_loops.py`). **A contact handles admissibility only**: it never adds near-wall lubrication,
which a drag that includes it declares (`Interaction.includes_lubrication`: the surrogate rung with `lubrication=`),
and the plan refuses two laws on one body that include it (`lubrication.twice`). Friction at the contact is left to
the near-contact physics' design; its seam is the sweeps' rows.
