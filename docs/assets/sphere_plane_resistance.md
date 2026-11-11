# The sphere-plane resistance table

**What it is.** The exact resistance of a rigid sphere near a plane wall in Stokes flow, as MIME computed it from the
bispherical solutions: five functions of the gap over the radius, `d`, for a unit sphere in a medium of unit viscosity,
the force and torque the sphere exerts on the fluid per unit motion about its centre, the wall's normal +z: `zz` (the
normal translation), `xx` (a translation along the wall), `ryry` (a rotation about an axis along it), `x_ry` (the force
along x per rotation about y, which is the torque about y per translation along x) and `rzrz` (the spin about the
normal). Packaged as `asset("sphere_plane_resistance")`, read by the near-contact law's sphere form
(`mime.nodes.hydro.near_contact`), which the mrs rung of `drag` 0.5.0 takes for a sphere near a wall
(`near_contact`).

**The file.** `src/mime/data/sphere_plane_resistance.json`: 97 knots uniform in `ln d` from `1e-3` to `10`, and the 96
midpoints between them, which every read checks (`mime.data.asset_bytes`), its sha256:

`5c6ed8ff9fd2f8e7e2b15027d1c8f742a650906702b73d87adb0d1f7aba7e0e1`

**How it was made.** `scripts/sphere_plane_table.py`, MIME's own, its parts built on GitHub's runners and merged; its
derivation record is `src/mime/data/derivations/sphere_plane_resistance.json`. The normal translation is Brenner's
(1961) series and the spin Jeffery's (1915), summed in closed form. The motions along the wall are Dean and O'Neill's
(1963) and O'Neill's (1964) problems, solved in their representation of a Stokes flow (`u = x p / (2 mu) + w`, the
pressure and the Cartesian components of `w` harmonic, each a bispherical series) by collocation of the no-slip
condition and the divergence on both surfaces, by least squares, rather than by the papers' recurrences; force and
torque are the traction integrated over the sphere.

**Its terms.** MIME's, under MIME's licence: computed by MIME, with no upstream data. Its tier is open: it ships in the
wheel.

**Its checks**, recorded by the build and in the derivation record: the same solver's normal and spin against Brenner's
and Jeffery's series within 2.1e-12; the coupling computed both ways (the force per rotation and the torque per
translation) within 5.3e-10; the no-slip condition at nodes not collocated within 3.3e-10; each entry's change with the
series' length and the nodes a quarter longer, and the traction's nodes doubled, within 3.8e-10 at `d` 1e-3 and 0.01.
Against an independent method of fundamental solutions (graded towards the contact, sharing no code with MIME,
`tests/differential/graded_mfs.py`), every entry within that method's own resolution from 0.3 to 0.005 radii
(`tests/oracles/test_near_contact.py`).

**How it is read.** Each entry as its singular terms, exactly (the normal's `1/d` and its fifth of the logarithm, Cox
and Brenner's; Goldman, Cox and Brenner's three logarithms), plus a remainder interpolated by a cubic spline in `ln d`:
within a radius of the wall each entry within 2.7e-7 of itself at the midpoints. Below `d` 1e-3 the remainder is
continued as `R0 + c1 d ln d + c2 d`, matched to the spline's value and slope at the edge (C1 there) and its value at the
next knot; past 10 radii the remainder is held at its value there.

**Its accuracy, by use.** The table is exact to its checks; what the near-contact law makes of it is gated through the
drag's node (`tests/oracles/test_near_contact.py`): against a plane, a sphere's resistance is the coarse MRS at the
hand-over gap plus the table's exact increment, so its error is the coarse MRS's own there.
