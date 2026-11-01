---
describes: ["0.1.0"]
---

# `ellipsoid`

A solid ellipsoid given by its three semi-axes, along its part frame's x, y and z, and centred on its centroid: an
egg-shaped or flattened particle, a cell, or a smooth body whose three widths differ. As a body's part it gives the
body its volume and inertia and its surface samples, its one patch `surface`, for a solver that integrates over the
surface. Its region `solid` has no registered shape (an ellipsoid's distance has no closed form), so nothing evaluates
it at run time, and a body whose hull it is declares its extent where a law reads one.

**When not to use it.** For a body with a feature the ellipsoid smooths away (a thread, a fin, an edge), or where a
law needs the distance to its surface at run time.

**Its derivative** is exact: its volume, inertia, area, samples and reach are smooth functions of its semi-axes.

**Its sampling.** At a spacing, for a solver whose unknowns sit at the samples (the MRS rung), it samples its surface
quasi-uniformly: as many samples as its quadrature (Gauss-Legendre in the cosine of the polar angle about z, the
trapezoid rule round it) has at that spacing, chosen from that quadrature at a finer spacing by farthest points and
Lloyd's relaxation (`mime.scene.quasi_uniform`), each a quadrature sample with its cell's area as its weight. Its hull
query, its reach along `d`, is the length of `(a d_x, b d_y, c d_z)`.

**What its sampling costs.** As on the sphere, the MRS rung's drag at a given count is further from the closed form
(Oberbeck's and Edwardes's, as Kim and Karrila give it) on the quasi-uniform sampling than on its quadrature's grid,
in translation and in rotation, on the two triaxial ellipsoids measured: the grid misses by three fifths to three
quarters as much (the numbers are MIME-ANO-126's). The ellipsoid samples quasi-uniformly all the same, as every kind
does (design #195, a labelled default the maintainer may overturn); the separated quadrature (phase 7m-b) chooses the
defaults again.

**Versions.** 0.1.0, its first version, gained its quadrature, its sampling and its regions in phase 7m-a, at the
same version: before it the MRS rung refused it, so no result of it moved.
