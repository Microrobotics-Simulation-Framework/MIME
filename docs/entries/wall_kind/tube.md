---
describes: 0.1.0
---

# `tube`

A straight circular tube with a no-slip wall, the medium filling its lumen. The boundary-element method adds the tube's
image system to the free-space kernel: the flow a point force makes inside the tube, less the free Stokeslet
[@LironShahar1978]. It is read from one table of the unit tube's image system, which the tube's radius and the
fluid's viscosity scale to any tube, a file fetched into the assets cache (`mime assets fetch unit_tube_R0.997`) and
read when an experiment with the wall kind is planned; a file that is not there is refused by name, never downloaded.

**What it reads.** The tube's centre, axis and radius, from the cylinder region the medium fills.

**Where it reads.** Within the table's readable range: from the axis to the last radial knot of the file, and a few
radii apart along the axis (the file's reach). A body whose samples leave that range is refused: the plan refuses a
body longer than the table reaches along the axis, and a run that leaves the range publishes `wall_accepted` false. The
numbers, the read's accuracy and the table's gate are MIME-ANO-117's.

**When not to use it.** For interactions between bodies several radii apart along the axis: from there the read's
error exceeds the tube's whole Green's function, which is fine for a body's drag and not for a long-range
interaction.

**The gradient.** The derivative is continuous except where a sample crosses the tube's axis: the read is continuous
through the axis and its derivative across it is not (`tube.axis`, a kink).

**Versions.** The first.
