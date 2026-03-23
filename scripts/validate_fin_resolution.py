#!/usr/bin/env python3
"""Fin resolution sensitivity study for pre-T2.6 gate.

For each candidate LBM resolution, instantiates the d2.8 UMR mask at the
corresponding lattice scale (vessel diameter 9.4mm as domain) and reports:
- Fin circumferential arc length in lattice units (the thin dimension)
- Fin radial extent in lattice units
- Whether angular fin patches are distinguishable at each z-slice
- Total UMR solid node count and fraction

Key geometry note: The 6 fins (2 sets x 3) each span 2.03mm axially in a
4.1mm body — they OVERLAP in z (spacing = 0.68mm < length = 2.03mm).
The z-band disconnection metric is therefore invalid; we instead count
angular bands (distinct fin patches around the circumference) at each z.

Output: summary table printed and written to docs/validation/fin_resolution_study.md
"""

import os

# Prefer GPU if available; fall back to CPU
if "JAX_PLATFORMS" not in os.environ:
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import math
import time
import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX backend: {jax.default_backend()}", flush=True)

from mime.nodes.robot.helix_geometry import (
    create_umr_mask,
    create_cylinder_body_mask,
    create_discontinuous_fins_mask,
)


# ---------------------------------------------------------------------------
# Physical parameters (d2.8 UMR, all in mm)
# ---------------------------------------------------------------------------

VESSEL_DIAMETER = 9.4
UMR_BODY_RADIUS = 0.87
UMR_BODY_LENGTH = 4.1
UMR_CONE_LENGTH = 1.9
UMR_CONE_END_RADIUS = 0.255
UMR_FIN_OUTER_RADIUS = 1.42
UMR_FIN_LENGTH = 2.03
UMR_FIN_WIDTH = 0.55       # circumferential arc at body surface (code interpretation)
UMR_FIN_THICKNESS = 0.15   # radial overlap zone (code interpretation)
UMR_HELIX_PITCH = 8.0
UMR_TOTAL_LENGTH = 6.0

RESOLUTIONS = [64, 96, 128, 192]


def scale_to_lattice(N):
    """Compute dx and scale all geometry to lattice units."""
    dx = VESSEL_DIAMETER / N
    nz = max(N, int(math.ceil(2.0 * UMR_TOTAL_LENGTH / dx)))
    if nz % 2 != 0:
        nz += 1
    return {
        "nx": N, "ny": N, "nz": nz, "dx": dx,
        "body_radius": UMR_BODY_RADIUS / dx,
        "body_length": UMR_BODY_LENGTH / dx,
        "cone_length": UMR_CONE_LENGTH / dx,
        "cone_end_radius": UMR_CONE_END_RADIUS / dx,
        "fin_outer_radius": UMR_FIN_OUTER_RADIUS / dx,
        "fin_length": UMR_FIN_LENGTH / dx,
        "fin_width": UMR_FIN_WIDTH / dx,
        "fin_thickness": UMR_FIN_THICKNESS / dx,
        "helix_pitch": UMR_HELIX_PITCH / dx,
    }


def count_angular_bands(mask_2d, cx, cy, r_check, n_sample=720):
    """Count distinct angular bands of solid at radius r_check in a 2D slice."""
    nx, ny = mask_2d.shape
    solid_ring = []
    for i in range(n_sample):
        angle = 2.0 * math.pi * i / n_sample
        xi = int(round(cx + r_check * math.cos(angle)))
        yi = int(round(cy + r_check * math.sin(angle)))
        if 0 <= xi < nx and 0 <= yi < ny:
            solid_ring.append(bool(mask_2d[xi, yi]))
        else:
            solid_ring.append(False)

    arr = np.array(solid_ring)
    if not np.any(arr):
        return 0, []

    # Count transitions (solid→empty) wrapping around
    transitions = 0
    band_sizes = []
    in_band = arr[0]
    count = 1 if in_band else 0
    for i in range(1, len(arr)):
        if arr[i] and not arr[i - 1]:
            transitions += 1
            count = 1
        elif arr[i] and arr[i - 1]:
            count += 1
        elif not arr[i] and arr[i - 1]:
            band_sizes.append(count)
            count = 0
    if arr[-1]:
        # Check wrap-around: if first element is also solid, merge
        if arr[0] and transitions > 0:
            band_sizes[0] += count
        else:
            band_sizes.append(count)
            if not arr[0]:
                transitions += 1
    # Bands = number of rising edges
    n_bands = len(band_sizes)
    # Convert band sizes from angular samples to degrees
    band_degrees = [b * 360.0 / n_sample for b in band_sizes]
    return n_bands, band_degrees


def measure_radial_extent(fins_only, cx, cy, z_idx, body_r, outer_r, dx):
    """Measure radial extent of fins at a given z-slice, scanning angles.

    Note: this measurement underestimates the true radial extent by ~0.5-1 lu
    per side due to voxel discretisation (int(round(...)) sampling) and because
    only a single z-slice is sampled. The actual outer radius is guaranteed by
    the geometry code's `r_perp < fin_outer_radius` bound. Use circumferential
    arc (angular band method) as the primary resolution metric.
    """
    nx, ny = fins_only.shape[0], fins_only.shape[1]
    extents = []
    for angle_deg in range(0, 360, 2):
        angle = math.radians(angle_deg)
        r_min = None
        r_max = None
        for r_step in np.linspace(max(body_r - 3, 0), min(outer_r + 3, nx / 2), 200):
            xi = int(round(cx + r_step * math.cos(angle)))
            yi = int(round(cy + r_step * math.sin(angle)))
            if 0 <= xi < nx and 0 <= yi < ny:
                if bool(fins_only[xi, yi, z_idx]):
                    if r_min is None:
                        r_min = r_step
                    r_max = r_step
        if r_min is not None:
            extents.append(r_max - r_min)
    return extents


def analyze_resolution(N):
    """Run full analysis at a given resolution."""
    params = scale_to_lattice(N)
    nx, ny, nz = params["nx"], params["ny"], params["nz"]
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    cx, cy, cz = center

    print(f"\n--- Resolution {nx}x{ny}x{nz} (dx = {params['dx']:.4f} mm) ---", flush=True)

    t0 = time.perf_counter()

    body_mask = create_cylinder_body_mask(
        nx, ny, nz,
        body_radius=params["body_radius"],
        body_length=params["body_length"],
        cone_length=params["cone_length"],
        cone_end_radius=params["cone_end_radius"],
        center=center,
    )

    fins_mask = create_discontinuous_fins_mask(
        nx, ny, nz,
        body_radius=params["body_radius"],
        fin_outer_radius=params["fin_outer_radius"],
        fin_length=params["fin_length"],
        fin_width=params["fin_width"],
        fin_thickness=params["fin_thickness"],
        helix_pitch=params["helix_pitch"],
        center=center,
        body_length=params["body_length"],
    )

    umr_mask = body_mask | fins_mask
    fins_only = fins_mask & ~body_mask

    body_count = int(jnp.sum(body_mask))
    fins_only_count = int(jnp.sum(fins_only))
    total_count = int(jnp.sum(umr_mask))
    total_nodes = nx * ny * nz
    solid_fraction = total_count / total_nodes

    # Angular band analysis: sample multiple z-slices through the fin region
    # Body z range: cz - body_length/2 to cz + body_length/2
    z_body_start = int(cz - params["body_length"] / 2)
    z_body_end = int(cz + params["body_length"] / 2)
    z_slices = np.linspace(z_body_start + 2, z_body_end - 2, 5, dtype=int)

    r_check = (params["body_radius"] + params["fin_outer_radius"]) / 2.0
    fins_only_np = np.array(fins_only)

    max_bands = 0
    all_band_info = []
    for z_idx in z_slices:
        n_bands, band_deg = count_angular_bands(
            fins_only_np[:, :, z_idx], cx, cy, r_check,
        )
        max_bands = max(max_bands, n_bands)
        all_band_info.append((z_idx, n_bands, band_deg))

    # Radial extent measurement
    z_mid = int(cz)
    radial_extents = measure_radial_extent(
        fins_only_np, cx, cy, z_mid,
        params["body_radius"], params["fin_outer_radius"], params["dx"],
    )
    mean_radial_lu = np.mean(radial_extents) if radial_extents else 0.0
    mean_radial_mm = mean_radial_lu * params["dx"]

    # Circumferential arc: from angular band sizes
    all_arcs_lu = []
    for _, _, band_deg in all_band_info:
        for deg in band_deg:
            arc_lu = r_check * math.radians(deg)
            all_arcs_lu.append(arc_lu)
    mean_arc_lu = np.mean(all_arcs_lu) if all_arcs_lu else 0.0
    mean_arc_mm = mean_arc_lu * params["dx"]

    elapsed = time.perf_counter() - t0

    result = {
        "N": N, "nz": nz, "dx_mm": params["dx"],
        "body_radius_lu": params["body_radius"],
        "fin_width_lu": params["fin_width"],
        "fin_outer_radius_lu": params["fin_outer_radius"],
        "body_nodes": body_count,
        "fin_only_nodes": fins_only_count,
        "total_nodes": total_count,
        "solid_fraction": solid_fraction,
        "max_angular_bands": max_bands,
        "mean_arc_lu": mean_arc_lu,
        "mean_arc_mm": mean_arc_mm,
        "mean_radial_lu": mean_radial_lu,
        "mean_radial_mm": mean_radial_mm,
        "fins_absent": fins_only_count == 0,
        "elapsed_s": elapsed,
    }

    print(f"  Body radius: {params['body_radius']:.2f} lu", flush=True)
    print(f"  Fin outer radius: {params['fin_outer_radius']:.2f} lu", flush=True)
    print(f"  Body nodes: {body_count}", flush=True)
    print(f"  Fin-only nodes: {fins_only_count}", flush=True)
    print(f"  Solid fraction: {solid_fraction:.4%}", flush=True)
    print(f"  Max angular bands (across z-slices): {max_bands}", flush=True)
    for z_idx, nb, bd in all_band_info:
        bd_str = ", ".join(f"{d:.0f}deg" for d in bd) if bd else "none"
        print(f"    z={z_idx}: {nb} bands [{bd_str}]", flush=True)
    print(f"  Mean circumferential arc: {mean_arc_lu:.1f} lu = {mean_arc_mm:.2f} mm", flush=True)
    print(f"  Mean radial extent: {mean_radial_lu:.1f} lu = {mean_radial_mm:.2f} mm", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)

    return result


def write_report(results, path):
    """Write markdown report."""
    lines = [
        "# Fin Resolution Sensitivity Study",
        "",
        "Pre-T2.6 gate check: UMR fin resolution at candidate LBM grid spacings.",
        "",
        "## Physical geometry",
        "",
        f"- Vessel diameter (domain): {VESSEL_DIAMETER} mm",
        f"- Body radius: {UMR_BODY_RADIUS} mm, length: {UMR_BODY_LENGTH} mm",
        f"- Fin outer radius: {UMR_FIN_OUTER_RADIUS} mm",
        f"- Fin length: {UMR_FIN_LENGTH} mm, width (circ. arc): {UMR_FIN_WIDTH} mm, thickness (radial overlap): {UMR_FIN_THICKNESS} mm",
        f"- Helix pitch: {UMR_HELIX_PITCH} mm (MIME-ANO-002: assumed)",
        "",
        "## Key geometry insight",
        "",
        "The 6 fins (2 sets x 3) each span 2.03mm axially in a 4.1mm body.",
        f"Fin spacing = {UMR_BODY_LENGTH/6:.2f}mm < fin length = {UMR_FIN_LENGTH}mm,",
        "so fins overlap axially. The correct distinguishability metric is",
        "angular bands per z-cross-section (not z-bands).",
        "",
        "The thinnest free-standing fin dimension in the code is the circumferential",
        f"arc length = fin_width = {UMR_FIN_WIDTH}mm at the body surface.",
        f"The `fin_thickness` parameter ({UMR_FIN_THICKNESS}mm) controls a radial overlap",
        "zone with the body, not a free-standing thin feature.",
        "",
        "## Resolution comparison",
        "",
        "| Resolution | dx (mm) | Circ. arc (lu) | Radial (lu) | Angular bands | Fin nodes | Solid % |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['N']}x{r['N']}x{r['nz']} "
            f"| {r['dx_mm']:.4f} "
            f"| {r['mean_arc_lu']:.1f} "
            f"| {r['mean_radial_lu']:.1f} "
            f"| {r['max_angular_bands']} "
            f"| {r['fin_only_nodes']} "
            f"| {r['solid_fraction']:.3%} |"
        )

    # Minimum viable resolution
    viable = [r for r in results
              if not r["fins_absent"] and r["max_angular_bands"] >= 2 and r["mean_arc_lu"] >= 2.0]
    if viable:
        min_v = min(viable, key=lambda r: r["N"])
        lines.extend([
            "",
            f"**Minimum viable resolution**: {min_v['N']}^3 "
            f"(dx = {min_v['dx_mm']:.4f} mm, "
            f"circ. arc = {min_v['mean_arc_lu']:.1f} lu, "
            f"radial = {min_v['mean_radial_lu']:.1f} lu)",
        ])
    else:
        lines.extend(["", "**WARNING**: No tested resolution produced viable fin resolution."])

    lines.extend([
        "",
        "## Parameter interpretation note",
        "",
        "The code uses `fin_width` (0.55mm) as circumferential arc and `fin_thickness`",
        "(0.15mm) as a radial overlap with the body. The paper's \"fin thickness\" likely",
        "refers to the blade thickness (circumferential), which would make the fin thinner",
        "than the code implements. If the interpretation is revised so that the circumferential",
        "extent = 0.15mm instead of 0.55mm, the resolution requirements increase significantly",
        "(0.15mm / dx at 128^3 = 2.0 lu — marginal). This should be resolved before T2.6.",
        "",
    ])

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"\nReport written to: {path}")


if __name__ == "__main__":
    results = []
    for N in RESOLUTIONS:
        results.append(analyze_resolution(N))

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    hdr = f"{'Resolution':<16} {'dx(mm)':<10} {'Arc(lu)':<10} {'Radial(lu)':<12} {'Ang.bands':<12} {'Fin nodes':<12}"
    print(hdr)
    print("-" * 90)
    for r in results:
        print(f"{r['N']}x{r['N']}x{r['nz']:<8} "
              f"{r['dx_mm']:<10.4f} "
              f"{r['mean_arc_lu']:<10.1f} "
              f"{r['mean_radial_lu']:<12.1f} "
              f"{r['max_angular_bands']:<12} "
              f"{r['fin_only_nodes']:<12}")

    report_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "validation", "umr_deboer2025", "fin_resolution_study.md",
    )
    write_report(results, report_path)
