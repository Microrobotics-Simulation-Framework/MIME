"""M3, the wall correction's accuracy at the floor (sim-to-real REGISTER section 6): the wall table's measured read
error propagated through the closed BEM at 7j-a's reference point ("A": closed surface, 0.178 mm spacing, blob 0.5,
1/4" tube, margin 0.1 mm, 10 Hz), as a perturbation of the assembled wall correction ``G`` (``corrected.matrix``) that
varies from entry to entry, not as a scaling. No measured speed enters; 7j-a's scripts are imported, never modified.

The reported quantity is the **phase-mean axial speed** over 7j-a's eight phases (``op.speeds(...)["axial"]``).

Two error models, 50 draws each, fixed seeds, both keeping ``G`` symmetric:

1. ``noise``: ``dG = A o S``. ``S`` is symmetric, its upper triangle (diagonal included) drawn independently and
   uniformly on [-1, 1] and mirrored; ``A_ij = 0.0023 x`` the largest absolute entry of the 3 x 3 block ``(i, j)`` of
   ``G`` (the gate's measured maximum read error, relative to a block's largest entry). One ``S`` per draw and per
   phase (the pairs differ from phase to phase).
2. ``smooth``: ``dG_ij = G_ij (phi(x_i) + phi(x_j)) / 2`` (every entry of a pair's 3 x 3 block). ``phi`` is a smooth
   random field of the samples' tube coordinates (rho, theta, z), one field per draw read at every phase's samples,
   Gaussian-correlated, built from random Fourier features (the theta wavenumbers integer, so the field is periodic).
   Its correlation lengths are the table's knot spacing along each axis (see :func:`lengths`); its amplitude is
   0.0023 at rho >= 0.157 R and 0.023 at rho < 0.157 R (the table's first radial interval, with its larger divergence
   error), blended by a cubic smoothstep over the interval's last 2 R/30 (so that the samples at rho >= 0.157 R carry the
   outer amplitude alone), and set per draw so that the largest |phi| over each region's samples at phase 0 is that
   amplitude. ``smooth_local`` (an extra arm, not in the register) takes the
   radial correlation length as the local knot interval (the coordinate is the knot index), 0.157 R in the first
   interval.

Stages (each a function of a seed range, so a chunk is the same as the whole): ``reference`` (nominal, the floor, the
x1.02 control, the pole scan, the construction's checks), ``noise``, ``smooth``, ``smooth_local``. ``analyze`` reads
their JSON. Run by hand or on Modal (``m3_modal.py``), never by CI:

    JAX_ENABLE_X64=1 python scripts/sim2real/m3_wall_read.py --stage reference --out DIR
    JAX_ENABLE_X64=1 python scripts/sim2real/m3_wall_read.py --dry          # a coarse mesh, no speeds printed
    python scripts/sim2real/m3_wall_read.py --analyze DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import bem  # noqa: E402
import operating_point as op  # noqa: E402

HERE = Path(__file__).resolve().parent
#: The table's measured read error (relative to a block's largest entry) and its first radial interval's divergence
#: error, REGISTER section 6; the interval's outer edge in the tube's radius.
READ_ERROR = 0.0023
FIRST_INTERVAL_ERROR = 0.023
INNER_EDGE = 0.157
BLEND_HALF_WIDTH = 0.997 / 30.0
DRAWS = 50
SEEDS = {"noise": 20261101, "smooth": 20261102, "smooth_local": 20261103}
FEATURES = 4000
CONTROL_SCALE = 1.02
#: 7j-a's scaling band (+-0.33%, closed) and the register's "near the pole" line, ten times it.
SCALING_BAND = 0.0033
NEAR_POLE = 10.0 * SCALING_BAND
POLE_SCALES = tuple(round(1.040 + 0.002 * k, 3) for k in range(16))
FLOOR_ULPS = (-2, -1, 1, 2, 3)
TABLE_FILE = "unit_tube_R0.997.npz"
#: The fine scan (the coordinator's addition after the coarse scan found the smallest eigenvalue negative at its first
#: point): the correction's scale from x1.000 to x1.040 in steps of 0.0025, at every phase.
SCAN_FINE_SCALES = tuple(round(1.0 + 0.0025 * k, 4) for k in range(17))
STAGES = ("reference", "noise", "smooth", "smooth_local", "scan_fine")


# ---- the wall correction, assembled and solved in two halves (a copy of ``op.bem_resistance``) -------------------------
def assemble_wall(built: Any, pose: Any, leaves: dict[str, Any]) -> dict[str, Any]:
    """The first half of ``op.bem_resistance``, unchanged: the samples at ``pose``, the wall's correction ``G`` at
    them (unscaled), the blob and the viscosity; and the samples in the tube's frame over its radius (for the field)."""
    import jax.numpy as jnp

    from mime.core import quaternion
    from mime.nodes.hydro.bem import blob_squared, wall_correction_named

    node = built.graph.get_node(op.DRAG_NODE)
    p = node._step_constants(leaves)  # noqa: SLF001
    held = node.samples(p)
    points = pose[:3] + jnp.matmul(held.points, quaternion.rotation_matrix(pose[3:]).T, precision="highest")
    wall = wall_correction_named(node.constants()["wall_kind"], node.constants()["wall_kind_version"])
    corrected = wall.correction(points, {name: p[f"wall_{name}"] for name in wall.parameters}, p["viscosity"])
    frame = jnp.asarray(p["wall_frame"], dtype=points.dtype).reshape(3, 3)
    local = jnp.matmul(points - jnp.asarray(p["wall_centre"], dtype=points.dtype), frame.T,
                       precision="highest") / jnp.asarray(p["wall_radius"], dtype=points.dtype)
    return {"points": points, "eps2": blob_squared(held.weights, p["blob"]), "viscosity": p["viscosity"],
            "centre": pose[:3], "G": corrected.matrix, "local": local}


def solve_resistance(parts: dict[str, Any], correction: Any) -> Any:
    """The second half of ``op.bem_resistance``: ``R = M^T (A + G_used)^-1 M``, with ``correction`` the wall's
    correction as used."""
    from mime.nodes.hydro.bem import resistance

    return resistance(parts["points"], parts["eps2"], parts["viscosity"], parts["centre"], correction)


def bem_resistance_perturbed(built: Any, pose: Any, leaves: dict[str, Any], *, correction_scale: float = 1.0,
                             perturb: Any = None) -> Any:
    """``op.bem_resistance`` with the assembled correction perturbed: ``perturb(G, parts)`` returns ``dG``, added to
    ``correction_scale * G``. With no perturbation it is ``op.bem_resistance`` (a control in ``reference``)."""
    parts = assemble_wall(built, pose, leaves)
    used = correction_scale * parts["G"]
    if perturb is not None:
        used = used + perturb(used, parts)
    return solve_resistance(parts, used)


# ---- the two error models' perturbations of G --------------------------------------------------------------------------
def noise_delta(G: Any, key: Any, relative: float = READ_ERROR) -> Any:
    """Model 1: ``A o S`` (module docstring); symmetric by construction when ``G`` is."""
    import jax
    import jax.numpy as jnp

    n3 = G.shape[0]
    n = n3 // 3
    drawn = jax.random.uniform(key, (n3, n3), dtype=G.dtype, minval=-1.0, maxval=1.0)
    upper = jnp.triu(drawn)
    symmetric = upper + jnp.triu(drawn, 1).T
    largest = jnp.max(jnp.abs(G.reshape(n, 3, n, 3)), axis=(1, 3))
    largest = jnp.maximum(largest, largest.T)  # block (i, j) and (j, i) hold the same entries up to G's own rounding
    return (symmetric.reshape(n, 3, n, 3) * (relative * largest)[:, None, :, None]).reshape(n3, n3)


def smooth_delta(G: Any, phi: Any) -> Any:
    """Model 2: ``G_ij (phi_i + phi_j) / 2`` on every entry of pair ``(i, j)``'s block."""
    n = phi.shape[0]
    factor = 0.5 * (phi[:, None] + phi[None, :])
    return (G.reshape(n, 3, n, 3) * factor[:, None, :, None]).reshape(G.shape)


# ---- model 2's field ---------------------------------------------------------------------------------------------------
def table_axes(path: str | Path | None = None) -> dict[str, Any]:
    """The wall table's knots along each axis, in the tube's radius, and the correlation lengths built from them: the
    mean spacing along each (range over intervals), as the register's R/30 for the radius; and the radial knots
    themselves (for ``smooth_local``)."""
    if path is None:
        from mime import assets

        path = assets.locate(assets.reference("unit_tube_R0.997"))
    with np.load(path) as held:
        radius = float(held["R_cyl"])
        rho, dphi, dz = (np.asarray(held[k], dtype=np.float64) for k in ("rho_grid", "dphi_grid", "dz_grid"))
    rho = rho / radius
    dz = dz / radius
    return {"R_cyl": radius, "rho_knots": rho, "ell_rho": float((rho[-1] - rho[0]) / (len(rho) - 1)),
            "ell_theta": float((dphi[-1] - dphi[0]) / (len(dphi) - 1)), "ell_z": float((dz[-1] - dz[0]) / (len(dz) - 1)),
            "counts": {"rho": len(rho), "dphi": len(dphi), "dz": len(dz)},
            "first_rho_interval": float(rho[1] - rho[0]), "last_rho_interval": float(rho[-1] - rho[-2]),
            "dphi_first_last_interval": [float(dphi[1] - dphi[0]), float(dphi[-1] - dphi[-2])],
            "dz_first_last_interval": [float((dz[1] - dz[0])), float(dz[-1] - dz[-2])]}


def tube_coordinates(local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rho, theta, z) of samples in the tube's frame over its radius."""
    return np.hypot(local[:, 0], local[:, 1]), np.arctan2(local[:, 1], local[:, 0]), local[:, 2]


class Field:
    """One draw's unit-variance smooth field: ``sqrt(2/K) sum cos(w_rho u_rho + m theta + w_z u_z + b)`` with
    ``u_rho`` the radius over its correlation length (or, ``local``, the radial knot index by piecewise-linear
    interpolation: the correlation length is then the local knot interval), ``u_z = z / ell_z``, ``w`` standard normal
    and ``m`` the integer nearest a normal of standard deviation ``1 / ell_theta``. Its covariance is the Gaussian
    ``exp(-d^2 / 2)`` in each of these coordinates."""

    def __init__(self, seed: int, draw: int, axes: dict[str, Any], local: bool) -> None:
        rng = np.random.default_rng([seed, draw])
        self.w_rho = rng.standard_normal(FEATURES)
        self.m = np.rint(rng.standard_normal(FEATURES) / axes["ell_theta"])
        self.w_z = rng.standard_normal(FEATURES)
        self.b = rng.uniform(0.0, 2.0 * math.pi, FEATURES)
        self.axes, self.local = axes, local

    def u_rho(self, rho: np.ndarray) -> np.ndarray:
        knots = self.axes["rho_knots"]
        if self.local:
            return np.interp(rho, knots, np.arange(len(knots), dtype=np.float64))
        return rho / self.axes["ell_rho"]

    def __call__(self, rho: np.ndarray, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
        phase = (self.u_rho(rho)[:, None] * self.w_rho + theta[:, None] * self.m + (z / self.axes["ell_z"])[:, None]
                 * self.w_z + self.b)
        return math.sqrt(2.0 / FEATURES) * np.cos(phase).sum(axis=1)


def blend_weight(rho: np.ndarray) -> np.ndarray:
    """1 well inside the first radial interval, 0 from its outer edge ``INNER_EDGE`` on: a cubic smoothstep over the
    interval's last 2 R/30 (so that the samples at ``rho >= 0.157 R`` carry only the outer amplitude)."""
    t = np.clip((rho - (INNER_EDGE - 2.0 * BLEND_HALF_WIDTH)) / (2.0 * BLEND_HALF_WIDTH), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


class Draw:
    """One draw of model 2: its field, scaled by the amplitudes ``0.0023`` outside and ``0.023`` inside, so that the
    largest ``|phi|`` over each region's samples at the reference coordinates (phase 0) is that amplitude (inside, the
    blend's weight enters the largest, so the realised maximum is that amplitude to within the outer term's share)."""

    def __init__(self, seed: int, draw: int, axes: dict[str, Any], local: bool, reference: tuple[np.ndarray, ...]) -> None:
        self.field = Field(seed, draw, axes, local)
        rho, theta, z = reference
        unit = self.field(rho, theta, z)
        outer = rho >= INNER_EDGE
        self.largest = {"outer": float(np.abs(unit[outer]).max()),
                        "inner": float((np.abs(unit) * blend_weight(rho))[~outer].max())}
        self.scale = {"outer": READ_ERROR / self.largest["outer"], "inner": FIRST_INTERVAL_ERROR / self.largest["inner"]}

    def phi(self, coordinates: tuple[np.ndarray, ...]) -> np.ndarray:
        rho = coordinates[0]
        inside = blend_weight(rho)
        return self.field(*coordinates) * (inside * self.scale["inner"] + (1.0 - inside) * self.scale["outer"])


# ---- the rung ----------------------------------------------------------------------------------------------------------
class Study:
    """A closed-surface rung at 7j-a's reference point (or a coarse one for the dry run), its assembly and its solves
    compiled once, and the samples' tube coordinates at each phase."""

    def __init__(self, spacing_mm: float = op.SPACING_MM, phases: tuple[float, ...] = op.PHASES) -> None:
        import jax
        import jax.numpy as jnp

        self.rung = bem.Rung("closed", spacing_mm=spacing_mm)
        self.phases = phases
        self.spacing_mm = spacing_mm
        self.drag = self.rung.leaves[op.DRAG_NODE]
        self.solves = 0
        built = self.rung.built
        self._assemble = jax.jit(lambda pose, drag: assemble_wall(built, pose, drag))
        self._solve = jax.jit(lambda parts, G: solve_resistance(parts, G))
        self._scaled = jax.jit(lambda parts, scale: solve_resistance(parts, scale * parts["G"]))
        self._noise = jax.jit(lambda parts, key: solve_resistance(parts, parts["G"] + noise_delta(parts["G"], key)))
        self._smooth = jax.jit(lambda parts, phi: solve_resistance(parts, parts["G"] + smooth_delta(parts["G"], phi)))
        self._noise_parts = jax.jit(lambda G, key: noise_delta(G, key))
        self._smooth_parts = jax.jit(lambda G, phi: smooth_delta(G, phi))
        self._jnp = jnp

    def parts(self, phase: float, offset: float | None = None) -> dict[str, Any]:
        return self._assemble(self.rung.pose(phase, offset), self.drag)

    def coordinates(self, phase: float) -> tuple[np.ndarray, ...]:
        return tube_coordinates(np.asarray(self.parts(phase)["local"], dtype=np.float64))

    def read(self, found: Any) -> np.ndarray:
        import jax

        self.solves += 1
        return np.asarray(jax.block_until_ready(found), dtype=np.float64)

    @staticmethod
    def axial(resistance: np.ndarray) -> float:
        return float(-resistance[0, 3] * 2.0 * math.pi * op.FREQUENCY_HZ / resistance[0, 0])

    def row(self, resistance: np.ndarray) -> dict[str, float]:
        return {"axial": self.axial(resistance), "min_eigenvalue_nd": bem.smallest_eigenvalue(resistance),
                "finite": bool(np.all(np.isfinite(resistance)))}


def _summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    speeds = [row["axial"] for row in rows]
    return {"per_phase": rows, "mean_axial": float(np.mean(speeds)),
            "min_eigenvalue_nd": float(min(row["min_eigenvalue_nd"] for row in rows)),
            "finite": all(row["finite"] for row in rows)}


# ---- stages ------------------------------------------------------------------------------------------------------------
def stage_reference(study: Study) -> dict[str, Any]:
    """The nominal, the floor, the control, the pole scan and the construction's checks."""
    started = time.perf_counter()
    out: dict[str, Any] = {"stage": "reference", "spacing_mm": study.spacing_mm, "samples": study.rung.samples,
                           "offset_m": study.rung.offset, "scaling_band": SCALING_BAND}
    nominal_rows: list[dict[str, float]] = []
    nominal_matrices = []
    for phase in study.phases:
        parts = study.parts(phase)
        resistance = study.read(study._solve(parts, parts["G"]))  # noqa: SLF001
        nominal_matrices.append(resistance)
        nominal_rows.append(study.row(resistance))
    out["nominal"] = _summary(nominal_rows)

    # The floor: the nominal under 1- to 3-ulp changes of the pose's offset (five of them), each at all phases.
    ulp = float(np.spacing(study.rung.offset))
    floor = []
    for k in FLOOR_ULPS:
        rows = []
        for phase in study.phases:
            parts = study.parts(phase, study.rung.offset + k * ulp)
            rows.append(study.row(study.read(study._solve(parts, parts["G"]))))  # noqa: SLF001
        floor.append({"ulps": k, **_summary(rows)})
    out["floor"] = {"ulp_m": ulp, "changes": floor}

    # The control: the correction scaled by x1.02, every phase.
    rows = []
    for phase in study.phases:
        parts = study.parts(phase)
        rows.append(study.row(study.read(study._scaled(parts, CONTROL_SCALE))))  # noqa: SLF001
    out["control"] = {"scale": CONTROL_SCALE, **_summary(rows)}

    # The pole scan, phase 0.
    parts = study.parts(study.phases[0])
    scan = []
    for scale in POLE_SCALES:
        resistance = study.read(study._scaled(parts, scale))  # noqa: SLF001
        scan.append({"scale": scale, **study.row(resistance)})
    out["pole_scan"] = scan
    out["checks"] = construction_checks(study, nominal_matrices)
    out["environment"] = op.environment()
    out["solves"] = study.solves
    out["seconds"] = time.perf_counter() - started
    return out


def stage_scan_fine(study: Study) -> dict[str, Any]:
    """Per phase and scale: the axial speed and the smallest nondimensional eigenvalue; the onset of indefiniteness at
    each phase (the first scale on the grid at which the eigenvalue is not positive, with the last positive one) and its
    minimum over the phases."""
    started = time.perf_counter()
    phases = []
    for phase in study.phases:
        parts = study.parts(phase)
        scan = [{"scale": scale, **study.row(study.read(study._scaled(parts, scale)))}  # noqa: SLF001
                for scale in SCAN_FINE_SCALES]
        first = next((i for i, row in enumerate(scan) if row["min_eigenvalue_nd"] <= 0.0), None)
        phases.append({"phase": phase, "scan": scan, "onset_scale": None if first is None else scan[first]["scale"],
                       "last_positive_scale": None if first in (None, 0) else scan[first - 1]["scale"]})
    onsets = [p["onset_scale"] for p in phases if p["onset_scale"] is not None]
    return {"stage": "scan_fine", "scales": list(SCAN_FINE_SCALES), "phases": phases,
            "onset_scale_min_over_phases": min(onsets) if onsets else None, "environment": op.environment(),
            "solves": study.solves, "seconds": time.perf_counter() - started}


def construction_checks(study: Study, nominal_matrices: list[np.ndarray]) -> dict[str, Any]:
    """The perturbation's construction against its own definition, and the copy of ``op.bem_resistance`` against it
    (equal to rounding with no perturbation, and unchanged by a zero perturbation)."""
    import jax
    import jax.numpy as jnp

    checks: dict[str, Any] = {}
    phase = study.phases[0]
    pose = study.rung.pose(phase)
    parts = study.parts(phase)
    G = np.asarray(parts["G"])
    checks["G_symmetry_relative"] = float(np.abs(G - G.T).max() / np.abs(G).max())
    own = study.rung.resistance(pose, own=True)
    copy = np.asarray(bem_resistance_perturbed(study.rung.built, pose, study.drag), dtype=np.float64)
    zero = np.asarray(bem_resistance_perturbed(study.rung.built, pose, study.drag,
                                               perturb=lambda g, p: jnp.zeros_like(g)), dtype=np.float64)
    checks["copy_against_op_bem_resistance_relative"] = float(np.abs(copy - own).max() / np.abs(own).max())
    checks["zero_perturbation_against_op_relative"] = float(np.abs(zero - own).max() / np.abs(own).max())
    checks["staged_against_nominal_relative"] = float(np.abs(nominal_matrices[0] - own).max() / np.abs(own).max())
    # Model 1's perturbation against its definition.
    key = jax.random.fold_in(jax.random.PRNGKey(SEEDS["noise"]), 0)
    delta = np.asarray(study._noise_parts(parts["G"], key))  # noqa: SLF001
    n = G.shape[0] // 3
    largest = np.abs(G.reshape(n, 3, n, 3)).max(axis=(1, 3))
    allowed = READ_ERROR * np.repeat(np.repeat(largest, 3, axis=0), 3, axis=1)
    checks["noise_symmetric_exactly"] = bool(np.array_equal(delta, delta.T))
    checks["noise_within_allowed"] = bool(np.all(np.abs(delta) <= allowed * (1.0 + 1e-12)))
    checks["noise_largest_over_allowed"] = float((np.abs(delta) / np.where(allowed > 0, allowed, 1.0)).max())
    checks["noise_mean_over_allowed"] = float((np.abs(delta) / np.where(allowed > 0, allowed, 1.0)).mean())
    checks["noise_relative_to_G_max"] = float(np.abs(delta).max() / np.abs(G).max())
    # Model 2's field and perturbation against their definitions.
    axes = table_axes()
    coords = study.coordinates(phase)
    draw = Draw(SEEDS["smooth"], 0, axes, False, coords)
    phi = draw.phi(coords)
    delta2 = np.asarray(study._smooth_parts(parts["G"], jnp.asarray(phi)))  # noqa: SLF001
    checks["smooth_asymmetry_relative_to_largest"] = float(np.abs(delta2 - delta2.T).max() / np.abs(delta2).max())
    checks["G_asymmetry_over_G_max"] = float(np.abs(G - G.T).max() / np.abs(G).max())
    checks["smooth_max_phi_outer"] = float(np.abs(phi[coords[0] >= INNER_EDGE]).max())
    checks["smooth_max_phi_inner"] = float(np.abs(phi[coords[0] < INNER_EDGE]).max())
    checks["samples_inner_outer"] = [int((coords[0] < INNER_EDGE).sum()), int((coords[0] >= INNER_EDGE).sum())]
    checks["rho_range_over_R"] = [float(coords[0].min()), float(coords[0].max())]
    checks["z_range_over_R"] = [float(coords[2].min()), float(coords[2].max())]
    pair = (3, 11)
    block = G[3 * pair[0]:3 * pair[0] + 3, 3 * pair[1]:3 * pair[1] + 3]
    expected = block * 0.5 * (phi[pair[0]] + phi[pair[1]])
    checks["smooth_block_matches_definition"] = float(np.abs(delta2[3 * pair[0]:3 * pair[0] + 3,
                                                                    3 * pair[1]:3 * pair[1] + 3] - expected).max())
    checks["field_correlation"] = field_correlation(axes)
    checks["table_axes"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in axes.items()}
    return checks


def field_correlation(axes: dict[str, Any], draws: int = 400, seed: int = 7) -> dict[str, float]:
    """The field's empirical correlation at one correlation length along each axis (expected ``exp(-1/2) = 0.607``),
    at a point pair a length apart, over ``draws`` fields; and its variance (expected 1)."""
    rng = np.random.default_rng(seed)
    base = (np.full(1, 0.5), np.full(1, 1.0), np.full(1, 0.3))
    steps = {"rho": (axes["ell_rho"], 0.0, 0.0), "theta": (0.0, axes["ell_theta"], 0.0), "z": (0.0, 0.0, axes["ell_z"])}
    found: dict[str, list[float]] = {name: [] for name in steps}
    found["variance"] = []
    for k in range(draws):
        field = Field(int(rng.integers(1 << 30)), k, axes, False)
        a = field(*base)[0]
        found["variance"].append(a * a)
        for name, (dr, dt, dz) in steps.items():
            found[name].append(a * field(base[0] + dr, base[1] + dt, base[2] + dz)[0])
    return {name: float(np.mean(values)) for name, values in found.items()}


def stage_noise(study: Study, draws: range) -> dict[str, Any]:
    import jax

    started = time.perf_counter()
    nominal_rows: list[dict[str, float]] = []
    results: dict[int, list[dict[str, float]]] = {d: [] for d in draws}
    base = jax.random.PRNGKey(SEEDS["noise"])
    for index, phase in enumerate(study.phases):
        parts = study.parts(phase)
        nominal_rows.append(study.row(study.read(study._solve(parts, parts["G"]))))  # noqa: SLF001
        for d in draws:
            key = jax.random.fold_in(base, d * len(study.phases) + index)
            results[d].append(study.row(study.read(study._noise(parts, key))))  # noqa: SLF001
    return {"stage": "noise", "seed": SEEDS["noise"], "relative_error": READ_ERROR, "nominal": _summary(nominal_rows),
            "draws": [{"draw": d, **_summary(results[d])} for d in draws], "environment": op.environment(),
            "solves": study.solves, "seconds": time.perf_counter() - started}


def stage_smooth(study: Study, draws: range, local: bool) -> dict[str, Any]:
    import jax.numpy as jnp

    name = "smooth_local" if local else "smooth"
    started = time.perf_counter()
    axes = table_axes()
    coordinates = [study.coordinates(phase) for phase in study.phases]
    fields = [Draw(SEEDS[name], d, axes, local, coordinates[0]) for d in draws]
    results: dict[int, list[dict[str, Any]]] = {d: [] for d in draws}
    nominal_rows: list[dict[str, float]] = []
    for index, phase in enumerate(study.phases):
        parts = study.parts(phase)
        nominal_rows.append(study.row(study.read(study._solve(parts, parts["G"]))))  # noqa: SLF001
        for d, draw in zip(draws, fields, strict=True):
            phi = draw.phi(coordinates[index])
            row = study.row(study.read(study._smooth(parts, jnp.asarray(phi))))  # noqa: SLF001
            outer = coordinates[index][0] >= INNER_EDGE
            row["max_abs_phi_outer"] = float(np.abs(phi[outer]).max())
            row["max_abs_phi_inner"] = float(np.abs(phi[~outer]).max()) if (~outer).any() else None
            results[d].append(row)
    return {"stage": name, "seed": SEEDS[name], "ell": {k: axes[k] for k in ("ell_rho", "ell_theta", "ell_z")},
            "local_radial": local, "nominal": _summary(nominal_rows),
            "draws": [{"draw": d, "scale": draw.scale, "largest_unit_field": draw.largest, **_summary(results[d])}
                      for d, draw in zip(draws, fields, strict=True)],
            "environment": op.environment(), "solves": study.solves, "seconds": time.perf_counter() - started}


def run_stage(stage: str, draws: int = DRAWS, spacing_mm: float = op.SPACING_MM,
              phases: tuple[float, ...] = op.PHASES) -> dict[str, Any]:
    op.require_float64()
    study = Study(spacing_mm, phases)
    if stage == "reference":
        return stage_reference(study)
    if stage == "scan_fine":
        return stage_scan_fine(study)
    if stage == "noise":
        return stage_noise(study, range(draws))
    if stage in ("smooth", "smooth_local"):
        return stage_smooth(study, range(draws), local=stage == "smooth_local")
    raise ValueError(f"no stage {stage!r}")


# ---- analysis ----------------------------------------------------------------------------------------------------------
def band(model: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """A model's band: the half-width (largest |change| of the phase-mean speed over the draws, relative to the
    nominal), its standard uncertainty by B1 (half-width / sqrt 3), the standard deviation, per-draw smallest
    eigenvalues and the pole flags, against the floor."""
    nominal = model["nominal"]["mean_axial"]
    changes = np.array([(d["mean_axial"] - nominal) / nominal for d in model["draws"]])
    per_phase = np.array([[(row["axial"] - n["axial"]) / n["axial"] for row, n in zip(d["per_phase"],
                                                                                   model["nominal"]["per_phase"], strict=True)]
                          for d in model["draws"]])
    floors = np.array([(c["mean_axial"] - reference["nominal"]["mean_axial"]) / reference["nominal"]["mean_axial"]
                       for c in reference["floor"]["changes"]])
    floor = float(np.abs(floors).max())
    half = float(np.abs(changes).max())
    eig = np.array([d["min_eigenvalue_nd"] for d in model["draws"]])
    return {"stage": model["stage"], "draws": len(changes), "half_width": half, "u_b1": half / math.sqrt(3.0),
            "std": float(changes.std(ddof=1)), "mean_change": float(changes.mean()),
            "floor_half_width": floor, "at_the_floor": bool(half <= floor),
            "ratio_to_floor": half / floor if floor > 0 else None,
            "largest_per_phase_change": float(np.abs(per_phase).max()),
            "nominal_vs_reference_nominal": float(model["nominal"]["mean_axial"] / reference["nominal"]["mean_axial"] - 1.0),
            "min_eigenvalue_nd": {"smallest": float(eig.min()), "median": float(np.median(eig)),
                                  "negative_draws": int((eig <= 0).sum()),
                                  "nominal_smallest": model["nominal"]["min_eigenvalue_nd"]},
            "near_pole": {"threshold": NEAR_POLE, "phase_mean_draws_above": int((np.abs(changes) > NEAR_POLE).sum()),
                          "per_phase_values_above": int((np.abs(per_phase) > NEAR_POLE).sum())},
            "all_finite": all(d["finite"] for d in model["draws"]),
            "control": None}


def analyze(directory: Path) -> dict[str, Any]:
    reference = json.loads((directory / "m3_reference.json").read_text())
    nominal = reference["nominal"]["mean_axial"]
    control = (reference["control"]["mean_axial"] - nominal) / nominal
    out: dict[str, Any] = {"nominal_mean_axial_m_s": nominal, "control_scale": CONTROL_SCALE, "control_change": control,
                           "floor_changes": [{"ulps": c["ulps"], "change": (c["mean_axial"] - nominal) / nominal,
                                              "largest_per_phase": max(abs(r["axial"] - n["axial"]) / abs(n["axial"])
                                                                       for r, n in zip(c["per_phase"],
                                                                                       reference["nominal"]["per_phase"],
                                                                                       strict=True))}
                                             for c in reference["floor"]["changes"]], "models": {}}
    for name in ("noise", "smooth", "smooth_local"):
        path = directory / f"m3_{name}.json"
        if path.exists():
            found = band(json.loads(path.read_text()), reference)
            found["control"] = {"change": control, "caught": bool(abs(control) > found["half_width"]),
                                "ratio_to_half_width": abs(control) / found["half_width"] if found["half_width"] > 0 else None}
            out["models"][name] = found
    fine = directory / "m3_scan_fine.json"
    if fine.exists():
        held = json.loads(fine.read_text())
        out["scan_fine"] = {"onset_scale_min_over_phases": held["onset_scale_min_over_phases"],
                            "per_phase": [{"phase": p["phase"], "onset_scale": p["onset_scale"],
                                           "last_positive_scale": p["last_positive_scale"]} for p in held["phases"]]}
    scan = reference["pole_scan"]
    out["pole_scan"] = [{"scale": s["scale"], "axial_mm_s": s["axial"] * 1e3, "min_eigenvalue_nd": s["min_eigenvalue_nd"]}
                        for s in scan]
    signs = [s["min_eigenvalue_nd"] > 0 for s in scan]
    out["pole_scan_eigenvalue_sign_changes_between"] = [[scan[i]["scale"], scan[i + 1]["scale"]]
                                                        for i in range(len(scan) - 1) if signs[i] != signs[i + 1]]
    speeds = [s["axial"] for s in scan]
    out["pole_scan_speed_sign_changes_between"] = [[scan[i]["scale"], scan[i + 1]["scale"]]
                                                   for i in range(len(scan) - 1) if speeds[i] * speeds[i + 1] < 0]
    out["pole_scan_largest_abs_speed_at_scale"] = scan[int(np.argmax(np.abs(speeds)))]["scale"]
    return out


# ---- command line ------------------------------------------------------------------------------------------------------
def dry_run() -> None:
    """The code paths at a coarse mesh (0.9 mm), three phases and two draws, nothing about a speed printed: shapes,
    finiteness, the construction's checks and the stages' agreement with one another."""
    op.require_float64()
    phases = op.PHASES[:3]
    study = Study(0.9, phases)
    print(f"dry run: {study.rung.samples} samples, {len(phases)} phases, device {op.environment()['devices']}")
    reference = stage_reference(study)
    checks = reference["checks"]
    shown = {k: v for k, v in checks.items() if k not in ("table_axes",)}
    print(json.dumps(shown, indent=1, sort_keys=True))
    print("table axes:", {k: (v if not isinstance(v, list) or len(v) < 4 else f"{len(v)} knots")
                          for k, v in checks["table_axes"].items()})
    print("reference: nominal finite", reference["nominal"]["finite"], "| floor changes", len(reference["floor"]["changes"]),
          "| control finite", reference["control"]["finite"], "| pole scan", len(reference["pole_scan"]), "scales, finite",
          all(s["finite"] for s in reference["pole_scan"]), "| solves", reference["solves"])
    for stage in ("noise", "smooth", "smooth_local"):
        found = stage_noise(study, range(2)) if stage == "noise" else stage_smooth(study, range(2), stage == "smooth_local")
        print(f"{stage}: {len(found['draws'])} draws, finite {all(d['finite'] for d in found['draws'])}, "
              f"perturbed differs from nominal {all(d['mean_axial'] != found['nominal']['mean_axial'] for d in found['draws'])}, "
              f"solves {found['solves']}")
    # The analysis path on the dry run's own JSON (speeds never printed).
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "m3_reference.json").write_text(json.dumps(reference, default=_plain))
        for stage in ("noise", "smooth", "smooth_local"):
            found = stage_noise(study, range(2)) if stage == "noise" else stage_smooth(study, range(2), stage == "smooth_local")
            (folder / f"m3_{stage}.json").write_text(json.dumps(found, default=_plain))
        summary = analyze(folder)
        print("analysis ran; models:", sorted(summary["models"]), "| keys per model:", sorted(next(iter(summary["models"].values()))))


def _plain(value: Any) -> Any:
    return np.asarray(value).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--out", default=None)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--analyze", default=None)
    args = parser.parse_args()
    if args.dry:
        dry_run()
    elif args.analyze:
        op.write_json(Path(args.analyze) / "m3_summary.json", analyze(Path(args.analyze)))
    elif args.stage and args.out:
        found = run_stage(args.stage, args.draws)
        op.write_json(Path(args.out) / f"m3_{args.stage}.json", found)
    else:
        parser.error("give --dry, --analyze DIR, or --stage and --out")


if __name__ == "__main__":
    main()
