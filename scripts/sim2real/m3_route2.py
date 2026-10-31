"""M3 route 2, stage 1: the wall table's error E = G_table - G_exact at operating point A's own sample pairs, measured on a
stratified subset (plans/sim2real/M3_ROUTE2_SCOPE.md, option (a), staged). Stages 2 and 3 (every pair, propagated through
the solve) are not here.

**What it computes.** ``G_table`` is the tube wall kind's read of the shipped table ``unit_tube_R0.997`` (the cubic read,
folded by the pair's parity, blended with its reciprocal read as ``mime.nodes.hydro.bem._tube`` blends it). ``G_exact`` is
the Fourier-Bessel image itself, the series of ``scripts/tube_table/build.py``, evaluated at the pair (source, target)
instead of on a tensor grid: the wall side once per source radius (``build.wall_side``), the target side
(``build.target_terms``) once per target, and a per-pair contraction over modes and wavenumbers in place of the grid's
two matrix products. ``build.py`` and ``full.py`` are imported and never edited (their hashes are in the table's
derivation record).

**The truncation.** The table's own truncation (2500 modes, 16000 nodes) is far beyond what A's pairs need (A's samples
reach 0.968 of the radius, the table 0.997). Each pair is evaluated with the builder's per-source rule (``H = (1 - rho_s)
+ (1 - rho_max)``, modes ``10 / H``, nodes ``64 / H``, ``k_max = 12 / H``, floors of ``full.py``) at A's ``rho_max``, scaled
by 3.0 (twice the table's own 1.5), **the source being the sample nearer the axis**. The scale is a convergence choice
made before any E was computed: at 1.5 a pair of near-wall neighbours (source at 0.96 of the radius) differed from a
three-times larger truncation by 4e-7, the random pairs by 1e-8; at 3.0 by 1e-10. Controls 3 and 4 measure what the
choice costs.

**Frames.** Everything is in the unit tube's own frame (axis z, lengths over the tube's radius, viscosity 1), where a
sample is ``(rho, phi, z)``; the world's ``corrected.matrix`` is ``F^T G F / (R mu)`` with ``F`` the tube frame's rows
(``wall_frame``). A block's entries are those of the same 3 x 3 up to a signed permutation, so a block's largest entry is
the same in both.

**The metrics.** ``|E|`` of a pair is the block's largest absolute miss over the exact block's largest entry, in two
forms: ``blend`` (the table's blended read, the one the solve uses, against the exact block, both Cartesian: the form
M3's error models use, ``A_ij`` = a fraction of the block's largest entry) and ``gate`` (the table's unblended read with
the pair's target and source as the builder has them, rows in the source's frame, as ``full.py::read_errors`` measures
it: the form the 0.002 is stated in).

    taskset -c N nice -n 10 env JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu MIME_ASSETS_DIR=... python m3_route2.py --controls --out DIR
    ... python m3_route2.py --stage1 --out DIR
    ... python m3_route2.py --pilot

Single-threaded by construction: every thread pool the numerical libraries have is set to one before they load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "tube_table"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
from scipy import special, stats  # noqa: E402

import build  # noqa: E402  (scripts/tube_table/build.py: imported, never edited)
import full  # noqa: E402  (scripts/tube_table/full.py)

#: A's samples reach this radius over the tube's at the eight phases (the scope's probe), the readable radius of the
#: per-source truncation rule.
RHO_MAX_A = 0.9683
INNER_EDGE, OUTER_EDGE = 0.157, 0.9
#: The base rule's scale (twice the table's own factor, 1.5: see the module docstring) and the finer rule's multiples of it (full.py's ``finer`` multiples).
BASE_FACTOR = 3.0
FINER_MULTIPLES = (1.25, 2.0, 1.5)
SEED = 20261031
#: The controls' tolerances, declared here before any of them ran.
TOLERANCE = {"kernel_vs_tests": 1e-9, "kernel_vs_builder_stored": 1e-9, "knots": 1e-6, "plant_detected": 1e-2,
             "reciprocity_common_truncation": 1e-9, "plumbing": 1e-9, "gate": (0.0015, 0.0025),
             "truncation_ratio": 10.0, "nodes": 1e-11, "weights": 1e-10, "rounding": 1e-12}


# ---- the truncation -----------------------------------------------------------------------------------------------
def pair_truncation(source: float, multiples: tuple[float, float, float] = (1.0, 1.0, 1.0),
                    factor: float = BASE_FACTOR, readable: float = RHO_MAX_A) -> build.Truncation:
    """The builder's per-source rule at ``readable`` (``full.truncation_for`` with A's ``rho_max`` for the table's
    ``R_MAX``), each of modes, nodes and ``k_max`` times ``factor`` times its multiple, with ``full.py``'s floors."""
    h = (1.0 - source) + (1.0 - readable)
    rule = full.RULE
    modes = max(int(math.ceil(factor * multiples[0] * rule["modes"] / h)), rule["floor_modes"])
    nodes = max(int(math.ceil(factor * multiples[1] * rule["wavenumbers"] / h)), rule["floor_wavenumbers"])
    k_max = max(factor * multiples[2] * rule["reach"] / h, rule["floor_k_max"])
    angles = 2 * int(math.ceil((modes + 25.0 / max(1.0 - source, 1e-3)) / 2.0))
    return build.Truncation(modes, nodes, k_max, max(angles, 2 * modes + 2))


def finer_truncation(source: float) -> build.Truncation:
    return pair_truncation(source, FINER_MULTIPLES)


def table_truncation() -> build.Truncation:
    """The shipped table's own single truncation (2500 modes, 16000 nodes, k_max 3000, 10834 wall points)."""
    return full.truncation_for(full.R_MAX, factor=1.5)


def wavenumbers(truncation: build.Truncation) -> tuple[np.ndarray, np.ndarray]:
    """``build.wavenumber_nodes`` with scipy's Gauss-Legendre rule for the nodes: numpy's ``leggauss`` is a dense
    eigenproblem (``O(count^3)``, minutes at the table's 16000) and A's every source radius has a count of its own.
    Control 0 holds the two equal."""
    t, weight = special.roots_legendre(truncation.wavenumbers)
    t_max = math.sqrt(truncation.k_max)
    t, weight = 0.5 * t_max * (t + 1.0), 0.5 * t_max * weight
    return t ** 2, weight * 2.0 * t


# ---- the pairs kernel -------------------------------------------------------------------------------------------------
def exact_group(rho_s: float, rho_t: np.ndarray, dphi: np.ndarray, dz: np.ndarray, truncation: build.Truncation,
                budget: float = 2e6) -> np.ndarray:
    """``(T, 3, 3)``: the image of a source at ``(rho_s, 0, 0)`` at each of ``T`` targets ``(rho_t, dphi, dz)``, unit
    tube, rows the velocity's cylindrical components at the target, columns the force's at the source (x along the
    source's direction from the axis, y across it, z along the axis): ``build.image_grid``'s blocks, pair by pair (the
    target's angle and offset are its own, signed, so no parity is folded here). One wall side per wavenumber chunk,
    shared by the targets."""
    rho_t = np.maximum(np.asarray(rho_t, dtype=np.float64), 0.0)
    dphi, dz = np.asarray(dphi, dtype=np.float64), np.asarray(dz, dtype=np.float64)
    count, top = len(rho_t), truncation.modes
    k_all, w_all = wavenumbers(truncation)
    size = max(1, min(int(budget // (count * (top + 1))), int(2e6 // truncation.angles), len(k_all)))
    mode = np.arange(top + 1, dtype=np.float64)[:, None]
    cos_n, sin_n = np.cos(mode * dphi[None, :]), np.sin(mode * dphi[None, :])
    total = np.zeros((count, 9))
    for start in range(0, len(k_all), size):
        piece = slice(start, min(start + size, len(k_all)))
        side = build.wall_side(rho_s, k_all[piece], w_all[piece], truncation)
        x = rho_t[:, None] * side.k[None, :]
        log_target, ratios = build.ladder_numpy(x, build.ratio_start(top + 2, x), top + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = build.target_terms(np, rho_t, side.k, side.coefficients, side.log_wall, log_target[:top + 1],
                                       ratios[:top + 1], ratios[1:top + 2], top)
        axial = side.k[:, None] * dz[None, :]
        cos_z, sin_z = np.cos(axial) * side.weight[:, None], np.sin(axial) * side.weight[:, None]
        for entry, u in enumerate(terms):
            row, col = divmod(entry, 3)
            by_angle = sin_n if build.SIN_IN_ANGLE[row, col] else cos_n
            by_axis = sin_z if build.SIN_ALONG[row, col] else cos_z
            across = np.einsum("ntk,nt->tk", u, by_angle)
            total[:, entry] += np.einsum("tk,kt->t", across, by_axis)
    return total.reshape(count, 3, 3)


def turn(angle: np.ndarray) -> np.ndarray:
    """``(P, 3, 3)``: Cartesian components to cylindrical ones at each angle (``build._turn``, vectorised)."""
    c, s = np.cos(angle), np.sin(angle)
    out = np.zeros((len(angle), 3, 3))
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1], out[:, 2, 2] = c, s, -s, c, 1.0
    return out


def cartesian_block(block: np.ndarray, phi_t: np.ndarray, phi_s: np.ndarray) -> np.ndarray:
    """A builder block (rows cylindrical at the target, columns at the source) in the tube frame's Cartesian axes."""
    return np.einsum("pba,pbc,pcd->pad", turn(phi_t), block, turn(phi_s))


class Samples:
    """A's samples in the unit tube's own frame and the pairs between them, with what the world's matrix needs."""

    def __init__(self, local: np.ndarray, points: np.ndarray, frame: np.ndarray, centre: np.ndarray, radius: float,
                 viscosity: float, wall: Any, parameters: dict[str, Any]) -> None:
        self.local, self.points, self.frame, self.centre = local, points, frame, centre
        self.radius, self.viscosity, self.wall, self.parameters = radius, viscosity, wall, parameters
        self.rho = np.hypot(local[:, 0], local[:, 1])
        self.phi = np.arctan2(local[:, 1], local[:, 0])
        self.z = local[:, 2]

    @property
    def count(self) -> int:
        return len(self.rho)

    def pairs(self, target: np.ndarray, source: np.ndarray) -> dict[str, np.ndarray]:
        return {"rho_t": self.rho[target], "rho_s": self.rho[source], "phi_t": self.phi[target],
                "phi_s": self.phi[source], "dphi": self.phi[target] - self.phi[source],
                "dz": self.z[target] - self.z[source]}


def phase_zero_samples() -> Samples:
    """A's samples at phase 0, as ``m3_wall_read.assemble_wall`` builds them (closed surface, 0.178 mm spacing, blob
    0.5, 1/4 inch tube, margin 0.1 mm), and the wall kind that reads them."""
    import jax.numpy as jnp

    import bem
    import operating_point as op
    from mime.core import quaternion
    from mime.nodes.hydro.bem import wall_correction_named

    rung = bem.Rung("closed")
    node = rung.node
    p = node._step_constants(rung.leaves[op.DRAG_NODE])  # noqa: SLF001
    held = node.samples(p)
    pose = rung.pose(0.0)
    points = np.asarray(pose[:3] + jnp.matmul(held.points, quaternion.rotation_matrix(pose[3:]).T, precision="highest"))
    frame = np.asarray(p["wall_frame"], dtype=np.float64).reshape(3, 3)
    centre = np.asarray(p["wall_centre"], dtype=np.float64)
    radius = float(p["wall_radius"])
    wall = wall_correction_named(node.constants()["wall_kind"], node.constants()["wall_kind_version"])
    parameters = {name: p[f"wall_{name}"] for name in wall.parameters}
    local = (points - centre) @ frame.T / radius
    return Samples(local, points, frame, centre, radius, float(p["viscosity"]), wall, parameters)


# ---- the table's read, in this layout --------------------------------------------------------------------------------
def load_table() -> Any:
    from mime import assets
    from mime.nodes.hydro.bem._tube import tube_table_from

    return tube_table_from(str(assets.locate(assets.reference("unit_tube_R0.997"))), scheme="cubic")


def read_source_frame(table: Any, rho_t: np.ndarray, rho_s: np.ndarray, dphi: np.ndarray, dz: np.ndarray, *,
                      signed: bool = True, chunk: int = 20000) -> np.ndarray:
    """``(P, 3, 3)``: the table's unblended read for a target at ``rho_t`` and a source at ``rho_s``, ``dphi`` from the
    source's direction to the target's and ``dz`` the target's offset, rows in the source's frame (``H``). The angle
    is folded into ``[0, pi]`` and the offset into ``[0, ...)``, the odd components taking the sign of ``sin(dphi)``
    and of ``dz`` (``signed=False``, a planted fault, signs nothing)."""
    out = np.empty((len(rho_t), 3, 3))
    for start in range(0, len(rho_t), chunk):
        piece = slice(start, start + chunk)
        sine, cosine, offset = np.sin(dphi[piece]), np.cos(dphi[piece]), dz[piece]
        angle, along = np.arctan2(np.abs(sine), cosine), np.abs(offset)
        read = full.read_at(table, rho_t[piece], rho_s[piece], angle, along)
        if signed:
            by_angle = np.where(sine >= 0, 1.0, -1.0)[:, None, None]
            by_along = np.where(offset >= 0, 1.0, -1.0)[:, None, None]
            read = read * (1.0 + (by_angle - 1.0) * build.SIN_IN_ANGLE) * (1.0 + (by_along - 1.0) * build.SIN_ALONG)
        out[piece] = read
    return out


def to_tube_axes(h: np.ndarray, phi_s: np.ndarray) -> np.ndarray:
    """A block with rows and columns in the source's frame, in the tube frame's Cartesian axes: ``R_s^T H R_s``."""
    rotation = turn(phi_s)
    return np.einsum("pba,pbc,pcd->pad", rotation, h, rotation)


def table_pair(table: Any, pairs: dict[str, np.ndarray], *, signed: bool = True, blend_weight: bool = True
               ) -> dict[str, np.ndarray]:
    """The table's read at pairs (target, source): ``unblended_h`` (rows in the source's frame, the gate's form),
    ``unblended`` (tube axes) and ``blended`` (tube axes: ``w W(t, s) + (1 - w) W(s, t)^T``, ``w = rho_s^2 / (rho_t^2 +
    rho_s^2)``, as the evaluator blends)."""
    h = read_source_frame(table, pairs["rho_t"], pairs["rho_s"], pairs["dphi"], pairs["dz"], signed=signed)
    forward = to_tube_axes(h, pairs["phi_s"])
    h_back = read_source_frame(table, pairs["rho_s"], pairs["rho_t"], -pairs["dphi"], -pairs["dz"], signed=signed)
    back = to_tube_axes(h_back, pairs["phi_t"])
    total = pairs["rho_t"] ** 2 + pairs["rho_s"] ** 2
    weight = np.where(total > 0, pairs["rho_s"] ** 2 / np.where(total > 0, total, 1.0), 0.5)
    if not blend_weight:
        weight = 1.0 - weight
    w = weight[:, None, None]
    return {"unblended_h": h, "unblended": forward, "blended": w * forward + (1.0 - w) * np.swapaxes(back, 1, 2)}


def relative(miss: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Each block's largest absolute miss over the reference block's largest entry."""
    return np.max(np.abs(miss), axis=(-2, -1)) / np.max(np.abs(reference), axis=(-2, -1))


# ---- exact values over pairs --------------------------------------------------------------------------------------
def exact_pairs(samples: Samples, target: np.ndarray, source: np.ndarray, truncation_of: Any,
                on_group: Any = None) -> dict[str, np.ndarray]:
    """The exact image at each pair ``(target[i], source[i])`` (indices into the samples), the source's wall side
    shared by the pairs of one source. ``truncation_of(rho_s)`` is the truncation. Returns the builder's cylindrical
    ``block``, its source-frame form ``h`` and the tube-axes block ``cart``."""
    pairs = samples.pairs(target, source)
    block = np.empty((len(target), 3, 3))
    order = np.argsort(source, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(source[order])) + 1)
    for done, group in enumerate(groups):
        s = int(source[group[0]])
        block[group] = exact_group(samples.rho[s], pairs["rho_t"][group], pairs["dphi"][group], pairs["dz"][group],
                                   truncation_of(float(samples.rho[s])))
        if on_group is not None:
            on_group(done, len(groups), s, len(group))
    return {"block": block, "h": full.in_source_frame(block, pairs["dphi"]),
            "cart": cartesian_block(block, pairs["phi_t"], pairs["phi_s"])}


# ---- the controls -----------------------------------------------------------------------------------------------------
def draw_pairs(samples: Samples, rng: np.random.Generator, per_stratum: int) -> dict[str, Any]:
    """Pair indices from the three strata (the stage's draw, at smaller counts): ``target``, ``source``, ``stratum``."""
    plan = StratumPlan(samples)
    target, source, stratum = [], [], []
    for name in STRATA:
        t, s = plan.draw(name, rng, sources=max(2, per_stratum // 10), per_source=10)
        target.append(t)
        source.append(s)
        stratum.append(np.full(len(t), STRATA.index(name)))
    return {"target": np.concatenate(target), "source": np.concatenate(source), "stratum": np.concatenate(stratum)}


def control_nodes() -> dict[str, Any]:
    """Control 0: scipy's Gauss-Legendre nodes against ``build.wavenumber_nodes`` (numpy's) at two counts."""
    out = {}
    for count in (256, 1515):
        truncation = build.Truncation(24, count, 30.0, 64)
        k_a, w_a = wavenumbers(truncation)
        k_b, w_b = build.wavenumber_nodes(truncation)
        out[str(count)] = {"nodes_rel": float(np.max(np.abs(k_a - k_b) / k_b)),
                           "weights_abs_over_largest": float(np.max(np.abs(w_a - w_b)) / np.max(w_b)),
                           "weights_rel": float(np.max(np.abs(w_a - w_b) / w_b))}
    out["pass"] = all(v["nodes_rel"] <= TOLERANCE["nodes"] and v["weights_abs_over_largest"] <= TOLERANCE["weights"]
                      for v in out.values() if isinstance(v, dict))
    return out


def control_kernel_vs_tests(rng: np.random.Generator) -> dict[str, Any]:
    """Control 1: the kernel against ``tests/tube_fourier_bessel.image`` (the independent implementation of the same
    series), 6 sources x 10 targets = 60 pairs, one truncation given to both; the sources' radii on the oracle's 1e-6
    grid (it bins them). **At ``check.py``'s COARSE counts (15 modes, 20 nodes, 64 angles)**: the oracle's own system is
    ill-conditioned at the smallest wavenumbers (the builder's recombined basis exists because of that), so with more
    nodes the oracle, not the kernel, carries a rounding of about 1e-6; ``scripts/tube_table/check.py`` makes the same
    comparison at these counts for the same reason. The criterion: the kernel equals the builder's own pairs routine to
    rounding (1e-12) and is no further from the oracle than that routine is (or than 1e-9). Planted faults (the target's angle negated, its offset negated, the
    transpose missing) must each be caught by an order of magnitude or more over the tolerance. (b) The kernel against the
    builder's own pairs routine (``build.image_points``, the grid path) at 30 modes x 120 nodes, where the oracle is not
    a reference: the per-pair contraction against the grid's. (c) The oracle against the kernel at those larger counts,
    recorded, not a criterion."""
    import tube_fourier_bessel as fb

    rho_s = np.round(rng.uniform(0.02, 0.9, 6), 6)
    angle_s = rng.uniform(-math.pi, math.pi, 6)
    sources = np.stack([rho_s * np.cos(angle_s), rho_s * np.sin(angle_s), rng.uniform(-0.5, 0.5, 6)], axis=1)
    rho_t = rng.uniform(0.0, 0.9, 10)
    rho_t[0] = 0.0
    angle_t = rng.uniform(-math.pi, math.pi, 10)
    targets = np.stack([rho_t * np.cos(angle_t), rho_t * np.sin(angle_t), rng.uniform(-2.0, 2.0, 10)], axis=1)
    phi_t = np.arctan2(targets[:, 1], targets[:, 0])

    def mine(truncation: build.Truncation, dphi_sign: float = 1.0, dz_sign: float = 1.0) -> np.ndarray:
        out = np.empty((10, 6, 3, 3))
        for j, source in enumerate(sources):
            phi_s = math.atan2(source[1], source[0])
            block = exact_group(math.hypot(*source[:2]), rho_t, dphi_sign * (phi_t - phi_s),
                                dz_sign * (targets[:, 2] - source[2]), truncation)
            out[:, j] = cartesian_block(block, phi_t, np.full(10, phi_s))
        return out

    def miss(found: np.ndarray, reference: np.ndarray) -> float:
        return float(np.max(relative(found - reference, reference)))

    coarse = dict(modes=15, wavenumbers=20, angles=64)
    truncation = build.series_truncation(**coarse, largest_source_radius=float(rho_s.max()))
    oracle = fb.image(targets, sources, 1.0, 1.0, **coarse)
    found = mine(truncation)
    faults = {"negated_angle": miss(mine(truncation, dphi_sign=-1.0), oracle),
              "negated_offset": miss(mine(truncation, dz_sign=-1.0), oracle),
              "transpose_missing": miss(np.swapaxes(found, -1, -2), oracle)}
    caught = all(v > TOLERANCE["plant_detected"] for v in faults.values())
    builder_coarse = build.image_points(targets, sources, truncation)
    kernel_vs_builder_coarse = miss(found, builder_coarse)
    builder_vs_oracle_coarse = miss(builder_coarse, oracle)
    # (b) and (c) at 30 modes x 120 nodes
    large = dict(modes=30, wavenumbers=120, angles=96)
    big = build.series_truncation(**large, largest_source_radius=float(rho_s.max()))
    grid_path = build.image_points(targets, sources, big)
    kernel_big = mine(big)
    against_grid = miss(kernel_big, grid_path)
    oracle_big = miss(kernel_big, fb.image(targets, sources, 1.0, 1.0, **large))
    oracle_vs_grid = miss(grid_path, fb.image(targets, sources, 1.0, 1.0, **large))
    return {"pairs": 60, "truncation": truncation.__dict__, "max_relative_miss": miss(found, oracle),
            "tolerance": TOLERANCE["kernel_vs_tests"], "planted_fault_misses": faults, "faults_caught": caught,
            "kernel_vs_builder_pairs_routine_coarse": kernel_vs_builder_coarse,
            "builder_pairs_routine_vs_oracle_coarse": builder_vs_oracle_coarse,
            "against_builder_pairs_routine_30x120": against_grid,
            "oracle_against_kernel_30x120_recorded": oracle_big,
            "oracle_against_builder_pairs_routine_30x120_recorded": oracle_vs_grid,
            "pass": bool(kernel_vs_builder_coarse <= TOLERANCE["rounding"] and caught
                         and miss(found, oracle) <= max(TOLERANCE["kernel_vs_tests"], builder_vs_oracle_coarse
                                                        + TOLERANCE["rounding"])
                         and against_grid <= TOLERANCE["rounding"])}


def control_kernel_vs_builder(gate_results: Path, key: str = "gate_0_any_0") -> dict[str, Any]:
    """Control 1b: the kernel at the table's own truncation against the blocks the table's builder stored for one of its
    gate jobs (``full.py``'s random off-knot targets; built on Modal, numpy/scipy there), all 27 targets of one source."""
    values, item = full.load(gate_results, key)
    rho, phi, z = np.meshgrid(item["rho"], item["phi"], item["z"], indexing="ij")
    truncation = build.Truncation(**item["truncation"])
    assert truncation == table_truncation(), "the stored job is not at the table's truncation"
    started = time.perf_counter()
    mine = exact_group(item["source"], rho.ravel(), phi.ravel(), z.ravel(), truncation)
    stored = values.reshape(-1, 3, 3)
    miss = float(np.max(relative(mine - stored, stored)))
    return {"job": key, "source": item["source"], "pairs": len(stored), "seconds": time.perf_counter() - started,
            "max_relative_miss": miss, "tolerance": TOLERANCE["kernel_vs_builder_stored"],
            "pass": miss <= TOLERANCE["kernel_vs_builder_stored"]}


def control_knots(samples: Samples, table: Any, rng: np.random.Generator) -> dict[str, Any]:
    """Control 2: exact against table **at knot coordinates**: 8 sources at table radii inside A's range, 20 targets
    each at table radii, angles and offsets (signed at random, so the fold is exercised). E there is the truncation's,
    1e-9 and not O(1). Planted faults (the read without its parity signs, the rows unturned) must each give O(1)."""
    rho_k, angle_k, along_k = full.knots("G1")
    inside = rho_k[(rho_k > 0.0) & (rho_k <= RHO_MAX_A)]
    source_r = inside[np.linspace(1, len(inside) - 1, 8).astype(int)]
    worst, worst_gate, rows = 0.0, 0.0, []
    plants = {"odd_components_unsigned": 0.0, "rows_unturned": 0.0}
    for rho_s in source_r:
        target_r = rng.choice(inside, 20)
        dphi = rng.choice(angle_k, 20) * rng.choice([-1.0, 1.0], 20)
        dz = rng.choice(along_k[along_k < 2.4], 20) * rng.choice([-1.0, 1.0], 20)
        phi_s = rng.uniform(-math.pi, math.pi, 20)
        pairs = {"rho_t": target_r, "rho_s": np.full(20, rho_s), "phi_s": phi_s, "phi_t": phi_s + dphi, "dphi": dphi,
                 "dz": dz}
        block = exact_group(float(rho_s), target_r, dphi, dz, pair_truncation(float(rho_s)))
        cart = cartesian_block(block, pairs["phi_t"], pairs["phi_s"])
        read = table_pair(table, pairs)
        blend = float(np.max(relative(read["blended"] - cart, cart)))
        gate = float(np.max(relative(read["unblended_h"] - full.in_source_frame(block, dphi), block)))
        worst, worst_gate = max(worst, blend), max(worst_gate, gate)
        rows.append({"source": float(rho_s), "blend": blend, "gate": gate})
        wrong = table_pair(table, pairs, signed=False)["blended"]
        plants["odd_components_unsigned"] = max(plants["odd_components_unsigned"],
                                                float(np.max(relative(wrong - cart, cart))))
        plants["rows_unturned"] = max(plants["rows_unturned"], float(np.max(relative(
            read["unblended_h"] - block, block))))
    caught = all(v > TOLERANCE["plant_detected"] for v in plants.values())
    return {"sources": rows, "pairs": 160, "max_blend": worst, "max_gate": worst_gate, "tolerance": TOLERANCE["knots"],
            "planted_fault_misses": plants, "faults_caught": caught,
            "pass": worst <= TOLERANCE["knots"] and worst_gate <= TOLERANCE["knots"] and caught}


def control_reciprocity(samples: Samples, rng: np.random.Generator) -> dict[str, Any]:
    """Control 4: ``G(t, s) = G(s, t)^T`` of the exact image. (i) 60 pairs with the two directions at one common
    truncation (the finer of the two sources' own): the identity, to rounding. (ii) The same pairs with each direction at
    its own source's rule (the stage's way, the source being the other sample): the truncation's share. The plant: the
    pair compared without the transpose."""
    n = samples.count
    a = rng.integers(0, n, 60)
    b = rng.integers(0, n, 60)
    keep = a != b
    a, b = a[keep], b[keep]
    rows = []
    for i, j in zip(a, b, strict=True):
        tr = max((pair_truncation(float(samples.rho[i]), FINER_MULTIPLES),
                  pair_truncation(float(samples.rho[j]), FINER_MULTIPLES)), key=lambda t: (t.modes, t.wavenumbers))
        forward = exact_pairs(samples, np.array([j]), np.array([i]), lambda r, tr=tr: tr)["cart"][0]
        back = exact_pairs(samples, np.array([i]), np.array([j]), lambda r, tr=tr: tr)["cart"][0]
        base_forward = exact_pairs(samples, np.array([j]), np.array([i]), pair_truncation)["cart"][0]
        base_back = exact_pairs(samples, np.array([i]), np.array([j]), pair_truncation)["cart"][0]
        rows.append((relative(forward - back.T, forward), relative(forward - back, forward),
                     relative(base_forward - base_back.T, base_forward)))
    identity = np.array([r[0] for r in rows])
    plant = np.array([r[1] for r in rows])
    own = np.array([r[2] for r in rows])
    return {"pairs": int(keep.sum()), "common_truncation_max": float(identity.max()),
            "own_truncation_max": float(own.max()), "own_truncation_median": float(np.median(own)),
            "planted_no_transpose_median": float(np.median(plant)),
            "tolerance_common": TOLERANCE["reciprocity_common_truncation"],
            "pass": bool(identity.max() <= TOLERANCE["reciprocity_common_truncation"]
                         and np.median(plant) > TOLERANCE["plant_detected"])}


def control_plumbing(samples: Samples, table: Any, rng: np.random.Generator, count: int = 200) -> dict[str, Any]:
    """Control 5: the table's read routed through this layout (``table_pair``'s blend, ``to_tube_axes``, then the tube
    frame's rows) against ``corrected.matrix`` itself (``wall.correction`` at the same samples, times ``R mu``), every
    entry of the ``3 count x 3 count`` matrix of ``count`` random samples (all pairs, the diagonal included). The
    planted faults must each miss: the read's odd components unsigned and the frame not applied by O(1) (above 1e-2),
    the blend's weight taken the other way round by more than 1e3 times the tolerance (that fault is only as large as the
    table's own non-reciprocity, up to 1.4e-3 of a block by the evaluator's docstring)."""
    import jax.numpy as jnp

    chosen = np.sort(rng.choice(samples.count, count, replace=False))
    library = np.asarray(samples.wall.correction(jnp.asarray(samples.points[chosen]), samples.parameters,
                                                 samples.viscosity).matrix, dtype=np.float64)
    library = library * samples.radius * samples.viscosity
    target = np.repeat(chosen, count)
    source = np.tile(chosen, count)
    pairs = samples.pairs(target, source)

    def matrix(turn_frame: bool = True, **options: Any) -> np.ndarray:
        local = table_pair(table, pairs, **options)["blended"]
        world = np.einsum("ai,pab,bj->pij", samples.frame, local, samples.frame) if turn_frame else local
        return world.reshape(count, count, 3, 3).transpose(0, 2, 1, 3).reshape(3 * count, 3 * count)

    scale = np.abs(library).max()
    miss = float(np.abs(matrix() - library).max() / scale)
    plants = {"blend_weight_reversed": float(np.abs(matrix(blend_weight=False) - library).max() / scale),
              "odd_components_unsigned": float(np.abs(matrix(signed=False) - library).max() / scale),
              "frame_not_applied": float(np.abs(matrix(turn_frame=False) - library).max() / scale)}
    same_parities = bool(np.array_equal(build.SIN_IN_ANGLE, _tube().ODD_IN_ANGLE)
                         and np.array_equal(build.SIN_ALONG, _tube().ODD_ALONG))
    caught = (plants["odd_components_unsigned"] > TOLERANCE["plant_detected"]
              and plants["frame_not_applied"] > TOLERANCE["plant_detected"]
              and plants["blend_weight_reversed"] > 1e3 * TOLERANCE["plumbing"])
    return {"samples": count, "entries": int(library.size), "max_relative_miss": miss,
            "tolerance": TOLERANCE["plumbing"], "planted_fault_misses": plants, "faults_caught": caught,
            "parity_arrays_equal_to_the_evaluators": same_parities,
            "pass": miss <= TOLERANCE["plumbing"] and caught and same_parities}


def _tube() -> Any:
    from mime.nodes.hydro.bem import _tube

    return _tube


def control_gate(table: Any, gate_results: Path) -> dict[str, Any]:
    """Control 6: the gate's 0.002 on the gate's own draws: the table's read (``full.read_errors``, the gate's own
    function) against the builder's stored blocks of its five runs, and the same draws read through this layout's
    ``read_source_frame``. Fails if the maximum is not within [0.0015, 0.0025]."""
    runs, mine = [], []
    for run in range(5):
        rt, rs, ph, zz, reference = full.reference_points(gate_results, f"gate_{run}_")
        error = full.read_errors(table, rt, rs, ph, zz, reference)
        runs.append({"max": float(error.max()), "p99": float(np.quantile(error, 0.99)), "points": int(error.size)})
        ours = read_source_frame(table, rt, rs, ph, zz)
        turned = full.in_source_frame(reference, ph)
        mine.append(float(np.max(relative(ours - turned, turned))))
    low, high = TOLERANCE["gate"]
    return {"runs": runs, "max": max(r["max"] for r in runs), "this_layout_max_per_run": mine,
            "window": [low, high], "pass": low <= max(r["max"] for r in runs) <= high
            and abs(max(mine) - max(r["max"] for r in runs)) < 1e-12}


# ---- the strata -------------------------------------------------------------------------------------------------------
STRATA = ("inner", "outer", "other")


class StratumPlan:
    """The pairs of each stratum, drawn uniformly. A pair's source is its sample nearer the axis (rank by radius), its
    target the other. ``inner``: a sample at rho < 0.157 (the source is then inner); ``outer``: a sample at rho > 0.9 and
    none inside 0.157; ``other``: both in [0.157, 0.9]. A source is drawn with probability proportional to its number of
    partners in the stratum, then its partners uniformly: uniform over the stratum's unordered pairs, clustered by
    source (a wall side per source is the dominant cost)."""

    def __init__(self, samples: Samples) -> None:
        self.samples = samples
        rho = samples.rho
        self.inner, self.outer = rho < INNER_EDGE, rho > OUTER_EDGE
        self.middle = ~self.inner & ~self.outer
        self.rank = np.argsort(np.argsort(rho))

    def partners(self, name: str, s: int) -> np.ndarray:
        after = self.rank > self.rank[s]
        if name == "inner":
            return np.flatnonzero(after) if self.inner[s] else np.empty(0, dtype=int)
        if name == "outer":
            if self.inner[s]:
                return np.empty(0, dtype=int)
            return np.flatnonzero(after & self.outer)
        return np.flatnonzero(after & self.middle) if self.middle[s] else np.empty(0, dtype=int)

    def degrees(self, name: str) -> np.ndarray:
        return np.array([len(self.partners(name, s)) for s in range(self.samples.count)])

    def draw(self, name: str, rng: np.random.Generator, sources: int, per_source: int) -> tuple[np.ndarray, np.ndarray]:
        degree = self.degrees(name)
        chosen = rng.choice(self.samples.count, sources, p=degree / degree.sum())
        target, source = [], []
        for s in chosen:
            partner = self.partners(name, int(s))
            target.append(rng.choice(partner, per_source))
            source.append(np.full(per_source, s))
        return np.concatenate(target), np.concatenate(source)

    def population(self) -> dict[str, int]:
        return {name: int(self.degrees(name).sum()) for name in STRATA}


def control_truncation(samples: Samples, table: Any, drawn: dict[str, Any], table_sources: int, table_targets: int
                       ) -> dict[str, Any]:
    """Control 3: the truncation's change against E. (a) The finer rule (``FINER_MULTIPLES`` times the base) on every
    pair of the draw: the change of the exact block over its largest entry, per stratum, beside E of the same pairs.
    (b) The table's own truncation (2500 modes, 16000 nodes) on ``table_sources`` x ``table_targets`` pairs of the draw.
    The criterion, per stratum and form: median and 95th percentile of the change each at most a tenth of E's."""
    t, s = drawn["target"], drawn["source"]
    base = exact_pairs(samples, t, s, pair_truncation)
    finer = exact_pairs(samples, t, s, finer_truncation)
    pairs = samples.pairs(t, s)
    read = table_pair(table, pairs)
    e_blend = relative(read["blended"] - base["cart"], base["cart"])
    e_gate = relative(read["unblended_h"] - base["h"], base["h"])
    change = relative(finer["cart"] - base["cart"], base["cart"])
    report: dict[str, Any] = {"pairs": len(t), "per_stratum": {}}
    ok = True
    for index, name in enumerate(STRATA):
        pick = drawn["stratum"] == index
        row = {"pairs": int(pick.sum())}
        for label, e in (("blend", e_blend), ("gate", e_gate)):
            ratio = {q: float(np.quantile(e[pick], q) / max(np.quantile(change[pick], q), 1e-300)) for q in (0.5, 0.95)}
            row[label] = {"E_median": float(np.median(e[pick])), "E_p95": float(np.quantile(e[pick], 0.95)),
                          "change_median": float(np.median(change[pick])),
                          "change_p95": float(np.quantile(change[pick], 0.95)),
                          "change_max": float(change[pick].max()), "E_over_change_median": ratio[0.5],
                          "E_over_change_p95": ratio[0.95]}
            ok = ok and ratio[0.5] > TOLERANCE["truncation_ratio"] and ratio[0.95] > TOLERANCE["truncation_ratio"]
        report["per_stratum"][name] = row
    report["finer_rule"] = {"pass": bool(ok)}
    # (b) the table's own truncation
    started = time.perf_counter()
    picks = []
    for index in range(3):
        members = np.flatnonzero(drawn["stratum"] == index)
        first_source = s[members[0]]
        same = members[s[members] == first_source][:table_targets]
        picks.append(same)
    pick = np.concatenate(picks)[:table_sources * table_targets]
    tt, ss = t[pick], s[pick]
    exact_table = exact_pairs(samples, tt, ss, lambda r: table_truncation())
    miss = relative(exact_table["cart"] - base["cart"][pick], exact_table["cart"])
    miss_finer = relative(finer["cart"][pick] - base["cart"][pick], base["cart"][pick])
    e_table = relative(read["blended"][pick] - exact_table["cart"], exact_table["cart"])
    report["table_truncation"] = {
        "pairs": int(len(pick)), "seconds": time.perf_counter() - started, "stratum": drawn["stratum"][pick].tolist(),
        "base_vs_table_truncation_change": miss.tolist(), "base_vs_finer_rule_change_same_pairs": miss_finer.tolist(),
        "E_against_table_truncation_exact": e_table.tolist(), "E_against_base_exact": e_blend[pick].tolist(),
        "min_E_over_change": float(np.min(e_table / np.maximum(miss, 1e-300))),
        "pass": bool(np.all(e_table > TOLERANCE["truncation_ratio"] * miss))}
    report["pass"] = bool(report["finer_rule"]["pass"] and report["table_truncation"]["pass"])
    return report


def run_controls(out: Path, only: list[str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    results: dict[str, Any] = {"seed": SEED, "tolerances": {k: v for k, v in TOLERANCE.items()}}
    if only is not None and (out / "controls.json").exists():  # a partial run adds to the record, never replaces it
        results = json.loads((out / "controls.json").read_text()) | results
    gate_results = Path(os.environ.get("TUBE_GATE_RESULTS", ROOT.parents[0] / "plans" / "tube_table" / "rebuild2" / "build"))
    print("building A's phase-0 samples", flush=True)
    samples = phase_zero_samples()
    table = load_table()
    results["samples"] = samples.count
    results["rho_range"] = [float(samples.rho.min()), float(samples.rho.max())]
    results["table_sha256"] = hashlib.sha256(Path(str(table_path())).read_bytes()).hexdigest()
    wanted = lambda name: only is None or name in only  # noqa: E731

    def step(name: str, function: Any, *args: Any) -> None:
        if not wanted(name):
            return
        t0 = time.perf_counter()
        results[name] = function(*args)
        results[name]["seconds"] = time.perf_counter() - t0
        print(name, "PASS" if results[name]["pass"] else "FAIL", f"{results[name]['seconds']:.0f}s", flush=True)
        (out / "controls.json").write_text(json.dumps(results, indent=1, default=float) + "\n")

    drawn = draw_pairs(samples, np.random.default_rng(SEED + 1), per_stratum=70)
    step("c0_nodes", control_nodes)
    step("c1_kernel_vs_tests", control_kernel_vs_tests, rng)
    step("c2_knots", control_knots, samples, table, rng)
    step("c4_reciprocity", control_reciprocity, samples, rng)
    step("c5_plumbing", control_plumbing, samples, table, rng)
    step("c6_gate", control_gate, table, gate_results)
    step("c3_truncation", control_truncation, samples, table, drawn, 3, 7)
    step("c1b_kernel_vs_builder", control_kernel_vs_builder, gate_results)
    results["all_pass"] = all(v["pass"] for k, v in results.items() if isinstance(v, dict) and "pass" in v)
    results["seconds"] = time.perf_counter() - started
    (out / "controls.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    return results


def table_path() -> Path:
    from mime import assets

    return assets.locate(assets.reference("unit_tube_R0.997"))


# ---- stage 1 ----------------------------------------------------------------------------------------------------------
PLAN = {"inner": (100, 100), "outer": (500, 20), "other": (500, 20)}
DIAGONAL = 200


def stage1(out: Path, plan: dict[str, tuple[int, int]] = PLAN, diagonal: int = DIAGONAL, scale: float = 1.0) -> dict:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED + 7)
    samples = phase_zero_samples()
    table = load_table()
    strata = StratumPlan(samples)
    target, source, stratum = [], [], []
    for index, name in enumerate(STRATA):
        sources, per = plan[name]
        t, s = strata.draw(name, rng, max(1, int(sources * scale)), per)
        target.append(t)
        source.append(s)
        stratum.append(np.full(len(t), index))
    own = rng.choice(samples.count, max(1, int(diagonal * scale)), replace=False)
    target.append(own)
    source.append(own)
    stratum.append(np.full(len(own), len(STRATA)))
    target, source, stratum = np.concatenate(target), np.concatenate(source), np.concatenate(stratum)
    print("pairs", len(target), "population", strata.population(), flush=True)

    def progress(done: int, total: int, s: int, count: int) -> None:
        if done % 25 == 0:
            print(f"group {done}/{total}  source rho {samples.rho[s]:.3f}  {time.perf_counter() - started:.0f}s", flush=True)

    exact = exact_pairs(samples, target, source, pair_truncation, progress)
    pairs = samples.pairs(target, source)
    read = table_pair(table, pairs)
    miss_blend = relative(read["blended"] - exact["cart"], exact["cart"])
    miss_gate = relative(read["unblended_h"] - exact["h"], exact["h"])
    miss_unblended = relative(read["unblended"] - exact["cart"], exact["cart"])
    distance = np.sqrt((samples.local[target] - samples.local[source]).__pow__(2).sum(axis=1))
    record = {"target": target, "source": source, "stratum": stratum, "rho_t": pairs["rho_t"], "rho_s": pairs["rho_s"],
              "dphi": pairs["dphi"], "dz": pairs["dz"], "distance": distance, "E_blend": miss_blend, "E_gate": miss_gate,
              "E_unblended_axes": miss_unblended, "block_max": np.max(np.abs(exact["cart"]), axis=(1, 2))}
    np.savez(out / "stage1_pairs.npz", **record)
    summary = summarise(record, strata.population())
    summary["seconds"] = time.perf_counter() - started
    summary["pairs"] = int(len(target))
    (out / "stage1_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    pair_json = {k: np.asarray(v).tolist() for k, v in record.items()}
    (out / "stage1_pairs.json").write_text(json.dumps(pair_json, default=float) + "\n")
    return summary


BIN_SEPARATION = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 5.0])
BIN_RHO = np.array([0.0, 0.157, 0.3, 0.5, 0.7, 0.9, 1.0])


def stats_of(values: np.ndarray) -> dict[str, float]:
    return {"median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)), "max": float(values.max()),
            "mean": float(values.mean())}


def summarise(record: dict[str, np.ndarray], population: dict[str, int]) -> dict[str, Any]:
    """Per stratum: |E| (both forms) median, 95th percentile and maximum; the share above the gate's 0.002; Spearman's
    rank correlation of |E| with the pair's separation, the larger and the smaller of its radii, and its axial offset."""
    names = [*STRATA, "diagonal"]
    out: dict[str, Any] = {"population_unordered_pairs": population, "strata": {}}
    for index, name in enumerate(names):
        pick = record["stratum"] == index
        row: dict[str, Any] = {"pairs": int(pick.sum()), "sources": int(len(np.unique(record["source"][pick])))}
        for label in ("E_blend", "E_gate"):
            e = record[label][pick]
            row[label] = {**stats_of(e), "share_above_0.002": float(np.mean(e > 0.002)),
                          "share_above_0.02": float(np.mean(e > 0.02))}
            row[label]["spearman"] = {}
            for axis, values in (("separation", record["distance"]), ("rho_max", np.maximum(record["rho_t"], record["rho_s"])),
                                 ("rho_min", np.minimum(record["rho_t"], record["rho_s"])),
                                 ("abs_dz", np.abs(record["dz"]))):
                if pick.sum() > 3 and np.ptp(values[pick]) > 0:
                    r = stats.spearmanr(values[pick], e)
                    row[label]["spearman"][axis] = {"rho": float(r.statistic), "p": float(r.pvalue)}
        for label in ("E_blend", "E_gate"):
            row[label]["binned_median"] = {
                axis: [{"from": float(lo), "to": float(hi), "pairs": int(sel.sum()),
                        "median": float(np.median(record[label][pick][sel])) if sel.sum() else None}
                       for lo, hi in zip(edges[:-1], edges[1:], strict=True)
                       for sel in [(values[pick] >= lo) & (values[pick] < hi)]]
                for axis, values, edges in (("separation", record["distance"], BIN_SEPARATION),
                                            ("rho_max", np.maximum(record["rho_t"], record["rho_s"]), BIN_RHO),
                                            ("rho_min", np.minimum(record["rho_t"], record["rho_s"]), BIN_RHO))}
        out["strata"][name] = row
    return out


def pilot() -> None:
    """Timing and the truncation's convergence on a handful of pairs (decides the stage's counts, nothing else)."""
    rng = np.random.default_rng(SEED + 99)
    samples = phase_zero_samples()
    table = load_table()
    drawn = draw_pairs(samples, rng, per_stratum=20)
    t, s = drawn["target"], drawn["source"]
    for label, trunc in (("base", pair_truncation), ("finer", finer_truncation)):
        t0 = time.perf_counter()
        r = exact_pairs(samples, t, s, trunc)
        print(label, f"{time.perf_counter() - t0:.1f}s for {len(t)} pairs", flush=True)
        if label == "base":
            base = r
        else:
            change = relative(r["cart"] - base["cart"], base["cart"])
            read = table_pair(table, samples.pairs(t, s))
            e = relative(read["blended"] - base["cart"], base["cart"])
            for index, name in enumerate(STRATA):
                pick = drawn["stratum"] == index
                print(name, "change median/max", np.median(change[pick]), change[pick].max(), "E median/max",
                      np.median(e[pick]), e[pick].max(), flush=True)


def cost_estimate(out: Path, sources: int = 16, targets: int = 200) -> dict[str, Any]:
    """The cost of every pair of one phase at the stage's truncation, from timings: for ``sources`` samples spread over
    the radius ranking, the wall side (a one-target call less the marginal target) and the marginal cost per target
    (``targets`` partners further from the axis, against one), scaled to the source's real number of partners and summed
    over all samples by the ranking's quantile (each timed source stands for its share of the ranking). One core."""
    samples = phase_zero_samples()
    rng = np.random.default_rng(SEED + 3)
    order = np.argsort(samples.rho)
    n = samples.count
    spots = np.linspace(0, n - 2, sources).astype(int)
    rows = []
    for position in spots:
        s = int(order[position])
        partners = order[position + 1:]
        pick = rng.choice(partners, min(targets, len(partners)), replace=False)
        pairs = samples.pairs(pick, np.full(len(pick), s))
        truncation = pair_truncation(float(samples.rho[s]))
        t0 = time.perf_counter()
        exact_group(samples.rho[s], pairs["rho_t"][:1], pairs["dphi"][:1], pairs["dz"][:1], truncation)
        one = time.perf_counter() - t0
        t0 = time.perf_counter()
        exact_group(samples.rho[s], pairs["rho_t"], pairs["dphi"], pairs["dz"], truncation)
        many = time.perf_counter() - t0
        marginal = max((many - one) / max(len(pick) - 1, 1), 0.0)
        wall = max(one - marginal, 0.0)
        rows.append({"rank": int(position), "rho": float(samples.rho[s]), "partners": int(len(partners)), "wall_s": wall,
                     "marginal_ms": marginal * 1e3, "source_seconds": wall + marginal * len(partners)})
    mean_source = float(np.mean([r["source_seconds"] for r in rows]))
    total = mean_source * n
    found = {"rows": rows, "unordered_pairs_all_samples": n * (n - 1) // 2 + n, "estimated_core_hours_one_phase": total / 3600,
             "estimated_core_hours_eight_phases": 8 * total / 3600,
             "mean_marginal_ms_per_pair": float(np.mean([r["marginal_ms"] for r in rows])),
             "note": "sources spread evenly over the radius ranking (rank 0 .. N-2); each stands for 1/16 of the sources; "
                     "the diagonal and the table reads are not counted"}
    (out / "cost_estimate.json").write_text(json.dumps(found, indent=1) + "\n")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT.parent / "plans" / "sim2real" / "m3" / "route2"))
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--stage1", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--cost", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        pilot()
    if args.cost:
        print(json.dumps(cost_estimate(out), indent=1))
    if args.controls:
        run_controls(out, args.only)
    if args.stage1:
        print(json.dumps(stage1(out, scale=args.scale), indent=1))


if __name__ == "__main__":
    main()
