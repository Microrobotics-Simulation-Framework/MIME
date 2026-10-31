"""MIME-ANO-119's fix checked at 7j-a's reference point "A" (FL-9 on the floor of the 1/4" tube, the closed surface at
0.178 mm, blob 0.5, margin 0.1 mm, 10 Hz, the phase-mean axial speed over eight phases): the screw kind 0.2.0, whose
caps are laid in rings with no two samples closer than ``CAP_SEPARATION`` spacings, against the polar grid of 0.1.0,
rebuilt here (``old_caps``) as its control. No measured speed enters. Predictions: ``predictions_capfix.json``,
committed before any run.

Stages, each a function of the rung alone, run in this order (the cheapest and the preregistered first):

- ``forward``: the speed at every phase through the library's own path (``op.bem_resistance``) and through this
  script's assembly with the kind's caps (a control: equal to rounding) and with 0.1.0's (a control: 7j-a's speed),
  the 6 x 6 resistance in SI at every phase kept for both (``resistance_si``, the library's path for the fixed caps);
  whether ``A + G`` (the free-space matrix and the wall's correction) has a Cholesky factor at each phase; the
  free-space resistance with each surface's caps; the samples' hashes, 0.1.0's against 7j-a's preregistered one.
- ``scan_fine``: the wall correction's scale from x1.000 to x1.040 in steps of 0.0025 at every phase: the speed and the
  resistance's smallest non-dimensional eigenvalue; the first indefinite scale at each phase and the least of them.
- ``derivative``: DEFINITION 5.3's check (central differences at a relative step of 1e-6 against the derivative,
  tolerance 1e-4, and the step ten times larger), for the body's radius, the margin and the tube's radius, each by
  its own forward-mode tangent; its planted control (the derivative off by 1% to 20%, 20 draws); a scan of steps
  (diagnostic).
- ``eigen``: 0.1.0's ``A + G`` at phase 0 (a control: the anomaly's negative eigenvalues, seen by this code), then
  the eigenvalues of ``A + G`` at every phase (the count below zero, the least and the largest), then of ``A`` alone
  with each surface's caps.

Run by hand or on Modal (``capfix_modal.py``), never by CI:

    JAX_ENABLE_X64=1 python scripts/sim2real/capfix.py --stage forward --out DIR [--spacing-mm 0.9]
"""

from __future__ import annotations

import argparse
import hashlib
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
#: 7j-a's reference rung: the closed surface's phase-mean axial speed at A (static, 0.1.0's caps), mm/s.
REFERENCE_SPEED_MM_S = 4.8173
#: 7j-a's preregistered hash of the closed surface's samples at 0.178 mm (0.1.0's caps), in the body's frame.
PREREGISTERED_SHA256 = "2da1ad96a7b693fbab304ca280c80f777441a5d271e4e2d044f8e0635c5e6387"
SCAN_FINE_SCALES = tuple(round(1.0 + 0.0025 * k, 4) for k in range(17))
DERIVATIVE_INPUTS = ("body_radius", "margin", "tube_radius")
STAGES = ("forward", "scan_fine", "derivative", "eigen")


# ---- 0.1.0's caps, as that version sampled them (the control's surface) ---------------------------------------------
def old_caps(kind: Any, parameters: dict[str, Any], counts: tuple[int, ...]) -> Any:
    """The screw kind 0.1.0's caps, its code copied: the trapezoid rule round the turn and Gauss-Legendre nodes in the
    fraction out, weighted by ``s rho^2 ds dtheta``."""
    import jax.numpy as jnp

    from mime.core import SCREW
    from mime.registry import Samples
    from mime.scene._screw import _turned  # noqa: PLC2701

    radius, length = parameters["radius"], parameters["length"]
    epsilon, starts = parameters["modulation"], kind.starts
    half = length / 2.0
    outward, around = counts
    node, weight = np.polynomial.legendre.leggauss(outward)
    fraction = jnp.asarray(np.repeat((node + 1.0) / 2.0, around))
    share = jnp.asarray(np.repeat(weight / 2.0, around))
    angle = np.tile(2.0 * math.pi * (np.arange(around) + 0.5) / around, outward)
    rho = radius * (1.0 + epsilon * jnp.sin(starts * jnp.asarray(angle)))
    found = []
    for end in (-1.0, 1.0):
        c, s = _turned(angle, SCREW.turn(parameters, end * half))
        points = jnp.stack([jnp.full_like(rho, end * half), fraction * rho * c, fraction * rho * s], axis=-1)
        normals = jnp.zeros_like(points).at[:, 0].set(end)
        found.append(Samples(points, normals, share * fraction * rho ** 2 * (2.0 * math.pi / around), rho))
    return Samples(*(jnp.concatenate([getattr(s, name) for s in found]) for name in
                     ("points", "normals", "weights", "rounding_length")))


def old_counts(node: Any, spacing_m: float) -> tuple[int, int]:
    """0.1.0's caps' counts at ``spacing_m``: ``ceil(R (1 + eps) / spacing)`` rings of the lateral's ``around``."""
    constants = node.constants()
    reach = constants["shape_radius"] * (1.0 + constants["shape_modulation"])
    lateral = dict(zip(constants["patches"], constants["counts"], strict=True))["lateral"]
    return max(1, math.ceil(reach / spacing_m)), int(lateral[0])


def node_samples(node: Any, p: dict[str, Any], *, old: tuple[int, int] | None) -> Any:
    """The node's samples (``BEMDragNode.samples``, its code followed), the caps 0.1.0's at the counts ``old`` where
    given."""
    import jax.numpy as jnp

    from mime.nodes.hydro.bem import SHAPE
    from mime.registry import Samples

    constants = node.constants()
    kind = node._kind  # noqa: SLF001
    parameters = {name: p[f"{SHAPE}{name}"] if name in node._traced else constants[f"{SHAPE}{name}"]  # noqa: SLF001
                  for name in (key.removeprefix(SHAPE) for key in constants if key.startswith(SHAPE))}
    found = [old_caps(kind, parameters, old) if old is not None and patch == "caps"
             else kind.samples(parameters, patch, tuple(count))
             for patch, count in zip(constants["patches"], constants["counts"], strict=True)]
    dtype = jnp.asarray(p["viscosity"]).dtype
    turn = jnp.asarray(p["part_rotation"], dtype=dtype).reshape(3, 3)
    centre = jnp.asarray(p["part_centre"], dtype=dtype)
    points = jnp.concatenate([jnp.asarray(s.points, dtype=dtype) for s in found])
    weights = jnp.concatenate([jnp.asarray(s.weights, dtype=dtype) for s in found])
    return Samples(jnp.matmul(points, turn.T, precision="highest") + centre, None, weights)


def assembled(built: Any, pose: Any, leaves: dict[str, Any], *, old: tuple[int, int] | None,
              wall: bool = True) -> dict[str, Any]:
    """``op.bem_resistance``'s first half with either surface's caps: the samples at ``pose``, the blob, the
    viscosity and the wall's correction ``G`` (zero in free space)."""
    import jax.numpy as jnp

    from mime.core import quaternion
    from mime.nodes.hydro.bem import blob_squared, wall_correction_named

    node = built.graph.get_node(op.DRAG_NODE)
    p = node._step_constants(leaves)  # noqa: SLF001
    held = node_samples(node, p, old=old)
    points = pose[:3] + jnp.matmul(held.points, quaternion.rotation_matrix(pose[3:]).T, precision="highest")
    if wall:
        kind = wall_correction_named(node.constants()["wall_kind"], node.constants()["wall_kind_version"])
        correction = kind.correction(points, {name: p[f"wall_{name}"] for name in kind.parameters},
                                     p["viscosity"]).matrix
    else:
        correction = jnp.zeros((3 * points.shape[0],) * 2, dtype=points.dtype)
    return {"points": points, "eps2": blob_squared(held.weights, p["blob"]), "viscosity": p["viscosity"],
            "centre": pose[:3], "G": correction}


def solved(parts: dict[str, Any], scale: Any = 1.0) -> Any:
    from mime.nodes.hydro.bem import resistance

    return resistance(parts["points"], parts["eps2"], parts["viscosity"], parts["centre"], scale * parts["G"])


def full_matrix(parts: dict[str, Any], with_wall: bool = True) -> Any:
    """``A + G`` (or ``A``), symmetrised, as the solve factorises it."""
    from mime.nodes.hydro.bem import stokeslet_matrix

    matrix = stokeslet_matrix(parts["points"], parts["eps2"], parts["viscosity"])
    if with_wall:
        matrix = matrix + parts["G"]
    return (matrix + matrix.T) / 2.0


class Study:
    """The closed rung at A, its assemblies and solves compiled once."""

    def __init__(self, spacing_mm: float = op.SPACING_MM) -> None:
        import jax
        import jax.numpy as jnp

        self.spacing_mm = spacing_mm
        self.rung = bem.Rung("closed", spacing_mm=spacing_mm)
        self.drag = self.rung.leaves[op.DRAG_NODE]
        built = self.rung.built
        self.solves = 0
        self.old_counts = old_counts(self.rung.node, spacing_mm * 1e-3)
        counts = {False: None, True: self.old_counts}
        self._parts = {old: jax.jit(lambda pose, drag, old=old: assembled(built, pose, drag, old=counts[old]))
                       for old in (False, True)}
        self._free = {old: jax.jit(lambda pose, drag, old=old: assembled(built, pose, drag, old=counts[old],
                                                                         wall=False))
                      for old in (False, True)}
        self._solve = jax.jit(solved)
        self._cholesky_finite = jax.jit(lambda parts: jnp.all(jnp.isfinite(jnp.linalg.cholesky(full_matrix(parts)))))
        self._eigenvalues = jax.jit(lambda parts, with_wall: jnp.linalg.eigvalsh(full_matrix(parts, with_wall)),
                                    static_argnums=1)

    def pose(self, phase: float) -> Any:
        return self.rung.pose(phase)

    def parts(self, phase: float, *, old: bool = False) -> dict[str, Any]:
        return self._parts[old](self.pose(phase), self.drag)

    def read(self, found: Any) -> np.ndarray:
        import jax

        self.solves += 1
        return np.asarray(jax.block_until_ready(found), dtype=np.float64)

    @staticmethod
    def axial(r: np.ndarray) -> float:
        return float(op.speeds(r)["axial"])

    def row(self, r: np.ndarray, *, matrix: bool = False) -> dict[str, Any]:
        """A phase's reading; with ``matrix``, the 6 x 6 resistance itself in SI (force and torque about the body's
        centre per unit translation and rotation, the body's frame turned to the world's at the pose)."""
        found = {"axial_mm_s": 1e3 * self.axial(r), "min_eigenvalue_nd": bem.smallest_eigenvalue(r),
                 "finite": bool(np.all(np.isfinite(r)))}
        return {**found, "resistance_si": r.tolist()} if matrix else found

    def samples_sha256(self, *, old: bool) -> dict[str, Any]:
        node = self.rung.node
        held = node_samples(node, node._step_constants(self.drag), old=self.old_counts if old else None)  # noqa: SLF001
        points = np.ascontiguousarray(np.asarray(held.points, dtype=np.float64))
        return {"sha256_points": hashlib.sha256(points.tobytes()).hexdigest(), "samples": int(points.shape[0]),
                "area_m2": float(np.sum(np.asarray(held.weights)))}


def _phase_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    speeds = [row["axial_mm_s"] for row in rows]
    return {"per_phase": rows, "mean_axial_mm_s": float(np.mean(speeds)),
            "min_eigenvalue_nd": float(min(row["min_eigenvalue_nd"] for row in rows)),
            "finite": all(row["finite"] for row in rows)}


# ---- stages ----------------------------------------------------------------------------------------------------------
def stage_forward(study: Study) -> dict[str, Any]:
    import jax

    started = time.perf_counter()
    out: dict[str, Any] = {"stage": "forward", "spacing_mm": study.spacing_mm, "offset_m": study.rung.offset,
                           "counts": [list(c) for c in study.rung.node.constants()["counts"]],
                           "old_cap_counts": list(study.old_counts),
                           "samples": {"fixed": study.samples_sha256(old=False), "old": study.samples_sha256(old=True)},
                           "kind_version": study.rung.node.constants()["kind_version"]}
    out["samples"]["old_equals_preregistered"] = out["samples"]["old"]["sha256_points"] == PREREGISTERED_SHA256
    library, own, old, positive_definite, cholesky_old = [], [], [], [], []
    for phase in op.PHASES:
        pose = study.pose(phase)
        library.append(study.row(study.read(study.rung._node_solve(pose, study.drag)), matrix=True))  # noqa: SLF001
        parts = study.parts(phase)
        own.append(study.row(study.read(study._solve(parts))))  # noqa: SLF001
        positive_definite.append(bool(np.asarray(jax.block_until_ready(study._cholesky_finite(parts)))))  # noqa: SLF001
        parts_old = study.parts(phase, old=True)
        old.append(study.row(study.read(study._solve(parts_old)), matrix=True))  # noqa: SLF001
        cholesky_old.append(bool(np.asarray(jax.block_until_ready(study._cholesky_finite(parts_old)))))  # noqa: SLF001
    out["fixed_library_path"] = _phase_summary(library)
    out["fixed_own_assembly"] = _phase_summary(own)
    out["old_own_assembly"] = _phase_summary(old)
    out["own_against_library_largest_relative"] = max(
        abs(a["axial_mm_s"] - b["axial_mm_s"]) / abs(b["axial_mm_s"]) for a, b in zip(own, library, strict=True))
    out["old_against_reference_relative"] = (out["old_own_assembly"]["mean_axial_mm_s"] - REFERENCE_SPEED_MM_S) \
        / REFERENCE_SPEED_MM_S
    out["fixed_against_reference_relative"] = (out["fixed_library_path"]["mean_axial_mm_s"] - REFERENCE_SPEED_MM_S) \
        / REFERENCE_SPEED_MM_S
    out["cholesky_of_A_plus_G"] = {"fixed": positive_definite, "old": cholesky_old}
    out["resistance_positive_definite_control"] = bem.positive_definite_control(
        np.asarray(study.read(study._solve(study.parts(op.PHASES[0])))))  # noqa: SLF001

    # The free-space resistance (no wall), each surface's caps, at phase 0's pose (free space has no preferred pose).
    pose = study.pose(op.PHASES[0])
    free = {label: study.read(study._solve(study._free[old](pose, study.drag)))  # noqa: SLF001
            for label, old in (("fixed", False), ("old", True))}
    diagonal = np.diag(free["fixed"]) / np.diag(free["old"]) - 1.0
    coupling = free["fixed"][0, 3] / free["old"][0, 3] - 1.0
    blocks = {}
    for name, (rows, columns) in {"force_translation": (slice(0, 3), slice(0, 3)),
                                  "force_rotation": (slice(0, 3), slice(3, 6)),
                                  "torque_rotation": (slice(3, 6), slice(3, 6))}.items():
        change = free["fixed"][rows, columns] - free["old"][rows, columns]
        blocks[name] = float(np.max(np.abs(change)) / np.max(np.abs(free["old"][rows, columns])))
    out["free_space"] = {"fixed": free["fixed"].tolist(), "old": free["old"].tolist(),
                         "diagonal_relative_change": diagonal.tolist(),
                         "axial_coupling_relative_change": float(coupling),
                         "largest_relative_change_diagonal_and_coupling": float(max(np.max(np.abs(diagonal)),
                                                                                    abs(coupling))),
                         "block_largest_change_over_block_largest_entry": blocks}
    out["environment"] = op.environment()
    out["solves"] = study.solves
    out["seconds"] = time.perf_counter() - started
    return out


def stage_scan_fine(study: Study) -> dict[str, Any]:
    """Per phase and scale: the axial speed and the resistance's smallest non-dimensional eigenvalue; the onset of
    indefiniteness at each phase (the first scale on the grid at which that eigenvalue is not positive) and its least
    over the phases (M3's ``scan_fine``, on the fixed surface)."""
    started = time.perf_counter()
    phases = []
    for phase in op.PHASES:
        parts = study.parts(phase)
        scan = [{"scale": scale, **study.row(study.read(study._solve(parts, scale)))}  # noqa: SLF001
                for scale in SCAN_FINE_SCALES]
        first = next((i for i, row in enumerate(scan) if row["min_eigenvalue_nd"] <= 0.0), None)
        phases.append({"phase": phase, "scan": scan, "onset_scale": None if first is None else scan[first]["scale"],
                       "last_positive_scale": None if first in (None, 0) else scan[first - 1]["scale"]})
    onsets = [p["onset_scale"] for p in phases if p["onset_scale"] is not None]
    return {"stage": "scan_fine", "scales": list(SCAN_FINE_SCALES), "phases": phases,
            "onset_scale_min_over_phases": min(onsets) if onsets else None,
            "phase_of_min_onset": None if not onsets else next(p["phase"] for p in phases
                                                                if p["onset_scale"] == min(onsets)),
            "environment": op.environment(), "solves": study.solves, "seconds": time.perf_counter() - started}


def stage_derivative(study: Study) -> dict[str, Any]:
    """DEFINITION 5.3's check as ``bem.py`` runs it (its step, tolerance, convergence rule and planted control), each
    input's derivative by its own forward-mode tangent (one tangent at a time, which an A100-40GB holds)."""
    import jax
    import jax.numpy as jnp

    started = time.perf_counter()
    inputs = bem.budget_inputs()["bem_rung"]
    theta = {name: jnp.asarray(spec["nominal"], dtype=jnp.float64) for name, spec in inputs.items()}
    single = jax.jit(bem._physical_residual(study.rung, "axial"))  # noqa: SLF001
    speed = float(np.asarray(single(theta))[0])

    def tangent(name: str) -> float:
        direction = {key: jnp.zeros_like(value) for key, value in theta.items()}
        direction[name] = jnp.ones_like(theta[name])
        _, found = jax.jvp(single, (theta,), (direction,))
        return float(np.asarray(found)[0])

    def central(name: str, relative: float) -> float:
        step = relative * float(theta[name])
        up = float(np.asarray(single({**theta, name: theta[name] + step}))[0])
        down = float(np.asarray(single({**theta, name: theta[name] - step}))[0])
        return (up - down) / (2 * step)

    checks: dict[str, Any] = {}
    for name in DERIVATIVE_INPUTS:
        ad = tangent(name)
        fd, coarse = central(name, bem.FD_RELATIVE_STEP), central(name, 10 * bem.FD_RELATIVE_STEP)
        miss, coarse_miss = abs(ad - fd) / abs(fd), abs(ad - coarse) / abs(coarse)
        known = {bem.FD_RELATIVE_STEP: fd, 10 * bem.FD_RELATIVE_STEP: coarse}

        def at(relative: float, name: str = name, known: dict[float, float] = known) -> float:
            return next((v for k, v in known.items() if math.isclose(k, relative)), None) or central(name, relative)

        scan = [{"relative_step": relative, "fd": (fd_at := at(relative)),
                 "relative_difference": abs(ad - fd_at) / abs(ad)} for relative in bem.FD_SCAN]
        checks[name] = {"ad": ad, "fd": fd, "relative_difference": miss, "fd_ten_times_the_step": coarse,
                        "relative_difference_ten_times_the_step": coarse_miss,
                        "agrees": miss <= bem.FD_TOLERANCE and (coarse_miss >= 50 * miss or coarse_miss <= 1e-8),
                        "scan": scan}
    rng = np.random.default_rng(bem.TABLE_PLANT_SEED + 1)
    plants = []
    for _ in range(bem.FD_PLANTS):
        name = str(rng.choice(sorted(checks)))
        factor = float(1.0 + rng.choice((-1.0, 1.0)) * rng.uniform(0.01, 0.2))
        planted = checks[name]["ad"] * factor
        plants.append({"input": name, "factor": factor,
                       "caught": abs(planted - checks[name]["fd"]) > bem.FD_TOLERANCE * abs(checks[name]["fd"])})
    return {"stage": "derivative", "speed_m_s": speed, "step_relative": bem.FD_RELATIVE_STEP,
            "tolerance": bem.FD_TOLERANCE, "checks": checks, "passes": all(c["agrees"] for c in checks.values()),
            "control": {"plants": plants, "caught_every_run": all(p["caught"] for p in plants)},
            "environment": op.environment(), "seconds": time.perf_counter() - started}


def _spectrum(values: np.ndarray) -> dict[str, Any]:
    return {"negative": int(np.sum(values < 0.0)), "least": float(values[0]), "largest": float(values[-1]),
            "least_positive": float(values[values > 0.0][0]) if np.any(values > 0.0) else None,
            "condition": float(values[-1] / values[0]) if values[0] > 0.0 else None}


def stage_eigen(study: Study, save: Any = None) -> dict[str, Any]:
    """The spectrum of ``A + G`` at every phase (fixed caps), of ``A`` alone (both caps), and of 0.1.0's ``A + G`` at
    phase 0 (the control), each saved as it is found where ``save`` is given."""
    started = time.perf_counter()
    out: dict[str, Any] = {"stage": "eigen", "units": "m/(N s)", "phases": []}

    def spectrum(parts: dict[str, Any], with_wall: bool) -> dict[str, Any]:
        return _spectrum(study.read(study._eigenvalues(parts, with_wall)))  # noqa: SLF001

    out["control_old_A_plus_G_phase0"] = spectrum(study.parts(op.PHASES[0], old=True), True)
    if save is not None:
        save(out)
    for phase in op.PHASES:
        out["phases"].append({"phase": phase, **spectrum(study.parts(phase), True)})
        if save is not None:
            save(out)
    out["negative_at_any_phase"] = any(p["negative"] > 0 for p in out["phases"])
    out["A_alone"] = {"fixed": spectrum(study.parts(op.PHASES[0]), False),
                      "old": spectrum(study.parts(op.PHASES[0], old=True), False)}
    out["environment"] = op.environment()
    out["seconds"] = time.perf_counter() - started
    return out


def run_stage(stage: str, study: Study, save: Any = None) -> dict[str, Any]:
    if stage == "forward":
        return stage_forward(study)
    if stage == "scan_fine":
        return stage_scan_fine(study)
    if stage == "derivative":
        return stage_derivative(study)
    if stage == "eigen":
        return stage_eigen(study, save)
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--spacing-mm", type=float, default=op.SPACING_MM)
    args = parser.parse_args()
    op.require_float64()
    payload = run_stage(args.stage, Study(args.spacing_mm))
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    op.write_json(target / f"capfix_{args.stage}.json", payload)


if __name__ == "__main__":
    main()
