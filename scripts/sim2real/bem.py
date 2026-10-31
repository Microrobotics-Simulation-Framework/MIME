"""Phase 7j-a's boundary-element rung at A's operating point, static: the BEM (``Drag.BEM`` in the tube's wall kind)
at the floored pose, read at eight phases of the body's turn, and every model-internal check made on it. Each part
returns a JSON-able dict; ``modal_app.py`` runs them on an A100, and any part runs here on the CPU too:

    JAX_ENABLE_X64=1 python scripts/sim2real/bem.py --part ladder --out DIR

Parts:

- ``ladder``: the BEM rung, closed and lateral, at the default discretisation and one finer spacing; the BEM with the
  surrogate rung's lubrication added by the script (the rung carries none); the node against this script's own
  assembly (a control: equal to rounding).
- ``offsets``: the BEM and the surrogate (``--weights``) at MIME-ANO-114's offsets and the floored one: the known
  gap of the surrogate, re-measured through the corrected table.
- ``blob``: the blob at 0.35, 0.5 and 0.7 of the spacing, at two spacings.
- ``margin``: the margin from 0.05 to 0.2 mm, the floored offset moving with it.
- ``table``: the wall's correction scaled by the table's measured read error (0.23%), and its control: tables scaled
  by 2% to 10% (20 draws) must each move the speed outside that band.
- ``budget``: the sensitivities by ``maddening.sysid.fim(..., scale="nominal")`` over the physical inputs (their
  ``ParamSpec`` bounds the plausible ranges of ``budget_inputs.json``), the raw Jacobian's signs, the same over
  ``gm.params``' own leaves, the finite-difference check of the derivative (DEFINITION 5.3's) with its planted
  control, and the viscosity worked example with its planted control.
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
import operating_point as op  # noqa: E402

HERE = Path(__file__).resolve().parent
FINER_SPACING_MM = 0.140
BLOBS = (0.35, 0.5, 0.7)
MARGINS_MM = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2)
#: MIME-ANO-114's offsets (fractions of the tube's radius) and the floored one script A rests at (margin 0.1 mm).
OFFSETS = (0.20, 0.25, 0.28, 0.30, 0.32)
#: The table's read error, its gate's measured maximum relative to a block's largest entry (MIME-ANO-117).
READ_ERROR = 0.0023
TABLE_PLANTS, TABLE_PLANT_SEED = 20, 20261030
#: The step is small because the speed is strongly curved in the shape's inputs at a coarse mesh (the dry run's FD
#: error fell as the step squared from 1e-3 to 1e-7): the check also records a step ten times larger, and requires the
#: miss to fall by fifty times or more between them (the derivative's own convergence).
FD_RELATIVE_STEP, FD_TOLERANCE = 1e-6, 1e-4
FD_PLANTS = 20
#: A sensitivity is read as zero when its column is under this fraction of the speed (the viscosity check).
ZERO_COLUMN = 1e-9
SOLVES = {"count": 0}


class Rung:
    """One BEM experiment at one discretisation, its static solve compiled once."""

    def __init__(self, patches: str = "closed", spacing_mm: float = op.SPACING_MM, blob: float = op.BLOB,
                 margin_mm: float = op.MARGIN_MM, tube_radius_mm: float = op.TUBE_RADIUS_M * 1e3) -> None:
        import jax

        from mime.build._maddening import node_state

        self.patches, self.spacing_mm, self.blob, self.margin_mm = patches, spacing_mm, blob, margin_mm
        self.tube_radius_m = tube_radius_mm * 1e-3
        self.built = op.experiment("bem", patches=patches, spacing_mm=spacing_mm, blob=blob, margin_mm=margin_mm,
                                   tube_radius_mm=tube_radius_mm).build()
        self.leaves = op.leaves_of(self.built)
        self.node = self.built.graph.get_node(op.DRAG_NODE)
        self.offset = float(op.floored_offset(self.leaves[op.BODY_NODE]))
        self._node_solve = jax.jit(lambda pose, drag: op.static_resistance(self.built, pose, drag))
        self._own_solve = jax.jit(lambda pose, drag, scale, extra: op.bem_resistance(
            self.built, pose, drag, correction_scale=scale, extra=extra))
        state = node_state(self.built.graph, op.DRAG_NODE)
        self._accepted = jax.jit(lambda pose, drag: self.node.update(state, {"pose": pose}, 1e-3,
                                                                     params=drag)["wall_accepted"])

    @property
    def samples(self) -> int:
        return int(self.node.samples(self.leaves[op.DRAG_NODE]).points.shape[0])

    def samples_sha256(self) -> str:
        """The samples in the body's frame, hashed as the preregistration hashes them (a platform's own rounding of
        the kind's quadrature changes the hash, not the mesh)."""
        import hashlib

        points = np.asarray(self.node.samples(self.leaves[op.DRAG_NODE]).points, dtype=np.float64)
        return hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()

    def label(self) -> str:
        return f"{self.patches}/{self.spacing_mm:.3f}mm/blob{self.blob}/margin{self.margin_mm}mm"

    def resistance(self, pose: Any, *, own: bool = False, scale: float = 1.0, extra: Any = None) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        SOLVES["count"] += 1
        drag = self.leaves[op.DRAG_NODE]
        if own or scale != 1.0 or extra is not None:
            found = self._own_solve(pose, drag, scale, jnp.zeros((6, 6)) if extra is None else extra)
        else:
            found = self._node_solve(pose, drag)
        return np.asarray(jax.block_until_ready(found), dtype=np.float64)

    def accepted(self, pose: Any) -> bool:
        return bool(np.asarray(self._accepted(pose, self.leaves[op.DRAG_NODE])))

    def pose(self, phase: float, offset: float | None = None) -> Any:
        return op.floored_pose(np.float64(self.offset if offset is None else offset), phase)

    def nearest_gap(self, pose: Any) -> dict[str, float]:
        """The samples' nearest approach to the wall at ``pose``, and the analytic crest's."""
        import jax.numpy as jnp

        from mime.core import quaternion

        held = self.node.samples(self.leaves[op.DRAG_NODE])
        points = np.asarray(pose[:3] + jnp.matmul(held.points, quaternion.rotation_matrix(pose[3:]).T,
                                                  precision="highest"))
        farthest = float(np.hypot(points[:, 1], points[:, 2]).max())
        gap = self.tube_radius_m - farthest
        return {"nearest_sample_gap_m": gap, "gap_over_spacing": gap / (self.spacing_mm * 1e-3),
                "gap_over_blob_length": gap / float(math.sqrt(float(_eps_squared(self)))),
                "nearest_sample_over_R": farthest / self.tube_radius_m}

    def phases(self, *, offset: float | None = None, scale: float = 1.0, extra: Any = None,
               phases: tuple[float, ...] = op.PHASES) -> dict[str, Any]:
        """The speeds at each phase and their means, with the resistance at each."""
        rows: list[dict[str, float]] = []
        matrices = []
        accepted = True
        for phase in phases:
            pose = self.pose(phase, offset)
            r = self.resistance(pose, scale=scale, extra=extra(pose) if callable(extra) else extra)
            matrices.append(r.tolist())
            rows.append({k: float(v) for k, v in op.speeds(r).items()} | {"min_eigenvalue_nd": smallest_eigenvalue(r)})
            accepted = accepted and bool(np.all(np.isfinite(r)))
        out: dict[str, Any] = {"label": self.label(), "samples": self.samples, "unknowns": 3 * self.samples,
                               "samples_sha256": self.samples_sha256(),
                               "offset_m": self.offset if offset is None else offset,
                               "offset_over_R": (self.offset if offset is None else offset) / op.TUBE_RADIUS_M,
                               "phases": list(phases), "per_phase": rows, "resistance": matrices,
                               "finite": accepted, **self.nearest_gap(self.pose(0.0, offset))}
        out["wall_accepted_phase0"] = self.accepted(self.pose(0.0, offset))
        out["positive_definite_every_phase"] = all(row["min_eigenvalue_nd"] > 0.0 for row in rows)
        for variant in op.VARIANTS:
            out[variant] = op.phase_mean([row[variant] for row in rows])
        return out


def nondimensional(r: np.ndarray, viscosity: float = 1e-3, radius: float = 1.56e-3) -> np.ndarray:
    """``r`` in units of ``mu R``, ``mu R^2`` and ``mu R^3`` (R the body's radius), symmetrically scaled so that its
    eigenvalues are comparable across the blocks."""
    scale = np.diag([math.sqrt(viscosity * radius)] * 3 + [math.sqrt(viscosity * radius ** 3)] * 3)
    inverse = np.linalg.inv(scale)
    return inverse @ r @ inverse


def smallest_eigenvalue(r: np.ndarray) -> float:
    """The non-dimensional resistance's smallest eigenvalue: a resistance dissipates, so it is positive definite,
    and a negative eigenvalue is a discretisation outside its validity (the check added after the first read, when
    the lateral surface's resistance at one phase and the closed one at a 0.05 mm margin came out indefinite)."""
    held = nondimensional(np.asarray(r, dtype=np.float64))
    return float(np.linalg.eigvalsh((held + held.T) / 2.0).min())


def positive_definite_control(r: np.ndarray, draws: int = 20, seed: int = TABLE_PLANT_SEED + 2) -> dict[str, Any]:
    """The check's control: ``r`` with its smallest direction pushed below zero by 1% to 50% of its eigenvalue
    (20 draws) must read as not positive definite on every draw."""
    rng = np.random.default_rng(seed)
    held = nondimensional(np.asarray(r, dtype=np.float64))
    values, vectors = np.linalg.eigh((held + held.T) / 2.0)
    caught = []
    for _ in range(draws):
        push = (1.0 + float(rng.uniform(0.01, 0.5))) * values[0]
        planted = held - push * np.outer(vectors[:, 0], vectors[:, 0])
        caught.append(float(np.linalg.eigvalsh(planted).min()) <= 0.0)
    return {"draws": draws, "caught_every_run": all(caught)}


def _eps_squared(rung: Rung) -> Any:
    from mime.nodes.hydro.bem import blob_squared

    held = rung.node.samples(rung.leaves[op.DRAG_NODE])
    return blob_squared(held.weights, rung.leaves[op.DRAG_NODE]["blob"])


def _lubrication(margin_mm: float = op.MARGIN_MM):
    """The surrogate rung's lubrication at a pose, from script A's own surrogate experiment at the margin."""
    import jax

    surrogate = op.experiment("surrogate", margin_mm=margin_mm).build()
    held = jax.jit(lambda pose: op.lubrication_matrix(surrogate, pose))
    return lambda pose: held(pose)


# ---- parts ---------------------------------------------------------------------------------------------------------
def part_ladder(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"cases": []}
    lubrication = _lubrication()
    for spacing in (op.SPACING_MM, FINER_SPACING_MM):
        for patches in op.PATCHES:
            rung = Rung(patches, spacing)
            started = time.perf_counter()
            case = rung.phases()
            case["seconds"] = time.perf_counter() - started
            if spacing == op.SPACING_MM:
                case["with_lubrication"] = {k: v for k, v in rung.phases(extra=lubrication).items()
                                            if k in op.VARIANTS or k == "per_phase"}
                pose = rung.pose(0.0)
                node, own = rung.resistance(pose), rung.resistance(pose, own=True)
                case["control_node_against_own_assembly"] = float(np.max(np.abs(node - own)) / np.max(np.abs(node)))
                case["control_positive_definite"] = positive_definite_control(node)
                lub = np.asarray(lubrication(pose))
                case["lubrication_share_of_diagonal"] = (np.diag(lub) / np.diag(node)).tolist()
            out["cases"].append(case)
    return out


def part_offsets(args: argparse.Namespace) -> dict[str, Any]:
    import jax

    surrogate = op.experiment("surrogate", weights=args.weights, lubrication=False).build()
    surrogate_solve = jax.jit(lambda pose: op.static_resistance(surrogate, pose))
    out: dict[str, Any] = {"weights": args.weights, "cases": []}
    for patches in op.PATCHES:
        rung = Rung(patches)
        for fraction in (*OFFSETS, rung.offset / op.TUBE_RADIUS_M):
            offset = fraction * op.TUBE_RADIUS_M
            case = {k: v for k, v in rung.phases(offset=offset).items() if k != "resistance"}
            rows = [{k: float(v) for k, v in op.speeds(np.asarray(surrogate_solve(rung.pose(phase, offset)))).items()}
                    for phase in op.PHASES]
            case["surrogate"] = {variant: op.phase_mean([row[variant] for row in rows]) for variant in op.VARIANTS}
            out["cases"].append(case)
    return out


def part_blob(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"cases": []}
    for spacing in (op.SPACING_MM, FINER_SPACING_MM):
        for patches in op.PATCHES:
            for blob in BLOBS:
                case = Rung(patches, spacing, blob).phases()
                case.pop("resistance")
                out["cases"].append(case)
    return out


def part_margin(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"cases": []}
    for patches in op.PATCHES:
        for margin in MARGINS_MM:
            case = Rung(patches, margin_mm=margin).phases()
            case.pop("resistance")
            case["margin_mm"] = margin
            out["cases"].append(case)
    return out


def part_table(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(TABLE_PLANT_SEED)
    out: dict[str, Any] = {"read_error": READ_ERROR, "cases": []}
    for patches in op.PATCHES:
        rung = Rung(patches)
        nominal = rung.phases()
        low, high = (rung.phases(scale=1.0 + s * READ_ERROR) for s in (-1.0, 1.0))
        case: dict[str, Any] = {"label": rung.label()}
        for variant in op.VARIANTS:
            centre = nominal[variant]["mean"]
            case[variant] = {"nominal": centre, "low": low[variant]["mean"], "high": high[variant]["mean"],
                             "band_half_width": max(abs(low[variant]["mean"] - centre),
                                                    abs(high[variant]["mean"] - centre))}
        band = case["axial"]["band_half_width"]
        pose = rung.pose(0.0)
        base = float(op.speeds(rung.resistance(pose))["axial"])
        plants = []
        for _ in range(TABLE_PLANTS):
            planted = float(1.0 + rng.choice((-1.0, 1.0)) * rng.uniform(0.02, 0.10))
            moved = abs(float(op.speeds(rung.resistance(pose, scale=planted))["axial"]) - base)
            # The band at phase 0 alone, for a like-for-like comparison with the plant at phase 0.
            plants.append({"scale": planted, "speed_moved": moved, "caught": moved > band})
        case["control"] = {"plants": plants, "caught_every_run": all(p["caught"] for p in plants),
                           "band_used": band}
        out["cases"].append(case)
    return out


# ---- the budget ------------------------------------------------------------------------------------------------------
def budget_inputs() -> dict[str, Any]:
    return json.loads((HERE / "budget_inputs.json").read_text())


def _physical_residual(rung: Rung, variant: str = "axial", extra: Any = None):
    """The phase-mean speed as a function of the physical inputs: each input written into every leaf of
    ``gm.params`` it reaches (the tube's radius into the wall's and the contact's, the body's radius and modulation
    into the shape's and the contact's extent), the floored offset following them."""
    import jax
    import jax.numpy as jnp

    drag0, body0 = rung.leaves[op.DRAG_NODE], rung.leaves[op.BODY_NODE]
    phases = jnp.asarray(op.PHASES)

    def residual(theta: dict[str, Any]) -> Any:
        drag = {**drag0, "viscosity": theta["viscosity"], "shape_radius": theta["body_radius"],
                "shape_length": theta["length"], "shape_wavenumber": theta["wavenumber"],
                "shape_modulation": theta["modulation"], "wall_radius": theta["tube_radius"]}
        body = {**body0, "surface_radius": theta["tube_radius"],
                "extent": (1.0 + theta["modulation"]) * theta["body_radius"], "margin": theta["margin"]}
        offset = op.floored_offset(body)

        def one(phase: Any) -> Any:
            pose = jnp.stack([0.0 * offset, 0.0 * offset, -offset, jnp.cos(phase / 2), jnp.sin(phase / 2),
                              0.0 * offset, 0.0 * offset])
            r = op.bem_resistance(rung.built, pose, drag, extra=extra(pose, theta) if extra is not None else None)
            found = op.speeds(r)
            return jnp.stack([found["axial"], found["lateral_free"], found["wall_held"]])

        rows = jax.lax.map(one, phases)
        means = jnp.mean(rows, axis=0)
        return means if variant == "all" else means[op.VARIANTS.index(variant)][None]

    return residual


def _leaf_residual(rung: Rung):
    """The phase-mean axial speed as a function of ``gm.params``' own leaves, each on its own: the modularity reading
    (one physical input reaching several leaves shows as several columns)."""
    import jax
    import jax.numpy as jnp

    drag0, body0 = rung.leaves[op.DRAG_NODE], rung.leaves[op.BODY_NODE]
    phases = jnp.asarray(op.PHASES)

    def residual(params: dict[str, Any]) -> Any:
        drag = {**drag0, **params["nodes"][op.DRAG_NODE]}
        body = {**body0, **params["nodes"][op.BODY_NODE]}
        offset = op.floored_offset(body)

        def one(phase: Any) -> Any:
            pose = jnp.stack([0.0 * offset, 0.0 * offset, -offset, jnp.cos(phase / 2), jnp.sin(phase / 2),
                              0.0 * offset, 0.0 * offset])
            return op.speeds(op.bem_resistance(rung.built, pose, drag))["axial"]

        return jnp.mean(jax.lax.map(one, phases))[None]

    return residual


def _report(report: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("param_names", "fim", "eigvals", "rank", "crb", "cond", "least_identifiable", "value_scaled",
                 "zero_scaled", "integer_excluded", "precision_limited"):
        value = getattr(report, name, None)
        if value is not None:
            try:
                out[name] = np.asarray(value).tolist()
            except (TypeError, ValueError):
                out[name] = str(value)
    return out


def part_budget(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from maddening.core.params import ParamSpec

    from mime.build._maddening import parameter_leaves
    from mime.build._maddening import _held as held  # noqa: PLC2701 - MADDENING's untyped calls go through here

    sysid: Any = held(__import__("maddening.sysid", fromlist=["fim"]))
    inputs = budget_inputs()["bem_rung"]
    rung = Rung(getattr(args, "patches", "closed"))
    theta = {name: jnp.asarray(spec["nominal"], dtype=jnp.float64) for name, spec in inputs.items()}
    specs = {name: ParamSpec(bounds=(spec["low"], spec["high"]), units=spec["units"]) for name, spec in inputs.items()}
    out: dict[str, Any] = {"label": rung.label(), "inputs": inputs}

    started = time.perf_counter()
    residual = _physical_residual(rung, "all")
    jacobian = jax.jacfwd(residual)(theta)
    SOLVES["count"] += 8
    out["jacobian_seconds"] = time.perf_counter() - started
    value = np.asarray(residual(theta))
    SOLVES["count"] += 8
    out["speed"] = dict(zip(op.VARIANTS, value.tolist(), strict=True))
    out["across_range"] = {variant: {name: float(np.asarray(jacobian[name])[i]) * (inputs[name]["high"]
                                                                                    - inputs[name]["low"])
                                     for name in inputs} for i, variant in enumerate(op.VARIANTS)}
    out["derivative"] = {variant: {name: float(np.asarray(jacobian[name])[i]) for name in inputs}
                         for i, variant in enumerate(op.VARIANTS)}

    # The linear reading's validity: the speed at each end of each range, against the derivative times the width.
    ends: dict[str, Any] = {}
    for name, spec in inputs.items():
        at = {end: np.asarray(residual({**theta, name: jnp.asarray(spec[end], dtype=jnp.float64)})).tolist()
              for end in ("low", "high")}
        SOLVES["count"] += 16
        secant = {variant: at["high"][i] - at["low"][i] for i, variant in enumerate(op.VARIANTS)}
        ends[name] = {"speed_at_low": dict(zip(op.VARIANTS, at["low"], strict=True)),
                      "speed_at_high": dict(zip(op.VARIANTS, at["high"], strict=True)), "secant_across_range": secant,
                      "secant_over_linear": {v: secant[v] / out["across_range"][v][name]
                                             if out["across_range"][v][name] != 0.0 else None for v in op.VARIANTS}}
    out["range_ends"] = ends

    started = time.perf_counter()
    report = sysid.fim(_physical_residual(rung, "axial"), theta, scale="nominal", specs=specs)
    SOLVES["count"] += 8
    out["fim_seconds"] = time.perf_counter() - started
    out["fim_axial"] = _report(report)

    # gm.params' own leaves, each on its own.
    graph_leaves = parameter_leaves(rung.built.graph)
    chosen = {op.DRAG_NODE: ("viscosity", "shape_radius", "shape_length", "shape_wavenumber", "shape_modulation",
                             "wall_radius"),
              op.BODY_NODE: ("surface_radius", "extent", "margin")}
    params = {"nodes": {node: {key: jnp.asarray(graph_leaves[node][key], dtype=jnp.float64) for key in keys}
                        for node, keys in chosen.items()}}
    leaf_bounds = budget_inputs()["leaf_bounds"]
    leaf_specs = {"nodes": {node: {key: ParamSpec(bounds=tuple(leaf_bounds[f"{node}.{key}"]))
                                   for key in keys} for node, keys in chosen.items()}}
    report = sysid.fim(_leaf_residual(rung), params, scale="nominal", specs=leaf_specs)
    SOLVES["count"] += 8
    out["fim_leaves_axial"] = _report(report)
    out["graph_leaves_holding_each_input"] = budget_inputs()["leaves_of_input"]

    # DEFINITION 5.3's check: the derivative against central finite differences, and its planted control.
    single = _physical_residual(rung, "axial")
    checks: dict[str, Any] = {}
    def central(name: str, relative: float) -> float:
        step = relative * float(theta[name])
        up = float(np.asarray(single({**theta, name: theta[name] + step}))[0])
        down = float(np.asarray(single({**theta, name: theta[name] - step}))[0])
        SOLVES["count"] += 16
        return (up - down) / (2 * step)

    for name in ("margin", "tube_radius", "body_radius"):
        fd, coarse = central(name, FD_RELATIVE_STEP), central(name, 10 * FD_RELATIVE_STEP)
        ad = out["derivative"]["axial"][name]
        miss, coarse_miss = abs(ad - fd) / abs(fd), abs(ad - coarse) / abs(coarse)
        checks[name] = {"ad": ad, "fd": fd, "relative_difference": miss, "fd_ten_times_the_step": coarse,
                        "relative_difference_ten_times_the_step": coarse_miss,
                        "agrees": miss <= FD_TOLERANCE and (coarse_miss >= 50 * miss or coarse_miss <= 1e-8)}
    rng = np.random.default_rng(TABLE_PLANT_SEED + 1)
    plants = []
    for _ in range(FD_PLANTS):
        name = str(rng.choice(sorted(checks)))
        factor = float(1.0 + rng.choice((-1.0, 1.0)) * rng.uniform(0.01, 0.2))
        planted = checks[name]["ad"] * factor
        plants.append({"input": name, "factor": factor,
                       "caught": abs(planted - checks[name]["fd"]) > FD_TOLERANCE * abs(checks[name]["fd"])})
    out["fd_check"] = {"step_relative": FD_RELATIVE_STEP, "tolerance": FD_TOLERANCE, "checks": checks,
                       "control": {"plants": plants, "caught_every_run": all(p["caught"] for p in plants)}}

    # The worked example: in a torque-locked Stokes model the speed does not depend on the viscosity. Its control
    # plants a resistance that does not scale with it (the lubrication's at the nominal viscosity, held fixed, as a
    # friction would be), which must give the viscosity a column.
    viscosity_column = abs(out["across_range"]["axial"]["viscosity"]) / abs(out["speed"]["axial"])
    lubrication = _lubrication()
    planted = _physical_residual(rung, "axial", extra=lambda pose, th: jax.lax.stop_gradient(lubrication(pose)))
    planted_jacobian = jax.jacfwd(planted)(theta)
    SOLVES["count"] += 8
    planted_speed = float(np.asarray(planted(theta))[0])
    SOLVES["count"] += 8
    planted_column = abs(float(np.asarray(planted_jacobian["viscosity"])[0])
                         * (inputs["viscosity"]["high"] - inputs["viscosity"]["low"])) / abs(planted_speed)
    out["viscosity_worked_example"] = {
        "column_over_speed": viscosity_column, "reads_as_zero": viscosity_column < ZERO_COLUMN,
        "control": {"planted_column_over_speed": planted_column, "caught": planted_column >= ZERO_COLUMN},
        "threshold": ZERO_COLUMN}
    return out


#: The finite-difference steps of the diagnostic scan (added after the first read: at 0.178 mm the closed surface's
#: differences at 1e-6 and 1e-5 disagreed with the derivative and with each other).
FD_SCAN = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def part_fd_scan(args: argparse.Namespace) -> dict[str, Any]:
    """A diagnostic, not a gate: the derivative against central differences over five steps, for the closed and the
    lateral surface, and the viscosity's column on each (the worked example's floor)."""
    import jax
    import jax.numpy as jnp

    inputs = budget_inputs()["bem_rung"]
    theta = {name: jnp.asarray(spec["nominal"], dtype=jnp.float64) for name, spec in inputs.items()}
    out: dict[str, Any] = {"steps": list(FD_SCAN), "cases": []}
    for patches in op.PATCHES:
        rung = Rung(patches)
        single = _physical_residual(rung, "axial")
        jacobian = jax.jacfwd(single)(theta)
        speed = float(np.asarray(single(theta))[0])
        SOLVES["count"] += 16
        case: dict[str, Any] = {"label": rung.label(), "speed": speed,
                                "viscosity_column_over_speed": abs(float(np.asarray(jacobian["viscosity"])[0])
                                                                   * (inputs["viscosity"]["high"]
                                                                      - inputs["viscosity"]["low"])) / abs(speed),
                                "inputs": {}}
        for name in ("margin", "tube_radius", "body_radius"):
            ad = float(np.asarray(jacobian[name])[0])
            row = []
            for relative in FD_SCAN:
                step = relative * float(theta[name])
                up = float(np.asarray(single({**theta, name: theta[name] + step}))[0])
                down = float(np.asarray(single({**theta, name: theta[name] - step}))[0])
                SOLVES["count"] += 16
                fd = (up - down) / (2 * step)
                row.append({"relative_step": relative, "fd": fd, "relative_difference": abs(ad - fd) / abs(ad)})
            case["inputs"][name] = {"ad": ad, "scan": row}
        out["cases"].append(case)
    return out


PARTS = {"fd_scan": part_fd_scan, "ladder": part_ladder, "offsets": part_offsets, "blob": part_blob, "margin": part_margin,
         "table": part_table, "budget": part_budget}


def run_part(name: str, weights: str = op.DEFAULT_WEIGHTS, patches: str = "closed") -> dict[str, Any]:
    """``patches`` chooses the surface of the budget part (the lateral one added after the first read, when the
    closed surface's derivative failed its finite-difference check)."""
    op.require_float64()
    SOLVES["count"] = 0
    started = time.perf_counter()
    out = PARTS[name](argparse.Namespace(weights=weights, patches=patches))
    out.update(part=name, environment=op.environment(), seconds=time.perf_counter() - started,
               bem_solves=SOLVES["count"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--part", choices=sorted(PARTS), required=True)
    parser.add_argument("--out", required=True, help="the directory the part's JSON is written to")
    parser.add_argument("--weights", default=op.DEFAULT_WEIGHTS, help="the surrogate's weights (an asset's name)")
    parser.add_argument("--patches", default="closed", choices=sorted(op.PATCHES), help="the budget part's surface")
    args = parser.parse_args()
    out = run_part(args.part, args.weights, args.patches)
    suffix = "" if args.patches == "closed" else f"_{args.patches}"
    op.write_json(Path(args.out) / f"bem_{args.part}{suffix}.json", out)


if __name__ == "__main__":
    main()
