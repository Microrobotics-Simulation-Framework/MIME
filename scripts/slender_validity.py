"""The rft rung's validity lines against MIME's MRS rung on the same body: each point of
``docs/data/slender_validity/spec.json`` built twice, with ``Drag.MRS`` (the reference, its settings the spec's) and with
``Drag.RFT`` at its defaults, each resistance read at the body's starting pose, the misses written to the record
(``record.json`` beside the spec), with each point's force points, build and step times. The rft node is read directly
where its build would refuse it (an indefinite resistance with a wall), so the record says what it would have
delivered. Float64, one process; run it in a heavy slot (an MRS of a slender filament is thousands of force points).

Usage: python scripts/slender_validity.py [record.json] [--only name,name] [--ladder]"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mime.build._maddening import node_state, parameter_leaves
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.interactions import Drag
from mime.nodes.hydro.mrs._quadrature import floored_least_eigenvalue
from mime.scene import Capsule, Ellipsoid, Helix, Scene, Tube, Z
from mime.scene._helix import HelixTurn
from mime.things import Body, Medium, Part, State, Structure
from mime.units import DIMENSIONLESS, Quantity, kg, kg_m2, m, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

SPEC = Path(__file__).resolve().parents[1] / "docs" / "data" / "slender_validity" / "spec.json"
VISCOSITY = 1e-3


def kind_of(point: dict[str, Any]) -> tuple[Any, np.ndarray]:
    """The point's kind and where its body's centre of mass sits so that its axis's middle is at the origin."""
    if point["kind"] == "helix":
        settings = {key: float(point[key]) for key in ("radius", "pitch", "turns", "filament_radius")}
        kind = Helix(radius=m(settings["radius"]), pitch=m(settings["pitch"]),
                     turns=Quantity(settings["turns"], DIMENSIONLESS), filament_radius=m(settings["filament_radius"]))
        return kind, np.asarray(HelixTurn(settings).centroid(), dtype=np.float64)
    if point["kind"] == "capsule":
        return Capsule(radius=m(point["radius"]), height=m(point["height"]), axis=(1.0, 0.0, 0.0)), np.zeros(3)
    if point["kind"] == "ellipsoid":
        return Ellipsoid(semi_axes=m(tuple(point["semi_axes"]))), np.zeros(3)
    raise ValueError(point["kind"])


def filament(point: dict[str, Any]) -> float:
    if point["kind"] == "ellipsoid":
        return float(min(point["semi_axes"]))
    return float(point["filament_radius"] if point["kind"] == "helix" else point["radius"])


def prolate(a: float, b: float, mu: float = VISCOSITY) -> np.ndarray:
    """A prolate spheroid's resistance about its centre, its long axis along x: Kim & Karrila's closed form."""
    e = math.sqrt(1 - b * b / (a * a))
    log = math.log((1 + e) / (1 - e))
    xa = 8 / 3 * e ** 3 / (-2 * e + (1 + e * e) * log)
    ya = 16 / 3 * e ** 3 / (2 * e + (3 * e * e - 1) * log)
    xc = 4 / 3 * e ** 3 * (1 - e * e) / (2 * e - (1 - e * e) * log)
    yc = 4 / 3 * e ** 3 * (2 - e * e) / (-2 * e + (1 + e * e) * log)
    return np.diag([6 * math.pi * mu * a * xa] + [6 * math.pi * mu * a * ya] * 2
                   + [8 * math.pi * mu * a ** 3 * xc] + [8 * math.pi * mu * a ** 3 * yc] * 2)


def experiment(point: dict[str, Any], law: Any) -> Experiment:
    kind, start = kind_of(point)
    built = Experiment(point["name"], scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    if point["wall"] == "tube":
        built.add_structure(Structure("tube", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=m(point["tube_radius"]),
                                                            wall=mm(0.1), length=mm(20.0))))
        built.add_medium(Medium("water", fills="tube.lumen"))
    else:
        built.add_medium(Medium("water"))
    built.declare("water.viscosity", mPa_s(1.0), reason="water")
    built.add_body(Body("body", dynamics=RigidInertial(), parts={"shell": Part(kind)}, initial=State(
        position=m(tuple(float(c) for c in start)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    built.declare("body.mass", kg(1e-9), reason="a sub-millimetre body's order")
    built.declare("body.inertia", kg_m2((1e-16, 1e-16, 1e-16)), reason="a sub-millimetre body's order")
    built.attach(law, body="body", medium="water")
    return built


def about_the_origin(resistance: np.ndarray, centre: np.ndarray) -> np.ndarray:
    d = -centre
    shift = np.eye(6)
    shift[:3, 3:] = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])
    return shift.T @ resistance @ shift


def read(built: Any) -> tuple[np.ndarray, Any]:
    node = built.graph.get_node("drag:body")
    out = node.update(node_state(built.graph, "drag:body"), {"pose": node_state(built.graph, "body")["pose"]}, 1e-3,
                      params=parameter_leaves(built.graph)["drag:body"])
    return np.asarray(out["resistance"], dtype=np.float64), node


def rft_direct(point: dict[str, Any]) -> tuple[np.ndarray, bool, int]:
    """The rft resistance at the start with no build check (the plan's arguments, the node made directly)."""
    from mime.build import BuildRefused

    try:
        resistance, node = read(experiment(point, Drag.RFT(wall_kind=point["wall"])).build())
        return resistance, False, node.point_count
    except BuildRefused as refused:
        codes = [r.code for r in refused.refusals]
        if codes != ["drag.rft-indefinite"]:
            raise
    from mime.nodes.hydro import RFTDragNode
    from mime.nodes.hydro.mrs import wall_correction_named
    from mime.registry import centreline_counts

    kind, start = kind_of(point)
    settings = {key: float(point[key]) for key in ("radius", "pitch", "turns", "filament_radius")}
    correction = wall_correction_named(point["wall"], "0.1.0")
    constants = correction.prepare({"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "radius": point["tube_radius"]},
                                   jnp.float64)
    counts = centreline_counts(kind.centreline(settings), settings, filament(point))
    node = RFTDragNode("drag:body", 1e-3, viscosity=VISCOSITY, kind=kind.name, kind_version=kind.version,
                       counts=counts, part_centre=(0.0, 0.0, 0.0),
                       part_rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), coefficients="lighthill",
                       wall_kind=point["wall"], wall_kind_version="0.1.0",
                       **{f"wall_{k}": v for k, v in constants.items()}, **{f"shape_{k}": v for k, v in settings.items()})
    out = node.update(node.initial_state(), {"pose": jnp.asarray([*start, 1.0, 0.0, 0.0, 0.0])}, 1e-3,
                      params=node.params_pytree())
    return np.asarray(out["resistance"], dtype=np.float64), True, node.point_count


def mrs(point: dict[str, Any], spacing: float, blob: float) -> tuple[np.ndarray, int, float, float]:
    t0 = time.time()
    built = experiment(point, Drag.MRS(spacing=m(spacing * filament(point)), blob=blob,
                                       quadrature_refinement=max(1, math.ceil(2.0 / blob - 1e-9)),
                                       wall_kind=point["wall"])).build()
    t1 = time.time()
    resistance, node = read(built)
    return resistance, node.sample_count, t1 - t0, time.time() - t1


def misses(rft_r: np.ndarray, mrs_r: np.ndarray, centre: np.ndarray) -> dict[str, float]:
    scale = np.sqrt(np.abs(np.diag(mrs_r)))
    scaled = (rft_r - mrs_r) / (scale[:, None] * scale[None, :])
    a, b = about_the_origin(rft_r, centre), about_the_origin(mrs_r, centre)
    speed_rft, speed_mrs = -a[0, 3] / a[0, 0], -b[0, 3] / b[0, 0]
    chiral = abs(b[0, 3]) > 1e-3 * math.sqrt(abs(b[0, 0] * b[3, 3]))  # a straight body has no speed to miss
    return {"speed_rft": float(speed_rft), "speed_mrs": float(speed_mrs),
            "speed_miss": float(abs(speed_rft / speed_mrs - 1.0)) if chiral else None,
            "resistance_miss": float(np.max(np.abs(scaled))),
            "translation_miss": float(np.max(np.abs(scaled[:3, :3]))),
            "rotation_miss": float(np.max(np.abs(scaled[3:, 3:]))),
            "coupling_miss": float(np.max(np.abs(scaled[:3, 3:]))),
            "diagonal_rft_over_mrs": (np.diag(rft_r) / np.diag(mrs_r)).tolist(),
            "rft_least": float(floored_least_eigenvalue(jnp.asarray(rft_r))),
            "mrs_least": float(floored_least_eigenvalue(jnp.asarray(mrs_r)))}


def rounded_up(value: float) -> float:
    """``value`` rounded up to two significant figures: the rule's."""
    exponent = math.floor(math.log10(abs(value))) - 1
    return round(math.ceil(value / 10.0 ** exponent - 1e-9) * 10.0 ** exponent, 12)


def families(spec: dict[str, Any], record: dict[str, Any]) -> list[dict[str, Any]]:
    """The rft rung's lines by the spec's rule: the record's points grouped by kind, shape ratios, wall and (with a
    wall) the gap, each family's line its largest miss rounded up to two significant figures, over the slenderness
    its points span; a refused point claims no line."""
    from mime.nodes.hydro._rft_drag import shape_ratios

    points = {p["name"]: p for p in spec["points"] + spec.get("exact_points", [])}
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in record["points"]:
        point = points[row["name"]]
        if row["rft_refused"]:
            continue
        ratios = tuple(sorted(shape_ratios(point["kind"], point).items()))
        gap = 0.0 if point["wall"] == "free_space" else round(
            (point["tube_radius"] - point["radius"] - filament(point)) / filament(point), 9)
        grouped.setdefault((point["kind"], ratios, point["wall"], gap), []).append(row)
    found = []
    for (kind, ratios, wall, gap), rows in sorted(grouped.items()):
        speeds = [row["speed_miss"] for row in rows if row["speed_miss"] is not None]
        found.append({"kind": kind, "shape": dict(ratios), "wall": wall, "gap": gap,
                      "slenderness": [min(r["slenderness"] for r in rows), max(r["slenderness"] for r in rows)],
                      "speed": rounded_up(max(speeds)) if speeds else None,
                      "resistance": rounded_up(max(r["resistance_miss"] for r in rows)),
                      "points": [r["name"] for r in rows]})
    return found


def main() -> int:
    jax.config.update("jax_enable_x64", True)  # the run is float64; importing the module changes no process setting
    parser = argparse.ArgumentParser()
    parser.add_argument("record", nargs="?", default=str(SPEC.with_name("record.json")))
    parser.add_argument("--only", default="")
    parser.add_argument("--ladder", action="store_true")
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    only = set(filter(None, args.only.split(",")))
    import hashlib

    record: dict[str, Any] = {"spec": "spec.json",
                              "spec_written": "after the runs, with its rule; the hash below is of the spec as it stands",
                              "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
                              "points": [], "ladder": []}
    path = Path(args.record)
    if path.exists():
        record = json.loads(path.read_text())
    done = {row["name"] for row in record["points"]}
    for point in spec["points"]:
        if (only and point["name"] not in only) or point["name"] in done:
            continue
        _, centre = kind_of(point)
        reference, count, built_s, step_s = mrs(point, 0.79, 0.25)
        t0 = time.time()
        rft_r, refused, rft_points = rft_direct(point)
        row = {"name": point["name"], "mrs_points": count, "mrs_build_s": built_s, "mrs_step_s": step_s,
               "rft_points": rft_points, "rft_s": time.time() - t0, "rft_refused": refused,
               **misses(rft_r, reference, centre)}
        if point["kind"] == "helix":
            turn = HelixTurn({k: float(point[k]) for k in ("radius", "pitch", "turns", "filament_radius")})
            row["slenderness"] = float(turn.length) / filament(point)
            if point["wall"] == "tube":
                row["confinement"] = (float(point["radius"]) + filament(point)) / float(point["tube_radius"])
        else:
            row["slenderness"] = (float(point["height"]) + 2 * filament(point)) / filament(point)
        record["points"].append(row)
        print(json.dumps(row), flush=True)
        path.write_text(json.dumps(record, indent=1) + "\n")
    for point in spec.get("exact_points", []):
        if (only and point["name"] not in only) or point["name"] in done:
            continue
        a, b = max(point["semi_axes"]), min(point["semi_axes"])
        long = int(np.argmax(point["semi_axes"]))
        order = [long] + [i for i in range(3) if i != long]
        exact = prolate(a, b)  # its long axis along x: permute into the kind's axes
        reference = np.zeros((6, 6))
        for i, k in enumerate(order):
            for j, kk in enumerate(order):
                reference[k, kk], reference[3 + k, 3 + kk] = exact[i, j], exact[3 + i, 3 + j]
        rft_r, refused, rft_points = rft_direct(point)
        row = {"name": point["name"], "reference": "exact", "rft_points": rft_points, "rft_refused": refused,
               **misses(rft_r, reference, np.zeros(3)), "slenderness": 2.0 * a / b}
        record["points"].append(row)
        print(json.dumps(row), flush=True)
        path.write_text(json.dumps(record, indent=1) + "\n")
    if args.ladder and not record["ladder"]:
        ladder = spec["resolution_ladder"]
        point = next(p for p in spec["points"] if p["name"] == ladder["point"])
        _, centre = kind_of(point)
        for spacing in ladder["spacings_in_filament_radii"]:
            for blob in (ladder["blobs"] if spacing == 0.79 else [0.25]):
                resistance, count, built_s, step_s = mrs(point, spacing, blob)
                b = about_the_origin(resistance, centre)
                record["ladder"].append({"spacing": spacing, "blob": blob, "points": count, "build_s": built_s,
                                         "speed": float(-b[0, 3] / b[0, 0]),
                                         "diagonal": np.diag(resistance).tolist()})
                print(json.dumps(record["ladder"][-1]), flush=True)
                path.write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
