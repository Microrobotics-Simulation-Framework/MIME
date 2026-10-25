"""Generate the 0.2 port's drag outputs on fixed random poses, once, in the port's own environment:

    MIME-port041/.venv/bin/python MIME-030/tests/differential/generate_port_drag_references.py <out_dir>

It imports the port's ``mime`` (MIME-port041 on MADDENING 0.4.1) and never MIME 0.3's, and evaluates the port's
``MLPResistanceNode`` and ``LubricationCorrectionNode`` with script A's parameters (nu 2.33, L 7.47 mm, R 1.56 mm,
the 1/4" tube's 3.175 mm, water's 1 mPa s; lubrication epsilon 0.15 mm, the body's outer radius 1.33 R for both its
radius and its nearest point, the tube along z). Float64, CPU. Its outputs are copied into ``references/`` and compared
with the 0.3 node by ``test_port_drag.py``. Not collected by pytest.

The poses, by kind: inside the surrogate's training range of offsets (up to 0.3 of the tube's radius); between that
and the clamp; past the clamp, near and beyond the wall; near the axis (1e-8 to 1e-5 m); and on it, with the body's
orientation the identity and drawn. Each at a drawn axial position, orientation and twist.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import mime  # noqa: E402
from _provenance import recorded_argv  # noqa: E402
from mime.nodes.environment.stokeslet.lubrication_node import LubricationCorrectionNode  # noqa: E402
from mime.nodes.environment.stokeslet.mlp_resistance_node import MLPResistanceNode  # noqa: E402

SEED, DT = 20261023, 5e-4
NU, L_MM, R_MM, TUBE_MM, MU, EPSILON_MM = 2.33, 7.47, 1.56, 3.175, 1e-3, 0.15
OUTER_MM = R_MM * 1.33
#: Pose kinds and how many of each: the radial offset in metres (or a fraction of the tube's radius).
KINDS = {"inside": 24, "between": 8, "past_clamp": 12, "near_axis": 12, "on_axis_identity": 2, "on_axis_turned": 2}


def unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def offset_of(kind: str, rng: np.random.Generator) -> float:
    tube = TUBE_MM * 1e-3
    clamp = (TUBE_MM / R_MM - 1.33) * 0.95 * R_MM * 1e-3
    if kind == "inside":
        return rng.uniform(0.0, 0.3) * tube
    if kind == "between":
        return rng.uniform(0.3 * tube, clamp)
    if kind == "past_clamp":
        return rng.uniform(clamp * 1.001, 1.5e-3)
    if kind == "near_axis":
        return 10.0 ** rng.uniform(-8.0, -5.0)
    return 0.0


def main(out: Path) -> None:
    rng = np.random.default_rng(SEED)
    weights = Path(mime.__file__).resolve().parents[2] / "data" / "dejongh_benchmark" / "mlp_cholesky_weights_v2.npz"
    mlp = MLPResistanceNode("mlp_drag", DT, mlp_weights_path=str(weights), nu=NU, L_UMR_mm=L_MM, R_cyl_UMR_mm=R_MM,
                            R_ves_mm=TUBE_MM, mu_Pa_s=MU)
    lub = LubricationCorrectionNode("lub", DT, R_ves_mm=TUBE_MM, R_max_body_mm=OUTER_MM, mu_Pa_s=MU,
                                    epsilon_mm=EPSILON_MM, a_eff_mm=OUTER_MM, vessel_axis=2)
    kinds, positions, orientations, velocities, omegas = [], [], [], [], []
    for index, (kind, count) in enumerate(KINDS.items()):
        for _ in range(count):
            offset, angle = offset_of(kind, rng), rng.uniform(0.0, 2 * np.pi)
            positions.append([offset * np.cos(angle), offset * np.sin(angle), rng.uniform(-0.02, 0.02)])
            orientations.append([1.0, 0.0, 0.0, 0.0] if kind == "on_axis_identity" else unit_quaternion(rng))
            velocities.append(rng.uniform(-5e-3, 5e-3, 3))
            omegas.append(rng.uniform(-80.0, 80.0, 3))
            kinds.append(index)
    arrays: dict[str, np.ndarray] = {"kind": np.array(kinds), "position": np.array(positions),
                                     "orientation": np.array(orientations), "velocity": np.array(velocities),
                                     "omega": np.array(omegas)}
    names = ("resistance", "drag_force", "drag_torque", "clamp_fired", "lub_resistance", "lub_force", "lub_torque",
             "min_gap")
    found: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for p, q, v, w in zip(positions, orientations, velocities, omegas, strict=True):
        given = {"robot_position": jnp.asarray(p), "robot_orientation": jnp.asarray(q),
                 "body_velocity": jnp.asarray(v), "body_angular_velocity": jnp.asarray(w)}
        drag = mlp.update(mlp.initial_state(), given, DT)
        corrected = lub.update(lub.initial_state(), {**{k: given[k] for k in ("robot_position", "body_velocity",
                                                                               "body_angular_velocity")},
                                                    "resistance_matrix": drag["resistance_matrix"]}, DT)
        found["resistance"].append(np.asarray(drag["resistance_matrix"]))
        found["drag_force"].append(np.asarray(drag["drag_force"]))
        found["drag_torque"].append(np.asarray(drag["drag_torque"]))
        found["clamp_fired"].append(np.asarray(drag["clamp_fired"]))
        found["lub_resistance"].append(np.asarray(corrected["corrected_resistance_matrix"]))
        found["lub_force"].append(np.asarray(corrected["drag_force"]))
        found["lub_torque"].append(np.asarray(corrected["drag_torque"]))
        found["min_gap"].append(np.asarray(corrected["min_gap"]))
    arrays.update({name: np.array(values) for name, values in found.items()})

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "port_drag.npz", allow_pickle=False, **arrays)
    repo = Path(mime.__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    import hashlib

    import maddening

    (out / "port_drag_provenance.json").write_text(json.dumps({
        "generator": "MIME-030/tests/differential/generate_port_drag_references.py",
        "argv": recorded_argv(sys.argv, repo.parent),
        "port": repo.name, "commit": commit, "maddening": maddening.__version__, "jax": jax.__version__,
        "numpy": np.__version__, "python": platform.python_version(), "x64": True, "device": jax.default_backend(),
        "seed": SEED, "kinds": KINDS, "dt": DT,
        "weights": {"path": str(weights.relative_to(repo.parent)),
                    "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()},
        "parameters": {"nu": NU, "L_UMR_mm": L_MM, "R_cyl_UMR_mm": R_MM, "R_ves_mm": TUBE_MM, "mu_Pa_s": MU,
                       "epsilon_mm": EPSILON_MM, "R_max_body_mm": OUTER_MM, "a_eff_mm": OUTER_MM,
                       "delta_floor_mm": 1e-6, "vessel_axis": 2}}, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
