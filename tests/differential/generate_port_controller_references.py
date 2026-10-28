"""Generate the 0.2 line's A0 controller's outputs on a recorded input sequence, once, in the port's own environment:

    MIME-port041/.venv/bin/python MIME-030-7b1/tests/differential/generate_port_controller_references.py <out_dir>

It loads the port's ``experiments/ar4_helical_drive/control/controller.py`` and its ``physics/params.py`` as the 0.2
runner did (the parameters executed, the controller module loaded from its file) and never MIME 0.3: each tick it
calls the controller's ``compute(params, step, state)`` with a state built from the sequence, and records the joint
positions its inverse kinematics reaches (``q_target``), its filtered target's position and orientation, and the joint
torques it writes. The sequence is drawn once from a seed: a body moving along x within and beyond the reach
envelope, turning about axes near x, with the arm's joint positions near its home pose; the same sequence twice, with
the orientation feedback off (A0's setting) and on. Run with ``JAX_ENABLE_X64=0`` (A0's float32 reference) and
``JAX_ENABLE_X64=1`` (its mixed one); the precision names the output. Compared with ``mime.control.ArmTracking`` by
``test_port_controller.py``. CPU. Not collected by pytest.
"""

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import mime
from _provenance import recorded_argv

TICKS, SEED = 120, 20261028
#: The 0.2 line's A0 rig, below the port's root.
RIG = Path("experiments") / "ar4_helical_drive"


def load_controller(rig: Path):
    spec = importlib.util.spec_from_file_location("a0_controller", rig / "control" / "controller.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameters(rig: Path) -> dict:
    """``params.py`` as the 0.2 runner executed it: its public names of plain types."""
    namespace: dict = {}
    exec((rig / "physics" / "params.py").read_text(), namespace)  # noqa: S102 - the port's own parameter file
    return {k: v for k, v in namespace.items() if not k.startswith("_")
            and isinstance(v, (int, float, str, bool, list, tuple, dict))}


def sequence(home: np.ndarray) -> dict[str, np.ndarray]:
    """The recorded inputs, one row per tick: the body's position, orientation (w, x, y, z) and angular velocity, and
    the arm's joint positions and velocities."""
    rng = np.random.default_rng(SEED)
    x = np.cumsum(rng.normal(0.0, 4e-3, TICKS)) - 0.02
    positions = np.stack([x, rng.normal(0.0, 3e-4, TICKS), -1e-3 + rng.normal(0.0, 2e-4, TICKS)], axis=1)
    axes = np.array([1.0, 0.0, 0.0]) + rng.normal(0.0, 0.15, (TICKS, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.cumsum(rng.uniform(0.0, 0.4, TICKS))
    orientations = np.concatenate([np.cos(angles / 2)[:, None], np.sin(angles / 2)[:, None] * axes], axis=1)
    spins = axes * rng.uniform(0.0, 60.0, (TICKS, 1))
    joints = home[None, :] + rng.normal(0.0, 0.02, (TICKS, home.size))
    return {"position": positions, "orientation": orientations, "angular_velocity": spins,
            "joint_positions": joints, "joint_velocities": rng.normal(0.0, 0.1, (TICKS, home.size))}


def run(rig: Path, params: dict, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    controller = load_controller(rig)._TrackingController(params)
    found: dict[str, list[np.ndarray]] = {"q_target": [], "target_position": [], "target_orientation": [],
                                          "torques": []}
    for tick in range(TICKS):
        state = {"body": {"position": jnp.asarray(inputs["position"][tick]),
                          "orientation": jnp.asarray(inputs["orientation"][tick]),
                          "angular_velocity": jnp.asarray(inputs["angular_velocity"][tick])},
                 "arm": {"joint_angles": jnp.asarray(inputs["joint_positions"][tick]),
                         "joint_velocities": jnp.asarray(inputs["joint_velocities"][tick])}}
        written = controller.compute(params, tick, state)
        found["q_target"].append(np.asarray(controller._q_target))
        found["target_position"].append(np.asarray(controller._target_pos_world))
        found["target_orientation"].append(np.asarray(controller._target_quat_world))
        found["torques"].append(np.asarray(written["arm"]["commanded_joint_torques"]))
    return {name: np.asarray(rows) for name, rows in found.items()}


def main(out: Path) -> None:
    repo = Path(mime.__file__).resolve().parents[2]
    rig = repo / RIG
    x64 = bool(jax.config.jax_enable_x64)
    label = "mixed_f64" if x64 else "f32"
    params = parameters(rig)
    inputs = sequence(np.asarray(params["ARM_HOME_RAD"], dtype=np.float64))
    arrays = {f"input/{name}": value for name, value in inputs.items()}
    for feedback in (False, True):
        held = {**params, "ENABLE_ORIENTATION_FEEDBACK": feedback}
        arm = "feedback_on" if feedback else "feedback_off"
        arrays |= {f"{arm}/{name}": value for name, value in run(rig, held, inputs).items()}
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"port_controller_{label}.npz", allow_pickle=False, **arrays)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    import maddening

    settings = {key: params[key] for key in sorted(params) if key.startswith(("CONTROL_", "ARM_HOME", "BASE_POSE",
                                                                              "END_EFFECTOR", "GRAVITY", "URDF"))}
    (out / f"port_controller_{label}_provenance.json").write_text(json.dumps({
        "generator": "MIME-030-7b1/tests/differential/generate_port_controller_references.py",
        "argv": recorded_argv(sys.argv, repo.parent), "port": repo.name, "commit": commit,
        "controller": f"{RIG.as_posix()}/control/controller.py, _TrackingController.compute",
        "parameters": settings, "maddening": maddening.__version__, "jax": jax.__version__,
        "numpy": np.__version__, "python": platform.python_version(), "x64": x64, "device": jax.default_backend(),
        "seed": SEED, "ticks": TICKS}, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
