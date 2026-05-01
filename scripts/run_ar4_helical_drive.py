"""Standalone runner for the AR4 + helical-UMR experiment.

Loads ``MICROROBOTICA/experiments/ar4_helical_drive/`` and steps the
graph for a short window — useful for iterative visualisation work
without launching the full MICROROBOTICA Qt viewer.

Outputs a per-frame JSON trajectory to stdout (or a file when --out is
given) with the world-frame pose of every URDF link, the magnet, the
motor rotor, and the microrobot. The schema matches what
MICROROBOTICA's ``MimePhysicsProcess`` expects on its ZMQ topic, so a
subsequent commit can swap stdout for ZMQ without changing the graph
side.

Runs on whatever JAX backend is available — GPU when present (the
project default), CPU otherwise.

Usage:

    .venv/bin/python scripts/run_ar4_helical_drive.py \\
        --duration 0.5 --out /tmp/ar4_run.jsonl

    # Verify it actually used the GPU:
    JAX_PLATFORMS=gpu .venv/bin/python scripts/run_ar4_helical_drive.py \\
        --duration 0.1 --out /tmp/ar4_run.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --- XLA setup (must be set before JAX is imported) -------------------------
# Avoid grabbing the entire GPU at JAX init so cuSolver can claim its
# own handles (otherwise ``jnp.linalg.solve`` in ``RobotArmNode`` fails
# with INTERNAL: gpusolverDnCreate). No-op on CPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
# Skip XLA's gemm/solver autotune. Picking the "best" GPU kernel for
# every shape costs many seconds on a 2060 and the difference at our
# matrix sizes (≤ 6×6 mass matrix, 6×6 inertias) is in the noise.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax
import jax.numpy as jnp
import numpy as np

# Persistent compile cache. The first ever run on a new (graph, shape,
# dtype) triple still pays the full XLA compile, but subsequent runs
# load the compiled artefact from disk in milliseconds. This is the
# single biggest knob for iterative visualisation work.
_DEFAULT_JAX_CACHE = Path.home() / ".cache" / "jax_compilation_cache"
_jax_cache = Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", str(_DEFAULT_JAX_CACHE)))
_jax_cache.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_jax_cache))
# Cache even small (sub-second) compiles — the per-step graph isn't huge
# but the first compile is multi-minute on this rig.
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = (
    REPO_ROOT.parent / "MICROROBOTICA"
    / "experiments" / "ar4_helical_drive"
)


def _load_params(experiment_dir: Path) -> dict:
    """Exec ``physics/params.py`` into a fresh namespace and return it."""
    params_path = experiment_dir / "physics" / "params.py"
    ns: dict = {}
    with params_path.open() as f:
        exec(f.read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_")}


def _load_setup(experiment_dir: Path):
    """Import ``physics/setup.py`` as a module and return its
    ``build_graph(params, experiment_dir)`` callable."""
    setup_path = experiment_dir / "physics" / "setup.py"
    sys.path.insert(0, str(experiment_dir / "physics"))
    try:
        # Use a fresh module name so reruns within one process do not
        # cache stale state.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_ar4_setup", str(setup_path),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build_graph
    finally:
        sys.path.pop(0)


def _make_link_poses_fn(tree):
    """Return a jit'd ``q -> (N, 7)`` link-pose evaluator with the
    static kinematic tree captured. Compiled once, cached forever.

    This avoids the trace-per-frame cost of calling ``link_world_poses``
    fresh on every sample (which re-traces on every call because the
    Python function takes a non-JAX tree as an argument)."""
    from mime.control.kinematics import link_world_poses

    @jax.jit
    def _eval(q, base_pose):
        return link_world_poses(tree, q, base_pose)
    return _eval


def _frame_payload(t: float, state: dict, link_poses_fn,
                   base_pose, ee_offset) -> dict:
    """Serialise one timestep into the MICROROBOTICA ResultFrame schema.

    Per-actor entries are keyed by the actor names declared in the
    experiment's ``experiment.yaml`` ``scene.actors`` map. The arm's
    per-link world poses are computed by a pre-jit'd closure
    (``link_poses_fn``) so this call does not retrace.
    """
    def to_list(x):
        return np.asarray(x).reshape(-1).tolist()

    arm = state["arm"]
    body = state["body"]
    rotor = state["motor"]
    magnet = state["ext_magnet"]

    payload: dict = {"t": float(t), "actors": {}}
    actors = payload["actors"]

    link_poses = np.asarray(link_poses_fn(arm["joint_angles"], base_pose))
    for i, pose in enumerate(link_poses):
        actors[f"arm_link_{i}"] = {
            "translate": to_list(pose[0:3]),
            "orient": to_list(pose[3:7]),
        }

    actors["motor_rotor"] = {
        "translate": to_list(rotor["rotor_pose_world"][0:3]),
        "orient": to_list(rotor["rotor_pose_world"][3:7]),
        "scalar:angle_rad": float(rotor["angle"]),
    }
    actors["magnet"] = {
        "translate": to_list(rotor["rotor_pose_world"][0:3]),
        "orient": to_list(rotor["rotor_pose_world"][3:7]),
        "vec:field_T": to_list(magnet["field_vector"]),
    }
    actors["microrobot"] = {
        "translate": to_list(body["position"]),
        "orient": to_list(body["orientation"]),
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT_DIR,
                    help="Path to experiment directory.")
    ap.add_argument("--duration", type=float, default=0.2,
                    help="Simulated seconds to run.")
    ap.add_argument("--motor-freq-hz", type=float, default=10.0,
                    help="Constant motor commanded velocity, in Hz.")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON-lines output file (default: stdout).")
    ap.add_argument("--frames", type=int, default=200,
                    help="Number of frames to emit. Linearly spaced over the duration.")
    ap.add_argument("--no-coupling-group", action="store_true",
                    help="Disable the body↔magnet Gauss-Seidel coupling group "
                         "in the dejongh-new-chain graph (falls back to "
                         "staggered back-edges; ~10× faster, 1-step phase lag).")
    args = ap.parse_args()

    experiment_dir = args.experiment.resolve()
    if not experiment_dir.exists():
        print(f"experiment dir not found: {experiment_dir}", file=sys.stderr)
        return 2

    print(f"# JAX backend: {jax.default_backend()}", file=sys.stderr)
    print(f"# JAX devices: {jax.devices()}", file=sys.stderr)

    params = _load_params(experiment_dir)
    if args.no_coupling_group:
        params["USE_COUPLING_GROUP"] = False
        print("# coupling group: DISABLED (back-edge staggering)", file=sys.stderr)
    build_graph = _load_setup(experiment_dir)
    # Two ``build_graph`` conventions are in the wild:
    #   * MICROROBOTICA-side experiments: ``build_graph(params, experiment_dir)``
    #   * MIME-side experiments (dejongh_confined-style): ``build_graph(params)``
    # The MIME convention resolves paths internally via ``__file__``.
    import inspect
    sig = inspect.signature(build_graph)
    if len(sig.parameters) >= 2:
        gm = build_graph(params, str(experiment_dir))
    else:
        gm = build_graph(params)
    print(f"# graph nodes: {list(gm._nodes.keys())}", file=sys.stderr)

    # Two conventions in the wild: the MICROROBOTICA-side AR4
    # experiment names the timestep ``TIMESTEP_S``; the MIME-side
    # dejongh_confined-style experiments name it ``DT_PHYS``.
    dt = float(params.get("TIMESTEP_S", params.get("DT_PHYS")))
    if dt is None or dt <= 0:
        raise SystemExit(
            "experiment params must define TIMESTEP_S or DT_PHYS"
        )
    n_steps = int(round(args.duration / dt))
    sample_every = max(1, n_steps // args.frames)
    omega_des = jnp.float32(2.0 * np.pi * args.motor_freq_hz)
    arm_node = gm._nodes["arm"].node
    n_dof = arm_node._num_joints
    base_pose = jnp.asarray(arm_node.params["base_pose_world"], dtype=jnp.float32)
    ee_offset = jnp.asarray(
        arm_node.params["end_effector_offset_in_link"], dtype=jnp.float32,
    )
    # Pre-jit the link-pose evaluator so the per-frame serialiser does
    # not retrace on every sample (the static kinematic tree is captured
    # in the closure).
    link_poses_fn = _make_link_poses_fn(arm_node._tree)

    # NOTE on coupling-group iteration count: the dejongh deliverable
    # A.2 confirmed Gauss-Seidel converges to float32 floor in ~10
    # iterations on this loop. Mutating ``CouplingGroup.max_iterations``
    # in-place would make XLA's while_loop body smaller, but it also
    # touches a frozen dataclass and risks marking the GraphManager
    # dirty (forcing a re-compile on next step). Leaving the
    # MADDENING-shipped default (20 iterations) for now — the persistent
    # compile cache amortises the extra trace-time cost over reruns.
    arm_torque_zero = jnp.zeros((n_dof,), dtype=jnp.float32)
    # Stable dtypes / shapes — promoted once, outside the step loop, so
    # JAX's traced cache key never sees Python-side weak-type drift.
    omega_des32 = jnp.asarray(omega_des, dtype=jnp.float32)
    external_inputs = {
        "motor": {"commanded_velocity": omega_des32},
        "arm": {"commanded_joint_torques": arm_torque_zero},
    }

    out_handle = args.out.open("w") if args.out else sys.stdout

    # ---- Pre-warm: pay the JIT-compile cost once, separately, so the
    # timing loop measures steady-state throughput.
    print("# pre-warming step (first call compiles + caches XLA)...",
          file=sys.stderr)
    t_compile_start = time.time()
    state = gm.step(external_inputs=external_inputs)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        state,
    )
    t_compile = time.time() - t_compile_start
    print(f"# pre-warm: {t_compile:.2f} s (compile + first call)",
          file=sys.stderr)

    try:
        t0 = time.time()
        sim_t = 0.0
        last_state = state  # carry pre-warm result through
        for k in range(n_steps):
            state = gm.step(external_inputs=external_inputs)
            sim_t += dt
            if k % sample_every == 0:
                # Block on the leaves we serialise so the JSON writer
                # does not race ahead of XLA's async dispatch.
                state["arm"]["joint_angles"].block_until_ready()
                state["motor"]["rotor_pose_world"].block_until_ready()
                state["body"]["position"].block_until_ready()
                payload = _frame_payload(
                    sim_t, state, link_poses_fn, base_pose, ee_offset,
                )
                out_handle.write(json.dumps(payload) + "\n")
            last_state = state
        # Final block to capture full wall time.
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            last_state,
        )
        wall = time.time() - t0
    finally:
        if args.out:
            out_handle.close()

    real_per_sim = wall / max(args.duration, 1e-12)
    per_step_ms = (wall / max(n_steps, 1)) * 1e3
    print(
        f"# {n_steps} steady-state steps in {wall:.2f} s wall — "
        f"{per_step_ms:.2f} ms/step, {real_per_sim:.1f} s wall / s simulated",
        file=sys.stderr,
    )
    print(f"# JAX compile cache: {_jax_cache}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
