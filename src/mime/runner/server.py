"""MIME experiment server — ZMQ-based subprocess for MICROROBOTICA.

Reads experiment.yaml, builds the MADDENING GraphManager, runs the
simulation loop, and publishes ResultFrame JSON over ZMQ PUB.
Commands (params, stop, reload) are received via ZMQ REP.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger("mime.runner")


def _load_module_from_path(module_name: str, file_path: Path):
    """Import a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _execute_params(params_path: Path) -> dict:
    """Execute params.py and return the namespace as a dict of scalars."""
    namespace: dict[str, Any] = {}
    exec(params_path.read_text(), namespace)
    # Filter to serializable types only
    return {
        k: v for k, v in namespace.items()
        if not k.startswith("_")
        and isinstance(v, (int, float, str, bool, list, tuple, dict))
    }


def _state_to_result_frame(
    sim_time: float,
    state: dict[str, dict],
    actor_config: dict,
    flow_field_config: dict | None = None,
) -> dict:
    """Convert GraphManager state to MICROROBOTICA ResultFrame JSON.

    The format matches result_frame.h: positions as {"x","y","z"},
    orientations as {"w","x","y","z"}, scalars as doubles,
    meshes.vertexColors as [{"x":R,"y":G,"z":B}, ...].
    """
    frame: dict[str, Any] = {
        "simTime": sim_time,
        "positions": {},
        "orientations": {},
        "scalars": {},
        "meshes": {},
    }

    for actor_name, actor_spec in actor_config.items():
        node_state = state.get(actor_name)
        if node_state is None:
            continue

        state_fields = actor_spec.get("state_fields", [])

        if "position" in state_fields:
            pos = np.asarray(node_state.get("position", np.zeros(3)))
            frame["positions"][actor_name] = {
                "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            }

        if "orientation" in state_fields:
            q = np.asarray(node_state.get("orientation", [1, 0, 0, 0]))
            frame["orientations"][actor_name] = {
                "w": float(q[0]), "x": float(q[1]),
                "y": float(q[2]), "z": float(q[3]),
            }

        if "field_vector" in state_fields:
            fv = np.asarray(node_state.get("field_vector", np.zeros(3)))
            frame["positions"][actor_name] = {
                "x": float(fv[0]), "y": float(fv[1]), "z": float(fv[2]),
            }

    # Extract scalar diagnostics
    for actor_name in state:
        node_state = state[actor_name]
        if "drag_torque" in node_state:
            dt = np.asarray(node_state["drag_torque"])
            frame["scalars"]["drag_torque_z"] = float(dt[2])
        if "angular_velocity" in node_state:
            av = np.asarray(node_state["angular_velocity"])
            frame["scalars"]["omega_z"] = float(av[2])

    # Flow field vertex colors (if configured)
    if flow_field_config is not None:
        lbm_state = state.get("lbm_fluid")
        if lbm_state is not None and "f" in lbm_state:
            try:
                from mime.nodes.environment.lbm.d3q19 import compute_macroscopic
                _, velocity = compute_macroscopic(lbm_state["f"])
                vel_np = np.asarray(velocity)
                # Take z-midplane slice (x-y plane perpendicular to vessel axis)
                nz = vel_np.shape[2]
                mid_z = nz // 2
                vel_slice = vel_np[:, :, mid_z, :]
                mag = np.sqrt(np.sum(vel_slice ** 2, axis=-1))

                # Apply viridis colormap
                try:
                    import matplotlib.cm as cm
                    cmap = cm.get_cmap("viridis")
                    vmin, vmax = float(mag.min()), float(mag.max())
                    if vmax - vmin > 1e-30:
                        normed = (mag - vmin) / (vmax - vmin)
                    else:
                        normed = np.zeros_like(mag)
                    colors = []
                    for val in normed.ravel():
                        rgba = cmap(float(val))
                        colors.append({
                            "x": float(rgba[0]),
                            "y": float(rgba[1]),
                            "z": float(rgba[2]),
                        })
                except ImportError:
                    colors = [{"x": 0.0, "y": 0.0, "z": 0.5}] * mag.size

                frame["meshes"]["flow_field"] = {"vertexColors": colors}
            except Exception as e:
                logger.debug("Flow field extraction failed: %s", e)

    return frame


def run_experiment(yaml_path: str) -> None:
    """Main entry point: parse YAML, build graph, run with ZMQ publishing."""
    yaml_path = Path(yaml_path).resolve()
    experiment_dir = yaml_path.parent

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading experiment: %s", yaml_path)

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # --- Load physics ---
    setup_path = experiment_dir / config["physics"]["setup"]
    params_path = experiment_dir / config["physics"]["params"]

    params = _execute_params(params_path)
    logger.info("Parameters: %s", {k: v for k, v in params.items()
                                    if isinstance(v, (int, float))})

    setup_module = _load_module_from_path("experiment_setup", setup_path)
    if not hasattr(setup_module, "build_graph"):
        raise AttributeError(
            f"{setup_path} must define build_graph(params: dict) -> GraphManager"
        )

    from maddening.core.graph_manager import GraphManager
    gm: GraphManager = setup_module.build_graph(params)
    logger.info("Graph built: %d nodes", len(gm._nodes))

    # --- Export graph.json ---
    graph_json_path = experiment_dir / "graph.json"
    graph_dict = gm.to_dict()
    with open(graph_json_path, "w") as f:
        json.dump(graph_dict, f, indent=2, default=str)
    logger.info("Graph topology exported: %s", graph_json_path)

    # --- Compile ---
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    logger.info("Graph compiled")

    # --- Scene actors config ---
    scene_config = config.get("scene", {})
    actor_config = scene_config.get("actors", {})
    analysis_config = scene_config.get("analysis", {})
    flow_field_config = analysis_config.get("flow_field") if analysis_config else None

    # --- ZMQ setup ---
    try:
        import zmq
    except ImportError:
        logger.error("pyzmq not installed. Install with: pip install pyzmq")
        sys.exit(1)

    ctx = zmq.Context()

    # REP socket for commands (port 5555)
    rep_socket = ctx.socket(zmq.REP)
    rep_socket.bind("tcp://*:5555")
    rep_socket.setsockopt(zmq.RCVTIMEO, 0)  # non-blocking poll

    # PUB socket for ResultFrame streaming (port 5556)
    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5556")

    logger.info("ZMQ sockets bound: REP=5555, PUB=5556")

    # --- Publish params on startup ---
    params_msg = json.dumps({"type": "params", "data": params})
    pub_socket.send_string(params_msg)

    # --- Run loop ---
    running = True
    sim_time = 0.0
    step_count = 0
    dt_physical = list(gm._nodes.values())[0].timestep

    def handle_signal(signum, frame):
        nonlocal running
        logger.info("Received signal %d, stopping", signum)
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("Starting simulation loop (dt=%.3e s)...", dt_physical)
    t0 = time.perf_counter()

    while running:
        # Poll for commands (non-blocking)
        try:
            cmd_raw = rep_socket.recv_string(zmq.NOBLOCK)
            cmd = json.loads(cmd_raw)
            cmd_type = cmd.get("command", cmd.get("type", ""))

            if cmd_type == "stop":
                logger.info("Received stop command")
                rep_socket.send_string(json.dumps({"status": "ok"}))
                running = False
                continue
            elif cmd_type == "params":
                new_params = cmd.get("data", {})
                params.update(new_params)
                logger.info("Updated params: %s", list(new_params.keys()))
                rep_socket.send_string(json.dumps({"status": "ok"}))
            elif cmd_type == "reload_controller":
                logger.info("Controller reload requested (not yet implemented)")
                rep_socket.send_string(json.dumps({"status": "ok"}))
            else:
                rep_socket.send_string(json.dumps({"status": "unknown_command"}))
        except zmq.Again:
            pass  # No command waiting

        # Step the simulation
        gm.step()
        step_count += 1
        sim_time += dt_physical

        # Build and publish ResultFrame
        full_state = {}
        for node_name in gm._nodes:
            full_state[node_name] = gm.get_node_state(node_name)

        result_frame = _state_to_result_frame(
            sim_time, full_state, actor_config, flow_field_config,
        )
        pub_socket.send_string(json.dumps(result_frame))

        # Progress logging
        if step_count % 1000 == 0:
            elapsed = time.perf_counter() - t0
            rate = step_count / elapsed
            logger.info(
                "Step %d: sim_time=%.4f s, %.1f steps/s",
                step_count, sim_time, rate,
            )

    # Cleanup
    elapsed_total = time.perf_counter() - t0
    logger.info(
        "Simulation stopped after %d steps (%.1f s, %.1f steps/s)",
        step_count, elapsed_total,
        step_count / max(elapsed_total, 1e-6),
    )
    pub_socket.close()
    rep_socket.close()
    ctx.term()
