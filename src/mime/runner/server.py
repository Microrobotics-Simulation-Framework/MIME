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
) -> dict:
    """Convert GraphManager state to MICROROBOTICA ResultFrame JSON.

    Generic: extracts positions, orientations, and field vectors based
    on actor_config. Experiment-specific scalars and flow data are
    added separately via hooks.
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
    t_build_start = time.perf_counter()
    gm: GraphManager = setup_module.build_graph(params)
    t_build = time.perf_counter() - t_build_start
    logger.info("Graph built: %d nodes (%.1fs)", len(gm._nodes), t_build)

    # --- Load controller ---
    controller_module = None
    control_config = config.get("control", {})
    if "controller" in control_config:
        controller_path = experiment_dir / control_config["controller"]
        controller_module = _load_module_from_path("controller", controller_path)
        if not hasattr(controller_module, "get_external_inputs"):
            logger.warning(
                "%s has no get_external_inputs() — running without controller",
                controller_path,
            )
            controller_module = None
        else:
            logger.info("Controller loaded: %s", controller_path)

    # --- Load experiment hooks ---
    from mime.runner.hooks import load_hooks, ExperimentHooks, HookContext
    hooks = load_hooks(config, experiment_dir)

    # --- Export graph.json ---
    graph_json_path = experiment_dir / "graph.json"
    graph_dict = gm.to_dict()
    with open(graph_json_path, "w") as f:
        json.dump(graph_dict, f, indent=2, default=str)
    logger.info("Graph topology exported: %s", graph_json_path)

    # --- Compile ---
    import warnings
    t_compile_start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    t_compile = time.perf_counter() - t_compile_start
    logger.info("Graph compiled (%.1fs)", t_compile)

    # --- Scene config ---
    scene_config = config.get("scene", {})
    actor_config = scene_config.get("actors", {})
    env_config = scene_config.get("environment", {})

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

    # --- Streaming setup (optional, enabled by MIME_ENABLE_STREAMING=1) ---
    streaming_observer = None
    if os.environ.get("MIME_ENABLE_STREAMING", "0") == "1":
        try:
            from mime.viz.stage_bridge import StageBridge
            from mime.viz.hydra_viewport import HydraStormViewport
            from mime.viz.streaming_observer import StreamingObserver
            from maddening.cloud.selkies_session import SelkiesSession
            from maddening.cloud.streaming import StreamConfig

            stream_config = config.get("stream", {})
            width = stream_config.get("width", 1280)
            height = stream_config.get("height", 720)
            target_fps = stream_config.get("target_fps", 2.0)

            bridge = StageBridge()
            # Register actors from scene config
            for actor_name, actor_spec in actor_config.items():
                fields = actor_spec.get("state_fields", [])
                prim_path = actor_spec.get("prim_path", f"/World/Actors/{actor_name}")
                if "position" in fields and "orientation" in fields:
                    bridge.register_robot(actor_name, prim_path=prim_path)
                elif "field_vector" in fields:
                    bridge.register_field(actor_name, prim_path=prim_path)

            viewport = HydraStormViewport(width=width, height=height)
            viewport.set_stage(bridge.stage)

            selkies = SelkiesSession()
            stream_info = selkies.start(StreamConfig(
                width=width, height=height,
                fps=int(target_fps),
                bitrate_kbps=stream_config.get("bitrate_kbps", 4000),
            ))

            dt_physical = list(gm._nodes.values())[0].timestep
            frame_skip = StreamingObserver.compute_frame_skip(
                dt_physical, target_fps,
            )

            streaming_observer = StreamingObserver(
                bridge, viewport, selkies, frame_skip=frame_skip,
            )

            logger.info(
                "Streaming enabled: %dx%d @ %d fps (frame_skip=%d), "
                "signaling: %s",
                width, height, int(target_fps), frame_skip,
                stream_info.signaling_url,
            )
        except ImportError as e:
            logger.warning("Streaming disabled: %s", e)
        except Exception as e:
            logger.warning("Streaming setup failed: %s", e)

    # --- Recording setup (optional) ---
    recorder = None
    recording_config = config.get("recording")
    if recording_config and recording_config.get("enabled", False):
        try:
            from mime.viz.stage_bridge import StageBridge
            from mime.viz.usd_recorder import USDRecorderObserver
            from mime.core.geometry import CylinderGeometry

            rec_bridge = StageBridge()

            # Register actors — use mesh_generator hook if available
            for actor_name, actor_spec in actor_config.items():
                fields = actor_spec.get("state_fields", [])
                prim_path = actor_spec.get(
                    "prim_path", f"/World/Actors/{actor_name}",
                )
                if "position" in fields and "orientation" in fields:
                    mesh_data = None
                    if hooks.mesh_generator:
                        try:
                            result = hooks.mesh_generator(params)
                            if result is not None:
                                mesh_data = result
                                logger.info(
                                    "Mesh hook: %d verts, %d tris",
                                    len(result[0]), len(result[1]),
                                )
                        except Exception as e:
                            logger.warning("Mesh hook failed: %s", e)
                    rec_bridge.register_robot(
                        actor_name, mesh_data=mesh_data, prim_path=prim_path,
                    )
                elif "field_vector" in fields:
                    rec_bridge.register_field(
                        actor_name, prim_path=prim_path,
                    )

            # Register environment geometry
            for env_name, env_spec in env_config.items():
                if env_spec.get("source") == "parametric":
                    geom = env_spec.get("geometry", {})
                    if geom.get("type") == "cylinder":
                        vessel_geom = CylinderGeometry(
                            diameter_m=geom["radius"] * 2,
                            length_m=geom["length"],
                            axis=geom.get("axis", "Z").lower(),
                        )
                        env_path = env_spec.get(
                            "prim_path", f"/World/Environment/{env_name}",
                        )
                        rec_bridge.add_parametric_geometry(
                            vessel_geom, prim_path=env_path,
                        )

            # Scene dressing via hook
            if hooks.scene_setup:
                try:
                    hooks.scene_setup(rec_bridge, params, actor_config, env_config)
                except Exception as e:
                    logger.warning("Scene setup hook failed: %s", e)

            # Flow extractor via hook
            flow_fn = None
            if hooks.flow_extractor:
                def _wrap_flow(state):
                    ctx = HookContext(
                        state=state, params=params, dt=dt_physical,
                        step=step_count, ext_inputs=ext_inputs,
                    )
                    return hooks.flow_extractor(ctx)
                flow_fn = _wrap_flow

            output_path = str(experiment_dir / recording_config["output"])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            recorder = USDRecorderObserver(
                rec_bridge,
                output_path,
                sampling_interval=recording_config.get("sampling_interval", 10),
                fps=recording_config.get("fps", 24.0),
                live=False,
                flow_extractor=flow_fn,
            )
            logger.info(
                "Recording enabled: %s (every %d steps)",
                output_path, recorder.sampling_interval,
            )
        except ImportError as e:
            logger.warning("Recording disabled: %s", e)
        except Exception as e:
            logger.warning("Recording setup failed: %s", e)

    # --- Run loop ---
    running = True
    sim_time = 0.0
    step_count = 0
    ext_inputs = None
    dt_physical = list(gm._nodes.values())[0].timestep

    def handle_signal(signum, frame):
        nonlocal running
        logger.info("Received signal %d, stopping", signum)
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("Starting simulation loop (dt=%.3e s)...", dt_physical)
    t0 = time.perf_counter()
    t_first_step = None  # JIT compilation timing

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
        t_step_start = time.perf_counter()
        if controller_module is not None:
            ext_inputs = controller_module.get_external_inputs(params, step_count)
            gm.step(ext_inputs)
        else:
            gm.step()
        step_count += 1
        sim_time += dt_physical

        # Log JIT compilation time (first step)
        if t_first_step is None:
            t_first_step = time.perf_counter() - t_step_start
            logger.info("First step (XLA JIT): %.1fs", t_first_step)

        # Build and publish ResultFrame
        full_state = {}
        for node_name in gm._nodes:
            full_state[node_name] = gm.get_node_state(node_name)

        result_frame = _state_to_result_frame(
            sim_time, full_state, actor_config,
        )

        # Derived scalars via hook
        if hooks.scalar_extractor:
            try:
                ctx = HookContext(
                    state=full_state, params=params, dt=dt_physical,
                    step=step_count, ext_inputs=ext_inputs,
                )
                result_frame["scalars"].update(hooks.scalar_extractor(ctx))
            except Exception as e:
                logger.debug("Scalar extractor failed: %s", e)

        pub_socket.send_string(json.dumps(result_frame))

        # Streaming observer (render + WebRTC push)
        if streaming_observer is not None:
            streaming_observer(
                sim_time, dt_physical, full_state, {}, {}, {},
            )

        # USD recording
        if recorder is not None:
            recorder(sim_time, dt_physical, full_state, {}, {}, {})

        # Progress logging
        if step_count % 1000 == 0:
            elapsed = time.perf_counter() - t0
            rate = step_count / elapsed
            logger.info(
                "Step %d: sim_time=%.4f s, %.1f steps/s",
                step_count, sim_time, rate,
            )

    # Cleanup — performance summary
    elapsed_total = time.perf_counter() - t0
    steps_per_sec = step_count / max(elapsed_total, 1e-6)
    sim_time_per_wall_sec = sim_time / max(elapsed_total, 1e-6)
    ms_per_step = elapsed_total / max(step_count, 1) * 1000
    logger.info("=== PERFORMANCE SUMMARY ===")
    logger.info("  Graph build:      %.1fs", t_build)
    logger.info("  Graph compile:    %.1fs", t_compile)
    logger.info("  XLA JIT (1st step): %.1fs", t_first_step or 0)
    logger.info("  Total steps:      %d", step_count)
    logger.info("  Wall time:        %.1fs", elapsed_total)
    logger.info("  Steps/s:          %.1f", steps_per_sec)
    logger.info("  ms/step:          %.1f", ms_per_step)
    logger.info("  Sim time:         %.3fs", sim_time)
    logger.info("  Sim/wall ratio:   %.4f (%.1fx slower than real-time)",
                sim_time_per_wall_sec,
                1.0 / max(sim_time_per_wall_sec, 1e-10))
    if recorder is not None:
        logger.info("  Recording samples: %d (interval=%d)",
                    recorder.sample_count, recorder.sampling_interval)
    # Save recording (runs on SIGTERM/SIGINT via signal handler → running=False)
    if recorder is not None:
        try:
            recorder.save()
            logger.info(
                "Recording saved: %s (%d samples)",
                recorder.output_path, recorder.sample_count,
            )
        except Exception as e:
            logger.error("Failed to save recording: %s", e)

    # Cleanup streaming
    if streaming_observer is not None:
        logger.info(
            "Streaming: rendered %d frames total",
            streaming_observer.render_count,
        )
        try:
            streaming_observer.viewport.close()
            streaming_observer.selkies.stop()
        except Exception:
            pass

    pub_socket.close()
    rep_socket.close()
    ctx.term()
