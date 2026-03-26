"""mime.runner — MIME experiment subprocess for MICROROBOTICA integration.

Launched by MICROROBOTICA's ExperimentRunner as a separate process:
    python3 -m mime.runner /path/to/experiment.yaml

Responsibilities:
- Parse experiment.yaml (physics/setup, physics/params, control, scene)
- Build GraphManager from physics/setup.py
- Export graph.json (graph topology for MICROROBOTICA's graph inspector)
- Generate scene/world.usda via StageBridge
- Run step loop with ZMQ publishing (ResultFrame JSON each step)
- Handle commands via ZMQ REQ/REP (params, stop, reload_controller)
"""
