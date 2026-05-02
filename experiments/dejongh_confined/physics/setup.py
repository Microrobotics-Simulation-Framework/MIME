"""Adapter from runner params dict → mime.experiments.dejongh.build_graph.

The MIME runner imports this module and calls ``build_graph(params)`` once
at startup, expecting a configured (but un-compiled) ``GraphManager``
back. All wiring is delegated to ``mime.experiments.dejongh.build_graph``;
this file translates the flat params namespace into kwargs and seeds the
initial pose.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from mime.experiments.dejongh import build_graph as _build_graph


def _resolve_weights(params: dict) -> Path:
    p = Path(params["MLP_WEIGHTS_PATH"])
    if p.is_absolute():
        return p
    # Resolve relative to the MIME repo root: this file lives at
    # MIME/experiments/dejongh_confined/physics/setup.py
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / p


def build_graph(params: dict):
    """Construct the de Jongh graph with the runner's params."""
    gm = _build_graph(
        design_name=params["DESIGN_NAME"],
        vessel_name=params["VESSEL_NAME"],
        mu_Pa_s=params["MU_PA_S"],
        delta_rho=params["DELTA_RHO_KG_M3"],
        dt=params["DT_PHYS"],
        use_lubrication=params["USE_LUBRICATION"],
        lubrication_epsilon_mm=params["LUB_EPSILON_MM"],
        mlp_weights_path=_resolve_weights(params),
    )

    # Seed the body pre-sunk so the viewer doesn't open on the
    # ~3 ms gravity-relaxation transient. RigidBodyNode.initial_state()
    # returns position/orientation/velocity/angular_velocity dict; the
    # graph manager exposes a per-node state we can patch in place
    # before the first step.
    body = gm._nodes["body"]
    init_pos = jnp.array([
        params["INIT_X_M"], params["INIT_Y_M"], params["INIT_Z_M"]
    ], dtype=jnp.float32)

    # GraphManager exposes ``set_initial_state`` for this — fall back to
    # mutating the node's state dict template if the API isn't present.
    if hasattr(gm, "set_initial_state"):
        gm.set_initial_state("body", {"position": init_pos})
    else:
        # Older MADDENING revisions keep the per-node initial dict as
        # ``_initial_state``; we override just the position field.
        if hasattr(body, "_initial_state") and body._initial_state is not None:
            body._initial_state["position"] = init_pos

    return gm
