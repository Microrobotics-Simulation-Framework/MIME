"""M5 integration test: target reach envelope and joint-limit safety.

The AR4 at the configured base position can only reach roughly
``x ∈ [-0.05, +0.05]`` m at the 20 cm standoff with home orientation.
Beyond that range the IK can't converge without joint-limit
violations.  The controller's M5 clamp must keep the target inside
the safe envelope even if the body's position is far outside.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

PARAMS_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments" / "ar4_helical_drive" / "physics" / "params.py"
)
CONTROLLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments" / "ar4_helical_drive" / "control" / "controller.py"
)


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_params() -> dict:
    ns: dict = {}
    with open(PARAMS_PATH) as fh:
        exec(fh.read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}


@pytest.mark.parametrize("body_x", [+0.45, -0.45, +0.20, -0.20])
def test_target_x_clamped_to_envelope(body_x):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    params = _load_params()
    # params.py uses CONTROL_TARGET_ALPHA=0.005 (slow LP for jitter
    # rejection in the live experiment); 200 iterations don't fully
    # converge at that rate. Bump alpha so the 1mm clamp tolerance
    # is exercised against a settled target.
    params["CONTROL_TARGET_ALPHA"] = 0.2
    controller = _load_module_from_path("ar4_controller", CONTROLLER_PATH)
    controller._controller_instance = None

    home = jnp.asarray(params["ARM_HOME_RAD"], dtype=jnp.float32)
    body_pos = jnp.array([body_x, 0.0, -1e-3], dtype=jnp.float32)
    state = {
        "arm": {"joint_angles": home, "joint_velocities": jnp.zeros_like(home)},
        "body": {
            "position": body_pos,
            "orientation": jnp.array([0.7071068, 0.0, 0.7071068, 0.0],
                                     dtype=jnp.float32),
        },
    }
    controller.get_external_inputs(params, 0, state=state)
    inst = controller._controller_instance
    body_quat = state["body"]["orientation"]
    body_angvel = jnp.zeros(3, dtype=jnp.float32)   # static body
    # Run the filter to convergence directly (avoids JAX dispatch cost).
    for _ in range(200):
        inst._update_target_from_body(body_pos, body_quat, body_angvel)

    target_x = float(inst.T_target_world[0, 3])
    x_min = params["CONTROL_X_MIN_M"]
    x_max = params["CONTROL_X_MAX_M"]
    # Float32 tolerance — the controller stores params as float32
    # so clamping to -0.05 lands at -0.05000000447034836 in
    # double-precision comparisons.
    eps = 1e-5
    assert x_min - eps <= target_x <= x_max + eps, (
        f"Target x = {target_x:.6f} not in safe envelope "
        f"[{x_min}, {x_max}] for body_x = {body_x}"
    )
    # When body is outside the envelope, target must be saturated to
    # the nearer edge.
    if body_x > x_max:
        assert abs(target_x - x_max) < 1e-3
    elif body_x < x_min:
        assert abs(target_x - x_min) < 1e-3
