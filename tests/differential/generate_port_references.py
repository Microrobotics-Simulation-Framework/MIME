"""Generate the 0.2 port's node outputs on fixed random inputs, once, in the port's own environment:

    MIME-port041/.venv/bin/python MIME-030/tests/differential/generate_port_references.py <out_dir>

It imports the port's ``mime`` (MIME-port041 on MADDENING 0.4.1) and never MIME 0.3's; its outputs are copied into
``references/`` and compared with the 0.3 nodes by ``test_port_nodes.py``. Float64, CPU. Not collected by pytest.
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
from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode  # noqa: E402
from mime.nodes.environment.gravity_node import GravityNode  # noqa: E402
from mime.nodes.robot.constraints import CylindricalVesselConstraint  # noqa: E402
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode  # noqa: E402
from mime.nodes.robot.rigid_body import RigidBodyNode  # noqa: E402

SAMPLES, SEED, DT = 64, 20261023, 5e-4
#: A's reference values (phase 0's provenance): the FL-9 body and its two magnets.
M_EFF, I_EFF, M_SINGLE, CONSTRAINT_RADIUS, HALF_LENGTH = 8.491116020725203e-05, 1.0331989974018431e-10, 8.4e-4, \
    1.0002e-3, 0.05


def unit_quaternions(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def main(out: Path) -> None:
    rng = np.random.default_rng(SEED)
    arrays: dict[str, np.ndarray] = {}

    field = ExternalMagneticFieldNode("field", DT)
    steps = rng.integers(0, 20000, size=SAMPLES)
    frequency, strength = rng.uniform(-40, 40, size=SAMPLES), rng.uniform(0, 5, size=SAMPLES)
    arrays.update(field_steps=steps, field_frequency_hz=frequency, field_strength_mt=strength)
    arrays["field_vector"] = np.array([np.asarray(field.update(
        {**field.initial_state(), "step": jnp.asarray(s, dtype=jnp.int32), "sim_time": jnp.asarray(0.0)},
        {"frequency_hz": jnp.asarray(f), "field_strength_mt": jnp.asarray(b)}, DT)["field_vector"])
        for s, f, b in zip(steps, frequency, strength, strict=True)])

    magnet = PermanentMagnetResponseNode("magnet", DT, n_magnets=2, m_single=M_SINGLE, moment_axis=(1.0, 0.0, 0.0))
    q, b = unit_quaternions(rng, SAMPLES), rng.uniform(-5e-3, 5e-3, size=(SAMPLES, 3))
    arrays.update(magnet_orientation=q, magnet_field=b)
    out_magnet = [magnet.update(magnet.initial_state(), {"field_vector": jnp.asarray(bb),
                                                         "field_gradient": jnp.zeros((3, 3)),
                                                         "orientation": jnp.asarray(qq)}, DT)
                  for qq, bb in zip(q, b, strict=True)]
    arrays["magnet_torque"] = np.array([np.asarray(o["magnetic_torque"]) for o in out_magnet])
    arrays["magnet_force"] = np.array([np.asarray(o["magnetic_force"]) for o in out_magnet])

    gravity = GravityNode("gravity", DT, delta_rho_kg_m3=410.0, volume_m3=6.022068099805108e-08, g=9.81,
                          direction=(0.0, -1.0, 0.0))
    arrays["gravity_force"] = np.asarray(gravity.update(gravity.initial_state(), {}, DT)["gravity_force"])

    body = RigidBodyNode("body", DT, use_inertial=True, use_analytical_drag=False, I_eff=I_EFF, m_eff=M_EFF,
                         constraint=CylindricalVesselConstraint(radius=CONSTRAINT_RADIUS, half_length=HALF_LENGTH,
                                                                axis=0))
    position = np.column_stack([rng.uniform(-0.02, 0.02, SAMPLES), rng.uniform(-1.5e-3, 1.5e-3, (SAMPLES, 2))])
    orientation = unit_quaternions(rng, SAMPLES)
    velocity, omega = rng.uniform(-5e-3, 5e-3, (SAMPLES, 3)), rng.uniform(-80, 80, (SAMPLES, 3))
    omega[:4] = 0.0  # bodies at rest: the 0.3 quaternion's declared fix changes derivatives there, not values
    force, torque = rng.uniform(-3e-4, 3e-4, (SAMPLES, 3)), rng.uniform(-3e-6, 3e-6, (SAMPLES, 3))
    arrays.update(body_position=position, body_orientation=orientation, body_velocity=velocity, body_omega=omega,
                  body_force=force, body_torque=torque)
    stepped = [body.update({"position": jnp.asarray(p), "orientation": jnp.asarray(o), "velocity": jnp.asarray(v),
                            "angular_velocity": jnp.asarray(w)},
                           {"magnetic_force": jnp.asarray(f), "magnetic_torque": jnp.asarray(t)}, DT)
               for p, o, v, w, f, t in zip(position, orientation, velocity, omega, force, torque, strict=True)]
    for name in ("position", "orientation", "velocity", "angular_velocity"):
        arrays[f"body_out_{name}"] = np.array([np.asarray(s[name]) for s in stepped])

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "port_nodes.npz", allow_pickle=False, **arrays)
    repo = Path(mime.__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    import maddening

    (out / "port_nodes_provenance.json").write_text(json.dumps({
        "generator": "MIME-030/tests/differential/generate_port_references.py", "argv": sys.argv,
        "port": repo.name, "commit": commit, "maddening": maddening.__version__, "jax": jax.__version__,
        "numpy": np.__version__, "python": platform.python_version(), "x64": True, "device": jax.default_backend(),
        "seed": SEED, "samples": SAMPLES, "dt": DT}, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
