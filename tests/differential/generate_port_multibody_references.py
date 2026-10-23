"""Generate the 0.2 port's rigid-body library outputs on the AR4 at fixed random states, once, in the port's own
environment:

    MIME-port041/.venv/bin/python MIME-030-lane2/tests/differential/generate_port_multibody_references.py <out_dir>

It imports the port's ``mime`` (``mime.control.kinematics``, MIME-port041 on MADDENING 0.4.1) and never MIME 0.3's;
its outputs are copied into ``references/`` and compared with ``mime.core.multibody`` by ``test_port_multibody.py``.
Float64, CPU. Not collected by pytest.
"""

import hashlib
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
from mime.control.kinematics import (  # noqa: E402
    ee_jacobian,
    joint_to_world_transforms,
    link_to_world_transforms,
    mass_matrix,
    parse_urdf,
    rnea,
)

SAMPLES, SEED = 16, 20261023
GRAVITY = (0.0, 0.0, -9.81)
#: The URDF's velocity limit on every joint is 1.0472 rad/s (1.5708 on the last); accelerations at a few times that
#: per second.
VELOCITY, ACCELERATION = 1.0, 5.0


def main(out: Path) -> None:
    repo = Path(mime.__file__).resolve().parents[2]
    urdf = repo / "experiments" / "ar4_helical_drive" / "assets" / "ar4.urdf"
    tree = parse_urdf(str(urdf))
    rng = np.random.default_rng(SEED)
    limits = np.asarray(tree.joint_limits)
    q = rng.uniform(limits[:, 0], limits[:, 1], size=(SAMPLES, tree.num_joints))
    qd = rng.uniform(-VELOCITY, VELOCITY, size=q.shape)
    qdd = rng.uniform(-ACCELERATION, ACCELERATION, size=q.shape)
    identity = jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    gravity = jnp.asarray(GRAVITY)

    arrays: dict[str, np.ndarray] = {"q": q, "qd": qd, "qdd": qdd, "gravity": np.asarray(GRAVITY)}
    arrays["joint_world"] = np.array([np.asarray(joint_to_world_transforms(tree, jnp.asarray(x))) for x in q])
    arrays["centre_of_mass_world"] = np.array([np.asarray(link_to_world_transforms(tree, jnp.asarray(x))[:, :3, 3])
                                               for x in q])
    arrays["jacobian"] = np.array([[np.asarray(ee_jacobian(tree, jnp.asarray(x), i, identity))
                                    for i in range(tree.num_joints)] for x in q])
    arrays["mass_matrix"] = np.array([np.asarray(mass_matrix(tree, jnp.asarray(x))) for x in q])
    arrays["inverse_dynamics"] = np.array([np.asarray(rnea(tree, jnp.asarray(x), jnp.asarray(v), jnp.asarray(a), gravity))
                                           for x, v, a in zip(q, qd, qdd, strict=True)])

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "port_multibody.npz", allow_pickle=False, **arrays)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    import maddening

    (out / "port_multibody_provenance.json").write_text(json.dumps({
        "generator": "MIME-030-lane2/tests/differential/generate_port_multibody_references.py", "argv": sys.argv,
        "port": str(repo), "commit": commit, "library": "mime.control.kinematics",
        "urdf": str(urdf.relative_to(repo)), "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "links": list(tree.link_names), "joints": list(tree.joint_names), "maddening": maddening.__version__,
        "jax": jax.__version__, "numpy": np.__version__, "python": platform.python_version(), "x64": True,
        "device": jax.default_backend(), "seed": SEED, "samples": SAMPLES, "gravity": list(GRAVITY),
        "velocity_bound": VELOCITY, "acceleration_bound": ACCELERATION}, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
