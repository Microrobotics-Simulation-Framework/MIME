"""Generate the 0.2 port's point-dipole field and gradient at fixed random magnets and points, once, in the port's own
environment:

    MIME-port041/.venv/bin/python MIME-030-lane1/tests/differential/generate_port_dipole_references.py <out_dir>

It imports the port's ``mime`` (``PermanentMagnetNode`` with ``field_model="point_dipole"``, MIME-port041 on MADDENING
0.4.1) and never MIME 0.3's: the node's update gives the field and its gradient (by ``jax.jacrev``) at a target point,
from the magnet's pose and moment, with the Earth's field set to zero and the amplitude to one. Its outputs are copied
into ``references/`` and compared with ``mime.nodes.actuation.point_dipole`` by ``test_port_dipole.py``. Float64, CPU.
Not collected by pytest.
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
from mime.nodes.actuation.permanent_magnet import MU_0, PermanentMagnetNode  # noqa: E402

SAMPLES, SEED = 32, 20261028
#: The magnet's moment (A m^2) and its radius and length (m): de Jongh's 35 x 20 mm rotor magnet.
MOMENT, RADIUS, LENGTH = 18.89, 17.5e-3, 20e-3
#: Where the targets are drawn: from 5 to 30 magnet radii from the magnet, the point dipole's far field.
NEAREST, FURTHEST = 5 * RADIUS, 30 * RADIUS


def unit(rng: np.random.Generator, size: int) -> np.ndarray:
    v = rng.normal(size=(size, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main(out: Path) -> None:
    repo = Path(mime.__file__).resolve().parents[2]
    rng = np.random.default_rng(SEED)
    centres = rng.uniform(-0.2, 0.2, size=(SAMPLES, 3))
    turns = rng.normal(size=(SAMPLES, 4))
    turns /= np.linalg.norm(turns, axis=1, keepdims=True)
    axes = unit(rng, SAMPLES)
    distances = rng.uniform(NEAREST, FURTHEST, size=SAMPLES)
    targets = centres + distances[:, None] * unit(rng, SAMPLES)
    fields, gradients, moments = [], [], []
    for centre, turn, axis, target in zip(centres, turns, axes, targets, strict=True):
        node = PermanentMagnetNode("magnet", 1e-3, dipole_moment_a_m2=MOMENT, magnet_radius_m=RADIUS,
                                   magnet_length_m=LENGTH, magnetization_axis_in_body=tuple(axis),
                                   field_model="point_dipole", earth_field_world_t=(0.0, 0.0, 0.0))
        pose = jnp.asarray(np.concatenate([centre, turn]))
        out_state = node.update(node.initial_state(), {"magnet_pose_world": pose,
                                                       "target_position_world": jnp.asarray(target),
                                                       "amplitude_scale": jnp.asarray(1.0)}, 1e-3)
        fields.append(np.asarray(out_state["field_vector"]))
        gradients.append(np.asarray(out_state["field_gradient"]))
        moments.append(np.asarray(node._m_world(pose)))
    arrays = {"centres": centres, "orientations": turns, "axes": axes, "targets": targets,
              "moments_world": np.asarray(moments), "fields": np.asarray(fields), "gradients": np.asarray(gradients)}
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "port_dipole.npz", allow_pickle=False, **arrays)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    import maddening

    (out / "port_dipole_provenance.json").write_text(json.dumps({
        "generator": "MIME-030-lane1/tests/differential/generate_port_dipole_references.py",
        "argv": recorded_argv(sys.argv, repo.parent),
        "port": repo.name, "commit": commit, "node": "mime.nodes.actuation.permanent_magnet.PermanentMagnetNode",
        "field_model": "point_dipole", "earth_field_world_t": [0.0, 0.0, 0.0], "amplitude_scale": 1.0,
        "mu_0": float(MU_0), "moment_a_m2": MOMENT, "magnet_radius_m": RADIUS, "magnet_length_m": LENGTH,
        "nearest_m": NEAREST, "furthest_m": FURTHEST, "maddening": maddening.__version__, "jax": jax.__version__,
        "numpy": np.__version__, "python": platform.python_version(), "x64": True, "device": jax.default_backend(),
        "seed": SEED, "samples": SAMPLES}, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
