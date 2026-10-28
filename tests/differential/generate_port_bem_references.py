"""Generate the 0.2 port's free-space BEM outputs, once, in the port's own environment, before any of it is ported:

    PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu MIME-port041/.venv/bin/python \
        MIME-030-7g/tests/differential/generate_port_bem_references.py <out_dir> [ci|full]

It imports the port's ``mime`` (MIME-port041 on MADDENING 0.4.1) and its ``scripts/dejongh_benchmark.py``, and never
MIME 0.3's. Float64, CPU. Its outputs are copied into ``references/`` and compared with ``mime.nodes.hydro.bem`` by
``test_port_bem.py``. Not collected by pytest.

**What each case records**, each body by its own 0.2 mesh (its samples: triangle centroids, unit normals from each flat
triangle, the triangles' areas), with the blob at half the mesh's mean spacing (the 0.2 default), the viscosity 1 and
the torque about the mesh's origin:

- ``R_library``: ``resistance.compute_resistance_matrix`` (JAX), the 6 x 6 the 0.2 node's free-space standalone mode
  holds, not symmetrised;
- ``R_node``, ``force``, ``torque``, ``traction``: ``StokesletFluidNode`` built in its free-space Schwarz mode on the
  same mesh (its LU of the body's matrix), its symmetrised resistance (``resistance_matrix_si``), and its ``update``
  for each of the six unit rigid motions (three translations, three rotations), at rest in still fluid;
- for the helix, also ``R_benchmark`` and ``pre_symmetrisation``: ``dejongh_benchmark.compute_freespace_R``, the
  free-space form of the MRS reference ``compute_R_matrix`` the G0 R-matrix anchor matched (the same assembly and
  solve less the wall table's correction, in NumPy and SciPy), symmetrised after its solve, and how far from
  symmetric it was before;
- for the helix, the vertex grid its triangles are cut from (``vertices``), so that a test can show the 0.2 mesh is a
  tessellation of the de Jongh kind's lateral surface.

**The cases.** ``ci``: the helix at 12 x 10 vertices, the sphere at two subdivisions (320 triangles), the prolate
spheroid at two. ``full``: the helix at 24 x 36 (G0-a's free-space helix and G0-b's anchor mesh, 1,680 samples), at
40 x 40 (the surrogate's training data, ``generate_mlp_training_data.py``, 3,120 samples: the 9,360-unknown system),
and the sphere at three subdivisions (G0-a's sphere, 1,280 samples). The helix is de Jongh's FL-9 (nu 2.33, L 7.47
mm, R 1.56 mm, modulation 0.33, two starts), its lengths divided by R as ``dejongh_benchmark`` divides them. The 0.2
line has no spheroid mesh: the generator stretches the 0.2 icosphere's vertices (``surface_mesh._subdivide_icosphere``
from the icosahedron ``sphere_surface_mesh`` starts with) by 2 along x and takes the 0.2 rule (centroids, flat
normals, areas): semi-axes 2, 1, 1."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402

import mime  # noqa: E402
from _provenance import recorded_argv  # noqa: E402
from mime.nodes.environment.stokeslet.dejongh_geometry import (  # noqa: E402
    EPSILON_DEFAULT,
    FL_L_UMR,
    FL_TABLE,
    N_STARTS_DEFAULT,
    R_CYL_DEFAULT,
    dejongh_fl_mesh,
)
from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode  # noqa: E402
from mime.nodes.environment.stokeslet.resistance import compute_resistance_matrix  # noqa: E402
from mime.nodes.environment.stokeslet.surface_mesh import (  # noqa: E402
    SurfaceMesh,
    _subdivide_icosphere,
    sphere_surface_mesh,
)

REPO = Path(mime.__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from dejongh_benchmark import compute_freespace_R  # noqa: E402

DESIGN = 9
SPHEROID_STRETCH = 2.0
CASES = {
    "ci": {"helix_coarse": ("helix", (12, 10)), "sphere_coarse": ("sphere", 2), "spheroid_coarse": ("spheroid", 2)},
    "full": {"helix_g0": ("helix", (24, 36)), "helix_training": ("helix", (40, 40)), "sphere_g0": ("sphere", 3)},
}


def helix_vertices(n_theta: int, n_zeta: int) -> np.ndarray:
    """The vertex grid ``dejongh_umr_surface`` cuts its triangles from, in mm, (n_theta, n_zeta, 3)."""
    nu, radius = FL_TABLE[DESIGN]["nu"], R_CYL_DEFAULT
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0, FL_L_UMR, n_zeta)
    th, ze = np.meshgrid(theta, zeta, indexing="ij")
    rho = radius * (1.0 + EPSILON_DEFAULT * np.sin(N_STARTS_DEFAULT * th))
    alpha = nu * ze / radius + th
    return np.stack([rho * np.cos(alpha), rho * np.sin(alpha), ze], axis=-1)


def triangles(vertices: np.ndarray, faces: np.ndarray) -> SurfaceMesh:
    """The 0.2 rule of ``sphere_surface_mesh``: centroids, flat unit normals, areas."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    return SurfaceMesh(points=(v0 + v1 + v2) / 3.0, normals=cross / (np.linalg.norm(cross, axis=1, keepdims=True)
                                                                      + 1e-30), weights=areas)


def spheroid_mesh(n_refine: int) -> SurfaceMesh:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0], [0, -1, phi], [0, 1, phi],
                      [0, -1, -phi], [0, 1, -phi], [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]],
                     dtype=np.float64)
    verts /= np.linalg.norm(verts[0])
    faces = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                      [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                      [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int32)
    for _ in range(n_refine):
        verts, faces = _subdivide_icosphere(verts, faces)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * np.array([SPHEROID_STRETCH, 1.0, 1.0])
    return triangles(verts, faces)


def node_outputs(mesh: SurfaceMesh, epsilon: float) -> dict[str, np.ndarray]:
    """The 0.2 node in its free-space Schwarz mode: its symmetrised resistance, and its update for each unit motion."""
    node = StokesletFluidNode("bem", 1e-3, mu=1.0, body_mesh=mesh, interface_mesh=mesh, epsilon=epsilon)
    count = mesh.n_points
    forces, torques, tractions = [], [], []
    for column in range(6):
        motion = np.zeros(6)
        motion[column] = 1.0
        out = node.update(node.initial_state(), {
            "body_velocity": jnp.asarray(motion[:3]), "body_angular_velocity": jnp.asarray(motion[3:]),
            "body_orientation": jnp.asarray([1.0, 0.0, 0.0, 0.0]), "background_flow": jnp.zeros((count, 3))}, 1e-3)
        forces.append(np.asarray(out["drag_force"]))
        torques.append(np.asarray(out["drag_torque"]))
        tractions.append(np.asarray(out["body_traction"]))
    return {"R_node": np.asarray(node.resistance_matrix_si()), "force": np.stack(forces), "torque": np.stack(torques),
            "traction": np.stack(tractions)}


def case(kind: str, size: object) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    if kind == "helix":
        n_theta, n_zeta = size
        mm = dejongh_fl_mesh(DESIGN, n_theta=n_theta, n_zeta=n_zeta)
        vertices = helix_vertices(n_theta, n_zeta)
        rebuilt = [np.stack([vertices[i, j], vertices[(i + 1) % n_theta, j], vertices[(i + 1) % n_theta, j + 1],
                             vertices[i, j], vertices[(i + 1) % n_theta, j + 1], vertices[i, j + 1]])
                   for i in range(n_theta) for j in range(n_zeta - 1)]
        centroids = np.stack([np.stack([(q[0] + q[1] + q[2]) / 3.0, (q[3] + q[4] + q[5]) / 3.0])
                              for q in rebuilt]).reshape(-1, 3)
        assert np.array_equal(centroids, mm.points), "the vertex grid is not the one the mesh was cut from"
        R_benchmark, pre = compute_freespace_R(mm)
        scale = R_CYL_DEFAULT
        mesh = SurfaceMesh(points=mm.points / scale, normals=mm.normals, weights=mm.weights / scale ** 2)
        found.update(vertices=vertices / scale, R_benchmark=R_benchmark, pre_symmetrisation=np.asarray(pre),
                     length_scale_mm=np.asarray(scale))
    elif kind == "sphere":
        mesh = sphere_surface_mesh(radius=1.0, n_refine=int(size))
    else:
        mesh = spheroid_mesh(int(size))
    epsilon = mesh.mean_spacing / 2.0
    R_library = compute_resistance_matrix(jnp.asarray(mesh.points), jnp.asarray(mesh.weights), jnp.zeros(3), epsilon,
                                          1.0)
    found.update(points=np.asarray(mesh.points), normals=np.asarray(mesh.normals), weights=np.asarray(mesh.weights),
                 epsilon=np.asarray(epsilon), R_library=np.asarray(R_library), **node_outputs(mesh, epsilon))
    return found


def main(out: Path, which: str = "ci") -> None:
    repo = REPO
    out.mkdir(parents=True, exist_ok=True)
    name = "port_bem" if which == "ci" else "port_bem_full"
    arrays: dict[str, np.ndarray] = {}
    for label, (kind, size) in CASES[which].items():
        for key, value in case(kind, size).items():
            arrays[f"{label}/{key}"] = value
    np.savez(out / f"{name}.npz", allow_pickle=False, **arrays)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    import maddening

    (out / f"{name}_provenance.json").write_text(json.dumps({
        "generator": "MIME-030-7g/tests/differential/generate_port_bem_references.py",
        "argv": recorded_argv(sys.argv, repo.parent),
        "port": repo.name, "commit": commit,
        "library": "mime.nodes.environment.stokeslet: resistance.compute_resistance_matrix (JAX), "
                   "fluid_node.StokesletFluidNode (free-space Schwarz mode), and scripts/dejongh_benchmark.py's "
                   "compute_freespace_R (NumPy, SciPy) for the helix",
        "cases": {label: {"kind": kind, "size": list(size) if isinstance(size, tuple) else size,
                          "samples": int(arrays[f"{label}/points"].shape[0])}
                  for label, (kind, size) in CASES[which].items()},
        "helix": {"design": f"FL-{DESIGN}", "nu": FL_TABLE[DESIGN]["nu"], "length_mm": FL_L_UMR,
                  "radius_mm": R_CYL_DEFAULT, "modulation": EPSILON_DEFAULT, "starts": N_STARTS_DEFAULT,
                  "lengths_divided_by": "radius_mm"},
        "spheroid_stretch": SPHEROID_STRETCH, "viscosity": 1.0, "blob": "half the mesh's mean spacing",
        "torque_about": "the mesh's origin", "x64": bool(jax.config.jax_enable_x64), "device": jax.default_backend(),
        "maddening": maddening.__version__, "jax": jax.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "python": platform.python_version(),
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "sha256": hashlib.sha256((out / f"{name}.npz").read_bytes()).hexdigest()}, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), *sys.argv[2:3])
