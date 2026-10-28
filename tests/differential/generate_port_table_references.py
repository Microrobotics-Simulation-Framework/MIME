"""Generate the 0.2 port's wall-table lookup outputs on its own grid, once, in the port's own environment:

    PYTHONDONTWRITEBYTECODE=1 MIME-port041/.venv/bin/python \
        MIME-030-7f/tests/differential/generate_port_table_references.py <wall table .npz> <out_dir>

It imports the port's ``mime`` (``cylinder_wall_table``, MIME-port041 on MADDENING 0.4.1) and never MIME 0.3's; its
outputs are copied into ``references/`` and compared with ``mime.core.tables`` by ``test_port_table.py``. Not collected
by pytest.

**The table.** A de Jongh wall table of the 0.2 line (``R_cyl`` 2.5, its grid 40 x 40 x 64 x 128 in rho, rho0, the
angle and the axial offset, a 3 x 3 value at each node) is too large to commit, so a block of it is kept: its own knots
and values over rho and rho0 from the eighth-last knot to the last (the near-wall end, where the 0.2 lookup's clip
began), the first six angles and the first seven axial offsets. The port's lookup reads that block as a table of its
own.

**The lookup.** The port reaches its lookup only through the assembly of a wall-correction matrix, which rotates each
3 x 3 block from cylindrical to Cartesian at its two points and flips the signs of the odd components for a negative
angle or offset. Each block is turned back here (the rotation inverted with the port's own rotation matrices, the
signs re-applied, with unit weights and the table's viscosity, so both scalings are one), giving the lookup's output
at the coordinates the port computed for each pair of points, which are recorded beside it. The NumPy lookup in
float64 and the JAX twin in float32, each with its own coordinates. The points lie inside the block on every axis, so
neither lookup's clip acts.

The full grid's knots are recorded too, for the per-lookup budget (``scripts/table_evaluator_budget.py``)."""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import mime
from _provenance import recorded_argv
from mime.nodes.environment.stokeslet.cylinder_greens_function_v2 import _cart_to_cyl, _rotation_matrices
from mime.nodes.environment.stokeslet.cylinder_wall_table import (
    PARITY_DPHI,
    PARITY_DZ,
    WallTable,
    assemble_image_correction_matrix_from_table,
    assemble_image_correction_matrix_from_table_jax,
    load_wall_table,
)

SEED = 20261028
#: The block of the table kept: rho and rho0, the angle, the axial offset.
RHO, ANGLE, OFFSET = slice(32, 40), slice(0, 6), slice(0, 7)
POINT_COUNT = 24


def signs(dphi_sign: np.ndarray, dz_sign: np.ndarray) -> np.ndarray:
    """The port's parity signs of each component, (pairs, 3, 3)."""
    found = np.ones((dphi_sign.size, 3, 3))
    for a in range(3):
        for b in range(3):
            if PARITY_DPHI[a, b] == -1:
                found[:, a, b] *= dphi_sign
            if PARITY_DZ[a, b] == -1:
                found[:, a, b] *= dz_sign
    return found


def lookup(assembled: np.ndarray, phi: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """The lookup's output from an assembled matrix: ``R_i G_ij R_j^T``, the signs re-applied."""
    count = phi.size
    blocks = np.asarray(assembled, dtype=np.float64).reshape(count, 3, count, 3).transpose(0, 2, 1, 3)
    turn = _rotation_matrices(phi)
    cylindrical = np.einsum("iab,ijbc,jdc->ijad", turn, blocks, turn)
    return cylindrical.reshape(count * count, 3, 3) * sign


def coordinates64(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The port's NumPy lookup's coordinates of every pair, its phi, and its signs (lines 233-249)."""
    rho, phi, z = _cart_to_cyl(points)
    count = len(points)
    dphi_mod = np.mod(phi[:, None] - phi[None, :] + np.pi, 2 * np.pi) - np.pi
    dz_raw = z[:, None] - z[None, :]
    dphi_sign = np.sign(dphi_mod)
    dphi_sign[dphi_sign == 0] = 1.0
    dz_sign = np.sign(dz_raw)
    dz_sign[dz_sign == 0] = 1.0
    query = np.stack([np.repeat(rho, count), np.tile(rho, count), np.abs(dphi_mod).ravel(), np.abs(dz_raw).ravel()],
                     axis=-1)
    return query, phi, signs(dphi_sign.ravel(), dz_sign.ravel())


def coordinates32(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The port's JAX twin's coordinates of every pair, in float32, its phi, and its signs (lines 344-367)."""
    held = jnp.asarray(points, dtype=jnp.float32)
    rho = jnp.sqrt(held[:, 0] ** 2 + held[:, 1] ** 2)
    phi = jnp.arctan2(held[:, 1], held[:, 0])
    z = held[:, 2]
    count = len(points)
    dphi_mod = jnp.mod(phi[:, None] - phi[None, :] + jnp.pi, 2 * jnp.pi) - jnp.pi
    dz_raw = z[:, None] - z[None, :]
    query = jnp.stack([jnp.broadcast_to(rho[:, None], (count, count)).ravel(),
                       jnp.broadcast_to(rho[None, :], (count, count)).ravel(), jnp.abs(dphi_mod).ravel(),
                       jnp.abs(dz_raw).ravel()], axis=-1)
    dphi_sign = np.where(np.asarray(dphi_mod) < 0, -1.0, 1.0).ravel()
    dz_sign = np.where(np.asarray(dz_raw) < 0, -1.0, 1.0).ravel()
    return np.asarray(query), np.asarray(phi, dtype=np.float64), signs(dphi_sign, dz_sign)


def main(source: Path, out: Path) -> None:
    repo = Path(mime.__file__).resolve().parents[2]
    full = load_wall_table(str(source))
    block = WallTable(R_cyl=full.R_cyl, mu=full.mu, rho_grid=full.rho_grid[RHO], dphi_grid=full.dphi_grid[ANGLE],
                      dz_grid=full.dz_grid[OFFSET], data=np.ascontiguousarray(full.data[RHO, RHO, ANGLE, OFFSET]))
    rng = np.random.default_rng(SEED)
    rho = rng.uniform(block.rho_grid[0], block.rho_grid[-1], POINT_COUNT)
    rho[:3] = block.rho_grid[[0, 3, -1]]
    phi = rng.uniform(0.0, block.dphi_grid[-1], POINT_COUNT)
    z = rng.uniform(0.0, block.dz_grid[-1], POINT_COUNT)
    points = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=-1)
    weights = np.ones(POINT_COUNT)

    query64, phi64, sign64 = coordinates64(points)
    numpy_lookup = lookup(assemble_image_correction_matrix_from_table(points, weights, block.R_cyl, block.mu, block),
                          phi64, sign64)
    query32, phi32, sign32 = coordinates32(points)
    jax_lookup = lookup(np.asarray(assemble_image_correction_matrix_from_table_jax(points, weights, block.R_cyl,
                                                                                    block.mu, block)),
                        phi32, sign32)
    for query, grid in ((query64, block), (query32, block)):
        for axis, knots in enumerate((grid.rho_grid, grid.rho_grid, grid.dphi_grid, grid.dz_grid)):
            assert knots[0] <= query[:, axis].min() and query[:, axis].max() <= knots[-1], "a point outside the block"

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "port_table.npz", allow_pickle=False, rho=block.rho_grid, angle=block.dphi_grid,
             offset=block.dz_grid, data=block.data, points=points, query64=query64, lookup64=numpy_lookup,
             query32=query32.astype(np.float32), lookup32=jax_lookup, full_rho=full.rho_grid,
             full_angle=full.dphi_grid, full_offset=full.dz_grid)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    import maddening

    (out / "port_table_provenance.json").write_text(json.dumps({
        "generator": "MIME-030-7f/tests/differential/generate_port_table_references.py",
        "argv": recorded_argv(sys.argv, repo.parent),
        "port": repo.name, "commit": commit,
        "library": "mime.nodes.environment.stokeslet.cylinder_wall_table: assemble_image_correction_matrix_from_table "
                   "(NumPy, float64) and assemble_image_correction_matrix_from_table_jax (JAX, float32)",
        "source_table": {"file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "R_cyl": float(full.R_cyl), "mu": float(full.mu), "shape": list(full.data.shape)},
        "block": {"rho": [RHO.start, RHO.stop], "angle": [ANGLE.start, ANGLE.stop],
                  "offset": [OFFSET.start, OFFSET.stop]},
        "points": POINT_COUNT, "pairs": POINT_COUNT ** 2, "seed": SEED, "maddening": maddening.__version__,
        "jax": jax.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "device": jax.default_backend()}, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
