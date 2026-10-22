"""§A6 contract stress-test — MIME's FVMFluidNode against ShardedUnstructuredNode.

The v0.3.0 plan's de-risk task #2 (Capita-Selecta scope, `capita-selecta-fvm-scope`):
read the MADDENING v0.3.0 §A6 ``ShardedUnstructuredNode`` contract *through a
hypothetical MIME FVMFluidNode* during the v0.3.0 cycle, so any contract flaw is a
cheap MADDENING fix now rather than a v0.4.0 API break cascading into a MIME rewrite.

MADDENING ships its own consumer-side mock (`tests/cloud/multigpu/
test_a6_contract_stress.py`); this is the MIME-side counterpart, and it stresses the
contract harder in two ways the MADDENING mock does not:

* it partitions the **real MIME ``FVMMesh`` face-graph** (cell-adjacency edges from
  the interior ``owner``/``neighbour`` arrays, real cell volumes), not a toy ring;
* it carries a **vector cell field** ``u: [N_cells, dim]`` alongside a scalar
  ``p: [N_cells]`` — the real FVM state shape — to confirm the partition / ghost
  exchange / gather handle non-scalar cell fields (the MADDENING mock used scalars
  only).

If any of this needs a contract change, that is exactly the signal to surface to the
MADDENING side now. No validated FVM physics is touched — this is a contract probe.

Run on 4 CPU virtual devices::

    JAX_PLATFORMS=cpu \\
    XLA_FLAGS="--xla_force_host_platform_device_count=4" \\
    pytest tests/verification/test_fvm_a6_contract.py -m 'slow or not slow'
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray

from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_3d

_N_DEVICES = len(jax.devices())
_NEEDS_MULTIDEVICE = pytest.mark.skipif(
    _N_DEVICES < 4,
    reason="needs >=4 (virtual) devices; run with JAX_PLATFORMS=cpu "
    "XLA_FLAGS=--xla_force_host_platform_device_count=4",
)


def _real_mesh(nx=4, ny=4, nz=3):
    """A small real MIME FVMMesh — the face-graph the contract must partition."""
    return make_cartesian_mesh_3d(nx, ny, nz, 1.0, 1.0, 0.75, origin=(0.0, 0.0, 0.0))


def _cell_adjacency_edges(mesh) -> np.ndarray:
    """Cell-adjacency edges (owner, neighbour) from the interior face graph —
    the input the §A6 partitioner consumes."""
    return np.stack(
        [np.asarray(mesh.owner), np.asarray(mesh.neighbour)], axis=1
    ).astype(np.int32)


def _contiguous_partition(n_cells: int, n_devices: int) -> np.ndarray:
    """A toy contiguous block partition (real meshes use PyMetis; partitioning is
    deferred to the sharding scale-out, F)."""
    base = n_cells // n_devices
    pa = np.concatenate([
        np.full(base + (1 if d < n_cells % n_devices else 0), d, dtype=np.int32)
        for d in range(n_devices)
    ])
    return pa


class _FVMShapeNode(SimulationNode):
    """The shape MIME's FVMFluidNode port would take against §A6.

    Real FVM state: a **vector** velocity ``u: [N_cells, dim]`` + a scalar pressure
    ``p: [N_cells]``; partitioned static cell volumes; a mass/volume domain integral.
    """

    def __init__(self, *, name, mesh, partition_assignment, timestep=1e-3):
        super().__init__(name=name, timestep=timestep)
        self._n_cells = int(mesh.N_cells)
        self._dim = int(mesh.dim)
        self._V = np.asarray(mesh.V, dtype=np.float32)
        self._pa = np.asarray(partition_assignment, dtype=np.int32)

    def state_fields(self) -> list[str]:
        return ["u", "p"]

    def domain_integral_fields(self) -> set[str]:
        return {"total_volume"}

    @property
    def static_data(self) -> dict:
        return {
            "cell_volumes": StaticArray(
                self._V, replication="partition",
                partition_assignment=self._pa,
            ),
        }

    def initial_state(self) -> dict:
        # A non-trivial vector field so a round-trip is meaningful: u[i] =
        # (i, 2i, 3i) scaled; p[i] = i.
        idx = jnp.arange(self._n_cells, dtype=jnp.float32)
        u = jnp.stack([idx, 2.0 * idx, 3.0 * idx], axis=1)[:, : self._dim]
        return {"u": u, "p": idx}

    def update(self, state, boundary_inputs, dt):
        return state  # unsharded path not exercised

    def update_padded(self, state_padded, boundary_inputs, dt,
                      *, static_padded=None, shard_info=None) -> dict:
        assert static_padded is not None and "cell_volumes" in static_padded
        assert shard_info is not None and 0 in shard_info
        n_local = shard_info[0][1]
        u = state_padded["u"]            # [n_local_max + n_ghost_max, dim]
        p = state_padded["p"]
        vol = static_padded["cell_volumes"]
        # Ghost-aware no-op (proves vector + scalar ghost reads are shape-correct).
        new_u = u + 0.0 * jnp.sum(u)
        new_p = p + 0.0 * jnp.sum(p)
        total_volume = jnp.sum(vol[:n_local])  # local partial; wrapper psums it
        return {"u": new_u, "p": new_p, "total_volume": total_volume}


def _wrap(mesh, n_devices=4):
    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.halo_unstructured import (
        build_unstructured_partition,
    )
    from maddening.cloud.multigpu.sharded_unstructured import (
        ShardedUnstructuredNode,
    )

    pa = _contiguous_partition(mesh.N_cells, n_devices)
    edges = _cell_adjacency_edges(mesh)
    layout = build_unstructured_partition(
        partition_assignment=pa, edges=edges, n_devices=n_devices)
    device_mesh = create_device_mesh(shape=(n_devices,))
    node = _FVMShapeNode(name="fvm_probe", mesh=mesh, partition_assignment=pa)
    return ShardedUnstructuredNode(node, device_mesh, layout), node, pa


@pytest.mark.slow
@_NEEDS_MULTIDEVICE
def test_real_fvm_mesh_partitions_and_domain_integral_is_exact():
    """The real FVMMesh face-graph partitions across 4 devices; the partitioned
    real cell volumes psum back to the exact total mesh volume — the §A6
    domain-integral contract on real MIME mesh data."""
    mesh = _real_mesh()
    sharded, node, _ = _wrap(mesh)
    out = sharded.update(sharded.initial_state(), {}, 1e-3)
    total = float(jax.device_get(out["total_volume"]))
    expected = float(jnp.sum(mesh.V))
    assert np.isclose(total, expected, atol=1e-6), f"{total} vs {expected}"


@pytest.mark.slow
@_NEEDS_MULTIDEVICE
def test_vector_cell_field_round_trips_through_partition():
    """A vector cell field ``u: [N_cells, dim]`` (the real FVM state shape)
    partitions and gathers losslessly — confirms the contract is not
    scalar-only."""
    mesh = _real_mesh()
    sharded, node, _ = _wrap(mesh)
    sharded_state = sharded.initial_state()
    gathered = sharded.gather_global(sharded_state)
    ref = node.initial_state()
    assert np.allclose(np.asarray(gathered["u"]), np.asarray(ref["u"]), atol=0)
    assert np.allclose(np.asarray(gathered["p"]), np.asarray(ref["p"]), atol=0)


@pytest.mark.slow
@_NEEDS_MULTIDEVICE
def test_sharded_cg_accepts_real_mesh_pressure_solve_surface():
    """The §A5 sharded iterative-solver surface MIME's PISO pressure-correction
    step will call accepts a sharded matvec built from the real mesh partition —
    with a ``halo_unstructured`` exchange inside the operator — and returns a
    correct finite solution. The pressure Poisson is SPD, so ``sharded_cg`` is
    the natural solver; the operator here is a well-conditioned diagonal
    placeholder (the real graph-Laplacian PISO operator is the M2 H1b work — a
    pure identity degenerates the Krylov space)."""
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P, NamedSharding

    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.halo_unstructured import (
        build_unstructured_partition, exchange_unstructured,
    )
    from maddening.cloud.multigpu.iterative_solver import sharded_cg

    mesh = _real_mesh()
    n_devices = 4
    pa = _contiguous_partition(mesh.N_cells, n_devices)
    edges = _cell_adjacency_edges(mesh)
    layout = build_unstructured_partition(
        partition_assignment=pa, edges=edges, n_devices=n_devices)
    device_mesh = create_device_mesh(shape=(n_devices,))

    def matvec(x):
        def shard_matvec(local):
            # Exercise the halo exchange inside the operator (the real PISO
            # matvec reads neighbour pressures across partition ghosts); the
            # SPD diagonal keeps the surface test well-conditioned.
            exchange_unstructured(local, layout=layout, mesh_axis="devices")
            return 2.0 * local
        return shard_map(
            shard_matvec, mesh=device_mesh,
            in_specs=(P("devices"),), out_specs=P("devices"),
            check_rep=False,
        )(x)

    b = jnp.ones(mesh.N_cells, dtype=jnp.float32)
    b_sharded = jax.device_put(b, NamedSharding(device_mesh, P("devices")))
    result = sharded_cg(matvec, b_sharded, mesh=device_mesh,
                        in_specs=P("devices"), max_iters=50, backend="loop")
    # A x = 2 x = b  ->  x = b / 2.
    assert bool(jax.device_get(result.converged))
    assert jnp.allclose(jax.device_get(result.value),
                        0.5 * np.ones(mesh.N_cells, dtype=np.float32), atol=1e-5)


def test_a6_contract_names_unchanged():
    """Pin every §A6 / §A5 symbol MIME's future FVMFluidNode port depends on. A
    rename here is a v0.4.0 contract break = a MIME rewrite — surface it then."""
    from maddening.cloud.multigpu.iterative_solver import (  # noqa: F401
        sharded_cg, sharded_gmres, SharedSolveResult,
    )
    from maddening.cloud.multigpu.halo_unstructured import (
        UnstructuredPartitionLayout,  # noqa: F401
        build_unstructured_partition,
        exchange_unstructured,  # noqa: F401
        partition_value,  # noqa: F401
        gather_value,  # noqa: F401
    )
    from maddening.cloud.multigpu.sharded_unstructured import (  # noqa: F401
        ShardedUnstructuredNode,
    )

    pa = np.zeros(4, dtype=np.int32)
    sa = StaticArray(np.zeros(4, dtype=np.float32),
                     replication="partition", partition_assignment=pa)
    assert sa.replication == "partition"

    layout = build_unstructured_partition(
        partition_assignment=pa, edges=np.zeros((0, 2), dtype=np.int32),
        n_devices=1)
    for attr in ("partition_assignment", "n_devices", "local_global_ids",
                 "n_local", "n_local_max", "ghost_global_ids", "n_ghost",
                 "n_ghost_max", "send_indices", "recv_local_index",
                 "send_counts"):
        assert hasattr(layout, attr), f"§A6 layout missing {attr!r}"
