"""The patch's two transfers as registered MADDENING mapping kinds (the lagged edge route): each rebuilt from its saved
spec by MADDENING's registry reads what the transfer function reads, and its ``apply_T`` is ``apply``'s transpose (the
adjoint identity on the terms' scale), for each component; the read anchored at points in the patch's frame."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from maddening.core.coupling.mapping_spec import build_mapping

from mime.mappings import BOUNDARY_KIND, READ_KIND, patch_boundary_kind, patch_read_kind
from mime.mappings._workarounds import in_patch_frame, origin_geometry
from mime.nodes.flow.fvm._grid import MACGrid
from mime.nodes.flow.fvm._nested import NestedMACGrid
from mime.nodes.flow.fvm._patch_transfers import patch_boundary, patch_read


@pytest.fixture(autouse=True)
def float64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def adjoint_error(kind, x, y, geom):
    bx = kind.apply(x, geom=geom)
    bty = kind.apply_T(y, geom=geom)
    left, right = float(jnp.sum(bx * y)), float(jnp.sum(x * bty))
    return abs(left - right) / (float(jnp.sum(jnp.abs(bx * y))) + float(jnp.sum(jnp.abs(x * bty))))


@pytest.mark.parametrize("component", [0, 1, 2])
def test_the_boundary_kind_rebuilt_from_its_spec_reads_the_transfer_and_is_its_transpose(component):
    rng = np.random.default_rng(component)
    coarse = MACGrid((9, 8, 10), 2e-4)
    patch = NestedMACGrid((6, 7, 5), 6e-5)
    origin = jnp.asarray([3, 2, 7], jnp.int32)
    kind = patch_boundary_kind(nx=9, ny=8, nz=10, spacing=2e-4, open_z=False, corner_x=0.0, corner_y=0.0, corner_z=0.0,
                               px=6, py=7, pz=5, fine=6e-5, component=component)
    rebuilt = build_mapping(kind.spec, lambda reference: None)
    assert kind.spec.kind == BOUNDARY_KIND and type(rebuilt) is type(kind)
    field = tuple(jnp.asarray(rng.normal(size=coarse.cells)) for _ in range(3))
    expected, _ = patch_boundary(patch, origin, coarse, field, jnp.zeros(3))
    found = rebuilt.apply(field[component].reshape(-1), geom=origin_geometry(origin, jnp.float64))
    assert float(jnp.max(jnp.abs(found - expected[component]))) == 0.0
    y = jnp.asarray(rng.normal(size=kind.n_target))
    assert adjoint_error(rebuilt, field[component].reshape(-1), y, origin_geometry(origin, jnp.float64)) <= 1e-14


@pytest.mark.parametrize("component", [0, 1, 2])
def test_the_read_kind_rebuilt_from_its_spec_reads_the_transfer_and_is_its_transpose(component):
    rng = np.random.default_rng(10 + component)
    patch = NestedMACGrid((6, 7, 5), 6e-5)
    origin = jnp.asarray([3, 2, 7], jnp.int32)
    points = jnp.asarray((np.asarray([3, 2, 7]) + rng.uniform(0.5, 4.5, size=(11, 3))) * 6e-5)
    kind = patch_read_kind(px=6, py=7, pz=5, fine=6e-5, component=component, count=11)
    rebuilt = build_mapping(kind.spec, lambda reference: None)
    assert kind.spec.kind == READ_KIND and type(rebuilt) is type(kind)
    field = tuple(jnp.asarray(rng.normal(size=patch.padded_shape(c))) for c in range(3))
    expected, _ = patch_read(patch, origin, field, points)
    local = in_patch_frame(points, origin, 6e-5)
    found = rebuilt.apply(field[component].reshape(-1), geom=local)
    assert float(jnp.max(jnp.abs(found - expected[:, component]))) <= 1e-15 * float(jnp.max(jnp.abs(expected)))
    y = jnp.asarray(rng.normal(size=11))
    assert adjoint_error(rebuilt, field[component].reshape(-1), y, local) <= 1e-14
