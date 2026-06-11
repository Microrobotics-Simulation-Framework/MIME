"""DefectCorrectionFluidNode — v0.2 static_data channel adoption.

Pins that the BEM + LBM static arrays are declared on the static_data
channel and that static_data_hash is stable.
"""
import pytest

from maddening.core.static_data import StaticArray


def _make_node():
    from mime.nodes.environment.stokeslet import sphere_surface_mesh
    from mime.nodes.environment.defect_correction.fluid_node import (
        DefectCorrectionFluidNode,
    )
    body_mesh = sphere_surface_mesh(radius=0.1, n_refine=1)
    return DefectCorrectionFluidNode(
        name="defect", timestep=1e-3,
        mu=1e-3, rho=1000.0,
        body_mesh=body_mesh,
        body_radius=0.1, vessel_radius=0.3, dx=0.15,
    )


def test_defect_correction_static_data():
    node = _make_node()
    sd = node.static_data
    expected = {
        "bem_lu", "bem_piv", "body_points", "body_weights",
        "pipe_wall", "pipe_missing", "no_wall", "no_missing",
        "pipe_missing_flat", "no_missing_flat", "ib_idx", "ib_wts",
    }
    assert set(sd.keys()) == expected
    assert all(isinstance(v, StaticArray) for v in sd.values())
    assert all(v.replication == "replicate" for v in sd.values())
    # static_data_hash is stable across calls and non-zero.
    assert node.static_data_hash() == node.static_data_hash()
    assert node.static_data_hash() != 0
