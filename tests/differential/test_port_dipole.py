"""The point dipole against the 0.2 port's ``PermanentMagnetNode`` (``field_model="point_dipole"``, the Earth's field
off, its gradient by ``jax.jacrev``), at 32 random magnets and points in the far field, its outputs committed in
``references/`` by ``generate_port_dipole_references.py``. Float64.

One difference is declared before the comparison: the port takes mu_0 as 4 pi 1e-7, and 0.3 takes CODATA 2022's
1.25663706127e-6 (``mime.units.VACUUM_PERMEABILITY``), since the 2019 SI no longer fixes it; the port's outputs are
scaled by their ratio, about 1 - 5e-10, before they are compared within 64 float64 roundings of each output's scale.
The control: the same comparison without that scaling fails, and so does a moment off by a part in a million."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.actuation import point_dipole
from mime.units import VACUUM_PERMEABILITY

REFERENCES = Path(__file__).parent / "references"
ROUNDINGS = 64
EPS = float(np.finfo(np.float64).eps)


@pytest.fixture(scope="module", autouse=True)
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.fixture(scope="module")
def port():
    arrays = np.load(REFERENCES / "port_dipole.npz", allow_pickle=False)
    provenance = json.loads((REFERENCES / "port_dipole_provenance.json").read_text())
    return {key: arrays[key] for key in arrays.files}, provenance


def misfit(port, *, moment_scale: float = 1.0, mu_0_ratio: float | None = None) -> float:
    """The largest difference of the field and its gradient from the port's, in roundings of each sample's output's
    largest entry."""
    arrays, provenance = port
    ratio = VACUUM_PERMEABILITY / provenance["mu_0"] if mu_0_ratio is None else mu_0_ratio
    evaluate = jax.jit(point_dipole)
    worst = 0.0
    for moment, centre, target, field, gradient in zip(arrays["moments_world"], arrays["centres"], arrays["targets"],
                                                        arrays["fields"], arrays["gradients"], strict=True):
        found_field, found_gradient = evaluate(jnp.asarray(moment * moment_scale), jnp.asarray(centre),
                                               jnp.asarray(target))
        for found, expected in ((found_field, field * ratio), (found_gradient, gradient * ratio)):
            scale = float(np.max(np.abs(expected)))
            worst = max(worst, float(np.max(np.abs(np.asarray(found) - expected))) / (EPS * scale))
    return worst


def test_the_point_dipole_is_the_ports_with_the_declared_constant(port):
    found = misfit(port)
    print(f"\nworst misfit: {found:.3g} float64 roundings of each output's scale (allowed {ROUNDINGS})")
    assert found <= ROUNDINGS


def test_control_the_constant_undeclared_or_a_moment_off_by_a_part_in_a_million_fails(port):
    assert misfit(port, mu_0_ratio=1.0) > ROUNDINGS
    assert misfit(port, moment_scale=1.0 + 1e-6) > ROUNDINGS
