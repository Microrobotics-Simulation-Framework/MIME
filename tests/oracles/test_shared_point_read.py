"""An observation's read of a field that varies in space shares the slot a law reads at the same body point (the build's
``share_point_reads``), the refusal lifted for ``PhaseLag`` on a coil array: before, every observation's read of such a
field was refused (``observation.field-read``), since a slot of its own would change the provider's node.

On ``mount_cases``'s still sphere at Navion_2's centre, its magnet at its centre and the torque reading the field there:
``PhaseLag`` builds, its flux density is the torque's slot (the edge into it and the report's line say so), the
provider's node is the one built without it, and that field at the sphere is the host's ``actuation`` (float64) at the
point, within ``SLACK eps / c`` of its peak in float64 and float32 (``c`` the worst field cancellation of Navion_2's
sources at the point, the floor rounding puts on their sum; ``SLACK`` as in ``tests/oracles/test_calibrated_multipole.py``).
The control: with no law reading the field at the sphere, it is still refused."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import jax
import numpy as np
import pytest

from mount_cases import CURRENTS, exposure_sphere

from mime.build import BuildRefused
from mime.build._report import graph_config
from mime.data import asset
from mime.fields import CalibratedMultipole
from mime.fields._coil_array import calibration_of
from mime.interactions import Magnetic
from mime.observations import PhaseLag
from mime.units import m, ms

SLACK = 10.0
SLOT = "flux_density_magnetic_torque_sphere"


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def lag() -> PhaseLag:
    return PhaseLag("sphere", field="coils.magnetic", about=(0.0, 0.0, 1.0))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_phase_lag_on_a_coil_array_reads_the_torques_slot_and_its_field_is_actuations(dtype):
    with in_dtype(dtype):
        built = exposure_sphere("lag_on_coils", observe=lag(), latency=False, laws=(Magnetic.Torque,)).build()
        plain = exposure_sphere("lag_on_coils", latency=False, laws=(Magnetic.Torque,)).build()
        history = built.run(ms(5), history=True).history
    edges = graph_config(built.graph)["edges"]
    into = [(e["source_node"], e["source_field"]) for e in edges
            if e["target_node"] == "phase_lag:sphere" and e["target_field"] == "flux_density"]
    assert into == [("coils.magnetic", SLOT)]
    assert any("where magnetic.torque:sphere reads it" in line for line in built.report.checked["phase_lag:sphere"])
    nodes = [b.graph.get_node("coils.magnetic") for b in (built, plain)]
    assert sorted(nodes[0].initial_state()) == sorted(nodes[1].initial_state())
    made = CalibratedMultipole(calibration=asset("navion_2")).actuation(m((0.0, 0.0, 0.0))).field @ np.asarray(CURRENTS)
    field = np.asarray(history["coils.magnetic"][SLOT], dtype=np.float64)
    eps = float(np.finfo(np.dtype(dtype)).eps)
    tolerance = SLACK * eps / calibration_of(asset("navion_2")).cancellation(np.zeros((1, 3)))
    assert np.max(np.abs(field - made)) <= tolerance * np.max(np.abs(made))


def test_the_control_no_law_reading_the_field_at_the_sphere_is_still_refused():
    with pytest.raises(BuildRefused) as refused:
        exposure_sphere("lag_unread", observe=lag(), latency=False, laws=()).plan()
    assert "observation.field-read" in refused.value.codes()
