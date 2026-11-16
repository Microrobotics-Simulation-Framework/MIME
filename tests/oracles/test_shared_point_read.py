"""An observation's read of a field that varies in space shares the slot a law reads at the same body point (the build's
``share_point_reads``), the refusal lifted for ``PhaseLag`` on a coil array: before, every observation's read of such a
field was refused (``observation.field-read``), since a slot of its own would change the provider's node.

On ``mount_cases``'s still sphere at Navion_2's centre, its magnet at its centre and the torque reading the field there:
``PhaseLag`` builds, its flux density is the torque's slot (the edge into it and the report's line say so), the
provider's node is the one built without it, and that field at the sphere is the host's ``actuation`` (float64) at the
point, within ``SLACK eps / c`` of its peak in float64 and float32 (``c`` the worst field cancellation of Navion_2's
sources at the point, the floor rounding puts on their sum; ``SLACK`` as in ``tests/oracles/test_calibrated_multipole.py``).
The control: with no law reading the field at the sphere, it is still refused.

**At a nonzero ``magnet_centre`` on a turned body** the slot is read at the magnet's centre in the world, ``position + R
centre`` (the magnetic laws 0.2.0's point): an ``Exposure`` observing a sphere turned by a unit quaternion ``R``, its
magnet 1.4 cm off its position, reports ``|B|`` and the Frobenius norm of the gradient of the host's ``actuation``
(float64) at that point, within ``SLACK eps`` times the sum of the magnitudes the field is summed from
(``mount_cases.term_sums``), in float64 and float32, on a force law alone and on a torque and a force; and equal to the
norms of what the provider's slot holds. The controls, each told apart by more than twenty times the tolerance: the
field at the body's position (not at the magnet), and at ``position + centre`` with the turn left out."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import jax
import numpy as np
import pytest

from mount_cases import CURRENTS, exposure_sphere, term_sums

from mime.build import BuildRefused
from mime.build._report import graph_config
from mime.data import asset
from mime.fields import CalibratedMultipole
from mime.fields._coil_array import calibration_of
from mime.interactions import Magnetic
from mime.observations import Exposure, PhaseLag
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


POSITION = np.array([0.010, -0.008, 0.006])
CENTRE = np.array([0.012, -0.004, 0.007])
TURNED = np.array([0.8, 0.3, -0.4, 0.2]) / np.linalg.norm([0.8, 0.3, -0.4, 0.2])
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v + 2.0 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)


def host_norms(point: np.ndarray) -> tuple[float, float]:
    made = CalibratedMultipole(calibration=asset("navion_2")).actuation(m(tuple(point)))
    currents = np.asarray(CURRENTS)
    return float(np.linalg.norm(made.field @ currents)), float(np.linalg.norm(made.gradient @ currents))


@pytest.mark.parametrize("laws", [(Magnetic.Force,), (Magnetic.Torque, Magnetic.Force)], ids=["force", "torque_force"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_shared_slot_is_read_at_the_magnets_centre_of_a_turned_body(dtype, laws):
    centre = POSITION + rotate(TURNED, CENTRE)
    with in_dtype(dtype):
        built = exposure_sphere("shared_offset", observe=Exposure("sphere", field="coils.magnetic"), latency=False,
                                laws=laws, position=tuple(POSITION), orientation=tuple(TURNED),
                                centre=tuple(CENTRE)).build()
        history = built.run(ms(4), history=True).history
    seen = {k: np.asarray(v, dtype=np.float64) for k, v in history["exposure:sphere"].items()}
    slot = {k: np.asarray(v, dtype=np.float64) for k, v in history["coils.magnetic"].items()}
    law = laws[0].__name__.removeprefix("Magnetic").lower()
    field, gradient = slot[f"flux_density_magnetic_{law}_sphere"], slot[f"gradient_magnetic_{law}_sphere"]
    eps = float(np.finfo(np.dtype(dtype)).eps)
    terms = term_sums(IDENTITY, centre)
    tolerance = (SLACK * eps * np.sqrt(3.0) * np.max(terms[0]), SLACK * eps * 3.0 * np.max(terms[1]))
    expected = host_norms(centre)
    assert np.all(np.abs(seen["flux_density"][1:] - expected[0]) <= tolerance[0])
    assert np.all(np.abs(seen["field_gradient"][1:] - expected[1]) <= tolerance[1])
    assert np.allclose(seen["flux_density"][1:], np.linalg.norm(field[1:], axis=1), rtol=8 * eps, atol=0.0)
    assert np.allclose(seen["field_gradient"][1:], np.linalg.norm(gradient[1:].reshape(-1, 9), axis=1),
                       rtol=8 * eps, atol=0.0)
    for wrong in (POSITION, POSITION + CENTRE):
        other = host_norms(wrong)
        assert abs(other[0] - expected[0]) > 20 * tolerance[0] and abs(other[1] - expected[1]) > 20 * tolerance[1]
