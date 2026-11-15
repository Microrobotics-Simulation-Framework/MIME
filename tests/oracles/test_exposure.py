"""The exposure envelope (``mime.observations.Exposure``): what a body is exposed to, each step, at the point where a
law bound to it reads the field, against declared limits, with a flag per quantity and the maxima.

**What it bounds.** The three quantities are functions of the field the law reads, so each is held against the
field's own oracle: on ``mount_cases``'s still sphere at Navion_2's centre, ``flux_density`` and ``field_gradient`` against
the host's ``actuation`` (float64) at the sphere, ``|B|`` and the Frobenius norm of the gradient, within ``SLACK eps /
c`` (``c`` the worst field cancellation of Navion_2's sources at the point, the floor rounding puts on a sum of
sources that cancel; ``SLACK`` as in ``tests/oracles/test_calibrated_multipole.py``), in float64 and float32; and each
against the norm of what the law's slot holds, to the norm's rounding. ``rate_of_change`` is a one-step difference,
``|B_n - B_(n-1)| / dt``, of what the slot holds: an **estimate** of ``|dB/dt|`` along the body's path. On the mounted
rig (the array moving round the still sphere) it is held against the host's derivative of the field along the rotor's
turn, central differences of ``actuation`` at the rotor's pose: within ``dt`` times the largest second derivative
there, plus its rounding (``2 SLACK eps / c`` of the field over ``dt``); in float64 the miss is that first-order term,
of order ``dt``, and not rounding. In a uniform field turning at ``f`` it is the chord, ``2 B sin(pi f dt) / dt``,
to rounding, against ``|dB/dt| = 2 pi f B`` in the continuum.

**Step 0.** No earlier reading exists at step 0, so its rate of change is not measured: held at 0, it raises no flag
and sets no maximum. A field on from the start is never seen switching on; one switched on between step 0 and step 1
(its currents through a port that delays them a step) is seen at step 1, as ``|B_1| / dt``, and flagged there.

**The flags** are a report against a number the experiment declares, raised at a step where the quantity is above it:
at a limit just below the quantity's value (a thousandth below) the flag is raised at every step the value stands
above it, and at a limit just above, never. Their detection rates are measured on a runner (``detection_rate.py``),
each planted above its limit and clean below it.

**Refusals**, each with its control: a limit with no reason, a pending one, or one not above zero
(``exposure.declaration``; with a reason it builds); and an exposure at a body no law reads the field at
(``observation.field-read``; with the force attached it builds). **The read** shares the law's slot: the provider's
node and the physics' derived hash are what they are without the exposure."""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Iterator
from functools import cache
from typing import Any

import jax
import numpy as np
import pytest

from mount_cases import CURRENTS, ROTOR, ROTOR_AT, exposure_sphere, mounted_rig, still_sphere, \
    scene_experiment
from mount_cases import TURN as PLACEMENT_TURN

from mime import Apparatus
from mime.build import BuildRefused
from mime.build._derived import derive
from mime.build._plan import Section
from mime.data import asset
from mime.fields import CalibratedMultipole, UniformRotating
from mime.fields._coil_array import calibration_of
from mime.interactions import Magnetic
from mime.observations import Exposure
from mime.observations._exposure import Exposure as ExposureClass
from mime.units import Hz, T, T_per_m, T_per_s, m, mT, ms
from mime.validity import pending

jnp: Any = jax.numpy
SLACK = 10.0
DT = 1e-3
STEPS = 12
#: The plant a control's detection rate is measured with (``AP3B_PLANT``): ``declaration``, the check of a limit's
#: reason removed.
PLANT = os.environ.get("AP3B_PLANT", "")
OBSERVER = "exposure:sphere"


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def reasoned(**limits: Any) -> Exposure:
    return Exposure("sphere", field="coils.magnetic", **limits)


def run(experiment, steps: int = STEPS) -> dict[str, dict[str, np.ndarray]]:
    held = experiment.build().run(ms(steps), history=True).history
    return {node: {field: np.asarray(value, dtype=np.float64) for field, value in fields.items()}
            for node, fields in held.items()}


@cache
def host_field() -> tuple[np.ndarray, np.ndarray]:
    """The field and gradient at Navion_2's centre at ``CURRENTS``, float64, by the host's ``actuation``."""
    made = CalibratedMultipole(calibration=asset("navion_2")).actuation(m((0.0, 0.0, 0.0)))
    return made.field @ np.asarray(CURRENTS), made.gradient @ np.asarray(CURRENTS)


def floor(dtype: str) -> float:
    """``eps / c`` at Navion_2's centre: the share of the field that rounding in the sum of its sources costs."""
    return float(np.finfo(np.dtype(dtype)).eps) / calibration_of(asset("navion_2")).cancellation(np.zeros((1, 3)))


# -- the quantities -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_each_quantity_is_the_field_the_law_reads_at_the_body_against_actuation(dtype):
    with in_dtype(dtype):
        history = run(exposure_sphere(observe=reasoned()))
    seen, slot = history[OBSERVER], history["coils.magnetic"]
    field, gradient = slot["flux_density_magnetic_force_sphere"], slot["gradient_magnetic_force_sphere"]
    eps = float(np.finfo(np.dtype(dtype)).eps)
    assert np.allclose(seen["flux_density"], np.linalg.norm(field, axis=1), rtol=8 * eps, atol=0.0)
    assert np.allclose(seen["field_gradient"], np.linalg.norm(gradient.reshape(-1, 9), axis=1), rtol=8 * eps, atol=0.0)
    host, steep = host_field()
    assert seen["flux_density"][0] == 0.0 and seen["field_gradient"][0] == 0.0, "the field is off at step 0"
    tolerance = SLACK * floor(dtype)
    assert np.all(np.abs(seen["flux_density"][1:] - np.linalg.norm(host)) <= tolerance * np.linalg.norm(host))
    assert np.all(np.abs(seen["field_gradient"][1:] - np.linalg.norm(steep)) <= tolerance * np.linalg.norm(steep))
    assert np.all(seen["max_flux_density"][1:] == np.maximum.accumulate(seen["flux_density"])[1:])
    assert seen["max_field_gradient"][-1] == seen["field_gradient"].max()


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_step_0s_rate_of_change_is_not_measured_and_a_field_switched_on_at_step_1_is_caught_there(dtype):
    """Held at 0, step 0's rate of change raises no flag at any limit: a field on from the start is never seen
    switching on. The control, the field switched on between step 0 and step 1, is seen at step 1 as ``|B_1| / dt``
    and flagged there alone."""
    limit = T_per_s(1e-9, reason="below any change of the field")
    with in_dtype(dtype):
        from_start = run(exposure_sphere(observe=reasoned(rate_of_change=limit), latency=False))[OBSERVER]
        switched = run(exposure_sphere(observe=reasoned(rate_of_change=limit)))[OBSERVER]
    assert np.all(from_start["rate_of_change"] == 0.0) and np.all(from_start["above_rate_of_change"] == 0.0)
    assert from_start["max_rate_of_change"][-1] == 0.0
    assert switched["rate_of_change"][0] == 0.0 and switched["above_rate_of_change"][0] == 0.0
    expected = np.linalg.norm(host_field()[0]) / DT
    assert abs(switched["rate_of_change"][1] - expected) <= SLACK * floor(dtype) * expected
    assert list(np.flatnonzero(switched["above_rate_of_change"])) == [1], "flagged at step 1 alone"
    assert np.all(switched["rate_of_change"][2:] == 0.0), "a field that holds still has no rate of change"


@pytest.mark.parametrize("quantity", ["flux_density", "field_gradient", "rate_of_change"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_flag_is_raised_at_the_steps_its_quantity_stands_above_its_limit_and_never_below(quantity, dtype):
    host, steep = host_field()
    value = {"flux_density": np.linalg.norm(host), "field_gradient": np.linalg.norm(steep),
             "rate_of_change": np.linalg.norm(host) / DT}[quantity]
    unit = {"flux_density": T, "field_gradient": T_per_m, "rate_of_change": T_per_s}[quantity]
    raised = []
    for factor in (0.999, 1.001):
        with in_dtype(dtype):
            seen = run(exposure_sphere(observe=reasoned(**{quantity: unit(value * factor, reason="the test's")})))
        raised.append(list(np.flatnonzero(seen[OBSERVER][f"above_{quantity}"])))
    above = [1] if quantity == "rate_of_change" else list(range(1, STEPS))
    assert raised == [above, []], (quantity, raised)


@cache
def mounted_exposure(dtype: str) -> dict[str, dict[str, np.ndarray]]:
    with in_dtype(dtype):
        return run(mounted_rig("mounted_exposure", observe=Exposure("sphere", field="rig.coils")), 40)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v + 2.0 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def field_at_turn(start: np.ndarray, turn: float) -> np.ndarray:
    """The mounted array's field at the sphere, float64, with the rotor turned by ``turn`` from its pose ``start``:
    the array placed in the world there, by the host's ``actuation``."""
    rotor = np.concatenate([start[:3], multiply(start[3:], np.array([math.cos(turn / 2), 0.0, 0.0, math.sin(turn / 2)]))])
    placement = np.asarray((*ROTOR_AT, *PLACEMENT_TURN))
    world = np.concatenate([rotor[:3] + rotate(rotor[3:], placement[:3]), multiply(rotor[3:], placement[3:])])
    array = CalibratedMultipole(calibration=asset("navion_2"), at=m(tuple(world[:3])),
                                orientation=tuple(world[3:] / np.linalg.norm(world[3:])))
    return array.actuation(m((0.0, 0.0, 0.0))).field @ np.asarray(CURRENTS)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_rate_of_change_estimates_dB_dt_along_the_moving_chain_to_order_dt(dtype):
    history = mounted_exposure(dtype)
    seen = history[OBSERVER]
    field = history["rig.coils"]["flux_density_magnetic_force_sphere"]
    eps = float(np.finfo(np.dtype(dtype)).eps)
    difference = np.linalg.norm(np.diff(field, axis=0), axis=1) / DT
    assert np.allclose(seen["rate_of_change"][1:], difference, rtol=64 * eps, atol=64 * eps * np.max(field) / DT)
    frames, angles = history["rig.motor"]["frame_poses"][:, ROTOR], history["rig.motor"]["angle"]
    speed, h = float(np.mean(np.diff(angles)) / DT), 1e-4
    misses, first_order = [], []
    for step in range(1, len(angles)):
        start = frames[step]
        made = [field_at_turn(start, k * h) for k in (-1, 0, 1)]
        derivative = speed * (made[2] - made[0]) / (2 * h)
        second = speed ** 2 * (made[2] - 2 * made[1] + made[0]) / h ** 2
        bound = DT * np.linalg.norm(second) + 2 * SLACK * floor(dtype) * np.linalg.norm(made[1]) / DT
        misses.append(abs(seen["rate_of_change"][step] - np.linalg.norm(derivative)) / bound)
        first_order.append(abs(seen["rate_of_change"][step] - np.linalg.norm(derivative)) /
                           (DT * np.linalg.norm(second)))
    assert max(misses) <= 1.0, max(misses)
    if dtype == "float64":
        assert np.median(first_order) > 1e-3, "the miss is the estimate's order-dt term, not rounding"


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_in_a_turning_uniform_field_the_rate_of_change_is_the_chord(dtype):
    """``2 B sin(pi f dt) / dt`` from step 1, against ``2 pi f B`` in the continuum; the gradient is zero."""
    frequency, amplitude = 10.0, 1.2e-3
    exp = scene_experiment("exposure_uniform")
    exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(
        rotation_axis=(0.0, 0.0, 1.0), initial_direction=(1.0, 0.0, 0.0))},
        commands={"frequency": Hz(frequency), "amplitude": T(amplitude)}))
    still_sphere(exp)
    exp.attach(Magnetic.Torque(), body="sphere", field="field.magnetic")
    exp.observe(Exposure("sphere", field="field.magnetic"))
    with in_dtype(dtype):
        seen = run(exp)[OBSERVER]
    eps = float(np.finfo(np.dtype(dtype)).eps)
    chord = 2 * amplitude * math.sin(math.pi * frequency * DT) / DT
    assert np.all(np.abs(seen["rate_of_change"][1:] - chord) <= 256 * eps * amplitude / DT)
    assert np.all(seen["field_gradient"] == 0.0)
    assert abs(chord - 2 * math.pi * frequency * amplitude) > 1e-6 * chord, "the estimate differs from dB/dt"


# -- refusals -----------------------------------------------------------------------------------------------------

@pytest.fixture
def plant(monkeypatch):
    if PLANT == "declaration":
        monkeypatch.setattr(ExposureClass, "declaration_problems", lambda self: [])


@pytest.mark.parametrize("limit, held", [
    (mT(40), True), (mT(40, reason=pending()), True), (mT(0.0, reason="a ceiling"), True),
    (mT(40, reason="the protocol's ceiling"), False)],
    ids=["no_reason", "pending_reason", "not_above_zero", "control_with_its_reason"])
def test_a_limit_with_no_reason_or_not_above_zero_is_refused(plant, limit, held):
    experiment = exposure_sphere(observe=reasoned(flux_density=limit))
    if not held:
        experiment.plan()
        return
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    found = [r for r in refused.value.refusals if r.code == "exposure.declaration"]
    assert found and "flux_density" in found[0].message


@pytest.mark.parametrize("laws", [(), (Magnetic.Force,)], ids=["no_law_reads_it", "control_the_force_reads_it"])
def test_an_exposure_at_a_body_no_law_reads_the_field_at_is_refused(laws):
    experiment = exposure_sphere(observe=reasoned(), laws=laws)
    if laws:
        experiment.plan()
        return
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    found = [r for r in refused.value.refusals if r.code == "observation.field-read"]
    assert found and "no law reads the field there" in found[0].message


def test_an_exposure_with_its_limits_round_trips_through_save_load_and_verify(tmp_path):
    """The document holds each limit with its reason, in the unit the saved form writes (``unit_symbol``, which
    ``parse_unit`` reads back); the folder loads with its input hash, rebuilds its derived hash and verifies."""
    import yaml

    from mime.experiment import Experiment
    from mime.saved import DOCUMENT_FILE
    from mime.units import FIELD_RATE, parse_unit, unit_symbol

    observe = reasoned(flux_density=mT(40, reason="the protocol's ceiling"),
                       rate_of_change=T_per_s(2.0, reason="the protocol's ceiling on the rate"))
    with in_dtype("float64"):
        built = exposure_sphere("exposure_saved", observe=observe).build()
        folder = tmp_path / "exposure_saved"
        built.save(folder)
        document = yaml.safe_load((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))
        written = next(o for o in document["observations"] if o["name"] == "exposure")["settings"]
        assert written["rate_of_change"]["reason"] == "the protocol's ceiling on the rate"
        assert parse_unit(written["rate_of_change"]["unit"]) == FIELD_RATE == parse_unit(unit_symbol(FIELD_RATE))
        loaded = Experiment.load(folder)
        assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()
        again = loaded.build()
        assert again.report.derived_hash == built.report.derived_hash and again.verify().reproduced


def test_the_exposure_adds_no_slot_and_leaves_the_physics_as_it_was():
    with in_dtype("float64"):
        observed, plain = exposure_sphere(observe=reasoned()).build(), exposure_sphere().build()
    nodes = [built.graph.get_node("coils.magnetic") for built in (observed, plain)]
    assert sorted(nodes[0].initial_state()) == sorted(nodes[1].initial_state())
    assert nodes[0].boundary_input_spec().keys() == nodes[1].boundary_input_spec().keys()
    hashes = [derive(built.graph, [name for plan in built.plans.entries if plan.section is Section.OBSERVATION
                                   for name in plan.node_names()]).physics_hash for built in (observed, plain)]
    assert hashes[0] == hashes[1]
    assert any("where magnetic.force:sphere reads it" in line for line in observed.report.checked[OBSERVER])
