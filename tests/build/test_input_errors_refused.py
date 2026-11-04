"""Three input errors that gave a confident number are refused in every profile, each with its code and the line to
write, as ``docs/profiles.md`` says input errors are; each beside its control, the same experiment with the input in
its range, which builds (or runs):

- **a dynamic viscosity of zero or less** (``value.not-positive``), read by a drag or by nothing; below zero, the line
  writes its magnitude;
- **a body whose centre starts outside the region its medium fills** (``state.outside-region``), with the position
  that starts it on the region's axis; a body inside, anywhere along the axis, and one in a medium that fills no
  region, are taken;
- **an observation's window that starts at or after the run's end** (``run.window-past-end``), which read as a mean of
  zero: the reading is refused, its text says nothing was observed, and ``outputs.json`` writes no number for it; a
  window inside the run is read, and so is one a later run of the same graph reaches."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime.build import ArgumentRefused, BuildRefused
from mime.dynamics import RigidInertial
from mime.experiment import RUN_OUTPUTS
from mime.fields import UniformRotating
from mime.interactions import Drag, Gravity, Magnetic
from mime.observations import SwimSpeed
from mime.results import RUN_MANIFEST
from mime.scene import Scene, Sphere, Tube, X, Z
from mime.things import Part
from mime.units import A_m2, Hz, kg, kg_m2, kg_per_m3, m_per_s, mm, mm_per_s, mPa_s, ms, mT, rad_per_s, s, um


def refused(build) -> list:
    try:
        build()
    except BuildRefused as error:
        return list(error.refusals)
    return []


def coded(refusals, code: str) -> list:
    return [refusal for refusal in refusals if refusal.code == code]


def settling(*, viscosity=mPa_s(1.0), drag: bool = True, profile: str = "exploratory",
             position=um((0.0, 0.0, 0.0)), tube: bool = False) -> Experiment:
    """A steel ball settling in water (in a tube where ``tube``), its viscosity and position as given."""
    exp = Experiment("ball", scene=Scene.new(up=Z, units="m"), dt=ms(0.05), profile=profile)
    lumen = None
    if tube:
        lumen = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                  length=mm(100.0)))).lumen
    exp.add_medium(Medium("water", fills=lumen) if lumen is not None else Medium("water"))
    exp.declare("water.density", kg_per_m3(998.0), reason="water at 20 degC")
    exp.declare("water.viscosity", viscosity, reason="water at 20 degC")
    exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=position, orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"sphere": Part(Sphere(radius=um(50.0)))}))
    exp.declare("ball.parts.sphere.density", kg_per_m3(7850.0), reason="steel")
    if drag:
        exp.attach(Drag.MRS(spacing=um(20.0), blob=0.5), body="ball", medium="water")
    exp.attach(Gravity(), body="ball", medium="water")
    exp.observe(SwimSpeed("ball", along=-Z, after=s(0.0)))
    return exp


@pytest.mark.parametrize("profile", ["exploratory", "research"])
@pytest.mark.parametrize("drag", [True, False], ids=["read by the drag", "read by nothing"])
def test_a_viscosity_below_zero_is_refused_with_its_magnitude_to_write(profile, drag):
    (held,) = coded(refused(lambda: settling(viscosity=mPa_s(-1.0), drag=drag, profile=profile).build()),
                    "value.not-positive")
    assert "water.viscosity is -0.001 Pa s" in held.message and "zero or less" in held.message
    assert held.action.startswith("declare it above zero, in its unit: exp.declare('water.viscosity', mPa_s(1.0), "
                                  "reason=\"...\") if the minus sign was the slip")


def test_a_viscosity_of_zero_is_refused_with_a_typical_value_to_write():
    (held,) = coded(refused(lambda: settling(viscosity=mPa_s(0.0)).build()), "value.not-positive")
    assert "mPa_s(1.0)" in held.action and "water near 20 degC is about 1" in held.action


def test_control_a_viscosity_above_zero_builds():
    assert refused(lambda: settling(viscosity=mPa_s(1.0)).build()) == []
    assert refused(lambda: settling(viscosity=mPa_s(1e-3), drag=False).build()) == [], "small, and above zero"


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_a_body_whose_centre_starts_outside_the_lumen_is_refused_with_the_position_on_its_axis(profile):
    (held,) = coded(refused(lambda: settling(position=mm((10.0, 0.0, 5.0)), tube=True, drag=False,
                                             profile=profile).build()), "state.outside-region")
    assert "body 'ball' starts at (10, 0, 5) mm, outside tube.lumen" in held.message
    assert "1.825 mm beyond its wall" in held.message and "gravity:ball" in held.message
    assert "State(position=mm((10, 0, 0)))" in held.action


def test_control_a_body_inside_the_lumen_or_in_a_medium_filling_no_region_is_taken():
    for position in (mm((0.0, 0.0, 0.0)), mm((40.0, 0.0, 3.0)), mm((0.0, -2.0, 2.0))):
        assert coded(refused(lambda: settling(position=position, tube=True, drag=False).build()),
                     "state.outside-region") == []
    assert coded(refused(lambda: settling(position=mm((0.0, 0.0, 50.0)), drag=False).build()),
                 "state.outside-region") == []


def spinning(after) -> Experiment:
    """A magnetised body turned by a field, observed from ``after``, at a base step of 0.1 s."""
    exp = Experiment("window", scene=Scene.new(up=Z, units="m"), dt=ms(100))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.declare("umr.mass", kg(1e-6), reason="a millimetre body")
    exp.declare("umr.inertia", kg_m2((1e-12, 1e-12, 1e-12)), reason="a millimetre body")
    exp.declare("umr.magnetic_moment", A_m2((0.0, 1e-3, 0.0)), reason="a millimetre body's magnet")
    exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(rotation_axis=(1.0, 0.0, 0.0),
                                                                             initial_direction=(0.0, 1.0, 0.0))},
                                commands={"frequency": Hz(0.1), "amplitude": mT(1e-6)}))
    exp.attach(Magnetic.Torque(), body="umr", field="field.magnetic")
    exp.observe(SwimSpeed("umr", along=X, after=after))
    return exp


@pytest.mark.parametrize("after", [s(2.0), s(1.0)], ids=["after the end", "at the end"])
def test_a_window_that_starts_at_or_after_the_runs_end_is_refused_when_read_and_says_so_where_it_is_shown(after,
                                                                                                         tmp_path):
    result = spinning(after).build().run(s(1.0))
    reading = result["swim_speed:umr"].mean_speed
    with pytest.raises(ArgumentRefused) as error:
        reading.to(mm_per_s)
    (held,) = error.value.refusals
    assert held.code == "run.window-past-end" and "swim_speed:umr.mean_speed's window starts at" in held.message
    assert "observed nothing" in held.message and "run(s(...))" in held.action and "--duration" in held.action
    for read in (lambda: reading.value, lambda: reading.quantity, lambda: reading.quantity.to(mm_per_s)):
        with pytest.raises(ArgumentRefused) as error:
            read()  # each read is refused, the quantity's among them
        assert error.value.refusals[0].code == "run.window-past-end"
    assert "nothing observed: its window starts at" in result.text() and " = 0.0 m/s" not in result.text()
    written = spinning(after).build().run_to(tmp_path / "run", s(1.0))
    manifest = json.loads((tmp_path / "run" / RUN_MANIFEST).read_text())
    (entry,) = [o for held in manifest["observations"] for o in held["outputs"] if o["name"] == "mean_speed"]
    assert entry["value"] is None, "outputs.json writes no number for a window nothing was observed in"
    with np.load(tmp_path / "run" / RUN_OUTPUTS) as outputs:
        assert np.isnan(outputs[entry["array"]]).all(), "outputs.npz holds NaN for it, never the zero its node holds"
    assert "nothing observed" in written.result.text()


def test_control_a_window_inside_the_run_is_written_as_its_number_in_outputs_npz(tmp_path):
    spinning(s(0.5)).build().run_to(tmp_path / "run", s(1.0))
    manifest = json.loads((tmp_path / "run" / RUN_MANIFEST).read_text())
    (entry,) = [o for held in manifest["observations"] for o in held["outputs"] if o["name"] == "mean_speed"]
    with np.load(tmp_path / "run" / RUN_OUTPUTS) as outputs:
        held = outputs[entry["array"]]
    assert np.isfinite(held).all() and held.tolist() == pytest.approx(entry["value"])


def test_control_a_window_inside_the_run_is_read_and_a_later_run_reads_into_it():
    built = spinning(s(0.5)).build()
    assert built.run(s(1.0))["swim_speed:umr"].observed.to(s) == pytest.approx(0.5)
    later = spinning(s(1.5)).build()
    with pytest.raises(ArgumentRefused):
        later.run(s(1.0))["swim_speed:umr"].mean_speed.to(mm_per_s)
    reached = later.run(s(1.0))["swim_speed:umr"]  # the graph runs on from 1 s: its window is now inside
    assert reached.observed.to(s) == pytest.approx(0.5) and reached.mean_speed.to(mm_per_s) is not None
