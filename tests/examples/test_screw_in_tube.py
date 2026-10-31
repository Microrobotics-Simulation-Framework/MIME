"""``examples/screw_in_tube.py``: a magnetic screw swimming along a tube of glycerol, every value declared, as the
example runs it.

**What this acceptance test bounds, and what it does not.** It holds that the example builds, runs, swims forward in
step with the field and sits inside the range the rung is declared for. It does not bound the number's accuracy: the
example's speed at its starting spacing is far below the finest spacing the table measured, and that table's finest
spacing is itself no converged value (``docs/choosing_spacing.md``). The number the example prints is held to the
table's row at the same spacing (a run that drifted from the table fails here), not to a reference.

**The inertia.** The screw's parts give a tensor with a small product of inertia, which the build refuses and prints.
The example declares the diagonal the refusal prints: a test reads it back from the refusal and holds the example's
constant to it, so that the example cannot drift from what the build says. In a viscous regime the answer does not
depend on the inertia (the last test, at the coarsest spacing of the table, where the control is cheap)."""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from mime.build import BuildRefused
from mime.results._watch import WITHIN
from mime.units import s

ROOT = Path(__file__).resolve().parents[2]
TABLES = json.loads((ROOT / "docs" / "data" / "spacing_tables.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("example_screw_in_tube", ROOT / "examples" / "screw_in_tube.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def started(example):
    return example.run()


def test_the_screw_swims_forward_in_step_with_the_field_at_a_turning_reynolds_number_inside_the_range(example, started):
    result = started
    twist = np.asarray(result.history["umr"]["twist"])
    last_turn = twist[-round(1.0 / example.FREQUENCY_HZ / result.dt):]
    assert last_turn[:, 3].mean() == pytest.approx(2.0 * np.pi * example.FREQUENCY_HZ, rel=0.02), "synchronous"
    assert last_turn[:, 0].mean() > 0.0
    (watch,) = result.watches
    assert watch.status == WITHIN and watch.turning > watch.translation and watch.reynolds < 0.1


def test_the_speed_is_the_tables_row_at_the_same_spacing(example, started):
    row = next(r for r in TABLES["runs"]["screw"] if r["spacing_mm"] == example.START_SPACING_MM)
    speed = float(started["swim_speed:umr"].mean_speed.to(__import__("mime.units", fromlist=["m_per_s"]).m_per_s))
    assert speed == pytest.approx(row["speed_m_per_s"], rel=0.01)


def test_the_parts_own_tensor_is_refused_with_the_declaration_the_example_uses(example):
    from mime import Body, Experiment, State, Structure
    from mime.dynamics import RigidInertial
    from mime.scene import Scene, Screw, Tube, X, Z
    from mime.things import Part
    from mime.units import dimensionless, kg_per_m3, m_per_s, m_per_s2, mm, ms, rad_per_s

    exp = Experiment("refused", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(20.0))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                      length=mm(100.0))))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(Screw(
            radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
            starts=2))}))
    exp.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="FL-9 resin plus its magnets, mean density")
    with pytest.raises(BuildRefused) as refused:
        exp.build()
    (refusal,) = [r for r in refused.value.refusals if r.code == "parts.inertia-axes"]
    assert "in the body's frame (kg m^2)" in refusal.message and "principal moments" in refusal.message
    printed = re.search(r'kg_m2\(\(([^)]*)\)\)', refusal.action)
    assert printed is not None
    numbers = tuple(float(n) for n in printed.group(1).split(","))
    assert numbers == pytest.approx(example.INERTIA, rel=1e-9), "the example declares what the build prints"


def test_in_this_regime_the_answer_does_not_depend_on_the_inertia_declared(example):
    coarse = 0.83
    parts = example.screw_in_tube(coarse).build().run(s(example.DURATION_S))
    isotropic = example.screw_in_tube(coarse, inertia=(1.0e-10,) * 3).build().run(s(example.DURATION_S))
    a, b = (float(r["swim_speed:umr"].mean_speed.value) for r in (parts, isotropic))
    assert a > 0.0 and a == pytest.approx(b, rel=1e-3)
