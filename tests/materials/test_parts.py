"""The solids and the part resolver.

- Each solid's closed-form volume and inertia against an independent oracle, a Monte Carlo integral over its
  bounding box; the control: a solid planted with a wrong inertia misses the oracle.
- The resolver by the parallel-axis theorem: two boxes either side of the origin give the closed form; a part inside
  another adds nothing; the magnets' moments sum; a body whose parts put its centre of mass off its origin, or a
  part with no density, is refused.
- A body bound to an asset takes its parts' placement from the asset: the test robot's magnet, turned 90 degrees
  about z, is magnetised along the body's y.
"""

import math

import numpy as np
import pytest
from robot_asset import bound_robot_with_parts
from script_a import CONDITIONS, N45_RECORD

from mime import Body, Experiment, Medium, State, Structure
from mime.build import BuildRefused
from mime.dynamics import RigidInertial
from mime.interactions import Gravity
from mime.scene import Box, Cylinder, Ellipsoid, Scene, Tube, X, Z
from mime.things import Part, PartMass, PermanentMagnet, derive
from mime.units import VACUUM_PERMEABILITY, kg_per_m3, m_per_s, m_per_s2, mm, ms, rad_per_s

RNG_SEED, SAMPLES = 20261023, 400_000


def monte_carlo(inside, half: tuple[float, float, float]) -> tuple[float, np.ndarray]:
    """The volume and the inertia tensor per unit mass of the region ``inside`` accepts within the box of
    half-widths ``half``, by uniform sampling."""
    rng = np.random.default_rng(RNG_SEED)
    points = (rng.random((SAMPLES, 3)) * 2 - 1) * np.asarray(half)
    kept = points[inside(points)]
    volume = 8 * half[0] * half[1] * half[2] * len(kept) / SAMPLES
    square = np.sum(kept * kept, axis=1)
    tensor = np.mean(square)[None, None] * np.eye(3) - kept.T @ kept / len(kept)
    return volume, tensor


SOLIDS = {
    "cylinder": (Cylinder(radius=mm(1.5), length=mm(7.0), axis=(1.0, 1.0, 0.0)),
                 {"radius": 1.5e-3, "length": 7e-3, "axis": (math.sqrt(0.5), math.sqrt(0.5), 0.0)},
                 lambda p: (np.abs(p @ np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0])) <= 3.5e-3)
                 & (np.sum(p * p, axis=1) - (p @ np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0])) ** 2 <= 1.5e-3 ** 2),
                 (4e-3, 4e-3, 1.5e-3)),
    "ellipsoid": (Ellipsoid(semi_axes=mm((1.0, 2.0, 3.0))), {"semi_axes": (1e-3, 2e-3, 3e-3)},
                  lambda p: (p[:, 0] / 1e-3) ** 2 + (p[:, 1] / 2e-3) ** 2 + (p[:, 2] / 3e-3) ** 2 <= 1.0,
                  (1e-3, 2e-3, 3e-3)),
    "box": (Box(size=mm((1.0, 2.0, 4.0))), {"size": (1e-3, 2e-3, 4e-3)},
            lambda p: np.ones(len(p), dtype=bool), (0.5e-3, 1e-3, 2e-3)),
}


@pytest.mark.parametrize("name", list(SOLIDS))
def test_each_solids_volume_and_inertia_match_a_monte_carlo_integral(name):
    solid, settings, inside, half = SOLIDS[name]
    volume, tensor = monte_carlo(inside, half)
    closed = np.asarray(solid.inertia_per_mass(settings))
    assert solid.volume(settings) == pytest.approx(volume, rel=1e-2)
    assert np.max(np.abs(closed - tensor)) <= 1e-2 * np.max(np.abs(closed))


def test_the_control_a_wrong_inertia_misses_the_oracle():
    """A cylinder planted with r^2/3 about its axis, not r^2/2."""
    solid, settings, inside, half = SOLIDS["cylinder"]
    _, tensor = monte_carlo(inside, half)
    closed = np.asarray(solid.inertia_per_mass(settings))
    a = np.asarray(settings["axis"])
    planted = closed - (1.5e-3 ** 2 / 2 - 1.5e-3 ** 2 / 3) * np.outer(a, a)
    assert np.max(np.abs(planted - tensor)) > 1e-2 * np.max(np.abs(closed))


def box(name: str, at: tuple[float, float, float], inside: str | None = None) -> PartMass:
    solid, settings = Box(size=mm((1.0, 1.0, 1.0))), {"size": (1e-3, 1e-3, 1e-3)}
    return PartMass(name, "box", solid.volume(settings), at, solid.inertia_per_mass(settings), inside)


def test_the_resolver_sums_parts_by_the_parallel_axis_theorem():
    a = 2e-3
    parts = [box("left", (-a, 0.0, 0.0)), box("right", (a, 0.0, 0.0)), box("magnet", (0.0, 0.0, 0.0), "left")]
    found = derive("b", parts, {"left": 1000.0, "right": 1000.0}, {"magnet": (0.0, 1e6, 0.0)})
    m = 1000.0 * 1e-9
    own = m * (1e-6 + 1e-6) / 12
    assert found["mass"].value == pytest.approx(2 * m, rel=1e-15)
    assert found["inertia"].value == pytest.approx((2 * own, 2 * own + 2 * m * a * a, 2 * own + 2 * m * a * a),
                                                   rel=1e-14)
    assert found["buoyant_volume"].value == pytest.approx(2e-9, rel=1e-15)
    assert found["centre_of_buoyancy"].value == (0.0, 0.0, 0.0)
    assert found["magnetic_moment"].value == pytest.approx((0.0, 1e6 * 1e-9, 0.0), rel=1e-15)
    assert any("inside left" in line for line in found["mass"].details)


def test_a_centre_of_mass_off_the_origin_and_a_missing_density_are_problems():
    off = derive("b", [box("left", (-1e-3, 0.0, 0.0)), box("right", (2e-3, 0.0, 0.0))],
                 {"left": 1000.0, "right": 1000.0}, {})
    assert off["inertia"].value is None and off["inertia"].problems[0][0] == "parts.centre-of-mass"
    lacking = derive("b", [box("left", (0.0, 0.0, 0.0))], {}, {})
    assert lacking["mass"].problems[0][0] == "parts.density"
    assert lacking["buoyant_volume"].value == pytest.approx(1e-9, rel=1e-15)  # the volume needs no density


def python_first(parts: dict, magnets: dict, declared=()) -> Experiment:
    experiment = Experiment("parts", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5),
                            conditions=CONDITIONS)
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                    length=mm(100.0))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen))
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water")
    body = experiment.add_body(Body("b", dynamics=RigidInertial(), parts=parts, magnets=magnets, initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in declared:
        experiment.declare(path, value, reason="the test's")
    experiment.attach(Gravity(), body=body, medium=water)
    return experiment


def test_a_part_with_no_density_is_refused_naming_what_to_declare():
    experiment = python_first({"shell": Part(Box(size=mm((1.0, 1.0, 1.0))))}, {})
    with pytest.raises(BuildRefused) as caught:
        experiment.plan()
    found = [r for r in caught.value.refusals if r.code == "value.undeclared"]
    assert found and "b.parts.shell.density" in found[0].action
    declared = python_first({"shell": Part(Box(size=mm((1.0, 1.0, 1.0))))}, {},
                            [("b.parts.shell.density", kg_per_m3(1200.0))])
    assert {v.path: v.value for v in declared.plan().values}["b.mass"] == pytest.approx(1.2e-6, rel=1e-15)


def test_a_magnet_among_python_first_parts_needs_its_direction_declared():
    parts = {"shell": Part(Box(size=mm((2.0, 2.0, 2.0)))),
             "magnet": Part(Box(size=mm((1.0, 1.0, 1.0))), material=N45_RECORD, inside="shell")}
    with pytest.raises(BuildRefused) as caught:
        python_first(parts, {"magnet": PermanentMagnet()}).plan()
    assert "magnet.direction-missing" in [r.code for r in caught.value.refusals]
    plans = python_first(parts, {"magnet": PermanentMagnet(direction=(0.0, 0.0, 1.0))},
                         [("b.parts.shell.density", kg_per_m3(1200.0))]).plan()
    (magnet,) = plans.scene.magnets
    assert (magnet.key, magnet.in_body, magnet.source) == ("magnet", (0.0, 0.0, 1.0), "declared in the experiment")


def test_a_bound_bodys_parts_are_placed_by_its_asset():
    plans = bound_robot_with_parts().plan()
    values = {v.path: v.value for v in plans.values}
    moment = values["UMR.magnetic_moment"]
    volume = math.pi * 0.5e-3 ** 2 * 2e-3
    assert moment == pytest.approx((0.0, 1.345 / VACUUM_PERMEABILITY * volume, 0.0), abs=1e-15)
    (magnet,) = [part for part in plans.materials.parts if part.name == "Magnets"]
    assert np.allclose(magnet.rotation, ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), atol=1e-12)
    shell = 1410.0 * 4 / 3 * math.pi * 1.56e-3 ** 3
    assert values["UMR.mass"] == pytest.approx(shell, rel=1e-14)
    assert values["UMR.inertia"] == pytest.approx((2 / 5 * shell * 1.56e-3 ** 2,) * 3, rel=1e-14)
