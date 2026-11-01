"""The ``tube`` wall kind, registered on the shipped asset ``unit_tube_R0.997`` (``mime.walls.TubeWall``; the regularised-Stokeslet
method's correction ``mime.nodes.hydro.mrs._tube.TUBE_CORRECTION``): a small tube experiment builds and runs with it, the
confined sphere (the current version, sampled quasi-uniformly) drags as Haberman and Sayre say, a read past the table's readable range (0.997 of the radius, the
file's own ``readable_rho``) is refused, a body longer than the table reaches is refused when planned, and an asset that
is not in the cache fails by name, naming the command that stores it, and downloads nothing.

The confined MRS matrix on the current samplings (MIME-ANO-123's stated risk, a matrix conditioned with a wall's
correction or a derivative): the sphere and the capsule 1.5 mm off the axis of the 5 mm tube, turned, at 0.1 mm, keep
the matrix with the wall's correction within the sampling suite's free-space bound at blob 0.5, and the derivative of
their confined resistance in their radius equals central differences; the control, the kept grid sphere 0.1.0 there,
breaks the bound (measured in phase 7m-a's review: at most 175 on the current samplings, 1.95e4 on the grid, and the
derivative within 4e-10).

The asset must be in the cache (``mime assets fetch unit_tube_R0.997``; CI stores it once per job). The tests never
download it."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tube_asset
from mrs_cases import WATER, drag_node, free_space, resistance_at
from tube_cases import TUBE_RADIUS, haberman_sayre, polar_spacing, sphere_in_a_tube

import mime.nodes.hydro.mrs._tube as tube_module
from mime.build import BuildRefused
from mime.build._maddening import node_state, parameter_leaves
from mime.core import quaternion
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.interactions import Drag
from mime.nodes.hydro.mrs import blob_squared, stokeslet_matrix, wall_correction_named
from mime.registry import EntryKind, lookup
from mime.scene import KEPT, Capsule, Scene, Sphere, Tube, Z
from mime.testing import CONDITION_BOUNDS
from mime.things import Body, Medium, Part, State, Structure
from mime.units import kg, kg_m2, m, m_per_s, m_per_s2, mm, ms, rad_per_s

RATIO = 0.1
#: The confined sphere's tolerance and resolutions, as ``test_tube_wall.py`` states them (the oracle tube's).
SPHERE_TOLERANCE, POLAR = 5e-3, (10, 11)


def test_the_tube_wall_kind_is_registered_on_the_shipped_asset_with_the_files_readable_range():
    record = lookup(EntryKind.WALL_KIND, "tube", "0.1.0")
    assert record is not None and record.cls.matches == "cylinder" and record.cls.reads == ("centre", "axis", "radius")
    assert wall_correction_named("tube", "0.1.0") is tube_module.TUBE_CORRECTION
    assert tube_module.shipped_unit_tube().axes[0].readable_range == (0.0, 0.997)
    assert tube_module.shipped_unit_tube() is tube_module.shipped_unit_tube()
    assert tube_asset.path().name == tube_module.assets.reference(tube_module.UNIT_TUBE).sha256


def published_at(built, offset: float) -> tuple[bool, np.ndarray]:
    """``wall_accepted`` and the resistance of the drag node with the ball moved ``offset`` of the tube's radius along x."""
    node = drag_node(built, "ball")
    pose = jnp.asarray((offset * TUBE_RADIUS.value, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=jnp.float64)
    out = node.update(node_state(built.graph, "drag:ball"), {"pose": pose}, 1e-3,
                      params=parameter_leaves(built.graph)["drag:ball"])
    return bool(out["wall_accepted"]), np.asarray(out["resistance"])


@pytest.fixture(scope="module")
def built():
    return sphere_in_a_tube("registered", radius=RATIO, spacing=polar_spacing(RATIO, 6), wall_kind="tube").build()


def test_the_registered_tube_wall_kind_builds_runs_and_refuses_a_read_past_the_readable_range(built):
    """A sphere of a tenth of the radius reaches 0.9 of it at an offset of 0.8 (read) and 1.05 at 0.95 (refused: NaN,
    not accepted, as the file's range ends at 0.997)."""
    built.run(ms(3))
    inside, value = published_at(built, 0.8)
    outside, beyond = published_at(built, 0.95)
    print(f"\naccepted at 0.8 of the radius: {inside}; at 0.95: {outside}")
    assert inside and np.all(np.isfinite(value))
    assert not outside and not np.all(np.isfinite(beyond))


def drag_ratio(polar: int) -> float:
    spacing = polar_spacing(RATIO, polar)
    ball = Sphere(radius=m(RATIO * TUBE_RADIUS.value))
    confined = resistance_at(sphere_in_a_tube("confined", radius=RATIO, spacing=spacing, wall_kind="tube",
                                              hull=ball).build(), "ball")
    free = resistance_at(free_space("free", {"ball": ball},
                                    Drag.MRS(spacing=spacing, blob=0.5, wall_kind="free_space")).build(), "ball")
    return float(confined[2, 2] / free[2, 2])


def test_the_confined_sphere_on_the_shipped_table_drags_as_haberman_and_sayre_say():
    expected = haberman_sayre(RATIO)
    coarse, fine = POLAR
    found = [drag_ratio(polar) / expected - 1.0 for polar in POLAR]
    extrapolated = (fine * found[1] - coarse * found[0]) / (fine - coarse)
    print(f"\nK = {expected:.5f}; misses at {POLAR} polar nodes {np.round(found, 5)}, extrapolated {extrapolated:.5f}")
    assert expected - 1 >= 3 * SPHERE_TOLERANCE and abs(extrapolated) <= SPHERE_TOLERANCE


def test_a_body_longer_than_the_shipped_table_reaches_is_refused_when_planned():
    with pytest.raises(BuildRefused) as refused:
        sphere_in_a_tube("too_long", radius=RATIO, spacing=mm(1.0), wall_kind="tube",
                         hull=Capsule(radius=mm(0.3), height=mm(30.0), axis=(0.0, 0.0, 1.0))).plan()
    assert "drag.wall-range" in refused.value.codes()


def test_an_asset_that_is_not_in_the_cache_fails_by_name_and_downloads_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path))
    monkeypatch.setenv("MIME_OFFLINE", "1")
    with pytest.raises(BuildRefused) as refused:
        sphere_in_a_tube("missing", radius=RATIO, spacing=mm(1.0), wall_kind="tube").plan()
    message = str(refused.value)
    print(f"\n{message}")
    assert refused.value.codes() == ["asset.missing"]
    assert "unit_tube_R0.997" in message and "mime assets fetch unit_tube_R0.997" in message
    assert list(tmp_path.iterdir()) == []


#: Turned about an axis across the tube's, so that no sample's pattern lines up with the tube's.
TURNED = (math.cos(0.3), 0.6 * math.sin(0.3), 0.0, 0.8 * math.sin(0.3))


def off_axis(kind, spacing):
    """``kind`` 1.5 mm off the axis of the 5 mm tube, turned, in water, with the rung on the shipped table."""
    experiment = Experiment("off_axis", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=(0.0, 0.0, 1.0), inner_radius=TUBE_RADIUS,
                                                             wall=mm(0.5), length=mm(200.0))))
    experiment.add_medium(Medium("water", fills="tube.lumen"))
    experiment.declare("water.viscosity", WATER, reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(kind)}, initial=State(
        position=mm((1.5, 0.0, 0.0)), orientation=TURNED, velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("ball.mass", kg(1e-7), reason="a millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-14, 1e-14, 1e-14)), reason="a millimetre body's order")
    experiment.attach(Drag.MRS(spacing=spacing, blob=0.5, wall_kind="tube"), body="ball", medium="water")
    return experiment.build()


def confined_condition(built) -> float:
    """The condition number of the MRS matrix with the wall's correction, ``A + G``, on the node's samples at the
    body's pose, as its step assembles it (its symmetric part's extreme eigenvalues)."""
    node = drag_node(built, "ball")
    params = parameter_leaves(built.graph)["drag:ball"]
    pose = jnp.asarray(node_state(built.graph, "ball")["pose"], dtype=jnp.float64)
    held = node.samples(params)
    points = pose[:3] + jnp.matmul(held.points, quaternion.rotation_matrix(pose[3:]).T, precision="highest")
    wall = wall_correction_named("tube", "0.1.0")
    step = node._step_constants(params)  # noqa: SLF001 - the step's constants, the wall's table among them
    corrected = wall.correction(points, {name: step[f"wall_{name}"] for name in wall.parameters}, params["viscosity"])
    assert bool(corrected.accepted)
    matrix = np.asarray(stokeslet_matrix(points, blob_squared(held.weights, params["blob"]), params["viscosity"])
                        + corrected.matrix)
    values = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    return float(values[-1] / values[0]) if values[0] > 0.0 else math.inf


def radius_derivative(built) -> tuple[float, float]:
    """The derivative of a weighted sum of the confined resistance in the kind's radius, through the node's params
    (its sampling's choice held, as a study's step holds it), and its central difference at a millionth."""
    node = drag_node(built, "ball")
    params = parameter_leaves(built.graph)["drag:ball"]
    state = node_state(built.graph, "drag:ball")
    pose = jnp.asarray(node_state(built.graph, "ball")["pose"], dtype=jnp.float64)
    weights = jnp.asarray(np.random.default_rng(5).uniform(0.5, 1.5, (6, 6)))

    def loss(radius):
        return jnp.sum(weights * node.update(state, {"pose": pose}, 1e-3,
                                             params={**params, "shape_radius": radius})["resistance"])

    radius = params["shape_radius"]
    step = 1e-6 * float(radius)
    value = jax.jit(loss)
    central = (float(value(radius + step)) - float(value(radius - step))) / (2.0 * step)
    return float(jax.jit(jax.grad(loss))(radius)), central


CONFINED = {"sphere": Sphere(radius=mm(0.5)), "capsule": Capsule(radius=mm(0.3), height=mm(1.0), axis=(0.0, 1.0, 0.0))}


@pytest.mark.memory_heavy
@pytest.mark.parametrize("body", sorted(CONFINED))
def test_the_confined_matrix_stays_conditioned_and_its_radius_derivative_is_right_on_the_current_sampling(body):
    built = off_axis(CONFINED[body], mm(0.1))
    condition = confined_condition(built)
    automatic, central = radius_derivative(built)
    print(f"\n{body}: the confined matrix's condition {condition:.3g}; d/d radius {automatic:.12g} by AD, "
          f"{central:.12g} by central differences")
    assert condition <= CONDITION_BOUNDS[0.5]
    assert abs(automatic - central) <= 1e-8 * abs(central)


def test_control_the_kept_grid_sphere_breaks_the_bound_in_the_tube():
    assert confined_condition(off_axis(KEPT[("sphere", "0.1.0")][0](radius=mm(0.5)), mm(0.1))) > CONDITION_BOUNDS[0.5]
