"""The calibrated multipole (``calibrated_multipole`` 0.1.0) against ETH's ``mag_manip`` 3.0.0, per system (float64,
per module), from ``references/mpem_mag_manip.npz`` (its generator and provenance beside it):

- **The oracle**: the field per unit current (3 x K), the gradient's five rows of the actuation matrix, and the field and
  gradient at drawn currents, at 64 points in each declared region, from the node's own function and from the host's
  ``Actuation``, peak-normalised, each within ``10 eps / c`` of its peak, ``c`` the worst field cancellation at those
  points: two implementations summing the same near-cancelling sources in another order agree to that and no better.
  The tolerance is guarded: below 1e-9 on every system. The control: the directions read unnormalised (Navion_2's file
  holds them 1 + 4.9e-10 long) miss the oracle.
- **Planted errors fail on every draw**: a calibration's global sign, one coil's, its frame turned by 1 to 180 degrees,
  two of its axes swapped, its frame shifted by 1 to 100 mm. What this cannot do: a software oracle pins the sign and the
  frame to mag_manip's convention, not to hardware's (MIME-ANO-153); a measured field would be needed for that.
- **The exact transform**: the array placed at a drawn pose and read at the point the pose takes an oracle point to
  gives ``R B`` and ``R G R^T``; the turn applied the other way fails on every draw.
- **Maxwell, by construction**: the gradient is symmetric and trace-free to its rounding; an order-one source is MIME's
  point dipole of moment ``4 pi beta d / mu_0``; zero current gives zero field.
- **dev1's magnetic pair on a sphere** in the array's field, built: the force and torque its first step delivers are
  mag_manip's ``forceOnDipole`` and ``torqueOnDipole`` on the same moment, each system; with the energy ledger, the coil
  apparatus is outside and the pair supplied by it.
- **The background** moves the field by itself and the gradient by nothing; **the flags**: ``outside_region`` is
  mag_manip's ``pointInWorkspace`` negated at 48 probes, ``near_source`` rises within the guard. MADDENING's battery
  runs on the node in the node walk (``tests/saved/test_node_walk.py``, its case in ``tests/saved/node_cases.py``).
- **Allocation**: currents within the limits that make a field (and gradient) are recovered; out of reach, the limits
  hold and the miss is reported; a 3-coil array asked for both reports its rank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mime import Apparatus, Body, Experiment, State
from mime.build._maddening import _held, fed_inputs
from mime.build._refusals import ArgumentRefused
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.fields import CalibratedMultipole, allocate
from mime.fields._coil_array import calibration_of
from mime.interactions import Magnetic
from mime.nodes.actuation import CalibratedMultipoleNode, placed_unit_fields, point_dipole, read_calibration
from mime.nodes.actuation._multipole import Calibration
from mime.observations import EnergyLedger
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import VACUUM_PERMEABILITY, A, A_m2, T, kg, kg_m2, m, m_per_s, m_per_s2, mT, mT_per_m, ms, rad_per_s

REFERENCES = Path(__file__).resolve().parent / "references"
ORACLE = np.load(REFERENCES / "mpem_mag_manip.npz")
SYSTEMS = ("navion_2", "cardiomag", "octomag")
EPS = float(np.finfo(np.float64).eps)
#: The tolerance's factor over the predicted floor ``eps / c``: the 0.2 line's measured misses sat within 1.5 of it.
SLACK = 10.0
PROPERTY = settings(deadline=None)


def oracle(system: str, key: str) -> np.ndarray:
    return np.asarray(ORACLE[f"{system}_{key}"])


def calibration(system: str) -> Calibration:
    return calibration_of(asset(system))


def tolerance(system: str) -> float:
    """``SLACK eps / c``, ``c`` the worst cancellation at the oracle's own points."""
    return SLACK * EPS / calibration(system).cancellation(oracle(system, "points"))


def miss(found: np.ndarray, expected: np.ndarray) -> float:
    """The largest difference as a share of the reference's largest entry: an element's own share is undefined where
    the reference holds near-zero entries, as these matrices do."""
    return float(np.max(np.abs(found - expected)) / np.max(np.abs(expected)))


def leaves(held: Calibration) -> tuple[Any, Any]:
    return jnp.asarray(held.positions), jnp.asarray(held.coefficients[..., None] * held.directions)


at_point = jax.jit(placed_unit_fields)
#: The node's own function at many points at once, the array unplaced.
at_points = jax.jit(jax.vmap(placed_unit_fields, in_axes=(0, None, None, None, None)))
IDENTITY = (jnp.zeros(3), jnp.asarray((1.0, 0.0, 0.0, 0.0)))


def node_matrices(held: Calibration, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions, vectors = leaves(held)
    fields, gradients = at_points(jnp.asarray(points), *IDENTITY, positions, vectors)
    return np.asarray(fields), np.asarray(gradients)


def five(gradients: np.ndarray) -> np.ndarray:
    """``[Gxx, Gxy, Gxz, Gyy, Gyz]`` along a stack of gradients' first two axes after the stack's."""
    return np.stack([gradients[:, 0, 0], gradients[:, 0, 1], gradients[:, 0, 2], gradients[:, 1, 1],
                     gradients[:, 1, 2]], axis=1)


def oracle_misses(held: Calibration, system: str) -> tuple[float, float]:
    """The field matrix's and the gradient rows' misses against mag_manip, from the node's own function."""
    fields, gradients = node_matrices(held, oracle(system, "points"))
    return (miss(fields, oracle(system, "field_matrix")), miss(five(gradients), oracle(system, "actuation")[:, 3:]))


@pytest.mark.parametrize("system", SYSTEMS)
def test_the_tolerance_is_derived_per_system_and_is_a_test(system):
    assert tolerance(system) < 1e-9


@pytest.mark.parametrize("system", SYSTEMS)
def test_the_node_matches_mag_manip_per_unit_current_and_at_the_drawn_currents(system):
    held, tol = calibration(system), tolerance(system)
    field_miss, gradient_miss = oracle_misses(held, system)
    assert field_miss < tol and gradient_miss < tol, (system, field_miss, gradient_miss, tol)
    fields, gradients = node_matrices(held, oracle(system, "points"))
    currents = oracle(system, "currents")
    at_currents = np.concatenate([np.einsum("pik,pk->pi", fields, currents),
                                  five(np.einsum("pijk,pk->pij", gradients, currents))], axis=1)
    assert miss(at_currents[:, :3], oracle(system, "field_gradient5")[:, :3]) < tol
    assert miss(at_currents[:, 3:], oracle(system, "field_gradient5")[:, 3:]) < tol


@pytest.mark.parametrize("system", SYSTEMS)
def test_the_host_actuation_matches_mag_manip_and_the_node(system):
    provider, tol = CalibratedMultipole(calibration=asset(system)), tolerance(system)
    matrices = np.stack([provider.actuation(m(tuple(p))).matrix() for p in oracle(system, "points")])
    assert miss(matrices, oracle(system, "actuation")) < tol
    fields, gradients = node_matrices(calibration(system), oracle(system, "points"))
    assert miss(matrices[:, :3], fields) < tol and miss(matrices[:, 3:], five(gradients)) < tol


def test_control_directions_read_unnormalised_miss_the_oracle():
    held = calibration("navion_2")
    lengths = np.array([[np.linalg.norm(src["Source_Direction"]) for src in sources] for sources in raw_sources()])
    unnormalised = Calibration(held.system, held.positions, held.directions * lengths[..., None], held.coefficients,
                               held.sources, held.region)
    assert max(oracle_misses(unnormalised, "navion_2")) > 10.0 * tolerance("navion_2")


def raw_sources() -> list[list[dict[str, Any]]]:
    import yaml

    raw = yaml.safe_load((Path(__file__).resolve().parents[2] / "src/mime/data/emns_calibrations/navion_2.yaml")
                         .read_text())
    return [[raw[coil][name] for name in raw[coil]["Source_List"]] for coil in raw["Coil_List"]]


# -- planted errors -------------------------------------------------------------------------------------------------

def turned(held: Calibration, rotation: np.ndarray) -> Calibration:
    return Calibration(held.system, held.positions @ rotation.T, held.directions @ rotation.T, held.coefficients,
                       held.sources, held.region)


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def planted(held: Calibration, kind: str, draw: Any) -> Calibration:
    """``held`` with one error of ``kind``: as a calibration mishandled on its way in would be."""
    if kind == "global_sign":
        return Calibration(held.system, held.positions, held.directions, -held.coefficients, held.sources, held.region)
    if kind == "coil_sign":
        coefficients = held.coefficients.copy()
        coefficients[draw["coil"] % held.coils] *= -1.0
        return Calibration(held.system, held.positions, held.directions, coefficients, held.sources, held.region)
    if kind == "turned":
        return turned(held, axis_angle(np.asarray(draw["axis"]), np.radians(draw["degrees"])))
    if kind == "swapped":
        order = list(draw["axes"]) + [a for a in range(3) if a not in draw["axes"]]
        return turned(held, np.eye(3)[[order[1], order[0], order[2]]])
    shift = np.asarray(draw["axis"]) / np.linalg.norm(draw["axis"]) * draw["millimetres"] * 1e-3
    return Calibration(held.system, held.positions + shift, held.directions, held.coefficients, held.sources,
                       held.region)


AXES = st.tuples(*(st.floats(-1.0, 1.0) for _ in range(3))).filter(lambda v: float(np.linalg.norm(v)) > 0.1)
PLANTS = st.fixed_dictionaries({
    "system": st.sampled_from(SYSTEMS),
    "kind": st.sampled_from(("global_sign", "coil_sign", "turned", "swapped", "shifted")),
    "coil": st.integers(0, 7), "axis": AXES, "degrees": st.floats(1.0, 180.0), "millimetres": st.floats(1.0, 100.0),
    "axes": st.sampled_from(((0, 1), (0, 2), (1, 2)))})


@PROPERTY
@given(PLANTS)
def test_control_a_planted_frame_or_sign_error_in_a_calibration_fails_the_oracle_on_every_draw(draw):
    system = draw["system"]
    assert max(oracle_misses(planted(calibration(system), draw["kind"], draw), system)) > 10.0 * tolerance(system)


# -- the exact transform --------------------------------------------------------------------------------------------

def unit_quaternion(values: tuple[float, ...]) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64)
    return q / np.linalg.norm(q)


def matrix_of(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


POSES = st.tuples(st.sampled_from(SYSTEMS), st.tuples(*(st.floats(-0.5, 0.5) for _ in range(3))),
                  st.tuples(*(st.floats(-1.0, 1.0) for _ in range(4))).filter(lambda q: float(np.linalg.norm(q)) > 0.3),
                  st.integers(0, 63))


def placed_misses(system: str, at: np.ndarray, q: np.ndarray, index: int, function: Any) -> float:
    turn = matrix_of(q)
    point = oracle(system, "points")[index]
    positions, vectors = leaves(calibration(system))
    fields, gradients = (np.asarray(v) for v in function(jnp.asarray(turn @ point + at), jnp.asarray(at),
                                                         jnp.asarray(q), positions, vectors))
    expected_field = turn @ oracle(system, "field_matrix")[index]
    five_rows = oracle(system, "actuation")[index, 3:]
    expected_gradient = np.stack([turn @ unpacked(five_rows[:, k]) @ turn.T for k in range(five_rows.shape[1])], -1)
    return max(miss(fields, expected_field), miss(gradients, expected_gradient))


def unpacked(g5: np.ndarray) -> np.ndarray:
    xx, xy, xz, yy, yz = g5
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, -xx - yy]])


def turn_angle(q: np.ndarray) -> float:
    return 2.0 * float(np.degrees(np.arccos(min(1.0, abs(float(q[0]))))))


@PROPERTY
@given(POSES)
def test_the_array_placed_by_a_pose_gives_the_turned_field_and_gradient(drawn):
    system, at, q, index = drawn
    q = unit_quaternion(q)
    assert placed_misses(system, np.asarray(at), q, index, at_point) < 4.0 * tolerance(system)


@PROPERTY
@given(POSES)
def test_control_the_turn_applied_the_other_way_fails_on_every_draw(drawn):
    system, at, q, index = drawn
    q = unit_quaternion(q)
    # a turn of 0 or 180 degrees is its own inverse, so the other way is the same turn there: no evidence either way
    if not 1.0 <= turn_angle(q) <= 179.0:
        q = unit_quaternion((0.9, 0.3, -0.2, 0.25))
    assert placed_misses(system, np.asarray(at), q, index, wrong_way) > 10.0 * tolerance(system)


def _wrong_way(point, at, orientation, positions, vectors):
    from mime.core import quaternion
    from mime.nodes.actuation._multipole import unit_fields

    turn = quaternion.rotation_matrix(orientation).T
    fields, gradients = unit_fields(turn.T @ (point - at), positions, vectors)
    return turn @ fields, jnp.einsum("ia,abk,jb->ijk", turn, gradients, turn)


wrong_way = jax.jit(_wrong_way)


# -- Maxwell, the point dipole, zero current ------------------------------------------------------------------------

@pytest.mark.parametrize("system", SYSTEMS)
def test_the_gradient_is_symmetric_and_trace_free_by_construction(system):
    _, gradients = node_matrices(calibration(system), oracle(system, "points"))
    total = np.einsum("pijk,pk->pij", gradients, oracle(system, "currents"))
    scale = np.max(np.abs(total), axis=(1, 2))
    # each entry is a sum of sources' terms that cancel to c of their size, so it rounds at eps / c of itself
    floor = tolerance(system) * scale
    assert np.all(np.max(np.abs(total - np.swapaxes(total, 1, 2)), axis=(1, 2)) <= floor)
    assert np.all(np.abs(np.trace(total, axis1=1, axis2=2)) <= floor)


def test_an_order_one_source_is_mimes_point_dipole():
    rng = np.random.default_rng(20261109)
    for _ in range(64):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        coefficient, position = rng.uniform(-1e-3, 1e-3), rng.uniform(-0.3, 0.3, 3)
        point = position + rng.normal(size=3) * 0.1
        held = Calibration("one", position[None, None, :], direction[None, None, :], np.array([[coefficient]]), (1,),
                           np.array([[-1.0, 1.0]] * 3))
        fields, gradients = node_matrices(held, point[None])
        moment = jnp.asarray(4.0 * np.pi * coefficient * direction / VACUUM_PERMEABILITY)
        field, gradient = (np.asarray(v) for v in point_dipole(moment, jnp.asarray(position), jnp.asarray(point)))
        assert np.max(np.abs(fields[0, :, 0] - field)) <= 64 * EPS * np.max(np.abs(field))
        assert np.max(np.abs(gradients[0, :, :, 0] - gradient)) <= 64 * EPS * np.max(np.abs(gradient))


def test_the_reader_refuses_what_it_does_not_evaluate():
    base = (REFERENCES.parents[1].parent / "src/mime/data/emns_calibrations/navion_2.yaml").read_text()
    from mime.nodes.actuation import CalibrationError

    for edited, said in ((base.replace("A_Coeff: ~", "A_Coeff: [1.0]", 1), "interior"),
                         (base.replace("B_Coeff:\n", "B_Coeff:\n      - 1.0e-4\n", 1), "order 2"),
                         ("Coil_List: []\n", "Workspace_Dimensions")):
        with pytest.raises(CalibrationError, match=said):
            read_calibration(edited.encode())


# -- dev1's magnetic pair on a sphere -------------------------------------------------------------------------------

def sphere_in_the_array(system: str, index: int = 0, *, ledger: bool = False, torque: bool = True,
                        **settings: Any) -> Experiment:
    """A 1 mm sphere with a permanent moment at the oracle's point ``index``, in the array's field (its ``settings``) at
    the drawn currents there, with dev1's force (0.2.0) attached and its torque unless ``torque`` is false; no gravity,
    no medium."""
    exp = Experiment(f"sphere_{system}", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    exp.add_apparatus(Apparatus("coils", fields={"magnetic": CalibratedMultipole(calibration=asset(system),
                                                                                **settings)},
                                commands={"currents": A(tuple(oracle(system, "currents")[index]))}))
    exp.add_body(Body("sphere", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=m(1e-3)))},
                      initial=State(position=m(tuple(oracle(system, "points")[index])),
                                    orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("sphere.mass", kg(4e-6)), ("sphere.inertia", kg_m2((1.6e-12,) * 3)),
                        ("sphere.magnetic_moment", A_m2(tuple(oracle(system, "moments")[index]))),
                        ("sphere.magnet_centre", m((0.0, 0.0, 0.0)))):
        exp.declare(path, value, reason="a millimetre sphere with a magnet at its centre, the oracle's moment")
    if torque:
        exp.attach(Magnetic.Torque(), body="sphere", field="coils.magnetic")
    exp.attach(Magnetic.Force(), body="sphere", field="coils.magnetic")
    if ledger:
        exp.observe(EnergyLedger())
    return exp


@pytest.mark.parametrize("system", SYSTEMS)
def test_dev1s_magnetic_pair_on_a_sphere_delivers_mag_manips_force_and_torque(system):
    graph = sphere_in_the_array(system, 0).build().graph
    graph.reset_state()
    _, history = _held(graph).run_scan_with_history(1, fed_inputs(graph, None))
    force = np.asarray(history["magnetic.force:sphere"]["wrench"])[0]
    torque = np.asarray(history["magnetic.torque:sphere"]["wrench"])[0]
    tol = tolerance(system)
    assert miss(force[:3], oracle(system, "force")[0]) < tol and np.all(force[3:] == 0.0)
    assert miss(torque[3:], oracle(system, "torque")[0]) < tol and np.all(torque[:3] == 0.0)


def test_the_host_actuation_under_a_pose_and_a_background_is_the_nodes():
    """Review minor 8: the host helper at a placed, turned array with a background agrees with the node's step."""
    at, turn, background = np.array([0.02, -0.03, 0.01]), (0.8, 0.1, -0.5, 0.3), np.array([3e-5, -4e-5, 1.2e-4])
    unit = tuple(c / np.linalg.norm(turn) for c in turn)
    provider = CalibratedMultipole(calibration=asset("cardiomag"), at=m(tuple(at)), orientation=unit,
                                   background=T(tuple(background)))
    node = built_node("cardiomag", at=m(tuple(at)), orientation=unit, background=T(tuple(background)))
    currents = oracle("cardiomag", "currents")[0]
    point = at + np.array([0.01, 0.02, -0.01])
    found = provider.actuation(m(tuple(point))).at(A(tuple(currents)))
    out = step(node, point, currents)
    assert miss(np.asarray(found[0].value), out["flux_density"]) < 1e-12
    assert miss(np.asarray(found[1].value), out["gradient"].reshape(-1)) < 1e-12


def test_allocation_refuses_a_wanted_field_that_is_not_finite():
    actuation = CalibratedMultipole(calibration=asset("cardiomag")).actuation(m((0.0, 0.0, 0.0)))
    with pytest.raises(ArgumentRefused, match="not finite"):
        allocate(actuation, field=mT((float("nan"), 0.0, 0.0)), limits=A(20.0))


def test_the_sphere_is_pulled_along_the_force():
    """The force alone: with the torque the sphere (1.6e-12 kg m^2) turns about a radian in 2 ms, and its moment with
    it, so the force's direction would not hold over the run."""
    built = sphere_in_the_array("navion_2", 0, torque=False).build()
    history = built.run(ms(5), history=True).history
    moved = np.asarray(history["sphere"]["pose"])[-1, :3] - oracle("navion_2", "points")[0]
    force = oracle("navion_2", "force")[0]
    assert float(np.dot(moved, force)) / (np.linalg.norm(moved) * np.linalg.norm(force)) > 0.99


def test_with_the_ledger_the_coil_apparatus_is_outside_and_supplies_the_pair():
    """P1b's class for an external field: an apparatus with no chain is one unit, its node receives nothing from the
    components, so it is outside, and the potential pair reading it is supplied by it; no ledger change is needed."""
    report = sphere_in_the_array("navion_2", 0, ledger=True).build().report.text()
    outside = next(line for line in report.splitlines() if "outside:" in line)
    assert "coils" in outside
    supplied = [line for line in report.splitlines() if "supplied by coils" in line]
    assert any("magnetic.force:sphere" in line for line in supplied), report
    assert any("magnetic.torque:sphere" in line for line in supplied), report


# -- the background and the flags -----------------------------------------------------------------------------------

def built_node(system: str, **settings: Any) -> CalibratedMultipoleNode:
    node = sphere_in_the_array(system, **settings).build().graph.get_node("coils.magnetic")
    assert isinstance(node, CalibratedMultipoleNode)
    return node


def step(node: CalibratedMultipoleNode, point: np.ndarray, currents: np.ndarray) -> dict[str, np.ndarray]:
    """The node's step with every reader slot at ``point``, unturned, and the first slot's fields by their stems."""
    slots = node.constants()["slots"]
    pose = jnp.asarray((*point, 1.0, 0.0, 0.0, 0.0))
    out = node.update(node.initial_state(), {"delivered_currents": jnp.asarray(currents),
                                             **{f"points_{slot}": pose for slot in slots}}, 1e-3)
    return {key.removesuffix(f"_{slots[0]}"): np.asarray(value) for key, value in out.items()}


@pytest.mark.parametrize("system", SYSTEMS)
def test_a_background_moves_the_field_by_itself_and_the_gradient_by_nothing(system):
    background = np.array([3e-5, -4e-5, 1.2e-4])
    plain, shifted = built_node(system), built_node(system, background=T(tuple(background)))
    for point, currents in zip(oracle(system, "points")[:8], oracle(system, "currents")[:8], strict=True):
        a, b = step(plain, point, currents), step(shifted, point, currents)
        bound = EPS * (np.abs(a["flux_density"]) + np.abs(background))
        assert np.all(np.abs(b["flux_density"] - a["flux_density"] - background) <= bound)
        assert np.array_equal(b["gradient"], a["gradient"])


@pytest.mark.parametrize("system", SYSTEMS)
def test_outside_region_is_mag_manips_point_in_workspace_negated(system):
    node = built_node(system)
    for probe, inside in zip(oracle(system, "probes"), oracle(system, "in_workspace"), strict=True):
        assert step(node, probe, np.zeros(node.coils))["outside_region"] == (0.0 if inside else 1.0), probe


def test_near_source_is_the_gradients_departure_by_default_and_a_distance_when_declared():
    """Navion_2's region is smooth (its gradient reaches 13.5 times its median, the guard is 30): nowhere flagged in it,
    flagged at a fitted source; a declared ``source_clearance`` replaces the rule with the distance."""
    held = calibration("navion_2")
    node = built_node("navion_2")
    assert node.constants()["guard"] == "gradient"
    assert np.asarray(node.constants()["gradient_limit"]) == pytest.approx(held.gradient_limit())
    for point in oracle("navion_2", "points")[:16]:
        assert step(node, point, np.zeros(3))["near_source"] == 0.0
    source = held.positions[0, 0]
    out = step(node, source + np.array([0.0, 0.0, 1e-3]), np.zeros(3))
    assert out["near_source"] == 1.0 and out["source_distance"] == pytest.approx(1e-3)
    by_distance = built_node("navion_2", source_clearance=m(held.nearest_approach() * 1.5))
    assert by_distance.constants()["guard"] == "distance"
    assert step(by_distance, source + np.array([0.0, 0.0, held.nearest_approach() * 1.2]), np.zeros(3))["near_source"] == 1.0
    assert step(by_distance, oracle("navion_2", "points")[0], np.zeros(3))["near_source"] == 0.0


@pytest.mark.parametrize("system", ("navion_2", "cardiomag"))
def test_the_guard_raises_nowhere_in_a_smooth_region(system):
    """The measured basis of the factor: over the region's drawn points and its faces and corners the gradient stays
    under the guard in the two calibrations whose sources lie far from the region."""
    held = calibration(system)
    drawn = np.concatenate([oracle(system, "points"), oracle(system, "probes")[32:], held.grid(5)])
    node = built_node(system)
    assert all(step(node, point, np.zeros(node.coils))["near_source"] == 0.0 for point in drawn)


def octomags_source() -> np.ndarray:
    """OctoMag's fitted source nearest its declared region: 20 um outside its -x face, 0.8 and 0.6 mm from the face's
    edges (not at a corner, as the card once said)."""
    held = calibration("octomag")
    outside = np.maximum(0.0, np.maximum(held.region[:, 0] - held.positions, held.positions - held.region[:, 1]))
    distance = np.where(held.real, np.linalg.norm(outside, axis=-1), np.inf)
    k, j = np.unravel_index(np.argmin(distance), distance.shape)
    assert (k, j) == (4, 0) and distance[k, j] == pytest.approx(held.nearest_approach())
    return held.positions[k, j]


def test_octomags_guard_fires_at_its_source_and_its_nearest_corner_and_not_in_the_centre():
    """M1 of the review: the default must be able to fire inside the region it was designed for. A point 20 um from
    OctoMag's fitted source reads a gradient of hundreds of kT/m with the earlier distance default silent; here it is
    flagged, as is the region's nearest grid corner (1 mm from the source); the workspace centre is not. The control, on
    every run: with the guard removed (``source_clearance`` zero) the same point reads the same gradient, unflagged."""
    held = calibration("octomag")
    source = octomags_source()
    on_the_face = np.array([held.region[0, 0], source[1], source[2]])
    corner = np.array([-0.01, 0.01, -0.01])
    centre = np.zeros(3)
    currents = np.zeros(8)
    currents[4] = 16.0
    node, removed = built_node("octomag"), built_node("octomag", source_clearance=m(0.0))
    assert removed.constants()["guard"] == "distance"
    flagged = {label: step(node, point, currents) for label, point in
               (("source", on_the_face), ("corner", corner), ("centre", centre))}
    silent = step(removed, on_the_face, currents)
    assert np.linalg.norm(on_the_face - source) == pytest.approx(held.nearest_approach())
    assert (flagged["source"]["near_source"], flagged["corner"]["near_source"], flagged["centre"]["near_source"]) == \
        (1.0, 1.0, 0.0)
    assert silent["near_source"] == 0.0
    assert np.array_equal(silent["gradient"], flagged["source"]["gradient"])
    assert np.max(np.abs(silent["gradient"])) > 1e5, "the control reads the singular gradient silently, in T/m"
    assert np.max(np.abs(flagged["centre"]["gradient"])) < 10.0


@PROPERTY
@given(st.tuples(st.floats(-1, 1), st.floats(-1, 1), st.floats(-1, 1)).filter(lambda v: sum(c * c for c in v) > 1e-2),
       st.floats(2e-5, 3e-4))
def test_octomags_guard_fires_on_every_draw_within_300_um_of_its_source(direction, distance):
    node = built_node("octomag")
    unit = np.asarray(direction) / np.linalg.norm(direction)
    out = step(node, octomags_source() + distance * unit, np.zeros(8))
    assert out["near_source"] == 1.0, (direction, distance)


@pytest.mark.parametrize("system", SYSTEMS)
def test_zero_current_gives_the_oracles_zero_even_where_a_file_carries_an_offset_block(system):
    out = step(built_node(system), oracle(system, "points")[0], np.zeros(calibration(system).coils))
    assert not np.any(out["flux_density"]) and not np.any(out["gradient"])
    assert not np.any(oracle(system, "zero_current_field_gradient5")), "mag_manip gives zero at zero current"


# -- the Offset block: carried by CardioMag's file, evaluated by neither mag_manip nor MIME ------------------------

def offset_field(raw: dict[str, Any], point: np.ndarray) -> np.ndarray:
    """The field of a file's ``Offset`` block at ``point`` (T), the order-one sum, written out from the file."""
    field = np.zeros(3)
    for name in raw["Offset"]["Source_List"]:
        source = raw["Offset"][name]
        direction = np.asarray(source["Source_Direction"], dtype=float)
        direction = direction / np.linalg.norm(direction)
        offset = point - np.asarray(source["Source_Position"], dtype=float)
        distance = np.linalg.norm(offset)
        field += float(source["B_Coeff"][0]) * (3.0 * (direction @ offset) * offset / distance**5 - direction / distance**3)
    return field


def emns_file(system: str) -> bytes:
    return (REFERENCES.parents[1].parent / f"src/mime/data/emns_calibrations/{system}.yaml").read_bytes()


def test_only_cardiomags_file_carries_an_offset_block_and_the_reader_counts_it():
    assert {system: calibration(system).offset_sources for system in SYSTEMS} == {"navion_2": 0, "octomag": 0,
                                                                                   "cardiomag": 8}
    import yaml

    assert {system: "Offset" in yaml.safe_load(emns_file(system)) for system in SYSTEMS} == \
        {"navion_2": False, "octomag": False, "cardiomag": True}


def test_the_offset_block_is_about_1_5_mt_in_cardiomags_region_the_figure_the_card_and_validity_state():
    import yaml

    raw = yaml.safe_load(emns_file("cardiomag"))
    region = calibration("cardiomag").region
    sizes = [np.linalg.norm(offset_field(raw, point)) for point in region[:, 0] + np.random.default_rng(3).random((200, 3))
             * (region[:, 1] - region[:, 0])]
    assert 1.3e-3 < np.median(sizes) < 1.7e-3 and max(sizes) < 1.8e-3
    assert "about 1.5 mT" in " ".join(CalibratedMultipole.validity)


def test_mag_manip_does_not_evaluate_the_offset_block_of_the_shipped_file():
    """The archive's control: mag_manip on the shipped CardioMag file and on a copy whose Offset coefficients are 1000
    times larger give identical fields and gradients at zero and at drawn currents (upstream loads the block only if
    ``Coil_List`` names it). The block this test would otherwise be blind to is not small: scaled, it is 1.5 T."""
    import yaml

    assert float(oracle("cardiomag", "offset_scaled_difference")) == 0.0
    raw = yaml.safe_load(emns_file("cardiomag"))
    point = calibration("cardiomag").region[:, 0] / 2
    scaled = {"Offset": {**raw["Offset"], **{name: {**raw["Offset"][name], "B_Coeff": [1000.0 * raw["Offset"][name]["B_Coeff"][0]]}
                                             for name in raw["Offset"]["Source_List"]}}}
    assert np.linalg.norm(offset_field(scaled, point)) > 0.1


def test_the_reader_refuses_a_coil_list_that_names_the_offset_block_and_the_build_report_states_the_block():
    import yaml

    from mime.nodes.actuation import CalibrationError

    raw = yaml.safe_load(emns_file("cardiomag"))
    assert read_calibration(yaml.safe_dump(raw).encode()).offset_sources == 8, "the control: the file as it is reads"
    raw["Coil_List"] = [*raw["Coil_List"], "Offset"]
    with pytest.raises(CalibrationError, match="names an Offset block"):
        read_calibration(yaml.safe_dump(raw).encode())
    text = sphere_in_the_array("cardiomag", 0).build().report.text()
    assert "Offset block of eight sources" in text and "neither mag_manip nor this node evaluates" in text


# -- allocation -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("system", ("cardiomag", "octomag"))
def test_allocation_recovers_currents_that_make_a_field_and_gradient(system):
    provider = CalibratedMultipole(calibration=asset(system))
    for point, currents in zip(oracle(system, "points")[:8], oracle(system, "currents")[:8], strict=True):
        actuation = provider.actuation(m(tuple(point)))
        field, gradient = actuation.at(A(tuple(currents)))
        found = allocate(actuation, field=field, gradient=gradient, limits=A(10.0))
        assert found.rank == 8 and not any(found.at_limit)
        assert found.field_miss is not None and found.field_miss <= 1e-9 * float(np.max(np.abs(field.value)))
        assert np.allclose(np.asarray(found.currents.value), currents, rtol=0, atol=1e-6)


def test_allocation_holds_the_limits_out_of_reach_and_reports_the_miss():
    provider = CalibratedMultipole(calibration=asset("navion_2"))
    actuation = provider.actuation(m((0.0, 0.0, 0.0)))
    found = allocate(actuation, field=T((0.5, 0.0, 0.0)), limits=A(35.0))
    currents = np.asarray(found.currents.value)
    assert np.all(np.abs(currents) <= 35.0 * (1 + 1e-12)) and any(found.at_limit)
    assert found.field_miss is not None and found.field_miss > 0.1


def test_a_three_coil_array_asked_for_field_and_gradient_reports_its_rank():
    actuation = CalibratedMultipole(calibration=asset("navion_2")).actuation(m((0.01, 0.0, 0.0)))
    found = allocate(actuation, field=mT((10.0, 0.0, 0.0)),
                     gradient=mT_per_m((100.0, 0.0, 0.0, 0.0, -50.0, 0.0, 0.0, 0.0, -50.0)), limits=A(35.0))
    assert (found.rank, found.rows) == (3, 8) and found.gradient_miss is not None


def test_allocation_refuses_a_gradient_no_currents_make():

    actuation = CalibratedMultipole(calibration=asset("cardiomag")).actuation(m((0.0, 0.0, 0.0)))
    with pytest.raises(ArgumentRefused, match="symmetric and trace-free") as asymmetric:
        allocate(actuation, gradient=mT_per_m((1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)), limits=A(20.0))
    with pytest.raises(ArgumentRefused, match="limit") as unlimited:
        allocate(actuation, field=mT((1.0, 0.0, 0.0)), limits=A(0.0))
    assert [r.code for r in (*asymmetric.value.refusals, *unlimited.value.refusals)] == ["allocation.wanted"] * 2


def test_the_archive_is_the_one_its_provenance_names():
    import hashlib

    provenance = json.loads((REFERENCES / "mpem_mag_manip_provenance.json").read_text())
    assert hashlib.sha256((REFERENCES / "mpem_mag_manip.npz").read_bytes()).hexdigest() == provenance["archive_sha256"]
    assert provenance["mag_manip"] == "3.0.0"
    assert {name: asset(name).sha256 for name in SYSTEMS} == provenance["calibrations_sha256"]
