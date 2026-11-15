"""The forward models (``forward_model``, ``mime.registry.ForwardModel``) against their oracles, in float64 and
float32, with their derivatives in the commands and the parameters against central differences.

**What each bounds.** Each model is the node's own arithmetic, read from the node's leaves: on built rigs the models
agree with their nodes to the rounding of their terms in both dtypes (not bit for bit: the node runs inside a compiled
scan, whose fusions order the arithmetic otherwise), which says the model is the node, not that the node is right.
Against independent oracles:

- **The calibrated multipole** against ``mag_manip``'s reference (float64, per system): the field and gradient per unit
  current (the model's affine coefficients), and the field, gradient, force and torque at drawn currents and moments,
  each within ``10 eps / c`` of its peak (``c`` the worst cancellation at the reference's points), the bound of
  ``tests/oracles/test_calibrated_multipole.py``; and against the host's ``actuation`` (float64) at a drawn placement in
  both dtypes, within ``10 eps / c`` of the dtype's ``eps``. A software oracle's agreement: it grounds neither sign nor
  frame against hardware (MIME-ANO-153).
- **The point dipole** against the closed form written here in NumPy (float64), its gradient against central
  differences of that field, its force against central differences of ``m . B``, its torque ``m x B``, fixed in the world
  and on the AR4's rotor (its mount pose an input).
- **The kinematic arm** against ``mime.core.multibody.frame_pose`` of its own description, placed at its base, and
  against its node's published ``frame_poses``.
- **Affine in the currents** (the force and torque on a fixed moment too): ``f(u) = f(0) + J u`` to rounding at drawn
  ``u``; the planted control, a force with an unmodelled term quadratic in each coil's current, ``sum_k |J_k| u_k^2 / 1 A``
  (each coil's force per ampere, its magnitude, times the current squared), departs from affine by at least
  ``0.25 max |J_k|`` at every ``u`` the draws allow (each ``|u_k| >= 0.5 A``) whatever the coils' terms do to each other:
  no sum of signed terms to cancel, which a plant through the model's own sum (``u |u|``, then ``u (1 + |u| / 10 A)``)
  has (the first is exact where every ``|u_k| = 1 A``; the second vanished where three coils' columns summed to nothing,
  which CI's stress run found at ``u = (-1, -1, -1)``).
- **Derivatives**: forward mode in ``u`` and in ``theta`` against central differences, float64, the step's truncation
  and rounding bounding the tolerance.

**The tolerances' margins** (a declared round constant is not a derived bound: each is stated against the error observed
over 120 random draws of its own test's kind per case, both dtypes, 2026-11-15; a margin near one would be the sign to
derive the constant from the law's rounding). ``SLACK = 10`` over ``eps / c`` for the placement: the factor needed was
at most 2.4 (float32 Navion_2; 2.0 for OctoMag in float64), a margin of 4.2. The node arithmetic's ``16 eps / c``: at most
0.27 of it (float64), 60 times. The dipole's ``64 eps`` (field), ``128 eps`` (torque) and ``max(1e-8, 256 eps)`` (gradient,
force, which hold the central differences' truncation too): at most 0.12 of each, 8 times. The arm's ``256 eps``: float64
is exact, float32 at most 0.0074 of it (1.9 eps), 136 times; its node's ``64 eps``: 1.5 eps observed, 43 times. The
derivative's ``1e-5`` of the tangent's scale (the central difference's rounding is ``eps / h`` = 2e-10, its truncation
``h^2``): at most 1.9e-3 of it, 500 times. The affine line (``64 eps`` of the product's terms over the worst cancellation)
sits between the clean model's error (at most 1.3e-4 of it over 1200 draws with the corners, both dtypes) and the planted
quadratic term's (at least 355 times it in float32, 2e11 in float64, the same draws; by construction at least 7.8 times
in float32, the term being at least 0.25 max |J_k| against a line of 1.6e-3 of the largest term)."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from functools import cache
from typing import Any

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from believed_cases import model_of
from chain_cases import AR4, AR4_BASE, AR4_HOME, fixed_magnet, kinematic_rpm
from test_calibrated_multipole import SYSTEMS, calibration, miss, oracle, tolerance

from mime import Experiment
from mime.build import BuildRefused
from mime.core import quaternion
from mime.core.multibody import compose, frame_pose, read_urdf
from mime.data import asset
from mime.fields import CalibratedMultipole
from mime.fields._magnetic_model import gradient_rows
from mime.registry import ForwardModel
from mime.units import VACUUM_PERMEABILITY, m, ms

jnp: Any = jax.numpy
PROPERTY = settings(deadline=None, print_blob=True, max_examples=40)
#: The factor over the predicted rounding floor ``eps / c`` within which two summations of the same sources agree.
SLACK = 10.0
#: The plant a control's detection rate is measured with (``AP3_PLANT``): ``affine``, the model's force read at the
#: currents with a gain growing with them in the as-built arm of the affine test.
PLANT = os.environ.get("AP3_PLANT", "")
#: The current (A) the planted quadratic term is stated per: each coil's force per ampere times ``u^2 / 1 A``.
QUADRATIC_CURRENT = 1.0


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def coil_rig(system: str, *, at=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0)) -> Experiment:
    """The coils read by a sphere's magnetic force (a provider nothing reads is not planned), at a placement."""
    from believed_cases import believed_sphere

    return believed_sphere(truth=CalibratedMultipole(calibration=asset(system), at=m(at), orientation=orientation),
                           controller=None, port=None)


def physical(low: float, high: float) -> st.SearchStrategy[float]:
    """A signed value whose size is a body's or a coil's (``low`` to ``high``), or zero: never a size whose square
    underflows the dtype."""
    return st.one_of(st.just(0.0), st.builds(lambda size, sign: sign * size, st.floats(low, high),
                                             st.sampled_from((-1.0, 1.0))))


def as_numpy(values: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}


# -- the calibrated multipole -------------------------------------------------------------------------------------

@pytest.mark.parametrize("system", SYSTEMS)
def test_the_coil_models_coefficients_and_values_match_mag_manip(system):
    tol = tolerance(system)
    with in_dtype("float64"):
        model = model_of(coil_rig(system), "coils.magnetic")
        points, currents, moments = oracle(system, "points"), oracle(system, "currents"), oracle(system, "moments")
        fields, gradients, values, forces, torques = [], [], [], [], []
        for point, current, moment in zip(points, currents, moments, strict=True):
            coefficients, offset = model.affine({"point": point})
            assert not np.any(np.asarray(offset["flux_density"])) and not np.any(np.asarray(offset["gradient"]))
            fields.append(np.asarray(coefficients["flux_density"]["currents"]))
            gradients.append(np.asarray(gradient_rows(coefficients["gradient"]["currents"])))
            made = as_numpy(model({"currents": current}, {"point": point, "moment": moment}))
            values.append(np.concatenate([made["flux_density"], np.asarray(gradient_rows(made["gradient"]))]))
            forces.append(made["force"])
            torques.append(made["torque"])
    found = {"field": miss(np.stack(fields), oracle(system, "field_matrix")),
             "gradient": miss(np.stack(gradients), oracle(system, "actuation")[:, 3:]),
             "values": miss(np.stack(values), oracle(system, "field_gradient5")),
             "force": miss(np.stack(forces), oracle(system, "force")),
             "torque": miss(np.stack(torques), oracle(system, "torque"))}
    assert all(value < tol for value in found.values()), (system, found, tol)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("system", SYSTEMS)
@PROPERTY
@given(data=st.data())
def test_the_coil_model_at_a_placement_matches_the_hosts_actuation(system, dtype, data):
    """At a drawn placement (a shift of up to 5 cm, any turn), a drawn point of the region placed with it, drawn
    currents and moment: the model in ``dtype`` against ``actuation`` and its force and torque on the moment, in
    float64, within ``10 eps / c`` of the dtype's ``eps``."""
    held = calibration(system)
    if system == "octomag" and dtype == "float32":
        with in_dtype(dtype), pytest.raises(BuildRefused) as refused:
            coil_rig(system).plan()
        assert "refused in float32" in str(refused.value)  # its declared guard: no float32 model to compare
        return
    shift = np.asarray(data.draw(st.tuples(*[st.floats(-0.05, 0.05)] * 3), label="shift"))
    turn = np.asarray(data.draw(st.tuples(*[st.floats(-1.0, 1.0)] * 4).filter(
        lambda q: np.linalg.norm(q) > 0.1), label="turn"))
    turn = turn / np.linalg.norm(turn)
    local = np.asarray([data.draw(st.floats(*held.region[i]), label=f"x{i}") for i in range(3)])
    currents = np.asarray(data.draw(st.tuples(*[physical(0.01, 20.0)] * held.coils), label="currents"))
    moment = np.asarray(data.draw(st.tuples(*[physical(1e-6, 1e-2)] * 3), label="moment"))
    with in_dtype("float64"):
        rotation = np.asarray(quaternion.rotation_matrix(jnp.asarray(turn, dtype=jnp.float64)))
        point = shift + rotation @ local
        provider = CalibratedMultipole(calibration=asset(system), at=m(tuple(shift)), orientation=tuple(turn))
        actuation = provider.actuation(m(tuple(point)))
        field = actuation.field @ currents + actuation.background
        gradient = actuation.gradient @ currents
        expected = {"flux_density": field, "gradient": gradient, "force": np.einsum("i,ij->j", moment, gradient),
                    "torque": np.cross(moment, field)}
        # each judged on the scale of the terms it is summed from: a zero current makes a zero field exactly
        flux_scale = np.max(np.abs(actuation.field) @ np.abs(currents))
        gradient_scale = np.max(np.abs(actuation.gradient) @ np.abs(currents))
        scales = {"flux_density": flux_scale, "gradient": gradient_scale,
                  "force": 3 * np.max(np.abs(moment)) * gradient_scale, "torque": 2 * np.max(np.abs(moment)) * flux_scale}
    with in_dtype(dtype):
        model = model_of(coil_rig(system, at=tuple(shift), orientation=tuple(turn)), "coils.magnetic")
        made = as_numpy(model({"currents": currents}, {"point": point, "moment": moment}))
    floor = SLACK * float(np.finfo(np.dtype(dtype)).eps) / held.cancellation()
    for key, value in expected.items():
        assert np.max(np.abs(made[key] - value)) <= floor * scales[key], (key, made[key], value, floor)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_coil_model_is_its_nodes_arithmetic(dtype):
    """The model at the sphere's point and the currents delivered gives the node's field and gradient at each step of
    a closed-loop run: one function, read from the same leaves, to the rounding of its terms (the node runs inside a
    compiled scan, whose fusions order the arithmetic otherwise: bits are per compiled program)."""
    from believed_cases import believed_sphere

    eps = float(np.finfo(np.dtype(dtype)).eps)
    with in_dtype(dtype):
        experiment = believed_sphere()
        model = model_of(experiment, "coils.magnetic")
        history = experiment.build().run(ms(12), history=True).history
        node, poses = history["coils.magnetic"], np.asarray(history["sphere"]["pose"])
        delivered = np.asarray(history["ports:coils"]["currents"])
        slot = next(name[len("flux_density_"):] for name in node if name.startswith("flux_density_"))
        floor = 16 * eps / calibration("navion_2").cancellation()
        for step in range(1, poses.shape[0]):
            point = jnp.asarray(poses[step - 1][:3])
            made = model({"currents": jnp.asarray(delivered[step])}, {"point": point})
            coefficients, _ = model.affine({"point": point})
            for quantity in ("flux_density", "gradient"):
                terms = np.abs(np.asarray(coefficients[quantity]["currents"])) @ np.abs(delivered[step])
                found = np.asarray(made[quantity]) - np.asarray(node[f"{quantity}_{slot}"][step])
                assert np.max(np.abs(found)) <= floor * np.max(terms), (step, quantity)


def quadratic_term(coefficients: np.ndarray, currents: np.ndarray) -> np.ndarray:
    """The planted defect: ``sum_k |J_k| u_k^2 / 1 A``, an unmodelled term quadratic in each coil's current whose
    terms all add (the magnitude of each coil's force per ampere): no draw makes them cancel."""
    return np.abs(coefficients) @ (currents ** 2) / QUADRATIC_CURRENT


def affine_misses(model: ForwardModel, currents: np.ndarray, moment: np.ndarray, point: np.ndarray, dtype: str,
                  planted: bool) -> list[float]:
    """Each quantity's departure from ``f(0) + J u`` over its rounding line (``64 eps`` of the product's terms, over the
    worst cancellation), at ``currents``; ``planted`` reads the force at the grown currents. A ratio above one fails."""
    eps = float(np.finfo(np.dtype(dtype)).eps)
    at = {"point": point, "moment": moment}
    coefficients, offset = model.affine(at)
    made = as_numpy(model({"currents": currents}, at))
    if planted:
        made = dict(made, force=made["force"] + quadratic_term(np.asarray(coefficients["force"]["currents"],
                                                                          dtype=np.float64), currents))
    ratios = []
    for quantity, value in made.items():
        part = np.asarray(coefficients[quantity]["currents"], dtype=np.float64) * currents
        predicted = np.asarray(offset[quantity], dtype=np.float64) + part.sum(axis=-1)
        scale = np.max(np.abs(part)) * len(currents) / calibration("navion_2").cancellation()
        ratios.append(float(np.max(np.abs(value - predicted)) / (64 * eps * scale)))
    return ratios


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@PROPERTY
@given(data=st.data())
def test_the_coil_model_is_affine_in_its_currents_force_and_torque_included(dtype, data):
    """``f(u) = f(0) + J u`` for every quantity, the force and torque on a fixed moment among them, to the rounding of
    the product's terms; the planted control (the force with an added ``sum_k |J_k| u_k^2 / 1 A``) fails on every draw:
    its departure from affine is at least ``0.25 max |J_k|`` for every ``u`` drawn, with no sum of signed terms to
    cancel."""
    currents = np.asarray(data.draw(st.tuples(*[st.builds(lambda c, sign: sign * c, st.floats(0.5, 20.0),
                                                          st.sampled_from((-1.0, 1.0)))] * 3)))
    moment = np.asarray(data.draw(st.tuples(*[st.builds(lambda c, sign: sign * c, st.floats(1e-5, 1e-3),
                                                        st.sampled_from((-1.0, 1.0)))] * 3)))
    held = calibration("navion_2")
    point = np.asarray([data.draw(st.floats(*held.region[i])) for i in range(3)])
    with in_dtype(dtype):
        model = planned("coils")
        for planted in (False, True):
            ratios = affine_misses(model, currents, moment, point, dtype, planted or PLANT == "affine")
            assert any(ratio > 1.0 for ratio in ratios) is planted, (planted, ratios)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("currents, moment, centre", [
    ((1.0, 1.0, 1.0), (1e-4, -2e-4, 3e-4), None), ((-1.0, 1.0, -1.0), (1e-4, -2e-4, 3e-4), None),
    ((0.5, 0.5, 0.5), (1e-4, -2e-4, 3e-4), None), ((-20.0, 20.0, 0.5), (1e-4, -2e-4, 3e-4), None),
    ((1.0, 1.0, -1.0), (1e-4, -2e-4, 3e-4), None),
    ((-1.0, -1.0, -1.0), (-6.366703020483019e-4, -1e-5, -1e-5), (0.0, 0.0, -0.09375))],
    ids=["unit", "unit_signed", "smallest", "extremes", "unit_mixed", "the_draw_a_uniform_gain_plant_missed"])
def test_the_planted_departure_from_affine_is_there_at_the_draws_corners_where_a_square_plant_was_blind(
        dtype, currents, moment, centre):
    """The control's own premise, checked at the draws' corners (the unit currents among them, where ``u |u| = u`` made
    the earlier plant, ``u |u|``, exact, and blind on about 8 draws in 20,000): at each, the as-built model is within its
    line and the planted one is over it, with a margin (at least 10x, which the added term's lower bound gives)."""
    point = np.asarray(calibration("navion_2").region.mean(axis=1) if centre is None else centre)
    moment = np.asarray(moment)
    with in_dtype(dtype):
        model = planned("coils")
        clean = affine_misses(model, np.asarray(currents), moment, point, dtype, planted=False)
        planted = affine_misses(model, np.asarray(currents), moment, point, dtype, planted=True)
        old = np.asarray(model({"currents": np.asarray(currents) * np.abs(currents)}, {"point": point, "moment": moment})["force"])
        coefficients, offset = model.affine({"point": point, "moment": moment})
        linear = np.asarray(offset["force"]) + (np.asarray(coefficients["force"]["currents"]) * currents).sum(axis=-1)
    assert max(clean) < 1.0 and max(planted) > 10.0, (clean, planted)
    if all(abs(abs(c) - 1.0) < 1e-12 for c in currents):
        assert np.max(np.abs(old - linear)) <= 64 * float(np.finfo(np.dtype(dtype)).eps) * np.max(np.abs(linear)), \
            "the square plant is blind here: that is what the quadratic term replaces"


# -- the point dipole ---------------------------------------------------------------------------------------------

def closed_form(moment: np.ndarray, centre: np.ndarray, point: np.ndarray) -> np.ndarray:
    """``mu_0 / (4 pi) (3 (m . r) r / r^5 - m / r^3)``, float64, written here."""
    r = point - centre
    distance = np.linalg.norm(r)
    return VACUUM_PERMEABILITY / (4 * np.pi) * (3 * np.dot(moment, r) * r / distance**5 - moment / distance**3)


def central(fn: Any, x: np.ndarray, h: float) -> np.ndarray:
    """``dfn/dx`` by central differences, the derivative along its last axis."""
    columns = []
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h
        columns.append((fn(x + step) - fn(x - step)) / (2 * h))
    return np.stack(columns, axis=-1)


@cache
def planned(case: str) -> ForwardModel:
    """A rig's model, planned once: its function reads the run's dtype at each call, so one serves both dtypes."""
    rigs = {"fixed": (fixed_magnet, "stand.magnetic"), "on_the_rotor": (kinematic_rpm, "rpm.magnetic"),
            "arm": (kinematic_rpm, "rpm.arm"), "coils": (lambda: coil_rig("navion_2"), "coils.magnetic")}
    make, owner = rigs[case]
    return model_of(make(), owner)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("case", ["fixed", "on_the_rotor"])
@PROPERTY
@given(data=st.data())
def test_the_dipole_model_matches_its_closed_form_and_its_derivatives(case, dtype, data):
    """The model's field against the closed form, its gradient against central differences of that field, its force
    against central differences of ``m . B`` and its torque ``m x B``, at a drawn point 3 to 30 magnet radii from the
    magnet in a drawn direction, and for the magnet on the rotor at a drawn mount pose."""
    direction = np.asarray(data.draw(st.tuples(*[st.floats(-1.0, 1.0)] * 3).filter(lambda v: np.linalg.norm(v) > 0.1)))
    reach = data.draw(st.floats(0.06, 0.5))
    moment = np.asarray(data.draw(st.tuples(*[physical(1e-6, 1e-2)] * 3)))
    with in_dtype("float64"):
        model = planned(case)
        mounted = "mount_pose" in model.inputs
        mount = np.zeros(7)
        if mounted:
            turn = np.asarray(data.draw(st.tuples(*[st.floats(-1.0, 1.0)] * 4).filter(lambda q: np.linalg.norm(q) > 0.1)))
            mount = np.concatenate([np.asarray(data.draw(st.tuples(*[st.floats(-0.2, 0.2)] * 3))),
                                    turn / np.linalg.norm(turn)])
        placement = np.asarray(model.theta["placement"])
        part = np.asarray(compose(jnp.asarray(mount), jnp.asarray(placement))) if mounted else placement
        magnet = np.asarray(quaternion.rotate(jnp.asarray(part[3:]), jnp.asarray(model.theta["moment"])))
        point = part[:3] + reach * direction / np.linalg.norm(direction)
        field = closed_form(magnet, part[:3], point)
        gradient = central(lambda x: closed_form(magnet, part[:3], x), point, 1e-6 * reach)
        force = central(lambda x: np.dot(moment, closed_form(magnet, part[:3], x)), point, 1e-6 * reach)
    with in_dtype(dtype):
        x = {"point": point, "moment": moment, **({"mount_pose": mount} if mounted else {})}
        made = as_numpy(model({}, x))
    eps = float(np.finfo(np.dtype(dtype)).eps)
    # the central differences' truncation, (h/r)^2 of the field's scale, and the dtype's rounding of the closed form;
    # each judged on the scale of the terms it is made of (a moment along the field has no torque, exactly)
    loose = max(1e-8, 256 * eps)
    size, steep = np.max(np.abs(field)), np.max(np.abs(gradient))
    assert np.max(np.abs(made["flux_density"] - field)) <= 64 * eps * size
    assert np.max(np.abs(made["gradient"] - gradient)) <= loose * steep
    assert np.max(np.abs(made["force"] - force)) <= 3 * loose * np.max(np.abs(moment)) * steep
    assert np.max(np.abs(made["torque"] - np.cross(moment, field))) <= 128 * eps * np.max(np.abs(moment)) * size


# -- the kinematic arm --------------------------------------------------------------------------------------------

@cache
def ar4_tree() -> Any:
    return read_urdf(str(AR4)).tree


@cache
def frame_oracle() -> Any:
    """``frame_pose`` of each frame of the AR4's own description, placed at its base, compiled once."""
    tree = ar4_tree()
    base = (*AR4_BASE, 1.0, 0.0, 0.0, 0.0)
    return jax.jit(lambda q: {name: compose(jnp.asarray(base, dtype=q.dtype), frame_pose(tree, q, name))
                              for name in tree.frame_names})


@cache
def arm_model() -> Any:
    """The arm's model at its joint positions, compiled once (a dtype each, as its argument's)."""
    model = planned("arm")
    return jax.jit(lambda q: model({"joint_positions": q}, {})["frame_poses"])


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@PROPERTY
@given(data=st.data())
def test_the_arm_model_is_forward_kinematics_of_its_description(dtype, data):
    """At drawn joint positions within its limits, every frame's world pose against ``frame_pose`` of its own
    description (float64), placed at its base: positions within rounding of the arm's reach, orientations to rounding
    up to the quaternion's sign."""
    tree = ar4_tree()
    low, high = tree.articulation.lower, tree.articulation.upper
    q = np.asarray([data.draw(st.floats(max(-3.0, float(a)), min(3.0, float(b)))) for a, b in zip(low, high, strict=True)])
    with in_dtype("float64"):
        expected = {name: np.asarray(pose) for name, pose in frame_oracle()(jnp.asarray(q)).items()}
    with in_dtype(dtype):
        poses = np.asarray(arm_model()(jnp.asarray(q)), dtype=np.float64)
    eps = float(np.finfo(np.dtype(dtype)).eps)
    for row, name in enumerate(tree.frame_names):
        found, want = poses[row], expected[name]
        assert np.max(np.abs(found[:3] - want[:3])) < 256 * eps * 1.0, name
        assert min(np.max(np.abs(found[3:] - want[3:])), np.max(np.abs(found[3:] + want[3:]))) < 256 * eps, name


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_arm_model_is_its_nodes_published_poses(dtype):
    """The model at the joint positions the node holds gives the poses it publishes, to rounding of the arm's reach
    (bits are per compiled program)."""
    eps = float(np.finfo(np.dtype(dtype)).eps)
    with in_dtype(dtype):
        experiment = kinematic_rpm()
        model = model_of(experiment, "rpm.arm")
        history = experiment.build().run(ms(3), history=True).history
        node = history["rpm.arm"]
        for step in range(np.asarray(node["joint_positions"]).shape[0]):
            made = np.asarray(model({"joint_positions": jnp.asarray(node["joint_positions"][step])}, {})["frame_poses"])
            assert np.max(np.abs(made - np.asarray(node["frame_poses"][step]))) <= 64 * eps, step


# -- derivatives --------------------------------------------------------------------------------------------------

@cache
def compiled(case: str) -> tuple[Any, Any]:
    """The case's model and its forward-mode derivative, each compiled once: ``value(u, x, theta)`` and
    ``tangent(u, x, theta, du, dtheta)``."""
    model = planned("fixed" if case == "dipole" else case)

    def tangent(u: Any, x: Any, theta: Any, du: Any, dtheta: Any) -> Any:
        return jax.jvp(lambda uu, tt: model.fn(uu, x, tt), (u, theta), (du, dtheta))[1]

    return jax.jit(model.fn), jax.jit(tangent)


@pytest.mark.parametrize("case", ["coils", "dipole", "arm"])
@PROPERTY
@given(data=st.data())
def test_each_models_derivative_in_its_commands_and_parameters_is_central_differences(case, data):
    """Forward mode along a drawn direction in ``u`` and in every parameter at once, against central differences of
    the model along it, float64: within the differences' truncation and rounding."""
    with in_dtype("float64"):
        model = planned("fixed" if case == "dipole" else case)
        if case == "coils":
            region = calibration("navion_2").region
            x = {"point": jnp.asarray([data.draw(st.floats(*region[i])) for i in range(3)]),
                 "moment": jnp.asarray((1e-4, -2e-4, 3e-4))}
            u = {"currents": jnp.asarray(data.draw(st.tuples(*[physical(0.01, 20.0)] * 3)))}
        elif case == "dipole":
            x = {"point": jnp.asarray((0.03, -0.02, 0.0)), "moment": jnp.asarray((1e-4, -2e-4, 3e-4))}
            u = {}
        else:
            x = {}
            u = {"joint_positions": jnp.asarray(AR4_HOME) + jnp.asarray(data.draw(st.tuples(*[st.floats(-0.3, 0.3)] * 6)))}
        theta = {k: jnp.asarray(v) for k, v in model.theta.items()}
        du = {k: jnp.asarray(np.random.default_rng(data.draw(st.integers(0, 2**31))).normal(size=np.shape(v)))
              for k, v in u.items()}
        rng = np.random.default_rng(data.draw(st.integers(0, 2**31)))
        dtheta = {k: jnp.asarray(rng.normal(size=np.shape(v)) * (np.abs(np.asarray(v)) + 1e-3))
                  for k, v in theta.items()}
        value_of, tangent_of = compiled(case)
        tangent = tangent_of(u, x, theta, du, dtheta)
        h = 1e-6

        def along(t: float) -> dict[str, Any]:
            return value_of({k: v + t * du[k] for k, v in u.items()}, x, {k: v + t * dtheta[k] for k, v in theta.items()})

        plus, minus = along(h), along(-h)
        for quantity, value in tangent.items():
            difference = (np.asarray(plus[quantity]) - np.asarray(minus[quantity])) / (2 * h)
            scale = max(np.max(np.abs(difference)), np.max(np.abs(np.asarray(value))), 1e-300)
            assert np.max(np.abs(np.asarray(value) - difference)) <= 1e-5 * scale, (case, quantity)
