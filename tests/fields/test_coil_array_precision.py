"""The calibrated multipole in each precision, the declared guard on float32, and the background in both:

- **float32 is refused where the conditioning puts the floor above the guard**: OctoMag (``eps32 / c`` about 1.7e-3
  against the declared guard, 1e-4) is refused at the build, naming float64 as the route; Navion_2 and
  CardioMag build, and their float32 field and gradient at mag_manip's points are within ``4 eps32 / c`` of its peak
  (the guard's prediction, eps over the cancellation at those points). In float64 every system builds.
- **The background** moves the field by itself, to the rounding of the sum, and the gradient by nothing, in float32
  and float64; at the default zero the field is the coils' alone (the oracle's parity, ``tests/oracles``)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime import Apparatus, Body, Experiment, State
from mime.build import BuildRefused
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.fields import CalibratedMultipole
from mime.fields._coil_array import FLOAT32_BUDGET, calibration_of
from mime.interactions import Magnetic
from mime.nodes.actuation import CalibratedMultipoleNode
from mime.scene import Scene, Z
from mime.units import A, A_m2, T, kg, kg_m2, m, m_per_s, m_per_s2, ms, rad_per_s

ORACLE = np.load(Path(__file__).resolve().parents[1] / "oracles" / "references" / "mpem_mag_manip.npz")
EPS32 = float(np.finfo(np.float32).eps)


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def coils(system: str, **settings: Any) -> Experiment:
    """A body with a magnet in the array's field, the magnetic force attached."""
    exp = Experiment(f"coils_{system}", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    exp.add_apparatus(Apparatus("coils", fields={"magnetic": CalibratedMultipole(calibration=asset(system),
                                                                                **settings)},
                                commands={"currents": A(tuple(ORACLE[f"{system}_currents"][0]))}))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m(tuple(ORACLE[f"{system}_points"][0])), orientation=(1.0, 0.0, 0.0, 0.0),
        velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("umr.mass", kg(4e-6)), ("umr.inertia", kg_m2((1.6e-12,) * 3)),
                        ("umr.magnetic_moment", A_m2((1e-3, 0.0, 0.0))), ("umr.magnet_centre", m((0.0, 0.0, 0.0)))):
        exp.declare(path, value, reason="a millimetre body's magnet, its order")
    exp.attach(Magnetic.Force(), body="umr", field="coils.magnetic")
    return exp


def node_of(system: str, **settings: Any) -> CalibratedMultipoleNode:
    node = coils(system, **settings).build().graph.get_node("coils.magnetic")
    assert isinstance(node, CalibratedMultipoleNode)
    return node


def stepped(node: CalibratedMultipoleNode, point: np.ndarray, currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    (slot,) = node.constants()["slots"]
    out = node.update(node.initial_state(), {"delivered_currents": jnp.asarray(currents),
                                             f"points_{slot}": jnp.asarray((*point, 1.0, 0.0, 0.0, 0.0))}, 1e-3)
    return np.asarray(out[f"flux_density_{slot}"], np.float64), np.asarray(out[f"gradient_{slot}"], np.float64)


def test_octomag_is_refused_in_float32_and_builds_in_float64():
    with in_dtype("float32"):
        with pytest.raises(BuildRefused) as refused:
            coils("octomag").build()
    (refusal,) = [r for r in refused.value.refusals if "float32" in r.message]
    assert refusal.code == "setting.value" and "JAX_ENABLE_X64=1" in refusal.action
    assert calibration_of(asset("octomag")).floor(EPS32) > FLOAT32_BUDGET
    with in_dtype("float64"):
        assert node_of("octomag").coils == 8


@pytest.mark.parametrize("system", ["navion_2", "cardiomag"])
def test_a_well_conditioned_system_builds_in_float32_within_its_predicted_floor(system):
    held = calibration_of(asset(system))
    assert held.floor(EPS32) < FLOAT32_BUDGET
    points, currents = ORACLE[f"{system}_points"], ORACLE[f"{system}_currents"]
    floor = 4.0 * EPS32 / held.cancellation(points)
    with in_dtype("float32"):
        node = node_of(system)
        found = [stepped(node, p, c) for p, c in zip(points, currents, strict=True)]
    reference = ORACLE[f"{system}_field_gradient5"]
    fields = np.stack([f for f, _ in found])
    five = np.stack([[g[0, 0], g[0, 1], g[0, 2], g[1, 1], g[1, 2]] for _, g in found])
    assert np.max(np.abs(fields - reference[:, :3])) / np.max(np.abs(reference[:, :3])) < floor
    assert np.max(np.abs(five - reference[:, 3:])) / np.max(np.abs(reference[:, 3:])) < floor


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("system", ["navion_2", "cardiomag"])
def test_a_background_moves_the_field_by_itself_and_the_gradient_by_nothing(system, precision):
    background = (3e-5, -4e-5, 1.2e-4)
    with in_dtype(precision):
        plain, shifted = node_of(system), node_of(system, background=T(background))
        eps = float(jnp.finfo(jnp.asarray(0.0).dtype).eps)
        held = np.asarray(jnp.asarray(background), np.float64)
        for point, currents in zip(ORACLE[f"{system}_points"][:8], ORACLE[f"{system}_currents"][:8], strict=True):
            (field, gradient), (moved, same) = stepped(plain, point, currents), stepped(shifted, point, currents)
            assert np.all(np.abs(moved - field - held) <= 2 * eps * (np.abs(field) + np.abs(held)))
            assert np.array_equal(same, gradient)


#: What each calibration's card states of its conditioning (docs/assets/<name>.md), as it prints each (and in SI): the
#: field cancellation over the declared region, the rounding in float64 and float32, the least distance from a source
#: to the region and, for OctoMag, the furthest source's distance from the origin.
CARDS = {"navion_2": {"c": ("8.0e-3", 8.0e-3), "f64": ("2.8e-14", 2.8e-14), "f32": ("1.5e-5", 1.5e-5),
                      "near": ("0.116 m", 0.116)},
         "cardiomag": {"c": ("0.25", 0.25), "f64": ("8.9e-16", 8.9e-16), "f32": ("4.8e-7", 4.8e-7),
                       "near": ("0.163 m", 0.163)},
         "octomag": {"c": ("6.9e-5", 6.9e-5), "f64": ("3.2e-12", 3.2e-12), "f32": ("1.7e-3", 1.7e-3),
                     "near": ("20 um", 20e-6), "far": ("245 km", 245e3)}}


@pytest.mark.parametrize("system", sorted(CARDS))
def test_each_cards_conditioning_is_its_calibrations(system):
    held, stated = calibration_of(asset(system)), CARDS[system]
    card = (Path(__file__).resolve().parents[2] / "docs" / "assets" / f"{system}.md").read_text()
    found = {"c": held.cancellation(), "f64": held.floor(float(np.finfo(np.float64).eps)), "f32": held.floor(EPS32),
             "near": held.nearest_approach(), "far": float(np.max(np.linalg.norm(held.positions[held.real], axis=-1)))}
    flat = " ".join(card.split())
    for key, (printed, value) in stated.items():
        assert printed in flat, (system, key, printed)
        assert found[key] == pytest.approx(value, rel=0.06), (system, key, found[key])
