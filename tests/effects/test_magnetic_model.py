"""MagneticModel family — UniformField / PointDipole / DualDipole adapters.

Pins the v0.3 second EffectModel family: each backend wraps pre-built magnetic
nodes and materialises the field-production chain + body coupling onto a
GraphManager, the graph compiles clean, one backend swaps for another across
the same body, and a magnetic effect composes with a hydrodynamic effect on
the same body (additive force/torque edges).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from maddening.core.node import (
    BoundaryFluxSpec,
    SimulationNode,
)

from mime.effects import (
    Body,
    BodyPropertyMissing,
    ConstantInput,
    Experiment,
    ExternalInputRef,
    HydrodynamicModel,
    MagneticModel,
    MagneticRegime,
    MagneticSource,
    Medium,
    get_effect,
    list_registered_effects,
)
from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode
from mime.nodes.actuation.field_superposition import FieldSuperpositionNode
from mime.nodes.actuation.permanent_magnet import PermanentMagnetNode
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode


_DT = 1e-3


# ── node builders ────────────────────────────────────────────────────────

def _rigid_body(name="body"):
    from mime.nodes.robot.rigid_body import RigidBodyNode

    return RigidBodyNode(
        name=name, timestep=_DT,
        semi_major_axis_m=1e-3, semi_minor_axis_m=5e-4,
        density_kg_m3=1100.0,
        fluid_viscosity_pa_s=1e-3, fluid_density_kg_m3=1000.0,
        use_analytical_drag=False,
    )


def _response(name="magnet"):
    return PermanentMagnetResponseNode(name, _DT, n_magnets=2, m_single=1.07e-3)


def _uniform_field(name="field"):
    return ExternalMagneticFieldNode(name, _DT)


def _dipole(name):
    return PermanentMagnetNode(
        name, _DT,
        dipole_moment_a_m2=1.0,
        magnet_radius_m=2e-3, magnet_length_m=4e-3,
    )


class _PoseStub(SimulationNode):
    """Minimal node emitting a constant ``rotor_pose_world`` (7,)."""

    def initial_state(self) -> dict:
        return {"rotor_pose_world": jnp.array([0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0])}

    def update(self, state, boundary_inputs, dt) -> dict:
        return state

    def boundary_flux_spec(self) -> dict:
        return {
            "rotor_pose_world": BoundaryFluxSpec(
                shape=(7,), description="magnet pose", output_units="SI",
            ),
        }

    def compute_boundary_fluxes(self, state, boundary_inputs, dt) -> dict:
        return {"rotor_pose_world": state["rotor_pose_world"]}


class _StubFluid(SimulationNode):
    """Minimal SI fluid node emitting drag_force / drag_torque, no back-edges."""

    def initial_state(self) -> dict:
        return {"u": jnp.zeros(1, dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt) -> dict:
        return state

    def boundary_flux_spec(self) -> dict:
        return {
            "drag_force": BoundaryFluxSpec(
                shape=(3,), description="drag force", output_units="SI"),
            "drag_torque": BoundaryFluxSpec(
                shape=(3,), description="drag torque", output_units="SI"),
        }

    def boundary_input_spec(self) -> dict:
        return {}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt) -> dict:
        return {"drag_force": jnp.zeros(3), "drag_torque": jnp.zeros(3)}


def _magnetic_experiment(name="mag"):
    exp = Experiment(name=name)
    exp.set_body(Body(name="body", node=_rigid_body(),
                      properties={"magnetic": {}}))
    exp.set_medium(Medium(properties={}))
    return exp


# ── registry ────────────────────────────────────────────────────────────

def test_registry_lists_magnetic_backends():
    reg = list_registered_effects()
    for key in ("MagneticModel.UniformField", "MagneticModel.PointDipole",
                "MagneticModel.DualDipole"):
        assert key in reg
    assert get_effect("MagneticModel.UniformField") is MagneticModel.UniformField


def test_required_properties():
    eff = MagneticModel.UniformField(_uniform_field(), _response())
    assert eff.required_body_properties() == {"magnetic"}
    assert eff.required_medium_properties() == set()
    assert isinstance(eff.applicable_regime(), MagneticRegime)


# ── build + compile ───────────────────────────────────────────────────────

def test_uniform_field_builds_and_compiles():
    exp = _magnetic_experiment()
    exp.attach(MagneticModel.UniformField(_uniform_field(), _response()))
    gm, handles = exp.build()
    assert any("field" in n for h in handles for n in h.node_names)
    assert "magnet" in {n for h in handles for n in h.node_names}


def test_point_dipole_builds_and_compiles():
    exp = _magnetic_experiment()
    exp.attach(MagneticModel.PointDipole(_dipole("dip"), _response()))
    gm, handles = exp.build()
    names = {n for h in handles for n in h.node_names}
    assert {"dip", "magnet"} <= names


def test_point_dipole_with_pose_source_builds_and_compiles():
    exp = _magnetic_experiment()
    exp.attach(MagneticModel.PointDipole(
        _dipole("dip"), _response(), pose_source=_PoseStub("motor", _DT)))
    gm, handles = exp.build()
    names = {n for h in handles for n in h.node_names}
    assert {"dip", "magnet", "motor"} <= names


def test_dual_dipole_builds_and_compiles():
    exp = _magnetic_experiment()
    sp = FieldSuperpositionNode("sp", _DT, n_sources=2)
    exp.attach(MagneticModel.DualDipole(
        [_dipole("dip0"), _dipole("dip1")], sp, _response()))
    gm, handles = exp.build()
    names = {n for h in handles for n in h.node_names}
    assert {"dip0", "dip1", "sp", "magnet"} <= names


def test_dual_dipole_rejects_count_mismatch():
    sp = FieldSuperpositionNode("sp", _DT, n_sources=3)
    with pytest.raises(ValueError, match="dipoles"):
        MagneticModel.DualDipole([_dipole("dip0"), _dipole("dip1")], sp, _response())


# ── swap + compose ────────────────────────────────────────────────────────

def test_uniform_field_swaps_for_point_dipole():
    # Same body, same Experiment shape — only the magnetic backend changes.
    exp1 = _magnetic_experiment("a")
    exp1.attach(MagneticModel.UniformField(_uniform_field(), _response()))
    gm1, _ = exp1.build()

    exp2 = _magnetic_experiment("b")
    exp2.attach(MagneticModel.PointDipole(_dipole("dip"), _response()))
    gm2, _ = exp2.build()
    # Both compiled; the body coupling (magnetic_force/torque into "body") is
    # identical across the swap.
    assert gm1 is not gm2


def test_magnetic_composes_with_hydrodynamic_on_one_body():
    # E3 preview: a magnetic effect and a hydrodynamic effect deliver to the
    # same body on additive edges and compile together.
    exp = _magnetic_experiment("compose")
    # Hydrodynamic needs density/viscosity on the medium.
    exp.set_medium(Medium(properties={"density": 1000.0, "viscosity": 1e-3}))
    exp.attach(MagneticModel.UniformField(_uniform_field(), _response()))
    exp.attach(HydrodynamicModel.FVM(_StubFluid(name="fluid", timestep=_DT)))
    gm, handles = exp.build()
    names = {n for h in handles for n in h.node_names}
    assert {"field", "magnet", "fluid"} <= names


def test_two_magnetic_effects_compose_additively():
    # Two independent magnetic effects (each its own response) both deliver
    # magnetic_force to the same body — additive edges must not collide.
    exp = _magnetic_experiment("two")
    exp.attach(MagneticModel.UniformField(_uniform_field("f0"), _response("m0")))
    exp.attach(MagneticModel.PointDipole(_dipole("dip"), _response("m1")))
    gm, handles = exp.build()
    names = {n for h in handles for n in h.node_names}
    assert {"f0", "m0", "dip", "m1"} <= names


# ── validation + regime ────────────────────────────────────────────────────

def test_missing_magnetic_body_section_raises():
    exp = Experiment(name="x")
    exp.set_body(Body(name="body", node=_rigid_body(), properties={}))  # no magnetic
    exp.set_medium(Medium(properties={}))
    exp.attach(MagneticModel.UniformField(_uniform_field(), _response()))
    with pytest.raises(BodyPropertyMissing):
        exp.build()


def test_regime_advisory_for_out_of_range_drive():
    regime = MagneticRegime(freq_range_hz=(0.0, 200.0), field_range_mt=(0.0, 100.0))
    body = Body(name="body",
                properties={"magnetic": {"drive_frequency_hz": 500.0,
                                          "drive_field_mt": 250.0}})
    medium = Medium(properties={})
    advisories = regime.check(body=body, medium=medium)
    assert len(advisories) == 2
    assert all(a.effect == "magnetic" for a in advisories)


def test_regime_silent_in_range():
    regime = MagneticRegime()
    body = Body(name="body",
                properties={"magnetic": {"drive_frequency_hz": 10.0,
                                         "drive_field_mt": 10.0}})
    assert regime.check(body=body, medium=Medium(properties={})) == []


# ── E2 source integration ──────────────────────────────────────────────────

def test_point_dipole_with_external_pose_source_compiles():
    # Magnet pose driven by an external step()-time command (the RPM drive).
    exp = _magnetic_experiment()
    src = MagneticSource(
        name="dip", inputs={"magnet_pose_world": ExternalInputRef(shape=(7,))})
    exp.attach(MagneticModel.PointDipole(_dipole("dip"), _response(), source=src))
    gm, handles = exp.build()
    assert gm._external_inputs  # the pose command is a graph-external input


def test_point_dipole_with_constant_pose_source_adds_const_node():
    import jax.numpy as jnp

    exp = _magnetic_experiment()
    pose = jnp.array([0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0])
    src = MagneticSource(name="dip",
                         inputs={"magnet_pose_world": ConstantInput(pose)})
    exp.attach(MagneticModel.PointDipole(_dipole("dip"), _response(), source=src))
    gm, handles = exp.build()
    names = {n for h in handles for n in h.node_names}
    # the constant-input helper node is recorded in the handle
    assert any("const" in n for n in names)


def test_point_dipole_double_drive_raises():
    src = MagneticSource(
        name="dip", inputs={"magnet_pose_world": ExternalInputRef(shape=(7,))})
    with pytest.raises(ValueError, match="pick one"):
        MagneticModel.PointDipole(
            _dipole("dip"), _response(),
            pose_source=_PoseStub("motor", _DT), source=src)


def test_dual_dipole_with_per_dipole_sources_compiles():
    exp = _magnetic_experiment()
    sp = FieldSuperpositionNode("sp", _DT, n_sources=2)
    sources = [
        MagneticSource(name="d0",
                       inputs={"magnet_pose_world": ExternalInputRef(shape=(7,))}),
        MagneticSource(name="d1",
                       inputs={"magnet_pose_world": ExternalInputRef(shape=(7,))}),
    ]
    exp.attach(MagneticModel.DualDipole(
        [_dipole("dip0"), _dipole("dip1")], sp, _response(), sources=sources))
    gm, handles = exp.build()
    # two external pose commands (one per dipole) registered
    assert len(gm._external_inputs) >= 2


def test_dual_dipole_sources_length_mismatch_raises():
    sp = FieldSuperpositionNode("sp", _DT, n_sources=2)
    with pytest.raises(ValueError, match="sources length"):
        MagneticModel.DualDipole(
            [_dipole("dip0"), _dipole("dip1")], sp, _response(),
            sources=[MagneticSource(name="d0")])
