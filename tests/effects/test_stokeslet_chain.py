"""HydrodynamicModel.StokesletChain (E6d) — the de Jongh confined near-field.

Pins the resistance-chain backend (MLP resistance + optional lubrication →
body drag) and — the solver-core point — that its ``background_velocity`` input
is a cross-effect coupling port, so a far-field hydrodynamic effect can
``couple()`` its ambient velocity into the near-field body model: the two-scale
Schwarz coupling materialised through the EffectModel surface.

Uses stub nodes so the wiring is tested without the gitignored MLP weights; a
guarded smoke builds the real chain when the weights artefact is present.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from maddening.core.node import (
    BoundaryFluxSpec, BoundaryInputSpec, SimulationNode,
)

from mime.effects import (
    BaseEffectModel,
    Body,
    CouplingPort,
    EffectHandle,
    Experiment,
    HydrodynamicModel,
    HydrodynamicRegime,
    Medium,
    get_effect,
    list_registered_effects,
)

_DT = 1e-3


def _rigid_body(name="body"):
    from mime.nodes.robot.rigid_body import RigidBodyNode
    return RigidBodyNode(
        name=name, timestep=_DT,
        semi_major_axis_m=1e-3, semi_minor_axis_m=5e-4, density_kg_m3=1100.0,
        fluid_viscosity_pa_s=1e-3, fluid_density_kg_m3=1000.0,
        use_analytical_drag=False,
    )


class _MLPStub(SimulationNode):
    def initial_state(self):
        return {"drag_force": jnp.zeros(3), "drag_torque": jnp.zeros(3),
                "resistance_matrix": jnp.eye(6)}

    def boundary_input_spec(self):
        return {k: BoundaryInputSpec(shape=s, default=jnp.zeros(s))
                for k, s in (("robot_position", (3,)), ("robot_orientation", (4,)),
                             ("body_velocity", (3,)), ("body_angular_velocity", (3,)))}

    def update(self, state, bi, dt):
        return state

    def boundary_flux_spec(self):
        return {
            "drag_force": BoundaryFluxSpec(shape=(3,), description="", output_units="SI"),
            "drag_torque": BoundaryFluxSpec(shape=(3,), description="", output_units="SI"),
            "resistance_matrix": BoundaryFluxSpec(shape=(6, 6), description="", output_units="SI"),
        }

    def compute_boundary_fluxes(self, state, bi, dt):
        return {"drag_force": state["drag_force"], "drag_torque": state["drag_torque"],
                "resistance_matrix": state["resistance_matrix"]}


class _LubStub(SimulationNode):
    def initial_state(self):
        return {"drag_force": jnp.zeros(3), "drag_torque": jnp.zeros(3)}

    def boundary_input_spec(self):
        return {
            "resistance_matrix": BoundaryInputSpec(shape=(6, 6), default=jnp.eye(6)),
            "robot_position": BoundaryInputSpec(shape=(3,), default=jnp.zeros(3)),
            "body_velocity": BoundaryInputSpec(shape=(3,), default=jnp.zeros(3)),
            "body_angular_velocity": BoundaryInputSpec(shape=(3,), default=jnp.zeros(3)),
            "background_velocity": BoundaryInputSpec(shape=(3,), default=jnp.zeros(3)),
        }

    def update(self, state, bi, dt):
        return state

    def boundary_flux_spec(self):
        return {"drag_force": BoundaryFluxSpec(shape=(3,), description="", output_units="SI"),
                "drag_torque": BoundaryFluxSpec(shape=(3,), description="", output_units="SI")}

    def compute_boundary_fluxes(self, state, bi, dt):
        return {"drag_force": state["drag_force"], "drag_torque": state["drag_torque"]}


def _experiment():
    exp = Experiment(name="chain")
    exp.set_body(Body("body", node=_rigid_body(), properties={"hydrodynamic": {}}))
    exp.set_medium(Medium({"density": 1000.0, "viscosity": 1e-3}))
    return exp


# ── registry / build ──────────────────────────────────────────────────────

def test_registered():
    assert "HydrodynamicModel.StokesletChain" in list_registered_effects()
    assert get_effect("HydrodynamicModel.StokesletChain") is HydrodynamicModel.StokesletChain


def test_mlp_only_builds_and_compiles():
    exp = _experiment()
    exp.attach(HydrodynamicModel.StokesletChain(_MLPStub("mlp", _DT)))
    gm, handles = exp.build()
    assert {"mlp"} == set(n for h in handles for n in h.node_names)


def test_mlp_plus_lubrication_builds_and_compiles():
    exp = _experiment()
    exp.attach(HydrodynamicModel.StokesletChain(
        _MLPStub("mlp", _DT), lubrication_node=_LubStub("lub", _DT)))
    gm, handles = exp.build()
    assert {"mlp", "lub"} == set(n for h in handles for n in h.node_names)


def test_input_port_present_only_with_lubrication():
    chain_lub = HydrodynamicModel.StokesletChain(
        _MLPStub("mlp", _DT), lubrication_node=_LubStub("lub", _DT))
    assert "background_velocity" in chain_lub.input_ports
    chain_no_lub = HydrodynamicModel.StokesletChain(_MLPStub("mlp", _DT))
    assert chain_no_lub.input_ports == {}


# ── two-scale Schwarz coupling via the EffectModel surface ──────────────────

class _FarFieldStub(BaseEffectModel):
    """A stand-in far-field effect exposing an ambient velocity coupling port."""

    def __init__(self, node_name="farfield"):
        self._name = node_name

    @property
    def coupling_ports(self):
        return {"ambient_velocity": CouplingPort(
            node=self._name, field="velocity", shape=(3,))}

    def applicable_regime(self):
        return HydrodynamicRegime((0.0, 2000.0))

    def build(self, gm, *, body, medium):
        node = _FarFieldNode(self._name, _DT)
        gm.add_node(node)
        return EffectHandle(node_names=(self._name,))


class _FarFieldNode(SimulationNode):
    def initial_state(self):
        return {"velocity": jnp.array([0.0, 0.0, -1e-3])}  # a counter-flow

    def update(self, state, bi, dt):
        return state

    def boundary_flux_spec(self):
        return {"velocity": BoundaryFluxSpec(shape=(3,), description="ambient", output_units="SI")}

    def compute_boundary_fluxes(self, state, bi, dt):
        return {"velocity": state["velocity"]}


def test_two_scale_coupling_far_velocity_into_near_field():
    # The far-field ambient velocity couples into the near-field chain's
    # background_velocity — the two-scale Schwarz coupling, materialised.
    exp = _experiment()
    near = HydrodynamicModel.StokesletChain(
        _MLPStub("mlp", _DT), lubrication_node=_LubStub("lub", _DT))
    far = _FarFieldStub("farfield")
    exp.attach(near, name="near")
    exp.attach(far, name="far")
    exp.couple(near, target_port="background_velocity",
               source=far, source_port="ambient_velocity")
    gm, _ = exp.build()
    # the coupling edge farfield.velocity -> lub.background_velocity exists
    assert any(
        e for e in gm._edges
        if e.source_node == "farfield" and e.target_node == "lub"
        and e.target_field == "background_velocity"
    )


# ── real-node smoke (guarded on the gitignored weights) ─────────────────────

def test_real_chain_builds_when_weights_present():
    from mime.experiments.dejongh import default_mlp_weights_path
    if not default_mlp_weights_path().exists():
        pytest.skip(f"MLP weights artefact absent: {default_mlp_weights_path()}")
    from mime.nodes.environment.stokeslet.mlp_resistance_node import MLPResistanceNode
    from mime.nodes.environment.stokeslet.lubrication_node import LubricationCorrectionNode

    mlp = MLPResistanceNode(
        "mlp", _DT, mlp_weights_path=str(default_mlp_weights_path()),
        nu=1.0, L_UMR_mm=7.47, R_cyl_UMR_mm=1.56, R_ves_mm=2.0, mu_Pa_s=1e-3)
    lub = LubricationCorrectionNode(
        "lub", _DT, R_ves_mm=2.0, R_max_body_mm=1.56, mu_Pa_s=1e-3,
        epsilon_mm=0.05, a_eff_mm=1.56)
    exp = _experiment()
    exp.attach(HydrodynamicModel.StokesletChain(mlp, lubrication_node=lub))
    gm, handles = exp.build()
    assert {"mlp", "lub"} <= set(n for h in handles for n in h.node_names)
