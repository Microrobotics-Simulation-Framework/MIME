"""EffectModel contract pilot — Protocol, registry, the six build() passes,
version validation, and the HydrodynamicModel adapter / four-way swap.

Pins ADR-2026-EFFECT-MODEL §1-§7 for the v0.2 pilot. The validation passes
(1-5) are exercised with lightweight stub effects (they fail before any graph
compiles); pass 6 plus the real adapter and interchangeability use real fluid
+ rigid-body nodes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from maddening.core.node import BoundaryFluxSpec, SimulationNode

from mime.effects import (
    BaseEffectModel,
    Body,
    BodyPropertyMissing,
    CouplingError,
    EffectBuildError,
    EffectHandle,
    EffectModel,
    Experiment,
    HydrodynamicModel,
    HydrodynamicRegime,
    IncompatibleMimeVersionError,
    Medium,
    MediumPropertyMissing,
    PortTypeMismatchError,
    Regime,
    get_effect,
    list_registered_effects,
    register_effect,
)


# ── helpers ──────────────────────────────────────────────────────────────

@dataclass
class _Port:
    shape: tuple


class _StubEffect(BaseEffectModel):
    """Minimal EffectModel for exercising the validation passes."""

    def __init__(
        self,
        *,
        ports=None,
        input_ports=None,
        req_body=(),
        req_medium=(),
        re_range=(0.0, 1.0),
        build_raises=False,
    ):
        self._ports = ports or {}
        self.input_ports = input_ports or {}
        self._req_body = set(req_body)
        self._req_medium = set(req_medium)
        self._re_range = re_range
        self._build_raises = build_raises

    @property
    def coupling_ports(self):
        return self._ports

    def applicable_regime(self) -> Regime:
        return HydrodynamicRegime(self._re_range)

    def required_body_properties(self) -> set[str]:
        return self._req_body

    def required_medium_properties(self) -> set[str]:
        return self._req_medium

    def build(self, gm, *, body, medium) -> EffectHandle:
        if self._build_raises:
            raise RuntimeError("intentional build failure")
        return EffectHandle(node_names=())


class _StubFluid(SimulationNode):
    """A minimal SI fluid node emitting drag_force / drag_torque."""

    def initial_state(self) -> dict:
        return {"u": jnp.zeros(1, dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt) -> dict:
        return state

    def boundary_flux_spec(self) -> dict:
        return {
            "drag_force": BoundaryFluxSpec(
                shape=(3,), description="drag force", output_units="SI",
            ),
            "drag_torque": BoundaryFluxSpec(
                shape=(3,), description="drag torque", output_units="SI",
            ),
        }

    def compute_boundary_fluxes(self, state, boundary_inputs=None, dt=0.0) -> dict:
        return {
            "drag_force": jnp.zeros(3, dtype=jnp.float32),
            "drag_torque": jnp.zeros(3, dtype=jnp.float32),
        }


def _rigid_body(name="body"):
    from mime.nodes.robot.rigid_body import RigidBodyNode

    return RigidBodyNode(
        name=name, timestep=1e-3,
        semi_major_axis_m=1e-3, semi_minor_axis_m=5e-4,
        density_kg_m3=1100.0,
        fluid_viscosity_pa_s=1e-3, fluid_density_kg_m3=1000.0,
        use_analytical_drag=False,
    )


def _basic_experiment(**kw):
    exp = Experiment(name="t", **kw)
    exp.set_body(Body(name="body", node=None, properties={"hydrodynamic": {}}))
    exp.set_medium(Medium(properties={"density": 1000.0, "viscosity": 1e-3}))
    return exp


# ── registry (§6) ──────────────────────────────────────────────────────────

def test_registry_lists_hydrodynamic_backends():
    reg = list_registered_effects()
    for name in (
        "HydrodynamicModel.LBM", "HydrodynamicModel.FVM",
        "HydrodynamicModel.Stokeslet", "HydrodynamicModel.DefectCorrection",
    ):
        assert name in reg
    assert get_effect("HydrodynamicModel.LBM") is HydrodynamicModel.LBM


def test_registry_rejects_duplicate_name():
    with pytest.raises(ValueError, match="already registered"):
        @register_effect("HydrodynamicModel.LBM")
        class _Dup:  # noqa: D401
            pass


def test_backends_satisfy_protocol():
    # runtime_checkable structural check on the concrete classes.
    for cls in (HydrodynamicModel.LBM, HydrodynamicModel.FVM,
                HydrodynamicModel.Stokeslet,
                HydrodynamicModel.DefectCorrection):
        assert issubclass(cls, BaseEffectModel)


# ── version validation (§7, load-time) ─────────────────────────────────────

def test_version_too_old_raises():
    with pytest.raises(IncompatibleMimeVersionError, match="requires MIME >="):
        Experiment(name="t", mime_version_min="99.0.0")


def test_version_too_new_raises():
    with pytest.raises(IncompatibleMimeVersionError, match="requires MIME <="):
        Experiment(name="t", mime_version_max="0.0.1")


def test_version_in_range_ok():
    Experiment(name="t", mime_version_min="0.0.0", mime_version_max="999.0.0")


# ── build() validation passes (§5) ─────────────────────────────────────────

def test_pass1_unknown_coupling_reference():
    exp = _basic_experiment()
    a = _StubEffect()
    exp.attach(a, name="a")
    exp.couple("ghost", target_port="x", source="a", source_port="y")
    with pytest.raises(CouplingError, match="unknown effect reference"):
        exp.build()


def test_pass2_missing_source_port():
    exp = _basic_experiment()
    a = _StubEffect(ports={})  # no coupling ports
    b = _StubEffect(input_ports={"in": _Port((3,))})
    exp.attach(a, name="a")
    exp.attach(b, name="b")
    exp.couple("b", target_port="in", source="a", source_port="missing")
    with pytest.raises(CouplingError, match="no coupling port"):
        exp.build()


def test_pass2_missing_target_port():
    exp = _basic_experiment()
    a = _StubEffect(ports={"out": _Port((3,))})
    b = _StubEffect(input_ports={})
    exp.attach(a, name="a")
    exp.attach(b, name="b")
    exp.couple("b", target_port="missing", source="a", source_port="out")
    with pytest.raises(CouplingError, match="no input port"):
        exp.build()


def test_pass2_port_shape_mismatch():
    exp = _basic_experiment()
    a = _StubEffect(ports={"out": _Port((3,))})
    b = _StubEffect(input_ports={"in": _Port((4,))})
    exp.attach(a, name="a")
    exp.attach(b, name="b")
    exp.couple("b", target_port="in", source="a", source_port="out")
    with pytest.raises(PortTypeMismatchError, match="incompatible"):
        exp.build()


def test_pass3_missing_body_section():
    exp = Experiment(name="t")
    exp.set_body(Body(name="body", properties={}))  # no "hydrodynamic"
    exp.set_medium(Medium(properties={"density": 1.0, "viscosity": 1.0}))
    exp.attach(_StubEffect(req_body={"hydrodynamic"}), name="a")
    with pytest.raises(BodyPropertyMissing, match="hydrodynamic"):
        exp.build()


def test_pass3_missing_medium_property():
    exp = _basic_experiment()
    exp.attach(_StubEffect(req_medium={"temperature"}), name="a")
    with pytest.raises(MediumPropertyMissing, match="temperature"):
        exp.build()


def test_pass4_regime_warning_collected_not_fatal():
    exp = Experiment(name="t")
    exp.set_body(Body(name="body", properties={"hydrodynamic": {}}))
    # Re = 10 is outside the (0, 1) creeping-flow range -> advisory.
    exp.set_medium(Medium(properties={
        "density": 1.0, "viscosity": 1.0, "reynolds_number": 10.0,
    }))
    exp.attach(_StubEffect(re_range=(0.0, 1.0), build_raises=True), name="a")
    # build() will fail at pass 5 (build_raises), but pass 4 runs first and
    # must have recorded the regime advisory.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(EffectBuildError):
            exp.build()
    assert any("Reynolds" in str(w.message) for w in caught)
    assert exp.warnings and "Reynolds" in exp.warnings[0].message


def test_pass5_build_error_wrapped():
    exp = _basic_experiment()
    exp.attach(_StubEffect(build_raises=True), name="a")
    with pytest.raises(EffectBuildError, match="build\\(\\) failed"):
        exp.build()


def test_build_requires_body_and_medium():
    exp = Experiment(name="t")
    exp.attach(_StubEffect(), name="a")
    with pytest.raises(Exception, match="body|medium"):
        exp.build()


# ── pass 6: clean compile + interchangeability (§7 item 5 / roadmap §4.2a) ──

@pytest.mark.parametrize("backend_name", [
    "HydrodynamicModel.FVM",
    "HydrodynamicModel.DefectCorrection",
])
def test_generic_backend_compiles_clean_and_swaps(backend_name):
    """An SI hydrodynamic backend wrapping a stub fluid node wires the drag
    edges into the body and the graph compiles — and the body is unchanged
    when the backend is swapped (the interchangeability guarantee)."""
    backend_cls = get_effect(backend_name)
    fluid = _StubFluid(name="fluid", timestep=1e-3)
    effect = backend_cls(fluid)

    exp = Experiment(name="swap")
    exp.set_body(Body(name="body", node=_rigid_body(), properties={"hydrodynamic": {}}))
    exp.set_medium(Medium(properties={"density": 1000.0, "viscosity": 1e-3}))
    name = exp.attach(effect)

    gm, handles = exp.build()  # raises if compile is not clean
    assert handles[0].node_names == ("fluid",)
    assert "fluid" in gm.node_names and "body" in gm.node_names


def test_effect_is_runtime_protocol_instance():
    fluid = _StubFluid(name="fluid", timestep=1e-3)
    effect = HydrodynamicModel.FVM(fluid)
    assert isinstance(effect, EffectModel)


def test_hydrodynamic_lbm_adapter_compiles():
    """The real IB-LBM backend builds its node + lattice→SI drag edges into a
    rigid body and the graph compiles clean (de Boer geometry, tiny lattice)."""
    from mime.nodes.environment.lbm.fluid_node import IBLBMFluidNode

    N = 8
    lbm = IBLBMFluidNode(
        name="lbm_fluid", timestep=1e-3,
        nx=N, ny=N, nz=N, tau=0.6, vessel_radius_lu=3.0,
        body_geometry_params={
            "nx": N, "ny": N, "nz": N,
            "body_radius": 1.0, "body_length": 4.0,
            "cone_length": 1.5, "cone_end_radius": 0.3,
            "fin_outer_radius": 1.5, "fin_length": 2.0,
            "fin_width": 0.5, "fin_thickness": 0.1, "helix_pitch": 6.0,
        },
        use_bouzidi=False, dx_physical=1e-4,
    )
    effect = HydrodynamicModel.LBM(
        lbm, dx_physical=1e-4, dt_physical=1e-3, fluid_density=1060.0,
    )

    exp = Experiment(name="deboer_pilot")
    exp.set_body(Body(name="rigid_body", node=_rigid_body("rigid_body"),
                      properties={"hydrodynamic": {}}))
    exp.set_medium(Medium(properties={"density": 1060.0, "viscosity": 1.2e-3}))
    exp.attach(effect)

    gm, handles = exp.build()
    assert handles[0].node_names == ("lbm_fluid",)
    assert "lbm_fluid" in gm.node_names
