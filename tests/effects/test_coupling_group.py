"""Experiment.add_coupling_group — the E6a composition surface.

Pins the generalisation of the hand-built coupling groups the directory
experiments wire today (umr_confinement Schwarz IQN-ILS; dejongh_new_chain
Gauss-Seidel) into the Experiment surface: members may be node-name strings,
attached EffectModels (expanded to their node names), or the Body; extra
kwargs (acceleration, subcycling, ...) flow through to MADDENING.
"""

from __future__ import annotations

import pytest

from mime.effects import (
    Body,
    CouplingError,
    Experiment,
    MagneticModel,
    Medium,
)
from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode

_DT = 1e-3


def _rigid_body(name="body"):
    from mime.nodes.robot.rigid_body import RigidBodyNode

    return RigidBodyNode(
        name=name, timestep=_DT,
        semi_major_axis_m=1e-3, semi_minor_axis_m=5e-4,
        density_kg_m3=1100.0,
        fluid_viscosity_pa_s=1e-3, fluid_density_kg_m3=1000.0,
        use_analytical_drag=False,
    )


def _magnetic_experiment():
    # body ↔ response form a cycle (body→orientation→response,
    # response→magnetic_force→body), so a coupling group over them is valid.
    exp = Experiment(name="cg")
    exp.set_body(Body(name="body", node=_rigid_body(),
                      properties={"magnetic": {}}))
    exp.set_medium(Medium(properties={}))
    return exp


def _uniform_effect():
    return MagneticModel.UniformField(
        ExternalMagneticFieldNode("field", _DT),
        PermanentMagnetResponseNode("magnet", _DT, n_magnets=2),
    )


def test_coupling_group_with_node_name_strings():
    exp = _magnetic_experiment()
    exp.attach(_uniform_effect())
    exp.add_coupling_group(["body", "magnet"], max_iterations=5, tolerance=1e-5)
    gm, _ = exp.build()
    assert len(gm._coupling_groups) == 1
    grp = gm._coupling_groups[0]
    assert set(grp.nodes) == {"body", "magnet"}
    assert grp.max_iterations == 5
    assert grp.tolerance == 1e-5


def test_coupling_group_expands_effect_to_its_nodes():
    exp = _magnetic_experiment()
    eff = _uniform_effect()
    exp.attach(eff)
    # Reference the effect itself + the body; the effect expands to its nodes.
    exp.add_coupling_group([exp.body, eff])
    gm, _ = exp.build()
    grp = gm._coupling_groups[0]
    # field + magnet (response) from the effect, plus body.
    assert {"body", "field", "magnet"} <= set(grp.nodes)


def test_coupling_group_dedups_preserving_order():
    exp = _magnetic_experiment()
    eff = _uniform_effect()
    exp.attach(eff)
    # "magnet" appears both explicitly and via the effect expansion.
    exp.add_coupling_group(["body", "magnet", eff])
    gm, _ = exp.build()
    nodes = list(gm._coupling_groups[0].nodes)
    assert nodes[0] == "body" and nodes[1] == "magnet"
    assert len(nodes) == len(set(nodes))  # no duplicates


def test_coupling_group_passes_kwargs_through():
    exp = _magnetic_experiment()
    exp.attach(_uniform_effect())
    exp.add_coupling_group(
        ["body", "magnet"],
        max_iterations=3, tolerance=1e-4,
        acceleration="iqn-ils", subcycling=True,
        convergence_norm="interface",
    )
    gm, _ = exp.build()
    grp = gm._coupling_groups[0]
    assert grp.acceleration == "iqn-ils"
    assert grp.subcycling is True
    assert grp.convergence_norm == "interface"


def test_coupling_group_unknown_member_raises():
    exp = _magnetic_experiment()
    exp.attach(_uniform_effect())
    stray = _uniform_effect()  # not attached
    exp.add_coupling_group(["body", stray])
    with pytest.raises(CouplingError, match="not an attached effect"):
        exp.build()


def test_external_input_dtype_forwarded():
    import jax.numpy as jnp

    exp = _magnetic_experiment()
    exp.attach(_uniform_effect())
    # frequency drive as an external float32 input on the field node.
    exp.add_external_input("field", "frequency_hz", shape=(), dtype=jnp.float32)
    gm, _ = exp.build()  # compiles clean with the typed external input
    assert gm is not None
