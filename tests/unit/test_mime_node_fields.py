"""MIME's nodes' typed state: each node's TypedDicts name exactly the fields it holds and the inputs it declares, as
the toy nodes' do (``test_node_fields.py``). Every node class in ``mime.nodes`` must be listed here."""

import importlib
import inspect
import pkgutil
from pathlib import Path

import jax.numpy as jnp
import pytest
from maddening.core.node import SimulationNode
from typing_extensions import NoExtraItems

import mime.nodes
from mime.core import MimeNode
from mime.core.multibody import read_urdf
from mime.data import asset
from mime.nodes import actuation, body, hydro, magnetic

DT = 1e-3
TUBE = dict(surface_shape="cylinder", extent=1e-3, margin=1e-4, velocity_rule="remove_outward",
            surface_centre=(0.0, 0.0, 0.0), surface_axis=(1.0, 0.0, 0.0), surface_radius=3e-3)

PLANAR = read_urdf(Path(__file__).parents[1] / "descriptions" / "three_link_planar.urdf").tree.articulation
IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

TYPED = [
    (body.RigidInertialNode("b", DT, mass=1e-4, inertia=(1e-10,) * 3, initial_pose=(0, 0, 0, 1, 0, 0, 0),
                            initial_twist=(0.0,) * 6, **TUBE), body.RigidInertialState, body.RigidInertialInputs,
     set()),
    (body.GravityNode("g", DT, mass=1e-4, buoyant_volume=6e-8, fluid_density=1e3, gravity=(0, 0, -9.81),
                      centre_of_buoyancy=(0, 0, 1e-4)), body.Wrench, body.GravityInputs, set()),
    (body.SwimSpeedNode("s", DT, along=(1, 0, 0)), body.SwimSpeedState, body.SwimSpeedInputs, set()),
    (magnetic.MagneticTorqueNode("t", DT, moment=(0, 1e-3, 0)), body.Wrench, magnetic.MagneticTorqueInputs, set()),
    (actuation.UniformRotatingFieldNode("f", DT, slots=("r",), rotation_axis=(0, 0, 1), initial_direction=(1, 0, 0)),
     actuation.UniformRotatingFieldState, actuation.UniformRotatingFieldInputs, {"flux_density_r", "gradient_r"}),
    (actuation.ConstantNode("c", DT, value=10.0), actuation.Command, None, set()),
    (hydro.SurrogateDragNode("d", DT, weights="dejongh_mlp_v2", weights_sha256=asset("dejongh_mlp_v2").sha256,
                             viscosity=1e-3, helix_wavenumber=2.33, length=7.47e-3, radius=1.56e-3,
                             vessel_centre=(0, 0, 0), vessel_axis=(1, 0, 0), vessel_radius=3.175e-3,
                             lubrication_epsilon=1.5e-4), hydro.SurrogateDragState, hydro.SurrogateDragInputs, set()),
    (actuation.KinematicArmNode("a", DT, **actuation.KinematicArmNode.described(PLANAR), placement=IDENTITY,
                                initial_joint_positions=(0.0, 0.0, 0.0)), actuation.ArmState, actuation.ArmInputs,
     set()),
    (actuation.KinematicMotorNode("m", DT, axis=(0, 0, 1), prescribed="velocity", placement=IDENTITY, mount_index=0,
                                  mount_count=1, initial_angle=0.0), actuation.MotorState, actuation.MotorInputs,
     set()),
    (actuation.PointDipoleFieldNode("p", DT, slots=("r",), moment=(0, 0, 1), placement=IDENTITY, magnet_radius=1e-2),
     actuation.PointDipoleFieldState, actuation.PointDipoleInputs,
     {"flux_density_r", "gradient_r", "points_r"}),
    (magnetic.MagneticForceNode("f", DT, moment=(0, 1e-3, 0), magnet_centre=(0, 0, 0)), body.Wrench,
     magnetic.MagneticForceInputs, set()),
]


def _declared(kind) -> set[str]:
    return set() if kind is None else set(kind.__annotations__)


def _extra(kind) -> bool:
    return kind is None or getattr(kind, "__extra_items__", NoExtraItems) is not NoExtraItems


def _node_classes() -> set[type]:
    """Every node class in ``mime.nodes`` that steps: a base that writes no step of its own (``MountedNode``) is
    not one."""
    found: set[type] = set()
    for info in pkgutil.walk_packages(mime.nodes.__path__, "mime.nodes."):
        module = importlib.import_module(info.name)
        found |= {cls for _, cls in inspect.getmembers(module, inspect.isclass)
                  if issubclass(cls, SimulationNode) and cls.__module__ == module.__name__
                  and getattr(cls, "advance", None) is not MimeNode.advance}
    return found


def test_every_mime_node_class_has_its_typed_state_listed():
    assert _node_classes() == {type(node) for node, *_ in TYPED}


@pytest.mark.parametrize(("node", "state", "inputs", "constructed"), TYPED, ids=lambda v: type(v).__name__)
def test_each_nodes_typeddicts_name_the_fields_it_holds_and_the_inputs_it_declares(node, state, inputs,
                                                                                  constructed):
    initial = node.initial_state()
    given = {name: jnp.zeros(spec.shape) for name, spec in node.boundary_input_spec().items()}
    stepped = node.update(initial, given, DT)
    for fields in (set(initial), set(stepped)):
        assert _declared(state) <= fields
        extra = fields - _declared(state)
        assert extra <= constructed and (not extra or _extra(state)), extra
    declared_inputs = set(node.boundary_input_spec())
    assert _declared(inputs) <= declared_inputs | _declared(inputs)
    assert declared_inputs <= _declared(inputs) | constructed, declared_inputs - _declared(inputs)


def test_every_input_of_every_mime_node_declares_its_units():
    for node, *_ in TYPED:
        for name, spec in node.boundary_input_spec().items():
            assert spec.expected_units, f"{type(node).__name__}.{name}"
