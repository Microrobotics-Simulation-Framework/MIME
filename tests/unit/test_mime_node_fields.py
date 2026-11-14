"""MIME's nodes' typed state: each node's TypedDicts name exactly the fields it holds and the inputs it declares, as
the toy nodes' do (``test_node_fields.py``). Every node class in ``mime.nodes`` must be listed here."""

import importlib
import inspect
import pkgutil
from pathlib import Path

import jax.numpy as jnp
import pytest
from rft_cases import r6_node
from maddening.core.node import SimulationNode
from typing_extensions import NoExtraItems

import mime.nodes

import mime.scene  # noqa: F401 - registers the sphere kind the MRS node is made with
from mime.core import MimeNode
from mime.nodes.flow import fvm as flow
from mime.nodes.flow._body_force import BodyForceNode, BodyForceState
from mime.nodes.flow._womersley import PrescribedFlowInputs, PrescribedFlowNode, PrescribedFlowState
from mime.nodes.hydro._point_drag import PointDragInputs, PointDragNode, PointDragState
from mime.nodes.flow.fvm._resolved_node import BODY_FIELDS, BODY_INPUTS, ResolvedFlowInputs, ResolvedFlowNode, \
    ResolvedFlowState, ResolvedSpec
from mime.core.multibody import read_urdf
from mime.data import asset
from mime.nodes import actuation, body, control, hydro, magnetic
from mime.nodes.hydro import mrs, near_field

DT = 1e-3
TUBE = dict(surface_shape="cylinder", extent=1e-3, margin=1e-4, velocity_rule="remove_outward",
            surface_centre=(0.0, 0.0, 0.0), surface_axis=(1.0, 0.0, 0.0), surface_radius=3e-3)

#: An MRS sphere's arguments at a coarse count, its quadrature the force points themselves (one to a cell).
SPHERE = dict(blob=0.5, kind="sphere", kind_version="0.1.0", patches=("surface",), counts=((3, 5),),
              part_centre=(0.0, 0.0, 0.0), part_rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1), shape_radius=1e-3)


def _joint_node():
    """A medium's near field of two coarse spheres, each member's arguments under ``m<i>_``."""
    count = mrs.MRSDragNode("c", DT, viscosity=1e-3, wall_kind="free_space", wall_kind_version="0.1.0",
                            **SPHERE).sample_count
    member = {**SPHERE, "quadrature_owners": tuple(range(count))}
    return near_field.JointDragNode("n", DT, members=("a", "b"), viscosity=1e-3, wall_kind="free_space",
                                          wall_kind_version="0.1.0",
                                          **{f"m{i}_{key}": value for i in range(2) for key, value in member.items()})


PLANAR = read_urdf(Path(__file__).parents[1] / "descriptions" / "three_link_planar.urdf").tree.articulation
IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

TYPED = [
    (body.RigidInertialNode("b", DT, mass=1e-4, inertia=(1e-10,) * 3, initial_pose=(0, 0, 0, 1, 0, 0, 0),
                            initial_twist=(0.0,) * 6, **TUBE), body.RigidInertialState, body.RigidInertialInputs,
     set()),
    (body.RigidInertialMidpointNode("b", DT, mass=1e-4, inertia=(1e-10,) * 3, initial_pose=(0, 0, 0, 1, 0, 0, 0),
                                    initial_twist=(0.0,) * 6, **TUBE), body.RigidInertialState,
     body.RigidInertialInputs, set()),
    (body.RigidInertialTensorNode("b", DT, mass=1e-4, inertia=(1e-10, 0.0, 0.0, 0.0, 1e-10, 0.0, 0.0, 0.0, 1e-10),
                                  initial_pose=(0, 0, 0, 1, 0, 0, 0), initial_twist=(0.0,) * 6, **TUBE),
     body.RigidInertialState, body.RigidInertialInputs, set()),
    (body.GravityNode("g", DT, mass=1e-4, buoyant_volume=6e-8, fluid_density=1e3, gravity=(0, 0, -9.81),
                      centre_of_buoyancy=(0, 0, 1e-4)), body.Wrench, body.GravityInputs, set()),
    (body.SwimSpeedNode("s", DT, along=(1, 0, 0)), body.SwimSpeedState, body.SwimSpeedInputs, set()),
    (body.PhaseLagNode("l", DT, moment=(0, 1e-3, 0), about=(1, 0, 0)), body.PhaseLagState, body.PhaseLagInputs,
     set()),
    (magnetic.MagneticTorqueNode("t", DT, moment=(0, 1e-3, 0)), body.Wrench, magnetic.MagneticTorqueInputs, set()),
    (actuation.UniformRotatingNode("f", DT, slots=("r",), rotation_axis=(0, 0, 1), initial_direction=(1, 0, 0)),
     actuation.UniformRotatingState, actuation.UniformRotatingInputs, {"flux_density_r", "gradient_r"}),
    (actuation.IntegratingRotatingNode("f", DT, slots=("r",), rotation_axis=(0, 0, 1), initial_direction=(1, 0, 0)),
     actuation.IntegratingRotatingState, actuation.UniformRotatingInputs, {"flux_density_r", "gradient_r"}),
    (actuation.ConstantNode("c", DT, value=10.0), actuation.Command, None, set()),
    (actuation.RampNode("c", DT, start=10.0, rate=-1.0, duration=2.0, hold=0.5), actuation.RampState, None, set()),
    (flow.ImplicitFlowNode("w", DT, cells_x=11, cells_y=11, cells_z=2, spacing=1e-4, viscosity=1e-3, density=1e3,
                           tube_centre=(5.5e-4, 5.5e-4, 0.0), tube_axis=(0, 0, 1), tube_radius=2.5e-4),
     flow.FlowState, flow.FlowInputs, set()),
    (ResolvedFlowNode("w", DT, cells_x=11, cells_y=11, cells_z=8, spacing=1e-4, viscosity=1e-3, density=1e3,
                      tube_centre=(5.5e-4, 5.5e-4, 0.0), tube_axis=(0, 0, 1), tube_radius=4e-4,
                      bodies=[ResolvedSpec("ball", "drag:ball", "sphere", {"centre": (0.0, 0.0, 0.0), "radius": 1.5e-4},
                                           (0.0, 0.0, 0.0), (1, 0, 0, 0, 1, 0, 0, 0, 1), 9, 1.4e-11)]),
     ResolvedFlowState, ResolvedFlowInputs,
     {f"body_0_{field}" for field in (*BODY_FIELDS, *BODY_INPUTS)}),
    (flow.OpenFlowNode("w", DT, cells_x=11, cells_y=11, cells_z=4, spacing=1e-4, viscosity=1e-3, density=1e3,
                       tube_centre=(5.5e-4, 5.5e-4, 0.0), tube_axis=(0, 0, 1), tube_radius=2.5e-4,
                       openings=("tube.start", "tube.end"), opening_laws=("inflow", "windkessel"),
                       opening_parameters=[[1e-12, 0.0, 0.0, 0.0], [1e8, 1e-13, 1e9, 0.0]]),
     flow.OpenFlowState, flow.FlowInputs, set()),
    (BodyForceNode("f", DT, along=(0, 0, 1), steady=1.0), BodyForceState, None, set()),
    (PointDragNode("p", DT, radius=5e-5, viscosity=1e-3, density=1e3, spacing=1e-4, width_cells=1.0,
                   origin=(0.0, 0.0, 0.0), initial_points=(5e-4, 5e-4, 1e-4)), PointDragState, PointDragInputs, set()),
    (flow.ExplicitFlowNode("w", DT, cells_x=11, cells_y=11, cells_z=2, spacing=1e-4, viscosity=1e-3, density=1e3,
                           tube_centre=(5.5e-4, 5.5e-4, 0.0), tube_axis=(0, 0, 1), tube_radius=2.5e-4),
     flow.ExplicitFlowState, flow.FlowInputs, set()),
    (PrescribedFlowNode("w", DT, viscosity=1e-3, density=1e3, tube_centre=(0, 0, 0), tube_axis=(0, 0, 1),
                        tube_radius=2.5e-4), PrescribedFlowState, PrescribedFlowInputs, set()),
    (hydro.SurrogateDragNode("d", DT, weights="dejongh_mlp_v2", weights_sha256=asset("dejongh_mlp_v2").sha256,
                             viscosity=1e-3, helix_wavenumber=2.33, length=7.47e-3, radius=1.56e-3,
                             outer_radius=1.33 * 1.56e-3, vessel_shape="cylinder", vessel_centre=(0, 0, 0), vessel_axis=(1, 0, 0), vessel_radius=3.175e-3,
                             lubrication_epsilon=1.5e-4), hydro.SurrogateDragState, hydro.SurrogateDragInputs, set()),
    (actuation.KinematicArmNode("a", DT, **actuation.KinematicArmNode.described(PLANAR), placement=IDENTITY,
                                initial_joint_positions=(0.0, 0.0, 0.0)), actuation.ArmState, actuation.ArmInputs,
     set()),
    (actuation.KinematicMotorNode("m", DT, axis=(0, 0, 1), prescribed="velocity", placement=IDENTITY, mount_index=0,
                                  mount_count=1, initial_angle=0.0), actuation.MotorState, actuation.MotorInputs,
     set()),
    (actuation.DynamicArmNode("a", DT, **actuation.DynamicArmNode.described(PLANAR, (0.05, 0.05, 0.02), (0.0, 0.0, 0.0)),
                              placement=IDENTITY, gravity=(0.0, 0.0, -9.81), initial_joint_positions=(0.0, 0.0, 0.0)),
     actuation.DynamicArmState, actuation.DynamicArmInputs, set()),
    (actuation.DynamicMotorNode("m", DT, axis=(0, 0, 1), inertia=1e-5, damping=1e-4, placement=IDENTITY, mount_index=0,
                                mount_count=1, initial_angle=0.0), actuation.DynamicMotorState,
     actuation.DynamicMotorInputs, set()),
    (actuation.SourceReactionNode("r", DT, mount_index=1, mount_count=2), actuation.SourceReactionState,
     actuation.SourceReactionInputs, set()),
    (actuation.PointDipoleNode("p", DT, slots=("r",), moment=(0, 0, 1), placement=IDENTITY, magnet_radius=1e-2),
     actuation.PointDipoleState, actuation.PointDipoleInputs,
     {"flux_density_r", "gradient_r", "points_r"}),
    (actuation.CalibratedMultipoleNode("a", DT, slots=("r",), sources=(1,), source_positions=(0.0, 0.0, 0.1),
                                       source_directions=(0.0, 0.0, 1.0), coefficients=(1e-9,), at=(0.0, 0.0, 0.0),
                                       orientation=(1.0, 0.0, 0.0, 0.0), background=(0.0, 0.0, 0.0),
                                       region=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), gradient_limit=(1.0,)),
     actuation.CalibratedMultipoleState, None,
     {"flux_density_r", "gradient_r", "source_distance_r", "points_r", "delivered_currents"}),
    (magnetic.MagneticForceNode("f", DT, moment=(0, 1e-3, 0), magnet_centre=(0, 0, 0)), body.Wrench,
     magnetic.MagneticForceInputs, set()),
    (mrs.MRSDragNode("e", DT, viscosity=1e-3, blob=0.5, kind="sphere", kind_version="0.1.0", patches=("surface",),
                     counts=((3, 5),), part_centre=(0.0, 0.0, 0.0), part_rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                     wall_kind="free_space", wall_kind_version="0.1.0", shape_radius=1e-3), mrs.MRSDragState, mrs.MRSDragInputs, set()),
    (_joint_node(), near_field.JointDragState, near_field.JointDragInputs,
     {f"{field}_{i}" for i in range(2) for field in ("resistance", "background_load", "pose", "twist", "mass_matrix")}),
    (r6_node(tube=False), hydro.RFTDragState, hydro.RFTDragInputs, set()),
    (control.ControllerNode("k", DT, step=lambda state, reads: (state, {"angle": reads["pose"][4]}),
                            controller="toy 0.0.0", reads={"pose": ("m; 1", (7,))},
                            initial_state={"filtered": jnp.zeros(())}, initial_writes={"angle": jnp.zeros(())}),
     control.ControllerState, control.ControllerInputs, {"state_filtered", "write_angle", "read_pose"}),
    (control.DelayNode("l", DT, latency=2, units={"angle": "1"}, initial_writes={"angle": jnp.zeros(())}),
     control.DelayState, control.DelayInputs, {"angle", "waiting_angle"}),
    (control.CommandStageNode("ports:a", DT, commands=["c"], units={"c": "A"}, shapes={"c": (3,)}, periods={"c": 2},
                              latencies={"c": 1}, bounded={"c": "trip"}, slewed={"c": True},
                              responses={"c": ("response.first_order_lag", "0.1.0")}, carries={"c": ("lagged",)},
                              rest_c=0.0, low_c=-1.0, high_c=1.0, slew_c=10.0, response_c_tau=0.01),
     control.StageState, control.StageInputs, {"c", "written_c", "buffer_c", "tripped_c", "amplified_c",
                                               "response_c_lagged"}),
    (control.MeasureStageNode("measure:a", DT, ports=["p"], units={"p": "1"}, shapes={"p": (2,)}, periods={"p": 2},
                              latencies={"p": 1}, aged={"p": True}, seeds={"p": 1}, noisy={"p": True},
                              quantised={"p": True}, initial_p=0.0, initial_age_p=0, sigma_p=0.1, quantum_p=0.01),
     control.StageState, control.StageInputs, {"p", "p_age", "buffer_p", "stamps_p", "age_p"}),
]


def _declared(kind) -> set[str]:
    return set() if kind is None else set(kind.__annotations__)


def _required(kind) -> set[str]:
    """The fields a node always holds: its typed state's, but those it marks ``NotRequired`` (a joint near field's wall
    distances, held only beside a wall that reports them)."""
    from mime.core._field_kinds import _typed

    typed = None if kind is None else _typed(kind)
    return set() if typed is None else set(typed[1])


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
        assert _required(state) <= fields
        extra = fields - _declared(state)
        assert extra <= constructed and (not extra or _extra(state)), extra
    declared_inputs = set(node.boundary_input_spec())
    assert _declared(inputs) <= declared_inputs | _declared(inputs)
    assert declared_inputs <= _declared(inputs) | constructed, declared_inputs - _declared(inputs)


def test_every_input_of_every_mime_node_declares_its_units():
    for node, *_ in TYPED:
        for name, spec in node.boundary_input_spec().items():
            assert spec.expected_units, f"{type(node).__name__}.{name}"


@pytest.mark.parametrize(("node", "state", "inputs", "constructed"), TYPED, ids=lambda v: type(v).__name__)
def test_each_nodes_typed_state_is_the_one_its_fields_kinds_are_read_from(node, state, inputs, constructed):
    """``mime verify`` reads a saved field's kind from the closed TypedDict, near its node's class, that names exactly the
    fields it holds (``mime.core.field_kinds``): for every node here, the state listed above."""
    from mime.core._field_kinds import _names_exactly, _states_near

    assert state in _states_near(type(node)) and _names_exactly(state, frozenset(node.initial_state()))
