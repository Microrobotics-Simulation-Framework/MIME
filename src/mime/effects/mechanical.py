"""Mechanical body-load / kinematic-carrier effects (gate M3).

Two thin adapters that move the last hand-wired pieces of the de Jongh / ar4
experiments onto the effects-first ``Experiment.attach`` path:

* :class:`GravityEffect` — a buoyancy-corrected gravity load (``GravityNode``)
  delivered additively to the body's ``external_force``.
* :class:`RobotArmEffect` — a ``RobotArmNode`` whose end-effector pose drives an
  upstream magnet carrier (e.g. the motor's ``parent_pose_world``), so the arm
  holds the rotating magnet exactly as ``ar4_helical_drive`` wires it by hand.

Both are deliberately minimal — they wrap one node and one edge each, following
the magnetic/hydrodynamic adapter pattern. Neither reads the fluid medium nor
constrains the regime (a null regime that never warns).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mime.effects.protocol import BaseEffectModel, EffectHandle, Regime, RegimeWarning
from mime.effects.registry import register_effect

if TYPE_CHECKING:  # pragma: no cover
    from maddening.core.graph_manager import GraphManager
    from mime.effects.body_medium import Body, Medium


class _NullRegime(Regime):
    """A regime that is always applicable (mechanical loads have no envelope)."""

    def check(self, *, body, medium, sources=()) -> list[RegimeWarning]:
        return []


@register_effect("GravityEffect")
class GravityEffect(BaseEffectModel):
    """Buoyancy-corrected gravitational body load.

    Wraps ``GravityNode`` (``F = (ρ_robot − ρ_fluid)·V·g`` in world frame) and
    wires ``gravity_force → body.external_force`` additively so it composes with
    the hydrodynamic and magnetic loads on the same body.

    Parameters
    ----------
    timestep : float
        Node timestep (s).
    delta_rho_kg_m3, volume_m3 : float
        Density excess over the fluid and robot volume (de Jongh FL-9 defaults).
    direction : tuple[float, float, float]
        Gravity direction in the world frame (unit; default −z).
    name : str
        Node name (default ``"gravity"``).
    """

    def __init__(self, *, timestep: float, delta_rho_kg_m3: float = 410.0,
                 volume_m3: float = 5.7e-8, direction=(0.0, 0.0, -1.0),
                 name: str = "gravity"):
        self._dt = float(timestep)
        self._delta_rho = float(delta_rho_kg_m3)
        self._volume = float(volume_m3)
        self._direction = tuple(direction)
        self._name = name

    def applicable_regime(self) -> Regime:
        return _NullRegime()

    def build(self, gm: "GraphManager", *, body: "Body",
              medium: "Medium") -> EffectHandle:
        from mime.nodes.environment.gravity_node import GravityNode
        node = GravityNode(self._name, self._dt, delta_rho_kg_m3=self._delta_rho,
                           volume_m3=self._volume, direction=self._direction)
        gm.add_node(node)
        gm.add_edge(self._name, body.name, "gravity_force", "external_force",
                    additive=True)
        return EffectHandle(node_names=(self._name,))


@register_effect("RobotArmEffect")
class RobotArmEffect(BaseEffectModel):
    """A robot arm that carries an upstream node's parent pose.

    Adds a ``RobotArmNode`` and wires its ``end_effector_pose_world`` into the
    ``parent_pose_world`` of a named carrier node (e.g. the ``MotorNode`` created
    by ``MagneticModel.PointDipole``), reproducing ar4's arm→motor wiring as an
    attachable effect. **Attach order matters:** attach the magnetic drive (which
    creates the carrier) *before* this effect.

    Parameters
    ----------
    urdf_path : str
        Path to the arm URDF.
    carrier_node : str
        Name of the node whose ``parent_pose_world`` the end-effector drives
        (default ``"motor"``).
    end_effector_link_name : str
        URDF link the magnet rides on.
    base_pose_world : tuple
        7-vector base pose of the arm in the world.
    timestep : float
        Node timestep (s).
    end_effector_offset_in_link : tuple
        7-vector tool offset (default identity).
    auto_gravity_compensation : bool
        Add gravity compensation to commanded torques (default True).
    name : str
        Arm node name (default ``"arm"``).
    """

    def __init__(self, *, urdf_path: str, carrier_node: str = "motor",
                 end_effector_link_name: str, base_pose_world, timestep: float,
                 end_effector_offset_in_link=(0, 0, 0, 1, 0, 0, 0),
                 auto_gravity_compensation: bool = True, name: str = "arm"):
        self._urdf = str(urdf_path)
        self._carrier = carrier_node
        self._ee_link = end_effector_link_name
        self._base_pose = tuple(base_pose_world)
        self._dt = float(timestep)
        self._ee_offset = tuple(end_effector_offset_in_link)
        self._auto_gc = bool(auto_gravity_compensation)
        self._name = name

    def applicable_regime(self) -> Regime:
        return _NullRegime()

    def build(self, gm: "GraphManager", *, body: "Body",
              medium: "Medium") -> EffectHandle:
        from mime.nodes.actuation.robot_arm import RobotArmNode
        arm = RobotArmNode(
            name=self._name, timestep=self._dt, urdf_path=self._urdf,
            end_effector_link_name=self._ee_link,
            end_effector_offset_in_link=self._ee_offset,
            base_pose_world=self._base_pose,
            auto_gravity_compensation=self._auto_gc,
        )
        gm.add_node(arm)
        gm.add_edge(self._name, self._carrier,
                    "end_effector_pose_world", "parent_pose_world")
        return EffectHandle(node_names=(self._name,))
