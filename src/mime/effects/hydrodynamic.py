"""HydrodynamicModel family — the v0.2 EffectModel pilot.

ADR-2026-EFFECT-MODEL §7 item 1: v1.0 ships ``HydrodynamicModel`` with four
swappable backends (LBM / FVM / Stokeslet / DefectCorrection). This module
is the *pilot* — it proves the Protocol surface against the one family that
already shares a contract (``FLUID_NODE_CONTRACT.md``): each backend wraps an
existing fluid node and materialises it plus its body-coupling edges onto a
GraphManager, so one backend can be swapped for another across the same
graph edges.

The fluid nodes emit ``drag_force`` / ``drag_torque`` on the contract names;
the wrapper wires them into the rigid body. The LBM backend carries the
``lbm_to_si_*`` edge transforms (its outputs are in lattice units); the BEM
backends are already SI. No node is rewritten — this is an adapter layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from mime.effects.protocol import (
    BaseEffectModel,
    CouplingPort,
    EffectHandle,
    HydrodynamicRegime,
)
from mime.effects.registry import register_effect

if TYPE_CHECKING:  # pragma: no cover
    from maddening.core.graph_manager import GraphManager
    from maddening.core.node import SimulationNode

    from mime.effects.body_medium import Body, Medium


# Shared fluid-node-contract back-edges (FLUID_NODE_CONTRACT.md): map the
# rigid body's *output* field names to the contract `body_*` *input* names the
# fluid node reads. The generic backend wires only the subset a node actually
# declares (LBM consumes orientation + angular velocity; the BEM/FVM family
# also consumes velocity / position), so swapping a backend re-wires exactly
# the back-edges that backend needs.
_BODY_BACK_EDGES = {
    "body_position": "position",
    "body_velocity": "velocity",
    "body_angular_velocity": "angular_velocity",
    "body_orientation": "orientation",
}


def _signed(transform, sign):
    """Compose a ±1 sign onto an edge transform (used to normalise a backend's
    raw drag to the contract 'force on body' convention)."""
    if sign == 1.0:
        return transform
    if transform is None:
        return lambda x: -x
    return lambda x, _t=transform: -_t(x)


class _HydrodynamicEffect(BaseEffectModel):
    """Common adapter: wrap a fluid node + an edge-builder into an EffectModel.

    Parameters
    ----------
    node : SimulationNode
        A pre-constructed fluid node (IBLBM / FVM / Stokeslet /
        DefectCorrection). Backend-specific construction params belong on the
        node itself (ADR decision #6: parameters live in __init__).
    edge_builder : callable | None
        ``(fluid_name, body_name) -> list[EdgeSpec]`` for backends with a
        bespoke wiring helper (e.g. the LBM unit transforms, the Stokeslet
        helper). When None, generic SI wiring is used: forward
        ``drag_force`` / ``drag_torque`` edges into the body, plus the
        contract ``body_*`` back-edges the node declares (see
        ``_BODY_BACK_EDGES``) — valid for the SI backends (FVM /
        DefectCorrection).
    """

    def __init__(
        self,
        node: "SimulationNode",
        *,
        edge_builder: Optional[Callable[[str, str], list]] = None,
        re_range: tuple[float, float] = (0.0, 1.0),
        native_drag_sign: float = 1.0,
    ):
        self._node = node
        self._edge_builder = edge_builder
        self._re_range = re_range
        # Multiplier applied to the node's raw drag_force / drag_torque so the
        # body always receives the **force on the body** (opposing its motion)
        # — the contract sign (FLUID_NODE_CONTRACT.md). Some nodes report the
        # *reaction* (`+R·motion`, the force on the fluid): IBLBM and the
        # standalone Stokeslet do — verified on a clean achiral sphere, and
        # confirmed anti-dissipative (a de-Boer UMR diverges when fed `+R·ω`
        # without the omega-max clamp). Those backends pass `native_drag_sign=
        # -1`; FVM already returns the force on the body and passes `+1`.
        self._native_drag_sign = float(native_drag_sign)

    def applicable_regime(self) -> HydrodynamicRegime:
        return HydrodynamicRegime(self._re_range)

    def required_medium_properties(self) -> set[str]:
        return {"density", "viscosity"}

    def build(self, gm: "GraphManager", *, body: "Body", medium: "Medium") -> EffectHandle:
        gm.add_node(self._node)
        fluid_name = self._node.name
        sign = self._native_drag_sign
        if self._edge_builder is not None:
            for e in self._edge_builder(fluid_name, body.name):
                tf = getattr(e, "transform", None)
                # Normalise the forward drag edges to "force on body".
                if (e.source_node == fluid_name
                        and e.source_field in ("drag_force", "drag_torque")):
                    tf = _signed(tf, sign)
                gm.add_edge(
                    e.source_node, e.target_node,
                    e.source_field, e.target_field,
                    transform=tf,
                    additive=getattr(e, "additive", False),
                )
        else:
            # Generic SI wiring (FVM / DefectCorrection). Forward: the
            # hydrodynamic load → body, normalised to "force on body".
            drag_tf = _signed(None, sign)
            gm.add_edge(fluid_name, body.name, "drag_force", "drag_force",
                        transform=drag_tf, additive=True)
            gm.add_edge(fluid_name, body.name, "drag_torque", "drag_torque",
                        transform=drag_tf, additive=True)
            # Back-edges: body kinematics → fluid, for the contract `body_*`
            # inputs this node declares (the SI fluid nodes need the body's
            # velocity / position to impose the immersed-boundary condition).
            declared = set(self._node.boundary_input_spec())
            for fluid_input, body_field in _BODY_BACK_EDGES.items():
                if fluid_input in declared:
                    gm.add_edge(body.name, fluid_name, body_field, fluid_input)
        return EffectHandle(node_names=(fluid_name,))


class HydrodynamicModel:
    """Namespace of swappable hydrodynamic backends (ADR §7 item 1)."""

    @register_effect("HydrodynamicModel.LBM")
    class LBM(_HydrodynamicEffect):
        """IB-LBM backend. Carries the lattice→SI drag edge transforms."""

        def __init__(
            self,
            node: "SimulationNode",
            *,
            dx_physical: float,
            dt_physical: float,
            fluid_density: float = 1060.0,
            re_range: tuple[float, float] = (0.0, 1.0),
        ):
            from mime.nodes.environment.lbm.fluid_node import (
                make_iblbm_rigid_body_edges,
            )

            def _edges(fluid_name: str, body_name: str) -> list:
                return make_iblbm_rigid_body_edges(
                    fluid_name, body_name, dx_physical, dt_physical,
                    fluid_density,
                )

            # IBLBM momentum exchange reports +R·motion (the reaction); flip to
            # force-on-body. Verified anti-dissipative otherwise (de-Boer UMR
            # diverges without the omega-max clamp).
            super().__init__(node, edge_builder=_edges, re_range=re_range,
                             native_drag_sign=-1.0)

    @register_effect("HydrodynamicModel.Stokeslet")
    class Stokeslet(_HydrodynamicEffect):
        """Regularised-Stokeslet BEM backend (SI; bespoke edge helper)."""

        def __init__(self, node: "SimulationNode", *, re_range=(0.0, 1.0)):
            from mime.nodes.environment.stokeslet.fluid_node import (
                make_stokeslet_rigid_body_edges,
            )

            def _edges(fluid_name: str, body_name: str) -> list:
                return make_stokeslet_rigid_body_edges(fluid_name, body_name)

            # Standalone Stokeslet reports +R·[U,ω] (the reaction); flip to
            # force-on-body. (Its Schwarz path already integrates traction =
            # force on body; this normalises the standalone path to match.)
            super().__init__(node, edge_builder=_edges, re_range=re_range,
                             native_drag_sign=-1.0)

    @register_effect("HydrodynamicModel.FVM")
    class FVM(_HydrodynamicEffect):
        """Finite-volume + IBM backend (SI; generic drag edges). The
        momentum-deficit force is already the force on the body (opposing
        motion), so no sign flip."""

        def __init__(self, node: "SimulationNode", *, re_range=(0.0, 1.0)):
            super().__init__(node, edge_builder=None, re_range=re_range,
                             native_drag_sign=+1.0)

    # Chain-specific body back-edges: the de Jongh resistance chain reads the
    # body kinematics under different field names than the single-node contract
    # (robot_position / robot_orientation), so the StokesletChain wires its own.
    _CHAIN_BACK_EDGES = {
        "position": "robot_position",
        "orientation": "robot_orientation",
        "velocity": "body_velocity",
        "angular_velocity": "body_angular_velocity",
    }

    @register_effect("HydrodynamicModel.StokesletChain")
    class StokesletChain(BaseEffectModel):
        """Confined-swimming resistance chain (de Jongh): an MLP resistance
        surrogate, optionally corrected by a near-wall lubrication model, →
        body drag. This is the **confined Stokes-flow near-field body model**
        for the two-scale Against-the-Current solver (E6d) — the variant the
        de Jongh / Li–Misra–Khalil validation runs on (``dejongh.py`` /
        ``dejongh_new_chain.py``), which is *not* a single ``StokesletFluidNode``.

        When a lubrication node is present, its ``background_velocity`` input —
        the ambient fluid velocity at the robot — is exposed as a cross-effect
        **input port**, so a far-field hydrodynamic effect (the inertial FVM
        vessel flow) can ``couple()`` its velocity into the near field: the
        two-scale Schwarz coupling, materialised through the EffectModel surface.

        Parameters
        ----------
        mlp_node : SimulationNode
            An ``MLPResistanceNode`` emitting ``drag_force`` / ``drag_torque`` /
            ``resistance_matrix``.
        lubrication_node : SimulationNode | None
            An optional ``LubricationCorrectionNode``; when given it takes the
            MLP resistance matrix and emits the corrected drag to the body.
        """

        def __init__(
            self,
            mlp_node: "SimulationNode",
            *,
            lubrication_node: Optional["SimulationNode"] = None,
            re_range: tuple[float, float] = (0.0, 1.0),
        ):
            self._mlp = mlp_node
            self._lub = lubrication_node
            self._re_range = re_range

        def applicable_regime(self) -> HydrodynamicRegime:
            return HydrodynamicRegime(self._re_range)

        def required_medium_properties(self) -> set[str]:
            return {"density", "viscosity"}

        @property
        def input_ports(self) -> dict[str, "CouplingPort"]:
            # The near-field body model consumes the far-field ambient velocity
            # at the robot — the two-scale coupling input (lubrication carries it).
            if self._lub is not None:
                return {"background_velocity": CouplingPort(
                    node=self._lub.name, field="background_velocity", shape=(3,))}
            return {}

        def _wire_body_inputs(self, gm, target, body):
            declared = set(target.boundary_input_spec())
            for body_field, node_input in HydrodynamicModel._CHAIN_BACK_EDGES.items():
                if node_input in declared:
                    gm.add_edge(body.name, target.name, body_field, node_input)

        def build(self, gm: "GraphManager", *, body: "Body", medium: "Medium") -> EffectHandle:
            gm.add_node(self._mlp)
            self._wire_body_inputs(gm, self._mlp, body)
            names = [self._mlp.name]
            if self._lub is not None:
                gm.add_node(self._lub)
                names.append(self._lub.name)
                gm.add_edge(self._mlp.name, self._lub.name,
                            "resistance_matrix", "resistance_matrix")
                self._wire_body_inputs(gm, self._lub, body)
                drag_src = self._lub.name
            else:
                drag_src = self._mlp.name
            # The MLP/lubrication drag is already the force on the body
            # (de Jongh wires it directly, validated) — additive, no sign flip.
            gm.add_edge(drag_src, body.name, "drag_force", "drag_force",
                        additive=True)
            gm.add_edge(drag_src, body.name, "drag_torque", "drag_torque",
                        additive=True)
            return EffectHandle(node_names=tuple(names))

    @register_effect("HydrodynamicModel.DefectCorrection")
    class DefectCorrection(_HydrodynamicEffect):
        """BEM/LBM defect-correction backend (SI; generic drag edges).

        Computes drag via ``compute_force_torque`` (a surface-traction
        integral), which is the *force on the body* (opposing) — the same
        convention as FVM and the Stokeslet Schwarz path — hence
        ``native_drag_sign=+1``. **Unverified at runtime:** this backend needs
        a heavy LBM far-field spin-up to exercise, so it is not in the v0.2
        runnable concept-proof; confirm the sign on first real use (tracked as
        E6g). If a coupled run proves anti-dissipative, flip to ``-1``.
        """

        def __init__(self, node: "SimulationNode", *, re_range=(0.0, 1.0)):
            super().__init__(node, edge_builder=None, re_range=re_range,
                             native_drag_sign=+1.0)

    @register_effect("HydrodynamicModel.TwoScale")
    class TwoScale(BaseEffectModel):
        """Composite two-scale Schwarz coupling (C3): an inertial FVM far-field
        ⊗ a confined Stokes-BEM near-field, attached as **one** effect instead
        of a hand-wired pair.

        Wraps :func:`mime.nodes.environment.two_scale.make_two_scale_coupling`
        (C1): the **near** field couples to the body — drag → body + body
        kinematics → near, via the standalone-Stokeslet convention
        (``make_stokeslet_rigid_body_edges`` + the −1 drag-sign flip from the
        reaction to force-on-body) — and the **far** field couples to the near
        field (``velocity_at_points`` → ``background_flow``, ``body_traction``×
        weights → reaction forcing) through a Schwarz coupling group.

        Parameters
        ----------
        far_node : FVMFluidNode
            The inertial far-field (Cartesian + IBM vessel wall on the v0.3.0
            path), built with ``n_sample_points = n_forcing_points = N_body``.
        near_node : StokesletFluidNode
            The confined-Stokes near-field in Schwarz mode (``interface_mesh``).
        body_points, body_weights : arrays
            BEM body-surface points (sample/forcing locations) and quadrature
            weights (traction → force).

        Drag-sign note: the coupled-body drag sign mirrors the standalone
        Stokeslet backend (−1); confirm on first running coupled-body use
        (E-phase, as for ``DefectCorrection``/E6g).

        **Geometry footgun (must read before stepping):** the underlying
        ``make_two_scale_coupling`` registers ``sample_points``/``forcing_points``
        as *external inputs* on the far node (the body-surface locations where the
        FVM velocity is sampled and the BEM reaction is spread). This effect builds
        the coupling but does **not** bake or pose-track them. If they are left
        unset the FVM samples/forces at the **origin** and the coupling diverges to
        NaN within a few steps. The experiment must supply them each step
        (constant for a held body; pose-transformed for a moving one — see
        ``mime.experiments.schwarz_vessel_helix.body_world_points``). An in-graph
        pose transform wired from the body is the proper fix (follow-on).
        """

        def __init__(
            self,
            far_node: "SimulationNode",
            near_node: "SimulationNode",
            *,
            body_points,
            body_weights,
            coupling_kwargs: Optional[dict] = None,
            re_range: tuple[float, float] = (0.0, 2000.0),
        ):
            self._far = far_node
            self._near = near_node
            self._body_points = body_points
            self._body_weights = body_weights
            self._coupling_kwargs = coupling_kwargs
            self._re_range = re_range

        def applicable_regime(self) -> HydrodynamicRegime:
            return HydrodynamicRegime(self._re_range)

        def required_medium_properties(self) -> set[str]:
            return {"density", "viscosity"}

        def build(self, gm: "GraphManager", *, body: "Body",
                  medium: "Medium") -> EffectHandle:
            from mime.nodes.environment.stokeslet.fluid_node import (
                make_stokeslet_rigid_body_edges,
            )
            from mime.nodes.environment.two_scale import make_two_scale_coupling

            # near field ↔ body: drag → body (reaction → force-on-body, −1) and
            # body kinematics → near, the standalone-Stokeslet convention.
            gm.add_node(self._near)
            for e in make_stokeslet_rigid_body_edges(self._near.name, body.name):
                tf = getattr(e, "transform", None)
                if (e.source_node == self._near.name
                        and e.source_field in ("drag_force", "drag_torque")):
                    tf = _signed(tf, -1.0)
                gm.add_edge(e.source_node, e.target_node,
                            e.source_field, e.target_field,
                            transform=tf, additive=getattr(e, "additive", False))

            # far field ↔ near field: the two-scale Schwarz coupling (C1).
            make_two_scale_coupling(
                gm, self._far, self._near,
                body_points=self._body_points,
                body_weights=self._body_weights,
                coupling_kwargs=self._coupling_kwargs)

            return EffectHandle(node_names=(self._far.name, self._near.name))
