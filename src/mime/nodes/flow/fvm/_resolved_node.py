"""The implicit flow node with resolved bodies on its grid (F4, ``Drag.Penalty``): :class:`ImplicitFlowNode`'s step,
each body's solid lifted onto the faces at the pose its input gives (``mime.nodes.flow.fvm._resolved``), and each body's
load delivered back to it.

**The coupling is lagged, declared** (the coupling spike's R3): a step solves the field with each body's penalty driving
its interior to the twist the body had at the step's start, and delivers the load the solved field puts on it, which the
body takes as its background load in the same step (the flow is stepped before the bodies: media before bodies in the
canonical order). R3 measured this stable from a density ratio of 1.4 down to 0.3 at Re 76 with the field formulation
of the load. The one-pass affine response, which pairs the exchange at one twist and closes the energy exactly, is the
next form (the one elimination of bodies and openings).

**A body's load** (B8, point 5): the penalty's force and torque at the solved field, ``rho h^3 sum nu/K (u' - u_b)``,
plus the interior's momentum and angular momentum change, ``(Q' - Q)/dt``, ``Q' = rho h^3 sum chi u'`` on the step's
indicator and ``Q`` the value the step before held (at a run's first step the start's field on the start's indicator),
so the change telescopes over a run. The steady limit stores no inertia, so its load is the penalty's alone.

**The ledger.** The node holds each body's penalty as a law in its step, booked under the body's drag
(``body_<i>`` its role): its work on the fluid, ``-dt rho <nu/K (u' - u_b), u'>``, which is the law's relative
dissipation less the exchange with the body at the twist the field was solved at; the body books the load it takes
under the same law. **Its sign is judged here, at one velocity** (``body_<i>_gift_steps``): the penalty's load on the
body at that twist, ``dt xi . F``, with its work on the fluid, is ``-dt rho <nu/K (u' - u_b), u' - u_b>`` (``body_<i>_power``,
never above its rounding floor), summed in ``body_<i>_dissipated``. The rest of the law's row is the lagged exchange's
unpaired work (the twist the body reached against the one the field was solved at) and the interior's momentum change
on the body's twist, whose energy the fluid's storage holds (the interior fluid, which moves with the body).

**The displaced share** (the record's Part D tail, item 2): the field's drive acts on its fluid alone, never inside a
solid, so the solved field's load on a body lacks the share ``rho V_b f`` a uniform drive ``f`` (a pressure drop's
stand-in, or the fluid's weight) puts on the fluid the body displaces. The node gives it apart (``body_<i>_share``, at the
hull's centroid, its volume the hull's, not the indicator's), delivered to the body as a part of its drag's own,
``<id>.displaced``: the solver never takes it."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.params import ParamSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import Symbol, SymbolKind, as_run_array, jnp, node_input, run_dtype, shape_named
from mime.core import ledger as books
from mime.core.ledger import IDENTITY_ULPS
from mime.nodes.flow.fvm._band import WallFields, band_named, interior_depth
from mime.nodes.flow.fvm._grid import MACGrid, Velocity
from mime.nodes.flow.fvm._node import EQUATIONS, FlowFields, ImplicitFlowNode, StepParts
from mime.nodes.flow.fvm._resolved import BodyFaces, ResolvedBody, body_faces, moments, placed, pose_turn, \
    rigid_velocity, scattered, united

#: The per-body fields a resolved node holds, ``body_<i>_<name>`` for its ``i``-th body.
BODY_FIELDS = ("load", "penalty", "momentum", "indicator", "share", "power", "dissipated", "gift_steps")
#: The per-body inputs, ``body_<i>_<name>``: the body's pose and twist at the step's start.
BODY_INPUTS = ("pose", "twist")
#: A body's books in the step (the module's docstring states them): the penalty's load, its interior's momentum, its
#: load, its indicator's volume, its displaced share, the law's power at one velocity, its dissipation and its gift
#: steps.
BODY_EQUATIONS = (r", \quad \mathbf{F}_p = \rho \sum \operatorname{penalty}_{b} (\mathbf{u}_{n+1} - \mathbf{u}_{b}), "
                  r"\quad \mathbf{Q}_b = \rho \sum \operatorname{chi}_{b} \mathbf{u}_{n+1}, \quad \mathbf{F}_b = "
                  r"\mathbf{F}_p + \frac{\mathbf{Q}_b - \mathbf{Q}_{b,n}}{h}, \quad V_\chi = \sum \operatorname{chi}_{b}, "
                  r"\quad \mathbf{S}_b = \rho V_{b} \mathbf{g}, \quad P_b = \mathbf{F}_p \cdot \mathbf{u}_{b} - \rho \sum "
                  r"\operatorname{penalty}_{b} (\mathbf{u}_{n+1} - \mathbf{u}_{b}) \cdot \mathbf{u}_{n+1}, \quad D_b = "
                  r"-\sum h P_b, \quad n_b = \sum \operatorname{gift}(P_b)")
#: Where the symbol rows' functions are.
_STEP = "mime.nodes.flow.fvm._resolved_node.ResolvedFlowNode.stepped"
_BOOKS = "mime.nodes.flow.fvm._resolved_node.ResolvedFlowNode.body_books"
#: The identity pose: a body's input before anything is fed.
REST_POSE = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


class ResolvedFlowState(FlowFields, extra_items=ReadOnly[Float[Array, "..."]]):  # noqa: F722 - a jaxtyping shape
    """The flow's fields, and for each resolved body ``i``: ``body_<i>_load`` (its load, force and torque about its centre,
    N and N m: what it takes this step), ``body_<i>_penalty`` (the penalty's part of it), ``body_<i>_momentum`` (its interior's
    momentum and angular momentum on the step's indicator, kg m/s and kg m^2/s: what the next step's change is taken
    from) and ``body_<i>_indicator`` (its indicator's volume on the axial faces, m^3)."""


class ResolvedFlowInputs(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):  # noqa: F722 - a jaxtyping shape
    """The drive's, and ``body_<i>_pose`` and ``body_<i>_twist`` for each resolved body, from its state at the step's start."""

    body_force: NotRequired[ReadOnly[Float[Array, "3"]]]
    waveform: NotRequired[ReadOnly[Float[Array, "7"]]]


@dataclass(frozen=True)
class ResolvedSpec:
    """A resolved body as the build gives it: its body's name, its solid region's shape and parameters in its part's
    frame (SI), the part's centroid in the body's frame and its rotation to the body's (nine numbers, rows), the
    window's side in cells, and its hull's volume (m^3)."""

    name: str
    law: str
    shape: str
    parameters: Mapping[str, float | tuple[float, ...]]
    centre: tuple[float, float, float]
    rotation: tuple[float, ...]
    window: int
    volume: float

    def arguments(self, index: int) -> dict[str, Any]:
        """The node's arguments for it, ``body_<index>_<name>``."""
        prefix = f"body_{index}_"
        held: dict[str, Any] = {f"{prefix}name": self.name, f"{prefix}law": self.law, f"{prefix}shape": self.shape,
                                f"{prefix}centre": tuple(float(v) for v in self.centre),
                                f"{prefix}rotation": tuple(float(v) for v in self.rotation),
                                f"{prefix}window": int(self.window), f"{prefix}volume": float(self.volume)}
        for key in shape_named(self.shape).parameter_names:
            value = self.parameters[key]
            held[f"{prefix}{key}"] = tuple(float(v) for v in value) if isinstance(value, tuple) else float(value)
        return held


@dataclass(frozen=True)
class ResolvedFields(WallFields):
    """The solid's fields with the bodies' (:class:`WallFields`, the union), each body's windows on the three face
    sets, its twist at the step's start, and the penalty's push towards it, ``nu/K u_b`` on each face set (m/s^2)."""

    faces: tuple[tuple[BodyFaces, BodyFaces, BodyFaces], ...] = ()
    twists: tuple[Array, ...] = ()
    pushed: tuple[Array, ...] = ()


class ResolvedFlowNode(ImplicitFlowNode):
    """:class:`ImplicitFlowNode` with resolved bodies on its grid, their coupling lagged (the module says how)."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-131",
        description="A medium's flow on a periodic box holding a straight tube and resolved rigid bodies: each body's "
                    "solid the wall's band lifted from its shape's signed distance at its pose, its penalty driving its "
                    "interior to its rigid twist; each body's load the penalty's plus its interior's momentum change.",
        governing_equations=EQUATIONS + BODY_EQUATIONS,
        assumptions=(
            "Incompressible Newtonian fluid, its viscosity and density uniform, on the MAC grid of the implicit flow "
            "node, its wall a smooth Brinkman band",
            "Each resolved body is rigid, its solid lifted by the wall's own band from its solid region's exact signed "
            "distance, evaluated on a window of faces about it; the solid is the union of the wall's and the bodies'",
            "A body's load is the penalty's force and torque plus its interior fluid's momentum change taken from the "
            "field (the field formulation; the rigid-interior shortcut is not used)",
            "The coupling is lagged: the field is solved at each body's twist at the step's start, and the body takes "
            "the load the solved field puts on it in the same step"),
        limitations=(
            "A periodic box only: no openings, the flow driven by a body force on the fluid",
            "The lagged coupling was measured stable from a density ratio of 1.4 down to 0.3 at Re 76 (the coupling "
            "spike's R3); its exchange is not paired at one twist, so its unpaired work is the coupling's own residual",
            "A body is resolved where its thinnest feature holds the cells the build's line asks; bodies at least two "
            "cells from the wall and from each other (closer, the bands overlap: the near-wall resolved-body phase)",
        ),
        references=(Reference("HarlowWelch1965"), Reference("Angot1999"), Reference("KempeFrohlich2012")),
    )

    symbols = (
        *ImplicitFlowNode.symbols,
        Symbol(r"\mathbf{F}_b", SymbolKind.STATE, "body_{slot}_load", "N; N m", _STEP),
        Symbol(r"\mathbf{F}_p", SymbolKind.STATE, "body_{slot}_penalty", "N; N m", _BOOKS),
        Symbol(r"\mathbf{Q}_b", SymbolKind.STATE, "body_{slot}_momentum", "kg m/s; kg m^2/s", _BOOKS),
        Symbol(r"V_\chi", SymbolKind.STATE, "body_{slot}_indicator", "m^3", _BOOKS),
        Symbol(r"\mathbf{S}_b", SymbolKind.STATE, "body_{slot}_share", "N; N m", _STEP),
        Symbol(r"P_b", SymbolKind.STATE, "body_{slot}_power", "W", _STEP),
        Symbol(r"D_b", SymbolKind.STATE, "body_{slot}_dissipated", "J", _STEP),
        Symbol(r"n_b", SymbolKind.STATE, "body_{slot}_gift_steps", "1", _STEP),
    )

    def __init__(self, name: str, timestep: float, *, bodies: Sequence[ResolvedSpec] | int = (),
                 origin: Sequence[float] = (0.0, 0.0, 0.0), **settings: Any) -> None:
        # ``bodies`` is saved as their count, each body's description under ``body_<i>_``: a node made again from its
        # own arguments takes them back that way.
        flat: dict[str, Any] = {}
        if not isinstance(bodies, int):
            for index, spec in enumerate(bodies):
                flat.update(spec.arguments(index))
        count = bodies if isinstance(bodies, int) else len(bodies)
        super().__init__(name, timestep, bodies=int(count), origin=tuple(float(v) for v in origin), **flat,
                         **settings)

    @property
    def count(self) -> int:
        """How many resolved bodies the node holds."""
        return int(self.constants().get("bodies", 0))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        held = super().param_specs()
        fixed = "a structural setting: the body's window was sized with it"
        for index in range(self.count):
            held[f"body_{index}_volume"] = ParamSpec(trainable=False, units="m^3", description=fixed)
        held["origin"] = ParamSpec(trainable=False, units="m", description="the grid's origin less the world's: a "
                                                                             "body's grid position is its own plus it")
        return held

    @override
    def own_inputs(self) -> dict[str, Any]:
        own = super().own_inputs()
        for index in range(self.count):
            own[f"body_{index}_pose"] = node_input("m; 1", (7,), description=f"resolved body {index}'s pose at the step's "
                                                                       f"start: position (m) and unit quaternion")
            own[f"body_{index}_twist"] = node_input("m/s; rad/s", (6,), description=f"resolved body {index}'s twist at the step's "
                                                                        f"start: velocity (m/s) and angular velocity "
                                                                        f"(rad/s)")
        return own

    def bodies(self, p: Mapping[str, Any]) -> list[ResolvedBody]:
        """Each resolved body as the step reads it, its shape's parameters from the step's values."""
        c = self.constants()
        held: list[ResolvedBody] = []
        for index in range(self.count):
            prefix = f"body_{index}_"
            shape = str(c[f"{prefix}shape"])
            parameters = {key: as_run_array(p[f"{prefix}{key}"]) for key in shape_named(shape).parameter_names}
            rotation = jnp.reshape(as_run_array(p[f"{prefix}rotation"]), (3, 3))
            held.append(ResolvedBody(shape, parameters, as_run_array(p[f"{prefix}centre"]), rotation,
                                     int(c[f"{prefix}window"]), as_run_array(p[f"{prefix}volume"])))
        return held

    @override
    def step_fields(self, p: Mapping[str, Any], grid: MACGrid, boundary_inputs: Mapping[str, Array]) -> WallFields:
        """The wall's fields united with each body's at its pose (``body_<i>_pose``), and the penalty's push towards its
        twist (``body_<i>_twist``)."""
        wall = super().step_fields(p, grid, boundary_inputs)
        c = self.constants()
        band, interior = band_named(c["band"]), interior_depth(c["interior"])
        dtype = run_dtype()
        nu = self.kinematic(p)
        origin = as_run_array(p["origin"])
        chi, inverse = list(wall.chi), list(wall.inverse_permeability)
        pushed = [jnp.zeros(grid.cells, dtype=dtype) for _ in range(3)]
        all_faces: list[tuple[BodyFaces, BodyFaces, BodyFaces]] = []
        twists: list[Array] = []
        for index, body in enumerate(self.bodies(p)):
            pose = jnp.asarray(boundary_inputs.get(f"body_{index}_pose", jnp.asarray(REST_POSE)), dtype=dtype)
            twist = jnp.asarray(boundary_inputs.get(f"body_{index}_twist", jnp.zeros(6)), dtype=dtype)
            position, turn = pose_turn(pose)
            faces = tuple(body_faces(grid, component, body, position + origin, turn, band, interior)
                          for component in range(3))
            for component in range(3):
                chi[component] = united(chi[component], faces[component])
                inverse[component] = scattered(inverse[component], faces[component],
                                               faces[component].inverse_permeability)
                push = nu * faces[component].inverse_permeability * rigid_velocity(faces[component], component, twist)
                pushed[component] = scattered(pushed[component], faces[component], push)
            all_faces.append((faces[0], faces[1], faces[2]))
            twists.append(twist)
        given = {item.name: getattr(wall, item.name) for item in dataclasses.fields(wall)}
        given.update(chi=(chi[0], chi[1], chi[2]), inverse_permeability=(inverse[0], inverse[1], inverse[2]))
        return ResolvedFields(**given, faces=tuple(all_faces), twists=tuple(twists), pushed=tuple(pushed))

    @override
    def forcing(self, fields: WallFields, boundary_inputs: Mapping[str, Array], scheme: Any, dtype: Any) -> Any:
        """The drive, and the bodies' penalty's push towards their twists (held over the step)."""
        base = super().forcing(fields, boundary_inputs, scheme, dtype)
        if not isinstance(fields, ResolvedFields) or not fields.pushed:
            return base
        pushed = fields.pushed

        def force(stage: int, time: Any) -> Velocity:
            held = base(stage, time)
            return (held[0] + pushed[0], held[1] + pushed[1], held[2] + pushed[2])

        return force

    def body_books(self, p: Mapping[str, Any], fields: ResolvedFields, velocity: Velocity, index: int) -> \
            tuple[Array, Array, Array]:
        """Body ``index``'s penalty load (force and torque), its interior's momentum and angular momentum, both at
        ``velocity``, and its indicator's volume on the axial faces."""
        grid = self.grid(p)
        rho, nu = as_run_array(p["density"]), self.kinematic(p)
        volume = grid.cell_volume
        twist = fields.twists[index]
        dtype = velocity[0].dtype
        force, torque = jnp.zeros(3, dtype=dtype), jnp.zeros(3, dtype=dtype)
        momentum, angular = jnp.zeros(3, dtype=dtype), jnp.zeros(3, dtype=dtype)
        for component in range(3):
            faces = fields.faces[index][component]
            on = placed(velocity[component], faces)
            density = rho * nu * faces.inverse_permeability * (on - rigid_velocity(faces, component, twist))
            total, moment = moments(faces, component, density, volume)
            force, torque = force.at[component].add(total), torque + moment
            total, moment = moments(faces, component, rho * faces.chi * on, volume)
            momentum, angular = momentum.at[component].add(total), angular + moment
        displaced = volume * jnp.sum(fields.faces[index][2].chi)
        return jnp.concatenate([force, torque]), jnp.concatenate([momentum, angular]), displaced

    @override
    def stepped(self, p: Mapping[str, Any], state: Mapping[str, Array], boundary_inputs: Mapping[str, Array],
                dt: float) -> tuple[dict[str, Array], StepParts]:
        held, parts = super().stepped(p, state, boundary_inputs, dt)
        fields = parts.fields
        if not isinstance(fields, ResolvedFields) or not self.count:
            return held, parts
        before = (state["velocity_x"], state["velocity_y"], state["velocity_z"])
        first = jnp.asarray(state["steps"]) == 0
        inertia = self.scheme.stores_inertia
        rho = as_run_array(p["density"])
        drive = jnp.asarray(boundary_inputs.get("body_force", jnp.zeros(3)), dtype=run_dtype())
        for index in range(self.count):
            penalty, momentum, displaced = self.body_books(p, fields, parts.stage.solved, index)
            started = self.body_books(p, fields, before, index)[1]
            previous = jnp.where(first, started, jnp.asarray(state[f"body_{index}_momentum"]))
            load = penalty + (momentum - previous) / dt if inertia else penalty
            force = rho * as_run_array(p[f"body_{index}_volume"]) * drive
            pose = jnp.asarray(boundary_inputs.get(f"body_{index}_pose", jnp.asarray(REST_POSE)), dtype=run_dtype())
            buoyancy = pose_turn(pose)[1] @ as_run_array(p[f"body_{index}_centre"])
            share = jnp.concatenate([force, jnp.cross(buoyancy, force)])
            # The law's power at one velocity: its load on the body at the twist the field was solved at, with its
            # work on the fluid; minus its relative dissipation, never above the products' rounding.
            _, work, size = self.penalty_work(p, fields, parts.stage.solved, index, dt)
            on_body = dt * penalty * fields.twists[index]
            power = jnp.sum(on_body) + work
            floor = IDENTITY_ULPS * jnp.finfo(power.dtype).eps * (jnp.sum(jnp.abs(on_body)) + size)
            held = {**held, f"body_{index}_load": load, f"body_{index}_penalty": penalty, f"body_{index}_momentum": momentum,
                    f"body_{index}_indicator": displaced, f"body_{index}_share": share, f"body_{index}_power": power / dt,
                    f"body_{index}_dissipated": jnp.asarray(state[f"body_{index}_dissipated"]) - power,
                    f"body_{index}_gift_steps": jnp.asarray(state[f"body_{index}_gift_steps"]) + (power > floor).astype(
                        power.dtype)}
        return held, parts

    @override
    def hosted_laws(self) -> tuple[tuple[str, str, str], ...]:
        """The flow's own laws, and each resolved body's penalty under its drag's id (``body_<i>``), dissipative."""
        c = self.constants()
        return (*super().hosted_laws(),
                *((f"body_{index}", str(c[f"body_{index}_law"]), books.DISSIPATIVE) for index in range(self.count)))

    @override
    def hosted_fields(self) -> tuple[str, ...]:
        return (*super().hosted_fields(), *(f"body_{index}" for index in range(self.count)))

    @override
    def hosted_work(self, energy: Mapping[str, Array]) -> dict[str, Array]:
        held = super().hosted_work(energy)
        held.update({f"body_{index}": energy[f"body_{index}"] for index in range(self.count)})
        return held

    def penalty_work(self, p: Mapping[str, Any], fields: ResolvedFields, solved: Velocity, index: int,
                     dt: float) -> tuple[Array, Array, Array]:
        """Body ``index``'s penalty over the step on the fluid: ``dt rho <nu/K u', u'>`` (its share of the step's
        penalty term), its work ``-dt rho <nu/K (u' - u_b), u'>``, and that work's magnitude."""
        rho, nu = as_run_array(p["density"]), self.kinematic(p)
        volume = self.grid(p).cell_volume
        twist = fields.twists[index]
        own = jnp.zeros((), dtype=solved[0].dtype)
        work = jnp.zeros((), dtype=solved[0].dtype)
        size = jnp.zeros((), dtype=solved[0].dtype)
        for component in range(3):
            faces = fields.faces[index][component]
            on = placed(solved[component], faces)
            sigma = nu * faces.inverse_permeability
            own = own + dt * rho * volume * jnp.sum(sigma * on * on)
            product = sigma * (on - rigid_velocity(faces, component, twist)) * on
            work = work - dt * rho * volume * jnp.sum(product)
            size = size + dt * rho * volume * jnp.sum(jnp.abs(product))
        return own, work, size

    @override
    def energy_terms(self, p: Mapping[str, Any], grid: MACGrid, before: Velocity, stage: Any, sigma: Velocity,
                     fields: WallFields, dt: float) -> dict[str, Array]:
        """The flow's terms with the bodies' penalty apart from the wall's: each body's work on the fluid under
        ``body_<i>``, the wall's penalty the rest of the step's."""
        terms = super().energy_terms(p, grid, before, stage, sigma, fields, dt)
        if not isinstance(fields, ResolvedFields):
            return terms
        penalty, scale = terms["penalty"], terms["scale"]
        for index in range(self.count):
            own, work, size = self.penalty_work(p, fields, stage.solved, index, dt)
            penalty = penalty - own
            terms[f"body_{index}"] = work
            scale = scale + size
        return {**terms, "penalty": penalty, "scale": scale}

    @override
    def initial_fields(self) -> dict[str, Array]:
        held = super().initial_fields()
        z6 = jnp.zeros(6, dtype=run_dtype())
        for index in range(self.count):
            zero = as_run_array(0.0)
            held.update({f"body_{index}_load": z6, f"body_{index}_penalty": z6, f"body_{index}_momentum": z6,
                         f"body_{index}_indicator": zero, f"body_{index}_share": z6, f"body_{index}_power": zero,
                         f"body_{index}_dissipated": zero, f"body_{index}_gift_steps": zero})
        return held


__all__ = ["BODY_FIELDS", "BODY_INPUTS", "REST_POSE", "ResolvedFields", "ResolvedFlowInputs", "ResolvedFlowNode",
           "ResolvedFlowState", "ResolvedSpec"]
