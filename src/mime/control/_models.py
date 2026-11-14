"""A controller's models of what it drives and observes, as its law reads them through its context: ``ctx.believed``
and ``ctx.allocator``.

**Believed models are declared data.** ``set_controller(..., believed={key: declaration})`` gives the controller its
own declaration of an apparatus's provider or chain segment (``<apparatus>.<field>``, ``<apparatus>.<segment>``), or of
a camera (its id): an entry of the same kind, taking the same commands, with its own settings (a calibration, a
placement, a camera's intrinsics). Where none is given the belief is the true declaration, and the build report says
``believed = true``. No node is built for a belief. ``ctx.believed(key)`` gives a provider's or a segment's forward model
(``mime.registry.ForwardModel``), its parameters bound to the belief's numbers, or a camera's projection
(:class:`CameraModel`); inside the compiled step each number the law's step reads is a leaf of the controller's node,
``believed_<apparatus>__<part>__<parameter>``, so that a derivative of the run reaches the belief and a fit moves it,
never the truth.

**The allocation.** ``ctx.allocator(apparatus, commands=..., objective=..., owner=...)`` is the exact allocation
(:func:`mime.control.exact_allocation`) through the believed model, its bounds the commands' ports' (:class:`Allocator`)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import jax
import numpy as np

from mime.build._control import BeliefPlan, believed_key
from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.control._allocation import EXACT_COMMANDS, exact_allocation
from mime.core import as_run_array
from mime.nodes.control import UNBOUNDED
from mime.registry import ChainSegment, FieldProvider, ForwardModel
from mime.units import Quantity, UnitError, parse_unit, si

if TYPE_CHECKING:
    from mime.build import PlanSet
    from mime.build._plan import EntryPlan
    from mime.build._ports import CommandPortPlan

jnp: Any = jax.numpy

#: The prefix of a belief's number among the names a controller's law is made again from inside the compiled step:
#: ``believed:<key>:<parameter>``.
BELIEVED = "believed:"
#: What an allocation regularises towards: zero (least effort) or the last command (smooth).
TOWARDS = ("zero", "last")


def leaf_name(key: str, parameter: str) -> str:
    """The name a belief's number is made again under: ``believed:<key>:<parameter>``."""
    return f"{BELIEVED}{key}:{parameter}"


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> ArgumentRefused:
    """A refusal of a controller's call on its context, raised inside its ``law(ctx)``, which keeps its code."""
    return ArgumentRefused([Refusal(code, category, message, action)])


@dataclass(frozen=True)
class CameraModel:
    """A controller's model of a camera: where world points fall on its detector (:meth:`project`), from its nine
    numbers (``fx``, ``fy``, ``cx``, ``cy``, ``k1``, ``k2``, ``p1``, ``p2``, ``k3``) and, for a camera fixed in the world,
    its ``position`` and ``orientation``, each in ``theta``; ``width``, ``height`` and where it is mounted are its
    structure. The camera's own ``Camera.project``, with its numbers as the controller's."""

    owner: str
    width: int
    height: int
    source: str
    world: bool
    theta: Mapping[str, Any]

    def project(self, points: Any, *, pose: tuple[Any, Any] | None = None) -> tuple[Any, Any]:
        """Where world points ``(..., 3)`` (m) fall on the detector, ``(..., 2)`` pixels ``(u, v)``, and their depths
        along its optical axis: ``pose`` is its optical frame's position and unit quaternion in the world, its own for
        a camera fixed in the world where none is given."""
        from mime.sensing import BrownConrady, Pinhole
        from mime.sensing._detector import CameraPose
        from mime.sensing._detector import project as projected

        held = self.theta
        if pose is None:
            if not self.world:
                raise TypeError(f"the camera {self.owner!r} is mounted on a frame, which moves with it: give "
                                f"project(..., pose=(position, orientation)), its optical frame in the world")
            pose = (held["position"], held["orientation"])
        pinhole = Pinhole(self.width, self.height, held["fx"], held["fy"], held["cx"], held["cy"],
                          BrownConrady(held["k1"], held["k2"], held["p1"], held["p2"], held["k3"], source=self.source))
        return projected(pinhole, CameraPose(pose[0], pose[1]), points)

    def bound(self, theta: Mapping[str, Any]) -> CameraModel:
        return dataclasses.replace(self, theta=MappingProxyType({**self.theta, **theta}))


def camera_model(owner: str, camera: Any) -> CameraModel:
    """The model of the camera ``camera`` (a ``mime.observations.Camera``) a controller reads: its declared numbers."""
    from mime.observations import World
    from mime.sensing import Pinhole

    detector = camera.detector
    pinhole = detector if isinstance(detector, Pinhole) else detector.pinhole()
    distortion = pinhole.distortion
    numbers = {"fx": pinhole.fx, "fy": pinhole.fy, "cx": pinhole.cx, "cy": pinhole.cy, "k1": distortion.k1,
               "k2": distortion.k2, "p1": distortion.p1, "p2": distortion.p2, "k3": distortion.k3}
    theta = {name: np.asarray(float(value), dtype=np.float64) for name, value in numbers.items()}
    world = isinstance(camera.mount, World)
    if world:
        theta["position"] = np.asarray(si(camera.mount.position, parse_unit("m"), what="the camera's position"),
                                       dtype=np.float64)
        theta["orientation"] = np.asarray(camera.mount.orientation, dtype=np.float64)
    return CameraModel(owner, int(pinhole.width), int(pinhole.height), str(distortion.source), world,
                       MappingProxyType(theta))


type Model = ForwardModel | CameraModel


class AllocationReport(NamedTuple):
    """What an allocation gives beside its commands: each command's place in the selected active set (``active``, by
    command, ``-1`` at its lower bound, ``0`` free, ``+1`` at its upper one), the minimised sum of scaled squares
    (``objective``), and what the commands make by the believed model, each quantity of the objective (``made``)."""

    active: Mapping[str, Any]
    objective: Any
    made: Mapping[str, Any]


def _slew_steps(controller: int, port: int) -> float:
    """The base steps of slew ``g`` the allocation's box gives one write, for a controller writing every ``P =
    controller`` base steps through a port sampling every ``R = port``: ``g = R * floor(P / R)`` for ``R <= P``, else
    ``R / ceil(R / P)``. It is the least time the stage is sure to have between two changes of what it is driving
    towards, per write that change can hold: with the port at the controller's period or a divisor of it, every write
    is sampled and the next arrives at least ``R * floor(P / R)`` steps later; with the port slower, a sample can take
    up to ``ceil(R / P)`` writes' moves at once, with ``R`` steps to make them. Either way it is ``P`` where one period
    divides the other. A latency, the controller's or the port's, delays every write alike and changes neither.

    **The assumption.** The box is ``last +/- slew * dt * g``: it takes the stage to have stood at ``last`` when the
    write was made, which holds by induction only if every earlier command was deliverable, each reached before the next
    arrived. It does not hold on the first tick from rest where the law's initial ``last`` is not the port's ``rest``,
    nor after a trip latches the port to its ``rest``: there the stage is somewhere else, and the commands asked for are
    more than it delivers. A trip is flagged by the run (``tripped_<command>``); a delivery limited by the slew is not,
    and is read from the port's recorded traffic (``written_<command>`` against the command)."""
    if port <= controller:
        return float(port * (controller // port))
    return port / math.ceil(port / controller)


@dataclass(frozen=True)
class Allocator:
    """The exact allocation over ``commands`` of a believed model affine in them: ``solve(wanted, at, last=None) ->
    (commands, report)``, traceable, the same function in every mode. ``wanted`` holds a value of each quantity of the
    objective, ``at`` the model's inputs (a world point; a moment, for a force or torque), ``last`` the commands the
    controller wrote last (by command), required where a port declares a slew or the objective is smooth.

    The rows are each quantity's independent entries over its scale (``scales``, SI, by quantity), the model's
    coefficients in the commands and its offset at zero command taken at ``at``. The bounds are the ports' (``low``,
    ``high``, SI, by command), intersected where a port declares a slew with ``last`` plus or minus ``slew`` times
    ``dt`` times the steps :func:`_slew_steps` gives (``box``, by command, the half-width in SI). ``effort`` is the
    regularisation's scale by command, towards zero or ``last`` (``toward``); without it the tie among several
    minimisers goes to the commands nearest that reference. ``owner`` is the switch's owner, which the controller
    declares (``allocation_switch(owner, orders)``)."""

    model: ForwardModel
    commands: tuple[str, ...]
    scales: Mapping[str, float]
    low: Mapping[str, Any]
    high: Mapping[str, Any]
    box: Mapping[str, Any]
    effort: Mapping[str, Any] | None
    toward: str
    owner: str

    def __call__(self, wanted: Mapping[str, Any], at: Mapping[str, Any],
                 last: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], AllocationReport]:
        if set(wanted) != set(self.scales):
            raise ValueError(f"the allocation of {', '.join(self.commands)} is asked for {sorted(wanted)}, and its "
                             f"objective is {sorted(self.scales)}: give a wanted value of each")
        if (self.box or self.toward == "last") and last is None:
            raise ValueError(f"the allocation of {', '.join(self.commands)} moves from the last command (a port's "
                             f"slew, or toward='last'): give solve(..., last={{command: value}})")
        dtype = as_run_array(0.0).dtype
        sizes = [self.model.commands[c].shape[0] for c in self.commands]
        coefficients, offset = self.model.affine(at, commands=self.commands, dtype=dtype)
        rows: list[Any] = []
        target: list[Any] = []
        zero: list[Any] = []
        scale: list[Any] = []
        for quantity in sorted(self.scales):
            block = [jnp.reshape(self.model.row_values(quantity, coefficients[quantity][c]), (-1, size))
                     for c, size in zip(self.commands, sizes, strict=True)]
            rows.append(jnp.concatenate(block, axis=1))
            target.append(self.model.row_values(quantity, jnp.asarray(wanted[quantity], dtype=dtype)))
            zero.append(self.model.row_values(quantity, offset[quantity]))
            scale.append(jnp.full((rows[-1].shape[0],), self.scales[quantity], dtype=dtype))

        def joined(by: Mapping[str, Any]) -> Any:
            return jnp.concatenate([jnp.broadcast_to(jnp.asarray(by[c], dtype=dtype), (size,))
                                    for c, size in zip(self.commands, sizes, strict=True)])

        lower, upper = joined(self.low), joined(self.high)
        before = joined(last) if last is not None else None
        if self.box and before is not None:
            half = joined({c: self.box.get(c, jnp.inf) for c in self.commands})
            lower, upper = jnp.clip(before - half, lower, upper), jnp.clip(before + half, lower, upper)
        reference = before if self.toward == "last" else None
        found = exact_allocation(jnp.concatenate(rows), jnp.concatenate(target), lower=lower, upper=upper,
                                 scales=jnp.concatenate(scale), owner=self.owner, offset=jnp.concatenate(zero),
                                 reference=reference,
                                 reference_scales=joined(self.effort) if self.effort is not None else None)
        split = np.cumsum([0, *sizes])
        commands = {c: found.commands[split[i]:split[i + 1]] for i, c in enumerate(self.commands)}
        active = {c: found.active[split[i]:split[i + 1]] for i, c in enumerate(self.commands)}
        made = self.model(commands, at)
        return commands, AllocationReport(active, found.objective, {q: made[q] for q in sorted(self.scales)})


class ControlModels:
    """What a controller's context holds of its models (``ctx.believed``, ``ctx.allocator``), made from the plans:
    each provider's, segment's and camera's declaration, true or believed, with ``given`` numbers in place of a
    belief's where its law is made again from the controller's leaves (``believed:<key>:<parameter>``). ``read``
    records, in order, each key whose model the law read."""

    def __init__(self, plans: PlanSet, given: Mapping[str, Any] | None = None) -> None:
        self._plans = plans
        self._given = dict(given or {})
        self.read: list[str] = []
        #: One report line per command whose allocation is boxed by a port's slew (:func:`_slew_steps`).
        self.slewed: list[str] = []

    def _parts(self) -> dict[str, EntryPlan]:
        from mime.build._plan import Section

        return {plan.owner: plan for plan in self._plans.entries
                if (plan.section is Section.PROVIDER and isinstance(plan.entry, FieldProvider))
                or (plan.section is Section.DYNAMICS and isinstance(plan.entry, ChainSegment))
                or plan.section is Section.OBSERVATION}

    def key(self, written: str) -> str:
        """``written`` as its full key, or ``controller.believed`` naming the keys there are."""
        parts = self._parts()
        apparatus: dict[str, list[str]] = {}
        observations: list[str] = []
        for owner, plan in parts.items():
            if plan.thing is not None and owner.startswith(f"{plan.thing}."):
                apparatus.setdefault(plan.thing, []).append(owner.partition(".")[2])
            else:
                observations.append(owner)
        key, problem = believed_key(written, {name: tuple(held) for name, held in apparatus.items()}, observations)
        if key is None or key not in parts:
            raise _refuse("controller.believed", RefusalClass.USER,
                          f"ctx.believed({written!r}) {problem or 'names no model'}",
                           f"name one in full: ctx.believed(\"<key>\"), of {sorted(parts) or 'none declared'}")
        return key

    def _declared(self, key: str) -> BeliefPlan | None:
        controller = self._plans.controller
        return next((belief for belief in controller.beliefs if belief.key == key), None) if controller else None

    def model(self, key: str) -> Model:
        """The model of ``key`` (a full key) at its declared numbers, the belief's where one is declared."""
        from mime.build._construct import entry_context

        plan = self._parts()[key]
        belief = self._declared(key)
        entry = belief.entry if belief is not None else plan.entry
        if not isinstance(entry, (FieldProvider, ChainSegment)):
            from mime.observations import Camera

            if not isinstance(entry, Camera):
                raise _refuse("controller.believed", RefusalClass.USER,
                              f"ctx.believed({key!r}) names the observation {key!r}, a {type(entry).__name__}, and "
                               f"a controller models a camera among observations",
                               "read a camera's model, ctx.believed(\"<camera id>\"), or an apparatus part's, "
                               "ctx.believed(\"<apparatus>.<field>\")")
            return camera_model(key, entry)
        context = entry_context(plan, self._plans)
        if belief is not None:
            context = dataclasses.replace(context, settings=dict(belief.settings), entry=belief.entry)
        made = entry.forward_model(context)
        if made is None:
            raise _refuse("controller.believed", RefusalClass.USER,
                          f"ctx.believed({key!r}): {type(entry).__name__} declares no forward model",
                           "read the model of a part whose entry declares one, ctx.believed(\"<apparatus>.<field>\") "
                           "(mime.registry.ForwardModel)")
        return made

    def believed(self, written: str) -> Model:
        key = self.key(written)
        made = self.model(key)
        given = {name: self._given[leaf_name(key, name)] for name in made.theta if leaf_name(key, name) in self._given}
        if key not in self.read:
            self.read.append(key)
        return made.bound(given) if given else made

    def leaves(self, keys: Sequence[str]) -> dict[str, Any]:
        """Each number of the models of ``keys`` at its declared value, by the name the law is made again under."""
        return {leaf_name(key, name): np.asarray(value) for key in keys for name, value in self.model(key).theta.items()}

    def allocator(self, apparatus: str, *, commands: Sequence[str], objective: Mapping[str, Quantity], owner: str,
                  effort: Quantity | Mapping[str, Quantity] | None = None, toward: str = "zero") -> Allocator:
        held = tuple(str(c) for c in commands)
        where = f"ctx.allocator({apparatus!r}, commands={list(held)})"
        parts = self._parts()
        owners = {c: self._owner_of(apparatus, c, parts) for c in held}
        if not held or None in owners.values() or len(set(owners.values())) != 1:
            raise _refuse("allocation.model", RefusalClass.USER,
                          f"{where}: the commands name no one provider or segment of the apparatus {apparatus!r}, "
                           f"and an allocation in this release is through one model",
                           f"allocate the commands of one of its parts, ctx.allocator({apparatus!r}, commands=(...)): "
                           f"{self._commands_of(apparatus, parts)}")
        key = cast(str, next(iter(owners.values())))
        model = self.believed(key)
        if not isinstance(model, ForwardModel):
            raise _refuse("allocation.model", RefusalClass.USER,
                          f"{where}: {key} is a camera", "allocate an apparatus's commands, ctx.allocator(\"<apparatus>\", "
                           "commands=(...))")
        local = tuple(c.partition(".")[2] if "." in c else c for c in held)
        beyond = [c for c in local if c not in model.affine_in]
        if beyond:
            raise _refuse("allocation.model", RefusalClass.USER,
                          f"{where}: {key}'s model is not affine in {', '.join(beyond)} (it is affine in "
                           f"{sorted(model.affine_in) or 'none of its commands'}), so its allocation is not exact",
                           "allocate commands a model is affine in, ctx.allocator(\"<apparatus>\", commands=(\"currents\",)); "
                           "the iterative path, which takes any model, arrives in a later 0.3.0 development release")
        count = sum(model.commands[c].shape[0] for c in local)
        if count > EXACT_COMMANDS:
            raise _refuse("allocation.exact-size", RefusalClass.USER,
                          f"{where}: {count} commands, and the exact path takes at most EXACT_COMMANDS = "
                           f"{EXACT_COMMANDS} ({3 ** EXACT_COMMANDS} active sets)",
                           "allocate fewer commands at once, ctx.allocator(..., commands=(...)); or wait for the iterative "
                           "path, which takes any number (a later 0.3.0 development release)")
        scales = self._scales(objective, model, where)
        if toward not in TOWARDS:
            raise _refuse("allocation.objective", RefusalClass.USER,
                          f"{where}: toward={toward!r}", f"write ctx.allocator(..., toward=...) as one of {TOWARDS}")
        low, high, box = self._bounds(apparatus, held, local, where)
        efforts = None if effort is None else self._efforts(effort, local, model, where)
        return Allocator(model, local, scales, low, high, box, efforts, toward, owner)

    def _owner_of(self, apparatus: str, command: str, parts: Mapping[str, EntryPlan]) -> str | None:
        head, _, tail = command.partition(".")
        if tail:
            owner = f"{apparatus}.{head}"
            plan = parts.get(owner)
            return owner if plan is not None and plan.segment is not None and tail in plan.segment.commands else None
        found = [owner for owner, plan in parts.items() if plan.thing == apparatus and
                 isinstance(plan.entry, FieldProvider) and command in type(plan.entry).commands]
        return found[0] if len(found) == 1 else None

    def _commands_of(self, apparatus: str, parts: Mapping[str, EntryPlan]) -> list[str]:
        found: list[str] = []
        for owner, plan in sorted(parts.items()):
            if plan.thing != apparatus:
                continue
            if isinstance(plan.entry, FieldProvider):
                found += sorted(type(plan.entry).commands)
            elif plan.segment is not None:
                found += [f"{owner.partition('.')[2]}.{c}" for c in sorted(plan.segment.commands)]
        return found

    def _scales(self, objective: Mapping[str, Quantity], model: ForwardModel, where: str) -> dict[str, float]:
        if not isinstance(cast(object, objective), Mapping) or not objective:
            raise _refuse("allocation.objective", RefusalClass.USER,
                          f"{where}: objective={objective!r} is not a mapping of quantities to scales",
                           f"give ctx.allocator(..., objective={{quantity: scale}}), of {sorted(model.quantities)}")
        scales: dict[str, float] = {}
        for quantity, scale in objective.items():
            signal = model.quantities.get(quantity)
            if signal is None:
                raise _refuse("allocation.objective", RefusalClass.USER,
                              f"{where}: the model of {model.owner} gives no {quantity!r}",
                               f"name its quantities in ctx.allocator(..., objective={{...}}): "
                               f"{sorted(model.quantities)}")
            try:
                value = si(scale, parse_unit(signal.units), what=f"the scale of {quantity}")
            except (UnitError, TypeError) as error:
                raise _refuse("allocation.objective", RefusalClass.USER,
                              f"{where}: {error}", f"give objective={{{quantity!r}: <a quantity in {signal.units}>}}") from None
            if not (isinstance(value, float) and math.isfinite(value) and value > 0.0):
                raise _refuse("allocation.objective", RefusalClass.USER,
                              f"{where}: the scale of {quantity} is {scale}, not one value above zero",
                               f"give objective={{{quantity!r}: <one value above zero>}}: the size at which a miss in "
                               f"its rows counts as much as one in the others'")
            scales[quantity] = value
        return scales

    def _bounds(self, apparatus: str, written: Sequence[str], local: Sequence[str],
                where: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        ports: dict[str, CommandPortPlan] = {p.command: p for p in self._plans.command_ports if p.apparatus == apparatus}
        controller = self._plans.controller
        period = controller.period if controller is not None else 1
        low: dict[str, Any] = {}
        high: dict[str, Any] = {}
        box: dict[str, Any] = {}
        missing: list[str] = []
        for path, command in zip(written, local, strict=True):
            port = ports.get(path)
            if port is not None and port.bounded != UNBOUNDED and port.low is not None and port.high is not None:
                low[command], high[command] = np.asarray(port.low), np.asarray(port.high)
            else:
                stated = self._stated(apparatus, path, command)
                if stated is None:
                    missing.append(path)
                    continue
                low[command], high[command] = np.asarray(stated[0]), np.asarray(stated[1])
            if port is not None and port.slew is not None:
                steps = _slew_steps(period, port.period)
                box[command] = np.asarray(port.slew) * self._plans.dt * steps
                self.slewed.append(f"    {path}: last +/- {np.asarray(box[command]).tolist()} per write ({steps:g} base "
                                   f"steps of the port's slew; the stage taken to stand at the last command)")
        if missing:
            raise _refuse("allocation.bounds", RefusalClass.USER,
                          f"{where}: {', '.join(missing)} has no bounds: its port declares none and its receiver "
                           f"states none, and the exact allocation chooses within bounds",
                           f"declare the hardware's bounds on its port, Apparatus({apparatus!r}, ports={{"
                           f"{missing[0]!r}: CommandPort(bounds=..., reason=...)}})")
        return low, high, box

    def _stated(self, apparatus: str, path: str, command: str) -> tuple[Any, Any] | None:
        head, _, tail = path.partition(".")
        for plan in self._plans.entries:
            if tail and plan.owner == f"{apparatus}.{head}" and plan.segment is not None:
                held = plan.segment.commands.get(tail)
                return held.bounds if held is not None else None
            if not tail and plan.thing == apparatus and isinstance(plan.entry, FieldProvider):
                held = type(plan.entry).commands.get(command)
                if held is not None:
                    return held.bounds
        return None

    def _efforts(self, effort: Quantity | Mapping[str, Quantity], local: Sequence[str], model: ForwardModel,
                 where: str) -> dict[str, Any]:
        given = effort if isinstance(effort, Mapping) else dict.fromkeys(local, effort)
        found: dict[str, Any] = {}
        for command in local:
            held = given.get(command)
            signal = model.commands[command]
            try:
                value = si(held, parse_unit(signal.units), what=f"the effort scale of {command}")
            except (UnitError, TypeError) as error:
                raise _refuse("allocation.objective", RefusalClass.USER,
                              f"{where}: {error}", f"give ctx.allocator(..., effort=<a quantity in {signal.units}>)") from None
            values = np.asarray(value, dtype=np.float64)
            if not (np.all(np.isfinite(values)) and np.all(values > 0.0)):
                raise _refuse("allocation.objective", RefusalClass.USER,
                              f"{where}: the effort scale of {command} is {held}",
                               "give ctx.allocator(..., effort=<a quantity above zero>), one value or one per element")
            found[command] = values
        return found


__all__ = ["BELIEVED", "TOWARDS", "AllocationReport", "Allocator", "CameraModel", "ControlModels", "camera_model",
           "leaf_name"]
