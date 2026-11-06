"""The containment watch: after a run, whether a body held by its centre (the contact 0.1.0) kept its hull inside the
region it is held in.

That contact holds the body's centre by its extent and never reads its orientation, so a body that turns across a
region narrower than it is long reaches a pose its hull cannot take there, and the run goes on with the body through
the wall, unflagged by the contact. The watch reads the body's pose after the run (at every step the run kept, else the
last step's alone) and the surface's signed distance at the hull's points (``mime.build._watch.ContainmentPlan``): the
largest, over the steps and the points, is how far the hull passed the surface. Like the Reynolds watch it adds no node
and no edge, so the graph and its derived hash are what they would be without it, and it raises a flag on the result in
every profile, never a refusal: a run that went through the wall is a result the experiment's author may want to read
(the flag says where), not one MIME may discard.

What it bounds and what it estimates. The depth is the depth of the hull's samples, which fall short of the body by up
to the plan's ``shortfall``: the body may be that much deeper than the depth read (the watch cannot report a hull inside
that is outside, by more than that). The flag rises past rounding alone (``ROUNDING_ULPS`` units in the last place of
the run's precision at the scale of the hull and its position), not past the contact's margin: a hull that comes within
the margin of the wall without passing it is held as the contact's extent leaves it, and is not the mistake."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np
from numpy.typing import NDArray

from mime.build._watch import ContainmentPlan
from mime.core import jnp, shape_named

#: The units in the last place of the run's precision, at the scale of the hull and the body's position, that the
#: hull may pass the surface by before the flag rises: the rounding of the poses and of the distances read from them.
ROUNDING_ULPS = 16
#: The most numbers (steps times points times three) one block of the evaluation holds.
BLOCK = 20_000_000

WITHIN, PAST, UNKNOWN = "within", "past_the_wall", "not_computed"

type Final = Mapping[str, Mapping[str, NDArray[Any]]]


@dataclass(frozen=True)
class ContainmentRead:
    """One body's containment in a run: ``depth`` is the largest signed distance of the hull's points from the surface
    over the steps read (m, positive past the surface, negative inside it), ``first_step`` the first step at which it
    was past the surface by more than ``tolerance`` and ``worst_step`` the step of the largest depth (steps from the
    graph's start, as the run counts them), ``read`` how many steps were read, and ``peak`` whether they were every
    step the run kept (else the last step's alone, which says nothing of the steps before it). ``status`` is ``past_the_wall``
    (raised), ``within`` or ``not_computed`` (``missing`` says why)."""

    owner: str
    entry: str
    body: str
    surface: str
    status: str
    depth: float | None
    tolerance: float | None
    first_step: int | None
    worst_step: int | None
    read: int
    peak: bool
    margin: float
    shortfall: float | None
    dt: float
    missing: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        return f"containment:{self.owner}"

    @property
    def raised(self) -> bool:
        return self.status == PAST

    def text(self) -> str:
        what = f"{self.path} ({self.entry})"
        if self.status == UNKNOWN:
            return f"{what}: not computed, {'; '.join(self.missing)}"
        assert self.depth is not None and self.worst_step is not None
        scope = (f"the {self.read:,} steps the run kept" if self.peak else
                 "the last step only, which says nothing of the steps before it")
        short = "" if self.shortfall is None else (
            f"; the hull's samples fall up to {self.shortfall * 1e6:.3g} um short of the body, which may be that much "
            f"deeper")
        if self.status == WITHIN:
            return (f"{what}: {self.body}'s hull stayed inside {self.surface}, its deepest point "
                    f"{-self.depth * 1e3:.4g} mm from it over {scope}{short}")
        assert self.first_step is not None
        where = (f"at step {self.worst_step:,}, t = {self.worst_step * self.dt:.4g} s; first past it at step "
                 f"{self.first_step:,}, t = {self.first_step * self.dt:.4g} s" if self.peak else
                 f"at the last step, {self.worst_step:,}, t = {self.worst_step * self.dt:.4g} s")
        return (f"{what}: {self.body}'s hull passed {self.surface} by {self.depth * 1e3:.3g} mm ({where}), over "
                f"{scope}{short}. {self.entry} holds "
                f"{self.body}'s centre by its extent and never reads its orientation, so a body that turns across "
                f"the region reaches a pose its hull cannot take there, and this run has {self.body} through the "
                f"wall. Write Contact.Projection(margin=...), which holds the hull, its orientation read")

    def manifest(self) -> dict[str, Any]:
        return {"watch": self.path, "kind": "containment", "entry": self.entry, "owner": self.owner,
                "body": self.body, "surface": self.surface, "status": self.status, "raised": self.raised,
                "depth": self.depth, "tolerance": self.tolerance, "first_step": self.first_step,
                "worst_step": self.worst_step, "steps_read": self.read, "peak_over_run": self.peak,
                "margin": self.margin, "shortfall": self.shortfall, "dt": self.dt, "missing": list(self.missing),
                "text": self.text()}

    @classmethod
    def from_manifest(cls, held: Mapping[str, Any]) -> ContainmentRead:
        def number(name: str) -> float | None:
            value = held.get(name)
            return float(value) if value is not None else None

        def step(name: str) -> int | None:
            value = held.get(name)
            return int(value) if value is not None else None

        return cls(str(held["owner"]), str(held["entry"]), str(held["body"]), str(held["surface"]),
                   str(held["status"]), number("depth"), number("tolerance"), step("first_step"), step("worst_step"),
                   int(held["steps_read"]), bool(held.get("peak_over_run")), float(held["margin"]),
                   number("shortfall"), float(held["dt"]),
                   tuple(str(m) for m in cast(Sequence[object], held.get("missing", ()))))


def rotations(quaternions: NDArray[np.float64]) -> NDArray[np.float64]:
    """The rotation matrices ``(n, 3, 3)`` of unit quaternions ``(n, 4)``, ``(w, x, y, z)``."""
    w, x, y, z = (quaternions[:, i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def depths(plan: ContainmentPlan, poses: NDArray[np.float64]) -> NDArray[np.float64]:
    """The largest signed distance from the plan's surface, over the hull's points, at each of ``poses`` (``n x 7``:
    position, then the quaternion ``(w, x, y, z)``), in float64 whatever the run's precision (m, positive past the
    surface)."""
    shape = shape_named(plan.shape)
    vertices = plan.vertices
    block = max(1, BLOCK // (3 * len(vertices)))
    found = np.empty(len(poses), dtype=np.float64)
    with cast(Any, jax).enable_x64(True):
        parameters = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                      for name, value in plan.parameters.items()}
        for start in range(0, len(poses), block):
            held = poses[start:start + block]
            world = held[:, None, :3] + np.einsum("nij,kj->nki", rotations(held[:, 3:]), vertices)
            found[start:start + block] = np.asarray(
                jnp.max(shape.signed_distance(parameters, jnp.asarray(world, dtype=jnp.float64)), axis=-1))
    return found


def _unknown(plan: ContainmentPlan, dt: float, missing: tuple[str, ...]) -> ContainmentRead:
    return ContainmentRead(plan.owner, plan.entry, plan.body, plan.surface, UNKNOWN, None, None, None, None, 0, False,
                           plan.margin, None, dt, missing)


def read_containment(plan: ContainmentPlan, final: Final, history: Final | None, *, last_step: int, dt: float) -> \
        ContainmentRead:
    """``plan``'s watch in a run whose state at its end was ``final`` (after step ``last_step``) and which kept
    ``history`` (the state after each of the steps up to it, or ``None``)."""
    try:
        if history is not None and plan.node in history and plan.field in history[plan.node]:
            held = np.asarray(history[plan.node][plan.field])
            poses, peak = held.reshape(-1, 7), True
        else:
            held = np.asarray(final[plan.node][plan.field])
            poses, peak = held.reshape(1, 7), False
    except (KeyError, ValueError):
        return _unknown(plan, dt, (f"the body's pose ({plan.node}.{plan.field}) is not in the run's state",))
    precision = np.finfo(held.dtype if np.issubdtype(held.dtype, np.floating) else np.float64).eps
    poses = poses.astype(np.float64)
    if not np.isfinite(poses).all():
        return _unknown(plan, dt, ("the body's pose is not finite",))
    first_step = last_step - len(poses) + 1
    try:
        depth = depths(plan, poses)
    except Exception as error:  # noqa: BLE001 - a watch reports what it could not read, never refuses a run
        return _unknown(plan, dt, (f"the surface's signed distance could not be read at the hull's points: {error!r}",))
    scale = float(np.abs(poses[:, :3]).max()) + float(np.linalg.norm(plan.vertices, axis=1).max())
    tolerance = ROUNDING_ULPS * float(precision) * scale
    past = np.flatnonzero(depth > tolerance)
    worst = int(np.argmax(depth))
    status = PAST if len(past) else WITHIN
    return ContainmentRead(plan.owner, plan.entry, plan.body, plan.surface, status, float(depth[worst]), tolerance,
                           first_step + int(past[0]) if len(past) else None, first_step + worst, len(poses), peak,
                           plan.margin, plan.shortfall, dt)


def read_containments(plans: Sequence[Any], final: Final, history: Final | None, *, last_step: int,
                      dt: float) -> tuple[ContainmentRead, ...]:
    """Every containment watch among ``plans`` (the others are the Reynolds watch's)."""
    return tuple(read_containment(plan, final, history, last_step=last_step, dt=dt)
                 for plan in plans if isinstance(plan, ContainmentPlan))


__all__ = ["PAST", "UNKNOWN", "WITHIN", "ContainmentRead", "depths", "read_containment", "read_containments"]
