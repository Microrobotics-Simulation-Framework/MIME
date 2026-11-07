"""How ``verify`` judges a saved field where it claims no bits: in another environment, or from a reference no run
repeats bit for bit (one made on a GPU with XLA's deterministic mode off). From the tenth saved form, a field is within
tolerance when its largest difference over the reference steps is at most the lock's ``rtol`` times its **source
scale**, plus its ``atol``, which the tenth form writes as zero: no absolute part. The source scale is the field's own
largest magnitude over the reference, or the one its type gives (``mime.core.FieldKind``, read from the node's typed
state, never declared per node):

- a **twist** (``Twist``, [v; w]) by the speed of its body's surface points: per step ``|dv| + L |dw|`` against the
  reference's largest ``|v| + L |w|``, L the body's largest reach from its hull (the largest of its hull kind's reach
  over directions, from its parts), or, where it has no hull, its radius of gyration about its widest axis from the
  mass and inertia its node holds (the lock records both). L is computed, never declared. The angular part of a body
  that barely turns is rounding judged against itself otherwise: across CPU counts the tube demo's twist differs by
  1.9e-3 of its own largest magnitude, all of it in w (2e-6 rad/s against |w| 3.7e-4), and by 2.3e-7 of its surface
  points' speed;
- a **ratio** (``Ratio``, a share of the quantity it measures, as the MRS drag's asymmetry ``|R - R^T| / |R|`` is)
  against one, since it is already relative to that quantity: against itself it differs by 1.3e-3 across CPU counts,
  and by 1.25 % between a T4 and its CPU, where its absolute difference is 2.4e-7;
- **held points** (``HeldPoints``) only on the rows whose paired force (``HeldForces``) acts, in the reference or the
  rerun: a row that takes no force holds wherever its search ended, which one unit in the last place moves by
  millimetres (MIME-ANO-144; 32 % of its own scale between a T4 and its CPU).

A folder of the ninth form or earlier is judged as its form says, element by element: ``|a - b| <= atol + rtol |b|``
with its own numbers (``numpy.allclose``)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from mime.core import FieldKind, field_kinds, paired_forces
from mime.saved._versions import SOURCE_SCALE_ADDED

#: The first saved form whose ``reference.rtol`` is over each field's source scale (and whose ``atol`` is zero); a form
#: before it is judged element by element.
SCALED_FROM = SOURCE_SCALE_ADDED

#: float32: each field's largest difference at most 1e-5 of its source scale. Design #263, measured 2026-11-07, the tube
#: demo's eight reference steps: the largest judged spread 1.07e-6 (its least eigenvalue, a T4 against itself and against
#: its CPU, jax 0.11.1) and 2.4e-7 across 24, 4 and 2 usable CPUs (an i7-12850HX, jax 0.12.0.1): 9 and 42 times
#: below it.
FLOAT32_RTOL = 1e-5
#: float64: 2e-14 of the source scale. Design #263, measured 2026-11-07: 5.5e-16 across 24, 4 and 2 usable CPUs (the same
#: machine and demo), 36 times below it. It is the float32 tolerance times the ratio of the two resolutions (2^-29), and
#: rests on the spread being rounding, which scales with the resolution: the CPU's float64 spread over its float32 spread,
#: 2.3e-9, is near that ratio, 1.9e-9. No GPU was measured in float64.
FLOAT64_RTOL = 2e-14
#: The tenth form's absolute part: none.
SCALED_ATOL = 0.0


def tolerance_for(x64: bool) -> float:
    """The ``rtol`` a save writes for its precision."""
    return FLOAT64_RTOL if x64 else FLOAT32_RTOL


@dataclass(frozen=True)
class SourceScale:
    """What one saved field is judged against: its kind; a twist's size ``L`` (m) and where it came from, in words; and
    the label (``node.field``) of the forces held points pair with."""

    kind: FieldKind = FieldKind.OWN
    size: float | None = None
    size_from: str = ""
    forces: str | None = None


#: The scale of a field whose type gives none, and of every field judged under a form before the tenth.
OWN = SourceScale()


@dataclass(frozen=True)
class Judged:
    """A field's comparison: its largest difference over its source scale (``ratio``), the scale, the first step whose
    difference passes the tolerance (``None`` where none does), and the scale in words."""

    ratio: float
    scale: float
    first: int | None
    words: str

    @property
    def within(self) -> bool:
        return self.first is None


def _steps(array: np.ndarray, steps: bool) -> np.ndarray:
    held = np.asarray(array)
    return held if steps else held[None]


def _gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``|a - b|`` in float64, zero where both are NaN and infinite where one is."""
    x, y = a.astype(np.float64), b.astype(np.float64)
    both = np.isnan(x) & np.isnan(y)
    gap = np.abs(x - y)
    gap = np.where(both, 0.0, gap)
    return np.where(np.isnan(gap), np.inf, gap)


def _per_step(values: np.ndarray) -> np.ndarray:
    return values.reshape(values.shape[0], -1).max(axis=1, initial=0.0)


def _largest(a: np.ndarray) -> float:
    held = np.abs(a.astype(np.float64))
    return float(np.nanmax(held)) if held.size and not np.all(np.isnan(held)) else 0.0


def judge(saved: np.ndarray, found: np.ndarray, scale: SourceScale, rtol: float, atol: float, *,
          forces: tuple[np.ndarray, np.ndarray] | None = None, steps: bool = True) -> Judged:
    """``found`` against ``saved``, both of one shape (the first axis the steps where ``steps``), on ``scale``: within
    where every step's difference is at most ``rtol`` times the source scale plus ``atol``. A field that is not of
    floating point, an integer or a flag, is judged exactly. ``forces`` are the saved and found forces held points pair
    with."""
    a, b = _steps(saved, steps), _steps(found, steps)
    kind = scale.kind
    if a.dtype.kind not in "fc" or b.dtype.kind not in "fc":
        differs = np.any((a != b).reshape(a.shape[0], -1), axis=1) if a.size else np.zeros(a.shape[0], dtype=bool)
        each, source, words = np.where(differs, np.inf, 0.0), 1.0, "exactly"
    elif kind is FieldKind.TWIST and scale.size is not None and a.shape[-1:] == (6,):
        size = scale.size
        gap = _gap(a, b).reshape(a.shape[0], -1, 6)
        each = (np.linalg.norm(gap[..., :3], axis=-1) + size * np.linalg.norm(gap[..., 3:], axis=-1)).max(axis=1)
        held = a.astype(np.float64).reshape(a.shape[0], -1, 6)
        speed = np.linalg.norm(held[..., :3], axis=-1) + size * np.linalg.norm(held[..., 3:], axis=-1)
        source = float(np.nanmax(speed)) if speed.size else 0.0
        words = (f"its surface points' speed, {source:.3e} m/s (|v| + L |w|, L {size:.3e} m, "
                 f"{scale.size_from})")
    elif kind is FieldKind.RATIO:
        each, source, words = _per_step(_gap(a, b)), 1.0, "one, the quantity it is a share of"
    elif kind is FieldKind.HELD_POINTS and forces is not None and a.shape[:-1] == np.shape(_steps(forces[0], steps)):
        acting = (_steps(forces[0], steps) != 0) | (_steps(forces[1], steps) != 0)
        gap = np.where(acting[..., None], _gap(a, b), 0.0)
        each, source = _per_step(gap), _largest(a)
        words = f"its own largest magnitude, {source:.3e}, on the rows whose force acts"
    else:
        each, source = _per_step(_gap(a, b)), _largest(a)
        words = f"its own largest magnitude, {source:.3e}"
    bound = rtol * source + atol
    beyond = np.flatnonzero(each > bound)
    largest = float(np.max(each, initial=0.0))
    ratio = largest / source if source > 0 else (0.0 if largest == 0 else math.inf)
    return Judged(ratio, source, int(beyond[0]) if beyond.size else None, words)


def largest_reach(hull: Any) -> float:
    """The largest distance of a body's hull from the body's origin (m): the largest over directions of its kind's
    reach in the body's frame, the reach along ``d`` being ``max_p p . d``, whose largest over unit ``d`` is
    ``max_p |p|``. Over the six axes, the eight diagonals and 2048 directions of a Fibonacci sphere, it is the largest
    distance to within the directions' spacing, from below (``mime.scene.shortfall_directions``)."""
    import jax

    from mime.core import jnp
    from mime.scene._sampling import shortfall_directions

    kind = hull.kind
    parameters = {held.name: hull.settings.get(held.name, getattr(kind, held.name)) for held in dataclasses.fields(kind)}
    directions = shortfall_directions()
    turn, centre = np.asarray(hull.rotation, dtype=np.float64), np.asarray(hull.centre, dtype=np.float64)

    def along(direction: Any) -> Any:
        return kind.reach(parameters, direction)

    reach = jax.vmap(along)(jnp.asarray(directions @ turn))
    return float(np.max(np.asarray(reach) + directions @ centre))


def _gyration(node: object) -> float | None:
    """A body's radius of gyration about its widest axis, ``sqrt(max eig(I) / m)``, from its node's ``mass`` and
    ``inertia`` (three moments, or nine numbers row-major); ``None`` where it holds neither."""
    params = cast(Mapping[str, Any], getattr(node, "params", {}))
    mass, inertia = params.get("mass"), params.get("inertia")
    if mass is None or inertia is None:
        return None
    held = np.asarray(inertia, dtype=np.float64).reshape(-1)
    tensor = np.diag(held) if held.size == 3 else held.reshape(3, 3) if held.size == 9 else None
    if tensor is None or not float(mass) > 0:
        return None
    return float(math.sqrt(max(float(np.max(np.linalg.eigvalsh(0.5 * (tensor + tensor.T)))), 0.0) / float(mass)))


def twist_size(name: str, node: object, hulls: Mapping[str, Any]) -> tuple[float | None, str]:
    """The size ``L`` a twist held by ``node`` (named ``name``, its body's name) is judged by, and where it came from:
    its body's hull's largest reach, where its parts give a hull whose kind answers a reach, else its radius of
    gyration."""
    from mime.registry import implements

    hull = hulls.get(name)
    if hull is not None and implements(hull.kind, "reach"):
        return largest_reach(hull), f"the largest reach of its hull, part {hull.part!r}"
    radius = _gyration(node)
    if radius is not None:
        return radius, "its radius of gyration about its widest axis, from its mass and inertia"
    return None, ""


def source_scales(nodes: Mapping[str, object], fields: Mapping[str, Collection[str]],
                  hulls: Mapping[str, Any]) -> dict[str, SourceScale]:
    """Each saved field's source scale, by its label (``node.field``), from the node's typed state: ``fields`` are the
    fields saved for each node in ``nodes``, ``hulls`` the experiment's bodies' hulls by body. A twist with no size to
    judge it by, and held points with no forces to pair with, are judged against their own largest magnitude."""
    scales: dict[str, SourceScale] = {}
    for name, held in sorted(fields.items()):
        node = nodes.get(name)
        if node is None:
            continue
        kinds = field_kinds(node, held)
        forces = paired_forces(kinds)
        for field_name, kind in kinds.items():
            label = f"{name}.{field_name}"
            if kind is FieldKind.TWIST:
                size, said = twist_size(name, node, hulls)
                scales[label] = SourceScale(kind, size, said) if size is not None else OWN
            elif kind is FieldKind.HELD_POINTS:
                scales[label] = SourceScale(kind, forces=f"{name}.{forces}") if forces is not None else OWN
            elif kind is FieldKind.RATIO:
                scales[label] = SourceScale(kind)
    return scales


__all__ = ["FLOAT32_RTOL", "FLOAT64_RTOL", "OWN", "SCALED_ATOL", "SCALED_FROM", "Judged", "SourceScale", "judge",
           "largest_reach", "source_scales", "tolerance_for", "twist_size"]
