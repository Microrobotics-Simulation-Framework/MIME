"""How ``verify`` judges a saved field where it claims no bits: in another environment, or from a reference no run
repeats bit for bit (one made on a GPU with XLA's deterministic mode off).

From the tenth saved form every field is split into **blocks by unit**, read from the units MIME declares for it
(:class:`FieldUnits`: the node's symbol table, the edges that read it, the stage's binding, the observations and the
flags): a pose into its position and its quaternion, a twist into its velocity and its angular velocity, a wrench into
its force and its torque, a resistance into its translational, coupling and rotational blocks. A field of one unit is
one block, and so is a field whose units MIME does not declare. A block is within tolerance when, at every reference
step, its largest difference is at most its **bound**::

    bound = max(rtol * the block's largest magnitude over the reference, RESOLUTION_FACTOR * its resolution) + atol

The tenth form writes ``atol`` as zero. The **resolution** is measured at save: the reference steps are run once more,
by the same compiled program, from a start one unit in the last place off (every nonzero floating element of every
node's initial state, :func:`one_ulp_up`, and of the small parameters the compiled step reads,
``mime.build._maddening.params_one_ulp_up``, one ulp up), and each block's largest divergence from the reference is
recorded in the lock (``reference.resolution``). It is how far the run's own rounding moves that block, so a block that
rounding moves far beyond its share of its own magnitude (the angular velocity of a body that barely turns) is judged
on its resolution, and every other block on its magnitude. A resolution that measures zero, or is not finite, gives no
floor: the ``rtol`` term alone.

**One mask.** Held points (``mime.core.HeldPoints``) are judged only on the rows whose paired force (``HeldForces``)
acts, in the reference or the rerun, and so are their largest magnitude and their resolution: a row that takes no force
holds wherever its search ended, which one unit in the last place moves by millimetres (MIME-ANO-144).

A folder of the ninth form or earlier is judged as its form says, element by element: ``|a - b| <= atol + rtol |b|``
with its own numbers (``numpy.allclose``)."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np

from mime.saved._versions import SOURCE_SCALE_ADDED

#: The first saved form whose fields are judged by block, on their magnitude and their measured resolution (and whose
#: ``atol`` is zero); a form before it is judged element by element.
SCALED_FROM = SOURCE_SCALE_ADDED

#: float32: each block's largest difference at most 1e-5 of its largest magnitude, or its floor. Design #263; the table
#: it rests on is ``tests/saved/test_tolerance_ratchet.py``'s.
FLOAT32_RTOL = 1e-5
#: float64: 2e-14 of a block's largest magnitude, or its floor. Design #263; no GPU was measured in float64.
FLOAT64_RTOL = 2e-14
#: The tenth form's absolute part: none.
SCALED_ATOL = 0.0
#: k, the floor's factor over a block's measured resolution. Design #266; the table it rests on is
#: ``tests/saved/test_tolerance_ratchet.py``'s.
RESOLUTION_FACTOR = 30.0

#: The name of the block of a field whose units MIME does not declare: the field whole.
UNDECLARED = "units not declared"
#: A composite field's leading parts are vectors of three, in the world frame: a pose is [position; quaternion], a
#: twist [velocity; angular velocity], a wrench [force; torque], and a resistance maps a twist to a wrench.
PART = 3


def tolerance_for(x64: bool) -> float:
    """The ``rtol`` a save writes for its precision."""
    return FLOAT64_RTOL if x64 else FLOAT32_RTOL


@dataclass(frozen=True)
class FieldUnits:
    """What one saved field is judged by: the units MIME declares for it (``None`` where it declares none), which split
    it into blocks, and the label (``node.field``) of the forces its held points pair with."""

    units: str | None = None
    forces: str | None = None


#: A field whose units MIME does not declare, and that pairs with no forces: one block.
OWN = FieldUnits()


def unit_parts(units: str | None) -> tuple[str, ...]:
    """A field's declared units, part by part: a composite lists its parts separated by ``;`` (``"m; 1"``)."""
    return () if units is None else tuple(part.strip() for part in units.split(";"))


def _merged(held: tuple[tuple[str, np.ndarray], ...]) -> tuple[tuple[str, np.ndarray], ...]:
    found: dict[str, np.ndarray] = {}
    for name, mask in held:
        found[name] = found[name] | mask if name in found else mask
    return tuple(found.items())


def blocks(units: str | None, shape: tuple[int, ...]) -> tuple[tuple[str, np.ndarray], ...]:
    """A field's blocks for one step of ``shape``: each block's units, by which it is named, and its elements (a mask of
    ``shape``). Two parts, ``"a; b"``, split the last axis into its first three and the rest (a pose, a twist, a wrench,
    and each row of frame poses); three, ``"a; b; c"``, a square operator of six in blocks of three, ``[a b; b c]``
    (a resistance). One part is one block; so is a field whose units are not declared, or whose composite units do not
    fit its shape, named by what it declares."""
    whole = np.ones(shape, dtype=bool)
    parts = unit_parts(units)
    if len(parts) <= 1:
        return ((parts[0] if parts else UNDECLARED, whole),)
    if len(parts) == 2 and shape and shape[-1] > PART:
        first = np.zeros(shape, dtype=bool)
        first[..., :PART] = True
        return _merged(((parts[0], first), (parts[1], ~first)))
    if len(parts) == 3 and len(shape) >= 2 and shape[-1] == shape[-2] == 2 * PART:
        rows = np.arange(2 * PART)[:, None] < PART
        columns = np.arange(2 * PART)[None, :] < PART
        return _merged(((parts[0], np.broadcast_to(rows & columns, shape).copy()),
                        (parts[1], np.broadcast_to(rows != columns, shape).copy()),
                        (parts[2], np.broadcast_to(~rows & ~columns, shape).copy())))
    return ((str(units), whole),)


def _steps(array: np.ndarray, steps: bool) -> np.ndarray:
    held = np.asarray(array)
    return held if steps else held[None]


def _gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``|a - b|`` in float64, zero where both are NaN and infinite where one is."""
    x, y = a.astype(np.float64), b.astype(np.float64)
    both = np.isnan(x) & np.isnan(y)
    gap = np.where(both, 0.0, np.abs(x - y))
    return np.where(np.isnan(gap), np.inf, gap)


def _acting(a: np.ndarray, field: FieldUnits, forces: tuple[np.ndarray, np.ndarray] | None,
            steps: bool) -> np.ndarray:
    """Where each element counts: everywhere, or, for held points paired with forces, on the rows whose force acts in
    either run."""
    if field.forces is None or forces is None:
        return np.ones(a.shape, dtype=bool)
    saved, found = _steps(forces[0], steps), _steps(forces[1], steps)
    if a.shape[:-1] != saved.shape or saved.shape != found.shape:
        return np.ones(a.shape, dtype=bool)
    return np.broadcast_to(((saved != 0) | (found != 0))[..., None], a.shape)


def _per_step(values: np.ndarray, where: np.ndarray) -> np.ndarray:
    held = np.where(where, values, 0.0)
    return held.reshape(held.shape[0], -1).max(axis=1, initial=0.0)


def _largest(values: np.ndarray, where: np.ndarray) -> float:
    held = np.abs(values.astype(np.float64))
    held = np.where(where & ~np.isnan(held), held, 0.0)
    return float(np.max(held, initial=0.0))


def resolution(saved: np.ndarray, nudged: np.ndarray, field: FieldUnits, *,
               forces: tuple[np.ndarray, np.ndarray] | None = None, steps: bool = True) -> dict[str, float | None]:
    """Each block's largest divergence of ``nudged`` (the reference steps rerun from a start one ulp off) from ``saved``
    (the reference), by block name; ``None`` where it is not finite. A field that is not of floating point has none."""
    a, b = _steps(saved, steps), _steps(nudged, steps)
    if a.dtype.kind not in "fc" or b.dtype.kind not in "fc" or a.shape != b.shape:
        return {}
    gap, acting = _gap(a, b), _acting(a, field, forces, steps)
    found: dict[str, float | None] = {}
    for name, mask in blocks(field.units, a.shape[1:]):
        largest = float(np.max(np.where(mask[None] & acting, gap, 0.0), initial=0.0))
        found[name] = largest if math.isfinite(largest) else None
    return found


@dataclass(frozen=True)
class Block:
    """One block's comparison: its name (its units), its largest difference over the steps, its bound, what the bound
    is made of (its largest magnitude and its floor, the resolution it was given), and the first step beyond it
    (``None`` where none is)."""

    name: str
    largest: float
    bound: float
    magnitude: float
    floor: float | None
    first: int | None

    def words(self, rtol: float, factor: float) -> str:
        made = f"{rtol:g} of its largest magnitude {self.magnitude:.3e}"
        if self.floor is not None and factor * self.floor > rtol * self.magnitude:
            made = f"{factor:g} times its resolution {self.floor:.3e}"
        return (f"its block in {self.name}: largest difference {self.largest:.3e} against a bound of {self.bound:.3e} "
                f"({made})")


@dataclass(frozen=True)
class Judged:
    """A field's comparison, block by block, and the first step at which a block passes its bound (``None`` where
    none does)."""

    blocks: tuple[Block, ...]
    first: int | None
    exact: bool = False

    @property
    def within(self) -> bool:
        return self.first is None

    @property
    def beyond(self) -> Block | None:
        """The block that passes its bound first, and of those the one with the largest difference; ``None`` where none
        does."""
        held = [block for block in self.blocks if block.first is not None]
        return min(held, key=lambda block: (block.first, -block.largest)) if held else None


def floor_of(value: object) -> float | None:
    """A recorded resolution as a floor: a finite number above zero, else none."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def judge(saved: np.ndarray, found: np.ndarray, field: FieldUnits, rtol: float, atol: float, *,
          resolutions: Mapping[str, object] | None = None, factor: float = RESOLUTION_FACTOR,
          forces: tuple[np.ndarray, np.ndarray] | None = None, steps: bool = True) -> Judged:
    """``found`` against ``saved``, both of one shape (the first axis the steps where ``steps``), block by block: within
    where every step's largest difference in each block is at most ``max(rtol * its largest magnitude, factor * its
    resolution) + atol`` (``resolutions`` by block name, as the lock records them). A field that is not of floating
    point, an integer or a flag, is judged exactly. ``forces`` are the saved and found forces held points pair with."""
    a, b = _steps(saved, steps), _steps(found, steps)
    if a.dtype.kind not in "fc" or b.dtype.kind not in "fc":
        differs = np.any((a != b).reshape(a.shape[0], -1), axis=1) if a.size else np.zeros(a.shape[0], dtype=bool)
        beyond = np.flatnonzero(differs)
        first = int(beyond[0]) if beyond.size else None
        return Judged((Block("exactly", float(np.any(differs)), 0.0, 0.0, None, first),), first, exact=True)
    gap, acting = _gap(a, b), _acting(a, field, forces, steps)
    held = resolutions or {}
    judged: list[Block] = []
    for name, mask in blocks(field.units, a.shape[1:]):
        where = mask[None] & acting
        each = _per_step(gap, where)
        magnitude, floor = _largest(a, where), floor_of(held.get(name))
        bound = max(rtol * magnitude, factor * floor if floor is not None else 0.0) + atol
        beyond = np.flatnonzero(each > bound)
        judged.append(Block(name, float(np.max(each, initial=0.0)), bound, magnitude, floor,
                            int(beyond[0]) if beyond.size else None))
    firsts = [block.first for block in judged if block.first is not None]
    return Judged(tuple(judged), min(firsts) if firsts else None)


def one_ulp_up(state: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """``state`` with every nonzero, finite floating element one unit in the last place up, in its own dtype: the start
    the save's resolution is measured from. Zeros, integers and flags are kept, so a counter or a record that starts
    at zero starts there."""
    moved: dict[str, dict[str, np.ndarray]] = {}
    for node, fields in state.items():
        held: dict[str, np.ndarray] = {}
        for name, value in fields.items():
            array = np.asarray(value)
            if array.dtype.kind == "f":
                up = np.nextafter(array, np.asarray(np.inf, dtype=array.dtype))
                array = np.where((array != 0) & np.isfinite(array), up, array).astype(array.dtype)
            held[name] = array
        moved[node] = held
    return moved


#: In a symbol row's name, what stands for a reader slot's name (``flux_density_{slot}``).
_SLOT = "{slot}"


def symbol_units(node: object, fields: Collection[str]) -> dict[str, str]:
    """The units a node's symbol table states for its state fields, by field: a row named with ``{slot}`` names each
    field it matches with a slot's name in its place."""
    from mime.core import Symbol, SymbolKind

    found: dict[str, str] = {}
    rows: tuple[Symbol, ...] = getattr(type(node), "symbols", ())
    for row in rows:
        if row.kind is not SymbolKind.STATE:
            continue
        if _SLOT not in row.name:
            if row.name in fields:
                found.setdefault(row.name, row.units)
            continue
        head, _, tail = row.name.partition(_SLOT)
        pattern = re.compile(re.escape(head) + r"[A-Za-z0-9_]+" + re.escape(tail))
        for name in fields:
            if pattern.fullmatch(name):
                found.setdefault(name, row.units)
    return found


__all__ = ["FLOAT32_RTOL", "FLOAT64_RTOL", "OWN", "PART", "RESOLUTION_FACTOR", "SCALED_ATOL", "SCALED_FROM",
           "UNDECLARED", "Block", "FieldUnits", "Judged", "blocks", "floor_of", "judge", "one_ulp_up", "resolution",
           "symbol_units", "tolerance_for", "unit_parts"]
